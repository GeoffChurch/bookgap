"""Snapshot-based OpenAlex ingest (parallel to openalex_ingest.py).

This module exists because the OpenAlex API enforces a daily call cap that
makes full-corpus ingestion impractical. Instead, we download the public S3
snapshot directly.

High-level flow: list shard keys up to snapshot_date → download each shard
in turn → stream works through a Bucketer → flush buckets to per-scope JSONL
spill files on disk → finalize_all() reads each spill file, normalises,
filters, deduplicates, and writes the final per-scope Parquet. Spill files
survive crashes, so a resumed run skips already-completed shards but still
includes their data via the pre-existing spill files.

run() is the canonical entrypoint. See docs/superpowers/specs/2026-04-14-snapshot-ingest-design.md.
"""
from __future__ import annotations

import gzip
import json
import queue
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, IO, Iterator, Optional

import boto3
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.client import Config

from bookgap.util import CACHE_DIR, Filter, LOGGER


class DiskFullError(RuntimeError):
    """Raised when CACHE_DIR is too full to continue ingest."""


@dataclass
class SnapshotState:
    snapshot_date: str
    shards_completed: list[str] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


def load_state(path: Path, snapshot_date: str) -> SnapshotState:
    if not path.exists():
        return SnapshotState(snapshot_date=snapshot_date)
    data = json.loads(path.read_text())
    if data.get("snapshot_date") != snapshot_date:
        LOGGER.warning(
            "snapshot_date changed (%s -> %s); wiping resume state",
            data.get("snapshot_date"), snapshot_date,
        )
        return SnapshotState(snapshot_date=snapshot_date)
    return SnapshotState(**data)


def save_state(path: Path, state: SnapshotState) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(path)


def check_disk_or_abort(cache_dir: Path, abort_pct: float) -> None:
    usage = shutil.disk_usage(cache_dir)
    used_pct = 100.0 * usage.used / usage.total
    if used_pct >= abort_pct:
        raise DiskFullError(
            f"disk usage {used_pct:.1f}% >= {abort_pct}% abort threshold; "
            f"free {usage.free / 1e9:.1f} GB on {cache_dir}"
        )


def make_s3_client():
    # botocore does not set HTTP socket timeouts by default, so a dead
    # connection (remote-side FIN with unread bytes, i.e. CLOSE_WAIT) hangs
    # the download thread forever. We've observed this both with pyalex and
    # boto3; pyalex_client monkey-patches HTTPAdapter.send but that does not
    # affect botocore's own urllib3 stack. Set explicit connect/read timeouts
    # and let botocore retry a few times at the transfer layer.
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def taxonomy_path(snapshot_date: str) -> Path:
    """Path to the per-snapshot derived taxonomy file."""
    return CACHE_DIR / f"snapshot_taxonomy_{snapshot_date}.json"


@dataclass(frozen=True)
class PinnedTaxonomy:
    """Read-only view of the taxonomy derived from a works snapshot.

    Built bottom-up from the per-work primary_topic chain seen during ingest,
    so it reflects exactly the taxonomy state at snapshot_date. Records use
    the same nested-dict shape as OpenAlex's official taxonomy entity schema
    so callers can lookup parents via tax.subfields[sf_id]["field"]["id"]
    etc., without caring whether the source was the API or a snapshot.
    """
    snapshot_date: str
    fields: dict[str, dict]      # field_id (e.g. "31") -> field record
    subfields: dict[str, dict]   # subfield_id (e.g. "3102") -> subfield record
    topics: dict[str, dict]      # topic_id (e.g. "T10805") -> topic record


_TAXONOMY_CACHE: dict[str, PinnedTaxonomy] = {}


class _TaxonomyAccumulator:
    """Collects distinct domain/field/subfield/topic tuples during streaming.

    Each call to observe(work) extracts the work's primary_topic chain and
    threads it into the in-memory taxonomy. emit() returns the final dict
    structure ready to merge with any pre-existing taxonomy file.
    """

    def __init__(self) -> None:
        self.fields: dict[str, dict] = {}
        self.subfields: dict[str, dict] = {}
        self.topics: dict[str, dict] = {}

    @staticmethod
    def _short(record: dict) -> Optional[str]:
        rid = record.get("id")
        return rid.rsplit("/", 1)[-1] if isinstance(rid, str) else None

    def observe(self, work: dict) -> None:
        pt = work.get("primary_topic")
        if not isinstance(pt, dict):
            return
        topic_id = self._short(pt)
        if topic_id is None:
            return
        subfield = pt.get("subfield") or {}
        field = pt.get("field") or {}
        domain = pt.get("domain") or {}
        sf_id = self._short(subfield)
        f_id = self._short(field)
        if sf_id is None or f_id is None:
            return

        self.topics.setdefault(topic_id, {
            "id": pt.get("id"),
            "display_name": pt.get("display_name"),
            "subfield": {"id": subfield.get("id"), "display_name": subfield.get("display_name")},
        })

        sf_rec = self.subfields.setdefault(sf_id, {
            "id": subfield.get("id"),
            "display_name": subfield.get("display_name"),
            "field": {"id": field.get("id"), "display_name": field.get("display_name")},
            "topics": [],
            "_topic_ids": set(),
        })
        if topic_id not in sf_rec["_topic_ids"]:
            sf_rec["_topic_ids"].add(topic_id)
            sf_rec["topics"].append({"id": pt.get("id"), "display_name": pt.get("display_name")})

        f_rec = self.fields.setdefault(f_id, {
            "id": field.get("id"),
            "display_name": field.get("display_name"),
            "domain": {"id": domain.get("id"), "display_name": domain.get("display_name")},
            "subfields": [],
            "_subfield_ids": set(),
        })
        if sf_id not in f_rec["_subfield_ids"]:
            f_rec["_subfield_ids"].add(sf_id)
            f_rec["subfields"].append({"id": subfield.get("id"), "display_name": subfield.get("display_name")})

    def emit(self) -> dict[str, dict[str, dict]]:
        """Return JSON-serializable dicts. Strips the internal _id sets."""
        def clean(d: dict) -> dict:
            return {k: v for k, v in d.items() if not k.startswith("_")}
        return {
            "fields": {k: clean(v) for k, v in self.fields.items()},
            "subfields": {k: clean(v) for k, v in self.subfields.items()},
            "topics": {k: clean(v) for k, v in self.topics.items()},
        }


def _save_taxonomy(snapshot_date: str, accumulator: _TaxonomyAccumulator) -> None:
    """Write the accumulator's contents to taxonomy_path, merging with any
    existing file for the same snapshot_date so successive ingests
    accumulate into a single taxonomy."""
    out_path = taxonomy_path(snapshot_date)
    new = accumulator.emit()
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        if prior.get("snapshot_date") not in (None, snapshot_date):
            raise ValueError(
                f"Existing taxonomy at {out_path} has snapshot_date={prior.get('snapshot_date')!r}, "
                f"refusing to merge with snapshot_date={snapshot_date!r}"
            )
        for entity in ("fields", "subfields", "topics"):
            for k, v in (prior.get(entity) or {}).items():
                if k in new[entity]:
                    # Merge child lists from prior (some scopes may have been
                    # ingested before; their children should be retained).
                    if entity == "fields":
                        seen = {sf["id"] for sf in new[entity][k].get("subfields", [])}
                        for sf in v.get("subfields", []):
                            if sf.get("id") and sf["id"] not in seen:
                                new[entity][k].setdefault("subfields", []).append(sf)
                    elif entity == "subfields":
                        seen = {t["id"] for t in new[entity][k].get("topics", [])}
                        for t in v.get("topics", []):
                            if t.get("id") and t["id"] not in seen:
                                new[entity][k].setdefault("topics", []).append(t)
                else:
                    new[entity][k] = v
    out = {"snapshot_date": snapshot_date, **new}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    _TAXONOMY_CACHE.pop(snapshot_date, None)
    LOGGER.info(
        "snapshot_ingest: wrote derived taxonomy %s (%d fields, %d subfields, %d topics)",
        out_path, len(out["fields"]), len(out["subfields"]), len(out["topics"]),
    )


def load_taxonomy(snapshot_date: str) -> PinnedTaxonomy:
    """Load (and memoize) the per-snapshot derived taxonomy.

    Raises FileNotFoundError with a helpful message if the snapshot ingest
    for this snapshot_date has not produced a taxonomy file yet.
    """
    if snapshot_date in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[snapshot_date]
    path = taxonomy_path(snapshot_date)
    if not path.exists():
        raise FileNotFoundError(
            f"Derived taxonomy missing for snapshot_date={snapshot_date!r} at {path}. "
            f"Run bookgap.snapshot_ingest.run(work_filter=..., snapshot_date={snapshot_date!r}) "
            f"to ingest works (the taxonomy is built as a byproduct)."
        )
    raw = json.loads(path.read_text())
    if raw.get("snapshot_date") != snapshot_date:
        raise ValueError(
            f"Taxonomy file {path} has snapshot_date={raw.get('snapshot_date')!r}, expected {snapshot_date!r}"
        )
    tax = PinnedTaxonomy(
        snapshot_date=snapshot_date,
        fields=raw.get("fields", {}) or {},
        subfields=raw.get("subfields", {}) or {},
        topics=raw.get("topics", {}) or {},
    )
    _TAXONOMY_CACHE[snapshot_date] = tax
    return tax


def list_shards(client, bucket: str, prefix: str, snapshot_date: str) -> list[str]:
    """List snapshot shard keys not newer than snapshot_date.

    We use list_objects_v2 rather than parsing the snapshot manifest because
    the date partition is already encoded in the key path, so a single
    paginated listing suffices and avoids a second schema dependency.
    """
    paginator = client.get_paginator("list_objects_v2")
    shards: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".gz"):
                continue
            partition = _partition_date_from_key(key)
            if partition is None or partition > snapshot_date:
                continue
            shards.append(key)
    return sorted(shards)


_PARTITION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _partition_date_from_key(key: str) -> Optional[str]:
    # Keys look like: data/works/updated_date=2026-04-01/part_012.gz.
    # We require an exact YYYY-MM-DD partition; anything else is treated as
    # unrecognized and dropped by list_shards.
    for seg in key.split("/"):
        if seg.startswith("updated_date="):
            value = seg.split("=", 1)[1]
            if _PARTITION_DATE_RE.match(value):
                return value
            return None
    return None


# ---------------------------------------------------------------------------
# Shard streaming and bucket flushing (Task 4)
# ---------------------------------------------------------------------------

ScopeKey = tuple[str, str, str, str]  # (field, subfield, topic, type)


def _scope_key(work: dict) -> Optional[ScopeKey]:
    pt = work.get("primary_topic")
    if not pt:
        return None
    field_id = pt.get("field", {}).get("id", "").rsplit("/", 1)[-1]
    subfield_id = pt.get("subfield", {}).get("id", "").rsplit("/", 1)[-1]
    topic_id = pt.get("id", "").rsplit("/", 1)[-1]
    wtype = work.get("type", "")
    if not all([field_id, subfield_id, topic_id, wtype]):
        return None
    return (field_id, subfield_id, topic_id, wtype)


def stream_shard(
    fileobj: IO[bytes],
    target_fields: Optional[set[str]],
    prefilter_patterns: Optional[list[bytes]] = None,
) -> Iterator[tuple[ScopeKey, dict]]:
    """Yield (scope_key, work) tuples from a gzipped JSONL shard.

    Works without a resolvable primary_topic or malformed JSON lines are
    silently skipped. If target_fields is not None, only works whose
    field_id is in that set are yielded.

    If prefilter_patterns is provided, raw lines that do not contain at least
    one of the byte patterns are skipped without being parsed. This is a
    ~2.5x speedup when filtering by a small field set, since most lines in a
    snapshot shard belong to fields we're not targeting and full JSON parsing
    is the hot path. The prefilter is advisory: it may over-accept (parse a
    line that ends up not matching) but must never under-accept (reject a
    line that would have matched under full parsing). See the byte layout of
    OpenAlex snapshot records for safe pattern choices — e.g. for field 31,
    use b'/fields/31"' which anchors against the URL path tail of
    primary_topic.field.id and cannot collide with numeric substrings.
    """
    with gzip.GzipFile(fileobj=fileobj, mode="rb") as gz:
        for line in gz:
            if prefilter_patterns is not None and not any(p in line for p in prefilter_patterns):
                continue
            try:
                work = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            key = _scope_key(work)
            if key is None:
                continue
            if target_fields is not None and key[0] not in target_fields:
                continue
            yield key, work


Sink = Callable[[ScopeKey, list[dict]], None]


class Bucketer:
    """Accumulates works by scope key with two-level flushing.

    Per-bucket flush: when a bucket reaches flush_threshold rows, the sink
    is called with the bucket's contents and the bucket is cleared.

    Resident cap flush: if total resident rows across all buckets exceeds
    max_resident, the largest bucket is flushed first.

    The sink callable is (scope_key, list[dict]) -> None. It should write
    the rows out and return.
    """

    def __init__(self, sink: Sink, flush_threshold: int, max_resident: int):
        self._sink = sink
        self._flush_threshold = flush_threshold
        self._max_resident = max_resident
        self._buckets: dict[ScopeKey, list[dict]] = defaultdict(list)
        self._resident = 0

    def insert(self, key: ScopeKey, work: dict) -> None:
        bucket = self._buckets[key]
        bucket.append(work)
        self._resident += 1
        if len(bucket) >= self._flush_threshold:
            self._flush_bucket(key)
        elif self._resident > self._max_resident:
            # Flush the currently-largest bucket (ties broken by Python dict
            # insertion order, i.e. by first-seen scope).
            largest = max(self._buckets, key=lambda k: len(self._buckets[k]))
            self._flush_bucket(largest)

    def _flush_bucket(self, key: ScopeKey) -> None:
        rows = self._buckets.pop(key, [])
        if not rows:
            return
        self._resident -= len(rows)
        self._sink(key, rows)

    def flush_all(self) -> None:
        # Sort for stable, reproducible log/artifact output when debugging.
        for key in sorted(self._buckets.keys()):
            self._flush_bucket(key)


# ---------------------------------------------------------------------------
# Work normalization (Task 5)
# ---------------------------------------------------------------------------


def _normalize_snapshot_work(work: dict, scope_key: ScopeKey) -> Optional[dict]:
    """Transform a raw snapshot JSON record into the shape _filter_works expects.

    Returns None if the work has no abstract (pre-filter; mirrors the API path's
    dropped_no_abstract behavior).
    """
    from bookgap.openalex_ingest import resolve_abstract_inverted_index

    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return None
    try:
        abstract = resolve_abstract_inverted_index(inverted)
    except Exception as exc:
        ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
        work_id = ids.get("openalex") or work.get("id") or "<unknown>"
        LOGGER.warning(
            "snapshot_ingest: failed to decode abstract for %s, skipping: %s",
            work_id, exc,
        )
        return None
    work = dict(work)  # copy
    work["abstract"] = abstract
    work.pop("abstract_inverted_index", None)
    ids = work.get("ids")
    if isinstance(ids, dict):
        work["id"] = ids.get("openalex", work.get("id"))
        work.pop("ids", None)
    field_id, subfield_id, topic_id, _wtype = scope_key
    work["topic"] = topic_id
    work["field_id"] = field_id
    work["subfield_id"] = subfield_id
    work["topic_id"] = topic_id
    return work


# ---------------------------------------------------------------------------
# SpillSink (Task 5)
# ---------------------------------------------------------------------------


_SPILL_SUFFIX = ".parquet"
# zstd compression level 1 is fast and gives near-best compression on
# typed columnar data. Most of the win over the prior gzip-JSONL design
# comes from columnar layout + dictionary encoding (work types, topic IDs,
# language codes), not from the compression algorithm itself.
_SPILL_ZSTD_LEVEL = 1


class SpillSink:
    """Bucketer sink that writes per-scope works as zstd-compressed parquet shards.

    Each flush produces one new sequence-numbered shard file under
    spill_dir/<field>/<subfield>/<topic>/<wtype>_NNNN.parquet. No file
    handles persist between calls (each open is open-write-close), so
    file-descriptor limits aren't an issue regardless of scope count.

    On finalize_all(), all shards for a scope are concatenated, filtered
    and deduped through the existing pipeline, written as the final
    per-scope parquet (atomic), and the shards are deleted.

    disk_abort_root / disk_abort_pct: if set, check_disk_or_abort runs
    after each flush so growth within a single shard can't silently
    overshoot the per-shard check at the run() level.
    """

    def __init__(
        self,
        spill_dir: Path,
        disk_abort_root: Optional[Path] = None,
        disk_abort_pct: Optional[float] = None,
    ):
        self._spill_dir = spill_dir
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._disk_abort_root = disk_abort_root
        self._disk_abort_pct = disk_abort_pct
        self._files_written: set[ScopeKey] = set()
        self._next_seq: dict[ScopeKey, int] = {}
        # Discover existing shards from interrupted prior runs so finalize_all
        # picks them up. Also derive the next sequence number per scope so we
        # don't collide with prior shards.
        for spill_path in self._spill_dir.rglob(f"*{_SPILL_SUFFIX}"):
            try:
                rel = spill_path.relative_to(self._spill_dir)
                parts = rel.parts  # (field, subfield, topic, "<wtype>_NNNN.parquet")
                if len(parts) != 4:
                    continue
                field, subfield, topic, fname = parts
                stem = fname.removesuffix(_SPILL_SUFFIX)
                if "_" not in stem:
                    continue
                wtype, seq_str = stem.rsplit("_", 1)
                try:
                    seq = int(seq_str)
                except ValueError:
                    continue
                key = (field, subfield, topic, wtype)
                self._files_written.add(key)
                self._next_seq[key] = max(self._next_seq.get(key, 0), seq + 1)
            except ValueError:
                continue

    def _scope_dir(self, key: ScopeKey) -> Path:
        field, subfield, topic, _wtype = key
        return self._spill_dir / field / subfield / topic

    def _spill_path(self, key: ScopeKey, seq: int) -> Path:
        wtype = key[3]
        return self._scope_dir(key) / f"{wtype}_{seq:04d}{_SPILL_SUFFIX}"

    def _shards_for(self, key: ScopeKey) -> list[Path]:
        return sorted(self._scope_dir(key).glob(f"{key[3]}_*{_SPILL_SUFFIX}"))

    def __call__(self, key: ScopeKey, rows: list[dict]) -> None:
        """Write `rows` as one new parquet shard.

        Rows must already be normalized into the canonical work shape
        (display_name, abstract decoded, id, etc.). Normalization happens
        upstream in run() before bucketing.
        """
        if not rows:
            return
        from bookgap.openalex_ingest import works_to_table

        scope_dir = self._scope_dir(key)
        scope_dir.mkdir(parents=True, exist_ok=True)
        seq = self._next_seq.get(key, 0)
        path = self._spill_path(key, seq)
        self._next_seq[key] = seq + 1

        table = works_to_table(rows)
        pq.write_table(table, path, compression="zstd", compression_level=_SPILL_ZSTD_LEVEL)
        self._files_written.add(key)
        if self._disk_abort_root is not None and self._disk_abort_pct is not None:
            check_disk_or_abort(self._disk_abort_root, self._disk_abort_pct)

    def finalize_all(self) -> None:
        """For each scope: concat shards, filter+dedup, write final parquet, delete shards."""
        from bookgap.openalex_ingest import (
            _filter_works_iter,
            _deduplicate_works,
            _write_scope_parquet,
            _scope_parquet_path,
            ResolvedScope,
        )
        for key in sorted(self._files_written):
            shard_paths = self._shards_for(key)
            if not shard_paths:
                continue
            field, subfield, topic, wtype = key
            scope = ResolvedScope(field=field, subfield=subfield, topic=topic, work_type=wtype)

            tables = [pq.read_table(p) for p in shard_paths]
            combined = pa.concat_tables(tables)
            works = combined.to_pylist()

            filtered = list(_filter_works_iter(scope, iter(works)))
            deduped = _deduplicate_works(filtered, scope)
            parquet_path = _scope_parquet_path(scope)
            _write_scope_parquet(parquet_path, deduped)
            for p in shard_paths:
                p.unlink()
        self._files_written.clear()
        self._next_seq.clear()


# ---------------------------------------------------------------------------
# Top-level run() entrypoint (Task 5)
# ---------------------------------------------------------------------------


DEFAULT_DOWNLOAD_CONCURRENCY = 4
DEFAULT_DOWNLOAD_QUEUE_DEPTH = 6


def run(
    *,
    work_filter: "Filter | list[Filter]",
    snapshot_date: str,
    bucket: str = "openalex",
    prefix: str = "data/works/",
    shard_row_flush_threshold: int = 10_000,
    max_resident_rows: int = 1_000_000,
    disk_abort_pct: float = 90.0,
    state_path: Optional[Path] = None,
    spill_dir: Optional[Path] = None,
    s3_client=None,
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    download_queue_depth: int = DEFAULT_DOWNLOAD_QUEUE_DEPTH,
) -> None:
    """Top-level snapshot ingest entrypoint.

    work_filter may be a single Filter or a list of Filters. Each work in
    the snapshot is matched against any of the supplied filters via
    Filter.check_work; matching works are bucketed by their (field,
    subfield, topic, work_type) tuple and spilled to disk.

    The taxonomy is built bottom-up from the matched works themselves —
    every primary_topic chain seen contributes to the per-snapshot
    taxonomy file. No separate taxonomy snapshot is required, and the
    taxonomy is guaranteed to be consistent with the works.
    """
    state_path = state_path or (CACHE_DIR / "snapshot_state.json")
    spill_dir = spill_dir or (CACHE_DIR / "snapshot_tmp")
    check_disk_or_abort(CACHE_DIR, disk_abort_pct)

    filters = list(work_filter) if isinstance(work_filter, list) else [work_filter]
    if not filters:
        raise ValueError("work_filter must be a Filter or a non-empty list of Filters")
    for i, f in enumerate(filters):
        if f.work_types is None:
            raise ValueError(
                f"work_filter[{i}].work_types must be provided "
                f"(e.g. Filter(field='31', work_types=('book','article','preprint','book-chapter')))"
            )

    LOGGER.info(
        "snapshot_ingest: %d filter(s), snapshot_date=%s, filters=%s",
        len(filters), snapshot_date, [str(f) for f in filters],
    )

    state = load_state(state_path, snapshot_date)
    state.started_at = state.started_at or datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)

    client = s3_client or make_s3_client()
    shards = list_shards(client, bucket, prefix, snapshot_date)
    LOGGER.info(
        "snapshot_ingest: %d shards total, %d already done",
        len(shards),
        len(state.shards_completed),
    )

    sink = SpillSink(spill_dir, disk_abort_root=CACHE_DIR, disk_abort_pct=disk_abort_pct)
    bucketer = Bucketer(sink, flush_threshold=shard_row_flush_threshold, max_resident=max_resident_rows)
    taxonomy_acc = _TaxonomyAccumulator()

    tmp_shard_dir = CACHE_DIR / "snapshot_download_tmp"
    tmp_shard_dir.mkdir(parents=True, exist_ok=True)

    # Build a bytes-level prefilter from whatever's most specific in the
    # filters. For a topic-only filter, anchor on /T<topic>"; for subfield,
    # /subfields/<id>"; for field, /fields/<id>"; for domain, /domains/<id>".
    # Each filter contributes its own pattern; if any pattern matches the
    # raw line, we parse and run the full check_work. The prefilter is
    # advisory: it must not under-accept (drop a line that would have
    # matched), so we pick the most-specific level present per filter.
    def _prefilter_pattern(f: Filter) -> bytes:
        if f.topic is not None:
            return f'/{f.topic}"'.encode()
        if f.subfield is not None:
            return f'/subfields/{f.subfield}"'.encode()
        if f.field is not None:
            return f'/fields/{f.field}"'.encode()
        if f.domain is not None:
            return f'/domains/{f.domain}"'.encode()
        return b""  # accept-all
    prefilter_patterns = [_prefilter_pattern(f) for f in filters]
    if any(p == b"" for p in prefilter_patterns):
        prefilter_patterns = None  # one filter accepts everything → no prefilter
    # stream_shard's `target_fields` set is a coarse field-level filter
    # applied after JSON parse. Skip it (pass None) since check_work below
    # is the definitive predicate.
    target_fields_set = None

    completed_set = set(state.shards_completed)
    pending = [k for k in shards if k not in completed_set]
    LOGGER.info(
        "snapshot_ingest: %d pending shards, download_concurrency=%d queue_depth=%d",
        len(pending),
        download_concurrency,
        download_queue_depth,
    )

    # Producer: concurrent-downloader thread. Downloads shards with N workers
    # and streams completed (shard_key, tmp_path) tuples via a bounded queue.
    # Main thread consumes from the queue and does the parse+bucket work, so
    # download and parse overlap.
    done_q: queue.Queue = queue.Queue(maxsize=download_queue_depth)
    _SENTINEL = object()
    _producer_error: list[BaseException] = []

    def _download_one(shard_key: str) -> tuple[str, Path]:
        # Concurrent downloads share tmp_shard_dir, so we must disambiguate
        # per shard. Every OpenAlex shard has basename part_NNNN.gz, so using
        # Path(shard_key).name would collide across dates and cause a write
        # race. Encode the full key into a flat filename.
        safe_name = shard_key.replace("/", "__")
        tmp_path = tmp_shard_dir / safe_name
        client.download_file(bucket, shard_key, str(tmp_path))
        return shard_key, tmp_path

    def _producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=download_concurrency) as pool:
                in_flight: dict = {}
                pending_iter = iter(pending)
                # Seed the pool up to queue_depth.
                for _ in range(download_queue_depth):
                    try:
                        shard_key = next(pending_iter)
                    except StopIteration:
                        break
                    fut = pool.submit(_download_one, shard_key)
                    in_flight[fut] = shard_key
                while in_flight:
                    done = next(as_completed(in_flight))
                    shard_key = in_flight.pop(done)
                    try:
                        k, tmp_path = done.result()
                    except Exception as e:
                        LOGGER.error("snapshot_ingest: download failed for %s: %s", shard_key, e)
                        continue
                    finally:
                        # Refill the pool before blocking on the next done.
                        try:
                            next_key = next(pending_iter)
                            fut = pool.submit(_download_one, next_key)
                            in_flight[fut] = next_key
                        except StopIteration:
                            pass
                    done_q.put((k, tmp_path))
        except BaseException as e:
            _producer_error.append(e)
        finally:
            done_q.put(_SENTINEL)

    producer_thread = threading.Thread(target=_producer, name="snapshot_downloader", daemon=False)
    producer_thread.start()

    try:
        while True:
            item = done_q.get()
            if item is _SENTINEL:
                break
            shard_key, tmp_path = item
            check_disk_or_abort(CACHE_DIR, disk_abort_pct)
            matched = 0
            try:
                with tmp_path.open("rb") as f:
                    for key, work in stream_shard(f, target_fields_set, prefilter_patterns=prefilter_patterns):
                        # Definitive predicate: at least one filter accepts
                        # this work. Drops off-target work types, topics
                        # outside the requested subfield, etc.
                        if not any(filt.check_work(work) for filt in filters):
                            continue
                        # Observe taxonomy from the raw work (which still has
                        # the full primary_topic chain) before normalization
                        # strips/transforms fields.
                        taxonomy_acc.observe(work)
                        normalized = _normalize_snapshot_work(work, key)
                        if normalized is None:
                            continue
                        bucketer.insert(key, normalized)
                        matched += 1
            finally:
                tmp_path.unlink(missing_ok=True)
            LOGGER.info(
                "snapshot_ingest: shard %s done — %d works matched filters",
                shard_key,
                matched,
            )
            state.shards_completed.append(shard_key)
            save_state(state_path, state)
    finally:
        producer_thread.join()

    if _producer_error:
        raise _producer_error[0]

    bucketer.flush_all()
    sink.finalize_all()

    # Persist the derived taxonomy. Done after finalize_all so it only
    # records scopes whose parquets were actually written.
    _save_taxonomy(snapshot_date, taxonomy_acc)

    state.finished_at = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)
    LOGGER.info("snapshot_ingest: done")
