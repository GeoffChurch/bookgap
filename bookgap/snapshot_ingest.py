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
                work = json.loads(line)
            except json.JSONDecodeError:
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


class SpillSink:
    """Bucketer sink that spills each flush to a per-scope JSONL file on disk.

    On finalize_all(), reads each spill file back, normalizes+filters+dedups
    its works through the existing pipeline, writes the final per-scope
    parquet, and deletes the spill file.
    """

    def __init__(self, spill_dir: Path):
        self._spill_dir = spill_dir
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._files_written: set[ScopeKey] = set()
        # Discover any spill files left over from an interrupted prior run so
        # finalize_all() picks them up even if no new rows arrive for those scopes.
        for jsonl_path in self._spill_dir.rglob("*.jsonl"):
            try:
                rel = jsonl_path.relative_to(self._spill_dir)
                parts = rel.parts  # (field, subfield, topic, "<wtype>.jsonl")
                if len(parts) != 4:
                    continue
                field, subfield, topic, wtype_file = parts
                wtype = wtype_file.removesuffix(".jsonl")
                self._files_written.add((field, subfield, topic, wtype))
            except ValueError:
                continue

    def _spill_path(self, key: ScopeKey) -> Path:
        field, subfield, topic, wtype = key
        return self._spill_dir / field / subfield / topic / f"{wtype}.jsonl"

    def __call__(self, key: ScopeKey, rows: list[dict]) -> None:
        """Append rows as JSONL to the scope's spill file."""
        path = self._spill_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")
        self._files_written.add(key)

    def finalize_all(self) -> None:
        """For each scope with a spill file: normalize, filter, dedup, write parquet, delete spill."""
        from bookgap.openalex_ingest import (
            _filter_works_iter,
            _deduplicate_works,
            _write_scope_parquet,
            _scope_parquet_path,
            ResolvedScope,
        )
        for key in sorted(self._files_written):
            spill_path = self._spill_path(key)
            if not spill_path.exists():
                continue
            field, subfield, topic, wtype = key
            scope = ResolvedScope(field=field, subfield=subfield, topic=topic, work_type=wtype)

            def works_iter(sp=spill_path, k=key):
                with sp.open() as f:
                    for line in f:
                        raw = json.loads(line)
                        normalized = _normalize_snapshot_work(raw, k)
                        if normalized is not None:
                            yield normalized

            filtered = list(_filter_works_iter(scope, works_iter()))
            deduped = _deduplicate_works(filtered, scope)
            parquet_path = _scope_parquet_path(scope)
            _write_scope_parquet(parquet_path, deduped)
            spill_path.unlink()
        self._files_written.clear()


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
    disk_abort_pct: float = 95.0,
    state_path: Optional[Path] = None,
    spill_dir: Optional[Path] = None,
    s3_client=None,
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    download_queue_depth: int = DEFAULT_DOWNLOAD_QUEUE_DEPTH,
) -> None:
    """Top-level snapshot ingest entrypoint.

    work_filter may be a single Filter or a list of Filters. When a list is
    passed, each filter is resolved independently and the resulting scopes
    are unioned, so one streaming pass can populate multiple fields that
    aren't reachable from a single Filter (e.g. three non-contiguous fields
    under the same domain).
    """
    from bookgap.openalex_ingest import _resolve_filter_scopes

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

    # Resolve each filter into concrete scopes; union the results.
    resolved_scopes = []
    seen: set[tuple] = set()
    for f in filters:
        for s in _resolve_filter_scopes(f):
            key = (s.field, s.subfield, s.topic, s.work_type)
            if key not in seen:
                seen.add(key)
                resolved_scopes.append(s)
    if not resolved_scopes:
        raise ValueError(
            f"work_filter {filters} resolved to zero scopes; nothing to fetch"
        )
    target_fields: set[str] = {s.field for s in resolved_scopes}
    target_scopes: set[ScopeKey] = set(seen)
    LOGGER.info(
        "snapshot_ingest: target fields = %s (%d scopes from %d filter(s)), snapshot_date = %s",
        sorted(target_fields),
        len(resolved_scopes),
        len(filters),
        snapshot_date,
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

    sink = SpillSink(spill_dir)
    bucketer = Bucketer(sink, flush_threshold=shard_row_flush_threshold, max_resident=max_resident_rows)

    tmp_shard_dir = CACHE_DIR / "snapshot_download_tmp"
    tmp_shard_dir.mkdir(parents=True, exist_ok=True)

    # Build a bytes-level prefilter so the JSON parser can skip over works
    # whose field isn't in our target set without decoding them. See
    # stream_shard's docstring for the exact anchor pattern.
    prefilter_patterns = [f'/fields/{fid}"'.encode() for fid in sorted(target_fields)]

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
                    for key, work in stream_shard(f, target_fields, prefilter_patterns=prefilter_patterns):
                        # Field-level stream filter is a cheap prefilter; the
                        # definitive check is against the resolved (field,
                        # subfield, topic, work_type) tuple. Drops "other"
                        # work types and off-target topics under the target
                        # field.
                        if key not in target_scopes:
                            continue
                        bucketer.insert(key, work)
                        matched += 1
            finally:
                tmp_path.unlink(missing_ok=True)
            LOGGER.info(
                "snapshot_ingest: shard %s done — %d works matched target scopes",
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

    state.finished_at = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)
    LOGGER.info("snapshot_ingest: done")
