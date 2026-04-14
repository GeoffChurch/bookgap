"""Snapshot-based OpenAlex ingest (parallel to openalex_ingest.py).

Reads the OpenAlex S3 snapshot and writes per-scope parquet files using the
same pipeline as the API path. See docs/superpowers/specs/2026-04-14-snapshot-ingest-design.md.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, IO, Iterator, Optional

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from bookgap.util import LOGGER


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
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


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
) -> Iterator[tuple[ScopeKey, dict]]:
    """Yield (scope_key, work) tuples from a gzipped JSONL shard.

    Works without a resolvable primary_topic or malformed JSON lines are
    silently skipped. If target_fields is not None, only works whose
    field_id is in that set are yielded.
    """
    with gzip.GzipFile(fileobj=fileobj, mode="rb") as gz:
        for line in gz:
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
