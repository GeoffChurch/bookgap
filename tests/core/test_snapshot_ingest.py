import gzip
import io
import json
from pathlib import Path

import pytest

from bookgap.snapshot_ingest import (
    SnapshotState,
    load_state,
    save_state,
    check_disk_or_abort,
    DiskFullError,
    make_s3_client,
    list_shards,
)


def test_load_state_missing_returns_empty(tmp_path):
    state = load_state(tmp_path / "missing.json", snapshot_date="2026-04-01")
    assert state.snapshot_date == "2026-04-01"
    assert state.shards_completed == []


def test_load_state_wipes_on_date_change(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"snapshot_date": "2026-03-01", "shards_completed": ["a", "b"]}))
    state = load_state(path, snapshot_date="2026-04-01")
    assert state.shards_completed == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = SnapshotState(snapshot_date="2026-04-01", shards_completed=["shard1.gz"])
    save_state(path, state)
    loaded = load_state(path, snapshot_date="2026-04-01")
    assert loaded.shards_completed == ["shard1.gz"]


def test_disk_guard_aborts_at_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 96
        free = 4
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    with pytest.raises(DiskFullError):
        check_disk_or_abort(tmp_path, abort_pct=95.0)


def test_disk_guard_aborts_exactly_at_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 95
        free = 5
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    with pytest.raises(DiskFullError):
        check_disk_or_abort(tmp_path, abort_pct=95.0)


def test_disk_guard_passes_under_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 50
        free = 50
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    check_disk_or_abort(tmp_path, abort_pct=95.0)  # no raise


# ---------------------------------------------------------------------------
# Helpers for S3 tests
# ---------------------------------------------------------------------------

class FakePaginator:
    """Mimics boto3 paginator for list_objects_v2."""

    def __init__(self, pages):
        # pages: list of dicts, each with optional "Contents" key
        self._pages = pages

    def paginate(self, Bucket, Prefix):
        return iter(self._pages)


class FakeClient:
    """Minimal boto3 client stub."""

    def __init__(self, pages):
        self._paginator = FakePaginator(pages)

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return self._paginator


# ---------------------------------------------------------------------------
# S3 client tests
# ---------------------------------------------------------------------------

def test_make_s3_client_returns_unsigned_client():
    """make_s3_client() must produce a boto3 client with UNSIGNED signature."""
    from botocore import UNSIGNED
    client = make_s3_client()
    assert client.meta.config.signature_version == UNSIGNED


# ---------------------------------------------------------------------------
# list_shards tests
# ---------------------------------------------------------------------------

def test_list_shards_filters_newer_than_snapshot():
    """Keys whose updated_date partition is after snapshot_date must be excluded."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-03-15/part_000.gz"},  # before -> keep
            {"Key": "data/works/updated_date=2026-04-02/part_001.gz"},  # after -> drop
            {"Key": "data/works/updated_date=2026-04-01/part_002.gz"},  # equal -> keep
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == [
        "data/works/updated_date=2026-03-15/part_000.gz",
        "data/works/updated_date=2026-04-01/part_002.gz",
    ]


def test_list_shards_filters_non_gz_objects():
    """Non-.gz objects (e.g. manifest files) must be excluded."""
    pages = [
        {"Contents": [
            {"Key": "data/works/manifest"},
            {"Key": "data/works/updated_date=2026-03-01/part_000.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == ["data/works/updated_date=2026-03-01/part_000.gz"]


def test_list_shards_returns_sorted_order():
    """Results must be sorted regardless of page order."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-03-20/part_002.gz"},
            {"Key": "data/works/updated_date=2026-03-01/part_000.gz"},
            {"Key": "data/works/updated_date=2026-03-10/part_001.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == sorted(result)
    assert len(result) == 3


def test_list_shards_empty_when_no_match():
    """Return an empty list when all keys are newer than snapshot_date."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-05-01/part_000.gz"},
            {"Key": "data/works/updated_date=2026-06-15/part_001.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == []


def test_list_shards_handles_empty_pages():
    """Pages with no Contents key must be silently skipped."""
    pages = [
        {},
        {"Contents": [{"Key": "data/works/updated_date=2026-03-01/part_000.gz"}]},
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == ["data/works/updated_date=2026-03-01/part_000.gz"]


def test_partition_date_from_key_rejects_empty_value():
    from bookgap.snapshot_ingest import _partition_date_from_key
    assert _partition_date_from_key("data/works/updated_date=/part.gz") is None


def test_partition_date_from_key_rejects_time_suffix():
    from bookgap.snapshot_ingest import _partition_date_from_key
    # Anything with a time component is not an exact YYYY-MM-DD date partition.
    assert _partition_date_from_key("data/works/updated_date=2026-04-01T12:00:00/part.gz") is None


def test_list_shards_drops_keys_with_malformed_partition():
    # A key with an empty or time-suffixed partition should never make it
    # into the final shard list, regardless of snapshot_date.
    client = FakeClient(pages=[{
        "Contents": [
            {"Key": "data/works/updated_date=/part.gz"},
            {"Key": "data/works/updated_date=2026-04-01T12:00/part.gz"},
            {"Key": "data/works/updated_date=2026-04-01/part.gz"},
        ]
    }])
    shards = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert shards == ["data/works/updated_date=2026-04-01/part.gz"]


# ---------------------------------------------------------------------------
# Helpers for stream_shard / Bucketer tests
# ---------------------------------------------------------------------------


def _make_gzipped_jsonl(works: list[dict]) -> io.BytesIO:
    """Build an in-memory gzipped JSONL blob from a list of work dicts."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for work in works:
            gz.write((json.dumps(work) + "\n").encode())
    buf.seek(0)
    return buf


def _work(field_id: str, subfield_id: str, topic_id: str, wtype: str = "article") -> dict:
    """Convenience builder for a minimal OpenAlex work dict."""
    return {
        "id": f"https://openalex.org/W{topic_id}999",
        "type": wtype,
        "primary_topic": {
            "id": f"https://openalex.org/T{topic_id}",
            "field": {"id": f"https://openalex.org/fields/{field_id}"},
            "subfield": {"id": f"https://openalex.org/subfields/{subfield_id}"},
        },
    }


class RecordingSink:
    def __init__(self):
        self.calls = []  # list of (scope_key, list[dict])

    def __call__(self, scope_key, rows):
        self.calls.append((scope_key, list(rows)))


# ---------------------------------------------------------------------------
# stream_shard tests
# ---------------------------------------------------------------------------

def test_stream_shard_yields_scope_keyed_works():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        _work("31", "2100", "10805"),          # field 31 -> match
        _work("31", "2100", "10806"),          # field 31 -> match
        _work("99", "9900", "99001"),          # field 99 -> no match
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields={"31"}))

    assert len(results) == 2
    for key, work in results:
        field, subfield, topic, wtype = key
        assert field == "31"
        assert subfield == "2100"
        assert isinstance(topic, str) and topic
        assert wtype == "article"


def test_stream_shard_drops_works_without_primary_topic():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        {"id": "https://openalex.org/W1", "type": "article"},          # no primary_topic
        _work("31", "2100", "10805"),
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields=None))

    assert len(results) == 1
    assert results[0][0][0] == "31"


def test_stream_shard_drops_malformed_json_lines():
    from bookgap.snapshot_ingest import stream_shard

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"not json at all\n")
        gz.write((json.dumps(_work("31", "2100", "10805")) + "\n").encode())
    buf.seek(0)

    results = list(stream_shard(buf, target_fields=None))
    assert len(results) == 1


def test_stream_shard_no_target_fields_yields_all():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        _work("31", "2100", "10805"),
        _work("55", "3300", "20001"),
        _work("77", "4400", "30002", wtype="book"),
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields=None))

    assert len(results) == 3
    field_ids = {key[0] for key, _ in results}
    assert field_ids == {"31", "55", "77"}


# ---------------------------------------------------------------------------
# Bucketer tests
# ---------------------------------------------------------------------------

def test_bucketer_flushes_when_threshold_reached():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    threshold = 5
    bucketer = Bucketer(sink=sink, flush_threshold=threshold, max_resident=10_000)

    key = ("31", "2100", "10805", "article")
    work = _work("31", "2100", "10805")

    for _ in range(threshold):
        bucketer.insert(key, work)

    # Exactly at threshold -> one flush with all threshold rows
    assert len(sink.calls) == 1
    assert len(sink.calls[0][1]) == threshold
    assert sink.calls[0][0] == key


def test_bucketer_evicts_largest_bucket_when_resident_cap_hit():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    # max_resident = 5; threshold high so per-bucket flush won't fire
    bucketer = Bucketer(sink=sink, flush_threshold=1_000, max_resident=5)

    key_a = ("31", "2100", "10805", "article")
    key_b = ("55", "3300", "20001", "article")
    work_a = _work("31", "2100", "10805")
    work_b = _work("55", "3300", "20001")

    # Insert 3 into A, 2 into B (total = 5, exactly at cap — one more triggers eviction)
    for _ in range(3):
        bucketer.insert(key_a, work_a)
    for _ in range(2):
        bucketer.insert(key_b, work_b)

    # No flush yet (resident == max_resident, not exceeding)
    assert len(sink.calls) == 0

    # One more into B brings total to 6 > max_resident -> largest (A, size 3) flushed
    bucketer.insert(key_b, work_b)
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == key_a
    assert len(sink.calls[0][1]) == 3


def test_bucketer_flush_all_empties_all_buckets():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    bucketer = Bucketer(sink=sink, flush_threshold=1_000, max_resident=1_000)

    key_a = ("31", "2100", "10805", "article")
    key_b = ("55", "3300", "20001", "book")
    work_a = _work("31", "2100", "10805")
    work_b = _work("55", "3300", "20001", wtype="book")

    bucketer.insert(key_a, work_a)
    bucketer.insert(key_a, work_a)
    bucketer.insert(key_b, work_b)

    bucketer.flush_all()

    # Both buckets flushed; sink called twice
    assert len(sink.calls) == 2
    flushed_keys = {call[0] for call in sink.calls}
    assert flushed_keys == {key_a, key_b}

    # Totals match what was inserted
    total_rows = sum(len(call[1]) for call in sink.calls)
    assert total_rows == 3

    # Subsequent flush_all is a no-op
    sink.calls.clear()
    bucketer.flush_all()
    assert len(sink.calls) == 0
