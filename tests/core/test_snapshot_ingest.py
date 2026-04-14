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
