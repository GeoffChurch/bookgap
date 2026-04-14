import json
from pathlib import Path
import pytest

from bookgap.snapshot_ingest import (
    SnapshotState,
    load_state,
    save_state,
    check_disk_or_abort,
    DiskFullError,
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
