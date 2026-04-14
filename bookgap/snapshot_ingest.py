"""Snapshot-based OpenAlex ingest (parallel to openalex_ingest.py).

Reads the OpenAlex S3 snapshot and writes per-scope parquet files using the
same pipeline as the API path. See docs/superpowers/specs/2026-04-14-snapshot-ingest-design.md.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

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
