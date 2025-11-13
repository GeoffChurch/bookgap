# USAGE:
# python -m artifacts_scripts.clean_up_runs --delete-grace-period-hours 1

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from artifacts_scripts.common import find_run_directories

LOGGER = logging.getLogger(__name__)

DEFAULT_DELETE_GRACE_PERIOD_HOURS = 2.0


@dataclass(frozen=True)
class GracePeriodExpiredResult:
    expired: bool
    reason: str


def _latest_modification_timestamp(path: Path) -> float:
    try:
        latest = path.stat().st_mtime
    except FileNotFoundError:
        return float("-inf")

    for child in path.rglob("*"):
        try:
            child_mtime = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if child_mtime > latest:
            latest = child_mtime
    return latest


def _grace_period_is_expired(run_dir: Path, grace_period: timedelta) -> GracePeriodExpiredResult:
    if grace_period <= timedelta(0):
        return GracePeriodExpiredResult(True, "no grace period")

    latest_ts = _latest_modification_timestamp(run_dir)
    if latest_ts == float("-inf"):
        return GracePeriodExpiredResult(True, "no files found")

    last_modified = datetime.fromtimestamp(latest_ts)
    age = datetime.now() - last_modified
    if age < timedelta(0):
        age = timedelta(0)

    if age < grace_period:
        minutes = age.total_seconds() / 60
        grace_hours = grace_period.total_seconds() / 3600
        return GracePeriodExpiredResult(
            False,
            f"last modified {minutes:.1f} minutes ago, within {grace_hours:.2f} hour grace period",
        )

    return GracePeriodExpiredResult(True, "grace period expired")


def _delete_run_directory(run_dir: Path, grace_period: timedelta, dry_run: bool) -> str:
    grace_res = _grace_period_is_expired(run_dir, grace_period)
    if grace_res.expired:
        if dry_run:
            LOGGER.info("[DRY RUN] Would delete %s (%s)", run_dir, grace_res.reason)
        else:
            shutil.rmtree(run_dir)
    return grace_res.reason


def _cleanup_empty_date_dirs(outputs_dir: Path, dry_run: bool) -> None:
    if not outputs_dir.exists():
        return

    for date_dir in outputs_dir.iterdir():
        if not date_dir.is_dir():
            continue

        try:
            next(date_dir.iterdir())
        except StopIteration:
            if dry_run:
                LOGGER.info("[DRY RUN] Would delete empty date directory %s", date_dir)
            else:
                try:
                    date_dir.rmdir()
                    LOGGER.info("Deleted empty date directory %s", date_dir)
                except OSError as exc:
                    LOGGER.warning("Failed to delete empty directory %s: %s", date_dir, exc)


def clean_up_runs(
    outputs_dir: Path,
    delete_missing: bool,
    grace_period_hours: float,
    cleanup_empty_date_dirs: bool,
    dry_run: bool,
) -> None:
    if grace_period_hours < 0:
        LOGGER.error("delete-grace-period-hours must be non-negative")
        raise SystemExit(1)

    grace_period = timedelta(hours=grace_period_hours)

    for run_dir in find_run_directories(outputs_dir):
        csv_path = run_dir / "gaps_analysis_results.csv"
        if csv_path.exists():
            continue

        if delete_missing:
            reason = _delete_run_directory(run_dir, grace_period, dry_run)
        else:
            reason = "deletion disabled"
            LOGGER.info("Skipping deletion for %s (%s)", run_dir, reason)

        LOGGER.warning(
            "Missing gaps_analysis_results.csv in %s; %s", run_dir, reason
        )

    if cleanup_empty_date_dirs:
        _cleanup_empty_date_dirs(outputs_dir, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean up invalid or incomplete BookGap runs.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the root outputs directory containing run subdirectories.",
    )
    parser.add_argument(
        "--delete-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove run directories that are missing gaps_analysis_results.csv. Use "
            "--no-delete-missing to keep them."
        ),
    )
    parser.add_argument(
        "--delete-grace-period-hours",
        type=float,
        default=DEFAULT_DELETE_GRACE_PERIOD_HOURS,
        help=(
            "Only delete run directories missing results if their most recent modification "
            "is older than this many hours (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--cleanup-empty-date-dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove empty date directories under outputs after deletions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without removing anything.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    clean_up_runs(
        outputs_dir=args.outputs_dir,
        delete_missing=args.delete_missing,
        grace_period_hours=args.delete_grace_period_hours,
        cleanup_empty_date_dirs=args.cleanup_empty_date_dirs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
