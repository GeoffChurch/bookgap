"""Backfill primary_topic_id / primary_subfield_id / topic_label in the
per-scope parquet cache after fixing the snapshot-ingest round-trip bug.

Background: ``works_to_table`` only handled the nested ``primary_topic``
dict shape from the first-ingest path. snapshot_ingest's bucketer
writes a spill parquet and then reads it back via ``to_pylist()`` —
which produces flat keys (``primary_topic_id``, ``primary_subfield_id``,
``topic_label``). Calling ``works_to_table`` on the flat shape wrote
None to all three columns. Fixed in ``openalex_ingest.works_to_table``
to fall back to the flat keys; this script backfills existing parquets
without re-ingesting from S3 since the path itself encodes the topic.

For each parquet at ``cache/works_parquet/<field>/<subfield>/<topic>/<work_type>.parquet``:
  - primary_topic_id    -> "https://openalex.org/<topic>"
  - primary_subfield_id -> "https://openalex.org/subfields/<subfield>"
  - topic_label         -> "<topic>"

Atomic per-file: writes to a .tmp suffix and renames on success.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bookgap.util import CACHE_DIR


def backfill_one(parquet_path: Path, force: bool = False, dry_run: bool = False) -> tuple[bool, str]:
    """Backfill columns for one parquet. Returns (changed, reason)."""
    parts = parquet_path.relative_to(CACHE_DIR / "works_parquet").parts
    if len(parts) != 4:
        return False, "skip: not a scope-level parquet"
    field, subfield, topic, _wtype_with_suffix = parts

    table = pq.read_table(parquet_path)
    n = table.num_rows
    if n == 0:
        return False, "skip: empty"

    # Sanity-check: are these columns actually all null? If they're already
    # populated, we shouldn't overwrite — would corrupt anything that the
    # original ingest got right.
    pt_null_count = table.column("primary_topic_id").null_count
    sf_null_count = table.column("primary_subfield_id").null_count
    tl_null_count = table.column("topic_label").null_count
    if not force and pt_null_count == 0 and sf_null_count == 0 and tl_null_count == 0:
        return False, "skip: already populated"

    new_topic_id = f"https://openalex.org/{topic}"
    new_subfield_id = f"https://openalex.org/subfields/{subfield}"
    new_topic_label = topic

    new_table = (
        table
        .set_column(table.schema.get_field_index("primary_topic_id"),
                    "primary_topic_id",
                    pa.array([new_topic_id] * n, type=pa.string()))
    )
    new_table = new_table.set_column(
        new_table.schema.get_field_index("primary_subfield_id"),
        "primary_subfield_id",
        pa.array([new_subfield_id] * n, type=pa.string()),
    )
    new_table = new_table.set_column(
        new_table.schema.get_field_index("topic_label"),
        "topic_label",
        pa.array([new_topic_label] * n, type=pa.string()),
    )

    if dry_run:
        return True, f"would update {n} rows ({pt_null_count}/{sf_null_count}/{tl_null_count} null)"

    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(new_table, tmp, compression="zstd")
    os.replace(tmp, parquet_path)
    return True, f"updated {n} rows"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change but don't write")
    parser.add_argument("--force", action="store_true",
                        help="overwrite even non-null values (for re-runs)")
    parser.add_argument("--root", default=None,
                        help="parquet root (default: CACHE_DIR/works_parquet)")
    args = parser.parse_args()

    root = Path(args.root) if args.root else (CACHE_DIR / "works_parquet")
    if not root.exists():
        print(f"No parquet root at {root}")
        return

    paths = sorted(root.rglob("*.parquet"))
    print(f"Found {len(paths)} parquet files under {root}")
    n_changed = 0
    n_skipped = 0
    n_empty = 0
    n_already = 0
    for i, p in enumerate(paths, 1):
        if i % 500 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] processed; {n_changed} changed, "
                  f"{n_already} already populated, {n_empty} empty")
        try:
            changed, reason = backfill_one(p, force=args.force, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  ERROR on {p}: {exc}")
            continue
        if changed:
            n_changed += 1
        elif "empty" in reason:
            n_empty += 1
        elif "already populated" in reason:
            n_already += 1
        else:
            n_skipped += 1

    print()
    print(f"Summary:")
    print(f"  Changed:           {n_changed}")
    print(f"  Already populated: {n_already}")
    print(f"  Empty:             {n_empty}")
    print(f"  Other skips:       {n_skipped}")
    if args.dry_run:
        print(f"  (dry-run; no writes performed)")


if __name__ == "__main__":
    main()
