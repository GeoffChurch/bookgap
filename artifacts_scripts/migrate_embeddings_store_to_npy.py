"""One-shot migration: embeddings stores from zstd Parquet to memory-mappable .npy.

For every ``cache/embeddings_parquet/<corpus_fp>/<embedder_fp>/embeddings.parquet``
not named by ``--skip``, writes ``cache/embeddings_npy/<corpus_fp>/<embedder_fp>/
embeddings.npy`` (the path ``bookgap.cloud._embeddings_store_path`` now reads),
streaming one row group at a time so even the 64 GB domain-scale store never
sits in the heap. Each row group's ``row_id`` must equal its row positions, the
layout the .npy store assumes. The written array is re-read against every row
group and must match byte for byte before it is moved into place.

Parquet files are left where they are; delete them once satisfied.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.migrate_embeddings_store_to_npy \\
        --skip 469a666eca9848a2/00c71307b5d1398a 576eaea8718fce85/00c71307b5d1398a
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC_ROOT = Path("cache/embeddings_parquet")
DST_ROOT = Path("cache/embeddings_npy")


def _row_group_array(pf: pq.ParquetFile, rg: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    table = pf.read_row_group(rg, columns=["row_id", "embeddings"])
    row_ids = table.column("row_id").to_numpy()
    flat = table.column("embeddings").combine_chunks().values.to_numpy(zero_copy_only=False)
    return row_ids, np.asarray(flat, dtype=np.float32).reshape(len(row_ids), d)


def migrate(src: Path, dst: Path) -> None:
    pf = pq.ParquetFile(src)
    d = int((pf.schema_arrow.metadata or {})[b"bookgap:embedding_dim"].decode())
    n = pf.metadata.num_rows
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, d))
    offset = 0
    for rg in range(pf.num_row_groups):
        row_ids, arr = _row_group_array(pf, rg, d)
        if not np.array_equal(row_ids, np.arange(offset, offset + len(row_ids))):
            raise ValueError(f"{src}: row group {rg} row_id is not its row positions")
        out[offset:offset + len(arr)] = arr
        offset += len(arr)
    out.flush()
    del out
    check = np.load(tmp, mmap_mode="r")
    offset = 0
    for rg in range(pf.num_row_groups):
        _, arr = _row_group_array(pf, rg, d)
        if not np.array_equal(check[offset:offset + len(arr)], arr):
            raise ValueError(f"{src}: row group {rg} differs after writing {tmp}")
        offset += len(arr)
    del check
    os.replace(tmp, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], metavar="CORPUS_FP/EMBEDDER_FP",
                    help="stores not to migrate (e.g. ones whose vectors are known to be wrong)")
    args = ap.parse_args()
    skip = set(args.skip)

    stores = sorted(SRC_ROOT.glob("*/*/embeddings.parquet"))
    unknown = skip - {f"{p.parent.parent.name}/{p.parent.name}" for p in stores}
    if unknown:
        print(f"error: --skip names no store: {sorted(unknown)}", file=sys.stderr)
        return 2
    for src in stores:
        key = f"{src.parent.parent.name}/{src.parent.name}"
        dst = DST_ROOT / key / "embeddings.npy"
        if key in skip:
            print(f"skip     {key}")
            continue
        if dst.exists():
            print(f"exists   {key}")
            continue
        t = time.time()
        migrate(src, dst)
        print(f"migrated {key}  {dst.stat().st_size / 1e9:6.2f} GB  {time.time() - t:6.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
