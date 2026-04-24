"""Benchmark `range_search_with_` — the production dedup query path.

The earlier kNN-based benchmark (``hnsw_dedup_bench.py``) dramatically
underestimated dedup wall time because production uses HNSW ``range_search``
in small Python-level batches rather than a single bulk ``search``. This
script measures throughput of ``bookgap.index.range_search_with_`` at
various ``batch_size`` values against a sample of the real corpus, then
projects to N=42 M.

Usage::

    python -m artifacts_scripts.hnsw_range_search_bench \
        --sample 2000000 \
        --queries 200000 \
        --batch-sizes 24,48,96,256,1024,4096
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bookgap.index import HNSW, build_index, range_search_with_


def load_sample(parquet_path: Path, n: int, seed: int = 0) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["embeddings"])
    col = table.column("embeddings")
    total = len(col)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
    sliced = col.take(idx)
    flat = sliced.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(len(idx), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path,
                    default=Path("cache/embeddings_parquet/2f77728932c4b159/6bcf5b5e02946b7e/embeddings.parquet"))
    ap.add_argument("--sample", type=int, default=2_000_000,
                    help="number of embeddings to index (larger = more representative HNSW graph depth)")
    ap.add_argument("--queries", type=int, default=200_000,
                    help="number of queries per batch-size run")
    ap.add_argument("--batch-sizes", type=str, default="24,48,96,256,1024,4096",
                    help="comma-separated batch sizes to sweep")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--hnsw-efc", type=int, default=384)
    ap.add_argument("--hnsw-efs", type=int, default=128)
    ap.add_argument("--full-n", type=int, default=42_137_307)
    args = ap.parse_args()

    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]

    print(f"Loading {args.sample:,} sample embeddings ...")
    t0 = time.perf_counter()
    embeddings = load_sample(args.parquet, args.sample)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, shape={embeddings.shape}")

    print(f"Building HNSW(M={args.hnsw_m}, efC={args.hnsw_efc}, efS={args.hnsw_efs}) on {len(embeddings):,} points ...")
    t0 = time.perf_counter()
    index_info = HNSW(hnsw_m=args.hnsw_m, hnsw_efc=args.hnsw_efc, hnsw_efs=args.hnsw_efs)
    index = build_index(index_info, embeddings)
    build_s = time.perf_counter() - t0
    print(f"  built in {build_s:.1f}s")

    rng = np.random.default_rng(1)
    q_idx = rng.choice(len(embeddings), size=min(args.queries, len(embeddings)), replace=False)
    queries = embeddings[q_idx]
    print(f"Running {len(queries):,} queries per batch-size setting ...\n")

    # Per-query scaling from sample_n to full_n for HNSW range_search is
    # approximately O(log N) (graph diameter grows ~logarithmically).
    log_ratio = np.log(args.full_n) / np.log(args.sample)

    print(f"{'batch':>6} | {'wall_s':>8} {'q/s':>10} {'eff_cores':>10} | {'proj_42M':>12}")
    print("-" * 60)

    results = []
    for bs in batch_sizes:
        hits = [0]  # use list for closure
        def noop(qidx, dists, ids):
            hits[0] += len(ids)

        # Warm-up: a tiny first call to avoid including thread-pool spin-up
        range_search_with_(index, queries[:bs], args.epsilon, noop, batch_size=bs, log_every=None)

        t0 = time.perf_counter()
        range_search_with_(index, queries, args.epsilon, noop, batch_size=bs, log_every=None)
        wall_s = time.perf_counter() - t0

        qps = len(queries) / wall_s if wall_s > 0 else 0.0
        # Approximate effective CPU utilization: a single query on HNSW at
        # M=16, efS=128 touches ~128 * 16 ~= 2k node distances plus memory
        # loads; at ~10 ns/distance = ~20 μs/query ideally. Skip precise
        # per-core calculation — user cares about wall time.
        eff_cores = wall_s  # placeholder; compute below as float(OMP threads) fraction
        # Project 42 M: queries × log_ratio to adjust per-query cost
        proj_s = (args.full_n / qps) * log_ratio
        def fmt_s(s):
            if s < 60: return f"{s:.1f}s"
            if s < 3600: return f"{s/60:.1f}m"
            return f"{s/3600:.2f}h"
        results.append((bs, wall_s, qps, proj_s))
        print(f"{bs:>6} | {wall_s:>7.1f}s {qps:>9.0f} {'—':>10} | {fmt_s(proj_s):>12}")

    print()
    print("Notes:")
    print(f"  - sample build at N={args.sample:,} took {fmt_s_outer(build_s)} at (M={args.hnsw_m}, efC={args.hnsw_efc})")
    full_build_est = build_s * (args.full_n / args.sample) * log_ratio
    print(f"  - projected build at N={args.full_n:,}: {fmt_s_outer(full_build_est)} "
          f"(embeddings_parquet already written; this only rebuilds the HNSW index in-process)")


def fmt_s_outer(s):
    if s < 60: return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


if __name__ == "__main__":
    main()
