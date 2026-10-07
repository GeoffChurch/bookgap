"""HNSW build + k=2 NN benchmark for SqDistDeduplicator on the full corpus.

Loads N sample embeddings from the corpus embeddings store, sweeps
(M, efC, efS) configs, and reports build + query wall time plus recall
vs FAISS IndexFlatL2 ground truth. Times are then linearly projected to
the full 42 M corpus to estimate dedup wall time at each setting.

Recall is reported as both top-1 NN recall and "dedup mismatch" rate —
the fraction of points whose duplicate-vs-not classification (true
nearest sqdist < min_sqdist) flips when ANN replaces brute force, since
that's what actually matters for SqDistDeduplicator behaviour.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import faiss

from bookgap.cloud import read_embeddings_rows, read_embeddings_store


def load_sample(store_path: Path, n: int, seed: int = 0) -> np.ndarray:
    total = read_embeddings_store(store_path).shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
    return read_embeddings_rows(store_path, idx)


def bench_one(
    embeddings: np.ndarray,
    M: int,
    efC: int,
    efS: int,
    queries: np.ndarray,
    truth_idx: np.ndarray,
    truth_dist: np.ndarray,
    min_sqdist: float,
) -> dict:
    n, dim = embeddings.shape
    n_q = queries.shape[0]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS

    t0 = time.perf_counter()
    index.add(embeddings)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    D, I = index.search(queries, 2)
    query_s = time.perf_counter() - t0

    # Drop self-matches: when the query is in the index, the nearest is
    # itself (dist 0). The "real" nearest is the second column.
    ann_idx = I[:, 1]
    ann_dist = D[:, 1]

    recall_at_1 = float(np.mean(ann_idx == truth_idx))

    truth_dup = truth_dist < min_sqdist
    ann_dup = ann_dist < min_sqdist
    dedup_disagreement = float(np.mean(truth_dup != ann_dup))
    # Asymmetric: how often did we MISS a real duplicate (false negative)?
    missed_dup = float(np.mean(truth_dup & ~ann_dup))

    return {
        "M": M, "efC": efC, "efS": efS,
        "build_s": build_s, "query_s": query_s,
        "recall_at_1": recall_at_1,
        "dedup_disagreement": dedup_disagreement,
        "missed_duplicates": missed_dup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path,
                    default=Path("cache/embeddings_npy/2f77728932c4b159/6bcf5b5e02946b7e/embeddings.npy"))
    ap.add_argument("--sample", type=int, default=200_000,
                    help="number of embeddings to use for indexing (and queries)")
    ap.add_argument("--query-sample", type=int, default=5_000,
                    help="subset of points to query against the sample (recall is per-query)")
    ap.add_argument("--full-n", type=int, default=42_137_307,
                    help="total corpus size for time projection")
    ap.add_argument("--min-sqdist", type=float, default=0.01,
                    help="SqDistDeduplicator threshold")
    args = ap.parse_args()

    print(f"Loading {args.sample:,} sample embeddings from {args.store} ...")
    t0 = time.perf_counter()
    embeddings = load_sample(args.store, args.sample)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, shape={embeddings.shape}, dtype={embeddings.dtype}")

    rng = np.random.default_rng(1)
    q_idx = np.sort(rng.choice(embeddings.shape[0], size=min(args.query_sample, embeddings.shape[0]), replace=False))
    queries = embeddings[q_idx]

    print(f"Computing brute-force ground truth (k=2) for {len(queries):,} queries against {embeddings.shape[0]:,} points ...")
    t0 = time.perf_counter()
    truth_index = faiss.IndexFlatL2(embeddings.shape[1])
    truth_index.add(embeddings)
    truth_D, truth_I = truth_index.search(queries, 2)
    truth_idx = truth_I[:, 1]
    truth_dist = truth_D[:, 1]
    print(f"  ground truth in {time.perf_counter() - t0:.1f}s")
    print(f"  true near-duplicate rate (sqdist < {args.min_sqdist}): "
          f"{float(np.mean(truth_dist < args.min_sqdist)) * 100:.2f}%")
    print()

    configs = [
        (16, 512, 512),  # baseline
        (16, 384, 128),
        (16, 200, 64),   # current pick
    ]

    print(f"{'M':>3} {'efC':>4} {'efS':>4} | "
          f"{'build_s':>8} {'query_s':>8} | "
          f"{'recall@1':>9} {'mismatch':>9} {'missed_dup':>10} | "
          f"{'proj_build':>11} {'proj_query':>11} {'proj_total':>11}")
    print("-" * 110)

    sample_n = embeddings.shape[0]
    full_n = args.full_n
    n_ratio = full_n / sample_n
    log_ratio = np.log(full_n) / np.log(sample_n)
    # Build cost ~ O(N * log(N) * M * efC). Holding (M, efC) fixed,
    # build_full / build_sample ~ (full_n / sample_n) * (log full / log sample).
    build_factor = n_ratio * log_ratio
    # Query cost per-point ~ O(efS * log(N) * M); total queries scale
    # linearly with N, so total query factor is the same as build_factor.
    query_factor = build_factor

    for M, efC, efS in configs:
        r = bench_one(embeddings, M, efC, efS, queries, truth_idx, truth_dist, args.min_sqdist)
        proj_build = r["build_s"] * build_factor
        # Query benchmark used a subset; scale up to ALL points being queried
        # at full N: factor = (full_n / query_sample) * log_ratio
        per_query_at_full = (r["query_s"] / len(queries)) * log_ratio
        proj_query = per_query_at_full * full_n
        proj_total = proj_build + proj_query

        def fmt_s(s):
            if s < 60: return f"{s:.1f}s"
            if s < 3600: return f"{s/60:.1f}m"
            return f"{s/3600:.2f}h"

        print(f"{M:>3} {efC:>4} {efS:>4} | "
              f"{r['build_s']:>7.2f}s {r['query_s']:>7.2f}s | "
              f"{r['recall_at_1']*100:>8.2f}% {r['dedup_disagreement']*100:>8.3f}% "
              f"{r['missed_duplicates']*100:>9.3f}% | "
              f"{fmt_s(proj_build):>11} {fmt_s(proj_query):>11} {fmt_s(proj_total):>11}")


if __name__ == "__main__":
    main()
