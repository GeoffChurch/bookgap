"""Multi-cluster softmax post-hoc reassignment.

Phase 1 of the multicluster-softmax-membership backlog item. Reads an
enriched soft-EM CSV (the per-seed pairwise-posterior basin set) and
recomputes membership using a global softmax over kNN basin centroids,
with optional ∅ background (variant B) and optional iteration to
convergence. Emits per-config artifacts to --output-dir.

Spec: docs/superpowers/specs/2026-05-08-multicluster-softmax-phase1-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bookgap.index import build_index, FlatL2


def compute_self_tuning_sigma(
    emb: np.ndarray,
    k_sigma: int = 7,
) -> np.ndarray:
    """σ_i = sqrt(median squared distance to k_sigma nearest neighbors
    of i, excluding i itself).

    Matches gap_soft_em.py's formula. Returns shape (N,) float32.
    """
    index = build_index(FlatL2(), emb)
    D, _ = index.search(emb, k=k_sigma + 1)   # +1 for self
    # D[:, 0] is self (distance 0); take 1 .. k_sigma+1
    sigma_sq = np.median(D[:, 1:k_sigma + 1], axis=1)
    sigma_sq = np.maximum(sigma_sq, 1e-8).astype(np.float32)
    return np.sqrt(sigma_sq).astype(np.float32)


def load_enriched_csv(
    csv_path: Path,
    is_article: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Read an enriched soft-EM CSV. Build (N_articles, N_basins) M0
    dense matrix of original article-side soft-EM weights.

    Returns (M0, basin_meta) where basin_meta[k] holds metadata
    needed downstream (rank, n_books, book_anchored, etc.) for the
    k-th basin.
    """
    article_global_idx = np.where(is_article)[0]
    article_local_of_global = np.full(len(is_article), -1, dtype=np.int64)
    article_local_of_global[article_global_idx] = np.arange(len(article_global_idx))

    rows: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    N_articles = len(article_global_idx)
    N_basins = len(rows)
    M0 = np.zeros((N_articles, N_basins), dtype=np.float32)
    basin_meta: list[dict] = []

    for k, row in enumerate(rows):
        weights = json.loads(row["member_weights_json"])
        for w in weights:
            if w["type"] != "article":
                continue
            local = article_local_of_global[int(w["idx"])]
            if local >= 0:
                M0[local, k] = float(w["weight"])
        basin_meta.append({
            "rank": int(row["rank"]),
            "first_seed": int(row["seed_article"]),   # CSV header is seed_article, not first_seed
            "n_books": int(row.get("n_books", 0)),
            "n_articles": int(row.get("n_articles", 0)),
            "book_anchored": int(row.get("book_anchored", 0)),
        })

    return M0, basin_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True,
                   help="Topic scope path, e.g. 17/1702/T10181")
    p.add_argument("--enriched-csv", required=True,
                   help="Enriched soft-EM CSV path (input).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--perplexity", type=float, required=True)
    p.add_argument("--iter-mode", choices=["oneshot", "iterated"], required=True)
    p.add_argument("--k-knn", type=int, default=10,
                   help="Number of basin neighbors per article in the softmax.")
    p.add_argument("--k-sigma", type=int, default=7,
                   help="Self-tuning σ: median distance to k-th nearest neighbor.")
    p.add_argument("--k-marginal", type=int, default=200,
                   help="Article-kNN truncation for variant B's K̃_∅.")
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--conv-tol", type=float, default=1e-3,
                   help="L_inf convergence tolerance on membership matrix.")
    args = p.parse_args()
    raise NotImplementedError("driver implemented in later tasks")


if __name__ == "__main__":
    main()
