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


def softmax_with_optional_empty(
    log_K: np.ndarray,
    log_K_empty: np.ndarray | None,
    tau: float,
) -> np.ndarray:
    """Numerically-stable softmax over log_K / tau (and optional log_K_empty / tau).

    Args:
        log_K: (N, K) array of log K̃_k(a) for kNN basins.
        log_K_empty: (N,) or None. If provided, append as an extra
            column.
        tau: temperature.

    Returns:
        (N, K) or (N, K+1) array of probabilities. Rows sum to 1.
    """
    if log_K_empty is not None:
        logits = np.concatenate(
            [log_K / tau, (log_K_empty / tau)[:, None]], axis=1
        )
    else:
        logits = log_K / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def mean_entropy(probs: np.ndarray) -> float:
    """Mean per-row entropy (nats)."""
    p = np.clip(probs, 1e-30, 1.0)
    return float((-p * np.log(p)).sum(axis=1).mean())


def tune_tau_for_perplexity(
    log_K: np.ndarray,
    log_K_empty: np.ndarray | None,
    P_target: float,
    log_tau_low: float = -3.0,
    log_tau_high: float = 1.0,
    tol: float = 0.01,
    max_iters: int = 30,
) -> float:
    """Bisect on log τ until mean per-row entropy is within tol of
    log(P_target). Returns the tuned τ.

    Higher τ → flatter softmax → higher entropy → higher perplexity.
    """
    target_H = float(np.log(P_target))
    lo, hi = float(log_tau_low), float(log_tau_high)
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        probs = softmax_with_optional_empty(log_K, log_K_empty, np.exp(mid))
        H = mean_entropy(probs)
        if abs(H - target_H) < tol:
            return float(np.exp(mid))
        if H < target_H:
            lo = mid    # need higher τ → higher entropy
        else:
            hi = mid    # need lower τ → lower entropy
    return float(np.exp(0.5 * (lo + hi)))


def compute_log_K_empty(
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    k_marginal: int,
) -> np.ndarray:
    """Marginal-density background kernel K̃_∅(a) = (1/N) · Σ_y κ(a, y),
    truncated to a's k_marginal nearest articles.

    The kernel is essentially zero beyond ~5σ; this matches the
    truncation strategy gap_soft_em uses internally for K_G/K_B.

    Returns log(K̃_∅) of shape (N_articles,) in float64.
    """
    N = article_emb.shape[0]
    k = min(k_marginal, N)
    index = build_index(FlatL2(), article_emb.astype(np.float32))
    D, I = index.search(article_emb.astype(np.float32), k=k)  # squared L2

    sigma_a = article_sigma.astype(np.float64)
    sigma_y = sigma_a[I]                       # (N, k)
    d2 = D.astype(np.float64)                  # (N, k)
    kern = np.exp(-d2 / (sigma_a[:, None] * sigma_y + 1e-30))  # (N, k)

    K_emp = kern.sum(axis=1) / float(N)        # (N,) — divide by N (not k)
    # Sanity-check the kernel tail: log the value at the K-th neighbor
    tail = kern[:, -1]
    median_tail = float(np.median(tail))
    if median_tail > 1e-4:
        print(f"WARNING: K_empty kNN tail not negligible "
              f"(median tail κ = {median_tail:.2e}); consider larger k_marginal")
    return np.log(K_emp + 1e-30)


def compute_basin_centroids(
    article_emb: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Weighted mean of article embeddings, unit-normalized per basin.

    Args:
        article_emb: (N_articles, d) float32, unit-normalized.
        M: (N_articles, N_basins) float32, soft membership weights.

    Returns:
        (N_basins, d) float32, unit-normalized.
    """
    weight_sum = M.sum(axis=0)                  # (N_basins,)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    centroids = (M.T @ article_emb) / weight_sum[:, None]  # (N_basins, d)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    return (centroids / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def knn_basin_lookup(
    article_emb: np.ndarray,
    centroids: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each article, return indices of its top-k nearest basin
    centroids by cosine (= dot product on the unit sphere).

    Brute matmul + argpartition; basin set is small (~hundreds).
    """
    sims = article_emb @ centroids.T          # (N_articles, N_basins)
    k = min(k, centroids.shape[0])
    if k >= sims.shape[1]:
        return np.argsort(-sims, axis=1)
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    sorted_part = part[rows, np.argsort(-sims[rows, part], axis=1)]
    return sorted_part.astype(np.int64)


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
