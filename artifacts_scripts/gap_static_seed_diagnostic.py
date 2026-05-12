"""Static-seed cohesion diagnostic.

Resolves five algorithmic decisions for a proposed static-seed gap-
detection variant before committing to a full implementation. See
``docs/superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-design.md``
for context.

Pipeline diagnosed (per spec):

  per-seed (k*, weight vector) -> Jaccard confluence merge ->
  joint normalize -> rank by effective_size + coverage readout.

This script answers:
  1. LLR-vs-k or z-score-vs-k for the surprise curve?
  2. Self-tuning Gaussian / linear-rank / linear-distance decay?
  3. Static or one-step recentering?
  4. Is joint normalization sufficient as a redundancy handler?
  5. Jaccard or containment for the pre-norm confluence merge?

Usage::

    conda run -p .conda python -m artifacts_scripts.gap_static_seed_diagnostic \
        --scope 17/1702/T11303 \
        --out cache/static_seed_diagnostic/T11303
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

# Module-level constants (no magic numbers in section code).
K_SIGMA = 7
K_QUERY = 1500
K_GRID = (5, 10, 20, 40, 80, 160, 320, 640, 1280)


# ============================================================
# Math primitives
# ============================================================

def generalized_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized Jaccard on non-negative weight vectors.

    J(a, b) = sum(min(a, b)) / sum(max(a, b)).
    Returns 0.0 when both vectors are all-zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.minimum(a, b).sum()
    denom = np.maximum(a, b).sum()
    if denom <= 0:
        return 0.0
    return float(num / denom)


def generalized_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized Szymkiewicz-Simpson on non-negative weight vectors.

    C(a, b) = sum(min(a, b)) / min(sum(a), sum(b)).
    Returns 0.0 when either vector sums to zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.minimum(a, b).sum()
    denom = min(a.sum(), b.sum())
    if denom <= 0:
        return 0.0
    return float(num / denom)


def joint_normalize(M: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """Normalize each row of M to sum to 1; return (M_normalized, column_sums).

    Rows that sum to zero are left unchanged (no division). Column sums of
    the normalized matrix are returned as effective_size per cluster.
    """
    from scipy.sparse import diags
    M = M.tocsr()
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    inv = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    M_norm = (diags(inv) @ M).tocsr()
    eff = np.asarray(M_norm.sum(axis=0)).flatten()
    return M_norm, eff


def kernel_mass_matrix(
    D_art_NN: np.ndarray,
    I_art_NN_global: np.ndarray,
    sigma: np.ndarray,
    k_grid: Sequence[int],
    seed_global_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cumulative kernel mass M_k(seed) for each seed at each k in k_grid.

    Arguments:
      D_art_NN: (N_rows, K_query) squared L2 distances; row i corresponds
        to the i-th row of seed_global_idx (or i-th seed if seed_global_idx
        is None and rows correspond directly to sigma's indexing).
      I_art_NN_global: (N_rows, K_query) global indices of neighbors per seed.
      sigma: (N_total,) self-tuning bandwidths over all works (article+book).
      k_grid: sequence of k values; all values <= D_art_NN.shape[1].
      seed_global_idx: (N_rows,) global index of each seed row. If None,
        sigma is assumed to be already aligned to D_art_NN's rows.

    Returns (N_rows, len(k_grid)) cumulative kernel mass.
    """
    if seed_global_idx is None:
        sigma_seed = sigma[: D_art_NN.shape[0], None]
    else:
        sigma_seed = sigma[seed_global_idx][:, None]
    sigma_n = sigma[I_art_NN_global]
    W = np.exp(-D_art_NN / (sigma_seed * sigma_n))
    cum = np.cumsum(W, axis=1)
    cols = np.array([k - 1 for k in k_grid], dtype=np.int64)
    return cum[:, cols]


# ============================================================
# Setup
# ============================================================

def setup(scope: str, k_sigma: int = K_SIGMA, k_query: int = K_QUERY) -> dict:
    """Build the shared kernel/index state used by all five sections.

    Returns a dict with keys:
      emb, ids, types, is_article, is_book, article_global_idx, book_global_idx,
      sigma, D_art_NN_all, I_art_NN_all_global,
      full_index, article_index.

    All seeds (every article + every book) get a (k_query,)-row of nearest-
    article neighbors — these are the candidate cluster members under the
    static-seed variant's "articles-only" rule.
    """
    # Local imports keep the script importable without these (for unit tests).
    from artifacts_scripts.hdbscan_gap_probe import load_scope
    from bookgap.index import build_index, FlatL2

    emb, ids, types, _t, topic_ids, _refs = load_scope(scope, with_refs=False)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    is_article = types_arr == "article"
    book_global_idx = np.where(is_book)[0]
    article_global_idx = np.where(is_article)[0]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)

    full_index = build_index(FlatL2(), emb)
    article_emb = np.ascontiguousarray(emb[article_global_idx])
    article_index = build_index(FlatL2(), article_emb)

    D_sigma, _ = full_index.search(emb, k=k_sigma + 1)
    sigma_sq = np.median(D_sigma[:, 1: k_sigma + 1], axis=1)
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-8)).astype(np.float32)

    # For every work (book + article), find its top-k_query nearest articles.
    # article_index returns local indices into article_global_idx.
    D_art_NN_all, I_art_NN_local = article_index.search(emb, k=k_query)
    I_art_NN_all_global = article_global_idx[I_art_NN_local]

    return {
        "emb": emb,
        "ids": np.array(ids),
        "types": types_arr,
        "is_article": is_article,
        "is_book": is_book,
        "article_global_idx": article_global_idx,
        "book_global_idx": book_global_idx,
        "sigma": sigma,
        "D_art_NN_all": D_art_NN_all,
        "I_art_NN_all_global": I_art_NN_all_global,
        "full_index": full_index,
        "article_index": article_index,
    }


def main() -> None:
    raise NotImplementedError("main() wired in Task 7")


if __name__ == "__main__":
    main()
