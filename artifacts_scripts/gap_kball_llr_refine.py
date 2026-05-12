"""K-ball seed + LLR-vs-k refinement (Phase 3).

Takes K-ball CSV output and re-sizes each gap by LLR-vs-k argmax over
the corpus null. See spec:
``docs/superpowers/specs/2026-05-12-kball-llr-refinement-design.md``.

Pipeline:
  K-ball CSV -> per-gap LLR sweep -> optional one re-centering -> dedup
  + rerank -> refined CSV.

Reuses ``kernel_mass_matrix``, ``setup``, ``_gaussian_weights``, and
``K_GRID`` from ``gap_static_seed_diagnostic``. Reuses the K-ball CSV
loader pattern from ``gap_kball_cohesion``. Ports the local-CSLS-
balanced posterior inline.

Usage::

    .conda/bin/python -u -m artifacts_scripts.gap_kball_llr_refine \\
        --scope 17/1702/T11303 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T11303__pca__drop0.csv \\
        --out artifacts/gap_kball_llr_refine/T11303__llr_refined.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_static_seed_diagnostic import (
    K_GRID,
    K_QUERY,
    _gaussian_weights,
    kernel_mass_matrix,
    setup,
)


def binary_set_jaccard(a: set, b: set) -> float:
    """Set Jaccard |a ∩ b| / |a ∪ b|. Returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def load_kball_csv(path: Path) -> list[dict]:
    """Load K-ball CSV rows, parsing article_indices_json.

    Returns a list of dicts with keys:
      gap_idx (int), gap_size (int), topic_id (str), article_indices (list[int]).
    """
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_idx": int(r["gap_idx"]),
                "gap_size": int(r["gap_size"]),
                "topic_id": r["topic_id"],
                "article_indices": json.loads(r["article_indices_json"]),
            })
    return rows


def refine_gap_via_llr(
    member_indices: list[int],
    state: dict,
    null_log_mean: np.ndarray,
    recenter: bool = True,
) -> dict:
    """Refine a single K-ball gap by LLR-vs-k argmax around its centroid.

    member_indices: list of global article indices (from K-ball's article_indices_json).
    state: shared state dict from gap_static_seed_diagnostic.setup().
    null_log_mean: (len(K_GRID),) precomputed mean of log(corpus null kernel mass)
      at each k in K_GRID.
    recenter: if True, perform one re-centering step after picking k*.

    Returns dict with keys:
      refined_k_star (int), refined_member_indices (list[int] of length k*),
      llr_curve (list[float] of length len(K_GRID)),
      llr_peak (float = max of llr_curve), llr_peak_at_k (int).
    """
    emb = state["emb"]
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]

    members = np.asarray(member_indices, dtype=np.int64)
    centroid = emb[members].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        # Degenerate centroid (shouldn't happen with normalized embeddings).
        # Return the K-ball members verbatim, k* = original size.
        k_star = int(len(member_indices))
        return {
            "refined_k_star": k_star,
            "refined_member_indices": list(member_indices),
            "llr_curve": [0.0] * len(K_GRID),
            "llr_peak": 0.0,
            "llr_peak_at_k": k_star,
        }
    centroid = (centroid / norm).astype(np.float32)

    # Top-K_QUERY nearest articles to centroid.
    D_nn, I_nn_local = article_index.search(centroid[None, :], k=K_QUERY)
    D_nn = D_nn[0]
    I_nn_global = article_global_idx[I_nn_local[0]]

    # Kernel mass at each k in K_GRID. The centroid is virtual (not a work),
    # so we use the bandwidth of the nearest article seed as sigma_seed.
    sigma_seed = float(sigma[I_nn_global[0]])
    sigma_n = sigma[I_nn_global]
    W = np.exp(-D_nn / (sigma_seed * sigma_n))
    cum = np.cumsum(W)
    M_k = np.array([cum[k - 1] for k in K_GRID], dtype=np.float64)

    llr_curve = np.log(np.maximum(M_k, 1e-30)) - null_log_mean
    best_idx = int(np.argmax(llr_curve))
    k_star = int(K_GRID[best_idx])
    llr_peak = float(llr_curve[best_idx])

    refined_members = I_nn_global[:k_star].tolist()

    if recenter:
        # One step: kernel-weighted centroid of refined_members.
        member_emb = emb[refined_members]
        weights = _gaussian_weights(
            D_nn[:k_star], sigma_seed, sigma[I_nn_global[:k_star]],
        )
        denom = float(weights.sum())
        if denom > 1e-30:
            new_centroid = (weights[:, None] * member_emb).sum(axis=0) / denom
            new_norm = float(np.linalg.norm(new_centroid))
            if new_norm >= 1e-12:
                new_centroid = (new_centroid / new_norm).astype(np.float32)
                _, I_new_local = article_index.search(
                    new_centroid[None, :], k=k_star,
                )
                refined_members = article_global_idx[I_new_local[0]].tolist()

    return {
        "refined_k_star": k_star,
        "refined_member_indices": [int(x) for x in refined_members],
        "llr_curve": [float(x) for x in llr_curve],
        "llr_peak": llr_peak,
        "llr_peak_at_k": k_star,
    }


def main() -> None:
    raise NotImplementedError("CLI driver wired in Task 3")


if __name__ == "__main__":
    main()
