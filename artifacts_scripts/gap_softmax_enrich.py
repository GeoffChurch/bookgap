"""Multi-cluster softmax Phase 2 enrichment.

Replaces gap_book_existence_filter + gap_multi_scale_coverage with a
single one-shot multi-cluster softmax refinement layer over soft-EM's
basin set. Output is a drop-in enriched CSV with templated-leak
tagging, direct softmax effective_uncovered, and continuous
inherited_coverage_median_post (no coverage_label_v2 column —
downstream computes its own buckets from the raw flags + metric).

Spec: docs/superpowers/specs/2026-05-09-multicluster-softmax-phase2-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Soft-EM and Phase 2 CSVs both have member_weights_json fields > 131072 bytes
csv.field_size_limit(sys.maxsize)

from artifacts_scripts.gap_softmax_reassign import (
    compute_self_tuning_sigma,
    load_enriched_csv,
    softmax_e_step,
)


def load_soft_em_basins(
    csv_path: Path,
    is_article: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Read a soft-EM raw CSV → (M0, basin_meta).

    Thin alias around gap_softmax_reassign.load_enriched_csv; both raw
    soft-EM and Phase-1-enriched CSVs share the relevant columns
    (rank, seed_article, n_books, book_anchored,
    member_indices_json, member_weights_json), so the same loader works
    on either input.
    """
    return load_enriched_csv(csv_path, is_article)


def run_softmax_step(
    article_emb: np.ndarray,
    M0: np.ndarray,
    k_knn: int = 10,
    k_sigma: int = 7,
    perplexity: float = 2.0,
) -> np.ndarray:
    """One-shot softmax E-step over M0's basin set.

    Returns refined membership matrix M_new of shape
    (N_articles, N_basins). Only the k_knn basin columns per article
    are nonzero; the rest are 0 (kNN truncation). Rows sum to 1
    (Variant A — no ∅ class).
    """
    article_sigma = compute_self_tuning_sigma(article_emb, k_sigma=k_sigma)
    M_new, _ = softmax_e_step(
        article_emb=article_emb,
        article_sigma=article_sigma,
        M=M0,
        log_K_empty=None,
        k_knn=k_knn,
        P_target=perplexity,
    )
    return M_new


def crisp_filter(
    M_new: np.ndarray,
    tau_supp: float = 0.5,
    min_final_size: int = 5,
) -> dict[int, list[tuple[int, float]]]:
    """For each basin, return its supported members above tau_supp.

    Returns dict mapping basin_idx → list of (article_local_idx, weight)
    pairs, sorted by weight descending. Basins with fewer than
    min_final_size supported articles are dropped from the dict.
    """
    out: dict[int, list[tuple[int, float]]] = {}
    n_articles, n_basins = M_new.shape
    for b in range(n_basins):
        col = M_new[:, b]
        members = np.where(col > tau_supp)[0]
        if len(members) < min_final_size:
            continue
        weights = col[members]
        order = np.argsort(-weights)
        out[b] = [(int(members[i]), float(weights[i])) for i in order]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True,
                   help="Topic scope, e.g. 17/1702/T10181")
    p.add_argument("--input", required=True,
                   help="Soft-EM raw CSV path (output of gap_soft_em.py)")
    p.add_argument("--output", required=True,
                   help="Phase 2 enriched CSV path")
    p.add_argument("--k-knn", type=int, default=10)
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-marginal", type=int, default=200)
    p.add_argument("--perplexity", type=float, default=2.0)
    p.add_argument("--tau-supp", type=float, default=0.5)
    p.add_argument("--min-final-size", type=int, default=5)
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--vote-concentration-threshold", type=float, default=0.25)
    p.add_argument("--title-overlap-threshold", type=float, default=0.2)
    p.add_argument("--templated-concentration-threshold", type=float, default=0.5)
    args = p.parse_args()
    raise NotImplementedError("driver implemented in later tasks")


if __name__ == "__main__":
    main()
