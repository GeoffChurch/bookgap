"""Multi-cluster softmax Phase 1 comparison.

Reads original soft-EM enriched CSV + new (variant, P, iter, rule)
basin CSVs from gap_softmax_reassign.py, emits Jaccard / eff_rank /
coverage-label / leaderboard diff and a markdown report.

Spec: docs/superpowers/specs/2026-05-08-multicluster-softmax-phase1-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_basin_supports(csv_path: Path) -> list[frozenset[int]]:
    """Load a basin CSV -> list of frozensets of article global indices.

    Uses the member_indices_json column. Empty basins (n_articles=0)
    yield empty frozensets in the same position.
    """
    out: list[frozenset[int]] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            members = json.loads(row["member_indices_json"])
            out.append(frozenset(int(m) for m in members))
    return out


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def best_match_jaccard(
    src: list[frozenset[int]],
    tgt: list[frozenset[int]],
) -> dict:
    """For each non-empty basin in src, find the best Jaccard against
    tgt. Return distribution stats.

    Empty basins are skipped (they have no support; not informative).
    """
    scores: list[float] = []
    for s in src:
        if not s:
            continue
        scores.append(max((_jaccard(s, t) for t in tgt), default=0.0))
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0, "mean": float("nan"), "p25": float("nan"),
                "p50": float("nan"), "p75": float("nan"), "p95": float("nan"),
                "count_ge_0.95": 0, "count_lt_0.80": 0, "count_eq_1.0": 0}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "count_ge_0.95": int((arr >= 0.95).sum()),
        "count_lt_0.80": int((arr < 0.80).sum()),
        "count_eq_1.0": int((arr == 1.0).sum()),
    }


def effective_rank(p: np.ndarray) -> np.ndarray:
    """exp(H(p_row)) per row. p assumed nonneg, rows sum to 1.

    Returns shape (N,) float64. exp(0) = 1 for pure rows.
    """
    p_clip = np.clip(p, 1e-30, 1.0)
    H = -(p_clip * np.log(p_clip)).sum(axis=1)
    return np.exp(H)


def argmax_shift_rate(M_old: np.ndarray, M_new: np.ndarray) -> float:
    """Fraction of rows whose argmax column index differs."""
    return float((M_old.argmax(axis=1) != M_new.argmax(axis=1)).mean())


def eff_rank_summary(eff_old: np.ndarray, eff_new: np.ndarray) -> dict:
    delta = eff_old - eff_new   # positive = softmax sharpens
    return {
        "median_old": float(np.median(eff_old)),
        "median_new": float(np.median(eff_new)),
        "median_delta": float(np.median(delta)),
        "p25_delta": float(np.percentile(delta, 25)),
        "p75_delta": float(np.percentile(delta, 75)),
        "iqr_delta": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "frac_delta_ge_0.5": float((delta >= 0.5).mean()),
        "frac_delta_le_neg_0.5": float((delta <= -0.5).mean()),
    }


def save_eff_rank_scatter(
    out_path: Path,
    eff_old: np.ndarray,
    eff_new_oneshot: np.ndarray | None,
    eff_new_iterated: np.ndarray | None,
    title_suffix: str = "",
) -> None:
    """2D hexbin: x = eff_rank_old, y = eff_rank_new. Two panels:
    one-shot and iterated, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("one-shot", eff_new_oneshot),
        ("iterated", eff_new_iterated),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (name, eff_new) in zip(axes, panels):
        if eff_new is None:
            ax.text(0.5, 0.5, f"({name} missing)", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        ax.hexbin(eff_old, eff_new, gridsize=30, mincnt=1, cmap="viridis")
        max_val = max(eff_old.max(), eff_new.max())
        ax.plot([1, max_val], [1, max_val], "r--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{name}{title_suffix}")
        ax.set_xlabel("eff_rank old (normalized soft-EM)")
        ax.set_ylabel("eff_rank new (multi-cluster softmax)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    raise NotImplementedError("driver assembled in later tasks")


if __name__ == "__main__":
    main()
