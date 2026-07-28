"""Task B: empirical-null calibration of the gap statistic.

Replaces the parametric LCB (z = Delta - 0.5*sigma, sigma from max-ball counts)
with calibration against the EMPIRICAL null formed by the matched
hypothetical-removal controls: for each candidate, the upper-tail empirical
p-value of its raw statistic (Delta, the kNN-radius density log-ratio) among the
control points' statistics. Small p = "more gap-like than untreated locations".

Statistic choice: calibrate RAW Delta, not z — the LCB's sigma term is exactly
the parametric correction that calibration is meant to replace; calibrating z
would double-correct.

Pooling strategy (decided + measured, see report "Task B"):
  * POOLED nulls are order-preserving: p is a monotone transform of Delta, so the
    candidate ORDERING (rank/AUC) is identical to delta_only by construction —
    pooled calibration can only re-scale, never re-rank. It is kept as the sanity
    row.
  * STRATIFIED nulls re-rank: conditioning on the local bg scale (r_B,k terciles)
    compares each candidate to nulls with comparable evidence geometry — the
    nonparametric analog of the LCB's 1/n variance term, without imposing the
    parametric form. Main configuration: r_B,k terciles (quantiles over real bg),
    min 15 nulls per stratum with pooled fallback; sensitivity row swaps the
    stratifier to local bg counts (n_Bloc terciles).

Also provides the calibrated veto ceiling for Task A's fusion: the Tukey upper
fence (Q3 + mult*IQR) of the null Delta distribution — "more extreme than any
plausible in-field removal signature" — as the data-driven alternative to the
fixed log-C proximal-presence gate.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import numpy as np

MIN_NULL_PER_STRATUM = 15


def empirical_p(stats: np.ndarray, null_stats: np.ndarray) -> np.ndarray:
    """Conservative upper-tail empirical p-value: (1 + #{null >= s}) / (1 + n).

    A candidate drawn from the null distribution gets p ~ uniform on the grid
    {1/(n+1), ..., 1}; a candidate above every null gets the minimum 1/(n+1).
    """
    null_sorted = np.sort(np.asarray(null_stats, dtype=np.float64))
    n = null_sorted.size
    ge = n - np.searchsorted(null_sorted, np.asarray(stats, dtype=np.float64), side="left")
    return (1.0 + ge) / (1.0 + n)


def stratify(values: np.ndarray, real_mask: np.ndarray, n_strata: int = 3) -> np.ndarray:
    """Stratum index per candidate; band edges = quantiles over real candidates."""
    qs = np.quantile(values[real_mask], np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(qs, values, side="right")


def stratified_p(
    stats: np.ndarray,
    strata: np.ndarray,
    null_idx: np.ndarray,
    n_strata: int = 3,
    min_null: int = MIN_NULL_PER_STRATUM,
) -> np.ndarray:
    """Per-candidate empirical p against same-stratum nulls (pooled fallback).

    `null_idx` indexes into `stats`/`strata` (the matched-control candidates).
    Strata with fewer than `min_null` null points fall back to the pooled null.
    """
    p = np.empty(stats.shape[0], dtype=np.float64)
    null_strata = strata[null_idx]
    pooled_null = stats[null_idx]
    for s in range(n_strata):
        sel = strata == s
        if not sel.any():
            continue
        nl = stats[null_idx[null_strata == s]]
        if nl.size < min_null:
            nl = pooled_null
        p[sel] = empirical_p(stats[sel], nl)
    return p


def lexico_scores(p: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Float scores whose DESCENDING order is (p ascending, tiebreak descending).

    Small empirical nulls leave p on a coarse grid; ranks need a documented
    tie-break. AUC should be computed on -p directly (Mann-Whitney is tie-aware).
    """
    order = np.lexsort((-tiebreak, p))
    sc = np.empty(p.shape[0], dtype=np.float64)
    sc[order] = -np.arange(p.shape[0], dtype=np.float64)
    return sc


def calibrated_ceiling(null_deltas: np.ndarray, mult: float = 3.0) -> float:
    """Tukey upper fence of the null statistic: Q3 + mult*IQR."""
    q1, q3 = np.quantile(np.asarray(null_deltas, dtype=np.float64), [0.25, 0.75])
    return float(q3 + mult * (q3 - q1))
