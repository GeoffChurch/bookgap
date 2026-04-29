"""Region-depletion hypothesis testing for gap detection.

Provides binomial depletion tests, Benjamini-Hochberg FDR correction, and
effect-size computations on regions (HDBSCAN clusters or K-ball spheres),
asking whether a region has significantly fewer target-class members
(typically books) than expected under H0 of uniform sampling from the
corpus.

Used by ``artifacts_scripts/hdbscan_gap_probe.py`` and
``artifacts_scripts/kball_gap_probe.py`` to score detected gaps under a
shared statistical framework, so cross-algorithm comparisons use a
consistent definition of "significantly target-depleted."

The test is one-sided (left tail of the binomial). ``effect_size`` is
``(expected - observed) / expected`` ∈ (-∞, 1]: 1.0 = perfectly depleted
(0 observed), 0.0 = at base rate, negative = enriched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class RegionScore:
    """Per-region depletion score."""

    region_id: int
    size: int
    target_count: int
    expected: float
    p_value: float
    effect_size: float


def score_region_depletion(
    *,
    region_id: int,
    size: int,
    target_count: int,
    base_rate: float,
) -> RegionScore:
    """Score a single region for one-sided (left-tail) target-class depletion.

    H0: each member of the region is independently target-class with
    probability ``base_rate``. ``p_value = P(X <= target_count)`` is the
    left-tail of Binomial(size, base_rate); small values mean the region
    is significantly depleted of the target class.
    """
    expected = size * base_rate
    p_value = float(binom.cdf(target_count, size, base_rate))
    effect_size = (expected - target_count) / expected if expected > 0 else 0.0
    return RegionScore(
        region_id=region_id,
        size=size,
        target_count=target_count,
        expected=expected,
        p_value=p_value,
        effect_size=effect_size,
    )


def benjamini_hochberg_fdr_count(p_values: Sequence[float], alpha: float) -> int:
    """Count rejections at FDR level ``alpha`` via Benjamini-Hochberg.

    Sorts p-values ascending and finds the largest ``k`` such that the
    ``k``-th smallest p-value is ``<= (k / N) * alpha``. Everything ranked
    1..k is declared a discovery; returns ``k`` (zero if no discoveries).
    """
    p_arr = np.sort(np.asarray(p_values, dtype=np.float64))
    n = len(p_arr)
    if n == 0:
        return 0
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passes = p_arr <= thresholds
    if not passes.any():
        return 0
    return int(np.where(passes)[0].max() + 1)


def benjamini_hochberg_pass_mask(
    p_values: Sequence[float], alpha: float
) -> np.ndarray:
    """Return a per-input boolean mask of which p-values pass BH-FDR at ``alpha``.

    Same logic as :func:`benjamini_hochberg_fdr_count` but returns the per-
    input pass/fail decision (in input order, not sorted order) so the
    flag can be carried through to per-row CSV output.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    if n == 0:
        return np.zeros(0, dtype=bool)
    sort_perm = np.argsort(p_arr)
    sorted_ps = p_arr[sort_perm]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passes_sorted = sorted_ps <= thresholds
    if not passes_sorted.any():
        return np.zeros(n, dtype=bool)
    # Largest rank k where the sorted p-value still satisfies the BH
    # threshold; everything ranked 1..k+1 is a discovery.
    last_pass_rank = int(np.where(passes_sorted)[0].max())
    sorted_mask = np.zeros(n, dtype=bool)
    sorted_mask[: last_pass_rank + 1] = True
    pass_mask = np.zeros(n, dtype=bool)
    pass_mask[sort_perm] = sorted_mask
    return pass_mask


_DEFAULT_RAW_P_THRESHOLDS: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 0.01, 0.05)
_DEFAULT_FDR_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class DepletionSummary:
    """Aggregate depletion statistics across a list of regions."""

    n_regions: int
    n_strictly_zero: int
    raw_significance_counts: dict[float, int]
    fdr_significance_counts: dict[float, int]
    regions: list[RegionScore] = field(default_factory=list)


def score_regions(
    *,
    region_ids: Sequence[int],
    sizes: Sequence[int],
    target_counts: Sequence[int],
    base_rate: float,
    raw_p_thresholds: Sequence[float] = _DEFAULT_RAW_P_THRESHOLDS,
    fdr_alphas: Sequence[float] = _DEFAULT_FDR_ALPHAS,
) -> DepletionSummary:
    """Score a list of regions and compute summary statistics.

    Returns a :class:`DepletionSummary` with per-region scores, raw-
    significance counts at the requested ``p`` thresholds, and FDR-
    corrected counts at the requested ``alpha`` levels.
    """
    if not (len(region_ids) == len(sizes) == len(target_counts)):
        raise ValueError(
            f"region_ids ({len(region_ids)}), sizes ({len(sizes)}), "
            f"and target_counts ({len(target_counts)}) must match"
        )
    regions = [
        score_region_depletion(
            region_id=int(rid),
            size=int(sz),
            target_count=int(tc),
            base_rate=base_rate,
        )
        for rid, sz, tc in zip(region_ids, sizes, target_counts)
    ]
    p_values = [r.p_value for r in regions]
    return DepletionSummary(
        n_regions=len(regions),
        n_strictly_zero=sum(1 for r in regions if r.target_count == 0),
        raw_significance_counts={
            t: sum(1 for r in regions if r.p_value < t) for t in raw_p_thresholds
        },
        fdr_significance_counts={
            a: benjamini_hochberg_fdr_count(p_values, a) for a in fdr_alphas
        },
        regions=regions,
    )
