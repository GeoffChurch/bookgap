"""Tests for ``bookgap.depletion_test``."""
import numpy as np
import pytest

from bookgap.depletion_test import (
    benjamini_hochberg_fdr_count,
    score_region_depletion,
    score_regions,
)


class TestScoreRegionDepletion:
    def test_at_base_rate_pvalue_around_half(self):
        # Region of 100 with 50 books, base rate 0.5: expected=50, observed=50.
        # P(X <= 50 | n=100, p=0.5) ≈ 0.54 (just above half thanks to discrete
        # CDF including the 50 mass).
        r = score_region_depletion(
            region_id=0, size=100, target_count=50, base_rate=0.5,
        )
        assert r.expected == pytest.approx(50.0)
        assert 0.5 < r.p_value < 0.6
        assert abs(r.effect_size) < 1e-9

    def test_strictly_zero_high_significance(self):
        # 100 works at base rate 0.5, observed 0. P(X=0) = 0.5^100 ≈ 7.9e-31.
        r = score_region_depletion(
            region_id=0, size=100, target_count=0, base_rate=0.5,
        )
        assert r.target_count == 0
        assert r.p_value < 1e-25
        assert r.effect_size == pytest.approx(1.0)

    def test_enriched_negative_effect_size(self):
        # Region of 100 with 80 books, base rate 0.5: enriched.
        r = score_region_depletion(
            region_id=0, size=100, target_count=80, base_rate=0.5,
        )
        assert r.effect_size < 0
        assert r.effect_size == pytest.approx((50 - 80) / 50)
        # p_value high (depletion test, but observed > expected → not depleted).
        assert r.p_value > 0.99

    def test_zero_size_zero_expected_zero_effect(self):
        # Edge case: empty region. Expected=0, effect_size=0 by convention.
        r = score_region_depletion(
            region_id=0, size=0, target_count=0, base_rate=0.5,
        )
        assert r.expected == 0.0
        assert r.effect_size == 0.0
        # binom.cdf(0, 0, 0.5) = 1.0 (degenerate distribution at 0)
        assert r.p_value == pytest.approx(1.0)

    def test_id_passthrough(self):
        r = score_region_depletion(
            region_id=42, size=10, target_count=0, base_rate=0.1,
        )
        assert r.region_id == 42


class TestBenjaminiHochbergFDRCount:
    def test_empty_returns_zero(self):
        assert benjamini_hochberg_fdr_count([], 0.05) == 0

    def test_all_ones_no_discoveries(self):
        # No p-value passes any threshold.
        assert benjamini_hochberg_fdr_count([1.0, 1.0, 1.0], 0.05) == 0

    def test_all_tiny_all_discoveries(self):
        # All p-values are far below any reasonable threshold → all rejected.
        assert benjamini_hochberg_fdr_count([1e-10, 1e-10, 1e-10], 0.05) == 3

    def test_classic_example(self):
        # 5 ordered p-values; thresholds at α=0.05 are [0.01, 0.02, 0.03, 0.04, 0.05].
        # p=0.01 ≤ 0.01 ✓, p=0.012 ≤ 0.02 ✓, p=0.04 > 0.03 ✗, p=0.06 > 0.04 ✗, p=0.5 > 0.05 ✗
        # Largest k where passing: k=2. (Even though k=2 is the largest passing rank,
        # BH declares all k ≤ that rank as discoveries.)
        ps = [0.01, 0.012, 0.04, 0.06, 0.5]
        assert benjamini_hochberg_fdr_count(ps, 0.05) == 2

    def test_unsorted_input_handled(self):
        # Internal sort means input order shouldn't matter.
        ps_sorted = [0.01, 0.012, 0.04, 0.06, 0.5]
        ps_shuffled = [0.5, 0.012, 0.04, 0.01, 0.06]
        assert benjamini_hochberg_fdr_count(ps_sorted, 0.05) == benjamini_hochberg_fdr_count(ps_shuffled, 0.05)


class TestScoreRegions:
    def test_summary_counts(self):
        # 4 regions, base rate 0.5:
        #   r1: size=100, books=0  → p ≈ 7.9e-31, effect=1.0
        #   r2: size=100, books=30 → p ≈ 4e-5, effect=0.4
        #   r3: size=100, books=50 → p ≈ 0.54, effect=0
        #   r4: size=100, books=80 → p ≈ 1, effect=-0.6
        summary = score_regions(
            region_ids=[1, 2, 3, 4],
            sizes=[100, 100, 100, 100],
            target_counts=[0, 30, 50, 80],
            base_rate=0.5,
        )
        assert summary.n_regions == 4
        assert summary.n_strictly_zero == 1
        # Per-region scores in input order
        assert [r.region_id for r in summary.regions] == [1, 2, 3, 4]
        # p<1e-3 catches r1 and r2; p<0.01 same; p<0.05 same
        assert summary.raw_significance_counts[1e-3] == 2
        assert summary.raw_significance_counts[0.05] == 2
        # FDR: at α=0.05, both r1 and r2 should pass
        assert summary.fdr_significance_counts[0.05] >= 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            score_regions(
                region_ids=[1, 2],
                sizes=[10, 10, 10],
                target_counts=[0, 0],
                base_rate=0.5,
            )
