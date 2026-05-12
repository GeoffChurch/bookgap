"""Tests for the K-ball + LLR refinement pipeline's helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from artifacts_scripts.gap_kball_llr_refine import (
    binary_set_jaccard,
    load_kball_csv,
    refine_gap_via_llr,
)


def test_binary_set_jaccard_identical_sets_returns_one():
    a = {1, 2, 3, 4, 5}
    b = {1, 2, 3, 4, 5}
    assert binary_set_jaccard(a, b) == pytest.approx(1.0)


def test_binary_set_jaccard_disjoint_returns_zero():
    a = {1, 2, 3}
    b = {4, 5, 6}
    assert binary_set_jaccard(a, b) == pytest.approx(0.0)


def test_binary_set_jaccard_half_overlap_equal_size():
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    # |intersection| = 2, |union| = 6, J = 2/6 = 1/3
    assert binary_set_jaccard(a, b) == pytest.approx(2.0 / 6.0)


def test_binary_set_jaccard_subset_returns_size_ratio():
    a = {1, 2}
    b = {1, 2, 3, 4}
    # |∩| = 2, |∪| = 4, J = 0.5
    assert binary_set_jaccard(a, b) == pytest.approx(0.5)


def test_binary_set_jaccard_both_empty_returns_zero():
    assert binary_set_jaccard(set(), set()) == pytest.approx(0.0)


def test_load_kball_csv_parses_minimal_row(tmp_path):
    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text(
        "scope,algorithm,denoiser,drop_k,class_axis,radius_fraction,num_gaps_per_topic,"
        "topic_id,topic_name,gap_idx,gap_size,gap_size_percentile,gap_zscore,"
        "shapiro_log10_pval,article_indices_json\n"
        "17/1702/T11303,kball,pca,0,topic_x_type,0.75,200,T11303,Bayesian,"
        "0,3,0.5,1.0,-2.0,\"[12,34,56]\"\n"
    )
    rows = load_kball_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["gap_idx"] == 0
    assert rows[0]["gap_size"] == 3
    assert rows[0]["topic_id"] == "T11303"
    assert rows[0]["article_indices"] == [12, 34, 56]
