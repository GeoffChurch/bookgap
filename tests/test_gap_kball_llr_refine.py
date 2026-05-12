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


def test_local_csls_posterior_pure_gap_returns_post_near_one():
    """If members have all their kNN mass inside G, posterior ≈ 1."""
    from artifacts_scripts.gap_kball_llr_refine import compute_local_csls_posterior
    # 5 members, each with 4 neighbors entirely inside G (member_set itself).
    members = [0, 1, 2, 3, 4]
    # Each member's top-K kNN: list of global indices and squared distances.
    n_total = 6  # 5 articles + 1 book
    sigma = np.ones(n_total, dtype=np.float32)
    # Build a 5x4 D_NN where each member's 4 neighbors are the other 4 members.
    # Distances: small so kernel weights are large.
    D_NN = np.full((5, 4), 0.1, dtype=np.float32)
    I_NN = np.array([
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 4],
        [0, 1, 2, 3],
    ], dtype=np.int64)
    # No books found within K — boundary book list is empty.
    # For simplicity: book index is the 6th work; nearest_book per member is index 5.
    nearest_book_per_member = np.array([5, 5, 5, 5, 5], dtype=np.int64)
    # Background articles per book: take all 5 members (or whichever non-gap available).
    # With only article indices [0..4] in gap and book = 5, the "non-gap" article set
    # is empty in our synthetic — so K_B = 0 ⇒ post = 1 for every member.
    book_to_arts = {5: np.array([], dtype=np.int64)}
    posts = compute_local_csls_posterior(
        members=members,
        D_NN=D_NN,
        I_NN=I_NN,
        sigma=sigma,
        nearest_book_per_member=nearest_book_per_member,
        book_to_arts=book_to_arts,
        n_total=n_total,
    )
    np.testing.assert_allclose(posts, np.ones(5), rtol=1e-3, atol=1e-3)


def test_dedup_via_binary_jaccard_merges_near_duplicates():
    """Two gaps sharing 90% of their members at threshold 0.5 merge."""
    from artifacts_scripts.gap_kball_llr_refine import dedup_via_binary_jaccard
    gaps = [
        # gap 0: members 0..9, logit_sum = 100
        {"gap_idx": 0, "refined_member_indices": list(range(10)), "logit_sum": 100.0},
        # gap 1: members 0..8 + 100, logit_sum = 50 (overlaps with gap 0 at 9/11 = 0.82)
        {"gap_idx": 1, "refined_member_indices": list(range(9)) + [100], "logit_sum": 50.0},
        # gap 2: members 200..209, disjoint from 0 and 1
        {"gap_idx": 2, "refined_member_indices": list(range(200, 210)), "logit_sum": 80.0},
    ]
    annotated = dedup_via_binary_jaccard(gaps, threshold=0.5)
    by_idx = {g["gap_idx"]: g for g in annotated}
    # gap 0 has higher logit_sum so it's the survivor; gap 1 merged into 0.
    assert by_idx[0]["survivor"] == 1
    assert by_idx[0]["merged_into_gap_id"] == -1
    assert by_idx[1]["survivor"] == 0
    assert by_idx[1]["merged_into_gap_id"] == 0
    # gap 2 is disjoint; survivor on its own.
    assert by_idx[2]["survivor"] == 1
    assert by_idx[2]["merged_into_gap_id"] == -1
