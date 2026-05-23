"""Tests for post-proposal coverage analysis + PCA-on-all-articles projection."""
from __future__ import annotations

import numpy as np
import pytest

from bookgap.llm.coverage_analysis import (
    compute_post_proposal_coverage,
    PostProposalCoverage,
    compute_local_projection_2d,
    LocalProjection2DResult,
)


def test_post_proposal_coverage_handful_synthetic():
    """4 gap articles, 2 books, 1 proposal placed near 2 of the articles."""
    gap_articles = np.array([
        [0.0, 0.1, 0.0],   # nearest to book_0
        [0.0, -0.1, 0.0],  # nearest to book_0
        [1.0, 0.1, 0.0],   # nearest to proposal (placed at e_0)
        [1.0, -0.1, 0.0],  # nearest to proposal
    ], dtype=np.float32)
    books = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    proposal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    result = compute_post_proposal_coverage(
        gap_article_embeddings=gap_articles,
        book_embeddings=books,
        proposal_embedding=proposal,
    )
    assert isinstance(result, PostProposalCoverage)
    assert result.proposal_articles_covered == 2
    assert result.proposal_fraction_covered == pytest.approx(0.5)
    assert list(result.per_book_articles_covered_after) == [2, 0]


def test_post_proposal_coverage_no_flips():
    """Proposal sits FAR from gap articles → no flips."""
    gap_articles = np.array([[0.0, 0.0, 0.0],
                              [0.1, 0.0, 0.0]], dtype=np.float32)
    books = np.array([[0.0, 0.5, 0.0]], dtype=np.float32)
    proposal = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    result = compute_post_proposal_coverage(
        gap_article_embeddings=gap_articles,
        book_embeddings=books,
        proposal_embedding=proposal,
    )
    assert result.proposal_articles_covered == 0
    assert list(result.per_book_articles_covered_after) == [2]


def test_post_proposal_coverage_all_flips():
    """Proposal is at gap centroid → absorbs everything."""
    gap_articles = np.array([[0.0, 0.0, 0.0],
                              [0.1, 0.0, 0.0],
                              [-0.1, 0.0, 0.0]], dtype=np.float32)
    books = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    proposal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    result = compute_post_proposal_coverage(
        gap_article_embeddings=gap_articles,
        book_embeddings=books,
        proposal_embedding=proposal,
    )
    assert result.proposal_articles_covered == 3
    assert list(result.per_book_articles_covered_after) == [0]


def _project_args(rng, n_topic_articles=50, n_gap=10, n_books=5, d=8):
    """Shared synthetic-input builder."""
    all_articles = rng.normal(size=(n_topic_articles, d)).astype(np.float32)
    gap_indices = rng.choice(n_topic_articles, size=n_gap, replace=False)
    gap_articles = all_articles[gap_indices]
    seed = all_articles[gap_indices[0]]  # seed is one of the gap articles
    books = rng.normal(size=(n_books, d)).astype(np.float32) * 1.5
    proposal = rng.normal(size=d).astype(np.float32)
    return dict(
        all_topic_article_embeddings=all_articles,
        seed_embedding=seed,
        gap_article_embeddings=gap_articles,
        book_embeddings=books,
        proposal_embedding=proposal,
    )


def test_pca_projection_returns_correct_shapes():
    """LocalProjection2DResult has the expected shapes for each entity."""
    rng = np.random.default_rng(seed=42)
    args = _project_args(rng, n_topic_articles=50, n_gap=10, n_books=5, d=8)
    result = compute_local_projection_2d(**args)
    assert isinstance(result, LocalProjection2DResult)
    assert result.seed_2d.shape == (2,)
    assert result.proposal_2d.shape == (2,)
    assert result.gap_articles_2d.shape == (10, 2)
    assert result.books_2d.shape == (5, 2)


def test_pca_projection_is_norm_contracting():
    """Linear orthonormal projection: 2D distance from origin ≤ ND distance
    from the all-article mean."""
    rng = np.random.default_rng(seed=7)
    args = _project_args(rng, n_topic_articles=100, n_gap=20, n_books=10, d=12)
    result = compute_local_projection_2d(**args)
    center = args["all_topic_article_embeddings"].mean(axis=0)

    nd_gap = np.linalg.norm(args["gap_article_embeddings"] - center, axis=1)
    twod_gap = np.linalg.norm(result.gap_articles_2d, axis=1)
    # 2D distance is the projection magnitude — must be ≤ original distance.
    assert np.all(twod_gap <= nd_gap + 1e-5)

    nd_book = np.linalg.norm(args["book_embeddings"] - center, axis=1)
    twod_book = np.linalg.norm(result.books_2d, axis=1)
    assert np.all(twod_book <= nd_book + 1e-5)


def test_pca_projection_captures_dominant_directions():
    """If the article cloud has its variance concentrated along a known
    direction, PCA axis 0 should align with that direction."""
    d = 10
    n = 100
    rng = np.random.default_rng(seed=3)
    # Strongly correlated along e_0; weak noise elsewhere.
    main_axis = np.zeros(d, dtype=np.float32)
    main_axis[0] = 1.0
    weights = rng.normal(size=(n, 1)).astype(np.float32) * 5.0
    noise = rng.normal(size=(n, d)).astype(np.float32) * 0.01
    all_articles = weights * main_axis + noise

    args = dict(
        all_topic_article_embeddings=all_articles,
        seed_embedding=all_articles[0],
        gap_article_embeddings=all_articles[:5],
        book_embeddings=rng.normal(size=(3, d)).astype(np.float32),
        proposal_embedding=rng.normal(size=d).astype(np.float32),
    )
    result = compute_local_projection_2d(**args)
    # The article variance along the projection's first axis should be
    # essentially the full e_0 variance; the second axis captures noise.
    var_axis_0 = np.var(result.gap_articles_2d[:, 0])
    var_axis_1 = np.var(result.gap_articles_2d[:, 1])
    # Articles vary by ~25× on e_0 (weights * 5), so axis 0 should dominate.
    assert var_axis_0 > var_axis_1 * 100, (
        f"PCA axis 0 should capture dominant article variance; "
        f"var_axis_0={var_axis_0:.4f}, var_axis_1={var_axis_1:.4f}"
    )


def test_pca_projection_centered_at_article_mean():
    """Centering at the all-article mean: the article cloud's projection
    is centered around the 2D origin."""
    rng = np.random.default_rng(seed=11)
    args = _project_args(rng, n_topic_articles=200, n_gap=20, n_books=5, d=10)
    result = compute_local_projection_2d(**args)
    # Project all articles via the same axes to check center.
    center = args["all_topic_article_embeddings"].mean(axis=0)
    centered = args["all_topic_article_embeddings"] - center
    S = centered.T @ centered
    from scipy.linalg import eigh
    _, eigvecs = eigh(S)
    axes = eigvecs[:, -2:][:, ::-1]
    all_2d = (args["all_topic_article_embeddings"] - center) @ axes
    np.testing.assert_allclose(all_2d.mean(axis=0), [0.0, 0.0], atol=1e-5)


def test_pca_projection_handles_small_article_set():
    """3 articles, 1 book, 1 proposal — projection still produces valid output."""
    rng = np.random.default_rng(seed=13)
    args = dict(
        all_topic_article_embeddings=rng.normal(size=(3, 6)).astype(np.float32),
        seed_embedding=rng.normal(size=6).astype(np.float32),
        gap_article_embeddings=rng.normal(size=(2, 6)).astype(np.float32),
        book_embeddings=rng.normal(size=(1, 6)).astype(np.float32),
        proposal_embedding=rng.normal(size=6).astype(np.float32),
    )
    result = compute_local_projection_2d(**args)
    assert result.gap_articles_2d.shape == (2, 2)
    assert result.books_2d.shape == (1, 2)
    # No NaN/inf
    for arr in (result.seed_2d, result.proposal_2d,
                result.gap_articles_2d, result.books_2d):
        assert np.all(np.isfinite(arr))
