"""Tests for post-proposal coverage analysis + AEP projection."""
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
    # In 3D: place 2 articles near book_0 (at origin), 2 near where the
    # proposal will sit (at +e_0), books along e_0 axis.
    gap_articles = np.array([
        [0.0, 0.1, 0.0],   # nearest to book_0
        [0.0, -0.1, 0.0],  # nearest to book_0
        [1.0, 0.1, 0.0],   # nearest to proposal (placed at e_0)
        [1.0, -0.1, 0.0],  # nearest to proposal
    ], dtype=np.float32)
    books = np.array([
        [0.0, 0.0, 0.0],   # book_0
        [3.0, 0.0, 0.0],   # book_1, far away
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


def test_aep_preserves_radial_distance_exactly():
    """Each point's 2D distance from origin equals its ND distance from seed."""
    rng = np.random.default_rng(seed=42)
    seed = rng.normal(size=10).astype(np.float32)
    gap_articles = seed + 0.3 * rng.normal(size=(20, 10)).astype(np.float32)
    books = seed + 1.5 * rng.normal(size=(5, 10)).astype(np.float32)
    proposal = seed + 0.4 * rng.normal(size=10).astype(np.float32)
    green_mask = np.array([True] * 5 + [False] * 15)

    result = compute_local_projection_2d(
        seed_embedding=seed,
        gap_article_embeddings=gap_articles,
        green_mask=green_mask,
        book_embeddings=books,
        proposal_embedding=proposal,
        gap_radius=1.0,
    )

    nd_radii = np.linalg.norm(gap_articles - seed, axis=1)
    twod_radii = np.linalg.norm(result.gap_articles_2d, axis=1)
    np.testing.assert_allclose(twod_radii, nd_radii, atol=1e-5)

    nd_book_radii = np.linalg.norm(books - seed, axis=1)
    twod_book_radii = np.linalg.norm(result.books_2d, axis=1)
    np.testing.assert_allclose(twod_book_radii, nd_book_radii, atol=1e-5)


def test_aep_makes_greens_compact_when_possible():
    """Greens placed at +e_0 direction; non-greens scattered. After GEP,
    greens should cluster at similar 2D angles."""
    rng = np.random.default_rng(seed=0)
    seed = np.zeros(10, dtype=np.float32)
    # Greens: small noise around +e_0 direction
    greens = np.tile(np.eye(1, 10, 0).astype(np.float32), (10, 1)) \
             + 0.05 * rng.normal(size=(10, 10)).astype(np.float32)
    # Non-greens: scattered in other directions
    non_greens = rng.normal(size=(10, 10)).astype(np.float32)
    gap_articles = np.vstack([greens, non_greens])
    green_mask = np.array([True] * 10 + [False] * 10)
    books = rng.normal(size=(3, 10)).astype(np.float32) * 2.0
    proposal = np.eye(1, 10, 0).ravel().astype(np.float32)

    result = compute_local_projection_2d(
        seed_embedding=seed,
        gap_article_embeddings=gap_articles,
        green_mask=green_mask,
        book_embeddings=books,
        proposal_embedding=proposal,
        gap_radius=1.5,
    )

    # Green angles should be more concentrated than non-green angles
    green_angles = np.arctan2(result.gap_articles_2d[:10, 1],
                               result.gap_articles_2d[:10, 0])
    non_green_angles = np.arctan2(result.gap_articles_2d[10:, 1],
                                   result.gap_articles_2d[10:, 0])
    # Circular variance — lower for greens
    green_cv = 1 - np.abs(np.exp(1j * green_angles).mean())
    non_green_cv = 1 - np.abs(np.exp(1j * non_green_angles).mean())
    assert green_cv < non_green_cv, (
        f"greens should cluster more than non-greens; "
        f"green_cv={green_cv:.3f}, non_green_cv={non_green_cv:.3f}"
    )


def test_aep_falls_back_to_pca_when_no_greens():
    """When green_mask is all False, the GEP is degenerate; fall back
    to PCA of (gap_articles ∪ books ∪ proposal)."""
    rng = np.random.default_rng(seed=1)
    seed = np.zeros(8, dtype=np.float32)
    gap_articles = 0.3 * rng.normal(size=(15, 8)).astype(np.float32)
    green_mask = np.zeros(15, dtype=bool)
    books = rng.normal(size=(4, 8)).astype(np.float32)
    proposal = rng.normal(size=8).astype(np.float32)

    # Should not raise:
    result = compute_local_projection_2d(
        seed_embedding=seed,
        gap_article_embeddings=gap_articles,
        green_mask=green_mask,
        book_embeddings=books,
        proposal_embedding=proposal,
        gap_radius=1.0,
    )
    # Radial preservation still holds
    nd = np.linalg.norm(gap_articles - seed, axis=1)
    twod = np.linalg.norm(result.gap_articles_2d, axis=1)
    np.testing.assert_allclose(twod, nd, atol=1e-5)


def test_aep_seed_is_at_origin():
    """Seed always projects to (0, 0)."""
    seed = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    result = compute_local_projection_2d(
        seed_embedding=seed,
        gap_article_embeddings=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        green_mask=np.array([True]),
        book_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        proposal_embedding=np.array([0.4, 0.4, 0.5], dtype=np.float32),
        gap_radius=1.0,
    )
    assert result.seed_2d.tolist() == [0.0, 0.0]
