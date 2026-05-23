"""Pure NumPy functions: post-proposal coverage analysis + AEP/GEP
local projection from the gap seed.

No I/O, no Cloud, no Hydra — caller supplies all embeddings as arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, LinAlgError


@dataclass(frozen=True)
class PostProposalCoverage:
    proposal_articles_covered: int
    proposal_fraction_covered: float
    per_book_articles_covered_after: NDArray[np.int64]


def compute_post_proposal_coverage(
    *,
    gap_article_embeddings: NDArray[np.floating],
    book_embeddings: NDArray[np.floating],
    proposal_embedding: NDArray[np.floating],
) -> PostProposalCoverage:
    """For each gap article, determine whether the nearest-book among
    {existing books ∪ proposal} is the proposal. Returns counts.
    """
    assert gap_article_embeddings.ndim == 2
    assert book_embeddings.ndim == 2
    assert proposal_embedding.ndim == 1
    assert gap_article_embeddings.shape[1] == book_embeddings.shape[1] == proposal_embedding.shape[0]

    # Stack proposal at the end of the book matrix; nearest-index across
    # this augmented set tells us where each article ends up.
    augmented = np.vstack([book_embeddings, proposal_embedding[None, :]])
    # Cosine-distance ≡ Euclidean on normalized embeddings, ≡ argmin on
    # squared-Euclidean distances in general. The cloud's embeddings are
    # L2-normalized by default (Cloud.normalize_embeddings). Use squared
    # Euclidean for the assignment — robust regardless of normalization.
    diffs = gap_article_embeddings[:, None, :] - augmented[None, :, :]
    sq_dists = np.einsum("nbd,nbd->nb", diffs, diffs)
    nearest = np.argmin(sq_dists, axis=1)

    n_books = book_embeddings.shape[0]
    proposal_idx = n_books
    gap_size = gap_article_embeddings.shape[0]

    proposal_covered = int(np.sum(nearest == proposal_idx))
    per_book_after = np.zeros(n_books, dtype=np.int64)
    for b in range(n_books):
        per_book_after[b] = int(np.sum(nearest == b))

    return PostProposalCoverage(
        proposal_articles_covered=proposal_covered,
        proposal_fraction_covered=(proposal_covered / gap_size) if gap_size > 0 else 0.0,
        per_book_articles_covered_after=per_book_after,
    )


@dataclass(frozen=True)
class LocalProjection2DResult:
    seed_2d: NDArray[np.float64]            # shape (2,), always [0., 0.]
    proposal_2d: NDArray[np.float64]        # shape (2,)
    gap_articles_2d: NDArray[np.float64]    # shape (N_gap, 2)
    books_2d: NDArray[np.float64]           # shape (N_books, 2)


def compute_local_projection_2d(
    *,
    seed_embedding: NDArray[np.floating],
    gap_article_embeddings: NDArray[np.floating],
    green_mask: NDArray[np.bool_],
    book_embeddings: NDArray[np.floating],
    proposal_embedding: NDArray[np.floating],
    gap_radius: float,  # unused for math; included for caller convenience/forward compat
) -> LocalProjection2DResult:
    """Azimuthal equidistant projection from the seed, with angular
    axes (u₁, u₂) chosen by the green-vs-total GEP:

        maximize  trace(W^T S_green W) / trace(W^T S_total W)

    i.e. pick the directions where the greens carry the MOST variance
    relative to the rest. Under AEP, this is what makes greens cluster
    angularly: their lateral projections share a consistent direction
    (their dominant green axis), so after radial rescaling their angles
    stay tight. The non-greens, having little aligned variance on these
    axes, project to noisy lateral directions and spread out in angle.

    (Note: the plan's prose says "maximize total / green", but that
    actually disperses greens angularly under AEP — small lateral
    magnitudes get amplified by radial rescaling and their directions
    are dominated by noise. The objective implemented here is what the
    accompanying test specifies, which encodes the stated visual goal
    of green angular compactness.)

    Falls back to PCA on the total scatter if no green articles exist
    or the green scatter is too low-rank.
    """
    assert seed_embedding.ndim == 1
    assert gap_article_embeddings.ndim == 2
    assert book_embeddings.ndim == 2
    assert proposal_embedding.ndim == 1

    # Center everything at the seed.
    gap_c = gap_article_embeddings - seed_embedding
    books_c = book_embeddings - seed_embedding
    proposal_c = proposal_embedding - seed_embedding

    # Build M_total: all displayed entities (gap articles + books + proposal).
    M_total = np.vstack([gap_c, books_c, proposal_c[None, :]])
    # S_total: outer product (no scaling needed — only ratios matter).
    S_total = M_total.T @ M_total

    # Pick angular axes (u1, u2) ∈ ℝ^(D × 2).
    u1u2 = _choose_angular_axes(
        S_total=S_total,
        M_green=gap_c[green_mask] if green_mask.any() else None,
    )

    def _aep(centered: NDArray[np.floating]) -> NDArray[np.float64]:
        """Apply AEP to centered (i.e. x − seed) vectors."""
        # 2D lateral projection.
        lateral = centered @ u1u2  # shape (N, 2)
        # Radial: ND distance from seed.
        rho = np.linalg.norm(centered, axis=-1)  # shape (N,) or scalar
        # Angles.
        lateral_norms = np.linalg.norm(lateral, axis=-1)
        # Avoid /0: where lateral is zero, set angle to 0 (degenerate fallback).
        safe = lateral_norms > 1e-12
        scaled = np.zeros_like(lateral, dtype=np.float64)
        scaled[safe] = (lateral[safe].astype(np.float64)
                        * (rho[safe] / lateral_norms[safe])[:, None])
        # For non-safe points, leave at origin (will sit at the seed, which is
        # honest: their entire deviation from the seed lies in directions we
        # cannot show).
        return scaled

    def _aep_single(centered_vec: NDArray[np.floating]) -> NDArray[np.float64]:
        """AEP for a single vector. Returns shape (2,)."""
        result = _aep(centered_vec[None, :])
        return result[0]

    return LocalProjection2DResult(
        seed_2d=np.array([0.0, 0.0]),
        proposal_2d=_aep_single(proposal_c),
        gap_articles_2d=_aep(gap_c),
        books_2d=_aep(books_c),
    )


def _choose_angular_axes(
    *,
    S_total: NDArray[np.floating],
    M_green: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    """Solve `S_green v = λ S_total v` and return the top-2 generalized
    eigenvectors as columns (i.e. directions of largest green variance
    per unit total variance). Falls back to PCA of S_total if M_green is
    None, has fewer than 3 rows, or the GEP fails numerically.
    """
    d = S_total.shape[0]
    if M_green is None or M_green.shape[0] < 3:
        # Degenerate: PCA of S_total.
        return _top2_eigenvectors_psd(S_total)
    S_green = M_green.T @ M_green
    # Add small ridge to S_total for numerical stability (it appears as
    # the RHS metric matrix in the GEP and must be SPD).
    ridge = max(1e-8, 1e-6 * float(np.trace(S_total)) / d)
    S_total_reg = S_total + ridge * np.eye(d)
    try:
        eigvals, eigvecs = eigh(a=S_green, b=S_total_reg)
        # eigh returns eigvals in ascending order. We want the LARGEST
        # (largest ratio of green-scatter to total-scatter along v).
        top2 = eigvecs[:, -2:]
        # Reorder so column 0 corresponds to the largest eigenvalue.
        top2 = top2[:, ::-1].astype(np.float64)
        return top2
    except LinAlgError:
        # Fall back to PCA.
        return _top2_eigenvectors_psd(S_total)


def _top2_eigenvectors_psd(S: NDArray[np.floating]) -> NDArray[np.float64]:
    eigvals, eigvecs = eigh(S)
    top2 = eigvecs[:, -2:][:, ::-1].astype(np.float64)
    return top2
