"""Pure NumPy functions: post-proposal coverage analysis + AEP/GEP
local projection from the gap seed.

No I/O, no Cloud, no Hydra — caller supplies all embeddings as arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


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
    gap_size = gap_article_embeddings.shape[0]

    # Count per augmented-index, then split out proposal from books.
    counts = np.bincount(nearest, minlength=n_books + 1)
    per_book_after = counts[:n_books].astype(np.int64)
    proposal_covered = int(counts[n_books])

    return PostProposalCoverage(
        proposal_articles_covered=proposal_covered,
        proposal_fraction_covered=(proposal_covered / gap_size) if gap_size > 0 else 0.0,
        per_book_articles_covered_after=per_book_after,
    )


@dataclass(frozen=True)
class LocalProjection2DResult:
    seed_2d: NDArray[np.float64]            # shape (2,) — seed article's PCA projection
    proposal_2d: NDArray[np.float64]        # shape (2,)
    gap_articles_2d: NDArray[np.float64]    # shape (N_gap, 2)
    books_2d: NDArray[np.float64]           # shape (N_books, 2)


def compute_local_projection_2d(
    *,
    all_topic_article_embeddings: NDArray[np.floating],
    seed_embedding: NDArray[np.floating],
    gap_article_embeddings: NDArray[np.floating],
    book_embeddings: NDArray[np.floating],
    proposal_embedding: NDArray[np.floating],
) -> LocalProjection2DResult:
    """2D PCA on all topic articles; project the seed, gap articles, books,
    and proposal into the resulting plane.

    Articles are the natural reference cloud — they're the densest population
    in the topic and capture the dominant directions of within-topic
    semantic variation. Books and the proposal then land relative to that
    cloud, which is exactly the relationship the appendix wants to convey.

    Centering: at the all-article mean (the natural origin of the article
    cloud). PCA from ``scipy.linalg.eigh`` on a symmetric PSD matrix returns
    Euclidean-orthonormal eigenvectors, so 2D distances are honest
    lower-bounds for ND distances along the chosen plane.
    """
    assert all_topic_article_embeddings.ndim == 2
    assert seed_embedding.ndim == 1
    assert gap_article_embeddings.ndim == 2
    assert book_embeddings.ndim == 2
    assert proposal_embedding.ndim == 1

    center = all_topic_article_embeddings.mean(axis=0)
    centered = all_topic_article_embeddings - center
    S = centered.T @ centered
    _, eigvecs = eigh(S)
    axes = eigvecs[:, -2:][:, ::-1].astype(np.float64)

    def _project(x: NDArray[np.floating]) -> NDArray[np.float64]:
        return ((x - center) @ axes).astype(np.float64)

    return LocalProjection2DResult(
        seed_2d=_project(seed_embedding[None, :])[0],
        proposal_2d=_project(proposal_embedding[None, :])[0],
        gap_articles_2d=_project(gap_article_embeddings),
        books_2d=_project(book_embeddings),
    )
