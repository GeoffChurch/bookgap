"""Multi-view spectral clustering for the curriculum pipeline.

Combines embedding-similarity and citation-similarity views via the
co-regularized spectral framework of Kumar-Rai-Daumé (2011), with
CSLS hubness correction (Conneau et al. 2017) applied to each view
first.

Operates on dense (n, n) similarity matrices; assumes n ≤ ~10k.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


def csls_correct(
    sim_matrix: NDArray[np.floating],
    k: int,
    clip: bool = True,
) -> NDArray[np.floating]:
    """Apply CSLS hubness correction to a dense similarity matrix.

    For each row i, computes ``r_K(i) = mean of i's top-k similarities to
    other points (self excluded)``. The CSLS-corrected similarity is::

        CSLS(i, j) = 2 * sim(i, j) - r_K(i) - r_K(j)

    Reference: Conneau, Lample, Ranzato, Denoyer, Jégou (2017),
    "Word Translation Without Parallel Data".

    Parameters
    ----------
    sim_matrix : (n, n) symmetric similarity matrix.
    k : Number of neighbours to average for r_K. Must be < n.
    clip : If True, clip negative entries to 0 (PSD-friendly for
        downstream Laplacian operations).
    """
    n = sim_matrix.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than n={n}")
    sim_no_self = sim_matrix.astype(np.float64, copy=True)
    np.fill_diagonal(sim_no_self, -np.inf)
    # argpartition leaves the top-k unsorted in the last k columns.
    top_k_idx = np.argpartition(sim_no_self, -k, axis=1)[:, -k:]
    top_k_vals = np.take_along_axis(sim_no_self, top_k_idx, axis=1)
    r_k = top_k_vals.mean(axis=1)
    csls = 2.0 * sim_matrix - r_k[:, None] - r_k[None, :]
    if clip:
        csls = np.maximum(csls, 0.0)
    return csls


def build_embedding_similarity_view(
    embeddings: NDArray[np.floating],
    csls_k: int = 10,
) -> NDArray[np.floating]:
    """Build a CSLS-corrected dense embedding-similarity view.

    Embeddings are L2-normalised; pairwise similarity is cosine; CSLS
    hubness correction is applied with the given k. Result is symmetric,
    non-negative, shape (n, n).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # guard zero-norm rows
    emb_norm = embeddings / norms
    sim = emb_norm @ emb_norm.T
    return csls_correct(sim, k=csls_k, clip=True)
