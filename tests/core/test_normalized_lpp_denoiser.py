from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from scipy.linalg import eigh

from bookgap import denoisers


class _SimpleContext(denoisers.DenoiserContext):
    def __init__(self, embeddings: np.ndarray, adjacency: sp.csr_matrix):
        super().__init__()
        self._embeddings = embeddings
        self._adjacency = adjacency

    def A(self) -> sp.csr_matrix:
        return self._adjacency

    def X(self) -> np.ndarray:
        return self._embeddings

    def X_n(self, n: int) -> np.ndarray:  # pragma: no cover - not used by this context
        raise IndexError("No subsets provided in _SimpleContext")

    def num_subsets(self) -> int:
        return 0


def _build_chain_graph(num_nodes: int) -> sp.csr_matrix:
    row = []
    col = []
    for i in range(num_nodes):
        row.append(i)
        col.append(i)
        if i + 1 < num_nodes:
            row.extend([i, i + 1])
            col.extend([i + 1, i])
    data = np.ones(len(row))
    return sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes)) / 2


def test_normalized_lpp_matches_generalized_eigen_solution():
    rng = np.random.default_rng(1234)
    embeddings = rng.normal(size=(6, 4))

    # Build a simple chain graph so degrees vary
    adjacency = _build_chain_graph(embeddings.shape[0])

    context = _SimpleContext(embeddings, adjacency)
    projector, eigenvalues = denoisers.NormalizedLPPDenoiser().projector(context, n_components=2)

    # Manual generalized eigen-solution for comparison
    B = context.A_sym()
    degrees = np.asarray(B.sum(axis=1)).ravel()
    inv_sqrt_deg = np.zeros_like(degrees)
    mask = degrees > 0
    inv_sqrt_deg[mask] = np.power(degrees[mask], -0.5)
    D_inv_sqrt = sp.diags(inv_sqrt_deg)
    L_sym = sp.eye(B.shape[0]) - (D_inv_sqrt @ B @ D_inv_sqrt)
    D = sp.diags(degrees)

    emb_centered = context.X_centered()
    a_matrix = emb_centered.T @ L_sym @ emb_centered
    b_matrix = emb_centered.T @ D @ emb_centered

    evals_expected, evecs_expected = eigh(a_matrix, b_matrix)
    order = np.argsort(evals_expected)
    evals_expected = evals_expected[order][:2]
    evecs_expected = evecs_expected[:, order][:, :2]

    # Compare eigenvalues ignoring ordering
    assert np.allclose(np.sort(eigenvalues), np.sort(evals_expected[::-1]), atol=1e-6)

    # The returned projector columns should span the same subspace (allowing sign flips)
    span_distance = norm(projector @ projector.T - evecs_expected @ evecs_expected.T)
    assert span_distance < 1e-5

    # Verify orthonormality in the degree-weighted metric
    gram = projector.T @ (emb_centered.T @ D @ emb_centered) @ projector
    assert np.allclose(gram, np.eye(2), atol=1e-6)


def test_normalized_lpp_handles_zero_degree_nodes():
    embeddings = np.eye(3)
    adjacency = sp.csr_matrix((3, 3))  # All degrees zero
    context = _SimpleContext(embeddings, adjacency)

    projector, eigenvalues = denoisers.NormalizedLPPDenoiser().projector(context, n_components=2)

    # With zero edges, the Laplacian defaults to identity and the constraint to identity
    # The solution should effectively fall back to PCA of the embeddings (which are orthonormal basis vectors)
    assert projector.shape == (3, 2)
    emb_centered = context.X_centered()
    gram = projector.T @ (emb_centered.T @ emb_centered) @ projector
    assert np.allclose(gram[[0, 1], [1, 0]], 0, atol=1e-6)
    assert np.all(0.5 <= np.diag(np.abs(gram)))
    assert np.all(eigenvalues >= 0)
