"""Test equivalence of sparse and dense GEP solvers for CitationErrorLDA."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from bookgap.denoisers import (
    CitationErrorLDADenoiser,
    DenoiserContext,
    _solve_gep_sparse,
    scatter,
)


class _DenseContext(DenoiserContext):
    """Minimal context backed by a dense matrix, for testing."""

    def __init__(self, X: np.ndarray, A: sp.csr_matrix):
        super().__init__()
        self._X = X
        self._A = A

    def A(self) -> sp.csr_matrix:
        return self._A

    def X(self) -> np.ndarray:
        return self._X

    def X_n(self, n: int) -> np.ndarray:
        raise IndexError("no subsets")

    def num_subsets(self) -> int:
        return 0


class _SparseContext(DenoiserContext):
    """Minimal context backed by a sparse matrix, for testing."""

    def __init__(self, X: sp.csr_matrix, A: sp.csr_matrix):
        super().__init__()
        self._X = X
        self._A = A

    def A(self) -> sp.csr_matrix:
        return self._A

    def X(self) -> sp.csr_matrix:
        return self._X

    def S_C(self) -> np.ndarray:
        X_dense = np.asarray(self._X.todense(), dtype=np.float64)
        cited_means = self._A @ X_dense
        D_error = X_dense - cited_means
        return scatter(D_error, centered=False)

    def X_n(self, n: int) -> np.ndarray:
        raise IndexError("no subsets")

    def num_subsets(self) -> int:
        return 0


def _make_test_data(N: int = 50, V: int = 30, k: int = 5, density: float = 0.3, seed: int = 42):
    """Generate a small sparse TF matrix and citation graph for testing."""
    rng = np.random.RandomState(seed)

    # Sparse TF matrix (non-negative integers)
    X_dense = rng.poisson(lam=1.0, size=(N, V)).astype(np.float64)
    # Zero out some entries to make it sparse
    mask = rng.random((N, V)) > density
    X_dense[mask] = 0
    X_sparse = sp.csr_matrix(X_dense)

    # Random citation graph: sparse, row-stochastic with self-loops
    edges = rng.random((N, N)) < 0.05
    np.fill_diagonal(edges, True)
    A_dense = edges.astype(np.float64)
    row_sums = A_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_dense /= row_sums
    A_sparse = sp.csr_matrix(A_dense)

    return X_dense, X_sparse, A_sparse, k


def _subspace_similarity(P1: np.ndarray, P2: np.ndarray) -> float:
    """Measure how much the column spaces of P1 and P2 overlap.

    Returns the mean of the singular values of P1.T @ P2, which is 1.0
    when the column spaces are identical.
    """
    # Orthonormalize both
    Q1, _ = np.linalg.qr(P1)
    Q2, _ = np.linalg.qr(P2)
    Q1 = Q1[:, :P1.shape[1]]
    Q2 = Q2[:, :P2.shape[1]]
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(svals.mean())


class TestSparseGepEquivalence:
    """Verify sparse and dense GEP solvers produce equivalent subspaces."""

    def test_subspace_equivalence(self):
        """Sparse and dense solvers should find similar subspaces.

        Eigenvalues may differ because the two paths use different
        regularization strategies (ridge on S_signal vs ridge on S_C),
        but the subspaces they identify should overlap substantially.
        """
        X_dense, X_sparse, A, k = _make_test_data()

        # Dense path: use dense X so CitationErrorLDADenoiser takes primal path
        dense_ctx = _DenseContext(X_dense, A)
        denoiser = CitationErrorLDADenoiser()
        P_dense, eigvals_dense = denoiser.projector(dense_ctx, k)

        # Sparse path: use sparse X to trigger _solve_gep_sparse
        sparse_ctx = _SparseContext(X_sparse, A)
        P_sparse, eigvals_sparse = denoiser.projector(sparse_ctx, k)

        # Subspaces should overlap substantially (eigenvectors may differ in
        # sign/rotation due to different regularization, but the span should
        # capture the same low-citation-error directions).
        sim = _subspace_similarity(P_dense, P_sparse)
        assert sim > 0.7, f"Subspace similarity {sim:.3f} is too low (expected > 0.7)"

    def test_sparse_projector_rayleigh_quotient(self):
        """The sparse projector should have a small Rayleigh quotient S_C/S_signal."""
        _, X_sparse, A, k = _make_test_data(N=200, V=50, k=5, density=0.5)
        res = _solve_gep_sparse(X_sparse, A, k)
        P = res.P_or_U  # V x k

        # Compute S_C and S_T via dense for verification
        X_dense = X_sparse.toarray()
        X_c = X_dense - X_dense.mean(axis=0)
        E = X_dense - A @ X_dense
        E_c = E - E.mean(axis=0)
        S_T = X_c.T @ X_c / X_dense.shape[0]
        S_C = E_c.T @ E_c / X_dense.shape[0]

        # Rayleigh quotient: trace(P.T S_C P) / trace(P.T S_T P)
        # Should be small for a good citation-error-minimizing projection
        num = np.trace(P.T @ S_C @ P)
        den = np.trace(P.T @ S_T @ P)
        ratio = num / den if den > 0 else float("inf")

        # Ratio should be < 1 (citation error is less than total variance in projected space)
        assert ratio < 1.0, f"Rayleigh quotient {ratio:.4f} should be < 1.0"

    def test_eigenvalues_are_small(self):
        """The sparse solver should return the smallest eigenvalues."""
        _, X_sparse, A, k = _make_test_data()
        res = _solve_gep_sparse(X_sparse, A, k)
        # All eigenvalues should be non-negative (S_C is PSD)
        assert np.all(res.eigvals >= -1e-6), f"Negative eigenvalues: {res.eigvals}"
        # They should be sorted ascending
        assert np.all(np.diff(res.eigvals) >= -1e-10), "Eigenvalues not sorted ascending"

    def test_projector_shape(self):
        """Output projector should have shape (V, k)."""
        _, X_sparse, A, k = _make_test_data(V=30, k=5)
        res = _solve_gep_sparse(X_sparse, A, k)
        assert res.P_or_U.shape == (30, 5)

    def test_different_sizes(self):
        """Test with N > V and N < V."""
        for N, V in [(100, 50), (50, 100)]:
            _, X_sparse, A, k = _make_test_data(N=N, V=V, k=5)
            res = _solve_gep_sparse(X_sparse, A, k)
            assert res.P_or_U.shape == (V, k)
            assert len(res.eigvals) == k
