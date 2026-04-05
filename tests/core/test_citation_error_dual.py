from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from bookgap.denoisers import CitationErrorLDADenoiser, scatter, adj


class _SparseContext:
    """Minimal DenoiserContext for testing with N < d."""
    def __init__(self, X, adjacency):
        from bookgap.denoisers import CenteredMatrix
        self._X = X
        self._A = adjacency
        mean = np.asarray(X.mean(axis=0)) if sp.issparse(X) else X.mean(axis=0, keepdims=True)
        self._centered = CenteredMatrix(X=X, mean=np.atleast_2d(mean))

    def X(self):
        return self._X

    def A(self):
        return self._A

    def X_centered(self):
        return self._centered.X_centered_dense()

    def X_mean(self):
        return self._centered.mean

    def centered(self):
        return self._centered

    def A_sym(self):
        return self._A + self._A.T

    def S_T(self):
        X_c = self.X_centered()
        return scatter(X_c, centered=True)

    def S_C(self):
        X = np.asarray(self._X) if sp.issparse(self._X) else self._X
        cited_means = self._A @ X
        D_error = X - cited_means
        return scatter(D_error, centered=False)

    def num_subsets(self):
        return 0

    def X_n(self, n):
        raise IndexError


class TestCitationErrorLDADual:
    def test_dual_matches_primal_on_small_problem(self):
        """When N < d, the dual formulation should produce equivalent projections."""
        rng = np.random.default_rng(42)
        N, d = 20, 8  # N > d so primal works
        X = rng.normal(size=(N, d))

        # Chain graph with self-loops, row-normalized
        rows, cols = [], []
        for i in range(N):
            rows.append(i); cols.append(i)
            if i + 1 < N:
                rows.extend([i, i + 1]); cols.extend([i + 1, i])
        data = np.ones(len(rows))
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        A = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A

        ctx = _SparseContext(X, A)
        n_components = 4

        P_primal, eigvals_primal = CitationErrorLDADenoiser().projector(ctx, n_components)

        # Now make it N < d by padding with zeros (the projection into the
        # padded dimensions should be near-zero)
        X_wide = np.hstack([X, rng.normal(size=(N, 30)) * 0.001])
        A_wide = A  # same graph

        ctx_wide = _SparseContext(X_wide, A_wide)
        P_wide, eigvals_wide = CitationErrorLDADenoiser().projector(ctx_wide, n_components)

        # The eigenvalues should be similar (the padded noise dimensions
        # should have tiny eigenvalues and not be selected)
        assert P_wide.shape == (d + 30, n_components)
        assert eigvals_wide.shape == (n_components,)

        # Eigenvalues from dual should approximately match primal
        # (the padded noise dimensions don't affect the dominant eigenstructure)
        assert_allclose(eigvals_wide[:n_components], eigvals_primal, atol=0.1, rtol=0.5)

    def test_sparse_x_does_not_crash(self):
        """CitationErrorLDADenoiser should work with sparse X when N < V."""
        rng = np.random.default_rng(42)
        N, V = 30, 200
        X_sparse = sp.random(N, V, density=0.1, format='csr', random_state=rng)

        rows, cols = [], []
        for i in range(N):
            rows.append(i); cols.append(i)
            if i + 1 < N:
                rows.extend([i, i + 1]); cols.extend([i + 1, i])
        data = np.ones(len(rows))
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        A = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A

        ctx = _SparseContext(X_sparse, A)
        P, eigvals = CitationErrorLDADenoiser().projector(ctx, n_components=10)
        assert P.shape[1] == 10
        assert eigvals.shape == (10,)
