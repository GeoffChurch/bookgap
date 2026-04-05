# tests/core/test_centered_matrix.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from bookgap.denoisers import CenteredMatrix


class TestCenteredMatrixGram:
    def test_dense_gram_matches_explicit(self):
        """gram() on dense X should match X_c @ X_c.T computed explicitly."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 5))
        mean = X.mean(axis=0, keepdims=True)
        cm = CenteredMatrix(X=X, mean=mean)
        X_c = X - mean
        expected = X_c @ X_c.T
        assert_allclose(cm.gram(), expected, atol=1e-10)

    def test_sparse_gram_matches_dense(self):
        """gram() on sparse X should match gram() on equivalent dense X."""
        rng = np.random.default_rng(42)
        X_dense = rng.poisson(lam=0.5, size=(20, 100)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)
        mean = np.asarray(X_dense.mean(axis=0, keepdims=True))
        cm_dense = CenteredMatrix(X=X_dense, mean=mean)
        cm_sparse = CenteredMatrix(X=X_sparse, mean=mean)
        assert_allclose(cm_sparse.gram(), cm_dense.gram(), atol=1e-10)

    def test_sparse_n_less_than_d(self):
        """gram() works correctly when N < d (the TF-GEP regime)."""
        rng = np.random.default_rng(42)
        N, V = 50, 500
        X_dense = rng.poisson(lam=0.3, size=(N, V)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)
        mean = np.asarray(X_dense.mean(axis=0, keepdims=True))
        cm = CenteredMatrix(X=X_sparse, mean=mean)
        X_c = X_dense - mean
        expected = X_c @ X_c.T
        result = cm.gram()
        assert result.shape == (N, N)
        assert_allclose(result, expected, atol=1e-9)

    def test_x_centered_dense_with_sparse_input(self):
        """X_centered_dense() should work correctly with sparse X."""
        rng = np.random.default_rng(42)
        X_dense = rng.poisson(lam=0.5, size=(10, 20)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)
        mean = np.asarray(X_dense.mean(axis=0, keepdims=True))
        cm = CenteredMatrix(X=X_sparse, mean=mean)
        result = cm.X_centered_dense()
        expected = X_dense - mean
        assert_allclose(result, expected, atol=1e-10)
