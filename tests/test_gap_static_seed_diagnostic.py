"""Tests for the static-seed cohesion diagnostic's math primitives.

We verify the four load-bearing functions on synthetic inputs:
- generalized_jaccard / generalized_containment on hand-constructed weight vectors
- joint_normalize on a tiny seed/article overlap pattern
- kernel_mass_matrix on a 1D embedding (so distances are obvious)
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from artifacts_scripts.gap_static_seed_diagnostic import (
    generalized_jaccard,
    generalized_containment,
    joint_normalize,
    kernel_mass_matrix,
)


def test_generalized_jaccard_identical_vectors_returns_one():
    a = np.array([0.5, 0.3, 0.2, 0.0])
    b = np.array([0.5, 0.3, 0.2, 0.0])
    assert generalized_jaccard(a, b) == pytest.approx(1.0)


def test_generalized_jaccard_disjoint_supports_returns_zero():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    assert generalized_jaccard(a, b) == pytest.approx(0.0)


def test_generalized_jaccard_subset_small_in_large_below_half():
    # Small subset (norm 1) fully nested in large set (norm 4): J = 1/4
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    assert generalized_jaccard(a, b) == pytest.approx(0.25)


def test_generalized_containment_subset_in_superset_returns_one():
    # Small a fully covered (in min) by b -> C = sum(a)/sum(a) = 1.0
    a = np.array([0.5, 0.5, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    assert generalized_containment(a, b) == pytest.approx(1.0)


def test_generalized_containment_distinguishes_nested_from_jaccard():
    # Same vectors as the Jaccard subset test: J = 0.25 but C = 1.0.
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    assert generalized_jaccard(a, b) == pytest.approx(0.25)
    assert generalized_containment(a, b) == pytest.approx(1.0)


def test_joint_normalize_two_seeds_share_one_article_splits_evenly():
    # 3 articles, 2 seeds.
    # seed 0 claims articles 0,1 with weight 1.0 each.
    # seed 1 claims articles 1,2 with weight 1.0 each.
    # Article 1 is shared: should split 0.5/0.5. effective_size = [1.5, 1.5].
    rows = [0, 1, 1, 2]
    cols = [0, 0, 1, 1]
    vals = [1.0, 1.0, 1.0, 1.0]
    M = csr_matrix((vals, (rows, cols)), shape=(3, 2))
    M_norm, eff = joint_normalize(M)
    assert eff == pytest.approx([1.5, 1.5])
    assert M_norm[0, 0] == pytest.approx(1.0)
    assert M_norm[1, 0] == pytest.approx(0.5)
    assert M_norm[1, 1] == pytest.approx(0.5)
    assert M_norm[2, 1] == pytest.approx(1.0)


def test_kernel_mass_matrix_known_distances():
    # 3 seeds, each has top-2 article neighbors with known squared distances.
    # sigma = 1.0 for all -> kernel = exp(-d²).
    # k_grid = [1, 2] -> cumulative kernel sums at each k.
    D = np.array([
        [0.0, 1.0],  # seed 0: neighbor distances² 0, 1
        [0.5, 2.0],  # seed 1
        [1.0, 1.0],  # seed 2
    ], dtype=np.float32)
    # I_global irrelevant for kernel value; we just need sigma_neighbor lookup.
    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    I_global = np.array([
        [1, 2],
        [0, 2],
        [0, 1],
    ], dtype=np.int64)
    k_grid = [1, 2]
    M = kernel_mass_matrix(D, I_global, sigma, k_grid)
    # M[i, j] = cumulative sum of exp(-D[i, :j+1]) (assuming sigma_seed * sigma_n = 1)
    expected = np.array([
        [np.exp(-0.0), np.exp(-0.0) + np.exp(-1.0)],
        [np.exp(-0.5), np.exp(-0.5) + np.exp(-2.0)],
        [np.exp(-1.0), np.exp(-1.0) + np.exp(-1.0)],
    ])
    np.testing.assert_allclose(M, expected, rtol=1e-5)
