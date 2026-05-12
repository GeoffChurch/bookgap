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


def test_section1_returns_decision_on_synthetic_state(tmp_path):
    """End-to-end smoke: synthetic state -> section1 emits a decision JSON."""
    from artifacts_scripts.gap_static_seed_diagnostic import (
        section1_surprise_curves, K_GRID, K_QUERY,
    )
    rng = np.random.default_rng(0)
    N = 200
    art_idx = np.arange(150)
    book_idx = np.arange(150, 200)
    sigma = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    # Each seed's top-K_QUERY neighbors: random article ids; distances random.
    D = rng.uniform(0.0, 2.0, size=(N, K_QUERY)).astype(np.float32)
    D.sort(axis=1)
    I = rng.integers(0, 150, size=(N, K_QUERY)).astype(np.int64)
    state = {
        "sigma": sigma,
        "D_art_NN_all": D,
        "I_art_NN_all_global": art_idx[I % 150],
        "article_global_idx": art_idx,
        "book_global_idx": book_idx,
    }
    sample_seeds = {
        "cohesive": art_idx[:5].tolist(),
        "random": art_idx[5:15].tolist(),
        "book": book_idx[:3].tolist(),
    }
    decision = section1_surprise_curves(state, sample_seeds, tmp_path)
    assert decision["section"] == 1
    assert decision["recommended_metric"] in ("LLR", "z-score")
    assert (tmp_path / "curves_summary.csv").exists()
    assert (tmp_path / "decision.json").exists()


def test_section2_decay_comparison_returns_decision(tmp_path):
    from artifacts_scripts.gap_static_seed_diagnostic import (
        section2_decay_comparison, K_QUERY,
    )
    rng = np.random.default_rng(1)
    N = 100
    art_idx = np.arange(80)
    sigma = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    D = rng.uniform(0.0, 2.0, size=(N, K_QUERY)).astype(np.float32); D.sort(axis=1)
    I = rng.integers(0, 80, size=(N, K_QUERY)).astype(np.int64)
    state = {
        "sigma": sigma, "D_art_NN_all": D, "I_art_NN_all_global": art_idx[I % 80],
        "article_global_idx": art_idx, "book_global_idx": np.arange(80, 100),
    }
    sample_seeds = {"random": art_idx[:5].tolist()}
    k_star = {int(s): 20 for s in art_idx[:5]}
    decision = section2_decay_comparison(state, sample_seeds, k_star, tmp_path)
    assert decision["section"] == 2
    assert (tmp_path / "decay_summary.csv").exists()


def test_section3_recentering_returns_decision(tmp_path):
    """Synthetic state needs a real article_index; build a minimal FlatL2."""
    from bookgap.index import build_index, FlatL2
    from artifacts_scripts.gap_static_seed_diagnostic import (
        section3_recentering, K_QUERY,
    )
    rng = np.random.default_rng(2)
    N = 80
    D = 16
    emb = rng.standard_normal(size=(N, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    art_idx = np.arange(60)
    article_index = build_index(FlatL2(), emb[art_idx])
    sigma = rng.uniform(0.5, 1.5, size=N).astype(np.float32)
    Dmat = rng.uniform(0.0, 2.0, size=(N, K_QUERY)).astype(np.float32); Dmat.sort(axis=1)
    Imat = rng.integers(0, 60, size=(N, K_QUERY)).astype(np.int64)
    state = {
        "emb": emb, "sigma": sigma,
        "D_art_NN_all": Dmat, "I_art_NN_all_global": art_idx[Imat % 60],
        "article_global_idx": art_idx, "book_global_idx": np.arange(60, N),
        "article_index": article_index,
    }
    sample_seeds = {"random": art_idx[:5].tolist()}
    k_star = {int(s): 15 for s in art_idx[:5]}
    decision = section3_recentering(state, sample_seeds, k_star, tmp_path)
    assert decision["section"] == 3
    assert decision["recommended_step"] in ("static", "one-recentering")
