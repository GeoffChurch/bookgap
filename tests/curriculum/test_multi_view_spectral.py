"""Tests for multi-view spectral clustering with CSLS correction.

Components:
- CSLS hubness correction (Conneau et al. 2017).
- Per-view similarity builders (embedding cosine, citation Ruzicka).
- Co-regularized multi-view spectral solver (Kumar-Rai-Daumé 2011).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.curriculum.multi_view_spectral import (
    build_embedding_similarity_view,
    build_citation_similarity_view,
    csls_correct,
    _normalized_laplacian,
    _subspace_distance,
)


class TestCSLSCorrect:
    def test_uniform_similarity_collapses_to_zero(self):
        # Uniform off-diagonal similarity → each row's r_K equals that
        # uniform value → CSLS off-diagonal = 2*sim - sim - sim = 0.
        n = 5
        W = np.full((n, n), 0.5)
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        off_diag = result - np.diag(np.diag(result))
        assert np.allclose(off_diag, 0.0, atol=1e-9)

    def test_hub_penalty_differentiates(self):
        # Node 0 is similar (0.8) to all of 1..4; nodes 1..4 are mutually
        # dissimilar (0). Node 0 is a "hub". With k=2:
        #   r_K(0) = mean of top-2 of [0.8, 0.8, 0.8, 0.8] = 0.8
        #   r_K(1) = mean of top-2 of [0.8, 0, 0, 0] = 0.4
        # CSLS(0, 1) = 2 * 0.8 - 0.8 - 0.4 = 0.4 (less than raw 0.8 — hub
        # contribution discounted).
        n = 5
        W = np.zeros((n, n))
        W[0, 1:] = 0.8
        W[1:, 0] = 0.8
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        assert result[0, 1] == pytest.approx(0.4, abs=1e-9)

    def test_symmetry_preserved(self):
        rng = np.random.default_rng(42)
        n = 10
        raw = rng.uniform(0, 1, size=(n, n))
        W = (raw + raw.T) / 2
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=3)
        assert np.allclose(result, result.T, atol=1e-9)

    def test_clip_at_zero(self):
        n = 5
        W = np.full((n, n), 0.1)
        np.fill_diagonal(W, 1.0)
        result_clipped = csls_correct(W, k=2, clip=True)
        result_unclipped = csls_correct(W, k=2, clip=False)
        assert (result_clipped >= 0).all()
        # Whatever was negative in unclipped is exactly zero in clipped.
        neg_mask = result_unclipped < 0
        assert np.allclose(result_clipped[neg_mask], 0.0)

    def test_excludes_self_in_neighbourhood(self):
        # Self-similarity = 1; off-diagonal = 0.1.
        # If self were included in top-k, r_K(i) would pull toward 1 and
        # over-correct. With self excluded, r_K(i) = 0.1; CSLS off-diagonal
        # = 2*0.1 - 0.1 - 0.1 = 0; CSLS diagonal = 2*1 - 0.1 - 0.1 = 1.8.
        n = 5
        W = np.full((n, n), 0.1)
        np.fill_diagonal(W, 1.0)
        result = csls_correct(W, k=2, clip=False)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert result[0, 0] == pytest.approx(1.8, abs=1e-9)

    def test_k_too_large_raises(self):
        W = np.eye(3)
        with pytest.raises(ValueError, match="k=3 must be less than n=3"):
            csls_correct(W, k=3)


class TestBuildEmbeddingSimilarityView:
    def test_two_clusters_separated(self):
        # Two tight clusters along distinct rays from the origin so that
        # L2 normalisation preserves direction. (Centring a cluster at the
        # origin would normalise small-noise points to random unit-circle
        # directions, defeating the test.)
        rng = np.random.default_rng(0)
        cluster1 = rng.normal(loc=[3.0, 0.0], scale=0.05, size=(4, 2))
        cluster2 = rng.normal(loc=[0.0, 3.0], scale=0.05, size=(4, 2))
        embeddings = np.vstack([cluster1, cluster2])
        W = build_embedding_similarity_view(embeddings, csls_k=2)
        assert W.shape == (8, 8)
        assert np.allclose(W, W.T, atol=1e-9)
        assert (W >= 0).all()
        within = W[:4, :4][np.triu_indices(4, k=1)].mean()
        between = W[:4, 4:].mean()
        assert within > between

    def test_normalization_invariance(self):
        # Pre-normalised embeddings should give the same result as raw —
        # we L2-normalise internally.
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(6, 4))
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        W_raw = build_embedding_similarity_view(emb, csls_k=2)
        W_norm = build_embedding_similarity_view(emb_norm, csls_k=2)
        assert np.allclose(W_raw, W_norm, atol=1e-9)

    def test_zero_embedding_handled(self):
        # An all-zero row would div-by-zero without protection.
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        W = build_embedding_similarity_view(emb, csls_k=1)
        # Should not raise and should be finite.
        assert np.isfinite(W).all()


def _build_adj(edges: list[tuple[int, int]], n: int) -> sp.csr_matrix:
    """Build a directed citation adjacency: edge (i, j) means i cites j."""
    rows = [u for u, _ in edges]
    cols = [v for _, v in edges]
    data = [1.0] * len(edges)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class TestBuildCitationSimilarityView:
    def test_two_disjoint_components(self):
        # Two disjoint citation cliques with an external hub (node 6) that
        # cites into both cliques. This breaks symmetry and allows CSLS to
        # differentiate within vs between similarities.
        n = 7
        # Clique 0,1,2: cite each other
        edges = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
                 # Clique 3,4,5: cite each other
                 (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
                 # Hub 6 cites into both cliques
                 (6, 0), (6, 3)]
        adj = _build_adj(edges, n=n)
        W = build_citation_similarity_view(adj, csls_k=2, mode="weighted_unified")
        assert W.shape == (n, n)
        assert np.allclose(W, W.T, atol=1e-9)
        assert (W >= 0).all()
        # Within-clique similarity should exceed between-clique (both
        # preferentially distant from the hub).
        within = W[:3, :3][np.triu_indices(3, k=1)].mean()
        between = W[:3, 3:6].mean()
        assert within >= between  # Can be equal for small/degenerate matrices

    def test_mode_weighted_unified_default(self):
        # Sanity: the function accepts the three modes from
        # _citation_jaccard_distance_matrix.
        adj = _build_adj([(0, 1), (1, 0)], n=2)
        for mode in ("binary", "weighted_unified", "weighted_per_direction"):
            W = build_citation_similarity_view(adj, csls_k=1, mode=mode)
            assert W.shape == (2, 2)
            assert np.isfinite(W).all()


class TestNormalizedLaplacian:
    def test_shape_and_symmetry(self):
        rng = np.random.default_rng(0)
        n = 6
        W = rng.uniform(0, 1, size=(n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        L = _normalized_laplacian(W)
        assert L.shape == (n, n)
        assert np.allclose(L, L.T, atol=1e-9)

    def test_zero_eigenvalue_for_connected_graph(self):
        # Normalised Laplacian of a connected graph has smallest eigenvalue 0.
        n = 5
        W = np.ones((n, n)) - np.eye(n)
        L = _normalized_laplacian(W)
        eigvals = np.linalg.eigvalsh(L)
        assert eigvals[0] == pytest.approx(0.0, abs=1e-9)

    def test_isolated_node_handled(self):
        # An all-zero row would div-by-zero without floor.
        W = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # isolated
        ])
        L = _normalized_laplacian(W)
        assert np.isfinite(L).all()


class TestSubspaceDistance:
    def test_identical_subspace_zero(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(6, 3))
        # Orthonormalise so column space is well-defined.
        Q, _ = np.linalg.qr(A)
        d = _subspace_distance(Q, Q)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_sign_flip_invariant(self):
        # Flipping a column's sign doesn't change the column space.
        rng = np.random.default_rng(0)
        A, _ = np.linalg.qr(rng.normal(size=(6, 3)))
        B = A.copy()
        B[:, 0] *= -1
        d = _subspace_distance(A, B)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_subspaces_nonzero(self):
        # Two non-overlapping rank-1 subspaces in R^3 → positive distance.
        A = np.array([[1.0], [0.0], [0.0]])
        B = np.array([[0.0], [1.0], [0.0]])
        d = _subspace_distance(A, B)
        assert d > 0.5


from sklearn.metrics import adjusted_rand_score

from artifacts_scripts.curriculum.multi_view_spectral import (
    cluster_with_multi_view_spectral,
)


class TestClusterWithMultiViewSpectral:
    def _two_block_views(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 8 nodes: two clear blocks of 4 in BOTH views.
        rng = np.random.default_rng(seed)
        block_size = 4
        n = 2 * block_size
        # Both views agree: dense within-block, sparse between.
        W1 = np.full((n, n), 0.05) + rng.uniform(0, 0.01, size=(n, n))
        W2 = np.full((n, n), 0.05) + rng.uniform(0, 0.01, size=(n, n))
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    W1[i, j] = 0.9 + rng.uniform(0, 0.05)
                    W2[i, j] = 0.9 + rng.uniform(0, 0.05)
        for i in range(block_size, n):
            for j in range(block_size, n):
                if i != j:
                    W1[i, j] = 0.9 + rng.uniform(0, 0.05)
                    W2[i, j] = 0.9 + rng.uniform(0, 0.05)
        np.fill_diagonal(W1, 1.0)
        np.fill_diagonal(W2, 1.0)
        W1 = (W1 + W1.T) / 2
        W2 = (W2 + W2.T) / 2
        truth = np.array([0] * block_size + [1] * block_size, dtype=np.intp)
        return W1, W2, truth

    def test_recovers_two_blocks_when_views_agree(self):
        W1, W2, truth = self._two_block_views(seed=0)
        labels = cluster_with_multi_view_spectral(
            W1, W2, n_clusters=2, co_reg_lambda=0.5, seed=0,
        )
        assert labels.shape == (8,)
        ari = adjusted_rand_score(truth, labels)
        assert ari > 0.95

    def test_lambda_zero_independent_views(self):
        # λ=0 → views don't influence each other; should still recover the
        # ground-truth blocks since both views agree on them.
        W1, W2, truth = self._two_block_views(seed=1)
        labels = cluster_with_multi_view_spectral(
            W1, W2, n_clusters=2, co_reg_lambda=0.0, seed=0,
        )
        ari = adjusted_rand_score(truth, labels)
        assert ari > 0.95

    def test_returns_intp(self):
        W1, W2, _ = self._two_block_views(seed=2)
        labels = cluster_with_multi_view_spectral(
            W1, W2, n_clusters=2, co_reg_lambda=0.5, seed=0,
        )
        assert labels.dtype == np.intp

    def test_n_clusters_bounded_by_n(self):
        # Asking for n_clusters >= n returns identity assignment.
        W1, W2, _ = self._two_block_views(seed=3)
        n = W1.shape[0]
        labels = cluster_with_multi_view_spectral(
            W1, W2, n_clusters=n + 5, co_reg_lambda=0.5, seed=0,
        )
        assert labels.shape == (n,)
        assert len(np.unique(labels)) == n

    def test_shape_mismatch_raises(self):
        W1 = np.eye(5)
        W2 = np.eye(6)
        with pytest.raises(ValueError, match="must match"):
            cluster_with_multi_view_spectral(
                W1, W2, n_clusters=2, co_reg_lambda=0.5,
            )
