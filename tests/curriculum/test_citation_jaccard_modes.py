"""Tests for the three modes of ``_citation_jaccard_distance_matrix``.

Modes:
- ``"binary"``: ordinary Jaccard on the unweighted in+out neighbourhood.
- ``"weighted_unified"``: Ruzicka similarity on ``max(P, P^T)`` where
  ``P`` is the row-normalised random-walk transition matrix.
- ``"weighted_per_direction"``: Ruzicka on the concatenation of the
  per-node ℓ1-normalised out- and in-distributions.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.curriculum.build_curriculum import (
    _citation_jaccard_distance_matrix,
)


def _build_adj(edges: list[tuple[int, int]], n: int) -> sp.csr_matrix:
    """Build a directed citation adjacency: edge (i, j) means i cites j."""
    rows = [u for u, _ in edges]
    cols = [v for _, v in edges]
    data = [1.0] * len(edges)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# binary mode (regression — preserves the original behaviour)
# ---------------------------------------------------------------------------


class TestBinaryMode:
    def test_two_nodes_share_neighbour(self):
        # 0 → 2, 1 → 2: nodes 0 and 1 share out-neighbour 2.
        adj = _build_adj([(0, 2), (1, 2)], n=3)
        d = _citation_jaccard_distance_matrix(adj, mode="binary")
        # N(0) = {2}, N(1) = {2}, N(2) = {0, 1}.
        # Jaccard(0, 1) = 1/1 = 1 → d = 0.
        assert d[0, 1] == pytest.approx(0.0)
        # Jaccard(0, 2) = |{2} ∩ {0, 1}| / |{0, 1, 2}| = 0/3 → d = 1.
        assert d[0, 2] == pytest.approx(1.0)
        np.testing.assert_array_equal(np.diag(d), 0.0)

    def test_isolated_nodes_distance_one(self):
        adj = _build_adj([], n=3)
        d = _citation_jaccard_distance_matrix(adj, mode="binary")
        # All neighbourhoods empty → similarity 0 → distance 1 off-diag.
        off_diag = d - np.diag(np.diag(d))
        assert (off_diag == 1.0).sum() == 6


# ---------------------------------------------------------------------------
# weighted_unified mode
# ---------------------------------------------------------------------------


class TestWeightedUnified:
    def test_reduces_to_binary_when_degrees_equal(self):
        # Three nodes; each cites one other so out_deg = 1 for all.
        # 0 → 1, 1 → 2, 2 → 0.
        adj = _build_adj([(0, 1), (1, 2), (2, 0)], n=3)
        d_bin = _citation_jaccard_distance_matrix(adj, mode="binary")
        d_w = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        # All edges have weight 1 in P (since out_deg=1 each), and union
        # is via max so weights stay 1. Should equal the binary Jaccard.
        np.testing.assert_allclose(d_w, d_bin, atol=1e-12)

    def test_handcomputed_two_nodes_shared_target(self):
        # 0 → 2, 1 → 2 (and nothing else). out_deg = [1, 1, 0].
        # P[0, 2] = 1, P[1, 2] = 1, all else 0.
        # max(P, P^T): rows are
        #   0: {2: 1}                  (out)  + {} (in: nobody cites 0)
        #   1: {2: 1}                  (out)  + {} (in)
        #   2: {} (out, no out-edges) + {0: 1, 1: 1} (in)
        # Ruzicka(0, 1) = sum_min / sum_max = 1 / 1 = 1. d = 0.
        adj = _build_adj([(0, 2), (1, 2)], n=3)
        d = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        assert d[0, 1] == pytest.approx(0.0, abs=1e-12)
        # Ruzicka(0, 2): w_0 = {2: 1}, w_2 = {0: 1, 1: 1}. Disjoint
        # support → sum_min = 0, sum_max = 1 + 1 + 1 = 3. sim = 0, d = 1.
        assert d[0, 2] == pytest.approx(1.0, abs=1e-12)

    def test_dilutes_promiscuous_citers(self):
        # Build two nodes 0, 1 with the same shared in-neighbour 2 only —
        # but node 2 has very different out-degrees in the two cases. We
        # show that 2's contribution to (0, 1)'s Ruzicka similarity is
        # smaller when 2 cites many things.
        # Case A: 2 cites only 0 and 1 (out_deg(2) = 2).
        adj_a = _build_adj([(2, 0), (2, 1)], n=3)
        d_a = _citation_jaccard_distance_matrix(adj_a, mode="weighted_unified")
        # Case B: 2 cites 0, 1, 3, 4, 5, 6 (out_deg(2) = 6).
        adj_b = _build_adj(
            [(2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6)], n=7,
        )
        d_b = _citation_jaccard_distance_matrix(adj_b, mode="weighted_unified")
        # In both cases 0 and 1 share exactly the same support ({2}) and
        # the same weights (P[2, 0] = P[2, 1] = 1/out_deg(2)). Ruzicka
        # similarity is 1 → distance 0 in both.
        assert d_a[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert d_b[0, 1] == pytest.approx(0.0, abs=1e-12)
        # But 0's similarity to a *different* node — say node 3 in case
        # B (which is also cited by 2 but nothing else) — depends on
        # 2's out-degree: w_0[2] = 1/6, w_3[2] = 1/6, sum_min = 1/6,
        # sum_max = 1/6, sim = 1, d = 0. (Co-citation by the same set.)
        assert d_b[0, 3] == pytest.approx(0.0, abs=1e-12)

    def test_outdegree_mismatch_penalty(self):
        # u cites a single thing; v cites the same plus 5 others. Both
        # have only that one out-edge in common; no in-edges.
        # u: out = {0: 1};  v: out = {0: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
        # max(P, P^T) for u and v:
        #   w_u[0] = 1, w_v[0] = 1/6, w_v[2..6] = 1/6.
        # sum_min = min(1, 1/6) = 1/6
        # sum_max = max(1, 1/6) + 5 * 1/6 = 1 + 5/6 = 11/6
        # Ruzicka = (1/6) / (11/6) = 1/11.  d = 10/11.
        # u = 1, v = 7, target = 0, others = 2..6.
        adj = _build_adj(
            [(1, 0),  # u → 0
             (7, 0), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)],
            n=8,
        )
        d = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        sim = 1.0 - d[1, 7]
        assert sim == pytest.approx(1.0 / 11.0, abs=1e-12)

    def test_isolated_nodes_distance_one(self):
        adj = _build_adj([], n=3)
        d = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        off_diag = d - np.diag(np.diag(d))
        # With zero rows, denom = 0 → sim = 0 → d = 1 off-diag.
        assert (off_diag == 1.0).sum() == 6

    def test_symmetric(self):
        adj = _build_adj(
            [(0, 1), (1, 2), (2, 0), (3, 0), (3, 1), (3, 2)], n=4,
        )
        d = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        np.testing.assert_array_equal(np.diag(d), 0.0)
        assert d.min() >= 0.0
        assert d.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# weighted_per_direction mode
# ---------------------------------------------------------------------------


class TestWeightedPerDirection:
    def test_pure_out_overlap(self):
        # 0 → 2, 1 → 2. No in-edges to 0 or 1.
        # out-distributions: p_0 = {2: 1}, p_1 = {2: 1}.
        # in-distributions: empty for both 0 and 1.
        # Concat for 0: out + in (zeros) = single nonzero at "out:2" with weight 1.
        # Concat for 1: same. So Ruzicka(0, 1) = 1 → d = 0.
        adj = _build_adj([(0, 2), (1, 2)], n=3)
        d = _citation_jaccard_distance_matrix(
            adj, mode="weighted_per_direction",
        )
        assert d[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_separates_in_and_out_signals(self):
        # Asymmetric setup designed to show per-direction normalisation
        # behaves differently from weighted_unified.
        # Node 0 cites 2. Node 1 is cited by 2.
        # weighted_unified: w_0 = {2: 1} (out), w_1 = {2: 1} (in via P^T = 1/out_deg(2)=1)
        # → identical neighbourhoods → sim = 1.
        # weighted_per_direction: 0 has out-distribution {2: 1} only,
        # in-distribution empty. 1 has out-distribution empty,
        # in-distribution {2: 1}. The concatenated vectors put these in
        # disjoint blocks (out features vs in features), so sim = 0 and
        # d = 1.
        adj = _build_adj([(0, 2), (2, 1)], n=3)
        d_unif = _citation_jaccard_distance_matrix(adj, mode="weighted_unified")
        d_pdir = _citation_jaccard_distance_matrix(
            adj, mode="weighted_per_direction",
        )
        assert d_unif[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert d_pdir[0, 1] == pytest.approx(1.0, abs=1e-12)

    def test_handcomputed_co_citers(self):
        # Two papers (0, 1) cited by the same two papers (2, 3).
        # 2 → 0, 2 → 1, 3 → 0, 3 → 1.
        # in-distribution of 0 = {2: 1/2, 3: 1/2}.
        # in-distribution of 1 = {2: 1/2, 3: 1/2}.
        # out-distributions empty for both.
        # Concat for 0 = {in:2: 1/2, in:3: 1/2}, identical to 1.
        # → sim = 1, d = 0.
        adj = _build_adj([(2, 0), (2, 1), (3, 0), (3, 1)], n=4)
        d = _citation_jaccard_distance_matrix(
            adj, mode="weighted_per_direction",
        )
        assert d[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_isolated_nodes_distance_one(self):
        adj = _build_adj([], n=3)
        d = _citation_jaccard_distance_matrix(
            adj, mode="weighted_per_direction",
        )
        off_diag = d - np.diag(np.diag(d))
        assert (off_diag == 1.0).sum() == 6

    def test_symmetric(self):
        adj = _build_adj(
            [(0, 1), (1, 2), (2, 0), (3, 0), (3, 1), (3, 2)], n=4,
        )
        d = _citation_jaccard_distance_matrix(
            adj, mode="weighted_per_direction",
        )
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        np.testing.assert_array_equal(np.diag(d), 0.0)
        assert d.min() >= 0.0
        assert d.max() <= 1.0 + 1e-12


class TestCommon:
    def test_unknown_mode_raises(self):
        adj = _build_adj([(0, 1)], n=2)
        with pytest.raises(ValueError, match="Unknown citation_jaccard_mode"):
            _citation_jaccard_distance_matrix(adj, mode="bogus")

    def test_empty_graph_all_modes(self):
        adj = sp.csr_matrix((0, 0))
        for mode in ("binary", "weighted_unified", "weighted_per_direction"):
            d = _citation_jaccard_distance_matrix(adj, mode=mode)
            assert d.shape == (0, 0)
