"""Tests for Leiden community detection clustering option."""

import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.curriculum.build_curriculum import _cluster_with_leiden


class TestClusterWithLeiden:
    """``_cluster_with_leiden`` runs Leiden community detection on a sparse
    citation subgraph and returns ``(membership, n_communities)``."""

    def test_recovers_triangular_communities(self):
        """Three disconnected triangles bridged by one edge each should
        produce three communities."""
        # Triangles {0,1,2}, {3,4,5}, {6,7,8} with bridges 2-3, 5-6.
        edges_rows = [0, 1, 1, 2, 0, 2,    3, 4, 4, 5, 3, 5,    6, 7, 7, 8, 6, 8,    2, 3, 5, 6]
        edges_cols = [1, 0, 2, 1, 2, 0,    4, 3, 5, 4, 5, 3,    7, 6, 8, 7, 8, 6,    3, 2, 6, 5]
        data = [1.0] * len(edges_rows)
        adj = sp.csr_matrix((data, (edges_rows, edges_cols)), shape=(9, 9))

        membership, n_communities = _cluster_with_leiden(
            adj, resolution=1.0, seed=42,
        )
        assert n_communities == 3, f"expected 3 communities, got {n_communities}"
        # Within each triangle, nodes should share a community
        assert membership[0] == membership[1] == membership[2]
        assert membership[3] == membership[4] == membership[5]
        assert membership[6] == membership[7] == membership[8]
        # And the three triangles should be distinct communities
        assert len({membership[0], membership[3], membership[6]}) == 3

    def test_higher_resolution_yields_more_communities(self):
        """With sufficiently varied structure, higher resolution should
        generally produce at least as many communities as lower
        resolution. (Modularity-based community detection isn't strictly
        monotonic in resolution, but on this construction the trend
        should hold.)"""
        # 4 dense cliques (K_5) loosely connected — increasing resolution
        # should split them apart cleanly.
        edges_rows: list[int] = []
        edges_cols: list[int] = []
        for clique_start in (0, 5, 10, 15):
            for i in range(clique_start, clique_start + 5):
                for j in range(clique_start, clique_start + 5):
                    if i != j:
                        edges_rows.append(i)
                        edges_cols.append(j)
        # bridges: 4-5, 9-10, 14-15
        for u, v in [(4, 5), (9, 10), (14, 15)]:
            edges_rows.extend([u, v])
            edges_cols.extend([v, u])
        data = [1.0] * len(edges_rows)
        adj = sp.csr_matrix((data, (edges_rows, edges_cols)), shape=(20, 20))

        _, n_low = _cluster_with_leiden(adj, resolution=0.5, seed=42)
        _, n_high = _cluster_with_leiden(adj, resolution=2.0, seed=42)
        assert n_high >= n_low, (
            f"higher resolution should not yield fewer communities, got "
            f"{n_low} (low) vs {n_high} (high)"
        )

    def test_empty_graph(self):
        adj = sp.csr_matrix((0, 0))
        membership, n = _cluster_with_leiden(adj, resolution=1.0)
        assert membership.shape == (0,)
        assert n == 0

    def test_isolated_nodes(self):
        """Nodes with no edges should each be their own community."""
        adj = sp.csr_matrix((5, 5))
        membership, n = _cluster_with_leiden(adj, resolution=1.0, seed=42)
        assert membership.shape == (5,)
        # Each isolated node gets its own community
        assert n == 5
        assert set(membership.tolist()) == {0, 1, 2, 3, 4}

    def test_seed_reproducibility(self):
        edges_rows = [0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5]
        edges_cols = [1, 0, 2, 1, 2, 0, 4, 3, 5, 4, 5, 3]
        adj = sp.csr_matrix(
            ([1.0] * len(edges_rows), (edges_rows, edges_cols)),
            shape=(6, 6),
        )
        m1, n1 = _cluster_with_leiden(adj, resolution=1.0, seed=42)
        m2, n2 = _cluster_with_leiden(adj, resolution=1.0, seed=42)
        assert n1 == n2
        np.testing.assert_array_equal(m1, m2)
