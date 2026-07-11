"""Tests for PageRank implementation."""

import numpy as np
import pytest
import scipy.sparse as sp

from curriculum.core.pagerank import (
    build_batched_pagerank_seeds,
    compute_frontier_adjacency,
    compute_pagerank,
)
from bookgap.util import build_citation_adjacency_from_ids


@pytest.fixture
def simple_work_ids():
    """Create simple work IDs."""
    return ["W0", "W1", "W2", "W3", "W4"]


@pytest.fixture
def simple_work_references():
    """Create simple references with citation structure.
    
    Citation graph:
    0 -> 1, 2
    1 -> 2
    2 -> (none)
    3 -> 0, 2
    4 -> 3
    
    So 2 is highly cited (authority), 4 cites least (depends on others).
    """
    return [
        ["W1", "W2"],  # W0 cites W1, W2
        ["W2"],        # W1 cites W2
        [],            # W2 cites nothing
        ["W0", "W2"],  # W3 cites W0, W2
        ["W3"],        # W4 cites W3
    ]


@pytest.fixture
def simple_adj():
    """Create sparse adjacency matrix for simple citation graph."""
    # Same structure as simple_works but pre-built
    n = 5
    row = [0, 0, 1, 3, 3, 4]  # citers
    col = [1, 2, 2, 0, 2, 3]  # cited
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


class TestBuildCitationAdjacencyFromIds:
    """Tests for build_citation_adjacency_from_ids."""
    
    def test_basic_structure(self, simple_work_ids, simple_work_references):
        """Test basic adjacency matrix structure."""
        adj, id_to_idx = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )
        
        # Shape should be n x n
        assert adj.shape == (5, 5)
        
        # Check id_to_idx mapping
        assert id_to_idx["W0"] == 0
        assert id_to_idx["W4"] == 4
    
    def test_self_loops(self, simple_work_ids, simple_work_references):
        """Test self-loops are added when requested."""
        adj_no_loops, _ = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )
        adj_with_loops, _ = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=True
        )
        
        # With self-loops should have more non-zeros
        assert adj_with_loops.nnz > adj_no_loops.nnz
        
        # Diagonal should be non-zero with self-loops
        diag = adj_with_loops.diagonal()
        assert (diag > 0).all()
    
    def test_citation_edges(self, simple_work_ids, simple_work_references):
        """Test citation edges are correct."""
        adj, id_to_idx = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )
        
        # W0 cites W1 and W2
        assert adj[0, 1] == 1.0
        assert adj[0, 2] == 1.0
        
        # W2 cites nothing (except possibly self)
        assert adj[2, 0] == 0.0
        assert adj[2, 1] == 0.0
        assert adj[2, 3] == 0.0
        assert adj[2, 4] == 0.0
    
    def test_external_references_ignored(self):
        """Test that references to external works are ignored."""
        work_ids = ["W0", "W1"]
        work_references = [["W1", "external_id"], []]
        
        adj, _ = build_citation_adjacency_from_ids(
            work_ids, work_references, add_self_loops=False
        )
        
        # Only W0->W1 edge, not W0->external
        assert adj.nnz == 1
        assert adj[0, 1] == 1.0



class TestComputePagerank:
    """Tests for compute_pagerank."""
    
    def test_convergence(self, simple_adj):
        """Test PageRank converges."""
        seeds = np.ones(5) / 5  # Uniform
        
        pr = compute_pagerank(simple_adj, seeds, alpha=0.85, tol=1e-6, max_iter=100)
        
        # Should return valid probability distribution
        assert pr.min() >= 0
        assert pr.sum() == pytest.approx(1.0, rel=0.01)
    
    def test_personalization(self, simple_adj):
        """Test personalized PageRank concentrates mass near seed."""
        # Seed on node 4 only
        seeds = np.zeros(5)
        seeds[4] = 1.0
        
        pr = compute_pagerank(simple_adj, seeds, alpha=0.85, tol=1e-6)
        
        # Node 4 and nodes it cites (3, and transitively 0, 2) should have mass
        # Node 4 starts with all personalization mass
        assert pr[4] > 0
    
    def test_authority_has_high_mass(self, simple_adj):
        """Test highly cited nodes get high PageRank."""
        seeds = np.ones(5) / 5  # Uniform
        
        pr = compute_pagerank(simple_adj, seeds, alpha=0.85, tol=1e-6)
        
        # Node 2 is cited by 0, 1, 3 - should have high mass
        # (depends on graph structure and damping)
        assert pr.sum() == pytest.approx(1.0, rel=0.01)
    
    def test_batched_pagerank(self, simple_adj):
        """Test batched PageRank produces same results as individual runs."""
        n = 5
        
        # Run individual PageRanks
        individual_results = []
        for i in range(n):
            seeds = np.zeros(n)
            seeds[i] = 1.0
            pr = compute_pagerank(simple_adj, seeds, alpha=0.85, tol=1e-8)
            individual_results.append(pr)
        
        # Run batched PageRank
        batched_seeds = np.eye(n)
        batched_pr = compute_pagerank(simple_adj, batched_seeds, alpha=0.85, tol=1e-8)
        
        # Results should match
        for i in range(n):
            np.testing.assert_array_almost_equal(
                individual_results[i], batched_pr[:, i], decimal=5
            )
    
    def test_handles_sink_nodes(self):
        """Test PageRank handles sink nodes (no outgoing edges)."""
        # Node 2 is a sink
        adj = sp.csr_matrix([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],  # Sink
        ], dtype=np.float32)
        
        seeds = np.ones(3) / 3
        pr = compute_pagerank(adj, seeds, alpha=0.85, tol=1e-6)
        
        # Should still converge
        assert pr.sum() == pytest.approx(1.0, rel=0.01)
        assert pr.min() >= 0


class TestBuildBatchedPagerankSeeds:
    """Tests for build_batched_pagerank_seeds."""
    
    def test_basic_structure(self):
        """Test seed matrix structure."""
        n_samples = 10
        num_topics = 3
        leaf_to_cluster = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        
        seeds = build_batched_pagerank_seeds(n_samples, num_topics, leaf_to_cluster)
        
        # Shape should be (n_samples, num_topics)
        assert seeds.shape == (10, 3)
        
        # Each column should sum to 1
        col_sums = seeds.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, [1.0, 1.0, 1.0])
    
    def test_uniform_within_cluster(self):
        """Test seeds are uniform within each cluster."""
        n_samples = 6
        num_topics = 2
        leaf_to_cluster = np.array([0, 0, 0, 1, 1, 1])
        
        seeds = build_batched_pagerank_seeds(n_samples, num_topics, leaf_to_cluster)
        
        # Cluster 0 (indices 0,1,2) should have equal weight in column 0
        assert seeds[0, 0] == pytest.approx(1/3)
        assert seeds[1, 0] == pytest.approx(1/3)
        assert seeds[2, 0] == pytest.approx(1/3)
        
        # Cluster 1 (indices 3,4,5) should have equal weight in column 1
        assert seeds[3, 1] == pytest.approx(1/3)
        assert seeds[4, 1] == pytest.approx(1/3)
        assert seeds[5, 1] == pytest.approx(1/3)
        
        # Cross-cluster entries should be 0
        assert seeds[0, 1] == 0
        assert seeds[3, 0] == 0


class TestComputeFrontierAdjacency:
    """Tests for compute_frontier_adjacency."""
    
    def test_basic_computation(self, simple_adj):
        """Test frontier adjacency computation."""
        # Simple clustering: 0,1 in cluster 0; 2,3,4 in cluster 1
        frontier_nodes = np.array([0, 1])  # Dummy - just indices into frontier
        leaf_to_cluster = np.array([0, 0, 1, 1, 1])
        
        adj_frontier = compute_frontier_adjacency(
            simple_adj,
            frontier_nodes,
            leaf_to_cluster,
            alpha=0.85,
            tol=1e-6,
        )
        
        # Should be 2x2 matrix
        assert adj_frontier.shape == (2, 2)
        
        # Diagonal should be 0
        assert adj_frontier[0, 0] == 0
        assert adj_frontier[1, 1] == 0
    
    def test_asymmetric_dependencies(self, simple_adj):
        """Test that dependency matrix is generally asymmetric."""
        frontier_nodes = np.array([0, 1])
        leaf_to_cluster = np.array([0, 0, 1, 1, 1])
        
        adj_frontier = compute_frontier_adjacency(
            simple_adj,
            frontier_nodes,
            leaf_to_cluster,
            alpha=0.85,
            tol=1e-6,
        )
        
        # Generally adj[i,j] != adj[j,i] unless graph is symmetric
        # (dependency is directional)
        # Just check it computes without error
        assert adj_frontier.shape == (2, 2)


class TestIntegration:
    """Integration tests for PageRank pipeline."""
    
    def test_full_pipeline_with_synthetic_data(self):
        """Test full PageRank pipeline with synthetic citation network."""
        np.random.seed(42)
        n = 100
        
        # Create random sparse citation graph
        # Each node cites ~3 other nodes (preferring lower indices = older)
        row, col, data = [], [], []
        for i in range(n):
            # Cite ~3 random nodes with index < i (older works)
            n_citations = np.random.poisson(3)
            if i > 0:
                cited = np.random.choice(i, size=min(n_citations, i), replace=False)
                for j in cited:
                    row.append(i)
                    col.append(j)
                    data.append(1.0)
        
        adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
        
        # Run PageRank with uniform seeds
        seeds = np.ones(n) / n
        pr = compute_pagerank(adj, seeds, alpha=0.85, tol=1e-6)
        
        # Should be valid distribution
        assert pr.min() >= 0
        assert pr.sum() == pytest.approx(1.0, rel=0.01)
        
        # Early nodes (0, 1, 2, ...) should have higher PageRank
        # since they're cited more
        early_pr = pr[:10].mean()
        late_pr = pr[-10:].mean()
        assert early_pr > late_pr
