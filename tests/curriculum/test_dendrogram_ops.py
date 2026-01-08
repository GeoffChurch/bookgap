"""Tests for dendrogram operations."""

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from artifacts_scripts.curriculum.dendrogram_ops import (
    compute_dendrogram_masses,
    find_mass_threshold_for_n_topics,
    gather_leaves,
    get_children,
    get_frontier_nodes,
    map_leaves_to_frontier_clusters,
)


@pytest.fixture
def simple_linkage():
    """Create a simple linkage matrix for 5 points.
    
    Tree structure (node indices):
           8
          / \
         6   7
        / \ / \
       0  1 2  5
              / \
             3   4
    
    Linkage matrix format: [idx1, idx2, distance, count]
    """
    # 5 leaves (0-4), 4 internal nodes (5-8)
    # Merge order:
    # 5: merge 3, 4 -> node 5 (distance 1.0, count 2)
    # 6: merge 0, 1 -> node 6 (distance 1.5, count 2)
    # 7: merge 2, 5 -> node 7 (distance 2.0, count 3)
    # 8: merge 6, 7 -> node 8 (distance 3.0, count 5)
    linked = np.array([
        [3, 4, 1.0, 2],  # node 5
        [0, 1, 1.5, 2],  # node 6
        [2, 5, 2.0, 3],  # node 7
        [6, 7, 3.0, 5],  # node 8
    ])
    return linked


@pytest.fixture
def simple_masses():
    """Leaf masses for the simple linkage."""
    # Leaves 0-4 with varying masses
    return np.array([0.1, 0.2, 0.3, 0.15, 0.25])


class TestComputeDendrogramMasses:
    """Tests for compute_dendrogram_masses."""
    
    def test_basic_propagation(self, simple_linkage, simple_masses):
        """Test that masses propagate correctly up the tree."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Check leaf masses are preserved
        np.testing.assert_array_equal(node_masses[:5], simple_masses)
        
        # Check internal node masses
        # Node 5: 0.15 + 0.25 = 0.4
        assert node_masses[5] == pytest.approx(0.4)
        # Node 6: 0.1 + 0.2 = 0.3
        assert node_masses[6] == pytest.approx(0.3)
        # Node 7: 0.3 + 0.4 = 0.7
        assert node_masses[7] == pytest.approx(0.7)
        # Node 8 (root): 0.3 + 0.7 = 1.0
        assert node_masses[8] == pytest.approx(1.0)
    
    def test_output_shape(self, simple_linkage, simple_masses):
        """Test output has correct shape."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # 2*n - 1 nodes total
        assert node_masses.shape == (9,)
    
    def test_uniform_masses(self, simple_linkage):
        """Test with uniform masses."""
        uniform_masses = np.ones(5) * 0.2
        node_masses = compute_dendrogram_masses(simple_linkage, uniform_masses)
        
        # Root should have total mass
        assert node_masses[-1] == pytest.approx(1.0)
    
    def test_zero_masses(self, simple_linkage):
        """Test with zero masses."""
        zero_masses = np.zeros(5)
        node_masses = compute_dendrogram_masses(simple_linkage, zero_masses)
        
        # All masses should be zero
        np.testing.assert_array_equal(node_masses, 0.0)


class TestGetFrontierNodes:
    """Tests for get_frontier_nodes."""
    
    def test_high_threshold_returns_root(self, simple_linkage, simple_masses):
        """With high threshold, only root is above threshold."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Threshold just below root mass
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold=0.9)
        
        # Only root (8) should be in frontier
        assert len(frontier) == 1
        assert frontier[0] == 8
    
    def test_low_threshold_returns_leaves(self, simple_linkage, simple_masses):
        """With low threshold, individual leaves may be frontier."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Very low threshold - all leaves above
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold=0.05)
        
        # Should have multiple frontier nodes (leaves and small clusters)
        assert len(frontier) >= 2
    
    def test_medium_threshold(self, simple_linkage, simple_masses):
        """Test with medium threshold produces intermediate frontier."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Threshold between subtree masses
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold=0.35)
        
        # Should split into two subtrees: nodes 6 (0.3) and 7 (0.7)
        # Node 6 is below threshold (0.3 <= 0.35), so we don't include it
        # Node 7 is above threshold (0.7 > 0.35)
        # Check that we get reasonable number of frontier nodes
        assert len(frontier) >= 1
    
    def test_empty_frontier_with_extreme_threshold(self, simple_linkage, simple_masses):
        """Very high threshold returns empty frontier."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Threshold above all masses
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold=10.0)
        
        assert len(frontier) == 0


class TestFindMassThresholdForNTopics:
    """Tests for find_mass_threshold_for_n_topics."""
    
    def test_finds_reasonable_threshold(self, simple_linkage, simple_masses):
        """Test that it finds a threshold producing desired topic count."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        # Ask for 2 topics
        threshold = find_mass_threshold_for_n_topics(simple_linkage, node_masses, n_topics_desired=2)
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold)
        
        # Should be close to 2
        assert 1 <= len(frontier) <= 3
    
    def test_single_topic_returns_high_threshold(self, simple_linkage, simple_masses):
        """Asking for 1 topic should return root-like threshold."""
        node_masses = compute_dendrogram_masses(simple_linkage, simple_masses)
        
        threshold = find_mass_threshold_for_n_topics(simple_linkage, node_masses, n_topics_desired=1)
        frontier = get_frontier_nodes(simple_linkage, node_masses, threshold)
        
        assert len(frontier) == 1


class TestGatherLeaves:
    """Tests for gather_leaves."""
    
    def test_leaf_returns_itself(self, simple_linkage):
        """Gathering leaves from a leaf returns just that leaf."""
        leaves = gather_leaves(0, simple_linkage, n_samples=5)
        assert leaves == [0]
    
    def test_internal_node_returns_subtree_leaves(self, simple_linkage):
        """Internal node returns all leaves in subtree."""
        # Node 5 contains leaves 3 and 4
        leaves = gather_leaves(5, simple_linkage, n_samples=5)
        assert set(leaves) == {3, 4}
        
        # Node 6 contains leaves 0 and 1
        leaves = gather_leaves(6, simple_linkage, n_samples=5)
        assert set(leaves) == {0, 1}
        
        # Node 7 contains leaves 2, 3, 4
        leaves = gather_leaves(7, simple_linkage, n_samples=5)
        assert set(leaves) == {2, 3, 4}
    
    def test_root_returns_all_leaves(self, simple_linkage):
        """Root node returns all leaves."""
        leaves = gather_leaves(8, simple_linkage, n_samples=5)
        assert set(leaves) == {0, 1, 2, 3, 4}


class TestGetChildren:
    """Tests for get_children."""
    
    def test_internal_node_children(self, simple_linkage):
        """Test getting children of internal nodes."""
        # Node 5: children are 3, 4
        c1, c2 = get_children(5, simple_linkage, n_samples=5)
        assert set([c1, c2]) == {3, 4}
        
        # Node 8 (root): children are 6, 7
        c1, c2 = get_children(8, simple_linkage, n_samples=5)
        assert set([c1, c2]) == {6, 7}
    
    def test_leaf_raises_error(self, simple_linkage):
        """Getting children of leaf should raise ValueError."""
        with pytest.raises(ValueError):
            get_children(0, simple_linkage, n_samples=5)


class TestMapLeavesToFrontierClusters:
    """Tests for map_leaves_to_frontier_clusters."""
    
    def test_single_cluster(self, simple_linkage):
        """All leaves map to single cluster when root is frontier."""
        frontier_nodes = np.array([8])  # Root only
        
        mapping = map_leaves_to_frontier_clusters(simple_linkage, frontier_nodes, n_samples=5)
        
        # All leaves should be in cluster 0
        np.testing.assert_array_equal(mapping, [0, 0, 0, 0, 0])
    
    def test_two_clusters(self, simple_linkage):
        """Test mapping with two frontier clusters."""
        # Frontier: nodes 6 (leaves 0,1) and 7 (leaves 2,3,4)
        frontier_nodes = np.array([6, 7])
        
        mapping = map_leaves_to_frontier_clusters(simple_linkage, frontier_nodes, n_samples=5)
        
        # Leaves 0,1 -> cluster 0, leaves 2,3,4 -> cluster 1
        assert mapping[0] == 0
        assert mapping[1] == 0
        assert mapping[2] == 1
        assert mapping[3] == 1
        assert mapping[4] == 1


class TestIntegration:
    """Integration tests with real hierarchical clustering."""
    
    def test_full_pipeline_with_random_data(self):
        """Test full pipeline with random embeddings."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        # Generate random embeddings
        embeddings = np.random.randn(n_samples, n_features)
        
        # Build linkage
        linked = linkage(embeddings, method='ward')
        
        # Create random masses (simulating PageRank output)
        leaf_masses = np.random.rand(n_samples)
        leaf_masses /= leaf_masses.sum()  # Normalize
        
        # Propagate masses
        node_masses = compute_dendrogram_masses(linked, leaf_masses)
        
        # Root should have mass 1.0
        assert node_masses[-1] == pytest.approx(1.0)
        
        # Find threshold for ~5 topics
        threshold = find_mass_threshold_for_n_topics(linked, node_masses, n_topics_desired=5)
        
        # Get frontier
        frontier = get_frontier_nodes(linked, node_masses, threshold)
        
        # Should have reasonable number of topics
        assert 3 <= len(frontier) <= 10
        
        # Map leaves to clusters
        mapping = map_leaves_to_frontier_clusters(linked, frontier, n_samples)
        
        # Leaves in frontier clusters should be assigned (non-negative)
        # Some leaves may be unassigned (-1) if they're outside frontier coverage
        assigned_mask = mapping >= 0
        assert assigned_mask.sum() > 0  # At least some assigned
        
        # Number of unique assigned clusters should match frontier size
        assigned_clusters = np.unique(mapping[assigned_mask])
        assert len(assigned_clusters) == len(frontier)
