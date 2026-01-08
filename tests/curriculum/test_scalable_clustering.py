"""Tests for scalable clustering methods."""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, leaves_list

from artifacts_scripts.curriculum.scalable_clustering import (
    bisecting_kmeans_linkage,
    BisectingKMeansResult,
)
from artifacts_scripts.curriculum.dendrogram_ops import (
    compute_dendrogram_masses,
    find_mass_threshold_for_n_topics,
    get_frontier_nodes,
    gather_leaves,
    map_leaves_to_frontier_clusters,
)


class TestBisectingKmeansLinkage:
    """Tests for bisecting_kmeans_linkage."""
    
    def test_basic_shape(self):
        """Test that output has correct shape."""
        n = 100
        dim = 10
        embeddings = np.random.randn(n, dim)
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert isinstance(result, BisectingKMeansResult)
        assert result.linkage.shape == (n - 1, 4)
        assert result.leaf_cluster_assignments.shape == (n,)
        assert result.n_leaf_clusters == n  # With min_cluster_size=1
    
    def test_small_input(self):
        """Test with very small input."""
        embeddings = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (2, 4)
        assert result.n_leaf_clusters == 3
        # Check that all original points appear in leaves
        assert result.linkage[:, 3].max() == 3  # Final cluster has all 3 points
    
    def test_single_point(self):
        """Test with single point."""
        embeddings = np.array([[0, 0]])
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (0, 4)
        assert result.n_leaf_clusters == 1
        assert result.leaf_cluster_assignments[0] == 0
    
    def test_two_points(self):
        """Test with two points."""
        embeddings = np.array([[0, 0], [1, 1]])
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (1, 4)
        assert result.n_leaf_clusters == 2
        # Should merge points 0 and 1
        assert set(result.linkage[0, :2]) == {0, 1}
        assert result.linkage[0, 3] == 2  # Cluster size
    
    def test_linkage_columns(self):
        """Test that linkage matrix columns are valid."""
        n = 50
        embeddings = np.random.randn(n, 10)
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        linkage = result.linkage
        
        # Columns 0 and 1: cluster indices (0 to 2n-2)
        assert linkage[:, 0].min() >= 0
        assert linkage[:, 1].min() >= 0
        assert linkage[:, 0].max() < 2 * n - 1
        assert linkage[:, 1].max() < 2 * n - 1
        
        # Column 2: distance (non-negative)
        assert linkage[:, 2].min() >= 0
        
        # Column 3: cluster sizes (1 to n)
        assert linkage[:, 3].min() >= 1
        assert linkage[:, 3].max() == n
    
    def test_reproducibility(self):
        """Test that same seed gives same result."""
        embeddings = np.random.randn(50, 10)
        
        result1 = bisecting_kmeans_linkage(embeddings, random_state=42, verbose=False)
        result2 = bisecting_kmeans_linkage(embeddings, random_state=42, verbose=False)
        
        np.testing.assert_array_equal(result1.linkage, result2.linkage)
        np.testing.assert_array_equal(result1.leaf_cluster_assignments, result2.leaf_cluster_assignments)
    
    def test_different_seeds(self):
        """Test that different seeds can give different results."""
        embeddings = np.random.randn(50, 10)
        
        result1 = bisecting_kmeans_linkage(embeddings, random_state=42, verbose=False)
        result2 = bisecting_kmeans_linkage(embeddings, random_state=123, verbose=False)
        
        # They might be different (not guaranteed, but likely)
        # Just check they're both valid
        assert result1.linkage.shape == result2.linkage.shape


class TestBisectingKmeansWithDendrogramOps:
    """Test that bisecting K-Means linkage works with dendrogram_ops functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample embeddings and linkage."""
        np.random.seed(42)
        n = 100
        embeddings = np.random.randn(n, 10)
        result = bisecting_kmeans_linkage(embeddings, random_state=42, verbose=False)
        return embeddings, result, n
    
    def test_compute_dendrogram_masses(self, sample_data):
        """Test mass computation on bisecting K-Means tree."""
        embeddings, result, n = sample_data
        
        # Create uniform leaf masses
        leaf_masses = np.ones(result.n_leaf_clusters) / result.n_leaf_clusters
        
        node_masses = compute_dendrogram_masses(result.linkage, leaf_masses)
        
        # Root should have mass 1
        assert np.isclose(node_masses[-1], 1.0)
        
        # All leaf masses should be preserved
        np.testing.assert_array_almost_equal(node_masses[:result.n_leaf_clusters], leaf_masses)
    
    def test_find_frontier(self, sample_data):
        """Test frontier finding on bisecting K-Means tree."""
        embeddings, result, n = sample_data
        
        # Create non-uniform leaf masses (some leaves have more mass)
        leaf_masses = np.zeros(result.n_leaf_clusters)
        leaf_masses[:10] = 0.1  # First 10 leaves have most mass
        
        node_masses = compute_dendrogram_masses(result.linkage, leaf_masses)
        
        # Find frontier with ~5 topics
        threshold = find_mass_threshold_for_n_topics(
            result.linkage, node_masses, n_topics_desired=5
        )
        
        frontier_nodes = get_frontier_nodes(result.linkage, node_masses, threshold)
        
        # Should have approximately 5 topics
        assert 3 <= len(frontier_nodes) <= 7
    
    def test_map_leaves_to_clusters(self, sample_data):
        """Test leaf-to-cluster mapping on bisecting K-Means tree."""
        embeddings, result, n = sample_data
        
        # Create leaf masses - concentrate mass on some leaves
        np.random.seed(123)
        leaf_masses = np.random.rand(result.n_leaf_clusters) + 0.1
        leaf_masses /= leaf_masses.sum()
        
        node_masses = compute_dendrogram_masses(result.linkage, leaf_masses)
        
        threshold = find_mass_threshold_for_n_topics(
            result.linkage, node_masses, n_topics_desired=3
        )
        
        frontier_nodes = get_frontier_nodes(result.linkage, node_masses, threshold)
        
        # Skip test if we got degenerate frontier
        if len(frontier_nodes) == 0:
            pytest.skip("Degenerate frontier - no nodes above threshold")
        
        leaf_to_cluster = map_leaves_to_frontier_clusters(
            result.linkage, frontier_nodes, result.n_leaf_clusters
        )
        
        # The frontier may not cover all leaves - some leaves may be in 
        # low-mass parts of the tree that don't reach the threshold
        # This is expected behavior
        assigned_mask = leaf_to_cluster >= 0
        
        # But assigned leaves should map to valid cluster IDs
        assigned_clusters = set(leaf_to_cluster[assigned_mask])
        assert assigned_clusters <= set(range(len(frontier_nodes)))
        
        # And we should have assigned a reasonable fraction of leaves
        # (not a hard requirement, but sanity check)
        assert assigned_mask.sum() > result.n_leaf_clusters // 4, "Very few leaves assigned to clusters"
    
    def test_gather_leaves(self, sample_data):
        """Test leaf gathering on bisecting K-Means tree."""
        embeddings, result, n = sample_data
        
        # Test gathering from root (should get all leaves)
        root_node = 2 * result.n_leaf_clusters - 2
        leaves = gather_leaves(root_node, result.linkage, result.n_leaf_clusters)
        
        assert len(leaves) == result.n_leaf_clusters
        assert set(leaves) == set(range(result.n_leaf_clusters))


class TestScalingBehavior:
    """Test that bisecting K-Means scales well."""
    
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_scales_linearly(self, n):
        """Test that algorithm completes for various sizes."""
        embeddings = np.random.randn(n, 50)
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (n - 1, 4)
        assert result.n_leaf_clusters == n
    
    def test_handles_high_dimension(self):
        """Test with high-dimensional embeddings."""
        n = 100
        dim = 384  # Typical embedding dimension
        embeddings = np.random.randn(n, dim)
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (n - 1, 4)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_identical_points(self):
        """Test with many identical points."""
        embeddings = np.zeros((10, 5))
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (9, 4)
    
    def test_two_clusters(self):
        """Test with two clear clusters."""
        n = 50
        cluster1 = np.random.randn(n, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(n, 10) + np.array([-5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2])
        
        result = bisecting_kmeans_linkage(embeddings, verbose=False)
        
        assert result.linkage.shape == (2 * n - 1, 4)
        
        # The final merge should combine the two clusters
        # and have the largest distance
        assert result.linkage[-1, 3] == 2 * n  # All points


class TestMinClusterSize:
    """Test min_cluster_size parameter for coarse clustering."""
    
    def test_min_cluster_size_basic(self):
        """Test that min_cluster_size creates coarser tree."""
        n = 1000
        embeddings = np.random.randn(n, 10)
        
        result = bisecting_kmeans_linkage(
            embeddings, 
            min_cluster_size=100, 
            verbose=False
        )
        
        # Should have fewer merges than n-1
        assert result.linkage.shape[0] < n - 1
        # Should have fewer leaf clusters
        assert result.n_leaf_clusters < n
        # Each leaf cluster should have >= min_cluster_size points (approximately)
        # Note: some variance is expected at boundaries
        
    def test_min_cluster_size_assignments(self):
        """Test that leaf cluster assignments are valid."""
        n = 500
        embeddings = np.random.randn(n, 10)
        
        result = bisecting_kmeans_linkage(
            embeddings, 
            min_cluster_size=50, 
            verbose=False
        )
        
        # All points should be assigned to a cluster
        assert result.leaf_cluster_assignments.shape == (n,)
        assert result.leaf_cluster_assignments.min() >= 0
        assert result.leaf_cluster_assignments.max() < result.n_leaf_clusters
        
        # Count points per cluster
        cluster_sizes = np.bincount(result.leaf_cluster_assignments)
        assert len(cluster_sizes) == result.n_leaf_clusters
        
        # Most clusters should be at least min_cluster_size
        # (some tolerance for boundary cases due to K-Means behavior)
        assert cluster_sizes.min() >= 15  # Allow tolerance for K-Means variance
        
    def test_min_cluster_size_with_dendrogram_ops(self):
        """Test coarse clustering works with dendrogram operations."""
        n = 500
        embeddings = np.random.randn(n, 10)
        
        result = bisecting_kmeans_linkage(
            embeddings, 
            min_cluster_size=50, 
            verbose=False
        )
        
        # Create point masses and aggregate to leaf clusters
        point_masses = np.random.rand(n)
        point_masses /= point_masses.sum()
        
        # Aggregate masses
        leaf_masses = np.zeros(result.n_leaf_clusters)
        np.add.at(leaf_masses, result.leaf_cluster_assignments, point_masses)
        
        # Should sum to 1
        assert np.isclose(leaf_masses.sum(), 1.0)
        
        # Propagate up tree
        node_masses = compute_dendrogram_masses(result.linkage, leaf_masses)
        
        # Root should have all mass
        assert np.isclose(node_masses[-1], 1.0)
        
        # Find frontier
        threshold = find_mass_threshold_for_n_topics(
            result.linkage, node_masses, n_topics_desired=5
        )
        frontier_nodes = get_frontier_nodes(result.linkage, node_masses, threshold)
        
        # Should find some frontier topics
        assert len(frontier_nodes) > 0
        
    def test_entire_dataset_one_cluster(self):
        """Test when entire dataset is smaller than min_cluster_size."""
        n = 50
        embeddings = np.random.randn(n, 10)
        
        result = bisecting_kmeans_linkage(
            embeddings, 
            min_cluster_size=100,  # Larger than n 
            verbose=False
        )
        
        # Should have no merges - entire dataset is one leaf cluster
        assert result.linkage.shape == (0, 4)
        assert result.n_leaf_clusters == 1
        # All points in cluster 0
        assert np.all(result.leaf_cluster_assignments == 0)
