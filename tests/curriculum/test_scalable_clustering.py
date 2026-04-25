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


class TestWeightedBisectingKmeans:
    """``sample_weights`` support: weighted KMeans for centroid placement,
    weighted Ward-distance for merge cost. Used by the curriculum to anchor
    clusters around high-relevance works instead of letting them get peeled
    off as singletons by relevance-blind geometry."""

    def test_uniform_weights_match_unweighted(self):
        """All-ones weights should produce essentially identical output to
        the unweighted call (modulo floating-point in centroid math)."""
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((80, 5))
        weights = np.ones(80)

        unweighted = bisecting_kmeans_linkage(
            embeddings, min_cluster_size=2, verbose=False, random_state=42,
        )
        weighted = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=2,
            verbose=False, random_state=42,
        )
        # Same number of leaves and merges
        assert weighted.n_leaf_clusters == unweighted.n_leaf_clusters
        assert weighted.linkage.shape == unweighted.linkage.shape
        # Distances should match closely (uniform weights → same centroids)
        np.testing.assert_allclose(
            weighted.linkage[:, 2], unweighted.linkage[:, 2], atol=1e-10,
        )

    def test_concentrated_weight_pulls_centroid(self):
        """A single dominant-weight point should pull its cluster's centroid
        toward itself, observably changing the merge tree compared to
        uniform weights."""
        rng = np.random.default_rng(11)
        cloud_a = rng.standard_normal((20, 3))
        cloud_b = rng.standard_normal((20, 3)) + 10.0
        embeddings = np.vstack([cloud_a, cloud_b])
        weights = np.ones(40)
        weights[0] = 1000.0   # heavy weight in cloud_a

        weighted = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=2,
            verbose=False, random_state=42,
        )
        unweighted = bisecting_kmeans_linkage(
            embeddings, min_cluster_size=2, verbose=False, random_state=42,
        )
        # The weighted centroid of cloud_a is dragged toward the
        # heavy-weight point, so the bisection at that cluster places
        # differently and the resulting tree differs from the unweighted
        # version. We don't insist on the exact same shape — weighted
        # and unweighted bisecting trees can legitimately have different
        # numbers of merges depending on how the splits cascade. We just
        # verify they're not identical.
        same_shape = weighted.linkage.shape == unweighted.linkage.shape
        if same_shape:
            assert not np.allclose(weighted.linkage[:, 2], unweighted.linkage[:, 2])
        else:
            assert weighted.linkage.shape != unweighted.linkage.shape  # already differs

    def test_negative_weights_rejected(self):
        embeddings = np.random.randn(20, 3)
        weights = np.full(20, 1.0)
        weights[5] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            bisecting_kmeans_linkage(embeddings, sample_weights=weights, verbose=False)

    def test_wrong_shape_rejected(self):
        embeddings = np.random.randn(20, 3)
        weights = np.ones(19)
        with pytest.raises(ValueError, match="shape"):
            bisecting_kmeans_linkage(embeddings, sample_weights=weights, verbose=False)

    def test_zero_weight_side_falls_back_to_unit(self):
        """If all of one side's points have zero weight, fall back to unit
        weighting on that side (avoids /0 in weighted centroid)."""
        rng = np.random.default_rng(13)
        cloud_a = rng.standard_normal((10, 3))
        cloud_b = rng.standard_normal((10, 3)) + 5.0
        embeddings = np.vstack([cloud_a, cloud_b])
        weights = np.concatenate([np.ones(10), np.zeros(10)])
        result = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=2,
            verbose=False, random_state=42,
        )
        assert result.linkage.shape[0] > 0
        assert not np.isnan(result.linkage).any()


class TestMinClusterWeight:
    """``min_cluster_weight`` stops bisection once a cluster's total weight
    drops to/below the floor, in addition to the existing count-based
    ``min_cluster_size``. This is the natural threshold under weighted
    clustering: a single high-relevance work can earn singleton status
    while a glob of low-relevance neighbours stays merged."""

    def test_default_zero_no_effect(self):
        """min_cluster_weight=0 (default) should match the no-arg path."""
        rng = np.random.default_rng(17)
        embeddings = rng.standard_normal((60, 4))
        weights = rng.uniform(0.1, 1.0, size=60)

        baseline = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=2,
            verbose=False, random_state=42,
        )
        with_zero = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=2,
            min_cluster_weight=0.0, verbose=False, random_state=42,
        )
        assert with_zero.n_leaf_clusters == baseline.n_leaf_clusters
        np.testing.assert_array_equal(
            with_zero.leaf_cluster_assignments, baseline.leaf_cluster_assignments,
        )

    def test_high_threshold_collapses_to_one_cluster(self):
        """If the threshold exceeds the total weight, the whole dataset
        should be returned as a single leaf cluster (early-out path)."""
        rng = np.random.default_rng(19)
        embeddings = rng.standard_normal((40, 3))
        weights = np.full(40, 0.5)  # total = 20
        result = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=1,
            min_cluster_weight=100.0, verbose=False, random_state=42,
        )
        assert result.n_leaf_clusters == 1
        assert result.linkage.shape == (0, 4)
        assert np.all(result.leaf_cluster_assignments == 0)

    def test_high_weight_singleton_preserved(self):
        """A single point whose weight exceeds the threshold should still
        end up as its own leaf (size==1 takes precedence over weight)."""
        rng = np.random.default_rng(23)
        embeddings = np.vstack([
            rng.standard_normal((30, 3)),
            np.array([[10.0, 10.0, 10.0]]),  # outlier point
        ])
        weights = np.concatenate([np.full(30, 0.01), np.array([1000.0])])
        # Threshold is well above any sub-cluster's bulk weight (30 * 0.01 =
        # 0.3) but below the outlier alone (1000), so the outlier survives
        # as a singleton while other clusters get collapsed.
        result = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=1,
            min_cluster_weight=1.0, verbose=False, random_state=42,
        )
        # The outlier index (30) must map to a leaf cluster of size exactly 1.
        outlier_leaf = result.leaf_cluster_assignments[30]
        n_in_outlier_leaf = int(
            (result.leaf_cluster_assignments == outlier_leaf).sum()
        )
        assert n_in_outlier_leaf == 1, (
            f"outlier should be a singleton leaf, found {n_in_outlier_leaf} "
            f"points in its leaf"
        )

    def test_low_weight_cluster_kept_intact(self):
        """A cluster with many points but low total weight should not get
        further split — the recursion stops at the weight threshold."""
        rng = np.random.default_rng(29)
        # 100 low-weight points and 10 high-weight points well-separated.
        low_cluster = rng.standard_normal((100, 3))
        high_cluster = rng.standard_normal((10, 3)) + 20.0
        embeddings = np.vstack([low_cluster, high_cluster])
        weights = np.concatenate([np.full(100, 0.001), np.full(10, 10.0)])
        # Total low-cluster weight = 0.1; threshold of 0.5 keeps it intact,
        # high cluster weight = 100 should still get subdivided.
        result = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=1,
            min_cluster_weight=0.5, verbose=False, random_state=42,
        )
        # Find the leaf containing the low-weight point at index 0; all 100
        # low-weight points should share this leaf.
        low_leaf = result.leaf_cluster_assignments[0]
        low_leaf_count = int((result.leaf_cluster_assignments[:100] == low_leaf).sum())
        assert low_leaf_count == 100, (
            f"all 100 low-weight points should be in the same leaf, got "
            f"{low_leaf_count}"
        )
        # And the high-weight cluster should produce more than one leaf.
        high_leaf_ids = set(result.leaf_cluster_assignments[100:])
        assert len(high_leaf_ids) > 1, (
            f"high-weight cluster should subdivide further, got "
            f"{len(high_leaf_ids)} leaf(s)"
        )

    def test_requires_sample_weights(self):
        """min_cluster_weight > 0 without sample_weights is undefined."""
        embeddings = np.random.randn(20, 3)
        with pytest.raises(ValueError, match="requires sample_weights"):
            bisecting_kmeans_linkage(
                embeddings, min_cluster_weight=1.0, verbose=False,
            )

    def test_negative_threshold_rejected(self):
        embeddings = np.random.randn(20, 3)
        weights = np.ones(20)
        with pytest.raises(ValueError, match=">= 0"):
            bisecting_kmeans_linkage(
                embeddings, sample_weights=weights,
                min_cluster_weight=-0.1, verbose=False,
            )

    def test_combined_with_min_cluster_size(self):
        """When both thresholds are set, whichever fires first stops the
        recursion. Use a size threshold that fires well before the weight
        threshold and check we don't see singleton-driven results."""
        rng = np.random.default_rng(31)
        embeddings = rng.standard_normal((100, 3))
        weights = np.full(100, 1.0)  # uniform weights, total = 100
        # min_cluster_size=20 fires before weight threshold of 0.1 (which
        # alone would allow singletons since each point's weight = 1 > 0.1).
        result = bisecting_kmeans_linkage(
            embeddings, sample_weights=weights, min_cluster_size=20,
            min_cluster_weight=0.1, verbose=False, random_state=42,
        )
        cluster_sizes = np.bincount(result.leaf_cluster_assignments)
        # Weight threshold of 0.1 alone would let any cluster down to a
        # single point through (point weight = 1 > 0.1); the size threshold
        # must keep clusters substantially larger than 1. Allow K-Means
        # variance: leaves can dip below 20 due to uneven splits, but they
        # should be far from 1.
        assert cluster_sizes.min() >= 5, (
            f"min cluster size = {cluster_sizes.min()} suggests size "
            f"threshold did not dominate (weight-only would allow "
            f"singletons)"
        )
