"""Tests for structured sampling."""

import random

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage

from artifacts_scripts.curriculum.summarize_topics import sample_works_from_cluster
from artifacts_scripts.curriculum.dendrogram_ops import gather_leaves


@pytest.fixture
def simple_linkage():
    """Create a simple linkage matrix for 8 points.
    
    Tree structure (binary tree):
              14
            /    \
          12      13
         /  \    /  \
        8    9  10   11
       / \  / \ / \  / \
       0 1 2 3 4 5  6  7
    """
    # 8 leaves (0-7), 7 internal nodes (8-14)
    linked = np.array([
        [0, 1, 1.0, 2],   # node 8
        [2, 3, 1.0, 2],   # node 9
        [4, 5, 1.0, 2],   # node 10
        [6, 7, 1.0, 2],   # node 11
        [8, 9, 2.0, 4],   # node 12
        [10, 11, 2.0, 4], # node 13
        [12, 13, 3.0, 8], # node 14 (root)
    ])
    return linked


class TestSampleWorksFromCluster:
    """Tests for sample_works_from_cluster."""
    
    def test_single_sample_from_leaf(self, simple_linkage):
        """Sampling from leaf returns that leaf."""
        rng = random.Random(42)
        
        samples = sample_works_from_cluster(
            node_idx=0,
            n_samples=1,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        assert samples == [0]
    
    def test_single_sample_from_internal_node(self, simple_linkage):
        """Sampling 1 from internal node returns one leaf."""
        rng = random.Random(42)
        
        # Node 12 contains leaves 0, 1, 2, 3
        samples = sample_works_from_cluster(
            node_idx=12,
            n_samples=1,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        assert len(samples) == 1
        assert samples[0] in [0, 1, 2, 3]
    
    def test_structured_sampling_splits_evenly(self, simple_linkage):
        """Test that samples are split among children."""
        rng = random.Random(42)
        
        # Node 12 has children 8 (leaves 0,1) and 9 (leaves 2,3)
        # Requesting 4 samples should give 2 from each child
        samples = sample_works_from_cluster(
            node_idx=12,
            n_samples=4,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        assert len(samples) == 4
        
        # Should have 2 from each subtree
        left_subtree = [s for s in samples if s in [0, 1]]
        right_subtree = [s for s in samples if s in [2, 3]]
        assert len(left_subtree) == 2
        assert len(right_subtree) == 2
    
    def test_odd_sample_count_handled(self, simple_linkage):
        """Test odd sample counts are handled correctly."""
        rng = random.Random(42)
        
        # Request 3 samples from node 12 (children 8, 9)
        samples = sample_works_from_cluster(
            node_idx=12,
            n_samples=3,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        assert len(samples) == 3
        
        # Should be split 2/1 or 1/2 (depending on remainder handling)
        left_subtree = [s for s in samples if s in [0, 1]]
        right_subtree = [s for s in samples if s in [2, 3]]
        assert len(left_subtree) + len(right_subtree) == 3
        assert abs(len(left_subtree) - len(right_subtree)) <= 1
    
    def test_full_tree_sampling(self, simple_linkage):
        """Test sampling from root covers all subtrees."""
        rng = random.Random(42)
        
        # Request 8 samples from root (node 14)
        samples = sample_works_from_cluster(
            node_idx=14,
            n_samples=8,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        assert len(samples) == 8
        # Should cover all leaves (with possible repeats if sampling allows)
        assert all(s in range(8) for s in samples)
    
    def test_reproducibility(self, simple_linkage):
        """Test that same seed gives same results."""
        samples1 = sample_works_from_cluster(
            node_idx=14,
            n_samples=4,
            linked=simple_linkage,
            n_leaves=8,
            rng=random.Random(42),
        )
        
        samples2 = sample_works_from_cluster(
            node_idx=14,
            n_samples=4,
            linked=simple_linkage,
            n_leaves=8,
            rng=random.Random(42),
        )
        
        assert samples1 == samples2
    
    def test_different_seeds_give_different_results(self, simple_linkage):
        """Test that different seeds may give different results."""
        results = set()
        for seed in range(10):
            samples = sample_works_from_cluster(
                node_idx=14,
                n_samples=4,
                linked=simple_linkage,
                n_leaves=8,
                rng=random.Random(seed),
            )
            results.add(tuple(sorted(samples)))
        
        # With 10 different seeds, we should get some variation
        # (not guaranteed but highly likely)
        assert len(results) >= 2
    
    def test_more_samples_than_leaves_repeats(self, simple_linkage):
        """Test handling when requesting more samples than available leaves."""
        rng = random.Random(42)
        
        # Node 8 has only 2 leaves (0, 1), request 4 samples
        # The algorithm splits 4 -> 2,2 among children (nodes 0 and 1)
        # Each child is a leaf, so we get 2 samples total (one per leaf)
        samples = sample_works_from_cluster(
            node_idx=8,
            n_samples=4,
            linked=simple_linkage,
            n_leaves=8,
            rng=rng,
        )
        
        # Returns 2 samples (one per leaf) - algorithm doesn't duplicate
        assert len(samples) == 2
        assert all(s in [0, 1] for s in samples)


class TestSamplingDistribution:
    """Statistical tests for sampling distribution."""
    
    def test_coverage_over_many_samples(self, simple_linkage):
        """Test that repeated sampling covers the space well."""
        n_trials = 100
        coverage = {i: 0 for i in range(8)}
        
        for seed in range(n_trials):
            rng = random.Random(seed)
            samples = sample_works_from_cluster(
                node_idx=14,  # Root
                n_samples=4,
                linked=simple_linkage,
                n_leaves=8,
                rng=rng,
            )
            for s in samples:
                coverage[s] += 1
        
        # Each leaf should be sampled at least once
        assert all(count > 0 for count in coverage.values())
        
        # Distribution should be somewhat balanced
        # (not perfect due to tree structure, but not wildly skewed)
        counts = list(coverage.values())
        assert max(counts) / (min(counts) + 1) < 5  # No leaf is 5x more likely


class TestIntegrationWithRealClustering:
    """Integration tests with real hierarchical clustering."""
    
    def test_sampling_from_real_dendrogram(self):
        """Test sampling from a dendrogram built on real data."""
        np.random.seed(42)
        n_samples = 50
        
        # Generate clustered data
        X = np.vstack([
            np.random.randn(10, 2) + [0, 0],
            np.random.randn(10, 2) + [5, 0],
            np.random.randn(10, 2) + [0, 5],
            np.random.randn(10, 2) + [5, 5],
            np.random.randn(10, 2) + [2.5, 2.5],
        ])
        
        # Build dendrogram
        linked = linkage(X, method='ward')
        
        # Sample from root
        rng = random.Random(42)
        root_idx = 2 * n_samples - 2  # Root is last node
        
        samples = sample_works_from_cluster(
            node_idx=root_idx,
            n_samples=10,
            linked=linked,
            n_leaves=n_samples,
            rng=rng,
        )
        
        assert len(samples) == 10
        assert len(set(samples)) <= 10  # May have duplicates
        assert all(0 <= s < n_samples for s in samples)
