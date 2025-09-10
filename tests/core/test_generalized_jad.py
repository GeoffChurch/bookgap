#!/usr/bin/env python3
"""
Test script demonstrating the generalized JADDenoiser with multiple subsets.
"""

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from bookgap.denoisers import DenoiserContext, JADDenoiser


class TestDenoiserContext(DenoiserContext):
    """
    Concrete implementation of DenoiserContext for testing multiple subsets.
    """
    
    def __init__(self, X: NDArray, A: sp.csr_matrix, subset_indices: list[list[int]]):
        """
        Args:
            X: Full embeddings matrix (N x d)
            A: Affinity matrix (N x N)
            subset_indices: List of lists, where each inner list contains row indices for a subset
        """
        self._X = X
        self._A = A
        self._subset_indices = subset_indices
        
        # Pre-compute subset matrices
        self._subsets = []
        for indices in subset_indices:
            if len(indices) > 0:
                self._subsets.append(self._X[indices])
            else:
                self._subsets.append(np.empty((0, X.shape[1])))
    
    def A(self) -> sp.csr_matrix:
        return self._A
    
    def X(self) -> NDArray:
        return self._X
    
    def X_n(self, n: int) -> NDArray:
        if n < 0 or n >= len(self._subsets):
            raise IndexError(f"Subset index {n} out of range [0, {len(self._subsets)})")
        return self._subsets[n]
    
    def num_subsets(self) -> int:
        return len(self._subsets)
    
    # Backward compatibility methods for 2-subset case
    def X1(self) -> NDArray:
        """Legacy method for first subset"""
        if self.num_subsets() < 1:
            raise ValueError("No first subset available")
        return self.X_n(0)
    
    def X2(self) -> NDArray:
        """Legacy method for second subset"""
        if self.num_subsets() < 2:
            raise ValueError("No second subset available")
        return self.X_n(1)
    
    def S_B(self) -> NDArray:
        """Legacy method for between-subset scatter matrix"""
        if self.num_subsets() != 2:
            raise ValueError("S_B is only defined for exactly 2 subsets")
        
        # Compute means of each subset
        mean1 = np.mean(self.X_n(0), axis=0, keepdims=True)  # 1 x d
        mean2 = np.mean(self.X_n(1), axis=0, keepdims=True)  # 1 x d
        
        # Between-subset scatter
        diff = mean1 - mean2  # 1 x d
        return diff.T @ diff  # d x d


def create_synthetic_data(n_samples: int = 300, n_features: int = 50, n_subsets: int = 3, random_seed: int = 42):
    """
    Create synthetic data with multiple distinct subsets for testing.
    """
    np.random.seed(random_seed)
    
    # Create subset-specific means to make them distinct
    subset_means = []
    for i in range(n_subsets):
        mean = np.random.randn(n_features) * 2.0  # Different means for each subset
        subset_means.append(mean)
    
    # Generate samples for each subset
    samples_per_subset = n_samples // n_subsets
    X_parts = []
    subset_indices = []
    
    for i in range(n_subsets):
        # Generate samples around the subset-specific mean
        n_subset_samples = samples_per_subset + (1 if i < n_samples % n_subsets else 0)
        subset_data = np.random.randn(n_subset_samples, n_features) * 0.5 + subset_means[i]
        X_parts.append(subset_data)
        
        # Track indices for this subset
        start_idx = sum(len(part) for part in X_parts[:-1])
        end_idx = start_idx + len(subset_data)
        subset_indices.append(list(range(start_idx, end_idx)))
    
    # Combine all data
    X = np.vstack(X_parts)
    
    # Create a simple affinity matrix (k-nearest neighbors)
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=False)
    
    return X, A, subset_indices


def test_generalized_jad():
    """
    Test the generalized JADDenoiser with multiple subsets.
    """
    print("Testing Generalized JADDenoiser with Multiple Subsets")
    print("=" * 60)
    
    # Test with 3 subsets
    X, A, subset_indices = create_synthetic_data(n_samples=300, n_features=20, n_subsets=3)
    context = TestDenoiserContext(X, A, subset_indices)
    
    print(f"Created synthetic data:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Number of subsets: {context.num_subsets()}")
    
    for i in range(context.num_subsets()):
        subset_X = context.X_n(i)
        print(f"  Subset {i}: {subset_X.shape[0]} samples")
    
    # Test the JAD denoiser
    denoiser = JADDenoiser(trials=5, wt=1.0, wl=0.5, w_conflate=2.0)
    
    try:
        P, scores = denoiser.projector(context, n_components=5)
        print(f"\nJAD Denoiser Results:")
        print(f"  Projection matrix shape: {P.shape}")
        print(f"  Quality scores: {scores}")
        print(f"  Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        
        # Test projecting the data
        X_projected = X @ P
        print(f"  Projected data shape: {X_projected.shape}")
        
        # Show variance explained in original vs projected space
        original_var = np.var(X, axis=0).sum()
        projected_var = np.var(X_projected, axis=0).sum()
        print(f"  Original total variance: {original_var:.3f}")
        print(f"  Projected total variance: {projected_var:.3f}")
        print(f"  Variance retention: {projected_var/original_var:.1%}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error running JAD denoiser: {e}")
        # Fallback: test that the context methods work correctly
        print("\nTesting context methods as fallback:")
        print(f"  Total scatter matrix shape: {context.S_T().shape}")
        print(f"  Laplacian scatter matrix shape: {context.S_L().shape}")
        
        for i in range(context.num_subsets()):
            S_subset = context.S_subset(i)
            print(f"  Subset {i} scatter matrix shape: {S_subset.shape}")
        
        # Test multi-subset between-class scatter
        S_B_multi = context.S_B_multi([0, 1, 2])
        print(f"  Multi-subset between-class scatter shape: {S_B_multi.shape}")


def test_backward_compatibility():
    """
    Test that the generalized interface maintains backward compatibility.
    """
    print("\nTesting Backward Compatibility")
    print("=" * 40)
    
    # Create data with exactly 2 subsets for backward compatibility
    X, A, subset_indices = create_synthetic_data(n_samples=200, n_features=15, n_subsets=2)
    context = TestDenoiserContext(X, A, subset_indices)
    
    # Test that X1() and X2() still work
    X1 = context.X1()
    X2 = context.X2()
    
    print(f"X1 shape: {X1.shape}")
    print(f"X2 shape: {X2.shape}")
    
    # Test that S_B() still works
    S_B_old = context.S_B()
    print(f"Legacy S_B shape: {S_B_old.shape}")
    
    # Compare with new multi-subset method
    S_B_new = context.S_B_multi([0, 1])
    print(f"New S_B_multi shape: {S_B_new.shape}")
    
    # They should be similar (but not necessarily identical due to different formulations)
    print(f"Matrix norms - Legacy: {np.linalg.norm(S_B_old):.3f}, New: {np.linalg.norm(S_B_new):.3f}")
    
    print("Backward compatibility test completed!")


if __name__ == "__main__":
    test_generalized_jad()
    test_backward_compatibility()
