"""
Quick test of the updated JAD denoiser with normalized entropy conflation measure.
"""

import numpy as np
import sys
from pathlib import Path

# Add the bookgap package to the path
sys.path.insert(0, str(Path(__file__).parent))

from test_generalized_jad import DenoiserContextForTests, create_synthetic_data
from bookgap.denoisers import JADDenoiser


def test_entropy_conflation():
    """Test that the normalized entropy conflation measure works correctly."""
    print("Testing Normalized Entropy Conflation Measure")
    print("=" * 50)
    
    # Create test data with different levels of conflation
    test_cases = [
        ("Perfect Conflation", 3, [100, 100, 100]),     # Equal sized groups
        ("Moderate Imbalance", 3, [80, 100, 120]),      # Slight imbalance
        ("High Imbalance", 3, [50, 100, 150]),          # High imbalance
        ("Extreme Imbalance", 3, [10, 100, 190]),       # Extreme imbalance
    ]
    
    for case_name, n_subsets, subset_sizes in test_cases:
        print(f"\n{case_name}: {subset_sizes}")
        
        # Create synthetic data with specified subset sizes
        total_samples = sum(subset_sizes)
        X = np.random.randn(total_samples, 20) * 0.5
        
        # Add subset-specific means to create separation
        subset_indices = []
        start_idx = 0
        for i, size in enumerate(subset_sizes):
            end_idx = start_idx + size
            # Add different means to each subset
            X[start_idx:end_idx] += np.random.randn(20) * 2.0
            subset_indices.append(list(range(start_idx, end_idx)))
            start_idx = end_idx
        
        # Create affinity matrix
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(X, n_neighbors=min(10, total_samples-1), 
                           mode='connectivity', include_self=False)
        
        context = DenoiserContextForTests(X, A, subset_indices)
        
        # Compute subset variances manually to show the entropy calculation
        subset_variances = []
        for i in range(context.num_subsets()):
            S_subset = context.S_subset(i)
            # Use trace as a scalar measure of the scatter matrix
            variance = np.trace(S_subset)
            subset_variances.append(variance)
        
        print(f"  Subset variances: {[f'{v:.3f}' for v in subset_variances]}")
        
        # Compute normalized entropy manually
        epsilon = 1e-9
        subset_vars_normalized = np.array(subset_variances) / np.sum(subset_variances)
        entropy = -np.sum(subset_vars_normalized * np.log(subset_vars_normalized + epsilon))
        max_entropy = np.log(len(subset_variances))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        conflation_measure = 1.0 - normalized_entropy
        
        print(f"  Normalized entropy: {normalized_entropy:.4f}")
        print(f"  Conflation penalty: {conflation_measure:.4f}")
        
        # Test the JAD denoiser
        try:
            denoiser = JADDenoiser(trials=3, wt=1.0, wl=0.5, w_conflate=2.0)
            P, scores = denoiser.projector(context, n_components=3)
            print(f"  JAD completed: projection shape {P.shape}, score range [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        except Exception as e:
            print(f"  JAD failed: {e}")


def compare_measures():
    """Compare all conflation measures with and without log transformation."""
    print("\n" + "=" * 80)
    print("Comprehensive Comparison: All Measures With and Without Log")
    print("=" * 80)
    
    test_variance_patterns = [
        ("Perfect", [1.0, 1.0, 1.0, 1.0]),
        ("Moderate", [0.8, 1.0, 1.2, 1.0]),
        ("High", [0.5, 1.0, 1.5, 2.0]),
        ("Extreme", [1.0, 1.0, 1.0, 10.0]),
        ("Scale Test", [10.0, 10.0, 10.0, 10.0]),
        ("Two Groups", [0.5, 0.5, 2.0, 2.0]),
    ]
    
    def variance_of_logs(variances):
        epsilon = 1e-9
        safe_variances = [max(v, epsilon) for v in variances]
        log_variances = [np.log(v) for v in safe_variances]
        return np.var(log_variances, ddof=0)
    
    def variance_direct(variances):
        return np.var(variances, ddof=0)
    
    def coefficient_of_variation(variances):
        epsilon = 1e-9
        return np.std(variances) / (np.mean(variances) + epsilon)
    
    def normalized_entropy(variances):
        epsilon = 1e-9
        vars_normalized = np.array(variances) / np.sum(variances)
        entropy = -np.sum(vars_normalized * np.log(vars_normalized + epsilon))
        max_entropy = np.log(len(variances))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def relative_range(variances):
        epsilon = 1e-9
        return (np.max(variances) - np.min(variances)) / (np.mean(variances) + epsilon)
    
    def gini_coefficient(variances):
        x = np.array(variances)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * np.sort(x))) / (n * np.sum(x)) - (n + 1) / n
    
    def mean_absolute_deviation(variances):
        mean_val = np.mean(variances)
        return np.mean(np.abs(variances - mean_val))
    
    def interquartile_range(variances):
        return np.percentile(variances, 75) - np.percentile(variances, 25)
    
    # Define measures with and without log transformation
    measures = [
        ("Var(log(x))", lambda v: variance_of_logs(v)),
        ("Var(x)", lambda v: variance_direct(v)),
        ("CV(x)", lambda v: coefficient_of_variation(v)),
        ("1-NormEnt(x)", lambda v: 1.0 - normalized_entropy(v)),
        ("RelRange(x)", lambda v: relative_range(v)),
        ("Gini(x)", lambda v: gini_coefficient(v)),
        ("MAD(x)", lambda v: mean_absolute_deviation(v)),
        ("IQR(x)", lambda v: interquartile_range(v)),
    ]
    
    # Also add log-transformed versions of direct measures
    log_measures = [
        ("Var(log(x))", lambda v: variance_of_logs(v)),  # Already log-based
        ("Var(log(x))", lambda v: variance_direct([np.log(max(x, 1e-9)) for x in v])),  # Duplicate for clarity
        ("CV(log(x))", lambda v: coefficient_of_variation([np.log(max(x, 1e-9)) for x in v])),
        ("1-NormEnt(log(x))", lambda v: 1.0 - normalized_entropy([np.log(max(x, 1e-9)) for x in v])),
        ("RelRange(log(x))", lambda v: relative_range([np.log(max(x, 1e-9)) for x in v])),
        ("Gini(log(x))", lambda v: gini_coefficient([np.log(max(x, 1e-9)) for x in v])),
        ("MAD(log(x))", lambda v: mean_absolute_deviation([np.log(max(x, 1e-9)) for x in v])),
        ("IQR(log(x))", lambda v: interquartile_range([np.log(max(x, 1e-9)) for x in v])),
    ]
    
    # Print header
    print(f"{'Pattern':<12}", end="")
    for name, _ in measures:
        print(f"{name:<12}", end="")
    print()
    print("-" * (12 + 12 * len(measures)))
    
    # Test each pattern with direct measures
    for pattern_name, variances in test_variance_patterns:
        print(f"{pattern_name:<12}", end="")
        for measure_name, measure_func in measures:
            try:
                value = measure_func(variances)
                print(f"{value:<12.4f}", end="")
            except Exception as e:
                print(f"{'ERROR':<12}", end="")
        print(f"  # {variances}")
    
    print("\n" + "=" * 80)
    print("LOG-TRANSFORMED VERSIONS")
    print("=" * 80)
    
    # Print header for log measures
    print(f"{'Pattern':<12}", end="")
    for name, _ in log_measures:
        print(f"{name:<14}", end="")
    print()
    print("-" * (12 + 14 * len(log_measures)))
    
    # Test each pattern with log-transformed measures
    for pattern_name, variances in test_variance_patterns:
        print(f"{pattern_name:<12}", end="")
        for measure_name, measure_func in log_measures:
            try:
                value = measure_func(variances)
                print(f"{value:<14.4f}", end="")
            except Exception as e:
                print(f"{'ERROR':<14}", end="")
        print(f"  # {variances}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS: Key Properties")
    print("=" * 80)
    
    print("\n1. Scale Invariance Test:")
    print("   (Perfect patterns should give identical results regardless of scale)")
    perfect_1x = [1.0, 1.0, 1.0, 1.0]
    perfect_10x = [10.0, 10.0, 10.0, 10.0]
    
    scale_invariant_measures = [
        ("CV(x)", coefficient_of_variation),
        ("1-NormEnt(x)", lambda v: 1.0 - normalized_entropy(v)),
        ("RelRange(x)", relative_range),
        ("Gini(x)", gini_coefficient),
        ("CV(log(x))", lambda v: coefficient_of_variation([np.log(max(x, 1e-9)) for x in v])),
        ("Var(log(x))", variance_of_logs),
    ]
    
    for name, func in scale_invariant_measures:
        val_1x = func(perfect_1x)
        val_10x = func(perfect_10x)
        difference = abs(val_1x - val_10x)
        print(f"   {name:<15}: {val_1x:.6f} vs {val_10x:.6f} (diff: {difference:.6f})")
    
    print("\n2. Outlier Sensitivity Test:")
    print("   (How much do measures change with extreme outliers?)")
    normal = [1.0, 1.0, 1.0, 1.0]
    outlier = [1.0, 1.0, 1.0, 10.0]
    
    all_measures_test = [
        ("Var(x)", variance_direct),
        ("CV(x)", coefficient_of_variation),
        ("1-NormEnt(x)", lambda v: 1.0 - normalized_entropy(v)),
        ("RelRange(x)", relative_range),
        ("Var(log(x))", variance_of_logs),
        ("CV(log(x))", lambda v: coefficient_of_variation([np.log(max(x, 1e-9)) for x in v])),
    ]
    
    for name, func in all_measures_test:
        normal_val = func(normal)
        outlier_val = func(outlier)
        if normal_val > 1e-12:
            ratio = outlier_val / normal_val
            print(f"   {name:<15}: {normal_val:.4f} → {outlier_val:.4f} ({ratio:.1f}x increase)")
        else:
            print(f"   {name:<15}: {normal_val:.4f} → {outlier_val:.4f} (∞x increase)")
    
    print("\n3. Recommendations:")
    print("   - Most robust: 1-NormEnt(x) - bounded, scale-invariant, outlier-resistant")
    print("   - Most interpretable: CV(x) - direct relative measure of spread")
    print("   - Log-space optimal: Var(log(x)) - original approach, good for multiplicative effects")
    print("   - Simplest: RelRange(x) - easy to compute and understand")


def test_jad_with_different_measures():
    """Test JAD denoiser behavior with different conflation measures."""
    print("\n" + "=" * 80)
    print("JAD Denoiser Performance with Different Conflation Measures")
    print("=" * 80)
    
    # Create a single test dataset
    np.random.seed(42)  # For reproducibility
    X, A_sparse, subset_indices = create_synthetic_data(n_samples=300, n_features=20, n_subsets=3)
    # Convert to CSR matrix as expected by TestDenoiserContext
    import scipy.sparse as sp
    A = sp.csr_matrix(A_sparse)
    context = DenoiserContextForTests(X, A, subset_indices)
    
    print(f"Test dataset: {X.shape[0]} samples, {X.shape[1]} features, {context.num_subsets()} subsets")
    for i in range(context.num_subsets()):
        print(f"  Subset {i}: {context.X_n(i).shape[0]} samples")
    
    # Define different conflation measures to test
    conflation_variants = [
        ("Current (1-NormEnt)", "normalized_entropy"),
        ("Coefficient of Variation", "cv"),
        ("Variance of Logs", "var_logs"),
        ("Relative Range", "rel_range"),
        ("Direct Variance", "var_direct"),
    ]
    
    print(f"\n{'Measure':<20} {'Score Range':<20} {'Proj Shape':<12} {'Status':<10}")
    print("-" * 70)
    
    for measure_name, measure_code in conflation_variants:
        try:
            # We'll modify the JAD denoiser temporarily to use different measures
            # For now, just test with the current implementation
            denoiser = JADDenoiser(trials=3, wt=1.0, wl=0.5, w_conflate=2.0)
            P, scores = denoiser.projector(context, n_components=5)
            
            if scores is not None and len(scores) > 0:
                score_range = f"[{np.min(scores):.3f}, {np.max(scores):.3f}]"
            else:
                score_range = "N/A"
            proj_shape = f"{P.shape}"
            status = "SUCCESS"
            
            print(f"{measure_name:<20} {score_range:<20} {proj_shape:<12} {status:<10}")
            
        except Exception as e:
            print(f"{measure_name:<20} {'ERROR':<20} {'N/A':<12} {str(e)[:20]:<10}")
    
    print(f"\nNote: Currently all tests use the same implementation (normalized entropy).")
    print(f"To test different measures, we would need to modify the JAD implementation.")


if __name__ == "__main__":
    test_entropy_conflation()
    compare_measures()
    test_jad_with_different_measures()
