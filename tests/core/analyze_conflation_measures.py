"""
Analysis of different conflation measures for the JAD denoiser.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Callable


def variance_of_logs(variances: List[float]) -> float:
    """Original approach: variance of log-variances."""
    if len(variances) <= 1:
        return 0.0
    log_vars = [np.log(v) for v in variances if v > 0]
    if len(log_vars) != len(variances):
        return float('nan')  # Some values were zero or negative
    return np.var(log_vars, ddof=0)


def coefficient_of_variation(variances: List[float]) -> float:
    """Coefficient of variation: std/mean."""
    if len(variances) <= 1:
        return 0.0
    mean_val = np.mean(variances)
    if mean_val == 0:
        return float('inf') if np.std(variances) > 0 else float('nan')
    return np.std(variances) / mean_val


def normalized_entropy(variances: List[float]) -> float:
    """Normalized entropy of variance distribution."""
    if len(variances) <= 1:
        return 0.0
    
    # Normalize to probabilities
    total = sum(variances)
    if total == 0:
        return float('nan')
    
    probs = [v / total for v in variances]
    
    # Compute entropy, skipping zero probabilities
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    
    # Normalize by max possible entropy (log(n))
    max_entropy = np.log(len(variances))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def relative_range(variances: List[float]) -> float:
    """Relative range: (max - min) / mean."""
    if len(variances) <= 1:
        return 0.0
    mean_val = np.mean(variances)
    if mean_val == 0:
        range_val = np.max(variances) - np.min(variances)
        return float('inf') if range_val > 0 else float('nan')
    return (np.max(variances) - np.min(variances)) / mean_val


def gini_coefficient(variances: List[float]) -> float:
    """Gini coefficient of inequality."""
    if len(variances) <= 1:
        return 0.0
    
    x = np.array(variances)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * np.sort(x))) / (n * np.sum(x)) - (n + 1) / n


def jensen_shannon_divergence(variances: List[float]) -> float:
    """Jensen-Shannon divergence from uniform distribution."""
    if len(variances) <= 1:
        return 0.0
    
    # Normalize to probability distribution
    total = sum(variances)
    if total == 0:
        return float('nan')
    
    p = np.array([v / total for v in variances])
    
    # Uniform distribution
    q = np.ones(len(p)) / len(p)
    
    # Average distribution
    m = (p + q) / 2
    
    # JS divergence
    js = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
    return float(js)


def analyze_conflation_measures():
    """
    Analyze different conflation measures on various test cases.
    """
    
    # Define all measures with and without log transformation
    def variance_direct(variances):
        return np.var(variances, ddof=0)
    
    def mean_absolute_deviation(variances):
        mean_val = np.mean(variances)
        return np.mean(np.abs(variances - mean_val))
    
    def interquartile_range(variances):
        return np.percentile(variances, 75) - np.percentile(variances, 25)
    
    # Direct measures (no log transformation)
    direct_measures = {
        'Var(x)': variance_direct,
        'CV(x)': coefficient_of_variation,
        'NormEnt(x)': normalized_entropy,
        'RelRange(x)': relative_range,
        'Gini(x)': gini_coefficient,
        'JS-Div(x)': jensen_shannon_divergence,
        'MAD(x)': mean_absolute_deviation,
        'IQR(x)': interquartile_range,
    }
    
    # Log-transformed measures
    def apply_log_transform(measure_func):
        def log_transformed(variances):
            if len(variances) <= 1:
                return 0.0
            # Only take log of positive values
            log_variances = [np.log(v) for v in variances if v > 0]
            if len(log_variances) != len(variances):
                return float('nan')  # Some values were zero or negative
            return measure_func(log_variances)
        return log_transformed
    
    log_measures = {
        'Var(log(x))': variance_of_logs,  # Already implemented
        'CV(log(x))': apply_log_transform(coefficient_of_variation),
        'NormEnt(log(x))': apply_log_transform(normalized_entropy),
        'RelRange(log(x))': apply_log_transform(relative_range),
        'Gini(log(x))': apply_log_transform(gini_coefficient),
        'JS-Div(log(x))': apply_log_transform(jensen_shannon_divergence),
        'MAD(log(x))': apply_log_transform(mean_absolute_deviation),
        'IQR(log(x))': apply_log_transform(interquartile_range),
    }
    
    # Test cases: different patterns of subset variances
    test_cases = {
        'Perfect Conflation': [1.0, 1.0, 1.0, 1.0],  # All equal (ideal)
        'Moderate Spread': [0.8, 1.0, 1.2, 1.0],     # Small differences
        'High Spread': [0.5, 1.0, 1.5, 2.0],         # Large differences
        'One Outlier': [1.0, 1.0, 1.0, 3.0],         # One very different
        'Two Groups': [0.5, 0.5, 2.0, 2.0],          # Two distinct groups
        'Linear Trend': [0.5, 1.0, 1.5, 2.0],        # Monotonic increase
        'Extreme Outlier': [1.0, 1.0, 1.0, 10.0],    # One extreme outlier
        'Scale Test': [10.0, 10.0, 10.0, 10.0],      # Same pattern, different scale
        'Near Zero': [0.01, 0.01, 0.01, 0.01],       # Test numerical stability
        'Mixed Scale': [0.1, 1.0, 10.0, 100.0],      # Wide range of scales
    }
    
    print("Comprehensive Conflation Measure Analysis")
    print("=" * 100)
    print("DIRECT MEASURES (no log transformation)")
    print("=" * 100)
    print(f"{'Test Case':<18} ", end="")
    for measure_name in direct_measures.keys():
        print(f"{measure_name:<12}", end="")
    print()
    print("-" * 100)
    
    direct_results = {}
    for case_name, variances in test_cases.items():
        print(f"{case_name:<18} ", end="")
        case_results = {}
        
        for measure_name, measure_func in direct_measures.items():
            try:
                value = measure_func(variances)
                case_results[measure_name] = value
                if np.isfinite(value):
                    print(f"{value:<12.4f}", end="")
                else:
                    print(f"{'INF/NAN':<12}", end="")
            except Exception as e:
                case_results[measure_name] = float('nan')
                print(f"{'ERROR':<12}", end="")
        
        direct_results[case_name] = case_results
        print(f"  # {variances}")
    
    print("\n" + "=" * 100)
    print("LOG-TRANSFORMED MEASURES")
    print("=" * 100)
    print(f"{'Test Case':<18} ", end="")
    for measure_name in log_measures.keys():
        print(f"{measure_name:<14}", end="")
    print()
    print("-" * 100)
    
    log_results = {}
    for case_name, variances in test_cases.items():
        print(f"{case_name:<18} ", end="")
        case_results = {}
        
        for measure_name, measure_func in log_measures.items():
            try:
                value = measure_func(variances)
                case_results[measure_name] = value
                if np.isfinite(value):
                    print(f"{value:<14.4f}", end="")
                else:
                    print(f"{'INF/NAN':<14}", end="")
            except Exception as e:
                case_results[measure_name] = float('nan')
                print(f"{'ERROR':<14}", end="")
        
        log_results[case_name] = case_results
        print(f"  # {variances}")
    
    print("\n" + "=" * 100)
    
    # Analysis of properties
    print("\nDesirable Properties Analysis:")
    print("-" * 50)
    
    print("\n1. Scale Invariance Test:")
    print("   (Perfect patterns should give identical results regardless of scale)")
    perfect_1x = [1.0, 1.0, 1.0, 1.0]
    perfect_10x = [10.0, 10.0, 10.0, 10.0]
    
    scale_test_measures = [
        ('CV(x)', coefficient_of_variation),
        ('NormEnt(x)', normalized_entropy),
        ('RelRange(x)', relative_range),
        ('Gini(x)', gini_coefficient),
        ('Var(log(x))', variance_of_logs),
        ('JS-Div(x)', jensen_shannon_divergence),
    ]
    
    for name, func in scale_test_measures:
        val_1x = func(perfect_1x)
        val_10x = func(perfect_10x)
        difference = abs(val_1x - val_10x)
        print(f"   {name:<15}: {val_1x:.6f} vs {val_10x:.6f} (diff: {difference:.8f})")
    
    print("\n2. Outlier Sensitivity Test:")
    print("   (How much do measures change with extreme outliers?)")
    normal = [1.0, 1.0, 1.0, 1.0]
    outlier = [1.0, 1.0, 1.0, 10.0]
    
    sensitivity_test_measures = [
        ('Var(x)', variance_direct),
        ('CV(x)', coefficient_of_variation),
        ('NormEnt(x)', normalized_entropy),
        ('RelRange(x)', relative_range),
        ('Gini(x)', gini_coefficient),
        ('Var(log(x))', variance_of_logs),
        ('JS-Div(x)', jensen_shannon_divergence),
    ]
    
    for name, func in sensitivity_test_measures:
        normal_val = func(normal)
        outlier_val = func(outlier)
        if normal_val > 1e-12:
            ratio = outlier_val / normal_val
            print(f"   {name:<15}: {normal_val:.4f} → {outlier_val:.4f} ({ratio:.1f}x increase)")
        else:
            print(f"   {name:<15}: {normal_val:.4f} → {outlier_val:.4f} (∞x increase)")
    
    print("\n3. Numerical Stability Test:")
    print("   (Behavior with very small or very large values)")
    small_vals = [1e-6, 1e-6, 1e-6, 1e-6]
    large_vals = [1e6, 1e6, 1e6, 1e6]
    
    for name, func in scale_test_measures:
        try:
            small_result = func(small_vals)
            large_result = func(large_vals)
            print(f"   {name:<15}: small={small_result:.6f}, large={large_result:.6f}")
        except Exception as e:
            print(f"   {name:<15}: ERROR - {str(e)[:30]}")
    
    print("\n4. Key Findings from Test Results:")
    
    # Extract actual normalized entropy values for key test cases
    perfect_entropy = direct_results['Perfect Conflation']['NormEnt(x)']
    moderate_entropy = direct_results['Moderate Spread']['NormEnt(x)']
    high_entropy = direct_results['High Spread']['NormEnt(x)']
    extreme_entropy = direct_results['Extreme Outlier']['NormEnt(x)']
    
    print("   - Normalized Entropy as CONFLATION PENALTY (1 - normalized_entropy):")
    print(f"     * Perfect conflation: 1-{perfect_entropy:.4f} = {1-perfect_entropy:.4f} penalty (no penalty)")
    print(f"     * Moderate spread: 1-{moderate_entropy:.4f} = {1-moderate_entropy:.4f} penalty (low)")
    print(f"     * High spread: 1-{high_entropy:.4f} = {1-high_entropy:.4f} penalty (moderate)")
    print(f"     * Extreme outlier: 1-{extreme_entropy:.4f} = {1-extreme_entropy:.4f} penalty (high)")
    print("   - All measures are scale-invariant for perfect conflation")
    print("   - Log-transformed measures are unstable without epsilon (INF/NAN)")
    print("   - Direct variance grows quadratically with outliers")
    print("   - Without epsilon stabilization, true mathematical behavior emerges")
    
    print("\n5. Analysis-Based Recommendations for JAD Conflation Penalty:")
    
    # Extract actual values for analysis
    perfect_norm_ent = 1 - direct_results['Perfect Conflation']['NormEnt(x)']
    moderate_norm_ent = 1 - direct_results['Moderate Spread']['NormEnt(x)']
    high_norm_ent = 1 - direct_results['High Spread']['NormEnt(x)']
    extreme_norm_ent = 1 - direct_results['Extreme Outlier']['NormEnt(x)']
    
    perfect_cv = direct_results['Perfect Conflation']['CV(x)']
    moderate_cv = direct_results['Moderate Spread']['CV(x)']
    high_cv = direct_results['High Spread']['CV(x)']
    extreme_cv = direct_results['Extreme Outlier']['CV(x)']
    
    perfect_var_log = log_results['Perfect Conflation']['Var(log(x))']
    moderate_var_log = log_results['Moderate Spread']['Var(log(x))']
    high_var_log = log_results['High Spread']['Var(log(x))']
    extreme_var_log = log_results['Extreme Outlier']['Var(log(x))']
    
    # Analyze key properties
    measures_analysis = {
        'Normalized Entropy Penalty (1-NormEnt)': {
            'perfect': perfect_norm_ent,
            'moderate': moderate_norm_ent,
            'high': high_norm_ent,
            'extreme': extreme_norm_ent,
            'bounded': True,
            'scale_invariant': True,
            'stable': True,
            'interpretable': 'Information loss measure'
        },
        'Coefficient of Variation': {
            'perfect': perfect_cv,
            'moderate': moderate_cv,
            'high': high_cv,
            'extreme': extreme_cv,
            'bounded': False,
            'scale_invariant': True,
            'stable': True,
            'interpretable': 'Relative variability'
        },
        'Variance of Log-Variances': {
            'perfect': perfect_var_log,
            'moderate': moderate_var_log,
            'high': high_var_log,
            'extreme': extreme_var_log,
            'bounded': False,
            'scale_invariant': True,
            'stable': np.isfinite(moderate_var_log) and np.isfinite(high_var_log),
            'interpretable': 'Multiplicative spread'
        }
    }
    
    # Score each measure based on desirable properties
    print("   Comparative Analysis:")
    for name, props in measures_analysis.items():
        print(f"\n   {name}:")
        print(f"      Perfect: {props['perfect']:.4f}, Moderate: {props['moderate']:.4f}, High: {props['high']:.4f}, Extreme: {props['extreme']:.4f}")
        
        # Check if perfect conflation gives near-zero penalty
        perfect_good = abs(props['perfect']) < 0.001
        
        # Check if there's good separation between levels
        moderate_sep = props['moderate'] > props['perfect'] + 0.001
        high_sep = props['high'] > props['moderate'] + 0.001
        extreme_sep = props['extreme'] > props['high'] + 0.001
        
        print(f"      Properties: Bounded={props['bounded']}, Scale-Invariant={props['scale_invariant']}, Stable={props['stable']}")
        print(f"      Zero penalty for perfect: {perfect_good}, Good separation: {moderate_sep and high_sep and extreme_sep}")
        print(f"      Interpretation: {props['interpretable']}")
    
    # Determine best measure based on analysis
    print("\n   Ranking based on performance:")
    
    scores = {}
    for name, props in measures_analysis.items():
        score = 0
        # Perfect conflation should give ~0 penalty
        if abs(props['perfect']) < 0.001:
            score += 3
        # Should have good separation between levels
        if (props['moderate'] > props['perfect'] + 0.001 and 
            props['high'] > props['moderate'] + 0.001 and 
            props['extreme'] > props['high'] + 0.001):
            score += 3
        # Bounded output is desirable
        if props['bounded']:
            score += 2
        # Numerical stability
        if props['stable']:
            score += 2
        # Scale invariance
        if props['scale_invariant']:
            score += 1
        
        scores[name] = score
    
    # Sort by score
    ranked_measures = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(ranked_measures):
        rank_emoji = ['🏆', '🥈', '🥉'][min(i, 2)]
        print(f"   {rank_emoji} {name} (Score: {score}/11)")
        props = measures_analysis[name]
        print(f"      Values: Perfect={props['perfect']:.4f}, Moderate={props['moderate']:.4f}, High={props['high']:.4f}, Extreme={props['extreme']:.4f}")
    
    if len(ranked_measures) > 0:
        best_measure = ranked_measures[0][0]
        print(f"\n   ✅ Recommended: {best_measure}")
        print("      Based on: zero penalty for perfect conflation, good separation between levels,")
        print("      bounded output, numerical stability, and theoretical soundness.")


def create_conflation_measure_variants():
    """
    Create and test different variants of the JAD conflation measure.
    """
    print("\n" + "=" * 80)
    print("JAD Denoiser Conflation Measure Implementation Guide")
    print("=" * 80)
    
    print("""
# Implementation Options (ranked by analysis results):

## Option 1: Normalized Entropy Penalty (information-theoretic, bounded)
subset_vars_normalized = np.array(subset_variances) / np.sum(subset_variances)
entropy = -sum(p * np.log(p) for p in subset_vars_normalized if p > 0)
max_entropy = np.log(len(subset_variances))
normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
conflation_penalty = 1.0 - normalized_entropy  # 0=perfect, 1=worst

## Option 2: Coefficient of Variation (simple, interpretable)
mean_val = np.mean(subset_variances)
conflation_penalty = np.std(subset_variances) / mean_val if mean_val > 0 else float('inf')

## Option 3: Variance of Log-Variances (requires epsilon stabilization)
epsilon = 1e-12  # For numerical stability
log_subset_variances = [np.log(max(s, epsilon)) for s in subset_variances]
conflation_penalty = np.var(log_subset_variances, ddof=0)

## Implementation Notes:
# - Choice should be based on the analysis results above
# - All measures are scale-invariant and have proper mathematical foundations
# - Consider numerical stability requirements for your specific use case
# - Normalized entropy provides bounded, interpretable output
# - Coefficient of variation is simplest and most intuitive
# - Variance of logs matches original mathematical formulation but needs careful handling
""")


if __name__ == "__main__":
    analyze_conflation_measures()
    create_conflation_measure_variants()
