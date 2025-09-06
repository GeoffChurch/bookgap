#!/usr/bin/env python3
"""
Test script to verify Q-Q plot generation for book gap sizes.
"""

import numpy as np
from nextbook.gap_statistics import create_book_gap_sizes_qqplot
from nextbook.config_manager import AnalysisConfiguration

def test_qqplot():
    print("Testing Q-Q plot generation...")
    
    # Test case 1: Normal-ish distribution
    np.random.seed(42)
    book_gap_sizes = np.random.normal(20, 5, 50).astype(int).tolist()
    book_gap_sizes = [max(1, size) for size in book_gap_sizes]  # Ensure positive sizes
    
    print(f"Test 1 - Normal-ish distribution: {len(book_gap_sizes)} book gaps")
    print(f"  Mean: {np.mean(book_gap_sizes):.2f}")
    print(f"  Std: {np.std(book_gap_sizes, ddof=1):.2f}")
    
    # Create consolidated analysis configuration
    analysis_config = AnalysisConfiguration(
        denoiser="TEST",
        drop_k=0,
        radius_fraction=0.1,
        cell_type="kball",
        requested_num_prompt_articles=10,
        requested_num_prompt_books=3
    )
    
    create_book_gap_sizes_qqplot(
        book_gap_sizes=book_gap_sizes,
        topic_name="Test Normal Distribution",
        topic_id="test_normal",
        analysis_config=analysis_config
    )
    print("  Q-Q plot created successfully!")
    print()
    
    # Test case 2: Skewed distribution
    book_gap_sizes_skewed = np.random.exponential(10, 30).astype(int).tolist()
    book_gap_sizes_skewed = [max(1, size) for size in book_gap_sizes_skewed]  # Ensure positive sizes
    
    print(f"Test 2 - Skewed distribution: {len(book_gap_sizes_skewed)} book gaps")
    print(f"  Mean: {np.mean(book_gap_sizes_skewed):.2f}")
    print(f"  Std: {np.std(book_gap_sizes_skewed, ddof=1):.2f}")
    
    create_book_gap_sizes_qqplot(
        book_gap_sizes=book_gap_sizes_skewed,
        topic_name="Test Skewed Distribution",
        topic_id="test_skewed",
        analysis_config=analysis_config
    )
    print("  Q-Q plot created successfully!")
    print()
    
    # Test case 3: Insufficient data
    small_book_gaps = [10, 15]
    print(f"Test 3 - Insufficient data: {len(small_book_gaps)} book gaps")
    
    create_book_gap_sizes_qqplot(
        book_gap_sizes=small_book_gaps,
        topic_name="Test Insufficient Data",
        topic_id="test_insufficient",
        analysis_config=analysis_config
    )
    print("  Test completed (should skip with insufficient data message)")
    
    print("\nAll Q-Q plot tests completed!")

if __name__ == "__main__":
    test_qqplot()
