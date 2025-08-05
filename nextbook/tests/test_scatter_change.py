#!/usr/bin/env python3
"""
Quick test to verify that the scatter matrix change doesn't significantly affect JAD results.
"""

import numpy as np
from nextbook.denoisers import scatter

# Test with sample data similar to your dataset sizes
np.random.seed(42)

# Simulate book and article embeddings
n_articles = 1465427
n_books = 17183
n_dims = 384

# Create sample data (much smaller for testing)
test_articles = np.random.randn(1000, n_dims)
test_books = np.random.randn(20, n_dims)

# Compute scatter matrices
S_articles = scatter(test_articles)
S_books = scatter(test_books)

print(f"Articles scatter matrix: mean eigenvalue = {np.mean(np.linalg.eigvals(S_articles)):.6f}")
print(f"Books scatter matrix: mean eigenvalue = {np.mean(np.linalg.eigvals(S_books)):.6f}")

# Check the scaling factor
n_articles_factor = test_articles.shape[0] / (test_articles.shape[0] - 1)
n_books_factor = test_books.shape[0] / (test_books.shape[0] - 1)

print(f"Articles scaling factor (n/(n-1)): {n_articles_factor:.8f}")
print(f"Books scaling factor (n/(n-1)): {n_books_factor:.8f}")

# For your actual dataset sizes:
actual_articles_factor = n_articles / (n_articles - 1)
actual_books_factor = n_books / (n_books - 1)

print(f"\nActual dataset scaling factors:")
print(f"Articles: {actual_articles_factor:.10f}")
print(f"Books: {actual_books_factor:.8f}")
print(f"Difference from 1.0: Articles={actual_articles_factor-1:.2e}, Books={actual_books_factor-1:.2e}")
