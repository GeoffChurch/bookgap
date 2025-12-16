"""Sampling utilities for curriculum building.

Provides functions to select relevant items from topics using PageRank scores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_by_relevance(
    item_indices: NDArray[np.intp],
    relevance_scores: NDArray[np.floating],
    n_samples: int,
) -> NDArray[np.intp]:
    """Select top N most relevant items from a set using relevance scores.
    
    Args:
        item_indices: Array of item indices to sample from.
        relevance_scores: Array of relevance scores for all items (indexed by item_indices).
        n_samples: Number of items to select.
        
    Returns:
        Array of selected item indices, sorted by relevance (highest first).
    """
    # Get relevance scores for items in this set
    scores_for_items = relevance_scores[item_indices]
    
    # Select top N by relevance
    n_to_select = min(n_samples, len(item_indices))
    top_local_indices = np.argsort(scores_for_items)[-n_to_select:][::-1]  # Sort descending
    
    # Return the actual item indices (not positions in item_indices array)
    return item_indices[top_local_indices]
