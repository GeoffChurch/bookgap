"""
Shared types for gap analysis.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DistsAndIdxs:
    """Container for squared distances and indices."""
    sqdists: np.ndarray  # squared distances
    idxs: np.ndarray  # indices
