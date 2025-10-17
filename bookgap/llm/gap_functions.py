from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class DistsAndIdxs:
    """Container for squared distances and corresponding indices."""
    sqdists: NDArray[np.float32]  # Squared distances
    idxs: NDArray[np.intp]      # Corresponding indices






    
