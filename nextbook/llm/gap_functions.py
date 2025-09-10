from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from nextbook.cloud import SubCloud
from nextbook.cloud_index import get_kball_coverage, knn
from nextbook.index import hypothetical_voronoi_coverage
from nextbook.util import LOGGER


@dataclass(frozen=True)
class DistsAndIdxs:
    """Container for squared distances and corresponding indices."""
    sqdists: NDArray[np.float32]  # Squared distances
    idxs: NDArray[np.intp]      # Corresponding indices






    
