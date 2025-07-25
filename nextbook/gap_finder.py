from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from nextbook.cloud import SubCloud, get_ball_coverage
from nextbook.util import LOGGER

@dataclass(frozen=True)
class Gap:
    index: SubCloud
    query: SubCloud
    radius_fraction: float
    dists: NDArray[np.float32]
    idxs: NDArray[np.intp]

def gaps(index: SubCloud, query: SubCloud, radius_fraction):
    covered_points = get_ball_coverage(index, query, radius_fraction=radius_fraction).covered_points
    while True:
        highest_coverage_idx = np.argmax([len(idxs) for _dists, idxs in covered_points])
        dists, idxs = covered_points[highest_coverage_idx]
        if len(idxs) == 0:
            LOGGER.info("No more gaps found")
            break
        assert highest_coverage_idx in idxs
        yield Gap(index=index, query=query, radius_fraction=radius_fraction, dists=dists, idxs=idxs)
        for idx in idxs:
            # Prevent all covered points from being topic centers
            covered_points[idx] = (np.array([], dtype=dists.dtype), np.array([], dtype=idxs.dtype))
        for idx, (dists_, idxs_) in enumerate(covered_points):
            # Remove all points in idxs from all other covered points
            mask = np.isin(idxs_, idxs, invert=True)
            covered_points[idx] = (dists_[mask], idxs_[mask])
