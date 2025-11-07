"""
K-ball gap finding algorithm implementation.
"""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bookgap.subcloud import SubCloud
from bookgap.cloud_index import get_kball_coverage, knn
from bookgap.gap_types import DistsAndIdxs
from bookgap.index import kballs_with

class KballAlgorithm:
    """Wrapper class for k-ball gap finding algorithm."""
    
    @staticmethod
    def find_gaps(ambient: SubCloud, landmark: SubCloud, query: SubCloud, 
            radius_fraction: float, k: int, is_intra_cloud: bool):
        """
        Find gaps using k-ball algorithm.
        
        Args:
            ambient: The ambient space to count points in (e.g., articles)
            landmark: The landmarks used to define distances (e.g., books)
            query: The query points to analyze for gaps (e.g., articles)
            radius_fraction: Fraction of distance to k-th nearest landmark to use as radius
            k: Which nearest landmark to use (1 = nearest, 2 = second nearest, etc.)
            is_intra_cloud: Whether landmark and query are the same cloud
        """
        return KballAlgorithm.kball_gaps(ambient, landmark, query, radius_fraction, k, is_intra_cloud)
    
    @staticmethod
    def kball_gaps(ambient: SubCloud, landmark: SubCloud, query: SubCloud, 
            radius_fraction: float, k: int, is_intra_cloud: bool):
        """
        Find gaps: query points where there are many closer ambient points than the nearest landmark points.
        
        Args:
            ambient: The ambient space to count points in (e.g., articles)
            landmark: The landmarks used to define distances (e.g., books)
            query: The query points to analyze for gaps (e.g., articles)
            radius_fraction: Fraction of distance to k-th nearest landmark to use as radius
            k: Which nearest landmark to use (1 = nearest, 2 = second nearest, etc.)
            is_intra_cloud: Whether landmark and query are the same cloud
        """
        covered_ambient_points = get_kball_coverage(ambient=ambient, landmark=landmark,
                                                    query=query, radius_fraction=radius_fraction,
                                                    k=k, is_intra_cloud=is_intra_cloud)
        while True:
            if len(covered_ambient_points) == 0:
                break
            highest_coverage_query = np.argmax([len(idxs) for _dists, idxs in covered_ambient_points])
            dists, idxs = covered_ambient_points[highest_coverage_query]
            if len(idxs) == 0:
                break
            order = np.argsort(dists)
            dists = dists[order]
            idxs = idxs[order]

            yield KballGap(
                ambient=ambient,
                landmark=landmark,
                query=query,
                radius_fraction=radius_fraction,
                k=k,
                dists_idxs=DistsAndIdxs(sqdists=dists, idxs=idxs))

            # Prune the gap
            for idx, (dists_, idxs_) in enumerate(covered_ambient_points):
                # Remove all covered ambient points from other query point coverages (this completely zeros out the current gap)
                mask = np.isin(idxs_, idxs, invert=True)
                covered_ambient_points[idx] = (dists_[mask], idxs_[mask])
    
    @staticmethod
    def compute_gap_sizes_for_embeddings(
        *,
        ambient: SubCloud,
        landmark: SubCloud,
        embeddings: NDArray[np.float32],
        radius_fraction: float,
        k: int,
        is_intra_cloud: bool,
    ) -> list[int]:
        """Return k-ball coverage counts for arbitrary query embeddings."""
        query = np.ascontiguousarray(embeddings)
        if query.ndim != 2:
            raise ValueError(f"Query embeddings must be 2D (N, D); received shape {query.shape}")

        counts = [-1] * query.shape[0]
        effective_k = k + 1 if is_intra_cloud else k
        if effective_k <= 0:
            raise ValueError(f"Effective k must be positive; got {effective_k}")

        def kernel_fn(query_idx: int, _neighbor_dists, neighbor_ids) -> None:
            counts[query_idx] = len(neighbor_ids)

        kballs_with(
            ambient=ambient.index(),
            landmark=landmark.index(),
            query=query,
            radius_fraction=radius_fraction,
            k=effective_k,
            kernel_fn=kernel_fn,
            show_progress=False,
        )

        if (num_negative := sum(1 for count in counts if count < 0)) > 0:
            raise RuntimeError(f"kballs_with did not visit every query point; counts contain {num_negative} negative values")

        return counts

    @property
    def name(self):
        return "kball"



@dataclass(frozen=True)
class KballGap:
    ambient: SubCloud  # The ambient space to count points in (e.g., articles)
    landmark: SubCloud  # The landmarks used to define distances (e.g., books)
    query: SubCloud  # The query points to analyze for gaps (e.g., articles)
    radius_fraction: float
    k: int  # Which nearest landmark to use
    dists_idxs: DistsAndIdxs  # Squared distances and indices of the ambient points that are in the gap.

    def get_dists_idxs(self) -> DistsAndIdxs:
        """Return the distances and indices of articles in the gap."""
        return self.dists_idxs
    
    def get_ambient(self) -> SubCloud:
        """Return the ambient space (articles)."""
        return self.ambient
        
    def get_landmark(self) -> SubCloud:
        """Return the landmark space (books)."""
        return self.landmark

    @dataclass(frozen=True)
    class CoveredPoints:
        ambient: SubCloud
        landmark: SubCloud
        query: SubCloud
        radius_fraction: float
        landmark_to_covered_ambient_points: dict[np.intp, DistsAndIdxs]  # List of squared distances and idxs for each ambient point covered by the corresponding landmark point.
    def closest_query_points(self):
        landmark_to_covered_ambient_idxs = defaultdict(list)
        landmark_to_covered_ambient_dists = defaultdict(list)
        res = knn(index=self.landmark, query=self.ambient, k=1)
        closest_landmark_to_each_ambient = res.I[self.dists_idxs.idxs].squeeze(1)
        if np.any(closest_landmark_to_each_ambient < 0):
            assert np.all(closest_landmark_to_each_ambient < 0)
            return KballGap.CoveredPoints(
                ambient=self.ambient,
                landmark=self.landmark,
                query=self.query,
                radius_fraction=self.radius_fraction,
                landmark_to_covered_ambient_points={}
            )

        closest_landmark_sqdists = res.sq_dists[self.dists_idxs.idxs].squeeze(1)

        for dist, ambient_idx, landmark_idx in zip(closest_landmark_sqdists, self.dists_idxs.idxs, closest_landmark_to_each_ambient):
            landmark_to_covered_ambient_idxs[landmark_idx].append(ambient_idx)
            landmark_to_covered_ambient_dists[landmark_idx].append(dist)
        landmark_idxs = np.unique(closest_landmark_to_each_ambient).tolist()  # Ensure landmark_idxs are unique
        assert set(landmark_idxs) == set(closest_landmark_to_each_ambient)
        assert set(landmark_idxs) == set(landmark_to_covered_ambient_idxs.keys())
        assert set(landmark_idxs) == set(landmark_to_covered_ambient_dists.keys())
        landmark_to_covered_ambient_points = {landmark_idx: DistsAndIdxs(sqdists=np.array(landmark_to_covered_ambient_dists[landmark_idx], dtype=np.float32),
                                             idxs=np.array(landmark_to_covered_ambient_idxs[landmark_idx], dtype=np.intp))
                            for landmark_idx in landmark_idxs}  # Exclude -1 indices which indicate no nearest landmark point found
        return KballGap.CoveredPoints(
            ambient=self.ambient,
            landmark=self.landmark,
            query=self.query,
            radius_fraction=self.radius_fraction,
            landmark_to_covered_ambient_points=landmark_to_covered_ambient_points
        )

    def __post_init__(self):
        if not np.all(np.diff(self.dists_idxs.sqdists) >= 0):
            raise ValueError("Squared distances must be sorted in non-decreasing order")
        if not self.dists_idxs.sqdists.ndim == 1:
            raise ValueError("Squared distances must be a 1D array")
        if not self.dists_idxs.sqdists.shape == self.dists_idxs.idxs.shape:
            raise ValueError("Squared distances and indices must have the same shape")

    def get_covering_book_indices(self) -> list[np.intp]:
        """Get the indices of books that cover articles in the gap."""
        covered_points_analysis = self.closest_query_points()
        return list(covered_points_analysis.landmark_to_covered_ambient_points.keys())

    def get_book_to_articles_mapping(self) -> dict[np.intp, DistsAndIdxs]:
        """Get mapping from book indices to their covered articles with distances."""
        return self.closest_query_points().landmark_to_covered_ambient_points
