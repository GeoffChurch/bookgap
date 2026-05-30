"""
Voronoi gap finding algorithm implementation.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from bookgap.cloud_index import knn
from bookgap.gap_types import DistsAndIdxs
from bookgap.index import hypothetical_voronoi_coverage
from bookgap.subcloud import SubCloud
from bookgap.corpus import Corpus


class VoronoiAlgorithm:
    """Wrapper class for Voronoi gap finding algorithm."""
    
    @staticmethod
    def find_gaps(ambient: Corpus, landmark: Corpus, query: Corpus, radius_fraction=None, k=None, is_intra_cloud=None, deflation_strategy=None):
        # deflation_strategy is accepted for a uniform CellType.value.find_gaps
        # signature but intentionally ignored: Voronoi gaps come from its own
        # tessellation ordering (descending cell size), not k-ball candidate
        # deflation, so there is no greedy-vs-resid choice to make here.
        assert query != landmark, "Use actual_voronoi_coverage when query and landmark are the same cloud"
        query_embeddings = query.embeddings()
        cell_sizes = hypothetical_voronoi_coverage(
            ambient  = ambient.index(),
            landmark = landmark.index(),
            query    = query_embeddings,
            ambient_is_landmark = ambient==landmark)
        
        # Find nearest landmark to each ambient point
        k = 2 if landmark == ambient else 1  # Exclude self-neighbor if landmark and ambient are the same
        res = knn(index=landmark, query=ambient, k=k)
        ambient_to_nearest_landmark_sqdist = res.sq_dists[:, -1]
        ambient_nearest_landmark = res.I[:, -1]

        ambient_embeddings = ambient.embeddings()
        argsort = np.argsort(-cell_sizes)  # Descending order of voronoi cell sizes
        for query_idx in argsort:
            _count = cell_sizes[query_idx]
            embedding = query_embeddings[query_idx]
            
            # Find the ambient points that are closer to this query point than to any landmark
            ambient_sq_dist_to_query = np.sum((ambient_embeddings - embedding) ** 2, axis=1)
            reassigned_idxs = np.where(ambient_sq_dist_to_query < ambient_to_nearest_landmark_sqdist)[0]
            reassigned_dists = ambient_sq_dist_to_query[reassigned_idxs]
            
            order = np.argsort(reassigned_dists)
            reassigned_idxs = reassigned_idxs[order]
            reassigned_dists = reassigned_dists[order]
            
            if query == ambient:
                assert reassigned_idxs[0] == query_idx, f"Expected the closest ambient point to be the query point itself, got {reassigned_idxs[0]} instead"
                assert reassigned_dists[0] == 0, f"Expected the squared distance to the query point to be 0, got {reassigned_dists[0]} instead"
            
            
            yield VoronoiGap(
                ambient=ambient, landmark=landmark, query=query,
                ambient_idxs=reassigned_idxs,
                ambient_to_nearest_landmark_idxs=ambient_nearest_landmark[reassigned_idxs],
                ambient_to_query_sqdists=reassigned_dists,
                query_idx=query_idx)

    @staticmethod
    def compute_gap_sizes_for_embeddings(
        *,
        ambient: Corpus,
        landmark: Corpus,
        embeddings: NDArray[np.float32],
        radius_fraction: float | None,
        k: int | None,
        is_intra_cloud: bool | None,
        show_progress: bool = False,
    ) -> NDArray[np.int32]:
        """Return Voronoi coverage counts for arbitrary query embeddings."""

        if embeddings.size == 0:
            return np.zeros((0,), dtype=np.int32)

        query_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Query embeddings must be 2D (N, D); received shape {query_embeddings.shape}"
            )

        counts = hypothetical_voronoi_coverage(
            ambient=ambient.index(),
            landmark=landmark.index(),
            query=query_embeddings,
            ambient_is_landmark=ambient == landmark,
            show_progress=show_progress
        )

        return counts.astype(np.int32, copy=False)
    
    @property
    def name(self):
        return "voronoi"


@dataclass(frozen=True)
class VoronoiGap:
    ambient: SubCloud
    landmark: SubCloud
    query: SubCloud
    ambient_idxs: NDArray[np.intp]
    ambient_to_nearest_landmark_idxs: NDArray[np.intp]
    ambient_to_query_sqdists: NDArray[np.float32]  # Squared distances from query point to ambient points (sorted)
    query_idx: Optional[int] = None  # Index of the query point that defines this gap (if applicable)
        
    @property
    def dists_idxs(self) -> DistsAndIdxs:
        # Return the pre-computed sorted squared distances and indices
        return DistsAndIdxs(
            sqdists=self.ambient_to_query_sqdists,
            idxs=self.ambient_idxs
        )

    def get_dists_idxs(self) -> DistsAndIdxs:
        """Return the distances and indices of articles in the gap."""
        return self.dists_idxs
    
    def get_ambient(self) -> SubCloud:
        """Return the ambient space (articles)."""
        return self.ambient

    def get_landmark(self) -> SubCloud:
        """Return the landmark space (books)."""
        return self.landmark

    def get_covering_book_indices(self) -> list[np.intp]:
        """Get the indices of books that cover articles in the gap."""
        return list(np.unique(self.ambient_to_nearest_landmark_idxs))

    def get_book_to_articles_mapping(self) -> dict[np.intp, DistsAndIdxs]:
        """Get mapping from book indices to their covered articles with distances."""
        book_to_articles = {}
        book_coverage = defaultdict(list)
        
        # Group articles by covering books
        for gap_pos, landmark_idx in enumerate(self.ambient_to_nearest_landmark_idxs):
            article_idx = self.ambient_idxs[gap_pos]
            book_coverage[landmark_idx].append(article_idx)
        
        # Get embeddings for proper distance calculations
        landmark_embeddings = self.landmark.embeddings()
        ambient_embeddings = self.ambient.embeddings()
        
        # Compute actual book→article distances (not query→article!)
        for book_idx, article_indices in book_coverage.items():
            book_embedding = landmark_embeddings[book_idx] if book_idx >= 0 else np.full(ambient_embeddings.shape[1], np.nan)
            article_embeddings = ambient_embeddings[article_indices]
            
            # Compute squared distances from book to covered articles
            book_to_article_sqdists = np.sum((article_embeddings - book_embedding) ** 2, axis=1)
            
            # Sort by distance
            order = np.argsort(book_to_article_sqdists)
            sorted_sqdists = book_to_article_sqdists[order].astype(np.float32)
            sorted_idxs = np.array(article_indices, dtype=np.intp)[order]
            
            book_to_articles[book_idx] = DistsAndIdxs(
                sqdists=sorted_sqdists,
                idxs=sorted_idxs
            )
        
        return book_to_articles