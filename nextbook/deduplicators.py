"""
Deduplication strategies for removing duplicate works based on embeddings.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray
import tqdm

from nextbook.index import IndexInfo, range_search_, range_search_with_
from nextbook.util import TIMER, LOGGER


@dataclass(frozen=True)
class Deduplicator(ABC):
    """Abstract base class for deduplication strategies."""
    
    @abstractmethod
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo) -> NDArray[np.uint64]:
        """
        Deduplicate embeddings and return indices of elements to retain.
        
        Args:
            undeduped_embeddings: Array of embeddings to deduplicate, shape (N, D)
            index_info: Index configuration for building search indices
            
        Returns:
            Array of indices to retain, shape (M,) where M <= N
        """
        ...
    

@dataclass(frozen=True)
class FractionalDeduplicator(Deduplicator):
    """
    Deduplicator that removes a fixed fraction of duplicates based on nearest neighbor distances.
    This is prone to chain reactions that leave only the oldest point in a chain, which can be
    avoided by using SqDistDeduplicator. SqDistDeduplicator is also more efficient.
    """
    duplicate_fraction: float
    
    def __post_init__(self):
        assert 0 <= self.duplicate_fraction <= 1, f"duplicate_fraction must be in [0, 1], got {self.duplicate_fraction}"
    
    @TIMER.time(formatters=dict(undeduped_embeddings=lambda embeddings: f"<shape={embeddings.shape}>"))
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo):
        num_to_drop = int(self.duplicate_fraction * (len(undeduped_embeddings) - 1))
        LOGGER.info(f"Using duplicate_fraction={self.duplicate_fraction}, dropping {num_to_drop} points")

        if num_to_drop > 0 and len(undeduped_embeddings) > 1:
            # Compute distances to nearest older neighbor
            dist_to_nearest_older = []
            # Build index incrementally, but in batches for efficiency
            main_index = index_info.index_type.build(undeduped_embeddings[:1], index_info) # Start with oldest work
            batch_size = 8192
            
            desc = f"Finding nearest older neighbors for fractional deduplication"
            with tqdm.tqdm(total=len(undeduped_embeddings)-1, desc=desc) as pbar:
                for i in range(1, len(undeduped_embeddings), batch_size):
                    batch_start = i
                    batch_end = min(i + batch_size, len(undeduped_embeddings))
                    query_batch = undeduped_embeddings[batch_start:batch_end]

                    # Find nearest neighbor in the main_index (all points older than this batch)
                    D_outside, _ = main_index.search(query_batch, 1) # type: ignore

                    # For each point in the batch, find its nearest neighbor among older points *within the batch*
                    # and take the minimum of that and the distance to the nearest neighbor outside the batch.
                    if len(query_batch) > 1:
                        # Use a temporary flat index for within-batch searches, as it's fast to build.
                        import faiss
                        temp_index = faiss.IndexFlatL2(query_batch.shape[1])
                        temp_index.add(query_batch[0:1]) # type: ignore
                        
                        # First element of batch can only have older neighbors outside the batch
                        dist_to_nearest_older.append(D_outside[0, 0])

                        for j in range(1, len(query_batch)):
                            query_vec = query_batch[j:j+1]
                            D_inside, _ = temp_index.search(query_vec, 1) # type: ignore
                            dist_to_nearest_older.append(min(D_outside[j, 0], D_inside[0, 0]))
                            temp_index.add(query_vec) # type: ignore
                    elif len(query_batch) == 1: # batch size is 1, or last batch has 1 element
                        dist_to_nearest_older.append(D_outside[0, 0])

                    # Add the processed batch to the main index for the next iteration
                    main_index.add(query_batch) # type: ignore
                    pbar.update(len(query_batch))

            assert len(dist_to_nearest_older) == len(undeduped_embeddings) - 1, f"Expected {len(undeduped_embeddings) - 1} distances, got {len(dist_to_nearest_older)}"

            argsort = np.argsort(dist_to_nearest_older)
            retained_indices = np.concatenate(([0], argsort[num_to_drop:] + 1), dtype=np.uint64)  # +1 because we dropped the first element
        else:
            retained_indices = np.arange(len(undeduped_embeddings), dtype=np.uint64)

        LOGGER.info(f"Retaining {len(retained_indices)} points out of {len(undeduped_embeddings)}")
        return retained_indices


@dataclass(frozen=True)
class SqDistDeduplicator(Deduplicator):
    """Deduplicator that removes points with older neighbors within a squared distance threshold."""
    min_sqdist: float
    
    def __post_init__(self):
        assert self.min_sqdist >= 0, f"min_sqdist must be non-negative, got {self.min_sqdist}"
    
    def description(self) -> str:
        return f"squared distance deduplication (min_sqdist={self.min_sqdist})"
    
    @TIMER.time(formatters=dict(undeduped_embeddings=lambda embeddings: f"<shape={embeddings.shape}>"))
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo):
        LOGGER.info(f"Using min_sqdist={self.min_sqdist}")

        if len(undeduped_embeddings) <= 1:
            LOGGER.info(f"Only {len(undeduped_embeddings)} points, no deduplication needed")
            return np.arange(len(undeduped_embeddings), dtype=np.uint64)

        # Build index with all points
        index = index_info.index_type.build(undeduped_embeddings, index_info)
        
        # Track which points to keep (starts with all points)
        points_to_keep = np.ones(len(undeduped_embeddings), dtype=bool)
        
        # Process results point-by-point to avoid memory bottleneck
        def process_neighbors(query_idx: int, neighbor_dists: NDArray, neighbor_ids: NDArray):
            # If current point is already marked for removal, skip it
            if not points_to_keep[query_idx]:
                return
                
            # Check if any retained neighbors are older (have smaller indices)
            if len(neighbor_ids) > 0 and np.any(points_to_keep[neighbor_ids] & (neighbor_ids < query_idx)):
                # If any retained older neighbor is within the epsilon ball, mark this point for removal
                points_to_keep[query_idx] = False

        LOGGER.info(f"Finding all pairs within min_sqdist={self.min_sqdist}")
        range_search_with_(index, undeduped_embeddings, self.min_sqdist, process_neighbors)
        
        retained_indices = np.where(points_to_keep)[0]
        LOGGER.info(f"Retaining {len(retained_indices)} points out of {len(undeduped_embeddings)}")
        
        return retained_indices.astype(np.uint64)
