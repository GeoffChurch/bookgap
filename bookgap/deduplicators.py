"""
Deduplication strategies for removing duplicate works based on embeddings.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray
import tqdm

from bookgap.index import IndexInfo, range_search_, range_search_with_
from bookgap.util import TIMER, LOGGER


def _validate_embeddings_and_log_input(embeddings: NDArray, method_name: str) -> bool:
    """
    Shared validation and logging for deduplication input. 
    Returns True if deduplication should proceed, False for early return.
    """
    LOGGER.info(f"{method_name}: Processing {len(embeddings)} embeddings")
    
    if len(embeddings) <= 1:
        LOGGER.info(f"{method_name}: Only {len(embeddings)} points, no deduplication needed")
        return False
    return True


def _log_deduplication_results(retained_count: int, total_count: int, method_name: str) -> None:
    """Shared logging for deduplication results."""
    removed_count = total_count - retained_count
    removal_rate = (removed_count / total_count * 100) if total_count > 0 else 0
    LOGGER.info(f"{method_name}: Retained {retained_count}/{total_count} points (removed {removed_count}, {removal_rate:.1f}%)")


def _create_retained_indices(embeddings_count: int) -> NDArray[np.uint64]:
    """Create indices array for no-op deduplication."""
    return np.arange(embeddings_count, dtype=np.uint64)


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
        if not (0 <= self.duplicate_fraction <= 1):
            raise ValueError(f"duplicate_fraction must be in [0, 1], got {self.duplicate_fraction}")
    
    @TIMER.time(formatters=dict(undeduped_embeddings=lambda embeddings: f"<shape={embeddings.shape}>"))
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo):
        if not _validate_embeddings_and_log_input(undeduped_embeddings, "FractionalDeduplicator"):
            return _create_retained_indices(len(undeduped_embeddings))
            
        num_to_drop = int(self.duplicate_fraction * (len(undeduped_embeddings) - 1))
        LOGGER.info(f"FractionalDeduplicator: Using duplicate_fraction={self.duplicate_fraction}, will drop {num_to_drop} points")

        if num_to_drop == 0:
            retained_indices = _create_retained_indices(len(undeduped_embeddings))
        else:
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

            if len(dist_to_nearest_older) != len(undeduped_embeddings) - 1:
                raise ValueError(f"Expected {len(undeduped_embeddings) - 1} distances, got {len(dist_to_nearest_older)}")

            argsort = np.argsort(dist_to_nearest_older)
            retained_indices = np.concatenate(([0], argsort[num_to_drop:] + 1), dtype=np.uint64)  # +1 because we dropped the first element

        _log_deduplication_results(len(retained_indices), len(undeduped_embeddings), "FractionalDeduplicator")
        return retained_indices


@dataclass(frozen=True)
class SqDistDeduplicator(Deduplicator):
    """Deduplicator that removes points with older neighbors within a squared distance threshold."""
    min_sqdist: float
    
    def __post_init__(self):
        if self.min_sqdist < 0:
            raise ValueError(f"min_sqdist must be non-negative, got {self.min_sqdist}")
    
    def description(self) -> str:
        return f"squared distance deduplication (min_sqdist={self.min_sqdist})"
    
    @TIMER.time(formatters=dict(undeduped_embeddings=lambda embeddings: f"<shape={embeddings.shape}>"))
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo):
        if not _validate_embeddings_and_log_input(undeduped_embeddings, "SqDistDeduplicator"):
            return _create_retained_indices(len(undeduped_embeddings))

        LOGGER.info(f"SqDistDeduplicator: Using min_sqdist={self.min_sqdist}")

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
        _log_deduplication_results(len(retained_indices), len(undeduped_embeddings), "SqDistDeduplicator")

        return retained_indices.astype(np.uint64)


@dataclass(frozen=True)
class SimHashDeduplicator(Deduplicator):
    """LSH-style deduplicator: random hyperplane SimHash + exact-L2 verify.

    On the 42 M-work domain=3 corpus with GIST-small embeddings, the
    FAISS HNSW ``range_search``-based :class:`SqDistDeduplicator` spent
    many hours in pair-finding due to suboptimal cache behaviour at
    scale (see ``artifacts_scripts/dedup_methods_bench.py``). SimHash
    with k=32 bits × L=20 tables matched IVFFlat-exact at 99.98-99.9998 %
    recall across sparse and ultra-dense duplicate regimes while running
    in a small fraction of the wall time. The geometry reason: signature
    computation is a pure matrix multiply (cache-friendly, BLAS-optimised),
    bucket probes are O(1) hashes, and exact verification only fires on
    co-bucketed pairs — a tiny fraction of all (q, q') pairs.

    Semantics match :class:`SqDistDeduplicator`: a point is dropped iff
    some older-index (smaller-``query_idx``) retained point lies within
    ``sqrt(min_sqdist)`` L2 of it. ``index_info`` is accepted for
    interface parity and ignored.
    """
    min_sqdist: float
    k_bits: int = 32
    n_tables: int = 20
    seed: int = 0

    def __post_init__(self):
        if self.min_sqdist < 0:
            raise ValueError(f"min_sqdist must be non-negative, got {self.min_sqdist}")
        if not 1 <= self.k_bits <= 64:
            raise ValueError(f"k_bits must be 1..64 (uint64 signature), got {self.k_bits}")
        if self.n_tables < 1:
            raise ValueError(f"n_tables must be >= 1, got {self.n_tables}")

    def description(self) -> str:
        return (f"SimHash deduplication (min_sqdist={self.min_sqdist}, "
                f"k_bits={self.k_bits}, n_tables={self.n_tables})")

    @TIMER.time(formatters=dict(undeduped_embeddings=lambda embeddings: f"<shape={embeddings.shape}>"))
    def deduplicate(self, undeduped_embeddings: NDArray, index_info: IndexInfo):
        if not _validate_embeddings_and_log_input(undeduped_embeddings, "SimHashDeduplicator"):
            return _create_retained_indices(len(undeduped_embeddings))

        LOGGER.info(
            f"SimHashDeduplicator: min_sqdist={self.min_sqdist}, "
            f"k_bits={self.k_bits}, n_tables={self.n_tables}"
        )

        n, d = undeduped_embeddings.shape
        points_to_keep = np.ones(n, dtype=bool)
        rng = np.random.default_rng(self.seed)

        # Signature packing requires k_bits <= 64. We use uint64.
        # Per table: k_bits random Gaussian hyperplanes, normalized for
        # numerical stability. Sign of dot product → bit.
        pair_count = 0
        for t in range(self.n_tables):
            LOGGER.info(f"SimHashDeduplicator: table {t + 1}/{self.n_tables}")
            hyperplanes = rng.standard_normal((self.k_bits, d)).astype(np.float32)
            hyperplanes /= np.linalg.norm(hyperplanes, axis=1, keepdims=True)

            # (n, k_bits) projections → sign → bit → pack into uint64.
            projections = undeduped_embeddings @ hyperplanes.T
            bits = (projections > 0).astype(np.uint64)
            packed = np.zeros(n, dtype=np.uint64)
            for b in range(self.k_bits):
                packed |= bits[:, b] << np.uint64(b)

            # Bucket indices by signature in one vectorised pass: sort points
            # by their packed signature, then use unique()+index arithmetic to
            # yield contiguous runs per bucket. Beats the ~42M-iteration
            # Python setdefault loop.
            order = np.argsort(packed, kind="stable")
            sorted_sigs = packed[order]
            # boundaries[i]..boundaries[i+1] is a contiguous bucket in `order`
            change = np.nonzero(np.concatenate(([True], sorted_sigs[1:] != sorted_sigs[:-1], [True])))[0]
            bucket_starts = change[:-1]
            bucket_ends = change[1:]

            for start, end in zip(bucket_starts, bucket_ends):
                bsz = end - start
                if bsz < 2:
                    continue
                indices = order[start:end].astype(np.int64, copy=False)
                # Work on members that are still alive; if only one remains
                # there is nothing to compare against, skip.
                alive_mask = points_to_keep[indices]
                if alive_mask.sum() < 2:
                    continue
                # Pairwise L2² via a single matmul + norm trick:
                #   ||a-b||² = ||a||² + ||b||² - 2 a·b
                # For normalised embeddings (cloud's default) ||a||² = ||b||² = 1
                # so this simplifies to 2 - 2·(a·b); we compute it explicitly
                # anyway in case the caller passes non-normalised embeddings.
                chunk = undeduped_embeddings[indices]
                # To keep the B×B distance matrix bounded (a 10k bucket is
                # 10k × 10k × 4 B = 400 MB, a 100k one is 40 GB), chunk rows.
                MAX_ROWS_PER_STEP = 4096
                for row_start in range(0, bsz, MAX_ROWS_PER_STEP):
                    row_end = min(row_start + MAX_ROWS_PER_STEP, bsz)
                    row_block = chunk[row_start:row_end]           # (r, d)
                    # Only compare against strictly older (smaller global
                    # index) positions. Since `indices` is a sort by
                    # signature, not by global index, we compute dists to
                    # the whole bucket and mask by global-index ordering.
                    dots = row_block @ chunk.T                     # (r, bsz)
                    sq_norm_row = np.einsum("ij,ij->i", row_block, row_block)[:, None]
                    sq_norm_col = np.einsum("ij,ij->i", chunk, chunk)[None, :]
                    sq_dists = sq_norm_row + sq_norm_col - 2.0 * dots
                    # For each local row j (global = indices[row_start + j]),
                    # find any alive OLDER global index whose sq_dist < min.
                    in_ball = sq_dists < self.min_sqdist
                    # Older = strictly smaller global index
                    older_mask_full = indices[None, :] < indices[row_start:row_end][:, None]
                    alive_row_mask = points_to_keep[indices][None, :]
                    hit = in_ball & older_mask_full & alive_row_mask
                    # A row is dropped if any column is a hit and the row
                    # itself is currently alive.
                    any_hit = hit.any(axis=1)
                    rows_to_drop_local = np.nonzero(any_hit & points_to_keep[indices[row_start:row_end]])[0]
                    if rows_to_drop_local.size:
                        drop_global = indices[row_start + rows_to_drop_local]
                        points_to_keep[drop_global] = False
                        pair_count += int(rows_to_drop_local.size)

        LOGGER.info(f"SimHashDeduplicator: {pair_count} duplicate hits across {self.n_tables} tables")
        retained_indices = np.where(points_to_keep)[0]
        _log_deduplication_results(len(retained_indices), len(undeduped_embeddings), "SimHashDeduplicator")
        return retained_indices.astype(np.uint64)
