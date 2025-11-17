# class IndexMetric(enum.Enum):
#     L2 = ("L2", faiss.METRIC_L2)
#     INNER_PRODUCT = ("INNER_PRODUCT", faiss.METRIC_INNER_PRODUCT)
#     COSINE = ("COSINE", faiss.METRIC_INNER_PRODUCT, lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))
#     def __init__(self, name: str, metric, normalize_fn=lambda x: x):
#         self.metric_name = name
#         self.metric = metric
#         self.normalize_fn = normalize_fn
#     def __str__(self):
#         return self.metric_name
#     def __repr__(self):
#         return str(self)
    
import abc
from dataclasses import dataclass, field
import enum
from typing import Callable, Generic, Optional, TypeVar

import faiss
import numpy as np
from numpy.typing import NDArray
import tqdm

def init_flatl2_(d: int, _index_info: 'FlatL2') -> faiss.IndexFlatL2:
    assert isinstance(_index_info, FlatL2), f"Expected FlatL2 IndexInfo, got {_index_info}"
    return faiss.IndexFlatL2(d)
    
def init_hnsw_(d: int, index_info: 'HNSW') -> faiss.IndexHNSWFlat:
    assert isinstance(index_info, HNSW), f"Expected HNSW IndexInfo, got {index_info}"
    index = faiss.IndexHNSWFlat(
        d,
        index_info.hnsw_m,
        # index_info.index_metric.metric
        faiss.METRIC_L2
    )
    index.hnsw.efConstruction = index_info.hnsw_efc
    index.hnsw.efSearch = index_info.hnsw_efs
    return index

class IndexType(enum.Enum):
    HNSW = "HNSW"
    FLAT_L2 = "FLAT_L2"

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
    
    def build(self, vectors: NDArray, index_info: 'IndexInfo') -> faiss.Index:
        d = vectors.shape[1]
        if self is IndexType.HNSW:
            assert isinstance(index_info, HNSW), f"Expected HNSW IndexInfo, got {type(index_info)!r}"
            index = init_hnsw_(d, index_info)
        elif self is IndexType.FLAT_L2:
            assert isinstance(index_info, FlatL2), f"Expected FlatL2 IndexInfo, got {type(index_info)!r}"
            index = init_flatl2_(d, index_info)
        else:  # pragma: no cover - defensive guard for future enum members
            raise ValueError(f"Unsupported IndexType: {self}")
        index.add(np.asarray(vectors))  # type: ignore[arg-type]

        return index


@dataclass(frozen=True)
class IndexInfo(abc.ABC):
    index_type: IndexType = field(init=False)

    # @abc.abstractmethod
    # def normalize_vectors(self, vectors: NDArray) -> NDArray:
    #     ...

@dataclass(frozen=True)
class HNSW(IndexInfo):
    # index_type: IndexType = IndexType.HNSW
    hnsw_m: int
    hnsw_efc: int
    hnsw_efs: int
    # index_metric: IndexMetric
    def __post_init__(self):
        assert self.hnsw_m > 0, f"hnsw_m must be positive: {self.hnsw_m}"
        assert self.hnsw_efc > 0, f"hnsw_efc must be positive: {self.hnsw_efc}"
        assert self.hnsw_efs > 0, f"hnsw_efs must be positive: {self.hnsw_efs}"
        object.__setattr__(self, "index_type", IndexType.HNSW)

    # #@override
    # def normalize_vectors(self, vectors: NDArray) -> NDArray:
    #     return self.index_metric.normalize_fn(vectors)

@dataclass(frozen=True)
class FlatL2(IndexInfo):
    # index_type: IndexType = IndexType.FLAT_L2
    def __post_init__(self):
        object.__setattr__(self, "index_type", IndexType.FLAT_L2)

    # #@override
    # def normalize_vectors(self, vectors: NDArray) -> NDArray:
    #     return vectors  # No normalization needed for FlatL2

def build_index(index_info: IndexInfo, embeddings: NDArray) -> faiss.Index:
    return index_info.index_type.build(embeddings, index_info)


@dataclass(frozen=True)
class RangeSearchResult:
    lims: Optional[NDArray[np.uint64]]
    dists: Optional[NDArray[np.float32]]
    ids: Optional[NDArray[np.int64]]
    def __post_init__(self):
        if self.lims is not None and len(self.lims) > 0:
            assert self.lims[0] == 0, f"lims[0] != 0: {self.lims[0]}"
        if self.dists is not None and self.ids is not None:
            assert len(self.dists) == len(self.ids), f"len(dists) != len(ids): {len(self.dists)} != {len(self.ids)}"
        # assert len(self.lims) == len(self.dists) + 1, f"len(lims) != len(dists) + 1: {len(self.lims)} != {len(self.dists)} + 1"

def range_search_radii_(
    index: faiss.Index,
    query: NDArray,
    radii: NDArray,
    return_lims: bool = True,
    return_dists: bool = True,
    return_ids: bool = True
) -> RangeSearchResult:
    # Start with an empty array in each list so that concatenation works with no results
    lims = [np.array([], dtype=np.uint64)] if return_lims else None
    dists = [np.array([], dtype=np.float32)] if return_dists else None
    ids = [np.array([], dtype=np.int64)] if return_ids else None
    for i in tqdm.tqdm(range(len(query)), desc="Range search radii", mininterval=3, delay=3):
        batch = query[i:i+1]
        lims_, dists_, ids_ = index.range_search(batch, radii[i]) # type: ignore
        assert lims_.dtype == np.uint64
        assert dists_.dtype == np.float32
        assert ids_.dtype == np.int64
        if lims is not None:
            if i > 0:
                lims_ = lims_[1:]
                lims_ += lims[-1][-1]
            lims.append(lims_)
        if dists is not None:
            dists.append(dists_)
        if ids is not None:
            ids.append(ids_)
    return RangeSearchResult(
        lims=np.concatenate(lims) if lims is not None else None,
        dists=np.concatenate(dists) if dists is not None else None,
        ids=np.concatenate(ids) if ids is not None else None)

def range_search_radii_with(
    index: faiss.Index,
    query: NDArray,
    radii: NDArray,
    kernel_fn: Callable[[int, NDArray, NDArray], None],
    show_progress: bool | tqdm.tqdm = True
) -> None:
    """
    Range search that processes results point-by-point to avoid memory bottlenecks.
    Args:
        index: The FAISS index to search
        query: Query vectors, shape (N, D)
        radii: Search radii, shape (N,)
        kernel_fn: Function called as kernel_fn(query_idx, neighbor_dists, neighbor_ids)
                  for each query point
        show_progress: If True, show tqdm progress bar. If False, no progress bar. If tqdm instance, use it.
    """
    pbar = None
    cleanup_pbar = False
    if show_progress is True:
        pbar = tqdm.tqdm(desc="Range search radii with kernel", total=len(query), mininterval=3, delay=3)
        cleanup_pbar = True
    elif show_progress is not False:
        pbar = show_progress

    for i in range(len(query)):
        batch = query[i:i+1]
        lims_, dists_, ids_ = index.range_search(batch, radii[i]) # type: ignore
        assert lims_.dtype == np.uint64
        assert dists_.dtype == np.float32
        assert ids_.dtype == np.int64
        # batch only has one element, so lims_ should select the entire range of neighbors
        assert lims_.shape == (2,)
        assert lims_[0] == 0, f"{lims_}"
        assert lims_[1] == len(dists_), f"{lims_}, {len(dists_)}"
        kernel_fn(i, dists_, ids_)
        if pbar is not None:
            pbar.update(1)
    
    if cleanup_pbar:
        assert isinstance(pbar, tqdm.tqdm)
        pbar.close()

def range_search_(
    index: faiss.Index, 
    query: NDArray, 
    epsilon: float,
    return_lims: bool = True,
    return_dists: bool = True,
    return_ids: bool = True
) -> RangeSearchResult:
    # Need to use small batches to avoid crashing
    lims: Optional[list[NDArray[np.uint64]]] = [] if return_lims else None
    dists: Optional[list[NDArray[np.float32]]] = [] if return_dists else None
    ids: Optional[list[NDArray[np.int64]]] = [] if return_ids else None

    batch_size = 48
    with tqdm.tqdm(desc="Range search", total=len(query)) as pbar:
        for i in range(0, len(query), batch_size):
            batch = query[i:i + batch_size]
            lims_, dists_, ids_ = index.range_search(batch, epsilon) # type: ignore
            if lims is not None:
                if i > 0:
                    lims_ = lims_[1:]
                    lims_ += lims[-1][-1]
                lims.append(lims_)
            if dists is not None:
                dists.append(dists_)
            if ids is not None:
                ids.append(ids_)
            pbar.update(len(batch))
    
    return RangeSearchResult(
        lims=np.concatenate(lims) if lims is not None else None,
        dists=np.concatenate(dists) if dists is not None else None,
        ids=np.concatenate(ids) if ids is not None else None)


def range_search_with_(
    index: faiss.Index, 
    query: NDArray, 
    epsilon: float,
    kernel_fn: Callable[[int, NDArray, NDArray], None],
    batch_size: int = 24
) -> None:
    """
    Range search that processes results point-by-point to avoid memory bottlenecks.
    
    Args:
        index: The FAISS index to search
        query: Query vectors, shape (N, D)
        epsilon: Search radius
        kernel_fn: Function called as kernel_fn(query_idx, neighbor_dists, neighbor_ids)
                  for each query point
        batch_size: Number of queries to process at once
    """
    with tqdm.tqdm(desc="Range search with kernel", total=len(query), mininterval=3, delay=3) as pbar:
        for i in range(0, len(query), batch_size):
            batch = query[i:i + batch_size]
            lims_, dists_, ids_ = index.range_search(batch, epsilon) # type: ignore
            
            # Process each query in the batch
            for j in range(len(batch)):
                query_idx = i + j
                start_idx = lims_[j]
                end_idx = lims_[j + 1]
                
                neighbor_dists = dists_[start_idx:end_idx] if len(dists_) > 0 else np.array([])
                neighbor_ids = ids_[start_idx:end_idx] if len(ids_) > 0 else np.array([])
                
                kernel_fn(query_idx, neighbor_dists, neighbor_ids)
            
            pbar.update(len(batch))

def kballs_with(ambient: faiss.Index, landmark: faiss.Index, query: NDArray, 
                radius_fraction: float, k: int,
                kernel_fn: Callable[[int, NDArray, NDArray], None],
                show_progress: bool | tqdm.tqdm = True) -> None:
    D, _I = landmark.search(query, k=k) # type: ignore
    range_search_radii_with(ambient, query, radius_fraction * D[:, -1], kernel_fn,
                            show_progress=tqdm.tqdm(range(len(query)), desc="k-balls with kernel", total=len(query), mininterval=3, delay=3) if show_progress is True else show_progress)

T = TypeVar('T', bound=np.generic)
@dataclass(frozen=True)
class FlatWithLims(Generic[T]):
    concatenated: NDArray[T]
    lims: NDArray[np.integer]
    def __post_init__(self):
        assert np.issubdtype(self.lims.dtype, np.integer), f"lims must be integer type, got {self.lims.dtype}"
        if self.concatenated.size > 0:
            assert self.lims[0] == 0, f"lims[0] must be 0, got {self.lims[0]}"
            assert self.lims[-1] == len(self.concatenated), f"lims[-1] must be length of concatenated, got {self.lims[-1]} != {len(self.concatenated)}"
        assert np.all(self.lims >= 0)
        assert np.all(self.lims[1:] >= self.lims[:-1]), "lims must be non-decreasing"

def to_flat_with_lims(arrs: list[NDArray[T]]) -> FlatWithLims[T]:
    lims = np.array([0] + [len(arr) for arr in arrs], dtype=np.int64).cumsum()
    return FlatWithLims(concatenated=np.concatenate(arrs), lims=lims)

def from_flat_with_lims(flat_with_lims: FlatWithLims[T]) -> list[NDArray[T]]:
    lims = flat_with_lims.lims
    arr = flat_with_lims.concatenated
    return [arr[i:j] for i, j in zip(lims[:-1], lims[1:])]

def actual_voronoi_coverage(
    ambient: NDArray[np.float32],
    landmark: faiss.Index,
    ambient_is_landmark: bool):
    """
    Actual voronoi coverage: Return the number of ambient points covered by each landmark's Voronoi cell.
    """
    _dists, ambient_to_closest_landmark = landmark.search(ambient, k=2 if ambient_is_landmark else 1)  # type: ignore
    ambient_to_closest_landmark = ambient_to_closest_landmark[:, -1]  # Get the closest landmark for each ambient point
    return np.bincount(ambient_to_closest_landmark, minlength=landmark.ntotal)

def hypothetical_voronoi_coverage(
    ambient: faiss.Index,
    landmark: faiss.Index,
    query: NDArray,
    ambient_is_landmark: bool,
    show_progress: bool | tqdm.tqdm = True
) -> NDArray[np.uint64]:
    """
    Hypothetical voronoi coverage: Return the number of ambient points covered by each query point's hypothetical Voronoi cell among the landmarks if it became a landmark.
    Args:
        ambient: FAISS index for ambient points
        landmark: FAISS index for landmark points
        query: Embeddings for query points (shape [N, D])
        ambient_is_landmark: If the ambient points are also landmarks, we need to use k=2 to avoid counting the ambient point as its own nearest landmark.
    Returns:
        counts: np.ndarray of shape [N], number of ambient points covered by each query
    """
    counts = np.zeros(len(query), dtype=np.uint64)
    def go(_ambient_idx: int, _query_dists: NDArray[np.float32], query_ids: NDArray[np.int64]) -> None:
        counts[query_ids] += 1
    kballs_with(
        ambient=ambient,
        landmark=landmark,
        query=query,
        radius_fraction=1.0,
        k=2 if ambient_is_landmark else 1,
        kernel_fn=go,
        show_progress=tqdm.tqdm(desc="Voronoi coverage", total=len(query), mininterval=3, delay=3) if show_progress is True else show_progress
    )
    return counts