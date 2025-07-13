
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
from functools import partial

import faiss
from numpy.typing import NDArray

from nextbook.util import TIMER


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
    HNSW = partial(init_hnsw_)
    FLAT_L2 = partial(init_flatl2_)
    
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
    
    @TIMER.time(formatters=dict(vectors=lambda vs: f"<shape={vs.shape}>"))
    def build(self, vectors: NDArray, index_info: 'IndexInfo') -> faiss.Index:
        d = vectors.shape[1]
        index = self.value(d, index_info)
        # if isinstance(index_info, HNSW):
        #     normalized_vectors = index_info.index_metric.normalize_fn(vectors)
        #     index.add(normalized_vectors)
        # elif isinstance(index_info, FlatL2):
        #     index.add(vectors)
        
        # index.add(index_info.normalize_vectors(vectors))
        index.add(vectors)

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
