import abc
import argparse
from collections import defaultdict, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
import enum
from functools import lru_cache, partial
import logging
import matplotlib.pyplot as plt
import faiss
import itertools
import joblib
import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pyalex
import re
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde#, kstest
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
import time
import torch
import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Any, Collection, Dict, Iterable, List, Optional, Literal, Union#, override (Python 3.12+)
from util import CACHE_DIR, ks_2samp, logger, TIMER, MEMORY
import pandas as pd
from nextbook.denoisers import Denoiser

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (10, 6)
DPI = 150

pyalex.config.email = "gchurchill@cs.stonybrook.edu"

@dataclass(frozen=True)
class FlatWithLims:
    concatenated: NDArray
    lims: NDArray
def to_flat_with_lims(arrs: list[NDArray]) -> FlatWithLims:
    lims = np.array([0] + [len(arr) for arr in arrs], dtype=np.uint32).cumsum()
    return FlatWithLims(concatenated=np.concatenate(arrs), lims=lims)
def from_flat_with_lims(flat_with_lims: FlatWithLims) -> list[NDArray]:
    lims = flat_with_lims.lims
    arr = flat_with_lims.concatenated
    return [arr[i:j] for i, j in zip(lims[:-1], lims[1:])]

# class DimensionalityReduction(enum.Enum):
#     PCA = reduce_dimensions_pca
#     TSNE = reduce_dimensions_tsne
#     def __init__(self, func):
#         self.f = func
#     def __call__(self, embeddings: NDArray, **kwargs) -> NDArray:
#         return self.f(embeddings, **kwargs)

class Topic(enum.Enum):
    ROBOTICS = ("Robotics and Sensor-Based Localization",      "T10191")
    # MYTHS    = ("Folklore, Mythology, and Literature Studies", "T12586")
    # MEDIEVAL = ("Medieval Literature and History",             "T10595")
    # ALG_GEOM = ("Algebraic Geometry and Number Theory",        "T10061")
    # NLP      = ("Natural Language Processing Techniques",      "T10181")

    def __init__(self, display_name: str, topic_id: str):
        self._display_name = display_name
        self._topic_id = topic_id
        # Validate topic_id
        # topic = pyalex.Topics()[self._topic_id]
        # assert topic['display_name'] == self._display_name, f"Display name mismatch: {topic.display_name} != {self._display_name}"

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def topic_id(self) -> str:
        return self._topic_id
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)

class WorkType(enum.Enum):
    BOOK    = "book"
    ARTICLE = "article"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)

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


@dataclass(frozen=True)
class Cloud:
    name: str = field(init=False) # TODO maybe delete
    topic: Topic
    work_type: WorkType
    index_info: IndexInfo
    max_works: int
    duplicate_expectation: float
    normalize_embeddings: bool
    denoiser: Denoiser
    drop_k: int

    def __post_init__(self):
        object.__setattr__(self, "name", f"{self.topic} {self.work_type}")
        assert self.drop_k >= 0
        assert isinstance(self.denoiser, Denoiser)
        assert 0 <= self.duplicate_expectation <= 1, f"duplicate_expectation must be in [0, 1], got {self.duplicate_expectation}"
        if self.denoiser is Denoiser.NONE:
            assert self.drop_k == 0, f"If denoiser is NONE, drop_k must be 0 (got {self.drop_k})"
        # else:
        #     assert self.drop_k > 0, f"If denoiser is not NONE, drop_k must be > 0 (got {self.drop_k})"

    def works(self) -> List[Dict[str, Any]]:
        return self._data().works
    
    def index(self) -> faiss.Index:
        return self._data().index
    
    def embeddings(self) -> NDArray:
        return self._data().embeddings
    
    def _data(self) -> "Cloud.CloudData":
        return get_cloud_data(self)

    @dataclass(frozen=True)
    class CloudData:
        works: List[Dict[str, Any]]
        index: faiss.Index
        embeddings: NDArray
        def __post_init__(self):
            assert len(self.works) == self.embeddings.shape[0], f"Works and embeddings length mismatch: {len(self.works)} != {self.embeddings.shape[0]}"
            assert self.index.ntotal == len(self.works), f"Index and works length mismatch: {self.index.ntotal} != {len(self.works)}"
            assert self.index.d == self.embeddings.shape[1], f"Index and embeddings dimension mismatch: {self.index.d} != {self.embeddings.shape[1]}"

def build_index(index_info: IndexInfo, embeddings: NDArray) -> faiss.Index:
    return index_info.index_type.build(embeddings, index_info)

@TIMER.time(formatters=dict(index_info=lambda _: "<...>", clouds=lambda clouds: f"<{len(clouds)} clouds, num works={[len(cloud.works()) for cloud in clouds]}>"))
@MEMORY.cache
def build_index_from_clouds(index_info: IndexInfo, clouds: Sequence[Cloud]) -> faiss.Index:
    """
    Build a single index from multiple clouds.
    This is useful for combining multiple topics or work types into a single index.
    """
    if len(clouds) == 1 and clouds[0].index_info == index_info:
        # If there's only one cloud and it matches the index_info, return its index directly
        logger.info(f"Only one cloud provided, returning its index: {clouds[0].name}")
        return clouds[0].index()
    all_embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    logger.info(f"Building index from {len(clouds)} clouds with {index_info.index_type} index. Total embeddings shape: {all_embeddings.shape}")
    index = build_index(index_info, all_embeddings)
    return index

def build_citation_graph(works):
    id_to_idx = {work["id"]: i for i, work in enumerate(works)}
    n = len(works)
    citation_graph = np.eye(n, dtype=np.float32)  # add self-loops for well-defined normalization
    for i, work in enumerate(works):
        for ref_id in work["referenced_works"]:
            if ref_id in id_to_idx:
                citation_graph[i, id_to_idx[ref_id]] = 1.0
    citation_graph = citation_graph / citation_graph.sum(axis=1, keepdims=True)
    return citation_graph

def plot_eigenvalues(eigvals, n_components, out_path, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    # ax.scatter(range(eigvals.shape[0] - n_components), eigvals[:-n_components], s=1, color='red', alpha=0.5, label='Discarded components')
    # ax.scatter(range(eigvals.shape[0] - n_components, eigvals.shape[0]), eigvals[-n_components:], s=1, color='blue', alpha=0.5, label='Retained components')
    ax.scatter(range(len(eigvals)), eigvals, s=1, label='Eigenvalues')
    ax.set_yscale('log')
    ax.set_title(title)
    fig.legend()
    fig.savefig(out_path)
    plt.close(fig)

@lru_cache(maxsize=16)
@MEMORY.cache
def get_cloud_data(cloud: "Cloud") -> "Cloud.CloudData":
    def maybe_normalize(embeddings):
        if cloud.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.where(norms == 0, 1, norms)
        return embeddings
    logger.info(f"Getting cloud data for {cloud.name}")
    cloud_dir = ARTIFACTS_DIR / str(cloud.topic) / str(cloud.work_type)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    expected = get_query(cloud.work_type, cloud.topic).count()
    max_works = cloud.max_works
    expected = min(max_works, expected) if max_works is not None else expected
    works = sorted(tqdm.tqdm(itertools.islice(
        embed_works(cloud.work_type, cloud.topic),
        max_works),
        total=expected, disable=False), key=lambda w: w["publication_date"])
    logger.info(f"Cloud {cloud.name} -- Loaded {len(works)} works")
    d, = works[0]["embedding"].shape

    norms = [np.linalg.norm(work["embedding"]) for work in works]
    logger.info(f"Cloud {cloud.name} -- embedding norm stats:\n"
                f"\tmean={np.mean(norms):.4f}\n"
                f"\tmin={np.min(norms):.4f}\n"
                f"\tmed={np.median(norms):.4f}\n"
                f"\tmax={np.max(norms):.4f}\n"
                f"\tstd={np.std(norms):.4f}\n")

    undeduped_embeddings = np.empty((len(works), d), dtype=np.float32)
    for i, work in enumerate(works):
        undeduped_embeddings[i] = work["embedding"]
        del work["embedding"]

    # Dimensionality reduction/cleaning
    n_components = undeduped_embeddings.shape[1] - cloud.drop_k
    logger.info(f"Cloud {cloud.name} -- Running {cloud.denoiser.name} with drop_k={cloud.drop_k} (n_components={n_components}) on {len(undeduped_embeddings)} embeddings of shape {undeduped_embeddings.shape}")
    X_cleaned, eigvals = cloud.denoiser(undeduped_embeddings,
                                        np.empty(0) if cloud.denoiser is Denoiser.PCA else build_citation_graph(works),
                                        n_components)
    undeduped_embeddings = X_cleaned
    if len(eigvals) > 0:
        plot_eigenvalues(
            eigvals,
            n_components,
            cloud_dir / f"{cloud.denoiser.name.lower()}_eigenvalues.png",
            f"Eigenvalues for {cloud.denoiser.name} on {cloud.name} embeddings"
        )
    undeduped_embeddings = maybe_normalize(undeduped_embeddings)

    # Maybe normalize embeddings
    undeduped_embeddings = maybe_normalize(undeduped_embeddings)

    # Save for plotting
    dists_to_centroid = np.linalg.norm(undeduped_embeddings - undeduped_embeddings.mean(axis=0), axis=1)

    # --- Adaptive min_dist based on duplicate_expectation ---
    duplicate_expectation = cloud.duplicate_expectation
    num_to_drop = int(duplicate_expectation * (len(undeduped_embeddings) - 1))
    logger.info(f"Cloud {cloud.name} -- Using duplicate_expectation={duplicate_expectation}")

    if num_to_drop > 0 and len(undeduped_embeddings) > 1:
        # Compute distances to nearest older neighbor
        dist_to_nearest_older = []
        # Build index incrementally, but in batches for efficiency
        main_index = cloud.index_info.index_type.build(undeduped_embeddings[:1], cloud.index_info) # Start with oldest work
        batch_size = 512
        
        desc = f"Finding nearest older neighbors for {cloud.name}"
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
        retained_indices = np.concatenate(([0], argsort[num_to_drop:] + 1))  # +1 because we dropped the first element
    else:
        retained_indices = np.arange(len(undeduped_embeddings))

    logger.info(f"Cloud {cloud.name} -- Retaining {len(retained_indices)} points out of {len(undeduped_embeddings)} based on duplicate expectation")
    retained_works = [works[i] for i in retained_indices]
    deduped_embeddings = undeduped_embeddings[retained_indices]

    dists_to_centroid_after_removal = np.linalg.norm(deduped_embeddings - deduped_embeddings.mean(axis=0), axis=1)
    # Plot sorted distances to centroid before and after removing points
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.scatter(range(len(dists_to_centroid)), np.sort(dists_to_centroid), s=1, color='red', alpha=0.5, label='Before removal')
    ax.scatter(range(len(dists_to_centroid_after_removal)), np.sort(dists_to_centroid_after_removal), s=1, color='blue', alpha=0.5, label='After removal')
    ax.set_title(f"Distances to original centroid after removing {num_to_drop} ({cloud.duplicate_expectation * 100:.2f}%) suspected duplicates")
    fig.legend()
    fig.savefig(cloud_dir / "sorted_dists_to_centroid_deduping.png")
    plt.close(fig)

    # Rebuild the index with the retained points
    logger.info(f"Cloud {cloud.name} -- Rebuilding index with {len(works)} points")
    start_time = time.time()
    index = build_index(cloud.index_info, deduped_embeddings)
    logger.info(f"Cloud {cloud.name} -- Rebuilding index took {time.time() - start_time:.2f} seconds")
    return Cloud.CloudData(works=retained_works, index=index, embeddings=deduped_embeddings)

@dataclass(frozen=True)
class RangeSearchResult:
    lims: NDArray
    dists: NDArray
    ids: NDArray
    def __post_init__(self):
        assert self.lims[0] == 0, f"lims[0] != 0: {self.lims[0]}"
        assert len(self.dists) == len(self.ids), f"len(dists) != len(ids): {len(self.dists)} != {len(self.ids)}"
        # assert len(self.lims) == len(self.dists) + 1, f"len(lims) != len(dists) + 1: {len(self.lims)} != {len(self.dists)} + 1"

@TIMER.time(formatters=dict(index=lambda index: f"<{index.ntotal} vectors, d={index.d}>", query=lambda q: f"<shape={q.shape}>", radii=lambda r: f"<shape={r.shape}>"))
def range_search_radii_(index: faiss.Index, query: NDArray, radii: NDArray) -> RangeSearchResult:
    lims = []
    dists = []
    ids = []
    for i in tqdm.tqdm(range(len(query)), desc="Range search radii"):
        batch = query[i:i+1]
        lims_, dists_, ids_ = index.range_search(batch, radii[i]) # type: ignore
        if i > 0:
            lims_ = lims_[1:]
            lims_ += lims[-1][-1]
        lims.append(lims_)
        dists.append(dists_)
        ids.append(ids_)
    return RangeSearchResult(lims=np.concatenate(lims), dists=np.concatenate(dists), ids=np.concatenate(ids))

def range_search_(index: faiss.Index, query_embeddings: NDArray, epsilon: float) -> RangeSearchResult:
    # Need to use small batches to avoid crashing
    lims = []
    dists = []
    ids = []

    batch_size = 48
    with tqdm.tqdm(desc="Range search", total=len(query_embeddings)) as pbar:
        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i:i + batch_size]
            lims_, dists_, ids_ = index.range_search(batch, epsilon) # type: ignore
            if i > 0:
                lims_ = lims_[1:]
                lims_ += lims[-1][-1]
            lims.append(lims_)
            dists.append(dists_)
            ids.append(ids_)
            pbar.update(len(batch))
    return RangeSearchResult(lims=np.concatenate(lims), dists=np.concatenate(dists), ids=np.concatenate(ids))



@dataclass(frozen=True)
class FaissSearchOutsideEpsilonResult:
    D_outside: NDArray
    I_outside: NDArray
    lims_within: NDArray
    dists_within: NDArray
    ids_within: NDArray

@MEMORY.cache
def faiss_search_outside_epsilon(
        index: Cloud,
        query: Cloud,
        k: int,
        epsilon: float) -> FaissSearchOutsideEpsilonResult:
    """
    Find the k nearest neighbors of query that are not (strictly) within epsilon of the query.
    """
    query_embeddings = query.embeddings()
    if epsilon == 0:
        logger.info("Epsilon is 0, using k nearest neighbors search instead of range search")
        start_time = time.time()
        D, I = index.index().search(query_embeddings, k) # type: ignore
        logger.info(f"{k}-NN search took {time.time() - start_time:.2f} seconds")
        if k == 2:
            Dself, Dnn = D.T
            for i, idx in enumerate(I[:, 0]):
                dself, dnn = D[i]
                if idx != i and not np.allclose(dself, dnn):
                    logger.warning(f"Index mismatch ({dself=} {dnn=}): {idx} != {i}")
            assert np.all(Dself <= Dnn), f"Distance to self is greater than distance to nearest neighbor at {np.count_nonzero(Dself > Dnn)} points"
            argmax_dist_to_self = np.argmax(Dself)
            max_dist_to_self = Dself[argmax_dist_to_self]
            atol = 1e-5
            if not np.isclose(max_dist_to_self, 0, atol=atol):
                logger.warning(f"{np.sum(~np.isclose(Dself, 0, atol=atol))} points have distance to self not close to 0.")
        return FaissSearchOutsideEpsilonResult(
            D_outside=D,
            I_outside=I,
            lims_within=np.zeros((len(query_embeddings) + 1,), dtype=np.uint64),
            dists_within=np.empty((0,), dtype=np.float32),
            ids_within=np.empty((0,), dtype=np.int64))

    # Range search for all points strictly within epsilon
    logger.info(f"Range searching for all points strictly within epsilon")
    start_time = time.time()
    range_search_result = range_search_(index.index(), query.embeddings(), epsilon)
    lims_within, dists_within, ids_within = range_search_result.lims, range_search_result.dists, range_search_result.ids
    logger.info(f"Range searching took {time.time() - start_time:.2f} seconds")
    assert lims_within[0] == 0
    num_within = lims_within[1:] - lims_within[:-1]
    assert len(num_within) == len(query_embeddings)
    assert np.all(num_within >= 0), f"Negative number of points within epsilon: {num_within}"

    # Batch search all queries with the same number of points "j" within epsilon
    D_outside = np.empty((len(query_embeddings), k), dtype=np.float32)
    I_outside = np.empty((len(query_embeddings), k), dtype=np.int64)
    for j in np.unique(num_within).tolist():
        query_indices = num_within == j # Could pre-sort, but index.search is the bottleneck
        D, I = index.index().search(query_embeddings[query_indices], j + k) # type: ignore
        # Discard the first j points, which are within epsilon
        D_outside[query_indices] = D[:, j:]
        I_outside[query_indices] = I[:, j:]

    return FaissSearchOutsideEpsilonResult(
        D_outside=D_outside,
        I_outside=I_outside,
        lims_within=lims_within,
        dists_within=dists_within,
        ids_within=ids_within,
    )

def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _.-=\(\)]', ' ', str(s)).rstrip()

def get_query(work_type: WorkType, topic: Topic):
    # .sort(publication_date="asc", display_name="asc") \
    return pyalex.Works() \
            .filter(has_abstract=True,
                    # language="en",
                    type=str(work_type), **{"primary_topic.id": topic.topic_id}) \
            .select(", ".join(["display_name", "publication_date", "ids", "language", "abstract_inverted_index", "referenced_works"]))

def resolve_abstract_inverted_index(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not abstract_inverted_index:
        return ""
    pos_to_word = {}
    for word, positions in abstract_inverted_index.items():
        assert " " not in word, f"Word contains space: '{word}'"
        for pos in positions:
            assert pos not in pos_to_word, f"Duplicate position {pos} for word {word}"
            pos_to_word[pos] = word
    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "[[MISSING]]") for i in range(max_pos + 1)]
    return " ".join(words)




            
def get_works(work_type: WorkType, topic: Topic, cache_dir: Path):
    cache_dir = cache_dir / "api_data" / create_unicode_safe_filename(topic) / create_unicode_safe_filename(work_type)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        with state_file.open("w") as f:
            json.dump({"next_cursor": "*", "next_page": 1}, f)
    with state_file.open("r") as f:
        state = json.load(f)

    # First yield cached works
    for page in range(1, state["next_page"]):
        page_file = cache_dir / f"page_{page}.json"
        with page_file.open("r") as f:
            yield from json.load(f)
    
    # This isn't actually necessary, but it's clearer
    if state["next_cursor"] is None:
        return

    # Then yield new works
    last_time = time.time() # For rate limiting
    for i, page in enumerate(get_query(work_type, topic).paginate(
        cursor=state["next_cursor"], per_page=200, n_max=None) # type: ignore
        ):
        next_cursor = page.meta["next_cursor"]
        if next_cursor is None:
            with state_file.open("w") as f:
                json.dump({"next_cursor": None, "next_page": state["next_page"]}, f)
            return
        page = list(map(dict, page))
        for work in page:
            work["abstract"] = resolve_abstract_inverted_index(work["abstract_inverted_index"])
            del work["abstract_inverted_index"]
            work["id"] = work["ids"]["openalex"]
            del work["ids"]
            work["topic"] = str(topic)
        # Save page
        page_file = cache_dir / f"page_{state['next_page']}.json"
        with page_file.open("w") as f:
            json.dump(page, f, indent=2)
        # Save state
        state["next_page"] += 1
        state["next_cursor"] = next_cursor
        with state_file.open("w") as f:
            json.dump(state, f)
        # Yield works
        yield from page
        # Rate limit
        min_time_diff = 1.0 # Minimum time between requests
        time.sleep(max(0, min_time_diff - (time.time() - last_time)))
        last_time = time.time()

class RegexFilter:
    def __init__(self, regex: str, flags: int = re.IGNORECASE):
        self.regex = re.compile(regex, flags)
    def __call__(self, work: Dict) -> bool:
        return self.regex.search(f"{work['display_name']} {work['abstract']}") is not None
    def __repr__(self):
        return f"Contains {self.regex}"

class MissingFieldFilter:
    def __init__(self, field: str):
        self.field = field
    def __call__(self, work: Dict) -> bool:
        return work.get(self.field) is None
    def __repr__(self):
        return f"Missing {self.field}"

class ShortFieldFilter:
    def __init__(self, field: str, min_length: int):
        self.field = field
        self.min_length = min_length
    def __call__(self, work: Dict) -> bool:
        field = work.get(self.field)
        return field is not None and len(field) < self.min_length
    def __repr__(self):
        return f"len({self.field}) < {self.min_length}"

class WorkFilter(enum.Enum):
    BINDER_NAME_CANCEL_CREATE_EXPORT      = RegexFilter(r"BinderNameCancelCreateExport")
    ABSTRACT_NOT_AVAILABLE                = RegexFilter(r"An abstract is not available")
    SEARCH_FOR_OTHER_WORKS_BY_THIS_AUTHOR = RegexFilter(r"Search for other works by this author")
    RUSSIAN_TEXT_IGNORED                  = RegexFilter(r"Russian Text Ignored")
    MARITAL_ARMS                          = RegexFilter(r"Judging from repetitious appearances of her marital arms")
    NO_DISPLAY_NAME                       = MissingFieldFilter("display_name")
    NO_ABSTRACT                           = MissingFieldFilter("abstract")
    SHORT_DISPLAY_NAME                    = ShortFieldFilter("display_name", 4)
    SHORT_ABSTRACT                        = ShortFieldFilter("abstract", 10)
    NULL_REFERENCED_WORKS                 = MissingFieldFilter("referenced_works")
    ZERO_REFERENCED_WORKS                 = ShortFieldFilter("referenced_works", 1)
    
    def __call__(self, work: Dict) -> bool:
        return self.value(work)

    def __str__(self):
        return repr(self.value)

def filter_works(work_type: WorkType, topic: Topic, cache_dir: Path):
    stats: Dict[str, Union[int, float]]
    stats = {f"{filter}": 0 for filter in WorkFilter}
    stats["total"] = 0
    stats["num_removed"] = 0
    # invalid_dir = cache_dir / "filter" / str(topic) / str(work_type)
    # invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # with (invalid_dir / "invalid_works.txt").open("w") as f:
    for work in tqdm.tqdm(get_works(work_type, topic, cache_dir), total=get_query(work_type, topic).count(),
                          desc=f"Filtering {work_type} works in {topic}", disable=False):
        invalidators = []
        stats["total"] += 1
        for filter in WorkFilter:
            if filter(work):
                stats[f"{filter}"] += 1
                invalidators.append(f"{filter}")
        if len(invalidators) == 0:
            yield work
        else:
            stats["num_removed"] += 1
    #         else:
    #             f.write(f"{' '.join(invalidators)} invalidated {work['id']}: {work['display_name']} -- {work['abstract']}\n")

    # with (invalid_dir / "stats.json").open("w") as f:
    stats["fraction_removed"] = stats["num_removed"] / stats["total"]
    stats = OrderedDict(sorted(stats.items(), key=lambda x: (-x[1], x[0])))
    # json.dump(stats, f, indent=2, sort_keys=False)
    logger.info(f"{topic} {work_type} filter stats: {json.dumps(stats, indent=2, sort_keys=True)}")

@MEMORY.cache
def deduplicate_works_syntactic(work_type: WorkType, topic: Topic):
    works = list(filter_works(work_type, topic, CACHE_DIR))
    deduped = defaultdict(list)
    def normalize_work(work: Dict) -> str:
        return ' '.join(f"{work['display_name']} {work['abstract']}".lower().split())
    for work in works:
        key = normalize_work(work)
        deduped[key].append(work)

    # dedup_dir = OUTPUT_DIR / "deduped" / str(topic) / str(work_type)
    # dedup_dir.mkdir(parents=True, exist_ok=True)
    # with (dedup_dir / "stats.json").open("w") as f:
    #     counts = np.array([len(works) for works in deduped.values()])
    #     json.dump({"min": counts.min().item(), "max": counts.max().item(), "mean": counts.mean().item(), "std": counts.std().item(), "total": len(works)}, f, indent=2)
    counts = np.array([len(works) for works in deduped.values()])
    logger.info(f"{topic} {work_type} deduplication stats: {json.dumps({'min': counts.min().item(), 'max': counts.max().item(), 'mean': counts.mean().item(), 'std': counts.std().item(), 'total': len(works)}, indent=2)}")
    # for key, works in deduped.items():
    #     # if len(works) > 1:
    #     #     logger.debug(f"Duplicate works found: {len(works)} for key {key}\n"
    #     #                   f"{json.dumps(works, indent=2)}")
    #     yield works[0]
    return [work for (work, *_dups) in deduped.values()]

def batched(iterable, n): # TODO use itertools.batched when available
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable) # If already an iterator, `iterator is iterable == True`
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

@MEMORY.cache
def embed_works(work_type: WorkType, topic: Topic):
    logger.info(f"Embedding works for {work_type} in {topic.display_name}")
    works = iter(deduplicate_works_syntactic(work_type, topic))
    emb_cache_dir = CACHE_DIR / "embeddings" / create_unicode_safe_filename(topic) / create_unicode_safe_filename(work_type)
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    ret = []
    # Yield cached works and embeddings
    i = 1
    seen = 0
    while True:
        try:
            embeddings = np.load(emb_cache_dir / f"batch_{i}_embeddings.npy")
            with (emb_cache_dir / f"batch_{i}_ids.json").open("r") as ids_f:
                embedding_ids = json.load(ids_f)
            for embedding_id, embedding in zip(embedding_ids, embeddings, strict=True):
                # zip(works, embedding_ids, embeddings) was a nasty bug because it would
                # discard the work pulled just before embedding_ids raised StopIteration
                work = next(works) # So, manually call `next` here
                assert work["id"] == embedding_id, f"ID mismatch: {work['id']} != {embedding_id} at index {seen}"
                work["embedding"] = embedding
                seen += 1
                # yield work
                ret.append(work)
            i += 1
        except FileNotFoundError:
            break
    if seen > 0:
        logger.info(f"Yielded {seen} cached embeddings") # TODO no longer yielding
            
    # Yield new works and embeddings
    model = None # Lazy load the model in case `works` is already exhausted
    start_time = time.time()
    batch_size = 512
    seen = 0
    for batch in batched(tqdm.tqdm(works, desc="Embedding works", disable=False), batch_size):
        if model is None:
            logger.info(f"Embedding works for {work_type} in {topic.display_name}")
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            model = SentenceTransformer('avsolatorio/GIST-small-Embedding-v0')
        # Embed batch
        inputs = [f"{work['display_name']}\n{work['abstract']}" for work in batch]
        embeddings = model.encode(inputs, batch_size=batch_size, show_progress_bar=False)
        # Save batch
        with (emb_cache_dir / f"batch_{i}_embeddings.npy").open("wb") as embeddings_f, (emb_cache_dir / f"batch_{i}_ids.json").open("w") as ids_f:
            np.save(embeddings_f, embeddings)
            json.dump([work["id"] for work in batch], ids_f)
        # Yield works and embeddings
        for work, embedding in zip(batch, embeddings):
            work["embedding"] = embedding
            # yield work
            ret.append(work)
            seen += 1
        i += 1
    if seen > 0:
        logger.info(f"Yielded {seen} new embeddings in {time.time() - start_time:.2f} seconds")

    return ret

@TIMER.time()
@MEMORY.cache
def get_nearest_neighbor(index_cloud: Cloud, query_cloud: Cloud, min_dist: float) -> tuple[NDArray, NDArray]:
    logger.info(f"get_nearest_neighbor({index_cloud.name=}, {query_cloud.name=}, {min_dist=})")
    start_time = time.time()
    if min_dist == 0:
        # Search with k=2 to include self point and nearest non-self neighbor.
        D, I = index_cloud.index().search(index_cloud.embeddings(), k=2) # type: ignore
        D_first, D_second = D.T
        I_first, _I_second = I.T

        # Diagnostics for HNSW search quality
        identity_indices = np.arange(len(index_cloud.works()))
        
        # Discrepancy 1: Nearest neighbor is not the point itself.
        mismatch_mask = (I_first != identity_indices)
        
        # Discrepancy 2: Distance to first neighbor is not close to zero.
        atol = 1e-5
        nonzero_dist_mask = np.logical_not(np.isclose(D_first, 0, atol=atol))

        if np.any(mismatch_mask | nonzero_dist_mask):
            problem_indices = np.where(mismatch_mask | nonzero_dist_mask)[0]
            logger.warning(f"{index_cloud.name} -- Found {len(problem_indices)} points where the first nearest neighbor was not the point itself or had distance not close to 0.")

            # Log the largest distance to the first neighbor
            if len(problem_indices) > 0:
                worst_case_idx = problem_indices[np.argmax(D_first[problem_indices])]
                work_i = index_cloud.works()[worst_case_idx]
                work_found = index_cloud.works()[I_first[worst_case_idx]]
                logger.warning(
                    f"{index_cloud.name} -- Worst case: point {worst_case_idx} found neighbor {I_first[worst_case_idx]} "
                    f"with distance {D_first[worst_case_idx]:.4f} (should be close to 0).\n"
                    f"work_{worst_case_idx}:\n{json.dumps(work_i, indent=2)}\n"
                    f"found_neighbor_{I_first[worst_case_idx]}:\n{json.dumps(work_found, indent=2)}"
                )

            is_tie = np.isclose(D_first[problem_indices], D_second[problem_indices])
            significant_error_indices = problem_indices[~is_tie]

            if len(significant_error_indices) > 0:
                logger.warning(f"{index_cloud.name} -- Found {len(significant_error_indices)} significant HNSW errors (mismatches that are not near-ties).")
                # Log the most egregious case (largest distance to the incorrect first neighbor)
                worst_case_idx = significant_error_indices[np.argmax(D_first[significant_error_indices])]
                
                work_i = index_cloud.works()[worst_case_idx]
                work_found = index_cloud.works()[I_first[worst_case_idx]]
                logger.warning(
                    f"{index_cloud.name} -- Worst error example: point {worst_case_idx} found neighbor {I_first[worst_case_idx]} "
                    f"with distance {D_first[worst_case_idx]:.4f} (second neighbor dist: {D_second[worst_case_idx]:.4f}).\n"
                    f"work_{worst_case_idx}:\n{json.dumps(work_i, indent=2)}\n"
                    f"found_neighbor_{I_first[worst_case_idx]}:\n{json.dumps(work_found, indent=2)}"
                )
        
        # The actual nearest neighbor is the second result.
        Dnn = D_second
        # The original return signature is preserved for the caller in main().
        return Dnn, I
    else:
        res = faiss_search_outside_epsilon(index_cloud, query_cloud, 1, min_dist)
        logger.info(f"{index_cloud.name} -- Nearest neighbor search took {time.time() - start_time:.2f} seconds")
        Dnn, = res.D_outside.T
        I = res.I_outside
    return Dnn, I

@TIMER.time()
@MEMORY.cache
def get_rank_of_nearest_incloud_neighbor(selfcloud: Cloud, othercloud: Cloud, min_dist: float):
    nn_dir = CACHE_DIR / "nearest_neighbors" / create_unicode_safe_filename(f"{selfcloud.name} vs {othercloud.name}, min_dist: {min_dist}")
    nn_dir.mkdir(parents=True, exist_ok=True)
    
    selfcloud_ids = [work["id"] for work in selfcloud.works()]
    othercloud_ids = [work["id"] for work in othercloud.works()]

    selfcloud_ids_file = nn_dir / "selfcloud_ids.json"
    othercloud_ids_file = nn_dir / "othercloud_ids.json"
    def validate_ids() -> bool:
        if not selfcloud_ids_file.exists() or not othercloud_ids_file.exists():
            return False
        with selfcloud_ids_file.open("r") as sci_f, othercloud_ids_file.open("r") as oci_f:
            loaded_selfcloud_ids = json.load(sci_f)
            loaded_othercloud_ids = json.load(oci_f)
        if len(loaded_selfcloud_ids) != len(selfcloud_ids):
            logger.warning(f"{selfcloud.name} -- Selfcloud ID length mismatch: {len(loaded_selfcloud_ids)} != {len(selfcloud_ids)}")
            return False
        if len(loaded_othercloud_ids) != len(othercloud_ids):
            logger.warning(f"{selfcloud.name} -- Othercloud ID length mismatch: {len(loaded_othercloud_ids)} != {len(othercloud_ids)}")
            return False
        if (num_mismatches := np.count_nonzero(np.array(loaded_selfcloud_ids) != np.array(selfcloud_ids))) > 0:
            logger.warning(f"{selfcloud.name} -- Selfcloud mismatched IDs count: {num_mismatches}")
            return False
        if (num_mismatches := np.count_nonzero(np.array(loaded_othercloud_ids) != np.array(othercloud_ids))) > 0:
            logger.warning(f"{selfcloud.name} -- Othercloud mismatched IDs count: {num_mismatches}")
            return False
        logger.debug(f"{selfcloud.name} -- IDs match")
        return True
    ids_match = validate_ids()
    if not ids_match:
        selfcloud_ids_file.unlink(missing_ok=True)
        othercloud_ids_file.unlink(missing_ok=True)

    # Get nearest neighbors
    Dnn, I = get_nearest_neighbor(selfcloud, selfcloud, min_dist)

    closer_points_file = nn_dir / "closer_points.json"
    def load_range_search():
        if not ids_match or not closer_points_file.exists():
            return None
        with closer_points_file.open("r") as f:
            return json.load(f)
    def build_range_search():
        logger.info(f"{selfcloud.name} -- Range searching for all points in {othercloud.name} < distance to nearest neighbor")
        closer_points_file.unlink(missing_ok=True)
        start_time = time.time()
        closer_points = []
        for point, dist_ub in tqdm.tqdm(zip(selfcloud.embeddings(), Dnn), total=len(Dnn), desc=f"Range search in {othercloud.name} of {selfcloud.name} points"):
            lims, dists, ids = othercloud.index().range_search(point[np.newaxis], dist_ub) # type: ignore
            zero, num = lims
            assert zero == 0, f"Range search lims[0] != 0: {zero}"
            assert num == len(dists) == len(ids), f"Range search: not {num=} == {len(dists)=} == {len(ids)=}"
            closer_points.append(ids.tolist())
        logger.info(f"{selfcloud.name} -- Range searching took {time.time() - start_time:.2f} seconds")
        # Save range search results
        with closer_points_file.open("w") as f:
            json.dump(closer_points, f)
        return closer_points
    closer_points = load_range_search() or build_range_search()
    
    num_closer = np.array([len(closer) for closer in closer_points])
    
    artifacts_dir = ARTIFACTS_DIR / create_unicode_safe_filename(f"{selfcloud.name} vs {othercloud.name}") / f"min_dist: {min_dist}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Save statistics to file
    with (artifacts_dir / f"stats.json").open("w") as f:
        json.dump({
            "mean": np.mean(num_closer).item(),
            "median": np.median(num_closer).item(),
            "min": np.min(num_closer).item(),
            "max": np.max(num_closer).item(),
            "std": np.std(num_closer).item(),
            "num_points": len(num_closer),
            "num_points_in_other_cloud": len(othercloud.works()),
        }, f, indent=2)
    # # Log statistics
    # logger.info(f"Cloud {selfcloud.name} -- Statistics of number of points in {othercloud.name} within distance to nearest neighbor in {selfcloud.name}:\n"
    #              f"Mean: {np.mean(num_closer):.2f}\n"
    #              f"Median: {np.median(num_closer):.2f}\n"
    #              f"Min: {np.min(num_closer):.2f}\n"
    #              f"Max: {np.max(num_closer):.2f}\n"
    #              f"Std: {np.std(num_closer):.2f}\n"
    #              f"Num points: {len(num_closer)}\n"
    #              f"Num points in other cloud: {len(othercloud.items)}\n")
    counts = np.bincount(num_closer)
    # Build histogram of num_closer
    fig, hist_ax = plt.subplots(figsize=FIGSIZE)
    hist_ax.bar(range(len(counts)), counts)
    hist_ax.set_xlabel(f"Number of {othercloud.work_type}s between neighboring {selfcloud.work_type}s")
    hist_ax.set_ylabel("Count")
    hist_ax.set_xscale("log")
    hist_ax.set_yscale("log")
    title = f"{othercloud.work_type}s between neighboring {selfcloud.work_type}s for {selfcloud.topic.display_name}"
    if othercloud.topic != selfcloud.topic:
        logger.warning(f"{selfcloud.name} -- Topic mismatch: {selfcloud.topic} != {othercloud.topic}")
        title = f"{othercloud.topic} {title}"
    hist_ax.set_title(title)
    kde_ax = hist_ax.twinx()
    kde = gaussian_kde(num_closer)
    x = np.linspace(0, len(counts), 1000)
    kde_ax.plot(x, kde(x), color="red", label="KDE")
    # kde_ax.legend()
    fig.savefig(artifacts_dir / f"hist.png", dpi=DPI)
    plt.close(fig)

    # Save ids
    with selfcloud_ids_file.open("w") as sci_f, othercloud_ids_file.open("w") as oci_f:
        json.dump(selfcloud_ids, sci_f)
        json.dump(othercloud_ids, oci_f)

def get_stats(a: NDArray) -> dict:
    return {
        "mean": np.mean(a).item(),
        "median": np.median(a).item(),
        "min": np.min(a).item(),
        "max": np.max(a).item(),
        "std": np.std(a).item(),
        "count": len(a),
    }

# def get_points_by_coverage_voronoi(index_cloud: Cloud, query_cloud: Cloud, min_dist: float, output_dir: Path):
#     if index_cloud.topic != query_cloud.topic:
#         logger.warning(f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}")
#     logger.info(f"{index_cloud.topic} -- Calculating coverage of hypothetical {query_cloud.work_type}s in {index_cloud.work_type}s")
#     I_orig = faiss_search_outside_epsilon(index_cloud.index, query_cloud.embeddings, 1, min_dist).I_outside
#     logger.info(f"Statistics of nearest neighbors in {index_cloud.work_type}:\n{get_stats(np.unique(I_orig, return_counts=True)[1])}")
#     query = np.empty((query_cloud.embeddings.shape[0] + 1, query_cloud.embeddings.shape[1]), dtype=np.float32)
#     query[:-1] = query_cloud.embeddings
#     covered_points = []
#     for i, embedding in enumerate(tqdm.tqdm(index_cloud.embeddings, desc="Voronoi coverage")):
#         query[-1] = embedding
#         I = faiss_search_outside_epsilon(index_cloud.index, query, 1, min_dist).I_outside.squeeze(1)
#         covered_points.append(np.where(I == i)[0])
#         if covered_points[-1].size > 1:
#             logger.debug(f"Point {i} -- {len(covered_points[-1])} {index_cloud.work_type}s covered by {index_point['display_name']} {index_point['id']}")
#         index_point = index_cloud.items[i]
#     argsort = np.argsort([len(covered) for covered in covered_points])[::-1]
#     # Log the top 10 points
#     for i in argsort[:10]:
#         index_point = index_cloud.items[i]
#         covered = covered_points[i]
#         logger.info(f"Point {i} -- {len(covered)} {index_cloud.work_type}s covered by {index_point['display_name']} {index_point['id']}")

@TIMER.time(formatters=dict(index_clouds=lambda x: f"<{len(x)} clouds>"))
@MEMORY.cache
def get_avgref(index_clouds: Collection[Cloud], query_cloud: Cloud) -> tuple[NDArray, list[str]]:
    logger.info(f"get_avgref({query_cloud.name=}) for {len(index_clouds)} index clouds")
    index_id_to_cloud_idx = {work["id"]: (cloud, i) for cloud in index_clouds for i, work in enumerate(cloud.works())}
    ret = []
    ret_ids = []
    for work in query_cloud.works():
        referenced_embeddings = []
        for ref_id in work["referenced_works"]:
            if ref_id in index_id_to_cloud_idx:
                cloud, idx = index_id_to_cloud_idx[ref_id]
                referenced_embeddings.append(cloud.embeddings()[idx])
        # assert referenced_embeddings, f"No referenced works found for {work['id']} in {index_cloud.name}"
        if referenced_embeddings:
            ret.append(np.mean(referenced_embeddings, axis=0))
            ret_ids.append(work["id"])
    logger.info(f"get_avgref -- Found {len(ret)} avgrefs for {len(query_cloud.works())} query works ({len(ret) / len(query_cloud.works()) * 100:.2f}%)")
    return np.array(ret), ret_ids

# def plot_avgref(index_clouds: Collection[Cloud], query_cloud: Cloud, method: L
#     logger.info(f"plot_avgref({query_cloud.name=}) for {len(index_clouds)} index clouds")
#     avgref = get_avgref(index_clouds, query_cloud)
#     # Plotly interactive plot
#     query_embeddings, pca = reduce_dimensions_pca([query_cloud], dimensions=2)
#     avgref_embeddings = pca.transform(avgref)
    
@TIMER.time(formatters=dict(index_clouds=lambda x: f"<{len(x)} clouds>"))
def do_avgref_analysis(index_clouds: Sequence[Cloud], query_cloud: Cloud) -> None:
    assert len(index_clouds) == 1
    logger.info(f"do_avgref_analysis({query_cloud.name=}) for {len(index_clouds)} index clouds")
    
    artifacts_dir = ARTIFACTS_DIR / str(query_cloud.topic) / create_unicode_safe_filename(query_cloud.work_type) / "avgref_analysis"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    avgref_embeddings, avgref_ids = get_avgref(index_clouds, query_cloud)
    

    avgref_norms = np.linalg.norm(avgref_embeddings, axis=1, keepdims=True)
    logger.info(f"{query_cloud.name} -- Avgref norms: {get_stats(avgref_norms)}")
    if query_cloud.normalize_embeddings:
        logger.info(f"{query_cloud.name} -- Normalizing avgref embeddings")
        # Avoid division by zero for zero-norm vectors
        non_zero_mask = (avgref_norms > 1e-6).squeeze(1) # Use a small epsilon for floating point
        logger.info(f"{avgref_embeddings.shape=}, {avgref_norms.shape=}, {non_zero_mask.shape=}")
        logger.info(f"{avgref_embeddings.dtype=}, {avgref_norms.dtype=}, {non_zero_mask.dtype=}")
        avgref_embeddings[non_zero_mask] /= avgref_norms[non_zero_mask]

    logger.info(f"{query_cloud.name} -- Found {len(avgref_embeddings)} avgrefs for {len(query_cloud.works())} query works ({len(avgref_embeddings) / len(query_cloud.works()) * 100:.2f}%)")
    id_to_idx = {work["id"]: i for i, work in enumerate(query_cloud.works())}
    avgref_idxs = [id_to_idx[ref_id] for ref_id in avgref_ids if ref_id in id_to_idx]
    embeddings_with_avgrefs = query_cloud.embeddings()[avgref_idxs]

    # Average distance from centroid
    centroid = np.mean(embeddings_with_avgrefs, axis=0)
    embedding_dists_to_centroid = np.linalg.norm(embeddings_with_avgrefs - centroid, axis=1)
    avgref_dists_to_centroid = np.linalg.norm(avgref_embeddings - centroid, axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    all_data = np.concatenate([embedding_dists_to_centroid, avgref_dists_to_centroid])
    bins = np.histogram_bin_edges(all_data, bins=200)
    ax.stairs(*np.histogram(embedding_dists_to_centroid, bins=bins, density=False), fill=True, color='lightblue', alpha=0.7, label='Distance from works to centroid of works')
    ax.stairs(*np.histogram(avgref_dists_to_centroid, bins=bins, density=False), color='red', alpha=0.7, label='Distance from avgrefs to centroid of works')
    ax.set_xlabel("Euclidean distance to centroid")
    ax.set_ylabel("Count")
    ax.set_title(f"Euclidean distance to centroid for {query_cloud.name} ({len(avgref_embeddings)} avgrefs, {len(query_cloud.works())} query works)")
    ax.legend()
    fig.savefig(artifacts_dir / "distance_to_centroid.png", dpi=DPI)
    plt.close(fig)

    # Distribution of euclidean distances to avgref
    sq_dists = np.linalg.norm(embeddings_with_avgrefs - avgref_embeddings, axis=1) ** 2
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(sq_dists, bins=50, density=False, alpha=0.7, color='green', label='Squared Distance to avgref')

    # Overlay histogram of randomized permutation of idxs
    random_idxs = np.random.permutation(len(sq_dists))
    # random_embeddings_with_avgrefs = embeddings_with_avgrefs[random_idxs]
    # assert sorted(random_embeddings_with_avgrefs) == sorted(embeddings_with_avgrefs), "Randomized embeddings are not a permutation of original embeddings"
    random_avgref_embeddings = avgref_embeddings[random_idxs]
    assert np.all(random_avgref_embeddings.sort() == avgref_embeddings.sort()), "Randomized avgref embeddings are not a permutation of original avgref embeddings"
    # random_sq_dists = np.linalg.norm(random_embeddings_with_avgrefs - avgref_embeddings, axis=1) ** 2
    random_sq_dists = np.linalg.norm(embeddings_with_avgrefs - random_avgref_embeddings, axis=1) ** 2
    assert random_sq_dists.shape == sq_dists.shape, f"Randomized squared distances shape mismatch: {random_sq_dists.shape} != {sq_dists.shape}"
    ax.hist(random_sq_dists, bins=50, density=False, alpha=0.5, color='orange', label='Squared distance to random avgref')
    ax.legend()
    ax.set_xlabel("Squared Euclidean distance to avgref")
    ax.set_ylabel("Count")
    ax.set_title(f"Squared Euclidean distance to avgref for {query_cloud.name} ({len(avgref_embeddings)} avgrefs, {len(query_cloud.works())} query works)")
    fig.savefig(artifacts_dir / "euclidean_distance_distribution.png", dpi=DPI)
    plt.close(fig)
    # Distribution of rank of avgref in index clouds
    # Range search for each query point using the distance to avgref
    # index_cloud, = index_clouds
    
    # # Find subset of index_cloud that has at least one reference in index_cloud
    # _, index_avgref_ids = get_avgref(index_clouds, index_cloud)
    # index_id_to_idx = {work["id"]: i for i, work in enumerate(index_cloud.works())}
    # index_avgref_idxs = [index_id_to_idx[ref_id] for ref_id in index_avgref_ids if ref_id in index_id_to_idx]
    # # Build index on that subset
    # subset_index = build_index(index_cloud.index_info, index_cloud.embeddings()[index_avgref_idxs])
    # # ORIGINAL
    # # res = range_search_radii_(index_cloud.index(), embeddings_with_avgrefs, dists)
    # # SUBSET
    # res = range_search_radii_(subset_index, embeddings_with_avgrefs, random_sq_dists)

    # Build index on the full index cloud
    index_info = index_clouds[0].index_info
    assert all(cloud.index_info == index_info for cloud in index_clouds), "Index info mismatch across clouds"
    full_index = build_index_from_clouds(index_info, index_clouds)
    res = range_search_radii_(full_index, embeddings_with_avgrefs, sq_dists)

    # ORIGINAL
    # random_res = range_search_radii_(index_cloud.index(), random_embeddings_with_avgrefs, random_dists)
    # SUBSET
    # random_res = range_search_radii_(subset_index, random_embeddings_with_avgrefs, random_sq_dists)
    random_res = range_search_radii_(full_index, embeddings_with_avgrefs, random_sq_dists)

    ranks = res.lims[1:] - res.lims[:-1]
    random_ranks = random_res.lims[1:] - random_res.lims[:-1]
    logger.info(f"{query_cloud.name} -- Rank of avgref with respect to index works:\n{get_stats(ranks)}")
    logger.info(f"{query_cloud.name} -- Rank of random avgref with respect to index works:\n{get_stats(random_ranks)}")

    # --- Write statistics to global CSV ---
    # Prepare stats for both avgref and random avgref
    key_columns = [
        'topic',
        'work_type',
        'index_info',
        'duplicate_expectation',
        'normalize_embeddings',
        'drop_k',
        'denoiser',
    ]
    csv_path = Path("artifacts") / "avgref_rank_stats.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    existing_keys = set(tuple(row) for row in df[key_columns].to_numpy()) if not df.empty else set()

    # Compute stats for both nonrandom and random
    stats_nonrand = get_stats(ranks)
    stats_rand = get_stats(random_ranks)
    ks = ks_2samp(ranks, random_ranks, method='exact')
    logger.info(f"{query_cloud.name} -- KS test between ranks of avgref and random avgref: {ks}")
    row = {
        'topic': str(query_cloud.topic),
        'work_type': str(query_cloud.work_type),
        'index_info': str(index_info),
        'duplicate_expectation': query_cloud.duplicate_expectation,
        'normalize_embeddings': query_cloud.normalize_embeddings,
        'drop_k': query_cloud.drop_k,
        'denoiser': str(query_cloud.denoiser),
        'mean_nonrand': stats_nonrand['mean'],
        'median_nonrand': stats_nonrand['median'],
        'min_nonrand': stats_nonrand['min'],
        'max_nonrand': stats_nonrand['max'],
        'std_nonrand': stats_nonrand['std'],
        'count_nonrand': stats_nonrand['count'],
        'num_avgrefs': len(avgref_embeddings),
        'num_query_works': len(query_cloud.works()),
        'mean_rand': stats_rand['mean'],
        'median_rand': stats_rand['median'],
        'min_rand': stats_rand['min'],
        'max_rand': stats_rand['max'],
        'std_rand': stats_rand['std'],
        'count_rand': stats_rand['count'],
        'ks_log_pvalue': ks.log_pvalue,
    }
    key_tuple = tuple(row[k] for k in key_columns)
    if key_tuple not in existing_keys:
        new_df = pd.DataFrame([row])
        if not df.empty:
            assert list(df.columns) == list(new_df.columns), f"Column mismatch between existing and new DataFrame: {list(df.columns)=} != {list(new_df.columns)=}"
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        df = df.sort_values(by="mean_nonrand", ascending=False)
        df.to_csv(csv_path, index=False, float_format='%.5f')
    else:
        logger.info(f"Skipping existing row for {' '.join(f'{k}={row[k]}' for k in key_columns)}")
    # --- End CSV output ---

    # CDF of ranks
    cumsum = np.cumsum(np.bincount(ranks.astype(np.int64))) / len(ranks)
    cumsum_random = np.cumsum(np.bincount(random_ranks.astype(np.int64))) / len(random_ranks)

    # Plot CDF of ranks
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.plot(np.arange(len(cumsum)), cumsum, label='Rank of avgref', color='green')
    # ax.plot(np.arange(len(cumsum_random)), cumsum_random, label='Rank of random avgref', color='orange')
    ax.scatter(np.arange(len(cumsum)), cumsum, label='Rank of avgref', color='green', s=10, zorder=2)
    ax.scatter(np.arange(len(cumsum_random)), cumsum_random, label='Rank of random avgref', color='orange', s=10, zorder=1)

    # Overlay the ks statistic
    # ks = kstest(ranks, random_ranks)
    logger.info(f"{query_cloud.name} -- KS test between ranks of avgref and random avgref: {ks}")
    lower, higher = (cumsum_random, cumsum) if ks.statistic_sign == 1 else (cumsum, cumsum_random)
    ymin = lower[ks.statistic_location].item()
    ymax = higher[ks.statistic_location].item()
    assert np.isclose(ymax - ymin, ks.statistic), f"KS statistic mismatch: {ymax - ymin} != {ks.statistic} at location {ks.statistic_location}"
    logger.info(f"{query_cloud.name} -- KS statistic location: {ks.statistic_location}, ymin: {ymin}, ymax: {ymax}")
    # Overlay KS max diff
    ax.vlines(ks.statistic_location,
               ymin=ymin,
               ymax=ymax,
               color='purple', linestyle='dashed', linewidth=1, label=f"Max diff of {ymax-ymin:.2f} at rank {ks.statistic_location} (KS p-value = {ks.log_pvalue:.1e})",
                zorder=0)

    # # Overlay average ranks
    # ax.axvline(average_rank, color='blue', linestyle='dashed', linewidth=1, label=f"Average Rank = {average_rank:.2f} ({average_rank / max(ranks) * 100:.1f}% of max)", zorder=3)
    # ax.axvline(average_rank_random, color='red', linestyle='dashed', linewidth=1, label=f"Average Random Rank = {average_rank_random:.2f} ({average_rank_random / max(random_ranks) * 100:.1f}% of max)", zorder=3)


    ax.set_xlabel("Rank of avgref in index cloud")
    ax.set_ylabel("Cumulative count")
    ax.set_title(f"CDF of ranks of avgref for {len(avgref_embeddings)} {query_cloud.name} works")
    ax.legend()
    fig.savefig(artifacts_dir / "rank_cdf.png", dpi=DPI)
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / "rank_cdf_logx.png", dpi=DPI)
    # ax.set_yscale("log")
    plt.close(fig)

    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # ax.hist(ranks, bins=200, density=False, alpha=0.7, color='green', label='Rank of avgref')
    # ax.hist(random_ranks, bins=200, density=False, alpha=0.5, color='orange', label='Rank of random avgref')
    all_data = np.concatenate([ranks, random_ranks])
    bins = np.histogram_bin_edges(all_data, bins=200)
    ax.stairs(*np.histogram(ranks, bins=bins, density=False), fill=True, alpha=0.7, color='green', label='Rank of avgref')
    ax.stairs(*np.histogram(random_ranks, bins=bins, density=False), fill=True, alpha=0.5, color='orange', label='Rank of random avgref')
    # # Overlay expected rank
    # expected_rank = len(index_avgref_idxs) / 2
    # ax.axvline(expected_rank, color='red', linestyle='dashed', linewidth=1, label=f"Expected Rank ({len(index_avgref_idxs)} / 2 = {expected_rank})")
    ax.legend()
    ax.set_xlabel("Rank of avgref in index cloud")
    ax.set_ylabel("Count")
    ax.set_title(f"# works closer than avgref for {len(avgref_embeddings)} {query_cloud.name} works in {full_index.ntotal} works")
    fig.savefig(artifacts_dir / "rank_distribution.png", dpi=DPI)
    ax.set_xscale("log")
    fig.savefig(artifacts_dir / "rank_distribution_logx.png", dpi=DPI)
    # ax.set_yscale("log")
    plt.close(fig)


    # # Sanity check: the rank of a random work wrt another random work should be distributed uniformly from 0 to len(index_cloud.works())
    # # sanity_choices1 = np.random.choice(len(query_cloud.works()), size=len(query_cloud.works()), replace=True)
    # # sanity_choices2 = np.random.choice(len(index_cloud.works()), size=len(index_cloud.works()), replace=True)
    # sanity_choices1 = np.arange(len(query_cloud.works())) # Use all query works
    # sanity_choices2 = np.random.permutation(len(index_cloud.works())) # Randomly permute index works
    # assert sanity_choices1.shape == sanity_choices2.shape, f"Sanity choices shape mismatch: {sanity_choices1.shape} != {sanity_choices2.shape}"
    # assert sanity_choices1.shape == (len(index_cloud.works()),), f"Sanity choices shape mismatch: {sanity_choices1.shape} != ({len(query_cloud.works()),})"
    # assert index_cloud.embeddings().shape == index_cloud.embeddings()[sanity_choices1].shape, f"Index cloud embeddings shape mismatch: {index_cloud.embeddings().shape} != {index_cloud.embeddings()[sanity_choices1].shape}"
    # assert index_cloud.embeddings().shape == index_cloud.embeddings()[sanity_choices2].shape, f"Index cloud embeddings shape mismatch: {index_cloud.embeddings().shape} != {index_cloud.embeddings()[sanity_choices2].shape}"
    # sanity_sq_dists = np.linalg.norm(index_cloud.embeddings()[sanity_choices1] - index_cloud.embeddings()[sanity_choices2], axis=1) ** 2
    # assert sanity_sq_dists.shape == (len(index_cloud.works()),), f"Sanity squared distances shape mismatch: {sanity_sq_dists.shape} != ({len(index_cloud.works()),})"
    # sanity_res = range_search_radii_(
    #     index_cloud.index(),
    #     index_cloud.embeddings()[sanity_choices1],
    #     sanity_sq_dists
    # )
    # logger.info(f"{query_cloud.name} -- Sanity check rank of random work with respect to {index_cloud.name} works:\nlims ({sanity_res.lims.shape=}): {sanity_res.lims[:10]} ... {sanity_res.lims[-10:]}")
    # sanity_ranks = sanity_res.lims[1:] - sanity_res.lims[:-1]
    # logger.info(f"{query_cloud.name} -- Sanity check rank of random work with respect to {index_cloud.name} works:\n{get_stats(sanity_ranks)}")
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.hist(sanity_ranks, bins=200, density=False, alpha=0.7, color='blue', label='Rank of random work')
    # sanity_expected_rank = len(index_cloud.works()) / 2
    # ax.axvline(sanity_expected_rank, color='red', linestyle='dashed', linewidth=1, label=f'Expected Rank ({len(index_cloud.works())} / 2 = {sanity_expected_rank})') 
    # ax.legend()
    # ax.set_xlabel("Rank of random work in index cloud")
    # ax.set_ylabel("Count")
    # ax.set_title(f"# works closer than random work for {len(index_cloud.works())} {index_cloud.name} works")
    # fig.savefig(artifacts_dir / "sanity_rank_distribution.png", dpi=DPI)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # fig.savefig(artifacts_dir / "sanity_rank_distribution_log.png", dpi=DPI)
    # plt.close(fig)


@dataclass
class GetBallCoverageResult:
    index_works_per_query_work: NDArray
    ball_counts: NDArray # Number of index works within the ball for each query work
    covered_points: list[tuple[NDArray, NDArray]] # List of (dists, ids) tuples

@TIMER.time()
@MEMORY.cache
def get_ball_coverage(index_cloud: Cloud, query_cloud: Cloud, radius_fraction: float):
    assert index_cloud.topic == query_cloud.topic, f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}"
    topic = index_cloud.topic
    index_work_type = index_cloud.work_type
    query_work_type = query_cloud.work_type
    logger.info(f"{topic} -- Finding closest {query_work_type} to each {index_work_type} (each {query_work_type} gets a voronoi cell of {index_work_type}s)")
    
    # For each article (index_cloud), find the nearest book (query_cloud).
    res = faiss_search_outside_epsilon(query_cloud, index_cloud, k=1, epsilon=0)
    D = res.D_outside.squeeze(1) # Distances to nearest book, shape: (num_articles,)
    I = res.I_outside.squeeze(1) # Indices of nearest book, shape: (num_articles,)

    # Count, for each book, how many articles have it as their nearest neighbor.
    index_works_per_query_work = np.bincount(I, minlength=query_cloud.index().ntotal)
    
    title = f"{topic} - Number of {index_work_type}s covered by each {query_work_type} (radius fraction: {radius_fraction})"
    logger.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")

    # Now, for each article, count how many other articles are within a radius defined by its distance to the nearest book.
    D_scaled = D * radius_fraction
    # The query and radii must correspond. We query with articles and use the radii calculated for each article.
    res = range_search_radii_(index_cloud.index(), index_cloud.embeddings(), D_scaled)
    ball_counts = res.lims[1:] - res.lims[:-1]
    logger.info(f"Number of {index_work_type}s covered by each {index_work_type} (ball):\n{get_stats(ball_counts)}")

    # The `range_search_radii_` function already provides all the necessary information.
    # The manual loop below is redundant and has been replaced by unpacking the result.
    dists_list = from_flat_with_lims(FlatWithLims(res.dists, res.lims))
    ids_list = from_flat_with_lims(FlatWithLims(res.ids, res.lims))
    covered_points = list(zip(dists_list, ids_list))

    return GetBallCoverageResult(
        index_works_per_query_work=index_works_per_query_work,
        ball_counts=ball_counts,
        covered_points=covered_points)
# def get_ball_coverage(index_cloud: Cloud, query_cloud: Cloud, radius_fraction: float):
#     assert index_cloud.topic == query_cloud.topic, f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}"
#     topic = index_cloud.topic
#     index_work_type = index_cloud.work_type
#     query_work_type = query_cloud.work_type
#     logger.info(f"{topic} -- Finding closest {query_work_type} to each {index_work_type} (each {query_work_type} gets a voronoi cell of {index_work_type}s)")
#     res = faiss_search_outside_epsilon(query_cloud, index_cloud, k=1, epsilon=0)
#     D = res.D_outside.squeeze(1)
#     I = res.I_outside.squeeze(1)
#     Ip1 = np.ones(index_cloud.index().ntotal, dtype=np.int32) # Increment each index work by 1 so that np.unique counts even empty cells
#     Ip1[I] += 1 # Each index work is counted once for itself
#     _idxs, index_works_per_query_work_p1 = np.unique(Ip1, return_counts=True)
#     index_works_per_query_work = index_works_per_query_work_p1 - 1 # Decrement back to original counts, leaving 0 for empty cells
#     title = f"{topic} - Number of {index_work_type}s covered by each {query_work_type}"
#     logger.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")
#     # Now get stats for the balls
#     D_scaled = D * radius_fraction
#     res = range_search_radii_(index_cloud.index(), query_cloud.embeddings(), D_scaled)
#     ball_counts = res.lims[1:] - res.lims[:-1]
#     logger.info(f"{title} (ball):\n{get_stats(ball_counts)}")

#     covered_points = []
#     index_cloud_index = index_cloud.index()
#     for point, embedding in enumerate(tqdm.tqdm(index_cloud.embeddings(), desc=f"Ball coverage (for each {index_work_type}, count the number of {index_work_type}s within {radius_fraction} * distance to nearest {query_work_type})")):
#         lims, dists, ids = index_cloud_index.range_search(embedding[np.newaxis], D_scaled[point]) # type: ignore
#         zero, num = lims
#         assert zero == 0, f"Range search lims[0] != 0: {zero}"
#         assert num == len(dists) == len(ids), f"Range search: not {num=} == {len(dists)=} == {len(ids)=}"
#         covered_points.append((dists, ids))
#     return GetBallCoverageResult(
#         index_works_per_query_work=index_works_per_query_work,
#         ball_counts=ball_counts,
#         covered_points=covered_points)


class InferenceManager:
    def __init__(self):
        self.pipe = loadPipeline()

    def run(self, prompt: str, output_path: Path):
        start_time = time.time()
        llama_output = self.pipe(prompt, max_new_tokens=4096)
        elapsed_time = time.time() - start_time
        assert isinstance(llama_output, list)
        assert len(llama_output) == 1
        assert isinstance(llama_output[0], dict)
        assert set(llama_output[0].keys()) == {"generated_text"}
        assert len(llama_output[0]['generated_text']) >= len(prompt)
        generated_text = llama_output[0]['generated_text'][len(prompt):]
        with open(output_path, "w") as f:
            f.write(generated_text)
        logger.info(f"Inference took {elapsed_time:.2f} seconds for prompt of length {len(prompt)}, output length {len(generated_text)}. Saved to {output_path}")

@TIMER.time()
@lru_cache(maxsize=1)
def loadPipeline():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use "nf4" for NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
        bnb_4bit_use_double_quant=True, # Optional: for slightly higher precision
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading {model_id}")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Match the compute dtype
    )
    logger.info(f"Model {model_id} loaded in {time.time() - start_time:.2f} seconds")

    # 5. Create the text-generation pipeline with your pre-loaded, quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return pipe

@TIMER.time(formatters=dict(index_info=lambda _: "<...>"))
def get_points_by_coverage_ball(topic: Topic, index_work_type: WorkType, query_work_type: WorkType, radius_fraction: float, index_info: IndexInfo, max_works: int, duplicate_expectation: float, normalize_embeddings: bool, denoiser: Denoiser, drop_k: int):
    index_cloud = Cloud(topic=topic, work_type=index_work_type, index_info=index_info, max_works=max_works, duplicate_expectation=duplicate_expectation, normalize_embeddings=normalize_embeddings, denoiser=denoiser, drop_k=drop_k)
    query_cloud = Cloud(topic=topic, work_type=query_work_type, index_info=index_info, max_works=max_works, duplicate_expectation=duplicate_expectation, normalize_embeddings=normalize_embeddings, denoiser=denoiser, drop_k=drop_k)
    assert index_cloud.topic == query_cloud.topic, f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}"
    topic = index_cloud.topic
    index_work_type = index_cloud.work_type
    query_work_type = query_cloud.work_type
    artifacts_dir = ARTIFACTS_DIR / f"{topic}" / f"{index_work_type}s covered by {query_work_type}s"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # index_cloud = get_cloud(topic, index_work_type, **kwargs) 
    # query_cloud = get_cloud(topic, query_work_type, **kwargs)
    # index_cloud = Cloud(topic=topic, work_type=index_work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    # query_cloud = Cloud(topic=topic, work_type=query_work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    # output_dir = ARTIFACTS_DIR / create_unicode_safe_filename(f"{index_cloud.name} vs {query_cloud.name}") / f"min_dist: {min_dist}, radius_fraction: {radius_fraction}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # if index_cloud.topic != query_cloud.topic:
    #     logger.warning(f"Topic mismatch: {index_cloud.topic} != {query_cloud.topic}")
    # logger.info(f"{index_cloud.topic} -- Calculating coverage of hypothetical {query_cloud.work_type}s in {index_cloud.work_type}s")
    # logger.info(f"{topic} -- Finding closest {query_work_type} to each {index_work_type} (each {query_work_type} gets a voronoi cell of {index_work_type}s)")
    # res = faiss_search_outside_epsilon(query_cloud, index_cloud, k=1, epsilon=0)
    # D = res.D_outside.squeeze(1)
    # I = res.I_outside.squeeze(1)
    # _idxs, index_works_per_query_work = np.unique(I, return_counts=True)
    # # Add in zero counts for empty voronoi cells
    # index_works_per_query_work = np.concatenate((index_works_per_query_work, np.zeros(index_cloud.index().ntotal - len(index_works_per_query_work), dtype=np.int32)))
    # title = f"{topic} - Number of {index_work_type}s covered by each {query_work_type}"
    # logger.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")
    
    

    # # Now get stats for the balls
    # D_scaled = D * radius_fraction
    # res = range_search_radii_(index_cloud.index(), query_cloud.embeddings(), D_scaled)
    # ball_counts = res.lims[1:] - res.lims[:-1]
    # logger.info(f"{title} (ball):\n{get_stats(ball_counts)}")

    res = get_ball_coverage(index_cloud, query_cloud, radius_fraction)
    index_works_per_query_work = res.index_works_per_query_work
    ball_counts = res.ball_counts
    covered_points = res.covered_points
    title = f"{topic} - Number of {index_work_type}s covered by each {query_work_type}"

    fig, ax = plt.subplots(figsize=FIGSIZE)

    all_data = np.concatenate([index_works_per_query_work, ball_counts])
    bins = np.histogram_bin_edges(all_data, bins=50)
    ax.stairs(*np.histogram(index_works_per_query_work, bins=bins), color="lightblue", fill=True, zorder=1, label=f"Number of {index_work_type}s covered by each {query_work_type} (voronoi)")
    ax.stairs(*np.histogram(ball_counts, bins=bins), color="orange", fill=False, zorder=2, label=f"Number of {index_work_type}s covered by each {index_work_type} (ball of radius {radius_fraction} * dist to nearest {query_work_type})")
    ax.set_xlabel(f"Number of {index_work_type}s covered")
    ax.set_ylabel(f"Frequency")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.savefig(artifacts_dir / f"{index_work_type}s covered per {query_work_type}.png", dpi=DPI)
    plt.close(fig)
    
    # covered_points_dir = OUTPUT_DIR / "covered_points" / f"{topic}" / f"{index_work_type}s covered by {query_work_type}s" / f"radius_fraction: {radius_fraction}"
    # covered_points_dir.mkdir(parents=True, exist_ok=True)
    # covered_points_lims_file = covered_points_dir / "covered_point_lims.npy"
    # covered_points_ids_file = covered_points_dir / "covered_point_ids.npy"
    # covered_points_dists_file = covered_points_dir / "covered_point_dists.npy"
    


    # def load_covered_points() -> Optional[list[tuple[NDArray, NDArray]]]:
    #     if not covered_points_lims_file.exists() or not covered_points_ids_file.exists() or not covered_points_dists_file.exists():
    #         return None
    #     with covered_points_lims_file.open("rb") as f:
    #         lims = np.load(f)
    #     with covered_points_ids_file.open("rb") as f:
    #         ids = np.load(f)
    #     with covered_points_dists_file.open("rb") as f:
    #         dists = np.load(f)
    #     return list(zip(from_flat_with_lims(FlatWithLims(dists, lims)),
    #                     from_flat_with_lims(FlatWithLims(ids, lims))))
    # def build_covered_points() -> list[tuple[NDArray, NDArray]]:
    #     covered_points = []
    #     index_cloud_index = index_cloud.index()
    #     for point, embedding in enumerate(tqdm.tqdm(index_cloud.embeddings(), desc="Ball coverage")):
    #         lims, dists, ids = index_cloud_index.range_search(embedding[np.newaxis], D_scaled[point]) # type: ignore
    #         zero, num = lims
    #         assert zero == 0, f"Range search lims[0] != 0: {zero}"
    #         assert num == len(dists) == len(ids), f"Range search: not {num=} == {len(dists)=} == {len(ids)=}"
    #         covered_points.append((dists, ids))
    #     # Save covered points
    #     # with covered_points_lims_file.open("wb") as f:
    #     #     np.save(f, np.array([0] + [len(covered) for covered in covered_points], dtype=np.uint32).cumsum())
    #     # with covered_points_ids_file.open("wb") as f:
    #     #     np.save(f, np.concatenate([ids for _, ids in covered_points], axis=0))
    #     # with covered_points_dists_file.open("wb") as f:
    #     #     np.save(f, np.concatenate([dists for dists, _ in covered_points], axis=0))
    #     res_ids = to_flat_with_lims([ids for _, ids in covered_points])
    #     res_dists = to_flat_with_lims([dists for dists, _ in covered_points])
    #     assert np.array_equal(res_ids.lims, res_dists.lims), f"IDs and dists lims mismatch: {res_ids.lims} != {res_dists.lims}"
    #     with covered_points_lims_file.open("wb") as f:
    #         np.save(f, res_ids.lims)
    #     with covered_points_ids_file.open("wb") as f:
    #         np.save(f, res_ids.concatenated)
    #     with covered_points_dists_file.open("wb") as f:
    #         np.save(f, res_dists.concatenated)

    #     return covered_points
    # covered_points = load_covered_points() or build_covered_points()
    # argsort = np.argsort([len(covered) for covered in covered_points])[::-1]
    # # Log the top 10 points
    # for rank, point in enumerate(argsort[:10]):
    #     index_point = index_cloud.items[point]
    #     covered = []
    #     for covered_i in covered_ids[:10]:
    #         covered_point = index_cloud.items[covered_i]
    #         covered.append(f"{covered_point['display_name']} -- {covered_point['abstract']}")
    #     top10_dir = ARTIFACTS_DIR / "top10"
    #     top10_dir.mkdir(parents=True, exist_ok=True)
    #     with open(top10_dir / f"{index_cloud.name} - {query_cloud.name} - {rank}.txt", "w") as f:
    #         f.write("\n\n".join(f"{i}: {c}" for i, c in enumerate(covered)))
    #     logger.debug(f"{rank}: {len(covered)} {index_cloud.work_type}s covered by {index_point['display_name']} {index_point['id']}\n"
    #                   + "\n\t".join(f"{i}: {c}" for i, c in enumerate(covered))
    #                   + "\n\n")



    # pipe = loadPipeline()

    def output_covered_points(hole_rank, highest_coverage_id, covered_dists, covered_ids, original_coverage):
        assert len(covered_dists) == len(covered_ids), f"Covered dists and ids length mismatch: {len(covered_dists)} != {len(covered_ids)}"
        topic_dir = artifacts_dir / "highest coverage (ball)"
        topic_dir.mkdir(parents=True, exist_ok=True)
        
        order = np.argsort(covered_dists)
        # order = np.random.permutation(len(covered_dists))
        covered_ids = covered_ids[order]
        covered_dists = covered_dists[order]

        sample_idxs = np.random.choice(len(covered_ids) - 1, size=min(29, len(covered_ids) - 1), replace=False)
        sample_idxs = np.concatenate(([0], sample_idxs + 1)) # + 1 to skip 0
        assert np.unique(sample_idxs).size == sample_idxs.size, f"Sample indices are not unique: {sample_idxs}"
        sample_ids = covered_ids[sample_idxs]
        sample_dists = covered_dists[sample_idxs]

        # Collect titles and abstracts of the top 10 covered articles
        top_articles = []
        # for i, covered_i in enumerate(covered_ids[:]):
        #     covered_point = index_cloud.works()[covered_i]
        #     top_articles.append(f"TITLE: {covered_point['display_name']}\nABSTRACT: {covered_point['abstract']}")
        for sample_id in sample_ids:
            covered_point = index_cloud.works()[sample_id]
            # top_articles.append(f"TITLE: {covered_point['display_name']}\nABSTRACT: {covered_point['abstract']}")
            top_articles.append({"title": covered_point['display_name'], "abstract": covered_point['abstract']})
        articles_text = json.dumps(top_articles, indent=2, ensure_ascii=False)

        with open(topic_dir / f"{hole_rank}_articles.txt", "w") as f:
            f.write(articles_text)

        # # Run model inference asynchronously
        # prompt = (
        #     "You are a book editor. Given the following list of research articles (titles and abstracts), "
        #     "generate a suitable title, abstract, and table of contents for a textbook that would prepare a student to understand the topics in these articles. "
        #     # f'Format your answer in JSON, e.g. `{ {"title": "The Title", "abstract": "The abstract", "table_of_contents": "Part 1. ..."} }`.\n\n'
        #     "Format your answer as:\n"
        #     "==TITLE: <title goes here>\n"
        #     "==ABSTRACT: <abstract goes here>\n"
        #     "==TABLE OF CONTENTS:\n"
        #     "<table of contents goes here>\n"
        #     "==END TABLE OF CONTENTS\n\n"
        #     f"Articles:\n\n{articles_text}\n\n"
        # )
        # logger.info(f"Submitting inference job for hole rank {hole_rank}")
        # inference_manager.put(InferenceManager.Job(prompt=prompt, output_path=topic_dir / f"{hole_rank}_book_info.txt"))

        # Get top works closest to centroid
        with open(topic_dir / f"{hole_rank}_center.txt", "w") as f:
            for idx in range(min(20, len(covered_dists))):
                dist = covered_dists[idx]
                article = index_cloud.works()[covered_ids[idx]]
                f.write(f"{dist:.4f} -- {article['display_name']} -- {article['abstract']}\n")
        # # Compute convex hull
        # covered_embeddings = index_cloud.embeddings()[covered_ids]
        # if covered_embeddings.shape[0] > covered_embeddings.shape[1]:
        #     try:
        #         hull = ConvexHull(covered_embeddings)
        #         hull_idxs = hull.vertices
        #     except Exception as e:
        #         logger.warning(f"Convex hull computation failed: {e}")
        #         hull_idxs = np.arange(len(covered_embeddings))
        # else:
        #     logger.warning("Not enough points for convex hull; using all points.")
        #     hull_idxs = np.arange(len(covered_embeddings))
        # with open(top10_dir / f"{hole_rank}_convex_hull.txt", "w") as f:
        #     for idx in hull_idxs:
        #         dist = covered_dists[idx]
        #         article = index_cloud.works()[covered_ids[idx]]
        #         f.write(f"{dist:.4f} -- {article['display_name']} -- {article['abstract']}\n")

    # Iterate over the top 10 points, deleting covered points from the list
    logger.info(f"Determining top {index_work_type}s by number of {index_work_type}s within {radius_fraction} * distance to nearest {query_work_type}")
    num_centers = 5
    # original_coverage = np.array([len(ids) for _dists, ids in covered_points])
    def coverage(covered_points):
        return np.array([len(ids) for _dists, ids in covered_points])
    coverages = [coverage(covered_points)]
    for rank in tqdm.tqdm(range(num_centers), desc=f"Outputting highest coverage {index_work_type}s"):
        highest_coverage_id = np.argmax([len(ids) for _dists, ids in covered_points])
        dists, ids = covered_points[highest_coverage_id]
        output_covered_points(rank, highest_coverage_id, dists, ids, coverages[0][highest_coverage_id])
        # Remove the covered points from the list
        if highest_coverage_id not in ids:
            logger.warning(f"Top point {highest_coverage_id} not in its own covered points")
        # Remove the covered points from the list
        for covered_i in ids:
            covered_points[covered_i] = (np.empty(0, dtype=dists.dtype), np.empty(0, dtype=ids.dtype))
        # Remove the covered points from each remaining point's list
        ids = np.unique(ids)
        for coverage_to_plot in range(len(covered_points)):
            i_dists, i_ids = covered_points[coverage_to_plot]
            to_keep = np.isin(i_ids, ids)
            covered_points[coverage_to_plot] = (i_dists[~to_keep], i_ids[~to_keep])
        coverages.append(coverage(covered_points))

    # Plot a histogram comparing old to new ball coverages
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # pruned_coverage = np.array([len(ids) for _dists, ids in covered_points])
    # all_data = np.concatenate([original_coverage, pruned_coverage])
    all_data = np.concatenate(coverages)
    assert all_data.ndim == 1, f"all_data should be 1D, got {all_data.shape=}"
    bins = np.histogram_bin_edges(all_data, bins=50)
    # ax.stairs(*np.histogram(pruned_coverage, bins=bins), color="lightblue", fill=True, zorder=1, label="New coverage")
    # ax.stairs(*np.histogram(original_coverage, bins=bins), color="black", fill=False, zorder=2, label="Original coverage")

    # def i_to_rgb(i: int) -> tuple[float, float, float, float]:
    #     v = i / (len(coverages) - 1)
    #     return (v, v, v, 0.9)
    # for i, cov in enumerate(coverages):
    #     # ax.stairs(*np.histogram(cov, bins=bins), color=f"C{i}", fill=True, alpha=0.7, zorder=i+1, label=f"Coverage after pruning {i} {index_work_type}s")
    #     # Grayscale
    #     ax.stairs(*np.histogram(cov, bins=bins), color=i_to_rgb(i), fill=True, zorder=i+1, label=f"Coverage after pruning {i} {index_work_type}s")

    coverages_to_plot = sorted(set([min(i, len(coverages) - 1)
                                    for i in [0, 1, 2, len(coverages) - 1]]))

    cmap = plt.get_cmap("coolwarm")
    def i_to_color(i: int) -> tuple[float, float, float, float]:
        num_cycles = 1
        mod = len(coverages_to_plot) // num_cycles
        v = (i % mod) / (mod - 1)
        ret = cmap(v, alpha=0.75)
        # darkener = (i // mod) / (num_cycles - 1)
        # darkener *= 0.5
        # scalar = 1 - darkener
        # return (ret[0] * scalar, ret[1] * scalar, ret[2] * scalar, ret[3])
        return ret
    for i, coverage_to_plot in enumerate(coverages_to_plot):
        ax.stairs(*np.histogram(coverages[coverage_to_plot], bins=bins), color=i_to_color(i), fill=True, zorder=coverage_to_plot+1, label=f"Coverage after pruning {coverage_to_plot} {index_work_type}s")

    ax.set_xlabel(f"Number of {index_work_type}s covered")
    ax.set_ylabel(f"Number of {index_work_type}s with a given coverage of other {index_work_type}s")
    ax.set_title(f"{topic} - Number of {index_work_type}s covered by each {index_work_type}")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.savefig(artifacts_dir / f"{index_work_type}s covered per {index_work_type} (pruning overlaid).png", dpi=DPI)
    plt.close(fig)

    # # Plot a histogram comparing old to new voronoi coverages
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # ax.stairs(*np.histogram(original_coverage, bins=100), color="lightblue", fill=True, zorder=1, label="Original coverage")
    # # ax1 = ax.twinx()

# @MEMORY.cache
# def reduce_dimensions(clouds: Iterable[Cloud], dimensions: int, method: Literal["pca", "tsne"]) -> NDArray:
#     embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
#     logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using {method.upper()}")
#     start_time = time.time()
#     if method == "pca":
#         pca = PCA(n_components=dimensions)
#         transformed_embeddings = pca.fit_transform(embeddings)
#     elif method == "tsne":
#         tsne = TSNE(n_components=dimensions, perplexity=30, max_iter=1000)
#         transformed_embeddings = tsne.fit_transform(embeddings)
#     else:
#         raise ValueError(f"Unknown method: {method}")
#     logger.info(f"{method.upper()} took {time.time() - start_time:.2f} seconds")
#     return transformed_embeddings

@TIMER.time()
@MEMORY.cache
def reduce_dimensions_pca(clouds: Iterable[Cloud], dimensions: int) -> tuple[NDArray, PCA]:
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using PCA")
    pca = PCA(n_components=dimensions)
    transformed_embeddings = pca.fit_transform(embeddings)
    return transformed_embeddings, pca

@MEMORY.cache
def reduce_dimensions_tsne(clouds: Iterable[Cloud], dimensions: int) -> tuple[NDArray, TSNE]:
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {dimensions} dimensions using t-SNE")
    start_time = time.time()
    tsne = TSNE(n_components=dimensions, perplexity=30, max_iter=1000)
    transformed_embeddings = tsne.fit_transform(embeddings)
    logger.info(f"t-SNE took {time.time() - start_time:.2f} seconds")
    return transformed_embeddings, tsne

def plot_embeddings(clouds: Iterable[Cloud], colored_feature: str, method: Literal["pca", "tsne"], dimensions: int = 2, color_f = lambda colors: colors, **kwargs):
    """
    Plot the embeddings of the works in a 2D or 3D space using PCA or t-SNE, colored by the specified feature.
    Outputs a plotly html file.
    """
    assert dimensions in {2, 3}, f"Dimensions must be 2 or 3, got {dimensions}"
    # topics, work_types = list(topics), list(work_types)
    # assert topics == sorted(topics, key=lambda t: t.value), f"Topics must be sorted: {topics}"
    # assert work_types == sorted(work_types, key=lambda t: t.value), f"Work types must be sorted: {work_types}"
    artifacts_dir = ARTIFACTS_DIR / "plots" / create_unicode_safe_filename(colored_feature)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # clouds = {
    #     (topic, work_type): Cloud(topic=topic, work_type=work_type, index_info=kwargs["index_info"], max_works=kwargs["max_works"])
    #     for topic, work_type in itertools.product(topics, work_types)
    # }
    works = [work for cloud in clouds for work in cloud.works()]
    
    if method == "pca":
        transformed_embeddings, _ = reduce_dimensions_pca(clouds, dimensions)
    elif method == "tsne":
        transformed_embeddings, _ = reduce_dimensions_tsne(clouds, dimensions)
    else:
        raise ValueError(f"Unknown method: {method}")

    plot_title_prefix = ""
    labels = {}

    cloud_names = ", ".join(sorted(cloud.name for cloud in clouds))

    if method == "pca":
        plot_title_prefix = f"{cloud_names} PCA embeddings colored by {colored_feature}"
        labels = {"x": "PCA 1", "y": "PCA 2"}
        if dimensions == 3:
            labels["z"] = "PCA 3"
        output_file_suffix = f"_pca_{dimensions}d.html"
    elif method == "tsne":
        plot_title_prefix = f"{cloud_names} t-SNE embeddings colored by {colored_feature}"
        labels = {"x": "t-SNE 1", "y": "t-SNE 2"}
        output_file_suffix = "_tsne.html"
    else:
        raise ValueError(f"Unknown method: {method}")

    scatter_args = dict(
        x=transformed_embeddings[:, 0],
        y=transformed_embeddings[:, 1],
        color=color_f([work[colored_feature] for work in works]),
        hover_name=[f"{work['display_name']} ({work['id']})\\n{colored_feature}: {work[colored_feature]}" for work in works],
        title=plot_title_prefix,
        labels=labels,
    )
    if dimensions == 3:
        scatter_args["z"] = transformed_embeddings[:, 2]
    
    scatter_method = px.scatter if dimensions == 2 else px.scatter_3d

    # fig = scatter_method(**{k: v for k, v in scatter_args.items() if k in inspect.signature(scatter_method).parameters})
    fig = scatter_method(**scatter_args) # type: ignore

    fig.update_traces(marker=dict(size=5) if dimensions == 2 else dict(size=3, opacity=0.7))

    output_file = artifacts_dir / f"{create_unicode_safe_filename(cloud_names)}{output_file_suffix}"
    fig.write_html(output_file)

def plot_norms(clouds: Collection[Cloud]) -> None:
    # Plot histogram of norms of all embeddings in the clouds
    embeddings = np.concatenate([cloud.embeddings() for cloud in clouds], axis=0)
    logger.info(f"Plotting norms of {embeddings.shape} embeddings from {len(clouds)} clouds")
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Norms stats:\n"
                f"\tmin: {norms.min()}\n"
                f"\tmed: {np.median(norms)}\n"
                f"\tmax: {norms.max()}\n"
                f"\tmean: {norms.mean()}\n"
                f"\tstd: {norms.std()}\n"
                )
    assert norms.shape == (embeddings.shape[0],)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(norms, bins=100)
    ax.set_xlabel("Norm")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Norms of Embeddings")
    fig.savefig(ARTIFACTS_DIR / "norms_histogram.png", dpi=DPI)
    plt.close(fig)

def do_embedding_plots(config, clouds):
    color_f = lambda colors: [lang if lang in {"en", "fr", "ja", "ko", "nl", "ru", "tr"} else "other" for lang in colors]

    for work_type, topic in itertools.product(WorkType, Topic):
        cloud = clouds[(topic, work_type)]
        plot_embeddings([cloud], "language", method="pca", dimensions=2, color_f=color_f, **vars(config))
        # plot_embeddings([cloud], "language", method="pca", dimensions=3, color_f=color_f, **vars(config))
        plot_embeddings([cloud], "language", method="tsne", dimensions=2, color_f=color_f, **vars(config))

    # for work_type in WorkType:
    #     topic_clouds = [clouds[(topic, work_type)] for topic in Topic]
    #     plot_embeddings(topic_clouds, "topic", method="pca", dimensions=2, **vars(config))
    #     plot_embeddings(topic_clouds, "topic", method="pca", dimensions=3, **vars(config))
    #     plot_embeddings(topic_clouds, "topic", method="tsne", **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="pca", dimensions=2, color_f=color_f, **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="pca", dimensions=3, color_f=color_f, **vars(config))
    #     plot_embeddings(topic_clouds, "language", method="tsne", color_f=color_f, **vars(config))


def do_inference():
    # Synchronous inference for all *_articles.txt files
    inference_manager = InferenceManager()
    for articles_file in tqdm.tqdm(ARTIFACTS_DIR.glob("**/*_articles.txt"), desc="Running inference on articles files"):
        output_path = articles_file.with_name(articles_file.name.replace("_articles.txt", "_book_info.txt"))
        if output_path.exists():
            logger.info(f"Skipping {articles_file} as {output_path} already exists.")
            continue
        with open(articles_file, "r") as f:
            articles_text = f.read()
        prompt = (
            "Given the following list of research articles (titles and abstracts), "
            "generate a suitable title, abstract, and table of contents for a textbook "
            "that would prepare a student to understand the topics in these articles. "
            "Format your answer as:\n"
            "==TITLE: <title goes here>\n"
            "==ABSTRACT: <abstract goes here>\n"
            "==TABLE OF CONTENTS:\n"
            "<table of contents goes here>\n"
            "==END TABLE OF CONTENTS\n\n"
            "Provide no extraneous output, just the formatted answer.\n\n"
            f"Articles:\n\n{articles_text}\n\n"
        )
        logger.info(f"Running inference for {articles_file} -> {output_path}")
        inference_manager.run(prompt, output_path)


@dataclass(frozen=True)
class Config:
    # log_level: str
    max_works: int
    duplicate_expectation: float
    radius_fraction: float
    index_info: IndexInfo
    denoiser: Denoiser
    drop_k: int
    normalize_embeddings: bool = field(repr=False)

@TIMER.time()
def main():
    parser = argparse.ArgumentParser(description="Fetch, deduplicate, embed, and analyze OpenAlex works.")
    parser.add_argument("--drop_k", type=int, default=0, help="Number of top components to drop from embeddings")
    # Arguments for Config fields
    parser.add_argument("--log_level", type=str, default=logging._levelToName[logging.DEBUG], choices=sorted(logging._nameToLevel.keys(), key=lambda x: logging._nameToLevel[x]))
    parser.add_argument("--max_works", type=int, default=None, help="Maximum number of works to fetch")
    parser.add_argument("--duplicate_expectation", type=float, default=.05, help="Belief value in [0, 1] for expected fraction of duplicates (removed by semantic distance percentile)")
    parser.add_argument("--radius_fraction", type=float, default=0.75, help="Radius fraction for ball coverage")
    parser.add_argument("--denoiser", type=Denoiser._member_map_.__getitem__, choices=list(Denoiser), default=Denoiser.NONE, help=f"Denoiser method. Choices: {', '.join(map(str, Denoiser))}")
    
    # Arguments for IndexInfo fields (and its subtypes)
    parser.add_argument("--index_type", type=IndexType._member_map_.__getitem__, choices=list(IndexType), default=IndexType.HNSW, help=f"Faiss index type. Choices: {', '.join(map(str, IndexType))}")
    # HNSW specific arguments
    parser.add_argument("--hnsw_m", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--hnsw_efc", type=int, default=8192, help="HNSW efConstruction parameter")
    parser.add_argument("--hnsw_efs", type=int, default=1024, help="HNSW efSearch parameter")
    # parser.add_argument("--index_metric", type=IndexMetric._member_map_.__getitem__, choices=list(IndexMetric), default=IndexMetric.L2.name, help=f"Faiss index metric (used in HNSW). Choices: {', '.join(IndexMetric._member_names_)}")
    # parser.add_argument("--normalize_embeddings", action="store_true", default=False, help="Whether to normalize embeddings")

    parsed_args = parser.parse_args()


    logging_level = logging._nameToLevel[parsed_args.log_level]

    def get_explicit_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
        # Set all arguments to have unique default
        sentinel = object()
        parser.set_defaults(**{arg: sentinel for arg in vars(args)})
        # Re-parse the arguments to get explicit values
        explicit_args = vars(parser.parse_args())
        ret = [(k, v) for k, v in explicit_args.items() if v is not sentinel]
        ret = [(k, v) for k, v in ret if k not in {"log_level"}]
        return [(k, v) for k, v in explicit_args.items() if v is not sentinel]
    cmdline_args = get_explicit_args(parser, parsed_args)
            
    arg_string = create_unicode_safe_filename("_".join(f"{k}:{v}" for k, v in sorted(cmdline_args)))
    log_fname_parts = [time.strftime('%Y-%m-%d_%H-%M-%S')]
    if arg_string:
        log_fname_parts.append(arg_string)
    log_filename = Path("logs") / ("_".join(log_fname_parts) + ".log")
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure the named logger
    logger.setLevel(logging_level)
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s | %(funcName)-12s: %(message)s"))
    logger.addHandler(file_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO) # Set console level separately if needed
    console.setFormatter(logging.Formatter('%(levelname)-8s | %(funcName)-12s: %(message)s'))
    logger.addHandler(console)

    # Construct IndexInfo
    index_info_obj: IndexInfo
    if parsed_args.index_type == IndexType.HNSW:
        index_info_obj = HNSW(
            hnsw_m=parsed_args.hnsw_m,
            hnsw_efc=parsed_args.hnsw_efc,
            hnsw_efs=parsed_args.hnsw_efs,
        )
    elif parsed_args.index_type == IndexType.FLAT_L2:
        index_info_obj = FlatL2()
    else:
        raise ValueError(f"Unhandled index type: {parsed_args.index_type}")

    config = Config(
        # log_level=parsed_args.log_level,
        max_works=parsed_args.max_works,
        duplicate_expectation=parsed_args.duplicate_expectation,
        radius_fraction=parsed_args.radius_fraction,
        index_info=index_info_obj,
        denoiser=parsed_args.denoiser,
        drop_k=parsed_args.drop_k,
        # normalize_embeddings=parsed_args.normalize_embeddings
        normalize_embeddings=True,
    )

    global ARTIFACTS_DIR
    # ARTIFACTS_DIR /= create_unicode_safe_filename("_".join(f"{k}:{v}" for k, v in sorted(vars(config).items())))
    ARTIFACTS_DIR /= create_unicode_safe_filename(" ".join(f"{k}={v}" for k, v in sorted(vars(config).items())))
    logger.info(f"Artifacts directory: {ARTIFACTS_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting run with config: {config}")
    logger.info(f"Logging to: {log_filename}")

    clouds = {
        (topic, work_type): Cloud(
            topic=topic,
            work_type=work_type,
            index_info=config.index_info,
            max_works=config.max_works,
            duplicate_expectation=config.duplicate_expectation,
            normalize_embeddings=config.normalize_embeddings,
            denoiser=config.denoiser,
            drop_k=config.drop_k,
        )
        for topic, work_type in itertools.product(Topic, WorkType)
    }

    # Log the closest pairs of works in each cloud
    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Topic:
        work_type = WorkType.ARTICLE
        fname = ARTIFACTS_DIR / f"{topic}" / f"{work_type}" / "closest_pairs.txt"
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "w") as f:
            cloud =  clouds[(topic, work_type)]
            # Log the nearest neighbors in order of increasing distance
            Dnn, I = get_nearest_neighbor(cloud, cloud, 0)
            I = I[:, 1]
            sorted_indices = np.argsort(Dnn)
            # logger.info(f"Nearest neighbors in {cloud.name} ({work_type})\n")
            f.write(f"Nearest neighbors in {cloud.name} ({work_type})\n")
            for i in sorted_indices[:20]:
                selfwork, otherwork = cloud.works()[i], cloud.works()[I[i]]
                if otherwork["id"] == selfwork["id"]:
                    msg = f"Self match nearest neighbor: {selfwork['id']} == {otherwork['id']}"
                    logger.warning(msg)
                    f.write(f"WARNING: {msg}\n")
                if otherwork["id"] < selfwork["id"]:
                    continue
                # logger.info(f"Distance {i}: {Dnn[i]:.4f}\n"
                #             f"{selfwork['id']} | {selfwork['publication_date']} | {selfwork['display_name']} | {' '.join(selfwork['abstract'].split())}\n"
                #             f"{otherwork['id']} | {otherwork['publication_date']} | {otherwork['display_name']} | {' '.join(otherwork['abstract'].split())}\n")
                f.write(f"Distance {i}: {Dnn[i]:.4f}\n"
                        f"{selfwork['id']} | {selfwork['publication_date']} | {selfwork['display_name']} | {' '.join(selfwork['abstract'].split())}\n"
                        f"{otherwork['id']} | {otherwork['publication_date']} | {otherwork['display_name']} | {' '.join(otherwork['abstract'].split())}\n\n")

    for topic in Topic:
        get_points_by_coverage_ball(topic, WorkType.ARTICLE, WorkType.BOOK, config.radius_fraction, config.index_info, config.max_works, config.duplicate_expectation, config.normalize_embeddings, config.denoiser, config.drop_k)

    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Topic:
        work_type = WorkType.ARTICLE
        cloud = clouds[(topic, work_type)]
        do_avgref_analysis([cloud], cloud)

    # for work_type, topic in itertools.product(WorkType, Topic):
    for topic in Topic:
        work_type = WorkType.ARTICLE
        # Compute distribution of number of referenced_works
        cloud = clouds[(topic, work_type)]
        referenced_work_counts = np.array([len(work["referenced_works"]) for work in cloud.works()])
        logger.info(f"Number of referenced works in {topic} - {work_type}:\n{get_stats(referenced_work_counts)}")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.hist(referenced_work_counts, bins=100, color="lightblue", alpha=0.7)
        ax.set_xlabel("Number of referenced works")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{topic} - {work_type} - Number of referenced works")
        ax.set_yscale("log")
        fig_dir = ARTIFACTS_DIR / f"{topic}" / f"{work_type}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "referenced_works.png", dpi=DPI)
        plt.close(fig)


    do_embedding_plots(config, clouds)

    # do_inference()

if __name__ == "__main__":
    main()
