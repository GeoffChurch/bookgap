from collections import defaultdict
from dataclasses import asdict, dataclass
import enum
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import os
import shutil
from pathlib import Path
from bookgap.cloud_denoiser import ProjectiveDenoiserProjection, WorkDenoisers
import re
from typing import Any, Dict, List, Optional, TypeVar
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
import tqdm
from bookgap.embedder import BaseEmbedder, Embedder, SentenceTransformerEmbedder
from bookgap.index import IndexInfo, build_index
import scipy.sparse as sp
from bookgap.util import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    FilterPair,
    Filter,
    TIMER,
    MEMORY,
    LOGGER,
    build_citation_adjacency_from_ids,
)
from bookgap.work_store import WorksStore
from bookgap.deduplicators import Deduplicator
import scipy.linalg

from bookgap.dendrogram import DendrogramResult, get_cloud_dendrogram

from bookgap.openalex_ingest import (
    _resolve_filter_scopes,
    embed_works,
    fetch_works,
)

pyalex.config.email = "gchurchill@cs.stonybrook.edu"
pyalex.config.api_key = os.environ["OPENALEX_API_KEY"]

@dataclass(frozen=True)
class Cloud:
    filter_pair: FilterPair
    index_info: IndexInfo
    max_works: Optional[int]
    deduplicator: Optional[Deduplicator]
    normalize_embeddings: bool
    denoiser: str
    drop_k: int
    embedder: Any  # An embedder config dataclass (SentenceTransformerEmbedderConfig or TfGepEmbedderConfig)

    def __post_init__(self):
        from bookgap.configs.embedder.embedder_config import (
            SentenceTransformerEmbedderConfig,
            TfGepEmbedderConfig,
        )
        assert self.drop_k >= 0
        # Validate denoiser is a valid WorkDenoiser name
        valid_denoisers = [d.name for d in WorkDenoisers]
        if self.denoiser not in valid_denoisers:
            raise ValueError(f"denoiser must be one of {valid_denoisers}, got '{self.denoiser}'")
        # Normalize embedder to a config dataclass for hashability/caching.
        # Accept: legacy Embedder.ModelName enum, config dataclass, or
        # Hydra-instantiated embedder instance.
        embedder = self.embedder
        if isinstance(embedder, Embedder.ModelName):
            embedder = SentenceTransformerEmbedderConfig(model_name=embedder.value)
        elif isinstance(embedder, SentenceTransformerEmbedder):
            embedder = SentenceTransformerEmbedderConfig(model_name=embedder.model_name)
        else:
            try:
                from bookgap.tf_gep_embedder import TfGepEmbedder
                if isinstance(embedder, TfGepEmbedder):
                    embedder = TfGepEmbedderConfig(
                        n_components=embedder._n_components,
                        gep_type=embedder._gep_type,
                        ridge_fraction=embedder._ridge_fraction,
                        min_df=embedder._min_df,
                        max_df=embedder._max_df,
                        add_self_loops=embedder._add_self_loops,
                        normalize_adjacency=embedder._normalize_adjacency,
                    )
            except ImportError:
                pass
        if embedder is not self.embedder:
            object.__setattr__(self, "embedder", embedder)
        if not isinstance(self.embedder, (SentenceTransformerEmbedderConfig, TfGepEmbedderConfig)):
            raise TypeError(
                f"embedder must be a SentenceTransformerEmbedderConfig or TfGepEmbedderConfig, "
                f"got {type(self.embedder).__name__}"
            )

    def works(self) -> List[Dict[str, Any]]:
        return get_cloud_works(self)

    def work_ids(self) -> List[str]:
        return get_cloud_work_ids(self)
    
    def work_references(self) -> List[List[str]]:
        return get_cloud_work_references(self)
    
    def work_titles(self) -> List[str]:
        return get_cloud_work_titles(self)
    
    def work_abstracts(self) -> List[str]:
        return get_cloud_work_abstracts(self)
    
    def work_publication_dates(self) -> List[str]:
        return get_cloud_work_publication_dates(self)
    
    def index(self) -> faiss.Index:
        return get_cloud_index(self)
    
    def embeddings(self):
        return get_cloud_embeddings(self)

    def citation_graph(self, normalize: bool) -> sp.csr_matrix:
        return get_cloud_citation_graph(self, normalize=normalize)

    def denoiser_projection(self) -> ProjectiveDenoiserProjection:
        return get_cloud_projection(self)

    def dendrogram(
        self,
        linkage_method: str,
        *,
        min_cluster_size: int,
        max_depth: Optional[int],
    ) -> DendrogramResult:
        return get_cloud_dendrogram(
            self,
            linkage_method=linkage_method,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
        )

    def name(self) -> str:
        return f"{self.filter_pair.merged_name()}"

def _works_store_path(filter: Filter, max_works: Optional[int], embedder_cfg: Any) -> tuple[Path, str]:
    payload = {
        "filter": str(filter),
        "max_works": max_works,
        "embedder": asdict(embedder_cfg),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    store_dir = CACHE_DIR / "arrow" / "works" / fingerprint
    # Return directory path for partitioned dataset (not .parquet file)
    return store_dir, fingerprint


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_retained_idxs(cloud: Cloud) -> NDArray[np.uint64]:
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    embeddings = store.embeddings()

    retained_idxs = deduplicate_works_semantic(cloud, store, embeddings)
    LOGGER.info(f"Retained {len(retained_idxs)} works after deduplication")

    mask = _post_filter_mask(store, cloud.filter_pair.post, retained_idxs)
    retained_idxs = retained_idxs[mask]
    LOGGER.info(f"Retained {len(retained_idxs)} works after post-filtering")

    return retained_idxs


@lru_cache(maxsize=1)
@TIMER.time()
def get_cloud_works(cloud: Cloud) -> List[Dict[str, Any]]:
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_dicts(indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_ids(cloud: Cloud) -> List[str]:
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_list("id", indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_references(cloud: Cloud) -> List[List[str]]:
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_list("referenced_works", indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_citation_graph(cloud: Cloud, normalize: bool) -> sp.csr_matrix:
    """Return the citation adjacency/transition matrix for the cloud."""
    work_ids = get_cloud_work_ids(cloud)
    work_references = get_cloud_work_references(cloud)
    adj, _ = build_citation_adjacency_from_ids(work_ids, work_references, add_self_loops=True, normalize=normalize)
    LOGGER.info(f"Cloud {cloud.name()} citation graph: shape={adj.shape}, nnz={adj.nnz}")
    return adj


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_titles(cloud: Cloud) -> List[str]:
    """Get display names/titles for all works in the cloud."""
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_list("display_name", indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_abstracts(cloud: Cloud) -> List[str]:
    """Get abstracts for all works in the cloud.
    
    The abstract is already resolved from inverted index format during load_works.
    """
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_list("abstract", indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_publication_dates(cloud: Cloud) -> List[str]:
    """Get publication dates for all works in the cloud."""
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return store.to_list("publication_date", indices=retained_idxs.tolist())


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_index(cloud: Cloud) -> faiss.Index:
    embeddings = get_cloud_embeddings(cloud)
    return build_index(cloud.index_info, embeddings)


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_embeddings(cloud: Cloud):
    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return denoise(cloud, store.embeddings(indices=retained_idxs.tolist()))


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_projection(cloud: Cloud) -> ProjectiveDenoiserProjection:
    denoiser_enum = WorkDenoisers[cloud.denoiser]
    denoiser_impl = denoiser_enum.value

    store = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder)
    retained_idxs = get_cloud_retained_idxs(cloud)

    retained_works = store.to_dicts(indices=retained_idxs.tolist())
    retained_embeddings = store.embeddings(indices=retained_idxs.tolist())

    return denoiser_impl.compute_projection_artifact(cloud.drop_k, retained_works, retained_embeddings)


@lru_cache(maxsize=1)
@TIMER.time()
def load_works(filter: Filter, max_works: Optional[int], embedder_cfg: Any) -> WorksStore:
    """Load (or build) a WorksStore for *filter*, embedding via *embedder_cfg*.

    *embedder_cfg* is a config dataclass -- either
    ``SentenceTransformerEmbedderConfig`` (pretrained, per-scope batch cache)
    or ``TfGepEmbedderConfig`` (corpus-fitted, needs all works first).
    """
    store_path, fingerprint = _works_store_path(filter, max_works, embedder_cfg)

    if store_path.exists():
        try:
            store = WorksStore.load(store_path, expected_fingerprint=fingerprint)
            if "type" not in store.schema.names:
                LOGGER.warning(f"Cached works store at {store_path} missing 'type' column; rebuilding")
                shutil.rmtree(store_path, ignore_errors=True)
            else:
                LOGGER.info(f"Reusing cached works store at {store_path}")
                return store
        except Exception as e:  # pragma: no cover - defensive
            LOGGER.warning(f"Failed to load cached works store at {store_path}: {e}")

    if filter.work_types is None:
        raise ValueError("filter.work_types must be provided")

    from bookgap.configs.embedder.embedder_config import (
        SentenceTransformerEmbedderConfig,
        TfGepEmbedderConfig,
    )
    # -- Branch on embedder type --
    if isinstance(embedder_cfg, SentenceTransformerEmbedderConfig):
        works, undeduped_embeddings = _load_works_sentence_transformer(filter, max_works, embedder_cfg)
    elif isinstance(embedder_cfg, TfGepEmbedderConfig):
        works, undeduped_embeddings = _load_works_tf_gep(filter, embedder_cfg)
    else:
        raise TypeError(f"Unsupported embedder config type: {type(embedder_cfg).__name__}")

    # Drop works without a publication date to ensure sortable ordering
    valid_idxs = [i for i, work in enumerate(works) if work.get("publication_date")]
    if len(valid_idxs) != len(works):
        dropped = len(works) - len(valid_idxs)
        LOGGER.info(f"Dropped {dropped} works missing publication_date for {filter}")
        works = [works[i] for i in valid_idxs]
        undeduped_embeddings = undeduped_embeddings[valid_idxs]

    if max_works is None:
        argsort = np.argsort([work["publication_date"] for work in works])
    else:
        argsort = np.array(heapq.nsmallest(max_works, range(len(works)), key=lambda i: works[i]["publication_date"]))
    works = [works[i] for i in argsort]
    undeduped_embeddings = undeduped_embeddings[argsort]

    LOGGER.info(f"Loaded {len(works)} works")
    n, d = undeduped_embeddings.shape
    assert n == len(works), f"Embeddings and works length mismatch: {n} != {len(works)}"

    WHITESPACE_PAT = re.compile(r"\s")
    for work in works:
        work["display_name"] = WHITESPACE_PAT.sub(" ", work["display_name"])
        work["abstract"] = WHITESPACE_PAT.sub(" ", work["abstract"])

    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = WorksStore.write(
        store_path,
        fingerprint=fingerprint,
        works=works,
        embeddings=undeduped_embeddings,
    )
    LOGGER.info(f"Wrote works store to {store_path}")
    return store


def _load_works_sentence_transformer(
    filter: Filter,
    max_works: Optional[int],
    embedder_cfg: Any,
) -> tuple[list[dict[str, Any]], NDArray]:
    """Sentence-transformer path: per-scope batch caching via embed_works."""
    if max_works is not None:
        raise ValueError("max_works is not supported with per-scope embedding cache; pass None")

    model_name = Embedder.ModelName(embedder_cfg.model_name)

    works: list[dict[str, Any]] = []
    emb_chunks: list[NDArray] = []

    with tqdm.tqdm(_resolve_filter_scopes(filter), desc="Scopes", delay=3, mininterval=3) as scope_bar:
        for scope in scope_bar:
            scope_bar.set_postfix_str(str(scope))
            scope_works, scope_embeddings = embed_works(scope, model_name)
            works.extend(scope_works)
            emb_chunks.append(scope_embeddings)

    if emb_chunks:
        undeduped_embeddings = np.concatenate(emb_chunks, axis=0)
    else:
        undeduped_embeddings = np.empty((0, Embedder(model_name=model_name).embedding_dimension()))

    return works, undeduped_embeddings


def _load_works_tf_gep(
    filter: Filter,
    embedder_cfg: Any,
) -> tuple[list[dict[str, Any]], NDArray]:
    """TfGepEmbedder path: collect all works, build graph, fit, transform."""
    from bookgap.tf_gep_embedder import TfGepEmbedder

    works: list[dict[str, Any]] = []

    with tqdm.tqdm(_resolve_filter_scopes(filter), desc="Scopes", delay=3, mininterval=3) as scope_bar:
        for scope in scope_bar:
            scope_bar.set_postfix_str(str(scope))
            scope_works = fetch_works(scope).to_table().to_pylist()
            works.extend(scope_works)

    if not works:
        return works, np.empty((0, embedder_cfg.n_components), dtype=np.float32)

    # Build citation graph
    work_ids = [w["id"] for w in works]
    work_refs = [w.get("referenced_works", []) for w in works]
    citation_graph, _ = build_citation_adjacency_from_ids(
        work_ids, work_refs,
        add_self_loops=embedder_cfg.add_self_loops,
        normalize=embedder_cfg.normalize_adjacency,
    )

    # Instantiate, fit, and transform
    embedder = TfGepEmbedder(
        n_components=embedder_cfg.n_components,
        gep_type=embedder_cfg.gep_type,
        ridge_fraction=embedder_cfg.ridge_fraction,
        min_df=embedder_cfg.min_df,
        max_df=embedder_cfg.max_df,
        add_self_loops=embedder_cfg.add_self_loops,
        normalize_adjacency=embedder_cfg.normalize_adjacency,
    )
    embedder.fit(works, citation_graph=citation_graph)

    title_abstract_pairs = [(w["display_name"], w["abstract"]) for w in works]
    undeduped_embeddings = embedder.transform(title_abstract_pairs)

    return works, undeduped_embeddings

@TIMER.time(formatters=dict(embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["embeddings"]) # embeddings are ignored by the cache key because hashing is too slow. Use embeddings_key to disambiguate.
def deduplicate_works_semantic_(deduplicator: Optional[Deduplicator], index_info: IndexInfo, embeddings: NDArray, embeddings_key: tuple) -> NDArray[np.uint64]:
    if deduplicator is None:
        return np.arange(len(embeddings), dtype=np.uint64) # Retain all indices
    return deduplicator.deduplicate(embeddings, index_info)

def deduplicate_works_semantic(cloud: Cloud, store: WorksStore, embeddings):
    LOGGER.info(f"Cloud {cloud.name()} -- Applying deduplication to {len(embeddings)} embeddings with {cloud.deduplicator}")
    retained_idxs = deduplicate_works_semantic_(
        cloud.deduplicator,
        cloud.index_info,
        embeddings,
        embeddings_key=(cloud.filter_pair.pre, cloud.max_works, cloud.embedder),
    )

    artifacts_dir = ARTIFACTS_DIR / str(cloud.filter_pair.pre)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    work_types = store.to_list("type")
    original_counts = defaultdict(int)
    for wt in work_types:
        original_counts[wt or "UNKNOWN"] += 1
    new_counts = defaultdict(int)
    for idx in retained_idxs:
        wt = work_types[int(idx)] if int(idx) < len(work_types) else "UNKNOWN"
        new_counts[wt or "UNKNOWN"] += 1

    fname = f"deduplication_stats_{cloud.deduplicator}"
    if cloud.max_works is not None:
        fname += f"_maxworks={cloud.max_works}"
    path = artifacts_dir / f"{fname}.txt"
    with path.open("w") as f:
        total_original = sum(original_counts.values())
        total_new = sum(new_counts.values())
        f.write(f"Total works before deduplication: {total_original}\n")
        f.write(f"Total works after deduplication: {total_new}\n")
        f.write(f"Fraction of works retained: {total_new / total_original:.4f}\n")
        for work_type in sorted(original_counts.keys()):
            original_count = original_counts[work_type]
            new_count = new_counts[work_type]
            f.write(f"{work_type}: {original_count} -> {new_count} ({new_count / original_count:.4f})\n")
    with path.open("r") as f:
        LOGGER.info(f"Deduplication stats:\n{f.read()}")
    return retained_idxs


def _post_filter_mask(store: WorksStore, post_filter: Filter, retained_idxs: NDArray[np.uint64]) -> NDArray[np.bool_]:
    if not post_filter:
        return np.ones(len(retained_idxs), dtype=bool)
    types = store.to_list("type", indices=retained_idxs.tolist())
    topic_ids = store.to_list("primary_topic_id", indices=retained_idxs.tolist())
    subfield_ids = store.to_list("primary_subfield_id", indices=retained_idxs.tolist())

    mask = []
    for work_type, topic_id, subfield_id in zip(types, topic_ids, subfield_ids):
        work = {
            "type": work_type,
            "primary_topic": {"id": topic_id, "subfield": {"id": subfield_id}},
        }
        mask.append(post_filter.check_work(work))
    return np.array(mask, dtype=bool)

class RegexFilter:
    def __init__(self, regex: str, flags: int = re.IGNORECASE):
        self.regex = re.compile(regex, flags)
    def __call__(self, work: dict[str, Any]) -> bool:
        return self.regex.search(f"{work['display_name']} {work['abstract']}") is not None
    def __repr__(self):
        return f"Contains {self.regex}"

class MissingFieldFilter:
    def __init__(self, field: str):
        self.field = field
    def __call__(self, work: dict[str, Any]) -> bool:
        return work.get(self.field) is None
    def __repr__(self):
        return f"Missing {self.field}"

class ShortFieldFilter:
    def __init__(self, field: str, min_length: int):
        self.field = field
        self.min_length = min_length
    def __call__(self, work: dict[str, Any]) -> bool:
        field = work.get(self.field)
        return field is not None and len(field) < self.min_length
    def __repr__(self):
        return f"len({self.field}) < {self.min_length}"

class WorkFilter(enum.Enum):
    BINDER_NAME_CANCEL_CREATE_EXPORT      = RegexFilter(r"BinderNameCancelCreateExport")
    NO_ABSTRACT_AVAILABLE                 = RegexFilter(r"No abstract available")
    ABSTRACT_NOT_AVAILABLE                = RegexFilter(r"An abstract is not available")
    SEARCH_FOR_OTHER_WORKS_BY_THIS_AUTHOR = RegexFilter(r"Search for other works by this author")
    RUSSIAN_TEXT_IGNORED                  = RegexFilter(r"Russian Text Ignored")
    MARITAL_ARMS                          = RegexFilter(r"Judging from repetitious appearances of her marital arms")
    NO_DISPLAY_NAME                       = MissingFieldFilter("display_name")
    NO_ABSTRACT                           = MissingFieldFilter("abstract")
    SHORT_DISPLAY_NAME                    = ShortFieldFilter("display_name", min_length=4)
    SHORT_ABSTRACT                        = ShortFieldFilter("abstract", min_length=10)
    # NULL_REFERENCED_WORKS                 = MissingFieldFilter("referenced_works")
    # ZERO_REFERENCED_WORKS                 = ShortFieldFilter("referenced_works", min_length=1)
    
    def __call__(self, work: dict[str, Any]) -> bool:
        return self.value(work)

    def __str__(self):
        return repr(self.value)

def batched(iterable, n): # TODO use itertools.batched when available
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable) # If already an iterator, `iterator is iterable == True`
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

T = TypeVar('T', bound=np.generic)

def denoise(cloud: Cloud, embeddings: NDArray[T]) -> NDArray[T]:
    """Apply projective denoising with full orchestration."""
    # Pre-projection PCA for baseline singular values
    s_orig: NDArray = scipy.linalg.svdvals(embeddings - embeddings.mean(axis=0))  # type: ignore
    
    LOGGER.info(f"Top 10 original singular values: {' '.join(f'{s:.3f}' for s in s_orig[:10])}")

    # Dimensionality reduction/cleaning

    projection = get_cloud_projection(cloud)

    LOGGER.info(f"Cloud {cloud.name()} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={embeddings.shape[1] - cloud.drop_k}) on {len(embeddings)} embeddings of shape {embeddings.shape}")

    denoised_embeddings: NDArray[T] = projection.apply(embeddings)

    if cloud.normalize_embeddings:
        norms = np.linalg.norm(denoised_embeddings, axis=1, keepdims=True)
        denoised_embeddings = denoised_embeddings / norms

    s_post: NDArray = scipy.linalg.svdvals(denoised_embeddings - denoised_embeddings.mean(axis=0))  # type: ignore
    

    
    # Calculate variance decay
    total_variance_orig = np.sum(s_orig**2)
    total_variance_post = np.sum(s_post**2)
    variance_retained = total_variance_post / total_variance_orig
    
    LOGGER.info(f"Cloud {cloud.name()} -- Variance retained after denoising: {variance_retained:.2%}")
        
    return denoised_embeddings