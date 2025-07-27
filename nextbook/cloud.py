from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import enum
from functools import lru_cache
import itertools
import json
import matplotlib.pyplot as plt
import re
import time
from typing import Any, Dict, Generator, List, Optional, Union
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
from scipy import sparse as sp
import scipy.linalg
from sentence_transformers import SentenceTransformer
import tqdm
import nextbook.denoisers as denoisers
from nextbook.index import IndexInfo, build_index, range_search_, range_search_radii_, FlatWithLims, from_flat_with_lims
from nextbook.util import ARTIFACTS_DIR, CACHE_DIR, FilterPair, Filter, WorkType, TIMER, MEMORY, LOGGER, create_unicode_safe_filename, get_stats
from nextbook.deduplicators import Deduplicator

@dataclass(frozen=True)
class CloudData:
    works: List[Dict[str, Any]]
    index: faiss.Index
    embeddings: NDArray
    def __post_init__(self):
        assert len(self.works) == self.embeddings.shape[0], f"Works and embeddings length mismatch: {len(self.works)} != {self.embeddings.shape[0]}"
        assert self.index.ntotal == len(self.works), f"Index and works length mismatch: {self.index.ntotal} != {len(self.works)}"
        assert self.index.d == self.embeddings.shape[1], f"Index and embeddings dimension mismatch: {self.index.d} != {self.embeddings.shape[1]}"

@dataclass(frozen=True)
class Cloud:
    filter_pair: FilterPair
    index_info: IndexInfo
    max_works: Optional[int]
    deduplicator: Optional[Deduplicator]
    normalize_embeddings: bool
    denoiser: Optional[str]
    drop_k: int

    def __post_init__(self):
        assert self.drop_k >= 0
        if self.denoiser is None:
            assert self.drop_k == 0, f"If denoiser is NONE, drop_k must be 0 (got {self.drop_k})"
        # else:
        #     assert self.drop_k > 0, f"If denoiser is not NONE, drop_k must be > 0 (got {self.drop_k})"

    def works(self) -> List[Dict[str, Any]]:
        return self._data().works
    
    def index(self) -> faiss.Index:
        return self._data().index
    
    def embeddings(self) -> NDArray:
        return self._data().embeddings

    def name(self) -> str:
        return f"{self.filter_pair.merged_name()}"

    def _data(self) -> CloudData:
        return get_cloud_data(self)



# Work-aware denoiser system
class WorkDenoiser(ABC):
    """Base class for work-aware denoisers that can filter works and apply denoisers."""
    @abstractmethod
    def projector(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int) -> NDArray:
        """Apply the denoiser to works and embeddings."""
        ...

class SimpleWorkDenoiser(WorkDenoiser):
    """A simple work denoiser that applies a single denoiser implementation to all works."""
    
    def __init__(self, denoiser_impl, needs_citation_graph: bool = True):
        self.denoiser_impl = denoiser_impl
        self.needs_citation_graph = needs_citation_graph
    
    def projector(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int):
        # Center
        X_mean = embeddings.mean(axis=0)
        X_centered = embeddings - X_mean
        # Compute projection matrix
        if self.needs_citation_graph:
            citation_graph = build_citation_graph(works)
            P, _eigvals = self.denoiser_impl.projector(X_centered, citation_graph, n_components)
        else:
            P, _eigvals = self.denoiser_impl.projector(X_centered, n_components)
        return P


class TwoSubsetWorkDenoiser(WorkDenoiser):
    """Work denoiser that applies a denoiser to two filtered subsets of works."""
    
    def __init__(self, denoiser_impl: denoisers.TwoSubsetScatterDenoiser, filter1: Filter, filter2: Filter):
        self.denoiser_impl = denoiser_impl
        self.filter1 = filter1
        self.filter2 = filter2
    
    def projector(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int):
        indices1 = [i for i, work in enumerate(works) if self.filter1.check_work(work)]
        indices2 = [i for i, work in enumerate(works) if self.filter2.check_work(work)]
        # Union of both groups for centering
        union_indices = sorted(set(indices1) | set(indices2))
        if not union_indices:
            return np.eye(embeddings.shape[1], dtype=embeddings.dtype)
        
        # Filtered groups
        X1 = embeddings[indices1]
        X2 = embeddings[indices2]
        # Get projection matrix (pass full embeddings set for better scatter computation)
        P, _eigvals = self.denoiser_impl.projector(X1, X2, n_components, X=embeddings)
        return P

class WorkDenoisers(enum.Enum):
    # Simple denoisers
    PCA = SimpleWorkDenoiser(denoisers.PCADenoiser(), needs_citation_graph=False)
    GPCA = SimpleWorkDenoiser(denoisers.GPCADenoiser(), needs_citation_graph=True)
    REL_AMP = SimpleWorkDenoiser(denoisers.RelAmpDenoiser(), needs_citation_graph=True)
    NEIGHBOR_DIST_MIN = SimpleWorkDenoiser(denoisers.NeighborDistMinDenoiser(), needs_citation_graph=True)
    LPP = SimpleWorkDenoiser(denoisers.LPPDenoiser(), needs_citation_graph=True)
    AUG_LPP = SimpleWorkDenoiser(denoisers.AugmentedLPPDenoiser(), needs_citation_graph=True)
    ORTHO_MANIFOLD = SimpleWorkDenoiser(denoisers.OrthogonalManifoldDenoiser(), needs_citation_graph=True)
    NORM_LPP = SimpleWorkDenoiser(denoisers.NormalizedLPPDenoiser(), needs_citation_graph=True)
    W = SimpleWorkDenoiser(denoisers.WDenoiser(), needs_citation_graph=True)
    LDA = SimpleWorkDenoiser(denoisers.LDADenoiser(), needs_citation_graph=True)
    SYM_LDA = SimpleWorkDenoiser(denoisers.SymmetrizedLDADenoiser(), needs_citation_graph=True)
    PLS = SimpleWorkDenoiser(denoisers.PLSModeADenoiser(), needs_citation_graph=True)
    CCA = SimpleWorkDenoiser(denoisers.PLSModeBDenoiser(), needs_citation_graph=True)
    # W_LPP = SimpleWorkDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.LPPDenoiser()), needs_citation_graph=True)
    # W_NEIGHBOR_DIST_MIN = SimpleWorkDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.NeighborDistMinDenoiser()), needs_citation_graph=True)
    
    # Work-aware denoisers
    BOOK_ARTICLE_ANTI_LDA = TwoSubsetWorkDenoiser(
        denoisers.AntiLDADenoiser(),
        Filter(work_type=WorkType.BOOK.value), 
        Filter(work_type=WorkType.ARTICLE.value)
    )
    BOOK_ARTICLE_DOM_INVAR = TwoSubsetWorkDenoiser(
        denoisers.DomainInvariantDenoiser(),
        Filter(work_type=WorkType.BOOK.value), 
        Filter(work_type=WorkType.ARTICLE.value)
    ) 
    
    @TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>", embeddings=lambda embeddings: f"<{embeddings.shape}>"))
    def projector(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int):
        return self.value.projector(works, embeddings, n_components)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)

@TIMER.time()
@MEMORY.cache
def load_works(filter: Filter, max_works: Optional[int]):
    expected = get_query(filter).count()
    expected = min(max_works, expected) if max_works is not None else expected
    works = sorted(tqdm.tqdm(itertools.islice(
        embed_works(filter),
        max_works),
        total=expected, disable=False), key=lambda w: w["publication_date"])
    LOGGER.info(f"Loaded {len(works)} works")
    d, = works[0]["embedding"].shape

    undeduped_embeddings = np.empty((len(works), d), dtype=np.float32)
    for i, work in enumerate(works):
        undeduped_embeddings[i] = work["embedding"]
        del work["embedding"]
    return works, undeduped_embeddings

@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>", embeddings=lambda embeddings: f"<{embeddings.shape}>"))
def get_denoised_embeddings(cloud: Cloud, works: List[Dict[str, Any]], embeddings: NDArray) -> NDArray:
    def maybe_normalize(embeddings):
        if cloud.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.where(norms == 0, 1, norms)
        return embeddings
    
    if cloud.denoiser is None:
        return maybe_normalize(embeddings)
    
    cloud_dir = ARTIFACTS_DIR / str(cloud.filter_pair.pre) / f"{cloud.denoiser} drop_k={cloud.drop_k}"
    cloud_dir.mkdir(parents=True, exist_ok=True)

    # Pre-projection PCA for baseline singular values
    s_orig: NDArray = scipy.linalg.svdvals(embeddings - embeddings.mean(axis=0)) # type: ignore
    
    LOGGER.info(f"Cloud {cloud.name()} -- Top 10 original singular values: {' '.join(f'{s:.3f}' for s in s_orig[:10])}")

    # Dimensionality reduction/cleaning
    n_components = embeddings.shape[1] - cloud.drop_k # TODO where should we make sure this is >= 1 or 0?
    LOGGER.info(f"Cloud {cloud.name()} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={n_components}) on {len(embeddings)} embeddings of shape {embeddings.shape}")
    denoiser = WorkDenoisers[cloud.denoiser]

    P = denoiser.projector(works, embeddings, n_components)
    assert np.allclose(P.T @ P, np.eye(P.shape[1]), atol=1e-6), f"Projector {denoiser} did not produce orthogonal matrix P! {np.linalg.norm(P.T @ P - np.eye(P.shape[1]))=:.6f}"
    
    denoised_embeddings = embeddings @ P

    s_post: NDArray = scipy.linalg.svdvals(denoised_embeddings - denoised_embeddings.mean(axis=0)) # type: ignore
    
    # Plot singular value comparison        
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Scatter plot for original and post-denoising singular values
    ax.scatter(np.arange(len(s_orig)), s_orig, color='b', label='Original', alpha=0.7)
    ax.scatter(np.arange(len(s_post)), s_post, color='r', label='Post-denoising', alpha=0.7)
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate variance decay
    total_variance_orig = np.sum(s_orig**2)
    total_variance_post = np.sum(s_post**2)
    variance_retained = total_variance_post / total_variance_orig
    
    fig.suptitle(f'{cloud.denoiser} drop_k={cloud.drop_k} Denoising: {variance_retained:.2%} variance retained')
    
    plt.tight_layout()
    plt.savefig(cloud_dir / f"singular_values_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Cloud {cloud.name()} -- Variance retained after denoising: {variance_retained:.2%}")
    LOGGER.info(f"Cloud {cloud.name()} -- Saved singular value analysis to {cloud_dir}")
        

    return maybe_normalize(denoised_embeddings)

@TIMER.time(formatters=dict(embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["embeddings"]) # embeddings are ignored by the cache key because hashing is too slow. Use embeddings_key to disambiguate.
def deduplicate_works_semantic_(deduplicator: Optional[Deduplicator], index_info: IndexInfo, embeddings: NDArray, embeddings_key: tuple) -> NDArray[np.intp]:
    if deduplicator is None:
        return np.arange(len(embeddings), dtype=np.intp) # Retain all indices
    return deduplicator.deduplicate(embeddings, index_info)

def deduplicate_works_semantic(cloud, works, embeddings):
    LOGGER.info(f"Cloud {cloud.name()} -- Applying deduplication to {len(embeddings)} embeddings with {cloud.deduplicator}")
    retained_idxs = deduplicate_works_semantic_(cloud.deduplicator, cloud.index_info, embeddings, embeddings_key=(cloud.filter_pair.pre, cloud.max_works))
    # Write total and per-type retention stats
    artifacts_dir = ARTIFACTS_DIR / str(cloud.filter_pair.pre)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    original_counts = defaultdict(int)
    for work in works:
        original_counts[work.get("type", "UNKNOWN")] += 1
    new_counts = defaultdict(int)
    for i in retained_idxs:
        new_counts[works[i].get("type", "UNKNOWN")] += 1
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

@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_data(cloud: Cloud) -> CloudData:
    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works)
    
    # Apply deduplication in initial space because denoisers shouldn't need
    # to consider post-projection deduplication and so that different denoisers
    # will not result in different deduplicated works.
    retained_idxs = deduplicate_works_semantic(cloud, works, embeddings)
    works = [works[i] for i in retained_idxs]
    embeddings = embeddings[retained_idxs]
    LOGGER.info(f"Cloud {cloud.name()} -- Retained {len(works)} works after deduplication")

    # Apply post-filtering
    LOGGER.info(f"Cloud {cloud.name()} -- Applying post-filtering to {len(works)} works with {cloud.filter_pair.post}")
    retained_idxs = [i for i, work in enumerate(works) if cloud.filter_pair.post.check_work(work)]
    works = [works[i] for i in retained_idxs]
    embeddings = embeddings[retained_idxs]
    LOGGER.info(f"Cloud {cloud.name()} -- Retained {len(works)} works after post-filtering")

    embeddings = get_denoised_embeddings(cloud, works, embeddings)

    # Clean up work texts
    WHITESPACE_PAT = re.compile(r"\s")
    for work in works:
        work["display_name"] = WHITESPACE_PAT.sub(" ", work["display_name"])
        work["abstract"] = WHITESPACE_PAT.sub(" ", work["abstract"])

    # Rebuild the index with the retained points
    LOGGER.info(f"Cloud {cloud.name()} -- Rebuilding index with {len(works)} points")
    index = build_index(cloud.index_info, embeddings)
    return CloudData(works=works, index=index, embeddings=embeddings)



@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>"))
def build_citation_graph(works: List[Dict[str, Any]]) -> sp.csr_matrix:
    id_to_idx = {work["id"]: i for i, work in enumerate(works)}
    n = len(works)
    row = []
    col = []
    for i, work in enumerate(works):
        # Add self as reference for safe normalization and stability
        for ref_id in work.get("referenced_works", []) + [work["id"]]:
            if ref_id in id_to_idx:
                row.append(i)
                col.append(id_to_idx[ref_id])
    citation_graph = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    row_sums = np.asarray(citation_graph.sum(axis=1)).squeeze(1)
    assert row_sums.min() >= 1 # This is guaranteed by the self-references
    # In-place row normalization for CSR matrix
    citation_graph.data /= row_sums[citation_graph.indices]
    assert type(citation_graph) is sp.csr_matrix, f"Expected csr_matrix, got {type(citation_graph)}"
    LOGGER.info(f"Built citation graph with shape {citation_graph.shape} and {citation_graph.nnz} non-zero entries")
    return citation_graph


def get_works(pre_filter: Filter) -> Generator[dict[str, Any], None, None]:
    cache_dir = CACHE_DIR / "api_data" / create_unicode_safe_filename(pre_filter)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_file = cache_dir / "state.json"
    if not state_file.exists():
        with state_file.open("w") as f:
            json.dump({"next_cursor": "*", "next_page": 1}, f)
    with state_file.open("r") as f:
        state = json.load(f)

    # First yield cached works
    for page_num in range(1, state["next_page"]):
        page_file = cache_dir / f"page_{page_num}.json"
        with page_file.open("r") as f:
            yield from json.load(f)
    
    # This isn't actually necessary, but it's clearer
    if state["next_cursor"] is None:
        return

    # Then yield new works
    last_time = time.time() # For rate limiting
    for i, page in enumerate(get_query(pre_filter).paginate(
        cursor=state["next_cursor"], per_page=200,
        n_max=None # type: ignore
        )):
        next_cursor = page.meta["next_cursor"]
        if next_cursor is None:
            with state_file.open("w") as f:
                json.dump({"next_cursor": None, "next_page": state["next_page"]}, f)
            return
        works: list[dict[str, Any]] = list(map(dict, page))
        for work in works:
            work["abstract"] = resolve_abstract_inverted_index(work["abstract_inverted_index"])
            del work["abstract_inverted_index"]
            work["id"] = work["ids"]["openalex"]
            del work["ids"]
            work["topic"] = str(pre_filter)
        # Save works
        page_file = cache_dir / f"page_{state['next_page']}.json"
        with page_file.open("w") as f:
            json.dump(works, f, indent=2)
        # Save state
        state["next_page"] += 1
        state["next_cursor"] = next_cursor
        with state_file.open("w") as f:
            json.dump(state, f)
        # Yield works
        yield from works
        # Rate limit
        min_time_diff = 0.25 # Minimum time between requests
        time.sleep(max(0, min_time_diff - (time.time() - last_time)))
        last_time = time.time()

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

def filter_works(work_filter: Filter):
    class PerTypeCounter:
        def __init__(self):
            self.total: int = 0
            self.per_type: defaultdict[str, int] = defaultdict(int)
        def insert(self, work: dict[str, Any]):
            self.total += 1
            work_type = work.get('type', 'UNKNOWN')
            self.per_type[work_type] += 1
        
    filter_stats = {f"{filter}": PerTypeCounter() for filter in WorkFilter}
    total = PerTypeCounter()
    num_removed = PerTypeCounter()
    artifacts_dir = ARTIFACTS_DIR / str(work_filter)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    with (artifacts_dir / "invalid_works.txt").open("w") as f:
        for work in tqdm.tqdm(get_works(work_filter), total=get_query(work_filter).count(),
                            desc=f"Filtering works in {work_filter}", disable=False):
            invalidators = []
            total.insert(work)
            for filter in WorkFilter:
                if filter(work):
                    filter_stats[f"{filter}"].insert(work)
                    invalidators.append(f"{filter}")
            if len(invalidators) == 0:
                yield work
            else:
                num_removed.insert(work)
                f.write(f"{' '.join(invalidators)} invalidated {work['type']} {work['id']}: {work['display_name']} -- {' '.join(work['abstract'].split())}\n")

    # Write stats to file
    with (artifacts_dir / "invalid_works_stats.json").open("w") as f:
        stats = OrderedDict(total=total.total, num_removed=num_removed.total,
                            fraction_removed=num_removed.total / total.total if total.total > 0 else 0.0,
                            **OrderedDict(sorted((work_type, v.total) for work_type, v in filter_stats.items() if v.total > 0)))
        for work_type in total.per_type.keys():
            stats[f"{work_type}_total"] = total.per_type[work_type]
            stats[f"{work_type}_removed"] = num_removed.per_type[work_type]
            stats[f"{work_type}_fraction_removed"] = num_removed.per_type[work_type] / total.per_type[work_type] if total.per_type[work_type] > 0 else 0.0
            for filter_name in filter_stats.keys():
                stats[f"{work_type}_{filter_name}"] = filter_stats[filter_name].per_type[work_type]
        json.dump(stats, f, indent=2)
        LOGGER.info(f"{work_filter} filter stats: {json.dumps(stats, indent=2)}") 
                            
@MEMORY.cache
def deduplicate_works_syntactic(work_filter: Filter):
    works = list(filter_works(work_filter))
    deduped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    def normalize_work(work: dict[str, Any], _NONWORD_PAT=re.compile(r"\W")) -> str:
        s = f"{work['display_name']} {work['abstract']}"
        s = _NONWORD_PAT.sub("", s)
        s = s.lower()
        return s
    for work in works:
        key = normalize_work(work)
        deduped[key].append(work)


    artifacts_dir = ARTIFACTS_DIR / str(work_filter)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Find cross-type duplicates
    with (artifacts_dir / f"syntactic_deduplication_crosstype_duplicates.txt").open("w") as f:
        collisions = defaultdict(int)
        for dupgroup in deduped.values():
            work_types = tuple(sorted(set(work["type"] for work in dupgroup)))
            if len(work_types) > 1:
                collisions[work_types] += 1
        for work_types, count in sorted(collisions.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{work_types}: {count}\n")
    # Split duplicate groups into subgroups by type
    key_and_type_to_dupgroup: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, dupgroup in deduped.items():
        for work in dupgroup:
            work_type = work.get("type", "UNKNOWN")
            key_and_type_to_dupgroup[(key, work_type)].append(work)

    with (artifacts_dir / f"syntactic_deduplication_stats.txt").open("w") as f:
        f.write(f"Total works: {len(works)}\n")
        f.write(f"Unique works (before splitting by type): {len(deduped)} ({len(deduped) / len(works):.2%})\n")
        f.write(f"Number of works with duplicates (before splitting by type): {sum(1 for group in deduped.values() if len(group) > 1)}\n")
        f.write(f"Distribution of len(group) for all duplicate groups with at least 2 works (before splitting by type):\n")
        f.write(f"{get_stats(np.array([len(group) for group in deduped.values() if len(group) > 1]))}\n") 
        f.write("\n")
        f.write(f"Unique works (after splitting duplicate groups by type): {len(key_and_type_to_dupgroup)} ({len(key_and_type_to_dupgroup) / len(works):.2%})\n")
        f.write(f"Number of works with duplicates (after splitting duplicate groups by type): {sum(1 for group in key_and_type_to_dupgroup.values() if len(group) > 1)}\n")
        f.write(f"Distribution of len(group) for all duplicate groups with at least 2 works (after splitting by type):\n")
        f.write(f"{get_stats(np.array([len(subdupgroup) for subdupgroup in key_and_type_to_dupgroup.values() if len(subdupgroup) > 1]))}\n")
        f.write("\n")
        
        counts_by_type = defaultdict(list)
        for (key, work_type), subdupgroup in key_and_type_to_dupgroup.items():
            counts_by_type[work_type].append(len(subdupgroup))
        f.write(f"Counts by type:\n")
        for work_type, counts in sorted(counts_by_type.items()):
            f.write(f"\t{work_type} -- total works: {sum(counts)}\n")
            f.write(f"\t{work_type} -- unique works: {len(counts)} ({len(counts) / sum(counts):.2%})\n")
            f.write(f"\t{work_type} -- distribution of len(group) for all duplicate groups with at least 2 works:\n")
            f.write(f"{get_stats(np.array([count for count in counts if count > 1]))}\n")
            f.write("\n")

    return [subdupgroup[0] for subdupgroup in key_and_type_to_dupgroup.values()] # Return the first work of each deduplicated group

def batched(iterable, n): # TODO use itertools.batched when available
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable) # If already an iterator, `iterator is iterable == True`
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

@MEMORY.cache
def embed_works(work_filter: Filter):
    LOGGER.info(f"Embedding {work_filter} works")
    works = deduplicate_works_syntactic(work_filter)
    n_works = len(works)
    works = iter(works)
    emb_cache_dir = CACHE_DIR / "embeddings" / create_unicode_safe_filename(work_filter)
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    ret: list[dict[str, Any]] = []
    # Yield cached works and embeddings
    with tqdm.tqdm(total=n_works, desc="Getting embeddings", disable=False) as pbar:
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
                    pbar.update(1)
                i += 1
            except FileNotFoundError:
                break
        if seen > 0:
            LOGGER.info(f"Yielded {seen} cached embeddings") # TODO no longer yielding
                
        # Yield new works and embeddings
        model = None # Lazy load the model in case `works` is already exhausted
        start_time = time.time()
        batch_size = 512
        seen = 0
        LOGGER.info(f"Embedding {work_filter} works")
        for batch in batched(works, batch_size):
            if model is None:
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
                pbar.update(1)
                seen += 1
            i += 1
        if seen > 0:
            LOGGER.info(f"Yielded {seen} new embeddings in {time.time() - start_time:.2f} seconds")

    return ret

@dataclass(frozen=True)
class SubCloud:
    parent: Cloud
    post_filter: Filter

    def works(self) -> List[Dict[str, Any]]:
        return self._data().works
    
    def index(self) -> faiss.Index:
        return self._data().index
    
    def embeddings(self) -> NDArray:
        return self._data().embeddings

    @property
    def filter_pair(self) -> FilterPair:
        return self.parent.filter_pair

    def _data(self) -> CloudData:
        return get_subcloud_data(self)

@lru_cache(maxsize=2)
@TIMER.time()
def get_subcloud_data(subcloud: SubCloud) -> CloudData:
    works = subcloud.parent.works()
    embeddings = subcloud.parent.embeddings()
    # Apply post-filtering
    retained_idxs = [i for i, work in enumerate(works) if subcloud.post_filter.check_work(work)]
    filtered_works = [works[i] for i in retained_idxs]
    filtered_embeddings = embeddings[retained_idxs]
    index = build_index(subcloud.parent.index_info, filtered_embeddings)
    return CloudData(works=filtered_works, index=index, embeddings=filtered_embeddings)


def get_query(work_filter: Filter) -> pyalex.Works:
    # .sort(publication_date="asc", display_name="asc") 
    return pyalex.Works() \
            .filter(has_abstract=True,
                    # language="en",
                    **work_filter.to_dict()) \
            .select(", ".join(["display_name", "type", "publication_date", "ids", "language", "abstract_inverted_index", "referenced_works", "primary_topic"]))

def resolve_abstract_inverted_index(abstract_inverted_index: Dict[str, List[int]]) -> str:
    pos_to_word = {}
    for word, positions in abstract_inverted_index.items():
        assert " " not in word, f"Word contains space: '{word}'"
        for pos in positions:
            assert pos not in pos_to_word, f"Duplicate position {pos} for word {word}"
            pos_to_word[pos] = word
    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "[[MISSING]]") for i in range(max_pos + 1)]
    return " ".join(words)


@dataclass
class GetBallCoverageResult:
    index_works_per_query_work: NDArray
    ball_counts: NDArray  # Number of index works within the ball for each query work
    covered_points: list[tuple[NDArray[np.float32], NDArray[np.intp]]]  # List of (dists, ids) tuples

@TIMER.time()
@MEMORY.cache
def get_ball_coverage(index_subcloud: SubCloud, query_subcloud: SubCloud, radius_fraction: float) -> GetBallCoverageResult:
    """Get ball coverage statistics between two SubClouds."""
    # Both must be SubClouds of the same parent
    assert isinstance(index_subcloud, SubCloud) and isinstance(query_subcloud, SubCloud), "Both arguments must be SubClouds"
    assert index_subcloud.parent is query_subcloud.parent, "SubClouds must share the same parent Cloud"
    LOGGER.info(f"Finding closest query work to each index work (each query work gets a voronoi cell of index works)")

    # For each index_subcloud item, find the nearest query_subcloud item
    res = faiss_search_outside_epsilon(query_subcloud, index_subcloud, is_intra_cloud_search=False, k=1, epsilon=0)
    D = res.D_outside.squeeze(1)
    I = res.I_outside.squeeze(1)
    if query_subcloud.index().ntotal == 0:
        index_works_per_query_work = np.array([], dtype=np.intp)
    else:
        index_works_per_query_work = np.bincount(I, minlength=query_subcloud.index().ntotal)
    title = f"Number of index works covered by each query work (radius fraction: {radius_fraction})"
    LOGGER.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")

    # Now, for each article, count how many other articles are within a radius defined by its distance to the nearest book.
    D_scaled = D * radius_fraction
    res = range_search_radii_(index_subcloud.index(), index_subcloud.embeddings(), D_scaled)
    assert res.lims is not None and res.dists is not None and res.ids is not None
    ball_counts = res.lims[1:] - res.lims[:-1]
    LOGGER.info(f"Number of index works covered by each index work (ball):\n{get_stats(ball_counts)}")
    dists_list = from_flat_with_lims(FlatWithLims(res.dists, res.lims))
    ids_list = from_flat_with_lims(FlatWithLims(res.ids, res.lims))
    covered_points = list(zip(dists_list, ids_list))
    return GetBallCoverageResult(
        index_works_per_query_work=index_works_per_query_work,
        ball_counts=ball_counts,
        covered_points=covered_points)

@dataclass(frozen=True)
class FaissSearchOutsideEpsilonResult:
    D_outside: NDArray
    I_outside: NDArray
    lims_within: NDArray
    dists_within: NDArray
    ids_within: NDArray

@MEMORY.cache
def faiss_search_outside_epsilon(
    index: Union[Cloud, SubCloud],
    query: Union[Cloud, SubCloud],
    is_intra_cloud_search: bool,
    k: int,
    epsilon: float,
) -> FaissSearchOutsideEpsilonResult:
    """
    Find the k nearest neighbors of query that are not (strictly) within epsilon of the query.
    """
    query_embeddings = query.embeddings()
    if epsilon == 0:
        LOGGER.info("Epsilon is 0, using k-NN search instead of range search")
        start_time = time.time()
        
        if is_intra_cloud_search:
            # For intra-cloud, we need k+1 neighbors to discard the point itself.
            num_to_find = k + 1
            D_raw, I_raw = index.index().search(query_embeddings, num_to_find) # type: ignore
            
            # Check that the first neighbor is the point itself.
            # TODO may need to allow for some tolerance.
            if query.index().ntotal > 0:
                dist_to_self = D_raw[:, 0]
                max_dist_to_self = np.max(dist_to_self)
                atol = 1e-5
                if not np.isclose(max_dist_to_self, 0, atol=atol):
                    LOGGER.warning(f"Max distance to self was not close to 0: {max_dist_to_self:.6f}")
            
            # Check that distances are non-decreasing.
            assert np.all(np.diff(D_raw, axis=1) >= 0)

            # The results "outside epsilon" are the columns from index 1 onwards.
            D_outside = D_raw[:, 1:]
            I_outside = I_raw[:, 1:]

        else:
            # For inter-cloud search, a standard k-NN search is correct.
            D_outside, I_outside = index.index().search(query_embeddings, k) # type: ignore

        LOGGER.info(f"k-NN search took {time.time() - start_time:.2f} seconds")
        return FaissSearchOutsideEpsilonResult(
            D_outside=D_outside,
            I_outside=I_outside,
            lims_within=np.zeros((len(query_embeddings) + 1,), dtype=np.uint64),
            dists_within=np.empty((0,), dtype=np.float32),
            ids_within=np.empty((0,), dtype=np.int64))

    # Range search for all points strictly within epsilon
    LOGGER.info(f"Range searching for all points strictly within epsilon")
    start_time = time.time()
    range_search_result = range_search_(index.index(), query.embeddings(), epsilon)
    lims_within, dists_within, ids_within = range_search_result.lims, range_search_result.dists, range_search_result.ids
    assert lims_within is not None
    assert dists_within is not None
    assert ids_within is not None
    LOGGER.info(f"Range searching took {time.time() - start_time:.2f} seconds")
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

@dataclass(frozen=True)
class CrossCloudNNRanksResult:
    ranks: NDArray
    dists: Optional[NDArray]
    ids: Optional[NDArray]

    def __post_init__(self):
        assert self.ranks.ndim == 1
        if self.dists is not None:
            assert self.dists.ndim == 1
            assert len(self.dists) == len(self.ranks)
        if self.ids is not None:
            assert self.ids.ndim == 1
            assert len(self.ids) == len(self.ranks)

@TIMER.time()
@MEMORY.cache
def within_cloud_nn_ranks(
        ambient_subcloud: SubCloud,
        query_subcloud: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1):
    # Find nearest neighbor in query_subcloud for each query_subcloud point
    nns = faiss_search_outside_epsilon(query_subcloud, query_subcloud, is_intra_cloud_search=True, k=k, epsilon=0)
    D = nns.D_outside[:, -1]
    I = nns.I_outside[:, -1]
    
    # Sanity check: calculate squared distance to each such nearest neighbor
    query_embeddings = query_subcloud.embeddings()
    num_query_points = query_subcloud.index().ntotal
    if num_query_points > 0:
        sq_dists = np.linalg.norm(query_embeddings[I] - query_embeddings, axis=1) ** 2
        assert len(D) == len(I) == len(sq_dists) == num_query_points
        assert np.allclose(D, sq_dists)
    
    balls = range_search_radii_(
        ambient_subcloud.index(),
        query_embeddings,
        D,
        return_lims=True,
        return_dists=return_dists,
        return_ids=return_ids
    )
    assert balls.lims is not None
    ranks = balls.lims[1:] - balls.lims[:-1]
    assert len(ranks) == num_query_points, f"Ranks length mismatch: {len(ranks)} != {num_query_points}"
    assert np.all(ranks >= 0), f"Negative ranks found: {ranks[ranks < 0]}"
    return CrossCloudNNRanksResult(
        ranks=ranks,
        dists=balls.dists,
        ids=balls.ids
    )


@TIMER.time()
@MEMORY.cache
def cross_cloud_nn_ranks(
        ambient_subcloud: SubCloud,
        query_subcloud: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False,
        k: int = 1):
    # Find nearest neighbor in index_subcloud for each query_subcloud point
    nns = faiss_search_outside_epsilon(ambient_subcloud, query_subcloud, is_intra_cloud_search=False, k=k, epsilon=0)
    D = nns.D_outside[:, -1]
    I = nns.I_outside[:, -1]
    # Calculate squared distance to each such nearest neighbor
    if ambient_subcloud.index().ntotal > 0:
        sq_dists = np.linalg.norm(ambient_subcloud.embeddings()[I] - query_subcloud.embeddings(), axis=1) ** 2
        assert len(D) == len(I) == len(sq_dists) == query_subcloud.index().ntotal
        assert np.allclose(D, sq_dists)
    balls = range_search_radii_(
        query_subcloud.index(),
        query_subcloud.embeddings(),
        D,
        return_lims=True,
        return_dists=return_dists,
        return_ids=return_ids
    )
    assert balls.lims is not None
    ranks = balls.lims[1:] - balls.lims[:-1]
    assert len(ranks) == query_subcloud.index().ntotal, f"Ranks length mismatch: {len(ranks)} != {query_subcloud.index().ntotal}"
    assert np.all(ranks >= 0), f"Negative ranks found: {ranks[ranks < 0]}"
    return CrossCloudNNRanksResult(
        ranks=ranks,
        dists=balls.dists,
        ids=balls.ids
    )
