from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import enum
from functools import lru_cache
import itertools
import json
import matplotlib.pyplot as plt
from nextbook.plot_utils import PlotManager
import re
import time
from typing import Any, Dict, Generator, List, Optional, Protocol, TypeVar, runtime_checkable
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
from scipy import sparse as sp
import scipy.linalg
from sentence_transformers import SentenceTransformer
import tqdm
import nextbook.denoisers as denoisers
from nextbook.index import IndexInfo, build_index, HNSW, IndexType
from nextbook.util import ARTIFACTS_DIR, CACHE_DIR, FilterPair, Filter, SubField, WorkType, TIMER, MEMORY, LOGGER, create_unicode_safe_filename, get_stats
from nextbook.deduplicators import Deduplicator

T = TypeVar('T', bound=np.generic)

def get_embedding_model() -> SentenceTransformer:
    """Get a fresh instance of the embedding model used throughout the system."""
    return SentenceTransformer('avsolatorio/GIST-small-Embedding-v0')

def compute_embeddings_for_works(works: List[Dict[str, Any]], model: Optional[SentenceTransformer] = None) -> NDArray:
    """
    Compute embeddings for a list of works.
    
    Args:
        works: List of work dictionaries containing 'display_name' and 'abstract' fields
        model: Optional SentenceTransformer model. If None, a new model will be created.
        
    Returns:
        NDArray of embeddings with shape (len(works), embedding_dim)
    """
    if not works:
        # Return empty array with correct shape for the model
        if model is None:
            model = get_embedding_model()
        # Get embedding dimension by encoding a dummy text
        dummy_embedding = model.encode(["dummy"], show_progress_bar=False)
        return np.empty((0, dummy_embedding.shape[1]), dtype=np.float32)
    
    if model is None:
        model = get_embedding_model()
    
    # Create input texts from display_name and abstract
    inputs = [f"{work['display_name']}\n{work['abstract']}" for work in works]
    
    # Compute embeddings
    embeddings = model.encode(inputs, batch_size=512, show_progress_bar=False)
    
    return embeddings

@dataclass(frozen=True)
class CloudData:
    works: List[Dict[str, Any]]
    embeddings: NDArray
    def __post_init__(self):
        assert len(self.works) == self.embeddings.shape[0], f"Works and embeddings length mismatch: {len(self.works)} != {self.embeddings.shape[0]}"

@dataclass(frozen=True)
class Cloud:
    filter_pair: FilterPair
    index_info: IndexInfo
    max_works: Optional[int]
    deduplicator: Optional[Deduplicator]
    normalize_embeddings: bool
    denoiser: str
    drop_k: int

    def __post_init__(self):
        assert self.drop_k >= 0
        # Validate denoiser is a valid WorkDenoiser name
        valid_denoisers = [d.name for d in WorkDenoisers]
        if self.denoiser not in valid_denoisers:
            raise ValueError(f"denoiser must be one of {valid_denoisers}, got '{self.denoiser}'")

    def works(self) -> List[Dict[str, Any]]:
        return get_cloud_works(self)

    def work_ids(self) -> List[str]:
        return get_cloud_work_ids(self)
    
    def work_references(self) -> List[List[str]]:
        return get_cloud_work_references(self)
    
    def index(self) -> faiss.Index:
        return get_cloud_index(self)
    
    def embeddings(self):
        return get_cloud_embeddings(self)

    def name(self) -> str:
        return f"{self.filter_pair.merged_name()}"

@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_retained_idxs(cloud: Cloud) -> NDArray[np.uint64]:
    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works)
    
    # Apply deduplication in initial space because denoisers shouldn't need
    # to consider post-projection deduplication and so that different denoisers
    # will not result in different deduplicated works.
    retained_idxs = deduplicate_works_semantic(cloud, works, embeddings)
    LOGGER.info(f"Retained {len(retained_idxs)} works after deduplication")

    # Apply post-filtering
    LOGGER.info(f"Applying post-filtering to {len(retained_idxs)} works with {cloud.filter_pair.post}")
    mask = [cloud.filter_pair.post.check_work(works[i]) for i in retained_idxs]
    retained_idxs = retained_idxs[mask]
    LOGGER.info(f"Retained {len(retained_idxs)} works after post-filtering")

    return retained_idxs

@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_works(cloud: Cloud) -> List[Dict[str, Any]]:
    works, _ = load_works(cloud.filter_pair.pre, cloud.max_works)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return [works[i] for i in retained_idxs]

@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_ids(cloud: Cloud) -> List[str]:
    return [work["id"] for work in get_cloud_works(cloud)]

@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_references(cloud: Cloud) -> List[List[str]]:
    return [work.get("referenced_works", []).copy() for work in get_cloud_works(cloud)]

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
    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works)
    retained_idxs = get_cloud_retained_idxs(cloud)
    denoiser = WorkDenoisers[cloud.denoiser]
    return denoiser.value.denoise(cloud, [works[i] for i in retained_idxs], embeddings[retained_idxs])

class CloudDenoiserContext(denoisers.DenoiserContext):
    def __init__(self, works: List[Dict[str, Any]], embeddings: NDArray, *filters: Filter):
        self.works = works
        self.filters = list(filters)
        self._A: Optional[sp.csr_matrix] = None
        self._X: NDArray = embeddings
        self._X_subsets: List[Optional[NDArray]] = [None] * len(self.filters)

    def A(self) -> sp.csr_matrix:
        if self._A is None:
            self._A = build_citation_graph(self.works)
        return self._A
    
    def X(self) -> NDArray:
        return self._X
    
    def X_n(self, n: int) -> NDArray:
        """Get embeddings for the n-th subset (0-indexed)."""
        if n >= len(self.filters):
            raise IndexError(f"Subset {n} requested, but only {len(self.filters)} subsets available")
        
        if self._X_subsets[n] is None:
            self._X_subsets[n] = self._X[self._get_mask(self.filters[n])]
        
        result = self._X_subsets[n]
        assert result is not None, f"Subset {n} should not be None after initialization"
        return result
    
    def num_subsets(self) -> int:
        """Return the number of subsets available."""
        return len(self.filters)
    
    def _get_mask(self, filter: Filter) -> NDArray:
        """Get a boolean mask for works that match the filter."""
        return np.array([filter.check_work(work) for work in self.works], dtype=bool)
    
@runtime_checkable
class WorkDenoiser(Protocol):
    """Protocol for denoising embeddings."""
    def denoise(self, cloud: 'Cloud', works: List[Dict[str, Any]], embeddings: NDArray[T]) -> NDArray[T]:
        """Apply denoising to embeddings and return final denoised embeddings."""
        ...

class NoneDenoiser:
    """No-op denoiser that returns original embeddings unchanged."""
    
    def denoise(self, cloud: 'Cloud', works: List[Dict[str, Any]], embeddings: NDArray[T]) -> NDArray[T]:
        if cloud.drop_k != 0:
            raise ValueError(f"NoneDenoiser requires drop_k=0, got drop_k={cloud.drop_k}")
        
        LOGGER.info(f"NONE denoiser: returning original embeddings unchanged")
        return embeddings

class ProjectiveDenoiser:
    """Projective denoiser with full orchestration including plotting and analysis."""
    
    def __init__(self, denoiser_impl, *filters: Filter):
        self.denoiser_impl = denoiser_impl
        self.filters = filters

    def denoise(self, cloud: 'Cloud', works: List[Dict[str, Any]], embeddings: NDArray[T]) -> NDArray[T]:
        """Apply projective denoising with full orchestration."""
        cloud_dir = ARTIFACTS_DIR / str(cloud.filter_pair.pre) / f"{cloud.denoiser} drop_k={cloud.drop_k}"
        cloud_dir.mkdir(parents=True, exist_ok=True)

        # Pre-projection PCA for baseline singular values
        s_orig: NDArray = scipy.linalg.svdvals(embeddings - embeddings.mean(axis=0))  # type: ignore
        
        LOGGER.info(f"Cloud {cloud.name()} -- Top 10 original singular values: {' '.join(f'{s:.3f}' for s in s_orig[:10])}")

        # Dimensionality reduction/cleaning
        n_components = embeddings.shape[1] - cloud.drop_k
        LOGGER.info(f"Cloud {cloud.name()} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={n_components}) on {len(embeddings)} embeddings of shape {embeddings.shape}")
        
        # Get projector from the denoiser implementation
        context = CloudDenoiserContext(works, embeddings, *self.filters)
        P, _eigvals = self.denoiser_impl.projector(context, n_components)
        
        # Apply projection
        denoised_embeddings: NDArray[T] = embeddings @ P

        s_post: NDArray = scipy.linalg.svdvals(denoised_embeddings - denoised_embeddings.mean(axis=0))  # type: ignore
        
        # Plot singular value comparison        
        fig, ax = PlotManager.create_figure(figsize=(12, 5))
        
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
        PlotManager.save_and_close(fig, cloud_dir / f"singular_values_comparison.png")
        
        LOGGER.info(f"Cloud {cloud.name()} -- Variance retained after denoising: {variance_retained:.2%}")
        LOGGER.info(f"Cloud {cloud.name()} -- Saved singular value analysis to {cloud_dir}")
            
        return denoised_embeddings

class WorkDenoisers(enum.Enum):
    # No-op denoiser
    NONE = NoneDenoiser()
    
    # Simple denoisers
    PCA = ProjectiveDenoiser(denoisers.PCADenoiser())
    GPCA = ProjectiveDenoiser(denoisers.GPCADenoiser())
    REL_AMP = ProjectiveDenoiser(denoisers.RelAmpDenoiser())
    # NEIGHBOR_DIST_MIN = ProjectiveDenoiser(denoisers.NeighborDistMinDenoiser())
    LPP = ProjectiveDenoiser(denoisers.LPPDenoiser())
    AUG_LPP = ProjectiveDenoiser(denoisers.AugmentedLPPDenoiser())
    ORTHO_MANIFOLD = ProjectiveDenoiser(denoisers.OrthogonalManifoldDenoiser())
    NORM_LPP = ProjectiveDenoiser(denoisers.NormalizedLPPDenoiser())
    # W = ProjectiveDenoiser(denoisers.WDenoiser())
    # LDA = ProjectiveDenoiser(denoisers.LDADenoiser())
    # SYM_LDA = ProjectiveDenoiser(denoisers.SymmetrizedLDADenoiser())
    PLS = ProjectiveDenoiser(denoisers.PLSModeADenoiser())
    CCA = ProjectiveDenoiser(denoisers.PLSModeBDenoiser())
    # W_LPP = ProjectiveDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.LPPDenoiser()))
    # W_NEIGHBOR_DIST_MIN = ProjectiveDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.NeighborDistMinDenoiser()))
    
    ANTI_CSP = ProjectiveDenoiser(denoisers.AntiCSPDenoiser())

    ANTI_LDA = ProjectiveDenoiser(denoisers.AntiLDADenoiser(), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    DOM_INVAR = ProjectiveDenoiser(denoisers.DomainInvariantDenoiser(), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,))) 
    
    CCLIP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.5), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCLIP_1 = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.1), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCLIP_01 = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0.01), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    CCP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    LIP = ProjectiveDenoiser(denoisers.CCLIPDenoiser(alpha=1), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    
    # Joint Approximate Diagonalization denoisers
    # JAD_1_4 = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, wsubset1=0.25, wsubset2=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Optimal 1:4 book:article ratio
    # JAD_EQUAL = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, wsubset1=1.0, wsubset2=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Pre-sample-covariance optimal
    # JAD_PRIO_T = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, wsubset1=0.5, wsubset2=0.5), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))  # Equal emphasis
    JAD_EQUAL = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_HALF_L = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.5, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_CCP = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=0.0, w_conflate=1.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))
    JAD_LIP = ProjectiveDenoiser(denoisers.JADDenoiser(wt=1.0, wl=1.0, w_conflate=0.0), Filter(work_types=(WorkType.BOOK.value,)), Filter(work_types=(WorkType.ARTICLE.value,)))

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)

@lru_cache(maxsize=1)
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

    # Clean up work texts
    WHITESPACE_PAT = re.compile(r"\s")
    for work in works:
        work["display_name"] = WHITESPACE_PAT.sub(" ", work["display_name"])
        work["abstract"] = WHITESPACE_PAT.sub(" ", work["abstract"])

    return works, undeduped_embeddings

@TIMER.time(formatters=dict(embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["embeddings"]) # embeddings are ignored by the cache key because hashing is too slow. Use embeddings_key to disambiguate.
def deduplicate_works_semantic_(deduplicator: Optional[Deduplicator], index_info: IndexInfo, embeddings: NDArray, embeddings_key: tuple) -> NDArray[np.uint64]:
    if deduplicator is None:
        return np.arange(len(embeddings), dtype=np.uint64) # Retain all indices
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


@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>"))
def build_citation_graph(works: List[Dict[str, Any]]) -> sp.csr_matrix:
    id_to_idx = {work["id"]: i for i, work in enumerate(works)}
    n = len(works)
    row = []
    col = []
    for i, work in enumerate(works):
        # Add self-reference for safe normalization and stability. Don't use append on the original list!
        refs: list[str] = work.get("referenced_works", []) + [work["id"]]  
        for ref_id in refs:
            if ref_id in id_to_idx:
                row.append(i)
                col.append(id_to_idx[ref_id])
    data = np.ones(len(row), dtype=np.float32)
    citation_graph = sp.csr_matrix((data, (row, col)), shape=(n, n))
    row_sums = np.asarray(citation_graph.sum(axis=1)).squeeze(1)
    assert row_sums.min() >= 1 # This is guaranteed by the self-references
    
    # In-place row normalization
    assert citation_graph.data.dtype == np.float32, f"Expected float32 data type, got {citation_graph.data.dtype}"
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
                            desc=f"Filtering works in {work_filter}", mininterval=1, delay=1, disable=False):
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
    with tqdm.tqdm(desc="Loading cached embeddings", mininterval=1, delay=1) as cached_pbar:
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
                    cached_pbar.update(1)
                i += 1
            except FileNotFoundError:
                break
        if seen > 0:
            LOGGER.info(f"Yielded {seen} cached embeddings")
                
    # Yield new works and embeddings with their own progress bar
    new_works_count = n_works - seen
    with tqdm.tqdm(total=new_works_count, desc="Generating new embeddings", mininterval=1, delay=1) as new_pbar:
        model = None # Lazy load the model in case `works` is already exhausted
        start_time = time.time()
        batch_size = 512
        LOGGER.info(f"Embedding {work_filter} works")
        seen_new = 0
        for batch in batched(works, batch_size):
            if model is None:
                model = get_embedding_model()
            # Embed batch using the new function
            embeddings = compute_embeddings_for_works(list(batch), model)
            # Save batch
            with (emb_cache_dir / f"batch_{i}_embeddings.npy").open("wb") as embeddings_f, (emb_cache_dir / f"batch_{i}_ids.json").open("w") as ids_f:
                np.save(embeddings_f, embeddings)
                json.dump([work["id"] for work in batch], ids_f)
            # Yield works and embeddings
            for work, embedding in zip(batch, embeddings):
                work["embedding"] = embedding
                # yield work
                ret.append(work)
                new_pbar.update(1)
                seen_new += 1
            i += 1
        if seen_new > 0:
            LOGGER.info(f"Yielded {seen_new} new embeddings in {time.time() - start_time:.2f} seconds")

    return ret

@dataclass(frozen=True)
class SubCloud:
    parent: Cloud
    post_filter: Filter

    def works(self) -> List[Dict[str, Any]]:
        parent_works = self.parent.works()
        retained_idxs = get_subcloud_retained_idxs(self)
        return [parent_works[i] for i in retained_idxs]        
    
    def work_ids(self) -> List[str]:
        return get_subcloud_work_ids(self)
    
    def work_references(self) -> List[List[str]]:
        return get_subcloud_work_references(self)
    
    def index(self) -> faiss.Index:
        return get_subcloud_index(self)
    
    def embeddings(self):
        return get_subcloud_embeddings(self)
    
    @property
    def normalize_embeddings(self) -> bool:
        return self.parent.normalize_embeddings

# @lru_cache(maxsize=2)
# @TIMER.time()
# def get_subcloud_data(subcloud: SubCloud) -> CloudData:
#     works = subcloud.parent.works()
#     # Apply post-filtering
#     retained_idxs = [i for i, work in enumerate(works) if subcloud.post_filter.check_work(work)]
#     filtered_embeddings = subcloud.parent.embeddings()[retained_idxs]
#     index = build_index(subcloud.parent.index_info, filtered_embeddings)
#     return CloudData(works=[works[i] for i in retained_idxs], index=index, embeddings=filtered_embeddings)

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_retained_idxs(subcloud: SubCloud) -> NDArray[np.uint64]:
    return np.array([i for i, work in enumerate(subcloud.parent.works()) if subcloud.post_filter.check_work(work)], dtype=np.uint64)

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_work_ids(subcloud: SubCloud) -> List[str]:
    parent_work_ids = subcloud.parent.work_ids()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return [parent_work_ids[i] for i in retained_idxs]

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_work_references(subcloud: SubCloud) -> List[List[str]]:
    parent_work_references = subcloud.parent.work_references()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return [parent_work_references[i] for i in retained_idxs]

@lru_cache(maxsize=2)
@TIMER.time()
@MEMORY.cache
def get_subcloud_index(subcloud: SubCloud) -> faiss.Index:
    return build_index(subcloud.parent.index_info, get_subcloud_embeddings(subcloud))

@lru_cache(maxsize=2)
@MEMORY.cache
def get_subcloud_embeddings(subcloud: SubCloud):
    parent_embeddings = subcloud.parent.embeddings()
    retained_idxs = get_subcloud_retained_idxs(subcloud)
    return parent_embeddings[retained_idxs]


def get_query(work_filter: Filter) -> pyalex.Works:
    params = work_filter.to_dict()
    type_param = params.pop("type", None)

    # Build the query in one shot:
    # - If multiple types, use filter_or with all params so lists join with '|'
    # - If single type or no type, use standard AND filter
    if isinstance(type_param, list):
        if len(type_param) > 1:
            q = pyalex.Works().filter_or(has_abstract=True, type=type_param, **params)
        elif len(type_param) == 1:
            params["type"] = type_param[0]
            q = pyalex.Works().filter(has_abstract=True, **params)
        else:
            q = pyalex.Works().filter(has_abstract=True, **params)
    else:
        q = pyalex.Works().filter(has_abstract=True, **params)

    LOGGER.info(f"Query url: '{q.url}'")
    return q.select(
        ", ".join([
            "display_name", "type", "publication_date", "ids", "language", "abstract_inverted_index", "referenced_works", "primary_topic"
        ])
    )

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

def build_cloud(
    drop_k: int,
    max_works: Optional[int],
    deduplicator: Optional[Deduplicator],
    denoiser: str,
    index_type: IndexType,
    hnsw_m: int,
    hnsw_efc: int,
    hnsw_efs: int
) -> Cloud:
    """Build and return a Cloud object with the given parameters."""
    if index_type == IndexType.HNSW:
        index_info = HNSW(hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_efs=hnsw_efs)
    else:
        index_info = index_type.value()
    ai_filter_pair = FilterPair(
        Filter(subfield=SubField.AI.value, work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value)),
        Filter()
    )
    return Cloud(
        filter_pair=ai_filter_pair,
        index_info=index_info,
        max_works=max_works,
        deduplicator=deduplicator,
        normalize_embeddings=True,
        denoiser=denoiser,
        drop_k=drop_k
    )


def get_subclouds_for_topic(cloud: Cloud, topic_id: Optional[str]) -> tuple:
    """Return (books, articles) SubClouds for the given topic_id."""
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(WorkType.BOOK.value,)))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=(WorkType.ARTICLE.value,)))
    return books, articles
