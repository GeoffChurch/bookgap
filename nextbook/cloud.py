from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import enum
from functools import lru_cache
import itertools
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Generator, List, Optional, Union
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
from scipy import sparse as sp
from sentence_transformers import SentenceTransformer
import tqdm
import nextbook.denoisers as denoisers
from nextbook.index import IndexInfo, build_index, range_search_, range_search_radii_, FlatWithLims, from_flat_with_lims
from nextbook.util import CACHE_DIR, FilterPair, Filter, WorkType, TIMER, MEMORY, LOGGER, create_unicode_safe_filename, get_stats
@dataclass(frozen=True)
class Cloud:
    filter_pair: FilterPair
    index_info: IndexInfo
    max_works: Optional[int]
    duplicate_expectation: float
    normalize_embeddings: bool
    denoiser: Optional[str]
    drop_k: int

    def __post_init__(self):
        assert self.drop_k >= 0
        assert 0 <= self.duplicate_expectation <= 1, f"duplicate_expectation must be in [0, 1], got {self.duplicate_expectation}"
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


# Work-aware denoiser system
class WorkDenoiser:
    """Base class for work-aware denoisers that can filter works and apply denoisers."""
    
    def __call__(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        """Apply the denoiser to works and embeddings."""
        raise NotImplementedError


class SimpleWorkDenoiser(WorkDenoiser):
    """A simple work denoiser that applies a single denoiser implementation to all works."""
    
    def __init__(self, denoiser_impl, needs_citation_graph: bool = True):
        self.denoiser_impl = denoiser_impl
        self.needs_citation_graph = needs_citation_graph
    
    def __call__(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        # Center
        X_mean = embeddings.mean(axis=0)
        X_centered = embeddings - X_mean
        # Compute projection matrix
        if self.needs_citation_graph:
            citation_graph = build_citation_graph(works)
            P, extra = self.denoiser_impl.projector(X_centered, citation_graph, n_components)
        else:
            P, extra = self.denoiser_impl.projector(X_centered, n_components)
        # Project and decenter
        X_denoised = X_centered @ P
        X_denoised = X_denoised + X_mean
        return X_denoised, extra


class AntiLDAWorkDenoiser(WorkDenoiser):
    """Work denoiser that applies Anti-LDA to remove discriminative directions between two work types."""
    
    def __init__(self, filter1: Filter, filter2: Filter):
        self.filter1 = filter1
        self.filter2 = filter2
    
    def __call__(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        indices1 = [i for i, work in enumerate(works) if self.filter1.check_work(work)]
        indices2 = [i for i, work in enumerate(works) if self.filter2.check_work(work)]
        # Union of both groups for centering
        union_indices = sorted(set(indices1) | set(indices2))
        if not union_indices:
            # If union is empty, just return original embeddings
            return embeddings, None
        X_union = embeddings[union_indices]
        X_mean = X_union.mean(axis=0)
        # Center all embeddings
        X_centered = embeddings - X_mean
        # Centered filtered groups
        X1_centered = X_centered[indices1]
        X2_centered = X_centered[indices2]
        # Get projection matrix
        from nextbook.denoisers import AntiLDADenoiser
        anti_lda = AntiLDADenoiser()
        P, eigvals = anti_lda.projector(X1_centered, X2_centered, n_components)
        # Project all centered embeddings
        X_denoised = X_centered @ P
        return X_denoised, eigvals

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
    W_LPP = SimpleWorkDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.LPPDenoiser()), needs_citation_graph=True)
    W_NEIGHBOR_DIST_MIN = SimpleWorkDenoiser(denoisers.EmbeddedGraphDenoiser(denoisers.NeighborDistMinDenoiser()), needs_citation_graph=True)
    
    # Work-aware denoisers
    BOOK_ARTICLE_ANTI_LDA = AntiLDAWorkDenoiser(
        Filter(work_type=WorkType.BOOK.value), 
        Filter(work_type=WorkType.ARTICLE.value)
    )
    
    def __call__(self, works: List[Dict[str, Any]], embeddings: NDArray, n_components: int) -> tuple[NDArray, Optional[NDArray]]:
        return self.value(works, embeddings, n_components)
    
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
    if cloud.denoiser is not None:
        # cloud_dir = ARTIFACTS_DIR / str(cloud.topic) / str(cloud.work_type)
        # cloud_dir.mkdir(parents=True, exist_ok=True)

        # Dimensionality reduction/cleaning
        n_components = embeddings.shape[1] - cloud.drop_k
        LOGGER.info(f"Cloud {cloud.name()} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={n_components}) on {len(embeddings)} embeddings of shape {embeddings.shape}")
        denoiser = WorkDenoisers[cloud.denoiser]
        embeddings, eigvals = denoiser(works, embeddings, n_components)
        # if len(eigvals) > 0:
        #     plot_eigenvalues(
        #         eigvals,
        #         n_components,
        #         cloud_dir / f"{cloud.denoiser.lower()}_eigenvalues.png",
        #         f"Eigenvalues for {cloud.denoiser} on {cloud.name()} embeddings"
        #     )

    return maybe_normalize(embeddings)

@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>", undeduped_embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["works", "undeduped_embeddings"])
def deduplicate_works(cloud: Cloud, works: List[Dict[str, Any]], undeduped_embeddings: NDArray):
    # --- Adaptive min_dist based on duplicate_expectation ---
    duplicate_expectation = cloud.duplicate_expectation
    num_to_drop = int(duplicate_expectation * (len(undeduped_embeddings) - 1))
    LOGGER.info(f"Cloud {cloud.name()} -- Using duplicate_expectation={duplicate_expectation}")

    if num_to_drop > 0 and len(undeduped_embeddings) > 1:
        # Compute distances to nearest older neighbor
        dist_to_nearest_older = []
        # Build index incrementally, but in batches for efficiency
        main_index = cloud.index_info.index_type.build(undeduped_embeddings[:1], cloud.index_info) # Start with oldest work
        batch_size = 8192
        
        desc = f"Finding nearest older neighbors for {cloud.name()}"
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

    LOGGER.info(f"Cloud {cloud.name()} -- Retaining {len(retained_indices)} points out of {len(undeduped_embeddings)} based on duplicate expectation")
    retained_works = [works[i] for i in retained_indices]
    retained_embeddings = undeduped_embeddings[retained_indices]
    return retained_works, retained_embeddings

@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_data(cloud: Cloud) -> Cloud.CloudData:
    works, undeduped_embeddings = load_works(cloud.filter_pair.pre, cloud.max_works)
    
    # Apply post-filtering
    LOGGER.info(f"Cloud {cloud.name()} -- Applying post-filtering to {len(works)} works with {cloud.filter_pair.post}")
    retained_idxs = [i for i, work in enumerate(works) if cloud.filter_pair.post.check_work(work)]
    LOGGER.info(f"Cloud {cloud.name()} -- Retained {len(retained_idxs)} works after post-filtering")
    works = [works[i] for i in retained_idxs]
    undeduped_embeddings = undeduped_embeddings[retained_idxs]

    undeduped_embeddings = get_denoised_embeddings(cloud, works, undeduped_embeddings)

    # Deduplicate works based on the duplicate expectation
    retained_works, deduped_embeddings = deduplicate_works(cloud, works, undeduped_embeddings)

    # Clean up work texts
    for work in retained_works:
        work["display_name"] = work["display_name"].replace("\n", "\\n")
        work["abstract"] = work["abstract"].replace("\n", "\\n")

    # Rebuild the index with the retained points
    LOGGER.info(f"Cloud {cloud.name()} -- Rebuilding index with {len(works)} points")
    index = build_index(cloud.index_info, deduped_embeddings)
    return Cloud.CloudData(works=retained_works, index=index, embeddings=deduped_embeddings)


@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>"))
def build_citation_graph(works: List[Dict[str, Any]]) -> sp.csr_matrix:
    id_to_idx = {work["id"]: i for i, work in enumerate(works)}
    n = len(works)
    row = []
    col = []
    for i, work in enumerate(works):
        # Add self as reference for safe normalization and stability
        for ref_id in work["referenced_works"] + [work["id"]]:
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
    for page in range(1, state["next_page"]):
        page_file = cache_dir / f"page_{page}.json"
        with page_file.open("r") as f:
            yield from json.load(f)
    
    # This isn't actually necessary, but it's clearer
    if state["next_cursor"] is None:
        return

    # Then yield new works
    last_time = time.time() # For rate limiting
    for i, page in enumerate(get_query(pre_filter).paginate(
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
            work["topic"] = str(pre_filter)
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
        min_time_diff = 0.25 # Minimum time between requests
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

def filter_works(work_filter: Filter):
    stats: Dict[str, Union[int, float]]
    stats = {f"{filter}": 0 for filter in WorkFilter}
    stats["total"] = 0
    stats["num_removed"] = 0
    # invalid_dir = cache_dir / "filter" / str(topic) / str(work_type)
    # invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # with (invalid_dir / "invalid_works.txt").open("w") as f:
    for work in tqdm.tqdm(get_works(work_filter), total=get_query(work_filter).count(),
                          desc=f"Filtering works in {work_filter}", disable=False):
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
    LOGGER.info(f"{work_filter} filter stats: {json.dumps(stats, indent=2, sort_keys=True)}")

@MEMORY.cache
def deduplicate_works_syntactic(work_filter: Filter):
    works = list(filter_works(work_filter))
    deduped: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
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
    LOGGER.info(f"{work_filter} deduplication stats: {json.dumps({'min': counts.min().item(), 'max': counts.max().item(), 'mean': counts.mean().item(), 'std': counts.std().item(), 'total': len(works)}, indent=2)}")
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
def embed_works(work_filter: Filter):
    LOGGER.info(f"Embedding {work_filter} works")
    works = iter(deduplicate_works_syntactic(work_filter))
    emb_cache_dir = CACHE_DIR / "embeddings" / create_unicode_safe_filename(work_filter)
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    ret: list[dict[str, Any]] = []
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
        LOGGER.info(f"Yielded {seen} cached embeddings") # TODO no longer yielding
            
    # Yield new works and embeddings
    model = None # Lazy load the model in case `works` is already exhausted
    start_time = time.time()
    batch_size = 512
    seen = 0
    LOGGER.info(f"Embedding {work_filter} works")
    for batch in batched(tqdm.tqdm(works, desc="Embedding works", disable=False), batch_size):
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
            seen += 1
        i += 1
    if seen > 0:
        LOGGER.info(f"Yielded {seen} new embeddings in {time.time() - start_time:.2f} seconds")

    return ret

@dataclass(frozen=True)
class SubCloud:
    parent: Cloud
    post_filter: Filter

    @dataclass(frozen=True)
    class SubCloudData:
        works: List[Dict[str, Any]]
        index: faiss.Index
        embeddings: NDArray
        def __post_init__(self):
            assert len(self.works) == self.embeddings.shape[0], f"Works and embeddings length mismatch: {len(self.works)} != {self.embeddings.shape[0]}"
            assert self.index.ntotal == len(self.works), f"Index and works length mismatch: {self.index.ntotal} != {len(self.works)}"
            assert self.index.d == self.embeddings.shape[1], f"Index and embeddings dimension mismatch: {self.index.d} != {self.embeddings.shape[1]}"

    def works(self) -> List[Dict[str, Any]]:
        return self._data().works
    
    def index(self) -> faiss.Index:
        return self._data().index
    
    def embeddings(self) -> NDArray:
        return self._data().embeddings

    @property
    def filter_pair(self) -> FilterPair:
        return self.parent.filter_pair

    def _data(self) -> "SubCloud.SubCloudData":
        return get_subcloud_data(self)

@lru_cache(maxsize=2)
@TIMER.time()
@MEMORY.cache
def get_subcloud_data(subcloud: SubCloud) -> SubCloud.SubCloudData:
    parent_data = subcloud.parent._data()
    works = parent_data.works
    embeddings = parent_data.embeddings
    # Apply post-filtering
    retained_idxs = [i for i, work in enumerate(works) if subcloud.post_filter.check_work(work)]
    filtered_works = [works[i] for i in retained_idxs]
    filtered_embeddings = embeddings[retained_idxs]
    index = build_index(subcloud.parent.index_info, filtered_embeddings)
    return SubCloud.SubCloudData(works=filtered_works, index=index, embeddings=filtered_embeddings)


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
    covered_points: list[tuple[NDArray, NDArray]]  # List of (dists, ids) tuples

@TIMER.time()
@MEMORY.cache
def get_ball_coverage(index_subcloud: 'SubCloud', query_subcloud: 'SubCloud', radius_fraction: float) -> GetBallCoverageResult:
    """Get ball coverage statistics between two SubClouds."""
    # Both must be SubClouds of the same parent
    assert isinstance(index_subcloud, SubCloud) and isinstance(query_subcloud, SubCloud), "Both arguments must be SubClouds"
    assert index_subcloud.parent is query_subcloud.parent, "SubClouds must share the same parent Cloud"
    topic = index_subcloud.parent.filter_pair
    LOGGER.info(f"{topic} -- Finding closest {query_subcloud} to each {index_subcloud} (each {query_subcloud} gets a voronoi cell of {index_subcloud}s)")

    # For each index_subcloud item, find the nearest query_subcloud item
    res = faiss_search_outside_epsilon(query_subcloud, index_subcloud, k=1, epsilon=0)
    D = res.D_outside.squeeze(1)
    I = res.I_outside.squeeze(1)
    if query_subcloud.index().ntotal == 0:
        index_works_per_query_work = np.array([], dtype=np.intp)
    else:
        index_works_per_query_work = np.bincount(I, minlength=query_subcloud.index().ntotal)
    title = f"{topic} - Number of {index_subcloud}s covered by each {query_subcloud} (radius fraction: {radius_fraction})"
    LOGGER.info(f"{title} (voronoi):\n{get_stats(index_works_per_query_work)}")

    # Now, for each article, count how many other articles are within a radius defined by its distance to the nearest book.
    D_scaled = D * radius_fraction
    res = range_search_radii_(index_subcloud.index(), index_subcloud.embeddings(), D_scaled)
    assert res.lims is not None and res.dists is not None and res.ids is not None
    ball_counts = res.lims[1:] - res.lims[:-1]
    LOGGER.info(f"Number of {index_subcloud}s covered by each {index_subcloud} (ball):\n{get_stats(ball_counts)}")
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
        k: int,
        epsilon: float) -> FaissSearchOutsideEpsilonResult:
    """
    Find the k nearest neighbors of query that are not (strictly) within epsilon of the query.
    """
    query_embeddings = query.embeddings()
    if epsilon == 0:
        LOGGER.info("Epsilon is 0, using k nearest neighbors search instead of range search")
        start_time = time.time()
        D, I = index.index().search(query_embeddings, k) # type: ignore
        LOGGER.info(f"{k}-NN search took {time.time() - start_time:.2f} seconds")
        if k == 2:
            Dself, Dnn = D.T
            for i, idx in enumerate(I[:, 0]):
                dself, dnn = D[i]
                if idx != i and not np.allclose(dself, dnn):
                    LOGGER.warning(f"Index mismatch ({dself=} {dnn=}): {idx} != {i}")
            assert np.all(Dself <= Dnn), f"Distance to self is greater than distance to nearest neighbor at {np.count_nonzero(Dself > Dnn)} points"
            argmax_dist_to_self = np.argmax(Dself)
            max_dist_to_self = Dself[argmax_dist_to_self]
            atol = 1e-5
            if not np.isclose(max_dist_to_self, 0, atol=atol):
                LOGGER.warning(f"{np.sum(~np.isclose(Dself, 0, atol=atol))} points have distance to self not close to 0.")
        return FaissSearchOutsideEpsilonResult(
            D_outside=D,
            I_outside=I,
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
def cross_cloud_nn_ranks(
        ambient_subcloud: SubCloud,
        query_subcloud: SubCloud,
        return_dists: bool = False,
        return_ids: bool = False):
    # Find nearest neighbor in index_subcloud for each query_subcloud point
    nns = faiss_search_outside_epsilon(ambient_subcloud, query_subcloud, k=1, epsilon=0)
    D = nns.D_outside.squeeze(1)
    I = nns.I_outside.squeeze(1)
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
    