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
from nextbook.denoisers import Denoiser
from nextbook.index import IndexInfo, build_index
from nextbook.util import CACHE_DIR, Subjects, Subject, WorkType, TIMER, MEMORY, LOGGER, create_unicode_safe_filename


@dataclass(frozen=True)
class Cloud:
    name: str = field(init=False) # TODO maybe delete
    topic: Subjects
    work_type: WorkType
    index_info: IndexInfo
    max_works: int
    duplicate_expectation: float
    normalize_embeddings: bool
    denoiser: str
    drop_k: int

    def __post_init__(self):
        object.__setattr__(self, "name", f"{self.topic} {self.work_type}")
        assert self.drop_k >= 0
        assert 0 <= self.duplicate_expectation <= 1, f"duplicate_expectation must be in [0, 1], got {self.duplicate_expectation}"
        if Denoiser[self.denoiser] is Denoiser.NONE:
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

@TIMER.time()
@MEMORY.cache
def load_works(work_type: WorkType, filter: Subject, max_works: Optional[int]):
    expected = get_query(work_type, filter).count()
    expected = min(max_works, expected) if max_works is not None else expected
    works = sorted(tqdm.tqdm(itertools.islice(
        embed_works(work_type, filter),
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
    LOGGER.info(f"Getting cloud data for {cloud.name}")
    # cloud_dir = ARTIFACTS_DIR / str(cloud.topic) / str(cloud.work_type)
    # cloud_dir.mkdir(parents=True, exist_ok=True)

    # Dimensionality reduction/cleaning
    n_components = embeddings.shape[1] - cloud.drop_k
    LOGGER.info(f"Cloud {cloud.name} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={n_components}) on {len(embeddings)} embeddings of shape {embeddings.shape}")
    denoiser = Denoiser[cloud.denoiser]
    embeddings, eigvals = denoiser(
        embeddings,
        build_citation_graph(works) if denoiser.needs_citation_graph() else None,
        n_components)
    # if len(eigvals) > 0:
    #     plot_eigenvalues(
    #         eigvals,
    #         n_components,
    #         cloud_dir / f"{cloud.denoiser.lower()}_eigenvalues.png",
    #         f"Eigenvalues for {cloud.denoiser} on {cloud.name} embeddings"
    #     )

    return maybe_normalize(embeddings)

@TIMER.time(formatters=dict(works=lambda works: f"<{len(works)} works>", undeduped_embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["works", "undeduped_embeddings"])
def deduplicate_works(cloud: Cloud, works: List[Dict[str, Any]], undeduped_embeddings: NDArray):
    # --- Adaptive min_dist based on duplicate_expectation ---
    duplicate_expectation = cloud.duplicate_expectation
    num_to_drop = int(duplicate_expectation * (len(undeduped_embeddings) - 1))
    LOGGER.info(f"Cloud {cloud.name} -- Using duplicate_expectation={duplicate_expectation}")

    if num_to_drop > 0 and len(undeduped_embeddings) > 1:
        # Compute distances to nearest older neighbor
        dist_to_nearest_older = []
        # Build index incrementally, but in batches for efficiency
        main_index = cloud.index_info.index_type.build(undeduped_embeddings[:1], cloud.index_info) # Start with oldest work
        batch_size = 8192
        
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

    LOGGER.info(f"Cloud {cloud.name} -- Retaining {len(retained_indices)} points out of {len(undeduped_embeddings)} based on duplicate expectation")
    retained_works = [works[i] for i in retained_indices]
    retained_embeddings = undeduped_embeddings[retained_indices]
    return retained_works, retained_embeddings

@lru_cache(maxsize=16)
@TIMER.time()
@MEMORY.cache
def get_cloud_data(cloud: Cloud) -> Cloud.CloudData:
    works, undeduped_embeddings = load_works(cloud.work_type, cloud.topic.pre_filter, cloud.max_works)
    
    # Apply post-filtering
    LOGGER.info(f"Cloud {cloud.name} -- Applying post-filtering to {len(works)} works with {cloud.topic.post_filter}")
    retained_idxs = [i for i, work in enumerate(works) if cloud.topic.post_filter.check_work(work)]
    LOGGER.info(f"Cloud {cloud.name} -- Retained {len(retained_idxs)} works after post-filtering")
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
    LOGGER.info(f"Cloud {cloud.name} -- Rebuilding index with {len(works)} points")
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


def get_works(work_type: WorkType, pre_filter: Subject, cache_dir: Path) -> Generator[dict[str, Any], None, None]:
    cache_dir = cache_dir / "api_data" / create_unicode_safe_filename(pre_filter) / create_unicode_safe_filename(work_type)
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
    for i, page in enumerate(get_query(work_type, pre_filter).paginate(
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

def filter_works(work_type: WorkType, topic_filter: Subject, cache_dir: Path):
    stats: Dict[str, Union[int, float]]
    stats = {f"{filter}": 0 for filter in WorkFilter}
    stats["total"] = 0
    stats["num_removed"] = 0
    # invalid_dir = cache_dir / "filter" / str(topic) / str(work_type)
    # invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # with (invalid_dir / "invalid_works.txt").open("w") as f:
    for work in tqdm.tqdm(get_works(work_type, topic_filter, cache_dir), total=get_query(work_type, topic_filter).count(),
                          desc=f"Filtering {work_type} works in {topic_filter}", disable=False):
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
    LOGGER.info(f"{topic_filter} {work_type} filter stats: {json.dumps(stats, indent=2, sort_keys=True)}")

@MEMORY.cache
def deduplicate_works_syntactic(work_type: WorkType, topic_filter: Subject):
    works = list(filter_works(work_type, topic_filter, CACHE_DIR))
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
    LOGGER.info(f"{topic_filter} {work_type} deduplication stats: {json.dumps({'min': counts.min().item(), 'max': counts.max().item(), 'mean': counts.mean().item(), 'std': counts.std().item(), 'total': len(works)}, indent=2)}")
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
def embed_works(work_type: WorkType, topic_filter: Subject):
    LOGGER.info(f"Embedding works for {work_type} in {topic_filter}")
    works = iter(deduplicate_works_syntactic(work_type, topic_filter))
    emb_cache_dir = CACHE_DIR / "embeddings" / create_unicode_safe_filename(topic_filter) / create_unicode_safe_filename(work_type)
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
    for batch in batched(tqdm.tqdm(works, desc="Embedding works", disable=False), batch_size):
        if model is None:
            LOGGER.info(f"Embedding works for {work_type} in {topic_filter}")
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





def get_query(work_type: WorkType, topic_filter: Subject) -> pyalex.Works:
    # .sort(publication_date="asc", display_name="asc") 
    return pyalex.Works() \
            .filter(has_abstract=True,
                    # language="en",
                    type=str(work_type), **topic_filter.to_dict()) \
            .select(", ".join(["display_name", "publication_date", "ids", "language", "abstract_inverted_index", "referenced_works", "primary_topic"]))

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
