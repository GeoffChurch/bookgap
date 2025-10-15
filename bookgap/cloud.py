from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import enum
from functools import lru_cache
import heapq
import itertools
import json
from bookgap.cloud_denoiser import ProjectiveDenoiserProjection, WorkDenoisers
import re
import time
from typing import Any, Dict, Generator, List, Optional, TypeVar
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
import tqdm
from bookgap.embedder import Embedder
from bookgap.index import IndexInfo, build_index
from bookgap.util import ARTIFACTS_DIR, CACHE_DIR, FilterPair, Filter, TIMER, MEMORY, LOGGER, WorkType, create_unicode_safe_filename, get_stats
from bookgap.deduplicators import Deduplicator
import scipy.linalg

pyalex.config.email = "gchurchill@cs.stonybrook.edu"

@dataclass(frozen=True)
class Cloud:
    filter_pair: FilterPair
    index_info: IndexInfo
    max_works: Optional[int]
    deduplicator: Optional[Deduplicator]
    normalize_embeddings: bool
    denoiser: str
    drop_k: int
    embedder_model: Embedder.ModelName

    def __post_init__(self):
        assert self.drop_k >= 0
        # Validate denoiser is a valid WorkDenoiser name
        valid_denoisers = [d.name for d in WorkDenoisers]
        if self.denoiser not in valid_denoisers:
            raise ValueError(f"denoiser must be one of {valid_denoisers}, got '{self.denoiser}'")
        if not isinstance(self.embedder_model, Embedder.ModelName):
            raise TypeError("embedder_model must be an instance of Embedder.ModelName")

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

    def denoiser_projection(self) -> ProjectiveDenoiserProjection:
        return get_cloud_projection(self)

    def name(self) -> str:
        return f"{self.filter_pair.merged_name()}"

@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_retained_idxs(cloud: Cloud) -> NDArray[np.uint64]:
    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model)
    
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
    works, _ = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model)
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
    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return denoise(cloud, [works[i] for i in retained_idxs], embeddings[retained_idxs])


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_projection(cloud: Cloud) -> ProjectiveDenoiserProjection:
    denoiser_enum = WorkDenoisers[cloud.denoiser]
    denoiser_impl = denoiser_enum.value

    works, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model)
    retained_idxs = get_cloud_retained_idxs(cloud)

    retained_works = [works[i] for i in retained_idxs]
    retained_embeddings = embeddings[retained_idxs]

    return denoiser_impl.compute_projection_artifact(cloud.drop_k, retained_works, retained_embeddings)


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def load_works(filter: Filter, max_works: Optional[int], embedder_model: Embedder.ModelName):
    works, undeduped_embeddings = embed_works(filter, max_works, embedder_model)
    if max_works is None:
        argsort = np.argsort([work["publication_date"] for work in works])
    else:
        argsort = np.array(heapq.nsmallest(max_works, range(len(works)), key=lambda i: works[i]["publication_date"]))
    works = [works[i] for i in argsort]
    undeduped_embeddings = undeduped_embeddings[argsort]
    
    LOGGER.info(f"Loaded {len(works)} works")
    n, d = undeduped_embeddings.shape
    assert n == len(works), f"Embeddings and works length mismatch: {n} != {len(works)}"

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
    retained_idxs = deduplicate_works_semantic_(
        cloud.deduplicator,
        cloud.index_info,
        embeddings,
        embeddings_key=(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model),
    )
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

def _get_works(pre_filter: Filter) -> Generator[dict[str, Any], None, None]:
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

    time.sleep(1.0) # Initial 1 second wait in case of earlier requests

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
        min_time_diff = 1.0 # Minimum time between requests
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

def _filter_works(work_filter: Filter):
    class PerTypeCounter:
        def __init__(self) -> None:
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
        total_expected = get_query(work_filter).count()
        for work in tqdm.tqdm(_get_works(work_filter), total=total_expected,
                            desc=f"Filtering works in {work_filter}", mininterval=3, delay=3, smoothing=1):
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
    works = list(_filter_works(work_filter))
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
        collisions: defaultdict[tuple[str, ...], int] = defaultdict(int)
        for dupgroup in deduped.values():
            work_types: tuple[str, ...] = tuple(sorted(set(work["type"] for work in dupgroup)))
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

def embed_works(
    work_filter: Filter,
    max_works: Optional[int],
    model_name: Embedder.ModelName,
):
    LOGGER.info(f"Embedding {work_filter} works with model {model_name.value}")
    works = deduplicate_works_syntactic(work_filter)

    with Embedder(model_name=model_name) as embedder:
        emb_cache_dir = CACHE_DIR / "embeddings" / create_unicode_safe_filename(f"{work_filter}, {embedder.model_name}")
        emb_cache_dir.mkdir(parents=True, exist_ok=True)

        def validate_batch(batch_works: List[dict[str, Any]], expected_ids: List[str]):
            actual_ids = [work["id"] for work in batch_works]
            if len(actual_ids) != len(expected_ids):
                msg = f"ID length mismatch in batch. Expected {len(expected_ids)} IDs, got {len(actual_ids)} IDs"
                LOGGER.error(msg)
                raise ValueError(msg)
            try:
                mismatch_idx = next(i for i, (a, b) in enumerate(zip(actual_ids, expected_ids)) if a != b)
                msg = f"ID mismatch at index {mismatch_idx} in batch. Expected {expected_ids[mismatch_idx]}, got {actual_ids[mismatch_idx]}"
                LOGGER.error(msg)
                raise ValueError(msg)
            except StopIteration:
                return

        def batches() -> Generator[NDArray, None, None]:
            seen = 0
            cache_exhausted = False
            

            for i in itertools.count():
                emb_path = emb_cache_dir / f"batch_{i}_embeddings.npy"
                ids_path = emb_cache_dir / f"batch_{i}_ids.json"

                if emb_path.exists() or ids_path.exists():
                    if cache_exhausted:
                        raise ValueError(f"Cache exhausted but found unexpected cache files {emb_path} or {ids_path}.")
                    with emb_path.open("rb") as f:
                        embeddings: NDArray = np.load(f)
                    with ids_path.open("r") as f:
                        validate_batch(works[seen:seen + len(embeddings)], json.load(f))
                else:
                    cache_exhausted = True
                    BATCH_SIZE = 4096
                    batch_works = works[seen:seen + BATCH_SIZE]
                    if len(batch_works) == 0:
                        return
                    embeddings = embedder.embed_texts([(work["display_name"], work["abstract"]) for work in batch_works])
                    # Save batch
                    with emb_path.open("wb") as embeddings_f, ids_path.open("w") as ids_f:
                        np.save(embeddings_f, embeddings)
                        json.dump([work["id"] for work in batch_works], ids_f)

                seen += len(embeddings)
                yield embeddings
            

        n_works = len(works)
        if max_works is not None:
            n_works = min(n_works, max_works)

        dim = embedder.embedding_dimension()
        embeddings = np.empty(shape=(n_works, dim), dtype=embedder.embedding_dtype())
        idx = 0
        with tqdm.tqdm(total=n_works, desc=f"Embedding works for {work_filter}", mininterval=3, delay=3, smoothing=1) as pbar:
            for batch in batches():
                remaining = n_works - idx
                if len(batch) > remaining:
                    batch = batch[:remaining]

                embeddings[idx:idx + len(batch)] = batch
                idx += len(batch)
                pbar.update(len(batch))
                if idx >= n_works:
                    assert idx == n_works
                    break
        return works[:max_works], embeddings

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


T = TypeVar('T', bound=np.generic)

def denoise(cloud: Cloud, works: List[Dict[str, Any]], embeddings: NDArray[T]) -> NDArray[T]:
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