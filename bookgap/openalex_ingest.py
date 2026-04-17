import enum
import itertools
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import re
import tqdm

from numpy.typing import NDArray

from bookgap.embedder import Embedder
from bookgap.pyalex_client import pyalex, RATE_LIMITER
from bookgap.util import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    Filter,
    LOGGER,
    MEMORY,
    TIMER,
    create_unicode_safe_filename,
    get_stats,
)
from bookgap.openalex_scopes import resolve_filter_upward, get_subfield_topics, _suffix

@dataclass(frozen=True)
class ResolvedScope:
    field: str
    subfield: str
    topic: str
    work_type: str

    def __str__(self) -> str:
        return f"{self.field}/{self.subfield}/{self.topic}/{self.work_type}"

    def filter(self) -> Filter:
        """Return a single-type filter corresponding to this scope."""
        return Filter(field=self.field, subfield=self.subfield, topic=self.topic, work_types=(self.work_type,))

    __repr__ = __str__


def _reset_state_if_empty(state_file: Path, cache_dirs: list[Path]) -> dict[str, Any]:
    """Ensure state is usable even if previous run wrote an empty, finished state."""
    if not state_file.exists():
        state = {"next_cursor": "*", "next_page": 1}
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w") as f:
            json.dump(state, f)
        return state

    with state_file.open("r") as f:
        state = json.load(f)

    has_pages = any((cache_dir / f"page_1.json").exists() for cache_dir in cache_dirs)
    if state.get("next_cursor") is None and not has_pages:
        raise RuntimeError(f"Cache state at {state_file} was finished but empty")
    return state


def _primary_topic_id(work: dict[str, Any]) -> str:
    # IDs are of the form https://openalex.org/Txxxxx; keep the suffix
    return _suffix(work["primary_topic"]["id"])


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
            "display_name", "type", "publication_date", "ids", "language", "abstract_inverted_index", "referenced_works", "primary_topic", "primary_location", "created_date", "updated_date",
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


@TIMER.time()
@MEMORY.cache
def _resolve_filter_scopes(work_filter: Filter, snapshot_date: str) -> list[ResolvedScope]:
    """Resolve a filter into concrete (field, subfield, topic, work_type) scopes.

    Reads the snapshot-derived taxonomy at cache/snapshot_taxonomy_<date>.json
    (built as a byproduct of bookgap.snapshot_ingest.run). There is no live
    API path. The taxonomy reflects exactly the scopes for which works were
    ingested at snapshot_date.
    """
    from bookgap.snapshot_ingest import load_taxonomy

    if work_filter.work_types is None:
        raise ValueError("work_types must be provided to resolve scopes")

    resolved_filter = resolve_filter_upward(work_filter, snapshot_date=snapshot_date)
    tax = load_taxonomy(snapshot_date)

    def _topics_in_subfield(subfield_id: str) -> list[str]:
        return get_subfield_topics(subfield_id, snapshot_date=snapshot_date)

    def _subfields_in_field(field_id: str) -> list[str]:
        field_data = tax.fields.get(field_id)
        if field_data is None:
            raise ValueError(f"Field {field_id} not in snapshot taxonomy ({snapshot_date})")
        ids = sorted({_suffix(sf["id"]) for sf in field_data.get("subfields", []) if isinstance(sf.get("id"), str)})
        if not ids:
            raise ValueError(f"No subfields recorded for field {field_id} at snapshot {snapshot_date}")
        return ids

    def _fields_in_domain(domain_id: str) -> list[str]:
        out: list[str] = []
        for fid, f in tax.fields.items():
            dom = f.get("domain")
            if isinstance(dom, dict) and _suffix(dom.get("id", "")) == domain_id:
                out.append(fid)
        out = sorted(set(out))
        if not out:
            raise ValueError(f"No fields recorded for domain {domain_id} at snapshot {snapshot_date}")
        return out

    if resolved_filter.topic is not None:
        field_id = resolved_filter.field
        subfield_id = resolved_filter.subfield
        assert field_id is not None and subfield_id is not None
        return [
            ResolvedScope(field=field_id, subfield=subfield_id, topic=resolved_filter.topic, work_type=wt)
            for wt in work_filter.work_types
        ]

    if resolved_filter.subfield is not None:
        field_id = resolved_filter.field
        subfield_id = resolved_filter.subfield
        assert field_id is not None
        return [
            ResolvedScope(field=field_id, subfield=subfield_id, topic=topic_id, work_type=wt)
            for topic_id in _topics_in_subfield(subfield_id)
            for wt in work_filter.work_types
        ]

    if resolved_filter.field is not None:
        field_id = resolved_filter.field
        scopes: list[ResolvedScope] = []
        for subfield_id in _subfields_in_field(field_id):
            for topic_id in _topics_in_subfield(subfield_id):
                for wt in work_filter.work_types:
                    scopes.append(ResolvedScope(
                        field=field_id, subfield=subfield_id, topic=topic_id, work_type=wt,
                    ))
        return scopes

    domain_id = resolved_filter.domain
    assert domain_id is not None
    scopes = []
    for field_id in _fields_in_domain(domain_id):
        for subfield_id in _subfields_in_field(field_id):
            for topic_id in _topics_in_subfield(subfield_id):
                for wt in work_filter.work_types:
                    scopes.append(ResolvedScope(
                        field=field_id, subfield=subfield_id, topic=topic_id, work_type=wt,
                    ))
    return scopes


def _get_works_for_type(scope: ResolvedScope) -> Generator[dict[str, Any], None, None]:
    """Fetch and cache works for a single scope work type.

    Cache structure cache/api_data/<field>/<subfield>/<topic_id>/<work_type>/page_*.json
    """

    state_dir = CACHE_DIR / "api_data" / scope.field / scope.subfield / scope.topic / scope.work_type
    cache_dirs = [state_dir]
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "state.json"
    state = _reset_state_if_empty(state_file, cache_dirs)

    def yield_cached_pages():
        next_page = int(state.get("next_page", 1))
        for page_num in range(1, next_page):
            page_file = state_dir / f"page_{page_num}.json"
            if page_file.exists():
                with page_file.open("r") as f:
                    yield from json.load(f)

    yield from yield_cached_pages()

    if state.get("next_cursor") is None:
        return

    RATE_LIMITER.wait()

    # Then yield new works
    cursor = state["next_cursor"]
    empty_page_retries = 0
    max_empty_page_retries = 10
    while cursor is not None:
        RATE_LIMITER.wait()
        page_iter = get_query(scope.filter()).paginate(cursor=cursor, per_page=200, n_max=1) # type: ignore
        page = next(iter(page_iter), None)
        if page is None:
            empty_page_retries += 1
            LOGGER.warning(
                f"No page returned for {scope.filter()} at cursor={cursor}; retry {empty_page_retries}/{max_empty_page_retries}"
            )
            if empty_page_retries > max_empty_page_retries:
                raise RuntimeError(
                    f"No page returned for {scope.filter()} after {max_empty_page_retries} retries"
                )
            RATE_LIMITER.wait()
            continue

        works: list[dict[str, Any]] = list(map(dict, page))
        filtered_works: list[dict[str, Any]] = []
        dropped_no_abstract = 0
        for work in works:
            inverted = work.get("abstract_inverted_index")
            if not inverted:
                dropped_no_abstract += 1
                continue
            work["abstract"] = resolve_abstract_inverted_index(inverted)
            del work["abstract_inverted_index"]
            work["id"] = work["ids"]["openalex"]
            del work["ids"]
            if (work_topic := _primary_topic_id(work)) != scope.topic:
                raise RuntimeError(
                    f"Work {work['id']} primary topic {work_topic} differs from requested topic {scope.topic}"
                )
            work["topic"] = scope.topic
            work["field_id"] = scope.field
            work["subfield_id"] = scope.subfield
            work["topic_id"] = scope.topic
            filtered_works.append(work)
        if dropped_no_abstract:
            LOGGER.info(f"Skipped {dropped_no_abstract} works without abstracts for {scope.filter()}")

        next_cursor = page.meta["next_cursor"]
        if not filtered_works:
            if next_cursor is not None:
                empty_page_retries += 1
                if empty_page_retries > max_empty_page_retries:
                    raise RuntimeError(
                        f"Received empty page for {scope.filter()} before pagination finished after {max_empty_page_retries} retries"
                    )
                LOGGER.warning(
                    f"Empty page for {scope.filter()} at cursor={cursor} (next_cursor={next_cursor}); retry {empty_page_retries}/{max_empty_page_retries}"
                )
                cursor = next_cursor
                RATE_LIMITER.wait()
                continue
            state["next_cursor"] = next_cursor
            with state_file.open("w") as f:
                json.dump(state, f)
            return

        empty_page_retries = 0

        # Save works, sharded by topic
        page_file = state_dir / f"page_{state['next_page']}.json"
        with page_file.open("w") as f:
            json.dump(filtered_works, f, indent=2)

        state["next_page"] = int(state.get("next_page", 1)) + 1
        state["next_cursor"] = next_cursor
        with state_file.open("w") as f:
            json.dump(state, f)

        yield from filtered_works

        RATE_LIMITER.wait()
        cursor = next_cursor


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

    def __call__(self, work: dict[str, Any]) -> bool:
        return self.value(work)

    def __str__(self):
        return repr(self.value)


def _filter_works_iter(scope: ResolvedScope, works_iter):
    """Filter works from an externally-provided iterator for a single scope.

    Identical semantics to _filter_works, but takes the raw work iterator as a
    parameter instead of calling _get_works_for_type internally. Used by both
    the API ingest path (via _filter_works) and the snapshot ingest path.
    """
    filters = list(WorkFilter)

    class PerTypeCounter:
        def __init__(self) -> None:
            self.total: int = 0
            self.per_type: dict[str, int] = {}
        def insert(self, work: dict[str, Any]):
            self.total += 1
            work_type = work.get('type', 'UNKNOWN')
            self.per_type[work_type] = self.per_type.get(work_type, 0) + 1

    filter_stats = {f"{filter}": PerTypeCounter() for filter in filters}
    total = PerTypeCounter()
    num_removed = PerTypeCounter()
    artifacts_dir = ARTIFACTS_DIR / str(scope)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "invalid_works.txt").open("w") as f:
        for work in tqdm.tqdm(works_iter,
                desc=f"Filtering works in {scope}", mininterval=3, delay=3, smoothing=1):
            invalidators = []
            total.insert(work)
            for filter in filters:
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
        stats = {
            "total": total.total,
            "num_removed": num_removed.total,
            "fraction_removed": num_removed.total / total.total if total.total > 0 else 0.0,
        }
        json.dump(stats, f, indent=2)
        LOGGER.info(f"{scope} filter stats: {json.dumps(stats, indent=2)}")


def _filter_works(scope: ResolvedScope):
    """Filter works for a single concrete scope (field/subfield/topic/work_type)."""
    yield from _filter_works_iter(scope, _get_works_for_type(scope))


def _deduplicate_works(works: list[dict[str, Any]], scope: ResolvedScope) -> list[dict[str, Any]]:
    """Stateless syntactic dedup for a pre-filtered list of works.

    Writes the per-scope syntactic_deduplication_stats.txt artifact and returns
    a list containing one representative per normalized-text equivalence class.
    """
    deduped: dict[str, list[dict[str, Any]]] = {}
    def normalize_work(work: dict[str, Any], _NONWORD_PAT=re.compile(r"\W")) -> str:
        s = f"{work['display_name']} {work['abstract']}"
        s = _NONWORD_PAT.sub("", s)
        s = s.lower()
        return s
    for work in works:
        key = normalize_work(work)
        deduped.setdefault(key, []).append(work)

    artifacts_dir = ARTIFACTS_DIR / create_unicode_safe_filename(f"{scope}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    total_works = len(works)
    unique_works = len(deduped)
    dup_groups = [group for group in deduped.values() if len(group) > 1]
    with (artifacts_dir / f"syntactic_deduplication_stats.txt").open("w") as f:
        f.write(f"Total works: {total_works}\n")
        f.write(f"Unique works: {unique_works} ({unique_works / total_works if total_works else 0:.2%})\n")
        f.write(f"Number of duplicate groups: {len(dup_groups)}\n")
        f.write("Distribution of len(group) for duplicate groups (len>=2):\n")
        f.write(f"{get_stats(np.array([len(group) for group in dup_groups]))}\n")
        f.write("\n")

    return [group[0] for group in deduped.values()]  # Return the first work of each deduplicated group


@MEMORY.cache
def deduplicate_works_syntactic(scope: ResolvedScope):
    return _deduplicate_works(list(_filter_works(scope)), scope)


def batched(iterable, n): # TODO use itertools.batched when available
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def _scope_parquet_path(scope: ResolvedScope) -> Path:
    """Return the path for a scope's per-scope Parquet file."""
    return CACHE_DIR / "works_parquet" / scope.field / scope.subfield / scope.topic / f"{scope.work_type}.parquet"


def _safe_date_array(values: list) -> pa.Array:
    """Coerce date values to date32, tolerant of strings/None."""
    from datetime import datetime, date
    coerced = []
    for v in values:
        if v is None:
            coerced.append(None)
        elif isinstance(v, date):
            coerced.append(v)
        elif isinstance(v, datetime):
            coerced.append(v.date())
        else:
            try:
                coerced.append(datetime.fromisoformat(str(v).replace("Z", "+00:00")).date())
            except Exception:
                coerced.append(None)
    return pa.array(coerced, type=pa.date32())


SCOPE_PARQUET_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("display_name", pa.string()),
    ("abstract", pa.string()),
    ("type", pa.string()),
    ("publication_date", pa.date32()),
    ("created_date", pa.date32()),
    ("updated_date", pa.date32()),
    ("language", pa.string()),
    ("topic_label", pa.string()),
    ("primary_topic_id", pa.string()),
    ("primary_subfield_id", pa.string()),
    ("referenced_works", pa.list_(pa.string())),
])


def works_to_table(works: list[dict[str, Any]]) -> pa.Table:
    """Build a pyarrow Table for a list of normalized work dicts.

    The resulting table has SCOPE_PARQUET_SCHEMA — the canonical layout
    used by both _write_scope_parquet (final per-scope outputs) and
    snapshot_ingest's intermediate spill shards.
    """
    if not works:
        return pa.table({
            name: pa.array([], type=field.type)
            for name, field in zip(SCOPE_PARQUET_SCHEMA.names, SCOPE_PARQUET_SCHEMA)
        })

    primary_topic_ids: list = []
    primary_subfield_ids: list = []
    for w in works:
        pt = w.get("primary_topic")
        if isinstance(pt, dict):
            primary_topic_ids.append(pt.get("id"))
            sf = pt.get("subfield")
            primary_subfield_ids.append(sf.get("id") if isinstance(sf, dict) else None)
        else:
            primary_topic_ids.append(None)
            primary_subfield_ids.append(None)

    refs: list = []
    for w in works:
        r = w.get("referenced_works")
        refs.append([str(x) for x in r] if isinstance(r, list) else [])

    return pa.table({
        "id": pa.array([w.get("id") for w in works], type=pa.string()),
        "display_name": pa.array([w.get("display_name", "") for w in works], type=pa.string()),
        "abstract": pa.array([w.get("abstract", "") for w in works], type=pa.string()),
        "type": pa.array([w.get("type", "UNKNOWN") for w in works], type=pa.string()),
        "publication_date": _safe_date_array([w.get("publication_date") for w in works]),
        "created_date": _safe_date_array([w.get("created_date") for w in works]),
        "updated_date": _safe_date_array([w.get("updated_date") for w in works]),
        "language": pa.array([w.get("language") for w in works], type=pa.string()),
        "topic_label": pa.array([w.get("topic") for w in works], type=pa.string()),
        "primary_topic_id": pa.array(primary_topic_ids, type=pa.string()),
        "primary_subfield_id": pa.array(primary_subfield_ids, type=pa.string()),
        "referenced_works": pa.array(refs, type=pa.list_(pa.string())),
    })


def _write_scope_parquet(path: Path, works: list[dict[str, Any]]) -> None:
    """Write a list of work dicts to a single Parquet file."""
    table = works_to_table(works)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, path)


@TIMER.time()
def fetch_works(scope: ResolvedScope) -> ds.Dataset:
    """Fetch, deduplicate, and cache works for a scope as Parquet. Returns lazy Dataset."""
    parquet_path = _scope_parquet_path(scope)
    if not parquet_path.exists():
        works = deduplicate_works_syntactic(scope)
        _write_scope_parquet(parquet_path, works)
    return ds.dataset(parquet_path, format="parquet")


class MissingScopeParquetError(FileNotFoundError):
    """Raised by corpus_dataset when a required per-scope parquet is missing.

    This protects reproducibility: corpus_dataset never silently falls back
    to a live API fetch. Use bookgap.snapshot_ingest.run() to populate the
    missing parquets, or bookgap.openalex_ingest.fetch_all_scopes() for the
    live-API path (separate, explicit step).
    """


@lru_cache(maxsize=4)
def corpus_dataset(work_filter: Filter, snapshot_date: str) -> ds.Dataset:
    """Lazy Dataset spanning all scopes for a filter at a snapshot date.

    Resolves scopes from the snapshot-derived taxonomy and reads only the
    parquets that already exist on disk. If any required parquet is
    missing, raises MissingScopeParquetError listing the gaps. The snapshot
    is the single source of truth: there is no live API fallback.
    """
    scopes = _resolve_filter_scopes(work_filter, snapshot_date)
    paths: list[Path] = []
    missing: list[ResolvedScope] = []
    for scope in scopes:
        path = _scope_parquet_path(scope)
        if path.exists():
            paths.append(path)
        else:
            missing.append(scope)
    if missing:
        sample = ", ".join(str(s) for s in missing[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise MissingScopeParquetError(
            f"{len(missing)} of {len(scopes)} scope parquets are missing for filter "
            f"{work_filter} at snapshot {snapshot_date}: {sample}{more}. "
            f"Run bookgap.snapshot_ingest.run(work_filter=..., snapshot_date={snapshot_date!r}) "
            f"to populate them."
        )
    return ds.dataset(paths, format="parquet")


def fetch_all_scopes(work_filter: Filter, snapshot_date: str) -> None:
    """Ensure all scope Parquet files exist without loading any data.

    Iterates the scopes resolved against the snapshot taxonomy and runs the
    live API path for any that don't have a parquet yet. This is the
    explicit live-fetch entry point, complementary to snapshot_ingest.run.
    """
    scopes = _resolve_filter_scopes(work_filter, snapshot_date)
    failed: list[tuple[ResolvedScope, str]] = []
    for scope in tqdm.tqdm(scopes, desc="Fetching scopes"):
        if not _scope_parquet_path(scope).exists():
            try:
                RATE_LIMITER.wait()
                fetch_works(scope)
            except Exception as e:
                LOGGER.error(f"Failed to fetch {scope}: {e}")
                failed.append((scope, str(e)))
    if failed:
        LOGGER.warning(f"{len(failed)} scope(s) failed: {[str(s) for s, _ in failed]}")


def embed_works(
    scope: ResolvedScope,
    model_name: Embedder.ModelName,
):
    LOGGER.info(f"Embedding {scope} works with model {str(model_name)}")
    works = fetch_works(scope).to_table().to_pylist()
    n_works = len(works)
    with Embedder(model_name=model_name) as embedder:
        dim = embedder.embedding_dimension()
        embeddings = np.empty(shape=(n_works, dim), dtype=embedder.embedding_dtype())

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

        def scope_batches(scope_dir: Path, scope_works: List[dict[str, Any]]) -> Generator[NDArray, None, None]:
            seen = 0
            cache_exhausted = False
            for i in itertools.count():
                emb_path = scope_dir / f"batch_{i}_embeddings.npy"
                ids_path = scope_dir / f"batch_{i}_ids.json"

                if emb_path.exists() or ids_path.exists():
                    if cache_exhausted:
                        raise ValueError(f"Cache exhausted but found unexpected cache files {emb_path} or {ids_path}.")
                    with emb_path.open("rb") as f:
                        batch_embeddings: NDArray = np.load(f)
                    with ids_path.open("r") as f:
                        validate_batch(scope_works[seen : seen + len(batch_embeddings)], json.load(f))
                else:
                    cache_exhausted = True
                    BATCH_SIZE = 4096
                    batch_works = scope_works[seen : seen + BATCH_SIZE]
                    if len(batch_works) == 0:
                        return
                    batch_embeddings = embedder.embed_texts([(work["display_name"], work["abstract"]) for work in batch_works])
                    # Save atomically
                    tmp_emb_path = emb_path.with_name(emb_path.name + ".tmp")
                    with tmp_emb_path.open("wb") as embeddings_f:
                        np.save(embeddings_f, batch_embeddings)
                    os.replace(tmp_emb_path, emb_path)

                    tmp_ids_path = ids_path.with_name(ids_path.name + ".tmp")
                    with tmp_ids_path.open("w") as ids_f:
                        json.dump([work["id"] for work in batch_works], ids_f)
                    os.replace(tmp_ids_path, ids_path)

                seen += len(batch_embeddings)
                yield batch_embeddings

        scope_dir = CACHE_DIR / "embeddings" / embedder.model_name / scope.field / scope.subfield / scope.topic / scope.work_type
        scope_dir.mkdir(parents=True, exist_ok=True)

        scope_works: List[dict[str, Any]] = works
        idx_cursor = 0
        with tqdm.tqdm(total=n_works, desc=f"Embedding works for {scope}", mininterval=3, delay=3, smoothing=1) as pbar:
            for batch in scope_batches(scope_dir, scope_works):
                batch_len = len(batch)
                embeddings[idx_cursor : idx_cursor + batch_len] = batch
                idx_cursor += batch_len
                pbar.update(batch_len)

        return works, embeddings
