from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import lru_cache
import hashlib
import heapq
import json
import os
from pathlib import Path
from bookgap.cloud_denoiser import ProjectiveDenoiserProjection, WorkDenoisers
from bookgap.corpus import embedder_fingerprint
from typing import Any, Dict, Iterable, List, Optional, TypeVar
import faiss
import numpy as np
from numpy.typing import NDArray
from bookgap.pyalex_client import pyalex
import pyarrow.dataset as ds
import tqdm
from bookgap.embedder import BaseEmbedder, Embedder, SentenceTransformerEmbedder
from bookgap.index import IndexInfo, build_index
import scipy.sparse as sp
from bookgap.util import (
    ARTIFACTS_DIR,
    SCOPES_DIR,
    CACHE_DIR,
    FilterPair,
    Filter,
    TIMER,
    MEMORY,
    LOGGER,
    build_citation_adjacency_from_ids,
)
from bookgap.deduplicators import Deduplicator
import scipy.linalg
from sklearn.preprocessing import normalize as sk_normalize
import pyarrow as pa

from bookgap.dendrogram import DendrogramResult, get_cloud_dendrogram

from bookgap import jobs
from bookgap.openalex_ingest import (
    _resolve_filter_scopes,
    corpus_dataset,
    embed_works,
)

def _log_rss(tag: str) -> None:
    """Log current process RSS at a named transition.

    Used at big-allocation boundaries (pre/post full embeddings load,
    pre/post dedup, pre/post filtered load) so that if we ever OOM again
    the pipeline log tells us which step was over budget — far faster than
    another kernel-log forensic exercise.
    """
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        LOGGER.info(f"[rss:{tag}] {rss_gb:.1f} GB")
    except Exception:  # psutil not installed / /proc unreadable
        pass


def _embeddings_store_path(corpus_fingerprint: str, embedder_cfg: Any) -> Path:
    """Path for the embeddings store: an uncompressed float32 ``.npy`` array
    whose row ``i`` is corpus row position ``i``."""
    embedder_fp = embedder_fingerprint(embedder_cfg)
    return CACHE_DIR / "embeddings_npy" / corpus_fingerprint / embedder_fp / "embeddings.npy"


def write_embeddings_store(path: Path, embeddings: NDArray[np.float32]) -> None:
    """Write embeddings as an uncompressed float32 ``.npy`` array.

    Uncompressed so readers can memory-map it: a subset read then touches only
    the pages holding its rows, instead of decompressing the whole file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(embeddings, dtype=np.float32))
    os.replace(tmp, path)


def read_embeddings_store(path: Path) -> NDArray[np.float32]:
    """Every row of the store, as a read-only memory map.

    Nothing is read until rows are touched, and touched rows live in the page
    cache rather than the heap, so even the 64 GB domain-scale store can be
    scanned end to end without a heap copy.
    """
    return np.load(path, mmap_mode="r")


def read_embeddings_rows(
    path: Path,
    retained_positions: Optional[NDArray[np.uint64]] = None,
) -> NDArray[np.float32]:
    """The rows at *retained_positions* (every row if None), copied into memory
    in the order given.

    Reads through a memory map, so the cost scales with the rows requested,
    not with the size of the store.
    """
    store = read_embeddings_store(path)
    if retained_positions is None:
        return np.array(store)
    return np.asarray(store[np.asarray(retained_positions, dtype=np.intp)])


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
    # Pin taxonomy resolution and the corpus parquet set to a specific
    # OpenAlex snapshot for reproducibility. The snapshot must have been
    # ingested via bookgap.snapshot_ingest before constructing the Cloud.
    snapshot_date: str = field(default="2025-11-06")

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

    def work_titles_subset(self, subset_indices: NDArray[np.uint64]) -> List[str]:
        return get_cloud_work_titles_subset(self, subset_indices)


    def work_abstracts(self) -> List[str]:
        return get_cloud_work_abstracts(self)
    
    def work_publication_dates(self) -> List[str]:
        return get_cloud_work_publication_dates(self)

    def work_types(self) -> List[str]:
        return get_cloud_work_types(self)

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

def _corpus_fingerprint(filter: Filter, max_works: Optional[int]) -> str:
    """Stable fingerprint for a corpus defined by filter + max_works."""
    payload = {"filter": asdict(filter), "max_works": max_works}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _sorted_corpus_idxs(corpus: ds.Dataset, max_works: Optional[int]) -> NDArray[np.uint64]:
    """Return corpus-row indices sorted by publication_date, dropping nulls.

    If *max_works* is not None, only the earliest *max_works* are returned.
    """
    dates = corpus.to_table(columns=["publication_date"]).column("publication_date").to_pylist()
    valid_idxs = [i for i, d in enumerate(dates) if d is not None]
    valid_dates = [dates[i] for i in valid_idxs]
    if max_works is None:
        order = sorted(range(len(valid_idxs)), key=lambda j: (valid_dates[j], valid_idxs[j]))
    else:
        order = heapq.nsmallest(max_works, range(len(valid_idxs)), key=lambda j: (valid_dates[j], valid_idxs[j]))
    return np.array([valid_idxs[j] for j in order], dtype=np.uint64)


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def ensure_cloud_data(cloud: Cloud) -> tuple[str, NDArray[np.uint64], NDArray[np.uint64]]:
    """Ensure corpus is fetched, embeddings computed, and retained indices ready.

    Returns ``(corpus_fingerprint, sorted_idxs, retained_positions)`` where
    *sorted_idxs* are physical corpus-row indices for the date-sorted subset,
    and *retained_positions* are positions **into that subset** (and into the
    embeddings file) after post-filtering and semantic dedup.
    """
    corpus_fp = _corpus_fingerprint(cloud.filter_pair.pre, cloud.max_works)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)

    # 1. Date-sort + max_works truncation -> sorted_idxs (corpus row positions)
    sorted_idxs = _sorted_corpus_idxs(corpus, cloud.max_works)
    LOGGER.info(f"Cloud {cloud.name()} -- {len(sorted_idxs)} works after date-sort/truncation")

    # 2. Compute and store embeddings for the sorted subset
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)
    if not emb_path.exists():
        _compute_and_store_embeddings(cloud, corpus, sorted_idxs, emb_path)

    # 3. Semantic dedup (operates on the full sorted subset embeddings).
    # This is the one place that genuinely needs every row: the dedup
    # algorithm compares each point against every other. ``raw_embeddings``
    # is a memory map, so those rows sit in the page cache, not the heap;
    # it is still dropped promptly once dedup returns a small index array.
    _log_rss("before-read-embeddings-for-dedup")
    raw_embeddings = read_embeddings_store(emb_path)
    _log_rss("after-read-embeddings-for-dedup")
    dedup_positions = _deduplicate_semantic(cloud, corpus, raw_embeddings, sorted_idxs)
    LOGGER.info(f"Retained {len(dedup_positions)} works after deduplication")
    del raw_embeddings
    import gc
    gc.collect()
    _log_rss("after-drop-raw-embeddings")

    # 4. Post-filter (operates on deduped positions within sorted subset)
    post_mask = _post_filter_mask(corpus, cloud.filter_pair.post, sorted_idxs[dedup_positions])
    retained_positions = dedup_positions[post_mask]
    LOGGER.info(f"Retained {len(retained_positions)} works after post-filtering")

    return corpus_fp, sorted_idxs, retained_positions


def _compute_and_store_embeddings(
    cloud: Cloud,
    corpus: ds.Dataset,
    sorted_idxs: NDArray[np.uint64],
    emb_path: Path,
) -> None:
    """Compute embeddings for the sorted subset and write to *emb_path*."""
    from bookgap.configs.embedder.embedder_config import (
        SentenceTransformerEmbedderConfig,
        TfGepEmbedderConfig,
    )

    if isinstance(cloud.embedder, SentenceTransformerEmbedderConfig):
        embeddings = _embed_sentence_transformer(cloud.filter_pair.pre, cloud.embedder, sorted_idxs, cloud.snapshot_date)
    elif isinstance(cloud.embedder, TfGepEmbedderConfig):
        embeddings = _embed_tf_gep(corpus, cloud.embedder, sorted_idxs)
    else:
        raise TypeError(f"Unsupported embedder config type: {type(cloud.embedder).__name__}")

    assert embeddings.shape[0] == len(sorted_idxs), (
        f"Embeddings rows ({embeddings.shape[0]}) != sorted_idxs length ({len(sorted_idxs)})"
    )
    write_embeddings_store(emb_path, embeddings)
    LOGGER.info(f"Wrote embeddings ({embeddings.shape}) to {emb_path}")


def _embed_sentence_transformer(
    work_filter: Filter,
    embedder_cfg: Any,
    sorted_idxs: NDArray[np.uint64],
    snapshot_date: str,
) -> NDArray[np.float32]:
    """Sentence-transformer path: per-scope batch caching via embed_works.

    Returns embeddings re-ordered to match *sorted_idxs*.

    Loads the Embedder once around the whole scope loop (per-scope reload
    dominates wall time otherwise). Polls :func:`bookgap.jobs.stop_requested`
    between scopes so ``systemctl --user stop <scope>`` or Ctrl-C inside
    a :func:`bookgap.jobs.worker_lifecycle` block triggers an orderly
    shutdown after the current scope finishes; the per-scope cache stays
    consistent and a subsequent run resumes from where it left off.
    Outside a ``worker_lifecycle`` the poll is a no-op.
    """
    model_name = Embedder.ModelName(embedder_cfg.model_name)

    # Collect per-scope IDs + embeddings in corpus-file order. We deliberately
    # drop the full work dicts (with abstracts) as soon as embed_works
    # returns; retaining them across all 6k+ scopes peaks at ~80 GB on a
    # full-domain corpus and OOM-killed an earlier run on a 250 GB box.
    all_ids: list[str] = []
    emb_chunks: list[NDArray] = []

    with Embedder(model_name=model_name) as embedder:
        with tqdm.tqdm(_resolve_filter_scopes(work_filter, snapshot_date), desc="Scopes", delay=3, mininterval=3) as scope_bar:
            for scope in scope_bar:
                if jobs.stop_requested():
                    raise KeyboardInterrupt(
                        "Graceful stop requested between scopes; per-scope cache is consistent."
                    )
                scope_bar.set_postfix_str(str(scope))
                scope_works, scope_embeddings = embed_works(scope, embedder)
                all_ids.extend(w["id"] for w in scope_works)
                emb_chunks.append(scope_embeddings)
                del scope_works

    if emb_chunks:
        all_embeddings = np.concatenate(emb_chunks, axis=0)
    else:
        all_embeddings = np.empty((0, Embedder(model_name=model_name).embedding_dimension()), dtype=np.float32)
    # The per-scope arrays each retain a refcount through emb_chunks; drop
    # the list so concat's intermediate view + the scope arrays can both
    # be reclaimed before we move on to ID verification.
    del emb_chunks

    # Verify row ordering matches corpus dataset. Use pyarrow's vectorised
    # equality on the chunked column rather than to_pylist() so we don't
    # materialise a second 6 GB Python list of corpus IDs alongside all_ids.
    corpus = corpus_dataset(work_filter, snapshot_date)
    corpus_id_col = corpus.to_table(columns=["id"]).column("id")
    if len(all_ids) != len(corpus_id_col):
        raise RuntimeError(
            f"Scope concatenation length ({len(all_ids)}) != corpus dataset length ({len(corpus_id_col)}). "
            f"This indicates a scope resolution mismatch."
        )
    import pyarrow.compute as pc
    all_ids_arr = pa.array(all_ids, type=pa.string())
    eq_mask = pc.equal(all_ids_arr, corpus_id_col.combine_chunks())
    mismatch_count = pc.sum(pc.cast(pc.invert(eq_mask), pa.int64())).as_py()
    if mismatch_count:
        first_idx = pc.index(eq_mask, False).as_py()
        raise RuntimeError(
            f"Row ordering mismatch between scope concatenation and corpus dataset at row {first_idx}: "
            f"concat={all_ids[first_idx]}, corpus={corpus_id_col[first_idx].as_py()}. "
            f"{mismatch_count} total mismatches."
        )
    del all_ids, all_ids_arr, eq_mask, corpus_id_col

    # Select only the rows indicated by sorted_idxs
    return all_embeddings[sorted_idxs]


def _embed_tf_gep(
    corpus: ds.Dataset,
    embedder_cfg: Any,
    sorted_idxs: NDArray[np.uint64],
) -> NDArray[np.float32]:
    """TfGepEmbedder path: read columns from corpus, fit, transform."""
    from bookgap.tf_gep_embedder import TfGepEmbedder

    if not len(sorted_idxs):
        return np.empty((0, embedder_cfg.n_components), dtype=np.float32)

    # Read only the needed columns, then use take() to select only the needed rows
    # avoiding building Python lists for all corpus rows.
    full_table = corpus.to_table(columns=["id", "display_name", "abstract", "referenced_works"])
    table = full_table.take(pa.array(sorted_idxs, type=pa.int64()))

    ids = table.column("id").to_pylist()
    display_names = table.column("display_name").to_pylist()
    abstracts = table.column("abstract").to_pylist()
    refs = table.column("referenced_works").to_pylist()

    works: list[dict[str, Any]] = [
        {
            "id": ids[i],
            "display_name": display_names[i] or "",
            "abstract": abstracts[i] or "",
            "referenced_works": refs[i] or [],
        }
        for i in range(len(ids))
    ]

    # Build citation graph
    work_ids = [w["id"] for w in works]
    work_refs = [w["referenced_works"] for w in works]
    citation_graph, _ = build_citation_adjacency_from_ids(
        work_ids, work_refs,
        add_self_loops=embedder_cfg.add_self_loops,
        normalize=embedder_cfg.normalize_adjacency,
    )

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
    return embedder.transform(title_abstract_pairs)


def _read_corpus_column(
    corpus: ds.Dataset,
    column: str,
    positions: NDArray[np.uint64],
    chunk_size: int = 500_000,
) -> List:
    """Read a single column from the corpus at the given positions.

    Streams the Dataset batch by batch so each ``take`` operates on at most
    ``chunk_size`` rows. Each intermediate output stays well under pyarrow's
    32-bit StringArray offset limit (~2 GB), regardless of how many
    positions the caller requests — the non-chunked
    ``to_table() + take(all positions)`` pattern blows that limit on
    ~40 M × 1 KB-ish string columns (e.g. ``abstract``, ``display_name`` on
    the domain=3 corpus). Results are returned in the order of ``positions``.
    """
    n = len(positions)
    if n == 0:
        return []

    # Sort ascending for a single-pass row-group scan; inverse permutation
    # restores caller order at write-time.
    positions_arr = np.asarray(positions, dtype=np.int64)
    order = np.argsort(positions_arr, kind="stable")
    sorted_positions = positions_arr[order]

    out: List[Any] = [None] * n
    row_offset = 0
    pos_cursor = 0
    for batch in corpus.scanner(columns=[column], batch_size=chunk_size).to_batches():
        if jobs.stop_requested():
            raise KeyboardInterrupt(
                "Stop requested mid _read_corpus_column; aborting cleanly."
            )
        batch_end = row_offset + batch.num_rows
        while pos_cursor < n and sorted_positions[pos_cursor] < row_offset:
            pos_cursor += 1
        next_cursor = pos_cursor
        while next_cursor < n and sorted_positions[next_cursor] < batch_end:
            next_cursor += 1
        if next_cursor > pos_cursor:
            local_idx = sorted_positions[pos_cursor:next_cursor] - row_offset
            picked = batch.column(column).take(
                pa.array(local_idx, type=pa.int64())
            ).to_pylist()
            for k, caller_i in enumerate(order[pos_cursor:next_cursor]):
                out[int(caller_i)] = picked[k]
            pos_cursor = next_cursor
        row_offset = batch_end

    if pos_cursor < n:
        raise IndexError(
            f"position {int(sorted_positions[pos_cursor])} out of range for "
            f"corpus with {row_offset} rows"
        )
    return out


_ALL_DICT_COLUMNS: tuple[str, ...] = (
    "id", "display_name", "abstract", "type", "publication_date",
    "created_date", "updated_date", "language", "topic_label",
    "primary_topic_id", "primary_subfield_id", "referenced_works",
)


def _read_corpus_dicts(
    corpus: ds.Dataset,
    positions: NDArray[np.uint64],
    chunk_size: int = 500_000,
    columns: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Read full work dicts from the corpus at the given positions.

    Chunked over Dataset batches for the same reason as
    :func:`_read_corpus_column`. The ``abstract`` column at ~1 KB/work × 40 M
    retained works (~40 GB) is well past pyarrow's single-chunk string
    concatenation limit; without chunking, ``take(all positions)`` crashes
    with ``ArrowInvalid: offset overflow`` before any user code runs.

    ``columns`` selects which parquet columns to read (default: all 12).
    Restricting to a subset lets callers that only need a few fields
    (e.g., a denoiser that only consumes ``id`` + ``referenced_works``)
    skip the expensive ``abstract``/``display_name`` materialisation —
    the dominant memory cost of this function on a full-corpus call. The
    output dict has whichever top-level keys the requested columns can
    populate (e.g., ``"primary_topic"`` requires both
    ``primary_topic_id`` and ``primary_subfield_id``).

    Polls :func:`bookgap.jobs.stop_requested` between batches so a
    SIGTERM mid-read aborts cleanly instead of waiting for the kernel
    to escalate to SIGKILL after the chunked allocations have already
    exceeded available memory.

    Results are returned in the order of ``positions``.
    """
    cols = tuple(columns) if columns is not None else _ALL_DICT_COLUMNS
    cols_set = frozenset(cols)
    n = len(positions)
    if n == 0:
        return []

    positions_arr = np.asarray(positions, dtype=np.int64)
    order = np.argsort(positions_arr, kind="stable")
    sorted_positions = positions_arr[order]

    works: List[Optional[Dict[str, Any]]] = [None] * n
    row_offset = 0
    pos_cursor = 0
    for batch in corpus.scanner(columns=list(cols), batch_size=chunk_size).to_batches():
        if jobs.stop_requested():
            raise KeyboardInterrupt(
                "Stop requested mid _read_corpus_dicts; aborting cleanly."
            )
        batch_end = row_offset + batch.num_rows
        while pos_cursor < n and sorted_positions[pos_cursor] < row_offset:
            pos_cursor += 1
        next_cursor = pos_cursor
        while next_cursor < n and sorted_positions[next_cursor] < batch_end:
            next_cursor += 1
        if next_cursor > pos_cursor:
            local_idx = sorted_positions[pos_cursor:next_cursor] - row_offset
            sub = batch.take(pa.array(local_idx, type=pa.int64()))
            col = {c: sub.column(c).to_pylist() for c in cols}
            for k, caller_i in enumerate(order[pos_cursor:next_cursor]):
                d: Dict[str, Any] = {}
                if "id" in cols_set:
                    d["id"] = col["id"][k]
                if "display_name" in cols_set:
                    d["display_name"] = col["display_name"][k] or ""
                if "abstract" in cols_set:
                    d["abstract"] = col["abstract"][k] or ""
                if "type" in cols_set:
                    d["type"] = col["type"][k] or "UNKNOWN"
                if "publication_date" in cols_set:
                    pub = col["publication_date"][k]
                    if isinstance(pub, (date, datetime)):
                        pub = pub.isoformat()
                    d["publication_date"] = pub
                if "created_date" in cols_set:
                    created = col["created_date"][k]
                    if isinstance(created, (date, datetime)):
                        created = created.isoformat()
                    d["created_date"] = created
                if "updated_date" in cols_set:
                    updated = col["updated_date"][k]
                    if isinstance(updated, (date, datetime)):
                        updated = updated.isoformat()
                    d["updated_date"] = updated
                if "language" in cols_set:
                    d["language"] = col["language"][k]
                if "topic_label" in cols_set:
                    d["topic"] = col["topic_label"][k]
                if "primary_topic_id" in cols_set or "primary_subfield_id" in cols_set:
                    d["primary_topic"] = {
                        "id": col["primary_topic_id"][k] if "primary_topic_id" in cols_set else None,
                        "subfield": {
                            "id": col["primary_subfield_id"][k] if "primary_subfield_id" in cols_set else None
                        },
                    }
                if "referenced_works" in cols_set:
                    d["referenced_works"] = col["referenced_works"][k] or []
                works[int(caller_i)] = d
            pos_cursor = next_cursor
        row_offset = batch_end

    if pos_cursor < n:
        raise IndexError(
            f"position {int(sorted_positions[pos_cursor])} out of range for "
            f"corpus with {row_offset} rows"
        )
    # After the scan every slot is filled; narrow the optional for callers.
    return [w for w in works if w is not None]


@lru_cache(maxsize=1)
@TIMER.time()
def get_cloud_works(cloud: Cloud) -> List[Dict[str, Any]]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    return _read_corpus_dicts(corpus, sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_ids(cloud: Cloud) -> List[str]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    return _read_corpus_column(corpus, "id", sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_references(cloud: Cloud) -> List[List[str]]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    refs = _read_corpus_column(corpus, "referenced_works", sorted_idxs[retained_positions])
    return [r or [] for r in refs]


def _citation_adjacency_streamed(
    cloud: Cloud,
    add_self_loops: bool = True,
    normalize: bool = False,
    chunk_size: int = 500_000,
) -> sp.csr_matrix:
    """Build the citation adjacency matrix without materialising all
    referenced_works as Python strings at once.

    The naive path — ``_read_corpus_column(..., 'referenced_works')`` then
    ``build_citation_adjacency_from_ids`` — holds ~1 B Python reference
    strings in memory simultaneously (~100 GB at 40 M-work corpus scale),
    which OOM-killed the pipeline. This streamed version:

      - builds a ``{work_id: row_idx}`` dict once from the work_ids column
        (~4 GB at 40 M × ~50 bytes Python-string-plus-dict-overhead),
      - iterates over the corpus in row-group-scale chunks,
      - for each chunk, materialises only that chunk's referenced_works
        to Python (~250 MB at chunk_size=500k × 30 refs × ~100 bytes),
      - resolves references against the dict, appending to per-chunk
        int32 numpy arrays,
      - discards each chunk's Python objects before loading the next.

    Peak scratch beyond the id_to_idx dict: one chunk's referenced_works
    (~250 MB) + growing row/col int32 arrays (~6 GB at final size for
    1.6 B edges). Total peak ~11 GB, vs ~100 GB for the naive path.
    """
    _log_rss("before-citation-graph")
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)

    global_positions = sorted_idxs[retained_positions].astype(np.int64)
    n = len(global_positions)

    # Step 1: stream the "id" column in batches to build the id→row_idx
    # dict. We need the corpus-global row positions of retained works so we
    # can mask later batches by range; we don't materialise the full id
    # column up front (though at 40 M short strings it'd only be ~4 GB, the
    # referenced_works column on the same pass would be ~100 GB which we
    # definitely can't materialise).
    import gc
    id_to_idx: Dict[str, int] = {}
    row_offset = 0
    scanner = corpus.scanner(columns=["id"], batch_size=chunk_size)
    for batch in scanner.to_batches():
        batch_end = row_offset + batch.num_rows
        # Find retained global positions falling in this batch range.
        in_batch = np.asarray(
            (global_positions >= row_offset) & (global_positions < batch_end)
        ).nonzero()[0]
        if in_batch.size:
            globals_here = global_positions[in_batch]
            local_idx = (globals_here - row_offset).astype(np.int64)
            ids_py = batch.column("id").take(pa.array(local_idx, type=pa.int64())).to_pylist()
            for k, wid in enumerate(ids_py):
                id_to_idx[wid] = int(in_batch[k])
        row_offset = batch_end
    del scanner
    _log_rss("citation-graph:after-id-map")

    # Step 2: stream referenced_works in the same batched fashion, never
    # materialising more than ``chunk_size`` rows' worth of Python string
    # lists at once.
    row_chunks: list[NDArray[np.int32]] = []
    col_chunks: list[NDArray[np.int32]] = []
    row_offset = 0
    batch_idx = 0
    scanner = corpus.scanner(columns=["referenced_works"], batch_size=chunk_size)
    for batch in scanner.to_batches():
        batch_end = row_offset + batch.num_rows
        in_batch = np.asarray(
            (global_positions >= row_offset) & (global_positions < batch_end)
        ).nonzero()[0]
        if in_batch.size:
            globals_here = global_positions[in_batch]
            local_idx = (globals_here - row_offset).astype(np.int64)
            refs_py = (
                batch.column("referenced_works")
                .take(pa.array(local_idx, type=pa.int64()))
                .to_pylist()
            )
            # Bound-preallocate per-sub-chunk row/col buffers.
            max_edges = in_batch.size + sum(len(r or ()) for r in refs_py)
            row_buf = np.empty(max_edges, dtype=np.int32)
            col_buf = np.empty(max_edges, dtype=np.int32)
            cursor = 0
            for k, refs in enumerate(refs_py):
                my_row = int(in_batch[k])
                if add_self_loops:
                    row_buf[cursor] = my_row
                    col_buf[cursor] = my_row
                    cursor += 1
                if refs:
                    for ref_id in refs:
                        target = id_to_idx.get(ref_id)
                        if target is not None:
                            row_buf[cursor] = my_row
                            col_buf[cursor] = target
                            cursor += 1
            row_chunks.append(row_buf[:cursor])
            col_chunks.append(col_buf[:cursor])
            del refs_py, row_buf, col_buf
        row_offset = batch_end
        batch_idx += 1
        if batch_idx % 10 == 0:
            _log_rss(f"citation-graph:batch-{batch_idx}")
    del scanner, id_to_idx
    gc.collect()

    row = np.concatenate(row_chunks)
    col = np.concatenate(col_chunks)
    del row_chunks, col_chunks
    _log_rss("citation-graph:after-chunks")

    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    del row, col, data
    _log_rss("citation-graph:after-csr")

    if normalize:
        adj = sk_normalize(adj, norm="l1", axis=1, copy=False)

    LOGGER.info(
        f"Built citation adjacency: {n} works, {adj.nnz} edges "
        f"(density: {adj.nnz / (n * n):.2e})"
    )
    return adj


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_citation_graph(cloud: Cloud, normalize: bool) -> sp.csr_matrix:
    """Return the citation adjacency/transition matrix for the cloud."""
    adj = _citation_adjacency_streamed(
        cloud, add_self_loops=True, normalize=normalize
    )
    LOGGER.info(f"Cloud {cloud.name()} citation graph: shape={adj.shape}, nnz={adj.nnz}")
    return adj


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_titles(cloud: Cloud) -> List[str]:
    """Get display names/titles for all works in the cloud.

    Memory-heavy at full-corpus scale: materialises every retained title as a
    Python string, and the intermediate ``table.take()`` concatenation hits
    pyarrow's 32-bit StringArray offset limit (~2 GB) once the combined
    ``display_name`` bytes exceed it. Callers that only need a handful of
    titles should use :func:`get_cloud_work_titles_subset` instead.
    """
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    return _read_corpus_column(corpus, "display_name", sorted_idxs[retained_positions])


def get_cloud_work_titles_subset(
    cloud: Cloud, subset_indices: NDArray[np.uint64]
) -> List[str]:
    """Read display_name for an arbitrary subset of retained works.

    ``subset_indices`` are positions in the same retained-subset space as
    the result of :meth:`Cloud.work_titles` (i.e. 0..N_retained-1). Returns
    titles in the order given. Not memoised — array-valued args don't cache
    usefully and the selective read is fast.
    """
    _, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    global_positions = sorted_idxs[retained_positions[subset_indices]]
    return _read_corpus_column(corpus, "display_name", global_positions)


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_abstracts(cloud: Cloud) -> List[str]:
    """Get abstracts for all works in the cloud."""
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    return _read_corpus_column(corpus, "abstract", sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_publication_dates(cloud: Cloud) -> List[str]:
    """Get publication dates for all works in the cloud."""
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    dates = _read_corpus_column(corpus, "publication_date", sorted_idxs[retained_positions])
    return [d.isoformat() if isinstance(d, (date, datetime)) else d for d in dates]


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_types(cloud: Cloud) -> List[str]:
    """Get OpenAlex ``type`` (book / article / preprint / …) for all works.

    Note: missing ``type`` values come through as Python ``None`` here,
    unlike the full-dict path :func:`get_cloud_works`, where
    :func:`_read_corpus_dicts` rewrites ``None`` to the sentinel string
    ``"UNKNOWN"``. Callers comparing against concrete WorkType strings
    (``"book"``, ``"article"``, ``"preprint"``) get the same behavior in
    both paths — neither ``None`` nor ``"UNKNOWN"`` matches — but anyone
    depending on the sentinel literal would be surprised.
    """
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    return _read_corpus_column(corpus, "type", sorted_idxs[retained_positions])


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
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)
    # Allocates exactly one N_retained × D output array (~62 GB on the
    # domain=3 corpus); the store itself is only memory-mapped.
    _log_rss("before-streamed-embeddings-read")
    retained_embeddings = read_embeddings_rows(emb_path, retained_positions)
    _log_rss("after-streamed-embeddings-read")
    return denoise(cloud, retained_embeddings)


def cloud_embeddings_store_positions(cloud: Cloud) -> tuple[Path, NDArray[np.uint64]]:
    """Return ``(emb_path, retained_positions)`` for *cloud*.

    *retained_positions* are ascending row positions into the embeddings
    store at *emb_path* (post-dedup, post-filter). Row ``i`` of
    ``cloud.embeddings()`` corresponds to store row ``retained_positions[i]``
    (the parent's denoising is applied row-wise afterwards). Exposed so a
    :class:`~bookgap.subcloud.SubCloud` can read only its own rows without
    ever materialising the full parent array.
    """
    corpus_fp, _sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)
    return emb_path, retained_positions


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_projection(cloud: Cloud) -> ProjectiveDenoiserProjection:
    denoiser_enum = WorkDenoisers[cloud.denoiser]
    denoiser_impl = denoiser_enum.value

    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date)
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)

    # Honor the denoiser's declaration of which work-dict columns it
    # actually consumes. NoneDenoiser declares the empty set → skip the
    # whole _read_corpus_dicts materialisation (saves ~40 GB of Python
    # strings + ~30 GB transient pyarrow buffers on a 40M-retained-work
    # corpus). Other denoisers default to None = "all columns" until
    # they're audited and tightened.
    required_cols = getattr(denoiser_impl, "required_corpus_columns", None)
    if required_cols is not None and len(required_cols) == 0:
        retained_works: List[Dict[str, Any]] = []
    else:
        retained_works = _read_corpus_dicts(
            corpus, sorted_idxs[retained_positions], columns=required_cols,
        )
    retained_embeddings = read_embeddings_rows(emb_path, retained_positions)

    return denoiser_impl.compute_projection_artifact(cloud.drop_k, retained_works, retained_embeddings)

@TIMER.time(formatters=dict(embeddings=lambda embeddings: f"<{embeddings.shape}>"))
@MEMORY.cache(ignore=["embeddings"]) # embeddings are ignored by the cache key because hashing is too slow. Use embeddings_key to disambiguate.
def deduplicate_works_semantic_(deduplicator: Optional[Deduplicator], index_info: IndexInfo, embeddings: NDArray, embeddings_key: tuple) -> NDArray[np.uint64]:
    if deduplicator is None:
        return np.arange(len(embeddings), dtype=np.uint64) # Retain all indices
    return deduplicator.deduplicate(embeddings, index_info)

def _deduplicate_semantic(
    cloud: Cloud,
    corpus: ds.Dataset,
    embeddings: NDArray[np.float32],
    sorted_idxs: NDArray[np.uint64],
) -> NDArray[np.uint64]:
    """Run semantic dedup and log stats.  Returns positions into the embeddings array."""
    LOGGER.info(
        f"Cloud {cloud.name()} -- Applying deduplication to {len(embeddings)} "
        f"embeddings with {cloud.deduplicator}"
    )
    retained_positions = deduplicate_works_semantic_(
        cloud.deduplicator,
        cloud.index_info,
        embeddings,
        embeddings_key=(cloud.filter_pair.pre, cloud.max_works, cloud.embedder),
    )

    # Log dedup stats by work type
    artifacts_dir = SCOPES_DIR / str(cloud.filter_pair.pre)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Read only the type column for the sorted subset using take()
    type_table = corpus.to_table(columns=["type"])
    subset_types = type_table.take(pa.array(sorted_idxs, type=pa.int64())).column("type").to_pylist()

    original_counts: dict[str, int] = defaultdict(int)
    for wt in subset_types:
        original_counts[wt or "UNKNOWN"] += 1
    new_counts: dict[str, int] = defaultdict(int)
    for pos in retained_positions:
        new_counts[subset_types[int(pos)] or "UNKNOWN"] += 1

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
            new_count = new_counts.get(work_type, 0)
            f.write(f"{work_type}: {original_count} -> {new_count} ({new_count / original_count:.4f})\n")
    with path.open("r") as f:
        LOGGER.info(f"Deduplication stats:\n{f.read()}")
    return retained_positions


def _post_filter_mask(
    corpus: ds.Dataset,
    post_filter: Filter,
    corpus_row_idxs: NDArray[np.uint64],
) -> NDArray[np.bool_]:
    """Boolean mask over *corpus_row_idxs* where post_filter passes."""
    if not post_filter:
        return np.ones(len(corpus_row_idxs), dtype=bool)
    if post_filter.field is not None or post_filter.domain is not None:
        raise NotImplementedError(
            "post_filter with field or domain constraints is not supported "
            "with per-scope Parquet storage. Use pre-filter instead."
        )
    full_table = corpus.to_table(columns=["type", "primary_topic_id", "primary_subfield_id"])
    table = full_table.take(pa.array(corpus_row_idxs, type=pa.int64()))
    types = table.column("type").to_pylist()
    topic_ids = table.column("primary_topic_id").to_pylist()
    subfield_ids = table.column("primary_subfield_id").to_pylist()

    mask = []
    for i in range(len(types)):
        work = {
            "type": types[i],
            "primary_topic": {"id": topic_ids[i], "subfield": {"id": subfield_ids[i]}},
        }
        mask.append(post_filter.check_work(work))
    return np.array(mask, dtype=bool)

T = TypeVar('T', bound=np.generic)

def _centered_singular_values(X: NDArray) -> NDArray:
    """All singular values of (X - X.mean(axis=0)) in descending order,
    without materialising the centered matrix or running a full LAPACK gesdd.

    Uses the identity (X_c)^T (X_c) = X^T X - n μ μ^T, then takes the
    eigenvalues of that d×d Gram matrix. For our shape (40M × 384) this
    is one BLAS matmul (~30 s, reads X once) plus a microsecond 384×384
    eigh. The old ``svdvals(X - X.mean(0))`` path allocated a 62 GB
    centered copy AND a 62 GB LAPACK workspace on top of the 62 GB input —
    cumulative ~200 GB, reliably OOM-killed the process. The returned
    array has ``min(X.shape[1], X.shape[0])`` entries so callers that
    previously used the full vector (e.g. for total-variance ratios) work
    unchanged.
    """
    n, d = X.shape
    mean = X.mean(axis=0)
    gram = X.T @ X
    gram -= n * np.outer(mean, mean)
    # Symmetrise to kill asymmetric float-noise from the matmul;
    # eigvalsh on a slightly-non-symmetric matrix silently misbehaves.
    gram = 0.5 * (gram + gram.T)
    eigvals = scipy.linalg.eigvalsh(gram)  # ascending
    eigvals = np.clip(eigvals[::-1], 0.0, None)  # descending, non-negative
    return np.sqrt(eigvals)


def denoise(cloud: Cloud, embeddings: NDArray[T]) -> NDArray[T]:
    """Apply projective denoising with full orchestration."""
    if len(embeddings) == 0:
        # An empty subset (e.g. a topic left with no books after dedup) has no
        # spectrum to log, and the diagnostic below would eigensolve a NaN Gram.
        return get_cloud_projection(cloud).apply(embeddings)

    # Baseline singular values (top-10) for diagnostic logging. Uses the
    # Gram-matrix trick so memory is O(d²) not O(nd) in scratch.
    s_orig = _centered_singular_values(embeddings)
    LOGGER.info(f"Top 10 original singular values: {' '.join(f'{s:.3f}' for s in s_orig[:10])}")

    # Dimensionality reduction/cleaning

    projection = get_cloud_projection(cloud)

    LOGGER.info(f"Cloud {cloud.name()} -- Running {cloud.denoiser} with drop_k={cloud.drop_k} (n_components={embeddings.shape[1] - cloud.drop_k}) on {len(embeddings)} embeddings of shape {embeddings.shape}")

    denoised_embeddings: NDArray[T] = projection.apply(embeddings)

    # What the projection keeps, measured before renormalizing: the projection is
    # orthonormal, so this is at most 100%. Measured after, it is not a fraction at
    # all, since renormalizing scales each vector back up by 1/||projection|| (it
    # read ~250% for GIST, whose dropped directions carry much of every vector's
    # length). The mean norm kept is what that scaling undoes.
    variance_retained = np.sum(_centered_singular_values(denoised_embeddings) ** 2) / np.sum(s_orig ** 2)
    norm_retained = float(np.mean(np.linalg.norm(denoised_embeddings, axis=1) / np.linalg.norm(embeddings, axis=1)))
    LOGGER.info(f"Cloud {cloud.name()} -- Projection keeps {variance_retained:.2%} of centered variance, "
                f"{norm_retained:.2%} of the mean embedding norm")

    if cloud.normalize_embeddings:
        norms = np.linalg.norm(denoised_embeddings, axis=1, keepdims=True)
        denoised_embeddings = denoised_embeddings / norms

    return denoised_embeddings