from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
import enum
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
from bookgap.cloud_denoiser import ProjectiveDenoiserProjection, WorkDenoisers
import re
from typing import Any, Dict, List, Optional, TypeVar
import faiss
import numpy as np
from numpy.typing import NDArray
import pyalex
import pyarrow.dataset as ds
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
from bookgap.deduplicators import Deduplicator
import scipy.linalg
import pyarrow as pa
import pyarrow.parquet as pq

from bookgap.dendrogram import DendrogramResult, get_cloud_dendrogram

from bookgap.openalex_ingest import (
    _resolve_filter_scopes,
    corpus_dataset,
    embed_works,
)

pyalex.config.email = "gchurchill@cs.stonybrook.edu"
pyalex.config.api_key = os.environ["OPENALEX_API_KEY"]


def _embeddings_store_path(corpus_fingerprint: str, embedder_cfg: Any) -> Path:
    """Path for the embeddings Parquet file."""
    embedder_fp = hashlib.sha256(
        json.dumps(asdict(embedder_cfg), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return CACHE_DIR / "embeddings_parquet" / corpus_fingerprint / embedder_fp / "embeddings.parquet"


def write_embeddings_store(path: Path, embeddings: NDArray[np.float32]) -> None:
    """Write embeddings to a Parquet file with zstd compression."""
    n = embeddings.shape[0]
    d = embeddings.shape[1] if embeddings.ndim == 2 else 0

    if n > 0:
        flat = pa.array(embeddings.reshape(-1).astype(np.float32), type=pa.float32())
        emb_col = pa.FixedSizeListArray.from_arrays(flat, list_size=d)
    else:
        # Empty: create a typed but zero-length column
        emb_type = pa.list_(pa.float32(), d) if d > 0 else pa.list_(pa.float32())
        emb_col = pa.array([], type=emb_type)

    table = pa.table({
        "row_id": pa.array(range(n), type=pa.int64()),
        "embeddings": emb_col,
    })
    metadata = {b"bookgap:embedding_dim": str(d).encode()}
    table = table.replace_schema_metadata(metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def read_embeddings_store(path: Path) -> NDArray[np.float32]:
    """Read embeddings from a Parquet file."""
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    d = int(metadata.get(b"bookgap:embedding_dim", b"0").decode())
    n = len(table)
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    arr = table.column("embeddings").combine_chunks().values
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(n, d)


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
def ensure_cloud_data(cloud: Cloud) -> tuple[str, NDArray[np.uint64], NDArray[np.uint64]]:
    """Ensure corpus is fetched, embeddings computed, and retained indices ready.

    Returns ``(corpus_fingerprint, sorted_idxs, retained_positions)`` where
    *sorted_idxs* are physical corpus-row indices for the date-sorted subset,
    and *retained_positions* are positions **into that subset** (and into the
    embeddings file) after post-filtering and semantic dedup.
    """
    corpus_fp = _corpus_fingerprint(cloud.filter_pair.pre, cloud.max_works)
    corpus = corpus_dataset(cloud.filter_pair.pre)

    # 1. Date-sort + max_works truncation -> sorted_idxs (corpus row positions)
    sorted_idxs = _sorted_corpus_idxs(corpus, cloud.max_works)
    LOGGER.info(f"Cloud {cloud.name()} -- {len(sorted_idxs)} works after date-sort/truncation")

    # 2. Compute and store embeddings for the sorted subset
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)
    if not emb_path.exists():
        _compute_and_store_embeddings(cloud, corpus, sorted_idxs, emb_path)

    # 3. Semantic dedup (operates on the full sorted subset embeddings)
    raw_embeddings = read_embeddings_store(emb_path)
    dedup_positions = _deduplicate_semantic(cloud, corpus, raw_embeddings)
    LOGGER.info(f"Retained {len(dedup_positions)} works after deduplication")

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
        embeddings = _embed_sentence_transformer(cloud.filter_pair.pre, cloud.embedder, sorted_idxs)
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
) -> NDArray[np.float32]:
    """Sentence-transformer path: per-scope batch caching via embed_works.

    Returns embeddings re-ordered to match *sorted_idxs*.
    """
    model_name = Embedder.ModelName(embedder_cfg.model_name)

    # Collect per-scope works + embeddings in corpus-file order
    all_works: list[dict[str, Any]] = []
    emb_chunks: list[NDArray] = []

    with tqdm.tqdm(_resolve_filter_scopes(work_filter), desc="Scopes", delay=3, mininterval=3) as scope_bar:
        for scope in scope_bar:
            scope_bar.set_postfix_str(str(scope))
            scope_works, scope_embeddings = embed_works(scope, model_name)
            all_works.extend(scope_works)
            emb_chunks.append(scope_embeddings)

    if emb_chunks:
        all_embeddings = np.concatenate(emb_chunks, axis=0)
    else:
        all_embeddings = np.empty((0, Embedder(model_name=model_name).embedding_dimension()), dtype=np.float32)

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


def _read_corpus_column(corpus: ds.Dataset, column: str, positions: NDArray[np.uint64]) -> List:
    """Read a single column from the corpus at the given positions."""
    table = corpus.to_table(columns=[column])
    subset = table.take(pa.array(positions, type=pa.int64()))
    return subset.column(column).to_pylist()


def _read_corpus_dicts(corpus: ds.Dataset, positions: NDArray[np.uint64]) -> List[Dict[str, Any]]:
    """Read full work dicts from the corpus at the given positions."""
    columns = [
        "id", "display_name", "abstract", "type", "publication_date",
        "created_date", "updated_date", "language", "topic_label",
        "primary_topic_id", "primary_subfield_id", "referenced_works",
    ]
    full_table = corpus.to_table(columns=columns)
    table = full_table.take(pa.array(positions, type=pa.int64()))

    ids = table.column("id").to_pylist()
    display_names = table.column("display_name").to_pylist()
    abstracts = table.column("abstract").to_pylist()
    types = table.column("type").to_pylist()
    pub_dates = table.column("publication_date").to_pylist()
    created_dates = table.column("created_date").to_pylist()
    updated_dates = table.column("updated_date").to_pylist()
    languages = table.column("language").to_pylist()
    topic_labels = table.column("topic_label").to_pylist()
    primary_topic_ids = table.column("primary_topic_id").to_pylist()
    primary_subfield_ids = table.column("primary_subfield_id").to_pylist()
    references = table.column("referenced_works").to_pylist()

    works: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        pub = pub_dates[i]
        if isinstance(pub, (date, datetime)):
            pub = pub.isoformat()
        created = created_dates[i]
        if isinstance(created, (date, datetime)):
            created = created.isoformat()
        updated = updated_dates[i]
        if isinstance(updated, (date, datetime)):
            updated = updated.isoformat()
        works.append({
            "id": ids[i],
            "display_name": display_names[i] or "",
            "abstract": abstracts[i] or "",
            "type": types[i] or "UNKNOWN",
            "publication_date": pub,
            "created_date": created,
            "updated_date": updated,
            "language": languages[i],
            "topic": topic_labels[i],
            "primary_topic": {
                "id": primary_topic_ids[i],
                "subfield": {"id": primary_subfield_ids[i]},
            },
            "referenced_works": references[i] or [],
        })
    return works


@lru_cache(maxsize=1)
@TIMER.time()
def get_cloud_works(cloud: Cloud) -> List[Dict[str, Any]]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    return _read_corpus_dicts(corpus, sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_ids(cloud: Cloud) -> List[str]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    return _read_corpus_column(corpus, "id", sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_references(cloud: Cloud) -> List[List[str]]:
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    refs = _read_corpus_column(corpus, "referenced_works", sorted_idxs[retained_positions])
    return [r or [] for r in refs]


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
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    return _read_corpus_column(corpus, "display_name", sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_abstracts(cloud: Cloud) -> List[str]:
    """Get abstracts for all works in the cloud."""
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    return _read_corpus_column(corpus, "abstract", sorted_idxs[retained_positions])


@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_publication_dates(cloud: Cloud) -> List[str]:
    """Get publication dates for all works in the cloud."""
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    dates = _read_corpus_column(corpus, "publication_date", sorted_idxs[retained_positions])
    return [d.isoformat() if isinstance(d, (date, datetime)) else d for d in dates]


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
    all_embeddings = read_embeddings_store(emb_path)
    return denoise(cloud, all_embeddings[retained_positions])


@lru_cache(maxsize=1)
@TIMER.time()
@MEMORY.cache
def get_cloud_projection(cloud: Cloud) -> ProjectiveDenoiserProjection:
    denoiser_enum = WorkDenoisers[cloud.denoiser]
    denoiser_impl = denoiser_enum.value

    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    corpus = corpus_dataset(cloud.filter_pair.pre)
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)

    retained_works = _read_corpus_dicts(corpus, sorted_idxs[retained_positions])
    all_embeddings = read_embeddings_store(emb_path)
    retained_embeddings = all_embeddings[retained_positions]

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
    artifacts_dir = ARTIFACTS_DIR / str(cloud.filter_pair.pre)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Only read the "type" column we need for stats
    all_types = corpus.to_table(columns=["type"]).column("type").to_pylist()
    # sorted_idxs aren't directly available here; compute from ensure_cloud_data's
    # cached sorted_idxs via _sorted_corpus_idxs.  But we need sorted_idxs to map
    # positions back to corpus rows.  To avoid recomputation, we accept that
    # the embeddings array is 1:1 with the sorted subset and the caller passes
    # the corpus.  We reconstruct sorted_idxs here (cheap, only reads dates).
    sorted_idxs = _sorted_corpus_idxs(corpus, cloud.max_works)

    original_counts: dict[str, int] = defaultdict(int)
    for idx in sorted_idxs:
        wt = all_types[int(idx)] or "UNKNOWN"
        original_counts[wt] += 1
    new_counts: dict[str, int] = defaultdict(int)
    for pos in retained_positions:
        wt = all_types[int(sorted_idxs[int(pos)])] or "UNKNOWN"
        new_counts[wt] += 1

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