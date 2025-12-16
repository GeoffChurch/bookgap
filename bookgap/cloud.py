from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import enum
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import os
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
import scipy.sparse as sp
from bookgap.util import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    FilterPair,
    Filter,
    TIMER,
    MEMORY,
    LOGGER,
    create_unicode_safe_filename,
    get_stats,
    build_citation_adjacency_from_ids,
)
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
    
    def work_titles(self) -> List[str]:
        return get_cloud_work_titles(self)
    
    def work_abstracts(self) -> List[str]:
        return get_cloud_work_abstracts(self)
    
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
    ) -> "DendrogramResult":
        return get_cloud_dendrogram(
            self,
            linkage_method=linkage_method,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
        )

    def name(self) -> str:
        return f"{self.filter_pair.merged_name()}"


@dataclass(frozen=True)
class DendrogramResult:
    """Result of dendrogram building.

    This mirrors scipy's linkage conventions:
    - leaves are indexed 0..n_leaf_clusters-1
    - internal nodes are indexed n_leaf_clusters..(2*n_leaf_clusters-2)
    - linkage has shape (n_leaf_clusters-1, 4) when n_leaf_clusters > 1

    For bisecting-kmeans, leaves may represent multi-point clusters. Use
    leaf_cluster_assignments to aggregate point masses before dendrogram ops.
    """

    linkage: NDArray[np.floating]
    leaf_cluster_assignments: NDArray[np.intp]
    n_leaf_clusters: int


def _dendrogram_cache_fingerprint(cloud: Cloud, linkage_method: str) -> str:
    # Keep this stable and human-inspectable; order matters.
    payload = {
        "cloud": {
            "filter_pre": str(cloud.filter_pair.pre),
            "filter_post": str(cloud.filter_pair.post),
            "max_works": cloud.max_works,
            "deduplicator": repr(cloud.deduplicator),
            "index_info": repr(cloud.index_info),
            "normalize_embeddings": cloud.normalize_embeddings,
            "denoiser": cloud.denoiser,
            "drop_k": cloud.drop_k,
            "embedder_model": cloud.embedder_model.value,
        },
        "linkage_method": linkage_method,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _dendrogram_cache_dir(cloud: Cloud, linkage_method: str):
    fingerprint = _dendrogram_cache_fingerprint(cloud, linkage_method)
    h = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return (CACHE_DIR / "dendrograms" / h), fingerprint


def _max_depth_subsumes(cached: Optional[int], requested: Optional[int]) -> bool:
    if requested is None:
        return cached is None
    if cached is None:
        return True
    return cached >= requested


def _depth_value(depth: Optional[int]) -> float:
    return float("inf") if depth is None else float(depth)


def _parse_cached_dendrogram_filename(name: str) -> Optional[tuple[int, Optional[int]]]:
    # Expected: dendrogram_min=<int>_max=<int|none>.npz
    if not (name.startswith("dendrogram_min=") and name.endswith(".npz")):
        return None
    core = name[: -len(".npz")]
    parts = core.split("_")
    if len(parts) != 2:
        return None
    try:
        min_part = parts[0]
        max_part = parts[1]
        min_cluster_size = int(min_part.split("=", 1)[1])
        raw_max = max_part.split("=", 1)[1]
        if raw_max.lower() == "none":
            max_depth: Optional[int] = None
        else:
            max_depth = int(raw_max)
        return min_cluster_size, max_depth
    except Exception:
        return None


def _truncate_bisecting_kmeans_dendrogram(
    *,
    linkage: NDArray[np.floating],
    leaf_cluster_assignments: NDArray[np.intp],
    n_leaf_clusters: int,
    min_cluster_size: int,
    max_depth: Optional[int],
) -> DendrogramResult:
    """Derive a coarser bisecting-kmeans dendrogram view without re-running k-means.

    This does NOT write anything back to disk.
    """
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    # Degenerate: a single leaf cluster
    if n_leaf_clusters <= 1 or linkage.shape[0] == 0:
        return DendrogramResult(
            linkage=np.zeros((0, 4), dtype=np.float64),
            leaf_cluster_assignments=np.zeros(len(leaf_cluster_assignments), dtype=np.intp),
            n_leaf_clusters=1 if len(leaf_cluster_assignments) > 0 else 0,
        )

    n_merges = int(linkage.shape[0])
    total_nodes = n_leaf_clusters + n_merges
    root = n_leaf_clusters + n_merges - 1

    # Node metadata (leaves + internal)
    left_child = np.full(total_nodes, -1, dtype=np.int64)
    right_child = np.full(total_nodes, -1, dtype=np.int64)
    node_dist = np.zeros(total_nodes, dtype=np.float64)
    node_size = np.zeros(total_nodes, dtype=np.int64)

    # Leaf sizes from assignments
    leaf_sizes = np.bincount(leaf_cluster_assignments.astype(np.int64), minlength=n_leaf_clusters).astype(np.int64)
    node_size[:n_leaf_clusters] = leaf_sizes

    parent = np.full(total_nodes, -1, dtype=np.int64)

    for i in range(n_merges):
        parent_id = n_leaf_clusters + i
        c1 = int(linkage[i, 0])
        c2 = int(linkage[i, 1])
        left_child[parent_id] = c1
        right_child[parent_id] = c2
        node_dist[parent_id] = float(linkage[i, 2])
        node_size[parent_id] = int(linkage[i, 3])
        parent[c1] = parent_id
        parent[c2] = parent_id

    # Depth from root (root=0)
    depth = np.full(total_nodes, -1, dtype=np.int64)
    depth_stack: list[tuple[int, int]] = [(root, 0)]
    while depth_stack:
        node, d = depth_stack.pop()
        depth[node] = d
        if node >= n_leaf_clusters:
            depth_stack.append((int(left_child[node]), d + 1))
            depth_stack.append((int(right_child[node]), d + 1))

    # Identify cut nodes (disjoint) via preorder traversal
    cut_nodes: list[int] = []
    cut_set = np.zeros(total_nodes, dtype=bool)
    cut_stack: list[int] = [root]
    while cut_stack:
        node = int(cut_stack.pop())
        if node < n_leaf_clusters:
            cut_set[node] = True
            cut_nodes.append(node)
            continue
        if node_size[node] <= min_cluster_size:
            cut_set[node] = True
            cut_nodes.append(node)
            continue
        if max_depth is not None and depth[node] >= max_depth:
            cut_set[node] = True
            cut_nodes.append(node)
            continue
        cut_stack.append(int(right_child[node]))
        cut_stack.append(int(left_child[node]))

    # Assign new leaf ids to cut nodes
    cut_node_to_leaf_id = np.full(total_nodes, -1, dtype=np.int64)
    for new_leaf_id, node in enumerate(cut_nodes):
        cut_node_to_leaf_id[node] = new_leaf_id
    n_new_leaves = len(cut_nodes)

    # Map fine leaf cluster -> new leaf id by climbing to nearest cut ancestor
    fine_leaf_to_new_leaf = np.full(n_leaf_clusters, -1, dtype=np.int64)
    for leaf_id in range(n_leaf_clusters):
        node = leaf_id
        while cut_node_to_leaf_id[node] < 0:
            node = int(parent[node])
            if node < 0:
                raise RuntimeError("Failed to find cut ancestor for leaf")
        fine_leaf_to_new_leaf[leaf_id] = int(cut_node_to_leaf_id[node])
    new_leaf_cluster_assignments = fine_leaf_to_new_leaf[leaf_cluster_assignments.astype(np.int64)].astype(np.intp)

    # Build contracted linkage (postorder) for the induced tree above cuts
    linkage_rows: list[list[float]] = []
    next_internal_id = [n_new_leaves]

    def build(node: int) -> int:
        leaf_id = int(cut_node_to_leaf_id[node])
        if leaf_id >= 0:
            return leaf_id
        if node < n_leaf_clusters:
            # Shouldn't happen (leaves always cut), but be defensive.
            raise RuntimeError("Encountered uncropped leaf")
        left_id = build(int(left_child[node]))
        right_id = build(int(right_child[node]))
        if left_id == right_id:
            return left_id
        my_id = next_internal_id[0]
        next_internal_id[0] += 1
        linkage_rows.append([float(left_id), float(right_id), float(node_dist[node]), float(node_size[node])])
        return my_id

    build(root)

    if n_new_leaves <= 1 or not linkage_rows:
        new_linkage = np.zeros((0, 4), dtype=np.float64)
    else:
        new_linkage = np.asarray(linkage_rows, dtype=np.float64)

    return DendrogramResult(
        linkage=new_linkage,
        leaf_cluster_assignments=new_leaf_cluster_assignments,
        n_leaf_clusters=n_new_leaves,
    )


@TIMER.time()
def get_cloud_dendrogram(
    cloud: Cloud,
    *,
    linkage_method: str,
    min_cluster_size: int,
    max_depth: Optional[int],
) -> DendrogramResult:
    """Get a dendrogram for this cloud.

    Caching/subsumption is currently implemented only for linkage_method="bisecting_kmeans".
    """
    if linkage_method != "bisecting_kmeans":
        raise ValueError("Cloud dendrogram caching is only supported for linkage_method='bisecting_kmeans'")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    cache_dir, fingerprint = _dendrogram_cache_dir(cloud, linkage_method)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_enabled = True

    # Sanity check (collision guard)
    sanity_path = cache_dir / "sanity.pkl"
    if sanity_path.exists():
        try:
            import pickle

            with sanity_path.open("rb") as f:
                saved = pickle.load(f)
            if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
                LOGGER.warning(
                    "Dendrogram cache sanity mismatch (possible hash collision); ignoring cached entries"
                )
                cache_enabled = False
        except Exception as e:
            LOGGER.warning(f"Failed reading dendrogram cache sanity file: {e}")
            cache_enabled = False
    else:
        try:
            import pickle

            tmp = sanity_path.with_name(sanity_path.name + ".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"fingerprint": fingerprint}, f)
            os.replace(tmp, sanity_path)
        except Exception as e:
            LOGGER.warning(f"Failed writing dendrogram cache sanity file: {e}")
            cache_enabled = False

    # Enumerate cached dendrograms
    entries: list[tuple[int, Optional[int], Any]] = []  # (min_cluster_size, max_depth, path)
    if cache_enabled:
        for p in cache_dir.glob("dendrogram_min=*_max=*.npz"):
            parsed = _parse_cached_dendrogram_filename(p.name)
            if parsed is None:
                continue
            cached_min, cached_max = parsed
            entries.append((cached_min, cached_max, p))

    def load_npz(path):
        with np.load(path) as data:
            linkage = data["linkage"].astype(np.float64, copy=False)
            leaf_cluster_assignments = data["leaf_cluster_assignments"].astype(np.intp, copy=False)
            n_leaf_clusters = int(data["n_leaf_clusters"])
        return DendrogramResult(linkage=linkage, leaf_cluster_assignments=leaf_cluster_assignments, n_leaf_clusters=n_leaf_clusters)

    # Exact match first
    requested_name = f"dendrogram_min={min_cluster_size}_max={'none' if max_depth is None else max_depth}.npz"
    exact_path = cache_dir / requested_name
    if cache_enabled and exact_path.exists():
        cached = load_npz(exact_path)
        LOGGER.info(
            f"Dendrogram cache hit (exact): {exact_path} -> n_leaf_clusters={cached.n_leaf_clusters}"
        )
        return cached

    # Subsuming match: cached_min <= requested_min and cached_max subsumes requested_max
    subsuming: list[tuple[int, Optional[int], Any]] = [
        (cmin, cmax, path)
        for (cmin, cmax, path) in entries
        if cmin <= min_cluster_size and _max_depth_subsumes(cmax, max_depth)
    ]

    if cache_enabled and subsuming:
        # Prefer the closest (least refined) subsumer to reduce load/processing.
        subsuming.sort(key=lambda t: (-t[0], _depth_value(t[1])))
        cached_min, cached_max, path = subsuming[0]
        cached = load_npz(path)
        LOGGER.info(
            "Dendrogram cache hit (subsuming): "
            f"{path.name} (min={cached_min}, max={cached_max}, n_leaf_clusters={cached.n_leaf_clusters}) "
            f"-> requested (min={min_cluster_size}, max={max_depth})"
        )
        if cached_min == min_cluster_size and cached_max == max_depth:
            return cached
        # NOTE: We intentionally do NOT persist this derived truncation; in principle we
        # could garbage-collect cached entries subsumed by others, but we keep it simple.
        # TODO: Is it better to truncate or to expect the user to short-circuit traversal?
        derived = _truncate_bisecting_kmeans_dendrogram(
            linkage=cached.linkage,
            leaf_cluster_assignments=cached.leaf_cluster_assignments,
            n_leaf_clusters=cached.n_leaf_clusters,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
        )
        LOGGER.info(
            f"Dendrogram derived view: requested (min={min_cluster_size}, max={max_depth}) -> n_leaf_clusters={derived.n_leaf_clusters}"
        )
        return derived

    # Cache miss: build and persist
    from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage

    embeddings = cloud.embeddings()
    LOGGER.info(
        f"Building bisecting-kmeans dendrogram for cloud {cloud.name()} with min_cluster_size={min_cluster_size}, max_depth={max_depth}"
    )
    result = bisecting_kmeans_linkage(
        embeddings,
        min_cluster_size=min_cluster_size,
        max_depth=max_depth,
        verbose=True,
    )
    dendrogram = DendrogramResult(
        linkage=result.linkage,
        leaf_cluster_assignments=result.leaf_cluster_assignments,
        n_leaf_clusters=int(result.n_leaf_clusters),
    )
    LOGGER.info(
        f"Dendrogram built: requested (min={min_cluster_size}, max={max_depth}) -> n_leaf_clusters={dendrogram.n_leaf_clusters}"
    )

    if cache_enabled:
        try:
            # Important: np.savez will silently append ".npz" if given a string path
            # that doesn't end with ".npz". Use an explicit file handle to ensure
            # the temp path we later rename is exactly what was written.
            tmp_path = exact_path.with_name(exact_path.name + ".tmp")
            with tmp_path.open("wb") as f:
                np.savez(
                    f,
                    linkage=dendrogram.linkage,
                    leaf_cluster_assignments=dendrogram.leaf_cluster_assignments,
                    n_leaf_clusters=np.array(dendrogram.n_leaf_clusters, dtype=np.int64),
                )
            os.replace(tmp_path, exact_path)
            LOGGER.info(f"Saved dendrogram cache: {exact_path}")
        except Exception as e:
            LOGGER.warning(f"Failed saving dendrogram cache to {exact_path}: {e}")

    return dendrogram

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
    return [work.get("display_name", "") for work in get_cloud_works(cloud)]

@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_abstracts(cloud: Cloud) -> List[str]:
    """Get abstracts for all works in the cloud.
    
    The abstract is already resolved from inverted index format during load_works.
    """
    return [work.get("abstract", "") for work in get_cloud_works(cloud)]

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
    _, embeddings = load_works(cloud.filter_pair.pre, cloud.max_works, cloud.embedder_model)
    retained_idxs = get_cloud_retained_idxs(cloud)
    return denoise(cloud, embeddings[retained_idxs])


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