import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bookgap.util import CACHE_DIR, LOGGER, TIMER

from numpy.typing import NDArray


class DendrogramResult:
    """Result of dendrogram building.

    This mirrors scipy's linkage conventions:
    - leaves are indexed 0..n_leaf_clusters-1
    - internal nodes are indexed n_leaf_clusters..(2*n_leaf_clusters-2)
    - linkage has shape (n_leaf_clusters-1, 4) when n_leaf_clusters > 1

    For bisecting-kmeans, leaves may represent multi-point clusters. Use
    leaf_cluster_assignments to aggregate point masses before dendrogram ops.
    """

    def __init__(self, linkage: NDArray[np.floating], leaf_cluster_assignments: NDArray[np.intp], n_leaf_clusters: int):
        self.linkage = linkage
        self.leaf_cluster_assignments = leaf_cluster_assignments
        self.n_leaf_clusters = n_leaf_clusters


def _dendrogram_cache_fingerprint(cloud: Any, linkage_method: str) -> str:
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


def _dendrogram_cache_dir(cloud: Any, linkage_method: str):
    fingerprint = _dendrogram_cache_fingerprint(cloud, linkage_method)
    h = __import__("hashlib").sha256(fingerprint.encode("utf-8")).hexdigest()
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


def _setup_dendrogram_cache(cache_dir: Path, fingerprint: str) -> bool:
    """Initialize cache directory and validate sanity check.
    
    Returns:
        True if cache is enabled and valid, False otherwise.
    """
    sanity_path = cache_dir / "sanity.pkl"
    if sanity_path.exists():
        try:
            with sanity_path.open("rb") as f:
                saved = pickle.load(f)
            if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
                LOGGER.warning(
                    "Dendrogram cache sanity mismatch (possible hash collision); ignoring cached entries"
                )
                return False
        except Exception as e:
            LOGGER.warning(f"Failed reading dendrogram cache sanity file: {e}")
            return False
    else:
        try:
            tmp = sanity_path.with_name(sanity_path.name + ".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"fingerprint": fingerprint}, f)
            os.replace(tmp, sanity_path)
        except Exception as e:
            LOGGER.warning(f"Failed writing dendrogram cache sanity file: {e}")
            return False
    return True


def _load_dendrogram_npz(path: Path) -> DendrogramResult:
    """Load a dendrogram from an .npz file."""
    with np.load(path) as data:
        linkage = data["linkage"].astype(np.float64, copy=False)
        leaf_cluster_assignments = data["leaf_cluster_assignments"].astype(np.intp, copy=False)
        n_leaf_clusters = int(data["n_leaf_clusters"])
    return DendrogramResult(
        linkage=linkage, 
        leaf_cluster_assignments=leaf_cluster_assignments, 
        n_leaf_clusters=n_leaf_clusters
    )


def _find_cached_dendrogram(
    cache_dir: Path,
    min_cluster_size: int,
    max_depth: Optional[int],
    cache_enabled: bool,
) -> Optional[DendrogramResult]:
    """Find and return exact or subsuming cached dendrogram.
    
    Returns:
        Cached or derived dendrogram if found, None otherwise.
    """
    if not cache_enabled:
        return None
    
    # Check for exact match first
    requested_name = f"dendrogram_min={min_cluster_size}_max={'none' if max_depth is None else max_depth}.npz"
    exact_path = cache_dir / requested_name
    if exact_path.exists():
        cached = _load_dendrogram_npz(exact_path)
        LOGGER.info(
            f"Dendrogram cache hit (exact): {exact_path} -> n_leaf_clusters={cached.n_leaf_clusters}"
        )
        return cached
    
    # Enumerate all cached dendrograms
    entries: list[tuple[int, Optional[int], Path]] = []
    for p in cache_dir.glob("dendrogram_min=*_max=*.npz"):
        parsed = _parse_cached_dendrogram_filename(p.name)
        if parsed is None:
            continue
        cached_min, cached_max = parsed
        entries.append((cached_min, cached_max, p))
    
    # Find subsuming matches
    subsuming: list[tuple[int, Optional[int], Path]] = [
        (cmin, cmax, path)
        for (cmin, cmax, path) in entries
        if cmin <= min_cluster_size and _max_depth_subsumes(cmax, max_depth)
    ]
    
    if not subsuming:
        return None
    
    # Prefer the closest (least refined) subsumer
    subsuming.sort(key=lambda t: (-t[0], _depth_value(t[1])))
    cached_min, cached_max, path = subsuming[0]
    cached = _load_dendrogram_npz(path)
    LOGGER.info(
        "Dendrogram cache hit (subsuming): "
        f"{path.name} (min={cached_min}, max={cached_max}, n_leaf_clusters={cached.n_leaf_clusters}) "
        f"-> requested (min={min_cluster_size}, max={max_depth})"
    )
    
    # If exact parameters match, return as-is
    if cached_min == min_cluster_size and cached_max == max_depth:
        return cached
    
    # Otherwise, derive truncated view
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


def _save_dendrogram(path: Path, dendrogram: DendrogramResult, cache_enabled: bool) -> None:
    """Save dendrogram to disk atomically if caching is enabled."""
    if not cache_enabled:
        return
    
    try:
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("wb") as f:
            np.savez(
                f,
                linkage=dendrogram.linkage,
                leaf_cluster_assignments=dendrogram.leaf_cluster_assignments,
                n_leaf_clusters=np.array(dendrogram.n_leaf_clusters, dtype=np.int64),
            )
        os.replace(tmp_path, path)
        LOGGER.info(f"Saved dendrogram cache: {path}")
    except Exception as e:
        LOGGER.warning(f"Failed saving dendrogram cache to {path}: {e}")


def _build_dendrogram(cloud: Any, min_cluster_size: int, max_depth: Optional[int]) -> DendrogramResult:
    """Build a new dendrogram from cloud embeddings."""
    from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage

    embeddings = cloud.embeddings()
    LOGGER.info(
        f"Building bisecting-kmeans dendrogram for cloud {cloud.name()} "
        f"with min_cluster_size={min_cluster_size}, max_depth={max_depth}"
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
        f"Dendrogram built: requested (min={min_cluster_size}, max={max_depth}) -> "
        f"n_leaf_clusters={dendrogram.n_leaf_clusters}"
    )
    return dendrogram


@TIMER.time()
def get_cloud_dendrogram(
    cloud: Any,
    *,
    linkage_method: str,
    min_cluster_size: int,
    max_depth: Optional[int],
) -> DendrogramResult:
    """Get a dendrogram for this cloud.

    Caching/subsumption is currently implemented only for linkage_method="bisecting_kmeans".
    """
    # Validate parameters
    if linkage_method != "bisecting_kmeans":
        raise ValueError("Cloud dendrogram caching is only supported for linkage_method='bisecting_kmeans'")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    # Setup cache
    cache_dir, fingerprint = _dendrogram_cache_dir(cloud, linkage_method)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_enabled = _setup_dendrogram_cache(cache_dir, fingerprint)

    # Try to find cached dendrogram
    cached = _find_cached_dendrogram(cache_dir, min_cluster_size, max_depth, cache_enabled)
    if cached is not None:
        return cached

    # Build new dendrogram
    dendrogram = _build_dendrogram(cloud, min_cluster_size, max_depth)

    # Persist to cache
    requested_name = f"dendrogram_min={min_cluster_size}_max={'none' if max_depth is None else max_depth}.npz"
    exact_path = cache_dir / requested_name
    _save_dendrogram(exact_path, dendrogram, cache_enabled)

    return dendrogram
