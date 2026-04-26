"""Compute objective clustering quality metrics on existing curriculum runs.

For each run found under ``artifacts/curriculum/`` (each is a directory with
``citation_graph.npz``, ``filtered_work_ids.json``, ``point_to_cluster.npy``,
``topic_keywords.json``, etc.), compute:

Citation-graph metrics (intrinsic, no embeddings):
  - Modularity Q (Newman) on the symmetrised citation subgraph
  - Average conductance
  - Citation-flow capture (within-cluster edge fraction)
  - Number of effective topics (size > floor)

Keyword metrics (uses topic_keywords.json):
  - Topic coherence (NPMI top-K) — averaged over topics
  - Cross-topic keyword distinctness (avg pairwise Jaccard distance of
    top-K sets, equivalently 1 - "crowding")

Group runs by their target_work_ids (so cross-method comparisons stay
fair within a corpus).

Usage:
    python -m artifacts_scripts.curriculum.compare_clusterings
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp


def _log(msg: str) -> None:
    print(f"[compare] {msg}", file=sys.stderr, flush=True)


# -----------------------------------------------------------------
# Citation-graph metrics
# -----------------------------------------------------------------

def _load_subgraph_for_run(run_dir: Path) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Load citation subgraph (5000x5000) restricted to the top-k filtered
    works, plus the per-work cluster assignments. Symmetrises edges."""
    adj = sp.load_npz(run_dir / "citation_graph.npz")
    p2c = np.load(run_dir / "point_to_cluster.npy")
    # ``filtered_work_ids.json`` gives the order of the top-k indices
    # in the subgraph, but we need the full-corpus indices to slice
    # adj. Reconstruct from work_ids of the cloud once per call —
    # this is the slow part (cloud loading takes ~30s) but only
    # happens once per script invocation since we cache.
    return adj, p2c


def modularity_q(
    sub_adj: sp.spmatrix,
    membership: np.ndarray,
) -> float:
    """Newman's modularity Q on a symmetric weighted graph.

    Q = (1/2m) Σ_{ij} (A_ij - k_i k_j / 2m) δ(c_i, c_j)
      = Σ_c (e_c - a_c²)
    where e_c is the within-cluster edge fraction and a_c is the
    fraction of edge ends in cluster c.

    Range: [-1, 1]. Q > 0.3 indicates real community structure;
    Q > 0.5 is strong. Leiden literally optimises this (with a
    resolution-parameter generalisation), so Leiden should win on Q
    by definition unless the optimisation got stuck.
    """
    sym = (sub_adj + sub_adj.T) * 0.5
    sym = sym.tocoo()
    rows, cols, weights = sym.row, sym.col, sym.data
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    deg = np.asarray(sym.sum(axis=1)).flatten()
    valid = (membership >= 0)
    e_c: Dict[int, float] = defaultdict(float)
    a_c: Dict[int, float] = defaultdict(float)
    for i, j, w in zip(rows, cols, weights):
        if not (valid[i] and valid[j]):
            continue
        if membership[i] == membership[j]:
            e_c[int(membership[i])] += w
    for v in range(len(deg)):
        if valid[v]:
            a_c[int(membership[v])] += deg[v]
    Q = 0.0
    for c in set(e_c) | set(a_c):
        Q += e_c[c] / total - (a_c[c] / total) ** 2
    return float(Q)


def average_conductance(
    sub_adj: sp.spmatrix,
    membership: np.ndarray,
) -> float:
    """Mean of per-cluster conductance: (edges-leaving-cluster) /
    min(volume(cluster), volume(rest)).

    Conductance is in [0, 1]. Lower = better-separated cluster.
    Average over non-trivial clusters (size > 1, mass > 0).
    """
    sym = (sub_adj + sub_adj.T) * 0.5
    sym = sym.tocsr()
    deg = np.asarray(sym.sum(axis=1)).flatten()
    total_vol = float(deg.sum())
    if total_vol <= 0:
        return 0.0
    valid = membership >= 0
    cluster_ids = sorted(set(int(c) for c in membership[valid]))
    conductances: List[float] = []
    for c in cluster_ids:
        mask = (membership == c) & valid
        if mask.sum() <= 1:
            continue
        vol = float(deg[mask].sum())
        if vol <= 0:
            continue
        # Edges leaving = vol(c) - 2 * within-edge-weight(c)
        sub = sym[mask][:, mask]
        within = float(sub.sum())  # already counts both directions in sym
        cut = vol - within
        denom = min(vol, total_vol - vol)
        if denom <= 0:
            continue
        conductances.append(cut / denom)
    if not conductances:
        return 0.0
    return float(np.mean(conductances))


def cluster_size_gini(sizes: List[int]) -> float:
    """Gini coefficient of the cluster-size distribution.

    0 = perfectly uniform sizes; ~1 = one cluster has all the mass.
    Computed only over non-empty clusters.

    The mega-cluster pathology spectral exhibits on QC (one cluster of
    2847 + smaller fragments) shows up as Gini ≈ 0.7+, while a balanced
    Leiden/graph_bisect run is ~0.3-0.4. None of the existing
    metrics (Q, conductance, flow capture) catch this directly.
    """
    nz = [s for s in sizes if s > 0]
    n = len(nz)
    if n <= 1:
        return 0.0
    arr = np.sort(np.asarray(nz, dtype=np.float64))
    cum = np.cumsum(arr)
    total = cum[-1]
    if total <= 0:
        return 0.0
    # Gini = (2 * sum_i(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
    indices = np.arange(1, n + 1)
    return float((2.0 * np.sum(indices * arr) - (n + 1) * total) / (n * total))


def max_cluster_fraction(sizes: List[int], total: int = 5000) -> float:
    """Fraction of the corpus held by the largest cluster.

    Spectral on QC: 0.57 (mega-cluster). Leiden on QC: 0.15 (balanced).
    Higher = more mega-cluster-shaped.
    """
    if not sizes or total <= 0:
        return 0.0
    return float(max(sizes) / total)


def fragment_cluster_count(sizes: List[int], threshold: int = 20) -> int:
    """Count of clusters at or below the fragment threshold.

    Default threshold of 20 (small enough to be a "fragment" in a
    5000-doc corpus, large enough to admit genuine niche subtopics).

    Spectral on QC: 5 clusters of size ≤16 (4 of size ≤16, includes
    a size-2). graph_bisect: 0. Leiden: 0.
    """
    return sum(1 for s in sizes if 0 < s <= threshold)


def citation_flow_capture(
    sub_adj: sp.spmatrix,
    membership: np.ndarray,
) -> float:
    """Fraction of edges (by weight) that fall within clusters.

    Higher = clusters capture more of the citation structure.
    Range [0, 1].
    """
    coo = sub_adj.tocoo()
    total = float(coo.data.sum())
    if total <= 0:
        return 0.0
    within = 0.0
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if membership[i] >= 0 and membership[j] >= 0 and membership[i] == membership[j]:
            within += w
    return float(within / total)


# -----------------------------------------------------------------
# Keyword metrics
# -----------------------------------------------------------------

def topic_coherence_npmi(
    topic_keywords: Dict[int, List[str]],
    *,
    top_k: int = 10,
) -> float:
    """Average pairwise NPMI of top-K keywords across topics.

    Mimno et al. NPMI top-K coherence uses an external reference
    corpus (e.g. Wikipedia). Here we use the keyword sets themselves
    as a proxy — pairs of keywords that frequently co-occur within
    the same topic's top-K across multiple topics get higher scores
    (which isn't exactly Mimno's metric but is a decent proxy when
    we lack external reference). For a cleaner version, would need
    to run NPMI over the actual sampled per-topic doc tokens.

    To do this *properly* with what we have: skip — it'd need
    re-loading the per-topic doc tokens. For now, return the
    Jaccard-distance-based distinctness instead and note the
    coherence-via-top-K as a TODO.
    """
    return float("nan")  # placeholder — see distinctness below


def keyword_distinctness(
    topic_keywords: Dict[int, List[str]],
    *,
    top_k: int = 12,
) -> Tuple[float, int]:
    """Cross-topic keyword distinctness.

    Returns (avg_topics_per_keyword, n_keywords_in_3_or_more_topics):
    - avg_topics_per_keyword: 1.0 means perfect distinctness; >1 means
      keyword overlap. Equivalent to the "crowding" metric we computed
      in earlier experiments.
    - n_keywords_in_3_or_more_topics: count of words appearing in 3+
      topics' top-K. Lower = more distinct topic-keyword sets.
    """
    counter: Counter[str] = Counter()
    for kws in topic_keywords.values():
        for w in kws[:top_k]:
            counter[w] += 1
    if not counter:
        return (0.0, 0)
    avg = sum(counter.values()) / len(counter)
    n_crowded = sum(1 for c in counter.values() if c >= 3)
    return (avg, n_crowded)


def avg_pairwise_jaccard_distance(
    topic_keywords: Dict[int, List[str]],
    *,
    top_k: int = 12,
) -> float:
    """Average pairwise Jaccard *distance* (1 - similarity) between
    top-K keyword sets across topics.

    Higher = more distinct topics. Bounded in [0, 1].
    """
    sets = [set(kws[:top_k]) for kws in topic_keywords.values() if kws]
    if len(sets) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            inter = sets[i] & sets[j]
            jaccard_dist = 1.0 - len(inter) / len(union)
            total += jaccard_dist
            pairs += 1
    if pairs == 0:
        return 0.0
    return float(total / pairs)


# -----------------------------------------------------------------
# Run discovery + per-run evaluation
# -----------------------------------------------------------------

def discover_runs(base: Path) -> List[Path]:
    """Find all run dirs that have the canonical output files."""
    runs: List[Path] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        needed = ["metadata.json", "citation_graph.npz", "point_to_cluster.npy",
                  "topic_keywords.json", "filtered_work_ids.json"]
        if all((d / f).exists() for f in needed):
            runs.append(d)
    return runs


def evaluate_run(run_dir: Path) -> Dict:
    """Return a dict of metrics for one run. Logs which step is slow."""
    _log(f"evaluating {run_dir.name}...")
    md = json.loads((run_dir / "metadata.json").read_text())
    targets = sorted(md.get("target_work_ids", []))
    sources = sorted(md.get("source_work_ids", []))
    method = md.get("linkage_method", "?")

    p2c = np.load(run_dir / "point_to_cluster.npy")
    n_topics = md.get("actual_n_topics", int(p2c.max()) + 1)

    # Top-k subgraph: we need the rows/cols of the full adj that
    # correspond to the top-k filtered works. Build via cloud.
    adj_full = sp.load_npz(run_dir / "citation_graph.npz")
    with open(run_dir / "filtered_work_ids.json") as f:
        filtered_ids = json.load(f)

    # Lazy-load cloud for id_to_idx mapping; cache once per process.
    global _cloud_id_to_idx
    if "_cloud_id_to_idx" not in globals() or _cloud_id_to_idx is None:
        _log("  loading cloud (one-time)...")
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from bookgap.configs.cloud.cloud_config import CloudConfig
        cloud = instantiate(OmegaConf.structured(CloudConfig()))
        _cloud_id_to_idx = {wid: i for i, wid in enumerate(cloud.work_ids())}
        _log(f"  cloud loaded ({len(_cloud_id_to_idx):,} works)")

    top_k_indices = np.array(
        [_cloud_id_to_idx[wid] for wid in filtered_ids],
        dtype=np.int64,
    )
    sub_adj = adj_full[top_k_indices][:, top_k_indices]

    # p2c is in *filtered-index* space (0..n_filtered-1), same as
    # sub_adj rows/cols.

    _log(f"  Q ...")
    Q = modularity_q(sub_adj, p2c)
    _log(f"  conductance ...")
    cond = average_conductance(sub_adj, p2c)
    _log(f"  citation flow ...")
    flow = citation_flow_capture(sub_adj, p2c)

    # Cluster sizes
    sizes = sorted(md.get("cluster_sizes", []), reverse=True)
    if not sizes:
        sizes = sorted(np.bincount(p2c[p2c >= 0]).tolist(), reverse=True)
    n_substantial = sum(1 for s in sizes if s >= 50)
    gini = cluster_size_gini(sizes)
    max_frac = max_cluster_fraction(sizes, total=int((p2c >= 0).sum()) or 5000)
    n_fragments = fragment_cluster_count(sizes, threshold=20)
    n_singletons = sum(1 for s in sizes if s == 1)

    # Keywords
    keywords_raw = json.loads((run_dir / "topic_keywords.json").read_text())
    topic_keywords = {
        int(k): [w for w, _ in v] if v and isinstance(v[0], (list, tuple)) else v
        for k, v in keywords_raw.items()
    }
    avg_topics_per_kw, n_crowded = keyword_distinctness(topic_keywords, top_k=12)
    avg_jaccard_dist = avg_pairwise_jaccard_distance(topic_keywords, top_k=12)

    return {
        "run_dir": run_dir.name,
        "method": method,
        "targets": tuple(targets),
        "sources": tuple(sources),
        "n_topics": n_topics,
        "modularity_Q": Q,
        "avg_conductance": cond,
        "citation_flow_capture": flow,
        "cluster_sizes": sizes,
        "n_substantial_topics": n_substantial,
        "gini": gini,
        "max_frac": max_frac,
        "n_fragments": n_fragments,
        "n_singletons": n_singletons,
        "avg_topics_per_kw": avg_topics_per_kw,
        "n_kws_in_3plus_topics": n_crowded,
        "avg_jaccard_dist": avg_jaccard_dist,
    }


_cloud_id_to_idx = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum-dir", type=Path,
        default=Path("artifacts/curriculum"),
    )
    args = parser.parse_args()

    runs = discover_runs(args.curriculum_dir)
    _log(f"found {len(runs)} runs")
    rows: List[Dict] = []
    for r in runs:
        try:
            rows.append(evaluate_run(r))
        except Exception as e:
            _log(f"  skipped {r.name}: {e}")

    # Group by (targets, sources) so we compare within-corpus
    print()
    by_corpus: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        by_corpus[(row["targets"], row["sources"])].append(row)

    for corpus_idx, ((targets, sources), corpus_rows) in enumerate(by_corpus.items()):
        target_summary = (targets[0].split("/")[-1] if targets else "?")
        if len(targets) > 1:
            target_summary += f" + {len(targets)-1} more"
        print(f"\n=== Corpus {corpus_idx + 1}: targets={target_summary}, "
              f"{len(sources)} sources, {len(corpus_rows)} runs ===")
        print(
            f"{'method':<22}  "
            f"{'Q':>6}  {'cond':>6}  {'flow':>6}  "
            f"{'gini':>5}  {'max%':>5}  {'#frag':>5}  {'#sing':>5}  "
            f"{'#≥50':>4}  {'tpc/kw':>6}  {'#≥3':>4}  {'jacc':>5}"
        )
        corpus_rows.sort(key=lambda r: -r["modularity_Q"])
        for row in corpus_rows:
            print(
                f"{row['method']:<22}  "
                f"{row['modularity_Q']:>6.3f}  "
                f"{row['avg_conductance']:>6.3f}  "
                f"{row['citation_flow_capture']:>6.3f}  "
                f"{row['gini']:>5.2f}  "
                f"{row['max_frac']*100:>4.0f}%  "
                f"{row['n_fragments']:>5}  "
                f"{row['n_singletons']:>5}  "
                f"{row['n_substantial_topics']:>4}  "
                f"{row['avg_topics_per_kw']:>6.2f}  "
                f"{row['n_kws_in_3plus_topics']:>4}  "
                f"{row['avg_jaccard_dist']:>5.2f}"
            )


if __name__ == "__main__":
    main()
