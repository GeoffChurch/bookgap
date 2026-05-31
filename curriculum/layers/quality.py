"""Intrinsic curriculum-quality basket (no LLM, no external corpus).

Lifts the pure scoring functions from
``artifacts_scripts/curriculum/{compare_clusterings.py, curriculum_quality.py}``
(the bookgap-coupled on-disk loaders are NOT lifted — those belong to the host
application). Everything here is arrays-in, metrics-out, numpy/scipy only, plus
a thin wrapper over the core ``compute_frontier_adjacency``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Graph-structure metrics (from compare_clusterings.py)
# ---------------------------------------------------------------------------

def modularity_q(sub_adj: sp.spmatrix, membership: np.ndarray) -> float:
    """Newman modularity Q on a symmetrized weighted graph (range [-1, 1])."""
    sym = (sub_adj + sub_adj.T) * 0.5
    sym = sym.tocoo()
    rows, cols, weights = sym.row, sym.col, sym.data
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    deg = np.asarray(sym.sum(axis=1)).flatten()
    valid = membership >= 0
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


def citation_flow_capture(sub_adj: sp.spmatrix, membership: np.ndarray) -> float:
    """Fraction of edge weight that falls within clusters (range [0, 1])."""
    coo = sub_adj.tocoo()
    total = float(coo.data.sum())
    if total <= 0:
        return 0.0
    within = 0.0
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if membership[i] >= 0 and membership[j] >= 0 and membership[i] == membership[j]:
            within += w
    return float(within / total)


def cluster_size_gini(sizes: Sequence[int]) -> float:
    """Gini coefficient of the (non-empty) cluster-size distribution."""
    nz = [s for s in sizes if s > 0]
    n = len(nz)
    if n <= 1:
        return 0.0
    arr = np.sort(np.asarray(nz, dtype=np.float64))
    cum = np.cumsum(arr)
    total = cum[-1]
    if total <= 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2.0 * np.sum(indices * arr) - (n + 1) * total) / (n * total))


def keyword_distinctness(
    topic_keywords: Dict[int, List[str]], *, top_k: int = 12,
) -> Tuple[float, int]:
    """(avg topics per keyword, # keywords in 3+ topics). Lower both = cleaner."""
    counter: Counter = Counter()
    for kws in topic_keywords.values():
        for w in kws[:top_k]:
            counter[w] += 1
    if not counter:
        return (0.0, 0)
    avg = sum(counter.values()) / len(counter)
    n_crowded = sum(1 for c in counter.values() if c >= 3)
    return (avg, n_crowded)


def avg_pairwise_jaccard_distance(
    topic_keywords: Dict[int, List[str]], *, top_k: int = 12,
) -> float:
    """Average pairwise Jaccard distance of top-K keyword sets (higher=distinct)."""
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
            total += 1.0 - len(inter) / len(union)
            pairs += 1
    return float(total / pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Rank-correlation helpers (from curriculum_quality.py)
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _size_entropy_norm(sizes: Sequence[int]) -> float:
    nz = np.array([s for s in sizes if s > 0], dtype=np.float64)
    n = len(nz)
    if n < 2:
        return float("nan")
    p = nz / nz.sum()
    h = -float((p * np.log(p)).sum())
    return h / math.log(n)


# ---------------------------------------------------------------------------
# Ordering / clustering / path metrics (from curriculum_quality.py)
# ---------------------------------------------------------------------------

def sim_to_target_ascent(
    topic_order: Sequence[int],
    topic_sim_to_target: Optional[Sequence[float]],
) -> Dict[str, float]:
    if topic_sim_to_target is None:
        return {"sim_to_target_spearman": float("nan"),
                "sim_to_target_kendall": float("nan")}
    positions = np.arange(len(topic_order), dtype=np.float64)
    sims = np.asarray([float(topic_sim_to_target[t]) for t in topic_order], dtype=np.float64)
    finite = np.isfinite(sims)
    if finite.sum() < 3:
        return {"sim_to_target_spearman": float("nan"),
                "sim_to_target_kendall": float("nan")}
    return {
        "sim_to_target_spearman": _spearman(positions[finite], sims[finite]),
        "sim_to_target_kendall": _kendall_tau(positions[finite], sims[finite]),
    }


def dag_prereq_completeness(
    topic_adj: Optional[np.ndarray], topic_order: Sequence[int],
) -> float:
    if topic_adj is None:
        return float("nan")
    adj = np.asarray(topic_adj, dtype=np.float64)
    n = adj.shape[0]
    if n < 2 or adj.shape[0] != adj.shape[1]:
        return float("nan")
    pos = np.full(n, n, dtype=np.int64)
    for r, t in enumerate(topic_order):
        if 0 <= int(t) < n:
            pos[int(t)] = r
    per_topic: List[float] = []
    for i in range(n):
        deps = adj[i].copy()
        deps[i] = 0.0
        total = float(deps.sum())
        if total <= 0:
            continue
        earlier = float(deps[pos < pos[i]].sum())
        per_topic.append(earlier / total)
    if not per_topic:
        return float("nan")
    return float(np.mean(per_topic))


def directionality_kendall(
    topic_order: Sequence[int], topic_values: Optional[Sequence[float]],
) -> float:
    if topic_values is None:
        return float("nan")
    positions = np.arange(len(topic_order), dtype=np.float64)
    vals = np.asarray([float(topic_values[t]) for t in topic_order], dtype=np.float64)
    finite = np.isfinite(vals)
    if finite.sum() < 3:
        return float("nan")
    return _kendall_tau(positions[finite], vals[finite])


def topic_coherence_npmi(
    topic_keywords: Dict[int, List[str]],
    topic_docs_tokens: Optional[Dict[int, List[List[str]]]],
    *,
    top_k: int = 10,
) -> float:
    if not topic_docs_tokens:
        return float("nan")
    docs: List[set] = []
    for toks_list in topic_docs_tokens.values():
        for toks in toks_list:
            if toks:
                docs.append(set(toks))
    n_docs = len(docs)
    if n_docs < 2:
        return float("nan")
    df: Counter = Counter()
    for d in docs:
        for w in d:
            df[w] += 1
    eps = 1e-12
    topic_scores: List[float] = []
    for t, kws in topic_keywords.items():
        words = [w for w in kws[:top_k] if df.get(w, 0) > 0]
        if len(words) < 2:
            continue
        pair_scores: List[float] = []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                wa, wb = words[a], words[b]
                co = sum(1 for d in docs if wa in d and wb in d)
                p_ab = (co + eps) / n_docs
                p_a = df[wa] / n_docs
                p_b = df[wb] / n_docs
                npmi = math.log(p_ab / (p_a * p_b)) / (-math.log(p_ab))
                pair_scores.append(npmi)
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))
    if not topic_scores:
        return float("nan")
    return float(np.mean(topic_scores))


def step_smoothness(
    topic_order: Sequence[int], topic_centroids: Optional[np.ndarray],
) -> Dict[str, float]:
    if topic_centroids is None:
        return {"step_smoothness_mean": float("nan"),
                "step_smoothness_max": float("nan")}
    C = np.asarray(topic_centroids, dtype=np.float64)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dists: List[float] = []
    for a, b in zip(topic_order[:-1], topic_order[1:]):
        if 0 <= int(a) < len(C) and 0 <= int(b) < len(C):
            dists.append(1.0 - float(C[int(a)] @ C[int(b)]))
    if not dists:
        return {"step_smoothness_mean": float("nan"),
                "step_smoothness_max": float("nan")}
    return {"step_smoothness_mean": float(np.mean(dists)),
            "step_smoothness_max": float(np.max(dists))}


def topic_dependency_adjacency(
    graph: sp.spmatrix,
    point_to_cluster: np.ndarray,
    n_topics: int,
    *,
    alpha: float = 0.85,
) -> Optional[np.ndarray]:
    """Inter-topic dependency matrix via the core absorbing PageRank."""
    try:
        from curriculum.core.pagerank import compute_frontier_adjacency
        adj_csr = graph.tocsr()
        frontier_nodes = np.arange(n_topics, dtype=np.int64)
        leaf_to_cluster = point_to_cluster.astype(np.int64)
        return compute_frontier_adjacency(
            adj_csr, frontier_nodes, leaf_to_cluster, alpha=alpha, tol=1e-6, max_iter=200,
        )
    except Exception:
        return None


METRIC_DIRECTION: Dict[str, str] = {
    "sim_to_target_spearman": "higher",
    "sim_to_target_kendall": "higher",
    "dag_prereq_completeness": "higher",
    "directionality_pubyear_kendall": "flag",
    "directionality_indegree_kendall": "flag",
    "topic_coherence_npmi": "higher",
    "cross_topic_distinctness_jaccard": "higher",
    "avg_topics_per_keyword": "lower",
    "n_keywords_in_3plus_topics": "lower",
    "modularity_q": "higher",
    "size_entropy_norm": "higher",
    "size_gini": "lower",
    "n_singletons": "lower",
    "step_smoothness_mean": "lower",
    "step_smoothness_max": "lower",
}


@dataclass
class CurriculumInputs:
    """Inputs to :func:`score_curriculum`. Missing-input metrics return NaN."""

    topic_order: Sequence[int]
    point_to_cluster: np.ndarray
    n_topics: int
    cluster_sizes: Sequence[int]
    topic_keywords: Dict[int, List[str]]
    graph: Optional[sp.spmatrix] = None
    topic_adj: Optional[np.ndarray] = None
    topic_centroids: Optional[np.ndarray] = None
    target_centroid: Optional[np.ndarray] = None
    topic_sim_to_target: Optional[Sequence[float]] = None
    topic_pub_years: Optional[Sequence[float]] = None
    topic_docs_tokens: Optional[Dict[int, List[List[str]]]] = None
    keyword_top_k: int = 10


def score_curriculum(inputs: CurriculumInputs) -> Dict[str, float]:
    """Compute the intrinsic metric basket for one curriculum (flat dict)."""
    out: Dict[str, float] = {}
    order = list(inputs.topic_order)
    n = inputs.n_topics

    sim_to_target = inputs.topic_sim_to_target
    if sim_to_target is None and inputs.topic_centroids is not None \
            and inputs.target_centroid is not None:
        C = np.asarray(inputs.topic_centroids, dtype=np.float64)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        tgt = np.asarray(inputs.target_centroid, dtype=np.float64)
        tgt = tgt / (np.linalg.norm(tgt) + 1e-12)
        sim_to_target = (C @ tgt).tolist()

    topic_adj = inputs.topic_adj
    if topic_adj is None and inputs.graph is not None:
        topic_adj = topic_dependency_adjacency(inputs.graph, inputs.point_to_cluster, n)

    out.update(sim_to_target_ascent(order, sim_to_target))
    out["dag_prereq_completeness"] = dag_prereq_completeness(topic_adj, order)
    out["directionality_pubyear_kendall"] = directionality_kendall(order, inputs.topic_pub_years)
    indeg = None
    if topic_adj is not None:
        indeg = np.asarray(topic_adj, dtype=np.float64).sum(axis=0).tolist()
    out["directionality_indegree_kendall"] = directionality_kendall(order, indeg)

    out["topic_coherence_npmi"] = topic_coherence_npmi(
        inputs.topic_keywords, inputs.topic_docs_tokens, top_k=inputs.keyword_top_k)
    out["cross_topic_distinctness_jaccard"] = avg_pairwise_jaccard_distance(
        inputs.topic_keywords, top_k=inputs.keyword_top_k)
    avg_tpk, n_crowded = keyword_distinctness(inputs.topic_keywords, top_k=inputs.keyword_top_k)
    out["avg_topics_per_keyword"] = avg_tpk
    out["n_keywords_in_3plus_topics"] = float(n_crowded)
    if inputs.graph is not None:
        out["modularity_q"] = modularity_q(inputs.graph, inputs.point_to_cluster)
    else:
        out["modularity_q"] = float("nan")
    sizes = list(inputs.cluster_sizes)
    out["size_entropy_norm"] = _size_entropy_norm(sizes)
    out["size_gini"] = cluster_size_gini(sizes)
    out["n_singletons"] = float(sum(1 for s in sizes if s == 1))

    out.update(step_smoothness(order, inputs.topic_centroids))
    return out
