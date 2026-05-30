#!/usr/bin/env python
# artifacts_scripts/curriculum/curriculum_quality.py
"""Intrinsic curriculum-quality metric basket (the "quality dashboard").

Replaces eyeballing each curriculum's topic ordering / clustering with a
reusable basket of *intrinsic* (no external data) metrics. Design:
``docs/backlog/curriculum-diffusion-and-quality-metrics.md`` §B, intrinsic
section. Extrinsic proxies (syllabus / LectureBank / LLM-judge) are deferred.

The core entry point is :func:`score_curriculum`, a *pure* function of explicit
inputs (no disk / Cloud access). Per-curriculum loaders that assemble those
inputs from on-disk artifacts live in
``artifacts_scripts/curriculum/curriculum_quality_loaders.py`` and the CLI
``__main__`` below prints a dashboard table over the existing curricula.

Metric basket (each metric documents its DIRECTION — higher/lower = better)
==========================================================================

ORDERING (depend on ``topic_order``)
  - ``sim_to_target_spearman`` / ``sim_to_target_kendall``
        Rank-correlation of curriculum position vs each topic's cosine to the
        TARGET centroid. The curriculum should *ascend* toward the goal, so a
        topic's similarity-to-target should INCREASE with its position.
        HIGHER (→ +1) = BETTER (monotone ascent). Needs embeddings + target.
  - ``dag_prereq_completeness``
        On the directed topic graph (adj[i,j] = topic i depends on / cites
        topic j, i.e. j is a prerequisite of i), the fraction of each topic's
        prerequisite in-neighbours (the topics it depends on) that are ordered
        EARLIER than it. Averaged over topics, weighted by dependency mass.
        HIGHER (→ 1.0) = BETTER (prerequisites taught first). No knobs.
  - ``directionality_pubyear_kendall`` (SANITY CHECK — report, don't optimize)
        Kendall-tau of curriculum order vs each topic's median publication year.
        A NEGATIVE tau means later topics are OLDER => frontier-first failure.
        Reported as a flag; we do NOT maximize recency. Needs pub_years.
  - ``directionality_indegree_kendall`` (SANITY CHECK — report, don't optimize)
        Kendall-tau of curriculum order vs each topic's in-degree on the graph
        (high in-degree = heavily-cited = foundational). Foundations should
        come FIRST, so order-position should correlate NEGATIVELY with in-degree
        => a POSITIVE tau here flags frontier-first (foundations taught last).
        Reported as a flag. Uses the graph only.

CLUSTERING (do not depend on order)
  - ``topic_coherence_npmi``
        Mean intra-topic NPMI of the top-K keyword pairs, using the curriculum's
        OWN per-topic member documents as the reference corpus (Mimno-style
        top-K NPMI, intrinsic). HIGHER = BETTER (keywords genuinely co-occur).
        Needs per-topic document tokens; NaN otherwise.
  - ``keyword_coherence_embed``
        Mean intra-topic pairwise cosine of the top-K keyword EMBEDDINGS (a
        no-reference-corpus coherence proxy). HIGHER = BETTER. Needs a keyword
        embedder; NaN otherwise.
  - ``cross_topic_distinctness_jaccard``
        Average pairwise Jaccard DISTANCE of top-K keyword sets across topics.
        HIGHER (→ 1.0) = BETTER (topics are lexically distinct). Reuses
        ``compare_clusterings.avg_pairwise_jaccard_distance``.
  - ``avg_topics_per_keyword`` / ``n_keywords_in_3plus_topics``
        Keyword "crowding": 1.0 = perfect distinctness; >1 = overlap. LOWER =
        BETTER. Reuses ``compare_clusterings.keyword_distinctness``.
  - ``modularity_q``
        Newman modularity of the topic partition on the (symmetrised) graph.
        HIGHER = BETTER community structure (but never optimize alone — max-Q
        fragments). Reuses ``compare_clusterings.modularity_q``.
  - ``size_entropy_norm`` / ``size_gini``
        Cluster-size balance. ``size_entropy_norm`` is Shannon entropy of the
        size distribution normalized to [0,1] (1 = perfectly uniform) — HIGHER
        = BETTER. ``size_gini`` is the Gini of sizes (0 = uniform, ~1 = one
        mega-cluster) — LOWER = BETTER. Reuses ``compare_clusterings``.
  - ``n_singletons``
        Count of size-1 topics (the seed-as-singleton / degenerate-OA failure).
        LOWER = BETTER.

PATH (depends on ``topic_order`` + embeddings)
  - ``step_smoothness_mean`` / ``step_smoothness_max``
        Mean / max consecutive-topic centroid distance (1 - cosine) along the
        curriculum order, in embedding space. A big jump = a missing
        prerequisite. LOWER = BETTER. Needs topic centroids; NaN otherwise.

All metrics are returned in a flat dict; metrics whose required inputs are
missing are ``float('nan')`` so the dashboard can mark them "n/a".
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

# Reuse the validated clustering metrics rather than re-deriving them.
from artifacts_scripts.curriculum.compare_clusterings import (
    avg_pairwise_jaccard_distance,
    cluster_size_gini,
    keyword_distinctness,
    modularity_q,
)


# Metric direction registry: name -> "higher" | "lower" | "flag"
# ("flag" = sanity check, reported but not a quality axis to optimize).
METRIC_DIRECTION: Dict[str, str] = {
    "sim_to_target_spearman": "higher",
    "sim_to_target_kendall": "higher",
    "dag_prereq_completeness": "higher",
    "directionality_pubyear_kendall": "flag",
    "directionality_indegree_kendall": "flag",
    "topic_coherence_npmi": "higher",
    "keyword_coherence_embed": "higher",
    "keyword_coherence_embed_min": "higher",
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


# -----------------------------------------------------------------
# Small intrinsic helpers (not in compare_clusterings)
# -----------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (ties broken by average rank)."""
    if len(x) < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b (handles ties). O(n^2) — fine for ~15 topics."""
    n = len(x)
    if n < 3:
        return float("nan")
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(x[i]) - float(x[j])
            dy = float(y[i]) - float(y[j])
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
                continue
            if sy == 0:
                ty += 1
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt(float((n0 - tx) * (n0 - ty)))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (1-based)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _size_entropy_norm(sizes: Sequence[int]) -> float:
    """Shannon entropy of the cluster-size distribution, normalized to [0,1].

    1.0 = perfectly uniform sizes; ->0 = mass concentrated in one cluster.
    Computed over non-empty clusters; NaN if <2 clusters.
    """
    nz = np.asarray([s for s in sizes if s > 0], dtype=np.float64)
    n = len(nz)
    if n < 2:
        return float("nan")
    p = nz / nz.sum()
    h = -float((p * np.log(p)).sum())
    return h / math.log(n)


# -----------------------------------------------------------------
# Ordering metrics
# -----------------------------------------------------------------

def sim_to_target_ascent(
    topic_order: Sequence[int],
    topic_sim_to_target: Optional[Sequence[float]],
) -> Dict[str, float]:
    """Rank-correlation of curriculum position vs cosine-to-target.

    ``topic_order[r]`` = topic id taught at position ``r`` (0 = first).
    ``topic_sim_to_target[t]`` = cosine of topic t's centroid to the target
    centroid. A monotone ascent toward the target is good, so position and
    sim-to-target should be POSITIVELY correlated. Higher (→ +1) = better.
    """
    if topic_sim_to_target is None:
        return {"sim_to_target_spearman": float("nan"),
                "sim_to_target_kendall": float("nan")}
    positions = np.arange(len(topic_order), dtype=np.float64)
    sims = np.asarray([float(topic_sim_to_target[t]) for t in topic_order],
                      dtype=np.float64)
    finite = np.isfinite(sims)
    if finite.sum() < 3:
        return {"sim_to_target_spearman": float("nan"),
                "sim_to_target_kendall": float("nan")}
    return {
        "sim_to_target_spearman": _spearman(positions[finite], sims[finite]),
        "sim_to_target_kendall": _kendall_tau(positions[finite], sims[finite]),
    }


def dag_prereq_completeness(
    topic_adj: Optional[np.ndarray],
    topic_order: Sequence[int],
) -> float:
    """Fraction of each topic's prerequisite-mass that is ordered earlier.

    ``topic_adj[i,j]`` = how much topic i depends on topic j (j prereq of i).
    For each topic i, its prerequisite in-neighbours are the topics j it
    depends on (``topic_adj[i, j] > 0``). The completeness for i is the
    dependency-weighted fraction of those j that appear EARLIER than i in the
    curriculum order. Averaged over topics with >0 outgoing dependency mass.
    Higher (→ 1.0) = better. No knobs.
    """
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
        deps[i] = 0.0  # ignore self-loops
        total = float(deps.sum())
        if total <= 0:
            continue
        earlier = float(deps[pos < pos[i]].sum())
        per_topic.append(earlier / total)
    if not per_topic:
        return float("nan")
    return float(np.mean(per_topic))


def directionality_kendall(
    topic_order: Sequence[int],
    topic_values: Optional[Sequence[float]],
) -> float:
    """Kendall-tau of curriculum *position* vs a per-topic value.

    Generic helper for the two directionality sanity checks (pub-year,
    in-degree). Returns NaN if values missing. See module docstring for the
    sign interpretation of each check.
    """
    if topic_values is None:
        return float("nan")
    positions = np.arange(len(topic_order), dtype=np.float64)
    vals = np.asarray([float(topic_values[t]) for t in topic_order],
                      dtype=np.float64)
    finite = np.isfinite(vals)
    if finite.sum() < 3:
        return float("nan")
    return _kendall_tau(positions[finite], vals[finite])


# -----------------------------------------------------------------
# Clustering metrics
# -----------------------------------------------------------------

def topic_coherence_npmi(
    topic_keywords: Dict[int, List[str]],
    topic_docs_tokens: Optional[Dict[int, List[List[str]]]],
    *,
    top_k: int = 10,
) -> float:
    """Mean intra-topic NPMI of top-K keyword pairs (intrinsic top-K coherence).

    Uses the curriculum's OWN per-topic member documents as the reference
    corpus: ``topic_docs_tokens[t]`` is a list of token-lists (one per member
    document of topic t). NPMI is computed over the pooled document set (all
    topics' docs) using boolean document co-occurrence — Mimno et al. top-K
    NPMI coherence, with the corpus standing in for an external reference.
    Higher = better (a topic's top keywords genuinely co-occur in its docs).
    NaN if no document tokens are available.
    """
    if not topic_docs_tokens:
        return float("nan")
    # Build the pooled document set (one bag-of-words set per document).
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


def keyword_coherence_embed(
    topic_keywords: Dict[int, List[str]],
    keyword_embed_fn,
    *,
    top_k: int = 10,
) -> Dict[str, float]:
    """Intra-topic pairwise cosine of top-K keyword EMBEDDINGS.

    ``keyword_embed_fn(list_of_strings) -> (n, d) float array``. A
    no-reference-corpus coherence proxy: a coherent topic has keyword
    embeddings that cluster tightly, while a grab-bag topic (e.g. the v1
    ``kubeflow/tensorflow/lightgbm/keras`` toolkit topic mixed with generic
    "learning") scatters. Returns the MEAN over topics and the MIN over topics
    (the single worst/least-coherent topic — this is what catches one bad
    grab-bag topic that the mean washes out). Higher = better. NaN if no
    embedder.
    """
    nan = {"keyword_coherence_embed": float("nan"),
           "keyword_coherence_embed_min": float("nan")}
    if keyword_embed_fn is None:
        return nan
    # Collect a deduped vocabulary so we embed each keyword once.
    vocab: List[str] = []
    seen = set()
    for kws in topic_keywords.values():
        for w in kws[:top_k]:
            if w not in seen:
                seen.add(w)
                vocab.append(w)
    if len(vocab) < 2:
        return nan
    emb = np.asarray(keyword_embed_fn(vocab), dtype=np.float64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    idx = {w: i for i, w in enumerate(vocab)}
    topic_scores: List[float] = []
    for kws in topic_keywords.values():
        words = [w for w in kws[:top_k] if w in idx]
        if len(words) < 2:
            continue
        vecs = emb[[idx[w] for w in words]]
        sims = vecs @ vecs.T
        n = len(words)
        # mean of the strict upper triangle
        s = (float(sims.sum()) - n) / (n * (n - 1))
        topic_scores.append(s)
    if not topic_scores:
        return nan
    return {"keyword_coherence_embed": float(np.mean(topic_scores)),
            "keyword_coherence_embed_min": float(np.min(topic_scores))}


# -----------------------------------------------------------------
# Path metrics
# -----------------------------------------------------------------

def step_smoothness(
    topic_order: Sequence[int],
    topic_centroids: Optional[np.ndarray],
) -> Dict[str, float]:
    """Consecutive-topic centroid distance (1 - cosine) along the order.

    ``topic_centroids[t]`` = embedding-space centroid of topic t. A large jump
    between adjacent topics flags a missing prerequisite. Lower = better.
    NaN if centroids missing.
    """
    if topic_centroids is None:
        return {"step_smoothness_mean": float("nan"),
                "step_smoothness_max": float("nan")}
    C = np.asarray(topic_centroids, dtype=np.float64)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dists: List[float] = []
    for a, b in zip(topic_order[:-1], topic_order[1:]):
        if 0 <= int(a) < len(C) and 0 <= int(b) < len(C):
            cos = float(C[int(a)] @ C[int(b)])
            dists.append(1.0 - cos)
    if not dists:
        return {"step_smoothness_mean": float("nan"),
                "step_smoothness_max": float("nan")}
    return {"step_smoothness_mean": float(np.mean(dists)),
            "step_smoothness_max": float(np.max(dists))}


# -----------------------------------------------------------------
# Topic-level graph aggregation
# -----------------------------------------------------------------

def topic_dependency_adjacency(
    graph: sp.spmatrix,
    point_to_cluster: np.ndarray,
    n_topics: int,
    *,
    alpha: float = 0.85,
) -> Optional[np.ndarray]:
    """Inter-topic dependency matrix via absorbing PageRank (prereq -> frontier).

    Thin wrapper over ``curriculum.pagerank.compute_frontier_adjacency`` so the
    DAG-prereq and directionality metrics use the SAME machinery the curriculum
    pipelines use to ORDER topics. Returns an ``(n_topics, n_topics)`` matrix
    where ``adj[i,j]`` = how much topic i depends on topic j (j prereq of i),
    or ``None`` on failure.
    """
    try:
        from artifacts_scripts.curriculum.pagerank import compute_frontier_adjacency
        adj_csr = graph.tocsr()
        frontier_nodes = np.arange(n_topics, dtype=np.int64)
        leaf_to_cluster = point_to_cluster.astype(np.int64)
        return compute_frontier_adjacency(
            adj_csr, frontier_nodes, leaf_to_cluster,
            alpha=alpha, tol=1e-6, max_iter=200,
        )
    except Exception:
        return None


# -----------------------------------------------------------------
# The basket
# -----------------------------------------------------------------

@dataclass
class CurriculumInputs:
    """Inputs to :func:`score_curriculum`. Most are optional; metrics whose
    inputs are missing return NaN."""
    topic_order: Sequence[int]
    point_to_cluster: np.ndarray
    n_topics: int
    cluster_sizes: Sequence[int]
    topic_keywords: Dict[int, List[str]]
    graph: Optional[sp.spmatrix] = None
    topic_adj: Optional[np.ndarray] = None        # precomputed inter-topic dep matrix
    topic_centroids: Optional[np.ndarray] = None   # (n_topics, d) embedding centroids
    target_centroid: Optional[np.ndarray] = None   # (d,) target embedding
    topic_sim_to_target: Optional[Sequence[float]] = None  # overrides centroid·target
    topic_pub_years: Optional[Sequence[float]] = None
    topic_docs_tokens: Optional[Dict[int, List[List[str]]]] = None
    keyword_embed_fn: Optional[object] = None
    keyword_top_k: int = 10


def score_curriculum(inputs: CurriculumInputs) -> Dict[str, float]:
    """Compute the full intrinsic metric basket for one curriculum.

    Returns a flat ``{metric_name: value}`` dict; see module docstring and
    :data:`METRIC_DIRECTION` for each metric's meaning and direction.
    """
    out: Dict[str, float] = {}
    order = list(inputs.topic_order)
    n = inputs.n_topics

    # --- topic centroids / sim-to-target ---
    sim_to_target = inputs.topic_sim_to_target
    if sim_to_target is None and inputs.topic_centroids is not None \
            and inputs.target_centroid is not None:
        C = np.asarray(inputs.topic_centroids, dtype=np.float64)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        tgt = np.asarray(inputs.target_centroid, dtype=np.float64)
        tgt = tgt / (np.linalg.norm(tgt) + 1e-12)
        sim_to_target = (C @ tgt).tolist()

    # --- inter-topic dependency adjacency ---
    topic_adj = inputs.topic_adj
    if topic_adj is None and inputs.graph is not None:
        topic_adj = topic_dependency_adjacency(
            inputs.graph, inputs.point_to_cluster, n)

    # ORDERING
    out.update(sim_to_target_ascent(order, sim_to_target))
    out["dag_prereq_completeness"] = dag_prereq_completeness(topic_adj, order)
    out["directionality_pubyear_kendall"] = directionality_kendall(
        order, inputs.topic_pub_years)
    # in-degree directionality: foundations (high in-degree) should be EARLY,
    # so position should anti-correlate with in-degree => positive tau = bad.
    indeg = None
    if topic_adj is not None:
        indeg = np.asarray(topic_adj, dtype=np.float64).sum(axis=0).tolist()
    out["directionality_indegree_kendall"] = directionality_kendall(order, indeg)

    # CLUSTERING
    out["topic_coherence_npmi"] = topic_coherence_npmi(
        inputs.topic_keywords, inputs.topic_docs_tokens, top_k=inputs.keyword_top_k)
    out.update(keyword_coherence_embed(
        inputs.topic_keywords, inputs.keyword_embed_fn, top_k=inputs.keyword_top_k))
    out["cross_topic_distinctness_jaccard"] = avg_pairwise_jaccard_distance(
        inputs.topic_keywords, top_k=inputs.keyword_top_k)
    avg_tpk, n_crowded = keyword_distinctness(
        inputs.topic_keywords, top_k=inputs.keyword_top_k)
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

    # PATH
    out.update(step_smoothness(order, inputs.topic_centroids))

    return out


# =====================================================================
# Per-curriculum loaders + dashboard CLI
# =====================================================================
# The scorer above is pure (arrays in, metrics out). Everything below assembles
# ``CurriculumInputs`` from each curriculum's on-disk artifacts and prints a
# dashboard table. To avoid repeatedly invoking the (separately-owned)
# ``joint_graph_curriculum.py`` while it may be under edit, the joint v1/v2
# artifacts are read from a one-time CACHE under
# ``cache/.curriculum_quality/{joint_v1,joint_v2,wiki_native}/`` (populated by
# running the committed script once up front).

import json
import re
import sys
from pathlib import Path
from typing import Any, Tuple

ART = Path("artifacts/curriculum")
QC_CACHE = Path("cache/.curriculum_quality")


def _log(msg: str) -> None:
    print(f"[curriculum_quality] {msg}", file=sys.stderr, flush=True)


def _parse_keywords(raw: Dict[str, Any]) -> Dict[int, List[str]]:
    """Normalize topic_keywords.json (``[[word, score], ...]`` or ``[word,...]``)
    into ``{topic_id: [word, ...]}``."""
    out: Dict[int, List[str]] = {}
    for k, v in raw.items():
        if v and isinstance(v[0], (list, tuple)):
            out[int(k)] = [w for w, _ in v]
        else:
            out[int(k)] = list(v)
    return out


def _recompute_topic_order(
    graph: sp.csr_matrix, point_to_cluster: np.ndarray, n_topics: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Recompute the prerequisite->frontier topic order from the graph using the
    SAME machinery the pipelines use. Returns (order, topic_adj)."""
    from artifacts_scripts.curriculum.build_curriculum import (
        _compute_linearization_order_from_adjacency,
    )
    topic_adj = topic_dependency_adjacency(graph, point_to_cluster, n_topics)
    if topic_adj is None:
        return np.arange(n_topics, dtype=np.int64), None
    order = _compute_linearization_order_from_adjacency(topic_adj)
    return order, topic_adj


_GIST_MODEL = None


def _gist_keyword_embedder():
    """Return a ``fn(list[str]) -> (n, 384) array`` over GIST-small (the same
    embedding space the curricula were built in). Underscores in mined bigrams
    are turned back into spaces so the encoder sees natural phrases. Loaded once
    per process; ``None`` if sentence-transformers is unavailable."""
    global _GIST_MODEL
    if _GIST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _log("loading GIST-small for keyword coherence (one-time)...")
            _GIST_MODEL = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
        except Exception as e:  # pragma: no cover
            _log(f"keyword embedder unavailable ({e})")
            return None

    def _embed(words: List[str]) -> np.ndarray:
        texts = [w.replace("_", " ") for w in words]
        return np.asarray(_GIST_MODEL.encode(texts, show_progress_bar=False))

    return _embed


def _tokenize_docs(texts: List[str]) -> List[List[str]]:
    from artifacts_scripts.curriculum.keyword_extraction import (
        tokenize, filter_stopwords,
    )
    out = []
    for t in texts:
        out.append(filter_stopwords(tokenize(t or "")))
    return out


# ---- wiki-native (bg-norm + frontier) -------------------------------------

def load_wiki_native(cache_dir: Path = QC_CACHE / "wiki_native",
                     keyword_embed_fn=None) -> CurriculumInputs:
    p2c = np.load(cache_dir / "point_to_cluster.npy")
    adj = sp.load_npz(cache_dir / "link_subgraph.npz").tocsr()
    kws = _parse_keywords(json.loads((cache_dir / "topic_keywords.json").read_text()))
    titles = json.loads((cache_dir / "top_titles.json").read_text())
    n_topics = int(p2c.max()) + 1
    sizes = [int((p2c == t).sum()) for t in range(n_topics)]

    # The persisted link_subgraph is the FULL 477k-node graph, but p2c is over
    # the top-4000 clustered pages whose titles are in `titles`. Map titles back
    # to subgraph rows to (a) slice the graph for modularity/ordering and (b)
    # pull GIST embeddings for centroids/target similarity.
    sub = np.load(
        "cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz",
        allow_pickle=True,
    )
    sub_titles = list(sub["titles"])
    title_to_row = {t: i for i, t in enumerate(sub_titles)}
    rows = np.array([title_to_row.get(t, -1) for t in titles], dtype=np.int64)
    ok = rows >= 0
    _log(f"wiki-native: matched {int(ok.sum())}/{len(titles)} titles to subgraph")
    sub_adj = adj[rows[ok]][:, rows[ok]]
    sub_p2c = p2c[ok]

    emb = np.asarray(sub["emb"])[rows[ok]]
    centroids = np.zeros((n_topics, emb.shape[1]), dtype=np.float64)
    for t in range(n_topics):
        m = sub_p2c == t
        if m.any():
            centroids[t] = emb[m].mean(axis=0)
    target = np.load("/tmp/nlp_dir.npy") if Path("/tmp/nlp_dir.npy").exists() else None

    # per-topic doc tokens for NPMI (page titles as documents)
    docs: Dict[int, List[List[str]]] = {}
    for t in range(n_topics):
        toks = _tokenize_docs([titles[i] for i in range(len(titles)) if p2c[i] == t])
        docs[t] = toks

    order, topic_adj = _recompute_topic_order(sub_adj, sub_p2c, n_topics)
    return CurriculumInputs(
        topic_order=order.tolist(),
        point_to_cluster=sub_p2c,
        n_topics=n_topics,
        cluster_sizes=sizes,
        topic_keywords=kws,
        graph=sub_adj,
        topic_adj=topic_adj,
        topic_centroids=centroids,
        target_centroid=target,
        topic_docs_tokens=docs,
        keyword_embed_fn=keyword_embed_fn,
    )


# ---- joint v1 / v2 --------------------------------------------------------

def load_joint(cache_dir: Path, keyword_embed_fn=None) -> CurriculumInputs:
    joint = sp.load_npz(cache_dir / "joint_graph.npz").tocsr()
    p2c = np.load(cache_dir / "point_to_cluster.npy")
    top_idx = np.load(cache_dir / "top_idx.npy")
    kws = _parse_keywords(json.loads((cache_dir / "topic_keywords.json").read_text()))
    n_topics = int(p2c.max()) + 1
    sizes = [int((p2c == t).sum()) for t in range(n_topics)]

    # Order + topic-adjacency from the joint graph (the script's own method).
    leaf = np.full(joint.shape[0], -1, dtype=np.int64)
    leaf[top_idx] = p2c.astype(np.int64)
    topic_adj = topic_dependency_adjacency(joint, leaf, n_topics)
    if topic_adj is not None:
        from artifacts_scripts.curriculum.build_curriculum import (
            _compute_linearization_order_from_adjacency,
        )
        order = _compute_linearization_order_from_adjacency(topic_adj)
    else:
        order = np.arange(n_topics, dtype=np.int64)

    # Modularity uses the clustered subgraph (rows/cols = top_idx), p2c-aligned.
    sub_adj = joint[top_idx][:, top_idx]

    # No per-node embeddings or titles are persisted for the joint top_idx set,
    # so embedding-space metrics (sim->target, step-smoothness) and NPMI docs
    # are left as NaN. Graph + keyword metrics carry the joint comparison.
    return CurriculumInputs(
        topic_order=order.tolist(),
        point_to_cluster=p2c,
        n_topics=n_topics,
        cluster_sizes=sizes,
        topic_keywords=kws,
        graph=sub_adj,
        topic_adj=topic_adj,
        keyword_embed_fn=keyword_embed_fn,
    )


# ---- OA citation curriculum ----------------------------------------------

def load_oa(run_dir: Path = ART / "gap_curriculum_T11273",
            with_embeddings: bool = False,
            keyword_embed_fn=None) -> CurriculumInputs:
    md = json.loads((run_dir / "metadata.json").read_text())
    p2c = np.load(run_dir / "point_to_cluster.npy")
    kws = _parse_keywords(json.loads((run_dir / "topic_keywords.json").read_text()))
    n_topics = md.get("actual_n_topics", int(p2c.max()) + 1)
    sizes = list(md.get("cluster_sizes", [int((p2c == t).sum()) for t in range(n_topics)]))

    # The summary.md "Topic Overview" table is emitted in curriculum order with
    # one row per cluster id (Rank N == cluster N-1), and carries a precomputed
    # Sim->Target column — reuse it for the ascent metric without re-embedding.
    sim_to_target = _parse_oa_sim_to_target(run_dir / "summary.md", n_topics)
    # The summary already orders topics 0..n-1 in curriculum order.
    order = list(range(n_topics))

    # Slice the citation graph to the clustered top-k for modularity / ordering.
    graph = None
    try:
        adj_full = sp.load_npz(run_dir / "citation_graph.npz").tocsr()
        filtered_ids = json.loads((run_dir / "filtered_work_ids.json").read_text())
        id_to_idx = _oa_id_to_idx()
        rows = np.array([id_to_idx[w] for w in filtered_ids], dtype=np.int64)
        graph = adj_full[rows][:, rows]
    except Exception as e:  # pragma: no cover - Cloud may be unavailable
        _log(f"OA: could not slice citation graph ({e}); graph metrics -> NaN")

    topic_adj = None
    if graph is not None:
        topic_adj = topic_dependency_adjacency(graph, p2c, n_topics)

    # per-topic doc tokens for NPMI from topic_work_ids titles (optional, heavy)
    docs = None
    if with_embeddings:
        docs = _oa_topic_docs(run_dir, n_topics)

    return CurriculumInputs(
        topic_order=order,
        point_to_cluster=p2c,
        n_topics=n_topics,
        cluster_sizes=sizes,
        topic_keywords=kws,
        graph=graph,
        topic_adj=topic_adj,
        topic_sim_to_target=sim_to_target,
        topic_docs_tokens=docs,
        keyword_embed_fn=keyword_embed_fn,
    )


def _parse_oa_sim_to_target(summary_path: Path, n_topics: int) -> Optional[List[float]]:
    if not summary_path.exists():
        return None
    sims: List[float] = []
    in_table = False
    for line in summary_path.read_text().splitlines():
        if line.startswith("| Rank | Size |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                if sims:
                    break
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6 or cells[0] in ("", ":---:"):
                continue
            try:
                sims.append(float(cells[5]))  # Sim->Target column
            except (ValueError, IndexError):
                continue
    if len(sims) != n_topics:
        return None
    return sims


_OA_ID_TO_IDX = None


def _oa_id_to_idx() -> Dict[str, int]:
    global _OA_ID_TO_IDX
    if _OA_ID_TO_IDX is None:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from bookgap.configs.cloud.cloud_config import CloudConfig
        _log("OA: loading cloud for id->idx (one-time)...")
        cloud = instantiate(OmegaConf.structured(CloudConfig()))
        _OA_ID_TO_IDX = {w: i for i, w in enumerate(cloud.work_ids())}
        _log(f"OA: cloud loaded ({len(_OA_ID_TO_IDX):,} works)")
    return _OA_ID_TO_IDX


def _oa_topic_docs(run_dir: Path, n_topics: int) -> Optional[Dict[int, List[List[str]]]]:
    """Per-topic title tokens from topic_work_ids.json via the corpus parquet.
    Heavy (Cloud title lookup); only used when --with-embeddings is set."""
    try:
        twi = json.loads((run_dir / "topic_work_ids.json").read_text())
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from bookgap.configs.cloud.cloud_config import CloudConfig
        cloud = instantiate(OmegaConf.structured(CloudConfig()))
        wids = cloud.work_ids()
        names = cloud.work_titles()
        name_by_id = dict(zip(wids, names))
        docs: Dict[int, List[List[str]]] = {}
        for t in range(n_topics):
            titles = [name_by_id.get(w, "") for w in twi.get(str(t), [])]
            docs[t] = _tokenize_docs(titles)
        return docs
    except Exception as e:  # pragma: no cover
        _log(f"OA: topic docs unavailable ({e})")
        return None


# ---- dashboard ------------------------------------------------------------

# Display order + short labels for the dashboard rows.
_DASHBOARD_ROWS = [
    ("sim_to_target_spearman", "Sim->Tgt ascent (Spearman)"),
    ("sim_to_target_kendall", "Sim->Tgt ascent (Kendall)"),
    ("dag_prereq_completeness", "DAG prereq-completeness"),
    ("directionality_indegree_kendall", "[flag] dir. tau vs in-degree"),
    ("directionality_pubyear_kendall", "[flag] dir. tau vs pub-year"),
    ("topic_coherence_npmi", "Topic coherence (NPMI)"),
    ("keyword_coherence_embed", "Keyword coherence (embed, mean)"),
    ("keyword_coherence_embed_min", "Keyword coherence (embed, worst)"),
    ("cross_topic_distinctness_jaccard", "Cross-topic distinctness (Jacc)"),
    ("avg_topics_per_keyword", "Keyword crowding (tpc/kw)"),
    ("n_keywords_in_3plus_topics", "#kw in 3+ topics"),
    ("modularity_q", "Modularity Q"),
    ("size_entropy_norm", "Size entropy (norm)"),
    ("size_gini", "Size Gini"),
    ("n_singletons", "#singleton topics"),
    ("step_smoothness_mean", "Step smoothness (mean)"),
    ("step_smoothness_max", "Step smoothness (max)"),
]


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  n/a"
    if abs(v) >= 100 or (v == int(v)):
        return f"{v:6.0f}"
    return f"{v:6.3f}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-embeddings", action="store_true",
                    help="enable OA NPMI doc tokens (loads the Cloud; slow).")
    ap.add_argument("--no-keyword-coherence", action="store_true",
                    help="skip the embedding-based keyword coherence (which "
                         "loads GIST-small). On by default since it is the "
                         "metric that catches grab-bag topics.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="optional path to dump the raw metric dicts as JSON.")
    args = ap.parse_args()

    kw_embed = None if args.no_keyword_coherence else _gist_keyword_embedder()

    loaders = [
        ("wiki-native", lambda: load_wiki_native(keyword_embed_fn=kw_embed)),
        ("joint-v1", lambda: load_joint(QC_CACHE / "joint_v1", keyword_embed_fn=kw_embed)),
        ("joint-v2", lambda: load_joint(QC_CACHE / "joint_v2", keyword_embed_fn=kw_embed)),
        ("OA-cite-T11273", lambda: load_oa(with_embeddings=args.with_embeddings,
                                           keyword_embed_fn=kw_embed)),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, loader in loaders:
        try:
            _log(f"scoring {name}...")
            inputs = loader()
            results[name] = score_curriculum(inputs)
        except Exception as e:
            _log(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)

    if not results:
        _log("no curricula scored")
        return

    names = list(results.keys())
    header = f"{'metric':<32}" + "".join(f"{n:>16}" for n in names)
    print()
    print(header)
    print("-" * len(header))
    for key, label in _DASHBOARD_ROWS:
        row = f"{label:<32}"
        for n in names:
            row += f"{_fmt(results[n].get(key, float('nan'))):>16}"
        print(row)
    print()
    print("Directions: higher=better unless noted. lower=better: keyword "
          "crowding, #kw-in-3+, size Gini, #singletons, step-smoothness. "
          "[flag]=sanity check (report, don't optimize): negative pub-year tau "
          "or POSITIVE in-degree tau => frontier-first ordering.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        _log(f"wrote raw metrics to {args.json_out}")


if __name__ == "__main__":
    main()
