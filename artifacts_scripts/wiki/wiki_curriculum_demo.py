#!/usr/bin/env python
# artifacts_scripts/wiki/wiki_curriculum_demo.py
"""Curriculum on the **cirrus-native** Wikipedia link graph (vs OpenAlex
citations), to test whether Wikipedia gives more *pedagogical* curricula.

Motivation (docs/backlog/curriculum-generalization-pedagogical-subspace.md):
Wikipedia is explicitly pedagogical — its link structure encodes "what you need
to know to understand X" more directly than a citation graph (which encodes
"what this paper built on", biased toward recent/citable work). This demo runs
the curriculum engine's PageRank -> top-k -> cluster -> keyword/summary steps on
a (Wikipedia link graph, Wikipedia GIST embeddings) pair.

**Coherence:** the link graph is parsed from the SAME CirrusSearch content dump
(``enwiki-20251229``) as the embeddings — every article carries its
``outgoing_link`` list natively, so there is NO cross-dump page_id/date mismatch
and NO bookgen dependency.

Dump-pass budget (the whole point of this rewrite)
--------------------------------------------------
The 43 GB gzip dump is streamed at most **twice per fresh run**, and **zero**
times once the per-run caches exist:

  1. ``load_or_build_title_maps`` — the one unavoidable *full-dump* pass
     (title + redirect maps). Cached at
     ``<store>/title_maps.parquet``; 0 dump passes when present.
  2. neighbor-resolution pass — buffers each *selected* page's resolved
     out-neighbor pids (this is the old standalone ``out_neighbor_pids`` pass,
     now the only thing that reads the dump before the graph build).
  3. ``build_link_graph`` pass — builds the directed CSR over the union node
     set ``sel ∪ neighbors``.

Abstracts (``opening_text``) are NOT a separate dump pass: they are already
materialized as the ``abstract`` column of ``<store>/articles.parquet`` (the
embedding store was built from cirrus ``opening_text``), so we read them
column-selectively for the node rows — zero dump access.

Subgraph cache
--------------
After the graph is built, ``(id_list, adj CSR, titles, abstracts, aligned
embeddings)`` is persisted to
``<store>/curriculum_subgraph_n{n_select}.npz``. ``--reuse-subgraph`` loads it
and goes straight to PageRank -> cluster -> keywords -> summary with **zero**
dump access, making seed/alpha/cluster tuning instant.

Observability
-------------
Phase progress is written to a *flushed* sidecar log
``artifacts/wiki_curriculum/run.log`` (``conda run`` buffers stdout, so plain
prints are invisible mid-run; the sidecar handle is flushed per phase).

Usage::

    conda run -p .conda python -m artifacts_scripts.wiki.wiki_curriculum_demo \
        --wiki-store cache/wiki/store/enwiki_full \
        --n-select 40000 --ranking-top-k 4000 --desired-n-topics 12 \
        --use-abstracts

Add ``--bg-norm`` (with optional ``--bg-alpha``, default 0.5) to rank the top-k
SELECTION by the background-normalized score ``p^a · log((p+ε)/(g+ε))`` instead
of raw seeded PR mass ``p`` — where ``g`` is a uniform-teleport (no-seed)
background PR on the same adjacency. This demotes globally-popular
bibliographic-infrastructure hubs (DOI/ISBN/PubMed/ArXiv/ISO), which have huge
``g``. Reuses the OpenAlex curriculum's scoring (see
``artifacts_scripts/curriculum/build_curriculum.py``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray

from artifacts_scripts.curriculum.keyword_extraction import (
    BackgroundMode,
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    extract_topic_keywords,
    filter_stopwords,
    mine_phrases_lattice,
    parse_with_phrases,
    tokenize,
)
from artifacts_scripts.curriculum.build_curriculum import (
    _compute_linearization_order_from_adjacency,
)
from artifacts_scripts.curriculum.pagerank import (
    compute_frontier_adjacency,
    compute_pagerank,
)
from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage
from bookgap.util import CACHE_DIR
from bookgap.wikipedia.corpus import WikipediaCorpus
from bookgap.wikipedia.link_graph import (
    _is_non_article_link,
    _iter_content_docs,
    _norm_title,
    build_link_graph,
    load_or_build_title_maps,
)

DEFAULT_DUMP = CACHE_DIR / "wiki" / "dumps" / "enwiki-20251229-cirrussearch-content.json.gz"
RUN_LOG = Path("artifacts/wiki_curriculum/run.log")

# Core NLP pages to seed PageRank from (resolved to page_ids via the title map).
# "Natural language processing" anchors the curriculum; the others pin a few
# canonical NLP entry points so the forward walk lands on prerequisites rather
# than one article's idiosyncratic out-links.
SEED_TITLES = [
    "Natural language processing",
    "Machine learning",
    "Language model",
    "Word embedding",
    "Parsing",
    "Computational linguistics",
]


class PhaseLog:
    """Tiny flushed sidecar logger. Every ``phase``/``info`` writes a line to
    ``run.log`` AND echoes to stdout, flushing both so progress is visible even
    under ``conda run``'s stdout buffering."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = open(path, "w", encoding="utf-8")
        self._t0 = time.time()
        self.info(f"run.log opened at {path}")

    def _emit(self, line: str) -> None:
        stamp = f"[{time.time() - self._t0:7.1f}s] {line}"
        self._fh.write(stamp + "\n")
        self._fh.flush()
        print(stamp, flush=True)

    def phase(self, name: str) -> None:
        self._emit(f"=== PHASE: {name}")

    def info(self, line: str) -> None:
        self._emit(f"    {line}")

    def close(self) -> None:
        self._emit("run complete")
        self._fh.close()


def _normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def _row_group_embeddings(pf: pq.ParquetFile, rg: int) -> NDArray[np.float32]:
    """Load one row group of the embeddings parquet as (n, d) float32."""
    tbl = pf.read_row_group(rg, columns=["embedding"])
    col = tbl.column("embedding").combine_chunks()
    d = col.type.list_size
    flat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return np.ascontiguousarray(flat.reshape(tbl.num_rows, d))


def select_nlp_pages(
    store_dir: Path,
    nlp_dir: NDArray[np.float32],
    n_select: int,
    log: PhaseLog,
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """Stream the 7.1M wiki embeddings (per parquet row group) and return the
    top ``n_select`` page rows by cosine to ``nlp_dir`` (mean-centered).

    Returns ``(row_indices, page_ids, embeddings)`` for the selected rows. The
    embeddings are raw GIST (NOT centered) so downstream clustering uses the
    same space as the OpenAlex curriculum. No dump access — embeddings only.
    """
    emb_path = store_dir / "embeddings.parquet"
    pf = pq.ParquetFile(emb_path)
    n_rg = pf.num_row_groups

    # Pass 1: streaming centroid (mean over all rows). Accumulate in float64.
    t = time.time()
    acc = np.zeros(384, dtype=np.float64)
    total = 0
    for rg in range(n_rg):
        E = _row_group_embeddings(pf, rg)
        En = _normalize(E)
        acc += En.sum(axis=0)
        total += En.shape[0]
    centroid = (acc / max(total, 1)).astype(np.float32)
    log.info(f"select: streamed {total} rows for centroid in {time.time()-t:.1f}s")

    # Pass 2: cosine of (centered, renormalized) row to nlp_dir; keep a running
    # top-n via argpartition over accumulated (score, row) chunks.
    t = time.time()
    all_scores: List[NDArray[np.float32]] = []
    for rg in range(n_rg):
        E = _row_group_embeddings(pf, rg)
        En = _normalize(E)
        Ec = _normalize(En - centroid)
        all_scores.append((Ec @ nlp_dir).astype(np.float32))
    scores_full = np.concatenate(all_scores)
    n_sel = min(n_select, scores_full.shape[0])
    part = np.argpartition(-scores_full, n_sel - 1)[:n_sel]
    sel_rows = part[np.argsort(-scores_full[part])].astype(np.int64)
    log.info(
        f"select: scored {scores_full.shape[0]} rows in {time.time()-t:.1f}s; "
        f"kept top-{n_sel} (cos {scores_full[sel_rows].min():.3f}.."
        f"{scores_full[sel_rows].max():.3f})"
    )

    page_ids_all = np.asarray(
        pq.read_table(store_dir / "articles.parquet", columns=["page_id"])
        .column("page_id")
        .to_numpy(),
        dtype=np.int64,
    )
    sel_page_ids = page_ids_all[sel_rows]

    # Load just the selected embedding rows (sorted for a single streaming pass).
    order = np.argsort(sel_rows)
    sel_rows_sorted = sel_rows[order]
    sel_emb_sorted = np.empty((n_sel, 384), dtype=np.float32)
    cursor = 0
    rg_start = 0
    for rg in range(n_rg):
        E = _row_group_embeddings(pf, rg)
        rg_end = rg_start + E.shape[0]
        mask = (sel_rows_sorted >= rg_start) & (sel_rows_sorted < rg_end)
        local = sel_rows_sorted[mask] - rg_start
        if local.size:
            sel_emb_sorted[cursor : cursor + local.size] = E[local]
            cursor += local.size
        rg_start = rg_end
    inv = np.empty_like(order)
    inv[order] = np.arange(n_sel)
    sel_emb = sel_emb_sorted[inv]
    return sel_rows, sel_page_ids, sel_emb


def resolve_out_neighbors(
    dump_path: Path,
    source_pids: set[int],
    title_to_pid: Dict[str, int],
    redirect_to_pid: Dict[str, int],
) -> set[int]:
    """ONE dump pass: resolved cirrus out-neighbor page_ids of ``source_pids``.

    For every article whose page_id is a source, resolves each ``outgoing_link``
    to a target page_id (title map, redirect fallback) and collects it. Grows
    the NLP node set by one link hop — prerequisites are often the linked-to
    pages just below a topic.
    """
    out: set[int] = set()
    for doc in _iter_content_docs(dump_path):
        pid = doc.get("__action_id") or doc.get("page_id")
        if pid is None or int(pid) not in source_pids:
            continue
        for raw in doc.get("outgoing_link") or ():
            nt = _norm_title(raw)
            if not nt or _is_non_article_link(nt):
                continue
            tgt = title_to_pid.get(nt) or redirect_to_pid.get(nt)
            if tgt is not None:
                out.add(tgt)
    return out


def materialize_embeddings(
    store_dir: Path,
    need_pids: set[int],
    sink: Dict[int, NDArray[np.float32]],
) -> None:
    """Stream the embeddings parquet once and fill ``sink[pid]`` for the rows
    in ``need_pids`` (selected pages already have theirs). No dump access."""
    if not need_pids:
        return
    page_ids_all = np.asarray(
        pq.read_table(store_dir / "articles.parquet", columns=["page_id"])
        .column("page_id")
        .to_numpy(),
        dtype=np.int64,
    )
    pid_to_global_row = {int(p): i for i, p in enumerate(page_ids_all)}
    need_rows = np.array(
        sorted(pid_to_global_row[p] for p in need_pids if p in pid_to_global_row),
        dtype=np.int64,
    )
    pf = pq.ParquetFile(store_dir / "embeddings.parquet")
    rg_start = 0
    for rg in range(pf.num_row_groups):
        E = _row_group_embeddings(pf, rg)
        rg_end = rg_start + E.shape[0]
        mask = (need_rows >= rg_start) & (need_rows < rg_end)
        for gr in need_rows[mask]:
            sink[int(page_ids_all[gr])] = E[gr - rg_start]
        rg_start = rg_end


def load_abstracts(store_dir: Path, want: set[int]) -> Dict[int, str]:
    """Read ``opening_text`` abstracts for ``want`` page_ids from the store's
    ``articles.parquet`` ``abstract`` column. ZERO dump access — abstracts were
    materialized into the store at embed time."""
    tbl = pq.read_table(store_dir / "articles.parquet", columns=["page_id", "abstract"])
    pids = tbl.column("page_id").to_numpy()
    abstracts = tbl.column("abstract").to_pylist()
    out: Dict[int, str] = {}
    for p, ab in zip(pids, abstracts):
        ip = int(p)
        if ip in want and ab:
            out[ip] = ab
    return out


def cluster_embeddings(
    embeddings: NDArray[np.float32],
    masses: NDArray[np.float64],
    desired_n_topics: int,
) -> NDArray[np.intp]:
    """Bisecting-K-Means on the (top-k) embeddings, cut to ~desired_n_topics.

    Mirrors the engine's default ``bisecting_kmeans`` dendrogram path plus a
    mass-threshold frontier cut, inlined so we don't need a ``Cloud``.
    """
    from artifacts_scripts.curriculum.dendrogram_ops import (
        compute_dendrogram_masses,
        find_mass_threshold_for_n_topics,
        get_frontier_nodes,
        map_leaves_to_frontier_clusters,
    )

    res = bisecting_kmeans_linkage(
        embeddings, sample_weights=None, min_cluster_size=1, verbose=False,
    )
    linked = res.linkage
    leaf_assign = res.leaf_cluster_assignments
    n_leaf = res.n_leaf_clusters
    leaf_masses = np.zeros(n_leaf, dtype=np.float64)
    np.add.at(leaf_masses, leaf_assign, masses)
    node_masses = compute_dendrogram_masses(linked, leaf_masses)
    threshold = find_mass_threshold_for_n_topics(linked, node_masses, desired_n_topics)
    frontier_nodes = get_frontier_nodes(linked, node_masses, threshold)
    leaf_to_frontier = map_leaves_to_frontier_clusters(linked, frontier_nodes, n_leaf)
    return leaf_to_frontier[leaf_assign]


def keywords_for_topics(
    texts: List[str],
    point_to_cluster: NDArray[np.intp],
    masses: NDArray[np.float64],
    n_topics: int,
    samples_per_topic: int = 80,
    top_k: int = 12,
) -> List[List[Tuple[str, float]]]:
    """Replicate ``summarize_topics``' keyword pipeline (lattice phrase mining
    + log-odds) directly on in-memory per-topic token lists."""
    topic_raw: List[List[List[str]]] = []
    for t in range(n_topics):
        idxs = np.where(point_to_cluster == t)[0]
        if idxs.size == 0:
            topic_raw.append([])
            continue
        order = np.argsort(masses[idxs])[::-1]
        pick = idxs[order[:samples_per_topic]]
        docs = []
        for i in pick:
            toks = tokenize(texts[int(i)])
            if toks:
                docs.append(toks)
        topic_raw.append(docs)

    flat = [doc for docs in topic_raw for doc in docs]
    phrase_vocab: set = set()
    parsed = flat
    for _ in range(6):
        new = mine_phrases_lattice(
            parsed, npmi_threshold=0.4, min_boundary_be=0.5,
            count_floor=3, max_phrases=5000,
        )
        new -= phrase_vocab
        if not new:
            break
        phrase_vocab |= new
        parsed = [parse_with_phrases(doc, phrase_vocab) for doc in flat]
    merged: List[List[List[str]]] = []
    off = 0
    for docs in topic_raw:
        merged.append(parsed[off : off + len(docs)])
        off += len(docs)

    topic_doc_tokens: List[List[List[str]]] = []
    for docs in merged:
        out = []
        for toks in docs:
            toks = filter_stopwords(toks)
            if toks:
                out.append(toks)
        topic_doc_tokens.append(out)

    return extract_topic_keywords(
        topic_doc_tokens,
        method=KeywordMethod.LOG_ODDS,
        top_k=top_k,
        doc_weighting=DocWeighting.EQUAL,
        topic_weighting=TopicWeighting.EQUAL,
        background_mode=BackgroundMode.MARGINAL,
        weight_by_freq=True,
    )


def save_subgraph(
    path: Path,
    id_list: List[int],
    adj: sp.csr_matrix,
    titles: List[str],
    abstracts: List[str],
    emb: NDArray[np.float32],
) -> None:
    """Persist the built subgraph so curriculum re-runs skip the dump entirely.

    One ``.npz`` holds the page-id list, the CSR adjacency (indptr/indices/data
    + shape), titles + abstracts (as object arrays), and the aligned embedding
    matrix. ``--reuse-subgraph`` reloads this and goes straight to PageRank.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        id_list=np.asarray(id_list, dtype=np.int64),
        adj_indptr=adj.indptr,
        adj_indices=adj.indices,
        adj_data=adj.data,
        adj_shape=np.asarray(adj.shape, dtype=np.int64),
        titles=np.asarray(titles, dtype=object),
        abstracts=np.asarray(abstracts, dtype=object),
        emb=emb,
    )


def load_subgraph(
    path: Path,
) -> Tuple[List[int], sp.csr_matrix, List[str], List[str], NDArray[np.float32]]:
    """Load a subgraph persisted by :func:`save_subgraph`."""
    z = np.load(path, allow_pickle=True)
    adj = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]),
        shape=tuple(int(x) for x in z["adj_shape"]),
    )
    id_list = [int(p) for p in z["id_list"]]
    titles = [str(s) for s in z["titles"]]
    abstracts = [str(s) for s in z["abstracts"]]
    emb = np.ascontiguousarray(z["emb"], dtype=np.float32)
    return id_list, adj, titles, abstracts, emb


def build_subgraph(
    args: argparse.Namespace, log: PhaseLog
) -> Tuple[List[int], sp.csr_matrix, List[str], List[str], NDArray[np.float32]]:
    """Build (or reuse) the cirrus link subgraph + aligned titles/abstracts/emb.

    Dump passes: 0 if ``--reuse-subgraph`` and the cache exists; otherwise
    ``load_or_build_title_maps`` (0 if its parquet exists, else 1 full pass) +
    neighbor-resolution pass (1) + ``build_link_graph`` pass (1) = at most 2
    dump passes after title maps.
    """
    subgraph_cache = args.wiki_store / f"curriculum_subgraph_n{args.n_select}.npz"
    if args.reuse_subgraph and subgraph_cache.exists():
        log.phase(f"reuse-subgraph (cache hit: {subgraph_cache.name}) — ZERO dump passes")
        id_list, adj, titles, abstracts, emb = load_subgraph(subgraph_cache)
        log.info(f"loaded {len(id_list)} nodes, {adj.nnz} edges from cache")
        return id_list, adj, titles, abstracts, emb

    # --- NLP direction -------------------------------------------------------
    log.phase("nlp-direction")
    if args.nlp_dir.exists():
        nlp_dir = np.load(args.nlp_dir).astype(np.float32)
        log.info(f"loaded NLP direction from {args.nlp_dir}")
    else:
        from artifacts_scripts.wiki.cross_corpus_gap_demo import load_openalex_nlp
        _, articles, _ = load_openalex_nlp()
        arts = _normalize(np.ascontiguousarray(articles.embeddings(), dtype=np.float32))
        nlp_dir = arts.mean(axis=0)
        nlp_dir = (nlp_dir / (np.linalg.norm(nlp_dir) or 1.0)).astype(np.float32)
        np.save(args.nlp_dir, nlp_dir)
        log.info("computed NLP direction from OpenAlex NLP articles")

    # --- Select NLP-relevant wiki pages (embeddings only, no dump) -----------
    log.phase(f"select-nlp-pages (top-{args.n_select}, streamed)")
    sel_rows, sel_page_ids, sel_emb = select_nlp_pages(
        args.wiki_store, nlp_dir, args.n_select, log
    )
    sel_pid_to_emb: Dict[int, NDArray[np.float32]] = {
        int(p): sel_emb[i] for i, p in enumerate(sel_page_ids)
    }

    # --- Title/redirect maps (CACHED full-dump pass) -------------------------
    title_cache = args.wiki_store / "title_maps.parquet"
    cached = title_cache.exists()
    log.phase(f"title-maps (cached? {'YES' if cached else 'NO — full dump pass'})")
    t = time.time()
    title_to_pid, redirect_to_pid = load_or_build_title_maps(args.dump, title_cache)
    pid_to_title = {pid: ttl for ttl, pid in title_to_pid.items()}
    log.info(
        f"{len(title_to_pid)} titles, {len(redirect_to_pid)} redirects "
        f"in {time.time()-t:.1f}s (cache {'hit' if cached else 'MISS, wrote ' + str(title_cache)})"
    )

    # --- Neighbor resolution (DUMP PASS 1 of <=2) ----------------------------
    embedded_pids = set(
        int(p)
        for p in pq.read_table(
            args.wiki_store / "articles.parquet", columns=["page_id"]
        ).column("page_id").to_numpy()
    )
    log.phase("resolve-out-neighbors (dump pass 1/2)")
    t = time.time()
    sel_set = set(int(p) for p in sel_page_ids)
    neigh = resolve_out_neighbors(args.dump, sel_set, title_to_pid, redirect_to_pid)
    neigh_embedded = (neigh & embedded_pids) - sel_set
    node_pids = sel_set | neigh_embedded
    id_list = sorted(node_pids)
    log.info(
        f"selected={len(sel_set)}, out-neighbors(embedded,new)={len(neigh_embedded)}, "
        f"node set={len(node_pids)} in {time.time()-t:.1f}s"
    )

    # --- Embeddings for new neighbor rows (no dump) --------------------------
    materialize_embeddings(
        args.wiki_store, node_pids - set(sel_pid_to_emb), sel_pid_to_emb
    )

    # --- Build the cirrus link subgraph (DUMP PASS 2 of <=2) -----------------
    log.phase(f"build-link-graph over {len(id_list)} nodes (dump pass 2/2)")
    t = time.time()
    lg = build_link_graph(
        args.dump, id_list,
        title_to_pid=title_to_pid, redirect_to_pid=redirect_to_pid,
    )
    adj = lg.adj
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    log.info(
        f"subgraph: {adj.shape[0]} nodes, {adj.nnz} directed edges "
        f"(dropped {lg.n_links_dropped} out-of-set) in {time.time()-t:.1f}s"
    )
    log.info(
        f"out-degree: mean={out_deg.mean():.1f}, max={int(out_deg.max())}, "
        f"frac-with-edge={(out_deg>0).mean():.2f}"
    )

    emb_aligned = np.stack([sel_pid_to_emb[p] for p in id_list]).astype(np.float32)
    titles_aligned = [pid_to_title.get(p, str(p)) for p in id_list]

    # --- Abstracts: from store articles.parquet (ZERO dump access) -----------
    if args.use_abstracts:
        log.phase("load-abstracts (from store articles.parquet — no dump)")
        t = time.time()
        abstr = load_abstracts(args.wiki_store, set(id_list))
        abstracts_aligned = [abstr.get(p, "") for p in id_list]
        log.info(
            f"joined {sum(1 for a in abstracts_aligned if a)} abstracts "
            f"in {time.time()-t:.1f}s"
        )
    else:
        abstracts_aligned = ["" for _ in id_list]

    # --- Persist subgraph cache (re-runs skip the dump entirely) -------------
    log.phase(f"save-subgraph -> {subgraph_cache.name}")
    save_subgraph(
        subgraph_cache, id_list, adj, titles_aligned, abstracts_aligned, emb_aligned
    )
    log.info(f"wrote {subgraph_cache} ({subgraph_cache.stat().st_size/1e6:.1f} MB)")

    return id_list, adj, titles_aligned, abstracts_aligned, emb_aligned


def compute_background_inflow(
    adj_csr: sp.csr_matrix,
    leaf_to_cluster_full: NDArray[np.int64],
    n_topics: int,
    alpha: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> NDArray[np.float64]:
    """Background analog of ``compute_frontier_adjacency``'s column sums.

    ``compute_frontier_adjacency`` builds ``A_fg[i,j]`` = absorbed PageRank mass
    that lands on topic *j*'s nodes when seeded from topic *i*'s nodes, with all
    OTHER topic nodes made absorbing (α=0) so only direct i→j flow is counted.
    Its column sum ``A_fg[:,k].sum()`` is the **seeded in-flow** to topic *k* —
    how much every other topic depends on *k*. That in-flow is inflated for a
    globally-popular topic (lots of topics link into it regardless of any
    genuine prerequisite relationship) — the topic-level version of the
    bibliographic-hub effect that node-selection bg-norm fixes.

    This computes the matching **background in-flow** ``inflow_bg(k)``: the same
    absorbing PageRank, but seeded from a UNIFORM distribution over all nodes
    (no NLP/topic seeding) — i.e. the un-seeded teleport ``g`` of the selection
    bg-norm, projected through the same absorbing kernel. A topic that absorbs a
    lot of mass purely from generic graph structure has a large ``inflow_bg``;
    dividing the seeded in-flow by it removes the popularity baseline, exactly
    mirroring ``p^a·log(p/g)`` at the topic level.

    The absorbing-α construction (all assigned nodes absorb) is identical to
    ``compute_frontier_adjacency`` — the only change is a single uniform source
    column instead of per-topic seed columns — so ``inflow_fg`` and
    ``inflow_bg`` are an apples-to-apples ratio.
    """
    n = adj_csr.shape[0]
    # Uniform global source (un-seeded teleport), single batch.
    uniform_seed = np.full((n, 1), 1.0 / n, dtype=np.float64)
    # Absorbing α: every node assigned to ANY topic absorbs (α=0); all other
    # (intermediary, -1) nodes follow links at α. This is the same rule as
    # compute_frontier_adjacency, minus the "restore α on the source topic's own
    # leaves" step — under a uniform source there is no single source topic, and
    # we want flow to be absorbed the first time it reaches ANY topic so the
    # in-flow is a direct (non-transitive) measure, matching A_fg's off-diagonal.
    alpha_vec = np.full((n, 1), alpha, dtype=np.float64)
    assigned = np.where(leaf_to_cluster_full != -1)[0]
    alpha_vec[assigned, :] = 0.0
    landed = compute_pagerank(
        adj_csr, uniform_seed, alpha=alpha_vec, tol=tol, max_iter=max_iter,
        normalize=True,
    ).ravel()
    inflow_bg = np.zeros(n_topics, dtype=np.float64)
    for k in range(n_topics):
        members = np.where(leaf_to_cluster_full == k)[0]
        if members.size:
            inflow_bg[k] = landed[members].sum()
    return inflow_bg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/enwiki_full"))
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--nlp-dir", type=Path, default=Path("/tmp/nlp_dir.npy"),
                    help="precomputed OpenAlex NLP centroid direction (384-D). "
                         "If missing, recomputed from the OpenAlex NLP slice.")
    ap.add_argument("--n-select", type=int, default=40000,
                    help="how many top-cosine NLP-relevant wiki pages to keep")
    ap.add_argument("--ranking-top-k", type=int, default=4000,
                    help="top-k pages by PageRank mass to cluster")
    ap.add_argument("--desired-n-topics", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping")
    ap.add_argument("--bg-norm", action="store_true",
                    help="rank the top-k SELECTION by the background-normalized "
                         "score p^a * log((p+eps)/(g+eps)) instead of raw seeded "
                         "PR mass p, where g is a uniform-teleport (no-seed) "
                         "background PR on the same adjacency. Demotes globally "
                         "popular hubs (DOI/ISBN/PubMed) that have huge g. Reuses "
                         "the OpenAlex curriculum's scoring (see "
                         "artifacts_scripts/curriculum/build_curriculum.py).")
    ap.add_argument("--bg-alpha", type=float, default=0.5,
                    help="exponent a in p^a * log(p/g). 0.0 = raw log-lift, "
                         "0.5 = Fightin'-Words variance-norm (default, matches the "
                         "OpenAlex curriculum default), 1.0 = pointwise KL.")
    ap.add_argument("--order-by",
                    choices=["lift", "mass", "frontier", "frontier-bgnorm"],
                    default=None,
                    help="how to ORDER the topics (decoupled from SELECTION):\n"
                         "  lift  = mean background-normalized lift (relevance "
                         "axis; default when --bg-norm is set);\n"
                         "  mass  = total raw forward-PR mass (default without "
                         "--bg-norm);\n"
                         "  frontier = directional pairwise-PR-DAG ordering via "
                         "compute_frontier_adjacency over the RAW forward-PR flow "
                         "(adj[i,j]=how much topic i depends on j), then "
                         "in-degree minus out-degree (prerequisites first). This "
                         "decouples ORDERING (raw forward PR) from SELECTION "
                         "(--bg-norm), so genuine foundations land first and "
                         "frontier seeds land last regardless of how the top-k "
                         "was chosen.\n"
                         "  frontier-bgnorm = like 'frontier', but the inter-topic "
                         "IN-FLOW is background-normalized so a topic is not "
                         "rewarded for being globally popular. Score(k) = "
                         "inflow_fg(k)^a * log((inflow_fg(k)+ε)/(inflow_bg(k)+ε)) "
                         "− outflow_fg(k), where inflow_fg/outflow_fg are the "
                         "seeded absorbing-PR in/out flows on the frontier "
                         "adjacency and inflow_bg is the same absorbing in-flow "
                         "under a UNIFORM (un-seeded) global source. Mirrors the "
                         "p^a·log(p/g) node-selection bg-norm, applied at the "
                         "topic level (a = --bg-alpha).")
    ap.add_argument("--use-abstracts", action="store_true",
                    help="join opening_text abstracts (from the store's "
                         "articles.parquet — NO extra dump pass) for keyword text")
    ap.add_argument("--reuse-subgraph", action="store_true",
                    help="if the subgraph .npz cache exists, load it and skip "
                         "ALL dump access (PageRank/cluster/keyword tuning only)")
    ap.add_argument("--out", type=Path, default=Path("docs/wikipedia-curriculum.md"))
    args = ap.parse_args()

    log = PhaseLog(RUN_LOG)
    log.info("Cirrus-native Wikipedia-link-graph curriculum demo")
    wiki = WikipediaCorpus(args.wiki_store)
    log.info(f"wiki store fingerprint: {wiki.embedder_fingerprint}")

    # --- Build (or reuse) the subgraph ---------------------------------------
    id_list, adj, titles_aligned, abstracts_aligned, emb_aligned = build_subgraph(
        args, log
    )
    pid_to_title = dict(zip(id_list, titles_aligned))
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    if args.use_abstracts and any(abstracts_aligned):
        texts_aligned = [
            f"{titles_aligned[i]}. {abstracts_aligned[i]}".strip()
            for i in range(len(id_list))
        ]
    else:
        texts_aligned = list(titles_aligned)

    # --- PageRank seeded at core NLP pages -----------------------------------
    log.phase("pagerank (forward, mass -> prerequisites)")
    pid_index = {pid: i for i, pid in enumerate(id_list)}
    # Resolve seed titles via the aligned title list (works in reuse mode too,
    # where no title map is loaded).
    title_to_local = {t: pid_index[p] for p, t in zip(id_list, titles_aligned)}
    seed_idx = [title_to_local[t] for t in SEED_TITLES if t in title_to_local]
    if not seed_idx:
        log.info("ERROR: no seed pages found in subgraph — aborting")
        log.close()
        raise SystemExit("No seed pages found in subgraph — aborting.")
    log.info(f"seed pages in subgraph: {[titles_aligned[i] for i in seed_idx]}")
    seeds = np.zeros(len(id_list), dtype=np.float64)
    for i in seed_idx:
        seeds[i] = 1.0 / len(seed_idx)

    # Forward PageRank: A[i,j]=1 means i links to j, so mass flows i->j and
    # accumulates on foundational (heavily-linked-to) prerequisite pages —
    # exactly the citation-graph semantics the engine assumes.
    adj_csr = adj.tocsr()
    t = time.time()
    masses = compute_pagerank(
        adj_csr, seeds, alpha=args.alpha, tol=1e-9, max_iter=200, normalize=True
    )
    log.info(f"pagerank converged in {time.time()-t:.1f}s")

    # --- Optional background normalization of the SELECTION score ------------
    # Reuses the OpenAlex curriculum's scoring (build_curriculum.py): rank the
    # top-k by p^a * log((p+eps)/(g+eps)) where p = seeded forward PR (=`masses`)
    # and g = a uniform-teleport (no-seed) background PR on the SAME adjacency,
    # at the same damping (alpha_bg = alpha_fg, the curriculum's canonical
    # default). Hubs like DOI/ISBN/PubMed have huge background mass g, so their
    # log-lift log(p/g) is small/negative and they sink in the ranking. Cluster
    # weights (`top_masses` below) stay RAW seeded PR mass, matching the
    # curriculum (background norm is a selection-only re-rank).
    selection_masses = masses
    if args.bg_norm:
        a_exp = float(args.bg_alpha)
        eps = 1e-12  # matches CurriculumConfig.background_eps
        log.phase(f"background-normalize (a={a_exp}, alpha_bg={args.alpha}, eps={eps:.0e})")
        t = time.time()
        uniform_seeds = np.full(len(id_list), 1.0 / len(id_list), dtype=np.float64)
        g_bg = compute_pagerank(
            adj_csr, uniform_seeds, alpha=args.alpha, tol=1e-9, max_iter=200,
            normalize=True,
        )
        p_fg = np.maximum(masses, 0.0)
        selection_masses = np.power(p_fg, a_exp) * np.log((p_fg + eps) / (g_bg + eps))
        log.info(
            f"background PR converged in {time.time()-t:.1f}s; "
            f"selection score min={float(selection_masses.min()):.3e} "
            f"max={float(selection_masses.max()):.3e}"
        )

    # --- Top-k by (possibly normalized) selection score, then cluster --------
    top_k = min(args.ranking_top_k, len(id_list))
    top_idx = np.argsort(selection_masses)[-top_k:][::-1]
    log.phase(f"cluster top-{top_k} (retains {100*masses[top_idx].sum():.1f}% PR mass)")
    top_emb = emb_aligned[top_idx]
    # Cluster weights stay RAW seeded PR mass (matches the OpenAlex curriculum;
    # background norm is a selection-only re-rank). When bg-norm is on, topic
    # ordering and representative-page selection use the background-normalized
    # selection score so the demotion of global hubs is reflected in the
    # prerequisite-first ordering, not just in which nodes survive the top-k cut.
    top_masses = masses[top_idx]
    top_rank = selection_masses[top_idx]
    top_texts = [texts_aligned[i] for i in top_idx]
    top_titles = [titles_aligned[i] for i in top_idx]

    t = time.time()
    point_to_cluster = cluster_embeddings(top_emb, top_masses, args.desired_n_topics)
    n_topics = int(point_to_cluster.max()) + 1
    log.info(f"clustered into {n_topics} topics in {time.time()-t:.1f}s")

    # --- Topic ordering (DECOUPLED from selection) ---------------------------
    # SELECTION (which nodes survive the top-k cut + the cluster weights) is
    # always governed by --bg-norm above. ORDERING is a separate axis:
    #   * lift  = mean background-normalized lift. A RELEVANCE axis — it puts the
    #             6 seeds first and genuine foundations last, which is backwards
    #             for a curriculum. Kept as the legacy bg-norm default.
    #   * mass  = total raw forward-PR mass (foundational pages accumulate mass
    #             under forward PR). Legacy default without --bg-norm.
    #   * frontier = directional pairwise-PR-DAG ordering: build the inter-topic
    #             dependency matrix with compute_frontier_adjacency over the RAW
    #             forward-PR flow (NOT the bg-normalized scores), where
    #             adj[i,j] = how much topic i depends on topic j, then sort by
    #             (in-degree − out-degree) so prerequisites come first. A target
    #             that depends on everything has high out-degree / low in-degree
    #             and is ordered LAST; a broadly-depended-on foundation is FIRST.
    order_by = args.order_by or ("lift" if args.bg_norm else "mass")

    # `topic_mass` is the score column shown in the report. For lift/mass it is
    # also the sort key; for frontier it is reported for context (the sort uses
    # the directional in−out score below). Its meaning tracks the SELECTION
    # score so it matches the report's column label (`score_label`): mean lift
    # under --bg-norm, total raw PR mass otherwise.
    if order_by == "lift" or (
        order_by in ("frontier", "frontier-bgnorm") and args.bg_norm
    ):
        topic_mass = np.array(
            [top_rank[point_to_cluster == t].mean() for t in range(n_topics)]
        )
    else:
        topic_mass = np.array(
            [top_masses[point_to_cluster == t].sum() for t in range(n_topics)]
        )

    frontier_in_out: NDArray[np.float64] | None = None
    if order_by in ("frontier", "frontier-bgnorm"):
        # Map EVERY subgraph node to a topic id (or -1 if it was not in the
        # top-k that got clustered). compute_frontier_adjacency expects this
        # full-graph leaf_to_cluster (length == n nodes) plus frontier_nodes
        # whose length is the topic count (the values are only used for |T|).
        # It then runs its OWN absorbing/batched forward PageRank seeded from
        # each topic's nodes over the raw adjacency — so the dependency matrix
        # is built from the RAW forward-PR flow, fully decoupled from the
        # bg-normalized selection score.
        log.phase(f"frontier-adjacency (raw forward-PR DAG over {n_topics} topics)")
        leaf_to_cluster_full = np.full(len(id_list), -1, dtype=np.int64)
        leaf_to_cluster_full[top_idx] = point_to_cluster.astype(np.int64)
        frontier_nodes = np.arange(n_topics, dtype=np.int64)
        t = time.time()
        adj_frontier = compute_frontier_adjacency(
            adj_csr,                 # RAW directed link subgraph
            frontier_nodes,
            leaf_to_cluster_full,
            alpha=args.alpha,        # same damping as the forward PR
            tol=1e-6,
            max_iter=200,
        )
        fa_secs = time.time() - t
        log.info(
            f"compute_frontier_adjacency took {fa_secs:.1f}s "
            f"({fa_secs/60:.1f} min) on the {adj_csr.shape[0]}-node subgraph"
        )
        inflow_fg = np.array([adj_frontier[:, k].sum() for k in range(n_topics)])
        outflow_fg = np.array([adj_frontier[k, :].sum() for k in range(n_topics)])
        if order_by == "frontier":
            # in-degree (depended-on) minus out-degree (depends-on); prereqs first.
            frontier_in_out = inflow_fg - outflow_fg
            order = _compute_linearization_order_from_adjacency(adj_frontier)
            log.info(
                "frontier in−out scores (topic: score): "
                + ", ".join(f"{t}:{frontier_in_out[t]:+.4f}" for t in range(n_topics))
            )
        else:
            # frontier-bgnorm: background-normalize the inter-topic IN-FLOW so a
            # topic is not rewarded for being globally popular. This is the
            # node-selection p^a·log(p/g) bg-norm applied at the topic level:
            #   p = inflow_fg(k)  (seeded absorbing in-flow on the frontier adj)
            #   g = inflow_bg(k)  (same absorbing in-flow under a UNIFORM source)
            #   score(k) = p^a · log((p+ε)/(g+ε)) − outflow_fg(k)
            # The −outflow_fg keeps the directional signal of `frontier` (a topic
            # that depends on everything has large outflow and is ordered last);
            # the bg-normalized in-flow replaces the raw inflow_fg so a hub topic
            # (high inflow only because everything links into it, large g too)
            # gets a small log-lift instead of a big raw in-degree. a = --bg-alpha
            # (canonical 0.5), ε = 1e-12, matching the selection bg-norm exactly.
            a_exp = float(args.bg_alpha)
            eps = 1e-12
            log.phase(
                f"frontier-bgnorm: background inter-topic in-flow "
                f"(a={a_exp}, eps={eps:.0e})"
            )
            t = time.time()
            inflow_bg = compute_background_inflow(
                adj_csr, leaf_to_cluster_full, n_topics,
                alpha=args.alpha, tol=1e-6, max_iter=200,
            )
            log.info(
                f"background inter-topic in-flow computed in {time.time()-t:.1f}s"
            )
            p_in = np.maximum(inflow_fg, 0.0)
            bgnorm_inflow = np.power(p_in, a_exp) * np.log((p_in + eps) / (inflow_bg + eps))
            frontier_in_out = bgnorm_inflow - outflow_fg
            order = np.argsort(-frontier_in_out)
            log.info(
                "frontier-bgnorm scores (topic: inflow_fg/inflow_bg → score): "
                + ", ".join(
                    f"{t}:{inflow_fg[t]:.4f}/{inflow_bg[t]:.4f}→{frontier_in_out[t]:+.4f}"
                    for t in range(n_topics)
                )
            )
    else:
        order = np.argsort(-topic_mass)

    # --- Keywords ------------------------------------------------------------
    log.phase(f"summary: keywords for {n_topics} topics")
    t = time.time()
    kw = keywords_for_topics(
        top_texts, point_to_cluster, top_masses, n_topics,
        samples_per_topic=80, top_k=12,
    )
    log.info(f"keywords extracted in {time.time()-t:.1f}s")

    # --- Report --------------------------------------------------------------
    lines: List[str] = []
    lines.append("# Wikipedia-link-graph curriculum (NLP) — cirrus-native\n")
    lines.append(
        "Curriculum built on the **cirrus-native Wikipedia link graph** parsed "
        "from the SAME CirrusSearch content dump (`enwiki-20251229`) as the "
        "**Wikipedia GIST embeddings** (store "
        f"`{args.wiki_store}`). One internally coherent dump — no cross-dump "
        "page_id mismatch, no bookgen. Compared against the OpenAlex-citation "
        "curriculum `artifacts/curriculum/gap_curriculum_T11273/summary.md`.\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"- **Subgraph selection:** top-{args.n_select} Wikipedia pages by cosine "
        "to the OpenAlex NLP-article centroid (streamed over all 7.1M embeddings; "
        "no dense index), unioned with their embedded cirrus out-neighbors.\n"
        f"- **Link subgraph (cirrus-native):** {adj.shape[0]} nodes, {adj.nnz} "
        f"directed article→article edges (mean out-degree {out_deg.mean():.1f}, "
        f"max {int(out_deg.max())}).\n"
        f"- **PageRank:** forward (mass→prerequisites), α={args.alpha}, seeded at "
        f"{[titles_aligned[i] for i in seed_idx]}.\n"
        + (
            f"- **Background normalization:** top-k SELECTION ranked by "
            f"`p^a · log((p+ε)/(g+ε))` (a={args.bg_alpha}, α_bg={args.alpha}, "
            f"ε=1e-12), where p = seeded forward PR and g = uniform-teleport "
            f"background PR on the same adjacency. Demotes globally-popular hubs. "
            f"Topic ordering uses the mean lift score; cluster weights stay raw "
            f"PR mass.\n"
            if args.bg_norm else ""
        )
        + f"- **Clustering:** bisecting K-Means on GIST embeddings of the top-"
        f"{top_k} pages by "
        f"{'background-normalized score' if args.bg_norm else 'PR mass'}, "
        f"cut to ~{args.desired_n_topics} topics.\n"
        + {
            "lift": "- **Ordering:** topics sorted by mean background-normalized "
                    "lift (a relevance axis — seeds first, foundations last).\n",
            "mass": "- **Ordering:** topics sorted by total PageRank mass "
                    "(foundational → frontier).\n",
            "frontier": "- **Ordering (directional):** inter-topic dependency "
                    "matrix built with `compute_frontier_adjacency` over the RAW "
                    "forward-PR flow (`adj[i,j]` = how much topic i depends on "
                    "topic j), then sorted by in-degree − out-degree "
                    "(`_compute_linearization_order_from_adjacency`) so "
                    "prerequisites come first and frontier targets come last. "
                    "Decoupled from the bg-norm SELECTION.\n",
            "frontier-bgnorm": "- **Ordering (directional, bg-normalized):** "
                    "inter-topic dependency matrix `compute_frontier_adjacency` "
                    "over the RAW forward-PR flow, but the in-flow to each topic "
                    "is background-normalized: "
                    f"`inflow_fg^a · log((inflow_fg+ε)/(inflow_bg+ε)) − "
                    f"outflow_fg` (a={args.bg_alpha}, ε=1e-12), where inflow_bg "
                    "is the same absorbing in-flow under a UNIFORM (un-seeded) "
                    "global source. Demotes globally-popular hub topics (the "
                    "topic-level analog of the node-selection bg-norm). Decoupled "
                    "from the bg-norm SELECTION.\n",
        }[order_by]
        + f"- **Keywords:** log-odds + lattice phrase mining (same as engine "
        f"`summarize_topics`), over page "
        f"{'title + opening_text' if (args.use_abstracts and any(abstracts_aligned)) else 'titles'}.\n"
    )

    # Under bg-norm, order representative pages within a topic by lift too, and
    # label the score column accordingly.
    sub_rank = top_rank if args.bg_norm else top_masses
    score_label = "lift" if args.bg_norm else "PR mass"
    dir_label = "bgnorm-in−out" if order_by == "frontier-bgnorm" else "in−out"
    lines.append("\n## Topics (prerequisite → frontier order)\n")
    if frontier_in_out is not None:
        lines.append(
            f"| Rank | Size | {score_label} | {dir_label} | Top keywords | Sample pages |"
        )
        lines.append("| :--: | ---: | ---: | ---: | :-- | :-- |")
    else:
        lines.append(f"| Rank | Size | {score_label} | Top keywords | Sample pages |")
        lines.append("| :--: | ---: | ---: | :-- | :-- |")
    for rank, t in enumerate(order):
        idxs = np.where(point_to_cluster == t)[0]
        m = float(topic_mass[t])
        size = int(idxs.size)
        sub_order = np.argsort(sub_rank[idxs])[::-1]
        sample_titles = [top_titles[idxs[j]] for j in sub_order[:6]]
        kw_str = ", ".join(w for w, _ in kw[t][:8])
        samp_str = "; ".join(s[:40] for s in sample_titles[:5])
        if frontier_in_out is not None:
            lines.append(
                f"| {rank+1} | {size} | {m:.4f} | {frontier_in_out[t]:+.4f} "
                f"| {kw_str} | {samp_str} |"
            )
        else:
            lines.append(f"| {rank+1} | {size} | {m:.4f} | {kw_str} | {samp_str} |")

    # Per-topic detail (keywords + 5 representative titles).
    lines.append("\n## Topic detail\n")
    for rank, t in enumerate(order):
        idxs = np.where(point_to_cluster == t)[0]
        m = float(topic_mass[t])
        size = int(idxs.size)
        sub_order = np.argsort(sub_rank[idxs])[::-1]
        sample_titles = [top_titles[idxs[j]] for j in sub_order[:5]]
        kw_str = ", ".join(w for w, _ in kw[t][:10])
        lines.append(f"### [{rank+1}] Topic (size={size}, {score_label}={m:.4f})\n")
        lines.append(f"**Keywords:** {kw_str}\n")
        lines.append("**Representative pages:**")
        for s in sample_titles:
            lines.append(f"- {s}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    log.info(f"wrote report to {args.out}")

    # Persist machine-readable artifacts alongside the doc for reuse.
    art_dir = Path("artifacts/curriculum/wiki_nlp_cirrus")
    art_dir.mkdir(parents=True, exist_ok=True)
    np.save(art_dir / "point_to_cluster.npy", point_to_cluster)
    np.save(art_dir / "top_masses.npy", top_masses)
    with open(art_dir / "topic_keywords.json", "w") as f:
        json.dump({int(t): kw[t] for t in range(n_topics)}, f, indent=2)
    with open(art_dir / "top_titles.json", "w") as f:
        json.dump(top_titles, f, indent=2)
    sp.save_npz(art_dir / "link_subgraph.npz", adj.tocsr())
    log.info(f"saved artifacts to {art_dir}/")
    log.close()


if __name__ == "__main__":
    main()
