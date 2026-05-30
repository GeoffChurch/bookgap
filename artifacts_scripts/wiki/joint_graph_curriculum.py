#!/usr/bin/env python
# artifacts_scripts/wiki/joint_graph_curriculum.py
"""Dictionary-bridged JOINT-graph curriculum over ``wiki-NLP ∪ OA-NLP``.

The "best-of-both" test (docs/backlog, ``best-of-both via dictionary-bridged
joint graph``): can ONE graph that fuses Wikipedia's pedagogical link backbone
with OpenAlex's coverage/frontier citation structure produce a curriculum that
keeps wiki's clean foundations→frontier ordering AND enriches the frontier with
niche/recent OA papers the wiki-only curriculum lacked?

Architecture — one scipy sparse adjacency over the union, then the validated
wiki-native pipeline on it
--------------------------------------------------------------------------------
Nodes: a tractable wiki-NLP subset (bg-norm top-k of the cached
``curriculum_subgraph_n40000.npz``) ∪ a tractable OA-NLP subset (top-n
T10181 articles by cosine to the NLP centroid, with an intra-set citation graph
built from their ``referenced_works``).

Edges (one CSR over the union, plus a node-type vector wiki=0/OA=1):
  1. wiki links (pedagogical), restricted to the wiki node set.
  2. OA citations (coverage/frontier), restricted to the OA node set.
  3. bt_cos-weighted bridge edges (below).

Bridges via bt_cos (back-translation cosine; higher coverage than mutual-NN)
--------------------------------------------------------------------------------
Per-corpus mean-center + unit-normalize both GIST sets (per slice). Cross-corpus
``sim = X_wiki @ X_oa.T`` (raw AND CSLS-corrected). Mirroring
``~/src/mycooc/diagnostics.py`` bt_cos:
  forward fwd[i]=argmax_j sim[i,j] (wiki→OA), backward bwd[j]=argmax_i sim[i,j].
  Per wiki i: round-trip k=bwd[fwd[i]]; bt_cos_i = cos(wiki_i, wiki_k). Add bridge
  wiki_i ↔ OA_fwd[i] weighted by bt_cos_i, kept if bt_cos_i > τ. Symmetric OA→wiki
  direction. Reports bridge coverage vs strict mutual-NN, swept over τ.

Pipeline on the union (reuses the wiki-native machinery)
--------------------------------------------------------------------------------
hub-suppressed PageRank seeded at the NLP core (wiki "Natural language
processing" + OA NLP seed works) → link-LPP on the JOINT affinity for the
clustering subspace (k≈64, hub-suppressed) → bisecting-KMeans cluster →
frontier-adjacency ordering (prerequisites first).

The decisive evaluation
--------------------------------------------------------------------------------
Identify OA-only frontier works (OA works with NO strong wiki bridge — low best
bt_cos). Do they land near their prerequisite wiki concepts (same cluster /
ordered after the relevant wiki foundations)? Concrete examples. Honest failure
modes (scatter if bridges too sparse; recency bias if OA citations dominate);
τ-sweep effect on frontier placement.

v2 — undoing OA mass starvation (the v1 bottleneck)
--------------------------------------------------------------------------------
v1 found only ~13% of top-k OA nodes reached a named topic vs ~43% of wiki,
because (a) the OA citation graph was restricted to edges *within the top-12k
cosine slice*, leaving ~2 edges/node, while (b) the dense wiki link backbone
hoarded the joint PageRank mass so the long OA tail was discarded before
clustering. v2 adds two structural knobs:

  * **Fix(1) ``--oa-expand`` (default on):** build the OA citation graph over the
    **1-hop citation neighborhood within the full 87k NLP slice** (add cited works
    present in the 87k as nodes), recovering OA's real citation degree
    (~2.1 → ~6.4 edges/node) at a tractable node count (~23k OA).
  * **Fix(3) ``--per-corpus-select`` (default on):** z-normalize the bg-norm
    selection score WITHIN each corpus before the top-k clustering-pool cut, so
    OA's presence in the pool tracks "high for its own corpus" rather than its
    (much smaller) absolute PageRank mass. Two further mass-cut levers,
    ``--teleport-split`` and ``--mass-rebalance``, are exposed but default OFF:
    on the bimodal joint mass distribution they destabilize the dendrogram cut
    (degenerate OA singletons / OA-takeover topics) rather than helping — see the
    v2 verdict in the results doc.

Reuses: ``wiki_curriculum_demo`` (subgraph load, cluster_embeddings,
keywords_for_topics, bg-norm selection logic), ``wiki_pedagogical_lpp``
(hub_suppressed_link_affinity, lpp_projection), ``curriculum.pagerank``
(compute_pagerank, compute_frontier_adjacency),
``curriculum.build_curriculum`` (_compute_linearization_order_from_adjacency).

Prerequisite: run ``_prep_oa_nlp_cache`` once to materialize
``/tmp/oa_nlp_slice.npz`` + ``/tmp/oa_nlp_refs.json`` (the OA-NLP slice + refs).

Usage::

    # v2 default (OA expansion + per-corpus pool selection on):
    conda run -p .conda python -m artifacts_scripts.wiki.joint_graph_curriculum \
        --n-wiki 12000 --n-oa 12000 --tau 0.5 0.6 0.7 \
        --bridge-scorer csls --lpp-k 64 --desired-n-topics 14

    # reproduce v1 exactly (no expansion, no rebalancing):
    conda run -p .conda python -m artifacts_scripts.wiki.joint_graph_curriculum \
        --no-oa-expand --mass-rebalance none --teleport-split -1 \
        --no-per-corpus-select --bridge-scorer csls
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray

from artifacts_scripts.curriculum.build_curriculum import (
    _compute_linearization_order_from_adjacency,
)
from artifacts_scripts.curriculum.pagerank import (
    compute_frontier_adjacency,
    compute_pagerank,
)
from artifacts_scripts.wiki.wiki_curriculum_demo import (
    SEED_TITLES,
    cluster_embeddings,
    keywords_for_topics,
    load_subgraph,
)
from artifacts_scripts.wiki.wiki_pedagogical_lpp import (
    PhaseLog,
    hub_suppressed_link_affinity,
    lpp_projection,
)

RUN_LOG = Path("artifacts/wiki_joint_curriculum/run.log")
ART_DIR = Path("artifacts/curriculum/wiki_oa_joint")
OA_NPZ = Path("/tmp/oa_nlp_slice.npz")
OA_REFS = Path("/tmp/oa_nlp_refs.json")

ALPHA = 0.85
BG_ALPHA = 0.5
BG_EPS = 1e-12

# OA seed works: a handful of canonical NLP entry-point titles to pin the
# PageRank core on the OA side (mirrors SEED_TITLES on the wiki side). Matched by
# case-insensitive substring against OA titles; whatever resolves is used.
OA_SEED_SUBSTRINGS = [
    "natural language processing",
    "neural probabilistic language model",
    "word embedding",
    "attention is all you need",
    "bert",
]


def _normalize_np(X: NDArray) -> NDArray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def _mean_match(X: NDArray) -> NDArray:
    """Per-slice mean-match: subtract own mean, then unit-normalize (the
    validated within-topic cross-corpus alignment baseline)."""
    return _normalize_np(X - X.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------------------
# Node sets
# ---------------------------------------------------------------------------

def select_wiki_nodes(
    adj: sp.csr_matrix, titles: List[str], emb: NDArray, n_wiki: int, log: PhaseLog,
) -> Tuple[NDArray[np.int64], sp.csr_matrix, List[str], NDArray]:
    """bg-norm top-n_wiki of the cached wiki subgraph (seeded forward PR `p` /
    uniform-teleport background PR `g`, select by p^0.5·log(p/g)), with wiki
    links restricted to that node set. Returns
    ``(orig_idx, sub_adj, sub_titles, sub_emb)``."""
    n = adj.shape[0]
    title_to_local = {t: i for i, t in enumerate(titles)}
    seed_idx = [title_to_local[t] for t in SEED_TITLES if t in title_to_local]
    if not seed_idx:
        raise SystemExit("No wiki seed pages found in subgraph.")
    log(f"  wiki seeds: {[titles[i] for i in seed_idx]}")
    seeds = np.zeros(n, dtype=np.float64)
    for i in seed_idx:
        seeds[i] = 1.0 / len(seed_idx)
    masses = compute_pagerank(adj, seeds, alpha=ALPHA, tol=1e-9, max_iter=200, normalize=True)
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    g_bg = compute_pagerank(adj, uniform, alpha=ALPHA, tol=1e-9, max_iter=200, normalize=True)
    p_fg = np.maximum(masses, 0.0)
    score = np.power(p_fg, BG_ALPHA) * np.log((p_fg + BG_EPS) / (g_bg + BG_EPS))
    k = min(n_wiki, n)
    sel = np.argsort(score)[-k:][::-1].astype(np.int64)
    sel_sorted = np.sort(sel)  # keep stable original ordering for the block
    sub_adj = adj[sel_sorted][:, sel_sorted].tocsr()
    sub_titles = [titles[i] for i in sel_sorted]
    sub_emb = emb[sel_sorted]
    log(f"  wiki node set: {k} nodes, {sub_adj.nnz} intra-set link edges "
        f"(retains {100*masses[sel_sorted].sum():.1f}% seeded PR mass)")
    return sel_sorted, sub_adj, sub_titles, sub_emb


def select_oa_nodes(
    emb: NDArray, titles: List[str], ids: List[str], refs: List[List[str]],
    n_oa: int, log: PhaseLog, expand: bool = True,
) -> Tuple[NDArray[np.int64], sp.csr_matrix, List[str], List[str], NDArray, Dict[str, float]]:
    """Build the OA node set + intra-set citation graph (A[i,j]=1 iff i cites j,
    + self loops) from ``referenced_works``. Returns ``(orig_idx, oa_adj,
    sub_titles, sub_ids, sub_emb, density_stats)``. Centroid + cosine selection on
    raw L2-normed GIST (matches the cross-corpus slice convention).

    **v2 Fix(1) — OA citation-neighborhood expansion (``expand=True``).** v1
    restricted OA citation edges to *within the top-n_oa cosine slice*, but most
    NLP-paper citations point OUTSIDE it (only ~2 edges/node survived). We instead
    take the top-n_oa cosine *seeds*, then expand the node set to the **1-hop
    citation neighborhood within the FULL 87k NLP slice**: any work cited by a seed
    that is itself present in the 87k is added as a node, and the citation graph is
    built over the union. This recovers OA's real citation degree (~6 edges/node)
    instead of the v1 artificial sparsity, while keeping the node count tractable
    (tens of thousands). With ``expand=False`` the v1 within-slice behavior is
    reproduced exactly (for the before/after comparison).

    ``density_stats`` reports the v1-equivalent within-seed-set edge count and the
    expanded edge count so the report can quote OA edges/node before vs after."""
    En = _normalize_np(emb)
    centroid = _normalize_np(En.mean(axis=0, keepdims=True))[0]
    cos = En @ centroid
    k_seed = min(n_oa, len(ids))
    seed = np.argsort(-cos)[:k_seed].astype(np.int64)
    seed_set = set(int(i) for i in seed)
    id_to_global = {wid: i for i, wid in enumerate(ids)}

    # v1-equivalent density: citation edges among the top-n_oa cosine seeds only.
    seed_ids = set(ids[i] for i in seed)
    v1_edges = sum(
        1 for i in seed for ref in (refs[i] or ()) if ref in seed_ids
    )

    if expand:
        # 1-hop expansion: add every seed-cited work that is itself in the 87k slice.
        node_globals = set(seed_set)
        for i in seed:
            for ref in (refs[i] or ()):
                g = id_to_global.get(ref)
                if g is not None:
                    node_globals.add(g)
        sel_sorted = np.array(sorted(node_globals), dtype=np.int64)
    else:
        sel_sorted = np.sort(seed)

    sub_ids = [ids[i] for i in sel_sorted]
    sub_titles = [titles[i] for i in sel_sorted]
    sub_emb = emb[sel_sorted]
    k = len(sel_sorted)

    # Build the citation graph over ALL nodes in the (possibly expanded) set.
    # An edge i→j is kept whenever both endpoints are in the node set, regardless
    # of whether i was a cosine seed or an expansion node — this is what gives the
    # expansion nodes real degree rather than only being citation targets.
    id_to_local = {wid: i for i, wid in enumerate(sub_ids)}
    rows: List[int] = []
    cols: List[int] = []
    for li, orig in enumerate(sel_sorted):
        rows.append(li); cols.append(li)  # self loop (matches cloud citation graph)
        for ref in (refs[orig] or ()):
            tgt = id_to_local.get(ref)
            if tgt is not None:
                rows.append(li); cols.append(tgt)
    oa_adj = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(k, k)
    )
    intra_edges = oa_adj.nnz - k  # minus self loops
    density_stats = {
        "n_seed": int(k_seed),
        "n_nodes": int(k),
        "n_expansion_nodes": int(k - k_seed),
        "v1_edges": int(v1_edges),
        "v1_edges_per_node": float(v1_edges) / max(k_seed, 1),
        "edges": int(intra_edges),
        "edges_per_node": float(intra_edges) / max(k, 1),
    }
    log(f"  OA node set: {k} nodes "
        f"({k_seed} cosine seeds + {k - k_seed} 1-hop cited)"
        if expand else f"  OA node set: {k} nodes (within-slice, no expansion)")
    log(f"    citation edges: v1-within-seed={v1_edges} "
        f"({density_stats['v1_edges_per_node']:.2f}/node) → "
        f"expanded={intra_edges} ({density_stats['edges_per_node']:.2f}/node); "
        f"cos→centroid {cos[seed].min():.3f}..{cos[seed].max():.3f}")
    return sel_sorted, oa_adj, sub_titles, sub_ids, sub_emb, density_stats


# ---------------------------------------------------------------------------
# bt_cos bridges
# ---------------------------------------------------------------------------

def csls(sim: torch.Tensor, k: int) -> torch.Tensor:
    mean0 = sim.topk(k, dim=0).values.mean(dim=0, keepdim=True)
    mean1 = sim.topk(k, dim=1).values.mean(dim=1, keepdim=True)
    return sim - 0.5 * (mean0 + mean1)


def bt_cos_bridges(
    wiki_emb: NDArray, oa_emb: NDArray, use_csls: bool, csls_k: int, log: PhaseLog,
) -> Dict[str, NDArray]:
    """Compute bt_cos in both directions on the per-slice-mean-matched cosine.

    Returns a dict with, for each direction, the best-match index, the matched
    similarity (cosine confidence of the cross-corpus match), and the bt_cos
    round-trip value. ``Xw``/``Xo`` are the mean-matched unit vectors so bt_cos
    is a genuine cosine in the wiki (resp. OA) slice.

    Keys:
      wiki_fwd  (n_wiki,)  -> OA index argmax_j sim[i,j]
      wiki_btcos(n_wiki,)  -> cos(wiki_i, wiki_{bwd[fwd[i]]})
      wiki_conf (n_wiki,)  -> sim[i, fwd[i]]
      oa_fwd, oa_btcos, oa_conf  -- symmetric (OA→wiki)
      mutual_wiki (bool n_wiki) -- strict mutual-NN flag (wiki side)
    """
    Xw = torch.tensor(_mean_match(wiki_emb))
    Xo = torch.tensor(_mean_match(oa_emb))
    sim = Xw @ Xo.T  # [n_wiki, n_oa], cosine in the shared identity space
    if use_csls:
        sim = csls(sim, csls_k)

    # wiki→OA→wiki round trip
    fwd = sim.argmax(dim=1)            # [n_wiki] -> oa
    bwd = sim.argmax(dim=0)            # [n_oa]   -> wiki
    rt = bwd[fwd]                      # [n_wiki] round-trip wiki index
    # bt_cos = cosine between the mean-matched wiki vectors (raw cosine, NOT CSLS).
    wiki_btcos = (Xw * Xw[rt]).sum(dim=1).numpy()
    wiki_conf = sim[torch.arange(sim.shape[0]), fwd].numpy()
    mutual_wiki = (rt == torch.arange(sim.shape[0])).numpy()

    # OA→wiki→OA round trip (symmetric)
    oa_fwd = bwd                       # [n_oa] -> wiki (already argmax over rows)
    oa_rt = fwd[oa_fwd]                # [n_oa] round-trip oa index
    oa_btcos = (Xo * Xo[oa_rt]).sum(dim=1).numpy()
    oa_conf = sim.T[torch.arange(sim.shape[1]), oa_fwd].numpy()

    log(f"  bridges ({'CSLS' if use_csls else 'raw'}): "
        f"wiki bt_cos median={np.median(wiki_btcos):.3f}, "
        f"OA bt_cos median={np.median(oa_btcos):.3f}, "
        f"strict mutual-NN rate (wiki)={mutual_wiki.mean():.4f}")
    return {
        "wiki_fwd": fwd.numpy(), "wiki_btcos": wiki_btcos, "wiki_conf": wiki_conf,
        "oa_fwd": oa_fwd.numpy(), "oa_btcos": oa_btcos, "oa_conf": oa_conf,
        "mutual_wiki": mutual_wiki,
    }


def build_bridge_edges(
    br: Dict[str, NDArray], n_wiki: int, tau: float,
) -> Tuple[List[int], List[int], List[float], float, float]:
    """Bridge edges (wiki-local i ↔ OA-local j, OA index offset by n_wiki) kept
    if bt_cos > tau, both directions, deduped. Returns
    ``(rows, cols, weights, wiki_coverage, oa_coverage)`` where coverage =
    fraction of that side's nodes with at least one surviving bridge.
    """
    pairs: Dict[Tuple[int, int], float] = {}
    # wiki→OA
    for i in range(n_wiki):
        if br["wiki_btcos"][i] > tau:
            j = int(br["wiki_fwd"][i])
            pairs[(i, j)] = max(pairs.get((i, j), 0.0), float(br["wiki_btcos"][i]))
    # OA→wiki
    n_oa = len(br["oa_btcos"])
    for j in range(n_oa):
        if br["oa_btcos"][j] > tau:
            i = int(br["oa_fwd"][j])
            pairs[(i, j)] = max(pairs.get((i, j), 0.0), float(br["oa_btcos"][j]))
    rows, cols, wts = [], [], []
    wiki_has = np.zeros(n_wiki, dtype=bool)
    oa_has = np.zeros(n_oa, dtype=bool)
    for (i, j), w in pairs.items():
        rows.append(i); cols.append(n_wiki + j); wts.append(w)
        wiki_has[i] = True; oa_has[j] = True
    return rows, cols, wts, float(wiki_has.mean()), float(oa_has.mean())


# ---------------------------------------------------------------------------
# Joint graph
# ---------------------------------------------------------------------------

def build_joint_graph(
    wiki_adj: sp.csr_matrix, oa_adj: sp.csr_matrix,
    bridge_rows: List[int], bridge_cols: List[int], bridge_wts: List[float],
) -> Tuple[sp.csr_matrix, NDArray[np.int64]]:
    """One DIRECTED CSR over the union [wiki block | OA block], plus a node-type
    vector (0=wiki, 1=OA). wiki links and OA citations keep their direction;
    bridges are added symmetrically (both i→j and j→i) so cross-corpus PageRank
    mass can flow either way. Bridge weight = bt_cos (wiki/OA intra-edges are
    unit-weighted, matching both source graphs)."""
    n_w = wiki_adj.shape[0]
    n_o = oa_adj.shape[0]
    n = n_w + n_o
    node_type = np.zeros(n, dtype=np.int64)
    node_type[n_w:] = 1

    wiki_coo = wiki_adj.tocoo()
    oa_coo = oa_adj.tocoo()
    rows = list(wiki_coo.row) + [n_w + r for r in oa_coo.row]
    cols = list(wiki_coo.col) + [n_w + c for c in oa_coo.col]
    data = list(wiki_coo.data.astype(np.float64)) + list(oa_coo.data.astype(np.float64))
    # symmetric bridges
    rows += bridge_rows + bridge_cols
    cols += bridge_cols + bridge_rows
    data += [float(w) for w in bridge_wts] * 2

    joint = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    joint.sum_duplicates()
    return joint, node_type


# ---------------------------------------------------------------------------
# Joint pipeline
# ---------------------------------------------------------------------------

def joint_seeded_pagerank(
    joint: sp.csr_matrix, wiki_titles: List[str], oa_titles: List[str], n_w: int,
    node_type: NDArray[np.int64], log: PhaseLog, teleport_split: float = -1.0,
) -> Tuple[NDArray, NDArray, List[int]]:
    """Hub-suppressed forward PR seeded at the NLP core (wiki SEED_TITLES + OA
    seed substrings), plus a background PR. Returns ``(masses, g_bg,
    seed_local_idx)``.

    **v2 Fix(3) — balanced teleport.** PageRank stationary mass tracks in-degree,
    so the dense wiki link backbone hoards mass even after row-normalization. The
    ``teleport_split`` knob rebalances BOTH the seed personalization vector AND the
    background-PR teleport vector across the two corpora:

      * ``teleport_split < 0`` (default ``-1.0``): v1 behavior — seed mass split
        uniformly across whatever named seeds resolve (so wiki's 6 seeds vs OA's
        ~3 already tilt toward wiki) and uniform background teleport over ALL
        nodes (which, with far denser wiki edges, pools on wiki).
      * ``0 ≤ teleport_split ≤ 1``: ``teleport_split`` of the teleport mass goes to
        the WIKI side, ``1-teleport_split`` to the OA side. Within each side mass
        is spread over that side's seeds (foreground) / uniformly over that side's
        nodes (background). ``0.5`` = even 50/50 split regardless of node count.

    The background PR ``g`` is rebalanced the SAME way so the ``p/g`` bg-norm ratio
    stays a within-graph contrast rather than silently re-tilting the cut."""
    n = joint.shape[0]
    title_to_w = {t: i for i, t in enumerate(wiki_titles)}
    wiki_seed_idx = [title_to_w[t] for t in SEED_TITLES if t in title_to_w]
    oa_lower = [t.lower() for t in oa_titles]
    oa_seeds: List[int] = []
    for sub in OA_SEED_SUBSTRINGS:
        for li, t in enumerate(oa_lower):
            if sub in t:
                oa_seeds.append(n_w + li)
                break
    seed_idx = wiki_seed_idx + oa_seeds

    seeds = np.zeros(n, dtype=np.float64)
    uniform = np.zeros(n, dtype=np.float64)
    is_wiki = node_type == 0
    is_oa = node_type == 1
    if teleport_split < 0:
        # v1: uniform over named seeds; uniform background over all nodes.
        for i in seed_idx:
            seeds[i] = 1.0 / len(seed_idx)
        uniform[:] = 1.0 / n
        log(f"  joint seeds: {len(seed_idx)} ({len(wiki_seed_idx)} wiki + "
            f"{len(oa_seeds)} OA); teleport=uniform-over-all (v1)")
    else:
        ws, os = teleport_split, 1.0 - teleport_split
        # Foreground: split seed mass ws:os across corpora, spread within side.
        if wiki_seed_idx:
            for i in wiki_seed_idx:
                seeds[i] = ws / len(wiki_seed_idx)
        if oa_seeds:
            for i in oa_seeds:
                seeds[i] = os / len(oa_seeds)
        # Background: ws of teleport over all wiki nodes, os over all OA nodes.
        nw, no = int(is_wiki.sum()), int(is_oa.sum())
        if nw:
            uniform[is_wiki] = ws / nw
        if no:
            uniform[is_oa] = os / no
        log(f"  joint seeds: {len(seed_idx)} ({len(wiki_seed_idx)} wiki + "
            f"{len(oa_seeds)} OA); balanced teleport split wiki:OA={ws:.2f}:{os:.2f}")
    masses = compute_pagerank(joint, seeds, alpha=ALPHA, tol=1e-9, max_iter=300, normalize=True)
    g_bg = compute_pagerank(joint, uniform, alpha=ALPHA, tol=1e-9, max_iter=300, normalize=True)
    return masses, g_bg, seed_idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/enwiki_full"))
    ap.add_argument("--n-select", type=int, default=40000,
                    help="wiki subgraph cache suffix (curriculum_subgraph_n{N}.npz)")
    ap.add_argument("--n-wiki", type=int, default=12000, help="wiki nodes (bg-norm top-k)")
    ap.add_argument("--n-oa", type=int, default=12000, help="OA nodes (top-cosine to NLP centroid)")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.6, 0.7],
                    help="bt_cos bridge thresholds to sweep (graph built at the FIRST)")
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--bridge-scorer", choices=["raw", "csls"], default="raw",
                    help="cosine for the bt_cos bridges (raw or CSLS-corrected)")
    ap.add_argument("--lpp-k", type=int, default=64, help="link-LPP subspace dim")
    ap.add_argument("--ranking-top-k", type=int, default=5000,
                    help="top-k joint nodes by bg-norm score to cluster")
    ap.add_argument("--desired-n-topics", type=int, default=14)
    # --- v2 rebalancing knobs (Fix 1 + Fix 3) ---------------------------------
    ap.add_argument("--oa-expand", dest="oa_expand", action="store_true", default=True,
                    help="Fix(1): expand OA nodes to the 1-hop citation neighborhood "
                         "within the full 87k slice (denser OA citation graph). On by default.")
    ap.add_argument("--no-oa-expand", dest="oa_expand", action="store_false",
                    help="reproduce v1: OA citation edges within the top-n_oa cosine slice only.")
    ap.add_argument("--mass-rebalance", choices=["none", "count"], default="none",
                    help="Fix(3) frontier-cut lever: per-corpus mass rebalancing for "
                         "the dendrogram mass-cut. 'count' rescales each corpus's "
                         "masses so the two corpora contribute mass proportional to "
                         "their node count in the clustered set. NOTE: on the bimodal "
                         "joint mass distribution this destabilizes the dendrogram cut "
                         "(spawns degenerate OA singletons) rather than helping — the "
                         "reliable OA-into-curriculum lever is --per-corpus-select at "
                         "the POOL stage. Default 'none' (off).")
    ap.add_argument("--teleport-split", type=float, default=-1.0,
                    help="Fix(3) secondary lever: fraction of PageRank teleport/seed "
                         "mass given to the WIKI side (rest to OA); applied to BOTH "
                         "the seeded foreground PR and the background PR. <0 = v1 "
                         "(uniform-over-all teleport). NOTE: tilting teleport is an "
                         "unstable lever (the dense wiki backbone re-concentrates the "
                         "mass regardless); --mass-rebalance is the reliable fix. "
                         "Default -1 (off).")
    ap.add_argument("--per-corpus-select", action="store_true", default=True,
                    help="Fix(3): z-normalize the bg-norm selection score WITHIN each "
                         "corpus before the top-k cut, so OA representation in the "
                         "clustered POOL doesn't depend on its absolute PR mass. On by default.")
    ap.add_argument("--no-per-corpus-select", dest="per_corpus_select", action="store_false",
                    help="reproduce v1: single global top-k cut on the raw bg-norm score.")
    ap.add_argument("--out", type=Path, default=ART_DIR / "report.md",
                    help="machine report path (default: the artifacts dir). The "
                         "curated results doc docs/wikipedia-openalex-joint-curriculum.md "
                         "is hand-authored from this report's numbers and is NOT "
                         "overwritten by re-runs.")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    log = PhaseLog(RUN_LOG)
    log("Dictionary-bridged JOINT-graph curriculum (wiki-NLP ∪ OA-NLP)")

    # --- Load both corpora ---------------------------------------------------
    log("load wiki subgraph (cached)")
    cache = args.wiki_store / f"curriculum_subgraph_n{args.n_select}.npz"
    w_ids, w_adj_full, w_titles_full, _w_abs, w_emb_full = load_subgraph(cache)
    w_adj_full = w_adj_full.tocsr()
    log(f"  wiki subgraph: {len(w_ids)} nodes, {w_adj_full.nnz} edges")

    log("load OA-NLP slice (cached npz/json)")
    if not OA_NPZ.exists():
        raise SystemExit(f"missing {OA_NPZ}; run _prep_oa_nlp_cache first.")
    z = np.load(OA_NPZ, allow_pickle=True)
    oa_emb_full = np.ascontiguousarray(z["emb"], dtype=np.float32)
    oa_titles_full = [str(s) for s in z["titles"]]
    oa_ids_full = [str(s) for s in z["ids"]]
    oa_refs_full = json.loads(OA_REFS.read_text())
    log(f"  OA-NLP slice: {oa_emb_full.shape[0]} works, fp={str(z['fp'])}")

    # --- Node sets -----------------------------------------------------------
    log("select wiki nodes (bg-norm top-k)")
    w_sel, wiki_adj, wiki_titles, wiki_emb = select_wiki_nodes(
        w_adj_full, w_titles_full, w_emb_full, args.n_wiki, log
    )
    log("select OA nodes (top-cosine + intra-set citation graph)")
    oa_sel, oa_adj, oa_titles, oa_ids, oa_emb, oa_density = select_oa_nodes(
        oa_emb_full, oa_titles_full, oa_ids_full, oa_refs_full, args.n_oa, log,
        expand=args.oa_expand,
    )
    n_w, n_o = len(wiki_titles), len(oa_titles)

    # --- bt_cos bridges + coverage sweep -------------------------------------
    log(f"bt_cos bridges ({args.bridge_scorer})")
    br_raw = bt_cos_bridges(wiki_emb, oa_emb, use_csls=False, csls_k=args.csls_k, log=log)
    br_csls = bt_cos_bridges(wiki_emb, oa_emb, use_csls=True, csls_k=args.csls_k, log=log)
    br = br_raw if args.bridge_scorer == "raw" else br_csls
    strict_mnn = float(br["mutual_wiki"].mean())

    coverage_table: List[Dict] = []
    for scorer_name, b in (("raw", br_raw), ("csls", br_csls)):
        for tau in args.tau:
            rows, cols, wts, wcov, ocov = build_bridge_edges(b, n_w, tau)
            coverage_table.append({
                "scorer": scorer_name, "tau": tau, "n_bridges": len(rows),
                "wiki_cov": wcov, "oa_cov": ocov,
            })
            log(f"  [{scorer_name} τ={tau}] bridges={len(rows)} "
                f"wiki_cov={wcov:.3f} oa_cov={ocov:.3f}")
    log(f"  strict mutual-NN coverage (wiki, no τ) = {strict_mnn:.4f}")

    # Build the JOINT graph at the FIRST (lowest) τ for the curriculum run; the
    # τ-sweep above quantifies bridge density, and a τ-sweep on frontier
    # placement is reported in the eval.
    tau0 = args.tau[0]
    rows, cols, wts, wcov0, ocov0 = build_bridge_edges(br, n_w, tau0)
    log(f"build joint graph at τ={tau0} ({len(rows)} bridges)")
    joint, node_type = build_joint_graph(wiki_adj, oa_adj, rows, cols, wts)
    log(f"  joint graph: {joint.shape[0]} nodes "
        f"({n_w} wiki + {n_o} OA), {joint.nnz} directed edges")

    # --- Joint embeddings (raw GIST, shared 384-d space) ---------------------
    joint_emb = np.vstack([wiki_emb, oa_emb]).astype(np.float32)
    joint_titles = list(wiki_titles) + list(oa_titles)

    # --- Hub-suppressed PageRank seeded at the NLP core ----------------------
    log("joint seeded PageRank (hub-suppressed via bg-norm) + background PR")
    masses, g_bg, seed_idx = joint_seeded_pagerank(
        joint, wiki_titles, oa_titles, n_w, node_type, log,
        teleport_split=args.teleport_split,
    )
    p_fg = np.maximum(masses, 0.0)
    sel_score = np.power(p_fg, BG_ALPHA) * np.log((p_fg + BG_EPS) / (g_bg + BG_EPS))

    # v2 Fix(3) — per-corpus selection normalization. The raw bg-norm score is an
    # absolute-mass quantity, so a single global top-k cut keeps mostly wiki (its
    # dense backbone carries higher mass). z-normalizing the score WITHIN each
    # corpus before the cut makes "high for its corpus" the selection criterion, so
    # OA's representation in the clustered set no longer depends on its absolute
    # mass — the rank cut admits each corpus's own top tail. This is the mass-cut
    # analog of "take top-k per corpus", but kept as a single ranking so the
    # downstream frontier mass-cut and clustering are untouched.
    if args.per_corpus_select:
        cut_score = sel_score.copy()
        for mask in (node_type == 0, node_type == 1):
            s = sel_score[mask]
            mu, sd = float(s.mean()), float(s.std())
            cut_score[mask] = (s - mu) / (sd if sd > 0 else 1.0)
        log("  per-corpus z-normalized selection score (Fix 3) for the top-k cut")
    else:
        cut_score = sel_score

    # --- link-LPP clustering subspace on the JOINT affinity ------------------
    log(f"link-LPP subspace on joint affinity (hub-suppressed, k={args.lpp_k})")
    W_link = hub_suppressed_link_affinity(joint)
    Xc = joint_emb.astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    P_link, eig_link = lpp_projection(joint_emb.astype(np.float64), W_link, args.lpp_k, log)
    Y = Xc @ P_link  # n × k, joint pedagogical subspace
    log(f"  LPP smallest λ: {np.round(eig_link[:6], 5).tolist()}")

    # --- Top-k by bg-norm score, cluster in LPP space ------------------------
    top_k = min(args.ranking_top_k, joint.shape[0])
    top_idx = np.argsort(cut_score)[-top_k:][::-1]
    log(f"cluster top-{top_k} joint nodes in LPP subspace "
        f"({100*node_type[top_idx].mean():.1f}% OA)")
    top_emb_lpp = np.ascontiguousarray(Y[top_idx], dtype=np.float32)
    top_masses = masses[top_idx]
    top_is_oa_pre = node_type[top_idx] == 1

    # v2 Fix(3) — per-corpus mass rebalancing for the dendrogram mass-cut.
    # `cluster_embeddings` keeps only high-mass dendrogram subtrees (everything
    # else → cluster -1). PageRank mass tracks in-degree, so even with a 50/50
    # teleport split the median wiki node carries ~120× the mass of the median OA
    # node (the dense wiki link backbone amplifies the teleport mass; the sparse OA
    # citation graph barely propagates it). The dendrogram cut therefore demotes
    # almost all OA below the topic-formation threshold REGARDLESS of teleport.
    # We undo this at the cut itself: rescale each corpus's masses so the two
    # corpora contribute mass proportional to their NODE COUNT in the clustered
    # set (`mass_rebalance="count"`) — i.e. "important within your own corpus" is
    # the cut criterion, not "high absolute mass". `"none"` keeps the raw masses
    # (v1). The rescaled masses are used ONLY for the dendrogram cut; the true
    # PageRank masses still drive frontier ordering, keyword salience, and the
    # per-topic sample ordering, so the wiki backbone's foundations→frontier
    # structure is unchanged — we only stop starving OA at the topic-formation gate.
    cut_masses = top_masses.copy()
    if args.mass_rebalance == "count":
        for mask in (~top_is_oa_pre, top_is_oa_pre):
            tot = float(top_masses[mask].sum())
            if tot > 0:
                # target mass for this corpus ∝ its share of clustered nodes
                target = float(mask.sum()) / len(top_idx)
                cut_masses[mask] = top_masses[mask] * (target / tot)
        log(f"  per-corpus mass rebalance='count' for the dendrogram cut "
            f"(equalizes each corpus's mass budget to its node share)")
    point_to_cluster = cluster_embeddings(top_emb_lpp, cut_masses, args.desired_n_topics)
    n_topics = int(point_to_cluster.max()) + 1
    log(f"  {n_topics} topics")

    # --- Frontier-adjacency ordering (prerequisites first) -------------------
    log("frontier-adjacency ordering")
    leaf_to_cluster_full = np.full(joint.shape[0], -1, dtype=np.int64)
    leaf_to_cluster_full[top_idx] = point_to_cluster.astype(np.int64)
    frontier_nodes = np.arange(n_topics, dtype=np.int64)
    adj_frontier = compute_frontier_adjacency(
        joint, frontier_nodes, leaf_to_cluster_full, alpha=ALPHA, tol=1e-6, max_iter=200,
    )
    inflow = np.array([adj_frontier[:, k].sum() for k in range(n_topics)])
    outflow = np.array([adj_frontier[k, :].sum() for k in range(n_topics)])
    frontier_in_out = inflow - outflow
    order = _compute_linearization_order_from_adjacency(adj_frontier)

    # --- Keywords ------------------------------------------------------------
    log("keywords")
    top_texts = [joint_titles[i] for i in top_idx]
    kw = keywords_for_topics(top_texts, point_to_cluster, top_masses, n_topics)

    # --- Per-topic OA share + composition ------------------------------------
    top_is_oa = node_type[top_idx] == 1
    topic_oa_frac = np.array(
        [top_is_oa[point_to_cluster == t].mean() if (point_to_cluster == t).any() else 0.0
         for t in range(n_topics)]
    )
    # Assignment diagnostic: the frontier mass-cut leaves low-mass nodes as
    # cluster -1 (unassigned intermediaries). How many of the top-k OA nodes
    # actually reach a NAMED frontier topic vs sink into the -1 mass? This is the
    # binding "does the OA frontier carry into the curriculum" number.
    oa_assigned = int(((point_to_cluster >= 0) & top_is_oa).sum())
    oa_total_topk = int(top_is_oa.sum())
    wiki_assigned = int(((point_to_cluster >= 0) & ~top_is_oa).sum())
    wiki_total_topk = int((~top_is_oa).sum())
    # Cleaner sub-metrics: "reach a named topic" alone counts degenerate
    # SINGLETON topics (a single OA node that the dendrogram mass-cut promoted to
    # its own "topic") as a success, which inflates the number without any
    # pedagogy. We additionally report OA reaching a NON-SINGLETON topic, and OA
    # reaching a WIKI-ANCHORED (mixed: ≥3 wiki members) topic — the latter is the
    # genuine best-of-both count (OA grafted onto a wiki-foundation-bearing topic).
    topic_sizes = np.array([(point_to_cluster == t).sum() for t in range(n_topics)])
    topic_n_wiki = np.array(
        [int((~top_is_oa)[point_to_cluster == t].sum()) for t in range(n_topics)]
    )
    nonsingleton = topic_sizes >= 2
    wiki_anchored = topic_n_wiki >= 3
    oa_in_nonsingleton = int(sum(
        ((point_to_cluster == t) & top_is_oa).sum()
        for t in range(n_topics) if nonsingleton[t]
    ))
    oa_in_wiki_anchored = int(sum(
        ((point_to_cluster == t) & top_is_oa).sum()
        for t in range(n_topics) if wiki_anchored[t]
    ))
    n_singletons = int((topic_sizes == 1).sum())
    assign_stats = {
        "oa_assigned": oa_assigned, "oa_total_topk": oa_total_topk,
        "wiki_assigned": wiki_assigned, "wiki_total_topk": wiki_total_topk,
        "top_k": len(top_idx),
        "oa_in_nonsingleton": oa_in_nonsingleton,
        "oa_in_wiki_anchored": oa_in_wiki_anchored,
        "n_singletons": n_singletons,
    }
    log(f"  assignment: OA {oa_assigned}/{oa_total_topk} reach a topic "
        f"({100*oa_assigned/max(oa_total_topk,1):.1f}%); "
        f"wiki {wiki_assigned}/{wiki_total_topk} "
        f"({100*wiki_assigned/max(wiki_total_topk,1):.1f}%)")
    log(f"  of those OA: {oa_in_nonsingleton} in non-singleton topics, "
        f"{oa_in_wiki_anchored} in wiki-anchored (≥3 wiki) topics; "
        f"{n_singletons} singleton topics total")

    # =====================================================================
    # DECISIVE EVALUATION: OA-only frontier-work placement
    # =====================================================================
    log("DECISIVE EVAL: OA-only frontier-work placement")
    # OA-only = OA node whose BEST cross-corpus match (raw bt_cos as the
    # round-trip-stable measure) is weak — i.e. no strong wiki counterpart.
    oa_best_btcos = br_raw["oa_btcos"]          # (n_o,)
    oa_best_conf = br_raw["oa_conf"]            # cross-corpus match cosine
    # Among the clustered OA nodes, find those with low bt_cos (niche/recent).
    clustered_oa_local = np.array(
        [i - n_w for i in top_idx if node_type[i] == 1], dtype=np.int64
    )
    clustered_oa_topidx = np.array(
        [pos for pos, i in enumerate(top_idx) if node_type[i] == 1], dtype=np.int64
    )
    # Build, for each clustered OA node, (oa_local, topic, btcos, conf, title).
    # Points the dendrogram mass-cut left unassigned carry cluster -1; exclude
    # them from the placement eval (they belong to no frontier topic).
    oa_records: List[Dict[str, Any]] = []
    for pos, oal in zip(clustered_oa_topidx, clustered_oa_local):
        t = int(point_to_cluster[pos])
        if t < 0:
            continue
        oa_records.append({
            "oa_local": int(oal),
            "topic": t,
            "btcos": float(oa_best_btcos[oal]),
            "conf": float(oa_best_conf[oal]),
            "wiki_match": wiki_titles[int(br_raw["oa_fwd"][oal])],
            "title": oa_titles[int(oal)],
        })
    oa_records.sort(key=lambda r: float(r["btcos"]))  # weakest bridge first = OA-only frontier
    n_oa_clustered = len(oa_records)
    btcos_q25 = float(np.quantile([float(rr["btcos"]) for rr in oa_records], 0.25)) \
        if oa_records else 0.0
    oa_only = [r for r in oa_records if float(r["btcos"]) <= btcos_q25]
    log(f"  {n_oa_clustered} OA nodes in the clustered set; "
        f"{len(oa_only)} are OA-only (bottom-quartile bt_cos)")

    # For a sample of OA-only works, list the topic's wiki foundations (highest
    # bg-norm wiki members of that topic) to check prerequisite adjacency.
    def topic_wiki_foundations(t: int, k: int = 6) -> List[str]:
        members = np.where(point_to_cluster == t)[0]
        wiki_members = [(top_masses[m], joint_titles[top_idx[m]])
                        for m in members if node_type[top_idx[m]] == 0]
        wiki_members.sort(key=lambda x: -x[0])
        return [t for _, t in wiki_members[:k]]

    # rank position of each topic in the prerequisite→frontier order. Any topic
    # missing from `order` (shouldn't happen, but guard) is ranked last.
    topic_rank = {int(t): r for r, t in enumerate(order)}
    for t in range(n_topics):
        topic_rank.setdefault(int(t), n_topics)

    eval_examples: List[Dict[str, Any]] = []
    for r in oa_only[:20]:
        t = int(r["topic"])
        eval_examples.append({
            **r,
            "topic_rank": topic_rank[t] + 1,
            "topic_oa_frac": float(topic_oa_frac[t]),
            "wiki_foundations": topic_wiki_foundations(t),
            "topic_keywords": [w for w, _ in kw[t][:8]],
        })

    # --- Report --------------------------------------------------------------
    log("write report")
    write_report(
        args, n_w, n_o, joint, wiki_adj, oa_adj, len(rows), tau0,
        coverage_table, strict_mnn, eig_link, n_topics, order, point_to_cluster,
        top_idx, top_masses, node_type, joint_titles, kw, frontier_in_out,
        topic_oa_frac, topic_rank, oa_records, eval_examples, n_oa_clustered,
        topic_wiki_foundations, assign_stats, oa_density,
    )

    # --- Persist artifacts ---------------------------------------------------
    ART_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(ART_DIR / "joint_graph.npz", joint)
    np.save(ART_DIR / "node_type.npy", node_type)
    np.save(ART_DIR / "point_to_cluster.npy", point_to_cluster)
    np.save(ART_DIR / "top_idx.npy", top_idx)
    with open(ART_DIR / "coverage_table.json", "w") as f:
        json.dump(coverage_table, f, indent=2)
    with open(ART_DIR / "topic_keywords.json", "w") as f:
        json.dump({int(t): kw[t] for t in range(n_topics)}, f, indent=2)
    with open(ART_DIR / "eval_examples.json", "w") as f:
        json.dump(eval_examples, f, indent=2)
    log(f"saved artifacts to {ART_DIR}/")
    log.close()


def write_report(
    args, n_w, n_o, joint, wiki_adj, oa_adj, n_bridges, tau0,
    coverage_table, strict_mnn, eig_link, n_topics, order, point_to_cluster,
    top_idx, top_masses, node_type, joint_titles, kw, frontier_in_out,
    topic_oa_frac, topic_rank, oa_records, eval_examples, n_oa_clustered,
    topic_wiki_foundations, assign_stats, oa_density,
) -> None:
    L: List[str] = []
    L.append("# Dictionary-bridged joint-graph curriculum (wiki-NLP ∪ OA-NLP)\n")
    L.append(
        "The **best-of-both** test: ONE scipy-sparse graph fusing Wikipedia's "
        "pedagogical link backbone with OpenAlex's citation coverage/frontier, "
        "bridged by **bt_cos** (back-translation cosine) cross-corpus edges, then "
        "run through the validated wiki-native curriculum pipeline. Does it keep "
        "wiki's clean foundations→frontier ordering AND enrich the frontier with "
        "niche/recent OA papers the wiki-only curriculum "
        "(`docs/wikipedia-curriculum.md`) lacked?\n"
    )

    L.append("## Joint-graph construction\n")
    L.append(
        f"- **Wiki nodes:** bg-norm top-{n_w} of the cached subgraph "
        f"`{args.wiki_store}/curriculum_subgraph_n{args.n_select}.npz` (seeded "
        f"forward PR `p` / uniform-teleport background PR `g`, select by "
        f"`p^0.5·log(p/g)`, α={ALPHA}), wiki links restricted to that set "
        f"({wiki_adj.nnz} edges).\n"
        f"- **OA nodes:** {oa_density['n_seed']} OpenAlex NLP-Techniques (T10181) "
        f"cosine seeds (top cosine to the NLP-article centroid, raw GIST-384)"
        + (
            f", **expanded (Fix 1) to the 1-hop citation neighborhood within the "
            f"full 87k NLP slice**: +{oa_density['n_expansion_nodes']} cited works "
            f"present in the 87k → {n_o} OA nodes total.\n"
            if args.oa_expand else
            f" (no expansion; {n_o} OA nodes).\n"
        )
        + f"  **OA citation density (the v1 bottleneck):** "
        f"v1-within-seed-set = {oa_density['v1_edges']} edges "
        f"({oa_density['v1_edges_per_node']:.2f}/node) → "
        f"{'expanded' if args.oa_expand else 'this run'} = {oa_adj.nnz - n_o} edges "
        f"({oa_density['edges_per_node']:.2f}/node). Same raw-GIST NLP slice used by "
        f"the cross-corpus alignment work, NOT the domain-scale gap curriculum.\n"
        f"- **Bridges:** bt_cos edges (below), {n_bridges} kept at τ={tau0} "
        f"(the curriculum-run threshold), added symmetrically, weighted by bt_cos.\n"
        f"- **Mass rebalancing (Fix 3):** per-corpus-select="
        f"{args.per_corpus_select}, mass-rebalance={args.mass_rebalance!r}, "
        f"teleport-split={args.teleport_split} (<0 = uniform-over-all, v1).\n"
        f"- **Joint graph:** {joint.shape[0]} nodes ({n_w} wiki + {n_o} OA), "
        f"{joint.nnz} directed edges, with a node-type vector (0=wiki, 1=OA).\n"
    )

    L.append("## bt_cos bridge coverage (vs strict mutual-NN)\n")
    L.append(
        "Per-slice mean-matched cosine `sim = X_wiki @ X_oa.T`; forward/backward "
        "argmax round-trip gives `bt_cos_i = cos(node_i, node_{rt(i)})`; a bridge "
        "survives if `bt_cos > τ`. **Coverage** = fraction of that side's nodes "
        f"with ≥1 surviving bridge. Strict mutual-NN coverage (no τ, the prior "
        f"~0.37 baseline) = **{strict_mnn:.4f}** on this slice.\n"
    )
    L.append("| scorer | τ | #bridges | wiki coverage | OA coverage |")
    L.append("| :-- | --: | --: | --: | --: |")
    for row in coverage_table:
        L.append(f"| {row['scorer']} | {row['tau']:.1f} | {row['n_bridges']} | "
                 f"{row['wiki_cov']:.3f} | {row['oa_cov']:.3f} |")
    L.append(
        f"\nbt_cos buys substantially more coverage than strict mutual-NN "
        f"({strict_mnn:.3f}): at τ=0.5 the round-trip-stable criterion admits many "
        f"non-mutual matches whose embedding is still self-consistent across the "
        f"round trip.\n"
    )

    L.append("## The joint curriculum (prerequisite → frontier order)\n")
    L.append(f"LPP subspace (k={args.lpp_k}) on the hub-suppressed JOINT link "
             f"affinity; smallest λ {np.round(eig_link[:5], 4).tolist()}. "
             f"Bisecting-KMeans into {n_topics} topics; frontier-adjacency "
             f"ordering.\n")
    L.append("| Rank | Size | OA frac | in−out | Top keywords | Sample nodes (★=OA) |")
    L.append("| :--: | ---: | ---: | ---: | :-- | :-- |")
    for rank, t in enumerate(order):
        idxs = np.where(point_to_cluster == t)[0]
        sub_order = idxs[np.argsort(top_masses[idxs])[::-1]]
        samp = []
        for m in sub_order[:5]:
            ti = top_idx[m]
            star = "★" if node_type[ti] == 1 else ""
            samp.append(f"{star}{joint_titles[ti][:38]}")
        kw_str = ", ".join(w for w, _ in kw[t][:7])
        L.append(f"| {rank+1} | {idxs.size} | {topic_oa_frac[t]:.2f} | "
                 f"{frontier_in_out[t]:+.4f} | {kw_str} | {'; '.join(samp)} |")

    L.append("\n## DECISIVE EVALUATION — do OA-only frontier works land near their "
             "wiki prerequisites?\n")
    oa_a = assign_stats["oa_assigned"]; oa_t = assign_stats["oa_total_topk"]
    wk_a = assign_stats["wiki_assigned"]; wk_t = assign_stats["wiki_total_topk"]
    oa_ns = assign_stats.get("oa_in_nonsingleton", oa_a)
    oa_wa = assign_stats.get("oa_in_wiki_anchored", oa_a)
    n_sing = assign_stats.get("n_singletons", 0)
    L.append(
        f"**The binding number — does the OA frontier even reach the curriculum?** "
        f"The dendrogram mass-cut assigns only high-mass subtrees to named topics; "
        f"everything else becomes a cluster `-1` intermediary. Of the **{oa_t} OA "
        f"nodes in the top-{assign_stats['top_k']}** clustering pool, **{oa_a} "
        f"({100*oa_a/max(oa_t,1):.1f}%) reach a named topic**; wiki: **{wk_a}/{wk_t} "
        f"({100*wk_a/max(wk_t,1):.1f}%)**. Stripping the degenerate single-OA-node "
        f"\"topics\" the mass-cut sometimes promotes ({n_sing} singletons total), "
        f"**{oa_ns} OA land in a non-singleton topic and {oa_wa} in a "
        f"wiki-anchored (≥3 wiki members) topic** — the latter is the genuine "
        f"best-of-both count (OA grafted onto a wiki-foundation-bearing topic).\n"
    )
    L.append(
        f"**OA-only frontier works** = clustered OA nodes with the weakest "
        f"cross-corpus bridge (bottom-quartile bt_cos) — niche/recent papers with "
        f"no strong wiki counterpart. Of {n_oa_clustered} OA nodes in the "
        f"clustered set, the bottom quartile are examined below. For each, we list "
        f"the topic it landed in, that topic's prerequisite→frontier rank, the "
        f"topic's wiki foundations (highest-mass wiki members), and the topic "
        f"keywords. The question: does the OA-only paper sit in a topic anchored "
        f"by the RIGHT wiki concepts, ordered at/after those foundations?\n"
    )
    L.append("| OA-only work (low bt_cos) | bt_cos | topic rank | topic OA frac | "
             "topic keywords | wiki foundations in same topic |")
    L.append("| :-- | --: | --: | --: | :-- | :-- |")
    for e in eval_examples[:14]:
        L.append(
            f"| {e['title'][:46]} | {e['btcos']:.2f} | {e['topic_rank']} | "
            f"{e['topic_oa_frac']:.2f} | {', '.join(e['topic_keywords'][:5])} | "
            f"{'; '.join(f[:28] for f in e['wiki_foundations'][:4])} |"
        )

    L.append("\n### Concrete examples\n")
    for e in eval_examples[:6]:
        L.append(f"- **OA-only:** {e['title']}  (bt_cos={e['btcos']:.2f}, best wiki "
                 f"match was *{e['wiki_match']}*)")
        L.append(f"  - landed in topic **rank {e['topic_rank']}/{n_topics}** "
                 f"(OA frac {e['topic_oa_frac']:.2f}); keywords: "
                 f"{', '.join(e['topic_keywords'][:6])}")
        L.append(f"  - wiki foundations in that topic: "
                 f"{', '.join(e['wiki_foundations'][:5]) or '(none — OA-dominated topic)'}\n")

    L.append("## Auto-generated metric summary\n")
    L.append("_Numbers only — the curated verdict (v1 + the v2 dual-eval) lives in "
             "the hand-authored results doc "
             "`docs/wikipedia-openalex-joint-curriculum.md`._\n")
    L.append(
        f"- **OA citation density:** v1-within-seed "
        f"{oa_density['v1_edges_per_node']:.2f}/node → this run "
        f"{oa_density['edges_per_node']:.2f}/node "
        f"({oa_density['n_nodes']} OA nodes, "
        f"oa_expand={args.oa_expand}).\n"
        f"- **OA into the curriculum:** {oa_a}/{oa_t} top-k OA reach a named topic "
        f"({100*oa_a/max(oa_t,1):.1f}%); {oa_ns} in non-singleton topics, "
        f"{oa_wa} in wiki-anchored (≥3 wiki) topics. Wiki: {wk_a}/{wk_t} "
        f"({100*wk_a/max(wk_t,1):.1f}%). {n_sing} singleton topics.\n"
        f"- **Rebalancing config:** per-corpus-select={args.per_corpus_select}, "
        f"mass-rebalance={args.mass_rebalance!r}, "
        f"teleport-split={args.teleport_split}.\n"
        f"- **Bridge coverage (CSLS τ={tau0}):** see the coverage table above; "
        f"strict mutual-NN baseline {strict_mnn:.3f}.\n"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
