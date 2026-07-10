#!/usr/bin/env python
# artifacts_scripts/wiki/wiki_pedagogical_lpp.py
"""Wikipedia **pedagogical subspace** via Locality Preserving Projections (LPP)
on the **link graph** affinity (NOT the usual kNN-of-embeddings).

Premise (docs/backlog/curriculum-generalization-pedagogical-subspace.md): the
Wikipedia link graph encodes human-curated "what you need to read to understand
X" structure that GIST cosine geometry does not. LPP finds the linear projection
``P`` (384×k) that minimizes ``Σ_ij W_ij ‖Pᵀxᵢ − Pᵀxⱼ‖²`` — i.e. it pulls
link-connected articles together in the projected space. If links carry
pedagogical signal beyond raw embedding similarity, reprojecting the curriculum
into ``Y = Pᵀ X`` should sharpen / make-more-prerequisite-coherent the topic
clustering relative to raw GIST.

**Reuses bookgap's LPP solver math.** ``bookgap.denoisers.LPPDenoiser`` already
solves the generalized eigenproblem ``X_cᵀ L X_c a = λ X_cᵀ D X_c a`` for the k
smallest eigenvalues via ``scipy.linalg.eigh`` (see ``_solve_eigh_robustly``),
where ``X_c`` is the mean-centered (n×d) embedding matrix and ``L = D − W``,
``D`` = degree of the symmetrized affinity. The novelty here is purely the
**affinity**: ``W`` = the symmetrized, **hub-suppressed** Wikipedia link graph
instead of ``adj + adj.T`` of a citation graph or a kNN-of-embeddings graph. We
call the same ``_solve_eigh_robustly`` so the numerics match the cloud denoiser
exactly; the only addition is discarding any trivial near-DC eigenvector (the
centered/``XᵀDX``-constrained 384-d problem does not normally produce one, but we
guard).

**Hub suppression is essential.** Without it the link Laplacian inherits the
DOI/ISBN/PubMed/ISO bibliographic-hub pathology documented in
``docs/wikipedia-curriculum.md``: those pages have huge in-degree (linked from
every citation template), so the unsuppressed Laplacian's largest-degree nodes
dominate ``D`` and the smoothness penalty collapses toward "don't separate the
hubs from everything." We downweight each edge into target ``j`` by
``1/log(indeg_j + e)`` (in-degree popularity), the same lever the curriculum's
background-normalized PageRank uses, then recompute ``D`` from the weighted ``W``.

**CPU-only, reuses the cached subgraph** ``curriculum_subgraph_n40000.npz`` (CSR
link adjacency + aligned GIST embeddings + page_ids + titles). No 43 GB dump, no
GPU. Whole script runs in ~2–3 min.

Experiments
-----------
1. **Top pedagogical axes** — for the first few LPP directions, list the
   articles at the extreme +/− ends of ``Y[:,axis]`` to read off what each axis
   captures (prerequisite-depth / topical structure / noise).
2. **Reproject + re-cluster** the bg-norm-selected curriculum top-k in the LPP
   space; compare cluster quality vs raw-GIST clustering (same bisecting-KMeans
   clusterer, same ~n_topics). PageRank selection + frontier ordering are
   graph-based and unaffected — LPP changes only the CLUSTERING + the
   representative/keyword selection, so the comparison is scoped to cluster
   quality (silhouette, link-graph modularity Q / flow-capture, keyword
   distinctness).
3. **LPP-on-links vs LPP-on-kNN-of-embeddings** — same nodes, same solver, but
   the affinity is the mutual-kNN graph of the GIST embeddings instead of the
   link graph. Does the human-curated link structure change the axes/clusters
   meaningfully? This is the crux of whether the link graph adds pedagogical
   signal a kNN graph would not.

Usage::

    uv run python -m artifacts_scripts.wiki.wiki_pedagogical_lpp \
        --wiki-store cache/wiki/store/enwiki_full \
        --ranking-top-k 4000 --desired-n-topics 12 --use-abstracts

Output: ``docs/wikipedia-pedagogical-lpp.md`` (+ machine-readable artifacts under
``artifacts/curriculum/wiki_pedagogical_lpp/``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import eigh
from sklearn.metrics import silhouette_score

from artifacts_scripts.curriculum.compare_clusterings import (
    avg_pairwise_jaccard_distance,
    citation_flow_capture,
    keyword_distinctness,
    modularity_q,
)
from artifacts_scripts.curriculum.pagerank import compute_pagerank
from artifacts_scripts.wiki.wiki_curriculum_demo import (
    SEED_TITLES,
    cluster_embeddings,
    keywords_for_topics,
    load_subgraph,
)
from bookgap.denoisers import _solve_eigh_robustly, subset_range

RUN_LOG = Path("artifacts/wiki_pedagogical_lpp/run.log")
ART_DIR = Path("artifacts/curriculum/wiki_pedagogical_lpp")
# Machine-generated full report (eigenspectrum + axes + metric tables + auto
# number summary). The CURATED results doc is docs/wikipedia-pedagogical-lpp.md,
# hand-authored from this report's numbers so re-runs don't clobber the
# interpretation. Override with --out to regenerate the machine report.
OUT_DOC = ART_DIR / "report.md"

# Match the curriculum demo defaults so the clustering comparison is
# apples-to-apples with docs/wikipedia-curriculum.md.
ALPHA = 0.85
BG_ALPHA = 0.5
BG_EPS = 1e-12


class PhaseLog:
    """Flushed sidecar logger (conda run buffers stdout)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._t0 = time.time()

    def __call__(self, line: str) -> None:
        stamp = f"[{time.time() - self._t0:7.1f}s] {line}"
        self._fh.write(stamp + "\n")
        self._fh.flush()
        print(stamp, flush=True)

    def close(self) -> None:
        self("done")
        self._fh.close()


# ---------------------------------------------------------------------------
# Affinities
# ---------------------------------------------------------------------------

def hub_suppressed_link_affinity(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric, hub-suppressed link affinity ``W``.

    Steps (all sparse, CPU):
      1. Booleanize the directed link adjacency (``A[i,j]=1`` iff i links to j).
      2. Hub-suppress: weight each directed edge ``i→j`` by ``1/log(indeg_j+e)``
         where ``indeg_j`` is j's global in-degree in the subgraph. Bibliographic
         hubs (DOI/ISBN/ISO) have indeg in the thousands, so their inbound edges
         are heavily damped; a niche prerequisite linked by a handful of pages
         keeps near-unit weight. This is the edge-level analog of the curriculum's
         ``1/g`` background-PageRank demotion.
      3. Symmetrize: ``W = (W_dir + W_dirᵀ)/2`` so the Laplacian is PSD and the
         GEP is real-symmetric (LPP requires an undirected affinity).
    """
    A = adj.copy()
    A.data[:] = 1.0  # boolean: edge present
    indeg = np.asarray(A.sum(axis=0)).ravel()  # global in-degree of each target
    supp = 1.0 / np.log(indeg + np.e)  # in [0,1]; ->0 for huge-indeg hubs
    W_dir = A.multiply(supp[np.newaxis, :]).tocsr()  # scale column (=target) j
    W = (W_dir + W_dir.T) * 0.5
    return W.tocsr()


def mutual_knn_affinity(X: NDArray[np.float32], k: int) -> sp.csr_matrix:
    """Symmetric kNN affinity of the (cosine) GIST embeddings — the *usual* LPP
    affinity, for the links-vs-kNN contrast.

    Cosine kNN (embeddings L2-normalized → inner-product search), binary edges,
    symmetrized by ``max`` (an edge survives if either endpoint is in the
    other's kNN). This is the embedding-similarity graph the link graph is being
    contrasted against; deliberately NO hub suppression (a kNN graph is
    near-k-regular by construction, so it has no popularity hubs to suppress —
    that asymmetry is itself part of the point).

    Uses ``faiss`` (the codebase's NN convention; brute sklearn kNN on 477k×384
    is ~1 hr). For 477k nodes an exact ``IndexFlatIP`` search is ~3 min when it
    gets all cores but degrades to ~10 min under CPU contention from a
    co-scheduled job (its OpenMP search-parallelism is unreliable on this box),
    so we use a high-recall **approximate** ``IVFFlat`` index instead: ``nlist``
    Voronoi cells with ``nprobe`` set high enough that recall@k is ~1.0. The
    binary affinity only needs the embedding-neighborhood community structure
    for the links-vs-kNN contrast, not ULP-exact neighbor ids, so near-exact
    recall is more than sufficient — and it finishes in seconds regardless of
    contention. Trained on the data itself with a fixed seed for reproducibility.
    """
    import faiss

    Xn = np.ascontiguousarray(
        X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12), dtype=np.float32
    )
    n, d = Xn.shape
    nlist = max(1, int(np.sqrt(n)))            # ~691 cells for 477k rows
    nprobe = min(nlist, 64)                    # high recall (covers the cell + 63 neighbors)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = 42                         # deterministic k-means for the cells
    index.train(Xn)
    index.add(Xn)
    index.nprobe = nprobe
    _, idx = index.search(Xn, k + 1)  # k+1 to drop the self-hit
    rows = np.repeat(np.arange(n), k + 1)
    cols = idx.ravel()
    valid = cols >= 0  # IVF can return -1 for under-filled probes; drop them
    rows, cols = rows[valid], cols[valid]
    self_mask = rows != cols  # drop self-edges
    rows, cols = rows[self_mask], cols[self_mask]
    A = sp.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)), shape=(n, n)
    )
    W = A.maximum(A.T)  # OR symmetrization (edge if either endpoint is a neighbor)
    return W.tocsr()


# ---------------------------------------------------------------------------
# LPP solver (reuses bookgap.denoisers numerics)
# ---------------------------------------------------------------------------

def lpp_projection(
    X: NDArray[np.float64],
    W: sp.csr_matrix,
    k: int,
    log: PhaseLog,
    dc_floor: float = 1e-6,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve LPP ``X_cᵀ L X_c a = λ X_cᵀ D X_c a`` for the k smallest
    non-trivial eigenvectors. Returns ``(P (d×k), eigvals (k,))``.

    Mirrors ``bookgap.denoisers.LPPDenoiser`` exactly (centered X, ``L = D−W``,
    smallest-eigenvalue ``eigh`` via the shared ``_solve_eigh_robustly``), with
    one addition: any eigenvector whose eigenvalue is below ``dc_floor`` is
    treated as trivial/DC and discarded before taking the k smallest. (In the
    centered + ``XᵀDX``-constrained 384-d problem this guard normally does not
    fire — the constant component lives in the discarded mean — but LPP on a
    disconnected/near-constant affinity can produce one, so we drop it to be
    safe.)
    """
    d = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - W

    A_mat = Xc.T @ (L @ Xc)  # d×d  (== X_cᵀ L X_c)
    B_mat = Xc.T @ (D @ Xc)  # d×d  (== X_cᵀ D X_c)
    A_mat = 0.5 * (A_mat + A_mat.T)  # enforce exact symmetry (PSD numerics)
    B_mat = 0.5 * (B_mat + B_mat.T)

    # Full spectrum so we can discard near-DC eigenvectors, then take k smallest.
    res = _solve_eigh_robustly(a_matrix=A_mat, b_matrix=B_mat, subset=None)
    eigvals, V = res.eigvals, res.P_or_U  # ascending
    n_dc = int((eigvals < dc_floor).sum())
    if n_dc:
        log(f"  discarding {n_dc} near-DC eigenvector(s) (λ<{dc_floor:.0e})")
    keep = np.arange(n_dc, n_dc + k)
    P = V[:, keep]
    return P, eigvals[keep]


def lpp_eigenspectrum(
    X: NDArray[np.float64], W: sp.csr_matrix, n: int = 16
) -> NDArray[np.float64]:
    """First ``n`` LPP eigenvalues (ascending) for spectrum reporting."""
    d = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - W
    XLX = Xc.T @ (L @ Xc)  # one big matmul each (don't recompute for symmetrize)
    XDX = Xc.T @ (D @ Xc)
    A_mat = 0.5 * (XLX + XLX.T)
    B_mat = 0.5 * (XDX + XDX.T)
    sub = subset_range(n=d, subset_size=n, smallest=True)
    res = _solve_eigh_robustly(a_matrix=A_mat, b_matrix=B_mat, subset=sub)
    return res.eigvals


# ---------------------------------------------------------------------------
# bg-norm top-k selection (replicated from wiki_curriculum_demo)
# ---------------------------------------------------------------------------

def bg_norm_selection(
    adj: sp.csr_matrix,
    id_list: List[int],
    titles: List[str],
    top_k: int,
    log: PhaseLog,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the demo's ``--bg-norm`` top-k: seeded forward PR ``p`` and
    uniform-teleport background PR ``g`` on the SAME adjacency (α=0.85),
    select top-k by ``p^0.5 · log((p+ε)/(g+ε))``. Returns
    ``(top_idx, raw_masses_topk, selection_score_topk)``."""
    n = len(id_list)
    title_to_local = {t: i for i, t in enumerate(titles)}
    seed_idx = [title_to_local[t] for t in SEED_TITLES if t in title_to_local]
    if not seed_idx:
        raise SystemExit("No seed pages found in subgraph.")
    log(f"  seeds in subgraph: {[titles[i] for i in seed_idx]}")
    seeds = np.zeros(n, dtype=np.float64)
    for i in seed_idx:
        seeds[i] = 1.0 / len(seed_idx)
    masses = compute_pagerank(adj, seeds, alpha=ALPHA, tol=1e-9, max_iter=200, normalize=True)
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    g_bg = compute_pagerank(adj, uniform, alpha=ALPHA, tol=1e-9, max_iter=200, normalize=True)
    p_fg = np.maximum(masses, 0.0)
    score = np.power(p_fg, BG_ALPHA) * np.log((p_fg + BG_EPS) / (g_bg + BG_EPS))
    k = min(top_k, n)
    top_idx = np.argsort(score)[-k:][::-1].astype(np.int64)
    return top_idx, masses[top_idx], score[top_idx]


# ---------------------------------------------------------------------------
# Cluster-quality scoring (reuses compare_clusterings)
# ---------------------------------------------------------------------------

def cluster_quality(
    coords_for_silhouette: NDArray[np.float64],
    point_to_cluster: NDArray[np.intp],
    sub_adj: sp.csr_matrix,
    topic_keywords: Dict[int, List[str]],
) -> Dict[str, float]:
    """Bundle of intrinsic + link-graph + keyword cluster-quality metrics.

    - ``silhouette``: mean silhouette in ``coords_for_silhouette`` (the space the
      clustering was performed in — raw GIST for the baseline, LPP-Y for LPP).
      Geometric separation/compactness of the discovered clusters.
    - ``modularity_Q`` / ``flow_capture``: on the (hub-suppressed) link subgraph
      restricted to the clustered nodes — do the clusters respect the
      human-curated link structure? This is the pedagogical-signal axis.
    - ``jaccard_dist`` / ``n_kw_3plus``: keyword distinctness across topics.
    """
    n_topics = int(point_to_cluster.max()) + 1
    sil = (
        float(silhouette_score(coords_for_silhouette, point_to_cluster))
        if n_topics > 1
        else float("nan")
    )
    Q = modularity_q(sub_adj, point_to_cluster)
    flow = citation_flow_capture(sub_adj, point_to_cluster)
    jacc = avg_pairwise_jaccard_distance(topic_keywords, top_k=12)
    _, n_crowded = keyword_distinctness(topic_keywords, top_k=12)
    sizes = np.bincount(point_to_cluster[point_to_cluster >= 0], minlength=n_topics)
    return {
        "n_topics": n_topics,
        "n_singletons": int((sizes == 1).sum()),
        "silhouette": sil,
        "modularity_Q": Q,
        "flow_capture": flow,
        "jaccard_dist": jacc,
        "n_kw_3plus": int(n_crowded),
    }


def topic_table(
    point_to_cluster: NDArray[np.intp],
    kw: List[List[Tuple[str, float]]],
    top_titles: List[str],
    masses: NDArray[np.float64],
) -> List[str]:
    """Markdown topic table ordered by total raw PR mass (same as the demo's
    default 'mass' ordering — ordering is graph-based and identical across
    embeddings; only the cluster membership/keywords differ)."""
    n_topics = int(point_to_cluster.max()) + 1
    topic_mass = np.array(
        [masses[point_to_cluster == t].sum() for t in range(n_topics)]
    )
    order = np.argsort(-topic_mass)
    lines = ["| Rank | Size | PR mass | Top keywords | Sample pages |",
             "| :--: | ---: | ---: | :-- | :-- |"]
    for rank, t in enumerate(order):
        idxs = np.where(point_to_cluster == t)[0]
        sub_order = idxs[np.argsort(masses[idxs])[::-1]]
        samp = "; ".join(top_titles[j][:40] for j in sub_order[:5])
        kw_str = ", ".join(w for w, _ in kw[t][:8])
        lines.append(
            f"| {rank+1} | {idxs.size} | {topic_mass[t]:.4f} | {kw_str} | {samp} |"
        )
    return lines


def extreme_axis_articles(
    Y: NDArray[np.float64], titles: List[str], axis: int, n: int = 12
) -> Tuple[List[str], List[str]]:
    """Titles at the extreme +/− ends of LPP ``Y[:, axis]``."""
    col = Y[:, axis]
    order = np.argsort(col)
    neg = [titles[i] for i in order[:n]]
    pos = [titles[i] for i in order[-n:][::-1]]
    return pos, neg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/enwiki_full"))
    ap.add_argument("--n-select", type=int, default=40000,
                    help="subgraph cache suffix (curriculum_subgraph_n{N}.npz)")
    ap.add_argument("--ranking-top-k", type=int, default=4000)
    ap.add_argument("--desired-n-topics", type=int, default=12)
    ap.add_argument("--k-sweep", type=int, nargs="+", default=[32, 64, 128],
                    help="LPP subspace dimensions to sweep")
    ap.add_argument("--knn-k", type=int, default=10,
                    help="k for the mutual-kNN-of-embeddings contrast affinity")
    ap.add_argument("--use-abstracts", action="store_true",
                    help="use title + opening_text for keyword text (else titles)")
    ap.add_argument("--out", type=Path, default=OUT_DOC,
                    help="machine report path (default: the artifacts dir). The "
                         "curated results doc docs/wikipedia-pedagogical-lpp.md "
                         "is hand-authored and NOT overwritten by re-runs.")
    args = ap.parse_args()

    log = PhaseLog(RUN_LOG)
    log("Wikipedia pedagogical LPP (link-graph affinity)")

    # --- Load cached subgraph (no dump) --------------------------------------
    cache = args.wiki_store / f"curriculum_subgraph_n{args.n_select}.npz"
    log(f"load subgraph {cache.name}")
    id_list, adj, titles, abstracts, emb = load_subgraph(cache)
    adj = adj.tocsr()
    n = len(id_list)
    X = np.ascontiguousarray(emb, dtype=np.float64)
    if args.use_abstracts and any(abstracts):
        texts = [f"{titles[i]}. {abstracts[i]}".strip() for i in range(n)]
    else:
        texts = list(titles)
    log(f"  {n} nodes, {adj.nnz} directed edges, X={X.shape}")

    # --- bg-norm top-k selection (graph-based; identical across embeddings) --
    log("bg-norm top-k selection (seeded fwd PR / uniform bg PR)")
    top_idx, top_masses, top_score = bg_norm_selection(
        adj, id_list, titles, args.ranking_top_k, log
    )
    top_titles = [titles[i] for i in top_idx]
    top_texts = [texts[i] for i in top_idx]
    log(f"  selected top-{len(top_idx)} (retains {100*top_masses.sum():.1f}% PR mass)")

    # --- Hub-suppressed link affinity ----------------------------------------
    log("build hub-suppressed link affinity W")
    t = time.time()
    W_link = hub_suppressed_link_affinity(adj)
    deg_link = np.asarray(W_link.sum(axis=1)).ravel()
    log(f"  W_link nnz={W_link.nnz}, deg mean={deg_link.mean():.3f} "
        f"max={deg_link.max():.1f} in {time.time()-t:.1f}s")
    # Diagnostic: do the worst bibliographic hubs really get suppressed?
    A_bool = adj.copy(); A_bool.data[:] = 1.0
    indeg = np.asarray(A_bool.sum(axis=0)).ravel()
    hub_order = np.argsort(-indeg)[:8]
    log("  highest-in-degree pages (indeg -> edge-weight 1/log(indeg+e)):")
    for i in hub_order:
        log(f"    {titles[i][:40]:<42} indeg={int(indeg[i])}  w={1.0/np.log(indeg[i]+np.e):.4f}")

    # --- Eigenspectrum -------------------------------------------------------
    log("LPP eigenspectrum (link affinity)")
    spectrum_link = lpp_eigenspectrum(X, W_link, n=16)
    log(f"  smallest 16 λ: {np.round(spectrum_link, 5).tolist()}")

    # --- Experiment 1: inspect top pedagogical axes --------------------------
    log("Experiment 1: top LPP axes (extreme +/- articles)")
    k_max = max(args.k_sweep)
    P_link, eig_link = lpp_projection(X, W_link, k_max, log)
    Xc = X - X.mean(axis=0, keepdims=True)
    Y_link_full = Xc @ P_link  # n × k_max
    axis_report: List[Tuple[int, float, List[str], List[str]]] = []
    for axis in range(4):
        pos, neg = extreme_axis_articles(Y_link_full, titles, axis, n=12)
        axis_report.append((axis, float(eig_link[axis]), pos, neg))
        log(f"  axis {axis} (λ={eig_link[axis]:.4f})")
        log(f"    +: {', '.join(pos[:6])}")
        log(f"    -: {', '.join(neg[:6])}")

    # --- Experiment 2: reproject top-k + re-cluster vs raw GIST --------------
    log("Experiment 2: reproject top-k + re-cluster vs raw GIST")
    top_emb_raw = X[top_idx]                      # n_topk × 384 (raw GIST)
    results: Dict[str, Dict] = {}

    # Baseline: cluster raw GIST (this reproduces docs/wikipedia-curriculum.md).
    log("  [baseline] cluster raw GIST top-k")
    p2c_raw = cluster_embeddings(
        np.ascontiguousarray(top_emb_raw, dtype=np.float32),
        top_masses, args.desired_n_topics,
    )
    n_topics_raw = int(p2c_raw.max()) + 1
    kw_raw = keywords_for_topics(top_texts, p2c_raw, top_masses, n_topics_raw)
    # Link subgraph restricted to the clustered nodes (hub-suppressed weights):
    sub_W_link = W_link[top_idx][:, top_idx].tocsr()
    tk_raw = {t: [w for w, _ in kw_raw[t]] for t in range(n_topics_raw)}
    results["raw_gist"] = {
        "metrics": cluster_quality(
            np.ascontiguousarray(top_emb_raw), p2c_raw, sub_W_link, tk_raw
        ),
        "p2c": p2c_raw, "kw": kw_raw,
    }

    # LPP-on-links, swept over k.
    for k in args.k_sweep:
        log(f"  [lpp-link k={k}] reproject + cluster")
        P_k, _ = lpp_projection(X, W_link, k, log)
        Y_topk = (Xc[top_idx] @ P_k)  # n_topk × k, in LPP space
        p2c = cluster_embeddings(
            np.ascontiguousarray(Y_topk, dtype=np.float32),
            top_masses, args.desired_n_topics,
        )
        nt = int(p2c.max()) + 1
        kwk = keywords_for_topics(top_texts, p2c, top_masses, nt)
        tk = {t: [w for w, _ in kwk[t]] for t in range(nt)}
        results[f"lpp_link_k{k}"] = {
            "metrics": cluster_quality(
                np.ascontiguousarray(Y_topk), p2c, sub_W_link, tk
            ),
            "p2c": p2c, "kw": kwk, "k": k,
        }

    # --- Experiment 3: LPP-on-links vs LPP-on-kNN-of-embeddings ---------------
    log(f"Experiment 3: contrast vs LPP-on-mutual-kNN (k={args.knn_k})")
    t = time.time()
    W_knn = mutual_knn_affinity(X.astype(np.float32), args.knn_k)
    log(f"  W_knn nnz={W_knn.nnz} built in {time.time()-t:.1f}s")
    spectrum_knn = lpp_eigenspectrum(X, W_knn, n=16)
    log(f"  kNN smallest 16 λ: {np.round(spectrum_knn, 5).tolist()}")
    # Pick a representative k (middle of the sweep) for the head-to-head.
    k_rep = sorted(args.k_sweep)[len(args.k_sweep) // 2]
    P_knn, eig_knn = lpp_projection(X, W_knn, k_rep, log)
    Y_knn_full = Xc @ P_knn
    knn_axes: List[Tuple[int, float, List[str], List[str]]] = []
    for axis in range(4):
        pos, neg = extreme_axis_articles(Y_knn_full, titles, axis, n=12)
        knn_axes.append((axis, float(eig_knn[axis]), pos, neg))

    # Subspace overlap: how much do link-LPP and kNN-LPP top-k_rep subspaces
    # agree? Principal angles via SVD of P_link_repᵀ P_knn — mean cos(angle).
    P_link_rep, _ = lpp_projection(X, W_link, k_rep, log)
    M = P_link_rep.T @ P_knn  # k_rep × k_rep
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    subspace_overlap = float(np.mean(sv))  # mean canonical correlation
    log(f"  link-LPP vs kNN-LPP (k={k_rep}) mean canonical corr = {subspace_overlap:.3f}")

    # kNN-LPP clustering (representative k) for the metric head-to-head.
    Y_knn_topk = Xc[top_idx] @ P_knn
    p2c_knn = cluster_embeddings(
        np.ascontiguousarray(Y_knn_topk, dtype=np.float32),
        top_masses, args.desired_n_topics,
    )
    nt_knn = int(p2c_knn.max()) + 1
    kw_knn = keywords_for_topics(top_texts, p2c_knn, top_masses, nt_knn)
    tk_knn = {t: [w for w, _ in kw_knn[t]] for t in range(nt_knn)}
    results[f"lpp_knn_k{k_rep}"] = {
        "metrics": cluster_quality(
            np.ascontiguousarray(Y_knn_topk), p2c_knn, sub_W_link, tk_knn
        ),
        "p2c": p2c_knn, "kw": kw_knn, "k": k_rep,
    }

    # --- Write report --------------------------------------------------------
    log("write report")
    write_report(
        args, n, adj, id_list, titles, top_idx, top_masses, top_titles,
        spectrum_link, spectrum_knn, axis_report, knn_axes,
        results, k_rep, subspace_overlap, indeg, hub_order,
    )

    # --- Persist artifacts ---------------------------------------------------
    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ART_DIR / "P_link.npy", P_link)
    np.save(ART_DIR / "eig_link.npy", spectrum_link)
    np.save(ART_DIR / "top_idx.npy", top_idx)
    for name, r in results.items():
        np.save(ART_DIR / f"p2c_{name}.npy", r["p2c"])
    with open(ART_DIR / "metrics.json", "w") as f:
        json.dump({k: v["metrics"] for k, v in results.items()}, f, indent=2)
    log(f"saved artifacts to {ART_DIR}/")
    log.close()


def write_report(
    args, n, adj, id_list, titles, top_idx, top_masses, top_titles,
    spectrum_link, spectrum_knn, axis_report, knn_axes,
    results, k_rep, subspace_overlap, indeg, hub_order,
) -> None:
    L: List[str] = []
    L.append("# Wikipedia pedagogical subspace via LPP on the link graph\n")
    L.append(
        "Locality Preserving Projections (LPP) fit on the **Wikipedia link "
        "graph** as the affinity (hub-suppressed), contrasted against the usual "
        "kNN-of-embeddings affinity, to test whether reprojecting the curriculum "
        "into the link-LPP subspace sharpens topic clustering. Reuses the cached "
        f"subgraph `{args.wiki_store}/curriculum_subgraph_n{args.n_select}.npz` "
        f"({n} nodes, {adj.nnz} directed link edges, GIST-384 embeddings). "
        "CPU-only, no dump.\n"
    )
    L.append("## Method\n")
    L.append(
        "- **LPP solver:** reuses `bookgap.denoisers` numerics — solve the "
        "generalized eigenproblem `X_cᵀ L X_c a = λ X_cᵀ D X_c a` for the k "
        "smallest non-trivial eigenvalues via the shared `_solve_eigh_robustly` "
        "(`scipy.linalg.eigh`), `X_c` = mean-centered (n×384) GIST, `L = D − W`, "
        "`D` = degree of the symmetrized affinity `W`. Identical to "
        "`LPPDenoiser`; the only addition is discarding any near-DC eigenvector "
        "(`λ < 1e-6`).\n"
        "- **Affinity (the novelty):** `W` = the **symmetrized, hub-suppressed "
        "link graph** — each directed edge `i→j` weighted by `1/log(indeg_j+e)` "
        "(j's global in-degree), then `(W_dir+W_dirᵀ)/2`. Hub suppression is "
        "essential: without it the Laplacian inherits the DOI/ISBN/ISO "
        "bibliographic-hub pathology from `docs/wikipedia-curriculum.md`.\n"
        "- **Clustering comparison:** the bg-norm-selected curriculum top-"
        f"{len(top_idx)} (seeded forward PR / uniform-teleport background PR, "
        "`p^0.5·log(p/g)`, α=0.85 — identical to `docs/wikipedia-curriculum.md`) "
        "is clustered by the same bisecting-KMeans into ~"
        f"{args.desired_n_topics} topics, once in **raw GIST** and once in each "
        "**LPP** subspace. PageRank selection + frontier ordering are graph-based "
        "and unaffected — LPP changes only the clustering geometry, so the "
        "comparison is scoped to cluster quality.\n"
    )

    # Hub suppression sanity table.
    L.append("\n### Hub suppression is doing its job\n")
    L.append("| Highest-in-degree page | in-degree | edge weight `1/log(indeg+e)` |")
    L.append("| :-- | ---: | ---: |")
    for i in hub_order:
        L.append(f"| {titles[i]} | {int(indeg[i])} | {1.0/np.log(indeg[i]+np.e):.4f} |")
    L.append(
        "\nThe bibliographic hubs that dominate the raw link graph carry the "
        "smallest edge weights, so they no longer pin the Laplacian.\n"
    )

    # Eigenspectrum.
    L.append("## Eigenspectrum\n")
    L.append("Smallest 16 LPP eigenvalues (ascending; small λ = directions that "
             "best preserve affinity locality):\n")
    L.append("| affinity | " + " | ".join(f"λ{i}" for i in range(8)) + " |")
    L.append("| :-- |" + " ---: |" * 8)
    L.append("| link (hub-suppressed) | " +
             " | ".join(f"{v:.4f}" for v in spectrum_link[:8]) + " |")
    L.append("| kNN-of-embeddings | " +
             " | ".join(f"{v:.4f}" for v in spectrum_knn[:8]) + " |")
    L.append(
        f"\nNo trivial near-zero eigenvalue appears for either affinity "
        f"(smallest link λ = {spectrum_link[0]:.4f}, kNN λ = {spectrum_knn[0]:.4f}); "
        "the constant/DC component lives in the discarded mean (centered `X_c`) "
        "and the `XᵀDX` constraint, so the guard does not fire. The link spectrum "
        "rises "
        + ("faster" if spectrum_link[7] > spectrum_knn[7] else "more slowly")
        + " than the kNN spectrum — "
        + (
            "the link affinity is harder to satisfy with few linear directions "
            "(its locality structure is less aligned with the GIST geometry)."
            if spectrum_link[7] > spectrum_knn[7]
            else "GIST geometry already aligns with the kNN affinity, so kNN-LPP "
            "directions are 'cheaper'."
        )
        + "\n"
    )

    # Experiment 1.
    L.append("## Experiment 1 — what the top link-LPP axes capture\n")
    L.append(
        "Articles at the extreme +/− ends of each of the first four link-LPP "
        "directions `Y[:,axis] = X_cᵀ Pᵀ`:\n"
    )
    for axis, lam, pos, neg in axis_report:
        L.append(f"### Axis {axis} (λ={lam:.4f})\n")
        L.append(f"- **+ end:** {', '.join(pos)}")
        L.append(f"- **− end:** {', '.join(neg)}\n")

    # Experiment 2 metric table.
    L.append("## Experiment 2 — reproject + re-cluster vs raw GIST\n")
    L.append(
        "Same nodes, same clusterer, same target topic count. `silhouette` is "
        "measured in the space the clustering ran in (raw GIST vs LPP-Y) — "
        "intrinsic compactness. `Q` and `flow` are Newman modularity and "
        "within-cluster edge fraction on the **hub-suppressed link subgraph** "
        "restricted to the clustered nodes (higher = clusters respect "
        "human-curated links = pedagogical signal). `jacc` = mean pairwise "
        "keyword Jaccard distance (higher = more distinct topics); `#kw3+` = "
        "keywords shared by ≥3 topics (lower = cleaner).\n"
    )
    hdr = ("| variant | k | #topics | #sing | silhouette | link-Q | link-flow "
           "| jacc-dist | #kw3+ |")
    L.append(hdr)
    L.append("| :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
    row_order = ["raw_gist"] + [f"lpp_link_k{k}" for k in sorted(args.k_sweep)] + \
                [f"lpp_knn_k{k_rep}"]
    for name in row_order:
        if name not in results:
            continue
        m = results[name]["metrics"]
        k = results[name].get("k", "—")
        L.append(
            f"| {name} | {k} | {m['n_topics']} | {m['n_singletons']} | "
            f"{m['silhouette']:.3f} | {m['modularity_Q']:.3f} | "
            f"{m['flow_capture']:.3f} | {m['jaccard_dist']:.3f} | {m['n_kw_3plus']} |"
        )
    L.append(
        "\n*Note on silhouette comparability:* silhouette across different spaces "
        "(raw 384-d GIST vs k-d LPP) is **not** directly comparable as an absolute "
        "number — it is reported per-space. The cross-space comparison that "
        "matters is **link-Q / link-flow** (same subgraph for every row) and the "
        "keyword distinctness.\n"
    )

    # Best LPP-link topic table for eyeballing.
    best_link = max(
        (nm for nm in results if nm.startswith("lpp_link")),
        key=lambda nm: results[nm]["metrics"]["modularity_Q"],
    )
    L.append(f"### Topic table — raw GIST (baseline)\n")
    L.extend(topic_table(results["raw_gist"]["p2c"], results["raw_gist"]["kw"],
                         top_titles, top_masses))
    L.append(f"\n### Topic table — {best_link} (best link-LPP by link-Q)\n")
    L.extend(topic_table(results[best_link]["p2c"], results[best_link]["kw"],
                         top_titles, top_masses))

    # Experiment 3.
    L.append("\n## Experiment 3 — link affinity vs kNN-of-embeddings affinity\n")
    L.append(
        f"Same nodes and solver; the only change is the affinity. Mean canonical "
        f"correlation between the link-LPP and kNN-LPP k={k_rep} subspaces is "
        f"**{subspace_overlap:.3f}** "
        + (
            "(near 1 → the two subspaces are nearly the same, so the link graph "
            "adds little beyond embedding similarity)."
            if subspace_overlap > 0.8
            else "(well below 1 → the link graph spans a materially different "
            "subspace than embedding-similarity kNN — the human-curated links "
            "carry geometry the GIST neighbors do not)."
        )
        + "\n"
    )
    L.append(f"### kNN-LPP top axes (k={k_rep}), for contrast\n")
    for axis, lam, pos, neg in knn_axes:
        L.append(f"**Axis {axis}** (λ={lam:.4f}) +: {', '.join(pos[:8])}")
        L.append(f"  − : {', '.join(neg[:8])}\n")

    # Verdict (templated from the numbers; manually curated prose appended below).
    rg = results["raw_gist"]["metrics"]
    bl = results[best_link]["metrics"]
    knn = results[f"lpp_knn_k{k_rep}"]["metrics"]
    L.append("## Auto-generated metric summary\n")
    L.append("_Numbers only — the curated verdict lives in the hand-authored "
             "results doc `docs/wikipedia-pedagogical-lpp.md`:_\n")
    L.append(
        f"- Best link-LPP variant by link-modularity: **{best_link}** "
        f"(link-Q {bl['modularity_Q']:.3f} vs raw-GIST {rg['modularity_Q']:.3f}, "
        f"link-flow {bl['flow_capture']:.3f} vs {rg['flow_capture']:.3f}).\n"
        f"- Keyword distinctness: link-LPP jacc {bl['jaccard_dist']:.3f} / "
        f"#kw3+ {bl['n_kw_3plus']} vs raw-GIST {rg['jaccard_dist']:.3f} / "
        f"{rg['n_kw_3plus']}.\n"
        f"- kNN-LPP (k={k_rep}) link-Q {knn['modularity_Q']:.3f}, "
        f"flow {knn['flow_capture']:.3f}, jacc {knn['jaccard_dist']:.3f}.\n"
        f"- link-LPP vs kNN-LPP subspace mean canonical corr {subspace_overlap:.3f}.\n"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
