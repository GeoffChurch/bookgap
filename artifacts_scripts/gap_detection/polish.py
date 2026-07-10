"""Polish-as-Track-1-phase-1 machinery (P1-P4, 2026-06-11 dispatch).

P1  Miner cross-species: attribution of mined itemsets to a planted removal
    (the species-agnostic analog of "rank of the planted combo"), per-doc
    pattern scores (doc layer, comparable to geometric detectors), and
    cross-cluster conjunction mining for corridors.
P2  Level-set residualization: differential candidate scores (after - before)
    fed to the existing component machinery; component-level recovery and
    natural-pocket tracking.
P3a Voltage all-pairs change field: joint fg+bg kNN graph, extended electrodes
    on every adjacent cluster pair, per-pair normalized current change
    aggregated by max into one species-agnostic anomaly map.
P3b Commute-distance (effective-resistance) embedding of a sparse graph via
    truncated eigendecomposition of the Laplacian (never densified).
P4  Low-df keyword tail: deterministic tf-idf rebuild, tail selection, and
    tail-tier mining to augment the scoped itemset lattice.

All functions are pure CPU numpy/scipy/sklearn; randomness through seeds only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

EPS = 1e-12


# ===================================================================== P1: miner machinery
def itemset_support_matrix(family_keys: list[frozenset], K_dense: np.ndarray) -> sp.csr_matrix:
    """(n_itemsets, n_docs) sparse bool: itemset i supported by doc j.

    K_dense: (n_docs, n_keywords) bool. Column indices in itemsets must be valid
    for K_dense. Built once and reused across trials (supports are fg-independent).
    """
    K_dense = np.asarray(K_dense, dtype=bool)
    indptr = [0]
    indices: list[np.ndarray] = []
    for S in family_keys:
        cols = sorted(S)
        v = K_dense[:, cols[0]].copy()
        for c in cols[1:]:
            v &= K_dense[:, c]
        rows = np.where(v)[0]
        indices.append(rows)
        indptr.append(indptr[-1] + rows.size)
    data = np.ones(indptr[-1], dtype=np.uint8)
    return sp.csr_matrix(
        (data, np.concatenate(indices) if indices else np.array([], dtype=np.int64),
         np.asarray(indptr)),
        shape=(len(family_keys), K_dense.shape[0]),
    )


def attribution_stats(
    M_fg: sp.csr_matrix, removed_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-itemset (n_pre, n_removed): pre-removal fg support and support lost.

    M_fg: itemset x fg-doc support matrix (pre-removal fg). removed_mask over fg docs.
    """
    n_pre = np.asarray(M_fg.sum(axis=1)).ravel().astype(np.int64)
    n_rem = np.asarray(M_fg[:, removed_mask].sum(axis=1)).ravel().astype(np.int64)
    return n_pre, n_rem


def rank_family_fast(
    keys: list[frozenset], sup_bg: np.ndarray, M_fg: sp.csr_matrix,
    fg_keep_mask: np.ndarray, n_bg: int, alpha: float = 1.0, min_len: int = 2,
) -> list[dict]:
    """Vectorized drop-in for detectors.fightin_words_rank (identical output).

    Uses the precomputed itemset x fg-doc support matrix instead of per-itemset
    column ANDs; ties resolved exactly as the original (stable sort over family
    insertion order). Unit-tested for exact agreement.
    """
    n_f = np.asarray(M_fg[:, fg_keep_mask].sum(axis=1)).ravel().astype(np.int64)
    n_fg = int(fg_keep_mask.sum())
    n_b = np.asarray(sup_bg, dtype=np.int64)
    d_b = np.log((n_b + alpha) / (n_bg - n_b + alpha))
    d_f = np.log((n_f + alpha) / (n_fg - n_f + alpha))
    sigma = np.sqrt(1.0 / (n_b + alpha) + 1.0 / (n_f + alpha))
    z = (d_b - d_f) / sigma
    lens = np.fromiter((len(S) for S in keys), dtype=np.int64, count=len(keys))
    sel = np.where(lens >= min_len)[0]
    order = sel[np.argsort(-z[sel], kind="stable")]
    return [
        {"itemset": keys[i], "z": float(z[i]),
         "bg_support": int(n_b[i]), "fg_support": int(n_f[i])}
        for i in order
    ]


def attributable_best_rank(
    ranked: list[dict], key_index: dict[frozenset, int],
    n_pre: np.ndarray, n_rem: np.ndarray,
    min_pre: int = 10, ratio: float = 0.8,
) -> tuple[int | None, dict | None]:
    """Best 1-based rank of an itemset attributable to the removal, plus its row.

    Attributable := pre-removal fg support >= min_pre AND the removal took
    >= ratio of that support. The analog, for geometric plants, of "rank of the
    planted combo": is there ANY lattice signature this removal left, and how
    high does the miner rank it among all emerging patterns?
    """
    for pos, r in enumerate(ranked, start=1):
        i = key_index.get(r["itemset"])
        if i is None:
            continue
        if n_pre[i] >= min_pre and n_rem[i] >= ratio * n_pre[i]:
            return pos, {**r, "n_pre": int(n_pre[i]), "n_removed": int(n_rem[i])}
    return None, None


def doc_pattern_scores(
    ranked: list[dict], key_index: dict[frozenset, int], M_bg_csc: sp.csc_matrix,
    agg: str = "max",
) -> np.ndarray:
    """Per-bg-doc score from the trial's itemset ranking.

    agg='max': each doc scores the max z among itemsets it supports (sharp);
    agg='sum': sum of positive z (burden; sensitivity row). Docs supporting no
    ranked itemset score -inf (max) / 0 (sum). M_bg_csc: itemset x bg-doc support.
    """
    z = np.full(M_bg_csc.shape[0], -np.inf)
    for r in ranked:
        i = key_index.get(r["itemset"])
        if i is not None:
            z[i] = r["z"]
    z = np.where(np.isfinite(z), z, -np.inf)
    n_docs = M_bg_csc.shape[1]
    if agg == "sum":
        zpos = np.maximum(np.where(np.isfinite(z), z, 0.0), 0.0)
        return np.asarray(M_bg_csc.T @ zpos).ravel()
    out = np.full(n_docs, -np.inf)
    indptr, rows = M_bg_csc.indptr, M_bg_csc.indices
    for d in range(n_docs):
        sl = slice(indptr[d], indptr[d + 1])
        if sl.start < sl.stop:
            out[d] = float(np.max(z[rows[sl]]))
    return out


def cluster_vocab(
    Kf_dense: np.ndarray, member_mask: np.ndarray, n_top: int = 40,
    alpha: float = 1.0, min_df_in: int = 5,
) -> np.ndarray:
    """Distinctive keyword columns for an fg cluster: log-odds vs the rest of fg.

    Requires the term in >= min_df_in cluster members (a vocabulary, not noise).
    """
    Kf_dense = np.asarray(Kf_dense, dtype=bool)
    inside = Kf_dense[member_mask].sum(axis=0).astype(np.float64)
    outside = Kf_dense[~member_mask].sum(axis=0).astype(np.float64)
    n_in = float(member_mask.sum())
    n_out = float((~member_mask).sum())
    score = np.log((inside + alpha) / (n_in - inside + alpha)) - np.log(
        (outside + alpha) / (n_out - outside + alpha)
    )
    score[inside < min_df_in] = -np.inf
    order = np.argsort(score)[::-1][:n_top]
    return order[np.isfinite(score[order])]


def cross_conjunctions(
    vocab_a: np.ndarray, vocab_b: np.ndarray, K_bg_dense: np.ndarray,
    min_support: int = 10, cap: int = 1600,
) -> dict[frozenset, int]:
    """Support-pruned pairwise conjunctions A^B, A from one flanking cluster's
    vocabulary, B from the other's. Candidate space = |vocab_a| x |vocab_b|
    (<= cap enforced), pruned to scoped-bg support >= min_support.
    """
    K_bg_dense = np.asarray(K_bg_dense, dtype=bool)
    out: dict[frozenset, int] = {}
    n = 0
    for a in vocab_a:
        va = K_bg_dense[:, int(a)]
        for b in vocab_b:
            if int(a) == int(b):
                continue
            n += 1
            if n > cap:
                return out
            s = int((va & K_bg_dense[:, int(b)]).sum())
            if s >= min_support:
                out[frozenset((int(a), int(b)))] = s
    return out


# ===================================================================== P2: residualization
def residual_scores(before: dict, after: dict, variant: str = "full") -> np.ndarray:
    """Differential candidate score: after - before for the given variant.

    For variant='delta_only' this equals log r_F_after - log r_F_before (the bg
    side cancels) — the pure fg-change field, which is also the orphan-core
    criterion; 'full' differences the production statistic (hab + LCB wrapped).
    """
    return np.asarray(after[variant], dtype=np.float64) - np.asarray(
        before[variant], dtype=np.float64
    )


def component_recovery(comps: list[dict], region: np.ndarray, thr: float = 0.1) -> dict:
    """Rank (1-based, by the given component order) of the first component with
    IoU >= thr against the region; best IoU in top-3 and anywhere."""
    from artifacts_scripts.gap_detection.detectors import iou

    ious = [iou(c["members"], region) for c in comps]
    rank = next((i + 1 for i, v in enumerate(ious) if v >= thr), None)
    return {
        "rank": rank,
        "recall_top3": bool(rank is not None and rank <= 3),
        "best_iou_top3": round(max(ious[:3], default=0.0), 4),
        "best_iou_any": round(max(ious, default=0.0), 4),
        "n_components": len(comps),
    }


def order_components_by_mass(comps: list[dict], scores: np.ndarray) -> list[dict]:
    """Re-order components by total positive differential mass (descending)."""
    def mass(c: dict) -> float:
        return float(np.maximum(scores[c["members"]], 0.0).sum())

    return sorted(comps, key=mass, reverse=True)


def pocket_status(
    comps: list[dict], scores: np.ndarray, pocket: np.ndarray,
    real_mask: np.ndarray, top_frac: float = 0.05, thr: float = 0.1,
) -> dict:
    """Where a known natural pocket lands under a scoring: rank of the best-IoU
    component (None if < thr everywhere) + how many pocket members survive the
    top-frac selection at all."""
    from artifacts_scripts.gap_detection.detectors import iou

    real_idx = np.where(real_mask)[0]
    m = max(int(np.ceil(top_frac * real_idx.size)), 3)
    top = set(real_idx[np.argsort(scores[real_idx])[::-1]][:m].tolist())
    in_top = sum(1 for p in pocket if int(p) in top)
    ious = [iou(c["members"], pocket) for c in comps]
    rank = next((i + 1 for i, v in enumerate(ious) if v >= thr), None)
    return {"rank": rank, "best_iou": round(max(ious, default=0.0), 4),
            "members_in_topfrac": int(in_top), "pocket_size": int(pocket.size)}


# ===================================================================== P3a: voltage field
def gaussian_knn_graph(X: np.ndarray, k: int = 10) -> sp.csr_matrix:
    """Union-symmetrized kNN graph with Gaussian conductances exp(-(d/sigma)^2),
    sigma = median kNN distance — the established voltage-bridging recipe."""
    k = min(k, max(X.shape[0] - 1, 1))
    D = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False).tocoo()
    sigma = float(np.median(D.data)) + EPS
    W = sp.coo_matrix(
        (np.exp(-((D.data / sigma) ** 2)), (D.row, D.col)), shape=D.shape
    ).tocsr()
    return W.maximum(W.T)


def voltage_on_graph(
    W: sp.csr_matrix, clamp0: np.ndarray, clamp1: np.ndarray,
    leakage: float = 1e-8, mid_weight: bool = True,
) -> np.ndarray:
    """Per-node mid-potential-weighted current for one electrode pair on a fixed
    graph (the solve half of detectors.voltage_current_scores)."""
    from artifacts_scripts.gap_detection.circuitrank import compute_current_flow, compute_graph_potential

    fixed: dict[int, float] = {int(i): 0.0 for i in clamp0}
    fixed.update({int(i): 1.0 for i in clamp1})
    phi = np.asarray(
        compute_graph_potential(
            W, fixed_potentials=fixed, normalize=False, edge_weighting="pagerank",
            leakage=leakage, gmres_tol=1e-8,
        )
    )
    cur = compute_current_flow(W, phi, normalize=False, edge_weighting="pagerank")
    if mid_weight:
        p = np.clip(phi, 0.0, 1.0)
        cur = cur * (4.0 * p * (1.0 - p))
    return cur


def flow_share(cur: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Per-node share of the total eligible current (scale-free across solves).

    Removing a foreground bridge LOWERS the total source->sink current (the
    effective conductance drops), so absolute per-node currents shrink almost
    everywhere; only the share field is comparable before vs after. Measured on
    the 2D toy: the share of the corridor band also DROPS on fg removal (the fg
    superhighway no longer pulls flow through the region), so the localizing
    statistic is the MAGNITUDE of the share change, not its sign.
    """
    tot = float(cur[eligible].sum())
    return cur / (tot + EPS)


def centroid_adjacency(cents: np.ndarray, n_neighbors: int = 3) -> list[tuple[int, int]]:
    """Cluster adjacency = mutual centroid kNN union: (a,b) adjacent if b is among
    a's n_neighbors nearest centroids or vice versa. Ordered pairs a < b."""
    n = cents.shape[0]
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pairs = set()
    for a in range(n):
        for b in np.argsort(d[a])[:n_neighbors]:
            pairs.add((min(a, int(b)), max(a, int(b))))
    return sorted(pairs)


# ===================================================================== P3b: commute embed
def bipartite_graph(K: sp.spmatrix) -> sp.csr_matrix:
    """(n_docs + n_keywords) node graph from binary doc x keyword incidence."""
    B = sp.csr_matrix(K, dtype=np.float64)
    n, m = B.shape
    return sp.bmat([[None, B], [B.T, None]], format="csr")


def commute_embedding(
    W: sp.spmatrix, dim: int = 128, seed: int = 0, tol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Effective-resistance (commute-distance) embedding of a sparse graph.

    Uses the exact spectral identity (Lovasz): with L_sym = I - N,
    N = D^-1/2 W D^-1/2, eigenpairs (lambda_k, u_k),

        R_eff(i, j) = sum_{k>=2} (u_k(i)/sqrt(d_i) - u_k(j)/sqrt(d_j))^2 / lambda_k

    so the coordinates x_i[k] = u_k(i) / sqrt(lambda_k * d_i) reproduce
    ||x_i - x_j||^2 = R_eff(i, j) exactly at full rank and as the rank-`dim`
    truncation here (the smallest-lambda terms dominate and are kept). The
    smallest lambda_k of L_sym are obtained as the LARGEST eigenvalues of N via
    matvec-only ARPACK — no factorization (shift-invert splu on this graph
    explodes with fill from high-degree hub nodes; measured 87 CPU-min before
    abort). The n x n resistance matrix is NEVER densified. Restricts to the
    largest connected component; nodes outside get NaN rows.
    """
    W = sp.csr_matrix(W, dtype=np.float64)
    n = W.shape[0]
    ncomp, labels = connected_components(W, directed=False)
    lcc = int(np.argmax(np.bincount(labels)))
    mask = labels == lcc
    Wl = W[mask][:, mask]
    deg = np.asarray(Wl.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, EPS))
    N = (sp.diags(dinv) @ Wl @ sp.diags(dinv)).tocsr()
    from scipy.sparse.linalg import eigsh

    k = min(dim + 1, N.shape[0] - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=N.shape[0])
    nu, vecs = eigsh(N, k=k, which="LA", v0=v0, tol=tol)
    order = np.argsort(nu)[::-1]  # descending: nu_1 ~ 1 is the trivial pair
    nu, vecs = nu[order], vecs[:, order]
    lam = 1.0 - nu[1:]  # smallest nontrivial normalized-Laplacian eigenvalues
    vecs = vecs[:, 1:]
    X = (vecs * dinv[:, None] / np.sqrt(np.maximum(lam, EPS))[None, :]).astype(np.float32)
    out = np.full((n, X.shape[1]), np.nan, dtype=np.float32)
    out[mask] = X
    meta = {
        "dim": int(X.shape[1]), "n_nodes": int(n), "n_lcc": int(mask.sum()),
        "n_components": int(ncomp), "lambda_min": float(lam[0]),
        "lambda_max": float(lam[-1]), "trivial_nu": float(nu[0]),
    }
    return out, meta


# ===================================================================== P4: low-df tail
def rebuild_tfidf(ds) -> tuple[sp.csr_matrix, np.ndarray]:
    """Deterministically rebuild the full tf-idf matrix for the cached doc sample.

    Replays data._build's pick/vectorize/trim steps (same seed, same call order)
    and ASSERTS identity: page_ids match the cached dataset and the top-2000-df
    keyword matrix reproduces ds.K exactly. Reads raw abstracts internally;
    emits matrices only.
    """
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer

    from artifacts_scripts.gap_detection.data import (
        MIN_ABSTRACT_CHARS, N_DOCS, SEED, WIKI_PARQUET,
    )

    rng = np.random.default_rng(SEED)
    tbl = pq.read_table(WIKI_PARQUET)
    page_ids = tbl.column("page_id").to_numpy()
    abstracts = tbl.column("abstract").to_pylist()
    del tbl
    lengths = np.fromiter((len(a or "") for a in abstracts), dtype=np.int64)
    eligible = np.where(lengths >= MIN_ABSTRACT_CHARS)[0]
    pick = rng.choice(eligible, size=min(N_DOCS + 2_000, eligible.size), replace=False)
    pick.sort()
    texts = [abstracts[i] for i in pick]
    del abstracts
    vec = TfidfVectorizer(
        stop_words="english", lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=10, max_df=0.3, max_features=60_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    keep = np.where(X.getnnz(axis=1) >= 5)[0]
    if keep.size > N_DOCS:
        keep = keep[rng.choice(keep.size, size=N_DOCS, replace=False)]
        keep.sort()
    X = X[keep]
    pick = pick[keep]
    if not (page_ids[pick] == ds.page_ids).all():
        raise RuntimeError("tfidf rebuild: page_id mismatch with cached dataset")
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    kw_cols = np.argsort(df)[::-1][:2000]
    kw_cols.sort()
    if (sp.csr_matrix((X[:, kw_cols] > 0).astype(np.uint8)) != ds.K).nnz != 0:
        raise RuntimeError("tfidf rebuild: top-2000 keyword matrix mismatch")
    return X.tocsr(), vec.get_feature_names_out()


def low_df_tail(
    X: sp.csr_matrix, df_range: tuple[int, int] = (3, 50), cap: int = 8000,
) -> tuple[np.ndarray, sp.csr_matrix]:
    """Low-df tail columns (rarest-first when over cap) + binary doc x tail matrix.

    The pipeline's TfidfVectorizer(min_df=10) bounds the realized floor: terms
    with corpus df < 10 pre-trim never reached X, so the achievable tail is
    df in [max(df_range[0], observed_min), df_range[1]].
    """
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    cols = np.where((df >= df_range[0]) & (df <= df_range[1]))[0]
    if cols.size > cap:
        cols = cols[np.argsort(df[cols], kind="stable")[:cap]]
    cols = np.sort(cols)
    K_tail = (X[:, cols] > 0).astype(np.uint8).tocsr()
    return cols, K_tail


def mine_tail_family(
    K_top_bg: np.ndarray, K_tail_bg: sp.csr_matrix, min_support: int = 10,
    top_offset: int = 0,
) -> dict[frozenset, int]:
    """Tail-tier itemsets over scoped bg: tail singletons, tail x tail pairs and
    tail x top-2000 pairs, all with support >= min_support.

    Column indexing of the returned itemsets: top-2000 columns keep their ids
    (offset 0); tail column j maps to top_offset + j. The support threshold is
    deliberately lower than the main lattice's 30 — tail terms have global
    df <= 50, so scoped support > 30 is nearly unreachable; document with results.
    """
    Kt = sp.csr_matrix(K_tail_bg, dtype=np.float64)
    n_tail = Kt.shape[1]
    sup1 = np.asarray(Kt.sum(axis=0)).ravel().astype(np.int64)
    fam: dict[frozenset, int] = {}
    keep_tail = np.where(sup1 >= min_support)[0]
    for j in keep_tail:
        fam[frozenset([top_offset + int(j)])] = int(sup1[j])
    # tail x tail pairs (sparse product; tail columns have <= 50 nnz each)
    C_tt = (Kt.T @ Kt).tocoo()
    for i, j, v in zip(C_tt.row, C_tt.col, C_tt.data):
        if i < j and v >= min_support:
            fam[frozenset([top_offset + int(i), top_offset + int(j)])] = int(v)
    # tail x top-2000 pairs
    Ktop = sp.csr_matrix(np.asarray(K_top_bg, dtype=np.float64))
    C_t2 = (Kt.T @ Ktop).tocoo()
    for i, j, v in zip(C_t2.row, C_t2.col, C_t2.data):
        if v >= min_support:
            fam[frozenset([top_offset + int(i), int(j)])] = int(v)
    return fam
