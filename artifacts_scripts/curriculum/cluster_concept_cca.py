"""Hybrid clustering: CCA shared subspace of (neural embedding, concept keyword TF-IDF).

docs/backlog/curriculum-keyword-concept-seeding.md. Keyword-TF-IDF clustering reconstructs the
structure + beats the embedder on specificity, but fragments by surface form (language). The embedder
is language-robust but manufactures generic "miscellaneous" blobs. CCA keeps the directions where the
two AGREE (= content) and collapses where they DISAGREE — and the language/surface directions are
T-specific (no embedding correlate, since the embedder is language-invariant) → canonical correlation
≈ 0 → dropped automatically. So the shared subspace inherits keyword-specificity AND language-
robustness. Cluster there.

Recipe: concept-internal TF-IDF → LSA(SVD) → ridge-regularized CCA with the GIST embedding (whitened
cross-covariance SVD); keep the top-k canonical directions (k by elbow in the ρ spectrum, or --cca-k);
cluster the ρ-weighted consensus canonical coordinates (agglomerative, cosine). The ρ spectrum is
printed so you can see how many directions genuinely agree (and the rest = collapsed disagreement).

Example:
  python -m artifacts_scripts.curriculum.cluster_concept_cca \\
      --concept-gap artifacts/book_gaps/jurafsky_concept_full/concept_gap.json --field 17 --n 25 \\
      --out artifacts/curriculum/jurafsky_trackA_cca
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.cluster_concept_cca"
_EPS = 0.5


def _rcca(E, T, reg):
    """Ridge-regularized CCA via whitened cross-covariance SVD.
    Returns (rho, Ea, Tb): canonical correlations and the two views' canonical coordinates."""
    E = E - E.mean(0); T = T - T.mean(0)
    n = len(E)
    See = E.T @ E / n + reg * np.eye(E.shape[1])
    Stt = T.T @ T / n + reg * np.eye(T.shape[1])
    Set = E.T @ T / n

    def isqrt(S):
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-12)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    Ei, Ti = isqrt(See), isqrt(Stt)
    U, s, Vt = np.linalg.svd(Ei @ Set @ Ti, full_matrices=False)
    A = Ei @ U                                    # (dE, r) canonical directions for E
    B = Ti @ Vt.T                                 # (dT, r) for T
    return np.clip(s, 0, 1), E @ A, T @ B


def _decontaminate(Xc, rows, cloud, vocab, model_name):
    """The bipartite-EDGE 'meet': down-weight each (work, term) cell by the embedder relevance of the term
    to the work. A reference-string/venue term is OFF-TOPIC for a real paper → cos(embed term, embed work)
    is low → it's suppressed inside that work (leaving its topical vocabulary), while the same term stays
    full-weight in works where it genuinely IS the topic. Surgical de-contamination, no detection / threshold
    / work-loss / whack-a-mole — the cell-level form of which work-suppression (row) and word-suppression
    (column) are the cruder marginals. Returns the reweighted Xc."""
    from sentence_transformers import SentenceTransformer
    EMw = normalize(np.asarray(cloud.embeddings(), dtype=np.float64)[rows])
    terms = np.asarray((Xc > 0).sum(0)).ravel().nonzero()[0]
    loc = np.full(Xc.shape[1], -1, dtype=np.int64); loc[terms] = np.arange(len(terms))
    st = SentenceTransformer(model_name, device="cpu")
    termE = normalize(np.asarray(st.encode([vocab[int(t)] for t in terms]), dtype=np.float64))
    ccent = normalize(EMw.mean(0)[None, :])[0]                              # concept (field) centroid
    rel_c = termE @ ccent                                                   # cos(term, field) per term
    coo = Xc.tocoo()
    rel_w = (termE[loc[coo.col]] * EMw[coo.row]).sum(1)                     # cos(term, this work)
    rel = np.maximum(rel_w - rel_c[loc[coo.col]], 0.0)                      # CONTRASTIVE: work-specific, not field-generic
    LOGGER.info("decontaminate: %d cells × %d terms reweighted by CONTRASTIVE term-work relevance (mean=%.3f)",
                len(coo.data), len(terms), float(rel.mean()))
    return sp.csr_matrix((coo.data * rel, (coo.row, coo.col)), shape=Xc.shape)


def _chapter_gaps(X, rows, cloud, labels, n_ch, min_size=5):
    """Per-chapter gap = lexical_coh - coupling_coh (members' keyword-TFIDF self-similarity vs
    bibliographic-coupling self-similarity, the latter over non-orphan members only). A real topic has gap
    ≈ 0 or NEGATIVE (cites a shared core at least as much as it shares words); a venue/paratext chapter has
    a large POSITIVE gap (lexically tight, citing nothing in common). Returns ({chapter: gap}, KW, CP,
    cpok). The signal is a GROUP property (an anomalously lexically-over-coherent set, citation-unbacked) —
    sharp at the chapter level, weak per-work/per-term (4 probes), so detection is cluster-mediated."""
    labels = np.asarray(labels)
    N = len(rows)
    Xc = X[rows].astype(np.float64)
    idf = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))
    KW = normalize(Xc.multiply(idf).tocsr())
    G = cloud.citation_graph(False).tocsr().astype(np.float64); G.setdiag(0.0); G.eliminate_zeros()
    Xo = G[rows]; cdf = np.asarray((Xo > 0).sum(0)).ravel(); Xo = Xo[:, cdf > 0]
    CP = normalize(Xo.multiply(np.log(N / (cdf[cdf > 0] + 1.0))).tocsr())
    cpok = np.asarray(CP.multiply(CP).sum(1)).ravel() > 1e-9

    def coh(V, idx):
        return float((V[idx] @ V[idx].T).sum() / (len(idx) ** 2)) if len(idx) >= 2 else float("nan")

    gaps = {}
    for c in range(n_ch):
        R = np.where(labels == c)[0]; Rc = R[cpok[R]]
        if len(R) < min_size or len(Rc) < min_size:           # too few coupling-bearers to assess lineage
            continue                                          # (recency-orphan or tiny — NOT paratext)
        gaps[c] = coh(KW, R) - coh(CP, Rc)
    return gaps, KW, CP, cpok


def _paratext_scores(X, rows, cloud, labels, n_ch, min_size=5, k_mad=3.0):
    """CONTINUOUS per-chapter paratext score p(c) ∈ [0,1] = sigmoid((gap - (med + k·MAD)) / MAD) — a smooth
    version of the hard outlier flag (p=0.5 at the old threshold). No boolean; drives the soft-EM."""
    gaps, *_ = _chapter_gaps(X, rows, cloud, labels, n_ch, min_size)
    p = np.zeros(n_ch)
    if len(gaps) < 4:
        return p
    g = np.array(list(gaps.values()))
    med = float(np.median(g)); mad = float(np.median(np.abs(g - med))) * 1.4826 + 1e-9
    for c, v in gaps.items():
        p[c] = 1.0 / (1.0 + np.exp(-(v - (med + k_mad * mad)) / mad))
    return p


def _detect_paratext(X, rows, cloud, labels, n_ch, min_size=5, k_mad=3.0):
    """Hard flag: chapters whose gap is a high robust-outlier (> median + k·MAD). Returns (flagged, KW, CP,
    cpok). Used by the dissolution + the refuted term-suppression; the soft-EM uses `_paratext_scores`."""
    gaps, KW, CP, cpok = _chapter_gaps(X, rows, cloud, labels, n_ch, min_size)
    if len(gaps) < 4:
        LOGGER.info("paratext filter: too few assessable chapters"); return [], KW, CP, cpok
    g = np.array(list(gaps.values()))
    med = float(np.median(g)); mad = float(np.median(np.abs(g - med))) * 1.4826 + 1e-9
    thresh = med + k_mad * mad
    flagged = [c for c, v in gaps.items() if v > thresh]
    for c in flagged:
        LOGGER.info("paratext: ch %d gap=%.3f > med+%.0f·MAD=%.3f", c, gaps[c], k_mad, thresh)
    if not flagged:
        LOGGER.info("paratext: none flagged (max gap %.3f <= thresh %.3f)", float(g.max()), thresh)
    return flagged, KW, CP, cpok


def _paratext_suppression(X, rows, labels, flagged, top_m=40):
    """UPSTREAM correction (vs dissolution): return a per-term multiplicative weight that ZEROES the
    distinctive (high-lift) terms of the flagged paratext chapters — the venue boilerplate, identified
    data-drivenly as the terms that characterize a paratext chapter. Re-clustering with these terms removed
    lets the paratext works re-home by their TOPICAL vocabulary, freeing the slot for a real chapter — so
    the count is preserved. 'Down-weight words acting paratextually, in the context that reveals them.'"""
    labels = np.asarray(labels)
    N = len(rows)
    Xb = (X[rows] > 0).astype(np.float64)
    bg = np.asarray(Xb.sum(0)).ravel() / N
    sup = np.ones(Xb.shape[1])
    killed = set()
    for c in flagged:
        mem = np.where(labels == c)[0]
        cnt = np.asarray(Xb[mem].sum(0)).ravel()
        lift = (cnt / max(len(mem), 1)) / (bg + 1e-9)
        ok = np.where(cnt >= 2)[0]
        top = ok[np.argsort(lift[ok])[::-1][:top_m]]
        sup[top] = 0.0; killed.update(top.tolist())
    LOGGER.info("paratext suppression: zeroed %d distinctive terms of %d flagged chapter(s)",
                len(killed), len(flagged))
    return sup


def _content_then_attach(X, rows, cloud, Z, labels, n_ch, Xc, idf_c, max_iter=10):
    """ITERATED, count-preserving. Paratext is MULTI-POPULATION, so PEEL it off one pass at a time —
    re-cluster the current content pool into n_ch, set aside any flagged venue chapter's WORKS (removing
    works is stable: a peeled work can't reconstitute, unlike a zeroed term), and repeat until the
    gap-detector flags nothing (the natural stopping condition; converges since the pool shrinks
    monotonically over finitely many venue populations). Then attach all peeled works by their DE-
    CONTAMINATED topical content: the peeled set being topically diverse, its SHARED terms (high within-set
    doc-freq) ARE the author/venue contaminants — down-weight each term by that frequency so each work is
    re-represented topically, and attach to the nearest content chapter. No global classifier, no blocklist."""
    rows = list(rows); N = len(rows)
    pool = np.arange(N)                                   # positions (into rows/Z) still treated as content
    peeled = []
    for _pass in range(max_iter):
        ncl = min(n_ch, len(pool) - 1)
        cl = AgglomerativeClustering(n_clusters=ncl, metric="cosine", linkage="average").fit_predict(Z[pool])
        flagged, *_ = _detect_paratext(X, [rows[p] for p in pool], cloud, cl, n_ch)
        if not flagged:
            break                                         # natural stopping condition
        venue_local = np.isin(cl, flagged)
        peeled += pool[venue_local].tolist(); pool = pool[~venue_local]
        LOGGER.info("content-then-attach pass %d: peeled %d works (%d venue chapter(s)); content pool=%d",
                    _pass + 1, int(venue_local.sum()), len(flagged), len(pool))
    cl = AgglomerativeClustering(n_clusters=min(n_ch, len(pool) - 1), metric="cosine",
                                 linkage="average").fit_predict(Z[pool])       # final clean content clustering
    present = sorted(set(cl.tolist())); remap = {c: i for i, c in enumerate(present)}
    out = np.full(N, -1); out[pool] = [present[c] for c in cl]
    if peeled:
        vidx = np.array(peeled)
        vdf = np.asarray((Xc[vidx] > 0).sum(0)).ravel() / len(vidx)            # within-peeled doc-freq = contaminant
        KWtop = normalize(Xc.multiply(idf_c * (1.0 - vdf)).tocsr())            # suppress shared terms per work
        Ind = sp.csr_matrix((np.ones(len(cl)), ([remap[c] for c in cl], np.arange(len(pool)))),
                            shape=(len(present), len(pool)))
        cent = normalize(Ind @ KWtop[pool])
        out[vidx] = [present[a] for a in np.asarray((KWtop[vidx] @ cent.T).todense()).argmax(1)]
        LOGGER.info("content-then-attach: %d content -> %d chapters; %d peeled works re-homed by topical "
                    "content (suppressed %d shared contaminants, vdf>0.3)", len(pool), len(present),
                    len(vidx), int((vdf > 0.3).sum()))
    return out


def _dissolve_paratext(X, rows, cloud, labels, n_ch, min_size=5, k_mad=3.0):
    """DISSOLVE flagged paratext chapters: redistribute their works to the nearest surviving chapter by
    citation-coupling (keyword fallback for orphans). Reduces the chapter count by the number dissolved;
    see `_paratext_suppression` + ``--suppress-paratext`` for the count-preserving upstream alternative."""
    labels = np.asarray(labels).copy()
    flagged, KW, CP, cpok = _detect_paratext(X, rows, cloud, labels, n_ch, min_size, k_mad)
    if not flagged:
        return labels
    N = len(rows)
    survive = np.array([c for c in range(n_ch) if c not in flagged and (labels == c).sum() > 0])

    def cents(V, restrict):
        r, col = [], []
        for i, c in enumerate(survive):
            mem = np.where(labels == c)[0]
            if restrict is not None:
                mem = mem[restrict[mem]]
            r += [i] * len(mem); col += mem.tolist()
        Ind = sp.csr_matrix((np.ones(len(r)), (r, col)), shape=(len(survive), N))
        return normalize(Ind @ V)

    CPc, KWc = cents(CP, cpok), cents(KW, None)
    moved = np.where(np.isin(labels, flagged))[0]
    for w in moved:
        sims = (CP[w] @ CPc.T) if cpok[w] else (KW[w] @ KWc.T)
        labels[w] = survive[int(np.asarray(sims.todense()).ravel().argmax())]
    LOGGER.info("paratext: dissolved %d chapter(s), redistributed %d works -> %d surviving",
                len(flagged), len(moved), len(survive))
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--lsa-dim", type=int, default=200, help="TruncatedSVD dims for the TF-IDF view")
    ap.add_argument("--cca-reg", type=float, default=1e-3, help="CCA ridge (relative to unit-scaled covariances)")
    ap.add_argument("--cca-k", type=int, default=0, help="# canonical directions to keep (0 = auto by elbow)")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--decontaminate", action="store_true",
                    help="the bipartite-EDGE 'meet': down-weight each (work, term) keyword cell by the "
                         "embedder relevance of the term to the work, so off-topic reference-string/venue "
                         "terms are suppressed INSIDE each work (kept where they ARE the topic). Surgical "
                         "de-contamination at the source — no detection/threshold/work-loss.")
    ap.add_argument("--label-filter", action="store_true",
                    help="re-rank each chapter's keyword label by lift × max(contrastive embedder relevance, "
                         "0) — a term must DISCRIMINATE this chapter (closer to its embedding centroid than "
                         "to any other's). Surfaces canonical topical phrasings, drops field-generic & stray "
                         "paratext terms. SAFE (only the displayed label changes; no work touched).")
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--view", choices=["embedding", "citation", "both"], default="embedding",
                    help="the robust 2nd view paired with keyword-TFIDF in the CCA: embedding (GIST), "
                         "citation (citation-similarity spectral features — embedder-free, but sparser → "
                         "weaker language-collapse for under-cited/foreign works), or both (concatenated).")
    ap.add_argument("--gcca-variant", choices=["maxvar", "sumcor"], default="maxvar",
                    help="[gcca] MAXVAR = top eigvecs of the summed per-view projectors (closed-form GEP, "
                         "global, self-blocks included); SUMCOR = top eigvecs of the OFF-diagonal cross-view "
                         "blocks (pairwise-correlation; linear relaxation of the non-convex JAD form).")
    ap.add_argument("--method",
                    choices=["cca", "discardclash", "citation_only", "concat_vocab", "link_plsa", "gcca"],
                    default="cca",
                    help="cca = retain the shared subspace (CCA). discardclash = keep the full keyword "
                         "space but REMOVE its top-k directions that are UNexplained by citations (= the "
                         "dominant citation-uncorrelated keyword variance ≈ language); embedder-free, "
                         "and right when the views are too independent for CCA-retain (small shared subspace). "
                         "citation_only = cluster on the citation view ALONE (no keywords/embeddings — the "
                         "'use one or the other' collaborative-filtering view).")
    ap.add_argument("--citation-source", choices=["full_coupling", "internal_ruzicka"], default="full_coupling",
                    help="full_coupling = bibliographic coupling over the FULL out-citation profile (IDF-"
                         "weighted, recency-clean; R²=0.154 vs keywords). internal_ruzicka = the refuted "
                         "near-identity feature on the internal subgraph (~1 edge/work; kept to reproduce "
                         "the dead-end).")
    ap.add_argument("--discard-k", type=int, default=6,
                    help="[discardclash] number of dominant citation-unexplained keyword directions to remove")
    ap.add_argument("--concat-alpha", type=float, default=0.5,
                    help="[concat_vocab] energy split terms:refs = alpha:(1-alpha) (each block L2-normed per "
                         "row first, so the ~120-terms vs ~8-refs count imbalance doesn't decide it). 0.5=equal.")
    ap.add_argument("--channel-weight", type=float, default=1.0,
                    help="[link_plsa] refs channel mass relative to terms (refs auto-scaled to equal total "
                         "mass first, then ×this). 1.0=equal-mass channels; >1 favors references.")
    ap.add_argument("--dissolve-paratext", action="store_true",
                    help="detect venue/paratext chapters (lexically coherent but NOT coupling-coherent) and "
                         "DISSOLVE them — redistribute their works to the nearest surviving chapter by "
                         "citation-coupling. Reduces the chapter count. No blocklist.")
    ap.add_argument("--suppress-paratext", action="store_true",
                    help="[cca] REFUTED (whack-a-mole): zero a paratext chapter's venue terms + re-cluster; "
                         "the venue cluster reconstitutes on the term register's long tail. Use --soft-paratext.")
    ap.add_argument("--content-then-attach", action="store_true",
                    help="[cca] count-preserving: detect the venue chapter, re-cluster the CONTENT works into "
                         "n (freed slot -> real topic), then attach each venue work by its TOPICAL content "
                         "with the venue cluster's SHARED terms (author/venue contaminants) down-weighted.")
    ap.add_argument("--soft-paratext", action="store_true",
                    help="[cca] CONTINUOUS, count-preserving soft-EM: a smooth per-chapter paratext score "
                         "(sigmoid of the lexical-vs-coupling gap) shrinks each work's keyword contribution "
                         "by its score while keeping its citation contribution, so venue works cluster by "
                         "what they CITE (their real topic). No flag/threshold/blocklist; saves a per-work "
                         "paratext weight (paratext_work_weights.json) for pipeline-wide reuse.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="cca", log_level="INFO", include_time=False)

    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}

    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]; ids = [w for w in ids if w in id2idx]
    N = len(rows)
    # T view: concept-internal TF-IDF -> LSA
    Xc = X[rows].astype(np.float64)
    idf_c = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))   # IDF on original presence
    if args.decontaminate:
        Xc = _decontaminate(Xc, rows, cloud, vocab, cfg.experiment.cloud.embedder.model_name)
    Tw = normalize(Xc.multiply(idf_c).tocsr())
    T = TruncatedSVD(n_components=min(args.lsa_dim, N - 1, Tw.shape[1] - 1), random_state=0).fit_transform(Tw)
    dim = min(args.lsa_dim, N - 1)

    def _coupling_features():
        # Bibliographic coupling over the FULL out-citation profile (not the starved internal subgraph).
        # Strip the PageRank self-loops (they spuriously couple a work with its own citers); IDF-weight the
        # cited-work columns (down-weight megahubs everyone cites); L2-normalize rows; LSA — mirrors T.
        G = cloud.citation_graph(False).tocsr().astype(np.float64)
        G.setdiag(0.0); G.eliminate_zeros()
        Xout = G[rows]
        col_df = np.asarray((Xout > 0).sum(0)).ravel(); keep = col_df > 0
        Xout = Xout[:, keep]; idf = np.log(N / (col_df[keep] + 1.0))
        Xw = normalize(Xout.multiply(idf).tocsr())
        LOGGER.info("full-coupling: %d works x %d cited targets (mean out-refs=%.1f)",
                    Xout.shape[0], Xout.shape[1], np.diff(Xout.indptr).mean())
        return normalize(TruncatedSVD(n_components=dim, random_state=0).fit_transform(Xw))

    def _citation_features():
        if args.citation_source == "full_coupling":
            return _coupling_features()
        from artifacts_scripts.curriculum.multi_view_spectral import build_citation_similarity_view
        Csim = build_citation_similarity_view(cloud.citation_graph(False)[rows][:, rows], csls_k=10)
        return normalize(TruncatedSVD(n_components=dim, random_state=0).fit_transform(Csim))

    labels = None                       # Z-clustering methods set Z below; link_plsa sets labels directly
    if args.method == "link_plsa":
        # GENERATIVE point on the concat↔CCA↔generative spectrum: Link-pLSA (KL-NMF) — each work's topic
        # mixture emits BOTH its terms and its references through SEPARATE channels; topics ARE the
        # chapters (argmax of the doc-topic matrix), no whitening / no separate clustering. Per-channel
        # mass-balancing handles the ~120-terms vs ~8-refs imbalance generatively (no α). KL-NMF is the MLE
        # of this two-channel pLSA; full Bayesian Link-LDA (Gibbs) is the further step. (RTM-proper models
        # doc-doc links → would hit the ~1-edge/work internal sparsity; references-as-tokens dodges it.)
        from sklearn.decomposition import NMF
        Gc = cloud.citation_graph(False).tocsr().astype(np.float64); Gc.setdiag(0.0); Gc.eliminate_zeros()
        Wt = (X[rows] > 0).astype(np.float64); Wt = Wt[:, np.asarray(Wt.sum(0)).ravel() > 0]   # terms channel
        Rt = Gc[rows]; Rt = Rt[:, np.asarray((Rt > 0).sum(0)).ravel() > 0]                      # refs channel
        bal = (Wt.sum() / max(Rt.sum(), 1.0)) * args.channel_weight                             # equal-mass ×cw
        M = sp.hstack([Wt, bal * Rt]).tocsr()
        nmf = NMF(n_components=args.n, beta_loss="kullback-leibler", solver="mu",
                  init="nndsvda", max_iter=300, random_state=0)
        Wdt = nmf.fit_transform(M)
        labels = Wdt.argmax(1)
        tag = f"link_plsa(cw={args.channel_weight})"
        LOGGER.info("link-pLSA: M %s [terms %d|refs %d], K=%d -> %d topics used, recon err=%.1f",
                    M.shape, Wt.shape[1], Rt.shape[1], args.n, len(set(labels.tolist())), nmf.reconstruction_err_)
    elif args.method == "concat_vocab":
        # "A reference is just another word": ONE incidence matrix over the combined vocabulary
        # (terms ⊕ cited-works), single LSA — no two-view CCA machinery. Each block is L2-normalized per
        # row then scaled by sqrt(alpha)/sqrt(1-alpha) so they share energy alpha:(1-alpha) regardless of
        # the ~120-terms vs ~8-refs count imbalance. Terms cover every work → 100% coverage (the uncited ⅓
        # keep their term block; no zero rows, unlike citation_only). Tests the unification directly: if a
        # single SVD here matches CCA-coupling, the whitening/agreement-selection wasn't load-bearing.
        Gc = cloud.citation_graph(False).tocsr().astype(np.float64); Gc.setdiag(0.0); Gc.eliminate_zeros()
        Xout = Gc[rows]; cdf = np.asarray((Xout > 0).sum(0)).ravel(); keep = cdf > 0
        Gg = normalize((Xout[:, keep]).multiply(np.log(N / (cdf[keep] + 1.0))).tocsr())
        a = args.concat_alpha
        M = sp.hstack([np.sqrt(a) * Tw, np.sqrt(1.0 - a) * Gg]).tocsr()
        Z = normalize(TruncatedSVD(n_components=dim, random_state=0).fit_transform(M))
        tag = f"concat_vocab(a={a})"
        LOGGER.info("concat-vocab: [terms %d | refs %d] alpha=%.2f -> M %s -> Z %s",
                    Tw.shape[1], Gg.shape[1], a, M.shape, Z.shape)
    elif args.method == "citation_only":
        # Cluster on the citation view ALONE — no keywords, no embedder. The "use one or the other"
        # collaborative view; viable now that full coupling has real structure (erank 5544, partic 1272).
        # Works with NO in-field out-citations have an all-zero coupling row (no collaborative signal) —
        # the honest cost of a citation-only view: drop them (keywords/embeddings could place them, this
        # can't). rows/ids are subset in lockstep so all downstream indexing stays consistent.
        Z = _citation_features()
        nz = np.linalg.norm(Z, axis=1) > 1e-9
        if not nz.all():
            LOGGER.info("citation-only: dropping %d/%d works with no coupling signal (%.1f%% coverage)",
                        int((~nz).sum()), len(nz), 100 * nz.mean())
            rows = [rows[i] for i in range(len(rows)) if nz[i]]
            ids = [ids[i] for i in range(len(ids)) if nz[i]]
            Z = Z[nz]
        tag = f"citation-only:{args.citation_source}"
        LOGGER.info("citation-only clustering on %s features: Z %s", args.citation_source, Z.shape)
    elif args.method == "discardclash":
        # Keep the full keyword space, REMOVE only its top-k directions unexplained by citations.
        # T_res = keyword variance not predictable from citations; its dominant PCs ≈ language (a big
        # citation-uncorrelated lexical split). Project T orthogonal to them. Embedder-free; symmetric
        # to CCA (CCA retains the shared, this removes the clash). See backlog.
        C = _citation_features()
        Tc = T - T.mean(0)
        coef, *_ = np.linalg.lstsq(C, T, rcond=None)
        T_res = T - C @ coef
        Tr = T_res - T_res.mean(0)
        # DIAGNOSTIC: how much keyword variance can citations actually explain? Near-0 ⇒ T_res≈T ⇒ the
        # "top residual dirs" are just T's dominant CONTENT axes, not a language clash (independence bites).
        r2 = 1.0 - float((Tr ** 2).sum() / (Tc ** 2).sum())
        _, sv, Vt = np.linalg.svd(Tr, full_matrices=False)
        _, _, VtT = np.linalg.svd(Tc, full_matrices=False)
        align = float((np.abs(Vt[:args.discard_k] @ VtT[:args.discard_k].T) ** 2).sum() / args.discard_k)
        Vlang = Vt[:args.discard_k]                              # (k, d_T) the "language" directions (claim)
        frac = float((sv[:args.discard_k] ** 2).sum() / (sv ** 2).sum())
        Z = normalize(T - (T @ Vlang.T) @ Vlang)                # keyword space minus those dirs
        LOGGER.info("discard-clash: removed top-%d dirs (%.1f%% of residual var) -> Z %s | citation R²=%.3f "
                    "(keyword var explained by citations); top-k residual-vs-content subspace overlap=%.3f",
                    args.discard_k, 100 * frac, Z.shape, r2, align)
        tag = f"discardclash k={args.discard_k}:{args.citation_source}"
    elif args.method == "gcca":
        # Multi-view CONSENSUS over keyword + embedding + coupling — BRANCHLESS (no heterogeneity threshold;
        # each view's influence is set by its agreement with the others, read off the eigenvalues).
        # Generalizes homogeneously to N views. MAXVAR = top eigvecs of the summed per-view projectors
        # (Σ Uᵢ Uᵢᵀ, computed via the small Gram + lift; closed-form, global). SUMCOR = top eigvecs of the
        # OFF-diagonal cross-view blocks only (pairwise correlation; the genuine constrained form is the
        # non-convex JAD, this is its linear relaxation).
        EMb = normalize(np.asarray(cloud.embeddings(), dtype=np.float64)[rows])
        Us = []
        for V in [T, EMb, _citation_features()]:
            Vc = V - V.mean(0)
            Uu, ss, _ = np.linalg.svd(Vc, full_matrices=False)
            Us.append(Uu[:, : max(2, int((ss > ss[0] * 1e-3).sum()))])
        sizes = [U.shape[1] for U in Us]
        Ucat = np.concatenate(Us, axis=1)
        Bgram = Ucat.T @ Ucat                                # block Gram; diagonal blocks = I (orthonormal)
        if args.gcca_variant == "sumcor":
            Boff = Bgram.copy(); s = 0
            for r in sizes:
                Boff[s:s + r, s:s + r] = 0.0; s += r         # zero self-blocks -> pairwise cross-view only
            evals, Vv = np.linalg.eigh(Boff)
        else:
            evals, Vv = np.linalg.eigh(Bgram)                # MAXVAR: self-projectors included
        order = np.argsort(evals)[::-1]
        k = max(2, min(2 * args.n, Bgram.shape[0]))
        Z = normalize(Ucat @ Vv[:, order[:k]])               # lift to the consensus embedding, cluster on it
        tag = f"gcca-{args.gcca_variant}"
        LOGGER.info("GCCA-%s: view dims %s, top eigenvalues %s (max agreement=%d views), k=%d",
                    args.gcca_variant, sizes, np.round(evals[order[:6]], 3).tolist(), len(Us), k)
    else:
        # CCA: retain the shared subspace of keyword-TFIDF and the robust 2nd view R.
        parts = []
        if args.view in ("embedding", "both"):
            parts.append(normalize(np.asarray(cloud.embeddings(), dtype=np.float32)[rows].astype(np.float64)))
        if args.view in ("citation", "both"):
            parts.append(_citation_features())
        R = parts[0] if len(parts) == 1 else normalize(np.concatenate(parts, axis=1))
        LOGGER.info("CCA views: R(%s) %s, T(LSA) %s; n=%d", args.view, R.shape, T.shape, N)

        def _cca_Z(Tloc):                                     # R is fixed; only the keyword view changes
            rho, Ea, Tb = _rcca(R, Tloc, args.cca_reg)
            if args.cca_k > 0:
                kk = args.cca_k
            else:
                d = rho[:-1] - rho[1:]
                kk = int(np.argmax(d[3:]) + 4) if len(d) > 4 else len(rho)
                kk = max(kk, int((rho > 0.3).sum()))
            kk = max(2, min(kk, Ea.shape[1]))
            return normalize(0.5 * (Ea[:, :kk] + Tb[:, :kk]) * rho[:kk]), Ea, Tb, rho, kk

        Z, Ea, Tb, rho, k = _cca_Z(T)
        LOGGER.info("canonical correlations rho (top 30): %s | keeping k=%d", np.round(rho[:30], 3).tolist(), k)
        tag = f"cca({args.view}:{args.citation_source}) k={k}" if args.view != "embedding" else f"cca(embedding) k={k}"
        if args.soft_paratext:
            # CONTINUOUS, count-preserving soft-EM. The paratext signal is a GROUP property (per-work/term
            # signals are all weak), so it stays cluster-mediated — but no boolean: a continuous per-chapter
            # score p(c)=sigmoid(gap) gives each work p_w, and we SHRINK its keyword (Tb) contribution by p_w
            # while keeping its citation (Ea) contribution, so a venue work clusters by what it CITES (its
            # real topic) instead of the venue terms it shares. p_w accumulates monotonically (sticky) ->
            # converges, no oscillation. Venue works disperse into real topics; the freed slot becomes a
            # real chapter (count preserved). p_w is saved for re-use as a pipeline-wide work weight.
            p_w = np.zeros(N)
            for _pass in range(6):
                Zp = normalize((Ea[:, :k] + (1.0 - p_w)[:, None] * Tb[:, :k]) * rho[:k])
                labels = AgglomerativeClustering(n_clusters=args.n, metric="cosine", linkage="average").fit_predict(Zp)
                pc = _paratext_scores(X, rows, cloud, labels, args.n)
                new_p_w = np.maximum(p_w, pc[labels])         # monotonic / sticky -> no oscillation
                LOGGER.info("soft-paratext pass %d: max chapter score=%.2f, works with p_w>0.5: %d",
                            _pass + 1, float(pc.max()), int((new_p_w > 0.5).sum()))
                if np.allclose(new_p_w, p_w, atol=1e-3):
                    break
                p_w = new_p_w
            (args.out).mkdir(parents=True, exist_ok=True)
            (args.out / "paratext_work_weights.json").write_text(
                json.dumps({ids[i]: float(p_w[i]) for i in range(N)}))
            tag = f"cca({args.view}:{args.citation_source}) k={k}+softpara"
        elif args.suppress_paratext:
            # UPSTREAM, count-preserving: cluster → detect paratext → zero its venue terms → re-CCA, iterated.
            sup = np.ones(Xc.shape[1])
            for _pass in range(5):
                labels = AgglomerativeClustering(n_clusters=args.n, metric="cosine", linkage="average").fit_predict(Z)
                flagged, *_ = _detect_paratext(X, rows, cloud, labels, args.n)
                if not flagged:
                    break
                LOGGER.info("suppress-paratext pass %d: %d paratext chapter(s) -> zero venue terms, re-CCA",
                            _pass + 1, len(flagged))
                sup = sup * _paratext_suppression(X, rows, labels, flagged)
                Tw_s = normalize(Xc.multiply(idf_c * sup).tocsr())
                T_s = TruncatedSVD(n_components=min(args.lsa_dim, N - 1, Tw_s.shape[1] - 1),
                                   random_state=0).fit_transform(Tw_s)
                Z, Ea, Tb, rho, k = _cca_Z(T_s)
            tag = f"cca({args.view}:{args.citation_source}) k={k}+suppress"
    if labels is None:
        labels = AgglomerativeClustering(n_clusters=args.n, metric="cosine", linkage="average").fit_predict(Z)
    if args.dissolve_paratext:
        labels = _dissolve_paratext(X, rows, cloud, labels, args.n)
    if args.content_then_attach:
        labels = _content_then_attach(X, rows, cloud, Z, labels, args.n, Xc, idf_c)

    BG = X.shape[0]

    def w2k(cl_rows, k_):
        cnt = np.asarray(X[cl_rows].sum(0)).ravel(); nz = cnt.nonzero()[0]; nS = len(cl_rows); out = {}
        for tid in nz:
            c = cnt[tid]
            lift = np.log(((c + _EPS) / (nS + _EPS)) / ((field_df[tid] + _EPS) / (BG + _EPS)))
            if lift > 0:
                out[int(tid)] = lift * np.log1p(c)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:k_])

    # chapter ORDERING by citation balance: a chapter that is net-CITED by the others is a
    # prerequisite → comes first (the one thing CCA-clustering lacked vs the embedding path).
    adj = cloud.citation_graph(False)[rows][:, rows].tocoo()
    net = np.zeros(args.n)
    for u, v in zip(adj.row, adj.col):                # u cites v → v is the prerequisite
        if labels[u] != labels[v]:
            net[labels[v]] += 1; net[labels[u]] -= 1
    args.out.mkdir(parents=True, exist_ok=True)
    clusters = [np.where(labels == c)[0] for c in np.argsort(net)[::-1]]    # net-cited (foundational) first
    clusters = [m for m in clusters if len(m) > 0]
    if args.label_filter:
        # CONTRASTIVE embedder label re-rank (recovers the embedder's discriminative judgement in the LABEL,
        # safely — no work touched). A term must be closer to THIS chapter's embedding centroid than to any
        # other's; field-generic / stray-paratext terms (~equidistant) drop, canonical topical phrasings rise.
        from sentence_transformers import SentenceTransformer
        EMB = normalize(np.asarray(cloud.embeddings(), dtype=np.float64))
        cents = normalize(np.vstack([EMB[[rows[m] for m in mem]].mean(0) for mem in clusters]))
        cand = [w2k([rows[m] for m in mem], args.top_k * 4) for mem in clusters]
        uniq = sorted({t for c in cand for t in c})
        st = SentenceTransformer(cfg.experiment.cloud.embedder.model_name, device="cpu")
        te = dict(zip(uniq, normalize(np.asarray(st.encode([vocab[t] for t in uniq]), dtype=np.float64))))
        labels_kw = []
        for ci, mk in enumerate(cand):
            terms = list(mk)
            sims = normalize(np.vstack([te[t] for t in terms])) @ cents.T          # (n_terms × n_clusters)
            own = sims[:, ci].copy(); sims[:, ci] = -1.0
            contr = own - sims.max(1)                                              # discriminative: this vs best other
            sc = {t: mk[t] * max(float(contr[j]), 0.0) for j, t in enumerate(terms)}
            labels_kw.append(dict(sorted(sc.items(), key=lambda x: -x[1])[:args.top_k]))
        LOGGER.info("label-filter: contrastive embedder re-rank of %d labels (%d unique candidate terms)",
                    len(clusters), len(uniq))
    else:
        labels_kw = [w2k([rows[m] for m in mem], args.top_k) for mem in clusters]
    twi = {}; kwj = {}; sm = [f"# {tag} curriculum — {args.concept_gap.parent.name}", ""]
    for new_i, members in enumerate(clusters):
        mk = labels_kw[new_i]
        twi[str(new_i)] = [ids[m] for m in members]; kwj[str(new_i)] = [[vocab[i], float(s)] for i, s in mk.items()]
        sm += [f"## Topic {new_i} ({len(members)} works)", "", ", ".join(f"`{vocab[i]}`" for i in list(mk)[:20]), ""]
    (args.out / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out / "topic_keywords.json").write_text(json.dumps(kwj))
    (args.out / "filtered_work_ids.json").write_text(json.dumps([w for m in clusters for w in (ids[x] for x in m)]))
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    (args.out / "summary.md").write_text("\n".join(sm) + f"\n---\n*Generated by `{cmd}`*\n")
    print(f"{tag} -> {args.n} topics (sizes {sorted((len(m) for m in clusters), reverse=True)}) -> {args.out}")
    for new_i in range(len(clusters)):
        print(f"  T{new_i}: " + ", ".join(vocab[i] for i in list(labels_kw[new_i])[:6]))


if __name__ == "__main__":
    main()
