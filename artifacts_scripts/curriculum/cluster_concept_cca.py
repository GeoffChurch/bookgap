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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--lsa-dim", type=int, default=200, help="TruncatedSVD dims for the TF-IDF view")
    ap.add_argument("--cca-reg", type=float, default=1e-3, help="CCA ridge (relative to unit-scaled covariances)")
    ap.add_argument("--cca-k", type=int, default=0, help="# canonical directions to keep (0 = auto by elbow)")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--view", choices=["embedding", "citation", "both"], default="embedding",
                    help="the robust 2nd view paired with keyword-TFIDF in the CCA: embedding (GIST), "
                         "citation (citation-similarity spectral features — embedder-free, but sparser → "
                         "weaker language-collapse for under-cited/foreign works), or both (concatenated).")
    ap.add_argument("--method", choices=["cca", "discardclash", "citation_only"], default="cca",
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
    idf_c = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))
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

    if args.method == "citation_only":
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
    else:
        # CCA: retain the shared subspace of keyword-TFIDF and the robust 2nd view R.
        parts = []
        if args.view in ("embedding", "both"):
            parts.append(normalize(np.asarray(cloud.embeddings(), dtype=np.float32)[rows].astype(np.float64)))
        if args.view in ("citation", "both"):
            parts.append(_citation_features())
        R = parts[0] if len(parts) == 1 else normalize(np.concatenate(parts, axis=1))
        LOGGER.info("CCA views: R(%s) %s, T(LSA) %s; n=%d", args.view, R.shape, T.shape, N)
        rho, Ea, Tb = _rcca(R, T, args.cca_reg)
        LOGGER.info("canonical correlations rho (top 30): %s", np.round(rho[:30], 3).tolist())
        if args.cca_k > 0:
            k = args.cca_k
        else:
            d = rho[:-1] - rho[1:]
            k = int(np.argmax(d[3:]) + 4) if len(d) > 4 else len(rho)
            k = max(k, int((rho > 0.3).sum()))
        k = max(2, min(k, Ea.shape[1]))
        LOGGER.info("keeping k=%d canonical directions (rho[k-1]=%.3f, rho[k]=%.3f); collapsing the rest",
                    k, rho[k - 1], rho[k] if k < len(rho) else 0.0)
        Z = normalize(0.5 * (Ea[:, :k] + Tb[:, :k]) * rho[:k])
        tag = f"cca({args.view}:{args.citation_source}) k={k}" if args.view != "embedding" else f"cca(embedding) k={k}"
    labels = AgglomerativeClustering(n_clusters=args.n, metric="cosine", linkage="average").fit_predict(Z)

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
    twi = {}; kwj = {}; sm = [f"# {tag} curriculum — {args.concept_gap.parent.name}", ""]
    for new_i, members in enumerate(clusters):
        cl_rows = [rows[m] for m in members]
        mk = w2k(cl_rows, args.top_k)
        twi[str(new_i)] = [ids[m] for m in members]; kwj[str(new_i)] = [[vocab[i], float(s)] for i, s in mk.items()]
        sm += [f"## Topic {new_i} ({len(members)} works)", "", ", ".join(f"`{vocab[i]}`" for i in list(mk)[:20]), ""]
    (args.out / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out / "topic_keywords.json").write_text(json.dumps(kwj))
    (args.out / "filtered_work_ids.json").write_text(json.dumps([w for m in clusters for w in (ids[x] for x in m)]))
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    (args.out / "summary.md").write_text("\n".join(sm) + f"\n---\n*Generated by `{cmd}`*\n")
    print(f"{tag} -> {args.n} topics (sizes {sorted((len(m) for m in clusters), reverse=True)}) -> {args.out}")
    for new_i, members in enumerate(clusters):
        print(f"  T{new_i}: " + ", ".join(vocab[i] for i in list(w2k([rows[m] for m in members], 6))[:6]))


if __name__ == "__main__":
    main()
