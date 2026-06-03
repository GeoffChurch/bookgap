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
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="cca", log_level="INFO", include_time=False)

    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    Emb = np.asarray(cloud.embeddings(), dtype=np.float32)

    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]; ids = [w for w in ids if w in id2idx]
    N = len(rows)
    # T view: concept-internal TF-IDF -> LSA
    Xc = X[rows].astype(np.float64)
    idf_c = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))
    Tw = normalize(Xc.multiply(idf_c).tocsr())
    T = TruncatedSVD(n_components=min(args.lsa_dim, N - 1, Tw.shape[1] - 1), random_state=0).fit_transform(Tw)
    # E view
    E = normalize(Emb[rows].astype(np.float64))
    LOGGER.info("CCA views: E %s, T(LSA) %s; n=%d", E.shape, T.shape, N)

    rho, Ea, Tb = _rcca(E, T, args.cca_reg)
    LOGGER.info("canonical correlations rho (top 30): %s", np.round(rho[:30], 3).tolist())
    if args.cca_k > 0:
        k = args.cca_k
    else:
        # elbow: largest relative drop in rho (after at least a few dims), or rho>0.3
        d = rho[:-1] - rho[1:]
        k = int(np.argmax(d[3:]) + 4) if len(d) > 4 else len(rho)
        k = max(k, int((rho > 0.3).sum()))            # at least the clearly-correlated ones
    k = max(2, min(k, Ea.shape[1]))
    LOGGER.info("keeping k=%d canonical directions (rho[k-1]=%.3f, rho[k]=%.3f); collapsing the rest "
                "(= disagreement/language)", k, rho[k - 1], rho[k] if k < len(rho) else 0.0)

    # shared representation = rho-weighted consensus of the two canonical coordinate sets
    Z = normalize(0.5 * (Ea[:, :k] + Tb[:, :k]) * rho[:k])
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

    args.out.mkdir(parents=True, exist_ok=True)
    clusters = sorted((np.where(labels == c)[0] for c in range(args.n)), key=lambda m: -len(m))
    twi = {}; kwj = {}; sm = [f"# CCA-hybrid curriculum (k={k}) — {args.concept_gap.parent.name}", ""]
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
    print(f"CCA k={k} -> {args.n} topics (sizes {sorted((len(m) for m in clusters), reverse=True)}) -> {args.out}")
    for new_i, members in enumerate(clusters):
        print(f"  T{new_i}: " + ", ".join(vocab[i] for i in list(w2k([rows[m] for m in members], 6))[:6]))


if __name__ == "__main__":
    main()
