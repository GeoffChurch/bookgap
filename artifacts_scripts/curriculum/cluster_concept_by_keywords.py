"""(b) Cluster a keyword-concept's works by concept-internal TF-IDF — no neural embedder.

docs/backlog/curriculum-keyword-concept-seeding.md. The embedder GROUPS diverse recent papers by their
generic framing (`natural_language_processing/tasks/papers`) into a "miscellaneous" cluster with a
generic label. Provocation: cluster from the text statistics we already have instead. Weight each
work's keyword vector by IDF computed *within the concept* (not the field): generic framing terms are
in nearly every concept work → IDF_concept ≈ 0 → down-weighted to nothing, so works that share ONLY
generic framing have no similarity left to co-cluster on, and scatter to their real subtopics. Cluster
the resulting (sparse, L2-normalized) TF-IDF vectors by agglomerative average-linkage on cosine.

Contrast with the embedding dendrogram + the keyword-coclustering dead-end: that used global/field
background and was judged on embedding-silhouette; this uses **concept-internal** IDF aimed at a
specific, isolated embedder failure (generic-framing co-clustering), so it's a fresh scoped test.
Evaluate with `topic_genericness.py` (does mean field-IDF rise / the misc cluster dissolve?) +
`gap_intrinsic_metrics.py`.

Example:
  python -m artifacts_scripts.curriculum.cluster_concept_by_keywords \\
      --concept-gap artifacts/book_gaps/jurafsky_concept_full/concept_gap.json --field 17 --n 25 \\
      --out artifacts/curriculum/jurafsky_trackA_kwcluster
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
from sklearn.preprocessing import normalize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.cluster_concept_by_keywords"
_EPS = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-gap", required=True, type=Path, help="concept_gap.json (has gap_article_ids)")
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=30, help="keywords per cluster")
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="kwcluster", log_level="INFO", include_time=False)

    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); titles = cloud.work_titles(); id2idx = {w: i for i, w in enumerate(wid)}

    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]
    ids = [w for w in ids if w in id2idx]
    Xc = X[rows].astype(np.float64)                                   # (n_concept, vocab) binary tf
    N = Xc.shape[0]
    df_c = np.asarray((Xc > 0).sum(0)).ravel()                        # concept-internal DF
    idf_c = np.log(N / (df_c + 1.0))                                  # generic-in-concept -> ~0
    TFIDF = normalize(Xc.multiply(idf_c).tocsr())                     # L2-normalized concept TF-IDF
    LOGGER.info("concept TF-IDF: %d works x %d vocab; clustering into %d (agglomerative, cosine)",
                N, Xc.shape[1], args.n)

    # cosine distance (1 - sim) via sparse dot; agglomerative average-linkage
    sim = (TFIDF @ TFIDF.T).toarray()
    np.clip(sim, -1.0, 1.0, out=sim)
    D = 1.0 - sim; np.fill_diagonal(D, 0.0)
    labels = AgglomerativeClustering(n_clusters=args.n, metric="precomputed",
                                     linkage="average").fit_predict(D)

    # re-extract per-cluster keywords via field-lift W2K (same as elsewhere)
    BG = X.shape[0]

    def w2k(cl_rows, k):
        cnt = np.asarray(X[cl_rows].sum(0)).ravel(); nz = cnt.nonzero()[0]; nS = len(cl_rows); out = {}
        for tid in nz:
            c = cnt[tid]
            lift = np.log(((c + _EPS) / (nS + _EPS)) / ((field_df[tid] + _EPS) / (BG + _EPS)))
            if lift > 0:
                out[int(tid)] = lift * np.log1p(c)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:k])

    args.out.mkdir(parents=True, exist_ok=True)
    twi = {}; kwj = {}; sizes = []
    sm = [f"# Keyword-TF-IDF clustered curriculum — {args.concept_gap.parent.name}", ""]
    clusters = [np.where(labels == c)[0] for c in range(args.n)]
    clusters.sort(key=lambda m: -len(m))
    for new_i, members in enumerate(clusters):
        cl_rows = [rows[m] for m in members]
        cl_ids = [ids[m] for m in members]
        mk = w2k(cl_rows, args.top_k)
        twi[str(new_i)] = cl_ids; kwj[str(new_i)] = [[vocab[i], float(s)] for i, s in mk.items()]
        sizes.append(len(members))
        sm += [f"## Topic {new_i} ({len(members)} works)", "",
               ", ".join(f"`{vocab[i]}`" for i in list(mk)[:20]), ""]
    (args.out / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out / "topic_keywords.json").write_text(json.dumps(kwj))
    (args.out / "filtered_work_ids.json").write_text(json.dumps([w for m in clusters for w in (ids[x] for x in m)]))
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    (args.out / "summary.md").write_text("\n".join(sm) + f"\n---\n*Generated by `{cmd}`*\n")
    print(f"clustered {N} works -> {args.n} topics (sizes {sorted(sizes, reverse=True)}) -> {args.out}")
    for new_i, members in enumerate(clusters[:args.n]):
        mk = w2k([rows[m] for m in members], 6)
        print(f"  T{new_i}: " + ", ".join(vocab[i] for i in list(mk)[:6]))


if __name__ == "__main__":
    main()
