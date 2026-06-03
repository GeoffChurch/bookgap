"""Multi-view clustering WITH a keyword/term view — the balanced fix.

Increment 3 of docs/backlog/curriculum-keyword-coclustering.md. Keyword-only deflation
(gap_keyword_curriculum.py) maxed keyword-distinctiveness but tanked embedding cohesion.
The fix is to use keywords AS A VIEW alongside embedding + citation, not alone — building
three CSLS-corrected similarity views over the pool and combining them.

Two ``--method`` combiners:
  * ``coreg`` (default) — the *co-regularized* 3-view spectral solver
    (cluster_with_multi_view_spectral_views): each view keeps its own Laplacian and the
    views are coupled via the mean-projector agreement term, so a view only bends the
    others where they *agree*. This is the principled balance.
  * ``kernelsum`` — REFUTED baseline: sum the three affinities and graph-bisect. Combining
    by raw sum let the views conflict → negative silhouette (see backlog inc 3). Kept
    runnable for the head-to-head record.

Then writes a judge/intrinsic/correspondence-ready run-dir (keywords via log-lift, order
by citation balance).

Example:
  python -m artifacts_scripts.curriculum.gap_multiview_term \\
      --pool-run-dir artifacts/curriculum/neuromorphic_trackA_sem --field 17 --n 13 \\
      --method coreg --out-dir artifacts/curriculum/neuromorphic_trackA_mvt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.multi_view_spectral import (
    build_citation_similarity_view, build_embedding_similarity_view, csls_correct,
)
from bookgap.util import LOGGER, setup_logging

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or to in for on with is are be by we our this that as at from using".split())


def _terms(text):
    t = [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]
    return set(t) | {f"{t[i]}_{t[i+1]}" for i in range(len(t) - 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-run-dir", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--method", choices=["coreg", "kernelsum"], default="coreg",
                    help="coreg = co-regularized 3-view spectral (principled); "
                         "kernelsum = refuted sum+graph-bisect baseline")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="mv_term", log_level="INFO", include_time=False)

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    titles = cloud.work_titles(); abstracts = cloud.work_abstracts()
    E_all = np.asarray(cloud.embeddings(), dtype=np.float32)

    fids = json.loads((args.pool_run_dir / "filtered_work_ids.json").read_text())
    poolpos = [id2idx[w] for w in fids if w in id2idx]
    P = len(poolpos)
    mass = np.load(args.pool_run_dir / "leaf_masses.npy")[:P].astype(float) if (args.pool_run_dir / "leaf_masses.npy").exists() else np.ones(P)
    doc_terms = [_terms(f"{titles[i]} {abstracts[i] or ''}") for i in poolpos]

    # three CSLS-corrected views over the pool
    W_emb = build_embedding_similarity_view(E_all[poolpos], csls_k=args.csls_k)
    W_cite = build_citation_similarity_view(cloud.citation_graph(False)[poolpos][:, poolpos], csls_k=args.csls_k)
    vocab = {}; rows = []; cols = []
    for d, s in enumerate(doc_terms):
        for t in s:
            rows.append(d); cols.append(vocab.setdefault(t, len(vocab)))
    X = normalize(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(P, len(vocab))))
    W_term = csls_correct((X @ X.T).toarray(), k=args.csls_k, clip=True)
    LOGGER.info("views built (P=%d, vocab=%d); method=%s into %d", P, len(vocab), args.method, args.n)

    if args.method == "coreg":
        # Principled: co-regularize the three views (each keeps its own Laplacian;
        # coupled via mean-projector agreement, so views bend each other only where
        # they agree). The auto-tuned λ transfers from the validated 2-view case.
        from artifacts_scripts.curriculum.multi_view_spectral import (
            cluster_with_multi_view_spectral_views,
        )
        labels = cluster_with_multi_view_spectral_views(
            [W_emb, W_cite, W_term], n_clusters=args.n,
        )
    else:
        # Refuted baseline: sum the affinities + recursive-Fiedler bisection (forces
        # balanced cuts; plain spectral collapses to one giant cluster on this dense,
        # tight affinity). Kept for the head-to-head record.
        from artifacts_scripts.curriculum.build_curriculum import _cluster_with_graph_bisect
        W = W_emb + W_cite + W_term
        k = max(args.csls_k, 15)
        keep = np.argsort(W, axis=1)[:, -k:]
        Ws = np.zeros_like(W)
        np.put_along_axis(Ws, keep, np.take_along_axis(W, keep, axis=1), axis=1)
        Ws = np.maximum(Ws, Ws.T)
        labels = _cluster_with_graph_bisect(sp.csr_matrix(Ws), n_clusters=args.n)

    # df/inverted index for log-lift keywords
    df = {}; inv = {}
    for d, s in enumerate(doc_terms):
        for t in s:
            df[t] = df.get(t, 0) + 1; inv.setdefault(t, []).append(d)
    eps = 0.5

    def w2k(S, k=25):
        nS = len(S); cnt = {}
        for d in S:
            for t in doc_terms[d]:
                cnt[t] = cnt.get(t, 0) + 1
        lifts = {t: np.log(((c + eps) / (nS + eps)) / ((df[t] + eps) / (P + eps))) * np.log1p(c)
                 for t, c in cnt.items() if c > 1}
        lifts = {t: v for t, v in lifts.items() if v > 0}
        return dict(sorted(lifts.items(), key=lambda x: -x[1])[:k])

    clusters = [list(np.where(labels == c)[0]) for c in range(args.n)]
    # order by citation balance (net-cited first)
    adj = cloud.citation_graph(False)[poolpos][:, poolpos].tocoo()
    net = np.zeros(args.n)
    for u, v in zip(adj.row, adj.col):
        if labels[u] != labels[v]:
            net[labels[v]] += 1; net[labels[u]] -= 1
    order = list(np.argsort(net)[::-1])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    twi = {}; sm = [f"# Multi-view(+term) curriculum — {args.pool_run_dir.name}", ""]
    for new_i, ci in enumerate(order):
        S = sorted(clusters[ci], key=lambda d: -mass[d])
        twi[str(new_i)] = [fids[d] for d in S]
        kw = w2k(set(clusters[ci]))
        sm += [f"## Topic {new_i}", "", "### Top Keywords", "", "| Keyword | Score |", "| :-- | :-- |"]
        sm += [f"| {t} | {s:.3f} |" for t, s in kw.items()]
        sm += ["", "### Top Works", ""] + [f"#### {j+1}. {titles[poolpos[d]]}" for j, d in enumerate(S[:8])] + [""]
    (args.out_dir / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out_dir / "filtered_work_ids.json").write_text(json.dumps([fids[d] for S in clusters for d in S]))
    (args.out_dir / "summary.md").write_text("\n".join(sm))
    print(f"wrote {args.n}-chapter multi-view(+term) curriculum -> {args.out_dir} "
          f"(sizes: {sorted([len(c) for c in clusters], reverse=True)})")


if __name__ == "__main__":
    main()
