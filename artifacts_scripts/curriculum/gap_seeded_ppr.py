"""Seeded bipartite personalized-PageRank curriculum (the cheap unification).

A direction from docs/backlog/curriculum-keyword-coclustering.md. Build the works×keywords
bipartite graph and, from a gap-seed work, run personalized PageRank with RESTART to the
seed: ``r = (1-α) s + α M r`` on the bipartite adjacency. The restart is the anchor — the
recurring anti-diffusion lesson says plain diffusion washes the seed out, but the teleport
term pins each basin to its seed, so this LINEAR walk should preserve per-seed co-clusters
the way the nonlinear iteration (gap_keyword_iteration.py) did, but far cheaper (sparse
matvecs, no per-round keyword re-mining).

The stationary r over work-nodes ranks the basin; r over keyword-nodes gives its
characteristic keywords (a genuine *co-cluster*: works + keywords co-defined). We DEFLATE
to build a curriculum: highest-mass remaining work → PPR → top-m remaining works → record
→ remove → repeat. Order chapters by citation balance. Writes a judge / intrinsic /
correspondence-ready run-dir, identical in shape to gap_keyword_curriculum.py for a clean
head-to-head.

Example:
  python -m artifacts_scripts.curriculum.gap_seeded_ppr \\
      --pool-run-dir artifacts/curriculum/neuromorphic_trackA_sem --field 17 --n 13 \\
      --out-dir artifacts/curriculum/neuromorphic_trackA_ppr
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

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.util import LOGGER, setup_logging

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or to in for on with is are be by we our this that as at from using".split())


def _terms(text):
    t = [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]
    return set(t) | {f"{t[i]}_{t[i+1]}" for i in range(len(t) - 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-run-dir", required=True, type=Path, help="run-dir whose selected works form the pool")
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--top-k", type=int, default=25, help="keywords to report per chapter")
    ap.add_argument("--top-m", type=int, default=120, help="works per basin")
    ap.add_argument("--alpha", type=float, default=0.85, help="PPR damping (restart prob = 1-alpha)")
    ap.add_argument("--iters", type=int, default=100, help="power-iteration steps")
    ap.add_argument("--min-df", type=int, default=2, help="drop terms rarer than this (hapax pruning)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="seeded_ppr", log_level="INFO", include_time=False)

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    titles = cloud.work_titles(); abstracts = cloud.work_abstracts()

    fids = json.loads((args.pool_run_dir / "filtered_work_ids.json").read_text())
    poolpos = [id2idx[w] for w in fids if w in id2idx]
    P = len(poolpos)
    mass = np.load(args.pool_run_dir / "leaf_masses.npy")[: P].astype(float) if (args.pool_run_dir / "leaf_masses.npy").exists() else np.ones(P)
    doc_terms = [_terms(f"{titles[i]} {abstracts[i] or ''}") for i in poolpos]

    # df for hapax pruning + log-lift keyword summaries.
    df = {}
    for s in doc_terms:
        for t in s:
            df[t] = df.get(t, 0) + 1
    vocab = {t: j for j, t in enumerate(t for t in df if df[t] >= args.min_df)}
    eps = 0.5

    # Bipartite incidence B (P works x V keywords), binary presence.
    rows, cols = [], []
    for d, s in enumerate(doc_terms):
        for t in s:
            j = vocab.get(t)
            if j is not None:
                rows.append(d); cols.append(j)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(P, len(vocab)))

    # Symmetric bipartite adjacency A = [[0, B], [Bᵀ, 0]] over P+V nodes; column-stochastic
    # transition M = A · diag(1/deg). PPR: r = (1-α) s + α M r, restart s on the seed work.
    A = sp.bmat([[None, B], [B.T, None]], format="csc")
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    M = (A @ sp.diags(1.0 / deg)).tocsr()
    N = M.shape[0]
    LOGGER.info("bipartite graph: P=%d works, V=%d keywords, nnz(B)=%d", P, len(vocab), B.nnz)

    def ppr_works(seed_d):
        s = np.zeros(N); s[seed_d] = 1.0
        r = s.copy()
        for _ in range(args.iters):
            r = (1.0 - args.alpha) * s + args.alpha * (M @ r)
        return r[:P]  # mass over work-nodes

    def w2k(S, k):
        nS = len(S); cnt = {}
        for d in S:
            for t in doc_terms[d]:
                cnt[t] = cnt.get(t, 0) + 1
        lifts = {}
        for t, c in cnt.items():
            if c <= 1:
                continue
            lift = np.log(((c + eps) / (nS + eps)) / ((df[t] + eps) / (P + eps)))
            if lift > 0:
                lifts[t] = lift * np.log1p(c)
        return dict(sorted(lifts.items(), key=lambda x: -x[1])[:k])

    # Deflation: one PPR basin per round, ranking restricted to remaining works.
    remaining = set(range(P))
    basins = []
    for r in range(args.n):
        if len(remaining) < args.top_m // 2:
            break
        seed = max(remaining, key=lambda d: mass[d])
        score = ppr_works(seed)
        ranked = [d for d in np.argsort(score)[::-1] if d in remaining]
        S = set(ranked[: args.top_m])
        basins.append(S)
        remaining -= S
        LOGGER.info("basin %d (seed %s): %d works, top kw: %s",
                    r, titles[poolpos[seed]][:48], len(S), list(w2k(S, 6)))

    # Order basins by citation balance (net-cited first = prerequisites).
    adj = cloud.citation_graph(False)[poolpos][:, poolpos].tocoo()
    blab = np.full(P, -1)
    for bi, S in enumerate(basins):
        for d in S:
            blab[d] = bi
    net = np.zeros(len(basins))
    for u, v in zip(adj.row, adj.col):       # u cites v -> v is prerequisite
        if blab[u] >= 0 and blab[v] >= 0 and blab[u] != blab[v]:
            net[blab[v]] += 1; net[blab[u]] -= 1
    order = list(np.argsort(net)[::-1])

    # Write run-dir (same shape as gap_keyword_curriculum.py).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    twi = {}
    sm = [f"# Seeded bipartite-PPR curriculum — {args.pool_run_dir.name}", ""]
    for new_i, bi in enumerate(order):
        S = sorted(basins[bi], key=lambda d: -mass[d])
        twi[str(new_i)] = [fids[d] for d in S]
        kw = w2k(set(basins[bi]), args.top_k)
        sm += [f"## Topic {new_i}", "", "### Top Keywords", "", "| Keyword | Score |", "| :-- | :-- |"]
        sm += [f"| {t} | {s:.3f} |" for t, s in kw.items()]
        sm += ["", "### Top Works", ""]
        sm += [f"#### {j+1}. {titles[poolpos[d]]}" for j, d in enumerate(S[:8])]
        sm += [""]
    (args.out_dir / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out_dir / "filtered_work_ids.json").write_text(json.dumps([fids[d] for S in basins for d in S]))
    (args.out_dir / "summary.md").write_text("\n".join(sm))
    print(f"wrote {len(basins)}-chapter seeded-PPR curriculum -> {args.out_dir} "
          f"(alpha={args.alpha}, top_m={args.top_m})")


if __name__ == "__main__":
    main()
