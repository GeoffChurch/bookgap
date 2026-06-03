"""Keyword-DEFINED curriculum via deflation of nonlinear keyword<->work basins.

Increment 2 of docs/backlog/curriculum-keyword-coclustering.md. The nonlinear
iteration (gap_keyword_iteration.py) gives one keyword-coherent basin per seed;
here we DEFLATE: pick the highest-mass remaining work, iterate to its basin, record
it as a chapter, remove its works, repeat -> N keyword-coherent chapters. Order them
by citation balance (prerequisites = net-cited come first). Writes a run-dir
(topic_work_ids.json + summary.md) that the judge / intrinsic-metrics /
chapter-correspondence all read, for a head-to-head vs the dendrogram curriculum.

Example:
  python -m artifacts_scripts.curriculum.gap_keyword_curriculum \\
      --pool-run-dir artifacts/curriculum/neuromorphic_trackA_sem --field 17 --n 13 \\
      --out-dir artifacts/curriculum/neuromorphic_trackA_kw
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
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
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--top-m", type=int, default=120)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="kw_curriculum", log_level="INFO", include_time=False)

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
    df = {}
    inv = {}
    for d, s in enumerate(doc_terms):
        for t in s:
            df[t] = df.get(t, 0) + 1
            inv.setdefault(t, []).append(d)
    eps = 0.5

    def w2k(S):
        nS = len(S); cnt = {}
        for d in S:
            for t in doc_terms[d]:
                cnt[t] = cnt.get(t, 0) + 1
        lifts = {}
        for t, c in cnt.items():
            lift = np.log(((c + eps) / (nS + eps)) / ((df[t] + eps) / (P + eps)))
            if lift > 0:
                lifts[t] = lift * np.log1p(c)
        return dict(sorted(lifts.items(), key=lambda x: -x[1])[: args.top_k])

    def k2w(kw, allowed):
        score = np.full(P, -1.0)
        for d in allowed:
            score[d] = 0.0
        for t, w in kw.items():
            for d in inv.get(t, []):
                if score[d] >= 0:
                    score[d] += w
        return set(np.argsort(score)[::-1][: args.top_m].tolist())

    # Deflation: one keyword-coherent basin per round.
    remaining = set(range(P))
    basins = []
    for r in range(args.n):
        if len(remaining) < args.top_m // 2:
            break
        seed = max(remaining, key=lambda d: mass[d])
        S = k2w(w2k({seed}), remaining)
        for _ in range(args.iters):
            S = k2w(w2k(S), remaining)
        basins.append(S)
        remaining -= S
        LOGGER.info("basin %d: %d works, top kw: %s", r, len(S), list(w2k(S))[:6])

    # Order basins by citation balance (net-cited first = prerequisites).
    adj = cloud.citation_graph(False)[poolpos][:, poolpos].tocoo()
    blab = np.full(P, -1)
    for bi, S in enumerate(basins):
        for d in S:
            blab[d] = bi
    nb = len(basins)
    net = np.zeros(nb)
    for u, v in zip(adj.row, adj.col):       # u cites v -> v is prerequisite
        if blab[u] >= 0 and blab[v] >= 0 and blab[u] != blab[v]:
            net[blab[v]] += 1; net[blab[u]] -= 1
    order = list(np.argsort(net)[::-1])      # most-net-cited (foundational) first

    # Write run-dir.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    twi = {}
    sm = [f"# Keyword-defined curriculum — {args.pool_run_dir.name}", ""]
    for new_i, bi in enumerate(order):
        S = sorted(basins[bi], key=lambda d: -mass[d])
        twi[str(new_i)] = [fids[d] for d in S]
        kw = w2k(set(basins[bi]))
        sm += [f"## Topic {new_i}", "", "### Top Keywords", "", "| Keyword | Score |", "| :-- | :-- |"]
        sm += [f"| {t} | {s:.3f} |" for t, s in kw.items()]
        sm += ["", "### Top Works", ""]
        sm += [f"#### {j+1}. {titles[poolpos[d]]}" for j, d in enumerate(S[:8])]
        sm += [""]
    (args.out_dir / "topic_work_ids.json").write_text(json.dumps(twi))
    (args.out_dir / "filtered_work_ids.json").write_text(json.dumps([fids[d] for S in basins for d in S]))
    (args.out_dir / "summary.md").write_text("\n".join(sm))
    print(f"wrote {len(basins)}-chapter keyword curriculum -> {args.out_dir}")


if __name__ == "__main__":
    main()
