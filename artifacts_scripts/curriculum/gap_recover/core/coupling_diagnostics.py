"""How aligned is a substrate's dependency graph with its text-embedding space?

Two intrinsic measurements over the generic gap-recover contract, used to explain WHY the
embedding-orphan-core plant-recover transfers in some substrates (Lean) and degenerates in others
(Python code) — see ``docs/code-gap-triangulation.md``:

1. **coupling** = mean over concepts C of
   ``cos(C, its operational citers) − cos(C, random foreground)``
   (positive = a concept's callers are embedding-nearer than random foreground).
2. **top-k composition** = fraction of a concept's ``k`` nearest foreground that are its OWN
   operational citers (high = the embedding neighbourhood is caller-dominated, so deleting callers
   moves the k-th nearest distance and an embedding orphan core forms).

CPU only; cosine == dot on the L2-normalised stored embeddings.

    python -m artifacts_scripts.curriculum.gap_recover.core.coupling_diagnostics \
        --data /tmp/lean_gap_new --label Lean
    python -m artifacts_scripts.curriculum.gap_recover.core.coupling_diagnostics \
        --data cache/code_gap/python/data --label Python
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .data import Data
from .kernels import knn_dists


def coupling(d: Data, rng: np.random.Generator, n_concepts: int, min_oper: int) -> dict:
    cand = np.where(d.is_B & (d.operational_users >= min_oper))[0]
    rng.shuffle(cand)
    cand = cand[:n_concepts]
    lifts, closer = [], []
    for c in cand:
        citers = d.subT.getrow(c).indices
        citers = citers[d.is_F[citers]]
        if citers.size < min_oper:
            continue
        if citers.size > 50:
            citers = rng.choice(citers, 50, replace=False)
        csim = d.emb[citers] @ d.emb[c]
        rsim = d.emb[rng.choice(d.F_idx, citers.size, replace=False)] @ d.emb[c]
        lifts.append(float(csim.mean() - rsim.mean()))
        closer.append(bool(csim.mean() > rsim.mean()))
    lifts = np.asarray(lifts)
    return {"n": int(lifts.size), "lift_mean": float(lifts.mean()),
            "lift_median": float(np.median(lifts)), "frac_closer": float(np.mean(closer))}


def topk_composition(d: Data, rng: np.random.Generator, n_concepts: int, min_oper: int,
                     k: int) -> dict:
    cand = np.where(d.is_B & (d.operational_users >= min_oper) & d.concept_pref)[0]
    rng.shuffle(cand)
    cand = cand[:n_concepts]
    fracs = []
    for c in cand:
        _, idx = knn_dists(d.emb[c:c + 1], d.Zf, k, return_idx=True)
        near_F = set(d.F_idx[idx[0]].tolist())
        citers = set(int(x) for x in d.subT.getrow(c).indices if d.is_F[x])
        fracs.append(len(near_F & citers) / k)
    fracs = np.asarray(fracs)
    return {"n": int(fracs.size), "k": k, "citer_frac_mean": float(fracs.mean()),
            "citer_frac_median": float(np.median(fracs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n-concepts", type=int, default=500)
    ap.add_argument("--min-oper", type=int, default=20, dest="min_oper")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Data(args.data)
    rng = np.random.default_rng(args.seed)
    cp = coupling(d, rng, args.n_concepts, min_oper=5)
    tk = topk_composition(d, rng, min(args.n_concepts, 200), args.min_oper, args.k)
    print(f"[{args.label or args.data}] nodes={d.N} B={int(d.is_B.sum())} F={int(d.is_F.sum())}")
    print(f"  coupling: citer−random cosine lift = {cp['lift_mean']:+.3f} "
          f"(median {cp['lift_median']:+.3f}, frac closer {cp['frac_closer']:.2f}, n={cp['n']})")
    print(f"  top-{tk['k']} composition: foreground that are C's own citers = "
          f"{tk['citer_frac_mean']:.2f} (median {tk['citer_frac_median']:.2f}, n={tk['n']})")


if __name__ == "__main__":
    sys.exit(main())
