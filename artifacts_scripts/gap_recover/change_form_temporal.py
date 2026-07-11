"""Change-form temporal gap test: do WIDENING gaps (ΔM1 across epochs) forecast arrivals
where STATIC gaps don't? Every static snapshot->arrival test is a universal negative
(docs/code-gap-triangulation.md); this is the one shot at a positive temporal signal.

Reuses the temporal-holdout helpers + the epoch births.npz. For candidates present at the
earlier epoch `prev`, compares three rankers on the SAME growth-matched null:
  random      — sanity (fill-ratio ~1, stratAUC ~0.5);
  static_M1   — M1 at T (the known negative);
  change_dM1  — ΔM1 = M1(T) - M1(prev)  (a gap that is actively WIDENING).
Metrics: fill-ratio (top-decile attach / matched-null attach, clustered bootstrap) and
within-(operational,degree)-stratum AUC that the ranker separates attach>0.

    python -m artifacts_scripts.gap_recover.change_form_temporal \\
        --data cache/code_gap/python/data --T 3 --prev 1 --label Python
"""
from __future__ import annotations

import argparse

import numpy as np

from artifacts_scripts.gap_recover.core.temporal_holdout import (
    Data,
    arrival_attachment,
    clustered_bootstrap_ratio,
    growth_matched_null,
    m1_at_T,
    stratified_auc,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--prev", type=int, default=1)
    ap.add_argument("--topfrac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    data = Data(a.data).load_births(a.data)
    m1_T, su_T, ou_T = m1_at_T(data, a.T)
    m1_P, _, _ = m1_at_T(data, a.prev)
    dM1 = m1_T - m1_P                         # >0 == the gap widened between prev and T
    deg_T = su_T + ou_T
    cit, _ = arrival_attachment(data, a.T)    # new operational citers born after T
    att = cit.astype(float)

    cand = np.where((data.birth <= a.prev) & data.is_B & (deg_T >= 1))[0]
    if cand.size < 20:
        print(f"[{a.label or a.data}] too few candidates ({cand.size})"); return
    n_top = max(int(round(cand.size * a.topfrac)), 1)
    label_pos = att[cand] > 0
    c1 = ou_T[cand].astype(float); c2 = deg_T[cand].astype(float)
    print(f"[{a.label or a.data}] T=e{a.T} prev=e{a.prev}  cand={cand.size} n_top={n_top} "
          f"arrivals_frac={label_pos.mean():.2f}", flush=True)

    rankers = {
        "random    ": rng.standard_normal(cand.size),
        "static_M1 ": m1_T[cand],
        "change_dM1": dM1[cand],
    }
    for name, score_c in rankers.items():
        top = cand[np.argsort(score_c)[::-1][:n_top]]
        pool = np.setdiff1d(cand, top)
        null_idx = growth_matched_null(rng, ou_T, deg_T, top, pool)
        ratio, lo, hi = clustered_bootstrap_ratio(
            att[top], data.mods[top], att[null_idx], data.mods[null_idx], rng, B=a.boot)
        sauc = stratified_auc(score_c, label_pos, c1, c2, 10)
        print(f"   {name}: fill-ratio={ratio:.2f} CI[{lo:.2f},{hi:.2f}]  stratAUC={sauc:.3f}  "
              f"top-attach-frac={(att[top] > 0).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
