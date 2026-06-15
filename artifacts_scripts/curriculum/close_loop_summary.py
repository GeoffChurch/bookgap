#!/usr/bin/env python3
"""Aggregate the close-the-loop ablation into one table.

Reads every cl_<slug>_metrics.json (co-primaries) + cl_<slug>_recon.json (A1's
absolute reconstruction: ARI/NMI/purity) and prints a per-book + pooled summary:

  reconstruction (placement)  A1 coverage vs A0 floor  -> A1-A0 delta
  prereq-violation (ordering) A1-frontier vs A2-good-edge vs random
  tau-to-human   (ordering)   A1-frontier vs A2-good-edge

Pooled deltas use a simple sign test across books (the panel is small and the eval
is the weak link — we report direction + per-book spread, not a t-test).

  .conda/bin/python -m artifacts_scripts.curriculum.close_loop_summary \
      --glob 'artifacts/book_gaps/cl_*_metrics.json' --out artifacts/book_gaps/cl_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="artifacts/book_gaps/cl_*_metrics.json")
    ap.add_argument("--out", type=Path, default=Path("artifacts/book_gaps/cl_summary.json"))
    args = ap.parse_args()

    rows = []
    for mf in sorted(glob.glob(args.glob)):
        m = json.loads(Path(mf).read_text())
        slug = Path(mf).stem.replace("cl_", "").replace("_metrics", "")
        recon_f = Path(mf).parent / f"cl_{slug}_recon.json"
        rec = json.loads(recon_f.read_text()) if recon_f.exists() else {}
        cov = m["reconstruction_coverage"]
        pv = m["prereq_violation_share"]
        tau = m["kendall_tau_to_human"]
        rows.append({
            "slug": slug, "book_id": m.get("book_id"),
            "n_book_ch": m["n_book_chapters"], "n_clusters": m["n_a1_clusters"],
            "cov_A1": cov["A1"], "cov_A0": cov.get("A0_mean"),
            "cov_delta": cov.get("A1_minus_A0"),
            "ari": rec.get("adjusted_rand_index"), "nmi": rec.get("normalized_mutual_info"),
            "purity": rec.get("chapter_purity"),
            "pv_A1": pv["A1_frontier"], "pv_A2": pv["A2_goodedge"], "pv_rand": pv["random_mean"],
            "tau_A1": tau["A1_frontier"], "tau_A2": tau["A2_goodedge"], "tau_n": tau["n_pairs"],
        })

    def _f(x, w=6, p=3):
        return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x is not None else f"{'--':>{w}}"

    print(f"\n{'book':16}{'bk_ch':>6}{'cl':>4} | {'covA1':>7}{'covA0':>7}{'Δcov':>7} "
          f"| {'purity':>7}{'ARI':>6} | {'pvA1':>6}{'pvA2':>6}{'pvR':>6} | {'τA1':>6}{'τA2':>6}")
    print("-" * 104)
    for r in rows:
        print(f"{r['slug']:16}{r['n_book_ch']:>6}{r['n_clusters']:>4} | "
              f"{_f(r['cov_A1'],7)}{_f(r['cov_A0'],7)}{_f(r['cov_delta'],7)} | "
              f"{_f(r['purity'],7)}{_f(r['ari'],6)} | "
              f"{_f(r['pv_A1'],6)}{_f(r['pv_A2'],6)}{_f(r['pv_rand'],6)} | "
              f"{_f(r['tau_A1'],6)}{_f(r['tau_A2'],6)}")

    def _signtest(deltas, label):
        d = [x for x in deltas if x is not None and not np.isnan(x)]
        if not d:
            print(f"  {label}: no data")
            return None
        pos = sum(x > 0 for x in d)
        print(f"  {label}: mean={np.mean(d):+.3f}  median={np.median(d):+.3f}  "
              f"sign {pos}/{len(d)} positive  (per-book: {['%+.2f' % x for x in d]})")
        return {"mean": float(np.mean(d)), "median": float(np.median(d)),
                "n_pos": int(pos), "n": len(d), "deltas": d}

    print("\nPOOLED (sign test across books):")
    pooled = {
        "placement_cov_A1_minus_A0": _signtest([r["cov_delta"] for r in rows],
                                               "placement  A1-A0 coverage"),
        "ordering_pv_A1_minus_A2": _signtest(
            [r["pv_A1"] - r["pv_A2"] if r["pv_A1"] is not None and r["pv_A2"] is not None else None
             for r in rows], "prereq     A1-A2 share (>0 = good-edge fewer violations)"),
        "ordering_tau_A2_minus_A1": _signtest(
            [r["tau_A2"] - r["tau_A1"] if r["tau_A1"] is not None and r["tau_A2"] is not None else None
             for r in rows], "tau-human  A2-A1 (>0 = good-edge matches human better)"),
    }
    args.out.write_text(json.dumps({"books": rows, "pooled": pooled}, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} books)")


if __name__ == "__main__":
    main()
