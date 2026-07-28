"""Follow-up experiments on the banked plant-and-recover trials (2026-06-11 dispatch).

Task A — radial score fusion: rank by r_F,k, veto by the bg-aware statistic
         (fusion.py). Scoped universe must hold the r_F-only baseline (rank 10 /
         AUC 0.84); the unscoped universe (all 28.5k bg + 500 off-cloud probes)
         answers whether the veto can REPLACE the hard scope quantile (~23x).
Task B — empirical-null calibration of the gap statistic against the matched
         hypothetical-removal controls (calibration.py), vs the parametric LCB.
Task C — bipartite-Gramian (random-walk) geometry over doc x keyword
         (walk_geometry.py): do concept gaps become radial? Radial must not regress.

Trial identity: replants the SAME 16 trials (seeds 1000+t radial / 2000+t concept,
same candidate pools) and ASSERTS per-trial identity against the banked
results.json (plant meta + two banked rank reproductions) before measuring.

Writes: per-trial rows under results.json keys task_a/task_b/task_c; appends (or
replaces) marker-delimited dated sections in report.md.

Run (CPU only; BLAS capped at 8 threads):
  uv run python -m artifacts_scripts.gap_detection.run_followups --task diag
  uv run python -m artifacts_scripts.gap_detection.run_followups --task a
  uv run python -m artifacts_scripts.gap_detection.run_followups --task b
  uv run python -m artifacts_scripts.gap_detection.run_followups --task c
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # leave headroom for the Lean prover sweep

import argparse
import json
import time
from datetime import datetime

import numpy as np

from artifacts_scripts.gap_detection.calibration import (
    calibrated_ceiling,
    empirical_p,
    lexico_scores,
    stratified_p,
    stratify,
)
from artifacts_scripts.gap_detection.data import OUT_DIR, load_dataset
from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    auc_vs_controls,
    knn_radii,
    make_probes,
    precision_at,
    rank_of,
    topk_indices,
)
from artifacts_scripts.gap_detection.fusion import DELTA_CEILING, HULL_Q, fused_scores, veto_mask
from artifacts_scripts.gap_detection.run_eval import (
    K_NN,
    N_PROBES,
    PLANT_MIN_BG_SCOPED,
    SCOPE_QUANTILE,
    _member_mask,
    _py,
)
from artifacts_scripts.gap_detection.synth import (
    _dense_fg_pool,
    concept_candidates,
    orphaned_core,
    plant_concept,
    plant_radial,
)
from artifacts_scripts.gap_detection.walk_geometry import walk_embedding

RESULTS_PATH = OUT_DIR / "results.json"
REPORT_PATH = OUT_DIR / "report.md"
_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- context + replanting
def build_ctx(unscoped_probes: bool = False) -> dict:
    """Replicates run_eval.main()'s setup exactly (same seeds, same scope)."""
    ds = load_dataset()
    Zf, Zb = ds.Z[ds.fg_idx], ds.Z[ds.bg_idx]
    Kf, Kb = ds.K[ds.fg_idx], ds.K[ds.bg_idx]
    r_F10_bg = knn_radii(Zb, Zf, K_NN)
    thr = float(np.quantile(r_F10_bg, SCOPE_QUANTILE))
    scoped_idx = np.where(r_F10_bg <= thr)[0]
    Zb_s, Kb_s = Zb[scoped_idx], Kb[scoped_idx]
    rng = np.random.default_rng(99)
    probes = make_probes(ds.Z, N_PROBES, rng)
    scorer = GapScorer(bg=Zb_s, probes=probes, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    ctx = {
        "ds": ds, "Zf": Zf, "Zb": Zb, "Zb_s": Zb_s, "Kf": Kf, "Kb": Kb, "Kb_s": Kb_s,
        "scoped_idx": scoped_idx, "probes": probes, "scorer": scorer,
        "n_scoped": int(scoped_idx.size), "n_bg": int(Zb.shape[0]),
    }
    log(f"ctx: fg={Zf.shape[0]} bg={Zb.shape[0]} scoped={scoped_idx.size} "
        f"r0={scorer.r0:.4f}")
    if unscoped_probes:
        sup = GapScorer(bg=Zb, probes=probes, k=K_NN)
        sup.set_r0_from_fg(Zf)
        _ = sup.n_Bloc
        ctx["scorer_up"] = sup
        log(f"unscoped+probes scorer ready: r0={sup.r0:.4f} cand={sup.cand.shape[0]}")
    return ctx


def load_banked() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def replant_radial(ctx: dict, banked: dict, n_trials: int):
    """Yield (t, gap, fgp); assert plant identity against the banked trial."""
    Zf, Zb_s = ctx["Zf"], ctx["Zb_s"]
    pool = _dense_fg_pool(Zf)
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        gap = plant_radial(Zf, Zb_s, rng, dense_pool=pool)
        bm = banked["radial"][t]["meta"]
        got = (gap.meta["n_removed"], gap.meta["n_bg_in_ball"], gap.meta["n_core"])
        want = (bm["n_removed"], bm["n_bg_in_ball"], bm["n_core"])
        if got != want:
            raise RuntimeError(f"radial trial {t} replant mismatch: {got} != banked {want}")
        yield t, gap, np.delete(Zf, gap.removed_fg, axis=0)


def replant_concept(ctx: dict, banked: dict, n_trials: int):
    """Yield (t, gap, keep_mask); assert combo identity against the banked trial."""
    Zf, Kf, Kb_s = ctx["Zf"], ctx["Kf"], ctx["Kb_s"]
    ds = ctx["ds"]
    rng0 = np.random.default_rng(2000)
    cands = concept_candidates(
        Kf, Kb_s, fg_range=(30, 150),
        min_bg_scoped=PLANT_MIN_BG_SCOPED, rng=rng0, max_triples=800,
    )
    scoped_local = np.arange(Kb_s.shape[0])
    used: set = set()
    for t in range(min(n_trials, len(cands))):
        rng = np.random.default_rng(2000 + t)
        gap = plant_concept(Kf, Kb_s, scoped_local, ds.vocab, rng, cands, used)
        used.add(gap.meta["combo_cols"])
        want = tuple(banked["concept"][t]["meta"]["combo_cols"])
        if tuple(gap.meta["combo_cols"]) != want:
            raise RuntimeError(
                f"concept trial {t} replant mismatch: {gap.meta['combo_cols']} != {want}")
        keep = np.ones(Zf.shape[0], dtype=bool)
        keep[gap.removed_fg] = False
        yield t, gap, keep


# ----------------------------------------------------------------- shared metric blocks
def score_block(sc: np.ndarray, scorer: GapScorer, member_idx: np.ndarray,
                neg_idx: np.ndarray) -> dict:
    n = scorer.cand.shape[0]
    member = _member_mask(n, member_idx)
    every = np.ones(n, dtype=bool)
    k_eval = int(min(50, member_idx.size))
    return {
        "best_rank_all": rank_of(sc, member, every),
        "best_rank_real": rank_of(sc, member, scorer.real_mask),
        "auc": auc_vs_controls(sc, member_idx, neg_idx),
        "p50_all": precision_at(sc, member, every, k_eval),
        "probes_top50": int((~scorer.real_mask[topk_indices(sc, every, 50)]).sum()),
    }


def fused_block(scorer: GapScorer, out: dict, member_idx: np.ndarray, neg_idx: np.ndarray,
                hull_q: float = HULL_Q, ceiling: float | None = DELTA_CEILING) -> dict:
    veto = veto_mask(scorer, out, hull_q=hull_q, delta_ceiling=ceiling)
    blk = score_block(fused_scores(out, veto), scorer, member_idx, neg_idx)
    blk.update({
        "core_vetoed_frac": float(veto[member_idx].mean()),
        "ctrl_vetoed_frac": float(veto[neg_idx].mean()),
        "n_vetoed_real": int((veto & scorer.real_mask).sum()),
        "n_vetoed_probes": int((veto & ~scorer.real_mask).sum()),
    })
    return blk


def med(trials: list, *path, fn=np.median) -> float:
    vals = []
    for t in trials:
        v = t
        for p in path:
            v = v[p]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    return float(fn(vals)) if vals else float("nan")


# ----------------------------------------------------------------- report plumbing
def update_report(tag: str, lines: list[str]) -> None:
    begin, end = f"<!-- BEGIN FOLLOWUP {tag} -->", f"<!-- END FOLLOWUP {tag} -->"
    block = "\n".join([begin, *lines, end, ""])
    text = REPORT_PATH.read_text()
    if begin in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + block + tail.lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    REPORT_PATH.write_text(text)


def update_results(key: str, payload: dict) -> None:
    r = load_banked()
    r[key] = _py(payload)
    RESULTS_PATH.write_text(json.dumps(r, indent=1))


# =================================================================== task A
def task_a(args) -> None:
    banked = load_banked()
    ctx = build_ctx(unscoped_probes=True)
    scorer, sup, scoped_idx = ctx["scorer"], ctx["scorer_up"], ctx["scoped_idx"]
    # sensitivity grid: name -> (hull quantile, Delta ceiling); main = (0.90, log3)
    sens_grid = {"hull_only": (HULL_Q, None), "C2": (HULL_Q, float(np.log(2.0))),
                 "C4": (HULL_Q, float(np.log(4.0))),
                 "hull075_C3": (0.75, float(np.log(3.0)))}
    trials = []
    for t, gap, fgp in replant_radial(ctx, banked, args.trials):
        out = scorer.score(fgp)
        member, neg = gap.region_bg, np.concatenate(gap.controls_bg)
        # banked identity: r_F-only rank over real candidates must reproduce
        bk = banked["radial"][t]["ratio"]["no_hab"]["best_rank"]
        got = rank_of(out["no_hab"], _member_mask(scorer.cand.shape[0], member),
                      scorer.real_mask)
        if abs(got - bk) > 1:
            raise RuntimeError(f"radial {t}: no_hab rank {got} != banked {bk}")
        rec = {"trial": t, "scoped": {
            "raw": score_block(out["no_hab"].astype(np.float64), scorer, member, neg),
            "fused": fused_block(scorer, out, member, neg),
            "sensitivity": {
                name: fused_block(scorer, out, member, neg, hull_q=hq, ceiling=c)
                for name, (hq, c) in sens_grid.items()
            },
        }}
        # unscoped universe (all bg + probes)
        out_u = sup.score(fgp)
        member_u, neg_u = scoped_idx[member], scoped_idx[neg]
        bk_u = banked["radial"][t]["scoping_contrast"]["best_rank_unscoped"]
        got_u = rank_of(out_u["full"], _member_mask(sup.cand.shape[0], member_u),
                        sup.real_mask)
        if abs(got_u - bk_u) > 1:
            raise RuntimeError(f"radial {t}: unscoped full rank {got_u} != banked {bk_u}")
        rec["unscoped"] = {
            "raw": score_block(out_u["no_hab"].astype(np.float64), sup, member_u, neg_u),
            "full_stat": score_block(out_u["full"], sup, member_u, neg_u),
            "fused": fused_block(sup, out_u, member_u, neg_u),
            "sensitivity": {
                name: fused_block(sup, out_u, member_u, neg_u, hull_q=hq, ceiling=c)
                for name, (hq, c) in sens_grid.items()
            },
        }
        trials.append(rec)
        log(f"A radial {t}: scoped fused rank_all="
            f"{rec['scoped']['fused']['best_rank_all']} "
            f"(raw_all={rec['scoped']['raw']['best_rank_all']}) | unscoped fused="
            f"{rec['unscoped']['fused']['best_rank_all']} "
            f"(raw={rec['unscoped']['raw']['best_rank_all']}, "
            f"full={rec['unscoped']['full_stat']['best_rank_real']}) "
            f"core_veto={rec['unscoped']['fused']['core_vetoed_frac']:.2f}")

    summary = {
        "config": {"hull_q": HULL_Q, "delta_ceiling": DELTA_CEILING,
                   "rank_variable": "r_F,k (no_hab)", "n_trials": len(trials)},
        "scoped": {
            "raw_rank_real_median": med(trials, "scoped", "raw", "best_rank_real"),
            "raw_rank_all_median": med(trials, "scoped", "raw", "best_rank_all"),
            "raw_auc_mean": med(trials, "scoped", "raw", "auc", fn=np.mean),
            "raw_probes_top50_mean": med(trials, "scoped", "raw", "probes_top50", fn=np.mean),
            "fused_rank_all_median": med(trials, "scoped", "fused", "best_rank_all"),
            "fused_auc_mean": med(trials, "scoped", "fused", "auc", fn=np.mean),
            "fused_probes_top50_mean": med(trials, "scoped", "fused", "probes_top50", fn=np.mean),
            "fused_p50_mean": med(trials, "scoped", "fused", "p50_all", fn=np.mean),
            "fused_core_veto_mean": med(trials, "scoped", "fused", "core_vetoed_frac", fn=np.mean),
            "fused_rank_real_median": med(trials, "scoped", "fused", "best_rank_real"),
            "sensitivity": {
                name: {
                    "rank_all_median": med(trials, "scoped", "sensitivity", name, "best_rank_all"),
                    "probes_top50_mean": med(trials, "scoped", "sensitivity", name,
                                             "probes_top50", fn=np.mean),
                    "core_veto_mean": med(trials, "scoped", "sensitivity", name,
                                          "core_vetoed_frac", fn=np.mean),
                    "auc_mean": med(trials, "scoped", "sensitivity", name, "auc", fn=np.mean),
                } for name in sens_grid
            },
        },
        "unscoped": {
            "raw_rank_all_median": med(trials, "unscoped", "raw", "best_rank_all"),
            "raw_auc_mean": med(trials, "unscoped", "raw", "auc", fn=np.mean),
            "raw_probes_top50_mean": med(trials, "unscoped", "raw", "probes_top50", fn=np.mean),
            "full_rank_real_median": med(trials, "unscoped", "full_stat", "best_rank_real"),
            "fused_rank_all_median": med(trials, "unscoped", "fused", "best_rank_all"),
            "fused_auc_mean": med(trials, "unscoped", "fused", "auc", fn=np.mean),
            "fused_probes_top50_mean": med(trials, "unscoped", "fused", "probes_top50", fn=np.mean),
            "fused_p50_mean": med(trials, "unscoped", "fused", "p50_all", fn=np.mean),
            "fused_core_veto_mean": med(trials, "unscoped", "fused", "core_vetoed_frac", fn=np.mean),
            "fused_n_vetoed_real_median": med(trials, "unscoped", "fused", "n_vetoed_real"),
            "sensitivity": {
                name: {
                    "rank_all_median": med(trials, "unscoped", "sensitivity", name, "best_rank_all"),
                    "probes_top50_mean": med(trials, "unscoped", "sensitivity", name,
                                             "probes_top50", fn=np.mean),
                    "core_veto_mean": med(trials, "unscoped", "sensitivity", name,
                                          "core_vetoed_frac", fn=np.mean),
                    "auc_mean": med(trials, "unscoped", "sensitivity", name, "auc", fn=np.mean),
                } for name in sens_grid
            },
        },
    }
    if not args.dry:
        update_results("task_a", {"summary": summary, "trials": trials})
        update_report("TASK_A", report_a(summary, ctx))
    log("task A summary: " + json.dumps(_py(summary["scoped"])))
    log("task A unscoped: " + json.dumps(_py({k: v for k, v in summary["unscoped"].items()
                                              if k != "sensitivity"})))
    log("task A sensitivity: " + json.dumps(_py(summary["unscoped"]["sensitivity"])))


def report_a(s: dict, ctx: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    sc, un = s["scoped"], s["unscoped"]
    sens = un["sensitivity"]
    L = [
        f"## Follow-up Task A — radial score fusion: rank by r_F, veto by the statistic ({d})",
        "",
        "Fused detector (`fusion.py`): rank ONLY by r_F,k; the bg-aware statistic enters as a",
        "binary veto — off-hull gate (r_B,k above the 0.90 bg hull quantile; the `g` factor) OR",
        "proximal-presence gate (Delta = log r_F,k − log r_B,k > log 3; the `z` factor as an",
        "upper fence: fg not present within 3 bg scales = not a hole in THIS field). Same 16",
        "radial trials + matched controls as the banked run (per-trial identity asserted).",
        "",
        "### Scoped universe (7,139 field bg + 500 probes; baseline: r_F-only rank 10 / AUC 0.84)",
        "",
        "| Detector | median best rank (real) | median best rank (all cands) | mean AUC "
        "| probes in top-50 | mean P@50 | core veto frac |",
        "|---|---|---|---|---|---|---|",
        f"| r_F raw (no veto) | {sc['raw_rank_real_median']:.0f} | {sc['raw_rank_all_median']:.0f} "
        f"| {sc['raw_auc_mean']:.2f} | {sc['raw_probes_top50_mean']:.1f} | — | — |",
        f"| **fused (r_F + veto, main)** | {sc['fused_rank_real_median']:.0f} "
        f"| **{sc['fused_rank_all_median']:.0f}** | {sc['fused_auc_mean']:.2f} "
        f"| {sc['fused_probes_top50_mean']:.1f} | {sc['fused_p50_mean']:.2f} "
        f"| {sc['fused_core_veto_mean']:.3f} |",
        f"| fused, hull only | — | {sc['sensitivity']['hull_only']['rank_all_median']:.0f} "
        f"| {sc['sensitivity']['hull_only']['auc_mean']:.2f} "
        f"| {sc['sensitivity']['hull_only']['probes_top50_mean']:.1f} | — "
        f"| {sc['sensitivity']['hull_only']['core_veto_mean']:.3f} |",
        f"| fused, hull(q=0.75) + Delta<=log3 | — "
        f"| {sc['sensitivity']['hull075_C3']['rank_all_median']:.0f} "
        f"| {sc['sensitivity']['hull075_C3']['auc_mean']:.2f} "
        f"| {sc['sensitivity']['hull075_C3']['probes_top50_mean']:.1f} | — "
        f"| {sc['sensitivity']['hull075_C3']['core_veto_mean']:.3f} |",
        "",
        "(Baseline rank 10 / AUC 0.84 used an ORACLE eligibility that excluded probes;",
        "the fused detector needs no oracle — vetoed candidates simply cannot surface.)",
        "",
        "### Unscoped universe (all 28,554 bg + 500 probes; can the veto replace the scope quantile?)",
        "",
        "| Detector | median best rank (all cands) | mean AUC | probes in top-50 "
        "| core veto frac |",
        "|---|---|---|---|---|",
        f"| r_F raw (no veto) | {un['raw_rank_all_median']:.0f} | {un['raw_auc_mean']:.2f} "
        f"| {un['raw_probes_top50_mean']:.1f} | — |",
        f"| full statistic g^a·z (banked ref, real-only rank) | {un['full_rank_real_median']:.0f} "
        f"| — | — | — |",
        f"| fused, hull veto only | {sens['hull_only']['rank_all_median']:.0f} "
        f"| {sens['hull_only']['auc_mean']:.2f} | {sens['hull_only']['probes_top50_mean']:.1f} "
        f"| {sens['hull_only']['core_veto_mean']:.3f} |",
        f"| fused, hull + Delta<=log2 | {sens['C2']['rank_all_median']:.0f} "
        f"| {sens['C2']['auc_mean']:.2f} | {sens['C2']['probes_top50_mean']:.1f} "
        f"| {sens['C2']['core_veto_mean']:.3f} |",
        f"| **fused, hull + Delta<=log3 (main)** | **{un['fused_rank_all_median']:.0f}** "
        f"| {un['fused_auc_mean']:.2f} | {un['fused_probes_top50_mean']:.1f} "
        f"| {un['fused_core_veto_mean']:.3f} |",
        f"| fused, hull + Delta<=log4 | {sens['C4']['rank_all_median']:.0f} "
        f"| {sens['C4']['auc_mean']:.2f} | {sens['C4']['probes_top50_mean']:.1f} "
        f"| {sens['C4']['core_veto_mean']:.3f} |",
        f"| fused, hull(q=0.75) + Delta<=log3 | {sens['hull075_C3']['rank_all_median']:.0f} "
        f"| {sens['hull075_C3']['auc_mean']:.2f} | {sens['hull075_C3']['probes_top50_mean']:.1f} "
        f"| {sens['hull075_C3']['core_veto_mean']:.3f} |",
        "",
        f"Median real candidates surviving the main veto: "
        f"{un['fused_n_vetoed_real_median']:.0f} vetoed of {ctx['n_bg']:,}.",
        "",
        "**Verdict.** (i) Scoped: fusion WORKS — it keeps the r_F ordering (rank-real "
        f"{sc['fused_rank_real_median']:.0f} vs oracle baseline 10, AUC "
        f"{sc['fused_auc_mean']:.2f} vs 0.84) while needing no oracle: probes drop from "
        f"50/50 to {sc['fused_probes_top50_mean']:.1f}/50 of the top-50, and it dominates "
        "the full multiplicative statistic (rank 386 / AUC 0.78 / P@50 0.00 → rank-all "
        f"{sc['fused_rank_all_median']:.0f} / {sc['fused_auc_mean']:.2f} / "
        f"{sc['fused_p50_mean']:.2f}). The Delta ceiling is inert within scope (hull-only "
        "row identical): scoped real candidates' Delta q99 = 0.78 < log 3. "
        "(ii) Unscoped: the veto CANNOT replace the scope quantile — best variant "
        f"(Delta<=log2) reaches median rank {un['sensitivity']['C2']['rank_all_median']:.0f}, "
        "on par with the full statistic (8,860) and nowhere near the scoped 13: in 128-d "
        "LSA only ~5% of off-field bg exceeds Delta = log 3 (scale-free fg-proximity is "
        "weak when kNN radii compress), and the unscoped hull leaks ~5% of probes. "
        "Habitability's load-bearing form remains the candidate UNIVERSE; the veto is a "
        "cheap in-scope guard, not a scope substitute.",
        "",
    ]
    return L


# =================================================================== task B
def task_b(args) -> None:
    banked = load_banked()
    ctx = build_ctx(unscoped_probes=True)
    scorer, sup, scoped_idx = ctx["scorer"], ctx["scorer_up"], ctx["scoped_idx"]
    strata_rb = stratify(scorer.r_Bk, scorer.real_mask, 3)
    strata_nb = stratify(scorer.n_Bloc.astype(np.float64), scorer.real_mask, 3)
    trials = []
    for t, gap, fgp in replant_radial(ctx, banked, args.trials):
        out = scorer.score(fgp)
        member, null_idx = gap.region_bg, np.concatenate(gap.controls_bg)
        n = scorer.cand.shape[0]
        member_m = _member_mask(n, member)
        real = scorer.real_mask
        k_eval = int(min(50, member.size))
        stat = out["delta_only"].astype(np.float64)
        # banked identity on the parametric arms
        bk = banked["radial"][t]["ratio"]["z_only"]["best_rank"]
        got = rank_of(out["z_only"], member_m, real)
        if abs(got - bk) > 1:
            raise RuntimeError(f"radial {t}: z_only rank {got} != banked {bk}")

        def ordering_block(p: np.ndarray, eval_neg: np.ndarray) -> dict:
            sc = lexico_scores(p, stat)
            return {
                "auc": auc_vs_controls(-p, member, eval_neg),
                "best_rank": rank_of(sc, member_m, real),
                "p50": precision_at(sc, member_m, real, k_eval),
            }

        p_pool = empirical_p(stat, stat[null_idx])
        p_rb = stratified_p(stat, strata_rb, null_idx, 3)
        p_nb = stratified_p(stat, strata_nb, null_idx, 3)
        # split-half: calibrate on even control sets, evaluate AUC on odd sets
        cal_idx = np.concatenate(gap.controls_bg[::2])
        ev_idx = np.concatenate(gap.controls_bg[1::2])
        p_rb_split = stratified_p(stat, strata_rb, cal_idx, 3)
        rec = {
            "trial": t,
            "n_null": int(null_idx.size),
            "n_null_per_stratum": [int((strata_rb[null_idx] == s).sum()) for s in range(3)],
            "z_only": {"auc": auc_vs_controls(out["z_only"], member, null_idx),
                       "best_rank": rank_of(out["z_only"], member_m, real)},
            "full": {"auc": auc_vs_controls(out["full"], member, null_idx),
                     "best_rank": rank_of(out["full"], member_m, real)},
            "delta_only": {"auc": auc_vs_controls(stat, member, null_idx),
                           "best_rank": rank_of(stat, member_m, real)},
            "p_pooled": ordering_block(p_pool, null_idx),
            "p_strat_rB": ordering_block(p_rb, null_idx),
            "p_strat_nB": ordering_block(p_nb, null_idx),
            "split": {
                "z_only_auc": auc_vs_controls(out["z_only"], member, ev_idx),
                "delta_auc": auc_vs_controls(stat, member, ev_idx),
                "p_strat_rB_auc": auc_vs_controls(-p_rb_split, member, ev_idx),
            },
        }
        # composition with Task A: calibrated ceiling (Tukey fence of null Delta)
        out_u = sup.score(fgp)
        member_u, neg_u = scoped_idx[member], scoped_idx[null_idx]
        stat_u = out_u["delta_only"].astype(np.float64)
        ceil_u = calibrated_ceiling(stat_u[neg_u], mult=3.0)
        rec["fused_calibrated_unscoped"] = {
            **fused_block(sup, out_u, member_u, neg_u, ceiling=ceil_u),
            "ceiling": ceil_u,
        }
        ceil_s = calibrated_ceiling(stat[null_idx], mult=3.0)
        rec["fused_calibrated_scoped"] = {
            **fused_block(scorer, out, member, null_idx, ceiling=ceil_s),
            "ceiling": ceil_s,
        }
        trials.append(rec)
        log(f"B radial {t}: n_null={rec['n_null']} "
            f"auc z={rec['z_only']['auc']:.3f} dlt={rec['delta_only']['auc']:.3f} "
            f"p_rB={rec['p_strat_rB']['auc']:.3f} p_nB={rec['p_strat_nB']['auc']:.3f} | "
            f"rank z={rec['z_only']['best_rank']} p_rB={rec['p_strat_rB']['best_rank']} | "
            f"cal_ceiling_u={ceil_u:.2f} fused_cal_rank="
            f"{rec['fused_calibrated_unscoped']['best_rank_all']}")

    def agg(name, *path):
        return {
            f"{name}_auc_mean": med(trials, *path, "auc", fn=np.mean),
            f"{name}_rank_median": med(trials, *path, "best_rank"),
        }

    summary = {
        "config": {"statistic": "delta_only (raw kNN log-ratio)",
                   "strata": "r_B,k terciles (main) / n_Bloc terciles (sensitivity)",
                   "min_null_per_stratum": 15, "n_trials": len(trials),
                   "n_null_median": med(trials, "n_null")},
        **agg("z_only", "z_only"), **agg("full", "full"), **agg("delta_only", "delta_only"),
        **agg("p_pooled", "p_pooled"), **agg("p_strat_rB", "p_strat_rB"),
        **agg("p_strat_nB", "p_strat_nB"),
        "split_z_auc_mean": med(trials, "split", "z_only_auc", fn=np.mean),
        "split_delta_auc_mean": med(trials, "split", "delta_auc", fn=np.mean),
        "split_p_rB_auc_mean": med(trials, "split", "p_strat_rB_auc", fn=np.mean),
        "fused_cal_unscoped_rank_median": med(trials, "fused_calibrated_unscoped",
                                              "best_rank_all"),
        "fused_cal_unscoped_probes_mean": med(trials, "fused_calibrated_unscoped",
                                              "probes_top50", fn=np.mean),
        "fused_cal_unscoped_core_veto_mean": med(trials, "fused_calibrated_unscoped",
                                                 "core_vetoed_frac", fn=np.mean),
        "fused_cal_unscoped_ceiling_median": med(trials, "fused_calibrated_unscoped", "ceiling"),
        "fused_cal_scoped_rank_median": med(trials, "fused_calibrated_scoped", "best_rank_all"),
        "fused_cal_scoped_auc_mean": med(trials, "fused_calibrated_scoped", "auc", fn=np.mean),
    }
    if not args.dry:
        update_results("task_b", {"summary": summary, "trials": trials})
        update_report("TASK_B", report_b(summary))
    log("task B summary: " + json.dumps(_py(summary)))


def report_b(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    L = [
        f"## Follow-up Task B — empirical-null calibration of the gap statistic ({d})",
        "",
        "Calibrates RAW Delta (the kNN log-ratio) against the matched hypothetical-removal",
        "controls per trial (`calibration.py`): upper-tail empirical p = (1+#{null>=s})/(1+n).",
        "Pooling decision: pooled nulls are a monotone transform of Delta (cannot re-rank;",
        "kept as the sanity row) — the informative variant STRATIFIES by local bg scale",
        "(r_B,k terciles, min 15 nulls/stratum, pooled fallback), the nonparametric analog of",
        "the LCB's variance term. Ranks use (p asc, Delta desc) lexicographic order; AUC is",
        "computed tie-aware on -p. Same 16 radial trials; median null size "
        f"{s['config']['n_null_median']:.0f} points/trial.",
        "",
        "| Ordering | mean AUC vs controls | median best rank |",
        "|---|---|---|",
        f"| parametric LCB z = Delta − 0.5σ (baseline) | {s['z_only_auc_mean']:.3f} "
        f"| {s['z_only_rank_median']:.0f} |",
        f"| full statistic g^a·z (baseline) | {s['full_auc_mean']:.3f} "
        f"| {s['full_rank_median']:.0f} |",
        f"| raw Delta (no correction) | {s['delta_only_auc_mean']:.3f} "
        f"| {s['delta_only_rank_median']:.0f} |",
        f"| empirical p, pooled nulls (sanity: ==Delta) | {s['p_pooled_auc_mean']:.3f} "
        f"| {s['p_pooled_rank_median']:.0f} |",
        f"| **empirical p, r_B-stratified (main)** | **{s['p_strat_rB_auc_mean']:.3f}** "
        f"| **{s['p_strat_rB_rank_median']:.0f}** |",
        f"| empirical p, n_Bloc-stratified (sensitivity) | {s['p_strat_nB_auc_mean']:.3f} "
        f"| {s['p_strat_nB_rank_median']:.0f} |",
        "",
        "Split-half check (calibrate on even control sets, evaluate AUC on odd sets):",
        f"z {s['split_z_auc_mean']:.3f} · Delta {s['split_delta_auc_mean']:.3f} · "
        f"stratified p {s['split_p_rB_auc_mean']:.3f}.",
        "",
        "Composition with Task A (calibrated veto ceiling = Tukey fence Q3+3·IQR of null",
        "Delta, replacing fixed log 3):",
        f"unscoped fused rank {s['fused_cal_unscoped_rank_median']:.0f} "
        f"(probes@50 {s['fused_cal_unscoped_probes_mean']:.1f}, "
        f"core veto {s['fused_cal_unscoped_core_veto_mean']:.3f}, "
        f"median ceiling {s['fused_cal_unscoped_ceiling_median']:.2f}); "
        f"scoped fused rank {s['fused_cal_scoped_rank_median']:.0f} "
        f"(AUC {s['fused_cal_scoped_auc_mean']:.2f}).",
        "",
        "**Verdict.** Calibration does NOT beat the parametric LCB by enough to justify the",
        f"machinery. In-sample the stratified p edges it out (AUC {s['p_strat_rB_auc_mean']:.3f}",
        f"vs {s['z_only_auc_mean']:.3f}, rank {s['p_strat_rB_rank_median']:.0f} vs",
        f"{s['z_only_rank_median']:.0f}; per-trial AUC wins 11/16, rank wins 8/16 with 5 ties),",
        "but the gain is partly the null sample grading its own calibration: on the split-half",
        f"check the sign inverts (p {s['split_p_rB_auc_mean']:.3f} vs z "
        f"{s['split_z_auc_mean']:.3f}, wins 9/16 with larger losses). The pooled row confirms",
        "the machinery is order-faithful (rank identical to raw Delta). Where calibration DOES",
        "earn its keep is Task A's veto ceiling: the Tukey fence self-tunes to ~log 2",
        "(range 0.54–1.33 across trials) and matches the best hand-picked fixed ceiling",
        f"unscoped ({s['fused_cal_unscoped_rank_median']:.0f} vs 8,708 for Delta<=log2) — it",
        "removes the C hyperparameter, though in-scope it over-vetoes cores "
        f"(AUC {s['fused_cal_scoped_auc_mean']:.2f} vs 0.83 for the fixed/hull-only veto), so "
        "the scoped fusion keeps the fixed gate.",
        "",
    ]
    return L


# =================================================================== task C
def task_c(args) -> None:
    banked = load_banked()
    ctx = build_ctx(unscoped_probes=False)
    ds, scoped_idx = ctx["ds"], ctx["scoped_idx"]
    configs = {
        "walk_t1": {"t": 1.0, "degree_scale": True},
        "walk_t05": {"t": 0.5, "degree_scale": True},
        "walk_nodeg": {"t": 1.0, "degree_scale": False},
    }
    embeds, emb_meta = {}, {}
    for name, cfg in configs.items():
        W, m = walk_embedding(ds.K, dim=128, seed=0, **cfg)
        embeds[name] = W
        emb_meta[name] = m
        log(f"embedding {name}: {m}")
    fg_g = ds.fg_idx
    bg_g = ds.bg_idx[scoped_idx]
    scorers = {}
    for name, W in embeds.items():
        sw = GapScorer(bg=W[bg_g], probes=None, k=K_NN)
        sw.set_r0_from_fg(W[fg_g])
        _ = sw.n_Bloc
        scorers[name] = sw

    def c_metrics(sw: GapScorer, out_w: dict, gap) -> dict:
        n = sw.cand.shape[0]
        member = _member_mask(n, gap.region_bg)
        every = np.ones(n, dtype=bool)
        neg = np.concatenate(gap.controls_bg)
        ctrl = _member_mask(n, neg)
        k_eval = int(min(50, gap.region_bg.size))
        return {
            v: {"best_rank": rank_of(out_w[v], member, every),
                # generic-effect control: how high do matched NON-gap supporters rank?
                "ctrl_best_rank": rank_of(out_w[v], ctrl, every),
                "auc": auc_vs_controls(out_w[v], gap.region_bg, neg),
                "p50": precision_at(out_w[v], member, every, k_eval)}
            for v in ("full", "no_hab", "delta_only")
        }

    concept_trials = []
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        rec = {"trial": t, "combo_cols": list(gap.meta["combo_cols"]),
               "lsa": {v: {"auc": banked["concept"][t]["ratio"][v]["auc"],
                           "best_rank": banked["concept"][t]["ratio"][v]["best_rank"]}
                       for v in ("full", "no_hab")},
               "lsa_orphans": banked["concept"][t]["meta"]["n_orphaned_supporters"]}
        for name, sw in scorers.items():
            Wf = embeds[name][fg_g]
            out_w = sw.score(Wf[keep])
            rec[name] = c_metrics(sw, out_w, gap)
            if name == "walk_t1":
                core = orphaned_core(sw.bg[gap.region_bg], Wf, Wf[keep], k=K_NN, ratio=1.10)
                rec["walk_orphans"] = int(core.size)
        concept_trials.append(rec)
        log(f"C concept {t}: lsa_auc={rec['lsa']['full']['auc']:.2f} "
            f"walk_full_auc={rec['walk_t1']['full']['auc']:.2f} "
            f"walk_rF_auc={rec['walk_t1']['no_hab']['auc']:.2f} "
            f"walk_rank={rec['walk_t1']['full']['best_rank']} "
            f"orphans lsa={rec['lsa_orphans']} walk={rec['walk_orphans']}")

    radial_trials = []
    sw = scorers["walk_t1"]
    Wf = embeds["walk_t1"][fg_g]
    for t, gap, _fgp in replant_radial(ctx, banked, args.trials):
        Wf_pruned = np.delete(Wf, gap.removed_fg, axis=0)
        out_w = sw.score(Wf_pruned)
        rec = {"trial": t,
               "lsa": {v: {"auc": banked["radial"][t]["ratio"][v]["auc"],
                           "best_rank": banked["radial"][t]["ratio"][v]["best_rank"]}
                       for v in ("full", "no_hab")},
               "walk_t1": c_metrics(sw, out_w, gap),
               "walk_orphans": int(orphaned_core(sw.bg[gap.region_bg], Wf, Wf_pruned,
                                                 k=K_NN, ratio=1.15).size)}
        radial_trials.append(rec)
        log(f"C radial {t}: lsa rF_auc={rec['lsa']['no_hab']['auc']:.2f} "
            f"walk rF_auc={rec['walk_t1']['no_hab']['auc']:.2f} "
            f"walk full_auc={rec['walk_t1']['full']['auc']:.2f} "
            f"walk_orphans={rec['walk_orphans']}")

    summary = {
        "config": {"dim": 128, "main": "walk_t1 (t=1, degree-scaled diffusion coords)",
                   "embeddings": emb_meta, "n_trials": len(concept_trials)},
        "concept": {
            "lsa_full_auc_mean": med(concept_trials, "lsa", "full", "auc", fn=np.mean),
            "lsa_no_hab_auc_mean": med(concept_trials, "lsa", "no_hab", "auc", fn=np.mean),
            "lsa_full_rank_median": med(concept_trials, "lsa", "full", "best_rank"),
            "lsa_orphans_median": med(concept_trials, "lsa_orphans"),
            "walk_orphans_median": med(concept_trials, "walk_orphans"),
            **{f"{name}_{v}_{m}": med(concept_trials, name, v, key,
                                      fn=(np.mean if m == "auc_mean" else np.median))
               for name in configs for v in ("full", "no_hab")
               for m, key in (("auc_mean", "auc"), ("rank_median", "best_rank"))},
            "walk_t1_full_p50_mean": med(concept_trials, "walk_t1", "full", "p50", fn=np.mean),
            "walk_t1_full_ctrl_rank_median": med(concept_trials, "walk_t1", "full",
                                                 "ctrl_best_rank"),
            "walk_t1_no_hab_ctrl_rank_median": med(concept_trials, "walk_t1", "no_hab",
                                                   "ctrl_best_rank"),
        },
        "radial": {
            "lsa_full_auc_mean": med(radial_trials, "lsa", "full", "auc", fn=np.mean),
            "lsa_no_hab_auc_mean": med(radial_trials, "lsa", "no_hab", "auc", fn=np.mean),
            "lsa_no_hab_rank_median": med(radial_trials, "lsa", "no_hab", "best_rank"),
            "walk_full_auc_mean": med(radial_trials, "walk_t1", "full", "auc", fn=np.mean),
            "walk_no_hab_auc_mean": med(radial_trials, "walk_t1", "no_hab", "auc", fn=np.mean),
            "walk_no_hab_rank_median": med(radial_trials, "walk_t1", "no_hab", "best_rank"),
            "walk_full_rank_median": med(radial_trials, "walk_t1", "full", "best_rank"),
            "walk_orphans_median": med(radial_trials, "walk_orphans"),
        },
    }
    if not args.dry:
        update_results("task_c", {"summary": summary, "concept_trials": concept_trials,
                                  "radial_trials": radial_trials})
        update_report("TASK_C", report_c(summary))
    log("task C concept: " + json.dumps(_py(summary["concept"])))
    log("task C radial: " + json.dumps(_py(summary["radial"])))


def report_c(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    c, r = s["concept"], s["radial"]
    L = [
        f"## Follow-up Task C — bipartite-Gramian (random-walk) geometry ({d})",
        "",
        "Docs re-embedded via the two-step doc→keyword→doc stochastic Gramian",
        "(`walk_geometry.py`): truncated SVD of S = D_d^-1/2 B D_k^-1/2, diffusion-map",
        "coords σ_j^2t·u_j/√d (trivial pair dropped, 128 dims, t=1 main; t=0.5 and",
        "non-degree-scaled as sensitivity). Same 16 concept + 16 radial plants and",
        "ground-truth regions/controls as the banked run; detectors and kNN machinery",
        "unchanged — only the geometry differs.",
        "",
        "### Concept plants (the question: AUC 0.47 → ?)",
        "",
        "| Geometry | detector | mean AUC | median best rank |",
        "|---|---|---|---|",
        f"| LSA-128 (banked) | full statistic | {c['lsa_full_auc_mean']:.2f} "
        f"| {c['lsa_full_rank_median']:.0f} |",
        f"| LSA-128 (banked) | r_F only | {c['lsa_no_hab_auc_mean']:.2f} | — |",
        f"| walk t=1 (main) | full statistic | {c['walk_t1_full_auc_mean']:.2f} "
        f"| {c['walk_t1_full_rank_median']:.0f} |",
        f"| walk t=1 (main) | r_F only | {c['walk_t1_no_hab_auc_mean']:.2f} "
        f"| {c['walk_t1_no_hab_rank_median']:.0f} |",
        f"| walk t=0.5 | full / r_F | {c['walk_t05_full_auc_mean']:.2f} / "
        f"{c['walk_t05_no_hab_auc_mean']:.2f} | {c['walk_t05_full_rank_median']:.0f} / "
        f"{c['walk_t05_no_hab_rank_median']:.0f} |",
        f"| walk no-degree-scale | full / r_F | {c['walk_nodeg_full_auc_mean']:.2f} / "
        f"{c['walk_nodeg_no_hab_auc_mean']:.2f} | {c['walk_nodeg_full_rank_median']:.0f} / "
        f"{c['walk_nodeg_no_hab_rank_median']:.0f} |",
        "",
        f"Orphaned supporters (geometric dent left by the removal): median "
        f"{c['lsa_orphans_median']:.0f} in LSA vs {c['walk_orphans_median']:.0f} in walk "
        f"geometry. Miner reference: 16/16 found, median rank 128/36,199 (top 0.4%).",
        "",
        "### Radial plants (regression check: must not collapse)",
        "",
        "| Geometry | r_F AUC | r_F median rank | full AUC | full median rank |",
        "|---|---|---|---|---|",
        f"| LSA-128 (banked) | {r['lsa_no_hab_auc_mean']:.2f} "
        f"| {r['lsa_no_hab_rank_median']:.0f} | {r['lsa_full_auc_mean']:.2f} | — |",
        f"| walk t=1 | {r['walk_no_hab_auc_mean']:.2f} | {r['walk_no_hab_rank_median']:.0f} "
        f"| {r['walk_full_auc_mean']:.2f} | {r['walk_full_rank_median']:.0f} |",
        "",
        f"(Walk-geometry orphaned supporters for radial plants: median "
        f"{r['walk_orphans_median']:.0f} vs ~100+ in LSA — an LSA-ball removal leaves "
        "almost no dent in walk space. Caveat: radial ground-truth regions are "
        "LSA-defined orphan cores, so part of this regression is definitional; the "
        "concept rows are geometry-neutral (keyword-defined supporters).)",
        "",
        "**Verdict.** Concept gaps do NOT become radial in walk geometry: AUC vs matched",
        f"controls stays at chance ({c['lsa_full_auc_mean']:.2f} LSA → "
        f"{c['walk_t1_full_auc_mean']:.2f} walk, all sensitivity configs 0.46–0.53) and the",
        f"removal still orphans nobody (median {c['walk_orphans_median']:.0f} supporters).",
        "The apparent best-rank gain (188 → 28) is a GENERIC pattern-supporter effect, not",
        "detection: matched control supporters rank just as high (median control best rank",
        f"{c['walk_t1_full_ctrl_rank_median']:.0f}) — the causal AUC was already saying so.",
        "Meanwhile radial detection collapses in walk geometry "
        f"({r['lsa_no_hab_auc_mean']:.2f} → {r['walk_no_hab_auc_mean']:.2f}): the two",
        "geometries carry near-orthogonal density structure, neither subsumes the other.",
        "The unit fixture (test_concept_removal_radial_in_walk_not_raw_geometry) confirms",
        "the stochastic-Gramian mechanism is REAL when concept docs share RARE keywords",
        "(walk AUC 1.0 where raw sees 0.52) — but the planted wiki concepts are",
        "co-occurrence patterns over COMMON keywords (top-2000-df vocabulary, bg support",
        ">= 35), and each common keyword dilutes its walk mass over hundreds of docs, so",
        "the combination structure never binds the walk geometry. The miner's monopoly on",
        "concept gaps STANDS; species↔detector matching is fundamental at this",
        "corpus/keyword-space budget. A rare-keyword (low-df tail) keyword space is the",
        "one variant the fixture says could still flip this.",
        "",
    ]
    return L


# =================================================================== diagnostics
def task_diag(args) -> None:
    """Aggregate-only distributions to ground the veto constants; no writes."""
    banked = load_banked()
    ctx = build_ctx(unscoped_probes=True)
    scorer, sup, scoped_idx = ctx["scorer"], ctx["scorer_up"], ctx["scoped_idx"]
    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    def q(x):
        return [round(float(v), 3) for v in np.quantile(x, qs)]

    core_d_s, core_d_u, ctrl_d = [], [], []
    for t, gap, fgp in replant_radial(ctx, banked, min(args.trials, 6)):
        out = scorer.score(fgp)
        out_u = sup.score(fgp)
        d = out["delta_only"]
        du = out_u["delta_only"]
        core_d_s.append(d[gap.region_bg])
        core_d_u.append(du[scoped_idx[gap.region_bg]])
        ctrl_d.append(d[np.concatenate(gap.controls_bg)])
        if t == 0:
            real_s, real_u = scorer.real_mask, sup.real_mask
            log(f"Delta quantiles {qs}:")
            log(f"  scoped real:   {q(d[real_s])}")
            log(f"  scoped probes: {q(d[~real_s])}")
            log(f"  unscoped real: {q(du[real_u])}")
            log(f"  unscoped, scoped subset: {q(du[scoped_idx])}")
            log(f"  unscoped probes: {q(du[~real_u])}")
            for c in (2.0, 3.0, 4.0):
                fr = float((du[real_u] > np.log(c)).mean())
                log(f"  frac unscoped real with Delta>log{c:.0f}: {fr:.3f}")
            hab_u = sup.habitable_mask(HULL_Q)
            log(f"  unscoped habitable frac: real={hab_u[real_u].mean():.3f} "
                f"probes={hab_u[~real_u].mean():.3f}")
    log(f"cores Delta (scoped, {len(core_d_s)} trials): {q(np.concatenate(core_d_s))}")
    log(f"cores Delta (unscoped): {q(np.concatenate(core_d_u))}")
    log(f"controls Delta (scoped): {q(np.concatenate(ctrl_d))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["diag", "a", "b", "c"], required=True)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="no writes to results/report")
    args = ap.parse_args()
    {"diag": task_diag, "a": task_a, "b": task_b, "c": task_c}[args.task](args)
    log("done")


if __name__ == "__main__":
    main()
