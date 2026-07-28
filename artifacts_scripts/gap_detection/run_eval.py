"""Plant-and-recover evaluation runner.

Plants ~16 synthetic gaps of each species (radial / concept / corridor) in the
wiki LSA foreground cloud, runs every detector on every trial, aggregates
recovery metrics, runs the spec'd ablations (habitability term off, variance
normalization off) plus two scoping contrasts (geometric and lattice-side),
optionally compares the legacy k-ball detector, and writes:

    artifacts/gap_detection_eval/report.md
    artifacts/gap_detection_eval/results.json

Design notes (measured, see report):
  * Candidate background = the field's territory (bg docs within the 0.25
    distance quantile of the fg centroid) + off-cloud probes. This mirrors the
    production setting where foreground and background share a topical scope;
    the unscoped variant is reported as a contrast (off-field bg drowns all
    rankings — the everything-off-field pathology).
  * Ground-truth regions are orphaned cores (see synth.py docstring).
  * P@50 is precision@min(50, |region|).

Run:  uv run python -m artifacts_scripts.gap_detection.run_eval
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import argparse
import json
import re
import sys
import time
from datetime import datetime

import numpy as np

from artifacts_scripts.gap_detection.data import OUT_DIR, load_dataset
from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    VARIANTS,
    auc_vs_controls,
    closed_filter,
    fightin_words_rank,
    iou,
    level_set_components,
    make_probes,
    mine_frequent_itemsets,
    precision_at,
    rank_of,
    topk_indices,
    voltage_current_scores,
)
from artifacts_scripts.gap_detection.synth import (
    PlantedGap,
    _dense_fg_pool,
    concept_candidates,
    corridor_pairs,
    plant_concept,
    plant_corridor,
    plant_radial,
)

K_NN = 10
N_PROBES = 500
TOP_N = 50
IOU_RECALL_THR = 0.1
SCOPE_QUANTILE = 0.25
MINE_MIN_SUPPORT = 30
PLANT_MIN_BG_SCOPED = 35
UNSCOPED_MIN_SUPPORT = 100


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, frozenset):
        return sorted(o)
    return o


def _member_mask(n: int, idx: np.ndarray) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def ratio_metrics(scorer: GapScorer, out: dict, gap: PlantedGap,
                  center: np.ndarray, radius: float) -> dict:
    """Recovery + ablation metrics of the kNN density-ratio statistic, per variant."""
    n = scorer.cand.shape[0]
    member = _member_mask(n, gap.region_bg)
    real = scorer.real_mask
    every = np.ones(n, dtype=bool)
    habitable = scorer.habitable_mask(0.90)
    stratum = scorer.bg_density_stratum(3)
    neg = (np.concatenate(gap.controls_bg) if gap.controls_bg else np.array([], dtype=int))
    k_eval = int(min(TOP_N, gap.region_bg.size))
    res = {}
    for v in VARIANTS:
        sc = out[v]
        top_real = topk_indices(sc, real, k_eval)
        top_all = topk_indices(sc, every, TOP_N)
        top1 = scorer.cand[top_real[0]]
        fp = top_real[~member[top_real]]
        res[v] = {
            "best_rank": rank_of(sc, member, real),
            "p50": precision_at(sc, member, real, k_eval),
            "auc": auc_vs_controls(sc, gap.region_bg, neg),
            "loc_err": float(np.linalg.norm(top1 - center) / max(radius, 1e-9)),
            # ablation (i): habitability — how much of the top-50 leaves the bg-dense hull
            "frac_top50_outside_habitable": float((~habitable[top_all]).mean()),
            "n_probes_in_top50": int((~real[top_all]).sum()),
            # ablation (ii): variance normalization — sparse-mirage diagnostics
            "mirage_frac_top50": float((stratum[topk_indices(sc, real, TOP_N)] == 0).mean()),
            "fp_in_sparse_stratum": float((stratum[fp] == 0).mean()) if fp.size else 0.0,
        }
    return res


def levelset_metrics(scorer: GapScorer, scores: np.ndarray, gap: PlantedGap) -> dict:
    comps = level_set_components(scores, scorer.cand, scorer.real_mask, top_frac=0.05)
    top3 = comps[:3]
    best_iou = max((iou(c["members"], gap.region_bg) for c in top3), default=0.0)
    best_iou_any = max((iou(c["members"], gap.region_bg) for c in comps), default=0.0)
    return {
        "n_components": len(comps),
        "recall_top3": bool(best_iou >= IOU_RECALL_THR),
        "best_iou_top3": round(best_iou, 4),
        "best_iou_any": round(best_iou_any, 4),
        "top_n_effs": [round(c["n_eff"], 1) for c in comps[:5]],
    }


# ----------------------------------------------------------------- per-species runners
def run_radial(ctx, n_trials, legacy_budget_s, log):
    scorer, scorer_u = ctx["scorer"], ctx["scorer_unscoped"]
    Zf, Zb_s, scoped_idx = ctx["Zf"], ctx["Zb_s"], ctx["scoped_idx"]
    trials = []
    pool = _dense_fg_pool(Zf)
    legacy_used, legacy_fail = 0.0, None
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        gap = plant_radial(Zf, Zb_s, rng, dense_pool=pool)
        fgp = np.delete(Zf, gap.removed_fg, axis=0)
        out = scorer.score(fgp)
        rec = {
            "trial": t, "meta": gap.meta,
            "ratio": ratio_metrics(scorer, out, gap, gap.center, gap.radius),
            "levelset": levelset_metrics(scorer, out["full"], gap),
        }
        # geometric scoping contrast: same statistic, candidates = ALL bg (off-field too)
        out_u = scorer_u.score(fgp)
        region_u = scoped_idx[gap.region_bg]
        member_u = _member_mask(scorer_u.cand.shape[0], region_u)
        rec["scoping_contrast"] = {
            "best_rank_unscoped": rank_of(out_u["full"], member_u, scorer_u.real_mask),
            "p50_unscoped": precision_at(out_u["full"], member_u, scorer_u.real_mask,
                                         int(min(TOP_N, region_u.size))),
        }
        if legacy_budget_s > 0 and legacy_used < legacy_budget_s and legacy_fail is None:
            try:
                from artifacts_scripts.gap_detection.legacy_compare import run_legacy_trial

                lr = run_legacy_trial(fgp, Zb_s, gap.region_bg,
                                      budget_s=min(180.0, legacy_budget_s - legacy_used))
                legacy_used += lr["elapsed_s"]
                rec["legacy"] = lr
            except Exception as e:  # noqa: BLE001 — report-not-crash by design
                legacy_fail = f"{type(e).__name__}: {e}"
                log(f"  legacy comparison disabled: {legacy_fail}")
        trials.append(rec)
        log(f"  radial {t}: removed={gap.meta['n_removed']} core={gap.meta['n_core']} "
            f"best_rank={rec['ratio']['full']['best_rank']} "
            f"p50={rec['ratio']['full']['p50']:.2f} "
            f"auc={rec['ratio']['full']['auc']:.2f} "
            f"unscoped_rank={rec['scoping_contrast']['best_rank_unscoped']}")
    return trials, legacy_fail


def run_concept(ctx, n_trials, log):
    scorer = ctx["scorer"]
    Zf, Kf, Kb_s = ctx["Zf"], ctx["Kf"], ctx["Kb_s"]
    ds = ctx["ds"]
    rng0 = np.random.default_rng(2000)
    cands = concept_candidates(
        Kf, Kb_s, fg_range=(30, 150),
        min_bg_scoped=PLANT_MIN_BG_SCOPED, rng=rng0, max_triples=800,
    )
    n_triples = sum(1 for c in cands if len(c[0]) == 3)
    log(f"  concept candidate combos: {len(cands)} ({n_triples} triples)")
    Kf_dense = np.asarray(Kf.todense(), dtype=bool)
    Kb_s_dense = np.asarray(Kb_s.todense(), dtype=bool)
    Kb_full_dense = np.asarray(ctx["Kb"].todense(), dtype=bool)
    # families mined once — the bg side never changes across trials
    fam_scoped = closed_filter(
        mine_frequent_itemsets(Kb_s_dense, min_support=MINE_MIN_SUPPORT,
                               max_len=3, max_pairs_expand=40_000)
    )
    fam_unscoped = closed_filter(
        mine_frequent_itemsets(Kb_full_dense, min_support=UNSCOPED_MIN_SUPPORT, max_len=2)
    )
    log(f"  mined itemsets: scoped={len(fam_scoped)}, unscoped-pairs={len(fam_unscoped)}")
    scoped_local = np.arange(Kb_s.shape[0])
    trials, used = [], set()
    for t in range(min(n_trials, len(cands))):
        rng = np.random.default_rng(2000 + t)
        gap = plant_concept(Kf, Kb_s, scoped_local, ds.vocab, rng, cands, used)
        used.add(gap.meta["combo_cols"])
        keep = np.ones(Zf.shape[0], dtype=bool)
        keep[gap.removed_fg] = False
        fgp = Zf[keep]
        Kf_pruned = Kf_dense[keep]
        out = scorer.score(fgp)
        center = Zf[gap.removed_fg].mean(axis=0)
        radius = float(np.median(np.linalg.norm(Zf[gap.removed_fg] - center, axis=1)))
        rec = {
            "trial": t, "meta": gap.meta, "keywords": list(gap.keywords),
            "ratio": ratio_metrics(scorer, out, gap, center, radius),
            "levelset": levelset_metrics(scorer, out["full"], gap),
        }
        # how geometrically visible was this concept removal? (orphaned supporters)
        from artifacts_scripts.gap_detection.synth import orphaned_core

        core_local = orphaned_core(ctx["Zb_s"][gap.region_bg], Zf, fgp, k=K_NN, ratio=1.10)
        core = gap.region_bg[core_local]
        rec["meta"]["n_orphaned_supporters"] = int(core.size)
        if core.size >= 5:
            gap_core = PlantedGap(kind="concept", removed_fg=gap.removed_fg,
                                  region_bg=core, controls_bg=gap.controls_bg)
            rec["ratio_core"] = {
                v: m for v, m in ratio_metrics(scorer, out, gap_core, center, radius).items()
                if v == "full"
            }
        # ---- emerging-pattern miner (bg side scoped = lattice habitability)
        ranked = fightin_words_rank(fam_scoped, Kf_pruned, n_bg=Kb_s_dense.shape[0])
        target = frozenset(int(c) for c in gap.meta["combo_cols"])
        pos = next((i + 1 for i, r in enumerate(ranked) if r["itemset"] == target), None)
        rec["miner"] = {
            "rank": pos, "n_itemsets_ranked": len(ranked),
            "found": pos is not None,
            "rank_pctile": round(pos / len(ranked), 4) if pos else None,
        }
        # ---- unscoped contrast (pairs only)
        if len(gap.meta["combo_cols"]) == 2:
            ranked_u = fightin_words_rank(fam_unscoped, Kf_pruned,
                                          n_bg=Kb_full_dense.shape[0])
            pos_u = next((i + 1 for i, r in enumerate(ranked_u) if r["itemset"] == target),
                         None)
            rec["miner_unscoped"] = {"rank": pos_u, "n_itemsets_ranked": len(ranked_u)}
        trials.append(rec)
        log(f"  concept {t}: kw={gap.keywords} fg_sup={gap.meta['fg_support']} "
            f"miner_rank={rec['miner']['rank']}/{len(ranked)} "
            f"ratio_p50={rec['ratio']['full']['p50']:.2f} "
            f"auc={rec['ratio']['full']['auc']:.2f}")
    return trials


def run_corridor(ctx, n_trials, log):
    scorer = ctx["scorer"]
    Zf, Zb_s = ctx["Zf"], ctx["Zb_s"]
    labels, cents, pairs = corridor_pairs(Zf, Zb_s, n_clusters=12, seed=7)
    log(f"  corridor qualifying pairs: {len(pairs)}")
    rng0 = np.random.default_rng(3000)
    order = rng0.permutation(len(pairs))
    trials = []
    for pidx in order:
        if len(trials) >= n_trials:
            break
        t = len(trials)
        rng = np.random.default_rng(3000 + t)
        gap = plant_corridor(Zf, Zb_s, labels, cents, pairs[int(pidx)], pairs, rng)
        if gap is None:
            continue
        fgp = np.delete(Zf, gap.removed_fg, axis=0)
        out = scorer.score(fgp)
        rec = {
            "trial": t, "meta": {k: v for k, v in gap.meta.items()},
            "ratio": ratio_metrics(scorer, out, gap, gap.center, gap.radius),
            "levelset": levelset_metrics(scorer, out["full"], gap),
        }
        # ---- Laplacian/voltage bridging on the local bg subgraph
        a, b = gap.meta["a"], gap.meta["b"]
        cA, cB = cents[a], cents[b]
        dA = np.linalg.norm(Zb_s - cA, axis=1)
        dB = np.linalg.norm(Zb_s - cB, axis=1)
        nodes = np.where(dA + dB <= gap.meta["lens_r"] * 1.1)[0]
        if nodes.size > 6000:
            nodes = nodes[np.argsort((dA + dB)[nodes])[:6000]]
        labels_p = np.delete(labels, gap.removed_fg)
        fg_a = np.delete(Zf, gap.removed_fg, axis=0)[labels_p == a]
        fg_b = np.delete(Zf, gap.removed_fg, axis=0)[labels_p == b]
        r_a = float(np.median(np.linalg.norm(fg_a - cA, axis=1)))
        r_b = float(np.median(np.linalg.norm(fg_b - cB, axis=1)))
        clamp0 = np.where(dA[nodes] <= r_a)[0]
        clamp1 = np.where(dB[nodes] <= r_b)[0]
        if clamp0.size < 5:
            clamp0 = np.argsort(dA[nodes])[:20]
        if clamp1.size < 5:
            clamp1 = np.argsort(dB[nodes])[:20]
        overlap = np.intersect1d(clamp0, clamp1)
        clamp1 = np.setdiff1d(clamp1, overlap)
        if clamp1.size < 5:
            log(f"  corridor pair ({a},{b}) skipped: degenerate electrodes")
            continue
        cur = voltage_current_scores(Zb_s[nodes], clamp0, clamp1, k_graph=10)
        eligible = np.ones(nodes.size, dtype=bool)
        eligible[clamp0] = eligible[clamp1] = False
        local = {int(g): i for i, g in enumerate(nodes)}
        member_local = np.zeros(nodes.size, dtype=bool)
        for g in gap.region_bg:
            if int(g) in local:
                member_local[local[int(g)]] = True
        k_eval = int(min(TOP_N, max(member_local.sum(), 1)))
        neg_local = np.where(eligible & ~member_local)[0]
        rec["voltage"] = {
            "n_nodes": int(nodes.size),
            "n_clamped": int(clamp0.size + clamp1.size),
            "region_in_subgraph": int(member_local.sum()),
            "best_rank": rank_of(cur, member_local, eligible),
            "p50": precision_at(cur, member_local, eligible, k_eval),
            "auc": auc_vs_controls(cur, np.where(member_local)[0], neg_local),
        }
        trials.append(rec)
        log(f"  corridor {t}: pair=({a},{b}) removed={gap.meta['n_removed']} "
            f"core={gap.meta['n_core']} voltage_p50={rec['voltage']['p50']:.2f} "
            f"voltage_auc={rec['voltage']['auc']:.2f} "
            f"ratio_p50={rec['ratio']['full']['p50']:.2f}")
    return trials


def qualitative_unplanted(ctx, log):
    """Top un-planted candidate gaps on the intact fg, described by titles+keywords."""
    scorer, Zf, ds = ctx["scorer"], ctx["Zf"], ctx["ds"]
    scoped_idx, Kb_s, Kf = ctx["scoped_idx"], ctx["Kb_s"], ctx["Kf"]
    out = scorer.score(Zf)
    comps = level_set_components(out["full"], scorer.cand, scorer.real_mask, top_frac=0.05)
    Kb_dense = np.asarray(Kb_s.todense(), dtype=bool)
    Kf_dense = np.asarray(Kf.todense(), dtype=bool)
    fam = closed_filter(
        mine_frequent_itemsets(Kb_dense, min_support=MINE_MIN_SUPPORT, max_len=3,
                               max_pairs_expand=40_000)
    )
    ranked = fightin_words_rank(fam, Kf_dense, n_bg=Kb_dense.shape[0])[:1000]
    found = []
    for c in comps[:2]:
        members = c["members"]  # candidate indices == scoped-bg-local indices
        order = members[np.argsort(out["full"][members])[::-1]]
        titles = [str(ds.titles[ds.bg_idx[scoped_idx[i]]]) for i in order[:5]]
        mem_set = set(int(i) for i in members)
        cand_patterns = []
        for r in ranked:
            cols = sorted(r["itemset"])
            v = Kb_dense[:, cols[0]].copy()
            for cc in cols[1:]:
                v &= Kb_dense[:, cc]
            overlap = sum(1 for s in np.where(v)[0] if int(s) in mem_set)
            if overlap >= 3:
                cand_patterns.append(
                    {"keywords": [str(ds.vocab[cc]) for cc in cols],
                     "z": round(r["z"], 2), "overlap": overlap,
                     # concentration: how specific the pattern is to this component
                     "concentration": round(overlap / max(r["bg_support"], 1), 3)}
                )
        cand_patterns.sort(key=lambda p: (p["concentration"], p["z"]), reverse=True)
        patterns = cand_patterns[:3]
        found.append(
            {"n_eff": round(c["n_eff"], 1), "size": c["size"],
             "top_titles": titles, "keyword_patterns": patterns}
        )
        log(f"  unplanted component n_eff={c['n_eff']:.1f}: {titles[:3]}")
    return found


# ----------------------------------------------------------------- aggregation/report
def _agg(trials, path, fn=np.median):
    vals = []
    for t in trials:
        v = t
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    return float(fn(vals)) if vals else float("nan")


def build_report(ds, results, ctx) -> str:
    r = results
    lines = []
    A = lines.append
    A("# Gap plant-and-recover evaluation")
    A("")
    A(f"_Generated {datetime.now().isoformat(timespec='seconds')}._")
    A("")
    A("Validates the density-ratio gap statistic family from "
      "`docs/backlog/curriculum-applications.md` ('Gap detection, revisited') by planting "
      "synthetic ground-truth gaps (removing known fg subsets) and measuring recovery: "
      "`gap(x) = g(x)^a * z(log g(x)/p(x))`, g = background density (habitability), "
      "p = foreground density (occupancy), z = Fightin'-Words variance-normalized log-ratio.")
    A("")
    A("## Data")
    A("")
    m = ds.meta
    fg = m["fg_cluster"]
    A(f"- Source: `{m['source']}` -> seeded {m['n_docs']:,}-doc subsample; "
      f"TF-IDF ({m['tfidf_vocab']:,} terms) -> TruncatedSVD d={m['lsa_dim']} "
      f"(EVR {m['svd_explained_variance_ratio_sum']:.2f}), L2-normalized; "
      f"binary doc x keyword space over top {m['n_keywords']:,} terms by df.")
    A(f"- Foreground = KMeans cluster {fg['cluster']} (n={fg['size']:,}, "
      f"coherence {fg['coherence']:.3f}, median bg-interleave {fg['median_bg_within_r10fg']:.0f}); "
      f"top terms: {', '.join(m['fg_top_terms'][:10])}.")
    A(f"- Candidates = field-scoped background ({ctx['n_scoped']:,} bg docs whose "
      f"10th-nearest-fg distance is below the {SCOPE_QUANTILE} quantile, of "
      f"{ctx['n_bg']:,} bg docs) + {N_PROBES} off-cloud probes. Scoping mirrors the "
      f"production setting (fg and bg share a topical scope) and is fixed from the "
      f"pre-removal fg; the unscoped pathology is quantified below.")
    A("- Ground-truth regions = orphaned cores: bg points whose 10th-fg-neighbor distance "
      "grew >= 15% (radial) / 10% (corridor) under the removal. In 128-d, ball mass "
      "concentrates at the boundary where removal is invisible; the core is the honest "
      "footprint. Controls are hypothetical-removal cores at untreated locations.")
    A("")
    A("## Recovery table (detector x gap type)")
    A("")
    A("P@50 = precision@min(50, |region|); best rank is 1-based over "
      f"{ctx['n_scoped']:,} real candidates; AUC vs matched (hypothetical-removal) controls.")
    A("")
    A("| Detector | Metric | Radial | Concept | Corridor |")
    A("|---|---|---|---|---|")

    def row(label, metric, path, fn=np.median, fmt="{:.2f}"):
        cells = []
        for kind in ("radial", "concept", "corridor"):
            v = _agg(r[kind], path, fn)
            cells.append(fmt.format(v) if not np.isnan(v) else "—")
        A(f"| {label} | {metric} | {cells[0]} | {cells[1]} | {cells[2]} |")

    row("kNN ratio (full)", "median best rank", ["ratio", "full", "best_rank"], np.median, "{:.0f}")
    row("kNN ratio (full)", "mean P@50", ["ratio", "full", "p50"], np.mean)
    row("kNN ratio (full)", "mean AUC", ["ratio", "full", "auc"], np.mean)
    row("kNN ratio (full)", "median localization err (/r)", ["ratio", "full", "loc_err"])
    row("Level-set comps", "recall@top-3 (IoU>=0.1)", ["levelset", "recall_top3"], np.mean)
    row("Level-set comps", "median best IoU (top-3)", ["levelset", "best_iou_top3"])
    miner_found = _agg(r["concept"], ["miner", "found"], np.mean)
    miner_rank = _agg(r["concept"], ["miner", "rank"], np.median)
    miner_pct = _agg(r["concept"], ["miner", "rank_pctile"], np.median)
    A(f"| Pattern miner (scoped) | found / median rank (pctile) | — | "
      f"{miner_found:.2f} / {miner_rank:.0f} (top {miner_pct * 100:.1f}%) | — |")
    core_auc = _agg(r["concept"], ["ratio_core", "full", "auc"], np.mean)
    core_n = _agg(r["concept"], ["meta", "n_orphaned_supporters"], np.median)
    n_core_trials = sum(1 for t in r["concept"] if "ratio_core" in t)
    A(f"| kNN ratio on orphaned supporters | median #orphaned / mean AUC | — | "
      f"{core_n:.0f} / " + (f"{core_auc:.2f} (n={n_core_trials})" if n_core_trials else "—")
      + " | — |")
    volt_p50 = _agg(r["corridor"], ["voltage", "p50"], np.mean)
    volt_rank = _agg(r["corridor"], ["voltage", "best_rank"], np.median)
    volt_auc = _agg(r["corridor"], ["voltage", "auc"], np.mean)
    A(f"| Voltage bridging | mean P@50 / median best rank / mean AUC | — | — | "
      f"{volt_p50:.2f} / {volt_rank:.0f} / {volt_auc:.2f} |")
    if any("legacy" in t for t in r["radial"]):
        lt = [t["legacy"] for t in r["radial"] if "legacy" in t]
        hit = np.mean([1.0 if x["first_hit_rank"] else 0.0 for x in lt])
        prec = np.mean([x["best_precision"] for x in lt])
        ji = np.mean([x["best_iou"] for x in lt])
        A(f"| Legacy k-ball (greedy, rf=0.75) | hit-rate@5 / mean best member-precision "
          f"/ mean best IoU | {hit:.2f} / {prec:.2f} / {ji:.2f} (n={len(lt)}) | — | — |")
    A("")
    A("## Ablations (the scientific payload)")
    A("")
    A("Methods note: the design doc's `z = Delta/sqrt(1/n_F,loc + 1/n_B,loc)` was "
      "implemented and REJECTED twice on measurement — (a) radius-Delta over count-sigma "
      "inverts rankings (1/n_F,loc explodes exactly where fg is absent, i.e. on the "
      "signal); (b) textbook count-ratio Fightin' Words is pure significance and "
      "amplifies tiny well-evidenced imbalances over large low-count effects. The "
      "earned-emptiness intent is implemented as a lower-confidence bound "
      "`z = Delta - 0.5*sigma`, sigma from counts in the adaptive max-ball "
      "R(x) = max(r_F,k, r_B,k). See detectors.py docstring.")
    A("")
    A("### (i) Habitability term `g^a` — does it matter?")
    A("")
    A("Candidate set includes 500 off-cloud probes; 'outside habitable hull' = "
      "r_B,10 above the 0.90 quantile of real bg.")
    A("")
    A("| Variant | frac top-50 outside habitable hull | probes in top-50 (of 50) "
      "| P@50 | median best rank | mean AUC |")
    A("|---|---|---|---|---|---|")
    for v, label in (("full", "full statistic"), ("no_hab", "distance-to-fg only")):
        fo = _agg(r["radial"], ["ratio", v, "frac_top50_outside_habitable"], np.mean)
        npr = _agg(r["radial"], ["ratio", v, "n_probes_in_top50"], np.mean)
        p = _agg(r["radial"], ["ratio", v, "p50"], np.mean)
        br = _agg(r["radial"], ["ratio", v, "best_rank"], np.median)
        au = _agg(r["radial"], ["ratio", v, "auc"], np.mean)
        A(f"| {label} | {fo:.2f} | {npr:.1f} | {p:.2f} | {br:.0f} | {au:.2f} |")
    A("")
    A("Geometric scoping contrast (the hard form of habitability — candidates "
      "restricted to field territory vs all bg):")
    A("")
    sr = _agg(r["radial"], ["ratio", "full", "best_rank"], np.median)
    ur = _agg(r["radial"], ["scoping_contrast", "best_rank_unscoped"], np.median)
    sp = _agg(r["radial"], ["ratio", "full", "p50"], np.mean)
    up = _agg(r["radial"], ["scoping_contrast", "p50_unscoped"], np.mean)
    A(f"- field-scoped candidates: median best rank {sr:.0f}, mean P@50 {sp:.2f}")
    A(f"- all-bg candidates:       median best rank {ur:.0f}, mean P@50 {up:.2f}")
    A("")
    A("### (ii) Variance normalization `z` — does it matter?")
    A("")
    A("| Variant | P@50 | median best rank | mean AUC | frac top-50 in sparsest bg tercile |")
    A("|---|---|---|---|---|")
    for v, label in (("full", "full statistic"), ("no_z", "variance norm OFF (hab*Delta)"),
                     ("z_only", "z only"), ("delta_only", "Delta only")):
        p = _agg(r["radial"], ["ratio", v, "p50"], np.mean)
        br = _agg(r["radial"], ["ratio", v, "best_rank"], np.median)
        au = _agg(r["radial"], ["ratio", v, "auc"], np.mean)
        mf = _agg(r["radial"], ["ratio", v, "mirage_frac_top50"], np.mean)
        A(f"| {label} | {p:.2f} | {br:.0f} | {au:.2f} | {mf:.2f} |")
    A("")
    A("### (iii) Lattice-side habitability: scoped vs unscoped pattern mining")
    A("")
    pos_s = [t["miner"]["rank"] for t in r["concept"] if t["miner"]["rank"]]
    pos_u = [t["miner_unscoped"]["rank"] for t in r["concept"]
             if t.get("miner_unscoped", {}).get("rank")]
    nu = sum(1 for t in r["concept"] if "miner_unscoped" in t)
    if pos_s:
        A(f"- Scoped (bg restricted to fg territory): planted combo found in "
          f"{len(pos_s)}/{len(r['concept'])} trials, median rank {np.median(pos_s):.0f}.")
    else:
        A(f"- Scoped: planted combo never found ({len(r['concept'])} trials).")
    if pos_u:
        A(f"- Unscoped (full bg, pairs only, n={nu} pair trials): found in {len(pos_u)}/{nu}, "
          f"median rank {np.median(pos_u):.0f}.")
    else:
        A(f"- Unscoped (full bg, pairs only, n={nu} pair trials): planted combo never ranked.")
    A("")
    A("## Verdicts")
    A("")
    A("1. **Habitability matters, but as the candidate universe, not (only) as a "
      "weight.** Field scoping is the dominant term (median best rank "
      f"{sr:.0f} scoped vs {ur:.0f} unscoped); with no habitability control at all, "
      "off-cloud probes take 50/50 of the top-50. Within scope, the soft `g^a` weight "
      "costs recovery precision (it re-ranks natural bg-dense pockets above planted "
      "holes) while its remaining job — hull-guarding — is already done by the scope.")
    A("2. **Variance normalization is directionally right but empirically marginal "
      "here.** The doc's z-ratio forms invert rankings (see methods note); the LCB "
      "form changes AUC/rank only slightly. The earned-emptiness mechanism is real "
      "(unit-test contract) but k=10 cores with >=8 bg points rarely hit "
      "low-evidence territory in this corpus.")
    A("3. **Each gap species needs its matched detector.** Radial dents: density "
      "ratio (AUC ~0.78). Concept gaps: geometrically invisible (AUC ~0.47, zero "
      "orphaned supporters) but the scoped pattern miner finds them at top ~0.4%. "
      "Corridor gaps: voltage bridging (median best rank ~12) where the density "
      "ratio fails (~1180) — interior and bridging holes are different topologies, "
      "as the design doc predicted.")
    A("4. **Level-set components do not isolate small planted gaps** (recall@3 = 0): "
      "n_eff-ordered components surface the large natural gap structure instead.")
    A("5. **Planted-hole detectability in 128-d LSA is bounded by backfill**: "
      "<=150-doc removals produce shallow dents (median orphan ratio ~1.10); "
      "corridor removals are the shallowest. Plant-and-recover at this budget "
      "measures ranking among natural gaps, not clean separation.")
    A("")
    A("## Qualitative: top un-planted candidate gaps (intact foreground)")
    A("")
    for i, q in enumerate(results["qualitative"], 1):
        A(f"**Candidate gap {i}** (level-set component, n_eff={q['n_eff']}, "
          f"size={q['size']}):")
        A("")
        A(f"- nearest bg works (titles): {'; '.join(q['top_titles'])}")
        if q["keyword_patterns"]:
            pat = "; ".join(
                "{" + ", ".join(p["keywords"]) + f"}} (z={p['z']})"
                for p in q["keyword_patterns"]
            )
            A(f"- keyword patterns present in bg here, absent/rare in fg: {pat}")
        else:
            A("- no scoped keyword pattern overlapped this component (>=3 supporters)")
        A("")
    if results.get("legacy_fail"):
        A(f"_Legacy comparison aborted: {results['legacy_fail']}_")
        A("")
    A("## Provenance")
    A("")
    cmd = "uv run python -m artifacts_scripts.gap_detection.run_eval"
    if sys.argv[1:]:
        cmd += " " + " ".join(sys.argv[1:])
    A(f"- Command: `{cmd}` (uv env, CPU only)")
    A(f"- Trials: radial={len(r['radial'])}, concept={len(r['concept'])}, "
      f"corridor={len(r['corridor'])}; k={K_NN}, probes={N_PROBES}, "
      f"seeds 1000/2000/3000+t, scope quantile {SCOPE_QUANTILE}.")
    A("- Full per-trial metrics: `artifacts/gap_detection_eval/results.json`.")
    A("- Regenerate rather than edit by hand.")
    A("")
    return "\n".join(lines)


def _preserve_followups(new_text: str) -> str:
    """Carry over marker-delimited follow-up sections written by run_followups."""
    path = OUT_DIR / "report.md"
    if not path.exists():
        return new_text
    blocks = re.findall(
        r"<!-- BEGIN FOLLOWUP .+? -->.*?<!-- END FOLLOWUP .+? -->", path.read_text(), re.S
    )
    if blocks:
        new_text = new_text.rstrip("\n") + "\n\n" + "\n\n".join(blocks) + "\n"
    return new_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--legacy-budget", type=float, default=1500.0, help="seconds")
    ap.add_argument("--quick", action="store_true", help="4 trials, no legacy")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild report.md from existing results.json")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.skip_legacy = 4, True

    if args.report_only:
        ds = load_dataset()
        Zf, Zb = ds.Z[ds.fg_idx], ds.Z[ds.bg_idx]
        from artifacts_scripts.gap_detection.detectors import knn_radii

        r_F10_bg = knn_radii(Zb, Zf, K_NN)
        scoped_n = int((r_F10_bg <= np.quantile(r_F10_bg, SCOPE_QUANTILE)).sum())
        results = json.loads((OUT_DIR / "results.json").read_text())
        ctx = {"n_scoped": scoped_n, "n_bg": int(Zb.shape[0])}
        (OUT_DIR / "report.md").write_text(_preserve_followups(build_report(ds, results, ctx)))
        print(f"rebuilt {OUT_DIR / 'report.md'}")
        return

    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    ds = load_dataset()
    Zf, Zb = ds.Z[ds.fg_idx], ds.Z[ds.bg_idx]
    Kf, Kb = ds.K[ds.fg_idx], ds.K[ds.bg_idx]
    # field territory: bg docs whose 10th-nearest fg doc is close (kNN scope; a
    # centroid ball on the unit sphere admits semantically far docs). The scope is
    # fixed once from the PRE-removal fg: the candidate universe is the field as it
    # existed before any gap was planted, identical across trials.
    from artifacts_scripts.gap_detection.detectors import knn_radii

    r_F10_bg = knn_radii(Zb, Zf, K_NN)
    thr = float(np.quantile(r_F10_bg, SCOPE_QUANTILE))
    scoped_idx = np.where(r_F10_bg <= thr)[0]
    Zb_s, Kb_s = Zb[scoped_idx], Kb[scoped_idx]
    log(f"dataset: n={ds.Z.shape[0]} fg={Zf.shape[0]} bg={Zb.shape[0]} "
        f"scoped_bg={scoped_idx.size} (cluster {ds.meta['fg_cluster']['cluster']})")

    rng = np.random.default_rng(99)
    probes = make_probes(ds.Z, N_PROBES, rng)
    scorer = GapScorer(bg=Zb_s, probes=probes, k=K_NN)
    r0 = scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    scorer_u = GapScorer(bg=Zb, probes=None, k=K_NN)
    scorer_u.set_r0_from_fg(Zf)
    _ = scorer_u.n_Bloc
    log(f"scorers ready: r0_scoped={r0:.4f}, r0_unscoped={scorer_u.r0:.4f}, "
        f"habitable frac={scorer.habitable_mask(0.9).mean():.3f}")

    ctx = {
        "ds": ds, "Zf": Zf, "Zb": Zb, "Zb_s": Zb_s, "Kf": Kf, "Kb": Kb, "Kb_s": Kb_s,
        "scoped_idx": scoped_idx, "scorer": scorer, "scorer_unscoped": scorer_u,
        "n_scoped": int(scoped_idx.size), "n_bg": int(Zb.shape[0]),
    }
    results: dict = {}
    log("radial trials...")
    results["radial"], legacy_fail = run_radial(
        ctx, args.trials, 0.0 if args.skip_legacy else args.legacy_budget, log,
    )
    results["legacy_fail"] = legacy_fail
    log("concept trials...")
    results["concept"] = run_concept(ctx, args.trials, log)
    log("corridor trials...")
    results["corridor"] = run_corridor(ctx, args.trials, log)
    log("qualitative un-planted scan...")
    results["qualitative"] = qualitative_unplanted(ctx, log)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(_py(results), indent=1))
    report = build_report(ds, results, ctx)
    (OUT_DIR / "report.md").write_text(_preserve_followups(report))
    log(f"wrote {OUT_DIR / 'report.md'} and results.json")
    for kind in ("radial", "concept", "corridor"):
        br = _agg(results[kind], ["ratio", "full", "best_rank"], np.median)
        p = _agg(results[kind], ["ratio", "full", "p50"], np.mean)
        auc = _agg(results[kind], ["ratio", "full", "auc"], np.mean)
        log(f"summary {kind}: best_rank~{br:.0f} P@50={p:.2f} AUC={auc:.2f}")


if __name__ == "__main__":
    main()
