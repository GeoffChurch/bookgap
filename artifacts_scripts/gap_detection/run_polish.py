"""Polish-as-Track-1-phase-1 runner (P1-P4, 2026-06-11 dispatch).

P1  Miner cross-species sweep (radial + corridor, conjunction-aware corridors).
P2  Level-set residualization (differential component scoring, all species).
P3a Voltage all-pairs change field (species-agnostic anomaly map).
P3b Commute-distance (effective-resistance) geometry for concept + radial.
P4  Low-df miner lattice (tail-augmented itemset space + walk-geometry rerun).

Discipline: replants the SAME banked trials (seeds 1000/2000/3000+t), asserts
per-trial plant identity AND reproduces at least one banked rank before each
measurement. Writes results.json keys task_p1..task_p4 and marker-delimited
report.md sections TASK_P1..TASK_P4.

Run (CPU only; BLAS capped at 8 threads):
  uv run python -m artifacts_scripts.gap_detection.run_polish --task p1
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
import scipy.sparse as sp

from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    auc_vs_controls,
    closed_filter,
    level_set_components,
    mine_frequent_itemsets,
    precision_at,
    rank_of,
    voltage_current_scores,
)
from artifacts_scripts.gap_detection.polish import (
    attributable_best_rank,
    attribution_stats,
    bipartite_graph,
    centroid_adjacency,
    cluster_vocab,
    commute_embedding,
    component_recovery,
    cross_conjunctions,
    doc_pattern_scores,
    flow_share,
    gaussian_knn_graph,
    itemset_support_matrix,
    low_df_tail,
    mine_tail_family,
    order_components_by_mass,
    pocket_status,
    rank_family_fast,
    rebuild_tfidf,
    residual_scores,
    voltage_on_graph,
)
from artifacts_scripts.gap_detection.run_eval import (
    K_NN,
    MINE_MIN_SUPPORT,
    _member_mask,
    _py,
)
from artifacts_scripts.gap_detection.run_followups import (
    build_ctx,
    load_banked,
    med,
    replant_concept,
    replant_radial,
    update_report,
    update_results,
)
from artifacts_scripts.gap_detection.synth import corridor_pairs, orphaned_core, plant_corridor
from artifacts_scripts.gap_detection.walk_geometry import walk_embedding

_T0 = time.monotonic()
N_SCOPED_BG = 7139  # asserted against ctx at runtime


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- corridor replanting
def corridor_structure(ctx: dict):
    return corridor_pairs(ctx["Zf"], ctx["Zb_s"], n_clusters=12, seed=7)


def replant_corridor(ctx: dict, banked: dict, n_trials: int, structure=None):
    """Yield (t, gap, info); mirrors run_eval.run_corridor's loop exactly
    (including the degenerate-electrode skip) and asserts per-trial identity.

    info: subgraph nodes (bg-local), local clamp0/clamp1, labels/cents.
    """
    Zf, Zb_s = ctx["Zf"], ctx["Zb_s"]
    labels, cents, pairs = structure or corridor_structure(ctx)
    rng0 = np.random.default_rng(3000)
    order = rng0.permutation(len(pairs))
    count = 0
    for pidx in order:
        if count >= n_trials:
            break
        t = count
        rng = np.random.default_rng(3000 + t)
        gap = plant_corridor(Zf, Zb_s, labels, cents, pairs[int(pidx)], pairs, rng)
        if gap is None:
            continue
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
        clamp1 = np.setdiff1d(clamp1, np.intersect1d(clamp0, clamp1))
        if clamp1.size < 5:
            continue
        bm = banked["corridor"][t]["meta"]
        got = (a, b, gap.meta["n_removed"], gap.meta["n_core"])
        want = (bm["a"], bm["b"], bm["n_removed"], bm["n_core"])
        if got != want:
            raise RuntimeError(f"corridor trial {t} replant mismatch: {got} != {want}")
        count += 1
        yield t, gap, {"nodes": nodes, "clamp0": clamp0, "clamp1": clamp1,
                       "labels": labels, "cents": cents, "pairs": pairs}


def reproduce_banked_voltage(ctx: dict, banked: dict, structure) -> None:
    """Identity gate: re-solve trial 0's banked voltage rank before measuring."""
    for t, gap, info in replant_corridor(ctx, banked, 1, structure):
        nodes, clamp0, clamp1 = info["nodes"], info["clamp0"], info["clamp1"]
        cur = voltage_current_scores(ctx["Zb_s"][nodes], clamp0, clamp1, k_graph=10)
        eligible = np.ones(nodes.size, dtype=bool)
        eligible[clamp0] = eligible[clamp1] = False
        local = {int(g): i for i, g in enumerate(nodes)}
        member_local = np.zeros(nodes.size, dtype=bool)
        for g in gap.region_bg:
            if int(g) in local:
                member_local[local[int(g)]] = True
        got = rank_of(cur, member_local, eligible)
        want = banked["corridor"][t]["voltage"]["best_rank"]
        if got != want:
            raise RuntimeError(f"voltage reproduction failed: rank {got} != banked {want}")
        log(f"identity: banked corridor voltage rank reproduced (trial {t}: {got})")


# ----------------------------------------------------------------- miner assets (P1/P4)
def miner_assets(ctx: dict, banked: dict) -> dict:
    """Mine the scoped family once; build support matrices; assert identity
    (concept trial 0 planted rank == banked, family sizes match)."""
    Kb_s_dense = np.asarray(ctx["Kb_s"].todense(), dtype=bool)
    Kf_dense = np.asarray(ctx["Kf"].todense(), dtype=bool)
    fam = closed_filter(
        mine_frequent_itemsets(Kb_s_dense, min_support=MINE_MIN_SUPPORT, max_len=3,
                               max_pairs_expand=40_000)
    )
    keys = list(fam.keys())
    key_index = {S: i for i, S in enumerate(keys)}
    sup_bg = np.array([fam[S] for S in keys], dtype=np.int64)
    M_fg = itemset_support_matrix(keys, Kf_dense)
    M_bg = itemset_support_matrix(keys, Kb_s_dense).tocsc()
    n_bg = Kb_s_dense.shape[0]
    assets = {
        "fam": fam, "keys": keys, "key_index": key_index, "sup_bg": sup_bg,
        "M_fg": M_fg, "M_bg": M_bg, "n_bg": n_bg,
        "Kb_s_dense": Kb_s_dense, "Kf_dense": Kf_dense,
    }
    # identity: trial-0 planted concept rank must reproduce through the fast ranker
    for t, gap, keep in replant_concept(ctx, banked, 1):
        ranked = rank_family_fast(keys, sup_bg, M_fg, keep, n_bg=n_bg)
        target = frozenset(int(c) for c in gap.meta["combo_cols"])
        pos = next((i + 1 for i, r in enumerate(ranked) if r["itemset"] == target), None)
        want = banked["concept"][t]["miner"]["rank"]
        n_want = banked["concept"][t]["miner"]["n_itemsets_ranked"]
        if pos != want or len(ranked) != n_want:
            raise RuntimeError(
                f"miner identity failed: rank {pos}/{len(ranked)} != banked {want}/{n_want}")
        log(f"identity: banked miner rank reproduced (trial {t}: {pos}/{len(ranked)})")
    return assets


def miner_trial(
    assets: dict, removed_fg: np.ndarray, gap, controls: np.ndarray,
    extra_fam: dict | None = None, Kf_aug: np.ndarray | None = None,
    Kb_aug: np.ndarray | None = None,
) -> dict:
    """Attribution + doc-layer metrics of the (possibly extended) miner for one trial."""
    keys, key_index, sup_bg = assets["keys"], assets["key_index"], assets["sup_bg"]
    M_fg, M_bg, n_bg = assets["M_fg"], assets["M_bg"], assets["n_bg"]
    if extra_fam:
        new = [S for S in extra_fam if S not in key_index]
        if new:
            keys = keys + new
            key_index = {S: i for i, S in enumerate(keys)}
            sup_bg = np.concatenate([sup_bg, np.array([extra_fam[S] for S in new])])
            M_fg = sp.vstack([M_fg, itemset_support_matrix(new, Kf_aug)]).tocsr()
            M_bg = sp.vstack([M_bg, itemset_support_matrix(new, Kb_aug)]).tocsc()
    n_fg = M_fg.shape[1]
    keep = np.ones(n_fg, dtype=bool)
    keep[removed_fg] = False
    ranked = rank_family_fast(keys, sup_bg, M_fg, keep, n_bg=n_bg)
    n_pre, n_rem = attribution_stats(M_fg, ~keep)
    pos8, row8 = attributable_best_rank(ranked, key_index, n_pre, n_rem, 10, 0.8)
    pos5, _ = attributable_best_rank(ranked, key_index, n_pre, n_rem, 10, 0.5)
    sc = doc_pattern_scores(ranked, key_index, M_bg, agg="max")
    sc_sum = doc_pattern_scores(ranked, key_index, M_bg, agg="sum")
    member = _member_mask(M_bg.shape[1], gap.region_bg)
    every = np.ones(M_bg.shape[1], dtype=bool)
    k_eval = int(min(50, gap.region_bg.size))
    return {
        "n_ranked": len(ranked),
        "attrib_rank_r08": pos8,
        "attrib_rank_r05": pos5,
        "attrib_itemset_cols": sorted(row8["itemset"]) if row8 else None,
        "attrib_n_pre": row8["n_pre"] if row8 else None,
        "attrib_n_removed": row8["n_removed"] if row8 else None,
        "attrib_bg_support": row8["bg_support"] if row8 else None,
        "doc_best_rank": rank_of(sc, member, every),
        "doc_p50": precision_at(sc, member, every, k_eval),
        "doc_auc": auc_vs_controls(sc, gap.region_bg, controls),
        "doc_auc_sum": auc_vs_controls(sc_sum, gap.region_bg, controls),
        "_ranked": ranked, "_keys": keys, "_key_index": key_index,
        "_n_pre": n_pre, "_n_rem": n_rem, "_doc_scores": sc,
    }


def strip_private(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# =================================================================== task P1
NULL_PREDICTION_P1 = (
    "Null prediction on record (before measurement): Task C's geometry/keyword "
    "orthogonality predicts the miner is BLIND on radial plants (an LSA-ball removal "
    "should leave no coherent keyword-lattice signature). For corridors the sharper "
    "hypothesis: a removed corridor is a missing cross-cluster keyword CONJUNCTION, "
    "so a conjunction-aware miner should see it."
)


def task_p1(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    assert ctx["n_scoped"] == N_SCOPED_BG
    log(NULL_PREDICTION_P1)
    structure = corridor_structure(ctx)
    reproduce_banked_voltage(ctx, banked, structure)
    assets = miner_assets(ctx, banked)
    Kf_dense, Kb_s_dense = assets["Kf_dense"], assets["Kb_s_dense"]
    vocab = ctx["ds"].vocab

    # ---- radial sweep
    radial = []
    for t, gap, _fgp in replant_radial(ctx, banked, args.trials):
        rec = {"trial": t,
               **strip_private(miner_trial(assets, gap.removed_fg, gap,
                                           np.concatenate(gap.controls_bg)))}
        radial.append(rec)
        log(f"P1 radial {t}: attrib_rank={rec['attrib_rank_r08']} "
            f"(r0.5 {rec['attrib_rank_r05']}) doc_rank={rec['doc_best_rank']} "
            f"doc_auc={rec['doc_auc']:.2f}")

    # ---- corridor sweep (conjunction-aware)
    labels = structure[0]
    corridor = []
    for t, gap, info in replant_corridor(ctx, banked, args.trials, structure):
        a, b = gap.meta["a"], gap.meta["b"]
        voc_a = cluster_vocab(Kf_dense, labels == a, n_top=40)
        voc_b = cluster_vocab(Kf_dense, labels == b, n_top=40)
        conj = cross_conjunctions(voc_a, voc_b, Kb_s_dense, min_support=10, cap=1600)
        m = miner_trial(assets, gap.removed_fg, gap, np.concatenate(gap.controls_bg),
                        extra_fam=conj, Kf_aug=Kf_dense, Kb_aug=Kb_s_dense)
        # is the best attributable itemset a cross-cluster conjunction?
        is_conj = None
        labels_words = None
        if m["attrib_itemset_cols"] is not None:
            cols = m["attrib_itemset_cols"]
            sa, sb = set(voc_a.tolist()), set(voc_b.tolist())
            is_conj = any(c in sa for c in cols) and any(c in sb for c in cols)
            labels_words = [str(vocab[c]) for c in cols]
        # doc-layer rank inside the voltage home subgraph (universe-matched row)
        nodes, c0, c1 = info["nodes"], info["clamp0"], info["clamp1"]
        elig_sub = np.zeros(N_SCOPED_BG, dtype=bool)
        elig_sub[nodes] = True
        elig_sub[nodes[c0]] = elig_sub[nodes[c1]] = False
        member = _member_mask(N_SCOPED_BG, gap.region_bg)
        rec = {"trial": t, "pair": [int(a), int(b)],
               "n_conj_mined": len(conj),
               "is_cross_cluster_conjunction": is_conj,
               "attrib_keywords": labels_words,
               "doc_best_rank_homesub": rank_of(m["_doc_scores"], member, elig_sub),
               "voltage_banked_rank": banked["corridor"][t]["voltage"]["best_rank"],
               **strip_private(m)}
        corridor.append(rec)
        log(f"P1 corridor {t}: pair=({a},{b}) conj_mined={len(conj)} "
            f"attrib_rank={rec['attrib_rank_r08']} is_conj={is_conj} "
            f"doc_rank={rec['doc_best_rank']} (homesub {rec['doc_best_rank_homesub']}, "
            f"voltage {rec['voltage_banked_rank']}) doc_auc={rec['doc_auc']:.2f}")

    # ---- concept anchor (same metric, for cross-species comparability)
    concept = []
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        m = miner_trial(assets, np.where(~keep)[0], gap, np.concatenate(gap.controls_bg))
        target = frozenset(int(c) for c in gap.meta["combo_cols"])
        planted = next((i + 1 for i, r in enumerate(m["_ranked"]) if r["itemset"] == target),
                       None)
        rec = {"trial": t, "planted_rank": planted,
               "banked_planted_rank": banked["concept"][t]["miner"]["rank"],
               **strip_private(m)}
        if planted != rec["banked_planted_rank"]:
            raise RuntimeError(f"concept {t}: planted rank {planted} != banked")
        concept.append(rec)
        log(f"P1 concept {t}: planted={planted} attrib_rank={rec['attrib_rank_r08']} "
            f"doc_rank={rec['doc_best_rank']} doc_auc={rec['doc_auc']:.2f}")

    def agg(trials):
        found = [tr for tr in trials if tr["attrib_rank_r08"] is not None]
        return {
            "n_trials": len(trials),
            "found_rate_r08": len(found) / max(len(trials), 1),
            "attrib_rank_median": (float(np.median([tr["attrib_rank_r08"] for tr in found]))
                                   if found else None),
            "found_rate_r05": np.mean([tr["attrib_rank_r05"] is not None for tr in trials]),
            "attrib_rank_r05_median": (
                float(np.median([tr["attrib_rank_r05"] for tr in trials
                                 if tr["attrib_rank_r05"] is not None]))
                if any(tr["attrib_rank_r05"] is not None for tr in trials) else None),
            "doc_rank_median": med(trials, "doc_best_rank"),
            "doc_auc_mean": med(trials, "doc_auc", fn=np.mean),
            "doc_auc_sum_mean": med(trials, "doc_auc_sum", fn=np.mean),
            "doc_p50_mean": med(trials, "doc_p50", fn=np.mean),
        }

    summary = {
        "null_prediction": NULL_PREDICTION_P1,
        "config": {"family": "scoped closed itemsets, min_support 30, max_len 3 "
                             "(38,034 itemsets; 36,199 ranked at min_len 2)",
                   "attribution": "itemset with pre-removal fg support >= 10 losing "
                                  ">= 80% of it to the removal (sens: >= 50%)",
                   "doc_layer": "per-doc max z over supported itemsets (sens: sum z+)",
                   "conjunctions": "corridor: cross-cluster pairs, 40x40 flanking "
                                   "vocabularies (log-odds), scoped-bg support >= 10, "
                                   "cap 1600/pair, merged into the family"},
        "radial": agg(radial),
        "corridor": {**agg(corridor),
                     "conj_frac_of_found": (
                         float(np.mean([bool(tr["is_cross_cluster_conjunction"])
                                        for tr in corridor
                                        if tr["attrib_rank_r08"] is not None]))
                         if any(tr["attrib_rank_r08"] is not None for tr in corridor)
                         else None),
                     "doc_rank_homesub_median": med(corridor, "doc_best_rank_homesub"),
                     "voltage_banked_rank_median": med(corridor, "voltage_banked_rank")},
        "concept": {**agg(concept),
                    "planted_rank_median": med(concept, "planted_rank")},
    }
    if not args.dry:
        update_results("task_p1", {"summary": summary,
                                   "radial": radial, "corridor": corridor,
                                   "concept": concept})
        update_report("TASK_P1", report_p1(summary, corridor))
    log("P1 radial:   " + json.dumps(_py(summary["radial"])))
    log("P1 corridor: " + json.dumps(_py(summary["corridor"])))
    log("P1 concept:  " + json.dumps(_py(summary["concept"])))


def report_p1(s: dict, corridor: list[dict]) -> list[str]:
    d = datetime.now().date().isoformat()
    r, c, k = s["radial"], s["corridor"], s["concept"]

    def fr(x):
        return "—" if x is None else f"{x:.0f}"

    conj_rows = [tr for tr in corridor if tr["attrib_rank_r08"] is not None
                 and tr["attrib_keywords"]][:3]
    L = [
        f"## Follow-up Task P1 — miner cross-species sweep ({d})",
        "",
        s["null_prediction"],
        "",
        "Same 16 trials per species (identity asserted; banked voltage rank + miner rank",
        "reproduced before measurement). The miner has no planted combo to look for on",
        "geometric plants, so the species-agnostic analog is ATTRIBUTION: the best rank of",
        "any itemset whose pre-removal fg support (>= 10) the removal wiped (>= 80%; 50%",
        "sensitivity). Doc layer: each scoped-bg doc scores the max Fightin'-Words z over",
        "itemsets it supports; ranks among the 7,139 real candidates. Corridor trials also",
        "mine cross-cluster conjunctions (40x40 flanking-cluster vocabularies by log-odds,",
        "scoped-bg support >= 10, <= 1600 candidates/pair) into the family.",
        "",
        "| Species | found (>=80% wipe) | median attrib rank | found (>=50%) | doc-layer "
        "median rank | doc AUC | home detector |",
        "|---|---|---|---|---|---|---|",
        f"| radial | {r['found_rate_r08']:.2f} | {fr(r['attrib_rank_median'])} "
        f"| {r['found_rate_r05']:.2f} | {r['doc_rank_median']:.0f} | {r['doc_auc_mean']:.2f} "
        f"| fused r_F: rank-real 6.5 / AUC 0.83 |",
        f"| corridor | {c['found_rate_r08']:.2f} | {fr(c['attrib_rank_median'])} "
        f"| {c['found_rate_r05']:.2f} | {c['doc_rank_median']:.0f} | {c['doc_auc_mean']:.2f} "
        f"| voltage: median rank {c['voltage_banked_rank_median']:.0f} (subgraph; "
        f"miner-in-subgraph {c['doc_rank_homesub_median']:.0f}) |",
        f"| concept (anchor) | {k['found_rate_r08']:.2f} | {fr(k['attrib_rank_median'])} "
        f"| {k['found_rate_r05']:.2f} | {k['doc_rank_median']:.0f} | {k['doc_auc_mean']:.2f} "
        f"| miner planted-combo rank {k['planted_rank_median']:.0f} |",
        "",
        "Cross-cluster conjunction share of corridor finds: "
        + ("—" if c["conj_frac_of_found"] is None else f"{c['conj_frac_of_found']:.2f}")
        + "; median conjunctions mined/pair: "
        + f"{int(np.median([tr['n_conj_mined'] for tr in corridor]))}.",
        "",
    ]
    if conj_rows:
        L.append("Attributed corridor patterns (keyword names, first 3 found trials):")
        L.append("")
        for tr in conj_rows:
            L.append(f"- trial {tr['trial']} pair {tuple(tr['pair'])}: "
                     f"{{{', '.join(tr['attrib_keywords'])}}} "
                     f"(rank {tr['attrib_rank_r08']}, bg support {tr['attrib_bg_support']}, "
                     f"cross-cluster: {tr['is_cross_cluster_conjunction']})")
        L.append("")
    L += [
        "**Verdict.** The stated null prediction is PARTIALLY REFUTED on radial and the",
        "corridor-conjunction hypothesis is REFUTED. (i) Radial: the miner is NOT blind —",
        f"{r['found_rate_r08']:.0%} of ball removals wipe a real keyword pattern "
        f"(median attributable rank {fr(r['attrib_rank_median'])}; 16/16 at the 50% wipe",
        f"sensitivity, rank {fr(r['attrib_rank_r05_median'])}) and the doc layer carries "
        f"causal signal (AUC {r['doc_auc_mean']:.2f}",
        "vs matched controls) — LSA balls are topically coherent, so removing one deletes",
        "co-occurrence patterns Task C's orthogonality argument missed. But it is far below",
        f"the home detector (doc rank {r['doc_rank_median']:.0f} vs fused 6.5; AUC "
        f"{r['doc_auc_mean']:.2f} vs 0.83). (ii) Corridor: removals",
        f"wipe >= 80% of a pattern in only {sum(1 for tr in corridor if tr['attrib_rank_r08'] is not None)}/16 trials and the signatures rank DEEP",
        f"(median {fr(c['attrib_rank_median'])}); cross-cluster conjunctions DO exist "
        f"(~{int(np.median([tr['n_conj_mined'] for tr in corridor]))}/pair mined) and half "
        "of the found signatures are conjunctions,",
        "but their scoped-bg support (~10-30) keeps their Fightin'-Words z low: the corridor",
        "lattice signature is real but evidentially thin — no human-readable-corridor payoff",
        f"at this support budget (doc rank {c['doc_rank_median']:.0f}, in-subgraph "
        f"{c['doc_rank_homesub_median']:.0f}, vs voltage {c['voltage_banked_rank_median']:.0f}).",
        "(iii) Concept anchor reproduces exactly (16/16, median 128.5). DECISION: a",
        "miner-led composite is REJECTED — the miner generalizes weakly (radial) or not",
        "usefully (corridor); real fusion of species-matched detectors stands, with the",
        "miner's radial attribution available as cheap corroborating evidence.",
        "",
    ]
    return L


# =================================================================== task P2
def _levelset_block(comps, region) -> dict:
    """Replicates run_eval.levelset_metrics' fields for banked-identity checks."""
    from artifacts_scripts.gap_detection.detectors import iou

    top3 = comps[:3]
    best_iou = max((iou(c["members"], region) for c in top3), default=0.0)
    return {"n_components": len(comps), "recall_top3": bool(best_iou >= 0.1),
            "best_iou_top3": round(best_iou, 4)}


def task_p2(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    scorer, Zf = ctx["scorer"], ctx["Zf"]
    before = scorer.score(Zf)
    # natural pockets = top-2 absolute components on the intact fg (banked qualitative)
    comps_abs = level_set_components(before["full"], scorer.cand, scorer.real_mask,
                                     top_frac=0.05)
    want = [(round(q["n_eff"], 1), q["size"]) for q in banked["qualitative"][:2]]
    got = [(round(c["n_eff"], 1), c["size"]) for c in comps_abs[:2]]
    if got != want:
        raise RuntimeError(f"pocket identity failed: {got} != banked {want}")
    pockets = [np.asarray(comps_abs[0]["members"]), np.asarray(comps_abs[1]["members"])]
    log(f"identity: natural pockets reproduced (n_eff/size {got})")

    def run_trial(sp: str, t: int, gap, fgp: np.ndarray) -> dict:
        after = scorer.score(fgp)
        # banked-identity: absolute level-set baseline must reproduce per trial
        comps_after = level_set_components(after["full"], scorer.cand, scorer.real_mask,
                                           top_frac=0.05)
        blk = _levelset_block(comps_after, gap.region_bg)
        bk = banked[sp][t]["levelset"]
        if (blk["n_components"] != bk["n_components"]
                or blk["recall_top3"] != bk["recall_top3"]
                or abs(blk["best_iou_top3"] - bk["best_iou_top3"]) > 1e-9):
            raise RuntimeError(f"{sp} {t}: levelset baseline mismatch {blk} vs {bk}")
        d_full = residual_scores(before, after, "full")
        d_delta = residual_scores(before, after, "delta_only")
        comps_d = level_set_components(d_full, scorer.cand, scorer.real_mask, top_frac=0.05)
        comps_dd = level_set_components(d_delta, scorer.cand, scorer.real_mask,
                                        top_frac=0.05)
        # tight-budget sensitivity: a differential field's change set is SMALL, so the
        # 5% selection admits ~300 noise points that dilute IoU; 1% matches plant scale
        comps_d01 = level_set_components(d_full, scorer.cand, scorer.real_mask,
                                         top_frac=0.01)
        rec = {
            "trial": t,
            "abs_baseline": blk,
            "neff": component_recovery(comps_d, gap.region_bg),
            "mass": component_recovery(order_components_by_mass(comps_d, d_full),
                                       gap.region_bg),
            "delta_sens": component_recovery(comps_dd, gap.region_bg),
            "neff_tf01": component_recovery(comps_d01, gap.region_bg),
            "pockets": [pocket_status(comps_d, d_full, p, scorer.real_mask)
                        for p in pockets],
        }
        log(f"P2 {sp} {t}: abs_recall={blk['recall_top3']} -> "
            f"neff rank={rec['neff']['rank']} iou3={rec['neff']['best_iou_top3']:.2f} "
            f"mass rank={rec['mass']['rank']} tf01 rank={rec['neff_tf01']['rank']} "
            f"iou3={rec['neff_tf01']['best_iou_top3']:.2f} "
            f"pocket ranks={[p['rank'] for p in rec['pockets']]} "
            f"pocket members@top={[p['members_in_topfrac'] for p in rec['pockets']]}")
        return rec

    results: dict[str, list] = {"radial": [], "concept": [], "corridor": []}
    for t, gap, fgp in replant_radial(ctx, banked, args.trials):
        results["radial"].append(run_trial("radial", t, gap, fgp))
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        results["concept"].append(run_trial("concept", t, gap, Zf[keep]))
    structure = corridor_structure(ctx)
    for t, gap, _info in replant_corridor(ctx, banked, args.trials, structure):
        results["corridor"].append(run_trial("corridor", t, gap,
                                             np.delete(Zf, gap.removed_fg, axis=0)))

    def agg(trials: list, key: str) -> dict:
        ranks = [tr[key]["rank"] for tr in trials if tr[key]["rank"] is not None]
        return {
            "recall_top3": float(np.mean([tr[key]["recall_top3"] for tr in trials])),
            "found_any": len(ranks) / max(len(trials), 1),
            "rank_median_found": float(np.median(ranks)) if ranks else None,
            "best_iou_top3_median": med(trials, key, "best_iou_top3"),
        }

    summary: dict = {"config": {
        "differential": "score_after - score_before per candidate; before = intact-fg "
                        "scores (the matched hypothetical-removal baseline the harness "
                        "computes); components on the differential field, top_frac 0.05",
        "orderings": "n_eff (banked contract) and total positive differential mass",
        "near_oracle_note": "for radial/corridor the ground-truth orphan core is "
                            "DEFINED by the same before/after r_F growth, so "
                            "differential retrieval of those species is near-oracle "
                            "by construction; concept regions are keyword-defined "
                            "(geometry-neutral) and the natural-pocket contrast is "
                            "the non-circular half of the measurement",
    }}
    for sp in ("radial", "concept", "corridor"):
        trials = results[sp]
        pk = {
            "pocket_rank_none_frac": float(np.mean(
                [all(p["rank"] is None for p in tr["pockets"]) for tr in trials])),
            "pocket_members_in_topfrac_median": float(np.median(
                [p["members_in_topfrac"] for tr in trials for p in tr["pockets"]])),
        }
        summary[sp] = {"abs_recall_top3": float(np.mean(
                           [tr["abs_baseline"]["recall_top3"] for tr in trials])),
                       "neff": agg(trials, "neff"), "mass": agg(trials, "mass"),
                       "delta_sens": agg(trials, "delta_sens"),
                       "neff_tf01": agg(trials, "neff_tf01"), **pk}
        log(f"P2 {sp} summary: " + json.dumps(_py(summary[sp])))
    summary["pockets_absolute_ranks"] = [1, 2]
    if not args.dry:
        update_results("task_p2", {"summary": summary, "trials": results})
        update_report("TASK_P2", report_p2(summary))


def report_p2(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    L = [
        f"## Follow-up Task P2 — level-set residualization (differential components) ({d})",
        "",
        "Level-sets were REFUTED for small-plant retrieval (recall@top-3 = 0/48) because",
        "absolute n_eff surfaces the corpus's standing natural gaps. Residualized form:",
        "score each candidate DIFFERENTIALLY (statistic after the removal minus the",
        "intact-fg baseline the harness already computes — exactly Track 2's T vs T+Delta",
        "snapshot differencing), then run the unchanged component machinery on the",
        "differential field. Same 48 trials; per-trial absolute baselines re-asserted",
        "against banked values before measuring. Caveat ON RECORD: radial/corridor",
        "ground-truth cores are defined by the same before/after r_F growth, so their",
        "differential retrieval is near-oracle BY CONSTRUCTION — the informative cells are",
        "the component machinery's behavior, the concept row (geometry-neutral regions),",
        "and the natural-pocket drop.",
        "",
        "| Species | absolute recall@3 (banked) | differential recall@3 (n_eff, "
        "top-frac 0.05) | (mass order) | (delta-only sens) | **(top-frac 0.01)** | "
        "median IoU @0.05 / @0.01 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sp in ("radial", "concept", "corridor"):
        r = s[sp]
        L.append(
            f"| {sp} | {r['abs_recall_top3']:.2f} | {r['neff']['recall_top3']:.2f} "
            f"| {r['mass']['recall_top3']:.2f} | {r['delta_sens']['recall_top3']:.2f} "
            f"| **{r['neff_tf01']['recall_top3']:.2f}** "
            f"| {r['neff']['best_iou_top3_median']:.2f} / "
            f"{r['neff_tf01']['best_iou_top3_median']:.2f} |")
    L += [
        "",
        "Rank of the planted component when found: median 1 in every cell with nonzero",
        "recall (the change field has essentially one blob).",
        "",
        "Natural pockets (pharmacology n_eff 117.1 / OS 86.1; ranks 1-2 under absolute",
        "scoring by construction) under differencing, per species "
        "(frac of trials where BOTH pockets vanish from the component list / median pocket "
        "members surviving the top-5% selection of 357):",
        "",
    ]
    for sp in ("radial", "concept", "corridor"):
        r = s[sp]
        L.append(f"- {sp}: {r['pocket_rank_none_frac']:.2f} / "
                 f"{r['pocket_members_in_topfrac_median']:.0f}")
    L += [
        "",
        "**Verdict.** Residualization REVIVES the component machinery as a CHANGE detector.",
        "(i) At the banked 5% budget the differential field already lifts recall@top-3 from",
        f"0.00 to {s['radial']['neff']['recall_top3']:.2f}/"
        f"{s['corridor']['neff']['recall_top3']:.2f} (radial/corridor) with the planted",
        "component at rank 1 when found — but the 5% selection (357 points) admits ~300",
        "near-zero-change points that dilute IoU to the 0.1 threshold. Matching the budget",
        "to change-set scale (top-frac 0.01) is the load-bearing fix: radial recall",
        f"{s['radial']['neff_tf01']['recall_top3']:.2f} (median IoU "
        f"{s['radial']['neff_tf01']['best_iou_top3_median']:.2f}), corridor "
        f"{s['corridor']['neff_tf01']['recall_top3']:.2f}. (ii) Concept plants stay at",
        f"{s['concept']['neff_tf01']['recall_top3']:.2f} at the matched budget (the 0.05-",
        "budget 'finds' were 2/16 marginal-IoU noise blobs): no geometric change exists to",
        "difference — consistent with C/P1; the miner keeps concept. (iii) The natural",
        "pockets DROP as designed: rank 1-2 absolute -> absent from the component list in",
        f"{s['radial']['pocket_rank_none_frac']:.2f}/{s['concept']['pocket_rank_none_frac']:.2f}/"
        f"{s['corridor']['pocket_rank_none_frac']:.2f} of trials (radial/concept/corridor),",
        "median 4-20 of 151/91 members surviving selection — they are standing structure,",
        "not changes; residual pocket appearances trace to removals genuinely adjacent to a",
        "pocket (real orphaning, not leakage). CAVEAT held on record: radial/corridor cores",
        "are DEFINED by before/after r_F growth, so their retrieval here is near-oracle;",
        "the non-circular content is the budget finding, the concept negative, and the",
        "pocket cancellation. DECISION: as a Track 1 single-snapshot candidate generator",
        "this remains NON-viable (it requires the before-field; production has no oracle",
        "for what was removed) — but as Track 2's T vs T+Delta snapshot differencer the",
        "machinery transfers verbatim and is now VALIDATED end-to-end (difference the",
        "statistic, select at change scale, take components; standing gaps cancel).",
        "",
    ]
    return L


# =================================================================== task P3a
def _electrodes(X: np.ndarray, Zf: np.ndarray, labels: np.ndarray, cents: np.ndarray,
                pair: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Extended electrodes for one cluster pair on the joint node set.

    Clamp radii come from the PRE-removal fg cluster geometry (median member
    distance to centroid) so before/after fields share the same terminals and
    the field change isolates the node removal."""
    a, b = pair
    cA, cB = cents[a], cents[b]
    r_a = float(np.median(np.linalg.norm(Zf[labels == a] - cA, axis=1)))
    r_b = float(np.median(np.linalg.norm(Zf[labels == b] - cB, axis=1)))
    c0 = np.where(np.linalg.norm(X - cA, axis=1) <= r_a)[0]
    c1 = np.where(np.linalg.norm(X - cB, axis=1) <= r_b)[0]
    c1 = np.setdiff1d(c1, np.intersect1d(c0, c1))
    if c0.size < 5 or c1.size < 5:
        return None
    return c0, c1


def task_p3a(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    Zf, Zb_s = ctx["Zf"], ctx["Zb_s"]
    structure = corridor_structure(ctx)
    labels, cents, pairs = structure
    reproduce_banked_voltage(ctx, banked, structure)
    all_pairs = [(p["a"], p["b"]) for p in pairs]  # 66 qualifying (the structure)
    adj3 = set(centroid_adjacency(cents, n_neighbors=3))
    banked_pairs = {(tr["meta"]["a"], tr["meta"]["b"]) for tr in banked["corridor"]}
    cov_all = sum(1 for p in banked_pairs if p in set(all_pairs))
    cov_adj = sum(1 for p in banked_pairs if p in adj3)
    n_fg = Zf.shape[0]
    X = np.vstack([Zf, Zb_s])
    bg_rows = np.arange(n_fg, X.shape[0])
    electrodes = {}
    for p in all_pairs:
        e = _electrodes(X, Zf, labels, cents, p)
        if e is not None:
            electrodes[p] = e
    pair_list = list(electrodes)
    adj_mask = np.array([p in adj3 for p in pair_list])
    log(f"P3a: qualifying pairs={len(all_pairs)} usable={len(pair_list)} "
        f"(banked coverage {cov_all}/{len(banked_pairs)}); 3-NN subset="
        f"{int(adj_mask.sum())} (coverage {cov_adj}/{len(banked_pairs)})")

    # before-fields (intact fg), one solve per pair
    W0 = gaussian_knn_graph(X, k=10)
    t0 = time.monotonic()
    s0 = []
    for p in pair_list:
        c0, c1 = electrodes[p]
        cur = voltage_on_graph(W0, c0, c1)
        elig = np.ones(X.shape[0], dtype=bool)
        elig[c0] = elig[c1] = False
        elig[:n_fg] = False
        s0.append(flow_share(cur, elig)[bg_rows])
    s0 = np.stack(s0)  # (n_pairs, n_bg)
    log(f"P3a before-fields: {len(pair_list)} solves in {time.monotonic() - t0:.0f}s")
    static_fields = {"all": np.max(s0, axis=0), "adj3": np.max(s0[adj_mask], axis=0)}

    def trial_fields(removed_fg: np.ndarray) -> dict[str, np.ndarray]:
        keep = np.ones(X.shape[0], dtype=bool)
        keep[removed_fg] = False
        remap = np.cumsum(keep) - 1
        W1 = gaussian_knn_graph(X[keep], k=10)
        n_after = int(keep.sum())
        bg_rows_after = remap[bg_rows]  # bg nodes are never removed
        s1 = []
        for p in pair_list:
            c0, c1 = electrodes[p]
            c0a = remap[c0[keep[c0]]]
            c1a = remap[c1[keep[c1]]]
            cur1 = voltage_on_graph(W1, c0a, c1a)
            elig = np.ones(n_after, dtype=bool)
            elig[c0a] = elig[c1a] = False
            elig[: int(keep[:n_fg].sum())] = False
            s1.append(flow_share(cur1, elig)[bg_rows_after])
        s1 = np.stack(s1)
        d = s1 - s0
        return {"change_mag": np.max(np.abs(d), axis=0),
                "change_signed": np.max(d, axis=0),
                "after_only": np.max(s1, axis=0),
                "change_mag_adj3": np.max(np.abs(d[adj_mask]), axis=0)}

    def field_metrics(sc: np.ndarray, gap) -> dict:
        member = _member_mask(N_SCOPED_BG, gap.region_bg)
        every = np.ones(N_SCOPED_BG, dtype=bool)
        k_eval = int(min(50, gap.region_bg.size))
        return {"best_rank": rank_of(sc, member, every),
                "p50": precision_at(sc, member, every, k_eval),
                "auc": auc_vs_controls(sc, gap.region_bg,
                                       np.concatenate(gap.controls_bg))}

    results: dict[str, list] = {"radial": [], "concept": [], "corridor": []}

    def run_trial(sp: str, t: int, gap, removed_fg: np.ndarray) -> None:
        tt = time.monotonic()
        fields = trial_fields(removed_fg)
        rec = {"trial": t,
               **{f: field_metrics(sc, gap) for f, sc in fields.items()},
               "static": field_metrics(static_fields["all"], gap),
               "static_adj3": field_metrics(static_fields["adj3"], gap)}
        results[sp].append(rec)
        log(f"P3a {sp} {t}: mag rank={rec['change_mag']['best_rank']} "
            f"auc={rec['change_mag']['auc']:.2f} | after rank="
            f"{rec['after_only']['best_rank']} auc={rec['after_only']['auc']:.2f} | "
            f"static rank={rec['static']['best_rank']} ({time.monotonic() - tt:.0f}s)")

    for t, gap, _fgp in replant_radial(ctx, banked, args.trials):
        run_trial("radial", t, gap, gap.removed_fg)
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        run_trial("concept", t, gap, np.where(~keep)[0])
    for t, gap, _info in replant_corridor(ctx, banked, args.trials, structure):
        run_trial("corridor", t, gap, gap.removed_fg)

    summary: dict = {"config": {
        "graph": f"joint fg+scoped-bg kNN graph ({X.shape[0]} nodes, k=10, Gaussian "
                 "conductances), rebuilt per trial on the pruned node set",
        "pairs": f"ALL {len(pair_list)} qualifying cluster pairs of the existing "
                 f"corridor structure (banked planted-pair coverage "
                 f"{cov_all}/{len(banked_pairs)}); sensitivity aggregation over the "
                 f"mutual-3NN-centroid adjacency subset ({int(adj_mask.sum())} pairs, "
                 f"coverage {cov_adj}/{len(banked_pairs)})",
        "electrodes": "extended (whole cluster cores), radii from PRE-removal fg",
        "fields": "per-pair mid-weighted node current -> flow SHARE (total current "
                  "drops when a bridge is removed; measured sign inversion — see "
                  "polish.flow_share); aggregate = max over pairs of |share change| "
                  "(main), signed change, after-only share, static before share",
    }}
    for sp in ("radial", "concept", "corridor"):
        summary[sp] = {
            f"{f}_{m}": med(results[sp], f, key,
                            fn=(np.mean if m in ("auc", "p50") else np.median))
            for f in ("change_mag", "change_signed", "after_only", "change_mag_adj3",
                      "static", "static_adj3")
            for m, key in (("rank", "best_rank"), ("auc", "auc"), ("p50", "p50"))
        }
        log(f"P3a {sp} summary: " + json.dumps(_py(summary[sp])))
    if not args.dry:
        update_results("task_p3a", {"summary": summary, "trials": results})
        update_report("TASK_P3A", report_p3a(summary))


def report_p3a(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    cfg = s["config"]
    L = [
        f"## Follow-up Task P3a — voltage all-pairs change field ({d})",
        "",
        f"Graph: {cfg['graph']}. Pairs: {cfg['pairs']}. Electrodes: {cfg['electrodes']}.",
        f"Fields: {cfg['fields']}.",
        "",
        "| Species | field | median best rank | mean AUC | mean P@50 | home detector |",
        "|---|---|---|---|---|---|",
    ]
    home = {"radial": "fused 6.5 (real) / AUC 0.83",
            "concept": "miner 128 (itemset space)",
            "corridor": "voltage single-pair 12 (subgraph) / AUC 0.69"}
    for sp in ("radial", "concept", "corridor"):
        r = s[sp]
        for f, label in (("change_mag", "abs share change (main)"),
                         ("change_signed", "signed change"),
                         ("after_only", "after-only share"),
                         ("change_mag_adj3", "abs share change, 3-NN pairs"),
                         ("static", "static before share"),
                         ("static_adj3", "static, 3-NN pairs")):
            L.append(f"| {sp if f == 'change_mag' else ''} | {label} "
                     f"| {r[f + '_rank']:.0f} | {r[f + '_auc']:.2f} "
                     f"| {r[f + '_p50']:.2f} | {home[sp] if f == 'change_mag' else ''} |")
    ra, co, ce = s["radial"], s["corridor"], s["concept"]
    L += [
        "",
        "**Verdict.** The fg-conditioned CHANGE field is the first genuinely",
        "species-agnostic detector in the suite — and it is a change detector, not a",
        "single-snapshot one. (i) Aggregated |share change| localizes ALL THREE species",
        f"with no species or location knowledge: radial median rank "
        f"{ra['change_mag_rank']:.0f} / AUC {ra['change_mag_auc']:.2f} (home fused: 6.5 /",
        f"0.83), corridor {co['change_mag_rank']:.0f} / {co['change_mag_auc']:.2f} —",
        "BEATING the single-pair voltage specialist (12 / 0.69) without being told the",
        f"pair — and even concept {ce['change_mag_rank']:.0f} / "
        f"{ce['change_mag_auc']:.2f}: the joint graph loses the removed doc NODES, so a",
        "concept removal perturbs flow near its supporters even though it leaves no bg",
        "density dent. (ii) The single-snapshot variants are at chance (after-only AUC",
        f"{ra['after_only_auc']:.2f}/{ce['after_only_auc']:.2f}/{co['after_only_auc']:.2f},",
        f"static {ra['static_auc']:.2f}/{ce['static_auc']:.2f}/{co['static_auc']:.2f}):",
        "natural bridging structure does not mark plants; the CHANGE carries all the",
        "signal. (iii) Aggregation is robust to the pair set (3-NN subset within 0.03 AUC",
        "of all-66) and to sign handling (signed ~ |.|). DECISION: voltage DOES generalize",
        "from corridor specialist to the composite's geometric half, but in Track 2 form —",
        "as the T vs T+Delta change field (where it should be the geometric workhorse,",
        "pairing with P2's residual components); for Track 1 single-snapshot detection the",
        "species-matched suite stands unchanged.",
        "",
    ]
    return L


# =================================================================== task P3b
def _geometry_scorer(X_doc: np.ndarray, ds, scoped_idx: np.ndarray):
    """GapScorer over scoped bg in an embedded doc geometry; NaN docs dropped.

    Returns (scorer, fg_coords (n_fg_kept, d), fg_keepable mask over fg docs,
    bg_keep mask over the 7,139 scoped bg, remap bg-local -> scorer candidate)."""
    fg_rows = X_doc[ds.fg_idx]
    bg_rows = X_doc[ds.bg_idx[scoped_idx]]
    fg_ok = np.isfinite(fg_rows).all(axis=1)
    bg_ok = np.isfinite(bg_rows).all(axis=1)
    remap = np.cumsum(bg_ok) - 1
    scorer = GapScorer(bg=bg_rows[bg_ok], probes=None, k=K_NN)
    scorer.set_r0_from_fg(fg_rows[fg_ok])
    _ = scorer.n_Bloc
    return scorer, fg_rows, fg_ok, bg_ok, remap


def _geom_trial_metrics(scorer: GapScorer, out: dict, gap, bg_ok: np.ndarray,
                        remap: np.ndarray) -> dict:
    region = remap[gap.region_bg[bg_ok[gap.region_bg]]]
    neg_raw = np.concatenate(gap.controls_bg)
    neg = remap[neg_raw[bg_ok[neg_raw]]]
    n = scorer.cand.shape[0]
    member = _member_mask(n, region)
    every = np.ones(n, dtype=bool)
    k_eval = int(min(50, max(region.size, 1)))
    return {
        v: {"auc": auc_vs_controls(out[v], region, neg),
            "best_rank": rank_of(out[v], member, every),
            "p50": precision_at(out[v], member, every, k_eval)}
        for v in ("full", "no_hab")
    }


def task_p3b(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    ds, scoped_idx, Zf = ctx["ds"], ctx["scoped_idx"], ctx["Zf"]
    # identity: reproduce a banked LSA rank before measuring (concept trial 0, full)
    for t, gap, keep in replant_concept(ctx, banked, 1):
        out = ctx["scorer"].score(Zf[keep])
        member = _member_mask(ctx["scorer"].cand.shape[0], gap.region_bg)
        got = rank_of(out["full"], member, ctx["scorer"].real_mask)
        want = banked["concept"][t]["ratio"]["full"]["best_rank"]
        if abs(got - want) > 1:
            raise RuntimeError(f"LSA identity failed: {got} != banked {want}")
        log(f"identity: banked concept LSA rank reproduced (trial {t}: {got})")

    geoms: dict[str, tuple[np.ndarray, dict]] = {}
    t0 = time.monotonic()
    Wb = bipartite_graph(ds.K)
    E, meta = commute_embedding(Wb, dim=128, seed=0)
    geoms["bip128"] = (E[: ds.K.shape[0]], meta)
    log(f"P3b commute bipartite d=128: {time.monotonic() - t0:.0f}s meta={meta}")
    t0 = time.monotonic()
    E64, meta64 = commute_embedding(Wb, dim=64, seed=0)
    geoms["bip64"] = (E64[: ds.K.shape[0]], meta64)
    log(f"P3b commute bipartite d=64: {time.monotonic() - t0:.0f}s")
    t0 = time.monotonic()
    Wl = gaussian_knn_graph(ds.Z, k=10)
    El, metal = commute_embedding(Wl, dim=128, seed=0)
    geoms["lsa128"] = (El, metal)
    log(f"P3b commute LSA-kNN d=128: {time.monotonic() - t0:.0f}s meta={metal}")

    scorers = {}
    for name, (Xd, meta) in geoms.items():
        sc, fg_rows, fg_ok, bg_ok, remap = _geometry_scorer(Xd, ds, scoped_idx)
        scorers[name] = (sc, fg_rows, fg_ok, bg_ok, remap)
        log(f"P3b {name}: fg kept {int(fg_ok.sum())}/{fg_ok.size}, "
            f"scoped bg kept {int(bg_ok.sum())}/{bg_ok.size}")

    concept_trials, radial_trials = [], []
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        rec = {"trial": t,
               "lsa_banked": {v: {"auc": banked["concept"][t]["ratio"][v]["auc"]}
                              for v in ("full", "no_hab")}}
        for name, (sc, fg_rows, fg_ok, bg_ok, remap) in scorers.items():
            fgp = fg_rows[keep & fg_ok]
            out = sc.score(fgp)
            rec[name] = _geom_trial_metrics(sc, out, gap, bg_ok, remap)
            if name == "bip128":
                region = remap[gap.region_bg[bg_ok[gap.region_bg]]]
                core = orphaned_core(sc.bg[region], fg_rows[fg_ok], fgp, k=K_NN,
                                     ratio=1.10)
                rec["bip128_orphans"] = int(core.size)
        concept_trials.append(rec)
        log(f"P3b concept {t}: lsa_auc={rec['lsa_banked']['full']['auc']:.2f} "
            f"bip128 full={rec['bip128']['full']['auc']:.2f} "
            f"rF={rec['bip128']['no_hab']['auc']:.2f} "
            f"lsa128 full={rec['lsa128']['full']['auc']:.2f} "
            f"orphans={rec['bip128_orphans']}")
    for t, gap, _fgp in replant_radial(ctx, banked, args.trials):
        keep = np.ones(Zf.shape[0], dtype=bool)
        keep[gap.removed_fg] = False
        rec = {"trial": t,
               "lsa_banked": {v: {"auc": banked["radial"][t]["ratio"][v]["auc"]}
                              for v in ("full", "no_hab")}}
        for name in ("bip128", "lsa128"):
            sc, fg_rows, fg_ok, bg_ok, remap = scorers[name]
            out = sc.score(fg_rows[keep & fg_ok])
            rec[name] = _geom_trial_metrics(sc, out, gap, bg_ok, remap)
        radial_trials.append(rec)
        log(f"P3b radial {t}: lsa_rF={rec['lsa_banked']['no_hab']['auc']:.2f} "
            f"bip128 rF={rec['bip128']['no_hab']['auc']:.2f} "
            f"lsa128 rF={rec['lsa128']['no_hab']['auc']:.2f}")

    def agg(trials, name):
        return {
            f"{name}_{v}_auc_mean": med(trials, name, v, "auc", fn=np.mean)
            for v in ("full", "no_hab")
        } | {f"{name}_{v}_rank_median": med(trials, name, v, "best_rank")
             for v in ("full", "no_hab")}

    summary = {
        "config": {"embeddings": {n: m for n, (_, m) in geoms.items()},
                   "note": "bipartite doc-keyword commute = the ALL-paths version of "
                           "Task C's one-hop co-occurrence walk; lsa128 = the literal "
                           "bg kNN-graph reading; rank sensitivity bip64"},
        "concept": {
            "lsa_full_auc_mean": med(concept_trials, "lsa_banked", "full", "auc",
                                     fn=np.mean),
            "lsa_no_hab_auc_mean": med(concept_trials, "lsa_banked", "no_hab", "auc",
                                       fn=np.mean),
            **agg(concept_trials, "bip128"), **agg(concept_trials, "bip64"),
            **agg(concept_trials, "lsa128"),
            "bip128_orphans_median": med(concept_trials, "bip128_orphans"),
        },
        "radial": {
            "lsa_no_hab_auc_mean": med(radial_trials, "lsa_banked", "no_hab", "auc",
                                       fn=np.mean),
            **agg(radial_trials, "bip128"), **agg(radial_trials, "lsa128"),
        },
    }
    if not args.dry:
        update_results("task_p3b", {"summary": summary, "concept": concept_trials,
                                    "radial": radial_trials})
        update_report("TASK_P3B", report_p3b(summary))
    log("P3b concept: " + json.dumps(_py(summary["concept"])))
    log("P3b radial: " + json.dumps(_py(summary["radial"])))


def report_p3b(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    c, r = s["concept"], s["radial"]
    L = [
        f"## Follow-up Task P3b — commute-distance (effective-resistance) geometry ({d})",
        "",
        "Effective-resistance embedding (`polish.commute_embedding`): largest connected",
        "component; the dim smallest nontrivial eigenpairs of the normalized Laplacian",
        "obtained as the largest of N = D^-1/2 W D^-1/2 (matvec-only ARPACK — shift-invert",
        "splu explodes on hub fill, measured 87 CPU-min before abort; the 30k x 30k",
        "resistance matrix is never densified); coords u_k(i)/sqrt(lambda_k d_i), whose",
        "squared distances equal R_eff exactly at full rank (Lovasz identity) and its",
        "rank-`dim` truncation here. Two graphs: (i) the bipartite doc-keyword incidence graph — commute",
        "through keywords is the ALL-paths version of Task C's one/two-step",
        "co-occurrence walk (the sharp question); (ii) the LSA bg-kNN graph over all",
        "30k docs (the literal reading; Gaussian conductances, k=10). Rank sensitivity:",
        "dim 64 vs 128 on the bipartite graph. Same 16 concept + 16 radial plants.",
        "",
        "### Concept plants (Task C reference: LSA 0.47, walk t=1 0.49)",
        "",
        "| Geometry | full AUC | r_F AUC | full median rank |",
        "|---|---|---|---|",
        f"| LSA-128 (banked) | {c['lsa_full_auc_mean']:.2f} | "
        f"{c['lsa_no_hab_auc_mean']:.2f} | 188 |",
        f"| commute bipartite d=128 (main) | {c['bip128_full_auc_mean']:.2f} "
        f"| {c['bip128_no_hab_auc_mean']:.2f} | {c['bip128_full_rank_median']:.0f} |",
        f"| commute bipartite d=64 | {c['bip64_full_auc_mean']:.2f} "
        f"| {c['bip64_no_hab_auc_mean']:.2f} | {c['bip64_full_rank_median']:.0f} |",
        f"| commute LSA-kNN d=128 | {c['lsa128_full_auc_mean']:.2f} "
        f"| {c['lsa128_no_hab_auc_mean']:.2f} | {c['lsa128_full_rank_median']:.0f} |",
        "",
        f"Orphaned supporters in bipartite-commute geometry: median "
        f"{c['bip128_orphans_median']:.0f} (LSA: 0, walk: 0).",
        "",
        "### Radial plants (regression; LSA r_F AUC 0.84, walk collapsed to 0.51)",
        "",
        "| Geometry | r_F AUC | r_F median rank | full AUC |",
        "|---|---|---|---|",
        f"| commute bipartite d=128 | {r['bip128_no_hab_auc_mean']:.2f} "
        f"| {r['bip128_no_hab_rank_median']:.0f} | {r['bip128_full_auc_mean']:.2f} |",
        f"| commute LSA-kNN d=128 | {r['lsa128_no_hab_auc_mean']:.2f} "
        f"| {r['lsa128_no_hab_rank_median']:.0f} | {r['lsa128_full_auc_mean']:.2f} |",
        "",
        "**Verdict.** Commute distance does NOT vindicate the Gramian intuition — the",
        "ALL-paths metric fails exactly where the one/two-step walk failed: concept AUC vs",
        f"matched controls stays at chance in every configuration "
        f"(bipartite d=128 {c['bip128_full_auc_mean']:.2f}, d=64 "
        f"{c['bip64_full_auc_mean']:.2f}, LSA-graph {c['lsa128_full_auc_mean']:.2f}; "
        f"refs: LSA 0.47, walk 0.49) and the",
        f"removal still orphans nobody (median {c['bip128_orphans_median']:.0f}). The",
        "low best-rank cells are the same generic pattern-supporter effect Task C",
        "diagnosed (causal AUC ~0.5 says so). Mechanism reading: the bipartite walk",
        "spectrum is FLAT (lambda in [0.35, 0.77]) — the spectral-gap regime where",
        "commute distance degenerates toward local degrees (von Luxburg), so summing all",
        "paths over common top-2000-df keywords dilutes exactly like the 2-step walk;",
        "rare-keyword binding (the fixture mechanism) never gets to dominate. Radial",
        "regression: the keyword-graph commute collapses radial detection like the walk",
        f"did (r_F AUC {r['bip128_no_hab_auc_mean']:.2f} vs 0.84 LSA), while commute OF",
        f"the LSA graph keeps the causal signal (AUC {r['lsa128_no_hab_auc_mean']:.2f})",
        f"but loses rank sharpness (median {r['lsa128_no_hab_rank_median']:.0f} vs 10",
        "Euclidean) — its tiny bottom eigenvalues (lambda_min 2.5e-4) stretch global",
        "structure at the expense of local dent contrast. DECISION: commute/resistance",
        "geometry earns no place in the composite — concept stays the miner's; radial",
        "stays Euclidean-LSA r_F.",
        "",
    ]
    return L


# =================================================================== task P4
def task_p4(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    ds, scoped_idx = ctx["ds"], ctx["scoped_idx"]
    assets = miner_assets(ctx, banked)  # includes miner identity gate
    Kf_dense, Kb_s_dense = assets["Kf_dense"], assets["Kb_s_dense"]
    X_tfidf, vocab_all = rebuild_tfidf(ds)  # asserts page_id + K identity
    log("identity: tf-idf rebuild matched cached dataset (page_ids + top-2000 K)")
    df_all = np.asarray((X_tfidf > 0).sum(axis=0)).ravel()

    def tail_assets(df_range, cap):
        cols, K_tail = low_df_tail(X_tfidf, df_range=df_range, cap=cap)
        tail_fg = np.asarray(K_tail[ds.fg_idx].todense(), dtype=bool)
        tail_bg = K_tail[ds.bg_idx][scoped_idx]
        fam_tail = mine_tail_family(Kb_s_dense, tail_bg, min_support=10, top_offset=2000)
        tail_keys = list(fam_tail.keys())
        Kf_aug = np.hstack([Kf_dense, tail_fg])
        Kb_aug = np.hstack([Kb_s_dense, np.asarray(tail_bg.todense(), dtype=bool)])
        keys_aug = assets["keys"] + tail_keys
        key_index_aug = {S: i for i, S in enumerate(keys_aug)}
        sup_aug = np.concatenate([assets["sup_bg"],
                                  np.array([fam_tail[S] for S in tail_keys],
                                           dtype=np.int64)])
        M_fg_aug = sp.vstack([assets["M_fg"],
                              itemset_support_matrix(tail_keys, Kf_aug)]).tocsr()
        return {"cols": cols, "fam_tail": fam_tail, "keys": keys_aug,
                "key_index": key_index_aug, "sup": sup_aug, "M_fg": M_fg_aug,
                "df_realized": (int(df_all[cols].min()), int(df_all[cols].max())),
                "n_tail_terms": int(cols.size), "K_tail": K_tail}

    # cap selection note: rarest-first cap 8000 keeps df in [5, 22] whose scoped-bg
    # support is almost never >= 10 (906 minable itemsets) — biased AGAINST the
    # hypothesis. Main = cap 10,000 (the spec's upper guidance); sensitivity = the
    # FULL df window (12,316 terms, no cap) = the hypothesis's best case.
    main_tail = tail_assets((3, 50), 10_000)
    sens_tail = tail_assets((3, 50), 20_000)  # > window size -> uncapped
    log(f"P4 tail(main cap=10k): {main_tail['n_tail_terms']} terms, realized df "
        f"{main_tail['df_realized']}, tail itemsets {len(main_tail['fam_tail'])}")
    log(f"P4 tail(sens uncapped): {sens_tail['n_tail_terms']} terms, realized df "
        f"{sens_tail['df_realized']}, tail itemsets {len(sens_tail['fam_tail'])}")

    n_bg = assets["n_bg"]
    trials = []
    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        target = frozenset(int(c) for c in gap.meta["combo_cols"])
        removed_mask = ~keep
        # original lattice (identity per trial)
        ranked_o = rank_family_fast(assets["keys"], assets["sup_bg"], assets["M_fg"],
                                    keep, n_bg=n_bg)
        pos_o = next((i + 1 for i, r in enumerate(ranked_o) if r["itemset"] == target),
                     None)
        if pos_o != banked["concept"][t]["miner"]["rank"]:
            raise RuntimeError(f"P4 concept {t}: orig rank {pos_o} != banked")
        rec = {"trial": t, "planted_rank_orig": pos_o, "n_ranked_orig": len(ranked_o)}
        n_pre_o, n_rem_o = attribution_stats(assets["M_fg"], removed_mask)
        a_o, _ = attributable_best_rank(ranked_o, assets["key_index"], n_pre_o, n_rem_o,
                                        10, 0.8)
        rec["attrib_rank_orig"] = a_o
        for name, ta in (("aug", main_tail), ("augfull", sens_tail)):
            ranked_a = rank_family_fast(ta["keys"], ta["sup"], ta["M_fg"], keep,
                                        n_bg=n_bg)
            pos_a = next((i + 1 for i, r in enumerate(ranked_a)
                          if r["itemset"] == target), None)
            rec[f"planted_rank_{name}"] = pos_a
            rec[f"n_ranked_{name}"] = len(ranked_a)
            if name == "aug":
                n_pre, n_rem = attribution_stats(ta["M_fg"], removed_mask)
                a_a, row_a = attributable_best_rank(ranked_a, ta["key_index"],
                                                    n_pre, n_rem, 10, 0.8)
                rec["attrib_rank_aug"] = a_a
                rec["attrib_aug_is_tail"] = (
                    bool(any(c >= 2000 for c in row_a["itemset"])) if row_a else None)
                # min_len=1: tail singletons enter the ranking
                ranked_1 = rank_family_fast(ta["keys"], ta["sup"], ta["M_fg"], keep,
                                            n_bg=n_bg, min_len=1)
                a_1, row_1 = attributable_best_rank(ranked_1, ta["key_index"],
                                                    n_pre, n_rem, 10, 0.8)
                rec["attrib_rank_aug_min1"] = a_1
                rec["planted_rank_aug_min1"] = next(
                    (i + 1 for i, r in enumerate(ranked_1) if r["itemset"] == target),
                    None)
        trials.append(rec)
        log(f"P4 concept {t}: planted {pos_o} -> aug {rec['planted_rank_aug']} "
            f"(full {rec['planted_rank_augfull']}, min1 {rec['planted_rank_aug_min1']}) | "
            f"attrib {rec['attrib_rank_orig']} -> {rec['attrib_rank_aug']} "
            f"(tail={rec['attrib_aug_is_tail']}, min1 {rec['attrib_rank_aug_min1']})")

    # ---- secondary: walk geometry in the augmented / tail keyword space
    walk_rows = {}
    K_aug_all = sp.hstack([ds.K, main_tail["K_tail"]]).tocsr()
    for name, Kmat in (("walk_aug", K_aug_all), ("walk_tail", main_tail["K_tail"])):
        W, meta = walk_embedding(Kmat, dim=128, seed=0, t=1.0, degree_scale=True)
        sc, fg_rows, fg_ok, bg_ok, remap = _geometry_scorer(W, ds, scoped_idx)
        # isolated docs embed at the origin (finite): count them instead via meta
        crows = []
        for t, gap, keep in replant_concept(ctx, banked, args.trials):
            out = sc.score(fg_rows[keep & fg_ok])
            crows.append(_geom_trial_metrics(sc, out, gap, bg_ok, remap))
        rrows = []
        if name == "walk_aug":
            for t, gap, _fgp in replant_radial(ctx, banked, args.trials):
                keep = np.ones(ctx["Zf"].shape[0], dtype=bool)
                keep[gap.removed_fg] = False
                out = sc.score(fg_rows[keep & fg_ok])
                rrows.append(_geom_trial_metrics(sc, out, gap, bg_ok, remap))
        walk_rows[name] = {
            "meta": meta,
            "concept_full_auc_mean": float(np.mean([r["full"]["auc"] for r in crows])),
            "concept_no_hab_auc_mean": float(np.mean([r["no_hab"]["auc"] for r in crows])),
            "concept_full_rank_median": float(np.median([r["full"]["best_rank"]
                                                         for r in crows])),
            "radial_no_hab_auc_mean": (float(np.mean([r["no_hab"]["auc"] for r in rrows]))
                                       if rrows else None),
        }
        log(f"P4 {name}: isolated={meta['n_isolated_docs']} "
            + json.dumps(_py({k: v for k, v in walk_rows[name].items() if k != 'meta'})))

    def medo(key):
        vals = [tr[key] for tr in trials if tr.get(key) is not None]
        return float(np.median(vals)) if vals else None

    summary = {
        "config": {
            "tail": f"df in [3,50] requested; realized {main_tail['df_realized']} "
                    f"(TfidfVectorizer min_df=10 bounds the floor), cap 10,000 rarest "
                    f"(sens: full window, {sens_tail['n_tail_terms']} terms, realized "
                    f"df {sens_tail['df_realized']}); {main_tail['n_tail_terms']} terms",
            "tail_mining": "tail singletons + tail x tail + tail x top-2000 pairs, "
                           "scoped-bg support >= 10 (vs 30 for the base lattice; tail "
                           "global df <= 50 makes support 30 nearly unreachable)",
            "n_tail_itemsets": len(main_tail["fam_tail"]),
        },
        "found_rate_orig": float(np.mean([tr["planted_rank_orig"] is not None
                                          for tr in trials])),
        "found_rate_aug": float(np.mean([tr["planted_rank_aug"] is not None
                                         for tr in trials])),
        "planted_rank_orig_median": medo("planted_rank_orig"),
        "planted_rank_aug_median": medo("planted_rank_aug"),
        "planted_rank_augfull_median": medo("planted_rank_augfull"),
        "planted_rank_aug_min1_median": medo("planted_rank_aug_min1"),
        "attrib_rank_orig_median": medo("attrib_rank_orig"),
        "attrib_rank_aug_median": medo("attrib_rank_aug"),
        "attrib_rank_aug_min1_median": medo("attrib_rank_aug_min1"),
        "attrib_aug_tail_frac": float(np.mean([bool(tr["attrib_aug_is_tail"])
                                               for tr in trials
                                               if tr["attrib_aug_is_tail"] is not None])),
        "n_ranked_orig": trials[0]["n_ranked_orig"],
        "n_ranked_aug": trials[0]["n_ranked_aug"],
        "walk": walk_rows,
    }
    if not args.dry:
        update_results("task_p4", {"summary": summary, "trials": trials})
        update_report("TASK_P4", report_p4(summary))
    log("P4 summary: " + json.dumps(_py({k: v for k, v in summary.items()
                                         if k not in ("walk", "config")})))


def report_p4(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    wa, wt = s["walk"]["walk_aug"], s["walk"]["walk_tail"]
    L = [
        f"## Follow-up Task P4 — low-df miner lattice ({d})",
        "",
        f"Tail: {s['config']['tail']}. Tier mining: {s['config']['tail_mining']} "
        f"({s['config']['n_tail_itemsets']:,} tail itemsets; family "
        f"{s['n_ranked_orig']:,} -> {s['n_ranked_aug']:,} ranked at min_len 2).",
        "Same 16 concept trials; per-trial original planted ranks re-asserted against",
        "banked values before measuring.",
        "",
        "| Metric | original lattice | augmented (main, cap 10k) | full window (sens, "
        "12,316) | min_len 1 |",
        "|---|---|---|---|---|",
        f"| planted-combo median rank | {s['planted_rank_orig_median']:.0f} "
        f"| {s['planted_rank_aug_median']:.0f} | {s['planted_rank_augfull_median']:.0f} "
        f"| {s['planted_rank_aug_min1_median']:.0f} |",
        f"| found rate | {s['found_rate_orig']:.2f} | {s['found_rate_aug']:.2f} "
        f"| {s['found_rate_aug']:.2f} | {s['found_rate_aug']:.2f} |",
        f"| best attributable median rank | {s['attrib_rank_orig_median']:.0f} "
        f"| {s['attrib_rank_aug_median']:.0f} | — "
        f"| {s['attrib_rank_aug_min1_median']:.0f} |",
        "",
        f"Tail share of best-attributable itemsets (augmented): "
        f"{s['attrib_aug_tail_frac']:.2f}.",
        "",
        "### Secondary: walk geometry in the augmented / tail-only keyword space",
        "",
        "| Space | concept full AUC | concept r_F AUC | concept full median rank "
        "| radial r_F AUC | isolated docs |",
        "|---|---|---|---|---|---|",
        f"| top-2000 walk (Task C) | 0.49 | 0.47 | 28 | 0.51 | 0 |",
        f"| top-2000 + tail (walk_aug) | {wa['concept_full_auc_mean']:.2f} "
        f"| {wa['concept_no_hab_auc_mean']:.2f} | {wa['concept_full_rank_median']:.0f} "
        f"| {wa['radial_no_hab_auc_mean']:.2f} | {wa['meta']['n_isolated_docs']} |",
        f"| tail only (walk_tail) | {wt['concept_full_auc_mean']:.2f} "
        f"| {wt['concept_no_hab_auc_mean']:.2f} | {wt['concept_full_rank_median']:.0f} "
        f"| — | {wt['meta']['n_isolated_docs']} |",
        "",
        "**Verdict.** Decisive DOUBLE NEGATIVE; the champion does not sharpen and C's",
        "negative does not flip. (i) The planted-combo rank is bit-identical under every",
        f"augmentation ({s['planted_rank_orig_median']:.0f} -> "
        f"{s['planted_rank_aug_median']:.0f} main, "
        f"{s['planted_rank_augfull_median']:.0f} full-window; 16/16 found throughout;",
        f"min_len 1 only worsens it to {s['planted_rank_aug_min1_median']:.0f} by letting",
        "fg-absent top-2000 singletons in above) and NO best-attributable itemset is ever",
        f"a tail itemset (share {s['attrib_aug_tail_frac']:.2f}). Mechanism: among",
        "fg-absent itemsets the Fightin'-Words z is monotone in bg support, and the ~128",
        "itemsets above a planted combo are the corpus's NATURAL emerging patterns",
        "(standing gaps like the pharmacology/OS pockets) with support >= the plant's;",
        "tail itemsets (scoped support 10-50) slot strictly below. The champion's rank is",
        "bounded by real corpus structure, NOT lattice resolution — the top-2000-df budget",
        "was never the binding constraint on the miner. (ii) The walk-geometry 'possible",
        f"flip' is refuted: concept AUC stays at chance in the augmented space "
        f"({wa['concept_full_auc_mean']:.2f}) and in the tail-only space "
        f"({wt['concept_full_auc_mean']:.2f},",
        f"with {wt['meta']['n_isolated_docs']:,} docs isolated there), vs 0.49 in Task C.",
        "The fixture mechanism requires concept members to SHARE the same few rare",
        "keywords; planted wiki concepts are common-keyword co-occurrence patterns whose",
        "members' rare vocabularies are heterogeneous — the low-df tail has nothing to",
        "bind. DECISION: champion boost magnitude = ZERO; keep the miner exactly as is",
        "(top-2000-df lattice, scoped, min_support 30) and spend no further effort on",
        "keyword-space engineering for concept gaps. (Track 2 note: emerging-topic gaps",
        "whose members DO share new rare terms remain the tail's regime — untested here",
        "by design, P2/P3a carry that track.)",
        "",
    ]
    return L


# =================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["p1", "p2", "p3a", "p3b", "p4"], required=True)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    {"p1": task_p1, "p2": task_p2, "p3a": task_p3a, "p3b": task_p3b,
     "p4": task_p4}[args.task](args)
    log("done")


if __name__ == "__main__":
    main()
