"""Track 1 finish runner (task_t1_comp, 2026-06-11 dispatch).

Part A (--part a): TYPE-BLIND COMPOSITE BENCHMARK. 48 mixed trials = the 48
banked plants (16 per species, seeds 1000/2000/3000+t) in a seeded balanced-deck
shuffle (uniform-without-replacement type draw; preserves banked per-type
comparability). The composite pipeline receives the AFTER state only (pruned fg
coords + pruned fg keyword rows) and never reads the species label: channels
geo (scoped fused r_F+veto), pat (miner doc layer), volt (single-pair voltage on
self-selected pairs, top-3 by band-emptiness, merged by max share). Arbitration
= max of in-trial channel percentiles (noisy-OR, no channel leads — P1's
miner-led arbitration stays rejected; the measured reason routing is not the
main rule: pair-emptiness false-fires on radial trials at the same 0.60-0.64
level as true corridors). Ablation rules: mean percentile, RRF (k=60), geo+volt
(miner dropped). Identity: per-trial plant identity asserted (replant_*), plus
banked rank reproductions per species (radial fused, concept miner itemset +
P1 doc layer, corridor voltage subgraph).

Part B (--part b): SENSITIVITY FRONTIER. Radial: removed-size {8..256, log2} x
detector k {5,10,20}, 16 seeds/cell, fused r_F+veto, ground truth fixed at the
banked contract (orphan core, k_gt=10, ratio 1.15, min_core relaxed to 3).
Corridor: sub-budget removals (m innermost band members) of the 16 banked pairs,
oracle-pair voltage, k_graph=10. Output: frontier table + the
minimum-recoverable-gap spec.

Run (CPU only; BLAS capped at 6 threads — a second analysis agent runs):
  uv run python -m artifacts_scripts.gap_detection.run_t1_composite --part a
  uv run python -m artifacts_scripts.gap_detection.run_t1_composite --part b
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")  # second analysis agent runs concurrently

import argparse
import json
import time
from datetime import datetime

import numpy as np

from artifacts_scripts.gap_detection.composite import (
    after_fg_pairs,
    fuse_max,
    fuse_mean,
    fuse_rrf,
    percentile_scores,
    voltage_pair_field,
)
from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    auc_vs_controls,
    precision_at,
    rank_of,
    topk_indices,
)
from artifacts_scripts.gap_detection.fusion import fused_scores, veto_mask
from artifacts_scripts.gap_detection.polish import doc_pattern_scores, rank_family_fast
from artifacts_scripts.gap_detection.run_eval import _member_mask, _py
from artifacts_scripts.gap_detection.run_followups import (
    build_ctx,
    load_banked,
    med,
    replant_concept,
    replant_radial,
    update_report,
    update_results,
)
from artifacts_scripts.gap_detection.run_polish import (
    N_SCOPED_BG,
    corridor_structure,
    miner_assets,
    replant_corridor,
    reproduce_banked_voltage,
)
from artifacts_scripts.gap_detection.synth import _dense_fg_pool, orphaned_core

_T0 = time.monotonic()
VOLT_J = 3            # pairs merged into the blind voltage channel
DECK_SEED = 4000      # balanced-deck shuffle seed (type sequence)
RULES = ("max", "mean", "rrf", "geo_volt")
FOUND_AT = 50

# Part B grids
FRONTIER_SIZES = (8, 16, 32, 64, 128, 256)
FRONTIER_KS = (5, 10, 20)
FRONTIER_TRIALS = 16
CORRIDOR_SIZES = (8, 16, 32, 64, 128)
AUC_SPEC_THR = 0.75


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def _fmt(v, spec: str = ".0f") -> str:
    return "—" if v is None else format(v, spec)


# ===================================================================== Part A machinery
def _pad_probes(x_real: np.ndarray, n_probes: int) -> np.ndarray:
    return np.concatenate([x_real, np.full(n_probes, -np.inf)])


def blind_channels(ctx: dict, assets: dict, keep: np.ndarray) -> tuple[dict, dict]:
    """All three channel score vectors over the candidate universe (real+probes).

    TYPE-BLIND BY CONSTRUCTION: consumes only the keep mask (= the after-state
    foreground) and corpus-side assets; no species label, no plant metadata.
    """
    scorer, Zf, Zb_s = ctx["scorer"], ctx["Zf"], ctx["Zb_s"]
    n_probes = scorer.cand.shape[0] - N_SCOPED_BG
    fgp = Zf[keep]
    out = scorer.score(fgp)
    veto = veto_mask(scorer, out)
    s_geo = fused_scores(out, veto)
    ranked = rank_family_fast(assets["keys"], assets["sup_bg"], assets["M_fg"], keep,
                              n_bg=assets["n_bg"])
    s_pat = _pad_probes(doc_pattern_scores(ranked, assets["key_index"], assets["M_bg"]),
                        n_probes)
    labels, cents, pairs = after_fg_pairs(fgp, Zb_s)
    fields, sel = [], []
    for pr in pairs[:VOLT_J]:
        r = voltage_pair_field(fgp, Zb_s, labels, cents, pr)
        if r is not None:
            fields.append(r[0])
            sel.append({"pair": [pr["a"], pr["b"]], "emptiness": round(pr["emptiness"], 4)})
    volt_real = (np.max(np.stack(fields), axis=0) if fields
                 else np.full(N_SCOPED_BG, -np.inf))
    s_volt = _pad_probes(volt_real, n_probes)
    diag = {"n_pairs_scored": len(pairs), "selected_pairs": sel,
            "top_emptiness": round(pairs[0]["emptiness"], 4) if pairs else None}
    return {"geo": s_geo, "pat": s_pat, "volt": s_volt}, diag


def composite_rules(ch: dict) -> dict[str, np.ndarray]:
    p = {k: percentile_scores(v) for k, v in ch.items()}
    return {
        "max": fuse_max([p["geo"], p["pat"], p["volt"]]),
        "mean": fuse_mean([p["geo"], p["pat"], p["volt"]]),
        "rrf": fuse_rrf([ch["geo"], ch["pat"], ch["volt"]]),
        "geo_volt": fuse_max([p["geo"], p["volt"]]),
    }


def _metrics(sc: np.ndarray, scorer: GapScorer, region: np.ndarray,
             controls: np.ndarray) -> dict:
    n = scorer.cand.shape[0]
    member = _member_mask(n, region)
    every = np.ones(n, dtype=bool)
    k_eval = int(min(FOUND_AT, region.size))
    rr = rank_of(sc, member, scorer.real_mask)
    return {
        "rank_real": rr,
        "rank_all": rank_of(sc, member, every),
        "auc": auc_vs_controls(sc, region, controls),
        "p50": precision_at(sc, member, scorer.real_mask, k_eval),
        "found50": bool(rr <= FOUND_AT),
        "probes_top50": int((~scorer.real_mask[topk_indices(sc, every, FOUND_AT)]).sum()),
    }


def part_a(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    assert ctx["n_scoped"] == N_SCOPED_BG
    scorer = ctx["scorer"]
    structure = corridor_structure(ctx)

    # ---- identity gates (one banked rank per species + per-trial plant identity)
    reproduce_banked_voltage(ctx, banked, structure)         # corridor: rank 32
    assets = miner_assets(ctx, banked)                       # concept: itemset rank 436
    for t, gap, fgp in replant_radial(ctx, banked, 1):       # radial: fused rank_real
        out = scorer.score(fgp)
        veto = veto_mask(scorer, out)
        got = rank_of(fused_scores(out, veto), _member_mask(scorer.cand.shape[0], gap.region_bg),
                      scorer.real_mask)
        want = banked["task_a"]["trials"][t]["scoped"]["fused"]["best_rank_real"]
        if got != want:
            raise RuntimeError(f"radial fused identity failed: {got} != banked {want}")
        log(f"identity: banked radial fused rank reproduced (trial {t}: {got})")

    # ---- the mixed deck: seeded balanced shuffle of the 48 banked trials
    deck = [("radial", t) for t in range(args.trials)] \
        + [("concept", t) for t in range(args.trials)] \
        + [("corridor", t) for t in range(args.trials)]
    order = np.random.default_rng(DECK_SEED).permutation(len(deck))
    deck = [deck[i] for i in order]
    deck_str = "".join({"radial": "R", "concept": "C", "corridor": "X"}[k] for k, _ in deck)
    log(f"deck: {deck_str}")

    Zf = ctx["Zf"]
    records: dict[tuple, dict] = {}

    def run_one(kind: str, t: int, gap, keep: np.ndarray) -> None:
        tt = time.monotonic()
        region, controls = gap.region_bg, np.concatenate(gap.controls_bg)
        ch, vdiag = blind_channels(ctx, assets, keep)
        rules = composite_rules(ch)
        rec = {
            "kind": kind, "trial": t, "volt_diag": vdiag,
            "channels": {c: _metrics(ch[c], scorer, region, controls)
                         for c in ("geo", "pat", "volt")},
            "rules": {r: _metrics(rules[r], scorer, region, controls) for r in RULES},
        }
        rec["channels"]["volt"]["region_coverage"] = float(
            np.isfinite(ch["volt"][region]).mean()) if region.size else 0.0
        records[(kind, t)] = rec
        m = rec["rules"]["max"]
        log(f"A {kind} {t}: comp_max rank={m['rank_real']} auc={m['auc']:.2f} "
            f"found50={m['found50']} | geo={rec['channels']['geo']['rank_real']} "
            f"pat={rec['channels']['pat']['rank_real']} "
            f"volt={rec['channels']['volt']['rank_real']} "
            f"(cov={rec['channels']['volt']['region_coverage']:.2f}) "
            f"({time.monotonic() - tt:.0f}s)")

    # ---- oracle home columns (computed OUTSIDE the blind path)
    for t, gap, fgp in replant_radial(ctx, banked, args.trials):
        keep = np.ones(Zf.shape[0], dtype=bool)
        keep[gap.removed_fg] = False
        run_one("radial", t, gap, keep)
        rec = records[("radial", t)]
        rec["oracle"] = {"detector": "fused r_F+veto (== geo channel)",
                         "rank": rec["channels"]["geo"]["rank_real"],
                         "auc": rec["channels"]["geo"]["auc"]}

    for t, gap, keep in replant_concept(ctx, banked, args.trials):
        run_one("concept", t, gap, keep)
        rec = records[("concept", t)]
        # P1 doc-layer identity (the pat channel IS the doc-space home detector)
        bk = banked["task_p1"]["concept"][t]["doc_best_rank"]
        got = rec["channels"]["pat"]["rank_real"]
        if got != bk:
            raise RuntimeError(f"concept {t}: pat doc rank {got} != banked P1 {bk}")
        rec["oracle"] = {"detector": "miner doc layer (== pat channel)",
                         "rank": got, "auc": rec["channels"]["pat"]["auc"],
                         "itemset_rank_banked": banked["concept"][t]["miner"]["rank"]}

    for t, gap, info in replant_corridor(ctx, banked, args.trials, structure):
        keep = np.ones(Zf.shape[0], dtype=bool)
        keep[gap.removed_fg] = False
        run_one("corridor", t, gap, keep)
        rec = records[("corridor", t)]
        # oracle-pair voltage in the SAME universe (field -inf outside subgraph)
        labels_p = np.delete(info["labels"], gap.removed_fg)
        pair = {"a": gap.meta["a"], "b": gap.meta["b"], "lens_r": gap.meta["lens_r"]}
        r = voltage_pair_field(Zf[keep], ctx["Zb_s"], labels_p, info["cents"], pair)
        assert r is not None
        sc_o, _ = r
        member = _member_mask(N_SCOPED_BG, gap.region_bg)
        sub_rank = rank_of(sc_o, member, np.isfinite(sc_o))
        bk = banked["corridor"][t]["voltage"]["best_rank"]
        if sub_rank != bk:
            raise RuntimeError(f"corridor {t}: oracle voltage rank {sub_rank} != banked {bk}")
        rec["oracle"] = {
            "detector": "single-pair voltage, planted pair",
            "rank": rank_of(_pad_probes(sc_o, scorer.cand.shape[0] - N_SCOPED_BG),
                            member, scorer.real_mask),
            "auc": auc_vs_controls(sc_o, gap.region_bg, np.concatenate(gap.controls_bg)),
            "subgraph_rank_banked": bk,
        }

    # ---- assemble + summarize
    trials = [{"mixed_id": i, **records[key]} for i, key in enumerate(deck)]

    def agg(kind: str) -> dict:
        rows = [r for r in trials if r["kind"] == kind]
        out = {
            "n": len(rows),
            "oracle_rank_median": med(rows, "oracle", "rank"),
            "oracle_auc_mean": med(rows, "oracle", "auc", fn=np.mean),
            "oracle_found50": float(np.mean([r["oracle"]["rank"] <= FOUND_AT
                                             for r in rows])),
        }
        for c in ("geo", "pat", "volt"):
            out[f"{c}_rank_median"] = med(rows, "channels", c, "rank_real")
            out[f"{c}_auc_mean"] = med(rows, "channels", c, "auc", fn=np.mean)
            out[f"{c}_found50"] = med(rows, "channels", c, "found50", fn=np.mean)
        out["volt_coverage_mean"] = med(rows, "channels", "volt", "region_coverage",
                                        fn=np.mean)
        for r in RULES:
            out[f"{r}_rank_median"] = med(rows, "rules", r, "rank_real")
            out[f"{r}_auc_mean"] = med(rows, "rules", r, "auc", fn=np.mean)
            out[f"{r}_found50"] = med(rows, "rules", r, "found50", fn=np.mean)
            out[f"{r}_p50_mean"] = med(rows, "rules", r, "p50", fn=np.mean)
        out["max_probes_top50_mean"] = med(rows, "rules", "max", "probes_top50", fn=np.mean)
        out["cost_rank"] = out["max_rank_median"] - out["oracle_rank_median"]
        out["cost_auc"] = out["max_auc_mean"] - out["oracle_auc_mean"]
        return out

    summary = {
        "config": {
            "deck_seed": DECK_SEED, "volt_pairs_merged": VOLT_J,
            "deck": deck_str,
            "arbitration_main": "max of in-trial channel percentiles (noisy-OR; "
                                "abstain/veto = 0; finite scores ranked among "
                                "themselves per channel)",
            "ablations": "mean percentile; RRF k=60; geo+volt (miner dropped)",
            "blindness": "channels consume only the after-state fg (coords + keyword "
                         "rows); species label never read in the scoring path",
        },
        "per_kind": {k: agg(k) for k in ("radial", "concept", "corridor")},
    }
    allrows = trials
    summary["overall"] = {
        "max_rank_median": med(allrows, "rules", "max", "rank_real"),
        "max_auc_mean": med(allrows, "rules", "max", "auc", fn=np.mean),
        "max_found50": med(allrows, "rules", "max", "found50", fn=np.mean),
        **{f"{r}_found50": med(allrows, "rules", r, "found50", fn=np.mean)
           for r in RULES},
        **{f"{r}_rank_median": med(allrows, "rules", r, "rank_real") for r in RULES},
        **{f"{r}_auc_mean": med(allrows, "rules", r, "auc", fn=np.mean) for r in RULES},
    }
    if not args.dry:
        existing = load_banked().get("task_t1_comp", {})
        existing["part_a"] = {"summary": summary, "trials": trials}
        update_results("task_t1_comp", existing)
        update_report("TASK_T1_COMP", report_a(summary))
    for k in ("radial", "concept", "corridor"):
        log(f"A summary {k}: " + json.dumps(_py(summary["per_kind"][k])))
    log("A overall: " + json.dumps(_py(summary["overall"])))


def report_a(s: dict) -> list[str]:
    d = datetime.now().date().isoformat()
    pk = s["per_kind"]
    home_note = {
        "radial": "fused r_F+veto 6.5 / 0.83 (banked A)",
        "concept": "miner doc layer 40 / 0.56 (P1; itemset-space 128.5/36,199)",
        "corridor": "voltage, planted pair 12 / 0.69 (banked)",
    }
    L = [
        f"## Track 1 finish, Part A — type-blind composite benchmark ({d})",
        "",
        "48 mixed trials = the 48 banked plants (16/species, identity asserted; banked",
        "ranks reproduced: radial fused rank-real, concept miner itemset rank + P1 doc",
        "rank, corridor voltage subgraph rank — all 16 corridor oracle ranks re-derived",
        "exactly) in a seeded balanced-deck shuffle "
        f"(seed {s['config']['deck_seed']}). The pipeline reads ONLY the after-state fg",
        "(coords + keyword rows): channels geo = scoped fused r_F+veto, pat = miner doc",
        "layer (max Fightin'-Words z over the scoped closed family), volt = single-pair",
        f"voltage on {s['config']['volt_pairs_merged']} SELF-selected pairs (band-emptiness "
        "= MWU AUC of fg vs bg |t-0.5| in the pair lens; after-fg KMeans-12), merged by",
        "max flow share. Arbitration (main) = max of in-trial channel percentiles —",
        "noisy-OR, no channel leads (P1: miner-led arbitration stays rejected; routing",
        "rejected on measurement: pair-emptiness false-fires on radial trials at the",
        "same 0.60-0.64 level as true corridors, so a corridor gate cannot be",
        "thresholded). Probes can enter only through geo and its veto holds.",
        "",
        "| Species | type-oracle home (rank / AUC / found@50) | composite max (rank / "
        "AUC / found@50) | cost of blindness (Δrank / ΔAUC / Δfound) | solo geo | solo "
        "pat | solo volt (rank / AUC / found@50; coverage) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ("radial", "concept", "corridor"):
        r = pk[k]
        L.append(
            f"| {k} | {r['oracle_rank_median']:.0f} / {r['oracle_auc_mean']:.2f} / "
            f"{r['oracle_found50']:.2f} ({home_note[k]}) "
            f"| **{r['max_rank_median']:.0f} / {r['max_auc_mean']:.2f} / "
            f"{r['max_found50']:.2f}** | {r['cost_rank']:+.0f} / {r['cost_auc']:+.2f} / "
            f"{r['max_found50'] - r['oracle_found50']:+.2f} "
            f"| {r['geo_rank_median']:.0f} / {r['geo_auc_mean']:.2f} / "
            f"{r['geo_found50']:.2f} "
            f"| {r['pat_rank_median']:.0f} / {r['pat_auc_mean']:.2f} / "
            f"{r['pat_found50']:.2f} "
            f"| {r['volt_rank_median']:.0f} / {r['volt_auc_mean']:.2f} / "
            f"{r['volt_found50']:.2f} ({r['volt_coverage_mean']:.2f}) |")
    co, ce = pk["corridor"], pk["concept"]
    L += [
        "",
        f"Corridor cost decomposition: pair-blindness {co['oracle_rank_median']:.0f} → "
        f"{co['volt_rank_median']:.0f} (self-selected pairs, region coverage "
        f"{co['volt_coverage_mean']:.2f}), arbitration interleaving "
        f"{co['volt_rank_median']:.0f} → {co['max_rank_median']:.0f}. Concept caveat: "
        "in DOC space every channel sits near chance AUC "
        f"(pat {ce['pat_auc_mean']:.2f}, geo {ce['geo_auc_mean']:.2f}) — the rank-"
        f"{ce['max_rank_median']:.0f} recovery rides the generic supporter effect (Task "
        "C's diagnosis); the miner's real concept power stays in ITEMSET space "
        "(16/16 @ 128.5/36,199), which a doc-ranked composite cannot express.",
    ]
    o = s["overall"]
    L += [
        "",
        "Arbitration ablation (all 48 mixed trials; median rank / mean AUC / found@50):",
        "",
        "| Rule | rank | AUC | found@50 |",
        "|---|---|---|---|",
        f"| **max percentile (main)** | **{o['max_rank_median']:.0f}** "
        f"| **{o['max_auc_mean']:.2f}** | **{o['max_found50']:.2f}** |",
        f"| mean percentile | {o['mean_rank_median']:.0f} | {o['mean_auc_mean']:.2f} "
        f"| {o['mean_found50']:.2f} |",
        f"| RRF (k=60) | {o['rrf_rank_median']:.0f} | {o['rrf_auc_mean']:.2f} "
        f"| {o['rrf_found50']:.2f} |",
        f"| geo+volt (miner dropped) | {o['geo_volt_rank_median']:.0f} "
        f"| {o['geo_volt_auc_mean']:.2f} | {o['geo_volt_found50']:.2f} |",
        "",
        f"Probes in composite top-50 (max rule, mean over 48): "
        f"{np.mean([pk[k]['max_probes_top50_mean'] for k in pk]):.1f}.",
        "",
    ]
    return L


# ===================================================================== Part B: frontier
def radial_frontier(ctx: dict, banked: dict, args) -> dict:
    """PAIRED design at the BANKED trials' own centers: removal size swept at the
    16 locations the banked construction machinery accepted (seeds 1000+t), via
    deterministic m-nearest-fg balls, so (i) the size axis is isolated from
    location difficulty, (ii) the locations are production-representative (the
    banked benchmark's own), (iii) the banked-budget row anchors to banked
    recovery. Matched controls per (center, m) = same-m hypothetical removals at
    the other banked centers. Ground truth fixed at the banked contract (orphan
    core, k_gt=10, ratio 1.15; min_core relaxed to 3, min_bg 15 as the
    witness/footprint gates, reported as the construct rate).

    Two rejected designs on record (run, then replaced): independent
    plant_radial resampling per cell (the v1 sweep) lets the accept loop hunt
    different-difficulty locations per size — rank zigzags 1074 (m=16) -> 4004
    (m=32) -> 13 (m=64) -> 734 (m=128) -> 1 (m=256), pure location lottery; v2
    (paired at 16 RANDOM dense-pool centers) shows the complementary fact that
    ≤32-doc removals at typical locations leave no witnessed footprint at all
    (0/16) and even 256-doc ones recover weakly (found@50 0.47) — the
    benchmark's resampling construction is itself a location filter, which is
    why the spec below conditions on the banked-accepted locations."""
    from sklearn.neighbors import NearestNeighbors

    Zf, Zb_s, probes = ctx["Zf"], ctx["Zb_s"], ctx["probes"]
    scorers: dict[int, GapScorer] = {}
    for k in FRONTIER_KS:
        sc = GapScorer(bg=Zb_s, probes=probes, k=k)
        sc.set_r0_from_fg(Zf)
        _ = sc.n_Bloc
        scorers[k] = sc
    # banked centers + identity gate: the k=10 frontier scoring path reproduces
    # the banked fused rank at trial 0, and the banked-budget removal at each
    # center is bit-identical to the banked plant
    centers, banked_m, banked_removed = [], [], []
    for t, gap, fgp in replant_radial(ctx, banked, args.trials):
        c_idx = int(np.where((Zf == gap.center).all(axis=1))[0][0])
        centers.append(c_idx)
        banked_m.append(int(gap.meta["n_removed"]))
        banked_removed.append(np.sort(gap.removed_fg))
        if t == 0:
            out = scorers[10].score(fgp)
            got = rank_of(fused_scores(out, veto_mask(scorers[10], out)),
                          _member_mask(scorers[10].cand.shape[0], gap.region_bg),
                          scorers[10].real_mask)
            want = banked["task_a"]["trials"][t]["scoped"]["fused"]["best_rank_real"]
            if got != want:
                raise RuntimeError(f"frontier identity failed: {got} != banked {want}")
            log(f"identity: banked radial fused rank reproduced through frontier path ({got})")
    centers = np.asarray(centers)
    max_m = max(FRONTIER_SIZES)
    nn_fg = NearestNeighbors(n_neighbors=max_m, algorithm="brute").fit(Zf)
    d_all, idx_all = nn_fg.kneighbors(Zf[centers])  # center is its own 0th neighbor
    nn_bg = NearestNeighbors(algorithm="brute").fit(Zb_s)
    # plant identity: at the banked budget, the deterministic ball at each banked
    # center reproduces the banked removed set EXACTLY
    for ci in range(args.trials):
        got_removed = np.sort(idx_all[ci, : banked_m[ci]])
        if not np.array_equal(got_removed, banked_removed[ci]):
            raise RuntimeError(f"frontier center {ci}: ball != banked removed set")
    log(f"banked centers recovered ({args.trials}); banked removed sets reproduced "
        f"exactly; banked budgets median {int(np.median(banked_m))}")

    def ball(ci: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        removed = idx_all[ci, :m]
        r = float(d_all[ci, m - 1])
        bg_in = nn_bg.radius_neighbors(Zf[centers[ci] : centers[ci] + 1], radius=r)[1][0]
        core = bg_in[orphaned_core(Zb_s[bg_in], Zf,
                                   np.delete(Zf, removed, axis=0), k=10, ratio=1.15)]
        return removed, bg_in, core

    MIN_BG, MIN_CORE = 15, 3
    cells = []
    for size in FRONTIER_SIZES:
        rows = []
        plants = [ball(ci, size) for ci in range(args.trials)]
        for ci, (removed, bg_in, core) in enumerate(plants):
            row = {"trial": ci, "n_removed": int(removed.size),
                   "n_bg_in_ball": int(bg_in.size), "n_core": int(core.size),
                   "constructed": bool(bg_in.size >= MIN_BG and core.size >= MIN_CORE)}
            if row["constructed"]:
                controls = []
                for cj, (_, c_bg, c_core) in enumerate(plants):
                    if cj == ci:
                        continue
                    c_core2 = np.setdiff1d(c_core, bg_in)
                    if c_bg.size >= MIN_BG and c_core2.size >= MIN_CORE:
                        controls.append(c_core2)
                    if len(controls) >= 5:
                        break
                if len(controls) < 2:
                    row["constructed"] = False
                else:
                    neg = np.concatenate(controls)
                    fgp = np.delete(Zf, removed, axis=0)
                    for k in FRONTIER_KS:
                        sc = scorers[k]
                        out = sc.score(fgp)
                        s = fused_scores(out, veto_mask(sc, out))
                        member = _member_mask(sc.cand.shape[0], core)
                        rr = rank_of(s, member, sc.real_mask)
                        row[f"k{k}"] = {
                            "rank": rr, "auc": auc_vs_controls(s, core, neg),
                            "found50": bool(rr <= FOUND_AT),
                            "core_veto_frac": float(veto_mask(sc, out)[core].mean()),
                        }
            rows.append(row)
        ok = [r for r in rows if r["constructed"]]
        cell = {"size": size, "n_constructed": len(ok), "n_trials": len(rows),
                "core_median": med(ok, "n_core") if ok else None,
                "rows": rows}
        for k in FRONTIER_KS:
            cell[f"k{k}_rank_median"] = med(ok, f"k{k}", "rank") if ok else None
            cell[f"k{k}_auc_mean"] = med(ok, f"k{k}", "auc", fn=np.mean) if ok else None
            cell[f"k{k}_found50"] = med(ok, f"k{k}", "found50", fn=np.mean) if ok else None
            cell[f"k{k}_core_veto_mean"] = (med(ok, f"k{k}", "core_veto_frac", fn=np.mean)
                                            if ok else None)
        cells.append(cell)
        parts = []
        for k in FRONTIER_KS:
            parts.append(f"k{k}: rank={_fmt(cell[f'k{k}_rank_median'])} "
                         f"auc={_fmt(cell[f'k{k}_auc_mean'], '.2f')} "
                         f"f50={_fmt(cell[f'k{k}_found50'], '.2f')}")
        log(f"B radial size={size}: constructed={len(ok)}/{len(rows)} "
            f"core_med={_fmt(cell['core_median'])} " + " ".join(parts))
    return {"cells": cells}


def corridor_frontier(ctx: dict, banked: dict, structure, args) -> dict:
    """Sub-budget corridor removals: m innermost band members of each banked pair."""
    from artifacts_scripts.gap_detection.synth import _axis_t, corridor_band_bg

    Zf, Zb_s = ctx["Zf"], ctx["Zb_s"]
    labels, cents, pairs = structure
    # the 16 banked pair configs, in banked trial order (plant identity via replant)
    banked_pairs = []
    for t, gap, info in replant_corridor(ctx, banked, args.trials, structure):
        banked_pairs.append({"t": t, "meta": dict(gap.meta)})
    all_pair_dicts = {(p["a"], p["b"]): p for p in pairs}

    def sub_removal(meta: dict, m: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = meta["a"], meta["b"]
        mem = np.where((labels == a) | (labels == b))[0]
        s = np.abs(_axis_t(Zf[mem], cents[a], cents[b]) - 0.5)
        inband = mem[s <= meta["w"]]
        order = np.argsort(s[s <= meta["w"]], kind="stable")
        removed = inband[order[:m]]
        band_bg = corridor_band_bg(Zb_s, cents, all_pair_dicts[(a, b)])
        return removed, band_bg

    cells = []
    for size in CORRIDOR_SIZES:
        rows = []
        for bp in banked_pairs:
            meta = bp["meta"]
            m = int(min(size, meta["n_removed"]))
            removed, band_bg = sub_removal(meta, m)
            fgp = np.delete(Zf, removed, axis=0)
            core = band_bg[orphaned_core(Zb_s[band_bg], Zf, fgp, k=10, ratio=1.05)]
            row = {"trial": bp["t"], "size_nominal": size, "size_realized": m,
                   "n_core": int(core.size), "constructed": bool(core.size >= 3)}
            if row["constructed"]:
                # matched controls: same-m hypothetical removals at other banked pairs
                rngc = np.random.default_rng(5000 + size + bp["t"])
                controls = []
                others = [q for q in banked_pairs if q["t"] != bp["t"]]
                rngc.shuffle(others)
                for q in others:
                    qm = int(min(size, q["meta"]["n_removed"]))
                    q_removed, q_band = sub_removal(q["meta"], qm)
                    q_fgp = np.delete(Zf, q_removed, axis=0)
                    q_core = q_band[orphaned_core(Zb_s[q_band], Zf, q_fgp, k=10, ratio=1.05)]
                    q_core = np.setdiff1d(q_core, band_bg)
                    if q_core.size >= 3:
                        controls.append(q_core)
                    if len(controls) >= 5:
                        break
                labels_p = np.delete(labels, removed)
                pair = {"a": meta["a"], "b": meta["b"], "lens_r": meta["lens_r"]}
                r = voltage_pair_field(fgp, Zb_s, labels_p, cents, pair)
                if r is None or len(controls) < 2:
                    row["constructed"] = False
                else:
                    sc, _ = r
                    member = _member_mask(N_SCOPED_BG, core)
                    rr = rank_of(sc, member, np.isfinite(sc))
                    row.update({
                        "rank": rr, "found50": bool(rr <= FOUND_AT),
                        "auc": auc_vs_controls(sc, core, np.concatenate(controls)),
                    })
            rows.append(row)
        ok = [r for r in rows if r["constructed"]]
        cell = {"size": size, "n_constructed": len(ok), "n_trials": len(rows),
                "realized_median": med(rows, "size_realized"),
                "core_median": med(ok, "n_core") if ok else None,
                "rank_median": med(ok, "rank") if ok else None,
                "auc_mean": med(ok, "auc", fn=np.mean) if ok else None,
                "found50": med(ok, "found50", fn=np.mean) if ok else None,
                "rows": rows}
        cells.append(cell)
        log(f"B corridor size={size}: constructed={len(ok)}/{len(rows)} "
            f"realized_med={_fmt(cell['realized_median'])} "
            f"core_med={_fmt(cell['core_median'])} rank={_fmt(cell['rank_median'])} "
            f"auc={_fmt(cell['auc_mean'], '.2f')}")
    return {"cells": cells}


def _spec_size(cells: list[dict], auc_key, found_key, min_construct: float = 0.5,
               min_found: float = 0.5):
    """Smallest size whose cell AND all larger cells are recoverable: construct
    rate >= min_construct, mean AUC >= AUC_SPEC_THR, found@50 >= min_found
    (AUC alone admits cells whose median rank is ~1000 — causal signal without
    production-visible recovery)."""
    def ok(c):
        rate = c["n_constructed"] / max(c["n_trials"], 1)
        auc, found = c.get(auc_key), c.get(found_key)
        return (rate >= min_construct and auc is not None and auc >= AUC_SPEC_THR
                and found is not None and found >= min_found)

    for i, c in enumerate(cells):
        if ok(c) and all(ok(c2) for c2 in cells[i:]):
            rate = c["n_constructed"] / max(c["n_trials"], 1)
            return c["size"], rate, c.get(auc_key), c.get(found_key)
    return None, None, None, None


def part_b(args) -> None:
    banked = load_banked()
    ctx = build_ctx()
    assert ctx["n_scoped"] == N_SCOPED_BG
    structure = corridor_structure(ctx)
    reproduce_banked_voltage(ctx, banked, structure)
    rad = radial_frontier(ctx, banked, args)
    cor = corridor_frontier(ctx, banked, structure, args)

    spec = {}
    for k in FRONTIER_KS:
        size, rate, auc, found = _spec_size(rad["cells"], f"k{k}_auc_mean", f"k{k}_found50")
        spec[f"radial_k{k}"] = {"min_size": size, "construct_rate": rate,
                                "auc": auc, "found50": found}
    size, rate, auc, found = _spec_size(cor["cells"], "auc_mean", "found50")
    spec["corridor_voltage"] = {"min_size": size, "construct_rate": rate,
                                "auc": auc, "found50": found}
    summary = {
        "config": {
            "radial": f"sizes {list(FRONTIER_SIZES)} x detector k {list(FRONTIER_KS)}, "
                      f"PAIRED at the {args.trials} BANKED trial centers (the existing "
                      "construction machinery's own accepted locations, seeds 1000+t; "
                      "banked removed sets reproduced exactly at the banked budgets): "
                      "deterministic m-nearest-fg balls at the SAME centers per size; "
                      "matched controls = same-m hypothetical removals at the other "
                      "banked centers; detector = fused r_F+veto; ground truth fixed at "
                      "the banked contract (orphan core, k_gt=10, ratio 1.15; "
                      "witness/footprint gates min_bg 15 / min_core 3 = construct rate). "
                      "Two designs run and rejected on record: per-cell independent "
                      "resampling (location lottery: rank zigzag 1074->4004->13->734->1 "
                      "across sizes) and pairing at random dense-pool centers (≤32-doc "
                      "removals leave no witnessed footprint, 0/16; found@50 only 0.47 "
                      "even at 256 — the resampling construction is itself a location "
                      "filter)",
            "corridor": f"sizes {list(CORRIDOR_SIZES)} (m innermost band members of the "
                        "16 banked pairs, capped at the pair's full band budget); "
                        "detector = oracle-pair voltage (k_graph=10); core ratio 1.05, "
                        "min_core 3; matched controls = same-m hypothetical removals at "
                        "other banked pairs",
            "spec_rule": "recoverable cell := construct rate >= 0.5 AND mean AUC >= "
                         f"{AUC_SPEC_THR} AND found@50 >= 0.5 among constructed trials, "
                         "required to HOLD for all larger sizes (AUC alone admits cells "
                         "with causal signal but median rank ~1000)",
        },
        "spec": spec,
    }
    if not args.dry:
        existing = load_banked().get("task_t1_comp", {})
        existing["part_b"] = {"summary": summary,
                              "radial_cells": rad["cells"], "corridor_cells": cor["cells"]}
        update_results("task_t1_comp", existing)
        update_report("TASK_T1_FRONTIER", report_b(summary, rad["cells"], cor["cells"]))
    log("B spec: " + json.dumps(_py(spec)))


def report_b(s: dict, rad_cells: list[dict], cor_cells: list[dict]) -> list[str]:
    d = datetime.now().date().isoformat()
    L = [
        f"## Track 1 finish, Part B — sensitivity frontier (minimum recoverable gap) ({d})",
        "",
        f"Radial: {s['config']['radial']}.",
        f"Corridor: {s['config']['corridor']}.",
        "",
        "| removed | constructed | core (med) | k=5 rank/AUC/found@50 | k=10 rank/AUC/"
        "found@50 | k=20 rank/AUC/found@50 |",
        "|---|---|---|---|---|---|",
    ]

    def f(c, k):
        if c[f"k{k}_rank_median"] is None:
            return "—"
        return (f"{c[f'k{k}_rank_median']:.0f} / {c[f'k{k}_auc_mean']:.2f} / "
                f"{c[f'k{k}_found50']:.2f}")

    for c in rad_cells:
        L.append(f"| {c['size']} | {c['n_constructed']}/{c['n_trials']} "
                 f"| {_fmt(c['core_median'])} "
                 f"| {f(c, 5)} | {f(c, 10)} | {f(c, 20)} |")
    L += [
        "",
        "| corridor removed (realized med) | constructed | core (med) | rank / AUC / "
        "found@50 (oracle-pair voltage, subgraph) |",
        "|---|---|---|---|",
    ]
    for c in cor_cells:
        cell = ("—" if c["rank_median"] is None else
                f"{c['rank_median']:.0f} / {c['auc_mean']:.2f} / {c['found50']:.2f}")
        L.append(f"| {c['size']} ({c['realized_median']:.0f}) "
                 f"| {c['n_constructed']}/{c['n_trials']} "
                 f"| {_fmt(c['core_median'])} "
                 f"| {cell} |")
    sp = s["spec"]
    k10, k5, k20 = sp["radial_k10"], sp["radial_k5"], sp["radial_k20"]
    cv = sp["corridor_voltage"]
    c16 = next(c for c in rad_cells if c["size"] == 16)
    c32 = next(c for c in rad_cells if c["size"] == 32)
    c128 = next(c for c in rad_cells if c["size"] == 128)
    L += [
        "",
        f"Spec rule: {s['config']['spec_rule']}. Banked-budget anchor: the size-128 row "
        f"(banked budgets median 124) reproduces banked recovery — k=10 rank "
        f"{c128['k10_rank_median']:.0f} / AUC {c128['k10_auc_mean']:.2f} vs banked "
        "6.5 / 0.83.",
        "",
        "**Minimum-recoverable-gap spec (the sentence production claims must quote):** "
        "on this corpus (30k docs, 1,446-doc fg, 128-d LSA), the minimum recoverable gap "
        f"is a **{k10['min_size']}-doc removal** (~{100 * k10['min_size'] / 1446:.0f}% of "
        f"the field): fused r_F+veto at k=5–10 recovers {k10['min_size']}-doc radial "
        f"pockets (construct {k10['construct_rate']:.2f}, AUC {k5['auc']:.2f}–"
        f"{k10['auc']:.2f}, found@50 {k10['found50']:.2f}–{k5['found50']:.2f}) and "
        f"oracle-pair voltage recovers {cv['min_size']}-doc corridor bands "
        f"(construct {cv['construct_rate']:.2f}, AUC {cv['auc']:.2f}, found@50 "
        f"{cv['found50']:.2f}); at k=20 the radial threshold doubles to "
        f"{k20['min_size']} (over-smoothing); below that, the footprint itself collapses "
        f"— 8-doc removals leave no orphan core anywhere, 16/32-doc removals leave cores "
        f"at only {c16['n_constructed']}/{c16['n_trials']} / "
        f"{c32['n_constructed']}/{c32['n_trials']} of the banked locations and rank "
        f"~{c16['k10_rank_median']:.0f}/{c32['k10_rank_median']:.0f} (k=10) — causal AUC "
        "without production-visible recovery.",
        "",
    ]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    {"a": part_a, "b": part_b}[args.part](args)
    log("done")


if __name__ == "__main__":
    main()
