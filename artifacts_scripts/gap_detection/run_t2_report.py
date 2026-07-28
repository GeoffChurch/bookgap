"""Render the TASK_T2_WIKI report section from results.json:task_t2_wiki."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from datetime import datetime

from artifacts_scripts.gap_detection.run_followups import load_banked, update_report


def verdict_for(fill: dict) -> str:
    lo = fill.get("ci_lo_clustered", fill["ci_lo"])
    hi = fill.get("ci_hi_clustered", fill["ci_hi"])
    if lo > 1.0:
        return "POSITIVE (clustered CI > 1)"
    if hi < 1.0:
        return "INVERTED (clustered CI < 1)"
    return "honest negative (clustered CI spans 1)"


def main() -> None:
    r = load_banked()["task_t2_wiki"]
    p0, p1, p2m, p3 = (r["phase0_gate"], r["phase1_creation_dates"],
                       r["phase2_t_corpus"], r["phase3"])
    cfg = p3["config"]
    d = datetime.now().date().isoformat()
    chosen = p0["choice"]
    cal = p1["full"]
    L = [
        f"## Track 2 — wiki temporal holdout: predict at T, score (T, T+Δ] arrivals ({d})",
        "",
        "First REAL-gap validation: the Track-1 suite predicts gap locations at a past",
        "snapshot T from ≤T data only; predictions are scored against which articles",
        "actually ARRIVED in (T, T+Δ], vs growth-matched null locations. P2 residual",
        "components and the P3a all-pairs change field act as OUTCOME instruments only.",
        "",
        "### Phase 0 — store discovery + arrival-density gate",
        "",
        "`wiki_400k` is a bounded cirrus-dump slice: the 400k OLDEST surviving page_ids",
        "(12..1,348,640; file order exactly id-sorted), creation span 2001-08-28..2004-12-31.",
        "The dispatched candidate years 2014–2022 therefore hold ZERO arrivals (gate rows",
        "recorded); the design slides T into the store's era. Density table (rough proxy,",
        "240 exact timestamps; in-scope = q0.25 scope rule from ≤T members of the banked",
        "subsample, banked-LSA embedding, identity 1.2e-07):",
        "",
        "| T | pre-T eligible | 2y arrivals (eligible) | in-scope | banked fg at T |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for row in p0["table"]:
        if row["delta_years"] != 2 or row["T"] in seen or row["T"].startswith(("2014", "2018", "2022")):
            continue
        seen.add(row["T"])
        ns = row["n_arrivals_in_scope"]
        L.append(f"| {row['T']} | {row['n_pre_store_eligible']:,} "
                 f"| {row['n_arrivals_eligible']:,} | {ns if ns is None else format(ns, ',')} "
                 f"| {row['n_fg_preT_banked']} |")
    L += [
        "| 2014/2018/2022 (dispatched) | 332,753 | 0 | — | 1,446 |",
        "",
        f"GATE (≥150 in-scope arrivals): PASSED. **Pre-committed on counts only: "
        f"T = {chosen['T']}, Δ = {chosen['delta_years']}y** (window {chosen['window']}; "
        f"store ceiling {chosen['store_ceiling']} truncates by 1 day; Δ=2y and 4y identical "
        "at this T). Rule: maximize in-scope arrivals subject to pre-T eligible ≥ 32k "
        "(size-matched 30k T-corpus).",
        "",
        "### Phase 1 — creation dates (page_id proxy, exact-API calibrated)",
        "",
        f"{cal['n_calibration']:,} exact first-revision timestamps (stratified + boundary "
        f"band; QC: {cal['qc']['missing']} missing / {cal['qc']['redirect']} redirects / 0 "
        "errors; every response cached). Isotonic id→date proxy, held-out validation: "
        f"median |err| {cal['validation_all']['median_abs_err_days']:.2f} d, p90 "
        f"{cal['validation_all']['p90_abs_err_days']:.1f} d; near-boundary (±1y of T) p90 "
        f"{cal['validation_near_boundary']['p90_abs_err_days']:.1f} d — far inside the "
        "180-day gate. Wrinkle rates (held-out |err|; the redirect-turned-article / "
        "history-merge / undeletion tail): "
        f">90d {p1['full']['wrinkle_rates_heldout']['gt_90d']:.1%}, "
        f">180d {p1['full']['wrinkle_rates_heldout']['gt_180d']:.1%}, "
        f">365d {p1['full']['wrinkle_rates_heldout']['gt_365d']:.1%}.",
        "",
        "### Phase 2 — leakage-free T-corpus",
        "",
        f"Uniform 30k sample of the {p2m['n_pool']:,}-doc pre-T eligible pool (banked pre-T "
        "docs force-included); TF-IDF/LSA-128/keywords/KMeans-40 ALL refit on ≤T docs "
        "(leakage guard unit-pinned). SANITY FINDING: the seeded procedure does NOT re-find "
        "the society/law/politics field at T — the banked fg FRAGMENTS "
        f"({', '.join(str(b['cluster']) + ':' + str(b['n_banked']) for b in p2m['banked_fragmentation_top5'])} "
        f"of {p2m['fg_overlap_sanity_final']['n_banked_fg_present']} present members), and "
        "the procedure's coherence pick is a generic-terminology cluster (recall 0.075) — "
        "the plant-era interleave≥5 gate excludes every topical cluster. Continuity rule "
        "(deterministic, ON RECORD): fg = max-banked-overlap size-qualified cluster = "
        f"academia/philosophy/economics (n={p2m['max_overlap_cluster']['size']}, "
        f"banked recall {p2m['fg_overlap_sanity_final']['recall_banked_in_t_fg']:.2f}; terms: "
        f"{', '.join((p2m.get('fg_final_terms') or [])[:8])}). Scope: "
        f"{p2m['scope']['n_scoped']:,} of {p2m['scope']['n_bg']:,} bg. Frozen-transform "
        f"embeddings: {p2m['arrivals']['n_eligible']:,} arrivals, "
        f"{p2m['growth_window']['n_eligible']:,} growth-window pages. Approximations on "
        "record: 2026 text for pre-T docs, deletion survivorship, proxy dates, 1-day window "
        "truncation.",
        "",
        "### Phase 3 — predictions vs arrivals (the measurement)",
        "",
        f"Pre-committed: K={cfg['K']} locations/detector, M={cfg['M']} matched controls "
        f"(covariates: {', '.join(cfg['covariates'])}; scope membership by construction; "
        "controls outside every flagged neighborhood and center-disjoint), radius = "
        f"{cfg['radius_rule']}, B={cfg['n_boot']} bootstrap. Fill = arrivals within the "
        "location radius. MATCHING CALIBRATION (against the machinery null only, never "
        "detector outcomes): the first-pass matcher (tercile strata + full disjointness "
        "d > r_f + r_c) biased the random-location null to 1.27 [1.02, 1.56] — on the unit "
        "sphere r_f + r_c approaches the diameter and annihilates the sparse tail, forcing "
        "dense controls (density SMD 0.68-1.72; banked under "
        "`task_t2_wiki.phase3_v1_sum_disjointness`). Final matcher: hard per-covariate "
        "calipers (0.2/0.5/0.5 sigma, progressive doubling) + center-disjointness "
        "d > max(r_f, r_c): random-null ratios over 5 seeds 1.04/0.77/0.84/0.96/1.13 (all "
        "CIs straddle 1), balance SMD <= 0.07.",
        "",
        "| Detector | fill ratio | naive 95% CI | clustered CI (n groups) | "
        "mean fill flagged/null | balance SMD (dens/dist/grow) | P@50 vs P2 | "
        "P@50 vs P3a | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("fused", "miner", "voltage", "levelset", "random50"):
        pd = p3["per_detector"][name]
        f = pd["fill"]
        pk = pd["precision_at_k"]
        b = pd["covariate_balance_smd"]
        lab = name if name != "random50" else "random50 (machinery null)"
        L.append(
            f"| {lab} | {f['ratio']:.2f} | [{f['ci_lo']:.2f}, {f['ci_hi']:.2f}] "
            f"| [{f['ci_lo_clustered']:.2f}, {f['ci_hi_clustered']:.2f}] ({f['n_groups']}) "
            f"| {f['mean_fill_flagged']:.1f} / {f['mean_fill_controls']:.1f} "
            f"| {b['log_rB10']:.2f}/{b['log_rF10']:.2f}/{b['log1p_growth']:.2f} "
            f"| {pk['vs_p2_residual_tf01']:.2f} | {pk['vs_p3a_peaks']:.2f} "
            f"| {verdict_for(f)} |")
    ch = p3["chance_hit_rates_poolwide"]
    L += [
        "",
        f"Chance hit rates (pool-wide, same neighborhood rule): vs P2 "
        f"{ch['vs_p2_residual_tf01']:.3f}, vs P3a {ch['vs_p3a_peaks']:.3f}. "
        f"NON-INDEPENDENCE caveat: the voltage predictor and the P3a instrument are "
        f"flow-share fields on near-identical graphs/electrodes — field-level Spearman "
        f"{p3['voltage_static_vs_p3a_spearman']:.2f} — so voltage's P@50 vs P3a is "
        "partly self-correlation, not validation.",
    ]
    ag = p3["instrument_agreement"]
    L += [
        "",
        f"Outcome instruments: P2 residual components (before−after full statistic, fg "
        f"grown by {cfg['p2_n_fg_arrivals']:,} frozen-KMeans-assigned arrivals, bg fixed; "
        f"top-frac {cfg['p2']['top_frac']}) -> {ag['n_p2_tf01']} member docs; P3a change "
        f"field ({cfg['p3a_pairs']} pairs, {cfg['p3a_arrival_nodes']:,} arrival nodes of "
        f"{cfg['n_in_scope_arrivals_T_geometry']:,} in-scope, cap "
        f"{cfg['p3a']['arrival_node_cap']:,}) -> top-1% peaks {ag['n_p3a_peaks']}. "
        f"Instrument agreement (sanity): Jaccard(P2 tf01, P3a peaks) = "
        f"{ag['jaccard_p2tf01_p3apeaks']}, (P2 tf05, P3a) = {ag['jaccard_p2tf05_p3apeaks']}.",
        "",
        "Realized-change regions, top components (titles only):",
        "",
    ]
    for i, q in enumerate(p3["realized_change_top10"][:6], 1):
        L.append(f"- {i}. n_eff={q['n_eff']}, size={q['size']}: "
                 + "; ".join(q["top_titles"][:4]))
    L += [
        "",
        "Top miner fg-absent patterns at T (keyword labels):",
        "",
        "- " + " · ".join("{" + ", ".join(it["keywords"]) + "}"
                          for it in p3["miner_top_patterns"][:6]),
        "",
        "**Verdict.** With the machinery null calibrated (random50 = "
        f"{p3['per_detector']['random50']['fill']['ratio']:.2f}, clustered CI "
        f"[{p3['per_detector']['random50']['fill']['ci_lo_clustered']:.2f}, "
        f"{p3['per_detector']['random50']['fill']['ci_hi_clustered']:.2f}]) and location "
        "near-duplication handled by the clustered bootstrap, exactly ONE predictor "
        "carries real-arrival signal: absolute LEVEL-SET components — fill ratio "
        f"{p3['per_detector']['levelset']['fill']['ratio']:.2f}, clustered CI "
        f"[{p3['per_detector']['levelset']['fill']['ci_lo_clustered']:.2f}, "
        f"{p3['per_detector']['levelset']['fill']['ci_hi_clustered']:.2f}] over "
        f"{p3['per_detector']['levelset']['fill']['n_groups']} location groups (caveat: "
        "all 50 locations needed caliper relaxation, max level 2; residual balance "
        "0.11/0.21/0.25σ). The Track-1 planted-gap champion (scoped fusion) is an honest "
        "negative on REAL arrivals (1.10, CI spans 1): the emptiest in-scope points fill "
        "at the matched-control rate. The miner's naive positive (1.26, naive CI "
        "[1.14, 1.39]) DISSOLVES under honesty corrections: its top-50 fg-absent patterns "
        "are generic register pairs ({use, used}, {called, used}, ...) whose supporter "
        "centroids collapse to 4 distinct neighborhoods — clustered CI [0.89, 1.31] = "
        "inconclusive; the Track-1 miner config needs pattern-diversity control before it "
        "can predict. Voltage corridors: 1.80 but CI [0.85, 2.99] spans 1 (heavy-tailed "
        "fill) — honest negative. INVERSION vs Track 1: the detector REFUTED for "
        "small-plant retrieval (level-sets surface large natural gaps) is the one whose "
        "structures actually FILL over (T, T+Δ]; extreme single-point emptiness (fused's "
        "specialty) does not out-fill growth-matched controls. The outcome instruments "
        "agree with NOTHING here including each other (Jaccard ≤ 0.005; all P@50 at/below "
        "the pool-wide chance rates 0.063/0.052 except voltage-vs-P3a 0.48 = "
        "self-correlation, Spearman 0.61): at arrivals ≈ 8.5× corpus size the 2-year "
        "regime is explosive growth, far from the marginal-change regime the plant "
        "validation modeled — P2/P3a remain validated as CHANGE LOCALIZERS, not as "
        "arrival-prediction referees. Design lessons banked for the OA phase: calibrate "
        "the matching machinery on random-location nulls FIRST; clustered bootstrap over "
        "location groups; diversity-dedupe pattern locations; expect field fragmentation "
        "across deep time.",
        "",
    ]
    update_report("TASK_T2_WIKI", L)
    print("report section TASK_T2_WIKI updated")


if __name__ == "__main__":
    main()
