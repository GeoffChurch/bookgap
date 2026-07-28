"""Render the TASK_T2_OA report section from results.json:task_t2_oa (+ phase2 meta)."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from datetime import datetime

from artifacts_scripts.gap_detection.oa_corpus import CACHE_DIR
from artifacts_scripts.gap_detection.run_followups import load_banked, update_report


def verdict_for(fill: dict) -> str:
    lo = fill.get("ci_lo_clustered", fill["ci_lo"])
    hi = fill.get("ci_hi_clustered", fill["ci_hi"])
    if lo > 1.0:
        return "POSITIVE (clustered CI > 1)"
    if hi < 1.0:
        return "INVERTED (clustered CI < 1)"
    return "honest negative (CI spans 1)"


def main() -> None:
    r = load_banked()["task_t2_oa"]
    p0 = r["phase0_gate"]
    p3 = r["phase3"]
    cfg = p3["config"]
    p2m = json.loads((CACHE_DIR / "t2_oa_phase2_meta.json").read_text())
    d = datetime.now().date().isoformat()
    ch = p0["choice"]
    gT, gE = cfg["graph_T"], cfg["graph_T_plus_delta"]

    L = [
        f"## Track 2 — OA NLP temporal holdout: predict at T, score (T, T+Δ] arrivals ({d})",
        "",
        "The wiki temporal holdout re-run in a MARGINAL-growth regime (OpenAlex NLP, T10181)",
        "with LEAK-FREE dated citation edges — the OA-specific win the wiki store could not give.",
        "The Track-1 suite predicts gap locations at T from ≤T data only; predictions are scored",
        "against which works actually ARRIVED in (T, T+Δ], vs growth-matched null locations. P2",
        "residual components and the P3a citation-graph flow-change field are OUTCOME instruments",
        "only. Every fitted object (TF-IDF, LSA-128, KMeans fg, scope quantiles, kNN scales) is",
        "refit on ≤T works ONLY; arrivals embed via the FROZEN T-transform; the dated graph at T",
        "excludes every citing act with date > T (toy-fixture leakage guards green).",
        "",
        "### Phase 0 — acquisition + arrival-density gate",
        "",
        f"Universe (LOCKED): OpenAlex `primary_topic.id:T10181` (NLP Techniques). Cursor-paged",
        f"pub-2008–2019 on the polite pool → **{p0['n_works_universe']:,} works** cached "
        f"(all dated). Abstract coverage {p0['abstract_coverage_any']:.1%} any / "
        f"{p0['abstract_coverage_ge200']:.1%} ≥200ch (field-wide; missingness model = drop "
        "no-/short-abstract works from corpus + embedding). Dated citation graph: "
        f"{p0['n_edges_intra_universe']:,} intra-universe edges, {p0['n_works_with_inuniverse_ref']:,} "
        "works carry ≥1 in-universe reference.",
        "",
        "| T | Δ | pre-T eligible (≥200ch) | arrivals (eligible) | in-scope | provisional fg |",
        "|---|---|---|---|---|---|",
    ]
    for row in p0["table"]:
        ns = row["n_arrivals_in_scope"]
        fg = row["n_fg_preT_provisional"]
        L.append(
            f"| {row['T']} | {row['delta_years']}y | {row['n_pre_eligible']:,} "
            f"| {row['n_arrivals_eligible']:,} | {ns if ns is None else format(ns, ',')} "
            f"| {fg if fg is None else format(fg, ',')} |")
    L += [
        "",
        f"GATE (≥{p0['choice']['gate_min_in_scope']} in-scope arrivals): **PASSED on the LOCKED "
        f"universe** (no fallback). **Pre-committed on counts only: T = {ch['T']}, Δ = "
        f"{ch['delta_years']}y** (window {ch['window']}). In-scope arrivals "
        f"{ch['n_arrivals_in_scope']:,} ≈ {ch['n_arrivals_in_scope'] / row['n_pre_eligible']:.2f}× "
        "the pre-T pool — a MARGINAL regime (vs the wiki run's explosive 8.5×), which is the "
        "point: there is no editorial-mass-production analogue for citation 'filling'.",
        "",
        "### Phase 1 — exact dates + dated citation graph",
        "",
        "`publication_date` is exact (no id→date proxy — simpler than wiki). The dated graph "
        "reconstructs the citation structure as it stood at T with NO future leakage: an edge "
        "(citing→cited) is present at T iff the CITING work's date ≤ T (papers cite earlier "
        "work). Graph stats:",
        "",
        "| snapshot | dated nodes | edges | non-isolated | density | max in-deg |",
        "|---|---|---|---|---|---|",
        f"| T ({ch['T']}) | {gT['n_nodes_dated']:,} | {gT['n_edges']:,} | "
        f"{gT['n_nonisolated']:,} | {gT['density']:.2e} | {gT['max_in_degree']} |",
        f"| T+Δ | {gE['n_nodes_dated']:,} | {gE['n_edges']:,} | {gE['n_nonisolated']:,} "
        f"| {gE['density']:.2e} | {gE['max_in_degree']} |",
        "",
        "### Phase 2 — leakage-free T-corpus (discovered fg)",
        "",
        f"Pre-T pool: {p2m['n_pre_dated']:,} dated → {p2m['n_pre_abstracted']:,} with abstract "
        f"≥200ch (the {p2m['n_pre_dated'] - p2m['n_pre_abstracted']:,} dropped = missingness "
        f"model). T-corpus n={p2m['corpus']['n_docs']:,}, TF-IDF vocab "
        f"{p2m['corpus']['tfidf_vocab']:,}, LSA-128 EVR {p2m['corpus']['svd_evr_sum']} — ALL "
        "refit on ≤T docs. fg = DISCOVERED KMeans cluster (most-coherent size-qualified; no "
        f"label, no continuity rule), size {p2m['corpus']['fg_cluster']['size']:,}; top terms: "
        f"{', '.join(p2m['fg_top_terms'][:12])}. Scope (T geometry): {p2m['scope']['n_scoped']:,} "
        f"of {p2m['scope']['n_bg']:,} bg (q{p2m['scope']['quantile']}). Frozen-transform "
        f"embeddings: {p2m['arrivals']['n_eligible']:,} arrivals, "
        f"{p2m['growth_window']['n_eligible']:,} growth-window works.",
        "",
        "### Phase 3 — predictions vs arrivals (the measurement)",
        "",
        f"Pre-committed: K={cfg['K']} locations/detector, M={cfg['M']} matched controls "
        f"(covariates: {', '.join(cfg['covariates'])}; scope by construction; controls outside "
        f"every flagged neighborhood + center-disjoint), radius = {cfg['radius_rule']}, "
        f"B={cfg['n_boot']} bootstrap, clustered over greedy location groups. Fill = "
        f"{cfg['fill']}. Matching machinery calibrated on random-location nulls FIRST (random50 "
        "below ≈ 1.0); the wiki v1 full-disjointness bug that inflated the null to 1.27 is "
        "avoided by the banked caliper + center-disjoint matcher (reused verbatim).",
        "",
        "| Detector | fill ratio | clustered CI (n groups) | mean fill flagged/null | "
        "balance SMD (dens/dist/grow) | P@50 vs P2 | P@50 vs P3a | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ("fused", "miner", "voltage_emb", "voltage_cite", "levelset", "random50")
    for name in order:
        if name not in p3["per_detector"]:
            continue
        pd = p3["per_detector"][name]
        f = pd["fill"]
        pk = pd["precision_at_k"]
        b = pd["covariate_balance_smd"]
        lab = name if name != "random50" else "random50 (machinery null)"
        L.append(
            f"| {lab} | {f['ratio']:.2f} | [{f['ci_lo_clustered']:.2f}, "
            f"{f['ci_hi_clustered']:.2f}] ({f['n_groups']}) "
            f"| {f['mean_fill_flagged']:.1f} / {f['mean_fill_controls']:.1f} "
            f"| {b['log_rB10']:.2f}/{b['log_rF10']:.2f}/{b['log1p_growth']:.2f} "
            f"| {pk['vs_p2_residual_tf01']:.2f} | {pk['vs_p3a_peaks']:.2f} "
            f"| {verdict_for(f)} |")
    chh = p3["chance_hit_rates_poolwide"]
    ag = p3["instrument_agreement"]
    L += [
        "",
        f"Chance hit rates (pool-wide, same rule): vs P2 {chh['vs_p2_residual_tf01']:.3f}, "
        f"vs P3a {chh['vs_p3a_peaks']:.3f}. Voltage is reported on BOTH graphs: `voltage_emb` "
        "(embedding-kNN bg graph, the wiki form) and `voltage_cite` (the dated CITATION graph "
        f"at T, {cfg['voltage_cite']['pairs_used']}/{cfg['voltage_cite']['pairs_qualifying']} "
        "pairs solved) — the OA-only corridor.",
        "",
        f"Outcome instruments: P2 residual components (before−after full statistic, fg grown by "
        f"{cfg['p2_n_fg_arrivals']:,} frozen-KMeans-assigned arrivals, bg fixed; top-frac "
        f"{cfg['p2']['top_frac']}) → {ag['n_p2_tf01']} member works; **P3a flow-change on the "
        f"DATED CITATION GRAPH difference** (T vs T+Δ, scoped-bg nodes; {cfg['p3a_pairs']} pairs; "
        f"{cfg['p3a_n_edges_T']:,} → {cfg['p3a_n_edges_Tdelta']:,} undirected edges) → top-1% "
        f"peaks {ag['n_p3a_peaks']}. Instrument agreement (sanity): Jaccard(P2 tf01, P3a peaks) "
        f"= {ag['jaccard_p2tf01_p3apeaks']}.",
        "",
        "Realized-change regions, top components (W-ids only — face-validity is read from fg "
        "top-terms, never raw abstracts):",
        "",
    ]
    for i, q in enumerate(p3["realized_change_top10"][:6], 1):
        L.append(f"- {i}. n_eff={q['n_eff']}, size={q['size']}: " + ", ".join(q["top_wids"][:4]))
    L += [
        "",
        "Top miner fg-absent abstract-term patterns at T (keyword labels):",
        "",
        "- " + " · ".join("{" + ", ".join(it["keywords"]) + "}"
                          for it in p3["miner_top_patterns"][:6]),
        "",
        f"**fg top terms (face-validity anchor):** {', '.join(cfg['fg_top_terms'][:15])}.",
        "",
        "### Face-validity (keyword labels of flagged regions; no raw abstracts)",
        "",
        "The fg is dependency/syntactic PARSING. Region labels (log-odds keyword terms vs the "
        "scoped pool; full pass in `t2_oa_descriptive.md`) confirm the flagged regions are "
        "genuine post-2015-active parsing-ADJACENT NLP, not noise:",
        "- **levelset** (the positive) flags: POS tagging/HMM taggers (size-128 component), "
        "synchronous grammars/tree transducers, **MT evaluation metrics {mt, ter, judgments}** "
        "and **statistical machine translation/domain adaptation {smt, translation, machine}**, "
        "Chinese word segmentation/CRF, morphological analysis — the prerequisite + sibling "
        "topics of parsing, all of which went neural after 2015.",
        "- **miner** independently surfaces the **machine-translation gap** — its top fg-absent "
        "patterns are {machine, translation}, {statistical, translation}, {machine, "
        "statistical, translation}: MT is the largest post-2015 NLP growth area and is absent "
        "from the parsing fg.",
        "- **P3a citation-change peaks + voltage_cite** flag the **CoNLL / treebank-benchmark** "
        "cluster {conll, treebanks, penn, joint, dependencies, benchmark} — precisely where "
        "citation activity concentrated post-2015 (Universal Dependencies / CoNLL parsing "
        "shared tasks).",
        "",
        "**Verdict.** With the matching machinery calibrated FIRST on random-location nulls "
        f"(random50 = {p3['per_detector']['random50']['fill']['ratio']:.2f}, clustered CI "
        f"[{p3['per_detector']['random50']['fill']['ci_lo_clustered']:.2f}, "
        f"{p3['per_detector']['random50']['fill']['ci_hi_clustered']:.2f}] — no wiki-v1 "
        "inflation) and location near-duplication handled by the clustered bootstrap, the OA "
        "marginal-growth regime gives a CLEANER replication of the wiki finding plus the "
        "leak-free citation win: (1) **absolute LEVEL-SET components are the one predictor that "
        f"out-fills growth-matched controls — {p3['per_detector']['levelset']['fill']['ratio']:.2f}, "
        f"clustered CI [{p3['per_detector']['levelset']['fill']['ci_lo_clustered']:.2f}, "
        f"{p3['per_detector']['levelset']['fill']['ci_hi_clustered']:.2f}]** over "
        f"{p3['per_detector']['levelset']['fill']['n_groups']} groups — and unlike wiki they "
        "COINCIDE with the realized-change instrument (P@50 vs P2 = "
        f"{p3['per_detector']['levelset']['precision_at_k']['vs_p2_residual_tf01']:.2f} vs "
        f"chance {chh['vs_p2_residual_tf01']:.2f}): the Track-1-REFUTED discovery detector is, "
        "again, the one whose structures actually fill, and here the fill is real arrivals in "
        "the parsing-adjacent neighborhood (tagging/MT/segmentation), not pop-culture "
        "mass-production. (2) The Track-1 planted-gap champion **scoped fusion is an honest "
        f"negative ({p3['per_detector']['fused']['fill']['ratio']:.2f}, CI spans 1)**: the "
        "emptiest single in-scope points fill at the matched-control rate — extreme emptiness "
        "does not predict marginal arrival. (3) The **miner** dissolves under clustering "
        f"({p3['per_detector']['miner']['fill']['ratio']:.2f}, 2 groups) exactly as in wiki, "
        "but its fg-absent patterns are face-valid (the MT gap) — it needs diversity control to "
        "PREDICT, yet it DESCRIBES the gap correctly. (4) **The OA-only win lands as a "
        "DISSOCIATION, not redundancy:** the dated citation graph densifies "
        f"{cfg['p3a_n_edges_T']:,} → {cfg['p3a_n_edges_Tdelta']:,} edges over (T, T+Δ]; the "
        "citation-flow-change field (P3a) and its predictor (voltage_cite) localize the "
        f"CoNLL/treebank benchmark cluster (self-agree P@50 {p3['per_detector']['voltage_cite']['precision_at_k']['vs_p3a_peaks']:.2f}), "
        f"but those nodes UNDER-fill with new abstracted papers (voltage_cite "
        f"{p3['per_detector']['voltage_cite']['fill']['ratio']:.2f}, clustered CI "
        f"[{p3['per_detector']['voltage_cite']['fill']['ci_lo_clustered']:.2f}, "
        f"{p3['per_detector']['voltage_cite']['fill']['ci_hi_clustered']:.2f}], INVERTED) — "
        "where CITATIONS reroute is NOT where new papers land. Net: the suite's arrival-"
        "prediction signal is carried by absolute level-sets in BOTH corpora; the citation "
        "graph adds an orthogonal, face-valid change-localizer (benchmark consolidation) that "
        "the embedding-only wiki run could not see. The editorial-mass-production confound that "
        "made the wiki 1.28 suspect is GONE here (no citation analogue), and the level-set "
        "positive survives at a larger 1.92 with tighter CI in the marginal regime.",
        "",
    ]
    update_report("TASK_T2_OA", L)
    print("report section TASK_T2_OA updated")


if __name__ == "__main__":
    main()
