"""Track 3 — book gap-audit demo runner (descriptive; below the validated frontier).

Audits the ds2 textbook ("Foundations of Data Science", 126 section-level fg
units) against an EXTERNAL OpenAlex field slice. Single snapshot, descriptive:
NO change detectors (no second snapshot exists). Detector roles per the banked
verdicts:
  * level-set components over the scoped bg, ordered by r_F — the "what coherent
    field regions sit far from every book section" discovery map;
  * miner fg-absent term patterns — the human-readable labels;
  * r_F ranking within scope — the per-doc ordering. Composite skipped
    (its arbitration adds little at this scale; brief).

SCALE RESCALING (documented; the validated config was 30K docs / 1,446-doc fg):
  * k_fg = 3 (fg has 126 units; k=10 would be 8% of the whole book — validated
    k=10 was 0.7% of fg). Sensitivity at k in {1, 5}.
  * LSA-64 (small fg + distance concentration; brief's hint). Sensitivity: LSA-128.
  * miner min_support scales WITH the mining universe (the scoped bg, which is
    LARGER than the lab's): max(30, 0.4% of scoped) — the brief's "lower
    min_support" hint assumed a smaller bg, not this one; 0.4% is the validated
    support fraction (30 / 7,139 scoped in t2).
  * scope quantile q0.25 (validated form, unchanged — it is a bg-side rule).

Phases (subcommands):
  topics — search /topics for the book's field, probe meta.count per candidate
           topic with the full works filter; PRINT the count table (pre-commit
           happens in the worklog before `fetch`).
  fetch  — cursor-page the pre-committed slice to cache/oa/t3_ds_book/.
  audit  — assemble bg, joint-embed fg+bg, scope, r_F, level-set components,
           miner labels, sensitivity checks; bank results.json:task_t3 and
           write artifacts/gap_detection_eval/t3_book_audit.md.
  report — render the report.md section from banked results.

Run: cd ~/src/bookgap && uv run python -m \
       artifacts_scripts.gap_detection.run_t3_book_audit <phase>

Context hygiene: prints titles, top-terms, counts, numbers ONLY.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import time

import numpy as np

from artifacts_scripts.gap_detection.book_audit import (
    CACHE_DIR,
    OUT_DIR,
    assemble_bg,
    embed_joint,
    fetch_works,
    full_statistic,
    load_book_fg,
    region_terms,
    search_topics,
    works_count,
    works_filter,
)
from artifacts_scripts.gap_detection.detectors import (
    closed_filter,
    fightin_words_rank,
    knn_radii,
    level_set_components,
    mine_frequent_itemsets,
)
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import update_report, update_results

_T0 = time.monotonic()

SEED = 42
# Window slid 2015->2019 on COUNTS ONLY to land inside the brief's 50-150K
# (2015: 207K, 2018: 162K, 2019: 144,868); recorded in the worklog pre-commit.
PUB_FROM, PUB_TO = "2019-01-01", "2025-12-31"
K_FG = 3                      # rescaled fg-side kNN (validated 10 @ fg 1,446)
K_BG = 10                     # bg-side k of the statistic (validated, unchanged)
K_FG_SENS = (1, 5)
LSA_DIM, LSA_DIM_SENS = 64, 128
SCOPE_QUANTILE = 0.25         # validated bg-side rule, unchanged
TOP_FRAC = 0.05               # validated level-set selection (primary)
TOP_FRAC_DIAG = 0.01          # P2's change-scale lesson: finer selection diagnostic
K_GRAPH = 10                  # validated component graph k
MIN_COMP_SIZE = 3
SUPPORT_FRAC = 0.004          # validated miner support fraction (30/7,139)
N_TOP_PATTERNS = 40
N_REPORT_COMPONENTS = 12
N_EXEMPLAR_TITLES = 5
# URL/citation/LaTeX debris tokens in OA abstracts — never topical vocabulary.
JUNK_TERMS = frozenset(
    "http https www doi org com html xml url et al ii iii "
    "mathbb mathcal mathrm mathbf frac cdot infty leq geq sqrt varepsilon "
    "ldots langle rangle epsilon delta tilde widetilde".split()
)
# A term frequent in >= this df-share BOTH in and out of scope is REGISTER
# (uniform across field areas = genre vocabulary, not a topic).
REGISTER_DF_SHARE = 0.04

# Candidate /topics searches: the book's chapter themes + the field's names.
TOPIC_QUERIES = (
    "statistical methods", "machine learning", "bayesian inference",
    "statistical learning theory", "regression analysis",
    "dimensionality reduction", "high-dimensional data", "information theory",
    "model selection", "sparse representation", "compressed sensing",
    "gaussian processes", "causal inference", "statistical inference",
    "data science", "matrix factorization", "probabilistic models",
    "experimental design", "monte carlo methods", "optimization algorithms",
    "neural network theory", "deep learning", "kernel methods",
    "support vector machines", "ensemble learning", "random forests",
    "nonparametric statistics", "survival analysis", "multivariate analysis",
    "bootstrap resampling", "clustering", "reinforcement learning",
    "probability theory", "random matrix",
)

# PRE-COMMITTED topic set (counts recorded in the worklog Phase-B entry BEFORE
# any pull/scoring; combined meta.count 144,868 at 2019-2025; selection rule:
# methods-area display name in a stats/ML/DS-foundations methods subfield,
# domain applications excluded — see worklog + cache/t3_commit.json).
COMMITTED_TOPICS: list[str] = [
    "T10136",  # Statistical Methods and Inference            15,423
    "T12535",  # Machine Learning and Data Classification     12,199
    "T12205",  # Time Series Analysis and Forecasting         11,654
    "T10968",  # Statistical Distribution Estimation and Appl 10,317
    "T10845",  # Advanced Causal Inference Techniques         10,269
    "T11871",  # Advanced Statistical Methods and Models       8,569
    "T11303",  # Bayesian Modeling and Causal Inference        8,025
    "T12101",  # Advanced Bandit Algorithms Research           7,580
    "T12536",  # Topological and Geometric Data Analysis       7,632
    "T11901",  # Bayesian Methods and Mixture Models           7,531
    "T11612",  # Stochastic Gradient Optimization Techniques   7,000
    "T12072",  # Machine Learning and Algorithms               6,779
    "T12814",  # Gaussian Processes and Bayesian Inference     6,267
    "T10637",  # Advanced Clustering Algorithms Research       5,664
    "T10243",  # Statistical Methods and Bayesian Inference    5,191
    "T12056",  # Markov Chains and Monte Carlo Methods         4,828
    "T11716",  # Random Matrices and Applications              4,816
    "T11798",  # Optimal Experimental Design Methods           3,012
    "T13030",  # Survey Sampling and Estimation Techniques     2,112
]


TOPIC_NAMES = {
    "T10136": "Statistical Methods and Inference",
    "T12535": "Machine Learning and Data Classification",
    "T12205": "Time Series Analysis and Forecasting",
    "T10968": "Statistical Distribution Estimation and Applications",
    "T10845": "Advanced Causal Inference Techniques",
    "T11871": "Advanced Statistical Methods and Models",
    "T11303": "Bayesian Modeling and Causal Inference",
    "T12101": "Advanced Bandit Algorithms Research",
    "T12536": "Topological and Geometric Data Analysis",
    "T11901": "Bayesian Methods and Mixture Models",
    "T11612": "Stochastic Gradient Optimization Techniques",
    "T12072": "Machine Learning and Algorithms",
    "T12814": "Gaussian Processes and Bayesian Inference",
    "T10637": "Advanced Clustering Algorithms Research",
    "T10243": "Statistical Methods and Bayesian Inference",
    "T12056": "Markov Chains and Monte Carlo Methods",
    "T11716": "Random Matrices and Applications",
    "T11798": "Optimal Experimental Design Methods",
    "T13030": "Survey Sampling and Estimation Techniques",
}


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- phase: topics
def phase_topics(args) -> None:
    seen: dict[str, dict] = {}
    for q in TOPIC_QUERIES:
        for t in search_topics(q, log=log):
            if t["id"] and t["id"] not in seen:
                seen[t["id"]] = {**t, "query": q}
    cands = sorted(seen.values(), key=lambda t: -(t["works_count"] or 0))
    log(f"{len(cands)} distinct candidate topics from {len(TOPIC_QUERIES)} queries")
    rows = []
    for t in cands:
        if (t["works_count"] or 0) < args.min_total:
            continue
        n = works_count(works_filter([t["id"]], PUB_FROM, PUB_TO), log=log)
        rows.append({**t, "filtered_count": n})
        print(f"{t['id']:>7} | {n:>7,} filtered | {t['works_count']:>8,} total | "
              f"{t['field']} / {t['subfield']} | {t['display_name']} "
              f"[q: {t['query']}]")
    (CACHE_DIR / "t3_topic_counts.json").write_text(json.dumps(_py(rows), indent=1))
    log(f"count table -> {CACHE_DIR / 't3_topic_counts.json'} ({len(rows)} probed)")


# ----------------------------------------------------------------- phase: fetch
def phase_fetch(args) -> None:
    assert COMMITTED_TOPICS, "pre-commit the topic set first (worklog, then edit module)"
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    n = works_count(filt, log=log)
    log(f"pre-committed slice: {len(COMMITTED_TOPICS)} topics, meta.count={n:,}")
    fetch_works(filt, log=log, max_pages=args.max_pages)


# ----------------------------------------------------------------- phase: audit
def _component_record(comp, Z, scoped_global, bg, K_scoped, vocab, full_scores,
                      r_F_scoped, fg_units, Zf, ranked_patterns) -> dict:
    """Numbers/titles/terms-only record for one level-set component."""
    mem_local = comp["members"]            # indices into the scoped array
    mem_global = scoped_global[mem_local]  # indices into bg arrays
    cen = Z[mem_global].mean(axis=0)
    cen /= max(np.linalg.norm(cen), 1e-12)
    # exemplar titles: members nearest the component centroid
    d_cen = np.linalg.norm(Z[mem_global] - cen[None, :], axis=1)
    ex = mem_global[np.argsort(d_cen)[:N_EXEMPLAR_TITLES]]
    # nearest book unit to the centroid + closest member approach
    d_fg_cen = np.linalg.norm(Zf - cen[None, :], axis=1)
    j = int(np.argmin(d_fg_cen))
    r1_members = knn_radii(Z[mem_global], Zf, 1)
    jm = int(np.argmin(r1_members))
    d_fg_min = float(r1_members[jm])
    j_close = int(np.argmin(np.linalg.norm(Zf - Z[mem_global[jm]][None, :], axis=1)))
    # vocabulary-gap miner patterns concentrated in this component
    mem_set = set(int(m) for m in mem_local)
    pats = []
    for p in ranked_patterns:
        rows = p["_supporter_rows"]
        if rows.size == 0:
            continue
        inside = sum(1 for r in rows if int(r) in mem_set)
        if inside >= 3:
            pats.append({"terms": p["terms"], "z": round(p["z"], 2),
                         "bg_support": p["bg_support"], "in_component": inside})
        if len(pats) >= 5:
            break
    return {
        "size": int(comp["size"]),
        "n_eff": round(float(comp["n_eff"]), 1),
        "mean_full": round(float(comp["mean_score"]), 4),
        "mean_r_F": round(float(np.mean(r_F_scoped[mem_local])), 4),
        "full_pctile_mean": round(
            float(np.mean([(full_scores < full_scores[m]).mean()
                           for m in mem_local])) * 100, 1),
        "terms": [t for t in region_terms(K_scoped, mem_local, vocab, n_top=14)
                  if t not in JUNK_TERMS][:10],
        "exemplar_titles": [str(bg["title"][i]) for i in ex],
        "exemplar_years": [int(bg["year"][i]) for i in ex],
        "nearest_unit": fg_units[j].label,
        "nearest_unit_dist": round(float(d_fg_cen[j]), 3),
        "closest_member_unit": fg_units[j_close].label,
        "closest_member_dist": round(d_fg_min, 3),
        "patterns": pats,
        "_members_global": mem_global,
    }


def phase_audit(args) -> None:
    assert COMMITTED_TOPICS, "pre-commit the topic set first"
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)

    # ---- fg
    fg_units = load_book_fg()
    fg_texts = [u.text for u in fg_units]
    lens = np.array([len(t) for t in fg_texts])
    log(f"fg: {len(fg_units)} units, chars med {int(np.median(lens))}")

    # ---- bg
    bg = assemble_bg(filt, log=log)
    yr = bg["year"]
    log(f"bg years: min {yr.min()} med {int(np.median(yr))} max {yr.max()}")

    # ---- joint embed (then the nnz>=5 trim on bg only; fg rows always kept)
    emb = embed_joint(fg_texts, bg["texts"], seed=SEED, lsa_dim=LSA_DIM)
    nnz = np.asarray((emb["X"] > 0).sum(axis=1)).ravel()
    n_fg = len(fg_texts)
    keep_bg = np.where(nnz[n_fg:] >= 5)[0]
    log(f"embed: tfidf vocab {len(emb['tfidf_vocab']):,}, LSA-{LSA_DIM} "
        f"EVR {emb['evr']:.3f}; bg nnz>=5 keeps {keep_bg.size:,}/{len(bg['texts']):,}; "
        f"fg nnz min/med {int(nnz[:n_fg].min())}/{int(np.median(nnz[:n_fg]))}")
    Zf = emb["Z"][:n_fg]
    Zb = emb["Z"][n_fg:][keep_bg]
    for k in ("wid", "title", "year", "topic", "cited_by"):
        bg[k] = bg[k][keep_bg]
    Kb = emb["K"][n_fg:][keep_bg]
    Kf = np.asarray(emb["K"][:n_fg].todense(), dtype=bool)
    vocab = emb["vocab"]

    # ---- scope by distance-to-fg (validated q0.25 rule), r_F ordering
    r_F = knn_radii(Zb, Zf, K_FG)
    thr = float(np.quantile(r_F, SCOPE_QUANTILE))
    scoped_global = np.where(r_F <= thr)[0]
    r_F_scoped = r_F[scoped_global]
    log(f"scope: r_F(k={K_FG}) q25 thr {thr:.4f} -> {scoped_global.size:,} of "
        f"{Zb.shape[0]:,} bg in scope; r_F quartiles all-bg "
        f"{np.quantile(r_F, [0.25, 0.5, 0.75]).round(3).tolist()}")

    # ---- the full statistic over the scoped pool (validated descriptive
    # substrate; identity-tested vs GapScorer, k_fg decoupled as documented).
    # ON RECORD (first pass, banked then replaced): components over RAW r_F at
    # top-5% give ONE connected blob (size 1559, n_eff 1559, incoherent terms) —
    # the scope-boundary SHELL, not a map; pure-distance level sets are an
    # annulus by construction. The validated descriptive form (t2 wiki + OA)
    # is components over the FULL statistic; peaks fragment into regions.
    Zs = Zb[scoped_global]
    stat = full_statistic(Zf, Zs, k_fg=K_FG, k_bg=K_BG)
    full = stat["full"]
    ones = np.ones(scoped_global.size, dtype=bool)
    comps05 = level_set_components(full, Zs, ones, top_frac=TOP_FRAC,
                                   k_graph=K_GRAPH, min_size=MIN_COMP_SIZE)
    comps01 = level_set_components(full, Zs, ones, top_frac=TOP_FRAC_DIAG,
                                   k_graph=K_GRAPH, min_size=MIN_COMP_SIZE)

    def _n_resolved(cs):  # components of size >= 5 (a usable map has several)
        return sum(1 for c in cs if c["size"] >= 5)

    map_frac, comps = (TOP_FRAC, comps05)
    if _n_resolved(comps01) > _n_resolved(comps05):
        map_frac, comps = (TOP_FRAC_DIAG, comps01)
    log(f"level-set(full): top5% -> {len(comps05)} comps ({_n_resolved(comps05)} "
        f"of size>=5), top1% -> {len(comps01)} ({_n_resolved(comps01)}); "
        f"map at top_frac={map_frac}")

    # ---- miner: fg-absent term patterns over the scoped pool, SPLIT into
    # topical vocabulary-gaps vs REGISTER. The book is a different GENRE from
    # paper abstracts: it lacks the whole research-paper register (propose,
    # simulation, demonstrate, novel...), so bare fg-absence is confounded
    # (v2 finding, on record: the top "vocab-gap" patterns were {simulation,
    # study}-style register pairs). Data-driven register test, no hand list:
    # a term is REGISTER iff it is frequent BOTH inside the scope and in the
    # off-scope rest of the field (df-share >= REGISTER_DF_SHARE in both) —
    # uniform across field areas = genre vocabulary, not a topic.
    K_scoped = np.asarray(Kb[scoped_global].todense(), dtype=bool)
    fg_term_df = np.asarray(Kf.sum(axis=0)).ravel()  # per keyword: # fg units
    off_scope = np.setdiff1d(np.arange(Zb.shape[0]), scoped_global)
    df_in = K_scoped.sum(axis=0) / max(scoped_global.size, 1)
    df_out = (np.asarray(Kb[off_scope].sum(axis=0)).ravel()
              / max(off_scope.size, 1))
    register_term = (df_in >= REGISTER_DF_SHARE) & (df_out >= REGISTER_DF_SHARE)
    topical_gap_term = ((fg_term_df == 0) & ~register_term
                        & np.array([t not in JUNK_TERMS for t in vocab]))
    log(f"term split: {int((fg_term_df == 0).sum())} fg-absent keywords -> "
        f"{int((register_term & (fg_term_df == 0)).sum())} register, "
        f"{int(topical_gap_term.sum())} topical-gap "
        f"(register df-share >= {REGISTER_DF_SHARE} in AND out of scope)")
    min_support = max(30, int(round(SUPPORT_FRAC * scoped_global.size)))
    fam = closed_filter(mine_frequent_itemsets(
        K_scoped, min_support=min_support, max_len=3, max_pairs_expand=40_000))
    ranked = fightin_words_rank(fam, Kf, n_bg=K_scoped.shape[0])
    fg_absent = [r for r in ranked if r["fg_support"] == 0]

    def _is_vocab_gap(itemset) -> bool:
        return any(topical_gap_term[c] for c in itemset)

    vocab_gap_raw = [r for r in fg_absent if _is_vocab_gap(r["itemset"])]
    register_raw = [r for r in fg_absent if not _is_vocab_gap(r["itemset"])]
    log(f"miner: min_support {min_support}, {len(fam)} closed itemsets, "
        f"{len(fg_absent)} fg-absent ({len(vocab_gap_raw)} topical vocab-gap, "
        f"{len(register_raw)} register/covered-term combos)")

    def _materialize(rows_ranked, n):
        out = []
        for r in rows_ranked[:n]:
            cols = sorted(r["itemset"])
            v = K_scoped[:, cols[0]].copy()
            for c in cols[1:]:
                v &= K_scoped[:, c]
            out.append({
                "terms": [str(vocab[c]) for c in cols], "z": float(r["z"]),
                "bg_support": int(r["bg_support"]),
                "fg_support": int(r["fg_support"]),
                "_supporter_rows": np.where(v)[0],
            })
        return out

    patterns = _materialize(vocab_gap_raw, N_TOP_PATTERNS)      # the labels
    register_patterns = _materialize(register_raw, 10)          # on record
    # topical fg-absent single TERMS (the book never says them), by scoped df
    bg_df = K_scoped.sum(axis=0)
    absent_cols = [int(c) for c in np.argsort(bg_df)[::-1] if topical_gap_term[c]]
    fg_absent_terms = [{"term": str(vocab[c]), "bg_df": int(bg_df[c])}
                       for c in absent_cols[:30]]
    register_sample = [str(vocab[c]) for c in
                       np.argsort(bg_df)[::-1]
                       if register_term[c] and fg_term_df[c] == 0][:12]

    # ---- per-doc orderings within scope: r_F (the brief's form; top-50 is the
    # boundary shell — ties at the scope threshold, on record) and the full
    # statistic (earned emptiness; the informative ordering at this scale).
    order_rf = np.argsort(r_F_scoped)[::-1]
    top50_rf_span = float(r_F_scoped[order_rf[0]] - r_F_scoped[order_rf[49]])
    scope_rf_iqr = float(np.subtract(*np.quantile(r_F_scoped, [0.75, 0.25])))
    order_full = np.argsort(full)[::-1]

    def _doc_row(i_local, score_name, score_val):
        g = scoped_global[i_local]
        d_units = np.linalg.norm(Zf - Zb[g][None, :], axis=1)
        j = int(np.argmin(d_units))
        return {
            "title": str(bg["title"][g]), "year": int(bg["year"][g]),
            score_name: round(float(score_val), 3),
            "d_nearest_unit": round(float(d_units[j]), 3),
            "nearest_unit": fg_units[j].label,
        }

    top_docs_rf = [_doc_row(i, "r_F", r_F_scoped[i]) for i in order_rf[:10]]
    top_docs_full = [_doc_row(i, "full", full[i]) for i in order_full[:15]]

    # ---- component records (ranked by n_eff of full-score mass)
    comp_recs = [
        _component_record(c, Zb, scoped_global, bg, K_scoped, vocab, full,
                          stat["r_F"], fg_units, Zf, patterns)
        for c in comps[:N_REPORT_COMPONENTS]
    ]

    # ---- sensitivity over the FULL ordering: k_fg and LSA dim. (First pass on
    # record: raw-r_F top-50 Jaccard was 0.000 in every config — boundary-shell
    # ties churn; the statistic's peaks are the stable object to compare.)
    from scipy.stats import spearmanr

    sens = {"k": {}, "lsa": {}}
    top50 = set(scoped_global[order_full[:50]].tolist())
    for k in K_FG_SENS:
        r_k = knn_radii(Zb, Zf, k)
        thr_k = float(np.quantile(r_k, SCOPE_QUANTILE))
        sg_k = np.where(r_k <= thr_k)[0]
        stat_k = full_statistic(Zf, Zb[sg_k], k_fg=k, k_bg=K_BG)
        t50 = set(sg_k[np.argsort(stat_k["full"])[::-1][:50]].tolist())
        shared = np.intersect1d(sg_k, scoped_global)
        pos_a = {g: i for i, g in enumerate(scoped_global)}
        pos_b = {g: i for i, g in enumerate(sg_k)}
        rho = float(spearmanr(
            [full[pos_a[g]] for g in shared],
            [stat_k["full"][pos_b[g]] for g in shared]).statistic)
        sens["k"][str(k)] = {
            "scope_jaccard": round(len(set(sg_k) & set(scoped_global))
                                   / len(set(sg_k) | set(scoped_global)), 3),
            "spearman_full_on_shared_scope": round(rho, 3),
            "top50_jaccard": round(len(top50 & t50) / max(len(top50 | t50), 1), 3),
        }
    emb128 = embed_joint(fg_texts, bg["texts"], seed=SEED, lsa_dim=LSA_DIM_SENS)
    # identical text list as the main embed (pre-trim list), so rows align
    Zf128 = emb128["Z"][:n_fg]
    Zb128 = emb128["Z"][n_fg:][keep_bg]
    r128 = knn_radii(Zb128, Zf128, K_FG)
    thr128 = float(np.quantile(r128, SCOPE_QUANTILE))
    sg128 = np.where(r128 <= thr128)[0]
    stat128 = full_statistic(Zf128, Zb128[sg128], k_fg=K_FG, k_bg=K_BG)
    t50_128 = set(sg128[np.argsort(stat128["full"])[::-1][:50]].tolist())
    shared = np.intersect1d(sg128, scoped_global)
    pos_a = {g: i for i, g in enumerate(scoped_global)}
    pos_b = {g: i for i, g in enumerate(sg128)}
    sens["lsa"][str(LSA_DIM_SENS)] = {
        "evr": round(emb128["evr"], 3),
        "scope_jaccard": round(len(set(sg128) & set(scoped_global))
                               / len(set(sg128) | set(scoped_global)), 3),
        "spearman_full_on_shared_scope": round(float(spearmanr(
            [full[pos_a[g]] for g in shared],
            [stat128["full"][pos_b[g]] for g in shared]).statistic), 3),
        "top50_jaccard": round(len(top50 & t50_128) / max(len(top50 | t50_128), 1), 3),
    }
    log(f"sensitivity: {json.dumps(sens)}")

    # ---- chapter coverage profile (which chapters the scoped field sits near)
    j_near = np.empty(scoped_global.size, dtype=np.int64)
    for s in range(0, scoped_global.size, 2048):
        blk = Zb[scoped_global[s:s + 2048]]
        d = np.linalg.norm(blk[:, None, :] - Zf[None, :, :], axis=2)
        j_near[s:s + 2048] = np.argmin(d, axis=1)
    ch_of = np.asarray([u.chapter for u in fg_units], dtype=object)
    ch_counts = {str(c): int(n) for c, n in
                 zip(*np.unique(ch_of[j_near], return_counts=True))}

    # ---- per-topic scope composition (which field areas the book's reach keeps)
    in_scope = np.zeros(Zb.shape[0], dtype=bool)
    in_scope[scoped_global] = True
    topic_rows = []
    for tid in COMMITTED_TOPICS:
        m = bg["topic"] == tid
        n_t = int(m.sum())
        topic_rows.append({
            "topic": tid, "n_used": n_t,
            "n_in_scope": int((m & in_scope).sum()),
            "scope_share": round(float((m & in_scope).sum() / max(n_t, 1)), 3),
            "median_r_F": round(float(np.median(r_F[m])), 3) if n_t else None,
        })
    topic_rows.sort(key=lambda r: -r["scope_share"])

    payload = {
        "validity": "descriptive demo — below the validated frontier",
        "seed": SEED,
        "fg": {
            "source": "~/src/learn-and-teach/outputs/2026-06-09/23-15-05/build/sections/ch*.tex",
            "n_units": len(fg_units), "n_chapters": 9,
            "chars_min_med_max": [int(lens.min()), int(np.median(lens)), int(lens.max())],
            "unit_rule": "split at \\section AND \\subsection + chapter intros; "
                         "<200-char units merged into previous; solutions.tex excluded",
        },
        "bg": {
            "filter": filt, "topics": COMMITTED_TOPICS,
            "pub_range": [PUB_FROM, PUB_TO], "counts": bg["counts"],
            "n_used": int(Zb.shape[0]),
            "year_min_med_max": [int(yr.min()), int(np.median(yr)), int(yr.max())],
        },
        "embed": {"lsa_dim": LSA_DIM, "evr": round(emb["evr"], 3),
                  "tfidf_vocab": int(len(emb["tfidf_vocab"])),
                  "n_keywords": int(vocab.size)},
        "scope": {"k_fg": K_FG, "k_bg": K_BG, "quantile": SCOPE_QUANTILE,
                  "thr": round(thr, 4), "n_scoped": int(scoped_global.size),
                  "r_F_quartiles_all_bg": np.quantile(r_F, [0.25, 0.5, 0.75]).round(4).tolist()},
        "levelset": {
            "score": "full = hab * (Delta - 0.5*sigma), k_fg=3/k_bg=10 "
                     "(identity-tested vs GapScorer at k_fg==k_bg)",
            "map_top_frac": map_frac, "k_graph": K_GRAPH,
            "n_components_top5pct": len(comps05),
            "n_components_top1pct": len(comps01),
            "rf_only_shell_note": "raw-r_F top-5% gives 1 connected component "
                                  "(size 1559, n_eff 1559, incoherent terms) — "
                                  "the scope-boundary shell; banked first pass, "
                                  "replaced by the validated full-statistic map",
            "components": [{k: v for k, v in c.items() if not k.startswith("_")}
                           for c in comp_recs]},
        "miner": {"min_support": min_support, "n_closed_itemsets": len(fam),
                  "n_fg_absent": len(fg_absent),
                  "n_vocab_gap": len(vocab_gap_raw),
                  "n_register": len(register_raw),
                  "register_df_share": REGISTER_DF_SHARE,
                  "register_term_sample": register_sample,
                  "top_vocab_gap": [{k: v for k, v in p.items()
                                     if not k.startswith("_")}
                                    for p in patterns],
                  "top_register": [{k: v for k, v in p.items()
                                    if not k.startswith("_")}
                                   for p in register_patterns],
                  "fg_absent_terms": fg_absent_terms},
        "per_doc": {"full_top": top_docs_full, "r_f_top": top_docs_rf,
                    "top50_rf_span": round(top50_rf_span, 5),
                    "scope_rf_iqr": round(scope_rf_iqr, 5)},
        "chapter_scope_profile": ch_counts,
        "topic_scope_composition": topic_rows,
        "sensitivity": sens,
        "runtime_s": round(time.monotonic() - _T0, 1),
    }
    update_results("task_t3", payload)
    log("banked results.json:task_t3")
    _write_audit_md(payload)
    log(f"wrote {OUT_DIR / 't3_book_audit.md'}")


# ----------------------------------------------------------------- audit markdown
# One-line owner framings per component rank (filled AFTER first inspection of
# the FINAL v4 labels/titles; deterministic re-runs render identical text).
FRAMINGS: dict[int, str] = {
    1: "The new-lifetime-distribution literature (exponentiated/Lindley/Lomax "
       "families, hazard shapes). The book covers the exponential family; named "
       "lifetime families, hazard rates, and reliability modeling are absent — "
       "a short 'beyond the exponential family' pointer is defensible; the "
       "distribution-mill literature itself is out of scope.",
    2: "Biased estimation under multicollinearity (ridge-type/Liu/Kibria-Lukman "
       "estimators for GLMs). The book has ridge as regularization (ch8) but not "
       "the estimation-theoretic multicollinearity tradition; a one-paragraph "
       "bridge (VIF, why OLS breaks) would connect ch3 to ch8 — the applied "
       "estimator-variant literature is out of scope.",
    3: "Survey sampling / finite-population estimation (ratio estimators, "
       "auxiliary variables, stratification). A real statistics subfield wholly "
       "absent — the book's estimation is model-based iid. Design-based "
       "inference deserves at least a remark in a foundations text; owner "
       "judges depth.",
    4: "Learning with noisy labels (robust training, annotation noise). Modern "
       "supervised-learning robustness; connects to the book's risk framework "
       "but is methods-frontier — plausibly out of scope for foundations.",
    5: "Censored-data Bayesian inference and reliability (progressive/Type-II "
       "censoring, credible intervals). Survival/censoring is classic "
       "statistics the book never mentions — a candidate gap for ch5/ch9; the "
       "specific reliability-journal literature is out of scope.",
    6: "Random matrix theory (spectral laws, eigenvalue distributions). The "
       "map's closest big region to the book (d=0.62 to ch1 spectral theory): "
       "Marchenko-Pastur / high-dimensional spectra would directly strengthen "
       "the PCA/SVD chapters. Strongest in-scope candidate.",
    7: "Bandits and regret bounds (sequential decision-making). A major modern "
       "field area adjacent to ch7's generalization bounds; the book stops at "
       "static iid learning — a 'beyond iid: sequential decisions' pointer is "
       "a reasonable ask; full treatment out of scope.",
    8: "Algebraic topology proper (homology, cohomology, complexes) — the pure "
       "wing; its applied face (persistent homology / TDA) is the nearer ask "
       "for a 'geometry of data' chapter that is currently linear-only. Owner "
       "judges whether a TDA pointer belongs.",
    9: "Applied interval estimation for skewed/zero-inflated families "
       "(coverage studies, Birnbaum-Saunders). The book covers credible vs "
       "confidence intervals conceptually (nearest unit d=0.58, the map's "
       "closest attachment) — this applied literature is out of scope.",
    10: "Bayes estimators of shape parameters under assorted priors/loss "
        "functions (LINEX-style journal literature). The applied wing of "
        "ch4-ch5 content; the loss-function variety beyond L1/L2/cross-entropy "
        "might merit a remark — the literature itself out of scope.",
    11: "Coded / privacy-preserving distributed matrix multiplication. Sits "
        "near ch1's SVD only through shared linear-algebra vocabulary; "
        "information-theoretic distributed computing — out of scope.",
    12: "Feature selection (filter/wrapper methods, relevance). The book "
        "reaches sparsity via lasso/elastic-net (ch8) but never treats feature "
        "selection as a task; a paragraph distinguishing selection from "
        "shrinkage is a defensible ask.",
}


def _write_audit_md(p: dict) -> None:
    L = [
        "# Track 3 — book gap audit: ds2 \"Foundations of Data Science\" vs its field",
        "",
        "**DESCRIPTIVE DEMO — BELOW THE VALIDATED FRONTIER.** The validated suite was",
        "tuned at 30K docs / 1,446-doc fg; here fg = "
        f"{p['fg']['n_units']} book sections, so parameters are rescaled "
        f"(k_fg={p['scope']['k_fg']}, LSA-{p['embed']['lsa_dim']}, miner support "
        f"{p['miner']['min_support']}) and this is a missing-TOPICS report, not a",
        "validated-recovery claim. Single snapshot; no change detectors.",
        "",
        f"- **fg**: {p['fg']['n_units']} section-level units from 9 chapters "
        f"({p['fg']['source']}; solutions.tex excluded).",
        f"- **bg**: OpenAlex slice, {len(p['bg']['topics'])} pre-committed topics "
        f"({', '.join(p['bg']['topics'])}), {p['bg']['pub_range'][0][:4]}–"
        f"{p['bg']['pub_range'][1][:4]}, English w/ abstracts: "
        f"{p['bg']['n_used']:,} works used.",
        f"- **Geometry**: joint TF-IDF → LSA-{p['embed']['lsa_dim']} "
        f"(EVR {p['embed']['evr']}). Scope = bg within the q{int(p['scope']['quantile']*100)} "
        f"distance-to-fg quantile (thr {p['scope']['thr']}): {p['scope']['n_scoped']:,} works.",
        "- **Reading**: each region is a coherent cluster of in-field works far from",
        "  every book section. The owner judges each: should the book cover this?",
        "",
        "## Ranked candidate gap regions (level-set components, by effective size)",
        "",
    ]
    for n, c in enumerate(p["levelset"]["components"], 1):
        pat = "; ".join(
            "{" + ", ".join(q["terms"]) + "}" for q in c["patterns"][:3]) or "—"
        titles = "\n".join(
            f"   - {t} ({y})" for t, y in zip(c["exemplar_titles"], c["exemplar_years"]))
        L += [
            f"### {n}. {{{', '.join(c['terms'][:6])}}}",
            f"- size {c['size']} (n_eff {c['n_eff']}), mean full-score "
            f"{c['mean_full']} (mean percentile {c['full_pctile_mean']}), "
            f"mean r_F {c['mean_r_F']}",
            f"- top terms: {', '.join(c['terms'])}",
            f"- vocabulary-gap patterns inside: {pat}",
            "- exemplar works (nearest the component centroid):",
            titles,
            f"- nearest book unit (centroid): {c['nearest_unit']} "
            f"(d={c['nearest_unit_dist']})",
            f"- closest member approach: {c['closest_member_unit']} "
            f"(d={c['closest_member_dist']})",
            f"- **Should the book cover this?** {FRAMINGS.get(n, '(owner judges)')}",
            "",
        ]
    L += [
        "## Miner labels: field vocabulary the book never uses",
        "",
        f"Of {p['miner']['n_fg_absent']:,} fg-absent frequent patterns in the scoped "
        f"pool, {p['miner']['n_vocab_gap']:,} contain a TOPICAL term the book never "
        f"utters (vocabulary gaps, below); the other {p['miner']['n_register']:,} "
        "are register/covered-term combos. The book is a different genre from "
        "paper abstracts — it lacks the whole research-paper register "
        f"(e.g. {', '.join(p['miner']['register_term_sample'][:8])}), so terms "
        f"frequent in >={p['miner']['register_df_share']:.0%} of docs both in and "
        "out of scope are classified register, not topic gaps (rule on record).",
        "",
        "Top vocabulary-gap patterns (bg-frequent, fg-absent):",
        "",
    ]
    for q in p["miner"]["top_vocab_gap"][:15]:
        L.append(f"- {{{', '.join(q['terms'])}}} — bg support {q['bg_support']}, "
                 f"fg support 0, z={q['z']:.1f}")
    L += ["", "Top topical terms the book never uses (by scoped-pool frequency):", "",
          "- " + ", ".join(f"{t['term']} ({t['bg_df']})"
                           for t in p["miner"]["fg_absent_terms"][:25]),
          "",
          "Reading the labels: the real signals here are **approximate Bayesian "
          "inference** ({approximate, posterior/inference/bayesian} — the book's "
          "Bayes chapter is conjugate-exact only), **mixture models** "
          "({distribution, mixture} — never treated), **covariate modeling** "
          "({covariates, regression}), and the term-level absences "
          "**nonparametric**, **mixture**, **adaptive**, **guarantees**. The "
          "rest ({commonly, used}, {real, simulated}, {case, special}) is "
          "phrase-level register that survives the df-share rule — read past it."]
    L += ["", "## Farthest individual in-scope works", "",
          "By the full statistic (earned emptiness):", ""]
    for d in p["per_doc"]["full_top"]:
        L.append(f"- {d['title']} ({d['year']}) — full {d['full']}, nearest unit: "
                 f"{d['nearest_unit']} (d={d['d_nearest_unit']})")
    L += [
        "",
        f"By raw r_F within scope (the boundary shell — top-50 span "
        f"{p['per_doc']['top50_rf_span']} vs scope IQR {p['per_doc']['scope_rf_iqr']}; "
        "shown for the record):",
        "",
    ]
    for d in p["per_doc"]["r_f_top"][:5]:
        L.append(f"- {d['title']} ({d['year']}) — r_F {d['r_F']}, nearest unit: "
                 f"{d['nearest_unit']} (d={d['d_nearest_unit']})")
    L += [
        "",
        "## Field-area composition of the book's topical reach",
        "",
        "Share of each pre-committed OpenAlex topic that falls inside the scope",
        "(the q25 distance-to-fg quantile). LOW share = the field area sits mostly",
        "beyond the book's territory; the components above show its in-reach edge.",
        "",
        "| topic | works used | in scope | share | median r_F |",
        "|---|---|---|---|---|",
    ]
    for t in p["topic_scope_composition"]:
        L.append(f"| {TOPIC_NAMES.get(t['topic'], t['topic'])} ({t['topic']}) | "
                 f"{t['n_used']:,} | {t['n_in_scope']:,} | {t['scope_share']} | "
                 f"{t['median_r_F']} |")
    L += [
        "",
        "**Topic-grain headline**: the field's **causal inference** (0.051), "
        "**clustering** (0.046), and **time series** (0.045) literatures sit "
        "almost entirely beyond the book's topical reach, with **experimental "
        "design** (0.116) and **bandits** (0.124) close behind — these are the "
        "curriculum-level absences. (At the other end, random matrices 0.715: "
        "the book's spectral chapters sit inside that literature's territory.) "
        "The owner question per area: does a 'Foundations of Data Science' text "
        "need a chapter — or at least a pointer — for it?",
    ]
    sens = p["sensitivity"]
    L += [
        "",
        "## Robustness (descriptive)",
        "",
        f"- k_fg sensitivity (vs k={p['scope']['k_fg']}, full-statistic ordering): "
        + "; ".join(
            f"k={k}: scope J {v['scope_jaccard']}, "
            f"Spearman {v['spearman_full_on_shared_scope']}, "
            f"top50 J {v['top50_jaccard']}" for k, v in sens["k"].items()),
        "- LSA-128 (vs 64): " + "; ".join(
            f"scope J {v['scope_jaccard']}, "
            f"Spearman {v['spearman_full_on_shared_scope']}, "
            f"top50 J {v['top50_jaccard']}" for v in sens["lsa"].values()),
        f"- level-set map: top-5% -> {p['levelset']['n_components_top5pct']} components, "
        f"top-1% -> {p['levelset']['n_components_top1pct']}; map shown at "
        f"top_frac={p['levelset']['map_top_frac']}. First-pass raw-r_F map on record: "
        "one boundary-shell component (see results.json rf_only_shell_note).",
        "- chapter scope profile (nearest chapter of scoped works): "
        + json.dumps(p["chapter_scope_profile"]),
        "",
        f"*Banked: results.json:task_t3. Seed {p['seed']}. Runtime {p['runtime_s']}s.*",
    ]
    (OUT_DIR / "t3_book_audit.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------- phase: frame
def phase_frame(args) -> None:
    """Re-render t3_book_audit.md from banked results (no recompute) — used to
    fill the per-region owner framings after first inspection of the labels."""
    from artifacts_scripts.gap_detection.run_followups import load_banked

    _write_audit_md(load_banked()["task_t3"])
    log(f"re-rendered {OUT_DIR / 't3_book_audit.md'} "
        f"({len(FRAMINGS)} framings filled)")


# ----------------------------------------------------------------- phase: report
def phase_report(args) -> None:
    from artifacts_scripts.gap_detection.run_followups import load_banked

    p = load_banked()["task_t3"]
    c0 = p["levelset"]["components"]
    L = [
        "## Track 3 — book gap-audit demo (descriptive; below the validated frontier) "
        "(2026-06-12)",
        "",
        f"First non-wiki/OA fg: the ds2 textbook \"Foundations of Data Science\" "
        f"({p['fg']['n_units']} section-level units, 9 chapters) audited against an "
        f"EXTERNAL OpenAlex slice — {len(p['bg']['topics'])} pre-committed topics "
        f"({', '.join(p['bg']['topics'])}), {p['bg']['pub_range'][0][:4]}–"
        f"{p['bg']['pub_range'][1][:4]}, {p['bg']['n_used']:,} English abstracted works "
        f"(the build is anchorless — no reference corpus exists). Joint TF-IDF → "
        f"LSA-{p['embed']['lsa_dim']} (EVR {p['embed']['evr']}); scope = "
        f"q{int(p['scope']['quantile']*100)} distance-to-fg "
        f"({p['scope']['n_scoped']:,} works); detectors per the banked verdicts "
        f"(scoped level-set discovery map + miner fg-absent labels + r_F ordering; "
        f"no change modes). Rescaled for the 10^2-unit fg: k_fg={p['scope']['k_fg']} "
        f"(bg-side k=10 unchanged), LSA-{p['embed']['lsa_dim']}, miner min_support "
        f"{p['miner']['min_support']} (0.4% of scope). Map = components over the "
        f"FULL statistic (identity-tested vs GapScorer; raw-r_F top-5% = one "
        f"boundary-shell blob, on record): top-5% -> "
        f"{p['levelset']['n_components_top5pct']} components, top-1% -> "
        f"{p['levelset']['n_components_top1pct']}; shown at "
        f"top_frac={p['levelset']['map_top_frac']}. Miner fg-absent patterns split "
        f"{p['miner']['n_vocab_gap']:,} vocabulary-gap / {p['miner']['n_register']:,} "
        f"paper-register (the t2-known genre stratum, excluded from labels). Full "
        f"ranked table + owner framings in `t3_book_audit.md`.",
        "",
        "| # | label (top terms) | size (n_eff) | nearest book unit | d |",
        "|---|---|---|---|---|",
    ]
    for n, c in enumerate(c0, 1):
        L.append(f"| {n} | {', '.join(c['terms'][:5])} | {c['size']} ({c['n_eff']}) | "
                 f"{c['nearest_unit']} | {c['nearest_unit_dist']} |")
    sens_k = p["sensitivity"]["k"]
    sens_l = next(iter(p["sensitivity"]["lsa"].values()))
    comp_rows = p["topic_scope_composition"]
    low = [f"{TOPIC_NAMES.get(t['topic'], t['topic'])} {t['scope_share']}"
           for t in comp_rows[-5:]]
    L += [
        "",
        "Miner top vocabulary-gap patterns: "
        + "; ".join("{" + ", ".join(q["terms"]) + "}"
                    for q in p["miner"]["top_vocab_gap"][:6]) + ". "
        "Top fg-absent terms: "
        + ", ".join(t["term"] for t in p["miner"]["fg_absent_terms"][:10]) + ".",
        "",
        "Topic-grain coverage (share of each pre-committed topic inside the q25 "
        "scope): lowest five — " + "; ".join(low) + " — whole field areas beyond "
        "the book's topical reach at the validated scope rule.",
        "",
        "Robustness: k_fg-sensitivity scope Jaccard "
        + "/".join(str(v["scope_jaccard"]) for v in sens_k.values())
        + f" (k={','.join(sens_k.keys())}), full-ordering top50 Jaccard "
        + "/".join(str(v["top50_jaccard"]) for v in sens_k.values())
        + f"; LSA-128 scope Jaccard {sens_l['scope_jaccard']}, top50 Jaccard "
        f"{sens_l['top50_jaccard']}. "
        f"Rows `results.json:task_t3`; artifact `t3_book_audit.md`.",
    ]
    update_report("TASK_T3_BOOK_AUDIT", L)
    log("report section TASK_T3_BOOK_AUDIT updated")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    t = sub.add_parser("topics")
    t.add_argument("--min-total", type=int, default=3000,
                   help="skip topics with fewer total works than this")
    t.set_defaults(fn=phase_topics)
    f = sub.add_parser("fetch")
    f.add_argument("--max-pages", type=int, default=None)
    f.set_defaults(fn=phase_fetch)
    a = sub.add_parser("audit")
    a.set_defaults(fn=phase_audit)
    fr = sub.add_parser("frame")
    fr.set_defaults(fn=phase_frame)
    r = sub.add_parser("report")
    r.set_defaults(fn=phase_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
