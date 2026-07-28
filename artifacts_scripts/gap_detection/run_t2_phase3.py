"""Track 2 Phase 3: predict at T, score against (T, T+Delta] arrivals.

PRE-COMMITTED design (before any outcome computation; see worklog Phase 0-2):
  K = 50 locations per detector; M = 20 matched controls per location;
  neighborhood radius = the location's 10-NN scale against the scoped-bg
  candidate set at T (self-excluded for locations that are candidates);
  matching covariates = log r_B,10 (local T density), log r_F,10 (distance to
  fg), log1p growth count in (T-2y, T] within the same neighborhood rule;
  scope membership by construction (controls drawn from the scoped universe);
  bootstrap B = 2000 over locations.

Predictors (T-only inputs; per banked Track-1 configs):
  fused    — rank scoped bg by r_F,10, Task-A fixed veto (hull q=0.90 OR
             Delta > log 3); top-50 candidates.
  miner    — scoped emerging patterns (top-2000-df, closed, min_support 30,
             max_len 3, min_len 2, Fightin'-Words vs T-fg); top-50 itemsets;
             location = L2-normalized supporter centroid in T-LSA.
  voltage  — single-pair corridor solves (extended electrodes, 4phi(1-phi)
             mid-weighting) on every qualifying T cluster pair's local bg
             subgraph; per-node max flow share across pairs; top-50 nodes.
  levelset — absolute level-set components (full statistic, top-frac 0.05,
             n_eff order, discovery mode); members flattened in (component,
             score) order; first 50.

Outcome instruments (computed BETWEEN T and T+Delta; never predictors):
  P2 residual components — before-minus-after full-statistic field (fg grows
  by arrivals the frozen T-KMeans assigns to the fg cluster; bg candidates
  fixed), components at the validated top-frac 0.01 (0.05 sensitivity row).
  P3a change field — max-over-pairs |flow-share change| on the joint
  fg+scoped-bg graph, after-graph augmented with a seeded sample of in-scope
  arrival nodes (electrodes/eligibility = T nodes only); peaks = top 1%.

Honest-negative discipline: a detector whose fill-rate-ratio CI contains 1.0
is reported as such.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.data import CACHE_DIR
from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    closed_filter,
    count_within_radius,
    fightin_words_rank,
    knn_radii,
    level_set_components,
    mine_frequent_itemsets,
    voltage_current_scores,
)
from artifacts_scripts.gap_detection.fusion import fused_scores, veto_mask
from artifacts_scripts.gap_detection.polish import flow_share, gaussian_knn_graph, voltage_on_graph
from artifacts_scripts.gap_detection.run_followups import load_banked, update_report, update_results
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.synth import corridor_pairs
from artifacts_scripts.gap_detection.t2_outcomes import (
    fill_counts,
    fill_rate_ratio,
    greedy_location_groups,
    jaccard,
    knn_scale,
    matched_controls,
    precision_vs_set,
)

_T0 = time.monotonic()
K_NN = 10
TOP_K = 50
M_CONTROLS = 20
N_BOOT = 2000
MINE_MIN_SUPPORT = 30
ARRIVAL_NODE_CAP = 8000
P3A_PEAK_FRAC = 0.01
P2_TOP_FRAC = 0.01
P2_TOP_FRAC_SENS = 0.05


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def load_phase2():
    d = np.load(CACHE_DIR / "t2_corpus.npz", allow_pickle=True)
    K = sp.load_npz(CACHE_DIR / "t2_corpus_K.npz").tocsr()
    arr = np.load(CACHE_DIR / "t2_arrivals.npz", allow_pickle=True)
    grow = np.load(CACHE_DIR / "t2_growth.npz", allow_pickle=True)
    import joblib

    models = joblib.load(CACHE_DIR / "t2_models.joblib")
    meta = json.loads((CACHE_DIR / "t2_phase2_meta.json").read_text())
    return d, K, arr, grow, models, meta


# ---------------------------------------------------------------- predictors
def predict_fused(scorer: GapScorer, out: dict) -> np.ndarray:
    veto = veto_mask(scorer, out)  # fixed gates per Task A
    sc = fused_scores(out, veto)
    return np.argsort(sc)[::-1][:TOP_K]


def predict_miner(Kb_s: np.ndarray, Kf: np.ndarray, Zb_s: np.ndarray):
    fam = closed_filter(
        mine_frequent_itemsets(Kb_s, min_support=MINE_MIN_SUPPORT, max_len=3,
                               max_pairs_expand=40_000)
    )
    ranked = fightin_words_rank(fam, Kf, n_bg=Kb_s.shape[0])
    locs, items = [], []
    for r in ranked[:TOP_K]:
        cols = sorted(r["itemset"])
        v = Kb_s[:, cols[0]].copy()
        for c in cols[1:]:
            v &= Kb_s[:, c]
        sup = np.where(v)[0]
        cent = Zb_s[sup].mean(axis=0)
        cent = cent / max(np.linalg.norm(cent), 1e-9)
        locs.append(cent.astype(np.float32))
        items.append({"cols": [int(c) for c in cols], "z": round(r["z"], 3),
                      "bg_support": int(r["bg_support"]), "fg_support": int(r["fg_support"]),
                      "n_supporters": int(sup.size)})
    return np.vstack(locs), items, len(ranked)


def predict_voltage(Zf: np.ndarray, Zb_s: np.ndarray):
    labels, cents, pairs = corridor_pairs(Zf, Zb_s, n_clusters=12, seed=7)
    agg = np.full(Zb_s.shape[0], -np.inf)
    n_used = 0
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        cA, cB = cents[a], cents[b]
        dA = np.linalg.norm(Zb_s - cA, axis=1)
        dB = np.linalg.norm(Zb_s - cB, axis=1)
        nodes = np.where(dA + dB <= pair["lens_r"] * 1.1)[0]
        if nodes.size > 6000:
            nodes = nodes[np.argsort((dA + dB)[nodes])[:6000]]
        r_a = float(np.median(np.linalg.norm(Zf[labels == a] - cA, axis=1)))
        r_b = float(np.median(np.linalg.norm(Zf[labels == b] - cB, axis=1)))
        clamp0 = np.where(dA[nodes] <= r_a)[0]
        clamp1 = np.where(dB[nodes] <= r_b)[0]
        if clamp0.size < 5:
            clamp0 = np.argsort(dA[nodes])[:20]
        if clamp1.size < 5:
            clamp1 = np.argsort(dB[nodes])[:20]
        clamp1 = np.setdiff1d(clamp1, np.intersect1d(clamp0, clamp1))
        if clamp1.size < 5:
            continue
        cur = voltage_current_scores(Zb_s[nodes], clamp0, clamp1, k_graph=10)
        elig = np.ones(nodes.size, dtype=bool)
        elig[clamp0] = elig[clamp1] = False
        share = flow_share(cur, elig)
        share[~elig] = -np.inf
        agg[nodes] = np.maximum(agg[nodes], share)
        n_used += 1
    top = np.argsort(agg)[::-1][:TOP_K]
    return top, n_used, len(pairs), labels, cents, pairs, agg


def predict_levelset(scorer: GapScorer, out: dict) -> np.ndarray:
    comps = level_set_components(out["full"], scorer.cand, scorer.real_mask, top_frac=0.05)
    sel: list[int] = []
    for c in comps:
        members = np.asarray(c["members"])
        order = members[np.argsort(out["full"][members])[::-1]]
        for mbr in order:
            if len(sel) >= TOP_K:
                break
            sel.append(int(mbr))
        if len(sel) >= TOP_K:
            break
    return np.asarray(sel[:TOP_K], dtype=np.int64)


# ---------------------------------------------------------------- instruments
def instrument_p2(scorer: GapScorer, Zf: np.ndarray, Z_arr: np.ndarray,
                  fg_label: int, kmeans) -> dict:
    pred = kmeans.predict(Z_arr)
    fg_add = Z_arr[pred == fg_label]
    before = scorer.score(Zf)
    after = scorer.score(np.vstack([Zf, fg_add]).astype(np.float32))
    d = before["full"] - after["full"]  # positive where the statistic DROPPED (filled)
    comps01 = level_set_components(d, scorer.cand, scorer.real_mask, top_frac=P2_TOP_FRAC)
    comps05 = level_set_components(d, scorer.cand, scorer.real_mask, top_frac=P2_TOP_FRAC_SENS)
    set01 = (np.concatenate([c["members"] for c in comps01])
             if comps01 else np.array([], dtype=np.int64))
    set05 = (np.concatenate([c["members"] for c in comps05])
             if comps05 else np.array([], dtype=np.int64))
    return {"n_fg_arrivals": int(fg_add.shape[0]), "field": d,
            "comps01": comps01, "comps05": comps05,
            "set01": set01, "set05": set05}


def instrument_p3a(Zf: np.ndarray, Zb_s: np.ndarray, Z_arr: np.ndarray,
                   arr_in_scope: np.ndarray, labels, cents, pairs, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    in_scope_idx = np.where(arr_in_scope)[0]
    if in_scope_idx.size > ARRIVAL_NODE_CAP:
        in_scope_idx = np.sort(rng.choice(in_scope_idx, size=ARRIVAL_NODE_CAP, replace=False))
    Za = Z_arr[in_scope_idx]
    n_fg = Zf.shape[0]
    X0 = np.vstack([Zf, Zb_s]).astype(np.float32)
    X1 = np.vstack([Zf, Zb_s, Za]).astype(np.float32)
    bg_rows = np.arange(n_fg, n_fg + Zb_s.shape[0])

    electrodes = {}
    for p in pairs:
        a, b = p["a"], p["b"]
        cA, cB = cents[a], cents[b]
        r_a = float(np.median(np.linalg.norm(Zf[labels == a] - cA, axis=1)))
        r_b = float(np.median(np.linalg.norm(Zf[labels == b] - cB, axis=1)))
        c0 = np.where(np.linalg.norm(X0 - cA, axis=1) <= r_a)[0]
        c1 = np.where(np.linalg.norm(X0 - cB, axis=1) <= r_b)[0]
        c1 = np.setdiff1d(c1, np.intersect1d(c0, c1))
        if c0.size >= 5 and c1.size >= 5:
            electrodes[(a, b)] = (c0, c1)
    W0 = gaussian_knn_graph(X0, k=10)
    W1 = gaussian_knn_graph(X1, k=10)
    log(f"P3a graphs: T={X0.shape[0]} nodes, T+D={X1.shape[0]} nodes "
        f"(arrival nodes {Za.shape[0]:,}/{int(arr_in_scope.sum()):,} in-scope, "
        f"cap {ARRIVAL_NODE_CAP}); pairs usable {len(electrodes)}/{len(pairs)}")
    d_max = np.zeros(Zb_s.shape[0])
    for i, (pair, (c0, c1)) in enumerate(electrodes.items()):
        cur0 = voltage_on_graph(W0, c0, c1)
        cur1 = voltage_on_graph(W1, c0, c1)
        elig0 = np.ones(X0.shape[0], dtype=bool)
        elig0[c0] = elig0[c1] = False
        elig0[:n_fg] = False
        elig1 = np.zeros(X1.shape[0], dtype=bool)
        elig1[: X0.shape[0]] = elig0  # arrival nodes are conduits, never eligible
        s0 = flow_share(cur0, elig0)[bg_rows]
        s1 = flow_share(cur1, elig1)[bg_rows]
        d_max = np.maximum(d_max, np.abs(s1 - s0))
        if (i + 1) % 20 == 0:
            log(f"  P3a pair {i + 1}/{len(electrodes)}")
    n_peak = max(int(np.ceil(P3A_PEAK_FRAC * Zb_s.shape[0])), 10)
    peaks = np.argsort(d_max)[::-1][:n_peak]
    return {"field": d_max, "peaks": peaks, "n_pairs": len(electrodes),
            "n_arrival_nodes": int(Za.shape[0])}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    d, K_corpus, arr, grow, models, meta = load_phase2()
    Z = d["Z"]
    fg_mask = d["fg_mask"]
    scoped_idx = d["scoped_idx"]
    scope_thr = float(d["scope_thr"])
    titles = d["titles"]
    Zf = Z[fg_mask]
    Zb = Z[~fg_mask]
    Zb_s = Zb[scoped_idx]
    bg_rows_global = np.where(~fg_mask)[0]
    scoped_global = bg_rows_global[scoped_idx]
    Kf_dense = np.asarray(K_corpus[fg_mask].todense(), dtype=bool)
    Kb_s_dense = np.asarray(K_corpus[~fg_mask][scoped_idx].todense(), dtype=bool)
    Z_arr = arr["Z"]
    arr_titles = arr["titles"]
    Z_grow = grow["Z"]
    fg_label = int(meta["corpus"]["kmeans_labels_chosen"])
    log(f"loaded: corpus {Z.shape[0]:,} (fg {Zf.shape[0]:,}, scoped bg {Zb_s.shape[0]:,}), "
        f"arrivals {Z_arr.shape[0]:,}, growth {Z_grow.shape[0]:,}")

    scorer = GapScorer(bg=Zb_s, probes=None, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    out = scorer.score(Zf)

    # ---- pool-side quantities (radius, covariates, fills) — outcome-blind design
    pool_radii = knn_scale(Zb_s, Zb_s, k=K_NN, self_rows=np.ones(Zb_s.shape[0], dtype=bool))
    pool_growth = count_within_radius(Zb_s, Z_grow, pool_radii).astype(np.float64)
    pool_C = np.column_stack([
        np.log(scorer.r_Bk), np.log(out["no_hab"]), np.log1p(pool_growth),
    ])
    log(f"pool ready: median radius {np.median(pool_radii):.3f}, "
        f"median growth count {np.median(pool_growth):.0f}")

    # ---- predictors (T-only)
    preds: dict[str, dict] = {}
    fused_idx = predict_fused(scorer, out)
    preds["fused"] = {"kind": "docs", "idx": fused_idx}
    log(f"fused: top-{TOP_K} picked (veto frac "
        f"{float(veto_mask(scorer, out).mean()):.3f})")

    miner_locs, miner_items, n_ranked = predict_miner(Kb_s_dense, Kf_dense, Zb_s)
    preds["miner"] = {"kind": "centroids", "X": miner_locs, "items": miner_items}
    log(f"miner: {n_ranked:,} itemsets ranked; top-{TOP_K} pattern centroids "
        f"(median supporters {int(np.median([it['n_supporters'] for it in miner_items]))})")

    volt_idx, n_pairs_used, n_pairs, labels12, cents12, pairs12, volt_field = (
        predict_voltage(Zf, Zb_s))
    preds["voltage"] = {"kind": "docs", "idx": volt_idx}
    log(f"voltage: {n_pairs_used}/{n_pairs} qualifying pairs solved; top-{TOP_K} nodes")

    level_idx = predict_levelset(scorer, out)
    preds["levelset"] = {"kind": "docs", "idx": level_idx}
    log(f"levelset: top-{TOP_K} flattened component members")

    # machinery null: random-50 scoped candidates through the IDENTICAL pipeline
    rng_null = np.random.default_rng(17)
    preds["random50"] = {"kind": "docs",
                         "idx": rng_null.choice(Zb_s.shape[0], size=TOP_K, replace=False)}

    # ---- location-side geometry/covariates
    for name, p in preds.items():
        if p["kind"] == "docs":
            p["X"] = Zb_s[p["idx"]]
            p["radii"] = pool_radii[p["idx"]]
            p["C"] = pool_C[p["idx"]]
        else:
            X = p["X"]
            p["radii"] = knn_scale(X, Zb_s, k=K_NN)
            growth = count_within_radius(X, Z_grow, p["radii"]).astype(np.float64)
            r_b = knn_radii(X, Zb_s, K_NN)
            r_f = knn_radii(X, Zf, K_NN)
            p["C"] = np.column_stack([np.log(r_b), np.log(r_f), np.log1p(growth)])

    # ---- fills + matched nulls + ratios (the primary outcome)
    results: dict = {"per_detector": {}}
    fill_pool = fill_counts(Zb_s, pool_radii, Z_arr).astype(np.float64)
    log(f"pool fill computed: median {np.median(fill_pool):.0f}, "
        f"q90 {np.quantile(fill_pool, 0.9):.0f}")
    for name, p in preds.items():
        f_flag = (fill_pool[p["idx"]] if p["kind"] == "docs"
                  else fill_counts(p["X"], p["radii"], Z_arr).astype(np.float64))
        excluded = np.zeros(Zb_s.shape[0], dtype=bool)
        if p["kind"] == "docs":
            excluded[p["idx"]] = True
        # per-location exclusion: pool docs inside any flagged neighborhood
        for i in range(p["X"].shape[0]):
            dd = np.linalg.norm(Zb_s - p["X"][i], axis=1)
            excluded |= dd <= p["radii"][i]
        ctr, info = matched_controls(
            p["X"], p["radii"], p["C"], Zb_s, pool_radii, pool_C,
            excluded=excluded, m_controls=M_CONTROLS, seed=11,
        )
        f_ctrl = fill_pool[ctr]
        groups = greedy_location_groups(p["X"], p["radii"])
        rr = fill_rate_ratio(f_flag, f_ctrl, n_boot=N_BOOT, seed=13, groups=groups)
        # covariate balance: standardized mean |difference| flagged vs own controls
        sd_pool = pool_C.std(axis=0) + 1e-12
        smd = np.abs(p["C"] - pool_C[ctr].mean(axis=1)) / sd_pool
        balance = {c: round(float(smd[:, j].mean()), 3)
                   for j, c in enumerate(["log_rB10", "log_rF10", "log1p_growth"])}
        results["per_detector"][name] = {
            "fill": rr, "matching": info, "covariate_balance_smd": balance,
            "n_excluded_pool": int(excluded.sum()),
        }
        log(f"{name}: fill ratio {rr['ratio']} CI [{rr['ci_lo']}, {rr['ci_hi']}] "
            f"clusteredCI [{rr['ci_lo_clustered']}, {rr['ci_hi_clustered']}] "
            f"n_groups {rr['n_groups']} "
            f"(flagged {rr['mean_fill_flagged']}, null {rr['mean_fill_controls']}; "
            f"relaxed {info['n_relaxed']}/{TOP_K} max_lvl {info['relax_level_max']}; "
            f"balance {balance})")

    # ---- outcome instruments
    log("instrument P2 (residual components)...")
    p2 = instrument_p2(scorer, Zf, Z_arr, fg_label, models["kmeans"])
    log(f"P2: fg-assigned arrivals {p2['n_fg_arrivals']:,}; "
        f"components tf{P2_TOP_FRAC}: {len(p2['comps01'])} "
        f"(member union {p2['set01'].size}), tf{P2_TOP_FRAC_SENS}: {len(p2['comps05'])} "
        f"({p2['set05'].size})")

    r_arr_f = knn_radii(Z_arr, Zf, K_NN)
    arr_in_scope = r_arr_f <= scope_thr
    log(f"in-scope arrivals (scope rule at T): {int(arr_in_scope.sum()):,}")
    p3 = instrument_p3a(Zf, Zb_s, Z_arr, arr_in_scope, labels12, cents12, pairs12)
    log(f"P3a: peaks {p3['peaks'].size} (top {P3A_PEAK_FRAC:.0%}), "
        f"pairs {p3['n_pairs']}")

    # instrument agreement (sanity row)
    agree = {
        "jaccard_p2tf01_p3apeaks": round(jaccard(p2["set01"], p3["peaks"]), 4),
        "jaccard_p2tf05_p3apeaks": round(jaccard(p2["set05"], p3["peaks"]), 4),
        "n_p2_tf01": int(p2["set01"].size), "n_p2_tf05": int(p2["set05"].size),
        "n_p3a_peaks": int(p3["peaks"].size),
    }
    log(f"instrument agreement: {json.dumps(agree)}")

    # ---- precision@K vs realized change (uniform neighborhood hit rule)
    tgt_p2_all = Zb_s[p2["set01"]] if p2["set01"].size else np.zeros((0, Z.shape[1]))
    tgt_p3_all = Zb_s[p3["peaks"]]
    chance = {
        "vs_p2_residual_tf01": round(precision_vs_set(Zb_s, pool_radii, tgt_p2_all), 4),
        "vs_p3a_peaks": round(precision_vs_set(Zb_s, pool_radii, tgt_p3_all), 4),
    }
    results["chance_hit_rates_poolwide"] = chance
    log(f"chance hit rates (pool-wide, same rule): {json.dumps(chance)}")
    # predictor/instrument non-independence check: static voltage field vs |dshare|
    from scipy.stats import spearmanr

    finite = np.isfinite(volt_field)
    rho = float(spearmanr(volt_field[finite], p3["field"][finite]).statistic)
    results["voltage_static_vs_p3a_spearman"] = round(rho, 4)
    log(f"voltage static share vs P3a |dshare| spearman (non-independence): {rho:.3f}")
    for name, p in preds.items():
        tgt_p2 = tgt_p2_all
        tgt_p3 = tgt_p3_all
        results["per_detector"][name]["precision_at_k"] = {
            "vs_p2_residual_tf01": round(precision_vs_set(p["X"], p["radii"], tgt_p2), 4),
            "vs_p3a_peaks": round(precision_vs_set(p["X"], p["radii"], tgt_p3), 4),
        }
        log(f"{name}: P@{TOP_K} vs P2 {results['per_detector'][name]['precision_at_k']['vs_p2_residual_tf01']}, "
            f"vs P3a {results['per_detector'][name]['precision_at_k']['vs_p3a_peaks']}")

    # ---- realized-change descriptive top-10 (titles/labels only)
    desc = []
    for c in p2["comps01"][:10]:
        members = np.asarray(c["members"])
        order = members[np.argsort(p2["field"][members])[::-1]]
        desc.append({
            "n_eff": round(c["n_eff"], 1), "size": int(c["size"]),
            "top_titles": [str(titles[scoped_global[i]]) for i in order[:5]],
        })
    results["realized_change_top10"] = desc
    for i, q in enumerate(desc[:10], 1):
        log(f"realized-change {i}: n_eff={q['n_eff']} size={q['size']} :: "
            + "; ".join(q["top_titles"][:3]))

    # ---- persist
    results["config"] = {
        "K": TOP_K, "M": M_CONTROLS, "n_boot": N_BOOT,
        "radius_rule": "10-NN scale vs scoped bg at T (self-excluded for candidates)",
        "covariates": ["log r_B10", "log r_F10", "log1p growth (T-2y,T] in own radius"],
        "fill": "arrival count within the location radius (all eligible arrivals)",
        "p2": {"top_frac": P2_TOP_FRAC, "sens": P2_TOP_FRAC_SENS,
               "direction": "before - after (statistic drop = filled)",
               "fg_growth": "arrivals assigned to fg cluster by frozen T-KMeans",
               "bg_side": "candidate bg fixed at T (banked P2 form)"},
        "p3a": {"peak_frac": P3A_PEAK_FRAC, "arrival_node_cap": ARRIVAL_NODE_CAP,
                "eligible": "scoped-bg rows only; arrival nodes are conduits"},
        "miner": {"min_support": MINE_MIN_SUPPORT, "n_ranked": n_ranked},
        "voltage": {"pairs_used": n_pairs_used, "pairs_qualifying": n_pairs},
        "p2_n_fg_arrivals": p2["n_fg_arrivals"],
        "p3a_pairs": p3["n_pairs"], "p3a_arrival_nodes": p3["n_arrival_nodes"],
        "n_in_scope_arrivals_T_geometry": int(arr_in_scope.sum()),
    }
    results["instrument_agreement"] = agree
    results["miner_top_patterns"] = [
        {**it, "keywords": [str(d["vocab"][c]) for c in it["cols"]]}
        for it in miner_items[:10]
    ]
    if not args.dry:
        banked = load_banked()
        row = banked.get("task_t2_wiki", {})
        row["phase3"] = _py(results)
        update_results("task_t2_wiki", row)
        log("banked phase3 rows")
    log("phase 3 done")


if __name__ == "__main__":
    main()
