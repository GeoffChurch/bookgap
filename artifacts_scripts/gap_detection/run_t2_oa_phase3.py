"""Track 2 OA phase — Phase 3: predict at T, score against (T, T+Δ] arrivals.

Mirrors run_t2_phase3.py (wiki) with the OpenAlex data layer + the OA-specific
win: the DATED CITATION GRAPH. Reuses the SAME outcome machinery
(t2_outcomes: matched nulls, fill-rate ratio, clustered bootstrap, precision@K).

PRE-COMMITTED design (identical to the wiki run; never tuned on outcomes):
  K = 50 locations/detector; M = 20 matched controls/location; radius = the
  location's 10-NN scale vs the scoped-bg set at T; covariates = log r_B,10,
  log r_F,10, log1p growth in (T-Δ, T]; bootstrap B = 2000; clustered over
  greedy location groups.

Predictors at T (T-only inputs):
  fused        — scoped-bg ranked by r_F,10, Task-A fixed veto; top-50.
  miner        — scoped fg-absent abstract-term patterns (Fightin' Words vs
                 T-fg), top-50 itemsets, location = supporter centroid.
  voltage_emb  — voltage corridors on the EMBEDDING-kNN bg graph (the wiki form).
  voltage_cite — voltage corridors on the DATED CITATION GRAPH at T (OA-only):
                 same cluster-pair electrodes, but flow solved on the citation
                 adjacency restricted to scoped-bg nodes. Reported separately.
  levelset     — absolute level-set components; first 50 flattened members.
  random50     — machinery null (random scoped candidates through the pipeline).

Outcome instruments (computed BETWEEN T and T+Δ; NEVER predictors):
  P2 residual components — before/after full-statistic field (fg grown by frozen-
    KMeans-assigned arrivals, bg fixed), components at top-frac 0.01 (0.05 sens).
  P3a flow-change field — max-over-pairs |flow-share change| on the DATED
    CITATION GRAPH difference (T-graph vs T+Δ-graph), restricted to scoped-bg
    nodes — the leak-free object wiki could not provide. (Embedding-kNN P3a also
    reported as a sensitivity row.)

Honest-negative discipline: a detector whose fill-ratio clustered CI contains 1.0
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
from artifacts_scripts.gap_detection.oa_corpus import CACHE_DIR
from artifacts_scripts.gap_detection.polish import flow_share, voltage_on_graph
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import load_banked, update_results
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
P3A_PEAK_FRAC = 0.01
P2_TOP_FRAC = 0.01
P2_TOP_FRAC_SENS = 0.05
ARRIVAL_NODE_CAP = 8000


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def load_phase2():
    d = np.load(CACHE_DIR / "t2_oa_corpus.npz", allow_pickle=True)
    K = sp.load_npz(CACHE_DIR / "t2_oa_corpus_K.npz").tocsr()
    arr = np.load(CACHE_DIR / "t2_oa_arrivals.npz", allow_pickle=True)
    grow = np.load(CACHE_DIR / "t2_oa_growth.npz", allow_pickle=True)
    graph = np.load(CACHE_DIR / "t2_oa_graph.npz", allow_pickle=True)
    import joblib

    models = joblib.load(CACHE_DIR / "t2_oa_models.joblib")
    meta = json.loads((CACHE_DIR / "t2_oa_phase2_meta.json").read_text())
    return d, K, arr, grow, graph, models, meta


# ---------------------------------------------------------------- predictors
def predict_fused(scorer: GapScorer, out: dict) -> np.ndarray:
    veto = veto_mask(scorer, out)
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
        if sup.size == 0:
            continue
        cent = Zb_s[sup].mean(axis=0)
        cent = cent / max(np.linalg.norm(cent), 1e-9)
        locs.append(cent.astype(np.float32))
        items.append({"cols": [int(c) for c in cols], "z": round(r["z"], 3),
                      "bg_support": int(r["bg_support"]), "fg_support": int(r["fg_support"]),
                      "n_supporters": int(sup.size)})
    return np.vstack(locs), items, len(ranked)


def predict_voltage_embedding(Zf: np.ndarray, Zb_s: np.ndarray):
    """Voltage corridors on the embedding-kNN bg graph (the wiki form)."""
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
        if nodes.size < 12:
            continue
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


def _cite_adjacency(graph, node_global_rows, t_days):
    """Undirected weighted citation adjacency at T over an ordered node set.

    Nodes = `node_global_rows` (universe row indices), relabeled to local 0..N-1.
    An edge exists iff a dated citing act (citing.date <= T) links the two
    within-universe works (either direction); weight = number of such acts.
    """
    src, dst, cdays = graph["src"], graph["dst"], graph["citing_days"]
    present = np.isfinite(cdays) & (cdays <= t_days)
    s, d = src[present], dst[present]
    n_univ = int(graph["n_works"])
    local = np.full(n_univ, -1, dtype=np.int64)
    local[node_global_rows] = np.arange(node_global_rows.size)
    ls, ld = local[s], local[d]
    keep = (ls >= 0) & (ld >= 0)
    ls, ld = ls[keep], ld[keep]
    N = node_global_rows.size
    A = sp.csr_matrix((np.ones(ls.size, np.float32), (ls, ld)), shape=(N, N))
    return (A + A.T).tocsr()  # undirected, weight = #acts


def predict_voltage_citation(Zf, Zb_s, W_cite, labels, cents, pairs):
    """Voltage corridors on the DATED CITATION GRAPH at T (OA-only).

    Electrodes are the SAME embedding cluster-pair cores (chosen in the embedding
    space, so the corridor question is identical), but the flow is solved on the
    citation adjacency W_cite over scoped-bg nodes. Nodes with zero citation
    degree are ineligible (no current can pass)."""
    S = Zb_s.shape[0]
    deg = np.asarray(W_cite.sum(axis=1)).ravel()
    agg = np.full(S, -np.inf)
    n_used = 0
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        cA, cB = cents[a], cents[b]
        dA = np.linalg.norm(Zb_s - cA, axis=1)
        dB = np.linalg.norm(Zb_s - cB, axis=1)
        r_a = float(np.median(np.linalg.norm(Zf[labels == a] - cA, axis=1)))
        r_b = float(np.median(np.linalg.norm(Zf[labels == b] - cB, axis=1)))
        c0 = np.where((dA <= r_a) & (deg > 0))[0]
        c1 = np.where((dB <= r_b) & (deg > 0))[0]
        c1 = np.setdiff1d(c1, np.intersect1d(c0, c1))
        if c0.size < 5 or c1.size < 5:
            continue
        cur = voltage_on_graph(W_cite, c0, c1)
        elig = (deg > 0)
        elig[c0] = elig[c1] = False
        share = flow_share(cur, elig)
        share = np.where(elig, share, -np.inf)
        agg = np.maximum(agg, share)
        n_used += 1
    top = np.argsort(agg)[::-1][:TOP_K]
    return top, n_used, agg


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
    d = before["full"] - after["full"]  # positive where the statistic dropped (filled)
    comps01 = level_set_components(d, scorer.cand, scorer.real_mask, top_frac=P2_TOP_FRAC)
    comps05 = level_set_components(d, scorer.cand, scorer.real_mask, top_frac=P2_TOP_FRAC_SENS)
    set01 = (np.concatenate([c["members"] for c in comps01])
             if comps01 else np.array([], dtype=np.int64))
    set05 = (np.concatenate([c["members"] for c in comps05])
             if comps05 else np.array([], dtype=np.int64))
    return {"n_fg_arrivals": int(fg_add.shape[0]), "field": d,
            "comps01": comps01, "comps05": comps05, "set01": set01, "set05": set05}


def instrument_p3a_citation(graph, scoped_global_rows, arr_global_rows, arr_in_scope,
                            t_days, end_days, labels, cents, pairs, Zf, Zb_s,
                            seed: int = 7) -> dict:
    """P3a flow-change on the DATED CITATION GRAPH difference (the OA win).

    Node set = scoped-bg (first S rows) ∪ in-scope ARRIVAL works (conduits). The
    arrivals are the works whose CITING acts have date in (T, T+Δ]; they exist in
    W1 (citing ≤ T+Δ) but NOT in W0 (citing ≤ T), so the difference localizes
    where new citations attach into the pre-T parsing structure — the leak-free
    object wiki could not provide. Electrodes = embedding cluster-pair cores
    (scoped-bg only); flow SHARE eligibility = scoped-bg nodes only (arrivals are
    conduits, never scored), exactly the wiki P3a augmentation but on citations."""
    rng = np.random.default_rng(seed)
    S = Zb_s.shape[0]
    in_scope_arr = arr_global_rows[arr_in_scope]
    if in_scope_arr.size > ARRIVAL_NODE_CAP:
        in_scope_arr = np.sort(rng.choice(in_scope_arr, size=ARRIVAL_NODE_CAP, replace=False))
    nodes = np.concatenate([scoped_global_rows, in_scope_arr])  # [scoped-bg | arrivals]
    W0 = _cite_adjacency(graph, nodes, t_days)        # citing ≤ T (no arrival edges)
    W1 = _cite_adjacency(graph, nodes, end_days)      # citing ≤ T+Δ (arrival edges in)
    deg0 = np.asarray(W0.sum(axis=1)).ravel()
    deg1 = np.asarray(W1.sum(axis=1)).ravel()
    scoped_slot = np.zeros(nodes.size, dtype=bool)
    scoped_slot[:S] = True
    d_max = np.zeros(S)
    n_pairs = 0
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        cA, cB = cents[a], cents[b]
        dA = np.linalg.norm(Zb_s - cA, axis=1)
        dB = np.linalg.norm(Zb_s - cB, axis=1)
        r_a = float(np.median(np.linalg.norm(Zf[labels == a] - cA, axis=1)))
        r_b = float(np.median(np.linalg.norm(Zf[labels == b] - cB, axis=1)))
        # electrodes are scoped-bg slots with citation degree in the T+Δ graph
        c0 = np.where((dA <= r_a) & (deg1[:S] > 0))[0]
        c1 = np.where((dB <= r_b) & (deg1[:S] > 0))[0]
        c1 = np.setdiff1d(c1, np.intersect1d(c0, c1))
        if c0.size < 5 or c1.size < 5:
            continue
        cur0 = voltage_on_graph(W0, c0, c1)
        cur1 = voltage_on_graph(W1, c0, c1)
        # eligible for SHARE normalization = scoped-bg conduit-free nodes (arrivals excluded)
        elig0 = scoped_slot & (deg0 > 0)
        elig0[c0] = elig0[c1] = False
        elig1 = scoped_slot & (deg1 > 0)
        elig1[c0] = elig1[c1] = False
        s0 = flow_share(cur0, elig0)[:S]
        s1 = flow_share(cur1, elig1)[:S]
        elig_s = (elig0 | elig1)[:S]
        d_max = np.where(elig_s, np.maximum(d_max, np.abs(s1 - s0)), d_max)
        n_pairs += 1
    n_peak = max(int(np.ceil(P3A_PEAK_FRAC * S)), 10)
    peaks = np.argsort(d_max)[::-1][:n_peak]
    return {"field": d_max, "peaks": peaks, "n_pairs": n_pairs,
            "n_arrival_nodes": int(in_scope_arr.size),
            "n_edges_T": int(W0.nnz // 2), "n_edges_Tdelta": int(W1.nnz // 2),
            "n_nodes_with_cite_T": int((deg0 > 0).sum()),
            "n_nodes_with_cite_Tdelta": int((deg1 > 0).sum())}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    d, K_corpus, arr, grow, graph, models, meta = load_phase2()
    Z = d["Z"]
    fg_mask = d["fg_mask"]
    scoped_idx = d["scoped_idx"]
    scope_thr = float(d["scope_thr"])
    wids = d["wids"]
    corpus_store_rows = d["store_rows"]  # universe row index per corpus doc
    Zf = Z[fg_mask]
    Zb = Z[~fg_mask]
    Zb_s = Zb[scoped_idx]
    bg_rows_global = np.where(~fg_mask)[0]
    scoped_local_to_corpus = bg_rows_global[scoped_idx]            # corpus-doc index
    scoped_global_rows = corpus_store_rows[scoped_local_to_corpus]  # universe row index
    Kf_dense = np.asarray(K_corpus[fg_mask].todense(), dtype=bool)
    Kb_s_dense = np.asarray(K_corpus[~fg_mask][scoped_idx].todense(), dtype=bool)
    Z_arr = arr["Z"]
    arr_global_rows = arr["store_rows"].astype(np.int64)  # universe row per arrival
    Z_grow = grow["Z"]
    fg_label = int(meta["fg_label"])
    t_days = float(meta["t_days"])
    end_days = float(meta["end_days"])
    log(f"loaded: corpus {Z.shape[0]:,} (fg {Zf.shape[0]:,}, scoped bg {Zb_s.shape[0]:,}), "
        f"arrivals {Z_arr.shape[0]:,}, growth {Z_grow.shape[0]:,}; "
        f"intra-universe edges {int(graph['src'].size):,}")

    scorer = GapScorer(bg=Zb_s, probes=None, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    out = scorer.score(Zf)

    # ---- pool-side quantities (outcome-blind)
    pool_radii = knn_scale(Zb_s, Zb_s, k=K_NN, self_rows=np.ones(Zb_s.shape[0], dtype=bool))
    pool_growth = count_within_radius(Zb_s, Z_grow, pool_radii).astype(np.float64)
    pool_C = np.column_stack([
        np.log(scorer.r_Bk), np.log(out["no_hab"]), np.log1p(pool_growth),
    ])
    log(f"pool ready: median radius {np.median(pool_radii):.3f}, "
        f"median growth {np.median(pool_growth):.0f}")

    # ---- citation adjacency at T (for the citation-voltage predictor)
    W_cite_T = _cite_adjacency(graph, scoped_global_rows, t_days)
    log(f"citation graph at T over scoped bg: {Zb_s.shape[0]:,} nodes, "
        f"{int(W_cite_T.nnz // 2):,} undirected edges, "
        f"{int((np.asarray(W_cite_T.sum(axis=1)).ravel() > 0).sum()):,} non-isolated")

    # ---- predictors
    preds: dict[str, dict] = {}
    fused_idx = predict_fused(scorer, out)
    preds["fused"] = {"kind": "docs", "idx": fused_idx}
    log(f"fused: top-{TOP_K} (veto frac {float(veto_mask(scorer, out).mean()):.3f})")

    miner_locs, miner_items, n_ranked = predict_miner(Kb_s_dense, Kf_dense, Zb_s)
    preds["miner"] = {"kind": "centroids", "X": miner_locs, "items": miner_items}
    log(f"miner: {n_ranked:,} itemsets; top-{TOP_K} centroids "
        f"(median supporters {int(np.median([it['n_supporters'] for it in miner_items]))})")

    volt_idx, n_used_e, n_pairs_e, labels12, cents12, pairs12, volt_field_e = (
        predict_voltage_embedding(Zf, Zb_s))
    preds["voltage_emb"] = {"kind": "docs", "idx": volt_idx}
    log(f"voltage_emb: {n_used_e}/{n_pairs_e} pairs solved; top-{TOP_K} nodes")

    volt_c_idx, n_used_c, volt_field_c = predict_voltage_citation(
        Zf, Zb_s, W_cite_T, labels12, cents12, pairs12)
    preds["voltage_cite"] = {"kind": "docs", "idx": volt_c_idx}
    log(f"voltage_cite: {n_used_c}/{len(pairs12)} pairs solved on the dated citation "
        f"graph; top-{TOP_K} nodes")

    level_idx = predict_levelset(scorer, out)
    preds["levelset"] = {"kind": "docs", "idx": level_idx}
    log(f"levelset: top-{TOP_K} flattened members")

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

    # ---- fills + matched nulls + ratios (primary outcome)
    results: dict = {"per_detector": {}}
    fill_pool = fill_counts(Zb_s, pool_radii, Z_arr).astype(np.float64)
    log(f"pool fill: median {np.median(fill_pool):.0f}, q90 {np.quantile(fill_pool, 0.9):.0f}")
    for name, p in preds.items():
        f_flag = (fill_pool[p["idx"]] if p["kind"] == "docs"
                  else fill_counts(p["X"], p["radii"], Z_arr).astype(np.float64))
        excluded = np.zeros(Zb_s.shape[0], dtype=bool)
        if p["kind"] == "docs":
            excluded[p["idx"]] = True
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
        sd_pool = pool_C.std(axis=0) + 1e-12
        smd = np.abs(p["C"] - pool_C[ctr].mean(axis=1)) / sd_pool
        balance = {c: round(float(smd[:, j].mean()), 3)
                   for j, c in enumerate(["log_rB10", "log_rF10", "log1p_growth"])}
        results["per_detector"][name] = {
            "fill": rr, "matching": info, "covariate_balance_smd": balance,
            "n_excluded_pool": int(excluded.sum()),
        }
        log(f"{name}: fill ratio {rr['ratio']} clusteredCI "
            f"[{rr['ci_lo_clustered']}, {rr['ci_hi_clustered']}] n_groups {rr['n_groups']} "
            f"(flagged {rr['mean_fill_flagged']}, null {rr['mean_fill_controls']}; "
            f"relaxed {info['n_relaxed']}/{TOP_K}; balance {balance})")

    # ---- outcome instruments
    log("instrument P2 (residual components)...")
    p2 = instrument_p2(scorer, Zf, Z_arr, fg_label, models["kmeans"])
    log(f"P2: fg-assigned arrivals {p2['n_fg_arrivals']:,}; "
        f"tf{P2_TOP_FRAC} comps {len(p2['comps01'])} (members {p2['set01'].size}), "
        f"tf{P2_TOP_FRAC_SENS} comps {len(p2['comps05'])} ({p2['set05'].size})")

    # which arrivals are in scope at T (scope rule, T geometry) — P3a conduits
    r_arr_f = knn_radii(Z_arr, Zf, K_NN)
    arr_in_scope = r_arr_f <= scope_thr
    log(f"in-scope arrivals (scope rule at T): {int(arr_in_scope.sum()):,} of {Z_arr.shape[0]:,}")
    log("instrument P3a (citation-graph flow change — the OA win)...")
    p3 = instrument_p3a_citation(
        graph, scoped_global_rows, arr_global_rows, arr_in_scope,
        t_days, end_days, labels12, cents12, pairs12, Zf, Zb_s)
    log(f"P3a (citation): peaks {p3['peaks'].size} (top {P3A_PEAK_FRAC:.0%}), "
        f"pairs {p3['n_pairs']}, arrival-conduit nodes {p3['n_arrival_nodes']:,}, "
        f"edges T {p3['n_edges_T']:,} -> T+Δ {p3['n_edges_Tdelta']:,}")

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
    log(f"chance hit rates (pool-wide): {json.dumps(chance)}")

    for name, p in preds.items():
        results["per_detector"][name]["precision_at_k"] = {
            "vs_p2_residual_tf01": round(precision_vs_set(p["X"], p["radii"], tgt_p2_all), 4),
            "vs_p3a_peaks": round(precision_vs_set(p["X"], p["radii"], tgt_p3_all), 4),
        }
        pk = results["per_detector"][name]["precision_at_k"]
        log(f"{name}: P@{TOP_K} vs P2 {pk['vs_p2_residual_tf01']}, vs P3a {pk['vs_p3a_peaks']}")

    # ---- realized-change descriptive top-10 (W-ids + fg-adjacency check; no text)
    desc = []
    for c in p2["comps01"][:10]:
        members = np.asarray(c["members"])
        order = members[np.argsort(p2["field"][members])[::-1]]
        desc.append({
            "n_eff": round(c["n_eff"], 1), "size": int(c["size"]),
            "top_wids": [str(wids[scoped_local_to_corpus[i]]) for i in order[:5]],
        })
    results["realized_change_top10"] = desc

    # ---- persist
    results["config"] = {
        "K": TOP_K, "M": M_CONTROLS, "n_boot": N_BOOT,
        "T": meta["choice"]["T"], "delta_years": meta["choice"]["delta_years"],
        "universe": meta["universe"], "topics": meta["topics"],
        "radius_rule": "10-NN scale vs scoped bg at T (self-excluded for candidates)",
        "covariates": ["log r_B10", "log r_F10", "log1p growth (T-Δ,T] in own radius"],
        "fill": "abstracted-arrival count within the location radius",
        "p2": {"top_frac": P2_TOP_FRAC, "sens": P2_TOP_FRAC_SENS,
               "direction": "before - after (statistic drop = filled)",
               "fg_growth": "arrivals assigned to fg cluster by frozen T-KMeans",
               "bg_side": "candidate bg fixed at T (banked P2 form)"},
        "p3a": {"peak_frac": P3A_PEAK_FRAC,
                "graph": "DATED CITATION GRAPH difference (T vs T+Δ); nodes = scoped-bg "
                         "∪ in-scope arrival conduits; arrivals appear only in W1",
                "eligible": "scoped-bg nodes with citation degree (arrivals are conduits)"},
        "miner": {"min_support": MINE_MIN_SUPPORT, "n_ranked": n_ranked},
        "voltage_emb": {"pairs_used": n_used_e, "pairs_qualifying": n_pairs_e},
        "voltage_cite": {"pairs_used": n_used_c, "pairs_qualifying": len(pairs12)},
        "p2_n_fg_arrivals": p2["n_fg_arrivals"],
        "p3a_pairs": p3["n_pairs"], "p3a_arrival_nodes": p3["n_arrival_nodes"],
        "n_in_scope_arrivals_T_geometry": int(arr_in_scope.sum()),
        "p3a_n_edges_T": p3["n_edges_T"], "p3a_n_edges_Tdelta": p3["n_edges_Tdelta"],
        "scope_thr": scope_thr,
        "fg_top_terms": meta["fg_top_terms"][:15],
        "graph_T": meta["graph_T"], "graph_T_plus_delta": meta["graph_T_plus_delta"],
    }
    results["instrument_agreement"] = agree
    results["miner_top_patterns"] = [
        {**it, "keywords": [str(d["vocab"][c]) for c in it["cols"]]}
        for it in miner_items[:10]
    ]
    if not args.dry:
        banked = load_banked()
        row = banked.get("task_t2_oa", {})
        row["phase3"] = _py(results)
        update_results("task_t2_oa", row)
        log("banked phase3 rows under task_t2_oa")
    log("phase 3 done")


if __name__ == "__main__":
    main()
