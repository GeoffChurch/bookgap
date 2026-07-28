"""Reroute-vs-arrival dissociation (task_oa_dissoc) — OA NLP temporal holdout.

Born from task_t2_oa: voltage_cite INVERTED (fill 0.71 [0.57, 0.88] — high
|flow-share change| regions of the dated citation graph UNDER-fill with new
abstracted papers) while self-agreeing with the P3a change instrument (P@50
0.74). Where citations reroute is NOT where papers land. Three questions:

Q1 — WHAT are the under-filling flow hubs? Characterize the voltage_cite top-50
  (and P3a peaks) with cache-resident metadata and test the INFRASTRUCTURE-NODE
  hypothesis (benchmark/dataset/resource papers: cited-by-everyone,
  landed-on-by-no-one) vs method clusters.

  PRE-SPECIFIED infrastructure operationalization (pool-quantile thresholds,
  frozen before looking at hub composition; title cues SECONDARY only):
      g_i  = in-universe citation in-degree growth of node i over (T, T+Δ]
             (dated graph: #directed edges with dst=i and T < citing_days ≤ T+Δ
             — leak-free and time-resolved, unlike the 2026 cited_by_count
             snapshot, which is reported as a secondary descriptor only)
      a_i  = abstracted-arrival fill of node i's 10-NN-radius neighborhood
             (the phase-3 fill outcome)
      IR_i = log1p(g_i) − log1p(a_i)          (continuous infrastructure ratio)
      INFRA(i) := g_i ≥ q75(g | scoped, citation-active at T+Δ)
                  AND a_i ≤ median(a | scoped)
  plus in-degree-growth CONCENTRATION: top-1 share / effective number of g over
  each node's radius-neighborhood (benchmark = growth concentrated on the hub).

Q2 — Does rerouting LEAD arrivals? Split Δ: W1 = (2015-01-01, 2017-01-01],
  W2 = (2017-01-01, 2019-01-01]. The W1 flow-share-change field uses ONLY citing
  acts ≤ 2017 and W1-dated conduits (window-split leakage guard unit-pinned).
  Test: do W1 reroute peaks out-fill growth-matched nulls in W2 ARRIVALS
  (t2_outcomes machinery, null-calibrated FIRST)? Strict variant adds W1-arrival
  fill as a 4th matching covariate ("leads, not lags"). Pre-specified verdict:
  leading-indicator CONFIRMED iff clustered CI_lo > 1 on the strict (4-cov)
  variant; INVERTED iff CI_hi < 1; else refuted (reroute ≠ early-warning).

Q3 — Formalize the two fields over the same scoped candidates: F = max-over-
  pairs |flow-share change| on the dated citation graph (T vs T+Δ); A =
  abstracted-arrival count in the own-radius neighborhood (and its covariate-
  residualized variant). Correlation, top-k overlap, 2x2 high/low decomposition
  with region examples (titles/top-terms only).

Identity discipline: every run re-derives the banked phase-3 objects and asserts
them (voltage_cite mean fill 5.5 / controls 7.717 / 42 groups / P@50 vs P3a
0.74; P3a 3 pairs, 78 peaks, 14,002 → 33,268 edges) before measuring anything new.

Context hygiene: abstracts are decoded inside the data layer only; this module
emits titles, top-terms, counts, numbers. Network use is limited to the
justified one-shot TITLE fetch (display_name was not in the phase-0 select
list): ≤ ~800 specific W-ids, 50/call on the polite pool, every batch cached to
cache/oa/titles/ (offline reruns).

Usage (repo root):
    uv run python -m artifacts_scripts.gap_detection.oa_dissoc --prep-titles
    uv run python -m artifacts_scripts.gap_detection.oa_dissoc --fetch-titles
    uv run python -m artifacts_scripts.gap_detection.oa_dissoc --q1
    uv run python -m artifacts_scripts.gap_detection.oa_dissoc --q2
    uv run python -m artifacts_scripts.gap_detection.oa_dissoc --q3
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
# A second agent runs concurrently: BLAS/n_jobs ≤ 6 (set BEFORE numpy import;
# downstream modules use setdefault so this wins).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "6")

import argparse
import gzip
import hashlib
import json
import re
import time

import numpy as np

from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    count_within_radius,
    effective_number,
    knn_radii,
)
from artifacts_scripts.gap_detection.oa_corpus import (
    MAILTO,
    REPO,
    USER_AGENT,
    iter_works,
    short_id,
    year_to_days,
)
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import load_banked, update_results
from artifacts_scripts.gap_detection.run_t2_oa_phase3 import (
    K_NN,
    TOP_K,
    _cite_adjacency,
    instrument_p3a_citation,
    load_phase2,
    predict_levelset,
    predict_voltage_citation,
)
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
CACHE_DIR = REPO / "artifacts/gap_detection_eval/cache"
TITLES_DIR = REPO / "cache/oa/titles"
WANTED_IDS_PATH = TITLES_DIR / "wanted_ids.json"
TITLES_MAP_PATH = TITLES_DIR / "titles_map.json.gz"
CITEDBY_PATH = CACHE_DIR / "oa_dissoc_citedby.npz"
SUB_KEY = "primary_topic.id:T10181"

M_CONTROLS = 20
N_BOOT = 2000
W1_END_DAYS = year_to_days(2017)  # (T, W1_END] = window W1
INFRA_G_QUANTILE = 0.75
INFRA_A_QUANTILE = 0.50
CALIPERS_4 = (0.2, 0.5, 0.5, 0.5)

# SECONDARY title-cue signal (descriptive only, pre-specified): benchmark /
# dataset / resource markers of CoNLL/UD-style infrastructure papers.
TITLE_CUE_RE = re.compile(
    r"\b(shared\s+task|treebank\w*|corpus|corpora|dataset\w*|data\s+set|"
    r"benchmark\w*|annotat\w*|conll|semeval|universal\s+dependenc\w*|"
    r"evaluation\s+campaign|resource\w*|toolkit\w*|database\w*|wordnet|"
    r"propbank|framenet|lexicon\w*)\b",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ small helpers
def region_terms(K_pool: np.ndarray, members: np.ndarray, vocab: np.ndarray,
                 n_top: int = 12, alpha: float = 1.0, min_df: int = 3) -> list[str]:
    """Top distinctive keyword terms of a region (log-odds vs the scoped rest);
    same form as describe_t2_oa.region_terms (duplicated to avoid importing a
    __main__-style script module)."""
    K_pool = np.asarray(K_pool, dtype=bool)
    members = np.asarray(members, dtype=np.int64)
    if members.size == 0:
        return []
    inside = K_pool[members].sum(axis=0).astype(np.float64)
    rest = ~np.isin(np.arange(K_pool.shape[0]), members)
    outside = K_pool[rest].sum(axis=0).astype(np.float64)
    n_in, n_out = float(members.size), float(rest.sum())
    score = (np.log((inside + alpha) / (n_in - inside + alpha))
             - np.log((outside + alpha) / (n_out - outside + alpha)))
    score[inside < min_df] = -np.inf
    order = np.argsort(score)[::-1][:n_top]
    return [str(vocab[j]) for j in order if np.isfinite(score[j])]


def in_degrees_at(graph, t_days: float) -> np.ndarray:
    """Directed in-universe citation in-degree per universe row at time t
    (#edges with dst=row and citing_days ≤ t)."""
    cd = graph["citing_days"]
    m = np.isfinite(cd) & (cd <= t_days)
    return np.bincount(graph["dst"][m], minlength=int(graph["n_works"])).astype(np.int64)


def window_conduit_mask(arr_days: np.ndarray, arr_in_scope: np.ndarray,
                        t0_days: float, t1_days: float) -> np.ndarray:
    """Conduit eligibility for a windowed flow-change field: in-scope arrivals
    DATED inside (t0, t1] only. Part of the window-split leakage guard (unit-
    pinned): a W2-dated arrival must never become a conduit node of the W1 field."""
    arr_days = np.asarray(arr_days, dtype=np.float64)
    return np.asarray(arr_in_scope, dtype=bool) & (arr_days > t0_days) & (arr_days <= t1_days)


def infrastructure_operationalization(
    g: np.ndarray, a: np.ndarray, pool_g_active: np.ndarray, pool_a: np.ndarray,
    g_quantile: float = INFRA_G_QUANTILE, a_quantile: float = INFRA_A_QUANTILE,
) -> dict:
    """The PRE-SPECIFIED infrastructure classification (module docstring).

    g, a: per-node citation in-degree growth over the window and abstracted-
    arrival fill. Thresholds come from POOL distributions only (outcome-blind
    w.r.t. any hub/control comparison): g_thr = q75 of g over citation-active
    scoped nodes, a_thr = median fill over all scoped nodes.
    """
    g = np.asarray(g, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    g_thr = float(np.quantile(np.asarray(pool_g_active, dtype=np.float64), g_quantile))
    a_thr = float(np.quantile(np.asarray(pool_a, dtype=np.float64), a_quantile))
    ir = np.log1p(g) - np.log1p(a)
    infra = (g >= g_thr) & (a <= a_thr)
    return {"IR": ir, "infra": infra, "g_thr": g_thr, "a_thr": a_thr}


def neighborhood_growth_concentration(
    idx: np.ndarray, Zb_s: np.ndarray, radii: np.ndarray, g_scoped: np.ndarray,
) -> dict:
    """Per node in `idx`: concentration of in-window in-degree growth over the
    scoped members of its own radius-neighborhood (incl. itself). Benchmark
    hub = growth concentrated on one paper (top-1 share → 1, n_eff → 1)."""
    idx = np.asarray(idx, dtype=np.int64)
    top1 = np.zeros(idx.size)
    neff = np.zeros(idx.size)
    nmem = np.zeros(idx.size, dtype=np.int64)
    gsum = np.zeros(idx.size)
    for j, i in enumerate(idx):
        d = np.linalg.norm(Zb_s - Zb_s[i], axis=1)
        mem = np.where(d <= radii[i])[0]
        nmem[j] = mem.size
        gg = g_scoped[mem].astype(np.float64)
        s = float(gg.sum())
        gsum[j] = s
        if s > 0:
            top1[j] = float(gg.max()) / s
            neff[j] = effective_number(gg)
    return {"top1_share": top1, "n_eff": neff, "n_members": nmem, "g_sum": gsum}


def degree_matched_controls(
    flag_idx: np.ndarray, deg_T_scoped: np.ndarray, excluded: np.ndarray,
    m_controls: int = M_CONTROLS, seed: int = 11,
) -> np.ndarray:
    """Per flagged node, the m nearest eligible scoped nodes in |Δ log1p in-deg
    at T| (hubness-at-T matched; outcome-blind, T-information only). Sharper
    control than the covariate matcher for a degree-driven statistic."""
    rng = np.random.default_rng(seed)
    ld = np.log1p(np.asarray(deg_T_scoped, dtype=np.float64))
    elig = np.where(~excluded)[0]
    out = np.full((flag_idx.size, m_controls), -1, dtype=np.int64)
    for j, i in enumerate(flag_idx):
        dd = np.abs(ld[elig] - ld[i]) + 1e-12 * rng.random(elig.size)
        out[j] = elig[np.argsort(dd, kind="stable")[:m_controls]]
    return out


def title_cue_stats(wid_list: list[str], titles: dict[str, str]) -> dict:
    """SECONDARY descriptive signal: fraction of titled works matching the
    benchmark/resource cue regex."""
    have = [w for w in wid_list if titles.get(w)]
    hits = [w for w in have if TITLE_CUE_RE.search(titles[w])]
    return {"n": len(wid_list), "n_titled": len(have), "n_cue": len(hits),
            "cue_frac": round(len(hits) / max(len(have), 1), 4)}


# ------------------------------------------------------------------ state build
def build_state(verbose: bool = True) -> dict:
    """Reconstruct the phase-3 objects deterministically from the frozen caches
    and ASSERT the banked numbers before any new measurement."""
    d, K_corpus, arr, grow, graph, models, meta = load_phase2()
    banked = load_banked()["task_t2_oa"]["phase3"]
    Z = d["Z"]
    fg_mask = d["fg_mask"]
    scoped_idx = d["scoped_idx"]
    scope_thr = float(d["scope_thr"])
    wids = d["wids"]
    corpus_store_rows = d["store_rows"]
    corpus_days = d["days"]
    vocab = d["vocab"]
    Zf = Z[fg_mask]
    Zb = Z[~fg_mask]
    Zb_s = Zb[scoped_idx]
    bg_rows_global = np.where(~fg_mask)[0]
    scoped_local_to_corpus = bg_rows_global[scoped_idx]
    scoped_global_rows = corpus_store_rows[scoped_local_to_corpus]
    K_scoped = np.asarray(K_corpus[~fg_mask][scoped_idx].todense(), dtype=bool)
    Z_arr = arr["Z"]
    arr_global_rows = arr["store_rows"].astype(np.int64)
    arr_days = arr["days"].astype(np.float64)
    arr_wids = arr["wids"]
    Z_grow = grow["Z"]
    t_days = float(meta["t_days"])
    end_days = float(meta["end_days"])
    assert abs(scope_thr - banked["config"]["scope_thr"]) < 1e-9
    if verbose:
        log(f"loaded: corpus {Z.shape[0]:,} (fg {Zf.shape[0]:,}, scoped bg {Zb_s.shape[0]:,}), "
            f"arrivals {Z_arr.shape[0]:,}, growth {Z_grow.shape[0]:,}")

    scorer = GapScorer(bg=Zb_s, probes=None, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    out = scorer.score(Zf)

    pool_radii = knn_scale(Zb_s, Zb_s, k=K_NN, self_rows=np.ones(Zb_s.shape[0], dtype=bool))
    pool_growth = count_within_radius(Zb_s, Z_grow, pool_radii).astype(np.float64)
    pool_C3 = np.column_stack([
        np.log(scorer.r_Bk), np.log(out["no_hab"]), np.log1p(pool_growth),
    ])
    pool_fill = fill_counts(Zb_s, pool_radii, Z_arr).astype(np.float64)

    # corridor electrodes (deterministic: KMeans seed 7, n_init 4 — phase-3 path)
    labels12, cents12, pairs12 = corridor_pairs(Zf, Zb_s, n_clusters=12, seed=7)
    W_cite_T = _cite_adjacency(graph, scoped_global_rows, t_days)
    vc_idx, n_used_c, vc_field = predict_voltage_citation(
        Zf, Zb_s, W_cite_T, labels12, cents12, pairs12)
    f_vc = pool_fill[vc_idx]
    assert round(float(f_vc.mean()), 3) == banked["per_detector"]["voltage_cite"]["fill"][
        "mean_fill_flagged"], "voltage_cite top-50 fill does not reproduce the banked run"
    vc_groups = greedy_location_groups(Zb_s[vc_idx], pool_radii[vc_idx])
    assert int(np.unique(vc_groups).size) == banked["per_detector"]["voltage_cite"]["fill"][
        "n_groups"], "voltage_cite location groups do not reproduce"
    if verbose:
        log(f"voltage_cite reproduced: {n_used_c} pairs, top-50 mean fill {f_vc.mean():.3f} "
            f"(banked 5.5), {np.unique(vc_groups).size} groups (banked 42)")

    # P3a full-window instrument (the banked OA win)
    r_arr_f = knn_radii(Z_arr, Zf, K_NN)
    arr_in_scope = r_arr_f <= scope_thr
    p3 = instrument_p3a_citation(
        graph, scoped_global_rows, arr_global_rows, arr_in_scope,
        t_days, end_days, labels12, cents12, pairs12, Zf, Zb_s)
    cfg = banked["config"]
    assert p3["n_pairs"] == cfg["p3a_pairs"]
    assert p3["n_arrival_nodes"] == cfg["p3a_arrival_nodes"]
    assert p3["n_edges_T"] == cfg["p3a_n_edges_T"]
    assert p3["n_edges_Tdelta"] == cfg["p3a_n_edges_Tdelta"]
    assert int(p3["peaks"].size) == banked["instrument_agreement"]["n_p3a_peaks"]
    pk = precision_vs_set(Zb_s[vc_idx], pool_radii[vc_idx], Zb_s[p3["peaks"]])
    assert round(pk, 4) == banked["per_detector"]["voltage_cite"]["precision_at_k"][
        "vs_p3a_peaks"], "voltage_cite P@50 vs P3a peaks does not reproduce"
    if verbose:
        log(f"P3a reproduced: {p3['n_pairs']} pairs, {p3['peaks'].size} peaks, edges "
            f"{p3['n_edges_T']:,} → {p3['n_edges_Tdelta']:,}; vc P@50 vs peaks {pk:.4f}")

    level_idx = predict_levelset(scorer, out)
    assert round(float(pool_fill[level_idx].mean()), 3) == banked["per_detector"]["levelset"][
        "fill"]["mean_fill_flagged"], "levelset top-50 fill does not reproduce"

    # in-universe citation in-degrees (directed, dated) for Q1 metadata
    deg_in_T = in_degrees_at(graph, t_days)
    deg_in_Td = in_degrees_at(graph, end_days)

    return {
        "Zf": Zf, "Zb_s": Zb_s, "K_scoped": K_scoped, "vocab": vocab,
        "wids": wids, "scoped_local_to_corpus": scoped_local_to_corpus,
        "scoped_global_rows": scoped_global_rows, "corpus_days": corpus_days,
        "scorer": scorer, "out": out,
        "pool_radii": pool_radii, "pool_growth": pool_growth, "pool_C3": pool_C3,
        "pool_fill": pool_fill,
        "labels12": labels12, "cents12": cents12, "pairs12": pairs12,
        "graph": graph, "W_cite_T": W_cite_T,
        "vc_idx": vc_idx, "vc_field": vc_field, "vc_groups": vc_groups,
        "p3a": p3, "level_idx": level_idx,
        "Z_arr": Z_arr, "arr_days": arr_days, "arr_wids": arr_wids,
        "arr_global_rows": arr_global_rows, "arr_in_scope": arr_in_scope,
        "t_days": t_days, "end_days": end_days,
        "deg_in_T": deg_in_T, "deg_in_Td": deg_in_Td,
        "banked": banked,
    }


def load_cited_by() -> np.ndarray:
    """Universe-level cited_by_count (2026 API snapshot; SECONDARY descriptor).
    Streamed once from the cached pages, then cached as npz."""
    if CITEDBY_PATH.exists():
        return np.load(CITEDBY_PATH)["cited_by"]
    log("streaming cached pages for cited_by_count (one-off)...")
    vals: list[int] = []
    for w in iter_works(SUB_KEY):
        if short_id(w.get("id")) is None:
            continue
        vals.append(int(w.get("cited_by_count") or 0))
    cited = np.asarray(vals, dtype=np.int64)
    np.savez_compressed(CITEDBY_PATH, cited_by=cited)
    log(f"cited_by cached: {cited.size:,} works")
    return cited


# ------------------------------------------------------------------ titles (justified fetch)
def scoped_wid(state: dict, scoped_node: int) -> str:
    return str(state["wids"][state["scoped_local_to_corpus"][scoped_node]])


def collect_wanted_ids(state: dict) -> dict:
    """All W-ids whose titles Q1/Q3 want: vc50 ∪ P3a peaks ∪ levelset50 ∪ the
    5 nearest controls per hub (both schemes) ∪ Q3 quadrant exemplars. Bounded
    ≤ ~800 ids by construction."""
    excluded = controls_exclusion(state, state["vc_idx"])
    ctr_cov, _ = matched_controls(
        state["Zb_s"][state["vc_idx"]], state["pool_radii"][state["vc_idx"]],
        state["pool_C3"][state["vc_idx"]], state["Zb_s"], state["pool_radii"],
        state["pool_C3"], excluded=excluded, m_controls=M_CONTROLS, seed=11)
    deg_T_scoped = state["deg_in_T"][state["scoped_global_rows"]]
    ctr_deg = degree_matched_controls(state["vc_idx"], deg_T_scoped, excluded)
    F, A, active = q3_fields(state)
    quad = q3_quadrants(F, A, active)
    want: set[int] = set()
    for s in (state["vc_idx"], state["p3a"]["peaks"], state["level_idx"],
              ctr_cov[:, :5].ravel(), ctr_deg[:, :5].ravel()):
        want.update(int(x) for x in np.asarray(s).ravel())
    for qname in ("HH", "HL", "LH"):
        want.update(int(x) for x in quad["exemplars"][qname])
    wid_list = sorted({scoped_wid(state, i) for i in want})
    return {"wids": wid_list, "n_nodes": len(want)}


def fetch_titles(wanted: list[str], batch: int = 50) -> dict[str, str]:
    """One-shot polite-pool fetch of display_name for specific W-ids (justified
    in the worklog: the phase-0 select list omitted display_name). Every batch
    response cached to cache/oa/titles/; reruns are offline."""
    TITLES_DIR.mkdir(parents=True, exist_ok=True)
    titles: dict[str, str] = {}
    if TITLES_MAP_PATH.exists():
        with gzip.open(TITLES_MAP_PATH, "rt", encoding="utf-8") as f:
            titles = json.load(f)
    missing = [w for w in wanted if w not in titles]
    if not missing:
        log(f"titles: all {len(wanted)} cached (offline)")
        return titles
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    n_calls = 0
    for i in range(0, len(missing), batch):
        ids = missing[i:i + batch]
        key = hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]
        bpath = TITLES_DIR / f"batch_{key}.json.gz"
        if bpath.exists():
            with gzip.open(bpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            params = {
                "filter": "ids.openalex:" + "|".join(ids),
                "select": "id,display_name", "per-page": str(batch),
                "mailto": MAILTO,
            }
            backoff = 1.0
            data = None
            for _ in range(8):
                try:
                    resp = session.get("https://api.openalex.org/works", params=params,
                                       timeout=60)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 64.0)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except requests.RequestException:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 64.0)
            if data is None:
                raise RuntimeError(f"title fetch failed for batch at offset {i}")
            with gzip.open(bpath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
            n_calls += 1
            time.sleep(0.05)
        for w in data.get("results", []):
            sid = short_id(w.get("id"))
            if sid:
                titles[sid] = w.get("display_name") or ""
    with gzip.open(TITLES_MAP_PATH, "wt", encoding="utf-8") as f:
        json.dump(titles, f)
    log(f"titles: {len(titles)} cached total ({n_calls} new polite-pool calls)")
    return titles


def load_titles() -> dict[str, str]:
    if TITLES_MAP_PATH.exists():
        with gzip.open(TITLES_MAP_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ------------------------------------------------------------------ Q1
def controls_exclusion(state: dict, flag_idx: np.ndarray) -> np.ndarray:
    """Phase-3 exclusion rule: flagged docs + every pool doc inside a flagged
    neighborhood."""
    Zb_s, pool_radii = state["Zb_s"], state["pool_radii"]
    excluded = np.zeros(Zb_s.shape[0], dtype=bool)
    excluded[flag_idx] = True
    for i in flag_idx:
        dd = np.linalg.norm(Zb_s - Zb_s[i], axis=1)
        excluded |= dd <= pool_radii[i]
    return excluded


def group_composition(name: str, idx: np.ndarray, g: np.ndarray, a: np.ndarray,
                      op: dict, conc: dict | None, age_y: np.ndarray,
                      deg_T: np.ndarray, cited_by: np.ndarray) -> dict:
    idx = np.asarray(idx, dtype=np.int64)
    return {
        "group": name, "n": int(idx.size),
        "infra_frac": round(float(op["infra"][idx].mean()), 4),
        "IR_median": round(float(np.median(op["IR"][idx])), 3),
        "g_median": float(np.median(g[idx])),
        "g_q90": float(np.quantile(g[idx], 0.9)),
        "fill_median": float(np.median(a[idx])),
        "deg_T_median": float(np.median(deg_T[idx])),
        "age_y_median": round(float(np.median(age_y[idx])), 2),
        "cited_by_2026_median": float(np.median(cited_by[idx])),
        "nbhd_top1_share_mean": (round(float(conc["top1_share"].mean()), 4)
                                 if conc is not None else None),
        "nbhd_g_neff_median": (round(float(np.median(conc["n_eff"])), 2)
                               if conc is not None else None),
    }


def boot_frac_diff(infra: np.ndarray, ir: np.ndarray, hub_idx: np.ndarray,
                   ctrl: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = 13) -> dict:
    """Clustered bootstrap (over hub location groups) for infra-fraction and
    median-IR differences hubs − their matched controls (rows of `ctrl`)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {gg: np.where(groups == gg)[0] for gg in uniq}

    def stats(rows: np.ndarray) -> tuple[float, float]:
        hub_rows = hub_idx[rows]
        ctrl_rows = ctrl[rows].ravel()
        df = float(infra[hub_rows].mean() - infra[ctrl_rows].mean())
        di = float(np.median(ir[hub_rows]) - np.median(ir[ctrl_rows]))
        return df, di

    point_f, point_i = stats(np.arange(hub_idx.size))
    bf, bi = [], []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([members[gg] for gg in gs])
        f_, i_ = stats(rows)
        bf.append(f_)
        bi.append(i_)
    bf, bi = np.asarray(bf), np.asarray(bi)
    return {
        "infra_frac_diff": round(point_f, 4),
        "infra_frac_diff_ci": [round(float(np.quantile(bf, q)), 4) for q in (0.025, 0.975)],
        "IR_median_diff": round(point_i, 4),
        "IR_median_diff_ci": [round(float(np.quantile(bi, q)), 4) for q in (0.025, 0.975)],
        "n_groups": int(uniq.size), "n_boot": int(n_boot),
    }


def run_q1(state: dict) -> dict:
    Zb_s = state["Zb_s"]
    sg = state["scoped_global_rows"]
    g_scoped = (state["deg_in_Td"] - state["deg_in_T"])[sg].astype(np.float64)
    deg_T_scoped = state["deg_in_T"][sg].astype(np.float64)
    deg_Td_scoped = state["deg_in_Td"][sg].astype(np.float64)
    a_scoped = state["pool_fill"]
    age_y = (state["t_days"] - state["corpus_days"][state["scoped_local_to_corpus"]]) / 365.25
    cited_by = load_cited_by()[sg].astype(np.float64)

    active = deg_Td_scoped > 0
    op = infrastructure_operationalization(
        g_scoped, a_scoped, pool_g_active=g_scoped[active], pool_a=a_scoped)
    log(f"operationalization thresholds: g_thr {op['g_thr']:.0f} "
        f"(q{INFRA_G_QUANTILE:.2f} over {int(active.sum()):,} citation-active), "
        f"a_thr {op['a_thr']:.0f} (median fill); pool infra rate "
        f"{float(op['infra'].mean()):.4f} (active-only {float(op['infra'][active].mean()):.4f})")

    vc_idx = state["vc_idx"]
    peaks = state["p3a"]["peaks"]
    level_idx = state["level_idx"]

    excluded = controls_exclusion(state, vc_idx)
    assert int(excluded.sum()) == state["banked"]["per_detector"]["voltage_cite"][
        "n_excluded_pool"], "voltage_cite exclusion mask does not reproduce"
    ctr_cov, cov_info = matched_controls(
        Zb_s[vc_idx], state["pool_radii"][vc_idx], state["pool_C3"][vc_idx],
        Zb_s, state["pool_radii"], state["pool_C3"],
        excluded=excluded, m_controls=M_CONTROLS, seed=11)
    assert round(float(a_scoped[ctr_cov].mean()), 3) == state["banked"]["per_detector"][
        "voltage_cite"]["fill"]["mean_fill_controls"], "cov-matched controls do not reproduce"
    ctr_deg = degree_matched_controls(vc_idx, deg_T_scoped, excluded)
    excluded_pk = controls_exclusion(state, peaks)
    ctr_cov_pk, _ = matched_controls(
        Zb_s[peaks], state["pool_radii"][peaks], state["pool_C3"][peaks],
        Zb_s, state["pool_radii"], state["pool_C3"],
        excluded=excluded_pk, m_controls=M_CONTROLS, seed=11)
    ctr_deg_pk = degree_matched_controls(peaks, deg_T_scoped, excluded_pk)
    log(f"controls: cov (n_relaxed {cov_info['n_relaxed']}/50), degree-matched "
        f"(median |Δlog1p degT| {np.median(np.abs(np.log1p(deg_T_scoped[ctr_deg]) - np.log1p(deg_T_scoped[vc_idx])[:, None])):.3f})")

    conc_vc = neighborhood_growth_concentration(vc_idx, Zb_s, state["pool_radii"], g_scoped)
    conc_pk = neighborhood_growth_concentration(peaks, Zb_s, state["pool_radii"], g_scoped)
    conc_cov = neighborhood_growth_concentration(
        np.unique(ctr_cov), Zb_s, state["pool_radii"], g_scoped)
    conc_deg = neighborhood_growth_concentration(
        np.unique(ctr_deg), Zb_s, state["pool_radii"], g_scoped)
    conc_lv = neighborhood_growth_concentration(level_idx, Zb_s, state["pool_radii"], g_scoped)

    groups = [
        group_composition("voltage_cite_top50", vc_idx, g_scoped, a_scoped, op, conc_vc,
                          age_y, deg_T_scoped, cited_by),
        group_composition("p3a_peaks", peaks, g_scoped, a_scoped, op, conc_pk,
                          age_y, deg_T_scoped, cited_by),
        group_composition("ctrl_cov_matched(unique)", np.unique(ctr_cov), g_scoped, a_scoped,
                          op, conc_cov, age_y, deg_T_scoped, cited_by),
        group_composition("ctrl_degree_matched(unique)", np.unique(ctr_deg), g_scoped,
                          a_scoped, op, conc_deg, age_y, deg_T_scoped, cited_by),
        group_composition("levelset_top50(contrast)", level_idx, g_scoped, a_scoped, op,
                          conc_lv, age_y, deg_T_scoped, cited_by),
        group_composition("pool_scoped", np.arange(Zb_s.shape[0]), g_scoped, a_scoped, op,
                          None, age_y, deg_T_scoped, cited_by),
        group_composition("pool_citation_active", np.where(active)[0], g_scoped, a_scoped,
                          op, None, age_y, deg_T_scoped, cited_by),
    ]
    for gr in groups:
        log("composition " + json.dumps(gr))

    diffs = {
        "vc50_vs_cov": boot_frac_diff(op["infra"], op["IR"], vc_idx, ctr_cov,
                                      state["vc_groups"]),
        "vc50_vs_deg": boot_frac_diff(op["infra"], op["IR"], vc_idx, ctr_deg,
                                      state["vc_groups"]),
        "peaks_vs_cov": boot_frac_diff(op["infra"], op["IR"], peaks, ctr_cov_pk,
                                       greedy_location_groups(Zb_s[peaks],
                                                              state["pool_radii"][peaks])),
        "peaks_vs_deg": boot_frac_diff(op["infra"], op["IR"], peaks, ctr_deg_pk,
                                       greedy_location_groups(Zb_s[peaks],
                                                              state["pool_radii"][peaks])),
    }
    for k, v in diffs.items():
        log(f"diff {k}: {json.dumps(v)}")

    # set-level in-degree-growth concentration (is the top-50 a few benchmarks?)
    set_conc = {}
    for nm, ii in (("voltage_cite_top50", vc_idx), ("p3a_peaks", peaks),
                   ("levelset_top50", level_idx)):
        gg = g_scoped[ii]
        s = float(gg.sum())
        set_conc[nm] = {
            "g_total": int(s), "top1_share": round(float(gg.max()) / max(s, 1e-9), 4),
            "top5_share": round(float(np.sort(gg)[::-1][:5].sum()) / max(s, 1e-9), 4),
            "n_eff": round(effective_number(gg), 2) if s > 0 else 0.0,
        }
    log("set-level growth concentration: " + json.dumps(set_conc))

    # titles (SECONDARY) + transcript table
    titles = load_titles()
    cue = {
        "voltage_cite_top50": title_cue_stats([scoped_wid(state, i) for i in vc_idx], titles),
        "p3a_peaks": title_cue_stats([scoped_wid(state, i) for i in peaks], titles),
        "ctrl_cov_top5each": title_cue_stats(
            [scoped_wid(state, i) for i in np.unique(ctr_cov[:, :5])], titles),
        "ctrl_deg_top5each": title_cue_stats(
            [scoped_wid(state, i) for i in np.unique(ctr_deg[:, :5])], titles),
        "levelset_top50": title_cue_stats([scoped_wid(state, i) for i in level_idx], titles),
    }
    log("title cues (secondary): " + json.dumps(cue))

    order = np.argsort(state["vc_field"][vc_idx])[::-1]
    pos_of = {int(v): j for j, v in enumerate(vc_idx)}
    table = []
    for i in vc_idx[order][:12]:
        w = scoped_wid(state, int(i))
        table.append({
            "wid": w, "title": (titles.get(w) or "")[:110],
            "age_y": round(float(age_y[i]), 1), "deg_T": int(deg_T_scoped[i]),
            "g_window": int(g_scoped[i]), "cited_by_2026": int(cited_by[i]),
            "fill": int(a_scoped[i]), "IR": round(float(op["IR"][i]), 2),
            "nbhd_top1": round(float(conc_vc["top1_share"][pos_of[int(i)]]), 3),
            "infra": bool(op["infra"][i]),
        })
    for row in table:
        log("hub " + json.dumps(row))

    terms = {
        "voltage_cite_top50": region_terms(state["K_scoped"], vc_idx, state["vocab"], 15),
        "p3a_peaks": region_terms(state["K_scoped"], peaks, state["vocab"], 15),
    }
    log("region terms: " + json.dumps(terms))

    return {
        "operationalization": {
            "g": "in-universe citation in-degree growth over (T, T+Δ] (dated graph)",
            "a": "abstracted-arrival fill in own 10-NN radius (phase-3 outcome)",
            "IR": "log1p(g) - log1p(a)",
            "rule": f"INFRA := g >= q{INFRA_G_QUANTILE} (citation-active pool) "
                    f"AND a <= median(pool)",
            "g_thr": op["g_thr"], "a_thr": op["a_thr"],
            "title_cues": "SECONDARY descriptive only; regex on display_name",
        },
        "composition": groups, "diffs": diffs, "set_concentration": set_conc,
        "title_cues": cue, "hub_table": table, "region_terms": terms,
        "controls": {"cov_info": cov_info,
                     "n_unique_cov": int(np.unique(ctr_cov).size),
                     "n_unique_deg": int(np.unique(ctr_deg).size)},
    }


# ------------------------------------------------------------------ Q2
def windowed_flow_change(state: dict, t0_days: float, t1_days: float) -> dict:
    """Flow-share-change field for the window (t0, t1]: graphs thresholded at
    citing_days ≤ t0 / ≤ t1; conduits = in-scope arrivals DATED in (t0, t1]."""
    mask = window_conduit_mask(state["arr_days"], state["arr_in_scope"], t0_days, t1_days)
    return instrument_p3a_citation(
        state["graph"], state["scoped_global_rows"], state["arr_global_rows"], mask,
        t0_days, t1_days, state["labels12"], state["cents12"], state["pairs12"],
        state["Zf"], state["Zb_s"])


def fill_run(state: dict, flag_idx: np.ndarray, fill_outcome: np.ndarray,
             pool_C: np.ndarray, calipers: tuple[float, ...], seed_ctrl: int = 11,
             seed_boot: int = 13) -> dict:
    """One matched-null fill-ratio measurement (phase-3 machinery verbatim,
    parameterized by outcome fill vector + covariate matrix)."""
    Zb_s, pool_radii = state["Zb_s"], state["pool_radii"]
    excluded = controls_exclusion(state, flag_idx)
    ctr, info = matched_controls(
        Zb_s[flag_idx], pool_radii[flag_idx], pool_C[flag_idx],
        Zb_s, pool_radii, pool_C, excluded=excluded,
        m_controls=M_CONTROLS, calipers=calipers, seed=seed_ctrl)
    groups = greedy_location_groups(Zb_s[flag_idx], pool_radii[flag_idx])
    rr = fill_rate_ratio(fill_outcome[flag_idx], fill_outcome[ctr],
                         n_boot=N_BOOT, seed=seed_boot, groups=groups)
    sd = pool_C.std(axis=0) + 1e-12
    smd = np.abs(pool_C[flag_idx] - pool_C[ctr].mean(axis=1)) / sd
    rr["balance_smd"] = [round(float(s), 3) for s in smd.mean(axis=0)]
    rr["n_relaxed"] = info["n_relaxed"]
    return rr


def run_q2(state: dict) -> dict:
    t0, t1, t2 = state["t_days"], W1_END_DAYS, state["end_days"]
    arr_days = state["arr_days"]
    w1_arr = (arr_days > t0) & (arr_days <= t1)
    w2_arr = (arr_days > t1) & (arr_days <= t2)
    Zb_s, pool_radii = state["Zb_s"], state["pool_radii"]
    fill_w1 = fill_counts(Zb_s, pool_radii, state["Z_arr"][w1_arr]).astype(np.float64)
    fill_w2 = fill_counts(Zb_s, pool_radii, state["Z_arr"][w2_arr]).astype(np.float64)
    log(f"windows: W1 arrivals {int(w1_arr.sum()):,} (in-scope conduits "
        f"{int(window_conduit_mask(arr_days, state['arr_in_scope'], t0, t1).sum()):,}), "
        f"W2 arrivals {int(w2_arr.sum()):,}; pool fill medians "
        f"W1 {np.median(fill_w1):.0f} / W2 {np.median(fill_w2):.0f}")

    p3_w1 = windowed_flow_change(state, t0, t1)
    fld = p3_w1["field"]
    n_pos = int((fld > 0).sum())
    w1_top = np.argsort(fld)[::-1][:TOP_K]
    w1_peaks = p3_w1["peaks"]
    log(f"W1 flow-change field: {p3_w1['n_pairs']} pairs, edges {p3_w1['n_edges_T']:,} → "
        f"{p3_w1['n_edges_Tdelta']:,}, conduits {p3_w1['n_arrival_nodes']:,}, "
        f"{n_pos:,} positive-field nodes")

    # field-stability descriptives vs the full-window instrument + predictor
    desc = {
        "jaccard_w1top50_vs_vc50": round(jaccard(w1_top, state["vc_idx"]), 4),
        "jaccard_w1top50_vs_fullpeaks": round(jaccard(w1_top, state["p3a"]["peaks"]), 4),
        "p_at_50_w1top50_vs_fullpeaks": round(precision_vs_set(
            Zb_s[w1_top], pool_radii[w1_top], Zb_s[state["p3a"]["peaks"]]), 4),
        "spearman_w1field_fullfield": None,
    }
    from scipy.stats import spearmanr

    el = (fld > 0) | (state["p3a"]["field"] > 0)
    desc["spearman_w1field_fullfield"] = round(
        float(spearmanr(fld[el], state["p3a"]["field"][el]).statistic), 4)
    log("W1-vs-full-field: " + json.dumps(desc))

    pool_C3 = state["pool_C3"]
    pool_C4 = np.column_stack([pool_C3, np.log1p(fill_w1)])

    runs: dict[str, dict] = {}
    # ---- NULL CALIBRATION FIRST (random locations through each machinery variant)
    for tag, C, cal, fill in (
        ("null_w2_4cov", pool_C4, CALIPERS_4, fill_w2),
        ("null_w2_3cov", pool_C3, (0.2, 0.5, 0.5), fill_w2),
        ("null_w1_3cov", pool_C3, (0.2, 0.5, 0.5), fill_w1),
    ):
        seeds = (17, 18, 19) if tag == "null_w2_4cov" else (17,)
        for s in seeds:
            ridx = np.random.default_rng(s).choice(Zb_s.shape[0], size=TOP_K, replace=False)
            rr = fill_run(state, ridx, fill, C, cal)
            runs[f"{tag}_seed{s}"] = rr
            log(f"{tag} seed{s}: ratio {rr['ratio']} clustered "
                f"[{rr['ci_lo_clustered']}, {rr['ci_hi_clustered']}] "
                f"(balance {rr['balance_smd']}, relaxed {rr['n_relaxed']})")
            ok = rr["ci_lo_clustered"] <= 1.0 <= rr["ci_hi_clustered"]
            if not ok:
                log(f"WARNING: machinery null off 1.0 for {tag} seed{s}")

    # ---- the measurements
    for tag, idx, C, cal, fill in (
        ("w1top50_to_W2_4cov_PRIMARY", w1_top, pool_C4, CALIPERS_4, fill_w2),
        ("w1top50_to_W2_3cov", w1_top, pool_C3, (0.2, 0.5, 0.5), fill_w2),
        ("w1top50_to_W1_samewindow", w1_top, pool_C3, (0.2, 0.5, 0.5), fill_w1),
        ("w1peaks78_to_W2_4cov_sens", w1_peaks, pool_C4, CALIPERS_4, fill_w2),
    ):
        rr = fill_run(state, idx, fill, C, cal)
        runs[tag] = rr
        log(f"{tag}: ratio {rr['ratio']} clustered "
            f"[{rr['ci_lo_clustered']}, {rr['ci_hi_clustered']}] n_groups {rr['n_groups']} "
            f"(flagged {rr['mean_fill_flagged']}, null {rr['mean_fill_controls']}; "
            f"balance {rr['balance_smd']})")

    prim = runs["w1top50_to_W2_4cov_PRIMARY"]
    verdict = ("CONFIRMED (leading indicator)" if prim["ci_lo_clustered"] > 1.0
               else "INVERTED (reroute peaks under-fill later too)"
               if prim["ci_hi_clustered"] < 1.0
               else "REFUTED as leading indicator (CI spans 1)")
    log(f"Q2 verdict (pre-specified rule on the 4-cov primary): {verdict}")
    return {
        "windows": {"W1": "(2015-01-01, 2017-01-01]", "W2": "(2017-01-01, 2019-01-01]",
                    "n_w1_arrivals": int(w1_arr.sum()), "n_w2_arrivals": int(w2_arr.sum()),
                    "n_w1_conduits": p3_w1["n_arrival_nodes"],
                    "w1_edges": [p3_w1["n_edges_T"], p3_w1["n_edges_Tdelta"]],
                    "n_positive_field": n_pos},
        "field_stability": desc, "runs": runs, "verdict": verdict,
        "rule": "CONFIRMED iff clustered CI_lo > 1 on w1top50_to_W2_4cov_PRIMARY; "
                "INVERTED iff CI_hi < 1; else REFUTED as leading indicator",
    }


# ------------------------------------------------------------------ Q3
def q3_fields(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two fields over the same scoped candidates + the citation-active mask
    (nodes with any undirected degree in the T+Δ instrument graph)."""
    F = state["p3a"]["field"].astype(np.float64)
    A = state["pool_fill"].astype(np.float64)
    # citation-active: same node set as the instrument (scoped ∪ conduits), W1 graph
    sg = state["scoped_global_rows"]
    mask = window_conduit_mask(state["arr_days"], state["arr_in_scope"],
                               state["t_days"], state["end_days"])
    conduits = state["arr_global_rows"][mask]
    nodes = np.concatenate([sg, conduits])
    W1 = _cite_adjacency(state["graph"], nodes, state["end_days"])
    deg1 = np.asarray(W1.sum(axis=1)).ravel()[: sg.size]
    return F, A, deg1 > 0


def q3_quadrants(F: np.ndarray, A: np.ndarray, active: np.ndarray,
                 q: float = 0.90, n_ex: int = 8) -> dict:
    """2x2 high/low decomposition within citation-active scoped nodes; exemplars
    by quadrant-corner extremity."""
    act = np.where(active)[0]
    f_thr = float(np.quantile(F[act], q))
    a_thr = float(np.quantile(A[act], q))
    hiF = F >= f_thr
    hiA = A >= a_thr
    quads = {
        "HH": act[(hiF & hiA)[act]], "HL": act[(hiF & ~hiA)[act]],
        "LH": act[(~hiF & hiA)[act]], "LL": act[(~hiF & ~hiA)[act]],
    }
    rF = (np.argsort(np.argsort(F)) + 1) / F.size  # rank-normalized for corner scores
    rA = (np.argsort(np.argsort(A)) + 1) / A.size
    corner = {"HH": rF + rA, "HL": rF - rA, "LH": rA - rF}
    exemplars = {}
    for k in ("HH", "HL", "LH"):
        m = quads[k]
        exemplars[k] = m[np.argsort(corner[k][m])[::-1][:n_ex]] if m.size else m
    return {"f_thr": f_thr, "a_thr": a_thr, "q": q,
            "counts": {k: int(v.size) for k, v in quads.items()},
            "quads": quads, "exemplars": exemplars}


def run_q3(state: dict) -> dict:
    from scipy.stats import spearmanr

    F, A, active = q3_fields(state)
    act = np.where(active)[0]
    # covariate-residualized arrival field (excess arrivals given density/dist/growth)
    X = np.column_stack([np.ones(A.size), state["pool_C3"]])
    beta, *_ = np.linalg.lstsq(X, np.log1p(A), rcond=None)
    A_resid = np.log1p(A) - X @ beta

    cors = {
        "spearman_F_A_all": round(float(spearmanr(F, A).statistic), 4),
        "spearman_F_A_active": round(float(spearmanr(F[act], A[act]).statistic), 4),
        "spearman_F_Aresid_all": round(float(spearmanr(F, A_resid).statistic), 4),
        "spearman_F_Aresid_active": round(float(spearmanr(F[act], A_resid[act]).statistic), 4),
        "n_active": int(act.size), "n_scoped": int(F.size),
    }
    log("correlations: " + json.dumps(cors))

    k50_F = np.argsort(F)[::-1][:TOP_K]
    k50_A = np.argsort(A)[::-1][:TOP_K]
    k1pc = max(int(np.ceil(0.01 * F.size)), 10)
    ov = {
        "jaccard_top50": round(jaccard(k50_F, k50_A), 4),
        "jaccard_top1pc": round(jaccard(np.argsort(F)[::-1][:k1pc],
                                        np.argsort(A)[::-1][:k1pc]), 4),
        "jaccard_top1pc_resid": round(jaccard(np.argsort(F)[::-1][:k1pc],
                                              np.argsort(A_resid)[::-1][:k1pc]), 4),
        "k_1pc": k1pc,
    }
    log("top-k overlap: " + json.dumps(ov))

    quad = q3_quadrants(F, A, active)
    log(f"2x2 within citation-active (q={quad['q']}, f_thr {quad['f_thr']:.4g}, "
        f"a_thr {quad['a_thr']:.0f}): counts {json.dumps(quad['counts'])}")
    titles = load_titles()
    quad_out = {}
    for k in ("HH", "HL", "LH"):
        m = quad["quads"][k]
        ex = quad["exemplars"][k]
        rows = []
        for i in ex[:5]:
            w = scoped_wid(state, int(i))
            rows.append({"wid": w, "title": (titles.get(w) or "")[:100],
                         "F": round(float(F[i]), 4), "fill": int(A[i])})
        quad_out[k] = {
            "n": int(m.size),
            "terms": region_terms(state["K_scoped"], m, state["vocab"], 12),
            "exemplars": rows,
        }
        log(f"{k} (n={m.size}): {', '.join(quad_out[k]['terms'])}")
        for r in rows:
            log(f"  {k} ex: " + json.dumps(r))

    return {"correlations": cors, "overlap": ov,
            "quadrants": {"thresholds": {"f_thr": quad["f_thr"], "a_thr": quad["a_thr"],
                                         "quantile": quad["q"]},
                          "counts": quad["counts"], "detail": quad_out}}


# ------------------------------------------------------------------ main
def bank(sub: str, payload: dict) -> None:
    banked = load_banked()
    row = banked.get("task_oa_dissoc", {})
    row[sub] = _py(payload)
    update_results("task_oa_dissoc", row)
    log(f"banked results.json:task_oa_dissoc.{sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-titles", action="store_true")
    ap.add_argument("--fetch-titles", action="store_true")
    ap.add_argument("--q1", action="store_true")
    ap.add_argument("--q2", action="store_true")
    ap.add_argument("--q3", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.fetch_titles:
        wanted = json.loads(WANTED_IDS_PATH.read_text())["wids"]
        fetch_titles(wanted)
        return

    state = build_state()
    if args.prep_titles:
        want = collect_wanted_ids(state)
        TITLES_DIR.mkdir(parents=True, exist_ok=True)
        WANTED_IDS_PATH.write_text(json.dumps(want))
        log(f"wanted ids written: {len(want['wids'])} W-ids ({want['n_nodes']} nodes) "
            f"-> {WANTED_IDS_PATH}")
        return
    if args.q1:
        res = run_q1(state)
        if not args.dry:
            bank("q1", res)
    if args.q2:
        res = run_q2(state)
        if not args.dry:
            bank("q2", res)
    if args.q3:
        res = run_q3(state)
        if not args.dry:
            bank("q3", res)
    log("done")


if __name__ == "__main__":
    main()
