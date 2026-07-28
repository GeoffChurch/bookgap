"""Track 2 OA phase — Phase 0 (acquisition + arrival-density gate) + Phase 1
(dates already exact; build the dated citation graph, report T / T+Δ stats).

Mirrors run_t2_phase01.py (wiki) with the OpenAlex data layer swapped in.

Stages (each cached; offline after the pages are on disk):
  --stage fetch    cursor-page the T10181 universe (pub 2008-2019) to cache/oa/
  --stage gate     works table + dated graph + arrival-density table over a
                   (T, Δ) grid; PRE-COMMIT T/Δ on COUNTS only; gate >=150 in-scope
  --stage graph    build + report the dated citation graph at T and T+Δ (Phase 1)

GATE (locked): >=150 in-scope (T, T+Δ] arrivals at the pre-committed (T,Δ), else
widen the universe to T10181+T10028+T10201 and re-run (record the widening).

Context hygiene: abstracts decoded internally; prints counts/aggregates only.
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

from artifacts_scripts.gap_detection.oa_corpus import (
    CACHE_DIR,
    FALLBACK_TOPICS,
    UNIVERSE_TOPIC,
    build_dated_graph,
    days_to_iso,
    decode_for_rows,
    fetch_universe,
    graph_stats,
    load_works_table,
    year_to_days,
)

_T0 = time.monotonic()
K_NN = 10
SCOPE_Q = 0.25
MIN_ABSTRACT_CHARS = 200
T_DAYS = year_to_days(2015)  # T = 2015-01-01 (LOCKED)
GATE_MIN_IN_SCOPE = 150


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def topics_for(universe: str) -> tuple[str, ...]:
    return FALLBACK_TOPICS if universe == "fallback" else (UNIVERSE_TOPIC,)


def stage_fetch(args) -> None:
    topics = topics_for(args.universe)
    sub_key = fetch_universe(topics=topics, log=log, max_pages=args.max_pages)
    n = 0
    from artifacts_scripts.gap_detection.oa_corpus import iter_works

    for _ in iter_works(sub_key):
        n += 1
    log(f"fetch complete: {n:,} works cached under key '{sub_key}'")


def _scope_quantile_threshold(Z_pre: np.ndarray, fg_mask_pre: np.ndarray) -> float | None:
    """q0.25 of pre-T bg's 10-NN-to-fg distance (the scope rule, T geometry)."""
    from artifacts_scripts.gap_detection.detectors import knn_radii

    Zf = Z_pre[fg_mask_pre]
    Zb = Z_pre[~fg_mask_pre]
    if Zf.shape[0] < 50 or Zb.shape[0] < 200:
        return None
    r_bg = knn_radii(Zb, Zf, K_NN)
    return float(np.quantile(r_bg, SCOPE_Q))


def _provisional_embedding(texts: list[str], seed: int = 42):
    """A ≤T provisional TF-IDF/LSA-128 used ONLY for the gate's in-scope counts and
    the gate's provisional fg discovery. (Phase 2 refits the production T-corpus;
    this provisional fit is never used as a predictor or outcome — it exists so the
    gate's 'in-scope' definition matches the eventual scope geometry.)"""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        stop_words="english", lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=10, max_df=0.3, max_features=60_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    nnz = np.asarray(X.getnnz(axis=1)).ravel()
    svd = TruncatedSVD(n_components=min(128, X.shape[1] - 1), random_state=seed)
    Z = normalize(svd.fit_transform(X).astype(np.float32)).astype(np.float32)
    return vec, svd, Z, nnz


def stage_gate(args) -> None:
    topics = topics_for(args.universe)
    sub_key = "primary_topic.id:" + "|".join(topics)
    tbl = load_works_table(sub_key, log=log)
    wid, days = tbl["wid"], tbl["days"]
    abs_chars = tbl["abstract_chars"]
    english_ok = tbl["english_ok"]
    has_date = np.isfinite(days)
    # eligibility = dated + abstract >=200ch + English (the ~5% non-English minority
    # forms a degenerate language cluster the coherence-max fg selector would grab).
    elig_chars = (abs_chars >= MIN_ABSTRACT_CHARS) & english_ok

    cov = float((abs_chars > 0).mean())
    cov200 = float((abs_chars >= MIN_ABSTRACT_CHARS).mean())
    cov_en = float(elig_chars.mean())
    n_nonen = int(((~english_ok) & (abs_chars >= MIN_ABSTRACT_CHARS)).sum())
    log(f"abstract coverage: any {cov:.3f}, >={MIN_ABSTRACT_CHARS}ch {cov200:.3f}, "
        f"+English {cov_en:.3f} ({n_nonen:,} non-English ≥200ch dropped)")

    # ----- dated citation graph (Phase 1 object; also informs the gate report)
    g = build_dated_graph(sub_key, tbl["row_of"], wid, days, log=log)

    # ----- provisional ≤T embedding for the gate's in-scope rule + fg discovery
    pre_mask = has_date & (days <= T_DAYS) & elig_chars
    pre_rows = np.where(pre_mask)[0]
    log(f"pre-T (<=2015) eligible (>= {MIN_ABSTRACT_CHARS}ch): {pre_rows.size:,}")
    # subsample pre-T pool for the provisional fit if very large (kept seeded)
    rng = np.random.default_rng(42)
    fit_rows = pre_rows
    if pre_rows.size > args.gate_fit_cap:
        fit_rows = np.sort(rng.choice(pre_rows, size=args.gate_fit_cap, replace=False))
    pre_texts = decode_for_rows(sub_key, fit_rows, wid)
    vec, svd, Z_pre, nnz_pre = _provisional_embedding(pre_texts)
    keep = nnz_pre >= 5
    Z_pre, fit_rows = Z_pre[keep], fit_rows[keep]
    log(f"provisional <=T embedding: {Z_pre.shape[0]:,} docs (nnz>=5), "
        f"EVR {float(svd.explained_variance_ratio_.sum()):.3f}")

    # provisional fg = discovered KMeans cluster (size 1-3k, most coherent)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors

    km = MiniBatchKMeans(n_clusters=40, random_state=42, n_init=5, batch_size=4096)
    labels = km.fit_predict(Z_pre)
    sizes = np.bincount(labels, minlength=40)
    cands = [c for c in range(40) if 1_000 <= sizes[c] <= 3_000]
    if not cands:
        cands = [int(np.argmin(np.abs(sizes - 2_000)))]
    best = max(
        cands,
        key=lambda c: float(np.mean(
            Z_pre[labels == c] @ km.cluster_centers_[c]
            / np.linalg.norm(km.cluster_centers_[c]))),
    )
    fg_mask_pre = labels == best
    n_fg_pre = int(fg_mask_pre.sum())
    log(f"provisional fg cluster {best}: size {n_fg_pre} (of {Z_pre.shape[0]:,} pre-T)")

    scope_thr = _scope_quantile_threshold(Z_pre, fg_mask_pre)
    log(f"provisional scope thr (q{SCOPE_Q}): {scope_thr}")

    # ----- frozen-transform embed arrivals for each candidate window, count in-scope
    from artifacts_scripts.gap_detection.detectors import knn_radii
    from sklearn.preprocessing import normalize

    Zf_pre = Z_pre[fg_mask_pre]

    def embed_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        texts = decode_for_rows(sub_key, rows, wid)
        X = vec.transform(texts)
        nz = np.asarray(X.getnnz(axis=1)).ravel()
        Z = normalize(svd.transform(X).astype(np.float32)).astype(np.float32)
        ok = nz >= 5
        return Z[ok], rows[ok]

    rows_table = []
    for ty in args.t_years:
        t_days = year_to_days(ty)
        n_pre_elig = int((has_date & (days <= t_days) & elig_chars).sum())
        # fg at T uses the provisional cluster only at the locked T; for other Ts
        # report pre-T eligible + arrivals (the pre-commit is on counts at T=2015).
        for dy in args.deltas:
            e_days = year_to_days(ty + dy)
            win = has_date & (days > t_days) & (days <= e_days) & elig_chars
            arr_rows = np.where(win)[0]
            n_arr = int(arr_rows.size)
            n_in_scope = None
            if scope_thr is not None and ty == 2015 and n_arr:
                # in-scope only computed at the LOCKED T (provisional fg is at T=2015)
                Z_arr, _ = embed_rows(arr_rows)
                r_arr = knn_radii(Z_arr, Zf_pre, K_NN)
                n_in_scope = int((r_arr <= scope_thr).sum())
            rows_table.append({
                "T": f"{ty}-01-01", "delta_years": dy,
                "window_end": days_to_iso(e_days),
                "n_pre_eligible": n_pre_elig,
                "n_arrivals_eligible": n_arr,
                "n_arrivals_in_scope": n_in_scope,
                "n_fg_preT_provisional": n_fg_pre if ty == 2015 else None,
                "scope_thr": scope_thr if ty == 2015 else None,
            })
            log(f"T={ty}-01 d={dy}y: pre={n_pre_elig:,} arr={n_arr:,} "
                f"in_scope={n_in_scope}")

    # ----- gate decision at the LOCKED (T=2015, Δ=4y)
    locked = next(r for r in rows_table if r["T"] == "2015-01-01" and r["delta_years"] == 4)
    in_scope = locked["n_arrivals_in_scope"]
    passed = in_scope is not None and in_scope >= GATE_MIN_IN_SCOPE
    log(f"GATE @ T=2015-01-01 Δ=4y: in_scope={in_scope} "
        f"(>= {GATE_MIN_IN_SCOPE}? {'PASS' if passed else 'FAIL'})")

    # ----- dated-graph stats at T and T+Δ (Phase 1 deliverable)
    dated_T = has_date & (days <= T_DAYS)
    dated_E = has_date & (days <= year_to_days(2019))
    stats_T = graph_stats(g["at"](T_DAYS), dated_T)
    stats_E = graph_stats(g["at"](year_to_days(2019)), dated_E)
    log(f"T-graph: {stats_T}")
    log(f"T+Δ-graph: {stats_E}")

    choice = {
        "universe": args.universe,
        "topics": list(topics),
        "T": "2015-01-01", "delta_years": 4,
        "window": "(2015-01-01, 2019-01-01]",
        "n_arrivals_in_scope": in_scope,
        "gate_min_in_scope": GATE_MIN_IN_SCOPE,
        "gate_passed": bool(passed),
        "provisional_fg_cluster_size": n_fg_pre,
        "scope_thr_provisional": scope_thr,
        "rule": "T/Δ pre-committed on COUNTS only (locked by brief); gate is the "
                ">=150 in-scope arrivals check at (2015-01-01, +4y].",
    }
    out = {
        "abstract_coverage_any": cov,
        "abstract_coverage_ge200": cov200,
        "abstract_coverage_ge200_english": cov_en,
        "n_nonenglish_ge200_dropped": n_nonen,
        "n_works_universe": int(wid.size),
        "n_dated": int(has_date.sum()),
        "n_edges_intra_universe": int(g["src"].size),
        "n_works_with_inuniverse_ref": int(g["n_with_refs"]),
        "graph_T": stats_T,
        "graph_T_plus_delta": stats_E,
        "table": rows_table,
        "choice": choice,
    }
    (CACHE_DIR / "t2_oa_gate_table.json").write_text(json.dumps(out, indent=1))
    (CACHE_DIR / "t2_oa_choice.json").write_text(json.dumps(choice, indent=1))
    log(f"wrote t2_oa_gate_table.json + t2_oa_choice.json (gate {'PASS' if passed else 'FAIL'})")

    # bank the Phase-0 summary under task_t2_oa (mirrors the wiki run's results layout)
    from artifacts_scripts.gap_detection.run_eval import _py
    from artifacts_scripts.gap_detection.run_followups import load_banked, update_results

    banked = load_banked()
    row = banked.get("task_t2_oa", {})
    row["phase0_gate"] = _py(out)
    update_results("task_t2_oa", row)
    log("banked phase0_gate under task_t2_oa")
    if not passed and args.universe != "fallback":
        log("GATE FAILED on the locked universe — rerun --stage fetch/gate "
            "--universe fallback (T10181+T10028+T10201) per the brief's fallback.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "gate", "graph"])
    ap.add_argument("--universe", default="locked", choices=["locked", "fallback"])
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--gate-fit-cap", type=int, default=80_000)
    ap.add_argument("--t-years", type=int, nargs="+", default=[2013, 2014, 2015, 2016])
    ap.add_argument("--deltas", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    {"fetch": stage_fetch, "gate": stage_gate, "graph": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
