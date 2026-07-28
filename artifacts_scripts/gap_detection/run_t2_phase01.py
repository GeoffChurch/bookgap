"""Track 2 wiki phase — Phase 0 (arrival-density gate) + Phase 1 (creation dates).

Stages (each cached; reruns are offline once the API responses are on disk):
  --stage rough    ~240 stratified exact dates -> rough isotonic curve (gate input)
  --stage gate     density table over candidate (T, Delta); writes gate rows
  --stage full     full stratified calibration (~2000) + boundary densification,
                   held-out validation, wrinkle rates, final proxy; assigns dates

Outputs: artifacts/gap_detection_eval/cache/t2_*.json|npz, results.json rows
under task_t2_wiki (via run_followups.update_results merging).
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

from artifacts_scripts.gap_detection.data import CACHE_DIR
from artifacts_scripts.gap_detection.wiki_temporal import (
    fetch_creation_dates,
    fit_id_date_proxy,
    invert_date_to_id,
    load_store_ids,
    proxy_error_report,
    ts_to_days,
    days_to_iso,
    year_to_days,
)

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def stratified_ids(ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n ids stratified uniformly over the store's id RANKS (equal-count strata)."""
    rng = np.random.default_rng(seed)
    order = np.sort(ids)
    edges = np.linspace(0, order.size - 1, n + 1).astype(np.int64)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        out.append(int(order[int(rng.integers(lo, hi))]))
    return np.unique(np.asarray(out, dtype=np.int64))


def collect_dates(recs: dict[int, dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """(ids, days) for records with a timestamp + wrinkle/qc counts."""
    ids, days = [], []
    qc = {"n": len(recs), "missing": 0, "redirect": 0, "no_ts": 0, "error": 0}
    for pid, r in sorted(recs.items()):
        if r.get("error"):
            qc["error"] += 1
            continue
        if r.get("missing"):
            qc["missing"] += 1
            continue
        if r.get("redirect"):
            qc["redirect"] += 1  # counted but still usable if ts present
        if not r.get("ts"):
            qc["no_ts"] += 1
            continue
        ids.append(pid)
        days.append(ts_to_days(r["ts"]))
    return np.asarray(ids, dtype=np.int64), np.asarray(days), qc


def stage_rough(args) -> None:
    ids, _ = load_store_ids()
    sample = stratified_ids(ids, args.n_rough, seed=420)
    log(f"rough calibration sample: {sample.size} ids")
    recs = fetch_creation_dates(sample.tolist(), log=log)
    cal_ids, cal_days, qc = collect_dates(recs)
    log(f"usable {cal_ids.size}/{sample.size}; qc={qc}")
    # monotonicity diagnostic: fraction of adjacent (id-sorted) pairs with date inversion
    inv = float((np.diff(cal_days) < 0).mean())
    log(f"adjacent-pair date-inversion rate (wrinkle proxy): {inv:.3f}")
    out = {
        "sample_ids": cal_ids.tolist(),
        "days": cal_days.tolist(),
        "qc": qc,
        "inversion_rate": inv,
    }
    (CACHE_DIR / "t2_rough_calibration.json").write_text(json.dumps(out))
    predict = fit_id_date_proxy(cal_ids, cal_days)
    for y in (2002, 2003, 2004, 2005):
        thr = invert_date_to_id(predict, year_to_days(y), int(ids.min()), int(ids.max()) + 1)
        log(f"  id threshold for {y}-01-01 ~ {thr:,}")
    lo, hi = float(np.min(cal_days)), float(np.max(cal_days))
    log(f"date range in store: {days_to_iso(lo)} .. {days_to_iso(hi)}")


def stage_gate(args) -> None:
    """Phase 0 density table. Needs: rough calibration + banked-LSA store embedding."""
    cal = json.loads((CACHE_DIR / "t2_rough_calibration.json").read_text())
    cal_ids = np.asarray(cal["sample_ids"], dtype=np.int64)
    cal_days = np.asarray(cal["days"])
    predict = fit_id_date_proxy(cal_ids, cal_days)
    ids, _ = load_store_ids()
    id_lo, id_hi = int(ids.min()), int(ids.max()) + 1

    z = np.load(CACHE_DIR / "t2_store_embed.npz")
    Z_store = z["Z"]            # (400k, 128) banked-transform LSA, L2-normalized
    eligible = z["eligible"]    # abstract>=200 chars AND nnz>=5 in banked vocab
    store_days = predict(ids.astype(np.float64))

    from artifacts_scripts.gap_detection.data import load_dataset
    from artifacts_scripts.gap_detection.detectors import knn_radii

    ds = load_dataset()
    sub_days = predict(ds.page_ids.astype(np.float64))
    fg_all = ds.fg_idx
    K_NN, SCOPE_Q = 10, 0.25

    rows = []
    t_grid = [(y, m) for y in args.t_years for m in ((1, 7) if args.half_years else (1,))]
    for (ty, tm) in t_grid:
        t_days = year_to_days(ty, tm)
        t_id = invert_date_to_id(predict, t_days, id_lo, id_hi)
        fg_pre = fg_all[sub_days[fg_all] <= t_days]
        bg_pre = ds.bg_idx[sub_days[ds.bg_idx] <= t_days]
        n_pre_store = int((store_days <= t_days).sum())
        n_pre_elig = int(((store_days <= t_days) & eligible).sum())
        row_base = {
            "T": f"{ty}-{tm:02d}-01", "t_id_threshold": int(t_id),
            "n_pre_store": n_pre_store, "n_pre_store_eligible": n_pre_elig,
            "n_fg_preT_banked": int(fg_pre.size), "n_bg_preT_banked": int(bg_pre.size),
        }
        scope_thr = None
        if fg_pre.size >= 50 and bg_pre.size >= 200:
            r_bg = knn_radii(ds.Z[bg_pre], ds.Z[fg_pre], K_NN)
            scope_thr = float(np.quantile(r_bg, SCOPE_Q))
        for dy in args.deltas:
            e_days = year_to_days(ty + dy, tm)
            win = (store_days > t_days) & (store_days <= e_days)
            n_arr = int(win.sum())
            n_arr_elig = int((win & eligible).sum())
            n_in_scope = None
            if scope_thr is not None and n_arr_elig:
                arr_idx = np.where(win & eligible)[0]
                r_arr = knn_radii(Z_store[arr_idx], ds.Z[fg_pre], K_NN)
                n_in_scope = int((r_arr <= scope_thr).sum())
            rows.append({**row_base, "delta_years": dy,
                         "window_end": days_to_iso(e_days),
                         "n_arrivals_store": n_arr,
                         "n_arrivals_eligible": n_arr_elig,
                         "n_arrivals_in_scope": n_in_scope,
                         "scope_thr": scope_thr})
            log(f"T={row_base['T']} d={dy}y: pre={n_pre_elig:,} arr={n_arr_elig:,} "
                f"in_scope={n_in_scope} fg_preT={fg_pre.size}")
    (CACHE_DIR / "t2_gate_table.json").write_text(json.dumps(rows, indent=1))
    log(f"wrote {CACHE_DIR / 't2_gate_table.json'}")


def stage_full(args) -> None:
    """Full calibration: ~n_full stratified + densification near the chosen T."""
    ids, _ = load_store_ids()
    pre = json.loads((CACHE_DIR / "t2_rough_calibration.json").read_text())
    sample = stratified_ids(ids, args.n_full, seed=421)
    boundary = json.loads((CACHE_DIR / "t2_choice.json").read_text())
    t_id = int(boundary["t_id_threshold"])
    # densify around the boundary: ids within a +-window of T's id threshold
    band = ids[(ids >= t_id - args.band) & (ids <= t_id + args.band)]
    rng = np.random.default_rng(422)
    if band.size > args.n_band:
        band = np.sort(rng.choice(band, size=args.n_band, replace=False))
    todo = np.unique(np.concatenate([
        sample, band, np.asarray(pre["sample_ids"], dtype=np.int64)]))
    log(f"full calibration: {sample.size} stratified + {band.size} boundary "
        f"(band +-{args.band} ids around {t_id:,}) -> {todo.size} unique")
    recs = fetch_creation_dates(todo.tolist(), log=log)
    cal_ids, cal_days, qc = collect_dates(recs)
    log(f"usable {cal_ids.size}/{todo.size}; qc={qc}")

    # held-out validation: even/odd split by rank
    order = np.argsort(cal_ids)
    cal_ids, cal_days = cal_ids[order], cal_days[order]
    fit_m = np.arange(cal_ids.size) % 2 == 0
    predict_fit = fit_id_date_proxy(cal_ids[fit_m], cal_days[fit_m])
    rep_all = proxy_error_report(predict_fit, cal_ids[~fit_m], cal_days[~fit_m])
    t_days = year_to_days(*[int(x) for x in boundary["T"].split("-")[:2]])
    near = ~fit_m & (np.abs(cal_days - t_days) <= 365.25)
    rep_near = proxy_error_report(predict_fit, cal_ids[near], cal_days[near])
    # wrinkle rate: held-out points whose proxy error exceeds thresholds
    err = np.abs(predict_fit(cal_ids[~fit_m]) - cal_days[~fit_m])
    wrinkles = {f"gt_{d}d": float((err > d).mean()) for d in (90, 180, 365)}
    log(f"validation all: {rep_all}")
    log(f"validation near-boundary (+-1y): {rep_near}")
    log(f"wrinkle rates (held-out err): {wrinkles}")

    # final proxy on ALL calibration points; date assignment for store + subsample
    predict = fit_id_date_proxy(cal_ids, cal_days)
    store_days = predict(ids.astype(np.float64))
    from artifacts_scripts.gap_detection.data import load_dataset

    ds = load_dataset()
    sub_days = predict(ds.page_ids.astype(np.float64))
    # exact dates override proxy where known
    exact = {int(i): float(d) for i, d in zip(cal_ids, cal_days)}
    for arr, idarr in ((store_days, ids), (sub_days, ds.page_ids)):
        for j, pid in enumerate(idarr):
            v = exact.get(int(pid))
            if v is not None:
                arr[j] = v
    np.savez_compressed(CACHE_DIR / "t2_dates.npz",
                        store_days=store_days, sub_days=sub_days,
                        cal_ids=cal_ids, cal_days=cal_days)
    out = {
        "n_calibration": int(cal_ids.size), "qc": qc,
        "validation_all": rep_all, "validation_near_boundary": rep_near,
        "wrinkle_rates_heldout": wrinkles,
        "n_exact_overrides_store": int(np.isin(ids, cal_ids).sum()),
        "n_exact_overrides_subsample": int(np.isin(ds.page_ids, cal_ids).sum()),
    }
    (CACHE_DIR / "t2_calibration_report.json").write_text(json.dumps(out, indent=1))
    log("wrote t2_dates.npz + t2_calibration_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rough", "gate", "full"])
    ap.add_argument("--n-rough", type=int, default=240)
    ap.add_argument("--n-full", type=int, default=2000)
    ap.add_argument("--n-band", type=int, default=600)
    ap.add_argument("--band", type=int, default=40_000)
    ap.add_argument("--t-years", type=int, nargs="+",
                    default=[2002, 2003, 2004, 2014, 2018, 2022])
    ap.add_argument("--half-years", action="store_true", default=True)
    ap.add_argument("--deltas", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    {"rough": stage_rough, "gate": stage_gate, "full": stage_full}[args.stage](args)


if __name__ == "__main__":
    main()
