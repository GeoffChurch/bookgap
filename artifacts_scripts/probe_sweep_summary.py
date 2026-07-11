"""Aggregate the per-(denoiser, drop_k) probe-sweep CSVs into a comparison table.

Reads CSVs from ``artifacts/hdbscan_gap_probe/`` and ``artifacts/kball_gap_probe/``
(produced by `hdbscan_gap_probe.py` and `kball_gap_probe.py`'s denoiser-bypass
mode), groups by (scope, denoiser, drop_k), and emits two tables to stdout:

  - HDBSCAN: cluster-depletion summary (qualifying count, p<0.05 count,
    FDR-discoveries at α=0.05/0.10/0.20, mean effect_size of significant
    clusters, max effect_size).
  - K-ball: gap-size summary (gaps found, mean/median gap_size, mean
    gap_size_percentile, mean |gap_zscore|).

Used as the final step of a probe sweep — see the "Run the actual probe
sweeps" backlog entry.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
HDBSCAN_DIR = ROOT / "artifacts" / "hdbscan_gap_probe"
KBALL_DIR = ROOT / "artifacts" / "kball_gap_probe"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _summarise_hdbscan(rows: list[dict[str, str]]) -> dict[str, object]:
    """Per-CSV summary: one row per qualifying cluster, summarise across them."""
    if not rows:
        return {"qualifying": 0, "p<0.05": 0, "fdr_05": 0, "fdr_10": 0, "fdr_20": 0,
                "mean_eff": float("nan"), "max_eff": float("nan"),
                "n_clusters_total": 0}
    n_qual = len(rows)
    p_lt_05 = sum(1 for r in rows if float(r["p_value"]) < 0.05)
    fdr_05 = sum(1 for r in rows if r["fdr_pass_05"] == "1")
    fdr_10 = sum(1 for r in rows if r["fdr_pass_10"] == "1")
    fdr_20 = sum(1 for r in rows if r["fdr_pass_20"] == "1")
    sig = [float(r["effect_size"]) for r in rows if float(r["p_value"]) < 0.05]
    mean_eff = sum(sig) / len(sig) if sig else float("nan")
    max_eff = max((float(r["effect_size"]) for r in rows), default=float("nan"))
    n_clusters_total = int(rows[0].get("n_clusters_total", 0))
    return {
        "qualifying": n_qual,
        "p<0.05": p_lt_05,
        "fdr_05": fdr_05,
        "fdr_10": fdr_10,
        "fdr_20": fdr_20,
        "mean_eff": mean_eff,
        "max_eff": max_eff,
        "n_clusters_total": n_clusters_total,
    }


def _summarise_kball(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {"n_gaps": 0, "mean_size": float("nan"), "median_size": float("nan"),
                "mean_pctile": float("nan"), "mean_abs_z": float("nan")}
    sizes = sorted(int(r["gap_size"]) for r in rows)
    n = len(sizes)
    mean_size = sum(sizes) / n
    median_size = sizes[n // 2]
    pctiles = [float(r["gap_size_percentile"]) for r in rows]
    abs_zs = [abs(float(r["gap_zscore"])) for r in rows]
    return {
        "n_gaps": n,
        "mean_size": mean_size,
        "median_size": median_size,
        "mean_pctile": sum(pctiles) / n,
        "mean_abs_z": sum(abs_zs) / n,
    }


def _key(filename: str) -> Optional[tuple[str, str, int]]:
    """Parse `{scope_safe}__{denoiser}__drop{drop_k}.csv` → (scope, denoiser, drop_k)."""
    stem = filename.removesuffix(".csv")
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    scope, denoiser, drop = parts
    if not drop.startswith("drop"):
        return None
    try:
        drop_k = int(drop[len("drop"):])
    except ValueError:
        return None
    return scope, denoiser, drop_k


def main() -> None:
    # HDBSCAN aggregate
    print("=" * 100)
    print(" HDBSCAN cluster-depletion sweep summary ")
    print("=" * 100)
    hdbscan_results: dict[tuple[str, str, int], dict[str, object]] = {}
    for csv_file in sorted(HDBSCAN_DIR.glob("*.csv")):
        key = _key(csv_file.name)
        if key is None:
            continue
        rows = _load_csv(csv_file)
        hdbscan_results[key] = _summarise_hdbscan(rows)

    if hdbscan_results:
        scopes = sorted({k[0] for k in hdbscan_results})
        for scope in scopes:
            print(f"\nscope: {scope}")
            print(f"  {'denoiser':14s} {'drop_k':>6s} {'qual':>6s} {'p<.05':>6s}"
                  f" {'fdr05':>6s} {'fdr10':>6s} {'fdr20':>6s} {'sig_eff':>8s}"
                  f" {'max_eff':>8s} {'#clust':>7s}")
            for k in sorted(hdbscan_results):
                if k[0] != scope:
                    continue
                _, denoiser, drop_k = k
                s = hdbscan_results[k]
                print(f"  {denoiser:14s} {drop_k:>6d} {s['qualifying']:>6d}"
                      f" {s['p<0.05']:>6d} {s['fdr_05']:>6d} {s['fdr_10']:>6d}"
                      f" {s['fdr_20']:>6d} {s['mean_eff']:>8.3f} {s['max_eff']:>8.3f}"
                      f" {s['n_clusters_total']:>7d}")
    else:
        print("  (no HDBSCAN CSVs found)")

    # K-ball aggregate (denoiser bypass + Cloud baseline)
    print("\n")
    print("=" * 100)
    print(" K-ball gap-detection sweep summary ")
    print("=" * 100)
    kball_results: dict[tuple[str, str, str], dict[str, object]] = {}
    for csv_file in sorted(KBALL_DIR.glob("*.csv")):
        key = _key(csv_file.name)
        if key is not None:
            scope, denoiser, drop_k = key
            kball_results[(scope, denoiser, str(drop_k))] = _summarise_kball(_load_csv(csv_file))
            continue
        # Cloud-baseline pattern: {scope}__{algo}__rf{rf}__num{n}.csv
        stem = csv_file.name.removesuffix(".csv")
        parts = stem.split("__")
        if len(parts) >= 2 and parts[1] == "kball":
            kball_results[(parts[0], "[cloud-baseline]", "n/a")] = _summarise_kball(
                _load_csv(csv_file)
            )

    if kball_results:
        scopes = sorted({k[0] for k in kball_results})
        for scope in scopes:
            print(f"\nscope: {scope}")
            print(f"  {'denoiser':18s} {'drop_k':>8s} {'gaps':>6s} {'mean_size':>10s}"
                  f" {'med_size':>10s} {'mean_pct':>10s} {'mean_|z|':>10s}")
            for k in sorted(kball_results):
                if k[0] != scope:
                    continue
                _, denoiser, drop_k = k
                s = kball_results[k]
                print(f"  {denoiser:18s} {drop_k:>8s} {s['n_gaps']:>6d}"
                      f" {s['mean_size']:>10.1f} {s['median_size']:>10.1f}"
                      f" {s['mean_pctile']:>10.3f} {s['mean_abs_z']:>10.2f}")
    else:
        print("  (no K-ball CSVs found)")


if __name__ == "__main__":
    main()
