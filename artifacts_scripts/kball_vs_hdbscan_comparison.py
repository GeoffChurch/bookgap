"""Side-by-side comparison: K-ball vs HDBSCAN gap detection.

Reads CSVs from ``artifacts/{hdbscan,kball}_gap_probe/`` (produced by the
denoiser sweep), groups by (denoiser, drop_k), and emits a comparison
table that supports the claim "K-ball is better than HDBSCAN for our
gap-detection use case." Outputs as both stdout markdown and a CSV at
``artifacts/kball_vs_hdbscan_comparison.csv`` for downstream paper-table
generation.

The comparison is structured as four sections:

1. **Volume comparison** — gaps per scope, topics covered, distribution.
2. **Per-gap statistical signal** — internal-to-each-algorithm metrics
   that aren't directly comparable but document the strength of evidence.
3. **Robustness across denoisers** — variance of headline metrics; this
   is the K-ball-is-denoiser-robust claim from the sweep.
4. **Best-of head-to-head** — the strongest K-ball config vs the
   strongest HDBSCAN config on each metric.

Phase 1 of the LLM-evaluated comparison plan: this script gives the
"easy" volume/signal numbers from existing CSVs. Phase 2+ would add
LLM-judge proposal-quality scores via Ollama generation + judging.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HDBSCAN_DIR = ROOT / "artifacts" / "hdbscan_gap_probe"
KBALL_DIR = ROOT / "artifacts" / "kball_gap_probe"
OUT_CSV = ROOT / "artifacts" / "kball_vs_hdbscan_comparison.csv"


def _load(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _parse_key(stem: str) -> tuple[str, str, str] | None:
    """Parse `{scope}__{denoiser}__drop{drop_k}` → (scope, denoiser, drop_k_str).
    Cloud-baseline pattern `{scope}__{algo}__rf{rf}__num{n}` returns None."""
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    scope, denoiser, drop = parts
    if not drop.startswith("drop"):
        return None
    return scope, denoiser, drop[len("drop"):]


def _summarise_kball(rows: list[dict[str, str]]) -> dict:
    """Per-CSV K-ball summary: gaps × topic-coverage × mean signal.

    Skips rows where ``gap_size_percentile`` is NaN (legitimate for topics
    where the book-as-query null distribution is degenerate — too few
    books for std/percentile to be defined).
    """
    if not rows:
        return {
            "total_gaps": 0, "topics_with_gaps": 0,
            "median_gaps_per_topic": 0,
            "mean_gap_size": float("nan"),
            "mean_percentile": float("nan"),
        }
    n = len(rows)
    gaps_per_topic: dict[str, int] = defaultdict(int)
    for r in rows:
        gaps_per_topic[r["topic_id"]] += 1
    valid_pcts = [
        float(r["gap_size_percentile"]) for r in rows
        if r["gap_size_percentile"].lower() != "nan"
    ]
    return {
        "total_gaps": n,
        "topics_with_gaps": len(gaps_per_topic),
        "median_gaps_per_topic": statistics.median(gaps_per_topic.values()),
        "mean_gap_size": sum(int(r["gap_size"]) for r in rows) / n,
        "mean_percentile": (
            sum(valid_pcts) / len(valid_pcts) if valid_pcts else float("nan")
        ),
        "n_valid_percentile": len(valid_pcts),
    }


def _summarise_hdbscan(rows: list[dict[str, str]]) -> dict:
    """Per-CSV HDBSCAN summary: qualifying × significant × effect."""
    if not rows:
        return {
            "qualifying": 0, "p_lt_05": 0, "fdr_05": 0,
            "mean_sig_effect": float("nan"),
            "n_clusters_total": 0,
        }
    n = len(rows)
    p_lt_05 = sum(1 for r in rows if float(r["p_value"]) < 0.05)
    fdr_05 = sum(1 for r in rows if r["fdr_pass_05"] == "1")
    sig_effects = [float(r["effect_size"]) for r in rows if float(r["p_value"]) < 0.05]
    return {
        "qualifying": n,
        "p_lt_05": p_lt_05,
        "fdr_05": fdr_05,
        "mean_sig_effect": (sum(sig_effects) / len(sig_effects)) if sig_effects else float("nan"),
        "n_clusters_total": int(rows[0].get("n_clusters_total", 0)),
    }


def main() -> None:
    # --- Aggregate ---
    kball_summaries: dict[tuple[str, str, str], dict] = {}
    cloud_baseline = None
    for path in sorted(KBALL_DIR.glob("*.csv")):
        stem = path.stem
        key = _parse_key(stem)
        rows = _load(path)
        if key is None:
            # Cloud-baseline (no denoiser) — has different filename pattern.
            scope = stem.split("__")[0]
            cloud_baseline = (scope, _summarise_kball(rows))
            continue
        kball_summaries[key] = _summarise_kball(rows)

    hdbscan_summaries: dict[tuple[str, str, str], dict] = {}
    for path in sorted(HDBSCAN_DIR.glob("*.csv")):
        key = _parse_key(path.stem)
        if key is None:
            continue
        hdbscan_summaries[key] = _summarise_hdbscan(_load(path))

    # Restrict to scope `17-1702` (the subfield sweep) — drop the smoke-test rows.
    SCOPE = "17-1702"
    kball_subset = {k: v for k, v in kball_summaries.items() if k[0] == SCOPE}
    hdbscan_subset = {k: v for k, v in hdbscan_summaries.items() if k[0] == SCOPE}

    print(f"## K-ball vs HDBSCAN gap detection on subfield {SCOPE}")
    print(f"  Sweep matrix: {len(kball_subset)} K-ball configs, {len(hdbscan_subset)} HDBSCAN configs")
    if cloud_baseline is not None:
        print(f"  + 1 K-ball Cloud baseline (no denoiser): {cloud_baseline[1]}")
    print()

    # --- Section 1: Volume comparison ---
    kball_volumes = [s["total_gaps"] for s in kball_subset.values()]
    kball_topics = [s["topics_with_gaps"] for s in kball_subset.values()]
    hdbscan_qual = [s["qualifying"] for s in hdbscan_subset.values()]
    hdbscan_fdr = [s["fdr_05"] for s in hdbscan_subset.values()]

    print("### 1. Volume comparison (across all configs)")
    print()
    print(f"| Metric | K-ball | HDBSCAN |")
    print(f"|---|---|---|")
    print(f"| Mean gaps/clusters per scope | {statistics.mean(kball_volumes):.1f} | "
          f"{statistics.mean(hdbscan_qual):.1f} (qualifying) / "
          f"{statistics.mean(hdbscan_fdr):.1f} (FDR-significant) |")
    print(f"| Median | {statistics.median(kball_volumes):.0f} | "
          f"{statistics.median(hdbscan_qual):.0f} / {statistics.median(hdbscan_fdr):.0f} |")
    print(f"| Std | {statistics.pstdev(kball_volumes):.2f} | "
          f"{statistics.pstdev(hdbscan_qual):.2f} / {statistics.pstdev(hdbscan_fdr):.2f} |")
    print(f"| Topics with ≥1 gap (mean) | "
          f"{statistics.mean(kball_topics):.1f} / 77 | "
          f"n/a (per-scope, not per-topic) |")
    print()

    # --- Section 2: Per-gap statistical signal ---
    kball_pcts = [s["mean_percentile"] for s in kball_subset.values()]
    hdbscan_eff = [s["mean_sig_effect"] for s in hdbscan_subset.values()
                   if not (s["mean_sig_effect"] != s["mean_sig_effect"])]  # not NaN

    print("### 2. Per-gap statistical signal")
    print()
    print(f"| Metric | K-ball | HDBSCAN |")
    print(f"|---|---|---|")
    print(f"| Mean per-gap strength | percentile = {statistics.mean(kball_pcts):.3f} "
          f"(vs book-as-query null) | effect_size = "
          f"{statistics.mean(hdbscan_eff):.3f} (vs corpus base rate) |")
    print(f"| Strength range across configs | "
          f"[{min(kball_pcts):.3f}, {max(kball_pcts):.3f}] | "
          f"[{min(hdbscan_eff):.3f}, {max(hdbscan_eff):.3f}] |")
    print()
    print("(Note: metrics are not directly comparable because the null "
          "distributions differ. K-ball's ~0.98 means \"gaps at the 98th "
          "percentile of typical book-gap sizes\"; HDBSCAN's ~0.05 means "
          "\"5% fewer books than expected under uniform sampling.\")")
    print()

    # --- Section 3: Robustness across denoisers ---
    print("### 3. Robustness across denoiser configs")
    print()
    print(f"| Metric | K-ball | HDBSCAN |")
    print(f"|---|---|---|")
    print(f"| Coefficient of variation in volume | "
          f"{statistics.pstdev(kball_volumes) / statistics.mean(kball_volumes):.3f} | "
          f"{statistics.pstdev(hdbscan_qual) / max(statistics.mean(hdbscan_qual), 1):.3f} |")
    print(f"| Coefficient of variation in mean signal | "
          f"{statistics.pstdev(kball_pcts) / statistics.mean(kball_pcts):.4f} | "
          f"{statistics.pstdev(hdbscan_eff) / max(statistics.mean(hdbscan_eff), 1e-9):.3f} |")
    print()

    # --- Section 4: Best-of head-to-head ---
    print("### 4. Best-of head-to-head (single best config per algorithm)")
    print()
    # Best K-ball: take max-percentile config (or smallest variance, here all ~0.98)
    best_kb = max(kball_subset.items(), key=lambda kv: kv[1]["mean_percentile"])
    # Best HDBSCAN: max mean_sig_effect among FDR-significant
    best_hd = max(
        hdbscan_subset.items(),
        key=lambda kv: kv[1]["mean_sig_effect"] if not (
            kv[1]["mean_sig_effect"] != kv[1]["mean_sig_effect"]
        ) else -1,
    )

    print(f"| Metric | K-ball ({best_kb[0][1]}, drop_k={best_kb[0][2]}) | "
          f"HDBSCAN ({best_hd[0][1]}, drop_k={best_hd[0][2]}) |")
    print(f"|---|---|---|")
    print(f"| Total gaps/clusters | {best_kb[1]['total_gaps']} | "
          f"{best_hd[1]['qualifying']} qualifying ({best_hd[1]['fdr_05']} FDR-sig) |")
    print(f"| Per-gap strength | percentile {best_kb[1]['mean_percentile']:.3f} | "
          f"effect_size {best_hd[1]['mean_sig_effect']:.3f} |")
    print(f"| Per-topic granularity | "
          f"{best_kb[1]['median_gaps_per_topic']:.0f} gaps/topic | "
          f"per-scope only |")
    print(f"| Topics covered | {best_kb[1]['topics_with_gaps']}/77 | "
          f"n/a |")
    print()

    # --- Save raw CSV ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm", "scope", "denoiser", "drop_k",
            "total_units", "fdr_significant", "topics_with_units",
            "median_per_topic", "mean_signal", "signal_metric",
        ])
        for k, s in sorted(kball_subset.items()):
            w.writerow([
                "kball", k[0], k[1], k[2],
                s["total_gaps"], "n/a", s["topics_with_gaps"],
                s["median_gaps_per_topic"], f"{s['mean_percentile']:.4f}",
                "gap_size_percentile",
            ])
        if cloud_baseline is not None:
            scope, s = cloud_baseline
            w.writerow([
                "kball", scope, "[cloud-baseline]", "n/a",
                s["total_gaps"], "n/a", s["topics_with_gaps"],
                s["median_gaps_per_topic"], f"{s['mean_percentile']:.4f}",
                "gap_size_percentile",
            ])
        for k, s in sorted(hdbscan_subset.items()):
            w.writerow([
                "hdbscan", k[0], k[1], k[2],
                s["qualifying"], s["fdr_05"], "n/a",
                "n/a", f"{s['mean_sig_effect']:.4f}",
                "binomial_effect_size",
            ])
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
