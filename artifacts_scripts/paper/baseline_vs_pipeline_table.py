"""Compare random-context baseline vs gap-aware production pipeline.

Joins the two per-topic CSVs on ``topic_id`` (inner join — any topic
missing from either side is logged and skipped), computes summary
statistics (mean + median of target gap coverage and proposal gap
percentile, both ×100), runs a two-sided sign test (binomial, p=0.5)
on the per-topic percentile differences, and writes a LaTeX tabular
block to ``paper/baseline_vs_pipeline_table.tex``.

Usage::

    uv run python -m artifacts_scripts.paper.baseline_vs_pipeline_table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


BASELINE_CSV = Path("outputs/random_baseline/results.csv")
PRODUCTION_CSV = Path(
    "outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10/"
    "gaps_analysis_results.csv"
)
PAPER_DIR = Path("paper")
TABLE_PATH = PAPER_DIR / "baseline_vs_pipeline_table.tex"

PCT_COL = "proposal_gap_size_percentile"
COV_COL = "gap_coverage_fraction"


def _load_per_topic(path: Path) -> pd.DataFrame:
    """Read a results CSV and reduce to one row per topic.

    The headline runs use ``num_gaps=1`` so there's only one row per
    topic anyway; keeping the first guards against historical
    multi-gap CSVs.
    """
    return (
        pd.read_csv(path)
        .drop_duplicates("topic_id", keep="first")
        .set_index("topic_id")
    )


def _format_pvalue(p: float) -> str:
    if p == 0.0:
        return "0"
    return f"{p:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    ap.add_argument("--production-csv", type=Path, default=PRODUCTION_CSV)
    ap.add_argument("--table-path", type=Path, default=TABLE_PATH)
    args = ap.parse_args()

    baseline = _load_per_topic(args.baseline_csv)
    production = _load_per_topic(args.production_csv)

    missing_in_baseline = production.index.difference(baseline.index).tolist()
    missing_in_production = baseline.index.difference(production.index).tolist()
    if missing_in_baseline:
        print(
            f"INFO: {len(missing_in_baseline)} topic(s) present in "
            f"production but not in baseline: {sorted(missing_in_baseline)}"
        )
    if missing_in_production:
        print(
            f"INFO: {len(missing_in_production)} topic(s) present in "
            f"baseline but not in production: {sorted(missing_in_production)}"
        )

    joined = baseline.join(production, lsuffix="_b", rsuffix="_p", how="inner")
    print(f"Joined on {len(joined)} topics")

    pct_pair = joined.dropna(subset=[f"{PCT_COL}_b", f"{PCT_COL}_p"])
    for tid in joined.index.difference(pct_pair.index):
        print(
            f"INFO: skipping {tid} — missing percentile "
            f"(baseline={joined.at[tid, f'{PCT_COL}_b']}, "
            f"prod={joined.at[tid, f'{PCT_COL}_p']})"
        )
    n = len(pct_pair)
    if n == 0:
        raise SystemExit("No topics with paired percentiles on both sides; aborting")

    b_pct_arr = pct_pair[f"{PCT_COL}_b"].to_numpy() * 100.0
    p_pct_arr = pct_pair[f"{PCT_COL}_p"].to_numpy() * 100.0

    # Coverage stats (stdout diagnostic only — not in the table). Uses
    # whatever topics have both-side coverage, which may be a subset of
    # the percentile-joined topics.
    cov_pair = joined.dropna(subset=[f"{COV_COL}_b", f"{COV_COL}_p"])
    b_cov_arr = cov_pair[f"{COV_COL}_b"].to_numpy() * 100.0
    p_cov_arr = cov_pair[f"{COV_COL}_p"].to_numpy() * 100.0

    summary = {
        "n": n,
        "n_coverage": len(cov_pair),
        "baseline_cov_mean": float(np.mean(b_cov_arr)),
        "baseline_cov_median": float(np.median(b_cov_arr)),
        "baseline_pct_mean": float(np.mean(b_pct_arr)),
        "baseline_pct_median": float(np.median(b_pct_arr)),
        "production_cov_mean": float(np.mean(p_cov_arr)),
        "production_cov_median": float(np.median(p_cov_arr)),
        "production_pct_mean": float(np.mean(p_pct_arr)),
        "production_pct_median": float(np.median(p_pct_arr)),
    }

    # Sign test (two-sided) on the per-topic percentile differences.
    # Preferred over Wilcoxon because it makes no symmetry-of-differences
    # assumption: it just asks how many topics one system beats the
    # other on. Ties are dropped per the standard sign-test convention;
    # in practice they land at the saturation points 0.0 (both
    # proposals worse than every existing book in the topic) or 1.0
    # (both better than every existing book).
    #
    # TIE_EPS_PCT (in percentage-point units) absorbs float-precision
    # noise from round-trip CSV serialization of rationals like 12/13.
    # Empirical noise floor on this data is ~5e-14 pp; the smallest
    # real per-topic difference is 0.53 pp; we sit a couple orders of
    # magnitude above the noise and many orders below any real signal.
    # Without the tolerance, parser-level ULP differences (Python's
    # float() vs pandas's default C parser) flip ~5 near-tied pairs
    # between tie and win/loss.
    TIE_EPS_PCT = 1e-13
    pct_diffs = p_pct_arr - b_pct_arr
    pct_wins = int(np.sum(pct_diffs > TIE_EPS_PCT))
    pct_losses = int(np.sum(pct_diffs < -TIE_EPS_PCT))
    pct_ties = int(np.sum(np.abs(pct_diffs) <= TIE_EPS_PCT))
    pct_nonzero = pct_wins + pct_losses
    if pct_nonzero == 0:
        pct_p = float("nan")
    else:
        pct_p = float(binomtest(
            pct_wins, pct_nonzero, p=0.5, alternative="two-sided"
        ).pvalue)

    summary["sign_test_percentile_p"] = pct_p
    summary["sign_test_wins"] = pct_wins
    summary["sign_test_losses"] = pct_losses
    summary["sign_test_ties"] = pct_ties

    # --- Print summary to stdout ---
    print()
    print(f"=== Summary: percentile over {n} topics; "
          f"coverage over {summary['n_coverage']} topics ===")
    print(f"{'':<22}{'cov mean':>12}{'cov median':>12}"
          f"{'pct mean':>12}{'pct median':>12}")
    print(f"{'Random baseline':<22}"
          f"{summary['baseline_cov_mean']:>12.1f}"
          f"{summary['baseline_cov_median']:>12.1f}"
          f"{summary['baseline_pct_mean']:>12.1f}"
          f"{summary['baseline_pct_median']:>12.1f}")
    print(f"{'Ours (gap-aware)':<22}"
          f"{summary['production_cov_mean']:>12.1f}"
          f"{summary['production_cov_median']:>12.1f}"
          f"{summary['production_pct_mean']:>12.1f}"
          f"{summary['production_pct_median']:>12.1f}")
    print()
    print(f"Sign test on proposal gap percentile (two-sided, n_nonzero="
          f"{pct_nonzero}):")
    print(f"  wins (production > baseline): {pct_wins}")
    print(f"  losses (baseline > production): {pct_losses}")
    print(f"  ties (|diff| <= {TIE_EPS_PCT:.0e} pp): {pct_ties}")
    print(f"  p = {_format_pvalue(pct_p)}")

    # --- Write LaTeX tabular block ---
    # We emit ONLY the tabular; the table/caption/label wrapper lives in
    # main.tex so prose can be edited without re-running this script.
    # Proposal gap percentile is the only cross-system metric reported:
    # target gap coverage compares baseline proposals against our
    # pipeline's specific K-ball, which the baseline never tries to fill,
    # so it would be a strawman of the baseline.
    tex = (
        "% Generated by artifacts_scripts/paper/baseline_vs_pipeline_table.py\n"
        f"% Sources: {args.baseline_csv}\n"
        f"%          {args.production_csv}\n"
        f"% Topics joined: {n}; sign test on proposal gap percentile —"
        f" wins={pct_wins}, losses={pct_losses}, ties={pct_ties},"
        f" two-sided p = {pct_p:.3g}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "                  & \\multicolumn{2}{c}{Proposal gap percentile (\\%)} \\\\\n"
        "\\cmidrule(lr){2-3}\n"
        "                  & mean & median \\\\\n"
        "\\midrule\n"
        f"Random baseline   & {summary['baseline_pct_mean']:.1f}"
        f" & {summary['baseline_pct_median']:.1f} \\\\\n"
        f"Ours (gap-aware)  & {summary['production_pct_mean']:.1f}"
        f" & {summary['production_pct_median']:.1f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    args.table_path.write_text(tex, encoding="utf-8")
    print(f"\nWrote LaTeX table to {args.table_path}")


if __name__ == "__main__":
    main()
