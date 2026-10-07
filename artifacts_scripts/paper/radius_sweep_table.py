"""The radius-fraction sweep table in the paper (tab:radius_sweep).

One row per run, ordered by the run's gap radius fraction p: the mean and median
target gap coverage percentage, the mean and median proposal gap percentile,
and the median target gap size, over the run's topics. Each metric skips the
topics where it is undefined, so a topic whose proposal has no gap percentile
still counts towards coverage. The best value in each of the four metric
columns is bold, ties included.

The runs must differ only in their radius fraction; anything else (embedder,
denoiser, drop_k, prompt articles and books, topic subset) is refused, since
the table attributes every difference to p.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.radius_sweep_table \\
        --output paper/artifacts/radius_sweep_table.tex RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from artifacts_scripts.common import latex_provenance

METRICS = (  # (column header, results column, statistic)
    (r"\makecell{cov.\\mean}", "gap_coverage_fraction", "mean"),
    (r"\makecell{cov.\\med}", "gap_coverage_fraction", "median"),
    (r"\makecell{pct.\\mean}", "proposal_gap_size_percentile", "mean"),
    (r"\makecell{pct.\\med}", "proposal_gap_size_percentile", "median"),
)


@dataclass(frozen=True)
class Row:
    radius_fraction: float
    metrics: tuple[float, ...]   # percentages, in METRICS order
    gap_median: float


def _identity(cfg) -> tuple:
    """Everything about a run except its radius fraction."""
    return (cfg.cloud.embedder.model_name, cfg.cloud.denoiser, cfg.cloud.drop_k,
            cfg.llm.num_articles, cfg.llm.num_books,
            tuple(cfg.gap.topic_subset) if cfg.gap.topic_subset is not None else None)


def read_row(run_dir: Path) -> tuple[tuple, Row]:
    cfg = OmegaConf.load(run_dir / "experiment_config.yaml")
    results = pd.read_csv(run_dir / "gaps_analysis_results.csv")
    metrics = tuple(100 * getattr(results[column].dropna(), stat)() for _, column, stat in METRICS)
    return _identity(cfg), Row(float(cfg.gap.radius_fraction), metrics, results["gap_size"].dropna().median())


def render(rows: list[Row]) -> str:
    rows = sorted(rows, key=lambda r: r.radius_fraction)
    shown = [[f"{v:.1f}" for v in r.metrics] for r in rows]
    best = [max(float(s[i]) for s in shown) for i in range(len(METRICS))]
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        "$p$ & " + " & ".join(h for h, _, _ in METRICS) + r" & \makecell{gap\\med} \\",
        r"\midrule",
    ]
    for row, cells in zip(rows, shown):
        p = Fraction(row.radius_fraction).limit_denominator(64)
        cells = [rf"\textbf{{{c}}}" if float(c) == best[i] else c for i, c in enumerate(cells)]
        lines.append(f"${p}$ & " + " & ".join(cells) + f" & {row.gap_median:.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    read = [read_row(d) for d in args.run_dirs]
    identities = {identity for identity, _ in read}
    if len(identities) != 1:
        print(f"error: the runs differ in more than their radius fraction: {sorted(map(str, identities))}", file=sys.stderr)
        return 2
    radii = [row.radius_fraction for _, row in read]
    if len(set(radii)) != len(radii):
        print(f"error: two runs share a radius fraction: {sorted(radii)}", file=sys.stderr)
        return 2
    args.output.write_text(render([row for _, row in read]) + latex_provenance(__spec__.name), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
