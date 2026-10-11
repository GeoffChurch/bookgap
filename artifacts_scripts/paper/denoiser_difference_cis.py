"""Paired differences between denoisers, with 95% intervals (the paper's Appendix B).

For every two runs that share an embedder, the per-topic difference in each
metric (target gap coverage percentage, proposal gap percentile) over the topics
both runs scored, within the fixed random 20 (``--topics random20``) or among all
77 (``--topics all``): n, the mean difference, and a two-sided 95% t-interval,
mean +/- t(0.975, n-1) * sd / sqrt(n). Paired, because both runs score the same
topics and topics vary far more than denoisers do; t rather than 1.96, because n
is as small as 20.

Reproduces the means the paper quoted from the May 2026 runs (+5.9 GIST LPP vs
NONE, +42.7 Gemma LPP vs NONE, +8.4 Gemma CCLIP(alpha=1) vs LPP, paired over the
19 or 20 shared topics). Of the widths quoted there, GIST's 9.7 is this
t-interval, but Gemma's 14.2 and 13.4 were 1.96 * sd / sqrt(n); the t-interval
gives 15.2 and 14.4.

The runs must share their prompt articles, books and radius fraction, and no two
may share an (embedder, denoiser) pair; anything else is refused.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.denoiser_difference_cis --topics all \\
        --output outputs/regen_2026-10/denoiser_difference_cis.md RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from artifacts_scripts.common import friendly_denoiser_name, friendly_embedder_name, markdown_provenance
from artifacts_scripts.gemma4_quantitative_sweep import TOPIC_SUBSET_RANDOM_20

METRICS = {"gap_coverage_fraction": "target gap coverage %", "proposal_gap_size_percentile": "proposal gap percentile"}
# NONE (no denoising) then LPP (the paper's denoiser) lead, and each difference is the
# later denoiser minus the earlier: LPP - NONE, then every other denoiser minus NONE and LPP.
LEADING = ("NONE", "LPP")


@dataclass(frozen=True)
class TopicSet:
    description: str
    keep: frozenset[str] | None   # the topics compared over; None keeps every topic a run scored


TOPIC_SETS = {
    "random20": TopicSet("in the fixed random-20 subset", frozenset(TOPIC_SUBSET_RANDOM_20)),
    "all": TopicSet("among all topics", None),
}


@dataclass(frozen=True)
class Run:
    embedder: str
    denoiser: str
    scores: dict[str, dict[str, float]]   # metric -> topic -> percentage


@dataclass(frozen=True)
class Difference:
    n: int
    mean: float
    half_width: float

    @property
    def excludes_zero(self) -> bool:
        return abs(self.mean) > self.half_width


def paired_difference(a: dict[str, float], b: dict[str, float]) -> Difference:
    """a - b over the topics both scored, with a two-sided 95% t-interval."""
    d = [a[t] - b[t] for t in sorted(set(a) & set(b))]
    n = len(d)
    return Difference(n, statistics.mean(d), stats.t.ppf(0.975, n - 1) * statistics.stdev(d) / math.sqrt(n))


def read_run(run_dir: Path, topics: TopicSet) -> tuple[tuple, Run]:
    cfg = OmegaConf.load(run_dir / "experiment_config.yaml")
    results = pd.read_csv(run_dir / "gaps_analysis_results.csv")
    if topics.keep is not None:
        results = results[results["topic_id"].isin(topics.keep)]
    scores = {m: {t: 100 * v for t, v in zip(results["topic_id"], results[m]) if not math.isnan(v)} for m in METRICS}
    shared = (cfg.llm.num_articles, cfg.llm.num_books, float(cfg.gap.radius_fraction))
    return shared, Run(cfg.cloud.embedder.model_name, cfg.cloud.denoiser, scores)


def order(denoiser: str) -> tuple:
    return (LEADING.index(denoiser) if denoiser in LEADING else len(LEADING), denoiser)


def name(run: Run) -> str:
    return friendly_denoiser_name(run.denoiser, latex=False)


def report(runs: list[Run]) -> str:
    out = []
    for embedder in sorted({r.embedder for r in runs}):
        mine = sorted((r for r in runs if r.embedder == embedder), key=lambda r: order(r.denoiser))
        for metric, label in METRICS.items():
            out += [f"## {friendly_embedder_name(embedder)}: {label}", "",
                    "| difference | n | mean | 95% CI | CI excludes 0 |", "|---|---:|---:|---|:---:|"]
            for x, y in combinations(mine, 2):  # combinations keeps `order`: x is the earlier
                d = paired_difference(y.scores[metric], x.scores[metric])
                out.append(f"| {name(y)} - {name(x)} | {d.n} | {d.mean:+.1f} +/- {d.half_width:.1f} | "
                           f"[{d.mean - d.half_width:.1f}, {d.mean + d.half_width:.1f}] | {'yes' if d.excludes_zero else ''} |")
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", choices=sorted(TOPIC_SETS), required=True,
                    help="compare over the fixed random 20 topics, or every topic the runs scored")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    topics = TOPIC_SETS[args.topics]
    read = [read_run(d, topics) for d in args.run_dirs]
    shared = {s for s, _ in read}
    if len(shared) != 1:
        print(f"error: the runs differ in prompt articles, books or radius fraction: {sorted(shared)}", file=sys.stderr)
        return 2
    keys = [(r.embedder, r.denoiser) for _, r in read]
    if len(set(keys)) != len(keys):
        print(f"error: more than one run for an (embedder, denoiser): {sorted(k for k in keys if keys.count(k) > 1)}", file=sys.stderr)
        return 2
    text = ("# Paired denoiser differences\n\n"
            f"Later denoiser minus earlier, per topic, over the topics both runs scored {topics.description}; "
            "95% t-intervals.\n\n" + report([r for _, r in read]) + markdown_provenance(__spec__.name))
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
