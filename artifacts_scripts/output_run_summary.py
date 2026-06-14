from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .common import (
    GAP_COVERAGE_FRACTION_COLUMN,
    PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN,
    PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
    RunMetadata,
    add_per_topic_gap_idx,
    find_run_directories,
    load_results_dataframe,
    load_run_metadata,
    relative_run_path,
    valid_rank_fraction_mask,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RunSummary:
    run_path: Path
    metadata: RunMetadata
    per_topic_gap_idx: int
    proposal_book_rank_fraction_mean: float
    proposal_article_rank_fraction_mean: float
    proposal_article_rank_fraction_median: float
    proposal_book_rank_fraction_std: float
    row_count: int
    gap_coverage_fraction_mean: float
    gap_coverage_fraction_median: float
    gap_coverage_fraction_std: float
    proposal_gap_size_percentile_mean: float
    proposal_gap_size_percentile_std: float
    target_gap_size_percentile_mean: float
    proposal_gap_zscore_mean: float
    proposal_gap_zscore_median: float
    target_gap_zscore_mean: float
    target_gap_zscore_median: float

def summarize_run(run_dir: Path, topic_subset: Optional[set] = None) -> Dict[int, RunSummary]:
    metadata = load_run_metadata(run_dir)
    if metadata is None:
        return {}

    df = load_results_dataframe(run_dir)
    if df is None:
        return {}

    if topic_subset is not None and "topic_id" in df.columns:
        before = len(df)
        df = df[df["topic_id"].isin(topic_subset)].copy()
        if len(df) < before:
            LOGGER.info(
                "Filtered %s from %d to %d rows by topic_subset",
                run_dir, before, len(df),
            )

    try:
        mask = valid_rank_fraction_mask(df)
    except KeyError as exc:
        LOGGER.warning("%s in %s", exc, run_dir)
        return {}

    valid_df = df.loc[mask].copy()
    if valid_df.empty:
        LOGGER.warning(
            "No valid values for %s in %s", PROPOSAL_BOOK_RANK_FRACTION_COLUMN, run_dir
        )
        return {}

    if "topic_id" not in valid_df.columns:
        raise ValueError(f"Column 'topic_id' missing from results in {run_dir}")
    if "gap_size" not in valid_df.columns:
        raise ValueError(f"Column 'gap_size' missing from results in {run_dir}")

    summaries: Dict[int, RunSummary] = {}

    try:
        valid_df = add_per_topic_gap_idx(valid_df)
    except ValueError as exc:
        raise ValueError(f"Invalid gap ordering in {run_dir}: {exc}") from exc

    for gap_idx, gap_df in valid_df.groupby("per_topic_gap_idx"):
        gap_idx_int = int(gap_idx)  # type: ignore[arg-type]
        book_values = gap_df[PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
        article_mean = float("nan")
        article_median = float("nan")
        coverage_mean = float("nan")
        coverage_median = float("nan")
        coverage_std = float("nan")
        gap_size_percentile_mean = float("nan")
        gap_size_percentile_std = float("nan")
        target_gap_size_percentile_mean = float("nan")
        proposal_gap_zscore_mean = float("nan")
        proposal_gap_zscore_median = float("nan")
        target_gap_zscore_mean = float("nan")
        target_gap_zscore_median = float("nan")

        if PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN in gap_df.columns:
            article_column = gap_df[PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN]
            article_mask = article_column.notna() & (article_column != -1)
            if article_mask.any():
                article_values = article_column.loc[article_mask]
                article_mean = float(article_values.mean())
                article_median = float(article_values.median())

        if GAP_COVERAGE_FRACTION_COLUMN in gap_df.columns:
            coverage_column = gap_df[GAP_COVERAGE_FRACTION_COLUMN]
            coverage_mask = coverage_column.notna() & (coverage_column >= 0)
            if coverage_mask.any():
                coverage_values = coverage_column.loc[coverage_mask]
                coverage_mean = float(coverage_values.mean())
                coverage_median = float(coverage_values.median())
                # Sample std across topics. With ~20 topics, SE_mean = std/sqrt(20).
                coverage_std = float(coverage_values.std(ddof=1))

        if "proposal_gap_size_percentile" in gap_df.columns:
            gap_size_percentile_column = gap_df["proposal_gap_size_percentile"]
            gap_size_percentile_mask = gap_size_percentile_column.notna()
            if gap_size_percentile_mask.any():
                gap_size_percentile_values = gap_size_percentile_column.loc[
                    gap_size_percentile_mask
                ]
                gap_size_percentile_mean = float(gap_size_percentile_values.mean())
                gap_size_percentile_std = float(gap_size_percentile_values.std(ddof=1))
        if "gap_size_percentile" in gap_df.columns:
            target_gap_size_percentile_column = gap_df["gap_size_percentile"]
            target_gap_size_percentile_mask = target_gap_size_percentile_column.notna()
            if target_gap_size_percentile_mask.any():
                target_gap_size_percentile_values = target_gap_size_percentile_column.loc[
                    target_gap_size_percentile_mask
                ]
                target_gap_size_percentile_mean = float(target_gap_size_percentile_values.mean())

        if "proposal_gap_zscore" in gap_df.columns:
            proposal_gap_zscore_column = gap_df["proposal_gap_zscore"]
            proposal_gap_zscore_mask = proposal_gap_zscore_column.notna()
            if proposal_gap_zscore_mask.any():
                proposal_gap_zscore_values = proposal_gap_zscore_column.loc[
                    proposal_gap_zscore_mask
                ]
                proposal_gap_zscore_mean = float(proposal_gap_zscore_values.mean())
                proposal_gap_zscore_median = float(proposal_gap_zscore_values.median())

        if "gap_zscore" in gap_df.columns:
            target_gap_zscore_column = gap_df["gap_zscore"]
            target_gap_zscore_mask = target_gap_zscore_column.notna()
            if target_gap_zscore_mask.any():
                target_gap_zscore_values = target_gap_zscore_column.loc[
                    target_gap_zscore_mask
                ]
                target_gap_zscore_mean = float(target_gap_zscore_values.mean())
                target_gap_zscore_median = float(target_gap_zscore_values.median())

        summaries[gap_idx_int] = RunSummary(
            run_path=run_dir,
            metadata=metadata,
            per_topic_gap_idx=gap_idx_int,
            proposal_book_rank_fraction_mean=float(book_values.mean()),
            proposal_article_rank_fraction_mean=article_mean,
            proposal_article_rank_fraction_median=article_median,
            proposal_book_rank_fraction_std=float(book_values.std()),
            row_count=int(gap_df.shape[0]),
            gap_coverage_fraction_mean=coverage_mean,
            gap_coverage_fraction_median=coverage_median,
            gap_coverage_fraction_std=coverage_std,
            proposal_gap_size_percentile_mean=gap_size_percentile_mean,
            proposal_gap_size_percentile_std=gap_size_percentile_std,
            target_gap_size_percentile_mean=target_gap_size_percentile_mean,
            proposal_gap_zscore_mean=proposal_gap_zscore_mean,
            proposal_gap_zscore_median=proposal_gap_zscore_median,
            target_gap_zscore_mean=target_gap_zscore_mean,
            target_gap_zscore_median=target_gap_zscore_median,
        )

    return summaries


def collect_summaries(
    outputs_dir: Path, topic_subset: Optional[set] = None,
) -> Dict[int, List[RunSummary]]:
    summaries: DefaultDict[int, List[RunSummary]] = defaultdict(list)
    for run_dir in find_run_directories(outputs_dir):
        summary = summarize_run(run_dir, topic_subset=topic_subset)
        for gap_idx, run_summary in summary.items():
            summaries[gap_idx].append(run_summary)

    for gap_summaries in summaries.values():
        gap_summaries.sort(key=lambda s: s.run_path)

    return dict(sorted(summaries.items()))

def gap_output_path(base_path: Path, gap_idx: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_gap{gap_idx}{base_path.suffix}")


def write_csvs(
    summaries_by_gap: Mapping[int, Iterable[RunSummary]],
    outputs_dir: Path,
    output_file: Path,
) -> None:
    if not summaries_by_gap:
        gap_output = gap_output_path(output_file, 0)
        gap_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records([]).to_csv(gap_output, index=False)
        LOGGER.info("Wrote empty summary to %s", gap_output)
        return

    for gap_idx, summaries in summaries_by_gap.items():
        records = []
        for summary in summaries:
            metadata = summary.metadata
            records.append(
                {
                    "run": str(relative_run_path(summary.run_path, outputs_dir)),
                    "embedder_model": metadata.embedder_model,
                    "denoiser": metadata.denoiser,
                    "drop_k": metadata.drop_k,
                    "llm_num_articles": metadata.num_articles,
                    "llm_num_books": metadata.num_books,
                    "proposal_book_rank_fraction_mean": summary.proposal_book_rank_fraction_mean,
                    "proposal_article_rank_fraction_mean": summary.proposal_article_rank_fraction_mean,
                    "proposal_article_rank_fraction_median": summary.proposal_article_rank_fraction_median,
                    "proposal_book_rank_fraction_std": summary.proposal_book_rank_fraction_std,
                    "row_count": summary.row_count,
                    "gap_coverage_fraction_mean": summary.gap_coverage_fraction_mean,
                    "gap_coverage_fraction_median": summary.gap_coverage_fraction_median,
                    "gap_coverage_fraction_std": summary.gap_coverage_fraction_std,
                    "proposal_gap_size_percentile_mean": summary.proposal_gap_size_percentile_mean,
                    "proposal_gap_size_percentile_std": summary.proposal_gap_size_percentile_std,
                    "target_gap_size_percentile_mean": summary.target_gap_size_percentile_mean,
                    "proposal_gap_zscore_mean": summary.proposal_gap_zscore_mean,
                    "proposal_gap_zscore_median": summary.proposal_gap_zscore_median,
                    "target_gap_zscore_mean": summary.target_gap_zscore_mean,
                    "target_gap_zscore_median": summary.target_gap_zscore_median,
                }
            )

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.sort_values(
                [
                    "proposal_gap_size_percentile_mean",
                    # "proposal_book_rank_fraction_mean",
                    "embedder_model",
                    "denoiser",
                    "drop_k",
                    "llm_num_articles",
                    "llm_num_books",
                    "run",
                ],
                na_position="last",
            )

        gap_output = gap_output_path(output_file, gap_idx)
        gap_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(gap_output, index=False, float_format="%.5f")
        LOGGER.info("Wrote %s for gap index %s", gap_output, gap_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize BookGap runs by denoiser and drop_k using the proposal_book_rank_fraction "
            "statistics."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the root outputs directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts") / "run_gap_summary.csv",
        help="Where to write the summary CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--topic-subset",
        type=str,
        default=None,
        help=(
            "Comma-separated list of topic IDs (e.g. 'T10028,T10181,...') to filter each "
            "run's gaps_analysis_results.csv to before computing means/medians. Used to "
            "compare 77-topic full runs against 20-topic subset runs on the same footing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    topic_subset = None
    if args.topic_subset:
        topic_subset = set(t.strip() for t in args.topic_subset.split(",") if t.strip())
        LOGGER.info("Filtering each run's CSV to %d topics: %s", len(topic_subset), sorted(topic_subset))
    summaries_by_gap = collect_summaries(args.outputs_dir, topic_subset=topic_subset)
    if not summaries_by_gap:
        LOGGER.warning("No summaries created from outputs in %s", args.outputs_dir)

    planned_indices = list(summaries_by_gap.keys()) or [0]
    if not args.overwrite:
        for gap_idx in planned_indices:
            candidate = gap_output_path(args.output_file, gap_idx)
            if candidate.exists():
                LOGGER.error(
                    "Output file %s already exists. Use --overwrite to replace it.", candidate
                )
                raise SystemExit(1)

    write_csvs(summaries_by_gap, args.outputs_dir, args.output_file)


if __name__ == "__main__":
    main()
