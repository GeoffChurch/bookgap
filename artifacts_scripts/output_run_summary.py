from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bookgap.cloud import get_cloud_projection
from bookgap.embedder import Embedder

from .common import (
    GAP_COVERAGE_FRACTION_COLUMN,
    PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN,
    PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
    RunMetadata,
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
    proposal_book_rank_fraction_mean: float
    proposal_article_rank_fraction_mean: float
    proposal_article_rank_fraction_median: float
    proposal_book_rank_fraction_std: float
    row_count: int
    gap_coverage_fraction_mean: float
    gap_coverage_fraction_median: float
    projector_skew: float


def _compute_projector_skew(run_dir: Path) -> float:
    config_path = run_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    cloud_cfg = OmegaConf.select(config, "cloud", default=None)

    embedder_model_value = OmegaConf.select(cloud_cfg, "embedder_model", default=None)
    enum_value = Embedder.ModelName[embedder_model_value]
    cloud_cfg.embedder_model = enum_value

    cloud = instantiate(cloud_cfg)
    projection = get_cloud_projection(cloud)

    matrix = projection.projection_matrix
    if matrix.size == 0:
        return 0.0

    gram = matrix.T @ matrix
    identity = np.eye(gram.shape[0], dtype=matrix.dtype)
    skew = np.linalg.norm(gram - identity, ord="fro")
    return float(skew)


def summarize_run(run_dir: Path) -> Optional[RunSummary]:
    metadata = load_run_metadata(run_dir)
    if metadata is None:
        return None

    df = load_results_dataframe(run_dir)
    if df is None:
        return None

    try:
        mask = valid_rank_fraction_mask(df)
    except KeyError as exc:
        LOGGER.warning("%s in %s", exc, run_dir)
        return None

    filtered = df.loc[mask, PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
    if filtered.empty:
        LOGGER.warning(
            "No valid values for %s in %s", PROPOSAL_BOOK_RANK_FRACTION_COLUMN, run_dir
        )
        return None

    article_mean = float("nan")
    article_median = float("nan")
    coverage_mean = float("nan")
    coverage_median = float("nan")
    if PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN in df.columns:
        article_column = df[PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN]
        article_mask = article_column.notna() & (article_column != -1)
        if article_mask.any():
            article_values = article_column.loc[article_mask]
            article_mean = float(article_values.mean())
            article_median = float(article_values.median())

    if GAP_COVERAGE_FRACTION_COLUMN in df.columns:
        coverage_column = df[GAP_COVERAGE_FRACTION_COLUMN]
        coverage_mask = coverage_column.notna() & (coverage_column >= 0)
        if coverage_mask.any():
            coverage_values = coverage_column.loc[coverage_mask]
            coverage_mean = float(coverage_values.mean())
            coverage_median = float(coverage_values.median())

    projector_skew = _compute_projector_skew(run_dir)

    return RunSummary(
        run_path=run_dir,
        metadata=metadata,
        proposal_book_rank_fraction_mean=float(filtered.mean()),
        proposal_article_rank_fraction_mean=article_mean,
        proposal_article_rank_fraction_median=article_median,
        proposal_book_rank_fraction_std=float(filtered.std()),
        row_count=int(filtered.shape[0]),
        gap_coverage_fraction_mean=coverage_mean,
        gap_coverage_fraction_median=coverage_median,
        projector_skew=projector_skew,
    )


def collect_summaries(outputs_dir: Path) -> List[RunSummary]:
    summaries: List[RunSummary] = []
    for run_dir in find_run_directories(outputs_dir):
        summary = summarize_run(run_dir)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda s: s.run_path)
    return summaries

def write_csv(summaries: Iterable[RunSummary], outputs_dir: Path, output_file: Path) -> None:
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
                "projector_skew_fro": summary.projector_skew,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(
            [
                "proposal_book_rank_fraction_mean",
                "embedder_model",
                "denoiser",
                "drop_k",
                "llm_num_articles",
                "llm_num_books",
                "run",
            ],
            na_position="last",
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format="%.5f")
    LOGGER.info("Wrote %s", output_file)


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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.output_file.exists() and not args.overwrite:
        LOGGER.error(
            "Output file %s already exists. Use --overwrite to replace it.", args.output_file
        )
        raise SystemExit(1)

    summaries = collect_summaries(args.outputs_dir)
    if not summaries:
        LOGGER.warning("No summaries created from outputs in %s", args.outputs_dir)

    write_csv(summaries, args.outputs_dir, args.output_file)


if __name__ == "__main__":
    main()
