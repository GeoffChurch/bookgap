"""In-process multi-year driver for historical gap analysis.

Replaces Hydra ``--multirun year=...``, which forks a fresh Python process
per year and pays ~10-30 s of module-import + Cloud-reinstantiation overhead
each time. This driver builds the experiment config once and loops over
years in the same process, calling :func:`_run_analysis` directly.

Joblib caches still kick in (per-year ``cloud.works()``, denoiser projection,
etc.) but staying in one process saves the per-year process-spawn tax.

Usage:
    uv run python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis_sweep
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

from pathlib import Path

from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.configs.config import ExperimentConfig
from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.util.filter_config import FilterConfig, FilterPairConfig
from bookgap.util import LOGGER, SubField, WorkType, create_unicode_safe_filename

from .config import HistoricalAnalysisConfig
from .historical_gap_analysis import _run_analysis


def _expected_summary_path(year: int, embedder_model: str, denoiser: str,
                            drop_k: int, radius_fraction: float) -> Path:
    """Reproduce the per-year output filename used by ``_run_analysis`` so we
    can skip years that already have a complete summary CSV on disk."""
    results_dir = Path(__file__).parent / "results"
    safe_embedder = create_unicode_safe_filename(embedder_model)
    safe_denoiser = create_unicode_safe_filename(denoiser)
    return results_dir / (
        f"historical_gap_analysis_summary_year{year}_"
        f"p{radius_fraction:.2f}_{safe_embedder}_{safe_denoiser}_drop{drop_k}.csv"
    )


def _build_experiment(denoiser: str, drop_k: int, radius_fraction: float) -> ExperimentConfig:
    return ExperimentConfig(
        gap=GapConfig(radius_fraction=radius_fraction),
        llm=LLMConfig(),
        batch=BatchConfig(),
        output=OutputConfig(),
        cloud=CloudConfig(
            filter_pair=FilterPairConfig(
                pre=FilterConfig(
                    subfield=SubField.AI.value,
                    work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
                ),
                post=FilterConfig(),
            ),
            denoiser=denoiser,
            drop_k=drop_k,
            embedder=SentenceTransformerEmbedderConfig(
                model_name="avsolatorio/GIST-small-Embedding-v0",
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-year", type=int, default=1990)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--denoiser", type=str, default="LPP")
    parser.add_argument("--drop-k", type=int, default=10)
    parser.add_argument("--radius-fraction", type=float, default=0.75)
    args = parser.parse_args()

    experiment = _build_experiment(args.denoiser, args.drop_k, args.radius_fraction)
    years = list(range(args.max_year, args.min_year - 1, -1))  # descending: newer first

    LOGGER.info("=" * 68)
    LOGGER.info("In-process historical sweep: %d years (%d..%d), %s drop_k=%d rf=%s",
                len(years), args.min_year, args.max_year,
                args.denoiser, args.drop_k, args.radius_fraction)
    LOGGER.info("=" * 68)

    embedder_model = experiment.cloud.embedder.model_name

    started = time.time()
    for i, year in enumerate(years, start=1):
        expected = _expected_summary_path(
            year, embedder_model, args.denoiser, args.drop_k, args.radius_fraction,
        )
        if expected.exists():
            LOGGER.info(">>> Year %d/%d: %d — already complete, skipping (%s)",
                        i, len(years), year, expected.name)
            continue
        year_started = time.time()
        LOGGER.info("")
        LOGGER.info(">>> Year %d/%d: %d", i, len(years), year)
        cfg = HistoricalAnalysisConfig(year=year, experiment=experiment)
        try:
            _run_analysis(cfg)
        except Exception:
            LOGGER.error("Year %d FAILED — continuing", year)
            LOGGER.error(traceback.format_exc())
            continue
        year_elapsed = time.time() - year_started
        total_elapsed = time.time() - started
        LOGGER.info(">>> Year %d done in %.1fs (cumulative: %.1fmin)",
                    year, year_elapsed, total_elapsed / 60)

    total = time.time() - started
    LOGGER.info("=" * 68)
    LOGGER.info("Sweep complete. %d years in %.1f min (avg %.1f s/year)",
                len(years), total / 60, total / len(years))
    LOGGER.info("=" * 68)


if __name__ == "__main__":
    main()
