"""Radius-fraction sweep for the §5.3 ``tab:radius_sweep`` table.

Runs three additional values of ``gap.radius_fraction`` so the §5.3 table
can be regenerated against the current production pipeline (GIST + LPP +
``gemma4:31b`` at 10 articles + 5 books, on all 77 AI topics). The
fourth cell, ``radius_fraction=0.75``, already exists at
``outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10/`` and
needs no re-run.

After each cell finishes, the lenient-recovery pass is applied so that
strict-parse failures (typically due to gemma4 producing a JSON array
wrapper instead of a single object) are recovered and their coverage
metrics added to ``gaps_analysis_results.csv`` — same pattern as the
p=0.75 cell.

Recommended invocation::

    uv run python -m bookgap.jobs run --prefix radius_sweep -- \\
        uv run python -m artifacts_scripts.radius_sweep

Wrapped in ``bookgap.jobs`` so the multi-hour sequential run survives
this session. Skip-if-done so re-invocation is safe.
"""
from __future__ import annotations

import datetime
import glob
import subprocess
import sys
import time
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.configs.config import ExperimentConfig
from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.util.filter_config import FilterConfig, FilterPairConfig
from bookgap.llm.generators.run_state import RunStateManager
from bookgap.llm.generators.workflow_functions import run_batch_workflow
from bookgap.llm.protocols import LLMBackend
from bookgap.util import LOGGER, SubField, WorkType


# Three new cells. p=0.75 already exists; we don't re-run it.
RADIUS_FRACTIONS = (0.5, 0.5625, 1.0)


def _slug(p: float) -> str:
    # 0.5 → p0.5, 0.5625 → p0.5625, 1.0 → p1.0
    return f"p{p}"


def _output_dir_for(p: float) -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / "gemma4_sweep" / (
        f"{stamp}_gist_a10_b5_dropk10_radius{_slug(p)}"
    )


def _cell_already_done(p: float) -> Path | None:
    pattern = (
        f"outputs/gemma4_sweep/*_gist_a10_b5_dropk10_radius{_slug(p)}/"
        f"gaps_analysis_results.csv"
    )
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]).parent if matches else None


def _build_config(p: float) -> ExperimentConfig:
    return ExperimentConfig(
        gap=GapConfig(radius_fraction=p),
        llm=LLMConfig(
            backend=LLMBackend.OLLAMA_LOCAL,
            num_articles=10,
            num_books=5,
        ),
        batch=BatchConfig(num_gaps=1),
        output=OutputConfig(),
        cloud=CloudConfig(
            filter_pair=FilterPairConfig(
                pre=FilterConfig(
                    subfield=SubField.AI.value,
                    work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value),
                ),
                post=FilterConfig(),
            ),
            denoiser="LPP",
            drop_k=10,
            embedder=SentenceTransformerEmbedderConfig(
                model_name="avsolatorio/GIST-small-Embedding-v0",
            ),
        ),
    )


def _run_one_cell(idx: int, total: int, p: float) -> Path:
    output_dir = _output_dir_for(p)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 68)
    LOGGER.info("Radius cell %d/%d  radius_fraction=%s", idx, total, p)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("=" * 68)

    config = _build_config(p)

    # Persist as both `experiment_config.yaml` (what the workflow itself
    # writes from Hydra) and `.hydra/config.yaml` (what
    # artifacts_scripts.common.find_run_directories scans). Same dual-write
    # pattern as gemma4_quantitative_sweep.py.
    structured = OmegaConf.structured(config)
    OmegaConf.save(structured, output_dir / "experiment_config.yaml")
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(structured, hydra_dir / "config.yaml")

    state_manager = RunStateManager(output_dir)
    results = run_batch_workflow(config, output_dir, state_manager)
    LOGGER.info("Radius cell %d/%d done  proposals=%d",
                idx, total, len(results.proposals))
    return output_dir


def _lenient_recover(run_dir: Path) -> None:
    """Run the lenient-recovery + post-loop CSV rebuild on the freshly
    completed cell. Same pattern as the p=0.75 cell — recovers
    array-wrapped outputs that the strict parser rejected.
    """
    LOGGER.info("Running lenient recovery on %s", run_dir)
    cmd = [
        sys.executable,
        "-m", "artifacts_scripts.lenient_recover_failed_proposals",
        "--run-dir", str(run_dir),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.error("Lenient recovery on %s exited %d (continuing)",
                     run_dir, result.returncode)


def main() -> None:
    total = len(RADIUS_FRACTIONS)
    started = time.time()
    for i, p in enumerate(RADIUS_FRACTIONS, start=1):
        existing = _cell_already_done(p)
        if existing is not None:
            LOGGER.info(
                "Radius cell %d/%d (p=%s) already complete at %s — skipping",
                i, total, p, existing,
            )
            continue
        cell_started = time.time()
        try:
            run_dir = _run_one_cell(i, total, p)
            _lenient_recover(run_dir)
        except Exception:
            LOGGER.error("Radius cell %d/%d FAILED — continuing to next",
                         i, total)
            LOGGER.error("Traceback:\n%s", traceback.format_exc())
            continue
        cell_elapsed = time.time() - cell_started
        total_elapsed = time.time() - started
        LOGGER.info("Radius cell %d/%d elapsed: %.1f min  (total: %.1f min)",
                    i, total, cell_elapsed / 60, total_elapsed / 60)
    LOGGER.info("=" * 68)
    LOGGER.info("Radius sweep complete. Total wall time: %.1f min",
                (time.time() - started) / 60)
    LOGGER.info("=" * 68)


if __name__ == "__main__":
    main()
