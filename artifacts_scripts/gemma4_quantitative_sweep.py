"""12-cell gemma4-quantitative-revision sweep, run as a single tracked job.

Replaces the §5.1/§5.2/§5.4 GPT-5+CCLIP numbers in the paper with
LPP+gemma4:31b throughout. Each cell uses plain LPP, AI subfield,
book+article work types, and the gemma4:31b proposal model. Varying:
embedder ({GIST, Gemma}) × (num_articles, num_books) in 6 combos.

First Gemma cell triggers the embedding cache build (~1-2 hr); the
subsequent 5 Gemma cells reuse the cache. Total ETA: ~10-14 hr.

Programmatic driver (not Hydra CLI) because `SentenceTransformerEmbedderConfig`
is frozen — required for Cloud's hash key — so Hydra CLI cannot override
`cloud.embedder.model_name`. Programmatic construction sidesteps this.

Plan: docs/superpowers/plans/2026-05-21-paper-gemma4-quantitative-revision.md
"""
from __future__ import annotations

import datetime
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


# Fixed random-20 topic subset (seed=42 over the AI subfield's 77 topics).
# Used for rapid iteration: each cell runs gap-detection + proposal-generation
# on only these 20 topics instead of all 77, ~74% time savings per cell.
# Comparisons against the 77-topic GIST LPP cells should filter those CSVs to
# this same subset at table-build time.
TOPIC_SUBSET_RANDOM_20 = (
    "T10028", "T10181", "T10201", "T10639", "T10664", "T10682", "T10711",
    "T10862", "T11424", "T11512", "T11598", "T11612", "T11901", "T12026",
    "T12262", "T13062", "T13904", "T14351", "T14381", "T14413",
)


# NOTE: The 6 LPP+GIST cells from the prior 77-topic sweep are NOT listed
# here. Their CSVs (full 77 topics) live under outputs/gemma4_sweep/
# *_gist_*_dropk10/ and are picked up at table-build time, filtered to
# the TOPIC_SUBSET_RANDOM_20 set above for apples-to-apples comparison
# with the cells below (which run on 20 topics).
CELLS = [
    # (embedder_model, denoiser, drop_k, num_articles, num_books)
    #
    # LPP / Gemma cells (drop_k=394 → retain 374 of 768, matching GIST's
    # retained dimensionality so the §5.4 comparison is at equal subspace size).
    ("google/embeddinggemma-300m",          "LPP", 394, 0, 1),
    ("google/embeddinggemma-300m",          "LPP", 394, 1, 1),
    ("google/embeddinggemma-300m",          "LPP", 394, 0, 5),
    ("google/embeddinggemma-300m",          "LPP", 394, 5, 0),
    ("google/embeddinggemma-300m",          "LPP", 394, 5, 5),
    ("google/embeddinggemma-300m",          "LPP", 394, 10, 5),
    # NONE / GIST baseline (no denoising; drop_k must be 0 for NoneDenoiser).
    # Populates the no-denoising column for the appendix bake-off tables.
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 0, 1),
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 1, 1),
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 0, 5),
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 5, 0),
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 5, 5),
    ("avsolatorio/GIST-small-Embedding-v0", "NONE", 0, 10, 5),
    # NONE / Gemma baseline.
    ("google/embeddinggemma-300m",          "NONE", 0, 0, 1),
    ("google/embeddinggemma-300m",          "NONE", 0, 1, 1),
    ("google/embeddinggemma-300m",          "NONE", 0, 0, 5),
    ("google/embeddinggemma-300m",          "NONE", 0, 5, 0),
    ("google/embeddinggemma-300m",          "NONE", 0, 5, 5),
    ("google/embeddinggemma-300m",          "NONE", 0, 10, 5),
]


def _build_config(
    embedder_model: str, denoiser: str, drop_k: int, num_articles: int, num_books: int,
    topic_subset=None,
) -> ExperimentConfig:
    return ExperimentConfig(
        gap=GapConfig(topic_subset=topic_subset),
        llm=LLMConfig(
            backend=LLMBackend.OLLAMA_LOCAL,
            num_articles=num_articles,
            num_books=num_books,
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
            denoiser=denoiser,
            drop_k=drop_k,
            embedder=SentenceTransformerEmbedderConfig(model_name=embedder_model),
        ),
    )


def _slug(embedder_model: str) -> str:
    # avsolatorio/GIST-small-Embedding-v0 → gist
    # google/embeddinggemma-300m           → gemma
    if "GIST" in embedder_model or "gist" in embedder_model:
        return "gist"
    if "gemma" in embedder_model.lower():
        return "gemma"
    return embedder_model.split("/")[-1]


def _run_one_cell(idx: int, total: int, embedder_model: str, denoiser: str,
                  drop_k: int, num_articles: int, num_books: int) -> None:
    slug = _slug(embedder_model)
    # Pre-existing LPP cells were named without a denoiser segment; preserve
    # that path so the skip-guard finds them. NONE (and any future denoiser)
    # gets the segment added so cells with the same (slug, a, b, drop_k) but
    # different denoiser don't collide. Add a topic_subset segment so 77-topic
    # full runs don't collide with restricted-subset runs.
    denoiser_segment = "" if denoiser == "LPP" else f"_{denoiser.lower()}"
    subset_segment = "" if TOPIC_SUBSET_RANDOM_20 is None else f"_t{len(TOPIC_SUBSET_RANDOM_20)}"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / "gemma4_sweep" / (
        f"{stamp}_{slug}_a{num_articles}_b{num_books}_dropk{drop_k}{denoiser_segment}{subset_segment}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 68)
    LOGGER.info("Cell %d/%d  embedder=%s  denoiser=%s  drop_k=%d  articles=%d  books=%d",
                idx, total, embedder_model, denoiser, drop_k, num_articles, num_books)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("=" * 68)

    config = _build_config(
        embedder_model, denoiser, drop_k, num_articles, num_books,
        topic_subset=TOPIC_SUBSET_RANDOM_20,
    )

    # Persist the config both as the "experiment_config.yaml" the workflow
    # would have written and as ".hydra/config.yaml" — the latter is what
    # artifacts_scripts.common.find_run_directories scans for, so writing
    # it here lets the LaTeX table-builder discover programmatic-driver
    # runs the same way it discovers Hydra-CLI runs.
    structured = OmegaConf.structured(config)
    OmegaConf.save(structured, output_dir / "experiment_config.yaml")
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(structured, hydra_dir / "config.yaml")

    state_manager = RunStateManager(output_dir)
    results = run_batch_workflow(config, output_dir, state_manager)
    LOGGER.info("Cell %d/%d done  proposals=%d", idx, total, len(results.proposals))


def _cell_already_done(embedder: str, denoiser: str, drop_k: int,
                       articles: int, books: int) -> bool:
    """Skip-if-done guard: True iff a sweep cell dir for this config exists
    AND it has a final ``gaps_analysis_results.csv``.

    Lets the script be re-invoked safely (e.g. after a kill+restart, or after
    extending CELLS with new cells like NONE-denoiser variants) without
    re-running the cells that already finished.
    """
    slug = _slug(embedder)
    denoiser_segment = "" if denoiser == "LPP" else f"_{denoiser.lower()}"
    subset_segment = "" if TOPIC_SUBSET_RANDOM_20 is None else f"_t{len(TOPIC_SUBSET_RANDOM_20)}"
    pattern = (
        f"outputs/gemma4_sweep/*_{slug}_a{articles}_b{books}_dropk{drop_k}"
        f"{denoiser_segment}{subset_segment}/gaps_analysis_results.csv"
    )
    import glob as _glob
    return bool(_glob.glob(pattern))


def main() -> None:
    total = len(CELLS)
    started = time.time()
    for i, (embedder, denoiser, drop_k, articles, books) in enumerate(CELLS, start=1):
        if _cell_already_done(embedder, denoiser, drop_k, articles, books):
            LOGGER.info(
                "Cell %d/%d (%s denoiser=%s a=%d b=%d drop_k=%d) already complete — skipping",
                i, total, embedder, denoiser, articles, books, drop_k,
            )
            continue
        cell_started = time.time()
        try:
            _run_one_cell(i, total, embedder, denoiser, drop_k, articles, books)
        except Exception:
            LOGGER.error("Cell %d/%d FAILED — continuing to next cell", i, total)
            LOGGER.error("Traceback:\n%s", traceback.format_exc())
            continue
        cell_elapsed = time.time() - cell_started
        total_elapsed = time.time() - started
        LOGGER.info("Cell %d/%d elapsed: %.1f min  (total so far: %.1f min)",
                    i, total, cell_elapsed / 60, total_elapsed / 60)
    LOGGER.info("=" * 68)
    LOGGER.info("Sweep complete. Total wall time: %.1f min", (time.time() - started) / 60)
    LOGGER.info("=" * 68)


if __name__ == "__main__":
    main()
