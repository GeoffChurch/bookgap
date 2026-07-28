"""Dispatch a batch of bake-off cells from a JSON spec.

Reads a JSON file containing a list of [embedder, denoiser, drop_k] triples
and runs each through `gemma4_quantitative_sweep._run_one_cell`. Honors the
existing skip-guard (cells with a completed `gaps_analysis_results.csv` are
skipped silently).

Usage:
    python -m artifacts_scripts.run_cells_from_json cells_to_run.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from bookgap.util import LOGGER

# Reuse the existing single-cell runner (+ skip guard) so we don't
# duplicate config plumbing.
from artifacts_scripts.gemma4_quantitative_sweep import (
    _cell_already_done, _run_one_cell,
)


# Headline config — mirrors gemma4_quantitative_sweep.py bake-off cells.
NUM_ARTICLES = 10
NUM_BOOKS = 5


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cells_path = Path(sys.argv[1])
    raw = json.loads(cells_path.read_text())
    cells: list[tuple[str, str, int]] = [tuple(c) for c in raw]

    total = len(cells)
    started = time.time()
    LOGGER.info("Dispatching %d cells from %s", total, cells_path)

    for i, (embedder, denoiser, drop_k) in enumerate(cells, start=1):
        if _cell_already_done(embedder, denoiser, drop_k, NUM_ARTICLES, NUM_BOOKS):
            LOGGER.info(
                "Cell %d/%d (%s denoiser=%s drop_k=%d) already complete — skipping",
                i, total, embedder, denoiser, drop_k,
            )
            continue
        cell_started = time.time()
        try:
            _run_one_cell(i, total, embedder, denoiser, drop_k, NUM_ARTICLES, NUM_BOOKS)
        except Exception:
            LOGGER.error("Cell %d/%d FAILED — continuing", i, total)
            LOGGER.error("Traceback:\n%s", traceback.format_exc())
            continue
        elapsed = time.time() - cell_started
        total_elapsed = time.time() - started
        LOGGER.info("Cell %d/%d elapsed: %.1f min  (total so far: %.1f min)",
                    i, total, elapsed / 60, total_elapsed / 60)

    LOGGER.info("=" * 68)
    LOGGER.info("Batch complete. Total wall time: %.1f min", (time.time() - started) / 60)
    LOGGER.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
