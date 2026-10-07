"""The 27 runs behind the paper's quantitative tables, regenerated from scratch.

Replaces the May 2026 gemma4_sweep runs the tables were built from. Those had
two faults: 58 prompts overflowed the 8192-token Ollama window and were
silently truncated, and the EmbeddingGemma vectors had been computed by an
environment that ran the model wrongly. Each cell here is built exactly as the
May sweep built it (``gemma4_quantitative_sweep._build_config``), with today's
defaults: a 16384-token window and prompts refused rather than truncated.

Cells, per embedder (GIST drop_k=10, EmbeddingGemma drop_k=394; NONE drops 0):

- the context ablation, Table I: LPP at (articles, books) in
  (0,1) (1,1) (0,5) (5,0) (5,5) (10,5)
- the denoiser comparison, Table II: NONE, CCP, CCLIP, LIP, ANTI_LDA,
  JAD_EQUAL at (10,5); LPP is the ablation's (10,5) cell
- GIST only, the radius sweep and the headline run: LPP (10,5) on all 77 topics
  at radius fractions 0.75 (the default), 0.5, 0.5625 and 1.0

Everything else runs on the fixed random-20 topic subset, as in May. GIST's
(10,5) LPP cell is the 77-topic headline run, which the tables read on the
20-topic subset. Each run directory gets ``provenance.json``: the command, the
bookgap commit and whether the tree was dirty, every installed package's
version, and the Ollama model's digest and window.

Usage (from the bookgap root; one chain per embedder, run side by side)::

    uv run python -m bookgap.jobs run --prefix regen_gist -- \\
        uv run python -m artifacts_scripts.paper.regen_sweep --embedder gist
    uv run python -m bookgap.jobs run --prefix regen_gemma -- \\
        uv run python -m artifacts_scripts.paper.regen_sweep --embedder gemma

A cell whose directory already holds a completed run is skipped, so a chain
can be restarted after a failure.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.metadata as md
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ollama
from omegaconf import OmegaConf

from artifacts_scripts.gemma4_quantitative_sweep import TOPIC_SUBSET_RANDOM_20, _build_config
from bookgap.llm.generators.run_state import RunStateManager
from bookgap.llm.generators.workflow_functions import run_batch_workflow

OUTPUT_ROOT = Path("outputs/regen_2026-10/runs")

EMBEDDERS = {
    "gist": ("avsolatorio/GIST-small-Embedding-v0", 10),
    "gemma": ("google/embeddinggemma-300m", 394),
}
ABLATION = ((10, 5), (0, 1), (1, 1), (0, 5), (5, 0), (5, 5))
DENOISERS = ("NONE", "CCP", "CCLIP", "LIP", "ANTI_LDA", "JAD_EQUAL")
RADII = (0.5, 0.5625, 1.0)


@dataclass(frozen=True)
class Cell:
    embedder: str          # key of EMBEDDERS
    denoiser: str
    articles: int
    books: int
    all_topics: bool       # all 77 topics, else TOPIC_SUBSET_RANDOM_20
    radius_fraction: float

    @property
    def drop_k(self) -> int:
        return 0 if self.denoiser == "NONE" else EMBEDDERS[self.embedder][1]

    @property
    def slug(self) -> str:
        topics = "t77" if self.all_topics else "t20"
        return (f"{self.embedder}_a{self.articles}_b{self.books}_dropk{self.drop_k}_"
                f"{self.denoiser.lower()}_{topics}_r{self.radius_fraction:g}")


def cells(embedder: str) -> list[Cell]:
    headline = embedder == "gist"
    out = [Cell(embedder, "LPP", a, b, headline and (a, b) == (10, 5), 0.75) for a, b in ABLATION]
    out += [Cell(embedder, d, 10, 5, False, 0.75) for d in DENOISERS]
    if headline:
        out += [Cell(embedder, "LPP", 10, 5, True, r) for r in RADII]
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _completed(cell: Cell) -> Path | None:
    for run_dir in sorted(OUTPUT_ROOT.glob(f"*_{cell.slug}")):
        state = run_dir / "run_state.json"
        if state.exists() and json.loads(state.read_text())["step"] == "COMPLETED":
            return run_dir
    return None


def run(cell: Cell) -> None:
    model_name = EMBEDDERS[cell.embedder][0]
    config = _build_config(
        model_name, cell.denoiser, cell.drop_k, cell.articles, cell.books,
        topic_subset=None if cell.all_topics else TOPIC_SUBSET_RANDOM_20,
    )
    config = dataclasses.replace(config, gap=dataclasses.replace(config.gap, radius_fraction=cell.radius_fraction))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = OUTPUT_ROOT / f"{stamp}_{cell.slug}"
    (out / ".hydra").mkdir(parents=True)
    structured = OmegaConf.structured(config)
    OmegaConf.save(structured, out / "experiment_config.yaml")
    OmegaConf.save(structured, out / ".hydra" / "config.yaml")
    model = next(m for m in ollama.list().models if m.model == config.llm.ollama_model)
    (out / "provenance.json").write_text(json.dumps({
        "command": [sys.executable, *sys.argv],
        "cell": dataclasses.asdict(cell),
        "bookgap_commit": _git("rev-parse", "HEAD"),
        "bookgap_dirty": _git("status", "--porcelain", "--untracked-files=no"),
        "packages": dict(sorted((d.metadata["Name"].lower(), d.version) for d in md.distributions())),
        "ollama_model": config.llm.ollama_model,
        "ollama_model_digest": model.digest,
        "ollama_num_ctx": config.llm.ollama_num_ctx,
        "started": stamp,
    }, indent=2) + "\n")

    log = logging.getLogger(__name__)
    log.info("cell %s -> %s", cell.slug, out)
    t0 = time.time()
    results = run_batch_workflow(config, out, RunStateManager(out))
    log.info("cell %s done: %d proposals in %.1f min", cell.slug, len(results.proposals), (time.time() - t0) / 60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", choices=sorted(EMBEDDERS), required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    todo = cells(args.embedder)
    for i, cell in enumerate(todo, start=1):
        done = _completed(cell)
        if done is not None:
            logging.getLogger(__name__).info("cell %d/%d %s already complete at %s", i, len(todo), cell.slug, done)
            continue
        logging.getLogger(__name__).info("cell %d/%d %s", i, len(todo), cell.slug)
        run(cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
