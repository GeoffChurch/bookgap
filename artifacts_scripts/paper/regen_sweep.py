"""The runs behind the paper's quantitative tables, regenerated from scratch.

Replaces the May 2026 gemma4_sweep runs the tables were built from. Those had
two faults: 58 prompts overflowed the 8192-token Ollama window and were
silently truncated, and the EmbeddingGemma vectors had been computed by an
environment that ran the model wrongly. Each cell here is built exactly as the
May sweep built it (``gemma4_quantitative_sweep._build_config``), with today's
defaults: prompts refused rather than truncated.

Cells, per embedder (GIST drop_k=10, EmbeddingGemma drop_k=394; NONE drops 0):

- the context ablation (Table III): LPP at (articles, books) in
  (0,1) (1,1) (0,5) (5,0) (5,5) (10,5)
- the denoiser comparison (Table II): NONE, CCP, CCLIP, LIP, ANTI_LDA,
  JAD_EQUAL at (10,5); LPP is the ablation's (10,5) cell
- GIST only, the radius sweep (Table VI) and the headline run: LPP (10,5) on
  all 77 topics at radius fractions 0.75 (the default), 0.5, 0.5625 and 1.0

``--topics`` picks the topics the table cells cover: ``random20``, the fixed
random 20 the paper uses (GIST's 10+5 LPP cell is still the 77-topic headline
run, which the tables read on the 20), or ``all`` 77, which narrows every
interval. Each set's runs live in their own directory, so a table never mixes
the two; the radius runs are all-topic either way and have one of their own.

A run's directory is named by its cell and a hash of everything that decides
its results (``run_identity``), so running the same cell again resumes it in
place: finished proposals are kept and generation continues from the next. A
cell is done once its run state is SCORED. A stop (``bookgap.jobs stop``) takes
effect between proposals, so a chain can be stopped and restarted at will and
loses at most the proposal in flight. Each run directory's ``provenance.json``
records the cell, the model's digest, and one entry per session that worked on
it: the command, the bookgap commit and whether the tree was dirty, every
installed package's version, the context window, and when it started.

Usage (from the bookgap root; one chain at a time, since chains sharing the GPU
slow each other)::

    uv run python -m bookgap.jobs run --prefix regen_gist -- \\
        uv run python -m artifacts_scripts.paper.regen_sweep --embedder gist --topics random20
    uv run python -m artifacts_scripts.paper.regen_sweep --embedder gemma --topics all --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata as md
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ollama
import yaml
from omegaconf import OmegaConf

from artifacts_scripts.gemma4_quantitative_sweep import TOPIC_SUBSET_RANDOM_20, _build_config
from bookgap import jobs
from bookgap.configs.config import ExperimentConfig
from bookgap.llm.generators.run_state import RunState, RunStateManager, RunStep
from bookgap.llm.generators.workflow_functions import run_batch_workflow

REGEN_ROOT = Path("outputs/regen_2026-10")
# The table cells of each topic set, and the radius runs (all-topic in both sets).
TOPIC_SET_ROOTS = {"random20": REGEN_ROOT / "runs", "all": REGEN_ROOT / "runs_t77"}
RADIUS_ROOT = REGEN_ROOT / "radius_runs"

EMBEDDERS = {
    "gist": ("avsolatorio/GIST-small-Embedding-v0", 10),
    "gemma": ("google/embeddinggemma-300m", 394),
}
ABLATION = ((10, 5), (0, 1), (1, 1), (0, 5), (5, 0), (5, 5))
DENOISERS = ("NONE", "CCP", "CCLIP", "LIP", "ANTI_LDA", "JAD_EQUAL")
RADII = (0.5, 0.5625, 1.0)
DEFAULT_RADIUS = 0.75


@dataclass(frozen=True)
class Cell:
    embedder: str          # key of EMBEDDERS
    denoiser: str
    articles: int
    books: int
    all_topics: bool       # all 77 topics, else TOPIC_SUBSET_RANDOM_20
    radius_fraction: float
    root: Path             # the directory this cell's run lives under

    @property
    def drop_k(self) -> int:
        return 0 if self.denoiser == "NONE" else EMBEDDERS[self.embedder][1]

    @property
    def slug(self) -> str:
        topics = "t77" if self.all_topics else "t20"
        return (f"{self.embedder}_a{self.articles}_b{self.books}_dropk{self.drop_k}_"
                f"{self.denoiser.lower()}_{topics}_r{self.radius_fraction:g}")


def cells(embedder: str, *, topics: str) -> list[Cell]:
    all_topics = topics == "all"
    root = TOPIC_SET_ROOTS[topics]
    headline = embedder == "gist"
    out = [Cell(embedder, "LPP", a, b, all_topics or (headline and (a, b) == (10, 5)), DEFAULT_RADIUS, root)
           for a, b in ABLATION]
    out += [Cell(embedder, d, 10, 5, all_topics, DEFAULT_RADIUS, root) for d in DENOISERS]
    if headline:
        out += [Cell(embedder, "LPP", 10, 5, True, r, RADIUS_ROOT) for r in RADII]
    return out


def config_for(cell: Cell) -> ExperimentConfig:
    config = _build_config(
        EMBEDDERS[cell.embedder][0], cell.denoiser, cell.drop_k, cell.articles, cell.books,
        topic_subset=None if cell.all_topics else TOPIC_SUBSET_RANDOM_20,
    )
    return dataclasses.replace(config, gap=dataclasses.replace(config.gap, radius_fraction=cell.radius_fraction))


def run_identity(config_yaml: str, model_digest: str) -> str:
    """A short hash of everything that decides a run's results: its resolved
    configuration and the Ollama model's weights, by digest.

    ``llm.ollama_num_ctx`` is left out. The generator refuses any prompt that does
    not fit the window and fails any reply that runs past it, so the window
    decides only whether a proposal is made, never what it says; the paper's runs
    used 16,384 and 65,536 tokens to the same effect. Takes the configuration as
    YAML so a run on disk and a cell built in code hash alike.
    """
    resolved = yaml.safe_load(config_yaml)
    del resolved["llm"]["ollama_num_ctx"]
    blob = json.dumps({"config": resolved, "ollama_model_digest": model_digest}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def run_dir(cell: Cell, config_yaml: str, model_digest: str) -> Path:
    return cell.root / f"{cell.slug}-{run_identity(config_yaml, model_digest)}"


def is_scored(out: Path) -> bool:
    # Read the state file directly: constructing a RunStateManager creates one.
    state = out / "run_state.json"
    return state.exists() and RunState.from_dict(json.loads(state.read_text())).step == RunStep.SCORED


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _session(config: ExperimentConfig, stamp: str) -> dict:
    return {
        "command": [sys.executable, *sys.argv],
        "bookgap_commit": _git("rev-parse", "HEAD"),
        "bookgap_dirty": _git("status", "--porcelain", "--untracked-files=no"),
        "packages": dict(sorted((d.metadata["Name"].lower(), d.version) for d in md.distributions())),
        "ollama_num_ctx": config.llm.ollama_num_ctx,
        "started": stamp,
    }


def run(cell: Cell, config: ExperimentConfig, config_yaml: str, model_digest: str) -> None:
    out = run_dir(cell, config_yaml, model_digest)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out / ".hydra").mkdir(parents=True, exist_ok=True)
    for path in (out / "experiment_config.yaml", out / ".hydra" / "config.yaml"):
        path.write_text(config_yaml)
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
    else:
        provenance = {"cell": {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cell).items()},
                      "ollama_model": config.llm.ollama_model, "ollama_model_digest": model_digest, "sessions": []}
    provenance["sessions"].append(_session(config, stamp))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    log = logging.getLogger(__name__)
    log.info("cell %s -> %s (session %d)", cell.slug, out, len(provenance["sessions"]))
    t0 = time.time()
    results = run_batch_workflow(config, out, RunStateManager(out))
    log.info("cell %s done: %d proposals in %.1f min", cell.slug, len(results.proposals), (time.time() - t0) / 60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", choices=sorted(EMBEDDERS), required=True)
    ap.add_argument("--topics", choices=sorted(TOPIC_SET_ROOTS), required=True,
                    help="the topics the table cells cover: the fixed random 20, or all 77")
    ap.add_argument("--dry-run", action="store_true",
                    help="list each cell's run directory and whether it is done, and run nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    log = logging.getLogger(__name__)
    todo = cells(args.embedder, topics=args.topics)
    digests = {m.model: m.digest for m in ollama.list().models}
    # Under `bookgap.jobs run` this process is the wrapper's child: without its own
    # handlers a stop would kill it mid-proposal instead of between proposals.
    with jobs.stop_signals():
        try:
            for i, cell in enumerate(todo, start=1):
                config = config_for(cell)
                config_yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
                out = run_dir(cell, config_yaml, digests[config.llm.ollama_model])
                status = "done" if is_scored(out) else "partial" if out.exists() else "new"
                log.info("cell %d/%d %s: %s %s", i, len(todo), cell.slug, status, out)
                if status == "done" or args.dry_run:
                    continue
                run(cell, config, config_yaml, digests[config.llm.ollama_model])
        except KeyboardInterrupt as stop:
            log.info("stopped (%s); rerun the same command to resume", stop)
            return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
