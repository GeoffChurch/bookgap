#!/usr/bin/env python3
"""Batch Proposal Generator - resumable entry point for LLM proposals."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.config_store import register_configs
from bookgap.llm.generators.run_state import RunStateManager
from bookgap.llm.generators.workflow_functions import run_batch_workflow
from bookgap.util import LOGGER


# Ensure configs are registered
register_configs()


def _parse_cli_args(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """Extract resume arguments while leaving Hydra overrides intact."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-output-dir", type=str, default=None)
    args, unknown = parser.parse_known_args(argv[1:])
    return args.resume_output_dir, [argv[0], *unknown]


def _ensure_unique_path(path: Path) -> Path:
    """Return a path with an added numeric suffix if the original exists."""

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _copy_resume_artifacts(source: Path, destination: Path) -> None:
    """Copy prior run artifacts into the new Hydra output directory."""

    if not source.exists():
        raise FileNotFoundError(f"Resume directory not found: {source}")

    LOGGER.info("Copying artifacts from %s into %s", source, destination)

    for item in source.iterdir():
        if item.name == ".hydra":
            target = _ensure_unique_path(destination / ".hydra_OLD")
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.name == "batch_generator.log":
            target = _ensure_unique_path(destination / "batch_generator_OLD.log")
            shutil.copy2(item, target)
        else:
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)


def _persist_config(cfg: ExperimentConfig, output_dir: Path) -> Path:
    """Write the structured experiment configuration to disk for auditing."""

    config_path = output_dir / "experiment_config.yaml"
    structured = OmegaConf.structured(cfg)
    OmegaConf.save(structured, config_path)
    return config_path


def _run_workflow(cfg: ExperimentConfig, resume_output_dir: Optional[Path]) -> None:
    """Execute the batch workflow using the provided configuration."""

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Log startup information
    LOGGER.info("=" * 60)
    LOGGER.info("BookGap Batch Proposal Generator")
    LOGGER.info("=" * 60)
    LOGGER.info("Working directory: %s", os.getcwd())
    LOGGER.info("Output directory: %s", output_dir)
    if resume_output_dir:
        LOGGER.info("Resuming from: %s", resume_output_dir)
    LOGGER.info("")

    if resume_output_dir:
        _copy_resume_artifacts(Path(resume_output_dir), output_dir)

    config_path = _persist_config(cfg, output_dir)
    LOGGER.info("Experiment configuration saved to %s", config_path)

    state_manager = RunStateManager(output_dir)
    LOGGER.info(
        "Loaded run state from %s (step=%s)",
        state_manager.state_path,
        state_manager.state.step.value,
    )

    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Execute workflow
    results = run_batch_workflow(cfg, output_dir, state_manager)

    # Log completion
    LOGGER.info("=" * 60)
    LOGGER.info("WORKFLOW COMPLETED SUCCESSFULLY")
    LOGGER.info("=" * 60)
    LOGGER.info("Generated proposals: %s", len(results.proposals))
    LOGGER.info("Output directory: %s", output_dir)


def main() -> None:
    """Parse CLI arguments, then delegate to the Hydra entry point."""

    original_argv = sys.argv[:]
    resume_output_dir, hydra_argv = _parse_cli_args(sys.argv)
    resume_path = Path(resume_output_dir).resolve() if resume_output_dir else None

    @hydra.main(version_base=None, config_path="configs", config_name="standard")
    def _hydra_entry(cfg: ExperimentConfig) -> None:
        _run_workflow(cfg, resume_path)

    try:
        sys.argv = hydra_argv
        _hydra_entry()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
