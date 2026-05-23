"""One-off: replay gap detection on an existing run and write
``contexts/*.json`` + ``prompts/*.json`` so old runs match the
post-2026-05-23 artifact layout. Slated for deletion after the headline
run and any other operator-selected runs have been backfilled and
verified.

Usage:

    conda run -p .conda python -m artifacts_scripts.backfill_prompts_and_contexts \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import typing
from pathlib import Path

from omegaconf import OmegaConf

from bookgap.configs.config import ExperimentConfig
from bookgap.gaps import compute_gaps_for_config
from bookgap.llm.artifacts.context_export import generate_all_contexts
from bookgap.llm.generators.workflow_functions import (
    build_contexts_for_gaps, build_prompt_records, generate_all_prompts,
)


def _build_dataclass(cls, value):
    """Construct a frozen dataclass from a plain dict.

    OmegaConf's structured-merge refuses frozen dataclasses with
    init=False fields (e.g. SentenceTransformerEmbedderConfig). This
    helper walks the type tree manually, instantiating each nested
    dataclass from the corresponding dict.
    """
    if value is None:
        return value
    if isinstance(cls, type) and issubclass(cls, enum.Enum) and isinstance(value, str):
        return cls[value] if value in cls.__members__ else cls(value)
    if not dataclasses.is_dataclass(cls):
        return value
    if not isinstance(value, dict):
        return value
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in value:
            continue
        ftype = hints.get(f.name, f.type)
        origin = typing.get_origin(ftype)
        args = typing.get_args(ftype)
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            ftype = non_none[0] if non_none else ftype
        kwargs[f.name] = _build_dataclass(ftype, value[f.name])
    return cls(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    loaded = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "experiment_config.yaml"),
        resolve=True,
    )
    assert isinstance(loaded, dict)
    config = _build_dataclass(ExperimentConfig, loaded)

    print(f"Replaying gap detection for {args.run_dir} ...")
    gaps = compute_gaps_for_config(config)  # joblib-cached
    print(f"  recovered {len(gaps)} gaps")

    contexts = build_contexts_for_gaps(gaps, config)
    print(f"  built {len(contexts)} ProposalContexts")

    ctx_paths = generate_all_contexts(gaps, args.run_dir)
    print(f"  wrote {len(ctx_paths)} contexts/*.json")

    prompt_records = build_prompt_records(contexts, config)
    prompt_paths = generate_all_prompts(prompt_records, args.run_dir)
    print(f"  wrote {len(prompt_paths)} prompts/*.json")


if __name__ == "__main__":
    main()
