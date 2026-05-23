"""One-off: walk an existing run, replay gap detection + proposal embedding,
and write v2 ContextRecord JSONs (contexts/gap_*_topic_*.json) with the
new ``proposal_coverage`` + ``local_projection_2d`` fields and per-book
``articles_covered_after_proposal``.

Slated for deletion after the headline run is backfilled and verified.

Usage::

    conda run -p .conda python -m artifacts_scripts.backfill_coverage_analysis \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import typing
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.data.data_config import (
    SimHashDeduplicatorConfig,
    SqDistDeduplicatorConfig,
)
from bookgap.configs.embedder.embedder_config import (
    SentenceTransformerEmbedderConfig,
    TfGepEmbedderConfig,
)
from bookgap.gaps import compute_gaps_for_config
from bookgap.llm.artifacts.context_export import generate_all_contexts
from bookgap.llm.data_structures import BatchResults
from bookgap.llm.generators.workflow_functions import (
    _build_proposal_analyses,
    _compute_proposal_embeddings,
    PendingProposal,
)
from bookgap.llm.protocols import ParsedProposal, Proposal


# ``CloudConfig.embedder`` and ``CloudConfig.deduplicator`` are typed ``Any``
# so the structured-config builder cannot infer the concrete class from the
# annotation. Resolve them via their ``_target_`` value instead.
_TARGET_TO_CLS = {
    "bookgap.embedder.SentenceTransformerEmbedder": SentenceTransformerEmbedderConfig,
    "bookgap.tf_gep_embedder.TfGepEmbedder": TfGepEmbedderConfig,
    "bookgap.deduplicators.SimHashDeduplicator": SimHashDeduplicatorConfig,
    "bookgap.deduplicators.SqDistDeduplicator": SqDistDeduplicatorConfig,
}


def _build_dataclass(cls, value):
    """Construct a frozen dataclass from a plain dict (OmegaConf merge
    refuses frozen dataclasses with init=False fields).
    """
    if value is None:
        return value
    if isinstance(cls, type) and issubclass(cls, enum.Enum) and isinstance(value, str):
        return cls[value] if value in cls.__members__ else cls(value)
    # Resolve ``Any``-typed fields whose dict carries a Hydra ``_target_``.
    if cls is typing.Any and isinstance(value, dict) and "_target_" in value:
        resolved = _TARGET_TO_CLS.get(value["_target_"])
        if resolved is not None:
            cls = resolved
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


def _load_proposals_from_ollama_jsonl(path: Path, config: ExperimentConfig) -> list[ParsedProposal]:
    """Re-parse proposals from ollama_results.jsonl AND compute their
    embeddings (since the JSONL doesn't store them).
    """
    if not path.exists():
        return []
    # First pass: collect raw proposals from the JSONL.
    pending: list[PendingProposal] = []
    custom_ids: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("proposal") is None:
                continue
            proposal = Proposal(**d["proposal"])
            pending.append(PendingProposal(proposal=proposal, custom_id=d["custom_id"]))
            custom_ids.append(d["custom_id"])
    if not pending:
        return []
    # Compute embeddings ONCE for all proposals (loads the Embedder model).
    embeddings = _compute_proposal_embeddings(pending, config=config)
    return [
        ParsedProposal(
            custom_id=cid,
            proposal=p.proposal,
            embedding=emb.astype(np.float32, copy=False),
        )
        for cid, p, emb in zip(custom_ids, pending, embeddings, strict=True)
    ]


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
    gaps = compute_gaps_for_config(config)
    print(f"  recovered {len(gaps)} gaps")

    proposals = _load_proposals_from_ollama_jsonl(
        args.run_dir / "ollama_results.jsonl", config=config,
    )
    print(f"  loaded + embedded {len(proposals)} parsed proposals")

    results = BatchResults(proposals=proposals, metadata=config)
    proposal_analyses = _build_proposal_analyses(gaps, results, config)
    print(f"  built proposal analyses for {len(proposal_analyses)} gaps")

    # Skip gaps without proposals — appendix never displays them.
    gaps_with_analysis = [
        g for g in gaps if (g.topic_id, g.gap_idx) in proposal_analyses
    ]
    paths = generate_all_contexts(
        gaps=gaps_with_analysis,
        output_dir=args.run_dir,
        proposal_analyses=proposal_analyses,
    )
    print(f"  wrote {len(paths)} v2 contexts/*.json")


if __name__ == "__main__":
    main()
