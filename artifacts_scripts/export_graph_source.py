#!/usr/bin/env python3
"""Materialize a bookgap Cloud as a GraphSource npz for the curriculum `generate` app.

This is the bookgap-side half of the Cloud -> GraphSource bridge: it composes
the same config surface the legacy curriculum pipeline uses to build a Cloud
(see ``scripts/capture_gap_parity_golden.py::compose_ai_cfg`` and
``artifacts_scripts/curriculum/build_curriculum.py::run_curriculum_building``),
instantiates it, reads off the Cloud's public array-accessor surface
(``work_ids`` / ``work_titles`` / ``work_abstracts`` / ``embeddings`` /
``citation_graph``), and writes the result via
``curriculum.sources.graph_npz.save_graph_npz`` plus a JSON sidecar recording
the cloud identity, the config used, and basic shape stats.

``curriculum`` engine code never imports bookgap — this script is the only
place the dependency runs bookgap -> curriculum, which is the allowed
direction (curriculum is the reusable engine; bookgap is one data producer
for it).

Usage:
    uv run python -m artifacts_scripts.export_graph_source \\
        --out artifacts/graph_sources/ai_1702.npz \\
        --config-overrides "cloud.filter_pair.pre.subfield='1702'" cloud.denoiser=LPP \\
        --target-work-ids https://openalex.org/W4391021820
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import OmegaConf

from bookgap.cloud import Cloud, ensure_cloud_data
from bookgap.configs.config_store import register_configs
from curriculum.sources.graph_npz import save_graph_npz


def compose_cloud_cfg(overrides: Sequence[str], *, config_name: str = "standard") -> Any:
    """Compose the bookgap config surface the legacy pipeline builds a Cloud from.

    Mirrors ``scripts/capture_gap_parity_golden.py::compose_ai_cfg``: registers
    the bookgap ConfigStore presets, then composes ``config_name`` (the
    ``standard`` ``ExperimentConfig``, whose top-level ``cloud`` field is a
    ``CloudConfig``) under ``overrides``, a Hydra dotlist
    (e.g. ``"cloud.filter_pair.pre.subfield='1702'"``, ``"cloud.denoiser=LPP"``).
    """
    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        return compose(config_name=config_name, overrides=list(overrides))


def assemble_arrays(
    cloud: Any,
) -> tuple[list[str], sp.csr_matrix, list[str], list[str], NDArray[np.float32]]:
    """Materialize a Cloud's public accessors into the GraphSource npz shape.

    Pure function over ``cloud.work_ids()`` / ``work_titles()`` /
    ``work_abstracts()`` / ``embeddings()`` / ``citation_graph(normalize=False)``
    — a duck-typed stub exposing just those five methods exercises this
    without touching any cache or corpus data, which is what makes this
    unit-testable in isolation from a real Cloud.

    Returns ``(ids, adj, titles, abstracts, emb)`` — the exact keyword shape
    ``curriculum.sources.graph_npz.save_graph_npz`` accepts. ``adj`` keeps the
    Cloud's citation convention (``adj[i, j] > 0`` == work ``i`` cites work
    ``j``), which already matches the GraphSource "``i`` points to ``j``"
    convention, so no transpose is needed.

    Raises ``ValueError`` if the accessors disagree on node count.
    """
    ids = list(cloud.work_ids())
    titles = list(cloud.work_titles())
    abstracts = list(cloud.work_abstracts())
    emb = np.ascontiguousarray(cloud.embeddings(), dtype=np.float32)
    adj = cloud.citation_graph(normalize=False).tocsr()

    n = len(ids)
    lengths = {"titles": len(titles), "abstracts": len(abstracts), "emb": emb.shape[0]}
    mismatched = {k: v for k, v in lengths.items() if v != n}
    if mismatched or adj.shape != (n, n):
        raise ValueError(
            f"Cloud array length mismatch: ids={n} {mismatched} adj.shape={adj.shape}"
        )

    return ids, adj, titles, abstracts, emb


def build_meta(
    cloud: Any,
    cfg: Any,
    *,
    ids: Sequence[str],
    adj: sp.csr_matrix,
    target_work_ids: Sequence[str],
    config_name: str,
    overrides: Sequence[str],
) -> dict[str, Any]:
    """Build the ``<out>.meta.json`` sidecar payload.

    ``target_work_ids`` is recorded verbatim, NOT resolved against ``ids``
    here — seed resolution happens at generate-time (by id, against the
    npz's own ``id_list``), so the exporter doesn't need to know whether the
    ids it was handed even exist in this particular cloud.
    """
    corpus_fingerprint, _, _ = ensure_cloud_data(cloud)
    embedder_name = getattr(cloud.embedder, "model_name", type(cloud.embedder).__name__)
    return {
        "cloud_name": cloud.name(),
        "cloud_fingerprint": corpus_fingerprint,
        "config_name": config_name,
        "config_overrides": list(overrides),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_nodes": len(ids),
        "n_edges": int(adj.nnz),
        "embedder_name": embedder_name,
        "target_work_ids": list(target_work_ids),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output GraphSource .npz path."
    )
    parser.add_argument(
        "--config-name",
        default="standard",
        help="Hydra config name to compose (default: 'standard', the "
        "ConfigStore-registered ExperimentConfig with a top-level `cloud` field).",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra dotlist overrides against --config-name, e.g. "
        "cloud.filter_pair.pre.subfield=\\'1702\\' cloud.denoiser=LPP",
    )
    parser.add_argument(
        "--target-work-ids",
        nargs="*",
        default=[],
        metavar="WORK_ID",
        help="Target work ids to record in the sidecar. NOT resolved here — "
        "the curriculum generate app resolves seeds by id at generate-time.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = compose_cloud_cfg(args.config_overrides, config_name=args.config_name)
    cloud: Cloud = instantiate(cfg.cloud)

    ids, adj, titles, abstracts, emb = assemble_arrays(cloud)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(args.out, ids=ids, adj=adj, titles=titles, abstracts=abstracts, emb=emb)

    meta = build_meta(
        cloud,
        cfg,
        ids=ids,
        adj=adj,
        target_work_ids=args.target_work_ids,
        config_name=args.config_name,
        overrides=args.config_overrides,
    )
    meta_path = args.out.parent / f"{args.out.name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))

    print(
        f"Wrote {args.out} ({meta['n_nodes']} nodes, {meta['n_edges']} edges) "
        f"and sidecar {meta_path}"
    )


if __name__ == "__main__":
    main()
