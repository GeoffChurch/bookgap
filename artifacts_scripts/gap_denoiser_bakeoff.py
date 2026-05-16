"""Two-phase denoiser bake-off over the LLR-refine harness.

Phase 0: citation-graph-construction shootout (self_loops x row_normalize),
judged on LIP. Phase 1: six-arm denoiser comparison on the frozen graph.
See docs/superpowers/specs/2026-05-15-denoiser-bakeoff-llr-design.md.
"""
from __future__ import annotations

import numpy as np

from bookgap.util import build_citation_adjacency_from_ids
from artifacts_scripts.hdbscan_gap_probe import _ProbeDenoiserContext


def build_scatter_context(
    emb: np.ndarray,
    ids: list[str],
    refs: list[list[str]],
    class_labels: list[str],
    self_loops: bool,
    row_normalize: bool,
) -> _ProbeDenoiserContext:
    """Build the scatter context for one graph construction.

    The adjacency comes from the production utility so a single code
    path expresses both the probe (self_loops=False, row_normalize=False)
    and production (True, True) constructions and the two mixed cells.
    """
    refs_clean = [r if r is not None else [] for r in refs]
    A, _id_to_idx = build_citation_adjacency_from_ids(
        ids, refs_clean,
        add_self_loops=self_loops,
        normalize=row_normalize,
    )
    return _ProbeDenoiserContext(emb, class_labels, A)
