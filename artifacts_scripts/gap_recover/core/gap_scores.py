"""Gap scores, incl. a SIZE-CONTROLLED variant (graph-intrinsic).

The coverage oracle (``docs/code-gap-triangulation.md`` §Follow-on) found the raw gap score

    Δ = log(structural_users + 1) − log(operational_users + 1)

partly proxies CLASS SIZE: high-Δ classes are small utility classes, and small classes are
trivially well-covered — so raw ρ(Δ, coverage) ≈ 0, the gap signal masked by a size
confound. The fix is to residualize Δ on size. But the detector only has the GRAPH (not
source LOC), so we use a **graph-intrinsic** size proxy: ``method_count(C)`` = the number of
F-nodes directly under C's namespace (its methods/functions). On click this tracks true
LOC-size at Spearman ρ = 0.91, and the size-controlled Δ recovers the de-confounded coverage
signal as a SINGLE score (ρ(Δ_resid, coverage) = −0.30, vs raw −0.03; see
``gap_size_control.py``).

All functions take plain arrays so they work on either a ``core.data.Data`` or an
``adapters.python.extract.ExtractResult``.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def raw_delta(structural_users: np.ndarray, operational_users: np.ndarray) -> np.ndarray:
    """Δ = log(structural_users+1) − log(operational_users+1) (high = structural-but-under-exercised)."""
    return (np.log(np.asarray(structural_users, float) + 1.0)
            - np.log(np.asarray(operational_users, float) + 1.0))


def method_count(names, is_F) -> np.ndarray:
    """Graph-intrinsic size proxy: per node, the number of F-nodes whose immediate
    namespace parent is that node (a class's direct methods/functions). Tracks source
    LOC-size closely (ρ ≈ 0.9 on click) WITHOUT needing the source — so the detector can
    size-control its own gap ranking. Aligned to ``names``."""
    is_F = np.asarray(is_F)
    parent: Counter = Counter()
    for i, n in enumerate(names):
        if is_F[i]:
            s = str(n)
            if "." in s:
                parent[s.rsplit(".", 1)[0]] += 1
    return np.array([parent.get(str(n), 0) for n in names], dtype=np.int64)


def size_controlled_delta(names, is_F, is_B, structural_users, operational_users) -> np.ndarray:
    """Δ residualized on log(method_count) over the B-nodes (the candidate set).

    Returns an array aligned to ``names``; non-B entries are NaN. The residual is the
    size-controlled gap score: high = genuinely under-exercised *for its size*, dissolving
    the size confound that masks raw Δ against the coverage oracle. Residualizing on the
    intact (baseline) method-count means Δ_resid responds to a structural deletion exactly
    as Δ does (size is invariant under deleting a concept's callers), so the modality /
    plant-recover results are unchanged when Δ_resid replaces Δ as the graph detector."""
    is_B = np.asarray(is_B, bool)
    delta = raw_delta(structural_users, operational_users)
    size = np.log(method_count(names, is_F).astype(float) + 1.0)
    out = np.full(len(delta), np.nan)
    cand = np.where(is_B)[0]
    if cand.size < 2:
        out[cand] = delta[cand]
        return out
    x, y = size[cand], delta[cand]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    out[cand] = y - A @ coef
    return out
