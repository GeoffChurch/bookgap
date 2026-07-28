"""Task A: radial score fusion — rank by r_F, veto by the bg-aware statistic.

Design (from the 2026-06-11 handoff follow-up): within the candidate universe the
plain distance-to-fg scorer r_F,k dominates planted-hole recovery (rank 10 / AUC
0.84) but is unsafe without a habitability control (off-cloud probes take 50/50 of
the top-50; off-field bg drowns the unscoped ranking). The fused detector keeps
r_F,k as the ONLY ranking variable and demotes the bg-aware statistic to a binary
VETO — never a multiplicative weight (g^a was measured to hurt within scope,
0.84 -> 0.78).

The veto is the statistic's two factors used as gates, one each way:

  1. off-hull gate (the `g` factor, binary): r_B,k(x) above the hull quantile of
     real bg => x is uninhabitable (no background lives there) => veto. Kills
     off-cloud probes. The threshold is a property of the bg corpus only.
  2. proximal-presence gate (the `z`/Delta factor, upper fence): Delta(x) =
     log r_F,k - log r_B,k > log C => the foreground is not even proximally
     present at x's own bg scale — x is another field's territory, not a hole in
     this one => veto. Kills the everything-off-field pathology. Scale-free
     (ratio of the two kNN radii), unlike the scope quantile it replaces.

  Default C: a hole dug into an interleaved fg/bg region (median bg-interleave 8
  of 10 here) starts near Delta ~ 0 and the orphan criterion is >= 15% r_F growth,
  so genuine in-field holes live at Delta well under log 2; off-field clusters sit
  at several bg scales from any fg. C = 3 is the measured midpoint (see report
  section "Task A", sensitivity over C in {2, 3, 4}). The Task B machinery offers
  a calibrated alternative (Tukey upper fence of the matched-control null).

Rejected-by-design: any soft combination (weight, product, sum) — that is the
measured-out `g^a` path.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import numpy as np

from artifacts_scripts.gap_detection.detectors import GapScorer

HULL_Q = 0.90
DELTA_CEILING = float(np.log(3.0))


def veto_mask(
    scorer: GapScorer,
    out: dict[str, np.ndarray],
    hull_q: float = HULL_Q,
    delta_ceiling: float | None = DELTA_CEILING,
) -> np.ndarray:
    """True = vetoed (off-hull / uninhabitable, or fg not proximally present).

    `out` must be the scorer's `score()` dict for the foreground being tested.
    `delta_ceiling=None` disables the proximal-presence gate (hull-only veto).
    """
    veto = ~scorer.habitable_mask(hull_q)
    if delta_ceiling is not None:
        veto |= out["delta_only"] > delta_ceiling
    return veto


def fused_scores(out: dict[str, np.ndarray], veto: np.ndarray) -> np.ndarray:
    """Rank by r_F,k (descending); vetoed candidates are pushed to the bottom."""
    sc = out["no_hab"].astype(np.float64).copy()  # no_hab IS r_F,k
    sc[veto] = -np.inf
    return sc
