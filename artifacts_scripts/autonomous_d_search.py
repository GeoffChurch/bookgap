"""Autonomous d-search planner for the denoiser bake-off.

For every (denoiser, embedder) pair, finds the retained-dimensionality `d`
that maximizes each tracked metric, via seed grid + bisection. Outputs a
list of new (embedder, denoiser, drop_k) cells to evaluate next.

Termination per pair:
- Adjacent d's within tolerance of the argmax (peak bracketed).
- ≥ MAX_EVALS_PER_PAIR distinct d's evaluated.
- Best d is at a boundary and there's no room to extend (e.g., d=1 floor or
  d=ambient ceiling — typically NONE).

Degeneracy detection: by seeding d=1 from the start, any metric whose argmax
is at d=1 is *flagged* in the output (the planner reports `degenerate=True`
in the summary); we still report its value but don't drive further bisection
toward d=1.

Usage:
    python -m artifacts_scripts.autonomous_d_search \\
        --csv outputs/gemma4_sweep/run_summary_gap0_gap0.csv \\
        --out cells_to_run.json \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# -- Configuration -----------------------------------------------------------

# Headline LLM config we search at (matches existing bake-off cells).
NUM_ARTICLES = 10
NUM_BOOKS = 5

# Embedder ambient dimensionalities.
AMBIENT_D = {
    "avsolatorio/GIST-small-Embedding-v0": 384,
    "google/embeddinggemma-300m": 768,
}

# Denoiser arms in the search. Excludes NONE (only meaningful at ambient d).
SEARCH_ARMS = (
    "LPP", "ANTI_LDA", "ANTI_CSP", "AUG_LPP", "CITATION_ERROR_LDA",
    "SYM_NORM_LPP", "CCP", "CCLIP", "LIP", "JAD_EQUAL", "PCA",
)

# Metrics we drive the search on.
METRICS = (
    "gap_coverage_fraction_mean",
    "proposal_gap_size_percentile_mean",
)

# Seed grid (retained-d values). Round-1/round-2 already evaluated d=374
# (GIST drop_k=10, Gemma drop_k=394) and d=128 (drop_k=256/640). Seed only
# adds the missing low-d points.
#
# d=1 is special: only PCA gets it. PCA at d=1 is the *degeneracy probe* — it
# keeps the largest-variance direction so the projection is well-defined; if
# coverage saturates at 100% there, we know the metric is degenerate at d=1
# regardless of denoiser. All other denoisers in the bake-off keep the
# *smallest*-Rayleigh-quotient eigenpair, which at d=1 puts the projection in
# (near-)null-space of the centered data — produces 0/0 normalization NaN that
# crashes the downstream `_centered_singular_values` diagnostic at
# `cloud.py:1118`. Worth ~25 min/iter of crash-cycle to skip. (The numerical
# fix — clip the norms before division — is also easy; choosing not to apply
# it here because the result would still be semantically near-zero.)
# All non-rank-1 arms seed at (16, 64) — including PCA, which historically had
# d=1 as a degeneracy floor probe but no longer needs it (degeneracy was
# confirmed: PCA d=1 streaming showed coverage saturating at 100% on every
# topic measured). With d=16 itself well below NONE on both embedders for
# every denoiser tested, anything d<16 just confirms the collapse. The
# bisection logic naturally avoids re-probing d<16 (argmax-and-better-neighbor
# never crosses below the lowest evaluated seed).
SEED_DS_PCA = (16, 64)
SEED_DS_OTHER = (16, 64)

# Per-embedder cross-embedder-comparison anchors. Gemma gets d=384 because
# GIST has d=384 natively via NONE — adding d=384 on Gemma lets denoised
# Gemma be compared against GIST's NONE baseline at the same retained dim.
# GIST gets no extras here (d=384 for any denoiser on GIST = drop_k=0 =
# rotation-equivalent of NONE). Cross-anchor cells are dispatched as
# INTERIOR cells (alongside d=64), not edges.
EMBEDDER_EXTRA_SEEDS: dict[str, tuple[int, ...]] = {
    "google/embeddinggemma-300m": (384,),
}

# High-edge "near-NONE" anchor for non-rank-1 arms: retained_d = ambient_d - 1
# (drop_k=1). For GIST this is d=383, for Gemma it's d=767. Symmetric to the
# rank-1 principled point (ANTI_LDA, CCP) which is at the same d. For full-rank
# methods this is "drop just the most-extremal eigenvector + keep the rest" —
# a near-NONE measurement that completes the curve. Without this anchor, the
# Gemma curves stop at d=576 or d=384 (depending on bisection extent), leaving
# a large unfilled gap between the sampled range and NONE at d=768; same gap
# for GIST between rightmost sample (often d=374) and NONE at d=384. Dispatched
# as EDGE cells in the edges-first ordering, so they land in one focused batch.
HIGH_EDGE_FOR_NON_RANK1 = True

# Rank-1 arms: their Rayleigh quotient's numerator is (effectively) rank 1, so
# only ONE direction has a non-trivial generalized eigenvalue and the other
# d-1 are in the null space of S_B (eigenvalue 0 or numerical-noise).
# "Smallest k" eigenvectors at any k>1 = "drop the discriminative direction
# + pick (k-1) arbitrary directions from a (d-1)-dim null space" = random
# projection. Varying drop_k measures random-projection-effect, not method.
# The principled operating point is drop_k = 1: drop the unique direction,
# keep the entire null space.
#
# Structural rank-1 classification (from formula inspection alone, see audit
# in autonomous_d_search.py docstring / denoisers.py review).
#   - CCP (CCLIPDenoiser α=0): numerator = S_B, rank 1 for binary classes.
#   - ANTI_LDA (TwoSubsetScatterDenoiser): numerator = S_B, rank 1 for binary.
# These have a (d-1)-dim null space; the single non-trivial GEP eigenpair is
# the class-discriminating direction. Empirically confirmed by
# `artifacts_scripts/denoiser_spectrum.py`:
#   - CCP: PR=1.06, elbow=1, top-5 = [152742, 4878, 0.17, 0.07, 0.06]
#   - ANTI_LDA: PR=1.75, elbow=2, top-5 = [414, 188, 5.9e-4, 5.6e-4, 4.7e-4]
#     (theoretically rank-1; numerically rank-≈2 from S_W ridge regularization)
#
# CCLIP_01 (α=0.01) is NOT in this list despite an empirically rank-1-like
# spectrum (PR=1.08). Its numerator is (1-α)·S_B + α·S_L which is structurally
# full-rank; the rank-1 *appearance* is contingent on α=0.01 being small enough
# that the S_L term is below the noise floor. We don't extrapolate from
# empirical perturbation behavior to a structural classification.
#
# Rank-1 arms get a single principled point at retained_d = ambient_d - 1
# (drop_k=1) and skip bisection. Historic drop_k=10/256/368 measurements are
# retained as random-projection baselines but no new k's are dispatched.
RANK1_ARMS = ("ANTI_LDA", "CCP")

# Denoiser-specific seed-grid overrides — currently empty.
# Earlier, JAD_EQUAL was special-cased with (1, 2, 3) based on its score
# spectrum having effective rank ~3 (PR=2.99 Gemma, PR=9.46 GIST). The
# diagnostic predicted JAD's principled-k might lie in that range. But the
# empirical d=16 collapse pattern argues against probing d<16 for anyone:
# JAD at d=16 was 14.66% (GIST) vs NONE 80.58% — well past the cliff. The
# spectrum-based prediction (d=3 might recover) is plausible but speculative;
# empirical coverage trumps it for the search. Re-enable if some denoiser
# proves the d=16-collapse pattern wrong.
DENOISER_SEED_OVERRIDE: dict[str, tuple[int, ...]] = {}

# Search knobs.
MAX_EVALS_PER_PAIR = 10
TOLERANCE_PP = 0.5            # if neighbor within this many pp, declare converged
COLLAPSE_THRESHOLD_PP = 5.0   # if a method is this far below its best on every
                              # metric for the last 2 evaluations going down,
                              # don't bisect further downward
MIN_D = 1
TOPIC_SUBSET_LEN = 20  # for filtering radius experiments / different subsets


# -- Data types --------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    embedder: str
    denoiser: str
    drop_k: int

    @property
    def retained_d(self) -> int:
        return AMBIENT_D[self.embedder] - self.drop_k

    def to_tuple(self) -> tuple[str, str, int]:
        return (self.embedder, self.denoiser, self.drop_k)


# -- CSV ingestion -----------------------------------------------------------

def _embedder_normalize(model: str) -> Optional[str]:
    for canonical in AMBIENT_D:
        if canonical in model or canonical.split("/")[-1].lower() in model.lower():
            return canonical
    return None


def load_evaluations(csv_path: Path) -> pd.DataFrame:
    """Load and clean the run summary CSV. Returns one row per (embedder,
    denoiser, retained_d) at the headline (a=10, b=5) config."""
    df = pd.read_csv(csv_path)
    # Drop radius experiments (separate sweep, not denoiser bake-off).
    df = df[~df["run"].str.contains("radiusp", na=False)]
    # Drop deprecated arms (legacy buggy variants kept for the appendix only).
    df = df[~df["denoiser"].str.contains("_DEPRECATED", na=False)]
    # Headline config only.
    df = df[(df["llm_num_articles"] == NUM_ARTICLES) & (df["llm_num_books"] == NUM_BOOKS)]
    # Normalize embedder model name to a canonical key.
    df = df.assign(emb=df["embedder_model"].apply(_embedder_normalize))
    df = df[df["emb"].notna()]
    # Retained-d.
    df = df.assign(retained_d=df.apply(lambda r: AMBIENT_D[r["emb"]] - int(r["drop_k"]), axis=1))
    # If multiple runs exist for the same (emb, denoiser, retained_d), pick
    # the most recent (highest-row-index after the CSV was generated in
    # discovery order).
    df = df.sort_values("run").drop_duplicates(
        subset=["emb", "denoiser", "retained_d"], keep="last",
    )
    return df


# -- Per-pair planning -------------------------------------------------------

@dataclass
class PairStatus:
    embedder: str
    denoiser: str
    evaluated: list[int]            # retained-d values evaluated
    metric_vals: dict[str, dict[int, float]]  # metric -> {d -> value}
    proposed_next_d: Optional[int]  # None if converged / capped / degenerate-floor
    note: str                       # short status word for logging


def _bisect_for_metric(
    ds: list[int],
    vals: dict[int, float],
    ambient_d: int,
    tolerance: float,
) -> Optional[int]:
    """Given evaluated d's (sorted) and their metric values, propose the next d
    to test for THIS metric. Return None if metric is converged / nothing to do.
    """
    if not ds:
        return None
    if len(ds) == 1:
        return None  # need more seed points first; handled by seed-injection
    # Best d so far.
    best_d = max(ds, key=lambda d: vals[d])
    best_v = vals[best_d]
    best_i = ds.index(best_d)
    left_d = ds[best_i - 1] if best_i > 0 else None
    right_d = ds[best_i + 1] if best_i + 1 < len(ds) else None

    # Boundary peaks: try to extend if there's room.
    if right_d is None and best_d < ambient_d:
        # Peak at the high end; extend upward midway to ambient_d.
        candidate = min(ambient_d, (best_d + ambient_d) // 2)
        if candidate != best_d and candidate not in vals:
            return candidate
        return None
    if left_d is None:
        # Peak at the low end (d=1 typically). Degenerate floor — don't extend
        # below MIN_D. Just declare bracketed at the floor.
        return None

    # Interior peak: convergence if either neighbor is within tolerance.
    left_v = vals[left_d]
    right_v = vals[right_d]
    if (best_v - left_v) <= tolerance or (best_v - right_v) <= tolerance:
        return None
    # Bisect toward the BETTER neighbor.
    better_neighbor = left_d if left_v >= right_v else right_d
    candidate = (best_d + better_neighbor) // 2
    if candidate <= MIN_D or candidate >= ambient_d or candidate in vals:
        return None
    return candidate


def _is_degenerate_at_d1(vals: dict[int, float]) -> bool:
    """Heuristic: argmax is at d=1 and metric is monotonically decreasing
    as d grows over the next few seed points."""
    if 1 not in vals:
        return False
    ds = sorted(vals.keys())
    if ds[0] != 1:
        return False
    # Check argmax.
    best_d = max(vals.keys(), key=lambda d: vals[d])
    if best_d != 1:
        return False
    # Confirm monotone decrease for the first 2-3 increases of d.
    monotone = True
    for i in range(min(3, len(ds) - 1)):
        if vals[ds[i + 1]] > vals[ds[i]]:
            monotone = False
            break
    return monotone


def _has_collapsed(
    metric_vals: dict[str, dict[int, float]],
    threshold_pp: float,
) -> bool:
    """True if for EVERY tracked metric, the lowest-d evaluation is ≥ threshold
    pp below the per-metric best. Indicates further downward bisection is
    pointless."""
    for metric, vals in metric_vals.items():
        if not vals:
            return False
        ds = sorted(vals.keys())
        if len(ds) < 2:
            return False
        best = max(vals.values()) * 100
        lowest_d = ds[0]
        lowest_v = vals[lowest_d] * 100
        if best - lowest_v < threshold_pp:
            return False
    return True


def plan_for_pair(
    embedder: str,
    denoiser: str,
    pair_df: pd.DataFrame,
) -> PairStatus:
    ambient_d = AMBIENT_D[embedder]
    evaluated = sorted(pair_df["retained_d"].astype(int).tolist())
    metric_vals: dict[str, dict[int, float]] = {}
    for metric in METRICS:
        metric_vals[metric] = {
            int(row["retained_d"]): float(row[metric])
            for _, row in pair_df.iterrows()
            if pd.notna(row.get(metric))
        }

    # Cap check.
    if len(evaluated) >= MAX_EVALS_PER_PAIR:
        return PairStatus(embedder, denoiser, evaluated, metric_vals, None, "capped")

    # Seed-injection: any missing seed d's still to inject?
    if denoiser in RANK1_ARMS:
        # Rank-1 arms: single principled point at retained_d = ambient_d - 1
        # (drop_k = 1). No bisection; just this one cell per embedder (plus
        # any embedder-specific extras for cross-embedder comparability).
        seed_grid: tuple[int, ...] = (ambient_d - 1,)
    elif denoiser in DENOISER_SEED_OVERRIDE:
        seed_grid = DENOISER_SEED_OVERRIDE[denoiser]
    elif denoiser == "PCA":
        seed_grid = SEED_DS_PCA
    else:
        seed_grid = SEED_DS_OTHER
    # Augment with embedder-specific cross-anchor (d=384 for Gemma).
    seed_grid = seed_grid + EMBEDDER_EXTRA_SEEDS.get(embedder, ())
    # Add the near-NONE high-edge anchor for non-rank-1 arms.
    if HIGH_EDGE_FOR_NON_RANK1 and denoiser not in RANK1_ARMS:
        seed_grid = seed_grid + (ambient_d - 1,)
    # Dedupe while preserving order (so smallest-first dispatch is intact).
    seed_grid = tuple(dict.fromkeys(seed_grid))
    missing_seeds = [d for d in seed_grid if d not in evaluated]
    if missing_seeds:
        # Inject in seed order (smallest first) so degeneracy info comes first.
        return PairStatus(
            embedder, denoiser, evaluated, metric_vals,
            missing_seeds[0], "seeding"
        )

    # Rank-1 arms have only one principled point; do not bisect.
    if denoiser in RANK1_ARMS:
        return PairStatus(embedder, denoiser, evaluated, metric_vals, None, "rank-1-done")

    # Per-metric bisection.
    proposals: set[int] = set()
    for metric in METRICS:
        if _is_degenerate_at_d1(metric_vals[metric]):
            continue  # don't bisect on a degenerate metric
        ds_for_metric = sorted(metric_vals[metric].keys())
        cand = _bisect_for_metric(ds_for_metric, metric_vals[metric], ambient_d, TOLERANCE_PP)
        if cand is not None:
            proposals.add(cand)

    if not proposals:
        # All non-degenerate metrics are bracketed/converged.
        return PairStatus(embedder, denoiser, evaluated, metric_vals, None, "converged")

    # Collapse early-stop: don't drive d down further if every metric has
    # already collapsed at the lowest evaluated d.
    if _has_collapsed(metric_vals, COLLAPSE_THRESHOLD_PP):
        # But we may still want to extend upward; filter proposals.
        lowest_evaluated = min(evaluated)
        proposals = {d for d in proposals if d > lowest_evaluated}
        if not proposals:
            return PairStatus(embedder, denoiser, evaluated, metric_vals, None, "collapsed")

    # Pick one proposal per round. Prefer proposal that's furthest from already
    # evaluated d's (in log space) — best info-gain heuristic.
    import math
    def log_min_dist(d):
        return min(abs(math.log(d) - math.log(e)) for e in evaluated)
    chosen = max(proposals, key=log_min_dist)
    return PairStatus(embedder, denoiser, evaluated, metric_vals, chosen, "bisecting")


# -- Top-level planner -------------------------------------------------------

def _is_edge_proposal(embedder: str, denoiser: str, proposed_d: int) -> bool:
    """An 'edge' cell is one at a pair's most-extreme principled seed value:
       - For rank-1 arms, the principled point at retained_d = ambient_d - 1.
       - For non-rank-1 arms, EITHER the lowest seed (currently d=16) OR the
         near-NONE high-edge anchor at ambient_d - 1.
    Embedder-specific cross-anchors (e.g., Gemma d=384) are NOT edges — they're
    interior comparison points dispatched alongside d=64. Bisection-proposed
    cells are NOT edges either."""
    if denoiser in RANK1_ARMS:
        # Only the structural principled point counts as the edge.
        return proposed_d == AMBIENT_D[embedder] - 1
    base = (DENOISER_SEED_OVERRIDE.get(denoiser)
            or (SEED_DS_PCA if denoiser == "PCA" else SEED_DS_OTHER))
    high_edge = AMBIENT_D[embedder] - 1
    return proposed_d == min(base) or proposed_d == high_edge


def plan_next_batch(csv_path: Path) -> tuple[list[Cell], list[PairStatus]]:
    """Plan the next batch. Edges first: if any pair has an un-evaluated edge
    seed, dispatch ONLY edge cells this iteration. Interior cells (d=64) and
    bisection cells wait until all edges are populated. This produces a more
    useful intermediate state — every pair gets its lowest-d datapoint before
    anyone gets refined."""
    df = load_evaluations(csv_path)
    edge_cells: list[Cell] = []
    interior_cells: list[Cell] = []
    statuses: list[PairStatus] = []
    for embedder in AMBIENT_D:
        for denoiser in SEARCH_ARMS:
            pair_df = df[(df["emb"] == embedder) & (df["denoiser"] == denoiser)]
            status = plan_for_pair(embedder, denoiser, pair_df)
            statuses.append(status)
            if status.proposed_next_d is None:
                continue
            drop_k = AMBIENT_D[embedder] - status.proposed_next_d
            cell = Cell(embedder, denoiser, drop_k)
            if _is_edge_proposal(embedder, denoiser, status.proposed_next_d):
                edge_cells.append(cell)
            else:
                interior_cells.append(cell)
    # Edges first: if any unfilled edges exist, dispatch only those.
    return (edge_cells if edge_cells else interior_cells), statuses


# -- CLI ---------------------------------------------------------------------

def _print_status_table(statuses: list[PairStatus]) -> None:
    print(f"{'embedder':<40} {'denoiser':<22} {'evaluated d':<28} {'next':<6} {'note':<12}")
    print("-" * 110)
    for s in statuses:
        emb_short = "GIST" if "GIST" in s.embedder else "Gemma"
        ds_str = ", ".join(str(d) for d in s.evaluated)
        if len(ds_str) > 26:
            ds_str = ds_str[:23] + "..."
        nxt = str(s.proposed_next_d) if s.proposed_next_d is not None else "—"
        print(f"{emb_short:<40} {s.denoiser:<22} {ds_str:<28} {nxt:<6} {s.note:<12}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to run_summary_gap0_gap0.csv")
    parser.add_argument("--out", type=Path,
                        default=Path("cache/autonomous_d_search/cells_to_run.json"),
                        help="Where to write the JSON cells list")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan but don't write JSON")
    args = parser.parse_args()

    cells, statuses = plan_next_batch(args.csv)

    _print_status_table(statuses)
    print()
    print(f"Proposed batch: {len(cells)} cells")
    for c in cells:
        emb_short = "GIST" if "GIST" in c.embedder else "Gemma"
        print(f"  {emb_short:<6} {c.denoiser:<22} drop_k={c.drop_k:<4} retained_d={c.retained_d}")

    if not args.dry_run:
        cells_json = [c.to_tuple() for c in cells]
        args.out.write_text(json.dumps(cells_json, indent=2))
        print(f"\nWrote {args.out} ({len(cells)} cells)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
