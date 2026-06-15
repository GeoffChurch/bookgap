"""Validate the SIZE-CONTROLLED gap score against the coverage oracle.

The coverage oracle showed raw Δ partly proxies class size, masking the gap signal
(ρ(Δ,coverage) ≈ 0). This shows that residualizing Δ on a GRAPH-INTRINSIC size proxy
(``core.gap_scores.method_count`` — methods per class, which the detector has, unlike source
LOC) recovers the de-confounded coverage signal AS A SINGLE SCORE. Reuses the per-class
coverage rows emitted by ``coverage_oracle.py`` (no need to re-run the suite).

    python -m artifacts_scripts.curriculum.gap_recover.gap_size_control \\
        --repo cache/code_gap/python/repos/click --import-name click \\
        --coverage-oracle-json artifacts/code_gap/python/coverage_oracle_click.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from artifacts_scripts.curriculum.gap_recover.adapters.python.extract import extract_repo
from artifacts_scripts.curriculum.gap_recover.core.gap_scores import (
    method_count,
    raw_delta,
    size_controlled_delta,
)
from artifacts_scripts.curriculum.gap_recover.coverage_oracle import partial_spearman, spearman


def _resid_on(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = np.log(np.asarray(z, float) + 1.0)
    A = np.vstack([z, np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--import-name", required=True)
    ap.add_argument("--coverage-oracle-json", required=True,
                    help="rows-bearing JSON written by coverage_oracle.py --out")
    a = ap.parse_args()

    rows = json.load(open(a.coverage_oracle_json))["rows"]
    cov = {r["qualname"]: (float(r["covered_fraction"]), float(r["n_stmt"])) for r in rows}

    r = extract_repo(a.repo, a.import_name)
    su, ou = r.structural_users(), r.operational_users()
    delta = raw_delta(su, ou)
    mc = method_count(r.names, r.is_F)
    scd = size_controlled_delta(r.names, r.is_F, r.is_B, su, ou)  # the shipped harness score

    idx = [i for i, n in enumerate(r.names) if r.is_B[i] and str(n) in cov]
    D = delta[idx]
    SCD = scd[idx]
    M = mc[idx].astype(float)
    C = np.array([cov[str(r.names[i])][0] for i in idx])
    L = np.array([cov[str(r.names[i])][1] for i in idx])
    n = len(idx)

    print(f"[{a.import_name}] n={n} classes (B-nodes matched to coverage)")
    print(f"  graph size proxy quality:  ρ(method_count, LOC n_stmt) = {spearman(M, L):+.3f}")
    print(f"  confounds:  ρ(method_count, Δ) = {spearman(M, D):+.3f}   "
          f"ρ(method_count, coverage) = {spearman(M, C):+.3f}")
    print("  gap-score vs coverage  (want a clean NEGATIVE as a single score):")
    print(f"    raw Δ                              ρ = {spearman(D, C):+.3f}   (masked by size)")
    print(f"    Δ_resid | method_count  [GRAPH]    ρ = {spearman(SCD, C):+.3f}   <- detector-usable, shipped")
    print(f"    Δ_resid | n_stmt        [LOC]      ρ = {spearman(_resid_on(D, L), C):+.3f}   (needs source)")
    print(f"    partial Δ;coverage | n_stmt        ρ = {partial_spearman(D, C, L)[0]:+.3f}   (reference target)")


if __name__ == "__main__":
    main()
