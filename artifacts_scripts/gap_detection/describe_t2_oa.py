"""Descriptive / face-validity pass for the OA NLP temporal holdout.

Recomputes the phase-3 predictors + instruments deterministically from the frozen
phase-2 caches, then LABELS each flagged region by its distinctive KEYWORD TERMS
(log-odds vs the rest of the scoped pool) — the allowed aggregate, and the signal
the brief's face-validity check wants: do flagged fg-adjacent regions match known
post-2015 NLP growth (embeddings / attention / NMT)? NO raw abstract text: only
top-2000-df keyword terms, counts, fills.

Usage: uv run python -m artifacts_scripts.gap_detection.describe_t2_oa
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
# Cap BLAS/OpenMP threads BEFORE numpy import (else the voltage solves oversubscribe
# the box — measured 44 cores / 4395% CPU when unset). Brief: CPU-only, n_jobs ≤ 8.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.detectors import GapScorer, knn_radii, level_set_components
from artifacts_scripts.gap_detection.run_t2_oa_phase3 import (
    K_NN,
    P2_TOP_FRAC,
    P3A_PEAK_FRAC,
    TOP_K,
    _cite_adjacency,
    fill_counts,
    instrument_p3a_citation,
    knn_scale,
    load_phase2,
    predict_voltage_citation,
)


def region_terms(K_pool: np.ndarray, members: np.ndarray, vocab: np.ndarray,
                 n_top: int = 12, alpha: float = 1.0, min_df: int = 3) -> list[str]:
    """Top distinctive keyword terms of a region: log-odds(member df) vs the rest
    of the scoped pool. K_pool is binary (scoped docs x keywords)."""
    K_pool = np.asarray(K_pool, dtype=bool)
    members = np.asarray(members, dtype=np.int64)
    if members.size == 0:
        return []
    inside = K_pool[members].sum(axis=0).astype(np.float64)
    rest = ~np.isin(np.arange(K_pool.shape[0]), members)
    outside = K_pool[rest].sum(axis=0).astype(np.float64)
    n_in, n_out = float(members.size), float(rest.sum())
    score = (np.log((inside + alpha) / (n_in - inside + alpha))
             - np.log((outside + alpha) / (n_out - outside + alpha)))
    score[inside < min_df] = -np.inf
    order = np.argsort(score)[::-1][:n_top]
    return [str(vocab[j]) for j in order if np.isfinite(score[j])]


def main() -> None:
    d, K_corpus, arr, grow, graph, models, meta = load_phase2()
    Z = d["Z"]
    vocab = d["vocab"]
    fg_mask = d["fg_mask"]
    scoped_idx = d["scoped_idx"]
    scope_thr = float(d["scope_thr"])
    Zf = Z[fg_mask]
    Zb_s = Z[~fg_mask][scoped_idx]
    K_scoped = np.asarray(K_corpus[~fg_mask][scoped_idx].todense(), dtype=bool)
    Z_arr = arr["Z"]
    arr_global_rows = arr["store_rows"].astype(np.int64)
    bg_rows_global = np.where(~fg_mask)[0]
    scoped_global_rows = d["store_rows"][bg_rows_global[scoped_idx]]
    t_days, end_days = float(meta["t_days"]), float(meta["end_days"])

    print("fg top terms (the discovered cluster):", ", ".join(meta["fg_top_terms"][:15]), "\n")

    scorer = GapScorer(bg=Zb_s, probes=None, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    out = scorer.score(Zf)

    pool_radii = knn_scale(Zb_s, Zb_s, k=K_NN, self_rows=np.ones(Zb_s.shape[0], dtype=bool))
    pool_fill = fill_counts(Zb_s, pool_radii, Z_arr)
    print(f"scoped pool: {Zb_s.shape[0]:,} candidates, median fill {np.median(pool_fill):.0f}, "
          f"mean {pool_fill.mean():.1f} arrivals/neighborhood\n")

    # ---- LEVELSET components (the positive detector) labeled by terms
    comps = level_set_components(out["full"], scorer.cand, scorer.real_mask, top_frac=0.05)
    print("=== LEVELSET components (top-frac 0.05; first 8) — keyword labels + fill ===")
    budget = TOP_K
    for rank, c in enumerate(comps[:8], 1):
        members = np.asarray(c["members"])
        order = members[np.argsort(out["full"][members])[::-1]]
        take = order[: max(budget, 8)]
        budget = max(budget - len(take), 0)
        terms = region_terms(K_scoped, members, vocab)
        fills = pool_fill[take]
        print(f"#{rank} size={len(members)} n_eff={c['n_eff']:.1f} fill/member median "
              f"{np.median(fills):.0f} (pool {np.median(pool_fill):.0f}): {', '.join(terms)}")
    print()

    # ---- P2 residual components (instrument) labeled
    pred = models["kmeans"].predict(Z_arr)
    fg_add = Z_arr[pred == int(meta["fg_label"])]
    after = scorer.score(np.vstack([Zf, fg_add]).astype(np.float32))
    dfield = out["full"] - after["full"]
    comps_p2 = level_set_components(dfield, scorer.cand, scorer.real_mask, top_frac=P2_TOP_FRAC)
    print(f"=== P2 residual components (before−after, top-frac {P2_TOP_FRAC}; fg grown by "
          f"{fg_add.shape[0]} arrivals) — keyword labels ===")
    for rank, c in enumerate(comps_p2[:8], 1):
        terms = region_terms(K_scoped, np.asarray(c["members"]), vocab)
        print(f"#{rank} size={c['size']} n_eff={c['n_eff']:.1f}: {', '.join(terms)}")
    print()

    # ---- P3a citation flow-change peaks (the OA instrument) labeled
    # corridor_pairs gives the same (labels, cents, pairs) the phase-3 voltage path uses,
    # WITHOUT the expensive embedding voltage solve (we only need the electrode geometry).
    from artifacts_scripts.gap_detection.synth import corridor_pairs

    labels12, cents12, pairs12 = corridor_pairs(Zf, Zb_s, n_clusters=12, seed=7)
    r_arr_f = knn_radii(Z_arr, Zf, K_NN)
    arr_in_scope = r_arr_f <= scope_thr
    p3 = instrument_p3a_citation(
        graph, scoped_global_rows, arr_global_rows, arr_in_scope,
        t_days, end_days, labels12, cents12, pairs12, Zf, Zb_s)
    peaks = p3["peaks"]
    print(f"=== P3a citation flow-change peaks (top {P3A_PEAK_FRAC:.0%}, "
          f"{p3['n_pairs']} pairs, edges T {p3['n_edges_T']:,} -> T+Δ {p3['n_edges_Tdelta']:,}) "
          f"— keyword labels of the peak region ===")
    print("peak region terms:", ", ".join(region_terms(K_scoped, peaks, vocab, n_top=15)))
    print()

    # ---- voltage_cite predicted nodes labeled (the OA corridor predictor)
    W_cite_T = _cite_adjacency(graph, scoped_global_rows, t_days)
    vc_idx, n_used_c, _ = predict_voltage_citation(Zf, Zb_s, W_cite_T, labels12, cents12, pairs12)
    print(f"=== voltage_cite predicted nodes ({n_used_c} pairs on the dated citation graph) "
          "— keyword labels ===")
    print("predicted-node terms:", ", ".join(region_terms(K_scoped, vc_idx, vocab, n_top=15)))


if __name__ == "__main__":
    main()
