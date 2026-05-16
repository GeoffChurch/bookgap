"""Two-phase denoiser bake-off over the LLR-refine harness.

Phase 0: citation-graph-construction shootout (self_loops x row_normalize),
judged on LIP. Phase 1: six-arm denoiser comparison on the frozen graph.
See docs/superpowers/specs/2026-05-15-denoiser-bakeoff-llr-design.md.
"""
from __future__ import annotations

import numpy as np
from bookgap.index import build_index, FlatL2

from bookgap.util import build_citation_adjacency_from_ids
from artifacts_scripts.hdbscan_gap_probe import (
    _ProbeDenoiserContext,
    _solve_keep_smallest_gep,
    _denoise_lip,
    _denoise_cclip,
)
from artifacts_scripts.kball_gap_probe import _greedy_kball_gaps_inline


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


ARMS = ("none", "pca", "lip", "cclip_auto",
        "cclip_norm_0.5", "cclip_norm_0")


def _thin_qr(basis: np.ndarray) -> np.ndarray:
    """Orthonormalize columns via thin QR (paper's 'Applying Denoising')."""
    Q, _R = np.linalg.qr(basis)
    return Q.astype(np.float64)


def solve_arm(arm: str, ctx: _ProbeDenoiserContext, n_keep: int):
    """Return (Q, alpha_used). Q is (d, n_keep) orthonormal, or None for 'none'.

    'none' -> (None, nan): the driver passes projection=None (full dim).
    """
    if arm == "none":
        return None, float("nan")
    if arm == "pca":
        S_T = ctx.S_T()
        S_T = 0.5 * (S_T + S_T.T)
        import scipy.linalg
        _w, V = scipy.linalg.eigh(S_T)
        basis = V[:, ::-1][:, :n_keep]  # keep largest-variance directions (contrast to the min-scatter lip/cclip arms)
        return _thin_qr(basis), float("nan")
    if arm == "lip":
        basis = _denoise_lip(ctx, n_keep)
        return _thin_qr(basis), float("nan")
    if arm == "cclip_auto":
        S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
        S_L = ctx.S_L()
        S_T = ctx.S_T()
        tr_B, tr_L = float(np.trace(S_B)), float(np.trace(S_L))
        alpha = tr_B / (tr_B + tr_L) if (tr_B + tr_L) > 0 else 0.5
        # raw (un-normalized) trace weighting — deliberately distinct from cclip_norm_* which trace-normalizes each matrix first
        A = (1.0 - alpha) * S_B + alpha * S_L
        basis = _solve_keep_smallest_gep(A, S_T, n_keep)
        return _thin_qr(basis), alpha
    if arm == "cclip_norm_0.5":
        basis, alpha = _denoise_cclip(ctx, n_keep, alpha=0.5)
        return _thin_qr(basis), alpha
    if arm == "cclip_norm_0":
        basis, alpha = _denoise_cclip(ctx, n_keep, alpha=0.0)
        return _thin_qr(basis), alpha
    raise ValueError(f"unknown arm {arm!r}")


def kball_seeds_from_state(
    state: dict,
    radius_fraction: float = 0.75,
    num_gaps: int = 1000,
    min_gap_size: int = 0,
) -> list[dict]:
    """Regenerate K-ball seeds in the projected space.

    Mirrors the dicts gap_kball_llr_refine.load_kball_csv yields:
    gap_idx, gap_size, topic_id, article_indices (global article idx).
    K-ball defaults: k=1, no self-match +1 correction (article-querying-article space; matches the standalone-validated kball_gap_probe path).
    """
    emb = state["emb"]
    art_gidx = np.asarray(state["article_global_idx"])
    bk_gidx = np.asarray(state["book_global_idx"])
    article_emb = np.ascontiguousarray(emb[art_gidx])
    book_emb = np.ascontiguousarray(emb[bk_gidx])

    gaps = _greedy_kball_gaps_inline(
        book_emb=book_emb,
        article_emb=article_emb,
        radius_fraction=radius_fraction,
        k=1,
        num_gaps=num_gaps,
        min_gap_size=min_gap_size,
    )
    topic_id = state.get("topic_id", "")
    seeds: list[dict] = []
    for gap_idx, gap_size, local_ids in gaps:
        global_ids = art_gidx[np.asarray(local_ids)].tolist()
        seeds.append({
            "gap_idx": int(gap_idx),
            "gap_size": int(gap_size),
            "topic_id": topic_id,
            "article_indices": global_ids,
        })
    return seeds


from artifacts_scripts.gap_kball_llr_refine import (  # noqa: E402
    refine_gap_via_llr,
    binary_set_jaccard,
    _build_fuzzy_jointnorm_memberships,
    _calibrate_stats_params,
    compute_gap_stats,
    dedup_via_containment_then_jaccard,
    _compute_corpus_null_stein_stratified,
    _null_for_gap,
)


def run_refine_pipeline(
    state: dict,
    seeds: list[dict],
    *,
    null_stratify_bins: int = 10,
    null_target_rel_se: float = 0.005,
    null_pilot_size: int = 200,
    calibrate_k_grid: tuple = (200, 400, 800, 1500),
    calibrate_m_grid: tuple = (25, 50, 100, 200, None),
    calibrate_tolerance_delta: float = 0.1,
    calibrate_tolerance_alpha: float = 0.05,
    calibrate_pilot_size: int = 100,
    null_mode: str = "dist",
    jaccard_threshold: float = 0.5,
    containment_threshold: float = 0.9,
    rank_by: str = "effective_size",
) -> list[dict]:
    """Run refine -> fuzzy -> calibrate -> stats -> dedup; return survivors.

    Mirrors gap_kball_llr_refine.main() between setup() and CSV writing,
    but takes a pre-built (already projected) ``state`` and in-process
    ``seeds`` instead of loading from disk.

    Parameters
    ----------
    state:
        Shared state dict as returned by gap_kball_llr_refine.setup() or
        an equivalent dict assembled by the bakeoff driver.
    seeds:
        List of K-ball seed dicts (keys: gap_idx, gap_size, topic_id,
        article_indices) as produced by kball_seeds_from_state().
    null_stratify_bins:
        Number of sigma-stratification bins for the corpus null.
    null_target_rel_se:
        Stein two-stage target relative SE for corpus null estimation.
    null_pilot_size:
        Pilot samples per bin for the corpus null Stein estimator.
    calibrate_k_grid:
        Candidate k_query values for _calibrate_stats_params.
    calibrate_m_grid:
        Candidate member_subsample values (None = "all") for calibration.
    calibrate_tolerance_delta:
        Absolute tolerance delta (δ) for calibration acceptance criterion.
    calibrate_tolerance_alpha:
        Max exceedance probability (α) for calibration acceptance criterion.
    calibrate_pilot_size:
        Number of pilot gaps for calibration.
    null_mode:
        Mode for _null_for_gap: "dist" (σ-histogram weighted, default) or
        "bin" (mean-σ single-bin lookup).
    jaccard_threshold:
        Jaccard overlap threshold for Jaccard-merge dedup step.
    containment_threshold:
        Containment threshold for containment-prune dedup step.
    rank_by:
        Gap stat key to rank by inside dedup (ties broken by this metric).
    """
    # Corpus-wide sigma-stratified null (returns null_strat, bin_edges, info).
    null_strat, null_bin_edges, _strat_info = _compute_corpus_null_stein_stratified(
        state,
        n_bins=null_stratify_bins,
        target_rel_se=null_target_rel_se,
        m_pilot_per_bin=null_pilot_size,
    )

    # Build book_index (article_index already lives in state; book_index
    # needs explicit construction, matching main()'s pattern).
    book_emb = np.ascontiguousarray(state["emb"][state["book_global_idx"]])
    book_index = build_index(FlatL2(), book_emb)

    # Phase 1: per-seed LLR refine.
    refined: list[dict] = []
    for row in seeds:
        gap_null = _null_for_gap(
            row["article_indices"],
            state["sigma"],
            null_strat,
            null_bin_edges,
            mode=null_mode,
        )
        result = refine_gap_via_llr(
            member_indices=row["article_indices"],
            state=state,
            null_log_mean=gap_null,
            recenter=True,
        )
        kball_set = set(row["article_indices"])
        refined_set = set(result["refined_member_indices"])
        refined.append({
            "kball_gap_id": row["gap_idx"],
            "kball_size": row["gap_size"],
            "refined_k_star": result["refined_k_star"],
            "refined_n_articles": len(result["refined_member_indices"]),
            "llr_peak": result["llr_peak"],
            "llr_peak_at_k": result["llr_peak_at_k"],
            "llr_curve": result["llr_curve"],
            "kball_size_delta": len(result["refined_member_indices"]) - row["gap_size"],
            "kball_jaccard": binary_set_jaccard(kball_set, refined_set),
            "refined_member_indices": result["refined_member_indices"],
        })

    # Phase 2: joint-normalized fuzzy gap memberships.
    M_norm_csc = _build_fuzzy_jointnorm_memberships(refined, state)

    # Phase 2b: calibrate (k_query, member_subsample) via pilot.
    # Real _calibrate_stats_params params: candidates_k, candidates_m,
    # tolerance_delta, tolerance_alpha, pilot_size (NOT k_grid/m_grid/tolerance/alpha).
    # It also requires book_index explicitly.
    k_query_resolved, m_resolved, _sweep = _calibrate_stats_params(
        refined_gaps=refined,
        state=state,
        M_norm_csc=M_norm_csc,
        book_index=book_index,
        candidates_k=list(calibrate_k_grid),
        candidates_m=list(calibrate_m_grid),
        tolerance_delta=calibrate_tolerance_delta,
        tolerance_alpha=calibrate_tolerance_alpha,
        pilot_size=calibrate_pilot_size,
    )

    # Phase 3: per-gap stats with fuzzy K_G + hard CSLS K_B.
    # main() passes k_per_book=k_query_resolved (same value as k_query).
    # main() passes book_global_idx=state["book_global_idx"] (array from state).
    # main() passes article_global_idx=state["article_global_idx"] (from state).
    # main() passes rng_seed=g["kball_gap_id"].
    for g_idx, g in enumerate(refined):
        mu_g = np.asarray(M_norm_csc[:, g_idx].todense()).flatten()
        stats = compute_gap_stats(
            refined_member_indices=g["refined_member_indices"],
            state=state,
            book_global_idx=state["book_global_idx"],
            book_index=book_index,
            article_index=state["article_index"],
            article_global_idx=state["article_global_idx"],
            fuzzy_gap_membership=mu_g,
            k_query=k_query_resolved,
            k_per_book=k_query_resolved,
            member_subsample_size=m_resolved,
            rng_seed=g["kball_gap_id"],
        )
        g.update({
            "logit_sum": stats["logit_sum"],
            "mean_logit": stats["mean_logit"],
            "median_logit": stats["median_logit"],
            "min_logit": stats["min_logit"],
            "effective_size": stats["effective_size"],
        })

    # Phase 4: containment+Jaccard dedup.
    # main() sets g["gap_idx"] = g["kball_gap_id"] before calling dedup
    # (dedup_via_containment_then_jaccard reads "gap_idx" internally).
    for g in refined:
        g["gap_idx"] = g["kball_gap_id"]
    refined = dedup_via_containment_then_jaccard(
        refined,
        jaccard_threshold=jaccard_threshold,
        containment_threshold=containment_threshold,
        rank_metric=rank_by,
    )

    # Return only survivors (dedup marks all gaps; filter to survivor==1).
    return [g for g in refined if g.get("survivor", 1) == 1]
