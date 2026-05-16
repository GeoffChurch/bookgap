"""Two-phase denoiser bake-off over the LLR-refine harness.

Phase 0: citation-graph-construction shootout (self_loops x row_normalize),
judged on LIP. Phase 1: six-arm denoiser comparison on the frozen graph.
See docs/superpowers/specs/2026-05-15-denoiser-bakeoff-llr-design.md.
"""
from __future__ import annotations

import numpy as np

from bookgap.util import build_citation_adjacency_from_ids
from artifacts_scripts.hdbscan_gap_probe import (
    _ProbeDenoiserContext,
    _solve_keep_smallest_gep,
    _denoise_lip,
    _denoise_cclip,
)


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


from artifacts_scripts.kball_gap_probe import _greedy_kball_gaps_inline


def kball_seeds_from_state(
    state: dict,
    radius_fraction: float = 0.75,
    num_gaps: int = 1000,
    min_gap_size: int = 0,
) -> list[dict]:
    """Regenerate K-ball seeds in the projected space.

    Mirrors the dicts gap_kball_llr_refine.load_kball_csv yields:
    gap_idx, gap_size, topic_id, article_indices (global article idx).
    K-ball defaults: k=1, is_intra_cloud=False (standalone-validated).
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
