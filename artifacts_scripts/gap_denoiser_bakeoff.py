"""Two-phase denoiser bake-off over the LLR-refine harness.

Phase 0: citation-graph-construction shootout (self_loops x row_normalize),
judged on LIP. Phase 1: six-arm denoiser comparison on the frozen graph.
See docs/superpowers/specs/2026-05-15-denoiser-bakeoff-llr-design.md.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from bookgap.index import build_index, FlatL2
from bookgap.depletion_test import score_region_depletion

from bookgap.util import build_citation_adjacency_from_ids
from artifacts_scripts.hdbscan_gap_probe import (
    _ProbeDenoiserContext,
    _solve_keep_smallest_gep,
    _denoise_lip,
    _denoise_lpp,
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


ARMS = ("none", "pca", "lip", "lpp", "cclip_auto",
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
    if arm == "lpp":
        basis = _denoise_lpp(ctx, n_keep)
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
    deflation: str = "greedy_hard",
) -> list[dict]:
    """Regenerate K-ball seeds in the projected space.

    Mirrors the dicts gap_kball_llr_refine.load_kball_csv yields:
    gap_idx, gap_size, topic_id, article_indices (global article idx).
    K-ball defaults: k=1, no self-match +1 correction (article-querying-article space; matches the standalone-validated kball_gap_probe path).

    When deflation == "greedy_hard" (the default), behaviour is byte-identical
    to the original _greedy_kball_gaps_inline call.
    """
    emb = state["emb"]
    art_gidx = np.asarray(state["article_global_idx"])
    bk_gidx = np.asarray(state["book_global_idx"])
    article_emb = np.ascontiguousarray(emb[art_gidx])
    book_emb = np.ascontiguousarray(emb[bk_gidx])

    from artifacts_scripts.gap_deflation_bakeoff import build_kball_candidates, deflate
    cands = build_kball_candidates(article_emb, book_emb, radius_fraction=radius_fraction, k=1)
    gaps = deflate(cands, deflation, article_emb, num_gaps=num_gaps,
                   min_gap_size=min_gap_size, sigma=state.get("sigma"),
                   book_emb=book_emb, radius_fraction=radius_fraction)
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
    # Dual-landmark: book-type landmarks first (canonical effective_size),
    # then book-chapter-type landmarks (effective_size_chapters).
    # Per-gap binomial depletion: enclosing-ball region via full_index.
    book_gidx = np.asarray(state["book_global_idx"])
    chap_gidx = _landmark_global_idx(state, "book-chapter")
    has_chapters = chap_gidx.size > 0
    if has_chapters:
        chap_emb = np.ascontiguousarray(state["emb"][chap_gidx])
        chap_index = build_index(FlatL2(), chap_emb)
    book_rate = _corpus_book_rate(state, "book")
    full_index = state["full_index"]
    types_arr = np.asarray(state["types"])
    # "book" is hardcoded for the depletion test (numerator+denominator
    # share the predicate); would need updating if setup() book_types differ.
    is_book_type = types_arr == "book"

    for g_idx, g in enumerate(refined):
        mu_g = np.asarray(M_norm_csc[:, g_idx].todense()).flatten()

        # First call: book-type landmarks (canonical path, same as original).
        stats_book = compute_gap_stats(
            refined_member_indices=g["refined_member_indices"],
            state=state,
            book_global_idx=book_gidx,
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
            "logit_sum": stats_book["logit_sum"],
            "mean_logit": stats_book["mean_logit"],
            "median_logit": stats_book["median_logit"],
            "min_logit": stats_book["min_logit"],
            "effective_size": stats_book["effective_size"],
        })
        g["effective_size_books"] = stats_book["effective_size"]

        # Second call: book-chapter-type landmarks (only landmark args differ).
        if has_chapters:
            stats_chap = compute_gap_stats(
                refined_member_indices=g["refined_member_indices"],
                state=state,
                book_global_idx=chap_gidx,
                book_index=chap_index,
                article_index=state["article_index"],
                article_global_idx=state["article_global_idx"],
                fuzzy_gap_membership=mu_g,
                k_query=k_query_resolved,
                k_per_book=k_query_resolved,
                member_subsample_size=m_resolved,
                rng_seed=g["kball_gap_id"],
            )
            g["effective_size_chapters"] = stats_chap["effective_size"]
        else:
            g["effective_size_chapters"] = 0.0

        # Per-gap binomial depletion: enclosing-ball around the gap centroid.
        # Region = works within centroid->1-NN-book radius (FAISS L2 = squared).
        # full_index supports range_search for both FlatL2 (topic-scale) and
        # HNSW (subfield/domain-scale); radius threshold is L2-squared in both.
        members = np.asarray(g["refined_member_indices"], dtype=np.int64)
        if members.size:
            # Non-unit centroid is intentional: defines an L2 enclosing ball (not
            # cosine); the SAME centroid is used for both the 1-NN-book radius and
            # the full_index region query, so radius and ball are consistent.
            centroid = state["emb"][members].mean(axis=0).astype(
                np.float32)[None, :]
            d_book, _ = book_index.search(np.ascontiguousarray(centroid), k=1)
            radius2 = float(d_book[0, 0])          # FAISS L2 distance = squared
            if radius2 > 0.0:
                lims, _D, I = full_index.range_search(
                    np.ascontiguousarray(centroid), radius2)
                region = I[lims[0]:lims[1]]
                size = int(region.size)
                target = int(is_book_type[region].sum()) if size else 0
            else:
                # Zero radius (degenerate: centroid coincides with a book) → guard.
                size = 0
                target = 0
            g["books_below_rate_pval"] = _depletion_pvalue(
                size, target, book_rate)
        else:
            g["books_below_rate_pval"] = 1.0

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
    return [g for g in refined if g["survivor"] == 1]


# ---------------------------------------------------------------------------
# Scorecard metrics + CSV writer
# ---------------------------------------------------------------------------

SCORECARD_FIELDS = [
    "phase", "topic", "cell_or_arm", "self_loops", "row_normalize",
    "n_keep", "alpha_used", "sig_count_p1e3", "top10_artifact_leaks",
    "top25_jaccard_vs_ref", "fourcorner_bc", "fourcorner_bC",
    "fourcorner_Bc", "fourcorner_BC",
    "mean_pairwise_jaccard_top20", "unique_coverage_at_20",
    "runtime_s",
]


def _mean_jaccard_top(a_sets, b_sets, top_n=25):
    """Mean best-match Jaccard of top-N gaps in a against top-N in b."""
    a = a_sets[:top_n]
    b = b_sets[:top_n]
    if not a or not b:
        return 0.0
    out = []
    for sa in a:
        best = 0.0
        for sb in b:
            u = len(sa | sb)
            best = max(best, (len(sa & sb) / u) if u else 0.0)
        out.append(best)
    return float(np.mean(out)) if out else 0.0


def _title_homogeneity_leak(members, titles_by_global_idx):
    """Approximate artifact screen: shared 3-token title prefix fraction.

    Flagged if >60% of members share the same first-3-token title
    prefix (templated dataset/language artifacts). Approximate by
    design; a screening signal, not ground truth.
    """
    if titles_by_global_idx is None or not members:
        return False
    from collections import Counter
    prefixes = []
    for gi in members:
        t = (titles_by_global_idx.get(gi) or "").lower().split()
        prefixes.append(" ".join(t[:3]))
    prefixes = [p for p in prefixes if p]
    if not prefixes:
        return False
    top = Counter(prefixes).most_common(1)[0][1]
    return (top / len(prefixes)) > 0.6


def scorecard_row(*, phase, topic, cell_or_arm, self_loops, row_normalize,
                  n_keep, alpha_used, survivors, ref_top_member_sets,
                  runtime_s, eff_low_high_split,
                  titles_by_global_idx=None):
    """Compute one scorecard row dict from a list of survivor gap dicts.

    Key reconciliation note (2026-05-15, updated): ``run_refine_pipeline``
    now emits all three dual-landmark keys per gap dict — ``effective_size_books``,
    ``effective_size_chapters``, and ``books_below_rate_pval`` — so the
    four-corner cross-tab is fully populated for dual-landmark callers.
    The ``.get()`` defaults below remain as defensive no-ops for non-dual
    callers (e.g. plain ``compute_gap_stats``) that do not produce the
    split keys:
      - ``effective_size_books``  → falls back to ``effective_size``
      - ``effective_size_chapters`` → falls back to ``0.0``
      - ``books_below_rate_pval``  → falls back to ``1.0`` (not significant)
    """
    ranked = sorted(survivors,
                    key=lambda g: g.get("effective_size", 0.0),
                    reverse=True)
    top10 = ranked[:10]
    sig = sum(1 for g in survivors
              if g.get("books_below_rate_pval", 1.0) < 1e-3)
    leaks = sum(1 for g in top10
                if _title_homogeneity_leak(
                    g.get("refined_member_indices", []),
                    titles_by_global_idx))
    this_top = [set(g.get("refined_member_indices", []))
                for g in ranked[:25]]
    jacc = _mean_jaccard_top(this_top, ref_top_member_sets, 25)

    # Dual-landmark four-corner.  Falls back gracefully when book/chapter
    # split keys are absent (effective_size_books → effective_size,
    # effective_size_chapters → 0.0) until dual-landmark is wired in.
    fc = {"bc": 0, "bC": 0, "Bc": 0, "BC": 0}
    s = eff_low_high_split
    for g in survivors:
        b_hi = g.get("effective_size_books",
                     g.get("effective_size", 0.0)) >= s
        c_hi = g.get("effective_size_chapters", 0.0) >= s
        key = ("B" if b_hi else "b") + ("C" if c_hi else "c")
        fc[key] += 1

    from artifacts_scripts.gap_deflation_bakeoff import inter_gap_metrics
    igm = inter_gap_metrics(ranked, top_k=20)

    return {
        "phase": phase, "topic": topic, "cell_or_arm": cell_or_arm,
        "self_loops": self_loops, "row_normalize": row_normalize,
        "n_keep": n_keep, "alpha_used": alpha_used,
        "sig_count_p1e3": sig, "top10_artifact_leaks": leaks,
        "top25_jaccard_vs_ref": round(jacc, 4),
        "fourcorner_bc": fc["bc"], "fourcorner_bC": fc["bC"],
        "fourcorner_Bc": fc["Bc"], "fourcorner_BC": fc["BC"],
        "mean_pairwise_jaccard_top20": round(igm["mean_pairwise_jaccard_top20"], 4),
        "unique_coverage_at_20": igm["unique_coverage_at_20"],
        "runtime_s": round(runtime_s, 1),
    }


def write_scorecard(rows: list[dict], out_path: Path) -> None:
    """Write scorecard rows to a CSV at ``out_path``, always writing header.

    Creates parent directories as needed.  Handles an empty ``rows`` list
    correctly — the header is always written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Task 7: Phase-0 deterministic selector, Phase-1 per-arm runner, CLI
# ---------------------------------------------------------------------------

import argparse
import time
from collections import Counter

GRAPH_CELLS = [(False, False), (False, True),
               (True, False), (True, True)]


def pick_phase0_winner(p0_rows: list[dict]) -> tuple[bool, bool]:
    """Deterministic: per-topic rank cells, majority-vote across topics.

    Order key per topic (best first): higher sig_count_p1e3, then fewer
    artifact_leaks, then higher (Bc + bc) four-corner mass.  Ties break
    toward simpler construction: self_loops=False before True, then
    row_normalize=False before True.
    """
    by_topic: dict[str, list[dict]] = {}
    for r in p0_rows:
        by_topic.setdefault(r["topic"], []).append(r)

    votes: Counter = Counter()
    for _topic, rows in by_topic.items():
        def rank_key(r):
            return (
                -r["sig_count_p1e3"],
                r["top10_artifact_leaks"],
                -(r["fourcorner_Bc"] + r["fourcorner_bc"]),
                bool(r["self_loops"]),    # False (0) sorts before True (1)
                bool(r["row_normalize"]),
            )
        best = sorted(rows, key=rank_key)[0]
        votes[(bool(best["self_loops"]), bool(best["row_normalize"]))] += 1

    if not votes:
        raise RuntimeError(
            "Phase 0: no successful cell evaluations across any scope; "
            "cannot pick a graph winner (check earlier [ERROR] lines)")

    # Majority vote; tie -> simpler construction (self_loops=False first, then row_normalize=False).
    return min(votes.items(),
               key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[0]


def _write_leaderboard(out_dir, topic, arm, ranked, titles_by_global_idx,
                       top_n: int = 20) -> None:
    """Per-(topic,arm) qualitative leaderboard: top-N gaps by effective_size
    with sampled in-gap article titles + member ids, for eyeballing what
    each gap is actually about. ``ranked`` is the effective_size-desc
    survivor list. No-op if ``out_dir`` is None.
    """
    if out_dir is None:
        return
    import csv as _csv
    import json as _json
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = str(topic).replace("/", "_")
    path = out_dir / f"{safe_topic}__{arm}__leaderboard.csv"
    cols = ["rank", "kball_gap_id", "refined_k_star", "n_members",
            "effective_size", "effective_size_books",
            "effective_size_chapters", "books_below_rate_pval",
            "sample_titles", "member_indices"]
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for rank, g in enumerate(ranked[:top_n], start=1):
            members = list(g.get("refined_member_indices", []))
            titles = []
            for gi in members[:8]:
                t = (titles_by_global_idx or {}).get(gi) or ""
                t = " ".join(str(t).split())  # collapse whitespace/newlines
                if t:
                    titles.append(t)
            w.writerow([
                rank,
                g.get("kball_gap_id", ""),
                g.get("refined_k_star", ""),
                len(members),
                round(float(g.get("effective_size", 0.0)), 3),
                round(float(g.get("effective_size_books", 0.0)), 3),
                round(float(g.get("effective_size_chapters", 0.0)), 3),
                g.get("books_below_rate_pval", ""),
                " || ".join(titles),
                _json.dumps(members[:50]),
            ])


def _run_one(scope, topic, emb, ids, refs, labels, titles_map,
             self_loops, row_normalize, arm, n_keep,
             phase, ref_top_member_sets, leaderboard_dir=None,
             index_type: str = "flat"):
    """One (graph, arm) evaluation -> (scorecard_row_dict, ranked_member_sets).

    Builds the scatter context, solves the arm basis, rebuilds state via
    setup() with the projection, regenerates K-ball seeds in projected
    space, runs the refine pipeline, and formats one scorecard row.
    """
    from artifacts_scripts.gap_static_seed_diagnostic import setup
    t0 = time.time()
    ctx = build_scatter_context(emb, ids, refs, labels,
                                self_loops=self_loops,
                                row_normalize=row_normalize)
    Q, alpha_used = solve_arm(arm, ctx, n_keep)
    state = setup(scope, index_type=index_type,
                  article_types=("article",), book_types=("book",),
                  projection=Q)
    state["topic_id"] = topic
    seeds = kball_seeds_from_state(state, radius_fraction=0.75,
                                   num_gaps=1000, min_gap_size=30)
    survivors = run_refine_pipeline(state, seeds)
    eff_books = [g.get("effective_size_books", 0.0) for g in survivors]
    split = float(np.median(eff_books)) if eff_books else 0.0
    ranked = sorted(survivors,
                    key=lambda g: g.get("effective_size", 0.0),
                    reverse=True)
    ranked_sets = [set(g.get("refined_member_indices", []))
                   for g in ranked[:25]]
    _write_leaderboard(leaderboard_dir, topic, arm, ranked, titles_map)
    row = scorecard_row(
        phase=phase, topic=topic, cell_or_arm=arm,
        self_loops=self_loops, row_normalize=row_normalize,
        n_keep=(n_keep if Q is not None else emb.shape[1]),
        alpha_used=alpha_used, survivors=survivors,
        ref_top_member_sets=ref_top_member_sets,
        runtime_s=time.time() - t0, eff_low_high_split=split,
        titles_by_global_idx=titles_map,
    )
    return row, ranked_sets


def main() -> None:
    """Two-phase denoiser bake-off driver.

    Phase 0: graph shootout across four (self_loops x row_normalize) cells
    using LIP, judged by the geometric/dual-landmark scorecard.  The winner
    cell is selected deterministically.

    Phase 1: six-arm denoiser comparison (none, pca, lip, cclip_auto,
    cclip_norm_0.5, cclip_norm_0) on the frozen graph construction chosen
    in Phase 0.  'none' runs first as the Jaccard reference.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scopes", required=True,
                   help="comma-separated F/SF/T scope keys, "
                        "e.g. 17/1702/T11303,17/1702/T10181,17/1702/T10462")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--n-keep", type=int, default=370)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freeze-graph", default=None,
                   help="skip Phase 0, e.g. 'self=T,rownorm=T'")
    p.add_argument("--index-type", choices=["flat", "hnsw"], default="flat",
                   help="flat = exact + deterministic (topic scale, default); "
                        "hnsw = approximate, for subfield/domain scale")
    args = p.parse_args()

    # Set index-type environment knob BEFORE Phase-0/Phase-1 loops so that
    # _build_topic_index (called inside kball_seeds_from_state) honours it.
    if args.index_type == "flat":
        os.environ["BOOKGAP_EXACT_INDEX"] = "1"
    # hnsw path intentionally leaves FAISS OMP at its multi-threaded
    # default. Forcing 1 thread serialized ALL downstream FAISS search
    # process-wide (~14x slowdown — as slow as exact FlatL2, defeating
    # the point of hnsw). With the fixed _build_topic_index params
    # (efc=512, efs=2000) recall is high enough that runs are materially
    # stable run-to-run WITHOUT single-thread construction; bit-level
    # thread determinism is explicitly not a requirement here. Use
    # --index-type flat for the exact ground-truth confirmation.

    from artifacts_scripts.hdbscan_gap_probe import load_scope

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_rows: list[dict] = []
    out_path = Path(args.out)

    # Load raw emb/ids/refs once (refs are expensive). NOTE: setup() still
    # re-loads from disk and rebuilds the index on every _run_one call
    # (projection differs per call, so index rebuild is unavoidable) — this
    # pre-load only amortizes the refs read + titles_map, not the index build.
    loaded: dict[str, tuple] = {}
    for scope in scopes:
        emb, ids, types, titles, topic_ids, refs = load_scope(
            scope, with_refs=True)
        labels = list(types)
        titles_map = {i: titles[i] for i in range(len(titles))}
        loaded[scope] = (emb, ids, refs, labels, titles_map, topic_ids)

    # ---- Phase 0: graph shootout on LIP ----
    if args.freeze_graph:
        kv = dict(x.split("=") for x in args.freeze_graph.split(","))
        frozen = (kv["self"].upper().startswith("T"),
                  kv["rownorm"].upper().startswith("T"))
    else:
        import traceback as _traceback
        p0_rows: list[dict] = []
        for scope in scopes:
            emb, ids, refs, labels, titles_map, _tids = loaded[scope]
            for sl, rn in GRAPH_CELLS:
                try:
                    row, _ = _run_one(
                        scope, scope, emb, ids, refs, labels, titles_map,
                        sl, rn, "lip", args.n_keep, "p0",
                        ref_top_member_sets=[],
                        leaderboard_dir=out_path / "leaderboards",
                        index_type=args.index_type)
                    p0_rows.append(row)
                except Exception as exc:
                    print(f"[ERROR] p0 scope={scope} cell_or_arm=lip "
                          f"self_loops={sl} row_normalize={rn}: {exc}",
                          flush=True)
                    _traceback.print_exc()
                    continue
        all_rows.extend(p0_rows)
        frozen = pick_phase0_winner(p0_rows)
    print(f"[phase0] frozen graph = self_loops={frozen[0]}, "
          f"row_normalize={frozen[1]}", flush=True)

    # ---- Phase 1: arm bake-off on the frozen graph ----
    import traceback as _traceback
    for scope in scopes:
        emb, ids, refs, labels, titles_map, _tids = loaded[scope]
        # 'none' runs first as the Jaccard reference for other arms.
        ref_sets: list = []
        try:
            ref_row, ref_sets = _run_one(
                scope, scope, emb, ids, refs, labels, titles_map,
                frozen[0], frozen[1], "none", args.n_keep, "p1",
                ref_top_member_sets=[],
                leaderboard_dir=out_path / "leaderboards",
                index_type=args.index_type)
            all_rows.append(ref_row)
        except Exception as exc:
            print(f"[ERROR] p1 scope={scope} cell_or_arm=none "
                  f"self_loops={frozen[0]} row_normalize={frozen[1]}: {exc}",
                  flush=True)
            _traceback.print_exc()
            # ref_sets stays [] — subsequent arms still run; their
            # top25_jaccard_vs_ref will degrade to 0.0, which is acceptable.
        for arm in arms:
            if arm == "none":
                continue
            try:
                row, _ = _run_one(
                    scope, scope, emb, ids, refs, labels, titles_map,
                    frozen[0], frozen[1], arm, args.n_keep, "p1",
                    ref_top_member_sets=ref_sets,
                    leaderboard_dir=out_path / "leaderboards",
                    index_type=args.index_type)
                all_rows.append(row)
            except Exception as exc:
                print(f"[ERROR] p1 scope={scope} cell_or_arm={arm} "
                      f"self_loops={frozen[0]} row_normalize={frozen[1]}: {exc}",
                      flush=True)
                _traceback.print_exc()
                continue
        write_scorecard(all_rows, out_path / "scorecard.csv")
        print(f"[checkpoint] wrote {len(all_rows)} rows after scope={scope}",
              flush=True)

    write_scorecard(all_rows, out_path / "scorecard.csv")
    print(f"[done] wrote {out_path / 'scorecard.csv'} "
          f"({len(all_rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Task 6B: dual-landmark effective_size + per-gap binomial depletion helpers
# ---------------------------------------------------------------------------

def _corpus_book_rate(state: dict, book_type: str = "book") -> float:
    """Fraction of corpus works that are of ``book_type``."""
    types = np.asarray(state["types"])
    n = len(types)
    return float((types == book_type).sum()) / n if n else 0.0


def _landmark_global_idx(state: dict, work_type: str) -> np.ndarray:
    """Global indices of all works whose type matches ``work_type``."""
    return np.where(np.asarray(state["types"]) == work_type)[0]


def _depletion_pvalue(size: int, target_count: int,
                      base_rate: float) -> float:
    """Left-tail binomial depletion p-value via the shared module.

    Guards: size <= 0 -> 1.0 (non-significant, degenerate region).
    """
    if size <= 0:
        return 1.0
    rs = score_region_depletion(
        region_id=0,
        size=int(size),
        target_count=int(target_count),
        base_rate=float(base_rate),
    )
    return float(rs.p_value)


if __name__ == "__main__":
    main()
