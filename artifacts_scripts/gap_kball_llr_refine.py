"""K-ball seed + LLR-vs-k refinement (Phase 3).

Takes K-ball CSV output and re-sizes each gap by LLR-vs-k argmax over
the corpus null. See spec:
``docs/superpowers/specs/2026-05-12-kball-llr-refinement-design.md``.

Pipeline:
  K-ball CSV -> per-gap LLR sweep -> optional one re-centering -> dedup
  + rerank -> refined CSV.

Reuses ``kernel_mass_matrix``, ``setup``, ``_gaussian_weights``, and
``K_GRID`` from ``gap_static_seed_diagnostic``. Reuses the K-ball CSV
loader pattern from ``gap_kball_cohesion``. Ports the local-CSLS-
balanced posterior inline.

Usage::

    .conda/bin/python -u -m artifacts_scripts.gap_kball_llr_refine \\
        --scope 17/1702/T11303 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T11303__pca__drop0.csv \\
        --out artifacts/gap_kball_llr_refine/T11303__llr_refined.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_static_seed_diagnostic import (
    K_GRID,
    K_QUERY,
    _gaussian_weights,
    kernel_mass_matrix,
    setup,
)

try:
    from bookgap.jobs import stop_requested as _stop_requested
except Exception:  # bookgap.jobs may not be importable in tests / one-offs
    def _stop_requested() -> bool:
        return False


def _abort_if_stop_requested(stage: str, i: int, n: int) -> None:
    """Raise SystemExit if the bookgap.jobs watchdog has flipped stop_requested.

    Inner loops should call this between iterations so the watchdog's
    RAM/disk thresholds can actually fire before the kernel OOMs us. See
    ``CLAUDE.md`` for the watchdog's default thresholds (80% RAM RLIMIT_AS
    with 95% inner ceiling; 90% disk).
    """
    if _stop_requested():
        print(f"[abort] stop_requested at {stage} {i}/{n}; exiting", flush=True)
        raise SystemExit(0)


def binary_set_jaccard(a: set, b: set) -> float:
    """Set Jaccard |a ∩ b| / |a ∪ b|. Returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def load_kball_csv(path: Path) -> list[dict]:
    """Load K-ball CSV rows, parsing article_indices_json.

    Returns a list of dicts with keys:
      gap_idx (int), gap_size (int), topic_id (str), article_indices (list[int]).
    """
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_idx": int(r["gap_idx"]),
                "gap_size": int(r["gap_size"]),
                "topic_id": r["topic_id"],
                "article_indices": json.loads(r["article_indices_json"]),
            })
    return rows


def refine_gap_via_llr(
    member_indices: list[int],
    state: dict,
    null_log_mean: np.ndarray,
    recenter: bool = True,
) -> dict:
    """Refine a single K-ball gap by LLR-vs-k argmax around its centroid.

    member_indices: list of global article indices (from K-ball's article_indices_json).
    state: shared state dict from gap_static_seed_diagnostic.setup().
    null_log_mean: (len(K_GRID),) precomputed mean of log(corpus null kernel mass)
      at each k in K_GRID.
    recenter: if True, perform one re-centering step after picking k*.

    Returns dict with keys:
      refined_k_star (int), refined_member_indices (list[int] of length k*),
      llr_curve (list[float] of length len(K_GRID)),
      llr_peak (float = max of llr_curve), llr_peak_at_k (int).
    """
    emb = state["emb"]
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]

    members = np.asarray(member_indices, dtype=np.int64)
    centroid = emb[members].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        # Degenerate centroid (shouldn't happen with normalized embeddings).
        # Return the K-ball members verbatim, k* = original size.
        k_star = int(len(member_indices))
        return {
            "refined_k_star": k_star,
            "refined_member_indices": list(member_indices),
            "llr_curve": [0.0] * len(K_GRID),
            "llr_peak": 0.0,
            "llr_peak_at_k": k_star,
        }
    centroid = (centroid / norm).astype(np.float32)

    # Top-K_QUERY nearest articles to centroid.
    D_nn, I_nn_local = article_index.search(centroid[None, :], k=K_QUERY)
    D_nn = D_nn[0]
    I_nn_global = article_global_idx[I_nn_local[0]]

    # Kernel mass at each k in K_GRID. The centroid is virtual (not a work),
    # so we use the bandwidth of the nearest article seed as sigma_seed.
    sigma_seed = float(sigma[I_nn_global[0]])
    sigma_n = sigma[I_nn_global]
    W = np.exp(-D_nn / (sigma_seed * sigma_n))
    cum = np.cumsum(W)
    M_k = np.array([cum[k - 1] for k in K_GRID], dtype=np.float64)

    llr_curve = np.log(np.maximum(M_k, 1e-30)) - null_log_mean
    best_idx = int(np.argmax(llr_curve))
    k_star = int(K_GRID[best_idx])
    llr_peak = float(llr_curve[best_idx])

    refined_members = I_nn_global[:k_star].tolist()

    if recenter:
        # One step: kernel-weighted centroid of refined_members.
        member_emb = emb[refined_members]
        weights = _gaussian_weights(
            D_nn[:k_star], sigma_seed, sigma[I_nn_global[:k_star]],
        )
        denom = float(weights.sum())
        if denom > 1e-30:
            new_centroid = (weights[:, None] * member_emb).sum(axis=0) / denom
            new_norm = float(np.linalg.norm(new_centroid))
            if new_norm >= 1e-12:
                new_centroid = (new_centroid / new_norm).astype(np.float32)
                _, I_new_local = article_index.search(
                    new_centroid[None, :], k=k_star,
                )
                refined_members = article_global_idx[I_new_local[0]].tolist()

    return {
        "refined_k_star": k_star,
        "refined_member_indices": [int(x) for x in refined_members],
        "llr_curve": [float(x) for x in llr_curve],
        "llr_peak": llr_peak,
        "llr_peak_at_k": k_star,
    }


def compute_local_csls_posterior(
    members: list[int],
    D_NN: np.ndarray,
    I_NN: np.ndarray,
    sigma: np.ndarray,
    nearest_book_per_member: np.ndarray,
    book_to_arts: dict[int, np.ndarray],
    n_total: int,
    fuzzy_gap_membership: np.ndarray,
    aggregate_member_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Local-CSLS-balanced per-member posterior with fuzzy gap membership.

    For each gap member i, posterior is:
      post_i = K_G(i) / (K_G(i) + K_B(i))

    K_G is the *expected* self-tuning-Gaussian kernel mass from i to its
    top-K neighbors, weighted by each neighbor's fuzzy membership in
    THIS gap:
        K_G(i) = Σ_j W_ij · μ_G(j) · 1[j ≠ i]

    K_B is the kernel mass of i's K-NN that are in the local-CSLS-
    balanced background B (one book contributes f(b) non-gap articles
    where f(b) = number of gap members whose 1-NN book is b).

    **Both K_G and K_B are fuzzy** via the same joint-normalized
    membership matrix. K_G weights each neighbor by μ_G(j); K_B weights
    each CSLS-background article by (1 − μ_G(j)) — i.e., a background
    article that's partially claimed by THIS gap (because it's near the
    centroid) contributes less to the bounding-object kernel mass. The
    symmetric (1 − μ_G) form keeps K_B's role as a size-matched
    alternative (per docs/cohesion_design_axes.md axis H) but avoids
    the structural asymmetry of a fuzzy numerator over a hard denominator.

    To make (1 − μ_G) meaningfully non-1 on background articles,
    ``fuzzy_gap_membership`` must be computed over the extended
    (top-K_QUERY around centroid) per-gap support, not just the gap's
    hard k\*-member set. That extended joint-normalization is built by
    ``_build_fuzzy_jointnorm_memberships``. See the Phase 3.5 follow-up
    in ``docs/static_seed_cohesion_diagnostic_phase3_5.md``.

    members: list of global indices for the gap.
    D_NN: (n_members, K) squared L2 distances from each member to its
      top-K article neighbors.
    I_NN: (n_members, K) global indices of those neighbors.
    sigma: (n_total,) self-tuning bandwidths.
    nearest_book_per_member: (n_members,) global index of each member's
      1-NN boundary book.
    book_to_arts: dict mapping global book index -> array of article
      global indices, sorted by distance from the book (closest first).
    n_total: total number of works in the topic (for bool-array sizing).
    fuzzy_gap_membership: (n_total,) array of per-article fuzzy
      memberships in THIS gap (in [0, 1]). Typically the gap's column
      of a joint-normalized centroid-kernel-weight matrix; can also be
      a boolean indicator for single-gap inspection.
    aggregate_member_indices: when provided, ``D_NN``/``I_NN`` rows
      correspond to a SUBSET of ``members`` (specifically the members at
      these indices). ``in_G_hard`` and the CSLS background are still
      built from the FULL ``members`` / ``nearest_book_per_member``, so
      a partial aggregation gets the same background context as a full
      one — only the per-member aggregation loop is restricted. Used by
      ``compute_gap_stats`` for member subsampling on large gaps.

    Returns posts: per-member posteriors in [0, 1]. Shape matches
    D_NN.shape[0] — i.e., ``len(aggregate_member_indices)`` when subset
    aggregation is in use, otherwise ``len(members)``.
    """
    from collections import Counter
    members_arr = np.asarray(members, dtype=np.int64)

    in_G_hard = np.zeros(n_total, dtype=bool)
    in_G_hard[members_arr] = True

    # Count books claimed by gap members (drives the CSLS-balanced background).
    # Always uses the FULL nearest_book_per_member so the background sizing
    # reflects the full gap context, even when aggregating over a subset.
    book_counts: Counter = Counter()
    for b in nearest_book_per_member:
        book_counts[int(b)] += 1

    # Build the background lookup (hard, per axis H bounding-object semantics)
    in_B = np.zeros(n_total, dtype=bool)
    for b, f_b in book_counts.items():
        if b < 0:
            continue
        in_B[b] = True
        arts = book_to_arts.get(b, np.array([], dtype=np.int64))
        taken = 0
        for a in arts:
            if taken >= f_b:
                break
            a_int = int(a)
            if not in_G_hard[a_int]:
                in_B[a_int] = True
                taken += 1

    # Which members do D_NN / I_NN rows correspond to?
    if aggregate_member_indices is None:
        seed_members = members_arr
    else:
        seed_members = members_arr[np.asarray(aggregate_member_indices, dtype=np.int64)]

    # Per-member kernel weights from i to its top-K neighbors
    sigma_seed = sigma[seed_members][:, None]
    sigma_n = sigma[I_NN]
    W = np.exp(-D_NN / (sigma_seed * sigma_n))
    self_mask = I_NN == seed_members[:, None]
    mu_at_neighbors = fuzzy_gap_membership[I_NN]
    # Fuzzy K_G: weight each neighbor by μ_G(j) and zero out self-contributions.
    K_G = (W * mu_at_neighbors * (~self_mask)).sum(axis=1)
    # Fuzzy K_B: weight each CSLS-background article by (1 − μ_G(j)) so that
    # a background article partially claimed by THIS gap contributes less.
    B_mask = in_B[I_NN]
    K_B = (W * B_mask * (1.0 - mu_at_neighbors)).sum(axis=1)
    denom = K_G + K_B + 1e-30
    return K_G / denom


def compute_gap_stats(
    refined_member_indices: list[int],
    state: dict,
    book_global_idx: np.ndarray,
    book_index,
    article_index,
    article_global_idx: np.ndarray,
    fuzzy_gap_membership: np.ndarray,
    k_query: int = K_QUERY,
    k_per_book: int = K_QUERY,
    member_subsample_size: int | None = None,
    rng_seed: int | None = None,
) -> dict:
    """Compute per-gap stats from the local-CSLS-balanced posterior.

    Per cohesion_design_axes.md axis H, K_G is fuzzy (weighted by
    ``fuzzy_gap_membership``) while K_B is hard (the bounding-object
    semantics don't admit fuzzification). See compute_local_csls_posterior.

    When ``member_subsample_size`` is set and the gap has more members
    than that, a uniform random subset of that many members is used for
    the expensive per-member HNSW query + posterior aggregation. The CSLS
    background is still built from ALL members' nearest books, so the
    sampled members get the same context they would in a full pass —
    only the aggregation domain shrinks. ``median_logit`` and
    ``min_logit`` are subsample-unbiased; ``logit_sum`` and
    ``effective_size`` are subsample-scaled (not extrapolated).

    Returns dict with: logit_sum, mean_logit, median_logit, min_logit,
    effective_size, min_post, mean_post.
    """
    emb = state["emb"]
    sigma = state["sigma"]
    full_index = state["full_index"]

    members = np.asarray(refined_member_indices, dtype=np.int64)
    if len(members) < 2:
        return {
            "logit_sum": 0.0, "mean_logit": 0.0,
            "median_logit": 0.0, "min_logit": 0.0,
            "effective_size": 0.0,
            "min_post": 0.0, "mean_post": 0.0,
        }

    # Always use ALL members to build the CSLS background — subsampling only
    # changes which members get per-member kNN + posterior aggregation, not
    # the background sizing.
    all_member_emb = np.ascontiguousarray(emb[members])
    _, I_book = book_index.search(all_member_emb, k=1)
    nearest_book = book_global_idx[I_book[:, 0]]

    unique_books = np.unique(nearest_book)
    if len(unique_books) > 0:
        book_emb_q = np.ascontiguousarray(emb[unique_books])
        _, I_book_arts_local = article_index.search(book_emb_q, k=k_per_book)
        book_to_arts = {
            int(b): article_global_idx[I_book_arts_local[r]]
            for r, b in enumerate(unique_books)
        }
    else:
        book_to_arts = {}

    # Decide which members to aggregate over
    if member_subsample_size is not None and len(members) > member_subsample_size:
        rng = np.random.default_rng(rng_seed if rng_seed is not None else 0)
        agg_indices = np.sort(
            rng.choice(len(members), member_subsample_size, replace=False),
        )
    else:
        agg_indices = None

    if agg_indices is None:
        agg_member_emb = all_member_emb
    else:
        agg_member_emb = np.ascontiguousarray(emb[members[agg_indices]])

    D_NN, I_NN_full = full_index.search(agg_member_emb, k=k_query + 1)
    D_NN = D_NN[:, 1:]
    I_NN_full = I_NN_full[:, 1:]

    n_total = sigma.shape[0]
    posts = compute_local_csls_posterior(
        members=list(map(int, members)),
        D_NN=D_NN.astype(np.float32),
        I_NN=I_NN_full.astype(np.int64),
        sigma=sigma,
        nearest_book_per_member=nearest_book.astype(np.int64),
        book_to_arts=book_to_arts,
        n_total=n_total,
        fuzzy_gap_membership=fuzzy_gap_membership,
        aggregate_member_indices=agg_indices,
    )

    # Clipped logits for size-extensive statistics (logit_sum / mean_logit).
    # Clipping prevents log(0) and log of (1 - 1) from blowing the sum to ±inf
    # when a single member has post ≈ 0 or ≈ 1.
    post_clip = np.clip(posts, 1e-3, 1 - 1e-3)
    logits_clip = np.log(post_clip / (1 - post_clip))
    # UN-clipped logits for median / min, which are robust to ±inf in the
    # tails without needing artificial clip thresholds. log(post / max(1-post, ε))
    # avoids divide-by-zero only when post is exactly 1; np.median handles
    # remaining -inf in the lower tail by sort order.
    with np.errstate(divide="ignore"):
        logits_raw = np.log(posts) - np.log(np.maximum(1.0 - posts, 1e-30))
    median_logit = float(np.median(logits_raw))
    finite_raw = logits_raw[np.isfinite(logits_raw)]
    min_logit = float(finite_raw.min()) if finite_raw.size else 0.0
    return {
        "logit_sum": float(logits_clip.sum()),
        "mean_logit": float(logits_clip.mean()),
        "median_logit": median_logit,
        "min_logit": min_logit,
        "effective_size": float(posts.sum()),
        "min_post": float(posts.min()),
        "mean_post": float(posts.mean()),
    }


def dedup_via_containment_then_jaccard(
    gaps: list[dict],
    jaccard_threshold: float = 0.5,
    containment_threshold: float = 0.9,
    rank_metric: str = "median_logit",
) -> list[dict]:
    """Two-stage dedup: containment pruning first, then Jaccard merging.

    Stage 1 (containment pruning): for each pair (A, B) where |A| > |B|,
    containment(B in A) ≥ ``containment_threshold``, and B's
    ``rank_metric`` is strictly greater than A's: drop A. This handles
    the "diffuse saturator wrapping a tight inner core" pattern that
    Jaccard misses (set-overlap is small, ~|B|/|A|, but B is fully
    contained in A and the tight core is the more informative cluster).

    Stage 2 (Jaccard merging): union-find on remaining gaps where
    ``binary_set_jaccard ≥ jaccard_threshold``. Keeps highest-
    ``rank_metric`` survivor per component. Handles symmetric near-
    duplicates that containment misses when sizes / metrics are tied.

    Empirical justification (cache/.../T10181_n1000_dedup_analysis):
    the two stages are essentially orthogonal — only 2/421 firings
    on T10181 1000-gap are caught by both criteria; the other 419
    are caught by exactly one.

    Uses an inverted index (article → gap indices) to enumerate only
    pairs sharing at least one article. On T10181 1000-gap this drops
    pair count from N²/2 = 500k (naive O(N²)) to 61k overlapping pairs
    — ~8× speedup before any per-pair savings.

    Annotates each input gap with:
      survivor: 1 if this gap survives both stages, 0 otherwise.
      merged_into_gap_id: gap_idx of the cluster this merged/pruned into
        (-1 for survivors).
      dedup_reason: "survivor" | "containment_pruned" | "jaccard_merged".

    Returns the annotated list (same order as input).
    """
    from collections import defaultdict
    from itertools import combinations

    n = len(gaps)
    sets = [frozenset(g["refined_member_indices"]) for g in gaps]
    sizes = [len(s) for s in sets]

    # Inverted index: article -> gap indices containing it.
    article_to_gap_indices: dict[int, list[int]] = defaultdict(list)
    for gi, s in enumerate(sets):
        for art in s:
            article_to_gap_indices[art].append(gi)

    # Enumerate unique overlapping pairs, accumulating intersection size.
    inter_counts: dict[tuple[int, int], int] = defaultdict(int)
    n_articles_seen = 0
    for glist in article_to_gap_indices.values():
        n_articles_seen += 1
        if n_articles_seen % 50_000 == 0:
            _abort_if_stop_requested(
                "dedup-pair-enum", n_articles_seen, len(article_to_gap_indices))
        if len(glist) < 2:
            continue
        for i, j in combinations(glist, 2):
            key = (i, j) if i < j else (j, i)
            inter_counts[key] += 1

    # Stage 1: containment pruning.
    dropped = [False] * n
    pruned_by_gap_idx: list[int | None] = [None] * n
    for (i, j), inter in inter_counts.items():
        si, sj = sizes[i], sizes[j]
        if si == sj:
            # Containment pruning with strict-greater tiebreak doesn't
            # fire on equal-size pairs; defer to Stage 2.
            continue
        smaller_idx, larger_idx = (i, j) if si < sj else (j, i)
        cont = inter / min(si, sj)
        if cont < containment_threshold:
            continue
        if gaps[smaller_idx][rank_metric] > gaps[larger_idx][rank_metric]:
            dropped[larger_idx] = True
            pruned_by_gap_idx[larger_idx] = gaps[smaller_idx]["gap_idx"]

    # Stage 2: Jaccard union-find on non-dropped gaps.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j), inter in inter_counts.items():
        if dropped[i] or dropped[j]:
            continue
        union_size = sizes[i] + sizes[j] - inter
        if union_size <= 0:
            continue
        if inter / union_size >= jaccard_threshold:
            union(i, j)

    components: dict[int, list[int]] = {}
    for idx in range(n):
        if dropped[idx]:
            continue
        components.setdefault(find(idx), []).append(idx)

    survivor_of_comp: dict[int, int] = {}
    for root, comp in components.items():
        best = max(comp, key=lambda i: gaps[i][rank_metric])
        survivor_of_comp[root] = best

    annotated: list[dict] = []
    for idx, g in enumerate(gaps):
        g_out = dict(g)
        if dropped[idx]:
            g_out["survivor"] = 0
            g_out["merged_into_gap_id"] = int(pruned_by_gap_idx[idx])
            g_out["dedup_reason"] = "containment_pruned"
        else:
            root = find(idx)
            survivor_idx = survivor_of_comp[root]
            if idx == survivor_idx:
                g_out["survivor"] = 1
                g_out["merged_into_gap_id"] = -1
                g_out["dedup_reason"] = "survivor"
            else:
                g_out["survivor"] = 0
                g_out["merged_into_gap_id"] = gaps[survivor_idx]["gap_idx"]
                g_out["dedup_reason"] = "jaccard_merged"
        annotated.append(g_out)
    return annotated


def dedup_via_binary_jaccard(
    gaps: list[dict],
    threshold: float = 0.5,
    rank_metric: str = "logit_sum",
) -> list[dict]:
    """Backward-compat shim: Jaccard-only dedup (no containment pruning).

    Calls ``dedup_via_containment_then_jaccard`` with
    ``containment_threshold=2.0`` (above the max possible value) so the
    containment stage never fires. Default ``rank_metric="logit_sum"``
    preserves the pre-Phase-3.5 tiebreak.
    """
    return dedup_via_containment_then_jaccard(
        gaps,
        jaccard_threshold=threshold,
        containment_threshold=2.0,
        rank_metric=rank_metric,
    )


def _topic_id_from_path(path: Path) -> str:
    """Derive a topic_id like 'T11303' from a K-ball CSV filename."""
    stem = path.stem
    parts = stem.split("__")[0].split("-")
    if len(parts) >= 3 and parts[-1].startswith("T"):
        return parts[-1]
    if stem.startswith("T") and "_" in stem:
        return stem.split("_")[0]
    return stem


def _build_fuzzy_jointnorm_memberships(
    refined_gaps: list[dict],
    state: dict,
) -> "csc_matrix":
    """Build joint-normalized fuzzy gap memberships over EXTENDED support.

    For each refined gap, compute the centroid of its members. Query the
    article_index for the top-K_QUERY nearest articles to that centroid
    (which includes the gap's hard members PLUS nearby non-members), and
    use the self-tuning-Gaussian kernel weight from centroid to each of
    those neighbors as the unnormalized μ_G value.

    The extended support is what makes the (1 − μ_G) factor in K_B
    meaningful: a CSLS-background article that's also near our centroid
    gets a nonzero μ_G in our gap (and thus a (1 − μ_G) < 1 weight in
    K_B), so it contributes less to the bounding-object mass. Without
    extending past hard members, μ_G is 0 on every background article
    and (1 − μ_G) is identically 1 — no effect.

    Returns the joint-normalized matrix in CSC form for cheap per-gap
    column extraction. To get gap g's fuzzy membership vector:

        mu_g = np.asarray(M_norm[:, g].todense()).flatten()
    """
    from scipy.sparse import csr_matrix, diags
    emb = state["emb"]
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    n_total = state["sigma"].shape[0]
    n_gaps = len(refined_gaps)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for g_idx, g in enumerate(refined_gaps):
        if g_idx % 100 == 0:
            _abort_if_stop_requested("fuzzy-build", g_idx, len(refined_gaps))
        members = np.asarray(
            g["refined_member_indices"], dtype=np.int64,
        )
        if len(members) == 0:
            continue
        member_emb = emb[members]
        centroid = member_emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            # Degenerate centroid (shouldn't happen with normalized embeddings).
            # Fall back to equal weights on the hard member set.
            for m in members:
                rows.append(int(m))
                cols.append(g_idx)
                vals.append(1.0)
            continue
        centroid_unit = (centroid / norm).astype(np.float32)
        # Extended support: top-K_QUERY articles near this gap's centroid.
        D_centroid, I_centroid_local = article_index.search(
            centroid_unit[None, :], k=K_QUERY,
        )
        D_centroid = D_centroid[0]
        I_centroid_global = article_global_idx[I_centroid_local[0]]
        sigma_centroid = float(sigma[members].mean())
        sigma_neighbors = sigma[I_centroid_global]
        w = np.exp(-D_centroid / (sigma_centroid * sigma_neighbors))
        for n_g, wi in zip(I_centroid_global, w):
            rows.append(int(n_g))
            cols.append(g_idx)
            vals.append(float(wi))

    M = csr_matrix((vals, (rows, cols)), shape=(n_total, n_gaps))
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    inv = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    M_norm = (diags(inv) @ M).tocsc()
    return M_norm


def _compute_corpus_null(
    state: dict,
    sample_size: int | None = 2000,
    rng_seed: int = 0,
) -> np.ndarray:
    """Precompute mean(log(M_k)) over article seeds at each k in K_GRID.

    The full-corpus null over N_articles seeds is the most-precise reference
    but is the slow path when extending K_GRID past the cached top-K_QUERY
    on large topics. Per the subsampling diagnostic
    (``cache/null_subsampling_diagnostic/T11303.csv``), m=2000 random article
    seeds gives relative SE ≤ 0.2% on every k in the current grid — orders
    of magnitude smaller than the LLR signal we care about. Default to
    sample_size=2000; pass ``sample_size=None`` to use every article seed.
    """
    art_idx = state["article_global_idx"]
    n_articles = len(art_idx)
    if sample_size is None or sample_size >= n_articles:
        seeds = art_idx
        effective_m = n_articles
    else:
        rng = np.random.default_rng(rng_seed)
        sample_local = rng.choice(n_articles, size=sample_size, replace=False)
        seeds = art_idx[sample_local]
        effective_m = sample_size
    M_null = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][seeds],
        I_art_NN_global=state["I_art_NN_all_global"][seeds],
        sigma=state["sigma"],
        k_grid=K_GRID,
        seed_global_idx=seeds,
    )
    print(f"  null computed from {effective_m} article seeds "
          f"(out of {n_articles} total)", flush=True)
    return np.log(np.maximum(M_null, 1e-30)).mean(axis=0)


def _compute_corpus_null_stein(
    state: dict,
    target_rel_se: float = 0.005,
    m_pilot: int = 200,
    m_min: int = 500,
    m_max: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Stein two-stage adaptive sub-sampling for the corpus null.

    Pilot pass: estimate σ_k on m_pilot random article seeds.
    Adaptive size: m_needed = max_k (σ_k / (target_rel_se · |null_k|))²
    so per-k relative SE on null_log_mean stays under ``target_rel_se``.
    Final pass: sample max(m_needed, m_min) seeds; reuse the pilot's
    kNN rows (since we already have them in ``state["D_art_NN_all"]``)
    and draw additional seeds only if needed.

    Returns (null_log_mean, info_dict). info_dict tracks pilot_sigma,
    pilot_mean, m_needed_per_k, m_used, so callers can log the chosen
    sample size.

    Notes:
    - Uses ``state["D_art_NN_all"]``; assumes setup() has already built
      the full top-K_QUERY kNN. This refactor does NOT save kNN compute
      (that's a Phase 4 setup() restructure). What it DOES do is pick m
      data-adaptively rather than guessing.
    - target_rel_se=0.005 ⇒ SE ≤ 0.5% of |null_k|. On T11303 / T10181
      this yields m_needed ≈ 500-1500 — well below the m=2000 fixed
      default and tighter precision.
    """
    art_idx = state["article_global_idx"]
    n_articles = len(art_idx)
    rng = np.random.default_rng(rng_seed)

    # Pilot pass
    m_pilot = min(m_pilot, n_articles)
    pilot_local = rng.choice(n_articles, size=m_pilot, replace=False)
    pilot_seeds = art_idx[pilot_local]
    M_pilot = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][pilot_seeds],
        I_art_NN_global=state["I_art_NN_all_global"][pilot_seeds],
        sigma=state["sigma"],
        k_grid=K_GRID,
        seed_global_idx=pilot_seeds,
    )
    log_M_pilot = np.log(np.maximum(M_pilot, 1e-30))
    pilot_sigma = log_M_pilot.std(axis=0, ddof=1)
    pilot_mean = log_M_pilot.mean(axis=0)

    # Per-k m_needed = (σ / (target_rel_se · |null|))²; take max across k
    abs_target_se = target_rel_se * np.maximum(np.abs(pilot_mean), 1e-30)
    m_needed_per_k = (pilot_sigma / abs_target_se) ** 2
    m_needed = int(np.ceil(m_needed_per_k.max()))

    cap = m_max if m_max is not None else n_articles
    m_used = max(m_min, min(m_needed, cap, n_articles))

    print(f"  stein pilot: m={m_pilot}; "
          f"σ_k={[f'{v:.3f}' for v in pilot_sigma]}; "
          f"m_needed={m_needed} (capped at {cap}); m_used={m_used}",
          flush=True)

    # Final pass — extend pilot to m_used by drawing additional unique seeds
    if m_used <= m_pilot:
        full_local = pilot_local[:m_used]
    else:
        remaining = np.setdiff1d(np.arange(n_articles), pilot_local)
        extra = rng.choice(remaining, size=m_used - m_pilot, replace=False)
        full_local = np.concatenate([pilot_local, extra])
    full_seeds = art_idx[full_local]

    M_full = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][full_seeds],
        I_art_NN_global=state["I_art_NN_all_global"][full_seeds],
        sigma=state["sigma"],
        k_grid=K_GRID,
        seed_global_idx=full_seeds,
    )
    null_log_mean = np.log(np.maximum(M_full, 1e-30)).mean(axis=0)
    info = {
        "method": "stein",
        "m_pilot": m_pilot,
        "m_used": m_used,
        "m_needed": m_needed,
        "target_rel_se": target_rel_se,
        "pilot_sigma": pilot_sigma.tolist(),
        "pilot_mean": pilot_mean.tolist(),
        "m_needed_per_k": m_needed_per_k.tolist(),
    }
    return null_log_mean, info


def _compute_corpus_null_stein_stratified(
    state: dict,
    n_bins: int = 10,
    target_rel_se: float = 0.005,
    m_pilot_per_bin: int = 200,
    m_min_per_bin: int = 200,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """σ-stratified Stein two-stage null: condition on local density bin.

    Each article has σ_i = distance to its k_sigma-th NN; small σ = dense
    local neighborhood, large σ = sparse. Binning σ into deciles (or
    n_bins quantile bins) gives a "local density" stratification. We then
    run Stein's two-stage adaptive sampling separately inside each bin to
    estimate ``null_log_mean[k, b] = E[log mass(i, k) | σ_i ∈ bin b]``.

    For a gap at phase 1, the appropriate null is the conditional
    expectation given the gap's σ distribution — either via a single-bin
    lookup using the gap's mean σ (``"bin"`` mode in ``_null_for_gap``)
    or via a σ-histogram weighted average (``"dist"`` mode), with the
    latter being a strict refinement of the former.

    Returns ``(null_strat, bin_edges, info)`` where:
      - ``null_strat`` is ``(len(K_GRID), n_bins)``
      - ``bin_edges`` is ``(n_bins + 1,)`` σ quantile edges
        (``bin_edges[0]`` = min σ, ``bin_edges[-1]`` = max σ).
      - ``info`` has per-bin diagnostics.
    """
    art_idx = state["article_global_idx"]
    sigma_full = state["sigma"]
    sigma_arts = sigma_full[art_idx]
    n_articles = len(art_idx)
    rng = np.random.default_rng(rng_seed)

    # Quantile bin edges over articles' σ
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(sigma_arts, quantiles)
    # Assign each article to a bin in [0, n_bins-1]
    # interior edges = bin_edges[1:-1] (length n_bins-1)
    bins_per_article = np.digitize(sigma_arts, bin_edges[1:-1], right=False)

    null_strat = np.zeros((len(K_GRID), n_bins), dtype=np.float64)
    info_per_bin: dict[int, dict] = {}

    for b in range(n_bins):
        in_bin_local = np.flatnonzero(bins_per_article == b)
        n_in_bin = len(in_bin_local)
        if n_in_bin < 2:
            print(f"  bin {b}: only {n_in_bin} articles — skipping "
                  f"(falling back to global mean)", flush=True)
            info_per_bin[b] = {"n_in_bin": n_in_bin, "m_used": 0,
                               "sigma_lo": float(bin_edges[b]),
                               "sigma_hi": float(bin_edges[b + 1])}
            continue

        # Pilot
        m_pilot_actual = min(m_pilot_per_bin, n_in_bin)
        pilot_local = rng.choice(in_bin_local, size=m_pilot_actual, replace=False)
        pilot_seeds = art_idx[pilot_local]
        M_pilot = kernel_mass_matrix(
            D_art_NN=state["D_art_NN_all"][pilot_seeds],
            I_art_NN_global=state["I_art_NN_all_global"][pilot_seeds],
            sigma=sigma_full,
            k_grid=K_GRID,
            seed_global_idx=pilot_seeds,
        )
        log_M_pilot = np.log(np.maximum(M_pilot, 1e-30))
        pilot_sigma = log_M_pilot.std(axis=0, ddof=1)
        pilot_mean = log_M_pilot.mean(axis=0)

        # Stein step
        abs_target_se = target_rel_se * np.maximum(np.abs(pilot_mean), 1e-30)
        m_needed_per_k = (pilot_sigma / abs_target_se) ** 2
        m_needed = int(np.ceil(m_needed_per_k.max()))
        m_used = max(m_min_per_bin, min(m_needed, n_in_bin))

        # Final pass — extend pilot if needed
        if m_used <= m_pilot_actual:
            full_local = pilot_local[:m_used]
        else:
            remaining = np.setdiff1d(in_bin_local, pilot_local)
            take_extra = min(m_used - m_pilot_actual, len(remaining))
            extra = rng.choice(remaining, size=take_extra, replace=False)
            full_local = np.concatenate([pilot_local, extra])
        full_seeds = art_idx[full_local]
        M_full = kernel_mass_matrix(
            D_art_NN=state["D_art_NN_all"][full_seeds],
            I_art_NN_global=state["I_art_NN_all_global"][full_seeds],
            sigma=sigma_full,
            k_grid=K_GRID,
            seed_global_idx=full_seeds,
        )
        null_strat[:, b] = np.log(np.maximum(M_full, 1e-30)).mean(axis=0)
        info_per_bin[b] = {
            "n_in_bin": n_in_bin,
            "m_used": int(len(full_seeds)),
            "m_needed": m_needed,
            "sigma_lo": float(bin_edges[b]),
            "sigma_hi": float(bin_edges[b + 1]),
        }
        print(f"  bin {b}: n={n_in_bin}, m_used={len(full_seeds)}, "
              f"σ∈[{bin_edges[b]:.3f}, {bin_edges[b + 1]:.3f}], "
              f"null[k=80]={null_strat[3, b]:.3f}",
              flush=True)

    # Fill missing bins with the closest populated bin's nulls
    populated = [b for b in range(n_bins) if info_per_bin[b]["m_used"] > 0]
    for b in range(n_bins):
        if info_per_bin[b]["m_used"] == 0:
            nearest = min(populated, key=lambda x: abs(x - b))
            null_strat[:, b] = null_strat[:, nearest]

    info = {
        "method": "stein_stratified",
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "per_bin": info_per_bin,
    }
    return null_strat, bin_edges, info


def _null_for_gap(
    member_indices: list[int],
    sigma: np.ndarray,
    null_strat: np.ndarray,
    bin_edges: np.ndarray,
    mode: str = "bin",
) -> np.ndarray:
    """Per-gap null_log_mean of shape (len(K_GRID),).

    ``mode="bin"``: look up by the gap's MEAN σ (single bin).
    ``mode="dist"``: weight by the gap's σ histogram across bins (strict
        refinement of "bin" — each member casts its own vote).
    """
    member_sigma = sigma[np.asarray(member_indices, dtype=np.int64)]
    n_bins = null_strat.shape[1]
    if mode == "bin":
        mean_sigma = float(np.mean(member_sigma))
        bin_id = int(np.digitize(mean_sigma, bin_edges[1:-1], right=False))
        bin_id = max(0, min(bin_id, n_bins - 1))
        return null_strat[:, bin_id]
    if mode == "dist":
        bin_ids = np.digitize(member_sigma, bin_edges[1:-1], right=False)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        weights = (np.bincount(bin_ids, minlength=n_bins)
                   / len(member_sigma)).astype(np.float64)
        return (null_strat * weights[None, :]).sum(axis=1)
    raise ValueError(f"unknown null mode: {mode}")


def _calibrate_stats_params(
    refined_gaps: list[dict],
    state: dict,
    M_norm_csc,
    book_index,
    candidates_k: list[int],
    candidates_m: list[int | None],
    tolerance_delta: float,
    tolerance_alpha: float,
    pilot_size: int,
    rng_seed: int = 42,
) -> tuple[int, int | None, list[dict]]:
    """Pilot-based joint calibration of (k_query_stats, member_subsample).

    Strategy: pick ``pilot_size`` random gaps. For each, compute the full
    top-``K_max`` per-member NN once at ``K_max = max(candidates_k)``,
    along with the full CSLS background. Treat
    ``(K=K_max, m=all)`` as ground truth for ``median_logit``. Then sweep
    candidate (K', m') pairs in increasing cost; for each candidate,
    truncate the cached top-K_max NN to top-K' and aggregate over a
    random m'-subsample of members (or all if m' is None). Pick the
    smallest (K', m') such that

        P_{g ∈ pilot}(|median_logit(K', m') - median_logit(K_max, all)| > δ) ≤ α

    Same statistical framing as the Stein two-stage we use for the
    corpus null: pilot estimates the truncation/subsampling error
    distribution, adoption is data-driven, no hand-tuned scale factors.

    The clever bit is that once you have D_NN/I_NN at K_max for one gap,
    every smaller K' truncation is array-slicing (no extra HNSW work),
    and every m' subsample is a row-select on the same array. So the
    pilot cost is dominated by the K_max baseline computation, not the
    sweep — total ≈ ``pilot_size × HNSW(k* members, K_max)``.

    Returns ``(k_query_chosen, m_chosen, sweep_table)`` where
    ``sweep_table`` is a list of dicts describing the realized tolerance
    distribution at each candidate, suitable for logging.
    """
    emb = state["emb"]
    sigma = state["sigma"]
    full_index = state["full_index"]
    book_global_idx = state["book_global_idx"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    n_total = sigma.shape[0]

    K_max = max(candidates_k)
    # Sort candidates by ascending cost — prefer smaller m (linear cost in
    # HNSW queries), then smaller K (cheaper post-processing).
    pairs: list[tuple[int, int | None]] = []
    for m_p in sorted(candidates_m, key=lambda x: float("inf") if x is None else x):
        for K_p in sorted(candidates_k):
            if K_p == K_max and (m_p is None):
                continue  # this IS the baseline
            pairs.append((K_p, m_p))

    n_gaps = len(refined_gaps)
    rng = np.random.default_rng(rng_seed)
    # Stratify pilot by gap size so we cover small/medium/large/saturator
    # rather than oversampling tiny gaps (which are abundant and uninformative).
    sizes = np.array([len(g["refined_member_indices"]) for g in refined_gaps])
    # Drop trivially small gaps (<2 members produce degenerate stats).
    eligible = np.flatnonzero(sizes >= 2)
    if len(eligible) <= pilot_size:
        pilot_idxs = eligible.tolist()
    else:
        # Decile stratification on sizes
        deciles = np.quantile(sizes[eligible], np.linspace(0, 1, 11))
        per_decile = max(1, pilot_size // 10)
        pilot_idxs: list[int] = []
        for lo, hi in zip(deciles[:-1], deciles[1:]):
            in_decile = eligible[(sizes[eligible] >= lo) & (sizes[eligible] <= hi)]
            if len(in_decile) == 0:
                continue
            take = min(per_decile, len(in_decile))
            pilot_idxs.extend(rng.choice(in_decile, take, replace=False).tolist())
        # Fill to pilot_size with random eligible
        remaining = pilot_size - len(pilot_idxs)
        if remaining > 0:
            extras = rng.choice(
                np.setdiff1d(eligible, np.array(pilot_idxs)),
                min(remaining, len(eligible) - len(pilot_idxs)),
                replace=False,
            )
            pilot_idxs.extend(extras.tolist())
    pilot_idxs = sorted(set(pilot_idxs))[:pilot_size]
    print(f"[calibrate] pilot of {len(pilot_idxs)} gaps "
          f"(size range {sizes[pilot_idxs].min()}–{sizes[pilot_idxs].max()}, "
          f"median {int(np.median(sizes[pilot_idxs]))})",
          flush=True)

    # Phase A: compute baseline (K_max, m=all) per pilot gap + cache D_NN/I_NN
    # and CSLS background so the sweep can reuse them without re-querying HNSW.
    print(f"[calibrate] computing baseline at K={K_max}, m=all on pilot...",
          flush=True)
    pilot_cache: list[dict] = []
    baselines: list[float] = []
    for pi, g_idx in enumerate(pilot_idxs):
        if (pi + 1) % 25 == 0:
            print(f"  pilot baseline {pi + 1}/{len(pilot_idxs)}", flush=True)
        g = refined_gaps[g_idx]
        members = np.asarray(g["refined_member_indices"], dtype=np.int64)
        mu_g = np.asarray(M_norm_csc[:, g_idx].todense()).flatten()

        all_member_emb = np.ascontiguousarray(emb[members])
        _, I_book = book_index.search(all_member_emb, k=1)
        nearest_book = book_global_idx[I_book[:, 0]].astype(np.int64)
        unique_books = np.unique(nearest_book)
        if len(unique_books) > 0:
            book_emb_q = np.ascontiguousarray(emb[unique_books])
            _, I_bart_local = article_index.search(book_emb_q, k=K_max)
            book_to_arts = {
                int(b): article_global_idx[I_bart_local[r]]
                for r, b in enumerate(unique_books)
            }
        else:
            book_to_arts = {}

        D_NN_max, I_NN_max = full_index.search(all_member_emb, k=K_max + 1)
        D_NN_max = D_NN_max[:, 1:].astype(np.float32)
        I_NN_max = I_NN_max[:, 1:].astype(np.int64)

        posts_baseline = compute_local_csls_posterior(
            members=list(map(int, members)),
            D_NN=D_NN_max, I_NN=I_NN_max, sigma=sigma,
            nearest_book_per_member=nearest_book,
            book_to_arts=book_to_arts, n_total=n_total,
            fuzzy_gap_membership=mu_g,
        )
        with np.errstate(divide="ignore"):
            logits = np.log(posts_baseline) - np.log(
                np.maximum(1 - posts_baseline, 1e-30),
            )
        baselines.append(float(np.median(logits)))
        pilot_cache.append({
            "members": members, "mu_g": mu_g,
            "D_NN": D_NN_max, "I_NN": I_NN_max,
            "nearest_book": nearest_book, "book_to_arts": book_to_arts,
        })
    baselines_arr = np.array(baselines)

    # Phase B: sweep (K', m') candidates in ascending cost
    print(f"[calibrate] sweeping {len(pairs)} (K', m') pairs "
          f"(δ={tolerance_delta}, α={tolerance_alpha})...",
          flush=True)
    sweep: list[dict] = []
    chosen: tuple[int, int | None] | None = None
    for K_p, m_p in pairs:
        deltas: list[float] = []
        for pi, pc in enumerate(pilot_cache):
            members = pc["members"]
            mu_g = pc["mu_g"]
            D_NN_full = pc["D_NN"]
            I_NN_full = pc["I_NN"]
            nearest_book = pc["nearest_book"]
            book_to_arts = pc["book_to_arts"]
            # Truncate to top-K' (array slice; no HNSW)
            D_NN_p = D_NN_full[:, :K_p]
            I_NN_p = I_NN_full[:, :K_p]
            # Subsample members
            if m_p is not None and len(members) > m_p:
                rng_g = np.random.default_rng(
                    rng_seed + 7919 * pilot_idxs[pi],
                )
                agg_idx = np.sort(rng_g.choice(len(members), m_p, replace=False))
                D_NN_pa = D_NN_p[agg_idx]
                I_NN_pa = I_NN_p[agg_idx]
            else:
                agg_idx = None
                D_NN_pa = D_NN_p
                I_NN_pa = I_NN_p
            posts = compute_local_csls_posterior(
                members=list(map(int, members)),
                D_NN=D_NN_pa, I_NN=I_NN_pa, sigma=sigma,
                nearest_book_per_member=nearest_book,
                book_to_arts=book_to_arts, n_total=n_total,
                fuzzy_gap_membership=mu_g,
                aggregate_member_indices=agg_idx,
            )
            with np.errstate(divide="ignore"):
                logits = np.log(posts) - np.log(
                    np.maximum(1 - posts, 1e-30),
                )
            deltas.append(abs(float(np.median(logits)) - baselines_arr[pi]))
        deltas_arr = np.array(deltas)
        p_high = float(np.quantile(deltas_arr, 1 - tolerance_alpha))
        passes = p_high <= tolerance_delta
        sweep.append({
            "k_query": K_p, "member_subsample": m_p,
            "p_high_abs_delta": p_high,
            "p_median_abs_delta": float(np.median(deltas_arr)),
            "p_max_abs_delta": float(deltas_arr.max()),
            "passes": passes,
        })
        print(f"  K={K_p:>4}  m={'all' if m_p is None else m_p:>4}  "
              f"p{int((1 - tolerance_alpha) * 100)}|Δ|={p_high:.4f}  "
              f"max|Δ|={deltas_arr.max():.3f}  "
              f"{'PASS' if passes else 'fail'}", flush=True)
        if passes and chosen is None:
            chosen = (K_p, m_p)
            # First pass in cost-ascending order is the adopted choice;
            # keep sweeping to log the full table.

    if chosen is None:
        print(f"[calibrate] no candidate met tolerance; falling back to "
              f"(K={K_max}, m=all)", flush=True)
        chosen = (K_max, None)

    print(f"[calibrate] CHOSEN: k_query_stats={chosen[0]}, "
          f"member_subsample={'all' if chosen[1] is None else chosen[1]}",
          flush=True)
    return chosen[0], chosen[1], sweep


def _write_refined_csv(out_path: Path, scope: str, gaps: list[dict]) -> None:
    """Write the refined CSV with the spec'd schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope", "kball_gap_id", "kball_size", "refined_k_star",
        "refined_n_articles", "llr_peak", "llr_peak_at_k",
        "logit_sum", "mean_logit", "median_logit", "min_logit",
        "effective_size",
        "kball_size_delta", "kball_jaccard",
        "survivor", "merged_into_gap_id", "dedup_reason",
        "article_indices_json",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for g in gaps:
            w.writerow({
                "scope": scope,
                "kball_gap_id": g["kball_gap_id"],
                "kball_size": g["kball_size"],
                "refined_k_star": g["refined_k_star"],
                "refined_n_articles": g["refined_n_articles"],
                "llr_peak": g["llr_peak"],
                "llr_peak_at_k": g["llr_peak_at_k"],
                "logit_sum": g["logit_sum"],
                "mean_logit": g["mean_logit"],
                "median_logit": g["median_logit"],
                "min_logit": g["min_logit"],
                "effective_size": g["effective_size"],
                "kball_size_delta": g["kball_size_delta"],
                "kball_jaccard": g["kball_jaccard"],
                "survivor": g["survivor"],
                "merged_into_gap_id": g["merged_into_gap_id"],
                "dedup_reason": g.get("dedup_reason", ""),
                "article_indices_json": json.dumps(g["refined_member_indices"]),
            })


def _plot_llr_curves(
    gaps: list[dict], top_n: int, out_path: Path,
    rank_by: str = "mean_logit",
) -> None:
    """Small-multiples plot of LLR-vs-k curves for the top-N gaps by ``rank_by``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = sorted(gaps, key=lambda g: -g[rank_by])[:top_n]
    n = len(top)
    if n == 0:
        return
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows), squeeze=False)
    for j, g in enumerate(top):
        ax = axes[j // cols][j % cols]
        ax.plot(K_GRID, g["llr_curve"], color="tab:orange")
        ax.axvline(g["refined_k_star"], color="red", linestyle="--", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_title(f"kb-{g['kball_gap_id']} (k*={g['refined_k_star']})", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"LLR-vs-k curves for top-{top_n} refined gaps by logit_sum")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_size_delta_hist(gaps: list[dict], out_path: Path) -> None:
    """Histogram of refined-vs-K-ball size delta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    deltas = np.array([g["kball_size_delta"] for g in gaps])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(deltas, bins=50, color="tab:blue", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("refined_n_articles - kball_size")
    ax.set_ylabel("count")
    ax.set_title(f"K-ball -> LLR-refined size delta (n={len(deltas)} gaps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True, help="e.g. 17/1702/T11303")
    p.add_argument("--kball-csv", required=True, type=Path,
                   help="Existing K-ball CSV path")
    p.add_argument("--out", required=True, type=Path,
                   help="Refined CSV output path")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Optional directory for diagnostic figures")
    p.add_argument("--no-recenter", action="store_true",
                   help="Disable the one-step re-centering after k* selection")
    p.add_argument("--jaccard-threshold", type=float, default=0.5,
                   help="Binary set Jaccard threshold for stage-2 dedup merge "
                        "(default 0.5; catches symmetric same-size near-"
                        "duplicates)")
    p.add_argument("--containment-threshold", type=float, default=0.9,
                   help="Stage-1 containment-pruning threshold (default 0.9; "
                        "catches the diffuse-saturator-wrapping-tight-inner "
                        "pattern). Pass 2.0 to disable containment pruning "
                        "and revert to Jaccard-only dedup.")
    p.add_argument("--null-method",
                   choices=["fixed", "stein"], default="stein",
                   help="How to size the article-seed subsample for the corpus "
                        "null. 'fixed' uses --null-sample-size as-is (default 2000). "
                        "'stein' (default) runs a 200-seed pilot, estimates σ_k, "
                        "and adaptively picks m to hit --null-target-rel-se. "
                        "On topic-scale data both give the same result; stein "
                        "is the principled choice and adapts to larger scopes "
                        "without retuning.")
    p.add_argument("--null-sample-size", type=int, default=2000,
                   help="Article-seed subsample size for the corpus null when "
                        "--null-method=fixed. Per the subsampling diagnostic, "
                        "m=2000 gives ≤0.2%% relative SE on every k in K_GRID. "
                        "Auto-disabled when N_articles ≤ m (full corpus used). "
                        "Pass 0 to force full corpus regardless of size. "
                        "Default 2000.")
    p.add_argument("--null-target-rel-se", type=float, default=0.005,
                   help="Stein-only: target relative SE for the corpus null "
                        "(SE / |null_k| ≤ this value at every k). Default 0.005 "
                        "(≤0.5%% relative noise on null_log_mean).")
    p.add_argument("--null-pilot-size", type=int, default=200,
                   help="Stein-only: pilot subsample size. Default 200.")
    p.add_argument("--null-stratify",
                   choices=["none", "bin", "dist"], default="dist",
                   help="σ-stratified null baseline. 'dist' (default) "
                        "weights each gap's null by the gap's σ histogram "
                        "across bins — i.e. conditional on the gap's "
                        "actual local density distribution. 'bin' is the "
                        "single-bin reduction (uses the gap's mean σ to "
                        "pick one bin). 'none' is the global Stein null "
                        "(pre-stratification). Empirically dist rescues "
                        "secret-saturator gaps in dense regions; runtime "
                        "overhead is <2s. Only affects phase 1 (k* "
                        "selection); phase 3 stats unchanged.")
    p.add_argument("--null-stratify-bins", type=int, default=10,
                   help="Number of σ quantile bins for --null-stratify={bin,dist}. "
                        "Default 10 (deciles).")
    p.add_argument("--rank-by",
                   choices=["median_logit", "min_logit", "mean_logit", "logit_sum"],
                   default="median_logit",
                   help="Column to rank survivors by. median_logit (default) "
                        "is robust to extreme per-member posteriors. "
                        "min_logit promotes tight clusters most aggressively "
                        "but is brittle to a single weak member. mean_logit "
                        "uses post-clipping to bound ±inf. logit_sum is "
                        "size-extensive (favors big clusters).")
    p.add_argument("--top-curves", type=int, default=20,
                   help="Plot LLR-vs-k for the top-N survivors by --rank-by")
    p.add_argument("--index-type", choices=["flat", "hnsw"], default="hnsw",
                   help="FAISS index type for setup's full_index + "
                        "article_index. 'hnsw' (default) is approximate "
                        "but much faster at subfield/field/domain scale; "
                        "'flat' is exact but O(N²). Empirically HNSW "
                        "M=32/efc=400/efs=max(K_QUERY,2000) gives recall "
                        "≥99% on test topics.")
    p.add_argument("--k-query-stats", default="auto",
                   help="Per-member top-K used by phase-3 stats. Integer "
                        "sets it explicitly; 'auto' (default) runs the "
                        "[calibrate] phase to pick it jointly with "
                        "--member-subsample from a Stein-style pilot.")
    p.add_argument("--member-subsample", default="auto",
                   help="When a gap has more than this many refined members, "
                        "phase-3 stats aggregates over a random subsample of "
                        "this size instead of all members (CSLS background "
                        "is still built from the full member set, so the "
                        "median_logit estimate is subsample-unbiased). "
                        "Integer or 'all' sets it explicitly; 'auto' (default) "
                        "runs the [calibrate] phase. Big saturator-style gaps "
                        "are the cost driver at phase-3 — capping them at "
                        "100-200 buys ~5-10× speedup with negligible "
                        "median_logit error.")
    p.add_argument("--calibrate-tolerance", type=float, default=0.1,
                   help="δ: maximum |median_logit| deviation from the "
                        "K_max/m=all baseline tolerated at the chosen "
                        "(K', m'). Calibration picks the smallest (K', m') "
                        "such that the (1-α)-quantile of |Δ| across the "
                        "pilot is ≤ δ. Default 0.1 logits.")
    p.add_argument("--calibrate-alpha", type=float, default=0.05,
                   help="α: exceedance rate for the pilot tolerance interval "
                        "(see --calibrate-tolerance). Default 0.05.")
    p.add_argument("--calibrate-pilot-size", type=int, default=100,
                   help="Number of gaps in the calibration pilot. Stratified "
                        "by gap size so small/medium/large/saturator gaps "
                        "are all represented. Default 100.")
    p.add_argument("--calibrate-k-grid", default="200,400,800,1500",
                   help="Comma-separated candidate K' values for the "
                        "calibration sweep. Default 200,400,800,1500.")
    p.add_argument("--calibrate-m-grid", default="25,50,100,200,all",
                   help="Comma-separated candidate member-subsample values "
                        "(use 'all' for no subsampling). Default "
                        "25,50,100,200,all.")
    p.add_argument("--article-types", default="article",
                   help="Comma-separated OpenAlex work-types treated as "
                        "ambient/query set (default: 'article'). Pass "
                        "'article,preprint' to include preprints.")
    p.add_argument("--book-types", default="book",
                   help="Comma-separated OpenAlex work-types treated as "
                        "landmark set (default: 'book'). Pass "
                        "'book-chapter' to use only chapters or "
                        "'book,book-chapter' to combine.")
    args = p.parse_args()

    article_types = tuple(t.strip() for t in args.article_types.split(",") if t.strip())
    book_types = tuple(t.strip() for t in args.book_types.split(",") if t.strip())

    print(f"[setup] loading scope {args.scope} "
          f"(index_type={args.index_type})...", flush=True)
    state = setup(
        args.scope, index_type=args.index_type,
        article_types=article_types, book_types=book_types,
    )
    n_total = state["sigma"].shape[0]
    print(f"  N_total={n_total}, "
          f"N_articles={len(state['article_global_idx'])}, "
          f"N_books={len(state['book_global_idx'])}", flush=True)

    print(f"[null] computing corpus-wide null log-mean per k "
          f"(method={args.null_method}, stratify={args.null_stratify})...",
          flush=True)
    null_strat: np.ndarray | None = None
    null_bin_edges: np.ndarray | None = None
    if args.null_stratify in ("bin", "dist"):
        null_strat, null_bin_edges, _strat_info = (
            _compute_corpus_null_stein_stratified(
                state,
                n_bins=args.null_stratify_bins,
                target_rel_se=args.null_target_rel_se,
                m_pilot_per_bin=args.null_pilot_size,
            )
        )
        # Provide a global fallback null in case downstream code expects it
        # (also used when --null-stratify=none on the same run).
        null_log_mean = null_strat.mean(axis=1)
    elif args.null_method == "stein":
        null_log_mean, stein_info = _compute_corpus_null_stein(
            state,
            target_rel_se=args.null_target_rel_se,
            m_pilot=args.null_pilot_size,
        )
    else:
        null_sample = None if args.null_sample_size == 0 else args.null_sample_size
        null_log_mean = _compute_corpus_null(state, sample_size=null_sample)

    print(f"[load] reading K-ball CSV {args.kball_csv}...", flush=True)
    kball_rows = load_kball_csv(args.kball_csv)
    print(f"  {len(kball_rows)} K-ball gaps", flush=True)

    # Build book_index for compute_gap_stats. setup() returns full_index +
    # article_index already; book_index needs construction here.
    from bookgap.index import build_index, FlatL2
    book_emb = np.ascontiguousarray(state["emb"][state["book_global_idx"]])
    book_index = build_index(FlatL2(), book_emb)

    # Phase 1: refine each K-ball gap (LLR-vs-k, optional re-centering).
    # Stats deferred to phase 3 since fuzzy K_G requires joint-normalized
    # memberships across the WHOLE refined gap set (cohesion-design-axes
    # axis H).
    print(f"[refine] running LLR-vs-k refinement on {len(kball_rows)} gaps...",
          flush=True)
    refined: list[dict] = []
    for i, row in enumerate(kball_rows):
        if i % 25 == 0:
            _abort_if_stop_requested("refine", i, len(kball_rows))
        if null_strat is not None and args.null_stratify in ("bin", "dist"):
            gap_null = _null_for_gap(
                row["article_indices"], state["sigma"],
                null_strat, null_bin_edges,
                mode=args.null_stratify,
            )
        else:
            gap_null = null_log_mean
        result = refine_gap_via_llr(
            member_indices=row["article_indices"],
            state=state,
            null_log_mean=gap_null,
            recenter=not args.no_recenter,
        )
        kball_set = set(row["article_indices"])
        refined_set = set(result["refined_member_indices"])
        kball_jaccard = binary_set_jaccard(kball_set, refined_set)
        refined.append({
            "kball_gap_id": row["gap_idx"],
            "kball_size": row["gap_size"],
            "refined_k_star": result["refined_k_star"],
            "refined_n_articles": len(result["refined_member_indices"]),
            "llr_peak": result["llr_peak"],
            "llr_peak_at_k": result["llr_peak_at_k"],
            "llr_curve": result["llr_curve"],
            "kball_size_delta": len(result["refined_member_indices"]) - row["gap_size"],
            "kball_jaccard": kball_jaccard,
            "refined_member_indices": result["refined_member_indices"],
        })
        if (i + 1) % 25 == 0:
            print(f"  refined {i + 1}/{len(kball_rows)}", flush=True)

    # Phase 2: build joint-normalized fuzzy gap memberships across all
    # refined gaps. Each article's per-gap membership sums to 1 across
    # the gaps that claim it. This drives K_G in the per-member posterior.
    print(f"[fuzzy] building joint-normalized fuzzy gap memberships...",
          flush=True)
    M_norm_csc = _build_fuzzy_jointnorm_memberships(refined, state)
    n_total = state["sigma"].shape[0]

    # Resolve k_query_stats and member_subsample. 'auto' → run calibration.
    k_query_resolved: int
    m_resolved: int | None
    needs_calibration = (args.k_query_stats == "auto") or (
        args.member_subsample == "auto"
    )
    if needs_calibration:
        candidates_k = [int(x) for x in args.calibrate_k_grid.split(",")]
        candidates_m_raw = [x.strip() for x in args.calibrate_m_grid.split(",")]
        candidates_m: list[int | None] = [
            None if x.lower() == "all" else int(x) for x in candidates_m_raw
        ]
        k_query_resolved, m_resolved, _sweep = _calibrate_stats_params(
            refined_gaps=refined,
            state=state,
            M_norm_csc=M_norm_csc,
            book_index=book_index,
            candidates_k=candidates_k,
            candidates_m=candidates_m,
            tolerance_delta=args.calibrate_tolerance,
            tolerance_alpha=args.calibrate_alpha,
            pilot_size=args.calibrate_pilot_size,
        )
        # Manual overrides win against calibration if specified.
        if args.k_query_stats != "auto":
            k_query_resolved = int(args.k_query_stats)
        if args.member_subsample != "auto":
            m_resolved = (
                None if args.member_subsample.lower() == "all"
                else int(args.member_subsample)
            )
    else:
        k_query_resolved = int(args.k_query_stats)
        m_resolved = (
            None if args.member_subsample.lower() == "all"
            else int(args.member_subsample)
        )

    # Phase 3: compute per-gap stats with fuzzy K_G + hard CSLS K_B
    # (per cohesion-design-axes axis H: K_B is a size-matched bounding
    # object, NOT a competing-class likelihood, so it stays hard).
    print(f"[stats] computing fuzzy-K_G posteriors for {len(refined)} gaps "
          f"(k_query_stats={k_query_resolved}, "
          f"member_subsample={'all' if m_resolved is None else m_resolved})...",
          flush=True)
    for g_idx, g in enumerate(refined):
        if g_idx % 25 == 0:
            _abort_if_stop_requested("stats", g_idx, len(refined))
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
        if (g_idx + 1) % 50 == 0:
            print(f"  stats {g_idx + 1}/{len(refined)}", flush=True)

    print(f"[dedup] containment ≥ {args.containment_threshold} (prune) "
          f"then Jaccard ≥ {args.jaccard_threshold} (merge); "
          f"rank_metric={args.rank_by}", flush=True)
    # The dedup function needs gap_idx; map kball_gap_id into that key.
    for g in refined:
        g["gap_idx"] = g["kball_gap_id"]
    refined = dedup_via_containment_then_jaccard(
        refined,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
        rank_metric=args.rank_by,
    )
    n_survivors = sum(1 for g in refined if g["survivor"] == 1)
    n_jaccard_merged = sum(1 for g in refined if g.get("dedup_reason") == "jaccard_merged")
    n_containment_pruned = sum(1 for g in refined if g.get("dedup_reason") == "containment_pruned")
    print(f"  {n_survivors} survivors / {len(refined)} gaps "
          f"(jaccard-merged: {n_jaccard_merged}, "
          f"containment-pruned: {n_containment_pruned})", flush=True)

    # Sort: survivors first, then by chosen --rank-by column desc.
    rank_key = args.rank_by
    refined.sort(key=lambda g: (-g["survivor"], -g[rank_key]))

    _write_refined_csv(args.out, args.scope, refined)
    print(f"[write] refined CSV written to {args.out}", flush=True)

    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        _plot_llr_curves(
            refined, args.top_curves, args.cache_dir / "llr_curves.png",
            rank_by=args.rank_by,
        )
        _plot_size_delta_hist(refined, args.cache_dir / "size_delta_hist.png")
        print(f"[figures] written to {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
