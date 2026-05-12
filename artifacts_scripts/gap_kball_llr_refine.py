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
) -> np.ndarray:
    """Local-CSLS-balanced per-member posterior (ported from gap_kball_cohesion_curve).

    For each gap member i, posterior is:
      post_i = K_G(i) / (K_G(i) + K_B(i))
    where K_G is the self-tuning Gaussian kernel mass of i's K-NN that
    are also in G, and K_B is the kernel mass of i's K-NN that are in
    the local-CSLS-balanced background B (one book contributes f(b)
    non-gap articles where f(b) = number of gap members whose 1-NN book
    is b).

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

    Returns posts: (n_members,) array of per-member posteriors in [0, 1].
    """
    from collections import Counter
    members_arr = np.asarray(members, dtype=np.int64)
    k_query = D_NN.shape[1]

    in_G = np.zeros(n_total, dtype=bool)
    in_G[members_arr] = True

    # Count books claimed by gap members
    book_counts: Counter = Counter()
    for b in nearest_book_per_member:
        book_counts[int(b)] += 1

    # Build the background lookup
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
            if not in_G[a_int]:
                in_B[a_int] = True
                taken += 1

    # Per-member kernel-weighted mass on G vs B
    sigma_seed = sigma[members_arr][:, None]
    sigma_n = sigma[I_NN]
    W = np.exp(-D_NN / (sigma_seed * sigma_n))
    self_mask = I_NN == members_arr[:, None]
    G_mask = in_G[I_NN] & ~self_mask
    B_mask = in_B[I_NN]
    K_G = (W * G_mask).sum(axis=1)
    K_B = (W * B_mask).sum(axis=1)
    denom = K_G + K_B + 1e-30
    return K_G / denom


def compute_gap_stats(
    refined_member_indices: list[int],
    state: dict,
    book_global_idx: np.ndarray,
    book_index,
    article_index,
    article_global_idx: np.ndarray,
    k_query: int = K_QUERY,
    k_per_book: int = K_QUERY,
) -> dict:
    """Compute logit_sum, mean_logit, effective_size for a refined gap.

    Returns dict with keys: logit_sum, mean_logit, effective_size, min_post,
    mean_post.
    """
    emb = state["emb"]
    sigma = state["sigma"]
    full_index = state["full_index"]

    members = np.asarray(refined_member_indices, dtype=np.int64)
    if len(members) < 2:
        return {
            "logit_sum": 0.0, "mean_logit": 0.0,
            "effective_size": 0.0,
            "min_post": 0.0, "mean_post": 0.0,
        }

    member_emb = np.ascontiguousarray(emb[members])
    D_NN, I_NN_full = full_index.search(member_emb, k=k_query + 1)
    D_NN = D_NN[:, 1:]
    I_NN_full = I_NN_full[:, 1:]

    # Find 1-NN book per member
    _, I_book = book_index.search(member_emb, k=1)
    nearest_book = book_global_idx[I_book[:, 0]]

    # Build book -> nearby-articles lookup for unique books only
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

    n_total = sigma.shape[0]
    posts = compute_local_csls_posterior(
        members=list(map(int, members)),
        D_NN=D_NN.astype(np.float32),
        I_NN=I_NN_full.astype(np.int64),
        sigma=sigma,
        nearest_book_per_member=nearest_book.astype(np.int64),
        book_to_arts=book_to_arts,
        n_total=n_total,
    )

    post_clip = np.clip(posts, 1e-3, 1 - 1e-3)
    logits = np.log(post_clip / (1 - post_clip))
    return {
        "logit_sum": float(logits.sum()),
        "mean_logit": float(logits.mean()),
        "effective_size": float(posts.sum()),
        "min_post": float(posts.min()),
        "mean_post": float(posts.mean()),
    }


def dedup_via_binary_jaccard(
    gaps: list[dict],
    threshold: float = 0.5,
) -> list[dict]:
    """Pairwise binary set Jaccard dedup over refined gaps.

    gaps: list of dicts, each with keys gap_idx, refined_member_indices,
      logit_sum (used as tiebreak).
    threshold: J >= threshold ⇒ merge.

    Annotates each input gap with two new keys:
      survivor: 1 if this gap is the survivor of its component, 0 otherwise.
      merged_into_gap_id: gap_idx of the survivor it merged into (-1 if survivor).

    Returns the annotated list (same order as input).
    """
    n = len(gaps)
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

    sets = [set(g["refined_member_indices"]) for g in gaps]
    for i in range(n):
        for j in range(i + 1, n):
            if binary_set_jaccard(sets[i], sets[j]) >= threshold:
                union(i, j)

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    survivor_of_comp: dict[int, int] = {}
    for root, comp in components.items():
        best = max(comp, key=lambda i: gaps[i]["logit_sum"])
        survivor_of_comp[root] = best

    annotated: list[dict] = []
    for idx, g in enumerate(gaps):
        root = find(idx)
        survivor_idx = survivor_of_comp[root]
        g_out = dict(g)
        if idx == survivor_idx:
            g_out["survivor"] = 1
            g_out["merged_into_gap_id"] = -1
        else:
            g_out["survivor"] = 0
            g_out["merged_into_gap_id"] = gaps[survivor_idx]["gap_idx"]
        annotated.append(g_out)
    return annotated


def main() -> None:
    raise NotImplementedError("CLI driver wired in Task 3")


if __name__ == "__main__":
    main()
