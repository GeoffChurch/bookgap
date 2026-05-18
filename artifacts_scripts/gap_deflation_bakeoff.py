"""Gap-deflation bake-off: swappable deflation strategies on the
byte-stable K-ball harness. See
docs/superpowers/specs/2026-05-17-gap-deflation-bakeoff-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bookgap.index import kballs_with
from artifacts_scripts.kball_gap_probe import _build_topic_index


@dataclass(frozen=True)
class Candidate:
    idx: int                 # article-local index of the ball's seed
    members: np.ndarray      # article-local indices in the k-ball (int64)
    centroid: np.ndarray     # mean embedding of members (float32, (d,))


def build_kball_candidates(
    article_emb: np.ndarray,
    book_emb: np.ndarray,
    radius_fraction: float = 0.75,
    k: int = 1,
) -> list[Candidate]:
    """Strategy-independent K-ball candidate pool.

    Mirrors the candidate construction inside
    kball_gap_probe._greedy_kball_gaps_inline (same indices, same
    kballs_with call) but stops before the greedy selection so any
    deflation strategy can consume the raw pool. members are
    article_emb-local indices.
    """
    book_index = _build_topic_index(book_emb)
    article_index = _build_topic_index(article_emb)
    coverage: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(article_emb)

    def _kernel(qi: int, dists: np.ndarray, ids: np.ndarray) -> None:
        coverage[qi] = (dists.copy(), ids.copy())

    kballs_with(
        ambient=article_index, landmark=book_index, query=article_emb,
        radius_fraction=radius_fraction, k=k, kernel_fn=_kernel,
        show_progress=False,
    )
    cands: list[Candidate] = []
    for i, cov in enumerate(coverage):
        if cov is None:
            members = np.array([], dtype=np.int64)
        else:
            members = cov[1].astype(np.int64)
        if members.size:
            # np.mean upcasts float32→float64; cast back to match article_emb dtype
            centroid = article_emb[members].mean(axis=0).astype(np.float32)
        else:
            centroid = article_emb[i].astype(np.float32)
        cands.append(Candidate(idx=i, members=members, centroid=centroid))
    return cands


# ---------------------------------------------------------------------------
# Deflation strategies
# ---------------------------------------------------------------------------

def _greedy_hard(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Greedy-prune deflation — byte-anchored to _greedy_kball_gaps_inline.

    The oracle iterates `for gap_idx in range(num_gaps)` and emits
    `(gap_idx, gap_size, best_ids.copy())` where `gap_idx` is the loop
    counter (0, 1, 2, …), NOT the argmax index.  That is matched here.

    Covered is maintained as a list of mutable id arrays, one per candidate,
    mirroring the oracle's list of (dists, ids) pairs — but since greedy
    ranking uses only `len(ids)`, the dists are irrelevant and we drop them.
    """
    # Mutable copy of each candidate's member set; ordering matches cands list
    covered = [c.members.copy() for c in cands]
    gaps: list[tuple[int, int, np.ndarray]] = []
    for gap_idx in range(num_gaps):
        if not any(len(ids) > 0 for ids in covered):
            break
        best = int(np.argmax([len(ids) for ids in covered]))
        best_ids = covered[best]
        if len(best_ids) == 0:
            break
        gap_size = int(len(best_ids))
        if min_gap_size > 0 and gap_size < min_gap_size:
            break
        gaps.append((gap_idx, gap_size, best_ids.copy()))
        pruned_set = set(best_ids.tolist())
        for i, ids in enumerate(covered):
            if ids.size:  # micro-opt vs oracle (omits this); result identical on empty arrays
                m = np.array([x not in pruned_set for x in ids.tolist()], dtype=bool)
                covered[i] = ids[m]
    return gaps


def _greedy_geom(cands, article_emb, *, num_gaps, min_gap_size,
                 sigma=None, **params):
    """Greedy, but every pick is taken on the RESIDUAL (unclaimed)
    article set: after each emit, drop all globally-claimed ids from
    every surviving candidate before the next argmax. v1 keeps the
    kball radius fixed (radius-vs-nearest-unclaimed-book recompute is
    intentionally deferred — residual-restricted pick is the behavior
    this arm isolates)."""
    alive = [c.members.copy() for c in cands]
    claimed: set[int] = set()
    gaps = []
    emitted = 0
    for _ in range(num_gaps):
        sizes = [len(ids) for ids in alive]
        if not any(sizes):
            break
        best = int(np.argmax(sizes))
        best_ids = alive[best]
        if len(best_ids) == 0 or (min_gap_size > 0 and len(best_ids) < min_gap_size):
            break
        # element-0 = sequential emission rank (uniform across arms; matches greedy_hard/oracle convention)
        gaps.append((emitted, int(len(best_ids)), best_ids.copy()))
        emitted += 1
        claimed |= set(best_ids.tolist())
        for i in range(len(alive)):
            if alive[i].size:
                m = np.array([x not in claimed for x in alive[i].tolist()],
                             dtype=bool)
                alive[i] = alive[i][m]
    return gaps


_FL_GAINS: list[float] = []


def _fl_marginal_trace() -> list[float]:
    return list(_FL_GAINS)


def _submodular_fl(cands, article_emb, *, num_gaps, min_gap_size,
                   sigma=None, **params):
    """Facility-location deflation — sparse, k-ball-restricted.

    sim(a,g) is defined ONLY for articles a in cands[g].members (zero outside
    the ball), so the working set is Σ|members_g| — never the full A×K matrix.

    sigma_g = median(sigma) is a fixed data-derived scalar (no tunable
    hyperparameter). sigma_a = sigma[a] per-article.

    Greedy rounds pick the candidate with the largest marginal coverage gain
    over the current best_cov vector. Per-candidate sim vectors are precomputed
    once before the loop (total memory ∝ Σ|members_g|, bounded by ball mass).
    """
    assert sigma is not None, "submodular_fl needs state['sigma']"
    sigma = np.asarray(sigma, dtype=np.float64)
    sig_g = float(np.median(sigma))
    A = len(article_emb)
    emb = article_emb.astype(np.float32)

    # Precompute per-candidate sim vectors — each has length |members_g|.
    # Total allocation: Σ|members_g| floats (sparse, bounded by ball mass).
    sim_vecs: list[np.ndarray] = []   # sim_vecs[g] shape (|members_g|,) float64
    for c in cands:
        if c.members.size == 0:
            sim_vecs.append(np.empty(0, dtype=np.float64))
            continue
        diff = emb[c.members].astype(np.float64) - c.centroid.astype(np.float64)
        d2 = np.einsum("ij,ij->i", diff, diff)
        denom = np.maximum(sigma[c.members] * sig_g, 1e-8)
        sim_vecs.append(np.exp(-d2 / denom))

    best_cov = np.zeros(A, dtype=np.float64)
    picked: set[int] = set()
    _FL_GAINS.clear()
    gaps = []
    emitted = 0

    for _ in range(num_gaps):
        # For each not-yet-picked candidate, gain = Σ max(0, sim(a,g) - best_cov[a])
        # over its ball members only.  Operate on small per-candidate vectors.
        best_gain = -np.inf
        best_j = -1
        for g, sv in enumerate(sim_vecs):
            if g in picked or sv.size == 0:
                continue
            gain_g = float(np.maximum(sv - best_cov[cands[g].members], 0.0).sum())
            if gain_g > best_gain:
                best_gain = gain_g
                best_j = g
        if best_j < 0 or not np.isfinite(best_gain) or best_gain <= 0:
            break
        # Emit: members where sim(a,g) > best_cov[a] (strict > per spec)
        sv = sim_vecs[best_j]
        mem = cands[best_j].members
        keep = sv > best_cov[mem]
        ids = mem[keep].astype(np.int64)
        if min_gap_size > 0 and ids.size < min_gap_size:
            break
        _FL_GAINS.append(best_gain)
        gaps.append((emitted, int(ids.size), ids))   # element-0 = emission rank
        emitted += 1
        # Update coverage for all members of the picked ball
        best_cov[mem] = np.maximum(best_cov[mem], sv)
        picked.add(best_j)
    return gaps


def _jaccard_sim_capped(member_sets):
    n = len(member_sets)
    S = np.eye(n, dtype=np.float64)
    for i in range(n):
        si = member_sets[i]
        for j in range(i + 1, n):
            sj = member_sets[j]
            u = len(si | sj)
            v = (len(si & sj) / u) if u else 0.0
            S[i, j] = S[j, i] = v
    return S


def _dpp(cands, article_emb, *, num_gaps, min_gap_size, sigma=None,
         dpp_pool_cap=4000, **params):
    # 1-2: cap the candidate pool (mandatory; dense K×K is O(corpus²))
    idxs = [i for i, c in enumerate(cands) if len(c.members) >= max(min_gap_size, 1)]
    if not idxs:
        return []
    idxs.sort(key=lambda i: (-len(cands[i].members), i))
    idxs = idxs[:dpp_pool_cap]
    msets = [set(cands[i].members.tolist()) for i in idxs]
    q = np.array([np.log1p(len(cands[i].members)) for i in idxs], dtype=np.float64)
    S = _jaccard_sim_capped(msets)
    L = (q[:, None] * q[None, :]) * S            # Diag(q) S Diag(q)
    n = L.shape[0]
    # Greedy MAP-DPP (Chen et al. 2018 fast Cholesky)
    cis = np.zeros((min(num_gaps, n), n))
    di2 = np.diag(L).astype(np.float64).copy()
    gaps = []
    emitted = 0
    for it in range(min(num_gaps, n)):
        j = int(np.argmax(np.where(di2 > 1e-12, di2, -np.inf)))
        if not np.isfinite(di2[j]) or di2[j] <= 1e-12:
            break
        orig = idxs[j]
        ids = cands[orig].members.astype(np.int64)
        if min_gap_size > 0 and ids.size < min_gap_size:
            break
        gaps.append((emitted, int(ids.size), ids))   # element-0 = emission rank
        emitted += 1
        eis = (L[j, :] - cis[:it].T @ cis[:it, j]) / np.sqrt(di2[j])
        cis[it, :] = eis
        di2 = di2 - eis ** 2
        di2[j] = -np.inf
    return gaps


def _softmax_mc(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,
    book_emb=None,
    k_basins: int = 8,
    target_perplexity: float = 1.75,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Multicluster-softmax deflation — memory-bounded (NO dense A×K array).

    Design:
    - Basins = candidate centroids ∪ book-anchored centroids.
      First K rows are candidate basins (emittable); remaining rows are
      book competitors (never emitted, but steal articles from gaps).
    - Per-article kNN-basin query via FAISS (k_basins nearest basins).
    - log-kernel: logk = -Dk / (sigma_a * sig_basin); sig_basin = median(sigma).
    - Temperature τ calibrated by target perplexity P via 1-D bisection (~40 iters).
    - Hard assignment: article → argmax basin over its k_basins retrieved.
      If argmax is a book basin (index ≥ K) → unassigned (book competitor wins).
    - Gaps = candidate basins with ≥ max(min_gap_size,1) assigned articles.
      Sorted by size desc; element-0 = sequential emission rank.
    """
    if not cands:
        return []

    assert sigma is not None, "_softmax_mc requires sigma"
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    sig_basin = float(np.median(sigma_arr))

    K = len(cands)
    cand_centroids = np.vstack([c.centroid for c in cands]).astype(np.float32)

    # Stack basins: [candidate centroids | book centroids]
    if book_emb is not None and len(book_emb) > 0:
        book_arr = np.asarray(book_emb, dtype=np.float32)
        basinC = np.vstack([cand_centroids, book_arr]).astype(np.float32)
    else:
        basinC = cand_centroids

    n_basins = len(basinC)
    k_q = min(k_basins, n_basins)

    # Build FAISS index over basin centroids
    bidx = _build_topic_index(basinC)

    # Query every article for its k_q nearest basins
    # article_emb: (A, d), bidx returns squared L2 distances
    A = len(article_emb)
    art_f32 = np.ascontiguousarray(article_emb.astype(np.float32))
    Dk, Ik = bidx.search(art_f32, k_q)   # (A, k_q) each

    # Compute log-kernel per (article, retrieved basin): logk[a, j] = -Dk[a,j] / (sigma_a * sig_basin)
    # Dk shape: (A, k_q) — no dense A×K allocation
    denom = np.maximum(sigma_arr[:, None] * sig_basin, 1e-8)   # (A, 1) broadcast → (A, k_q)
    logk = -Dk / denom   # (A, k_q) — only k_q basins per article, NOT K basins

    # Perplexity bisection to find temperature τ
    # perplexity(w) = exp(entropy(w)); target P = target_perplexity
    # Monotone in τ: higher τ → more uniform → higher perplexity.
    def _mean_perplexity(tau: float) -> float:
        w = logk / tau   # (A, k_q)
        # numerically stable softmax along axis=1
        w = w - w.max(axis=1, keepdims=True)
        e = np.exp(w)
        p = e / e.sum(axis=1, keepdims=True)
        # Shannon entropy per article (nats), then perplexity
        # clip to avoid log(0)
        ent = -(p * np.log(np.maximum(p, 1e-30))).sum(axis=1)   # (A,)
        return float(np.mean(np.exp(ent)))

    tau_lo, tau_hi = 1e-3, 1e3
    for _ in range(40):
        tau_mid = (tau_lo + tau_hi) * 0.5
        if _mean_perplexity(tau_mid) > target_perplexity:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid
    tau = (tau_lo + tau_hi) * 0.5

    # Hard assignment: argmax of logk/τ over retrieved basins
    scaled = logk / tau   # (A, k_q)
    best_local = np.argmax(scaled, axis=1)   # (A,) — index into columns of Ik
    best_basin = Ik[np.arange(A), best_local]   # (A,) — global basin index

    # Assign articles to candidate basins; book basins (index ≥ K) → unassigned
    assignments: list[list[int]] = [[] for _ in range(K)]
    for a in range(A):
        b = int(best_basin[a])
        if b < K:
            assignments[b].append(a)

    # Emit candidate basins with enough articles, sorted by size desc
    eff_min = max(min_gap_size, 1)
    eligible = [(len(assignments[g]), g) for g in range(K)
                if len(assignments[g]) >= eff_min]
    eligible.sort(key=lambda x: -x[0])

    gaps: list[tuple[int, int, np.ndarray]] = []
    for rank, (sz, g) in enumerate(eligible[:num_gaps]):
        ids = np.array(assignments[g], dtype=np.int64)
        gaps.append((rank, sz, ids))

    return gaps


_FL_GAINS_BD: list[float] = []


def _fl_bookdemand_trace() -> list[float]:
    return list(_FL_GAINS_BD)


def _fl_bookdemand(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,          # deliberately unused — pure set-overlap, no kernel/sigma
    book_emb=None,
    k_book: int = 10,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Facility-location deflation over BOOK-anchored demand.

    Coverage sim(g,b) = |members_g ∩ Nb[b]| / k_book — pure set overlap.
    No sigma/kernel used, which sidesteps the global-vs-article-local sigma
    index-space bug class entirely (sigma is accepted but ignored).

    Memory-bounded: inverted index maps article-local id → candidate list;
    per-candidate coverage stored as sparse dicts {book_idx: count/k_book}.
    No dense (B,K) or (A,K) matrix is ever built.
    """
    # Edge cases
    if book_emb is None:
        return []
    book_arr = np.asarray(book_emb, dtype=np.float32)
    if book_arr.size == 0 or not cands:
        return []

    from bookgap.index import build_index, FlatL2

    B = len(book_arr)
    A = len(article_emb)
    k_b = min(k_book, A)

    # Build exact article index and query books for their k_b nearest articles
    art_idx = build_index(FlatL2(), np.ascontiguousarray(article_emb.astype(np.float32)))
    _D, Nb = art_idx.search(np.ascontiguousarray(book_arr.astype(np.float32)), k=k_b)
    # Nb: shape (B, k_b) — article-local positional ids (consistent with Candidate.members)

    # Build inverted index: article-local id → list of candidate indices
    # O(Σ|members_g|) — bounded by total ball mass, no dense allocation
    inv: dict[int, list[int]] = {}
    for g, c in enumerate(cands):
        for a in c.members.tolist():
            if a not in inv:
                inv[a] = []
            inv[a].append(g)

    # Accumulate coverage counts: cand_cov[g] = {book_b: overlap_count}
    # Process each book's neighborhood; accumulate into sparse per-candidate dicts
    cand_cov: list[dict[int, float]] = [{} for _ in range(len(cands))]
    for b in range(B):
        for a in Nb[b].tolist():
            for g in inv.get(int(a), []):
                cand_cov[g][b] = cand_cov[g].get(b, 0) + 1
    # Normalise counts by k_b to get fractional coverage ∈ [0, 1]
    for g in range(len(cands)):
        for b in cand_cov[g]:
            cand_cov[g][b] /= k_b

    # Greedy facility-location: pick candidates to maximise Σ_b max(best_cov[b], cov(g,b))
    best_cov = np.zeros(B, dtype=np.float64)
    picked: set[int] = set()
    _FL_GAINS_BD.clear()
    gaps: list[tuple[int, int, np.ndarray]] = []
    emitted = 0

    for _ in range(num_gaps):
        best_gain = -np.inf
        best_j = -1
        # Iterate in candidate-index order; strict > for ties → lowest index wins
        for g in range(len(cands)):
            if g in picked or not cand_cov[g]:
                continue
            gain_g = sum(
                max(0.0, v - best_cov[b])
                for b, v in cand_cov[g].items()
            )
            if gain_g > best_gain:
                best_gain = gain_g
                best_j = g

        if best_j < 0 or not np.isfinite(best_gain) or best_gain <= 0:
            break

        ids = cands[best_j].members.astype(np.int64)
        if min_gap_size > 0 and ids.size < min_gap_size:
            break

        _FL_GAINS_BD.append(best_gain)
        gaps.append((emitted, int(ids.size), ids))
        emitted += 1

        # Update best_cov for books this candidate touches
        for b, v in cand_cov[best_j].items():
            if v > best_cov[b]:
                best_cov[b] = v
        picked.add(best_j)

    return gaps


def heldout_book_split(n_books: int, frac: float = 0.2,
                       seed: int = 42) -> np.ndarray:
    """Deterministic held-out book indices (sorted, no replacement)."""
    rng = np.random.RandomState(seed)
    n_hold = int(round(n_books * frac))
    return np.sort(rng.choice(n_books, size=n_hold, replace=False))


def heldout_recall_spread(heldout_book_emb, article_emb, gaps_members,
                          k_near: int = 10, cover_frac: float = 0.5):
    """recall = fraction of held-out books whose `k_near` nearest
    articles are >= `cover_frac` inside SOME gap's member set.
    spread = Shannon entropy of which gap claims each recalled book
    (0 when all recalled books map to one gap)."""
    from bookgap.index import build_index, FlatL2
    if len(heldout_book_emb) == 0 or len(article_emb) == 0 or not gaps_members:
        return 0.0, 0.0
    aidx = build_index(FlatL2(), np.ascontiguousarray(
        article_emb.astype(np.float32)))
    kk = min(k_near, len(article_emb))
    _D, I = aidx.search(np.ascontiguousarray(
        heldout_book_emb.astype(np.float32)), k=kk)
    member_sets = [set(int(x) for x in m) for m in gaps_members]
    claimed_by = []
    for nn in I:
        nnset = set(int(x) for x in nn)
        best_g, best_cov = -1, 0.0
        for gi, ms in enumerate(member_sets):
            cov = len(nnset & ms) / max(len(nnset), 1)
            if cov > best_cov:
                best_cov, best_g = cov, gi
        claimed_by.append(best_g if best_cov >= cover_frac else -1)
    recalled = [c for c in claimed_by if c >= 0]
    recall = len(recalled) / max(len(claimed_by), 1)
    if recalled:
        _, cnt = np.unique(recalled, return_counts=True)
        p = cnt / cnt.sum()
        spread = float(-(p * np.log(p)).sum())
    else:
        spread = 0.0
    return float(recall), spread


def _global_to_article_local(member_lists, article_global_idx):
    """Map GLOBAL work indices -> ARTICLE-LOCAL positions (0..A-1).
    Drops any id not in article_global_idx (non-article works)."""
    g2l = {int(g): i for i, g in enumerate(np.asarray(article_global_idx))}
    out = []
    for ml in member_lists:
        out.append([g2l[int(x)] for x in ml if int(x) in g2l])
    return out


def inter_gap_metrics(survivors: list[dict], top_k: int = 20) -> dict:
    sets = [set(g.get("refined_member_indices", [])) for g in survivors[:top_k]]
    js = [len(a & b)/len(a | b) for i,a in enumerate(sets) for b in sets[i+1:] if (a|b)]
    union = set().union(*sets) if sets else set()
    return {"mean_pairwise_jaccard_top20": float(np.mean(js)) if js else 0.0,
            "unique_coverage_at_20": len(union)}


# Every strategy: (cands, article_emb, *, num_gaps, min_gap_size, sigma, **params) -> list[(seed_idx, size, member_ids)]
DeflateFn = Callable[..., list]
_STRATEGIES: dict[str, DeflateFn] = {
    "greedy_hard": _greedy_hard,
    "greedy_geom": _greedy_geom,
    "submodular_fl": _submodular_fl,
}
_STRATEGIES["dpp"] = _dpp
_STRATEGIES["softmax_mc"] = _softmax_mc
_STRATEGIES["fl_bookdemand"] = _fl_bookdemand


def deflate(
    cands: list[Candidate],
    strategy: str,
    article_emb: np.ndarray,
    *,
    num_gaps: int = 1000,
    min_gap_size: int = 0,
    sigma: float | None = None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Dispatch to a named deflation strategy.

    Args:
        cands:        Candidate pool from build_kball_candidates.
        strategy:     One of the registered strategy names (e.g. "greedy_hard").
        article_emb:  Article embeddings array (passed to soft strategies).
        num_gaps:     Maximum gaps to emit.
        min_gap_size: Stop when the largest remaining ball < this (0 = off).
        sigma:        Smoothing bandwidth (soft strategies only).
        **params:     Extra keyword arguments forwarded to the strategy fn.

    Returns:
        List of (gap_idx, gap_size, member_ids) tuples.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown deflation strategy {strategy!r}")
    return _STRATEGIES[strategy](
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, **params)


# ---------------------------------------------------------------------------
# Bake-off driver
# ---------------------------------------------------------------------------

import argparse
import csv
import time
import traceback
from pathlib import Path


DEFLATION_SCORECARD_FIELDS = [
    "deflation", "scope",
    "mean_pairwise_jaccard_top20", "unique_coverage_at_20",
    "heldout_book_recall_at_20", "heldout_spread", "runtime_s",
]


def run_arm(
    scope: str,
    deflation: str,
    out_dir: Path,
    *,
    index_type: str = "hnsw",
    n_keep: int = 370,
) -> dict:
    """Run one deflation arm end-to-end; return a metrics row dict.

    Steps:
    1. Load raw corpus for ``scope`` (emb, ids, types, titles, refs).
    2. Build the FROZEN LIP/binary config (self_loops=True, row_normalize=False).
    3. Held-out books: split 20 % off *before* candidate generation so the
       gap finder must rediscover the held-out regions.
    4. Regenerate K-ball seeds on the masked book set.
    5. Run the full refine pipeline (LLR → fuzzy → calibrate → stats → dedup).
    6. Rank survivors by effective_size, write per-arm leaderboard CSV.
    7. Return inter-gap metrics + held-out recall/spread row dict.
    """
    from artifacts_scripts.hdbscan_gap_probe import load_scope
    from artifacts_scripts.gap_denoiser_bakeoff import (
        build_scatter_context,
        solve_arm,
        kball_seeds_from_state,
        run_refine_pipeline,
        _write_leaderboard,
    )
    from artifacts_scripts.gap_static_seed_diagnostic import setup

    t0 = time.time()

    # Step 1: load scope
    emb, ids, types, titles, topic_ids, refs = load_scope(scope, with_refs=True)
    labels = list(types)
    titles_map = {i: titles[i] for i in range(len(titles))}

    # Step 2: frozen LIP + binary graph config
    ctx = build_scatter_context(
        emb, ids, refs, labels,
        self_loops=True,
        row_normalize=False,
    )
    Q, _alpha = solve_arm("lip", ctx, n_keep)
    state = setup(
        scope,
        index_type=index_type,
        article_types=("article",),
        book_types=("book",),
        projection=Q,
    )
    state["topic_id"] = scope

    # Step 3: held-out books — split BEFORE candidate generation
    bgi = np.asarray(state["book_global_idx"])
    hidx_local = heldout_book_split(len(bgi), frac=0.2, seed=42)
    heldout_global = bgi[hidx_local]
    heldout_book_emb = state["emb"][heldout_global]

    heldout_set = set(heldout_global.tolist())
    kept_books = np.array([b for b in bgi if b not in heldout_set])
    state2 = dict(state)
    state2["book_global_idx"] = kept_books

    # Step 4: K-ball seeds using the masked book set
    seeds = kball_seeds_from_state(state2, deflation=deflation)

    # Step 5: full refine pipeline
    survivors = run_refine_pipeline(state2, seeds)

    # Step 6: rank and write leaderboard
    ranked = sorted(survivors,
                    key=lambda g: g.get("effective_size", 0.0),
                    reverse=True)
    leaderboard_dir = Path(out_dir) / "leaderboards"
    _write_leaderboard(leaderboard_dir, scope, deflation, ranked, titles_map)

    # Step 7: compute metrics row
    m = inter_gap_metrics(ranked)
    gm = [list(g.get("refined_member_indices", [])) for g in ranked]
    art_gidx = np.asarray(state["article_global_idx"])
    gm_local = _global_to_article_local(gm, art_gidx)
    rec, spread = heldout_recall_spread(
        heldout_book_emb,
        state["emb"][art_gidx],
        gm_local,
        k_near=10,
        cover_frac=0.5,
    )
    return {
        "deflation": deflation,
        "scope": scope,
        **m,
        "heldout_book_recall_at_20": rec,
        "heldout_spread": spread,
        "runtime_s": round(time.time() - t0, 1),
    }


def _write_deflation_scorecard(rows: list[dict], out_path: Path) -> None:
    """Write scorecard CSV at ``out_path``; creates parent dirs; always writes header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEFLATION_SCORECARD_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    """Gap-deflation bake-off driver.

    Loops over deflation arms for a single scope (or multiple scopes when
    called with repeated --scope flags in future extensions). Per-arm
    try/except ensures one bad arm does not abort the run; the scorecard CSV
    is checkpointed after every arm.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", default="17/1702/T10181",
                   help="F/SF/T scope key, e.g. 17/1702/T10181")
    p.add_argument("--arms",
                   default="greedy_hard,greedy_geom,submodular_fl,dpp,softmax_mc",
                   help="comma-separated deflation strategy names")
    p.add_argument("--index-type", choices=["flat", "hnsw"], default="hnsw")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory; scorecard written to <out>/deflation_scorecard.csv")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = out_dir / "deflation_scorecard.csv"

    all_rows: list[dict] = []
    for arm in arms:
        try:
            row = run_arm(
                args.scope, arm, out_dir,
                index_type=args.index_type,
            )
            all_rows.append(row)
        except Exception as exc:
            print(f"[ERROR] arm={arm}: {exc}", flush=True)
            traceback.print_exc()
            continue
        _write_deflation_scorecard(all_rows, scorecard_path)
        print(f"[checkpoint] arm={arm} done; wrote {len(all_rows)} rows to {scorecard_path}",
              flush=True)

    _write_deflation_scorecard(all_rows, scorecard_path)
    print(f"[done] wrote {scorecard_path} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
