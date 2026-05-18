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
