"""Production port of the validated gap-deflation strategies.

This module is a VERBATIM port of the relevant pieces of
``artifacts_scripts/gap_deflation_bakeoff.py`` (the byte-stable
investigation harness) into the production ``bookgap/`` package, so
production code never imports ``artifacts_scripts``.  The logic below
(``Candidate``, ``build_kball_candidates``, ``_resid_diag_mode``,
``_fl_bookdemand_resid``, the resid module-level diag globals, and the
``deflate`` dispatcher) is copied byte-for-byte from the bake-off so
the production resid path is byte-equivalent to the validated logic.
Do not paraphrase or "clean up" the ported bodies.

Imports are restricted to numpy, sklearn KMeans, and
``bookgap.index`` primitives — no ``artifacts_scripts`` import.
``bookgap.index`` only imports ``bookgap.util`` so there is no
circular import (``bookgap.index`` must never import this module).

The ``_build_topic_index`` helper is ported from
``artifacts_scripts/kball_gap_probe._build_topic_index`` (the bake-off's
``build_kball_candidates`` calls it) so candidate construction is
byte-equivalent to the validated harness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bookgap.index import build_index, kballs_with, HNSW, FlatL2


# --- Ported VERBATIM from artifacts_scripts/kball_gap_probe._build_topic_index
def _build_topic_index(topic_emb: np.ndarray):
    """Build a FAISS index for topic_emb.  HNSW(efc=512, efs=2000) for n >= 100, FlatL2 otherwise.

    If env BOOKGAP_EXACT_INDEX is set to "1", always use exact FlatL2
    regardless of n — used by the denoiser bake-off for fully
    deterministic, approximation-free K-ball seeds. Default behavior
    (env unset or any value other than "1") is unchanged for all other callers.

    efs sized to exceed typical K-ball gap neighborhoods (was 128, far below
    gap sizes → recall collapse).  Matches setup()'s proven recall floor
    (efc=512, efs=2000).
    """
    n = len(topic_emb)
    if os.environ.get("BOOKGAP_EXACT_INDEX", "0") == "1":
        return build_index(FlatL2(), topic_emb)
    if n >= 100:
        return build_index(HNSW(hnsw_m=32, hnsw_efc=512, hnsw_efs=2000), topic_emb)
    else:
        return build_index(FlatL2(), topic_emb)


# --- Ported VERBATIM from artifacts_scripts/gap_deflation_bakeoff.py ---------
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
# fl_bookdemand_resid (ported VERBATIM, incl. module-level diag globals)
# ---------------------------------------------------------------------------

_FL_GAINS_RESID: list[float] = []
_FL_DIAG_RESID: dict = {"n_books": 0, "n_orphans": 0, "orphan_frac": 0.0,
                        "M_bg": 0, "mode": "book-only", "ngaps": 0}


def _resid_diag_mode(n_books: int, m_bg: int) -> str:
    """3-way diag `mode` label (single source of truth; can't drift).

    - "background-only" : n_books == 0 (no book signal — O = all articles).
    - "book-only"       : M_bg == 0 with n_books > 0. Reached only when
                          orphans are fewer than the collective book
                          budget (|O| < k_book·n_books), i.e. books spend
                          their k_book budget on distinct manifold
                          regions. With clustered books this requires
                          books to be numerous *relative to* orphan mass,
                          not merely book-rich → pure book-anchored, the
                          fl_bookdemand limit.
    - "residual"        : both n_books > 0 and M_bg > 0 (mixed customers).
    """
    if n_books == 0:
        return "background-only"
    if m_bg == 0:
        return "book-only"
    return "residual"


def _fl_bookdemand_resid_trace() -> list[float]:
    return list(_FL_GAINS_RESID)


def _fl_bookdemand_resid_diag() -> dict:
    return dict(_FL_DIAG_RESID)


def _fl_bookdemand_resid(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,          # deliberately unused — see _fl_bookdemand: sidesteps
                          # the global-vs-article-local index-space bug class
    book_emb=None,
    k_book: int = 10,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Facility-location deflation over BOOK ∪ ORPHAN-BACKGROUND demand.

    Identical greedy-FL machinery to _fl_bookdemand_eb, but the background
    coreset is built ONLY over book-ORPHAN articles, so the operating
    point is intrinsic to orphan mass — there is NO s/M-vs-n_books knob
    and nothing fitted.

    - Book customers (one per training book b): demand Nb[b] = k_b nearest
      article-local articles to book_emb[b] — identical to
      _fl_bookdemand_eb. cov = Jaccard(members_g, Nb[b]).
    - Orphan set O = {articles} \\ ⋃_b Nb[b] (article-local). Small when
      books densely cover the manifold, large when sparse.
    - Background customers: KMeans(n_clusters=M_bg, random_state=0,
      n_init=1) over article_emb[O] ONLY.
      M_bg = clip(|O| // max(k_book·n_books, k_book), 0, 4000),
      clamped ≤ |O| — orphan density PER BOOK, so bg:book ≈
      |O|/(k_book·n_books²). Each centroid's demand Nc = k_b nearest
      article-local articles over the SAME single article index (built
      once, reused for book + orphan-centroid queries; neighborhoods may
      include non-orphans — fine). Weight 1 (no per-customer density
      weight: M_bg already encodes orphan mass; adding it would
      double-count and re-introduce a knob).

    The intrinsic regulator is orphan density PER BOOK
    (M_bg = |O| // max(k_book·n_books, k_book)): shrinkage DECREASES as
    books grow. M_bg→0 (pure book-anchored = fl_bookdemand limit)
    requires |O| < k_book·n_books — books collectively spending their
    k_book budget on distinct manifold regions; with clustered books
    that is reached only when books are numerous *relative to* orphan
    mass, not merely book-rich. Book-sparse ⇒ most articles orphaned and
    the per-book budget is tiny ⇒ background dominant (no collapse).

    cov(g,c) = Jaccard(members_g, demand(c)) — REUSED verbatim from
    _fl_bookdemand: isect / (|members_g| + k_b - isect). Greedy maximises
    Σ_c max(0, cov(g,c) - best_cov[c]); stop when best gain <= 0 or
    num_gaps; min_gap_size -> continue (peer-consistent, NOT break).
    `sigma` accepted but unused.

    Memory-bounded (mirrors _fl_bookdemand_eb exactly): one exact FlatL2
    over article_emb (built once, reused for book & orphan-centroid
    queries); inverted index article→candidate; per-candidate sparse
    {customer_id: cov} dict; best_cov is a 1-D dense vector length
    n_books+M_bg (NOT a matrix). No dense (A,K)/(C,K)/(A,K,d) array.

    Edges: not cands or A==0 → []. book_emb None/empty → n_books=0,
    O = ALL articles → background-only (NOT []). |O| < k_book·n_books
    (books spend their budget on distinct regions) → M_bg=0 → pure
    book-anchored (= fl_bookdemand limit).

    Diag is computed/printed IN-FUNCTION (dual-module-safe under
    `python -m`, per Addendum-1 fix 41ce35d) at the normal finalization
    and at every early return.
    """
    A = len(article_emb)
    if not cands or A == 0:
        _n_books_early = (0 if book_emb is None
                          else len(np.asarray(book_emb, dtype=np.float32)))
        _mode = _resid_diag_mode(_n_books_early, 0)
        _FL_GAINS_RESID.clear()
        _FL_DIAG_RESID.update(n_books=int(_n_books_early), n_orphans=0,
                              orphan_frac=0.0, M_bg=0, mode=_mode, ngaps=0)
        print(f"[fl_bookdemand_resid diag] EARLY-RETURN "
              f"reason={'no_cands' if not cands else 'A==0'} "
              f"n_books={int(_n_books_early)} n_orphans=0 "
              f"orphan_frac=0 M_bg=0 mode={_mode} ngaps=0", flush=True)
        return []

    from bookgap.index import build_index, FlatL2
    from sklearn.cluster import KMeans

    k_b = min(k_book, A)
    emb_f32 = np.ascontiguousarray(article_emb.astype(np.float32))

    # ONE exact article index, reused for BOTH book and orphan-centroid
    # queries (the single FlatL2 the memory contract pins).
    art_idx = build_index(FlatL2(), emb_f32)

    # --- Book customers (demand = Nb[b], k_b nearest article-local) ---------
    if book_emb is None:
        book_arr = np.zeros((0, article_emb.shape[1]), dtype=np.float32)
    else:
        book_arr = np.asarray(book_emb, dtype=np.float32)
    n_books = 0 if book_arr.size == 0 else len(book_arr)
    if n_books:
        _Db, Nb = art_idx.search(
            np.ascontiguousarray(book_arr.astype(np.float32)), k=k_b)
    else:
        Nb = np.zeros((0, k_b), dtype=np.int64)

    # --- Orphan set: article-local ids NOT in the union of any Nb row ------
    if n_books:
        covered = set(int(a) for a in np.asarray(Nb).reshape(-1).tolist())
    else:
        covered = set()
    orphans = np.fromiter(
        (i for i in range(A) if i not in covered),
        dtype=np.int64,
    )
    n_orphans = int(orphans.size)
    orphan_frac = float(n_orphans / A)

    # --- Background customers: KMeans over article_emb[O] ONLY ------------
    # Intrinsic regulator = orphan density PER BOOK (no tuned constant):
    # M_bg = |O| // max(k_book·n_books, k_book) ⇒ bg:book ≈ |O|/(k_book·
    # n_books²). Real books CLUSTER (they never blanket the manifold at
    # n_books ≥ A/k_book), so absolute |O| ≈ A even on book-rich topics;
    # the bare |O|//k_book reduces to ≈ A/(k_book·n_books) — algebraically
    # identical to the refuted A-keyed EB (over-shrinks everywhere).
    # Normalizing by n_books makes shrinkage DECREASE as books grow
    # (dense ⇒ ~0 ≈ fl_bookdemand; sparse ⇒ high, background fills),
    # still intrinsic. n_books==0 ⇒ max(0,k_book)=k_book ⇒ M_bg=|O|//
    # k_book (background-only path unchanged).
    M_bg = int(np.clip(n_orphans // max(k_book * n_books, k_book), 0, 4000))
    M_bg = min(M_bg, n_orphans)

    _mode = _resid_diag_mode(n_books, M_bg)

    if M_bg >= 1:
        # n_init=1, random_state=0: deterministic manifold-tiling probe
        # centroids over the orphan articles only (same rationale as
        # _fl_bookdemand_eb — not a production clustering).
        km = KMeans(n_clusters=M_bg, random_state=0, n_init=1).fit(
            emb_f32[orphans])
        centroids = km.cluster_centers_.astype(np.float32)
        # Reuse the SAME single FlatL2 index for the orphan-centroid query.
        _Dc, Nc = art_idx.search(np.ascontiguousarray(centroids), k=k_b)
    else:
        Nc = np.zeros((0, k_b), dtype=np.int64)

    # Customer ids: [0 .. n_books-1] = books, [n_books .. n_books+M_bg-1] bg.
    C = n_books + M_bg

    # Diagnostics (last call wins). # not thread-safe
    _FL_DIAG_RESID.update(
        n_books=int(n_books),
        n_orphans=int(n_orphans),
        orphan_frac=float(orphan_frac),
        M_bg=int(M_bg),
        mode=_mode,
        ngaps=0,
    )

    if C == 0:
        # No books AND no orphan-background (only reachable if A==0, already
        # handled, or n_orphans==0 with n_books==0 — impossible since
        # n_books==0 ⇒ O = all A ≥ 1). Kept defensively.
        _FL_GAINS_RESID.clear()
        print(f"[fl_bookdemand_resid diag] EARLY-RETURN reason=C==0 "
              f"n_books={int(n_books)} n_orphans={int(n_orphans)} "
              f"orphan_frac={orphan_frac:.4g} M_bg={int(M_bg)} "
              f"mode={_mode} ngaps=0",
              flush=True)
        return []

    # Inverted index: article-local id -> list of candidate indices.
    # O(Σ|members_g|) — bounded by total ball mass, no dense allocation.
    inv: dict[int, list[int]] = {}
    for g, c in enumerate(cands):
        for a in c.members.tolist():
            inv.setdefault(a, []).append(g)

    # Accumulate intersection counts per (candidate, customer) sparsely.
    cand_isect: list[dict[int, int]] = [{} for _ in range(len(cands))]

    def _accumulate(neighbor_rows, cust_offset):
        for ci, row in enumerate(neighbor_rows):
            cust = cust_offset + ci
            for a in row.tolist():
                for g in inv.get(int(a), []):
                    d = cand_isect[g]
                    d[cust] = d.get(cust, 0) + 1

    if n_books:
        _accumulate(Nb, 0)
    if M_bg:
        _accumulate(Nc, n_books)

    # Convert counts -> Jaccard (REUSED form from _fl_bookdemand):
    #   cov(g,c) = isect / (|members_g| + k_b - isect)   [O(1) per entry]
    cand_cov: list[dict[int, float]] = [{} for _ in range(len(cands))]
    for g in range(len(cands)):
        mem_size = len(cands[g].members)
        for cust, isect in cand_isect[g].items():
            union = mem_size + k_b - isect
            cand_cov[g][cust] = isect / union if union > 0 else 0.0

    # Greedy facility-location over the mixed customer set.
    best_cov = np.zeros(C, dtype=np.float64)   # 1-D dense, length n_books+M_bg
    picked: set[int] = set()
    _FL_GAINS_RESID.clear()
    gaps: list[tuple[int, int, np.ndarray]] = []
    emitted = 0

    for _ in range(num_gaps):
        best_gain = -np.inf
        best_j = -1
        # Candidate-index order; strict > → lowest index wins ties.
        for g in range(len(cands)):
            if g in picked or not cand_cov[g]:
                continue
            gain_g = 0.0
            for cust, v in cand_cov[g].items():
                delta = v - best_cov[cust]
                if delta > 0.0:
                    gain_g += delta
            if gain_g > best_gain:
                best_gain = gain_g
                best_j = g

        if best_j < 0 or not np.isfinite(best_gain) or best_gain <= 0:
            break

        ids = cands[best_j].members.astype(np.int64)
        if min_gap_size > 0 and ids.size < min_gap_size:
            # Skip undersized candidate (mark consumed so it isn't re-picked),
            # but do NOT emit and do NOT update best_cov — continue to next
            # round.  Peer-consistent with _fl_bookdemand (continue, not break).
            picked.add(best_j)
            continue

        _FL_GAINS_RESID.append(best_gain)
        gaps.append((emitted, int(ids.size), ids))
        emitted += 1

        # Update best_cov only over the customers this candidate touches.
        for cust, v in cand_cov[best_j].items():
            if v > best_cov[cust]:
                best_cov[cust] = v
        picked.add(best_j)

    _FL_DIAG_RESID.update(ngaps=len(gaps))
    print(f"[fl_bookdemand_resid diag] n_books={int(n_books)} "
          f"n_orphans={int(n_orphans)} orphan_frac={orphan_frac:.4g} "
          f"M_bg={int(M_bg)} mode={_mode} ngaps={len(gaps)}", flush=True)
    return gaps


# ---------------------------------------------------------------------------
# Dispatcher (subset of the bake-off's deflate(); production exposes only
# the validated fl_bookdemand_resid — greedy stays VERBATIM in
# kball_algorithm.py and is NOT routed through here).
# ---------------------------------------------------------------------------

DeflateFn = Callable[..., list]
_STRATEGIES: dict[str, DeflateFn] = {
    "fl_bookdemand_resid": _fl_bookdemand_resid,
}


def deflate(
    strategy: str,
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    book_emb=None,
    sigma: float | None = None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Dispatch to a named deflation strategy.

    Args:
        strategy:     One of the registered strategy names
                      (production exposes "fl_bookdemand_resid").
        cands:        Candidate pool from build_kball_candidates.
        article_emb:  Article embeddings array.
        num_gaps:     Maximum gaps to emit.
        min_gap_size: Stop when the largest remaining ball < this (0 = off).
        book_emb:     Book embeddings (forwarded to book-aware strategies).
        sigma:        Smoothing bandwidth (deliberately unused by resid).
        **params:     Extra keyword arguments forwarded to the strategy fn.

    Returns:
        List of (gap_idx, gap_size, member_ids) tuples.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown deflation strategy {strategy!r}")
    return _STRATEGIES[strategy](
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, book_emb=book_emb, **params)
