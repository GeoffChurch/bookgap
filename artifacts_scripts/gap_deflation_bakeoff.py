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

    Coverage cov(g,b) = Jaccard(members_g, Nb[b])
        = |members_g ∩ Nb[b]| / (|members_g| + k_b − |members_g ∩ Nb[b]|)

    Using Jaccard (rather than |intersection|/k_book) makes coverage
    saturation-resistant: a 500-member background ball overlapping a 10-article
    Nb scores ≤ 10/500 ≈ 0.02 and cannot dominate; a tight ball that IS the
    book's neighborhood scores near 1.  The submodular FL structure is
    unchanged: F(S) = Σ_b max_{g∈S} cov(g,b), greedy marginal gain, monotone
    non-increasing.

    No sigma/kernel used — sidesteps the global-vs-article-local sigma
    index-space bug class entirely (sigma is accepted but ignored).

    Memory-bounded: inverted index maps article-local id → candidate list;
    per-candidate coverage stored as sparse dicts {book_idx: jaccard_score}.
    |union| = |members_g| + k_b − |intersection| is computed O(1) from the
    existing intersection count — no new dense structure.
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

    # Accumulate intersection counts: cand_isect[g] = {book_b: intersection_count}
    # Process each book's neighborhood; accumulate into sparse per-candidate dicts
    cand_isect: list[dict[int, int]] = [{} for _ in range(len(cands))]
    for b in range(B):
        for a in Nb[b].tolist():
            for g in inv.get(int(a), []):
                cand_isect[g][b] = cand_isect[g].get(b, 0) + 1

    # Convert intersection counts to Jaccard scores:
    #   cov(g,b) = isect / (|members_g| + k_b − isect)
    # |union| = |members_g| + k_b − isect (inclusion-exclusion, O(1) per entry)
    cand_cov: list[dict[int, float]] = [{} for _ in range(len(cands))]
    for g in range(len(cands)):
        mem_size = len(cands[g].members)
        for b, isect in cand_isect[g].items():
            union = mem_size + k_b - isect
            cand_cov[g][b] = isect / union if union > 0 else 0.0

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
            # Skip undersized candidate (mark consumed so it isn't re-picked),
            # but do NOT emit and do NOT update best_cov — continue to next round.
            # This matches _greedy_hard/_submodular_fl peer behavior.
            picked.add(best_j)
            continue

        _FL_GAINS_BD.append(best_gain)
        gaps.append((emitted, int(ids.size), ids))
        emitted += 1

        # Update best_cov for books this candidate touches
        for b, v in cand_cov[best_j].items():
            if v > best_cov[b]:
                best_cov[b] = v
        picked.add(best_j)

    return gaps


# --- fl_bookdemand_eb: empirical-Bayes shrinkage demand ---------------------
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
# Module-global trace/diag stores (last call wins). # not thread-safe
_FL_GAINS_EB: list[float] = []
_FL_DIAG_EB: dict = {"n_books": 0, "M": 0, "ratio_bg_book": 0.0,
                     "shrinkage": 0.0, "s_target": None, "mode": "A-keyed"}


def _eb_diag_mode(eb_target_shrinkage: float | None, n_books: int) -> str:
    """3-way diag `mode` label (single source of truth; can't drift).

    - "nbooks-keyed"     : s given AND n_books > 0 (M from n_books·s/(1−s)).
    - "A-keyed"          : s is None — the deliberate refuted A-keyed control
                           (the `fl_bookdemand_eb` strategy).
    - "A-keyed-fallback" : s given BUT n_books == 0 — a variant silently
                           degraded to the A-keyed M because the topic has
                           no books to anchor the ratio to.
    """
    if eb_target_shrinkage is not None and n_books == 0:
        return "A-keyed-fallback"
    if eb_target_shrinkage is None:
        return "A-keyed"
    return "nbooks-keyed"


def _fl_bookdemand_eb_trace() -> list[float]:
    return list(_FL_GAINS_EB)


def _fl_bookdemand_eb_diag() -> dict:
    return dict(_FL_DIAG_EB)


def _fl_bookdemand_eb(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,          # deliberately unused — see _fl_bookdemand: sidesteps
                          # the global-vs-article-local index-space bug class
    book_emb=None,
    k_book: int = 10,
    eb_book_weight: float = 1.0,   # exposed-but-untuned: >1 up-weights books
    eb_target_shrinkage: float | None = None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Facility-location deflation over BOOK ∪ BACKGROUND-coreset demand.

    One greedy-FL objective with a *mixed* customer set so the regime
    adaptation is emergent (no `if`):

    - Book customers (one per training book b): demand Nb[b] = k_book
      nearest article-local articles to book_emb[b] — exactly as
      _fl_bookdemand.
    - Background customers (one per k-means centroid, count M): demand
      Nc[c] = k_book nearest article-local articles to centroid c.
      Centroids = KMeans(n_clusters=M, random_state=0, n_init=1).
    - M = clip(A // k_book, 64, 4000), clamped M <= A.

    cov(g,c) = Jaccard(members_g, demand(c)) — REUSED verbatim from
    _fl_bookdemand: isect / (|members_g| + k_b - isect).  Every customer
    weight 1.0; book customers carry eb_book_weight (default 1.0).
    Greedy maximises Σ_c w_c · max(0, cov(g,c) - best_cov[c]); stop when
    best gain <= 0 or num_gaps reached; min_gap_size -> continue
    (peer-consistent with _fl_bookdemand, NOT break).

    The shrinkage M/(M+n_books) falls out of the customer mixture: no
    tuned λ. No-books → background-only (graceful, NOT []). sigma unused.

    Memory-bounded (mirrors _fl_bookdemand exactly): one exact FlatL2
    over article_emb (built once, reused for book & background queries);
    inverted index article→candidate; per-candidate sparse
    {customer_id: cov} dict; best_cov is a 1-D dense vector length
    n_books+M (NOT a matrix). No dense (A,K)/(C,K)/(A,K,d) array.
    """
    if eb_target_shrinkage is not None and not (
            0.0 < eb_target_shrinkage < 1.0):
        raise ValueError(
            f"eb_target_shrinkage must be in the open interval (0, 1); "
            f"got {eb_target_shrinkage!r}")

    A = len(article_emb)
    if not cands or A == 0:
        # n_books unresolved here (no FL ran); derive it from book_emb so the
        # 3-way mode rule is identical to the finalization path.
        _n_books_early = (0 if book_emb is None
                          else len(np.asarray(book_emb, dtype=np.float32)))
        _eb_mode = _eb_diag_mode(eb_target_shrinkage, _n_books_early)
        _FL_GAINS_EB.clear()
        _FL_DIAG_EB.update(n_books=0, M=0, ratio_bg_book=0.0,
                           shrinkage=0.0, s_target=eb_target_shrinkage,
                           mode=_eb_mode)
        print(f"[fl_bookdemand_eb diag] EARLY-RETURN "
              f"reason={'no_cands' if not cands else 'A==0'} "
              f"n_books=0 M=0 ratio_bg_book=0 shrinkage=0 "
              f"s_target={eb_target_shrinkage} mode={_eb_mode} ngaps=0",
              flush=True)
        return []

    from bookgap.index import build_index, FlatL2
    from sklearn.cluster import KMeans

    k_b = min(k_book, A)
    emb_f32 = np.ascontiguousarray(article_emb.astype(np.float32))

    # ONE exact article index, reused for BOTH book and background queries.
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

    # --- Background customers (k-means coreset over article_emb) ------------
    # M tiles the manifold at ~book-neighborhood granularity.
    #
    # A-keyed (eb_target_shrinkage is None, OR no books to anchor to):
    #   M = clip(A // k_book, 64, 4000), clamped M <= A — keys the prior
    #   resolution to article density (the refuted control / curve point).
    # n_books-keyed (eb_target_shrinkage = s, 0<s<1, AND n_books > 0):
    #   M = round(n_books * s/(1-s)), clamped to [8, min(4000, A)] — pins
    #   bg:book = s/(1-s) and shrinkage M/(M+n_books) ≈ s INDEPENDENT of A.
    _eb_mode = _eb_diag_mode(eb_target_shrinkage, n_books)
    if eb_target_shrinkage is not None and n_books > 0:
        s = float(eb_target_shrinkage)
        M = int(round(n_books * (s / (1.0 - s))))
        M = max(8, min(M, 4000, A))
    else:
        # eb_target_shrinkage None, or set but n_books == 0 → A-keyed
        # fallback (no book signal to anchor the ratio to; preserves the
        # no-book graceful contract).
        M = int(np.clip(A // k_book, 64, 4000))
        M = min(M, A)
    # Unreachable in practice (A>=1 guaranteed above; clip lower bound 64);
    # kept defensively.
    if M < 1:
        _FL_GAINS_EB.clear()
        _FL_DIAG_EB.update(n_books=n_books, M=0, ratio_bg_book=0.0,
                           shrinkage=0.0, s_target=eb_target_shrinkage,
                           mode=_eb_mode)
        print(f"[fl_bookdemand_eb diag] EARLY-RETURN reason=M<1 "
              f"n_books={n_books} M=0 ratio_bg_book=0 shrinkage=0 "
              f"s_target={eb_target_shrinkage} mode={_eb_mode} ngaps=0",
              flush=True)
        return []
    # n_init=1 is deliberate: these are manifold-tiling probe centroids, not a
    # production clustering — they don't need global optimality (measured
    # 0.09% inertia vs n_init=10 for ~10x the wall time), and dropping to a
    # single seeded k-means++ init keeps the long real-topic job tractable.
    # Still deterministic (random_state=0, single seeded init).
    km = KMeans(n_clusters=M, random_state=0, n_init=1).fit(emb_f32)
    centroids = km.cluster_centers_.astype(np.float32)
    _Dc, Nc = art_idx.search(np.ascontiguousarray(centroids), k=k_b)

    # Customer ids: [0 .. n_books-1] = books, [n_books .. n_books+M-1] = bg.
    C = n_books + M
    cust_weight = np.ones(C, dtype=np.float64)
    if n_books:
        cust_weight[:n_books] = float(eb_book_weight)

    # Diagnostics (last call wins). # not thread-safe
    _FL_DIAG_EB.update(
        n_books=int(n_books),
        M=int(M),
        ratio_bg_book=float(M / max(n_books, 1)),
        shrinkage=float(M / (M + n_books)),
        s_target=eb_target_shrinkage,
        mode=_eb_mode,
    )

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
    best_cov = np.zeros(C, dtype=np.float64)   # 1-D dense, length n_books+M
    picked: set[int] = set()
    _FL_GAINS_EB.clear()
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
                    gain_g += cust_weight[cust] * delta
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

        _FL_GAINS_EB.append(best_gain)
        gaps.append((emitted, int(ids.size), ids))
        emitted += 1

        # Update best_cov only over the customers this candidate touches.
        for cust, v in cand_cov[best_j].items():
            if v > best_cov[cust]:
                best_cov[cust] = v
        picked.add(best_j)

    print(f"[fl_bookdemand_eb diag] n_books={int(n_books)} M={int(M)} "
          f"ratio_bg_book={float(M / max(n_books, 1)):.4g} "
          f"shrinkage={float(M / (M + n_books)):.4g} "
          f"s_target={eb_target_shrinkage} mode={_eb_mode} "
          f"ngaps={len(gaps)}", flush=True)
    return gaps


# --- fl_bookdemand_resid: intrinsic residual demand (no `s`, nothing fitted)
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
#       (Addendum 2, 2026-05-18). Same greedy-FL machinery as
#       _fl_bookdemand_eb, but the background coreset is built ONLY over
#       book-orphan articles, so the operating point is set intrinsically
#       by orphan mass — no s/M-vs-n_books knob, nothing calibrated.
# SEPARATE trace/diag globals (no cross-contamination with the eb arm).
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


# --- best-of-both reframes (Addendum 3, 2026-05-18) -------------------------
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
# A mixed book/manifold demand objective dilutes — no proportion
# (constant or intrinsic) is best at both regimes (resid proved this).
# Escape the box: stop mixing. Arm A unions the two PURE signals; arm B
# uses _submodular_fl as a robust GENERATOR then RANKS by book demand.
# Both are additive — they reuse _fl_bookdemand / _submodular_fl verbatim
# and never modify deflate/run_arm/kball_seeds_from_state or any arm.


def _fl_union_bd_sub(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Arm A — UNION (not mixture) of _fl_bookdemand and _submodular_fl.

    g1 = _fl_bookdemand(...) (book_emb flows via **params; book-anchored).
    g2 = _submodular_fl(...) (ignores book_emb; manifold coverage).
    Return g1 ++ g2 with element-0 renumbered 0..N-1 over the
    concatenation; each tuple's (size, member_ids) is left intact. No
    dedup here — the downstream run_refine_pipeline does containment+
    Jaccard dedup + effective_size ranking, which surfaces the
    book-anchored winners on dense and the manifold-coverage gaps on
    sparse automatically. No knob, no branch, no fitted constant.

    Edges: `not cands or A==0` → [] (both sub-arms already return [];
    the concat of [] is []). `sigma` is passed through to _submodular_fl
    (which needs it) and ignored by _fl_bookdemand (set-overlap only).
    """
    g1 = _fl_bookdemand(
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, **params)
    g2 = _submodular_fl(
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, **params)
    out: list[tuple[int, int, np.ndarray]] = []
    for r, (_old_rank, sz, ids) in enumerate(list(g1) + list(g2)):
        out.append((r, sz, ids))
    print(f"[fl_union_bd_sub diag] g1_bookdemand={len(g1)} "
          f"g2_submodular={len(g2)} total={len(out)}", flush=True)
    return out


# fl_gen_rank diag — SEPARATE globals (no cross-contamination with eb/resid).
# not thread-safe
_FL_DIAG_GENRANK: dict = {"n_books": 0, "pool_size": 0, "n_book_shaped": 0,
                          "mode": "gen-only", "ngaps": 0}


def _genrank_mode(n_books: int) -> str:
    """diag `mode` label (single source of truth; can't drift).

    - "gen-rank" : n_books > 0 — book demand re-ranks the submodular pool.
    - "gen-only" : n_books == 0 — bd is uninformative → pure submodular
                   order (graceful = the robust generator, no collapse).
    """
    return "gen-rank" if n_books > 0 else "gen-only"


def _fl_gen_rank_diag() -> dict:
    return dict(_FL_DIAG_GENRANK)


def _fl_gen_rank(
    cands: list[Candidate],
    article_emb: np.ndarray,
    *,
    num_gaps: int,
    min_gap_size: int,
    sigma=None,          # forwarded to _submodular_fl (the generator) which
                          # needs it; gen-rank's own scoring is pure
                          # set-overlap (no kernel/sigma) — see _fl_bookdemand
    book_emb=None,
    k_book: int = 10,
    **params,
) -> list[tuple[int, int, np.ndarray]]:
    """Arm B — decouple GENERATION from SELECTION.

    1. GENERATE the candidate pool P with _submodular_fl (robust,
       regime-agnostic, never collapses — its recall is the FLOOR).
    2. SCORE each pool gap g by book-anchored demand
         bd(g) = Σ_b Jaccard(members_g, Nb[b])
       Nb[b] = the k_b = min(k_book, A) nearest article-local articles to
       book b (exact FlatL2 over article_emb, queried with book_emb) —
       REUSE _fl_bookdemand's exact Jaccard form
         isect / (|members_g| + k_b − isect)
       and its sparse inverted-index pattern (article→which-pool-gaps;
       per-gap sparse {book: isect}). NO dense (A,K)/(C,K)/(A,K,d) array.
    3. EMIT: gaps with bd>0 sorted by bd DESC (stable; ties keep the
       original submodular order), THEN gaps with bd==0 in original
       submodular order (coverage backfill — NOT dropped, so sparse
       coverage is preserved). element-0 = this emission order; cap to
       num_gaps. Each tuple's (size, member_ids) is left intact.

    Self-weakening intrinsic property: book-sparse → bd ≈ noise → emit ≈
    submodular pool → ≈ submodular_fl's floor (no collapse). Book-dense →
    bd surfaces the book-shaped regions from the rich manifold pool → ≈
    fl_bookdemand. Best-of-both by separation of concerns — no mixing
    constant, no branch; the book signal self-attenuates exactly when it
    should.

    Edges: book_emb None/empty → all bd=0 → emit = P in submodular order
    (graceful = pure generator), mode "gen-only". `not cands or A==0` →
    [] (the submodular pool is [] and the concat is []). `sigma` is
    unused by gen-rank's own scoring (pure set-overlap, sidesteps the
    global-vs-article-local index-space bug class as in _fl_bookdemand);
    it IS forwarded to the _submodular_fl generator, which requires it.

    Determinism: _submodular_fl is deterministic; the rank uses a STABLE
    sort key (-bd[i], i) so ties fall back to the original submodular
    order. No RNG.

    Memory-bounded (mirrors _fl_bookdemand): one exact FlatL2 (built
    once), an inverted index article→pool-gap-index, a per-gap sparse
    {book: isect} dict, and a 1-D length-|P| `bd` vector. No dense
    (A,K)/(C,K)/(A,K,d) array anywhere.

    Diag is computed/printed IN-FUNCTION (dual-module-safe under
    `python -m`, per the Addendum-1 fix) at finalization AND at every
    early return.
    """
    A = len(article_emb)

    # Step 1: GENERATE the pool with the robust submodular generator.
    pool = _submodular_fl(
        cands, article_emb, num_gaps=num_gaps,
        min_gap_size=min_gap_size, sigma=sigma, **params)

    # Resolve book count (used by the 3-way mode rule everywhere).
    if book_emb is None:
        book_arr = np.zeros((0,), dtype=np.float32)
    else:
        book_arr = np.asarray(book_emb, dtype=np.float32)
    n_books = 0 if book_arr.size == 0 else len(book_arr)
    _mode = _genrank_mode(n_books)

    if not cands or A == 0 or not pool:
        _FL_DIAG_GENRANK.update(n_books=int(n_books), pool_size=len(pool),
                                n_book_shaped=0, mode=_mode, ngaps=0)
        print(f"[fl_gen_rank diag] EARLY-RETURN "
              f"reason={'no_cands' if not cands else ('A==0' if A == 0 else 'empty_pool')} "
              f"n_books={int(n_books)} pool_size={len(pool)} "
              f"n_book_shaped=0 mode={_mode} ngaps=0", flush=True)
        return []

    pool_size = len(pool)
    # member_ids array per pool gap (article-local; left intact on emit).
    pool_members = [np.asarray(ids, dtype=np.int64) for _, _, ids in pool]

    bd = np.zeros(pool_size, dtype=np.float64)   # 1-D length-|P| score vec

    if n_books > 0:
        from bookgap.index import build_index, FlatL2

        k_b = min(k_book, A)
        # ONE exact article index; query books for k_b nearest articles.
        art_idx = build_index(
            FlatL2(), np.ascontiguousarray(article_emb.astype(np.float32)))
        _Db, Nb = art_idx.search(
            np.ascontiguousarray(book_arr.astype(np.float32)), k=k_b)

        # Inverted index: article-local id → list of pool-gap indices.
        # O(Σ|members_p|) — bounded by pool member mass, no dense alloc.
        inv: dict[int, list[int]] = {}
        for p, mem in enumerate(pool_members):
            for a in mem.tolist():
                inv.setdefault(int(a), []).append(p)

        # Per-pool-gap sparse intersection counts {book: isect}.
        pool_isect: list[dict[int, int]] = [{} for _ in range(pool_size)]
        for b in range(n_books):
            for a in Nb[b].tolist():
                for p in inv.get(int(a), []):
                    d = pool_isect[p]
                    d[b] = d.get(b, 0) + 1

        # bd(g) = Σ_b Jaccard(members_g, Nb[b]), reusing _fl_bookdemand's
        # exact form: isect / (|members_g| + k_b − isect) [O(1) per entry].
        for p in range(pool_size):
            mem_size = int(pool_members[p].size)
            acc = 0.0
            for b, isect in pool_isect[p].items():
                union = mem_size + k_b - isect
                if union > 0:
                    acc += isect / union
            bd[p] = acc

    n_book_shaped = int(np.count_nonzero(bd > 0.0))

    # Step 3: EMIT. Stable sort by -bd (ties → original submodular order).
    order = sorted(range(pool_size), key=lambda i: (-bd[i], i))
    # bd>0 first (sorted desc, stable), then bd==0 in original order —
    # the (-bd, i) key already yields exactly this (all bd>0 sort ahead
    # of all bd==0, and within bd==0 the secondary key i preserves the
    # submodular order; backfill kept, nothing dropped).
    gaps: list[tuple[int, int, np.ndarray]] = []
    for rank, i in enumerate(order[:num_gaps]):
        _old_rank, sz, ids = pool[i]
        gaps.append((rank, sz, ids))

    _FL_DIAG_GENRANK.update(
        n_books=int(n_books), pool_size=int(pool_size),
        n_book_shaped=n_book_shaped, mode=_mode, ngaps=len(gaps))
    print(f"[fl_gen_rank diag] n_books={int(n_books)} "
          f"pool_size={int(pool_size)} n_book_shaped={n_book_shaped} "
          f"mode={_mode} ngaps={len(gaps)}", flush=True)
    return gaps


def heldout_book_split(n_books: int, frac: float = 0.2,
                       seed: int = 42) -> np.ndarray:
    """Deterministic held-out book indices (sorted, no replacement)."""
    rng = np.random.RandomState(seed)
    n_hold = int(round(n_books * frac))
    return np.sort(rng.choice(n_books, size=n_hold, replace=False))


# Shared top-k for the rank-sensitive metrics. inter_gap_metrics and the
# genuine held-out recall@20 in run_arm both slice ranked[:RANK_TOPK];
# deduplicating the literal keeps "@20" honest in both places.
RANK_TOPK = 20


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


# ---------------------------------------------------------------------------
# Ranking-stage book-aware selection — the BINDING stage.
# Spec: docs/superpowers/specs/2026-05-18-ranking-stage-book-aware-design.md
#
# The deflation stage only changes which gaps SURVIVE; the regime-blind
# `effective_size` ranking in run_arm decided the held-out recall metric
# and washed out every seed reorder (fl_gen_rank was byte-identical to
# submodular_fl). The self-weakening book signal belongs HERE, at ranking.
#
# Strictly additive: rank="effsize" is LITERALLY the prior run_arm line
# (a no-op refactor — the additive-proof gate). bookdemand / effsize_x_bd
# are opt-in. `bd(g)` reuses _fl_bookdemand's EXACT Jaccard form
# isect/(|m|+k_b−isect) and its sparse inverted-index pattern: NO dense
# (A,N)/(C,N)/(A,N,d) array. Book-sparse ⇒ bd≈0 ⇒ key ≈ effective_size
# (graceful floor, no branch/knob/fitted constant).
# ---------------------------------------------------------------------------

# Diag store (last call wins). not thread-safe.
_RANK_DIAG: dict = {"rank": "effsize", "n_books": 0,
                    "n_book_shaped": 0, "mode": "effsize"}


def _rank_diag() -> dict:
    return dict(_RANK_DIAG)


def _rank_book_demand(
    survivors: list[dict],
    *,
    book_emb,
    article_emb_for_rank: np.ndarray,
    k_book: int = 10,
    article_global_idx=None,
) -> np.ndarray:
    """bd(g) = Σ_b Jaccard(g_members, Nb[b]) for every survivor.

    g_members = g["refined_member_indices"] mapped GLOBAL→ARTICLE-LOCAL
    exactly as run_arm's gm_local path: when ``article_global_idx`` is
    given the members are remapped through the SAME
    ``_global_to_article_local`` helper run_arm uses for gm_local (global
    work ids → 0..A-1 positions, dropping non-article ids). When it is
    None the members are already article-local positions and any id
    outside 0..A-1 is dropped (same drop-non-member semantics).

    Nb[b] = the k_b = min(k_book, A) nearest article-local articles to
    KEPT book b (ONE exact FlatL2 over article_emb_for_rank, queried with
    the kept book embeddings). Jaccard reuses _fl_bookdemand's EXACT form
    isect/(|m|+k_b−isect) and its sparse inverted-index (article→survivor)
    + per-survivor sparse {book:isect}. No dense (A,N)/(C,N)/(A,N,d).

    book_emb None/empty OR no overlap ⇒ all-zero bd (graceful).
    """
    n = len(survivors)
    bd = np.zeros(n, dtype=np.float64)
    if book_emb is None or n == 0:
        return bd
    book_arr = np.asarray(book_emb, dtype=np.float32)
    art_arr = np.asarray(article_emb_for_rank, dtype=np.float32)
    A = len(art_arr)
    if book_arr.size == 0 or A == 0:
        return bd

    from bookgap.index import build_index, FlatL2

    B = len(book_arr)
    k_b = min(k_book, A)
    # ONE exact article index; query KEPT books for k_b nearest articles.
    art_idx = build_index(
        FlatL2(), np.ascontiguousarray(art_arr.astype(np.float32)))
    _D, Nb = art_idx.search(
        np.ascontiguousarray(book_arr.astype(np.float32)), k=k_b)

    # Members → article-local positions. Reuse run_arm's EXACT gm_local
    # mapping path when a global index is supplied; otherwise the members
    # are already article-local and out-of-range ids are dropped (same
    # drop-non-member semantics as _global_to_article_local).
    raw = [list(g.get("refined_member_indices", [])) for g in survivors]
    if article_global_idx is not None:
        g_members = _global_to_article_local(raw, article_global_idx)
    else:
        g_members = [[int(x) for x in ml if 0 <= int(x) < A]
                     for ml in raw]

    # Inverted index: article-local id → list of survivor indices.
    # O(Σ|members_g|) — bounded by survivor member mass, no dense alloc.
    inv: dict[int, list[int]] = {}
    for gi, mem in enumerate(g_members):
        for a in mem:
            inv.setdefault(a, []).append(gi)

    # Per-survivor sparse intersection counts {book: isect}.
    g_isect: list[dict[int, int]] = [{} for _ in range(n)]
    for b in range(B):
        for a in Nb[b].tolist():
            for gi in inv.get(int(a), []):
                dd = g_isect[gi]
                dd[b] = dd.get(b, 0) + 1

    # bd(g) = Σ_b isect / (|members_g| + k_b − isect)  [O(1) per entry],
    # the EXACT _fl_bookdemand Jaccard form.
    for gi in range(n):
        mem_size = len(g_members[gi])
        acc = 0.0
        for b, isect in g_isect[gi].items():
            union = mem_size + k_b - isect
            if union > 0:
                acc += isect / union
        bd[gi] = acc
    return bd


def _rank_survivors(
    survivors: list[dict],
    rank: str,
    *,
    book_emb,
    article_emb_for_rank: np.ndarray,
    k_book: int = 10,
    article_global_idx=None,
) -> list[dict]:
    """Rank-stage dispatcher (mirrors the _STRATEGIES deflation pattern).

    - "effsize"      : LITERALLY the prior run_arm expression
                       sorted(survivors,
                              key=lambda g: g.get("effective_size", 0.0),
                              reverse=True)
                       — verbatim; this branch is a no-op refactor and
                       must NEVER consult books (the additive-proof gate).
    - "bookdemand"   : stable sort by (-bd(g), -effective_size(g)).
    - "effsize_x_bd" : stable sort by -(effective_size(g)·(1+bd(g))).
    - unknown        : ValueError(f"unknown rank {rank!r}").

    Diag printed in-function (dual-module-safe under `python -m`).
    """
    if rank == "effsize":
        ranked = sorted(survivors,
                        key=lambda g: g.get("effective_size", 0.0),
                        reverse=True)
        _RANK_DIAG.update(rank=rank, n_books=0, n_book_shaped=0,
                          mode="effsize")
        print(f"[rank diag] rank={rank} n_books=0 n_book_shaped=0 "
              f"mode=effsize", flush=True)
        return ranked

    if rank not in ("bookdemand", "effsize_x_bd"):
        raise ValueError(f"unknown rank {rank!r}")

    book_arr = (None if book_emb is None
                else np.asarray(book_emb, dtype=np.float32))
    nb = 0 if (book_arr is None or book_arr.size == 0) else len(book_arr)
    bd = _rank_book_demand(
        survivors, book_emb=book_emb,
        article_emb_for_rank=article_emb_for_rank, k_book=k_book,
        article_global_idx=article_global_idx)
    nbs = int(np.count_nonzero(bd > 0.0))
    mode = "book-aware" if (rank != "effsize" and nb > 0) else "effsize"

    idx = list(range(len(survivors)))

    def _eff(i: int) -> float:
        return float(survivors[i].get("effective_size", 0.0))

    if rank == "bookdemand":
        # stable sort by (-bd, -effective_size) — bd>0 ahead of bd==0;
        # bd all 0 ⇒ pure effsize-stable (graceful, no special-case).
        order = sorted(idx, key=lambda i: (-bd[i], -_eff(i)))
    else:  # effsize_x_bd
        # self-weakening blend: book-sparse ⇒ bd≈0 ⇒ key ≈ effective_size
        # ⇒ identical order to "effsize" (graceful floor, no branch).
        order = sorted(idx, key=lambda i: -(_eff(i) * (1.0 + bd[i])))

    ranked = [survivors[i] for i in order]
    _RANK_DIAG.update(rank=rank, n_books=int(nb),
                      n_book_shaped=nbs, mode=mode)
    print(f"[rank diag] rank={rank} n_books={int(nb)} "
          f"n_book_shaped={nbs} mode={mode}", flush=True)
    return ranked


def inter_gap_metrics(survivors: list[dict], top_k: int = RANK_TOPK) -> dict:
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
_STRATEGIES["fl_bookdemand_eb"] = _fl_bookdemand_eb
_STRATEGIES["fl_bookdemand_resid"] = _fl_bookdemand_resid


def _make_eb_s(s: float) -> DeflateFn:
    """Factory for the n_books-keyed fl_bookdemand_eb_s* variants.

    Each delegates to the shared _fl_bookdemand_eb core with
    eb_target_shrinkage bound to ``s``; any incoming
    eb_target_shrinkage in params is popped first to avoid a
    duplicate-kwarg TypeError.  fl_bookdemand_eb itself (None path)
    stays untouched.
    """
    def _arm(cands, article_emb, *, num_gaps, min_gap_size,
             sigma=None, **params):
        params.pop("eb_target_shrinkage", None)
        return _fl_bookdemand_eb(
            cands, article_emb, num_gaps=num_gaps,
            min_gap_size=min_gap_size, sigma=sigma,
            eb_target_shrinkage=s, **params)
    return _arm


_STRATEGIES["fl_bookdemand_eb_s30"] = _make_eb_s(0.30)
_STRATEGIES["fl_bookdemand_eb_s50"] = _make_eb_s(0.50)
_STRATEGIES["fl_bookdemand_eb_s70"] = _make_eb_s(0.70)

# best-of-both reframes (Addendum 3): union (A) and gen→rank (B).
_STRATEGIES["fl_union_bd_sub"] = _fl_union_bd_sub
_STRATEGIES["fl_gen_rank"] = _fl_gen_rank


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
    "deflation", "rank", "scope",
    "mean_pairwise_jaccard_top20", "unique_coverage_at_20",
    "heldout_recall_at20", "heldout_spread_at20",
    "heldout_recall_fullset", "heldout_spread_fullset",
    "gaps_emitted", "runtime_s",
]


def run_arm(
    scope: str,
    deflation: str,
    out_dir: Path,
    *,
    index_type: str = "hnsw",
    n_keep: int = 370,
    rank: str = "effsize",
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
    ranked = _rank_survivors(survivors, rank, book_emb=state["emb"][np.asarray(state2["book_global_idx"])], article_emb_for_rank=state["emb"][np.asarray(state["article_global_idx"])], k_book=10, article_global_idx=state["article_global_idx"])
    leaderboard_dir = Path(out_dir) / "leaderboards"
    _write_leaderboard(leaderboard_dir, scope, deflation, ranked, titles_map)

    # Step 7: compute metrics row
    m = inter_gap_metrics(ranked)
    art_gidx = np.asarray(state["article_global_idx"])
    # full-set (OLD behaviour, verbatim): the byte-anchor gate + the
    # breadth-vs-quality delta. Identical inputs/call to the pre-change
    # `rec, spread` — just renamed to *_fullset.
    gm_full = [list(g.get("refined_member_indices", [])) for g in ranked]
    gm_full_local = _global_to_article_local(gm_full, art_gidx)
    rec_full, spr_full = heldout_recall_spread(
        heldout_book_emb,
        state["emb"][art_gidx],
        gm_full_local,
        k_near=10,
        cover_frac=0.5,
    )
    # top-20 (NEW PRIMARY): genuine rank-sensitive recall@20 over
    # ranked[:RANK_TOPK]; same mapping, same call shape as full-set.
    gm_top = [list(g.get("refined_member_indices", []))
              for g in ranked[:RANK_TOPK]]
    gm_top_local = _global_to_article_local(gm_top, art_gidx)
    rec20, spr20 = heldout_recall_spread(
        heldout_book_emb,
        state["emb"][art_gidx],
        gm_top_local,
        k_near=10,
        cover_frac=0.5,
    )
    return {
        "deflation": deflation,
        "rank": rank,
        "scope": scope,
        **m,
        "heldout_recall_at20": rec20,
        "heldout_spread_at20": spr20,
        "heldout_recall_fullset": rec_full,
        "heldout_spread_fullset": spr_full,
        "gaps_emitted": len(ranked),   # survivors after refine; low count (e.g. 1) signals saturation
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
    p.add_argument("--rank",
                   choices=["effsize", "bookdemand", "effsize_x_bd"],
                   default="effsize",
                   help="survivor-ranking axis (default effsize = prior "
                        "behavior, byte-identical to committed scorecards)")
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
                rank=args.rank,
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
