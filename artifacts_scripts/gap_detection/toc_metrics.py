"""Curriculum Showdown — objective ToC metric suite + ordering (Phase-0, reusable Phase 1).

Pure functions over numpy arrays; no I/O, no network. Pre-committed definitions
(worklog "Curriculum Showdown (Phase 0)", recorded before any sweep output):

* k-selection rule: max mean silhouette subject to no cluster <3% or >25% of
  mass; if no k is feasible, fewest violating clusters, then higher silhouette,
  then smaller k.
* ordering: exact minimum of forward-pointing citation flow
  sum_{pos(a)<pos(b)} F[a,b] over linear arrangements (bitmask DP, n<=16);
  ties broken by the arrangement maximizing sum(position * year-rank), i.e.
  zero-flow degenerates to ascending median publication year.
* metric 1 (field-mass coverage): share of corpus docs within `radius` of the
  NEAREST ToC representative (k=1; radius = the banked t3 q25 scope threshold).
* metric 2 (gap-region closure): a region is closed iff the MEDIAN over its
  member docs of distance-to-nearest-representative <= radius.
* metric 3 (prereq violations): # cross-chapter citation edges (w cites v)
  with chapter(w) ordered BEFORE chapter(v); reported with the cross-chapter
  edge total so a share can be formed.

Context hygiene: this module never touches text at all — vectors, ids, counts.
"""

from __future__ import annotations

import numpy as np

MIN_SHARE = 0.03
MAX_SHARE = 0.25


# ------------------------------------------------------------- k-selection rule
def n_violations(shares: list[float] | np.ndarray,
                 min_share: float = MIN_SHARE, max_share: float = MAX_SHARE) -> int:
    """# clusters whose mass share breaks the pre-committed [3%, 25%] band."""
    s = np.asarray(shares, dtype=np.float64)
    return int(((s < min_share) | (s > max_share)).sum())


def select_k(rows: dict[int, dict], *,
             min_share: float = MIN_SHARE, max_share: float = MAX_SHARE) -> tuple[int, str]:
    """Pre-committed rule. rows: {k: {"silhouette": float, "shares": [...]}}.

    Feasible k (no share outside [min_share, max_share]) with max silhouette;
    deterministic tie-break smaller k. If NO k is feasible: fewest violating
    clusters, then higher silhouette, then smaller k (the recorded fallback).
    """
    assert rows, "empty sweep"
    feas = [k for k in sorted(rows) if n_violations(rows[k]["shares"], min_share, max_share) == 0]
    if feas:
        k = max(feas, key=lambda k: (rows[k]["silhouette"], -k))
        return k, "feasible-max-silhouette"
    k = min(sorted(rows),
            key=lambda k: (n_violations(rows[k]["shares"], min_share, max_share),
                           -rows[k]["silhouette"], k))
    return k, "fallback-fewest-violations"


# ------------------------------------------------------------- citation flow + order
def citation_flow(src_cluster: np.ndarray, dst_cluster: np.ndarray, n_clusters: int) -> np.ndarray:
    """Edge lists (citing-cluster, cited-cluster) -> dense flow matrix F[a, b] =
    # edges from members of a to members of b. Diagonal kept (callers ignore it)."""
    F = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    if len(src_cluster):
        np.add.at(F, (np.asarray(src_cluster, dtype=np.int64),
                      np.asarray(dst_cluster, dtype=np.int64)), 1.0)
    return F


def order_min_forward_flow(F: np.ndarray, tie_key: np.ndarray) -> tuple[list[int], float]:
    """Exact min-forward-flow linear arrangement (bitmask DP over subsets, n<=16).

    Forward flow of an order = sum over pairs placed (a before b) of F[a, b]
    (a cites b later = a depends on unseen material). Lexicographic secondary
    objective: maximize sum(position * rank(tie_key)) — with zero flow this is
    exactly ascending tie_key (median publication year) order. Deterministic.
    Returns (order, minimal forward flow).
    """
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    assert F.shape == (n, n) and n <= 16, "DP is exact only for n <= 16"
    if n == 0:
        return [], 0.0
    # unique ranks: ascending tie_key, index breaks exact ties
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), np.asarray(tie_key)))] = np.arange(n)
    size = 1 << n
    INF = float("inf")
    dp_p = np.full(size, INF)            # primary: forward flow
    dp_s = np.full(size, INF)            # secondary: -sum(position * rank)
    parent = np.full(size, -1, dtype=np.int64)
    dp_p[0], dp_s[0] = 0.0, 0.0
    members = ((np.arange(size)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    for S in range(size):
        if dp_p[S] == INF:
            continue
        m = members[S]
        p = int(m.sum())                  # position of the next appended node
        col_costs = m @ F                 # cost of appending v = flow from S into v
        for v in range(n):
            if m[v]:
                continue
            T = S | (1 << v)
            cp = dp_p[S] + col_costs[v]
            cs = dp_s[S] - float(p * rank[v])
            if (cp < dp_p[T]) or (cp == dp_p[T] and cs < dp_s[T]):
                dp_p[T], dp_s[T], parent[T] = cp, cs, v
    order: list[int] = []
    S = size - 1
    while S:
        v = int(parent[S])
        order.append(v)
        S ^= 1 << v
    order.reverse()
    return order, float(dp_p[size - 1])


def forward_violations(F: np.ndarray, order: list[int]) -> dict:
    """Prereq-violation metric for a given chapter order.

    violations = total citation mass F[a, b] with a placed BEFORE b (a != b);
    cross = all cross-chapter mass. Returns counts + share."""
    F = np.asarray(F, dtype=np.float64)
    pos = {int(c): i for i, c in enumerate(order)}
    fwd = 0.0
    for a in order:
        for b in order:
            if a != b and pos[a] < pos[b]:
                fwd += F[a, b]
    cross = float(F.sum() - np.trace(F))
    return {"violations": float(fwd), "cross_edges": cross,
            "share": float(fwd / cross) if cross > 0 else 0.0}


# ------------------------------------------------------------- coverage / closure
def nearest_rep_dist(Z: np.ndarray, reps: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Euclidean distance from every row of Z to its nearest representative."""
    R = np.asarray(reps, dtype=np.float32)
    rsq = np.einsum("ij,ij->i", R, R)
    out = np.empty(Z.shape[0], dtype=np.float32)
    for s in range(0, Z.shape[0], chunk):
        q = Z[s:s + chunk].astype(np.float32)
        d2 = np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ R.T) + rsq[None, :]
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def coverage_at_radius(Z: np.ndarray, reps: np.ndarray, radius: float) -> float:
    """Metric 1: share of corpus rows within `radius` of the nearest representative."""
    return float((nearest_rep_dist(Z, reps) <= radius).mean())


def region_closure(Z: np.ndarray, regions: list[np.ndarray], reps: np.ndarray,
                   radius: float) -> list[bool]:
    """Metric 2, FIRST (degenerate) form — kept for the record: per region,
    closed iff MEDIAN member distance-to-nearest-rep <= radius. Computed
    2026-06-12 and found non-discriminating (the t3 regions sit inside the
    scope radius of BOTH ToCs by construction); superseded by
    region_closure_fullstat below (worklog revision entry)."""
    d = nearest_rep_dist(Z, reps)
    return [bool(np.median(d[np.asarray(m, dtype=np.int64)]) <= radius) for m in regions]


def gap_bar(full_scores: np.ndarray, top_frac: float = 0.05, min_size: int = 3) -> float:
    """The t3 selection level: the minimum full-statistic score inside the
    top-`top_frac` level set (the score at the boundary of gaphood)."""
    full_scores = np.asarray(full_scores, dtype=np.float64)
    order = np.argsort(full_scores)[::-1]
    m = max(int(np.ceil(top_frac * full_scores.size)), min_size)
    return float(full_scores[order[m - 1]])


def region_closure_fullstat(full_T: np.ndarray, regions_local: list[np.ndarray],
                            bar: float) -> list[bool]:
    """Metric 2, REVISED rule (pre-committed in the worklog before computing):
    region CLOSED by ToC T iff the mean of T's full statistic over the region's
    members falls below the gap bar — i.e. the region would no longer qualify
    as a gap if the t3 audit were re-run against T's representatives. The
    incumbent ToC (whose audit defined the regions) closes 0 by construction."""
    full_T = np.asarray(full_T, dtype=np.float64)
    return [bool(np.mean(full_T[np.asarray(m, dtype=np.int64)]) < bar)
            for m in regions_local]
