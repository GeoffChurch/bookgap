"""Track 2 Phase 3: outcome machinery — fill counts, growth-matched null
locations, fill-rate ratios with bootstrap CIs, precision@K vs realized-change.

Contracts (pinned by toy fixtures in test_gap_detection.py):
  * matched_controls: per flagged location, M controls from the candidate pool
    that are (a) outside EVERY flagged neighborhood (and not flagged docs),
    (b) spatially disjoint from their own flagged location's neighborhood
    (||x_c - x_f|| > r_f + r_c), (c) drawn from the same joint covariate
    stratum (quantile bins over the pool) when it holds >= M eligible
    candidates, else nearest eligible candidates in standardized covariate
    space (relaxations counted).
  * fill_rate_ratio: ratio of means with a location-level percentile bootstrap.
  * locations may be pool docs OR synthetic points (pattern centroids): all
    location-side quantities (radius, covariates) are computed by the same
    neighborhood rule, so both kinds are handled uniformly.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

from artifacts_scripts.gap_detection.detectors import count_within_radius

EPS = 1e-12


def knn_scale(loc_X: np.ndarray, ref_X: np.ndarray, k: int = 10,
              self_rows: np.ndarray | None = None) -> np.ndarray:
    """Neighborhood radius per location: distance to its k-th nearest reference
    point at T. `self_rows` marks locations that ARE reference points (their own
    row is excluded by taking the (k+1)-th neighbor)."""
    from sklearn.neighbors import NearestNeighbors

    kk = min(k + 1, ref_X.shape[0])
    nn = NearestNeighbors(n_neighbors=kk, algorithm="brute").fit(ref_X)
    d, _ = nn.kneighbors(loc_X)
    r = d[:, kk - 1].copy()                      # (k+1)-th for self rows
    if self_rows is None:
        self_rows = np.zeros(loc_X.shape[0], dtype=bool)
    r[~self_rows] = d[~self_rows, kk - 2]        # k-th for non-reference points
    return np.maximum(r, EPS)


def fill_counts(loc_X: np.ndarray, radii: np.ndarray, arr_X: np.ndarray) -> np.ndarray:
    """# arrivals within each location's radius."""
    if arr_X.shape[0] == 0:
        return np.zeros(loc_X.shape[0], dtype=np.int64)
    return count_within_radius(loc_X, arr_X, radii)


def quantile_bins(pool_col: np.ndarray, n_bins: int) -> np.ndarray:
    return np.quantile(pool_col, np.linspace(0, 1, n_bins + 1)[1:-1])


def covariate_strata(C: np.ndarray, n_bins: int = 3,
                     edges: list[np.ndarray] | None = None) -> np.ndarray:
    """Joint stratum id from per-covariate quantile bins (terciles by default).

    `edges` (per-column bin edges) lets locations be digitized against POOL
    quantiles rather than their own."""
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    n, m = C.shape
    out = np.zeros(n, dtype=np.int64)
    for j in range(m):
        e = quantile_bins(C[:, j], n_bins) if edges is None else edges[j]
        out = out * n_bins + np.searchsorted(e, C[:, j], side="right")
    return out


DEFAULT_CALIPERS = (0.2, 0.5, 0.5)  # sigma units: density, dist-to-fg, growth


def matched_controls(
    loc_X: np.ndarray,
    loc_radii: np.ndarray,
    loc_C: np.ndarray,
    pool_X: np.ndarray,
    pool_radii: np.ndarray,
    pool_C: np.ndarray,
    excluded: np.ndarray,
    m_controls: int = 20,
    calipers: tuple[float, ...] = DEFAULT_CALIPERS,
    max_relax: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Growth/density-matched null locations from the candidate pool.

    loc_*: the detector's K flagged locations (points, radii, raw covariates).
    pool_*: the scoped-bg candidate pool (scope membership by construction).
    excluded: pool mask — flagged docs AND any pool doc inside a flagged
    neighborhood (never eligible as controls).

    Matching: hard per-covariate calipers (sigma units, progressively doubled
    up to `max_relax` levels when a location's covariate cell is too thin, then
    dropped entirely), nearest standardized-Euclidean among survivors. The
    caliper widths were CALIBRATED against the random-location machinery null
    only (ratio must be ~1 on random flags), never against detector outcomes.

    Returns (controls (K, M) pool indices, info with per-location relax levels).
    """
    rng = np.random.default_rng(seed)
    loc_X = np.asarray(loc_X, dtype=np.float64)
    pool_X = np.asarray(pool_X, dtype=np.float64)
    mu, sd = pool_C.mean(axis=0), pool_C.std(axis=0) + EPS
    Cz_pool = (pool_C - mu) / sd
    Cz_loc = (np.atleast_2d(loc_C) - mu) / sd
    cal = np.asarray(calipers, dtype=np.float64)
    k = loc_X.shape[0]
    controls = np.full((k, m_controls), -1, dtype=np.int64)
    relax_levels = np.zeros(k, dtype=np.int64)
    for i in range(k):
        d_sp = np.linalg.norm(pool_X - loc_X[i], axis=1)
        # "neither center inside the other's neighborhood": the full-disjointness
        # rule d > r_loc + r_c annihilates the sparse tail on the unit sphere
        # (r_loc + r_c ~ the diameter), which forced sparse locations onto dense
        # controls and biased the random-location machinery null to ~1.2-1.4.
        disjoint = d_sp > np.maximum(loc_radii[i], pool_radii)
        elig = ~excluded & disjoint
        diff = np.abs(Cz_pool - Cz_loc[i])
        level = 0
        while True:
            if level >= max_relax:
                cand = elig
                break
            cand = elig & (diff <= cal * (2.0**level)).all(axis=1)
            if int(cand.sum()) >= m_controls:
                break
            level += 1
        relax_levels[i] = level
        pool_idx = np.where(cand)[0]
        if pool_idx.size < m_controls:
            raise ValueError("matched_controls: not enough eligible candidates")
        cd = np.linalg.norm(Cz_pool - Cz_loc[i], axis=1)
        # nearest in covariate space, seeded jitter to break exact ties
        order = pool_idx[np.argsort(cd[pool_idx] + 1e-12 * rng.random(pool_idx.size),
                                    kind="stable")]
        controls[i] = order[:m_controls]
    info = {
        "n_locations": int(k), "m_controls": int(m_controls),
        "calipers_sigma": [float(c) for c in cal],
        "n_relaxed": int((relax_levels > 0).sum()),
        "relax_level_median": float(np.median(relax_levels)),
        "relax_level_max": int(relax_levels.max()),
        "covariate_dims": int(pool_C.shape[1]),
    }
    return controls, info


def greedy_location_groups(loc_X: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Group overlapping locations: location i joins the first kept location j
    with ||x_i - x_j|| <= min(r_i, r_j); otherwise it starts a new group.

    Detectors can emit many near-duplicate locations (e.g. pattern centroids of
    overlapping itemsets); treating them as independent makes the bootstrap CI
    overconfident. Groups are the resampling unit for the clustered CI."""
    k = loc_X.shape[0]
    groups = np.full(k, -1, dtype=np.int64)
    reps: list[int] = []
    for i in range(k):
        for g, j in enumerate(reps):
            if np.linalg.norm(loc_X[i] - loc_X[j]) <= min(radii[i], radii[j]):
                groups[i] = g
                break
        if groups[i] < 0:
            groups[i] = len(reps)
            reps.append(i)
    return groups


def fill_rate_ratio(
    fill_flagged: np.ndarray, fill_controls: np.ndarray,
    n_boot: int = 2000, seed: int = 0, groups: np.ndarray | None = None,
) -> dict:
    """Ratio of mean fills (flagged / matched controls) + percentile bootstrap CI
    over locations (each location resampled WITH its control set). If `groups`
    is given, an additional CLUSTERED bootstrap resamples whole location groups
    (near-duplicate locations are not independent evidence)."""
    rng = np.random.default_rng(seed)
    f = np.asarray(fill_flagged, dtype=np.float64)
    c = np.asarray(fill_controls, dtype=np.float64)
    k = f.size

    def ratio(idx: np.ndarray) -> float:
        return float(f[idx].mean() / max(c[idx].mean(), EPS))

    point = ratio(np.arange(k))
    boots = np.array([ratio(rng.integers(0, k, size=k)) for _ in range(n_boot)])
    out = {
        "ratio": round(point, 4),
        "ci_lo": round(float(np.quantile(boots, 0.025)), 4),
        "ci_hi": round(float(np.quantile(boots, 0.975)), 4),
        "mean_fill_flagged": round(float(f.mean()), 3),
        "mean_fill_controls": round(float(c.mean()), 3),
        "n_locations": int(k),
        "n_boot": int(n_boot),
    }
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        cboots = []
        for _ in range(n_boot):
            gs = rng.choice(uniq, size=uniq.size, replace=True)
            idx = np.concatenate([members[g] for g in gs])
            cboots.append(ratio(idx))
        cboots = np.asarray(cboots)
        out.update({
            "n_groups": int(uniq.size),
            "ci_lo_clustered": round(float(np.quantile(cboots, 0.025)), 4),
            "ci_hi_clustered": round(float(np.quantile(cboots, 0.975)), 4),
        })
    return out


def precision_vs_set(
    loc_X: np.ndarray, radii: np.ndarray, target_X: np.ndarray,
) -> float:
    """Fraction of locations whose r-neighborhood contains >= 1 target point
    (the uniform hit rule across point- and centroid-valued locations)."""
    if target_X.shape[0] == 0:
        return 0.0
    hits = count_within_radius(loc_X, target_X, radii) >= 1
    return float(hits.mean())


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
