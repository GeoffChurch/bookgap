"""Synthetic ground-truth gap planting (plant-and-recover design).

Three gap species, each planted by REMOVING a known foreground subset:
  - radial:   all fg docs within radius r of a random fg-dense point (30-150 docs),
              verified to leave bg mass in the region (else resample);
  - concept:  all fg docs containing a chosen frequent keyword pair/triple
              (fg support 30-150, scoped-bg support >= threshold);
  - corridor: fg docs lying in the equidistance band between an adjacent KMeans
              cluster pair (dA ~ dB, i.e. distance-ratio near 1), verified for
              fg and bg mass.

High-dimensional honesty notes (measured on the 128-d LSA cloud, see report):
  * Ball mass concentrates at the boundary, where removal does not disturb fg
    neighborhoods. Ground-truth membership for radial/corridor gaps is therefore
    the ORPHANED CORE: bg points whose k-th-fg-neighbor distance grew by >= a
    factor (default 1.15 radial / 1.10 corridor) due to the removal.
  * Absolute detour cylinders ("corridors") are empty in 128-d: interpenetrating
    clusters put every point at detour >= 3. The corridor is instead the
    equidistance band |t - 0.5| <= w (t = projection onto the centroid axis;
    t = 0.5 is exactly the dA = dB bisector), w chosen adaptively so the removal
    lands in the 30-150 budget, plus a locality lens (dA+dB quantile).
  * Matched controls are HYPOTHETICAL-REMOVAL cores: the same selection function
    applied at an untreated location (what would have been orphaned had we
    removed there), giving causally matched negatives for AUC.

All randomness goes through the caller's np.random.Generator.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from artifacts_scripts.gap_detection.detectors import knn_radii

FG_REMOVE_RANGE = (30, 150)
MIN_BG_IN_REGION = 15
MIN_CORE = 8
N_CONTROLS = 5
RADIAL_CORE_RATIO = 1.15
# Corridor removals barely dent kNN densities in 128-d (flanking clusters backfill
# the band): measured orphan-ratio q90 <= 1.10 even at the maximum removal budget.
# The threshold is therefore lower; the weakness itself is a reported finding.
CORRIDOR_CORE_RATIO = 1.05


@dataclass
class PlantedGap:
    kind: str                                  # radial | concept | corridor
    removed_fg: np.ndarray                     # fg-local indices removed
    region_bg: np.ndarray                      # bg-local indices "inside" the gap
    controls_bg: list[np.ndarray] = field(default_factory=list)  # matched control regions
    center: np.ndarray | None = None
    radius: float | None = None
    keywords: tuple[str, ...] | None = None
    meta: dict = field(default_factory=dict)


def orphaned_core(
    Q: np.ndarray, fg_before: np.ndarray, fg_after: np.ndarray,
    k: int = 10, ratio: float = RADIAL_CORE_RATIO,
) -> np.ndarray:
    """Indices (into Q) of points whose k-th fg-neighbor distance grew by >= ratio."""
    if Q.shape[0] == 0:
        return np.array([], dtype=np.int64)
    r_before = knn_radii(Q, fg_before, k)
    r_after = knn_radii(Q, fg_after, k)
    return np.where(r_after / r_before >= ratio)[0]


# ------------------------------------------------------------------ radial
def _dense_fg_pool(Z_fg: np.ndarray, k: int = 10) -> np.ndarray:
    """fg points with above-median local fg density (small r_F,k)."""
    r = knn_radii(Z_fg, Z_fg, k, ref_is_query=True)
    return np.where(r <= np.median(r))[0]


def plant_radial(
    Z_fg: np.ndarray, Z_bg: np.ndarray, rng: np.random.Generator,
    remove_range: tuple[int, int] = FG_REMOVE_RANGE,
    min_bg: int = MIN_BG_IN_REGION, min_core: int = MIN_CORE,
    core_ratio: float = RADIAL_CORE_RATIO, k: int = 10, max_tries: int = 120,
    dense_pool: np.ndarray | None = None,
) -> PlantedGap:
    """Remove all fg docs within radius r of a random fg-dense point.

    region_bg = the orphaned core (bg in the ball whose fg neighborhood was
    materially disturbed); controls = hypothetical-removal cores elsewhere.
    """
    pool = _dense_fg_pool(Z_fg) if dense_pool is None else dense_pool
    nn_fg = NearestNeighbors(n_neighbors=remove_range[1] + 1, algorithm="brute").fit(Z_fg)
    nn_bg = NearestNeighbors(algorithm="brute").fit(Z_bg)

    def ball(center_idx: int, m: int) -> tuple[np.ndarray, float, np.ndarray]:
        d, idx = nn_fg.kneighbors(Z_fg[center_idx : center_idx + 1])
        r = float(d[0, m])
        removed = idx[0, : m + 1]
        bg_in = nn_bg.radius_neighbors(Z_fg[center_idx : center_idx + 1], radius=r)[1][0]
        return removed, r, bg_in

    for _ in range(max_tries):
        c_idx = int(rng.choice(pool))
        m = int(rng.integers(remove_range[0], remove_range[1] - 1))
        removed, r, bg_in = ball(c_idx, m)
        if bg_in.size < min_bg:
            continue
        fg_after = np.delete(Z_fg, removed, axis=0)
        core_local = orphaned_core(Z_bg[bg_in], Z_fg, fg_after, k=k, ratio=core_ratio)
        core = bg_in[core_local]
        if core.size < min_core:
            continue
        # matched controls: hypothetical-removal cores at other dense centers
        controls = []
        for cc in rng.choice(pool, size=min(N_CONTROLS * 4, pool.size), replace=False):
            if int(cc) == c_idx or len(controls) >= N_CONTROLS:
                continue
            c_removed, _, c_bg = ball(int(cc), m)
            if c_bg.size < min_bg:
                continue
            c_after = np.delete(Z_fg, c_removed, axis=0)  # hypothetical only
            c_core = c_bg[orphaned_core(Z_bg[c_bg], Z_fg, c_after, k=k, ratio=core_ratio)]
            c_core = np.setdiff1d(c_core, bg_in)
            if c_core.size >= min_core:
                controls.append(c_core)
        if len(controls) < 2:
            continue
        return PlantedGap(
            kind="radial", removed_fg=np.sort(removed), region_bg=np.sort(core),
            controls_bg=controls, center=Z_fg[c_idx].copy(), radius=r,
            meta={"n_removed": int(removed.size), "n_bg_in_ball": int(bg_in.size),
                  "n_core": int(core.size)},
        )
    raise RuntimeError("plant_radial: no valid configuration found (bg mass too thin)")


# ------------------------------------------------------------------ concept
def concept_candidates(
    K_fg: sp.csr_matrix, K_bg_scoped: sp.csr_matrix,
    fg_range: tuple[int, int] = FG_REMOVE_RANGE, min_bg_scoped: int = 30,
    max_triples: int = 800, rng: np.random.Generator | None = None,
) -> list[tuple[tuple[int, ...], int, int]]:
    """Keyword pairs/triples with fg support in range and scoped-bg support >= min.

    Returns [(keyword_cols, fg_support, scoped_bg_support)].
    """
    Ff = np.asarray(K_fg.todense(), dtype=bool)
    Fb = np.asarray(K_bg_scoped.todense(), dtype=bool)
    Cf = (Ff.astype(np.float32).T @ Ff.astype(np.float32))
    Cb = (Fb.astype(np.float32).T @ Fb.astype(np.float32))
    iu, ju = np.triu_indices(Ff.shape[1], k=1)
    okf = (Cf[iu, ju] >= fg_range[0]) & (Cf[iu, ju] <= fg_range[1])
    okb = Cb[iu, ju] >= min_bg_scoped
    sel = np.where(okf & okb)[0]
    out: list[tuple[tuple[int, ...], int, int]] = [
        ((int(iu[s]), int(ju[s])), int(Cf[iu[s], ju[s]]), int(Cb[iu[s], ju[s]])) for s in sel
    ]
    # Triples: extend a sample of qualifying-fg pairs (support monotonicity makes any
    # planted triple's sub-pairs at least as frequent, so the miner can reach them).
    pair_pool = np.where((Cf[iu, ju] >= fg_range[0]) & (Cb[iu, ju] >= min_bg_scoped))[0]
    if rng is not None and pair_pool.size:
        pair_pool = rng.permutation(pair_pool)[:max_triples]
    for s in pair_pool[:max_triples]:
        i, j = int(iu[s]), int(ju[s])
        vf = Ff[:, i] & Ff[:, j]
        vb = Fb[:, i] & Fb[:, j]
        nf3 = (vf[:, None] & Ff).sum(axis=0)
        nb3 = (vb[:, None] & Fb).sum(axis=0)
        ok3 = (nf3 >= fg_range[0]) & (nf3 <= fg_range[1]) & (nb3 >= min_bg_scoped)
        for l in np.where(ok3)[0]:
            if l <= j:
                continue
            out.append(((i, j, int(l)), int(nf3[l]), int(nb3[l])))
    return out


def plant_concept(
    K_fg: sp.csr_matrix, K_bg_scoped: sp.csr_matrix, scoped_bg_idx: np.ndarray,
    vocab: np.ndarray, rng: np.random.Generator,
    candidates: list[tuple[tuple[int, ...], int, int]],
    used: set[tuple[int, ...]] | None = None,
) -> PlantedGap:
    """Remove all fg docs containing a sampled frequent keyword combo."""
    used = used or set()
    avail = [c for c in candidates if c[0] not in used]
    if not avail:
        raise RuntimeError("plant_concept: candidate pool exhausted")
    combo, sup_f, sup_b = avail[int(rng.integers(len(avail)))]
    Ff = np.asarray(K_fg[:, list(combo)].todense(), dtype=bool)
    removed = np.where(Ff.all(axis=1))[0]
    Fb = np.asarray(K_bg_scoped[:, list(combo)].todense(), dtype=bool)
    region_local = np.where(Fb.all(axis=1))[0]
    region_bg = scoped_bg_idx[region_local]
    # matched controls: other candidate combos (closest fg-support), not removed
    pool = sorted(
        (c for c in avail if c[0] != combo), key=lambda c: abs(c[1] - sup_f)
    )[: N_CONTROLS * 3]
    rng.shuffle(pool)
    controls = []
    for cc, _, _ in pool:
        Fc = np.asarray(K_bg_scoped[:, list(cc)].todense(), dtype=bool)
        ctrl = scoped_bg_idx[np.where(Fc.all(axis=1))[0]]
        ctrl = np.setdiff1d(ctrl, region_bg)
        if ctrl.size >= 10:
            controls.append(ctrl)
        if len(controls) >= N_CONTROLS:
            break
    return PlantedGap(
        kind="concept", removed_fg=removed, region_bg=region_bg, controls_bg=controls,
        keywords=tuple(str(vocab[c]) for c in combo),
        meta={"combo_cols": tuple(int(c) for c in combo), "fg_support": sup_f,
              "scoped_bg_support": sup_b, "n_removed": int(removed.size)},
    )


# ------------------------------------------------------------------ corridor
def _axis_t(X: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> np.ndarray:
    """Projection onto the centroid axis, scaled so t(cA)=0, t(cB)=1.

    t = 0.5 is exactly the dA = dB perpendicular bisector ("distance-ratio near 1").
    """
    u = cB - cA
    return (X - cA) @ u / max(float(u @ u), 1e-12)


def corridor_pairs(
    Z_fg: np.ndarray, Z_bg: np.ndarray, n_clusters: int = 12, seed: int = 7,
    remove_range: tuple[int, int] = FG_REMOVE_RANGE, min_bg: int = MIN_BG_IN_REGION,
    lens_quantile: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """KMeans the fg; list adjacent cluster pairs with a removable equidistance band.

    For each pair: members = fg of the two clusters; the corridor is the band
    |t - 0.5| <= w with w set so the band holds ~max(30, 10% of members) docs
    (capped at 150); requires bg mass in the band's locality lens.

    Returns (labels, centroids, qualifying_pairs).
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4).fit(Z_fg)
    cents = km.cluster_centers_
    labels = km.labels_
    found: list[dict] = []
    for a in range(n_clusters):
        for b in range(a + 1, n_clusters):
            mem = np.where((labels == a) | (labels == b))[0]
            if mem.size < 2 * remove_range[0]:
                continue
            t_mem = _axis_t(Z_fg[mem], cents[a], cents[b])
            s = np.abs(t_mem - 0.5)
            # plant at the top of the removal budget: thin bands leave no measurable
            # dent in 128-d (neighbors backfill from the flanks)
            m = int(np.clip(mem.size // 2, remove_range[0], remove_range[1]))
            if m >= mem.size:
                continue
            w = float(np.sort(s)[m - 1])
            dA_m = np.linalg.norm(Z_fg[mem] - cents[a], axis=1)
            dB_m = np.linalg.norm(Z_fg[mem] - cents[b], axis=1)
            lens_r = float(np.quantile(dA_m + dB_m, lens_quantile))
            t_bg = _axis_t(Z_bg, cents[a], cents[b])
            dA_b = np.linalg.norm(Z_bg - cents[a], axis=1)
            dB_b = np.linalg.norm(Z_bg - cents[b], axis=1)
            bg_band = (np.abs(t_bg - 0.5) <= w) & (dA_b + dB_b <= lens_r)
            if int(bg_band.sum()) < min_bg:
                continue
            found.append(
                {"a": a, "b": b, "w": w, "m": m, "lens_r": lens_r,
                 "n_members": int(mem.size), "n_bg_band": int(bg_band.sum()),
                 "centroid_dist": float(np.linalg.norm(cents[a] - cents[b]))}
            )
    found.sort(key=lambda p: p["centroid_dist"])  # adjacency first
    return labels, cents, found


def corridor_band_bg(Z_bg: np.ndarray, cents: np.ndarray, pair: dict) -> np.ndarray:
    """bg-local indices inside the pair's equidistance band + locality lens."""
    cA, cB = cents[pair["a"]], cents[pair["b"]]
    t = _axis_t(Z_bg, cA, cB)
    dA = np.linalg.norm(Z_bg - cA, axis=1)
    dB = np.linalg.norm(Z_bg - cB, axis=1)
    return np.where((np.abs(t - 0.5) <= pair["w"]) & (dA + dB <= pair["lens_r"]))[0]


def plant_corridor(
    Z_fg: np.ndarray, Z_bg: np.ndarray, labels: np.ndarray, cents: np.ndarray,
    pair: dict, all_pairs: list[dict], rng: np.random.Generator,
    core_ratio: float = CORRIDOR_CORE_RATIO, min_core: int = MIN_CORE, k: int = 10,
) -> PlantedGap | None:
    """Remove the fg equidistance band of the given cluster pair.

    region_bg = orphaned core of the band bg; controls = hypothetical-removal
    cores of other qualifying pairs. Returns None if the removal leaves no
    detectable core (caller should try another pair).
    """
    a, b = pair["a"], pair["b"]
    cA, cB = cents[a], cents[b]
    mem = np.where((labels == a) | (labels == b))[0]
    t_mem = _axis_t(Z_fg[mem], cA, cB)
    removed = mem[np.abs(t_mem - 0.5) <= pair["w"]]
    band_bg = corridor_band_bg(Z_bg, cents, pair)
    fg_after = np.delete(Z_fg, removed, axis=0)
    core = band_bg[orphaned_core(Z_bg[band_bg], Z_fg, fg_after, k=k, ratio=core_ratio)]
    if core.size < min_core:
        return None
    controls = []
    others = [p for p in all_pairs if (p["a"], p["b"]) != (a, b)]
    rng.shuffle(others)
    for p in others:
        p_mem = np.where((labels == p["a"]) | (labels == p["b"]))[0]
        p_t = _axis_t(Z_fg[p_mem], cents[p["a"]], cents[p["b"]])
        p_removed = p_mem[np.abs(p_t - 0.5) <= p["w"]]
        p_band = corridor_band_bg(Z_bg, cents, p)
        p_after = np.delete(Z_fg, p_removed, axis=0)  # hypothetical only
        p_core = p_band[orphaned_core(Z_bg[p_band], Z_fg, p_after, k=k, ratio=core_ratio)]
        p_core = np.setdiff1d(p_core, band_bg)
        if p_core.size >= min_core:
            controls.append(p_core)
        if len(controls) >= N_CONTROLS:
            break
    if len(controls) < 2:
        return None
    return PlantedGap(
        kind="corridor", removed_fg=np.sort(removed), region_bg=np.sort(core),
        controls_bg=controls, center=((cA + cB) / 2.0),
        radius=float(np.linalg.norm(cA - cB)) / 2.0,
        meta={**pair, "n_removed": int(removed.size), "n_bg_band": int(band_bg.size),
              "n_core": int(core.size)},
    )
