"""Plant-and-recover probe for OSM POI completeness gaps.

FAST EXPLORATORY SPIKE — not production code.

Idea: a "completeness gap" is a feature the surrounding CONTEXT implies should
exist but is not mapped. Objective test = PLANT-AND-RECOVER: in a well-mapped
area, hold out (delete) a fraction of real POIs, fit a context-only expectation
model on the rest, then test whether the model ranks the held-out POI locations
above random empty locations (ranking AUC).

Context features for a location are built from a radius around it, EXCLUDING the
location's own immediate cell (to avoid trivially leaking the held-out label):
  - building density (count within radius)
  - road density (count within radius)
  - nearby OTHER-POI count within radius
  - nearby OTHER-POI count within a smaller inner radius (local clustering)
  - category diversity of nearby POIs

Two models are compared:
  - BASELINE: building-density only (the confound — "POIs cluster where buildings
    are"). If the rich model can't beat this, recovering specific omissions is weak.
  - CONTEXT: all features above.

We also report a POI-aware variant restricted to "destination" POIs (shops,
food, services) — excluding street furniture (bench, waste_basket, parking,
bicycle_parking, vending_machine) which is mechanically tied to roads/buildings
and inflates the trivial signal.

Regenerate:
    uv run \
        python -m artifacts_scripts.osm_spike.plant_recover
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from artifacts_scripts.osm_spike.fetch_osm import (
    BBOX_HEIDELBERG,
    fetch,
    parse_elements,
)

# POI categories that are "street furniture" — mechanically co-located with
# roads/buildings, so they inflate the trivial building-density signal.
FURNITURE = {
    "bench",
    "waste_basket",
    "bicycle_parking",
    "parking",
    "vending_machine",
    "toilets",
    "fountain",
    "drinking_water",
    "recycling",
    "post_box",
    "telephone",
    "parking_space",
    "motorcycle_parking",
    "charging_station",
    "bicycle_repair_station",
    "clock",
    "grit_bin",
    "hunting_stand",
}


def meters_per_deg(lat: float) -> tuple[float, float]:
    """Approx meters per degree of (lat, lon) at a given latitude."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return m_per_deg_lat, m_per_deg_lon


def to_xy(lats: np.ndarray, lons: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Local equirectangular projection to meters relative to (lat0, lon0)."""
    m_lat, m_lon = meters_per_deg(lat0)
    x = (lons - lon0) * m_lon
    y = (lats - lat0) * m_lat
    return np.column_stack([x, y])


def radius_counts(query_xy: np.ndarray, source_xy: np.ndarray, radius: float) -> np.ndarray:
    """Count source points within `radius` meters of each query point.

    Vectorized but memory-aware: chunk the queries. Returns int array len(query).
    """
    if len(source_xy) == 0:
        return np.zeros(len(query_xy), dtype=np.int32)
    out = np.empty(len(query_xy), dtype=np.int32)
    r2 = radius * radius
    chunk = 2000
    for i in range(0, len(query_xy), chunk):
        q = query_xy[i : i + chunk]
        # (chunk, n_source) squared distances
        d2 = (
            (q[:, 0:1] - source_xy[None, :, 0]) ** 2
            + (q[:, 1:2] - source_xy[None, :, 1]) ** 2
        )
        out[i : i + chunk] = (d2 <= r2).sum(axis=1)
    return out


def category_diversity(
    query_xy: np.ndarray,
    poi_xy: np.ndarray,
    poi_cat_idx: np.ndarray,
    n_cats: int,
    radius: float,
) -> np.ndarray:
    """Number of distinct POI categories within radius of each query point."""
    if len(poi_xy) == 0:
        return np.zeros(len(query_xy), dtype=np.int32)
    out = np.empty(len(query_xy), dtype=np.int32)
    r2 = radius * radius
    chunk = 1000
    for i in range(0, len(query_xy), chunk):
        q = query_xy[i : i + chunk]
        d2 = (
            (q[:, 0:1] - poi_xy[None, :, 0]) ** 2
            + (q[:, 1:2] - poi_xy[None, :, 1]) ** 2
        )
        within = d2 <= r2  # (chunk, n_poi)
        for j in range(q.shape[0]):
            cats = poi_cat_idx[within[j]]
            out[i + j] = len(np.unique(cats)) if cats.size else 0
    return out


@dataclass
class Features:
    """Feature matrix + names for a set of query locations."""

    X: np.ndarray
    names: list[str]


def build_features(
    query_xy: np.ndarray,
    building_xy: np.ndarray,
    road_xy: np.ndarray,
    fit_poi_xy: np.ndarray,
    fit_poi_cat_idx: np.ndarray,
    n_cats: int,
    *,
    inner_radius: float = 120.0,
    outer_radius: float = 300.0,
    self_exclude: float = 35.0,
) -> Features:
    """Context features for query locations.

    `fit_poi_xy` are the POIs VISIBLE to the model (the kept set after plant).
    Nearby-POI counts subtract any fit-POIs within `self_exclude` meters so a
    query sitting on top of a (kept) POI doesn't trivially see itself; held-out
    POIs are not in fit_poi_xy at all, so they get the genuine context-only read.
    """
    bld_out = radius_counts(query_xy, building_xy, outer_radius)
    bld_in = radius_counts(query_xy, building_xy, inner_radius)
    road_out = radius_counts(query_xy, road_xy, outer_radius)
    road_in = radius_counts(query_xy, road_xy, inner_radius)

    poi_out_raw = radius_counts(query_xy, fit_poi_xy, outer_radius)
    poi_in_raw = radius_counts(query_xy, fit_poi_xy, inner_radius)
    poi_self = radius_counts(query_xy, fit_poi_xy, self_exclude)
    poi_out = np.maximum(poi_out_raw - poi_self, 0)
    poi_in = np.maximum(poi_in_raw - poi_self, 0)

    diversity = category_diversity(query_xy, fit_poi_xy, fit_poi_cat_idx, n_cats, outer_radius)

    X = np.column_stack(
        [bld_out, bld_in, road_out, road_in, poi_out, poi_in, diversity]
    ).astype(np.float64)
    names = [
        "bld_outer",
        "bld_inner",
        "road_outer",
        "road_inner",
        "poi_outer",
        "poi_inner",
        "cat_diversity",
    ]
    return Features(X, names)


def sample_empty_locations(
    n: int,
    poi_xy: np.ndarray,
    building_xy: np.ndarray,
    bounds_xy: tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    min_dist_to_poi: float = 40.0,
    require_near_building: float = 80.0,
) -> np.ndarray:
    """Sample `n` random 'empty' locations: no POI within min_dist_to_poi.

    To make the negative set a FAIR test (not trivially separable on building
    density), we require empties to be near a building (within
    `require_near_building` m). Otherwise empties would sit in fields/water and
    the model would 'win' purely by building presence — the confound we want to
    control. This makes recovery genuinely about specific omissions in built-up
    space.
    """
    xmin, ymin, xmax, ymax = bounds_xy
    out = []
    tries = 0
    max_tries = n * 200
    while len(out) < n and tries < max_tries:
        tries += 1
        cand = np.array(
            [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]
        ).reshape(1, 2)
        # must be near a building (built-up space)
        if radius_counts(cand, building_xy, require_near_building)[0] < 1:
            continue
        # must be empty of POIs
        if radius_counts(cand, poi_xy, min_dist_to_poi)[0] > 0:
            continue
        out.append(cand[0])
    if len(out) < n:
        print(
            f"[warn] only sampled {len(out)}/{n} empties in {tries} tries",
            file=sys.stderr,
        )
    return np.array(out)


def run_seed(
    parsed: dict,
    seed: int,
    *,
    holdout_frac: float = 0.30,
    destinations_only: bool = False,
) -> dict:
    """One plant-and-recover trial. Returns dict of AUCs."""
    rng = np.random.default_rng(seed)

    pois = parsed["pois"]
    if destinations_only:
        pois = [p for p in pois if p["category"] not in FURNITURE]

    buildings = parsed["buildings"]
    roads = parsed["roads"]

    lat0 = float(np.mean([p["lat"] for p in pois]))
    lon0 = float(np.mean([p["lon"] for p in pois]))

    poi_xy = to_xy(
        np.array([p["lat"] for p in pois]),
        np.array([p["lon"] for p in pois]),
        lat0,
        lon0,
    )
    bld_xy = to_xy(
        np.array([b["lat"] for b in buildings]),
        np.array([b["lon"] for b in buildings]),
        lat0,
        lon0,
    )
    road_xy = to_xy(
        np.array([r["lat"] for r in roads]),
        np.array([r["lon"] for r in roads]),
        lat0,
        lon0,
    )

    # category index
    cats = sorted({p["category"] for p in pois})
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    poi_cat_idx = np.array([cat_to_idx[p["category"]] for p in pois])

    n = len(pois)
    perm = rng.permutation(n)
    n_hold = int(round(holdout_frac * n))
    hold_idx = perm[:n_hold]
    keep_idx = perm[n_hold:]

    keep_xy = poi_xy[keep_idx]
    keep_cat = poi_cat_idx[keep_idx]
    hold_xy = poi_xy[hold_idx]

    bounds = (
        float(poi_xy[:, 0].min()),
        float(poi_xy[:, 1].min()),
        float(poi_xy[:, 0].max()),
        float(poi_xy[:, 1].max()),
    )

    # negatives: empty, near-building locations (fair test on built-up space)
    empty_xy = sample_empty_locations(
        n_hold, poi_xy, bld_xy, bounds, rng
    )
    n_neg = len(empty_xy)
    hold_xy = hold_xy[:n_neg] if n_neg < n_hold else hold_xy  # balance if short

    # ----- TRAINING SET -----
    # Positives for training: the KEPT POIs (context-only features, self-excluded).
    # Negatives for training: a fresh set of empty near-building locations.
    train_neg_xy = sample_empty_locations(
        len(keep_xy), poi_xy, bld_xy, bounds, rng
    )

    feats_keep = build_features(
        keep_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats)
    )
    feats_train_neg = build_features(
        train_neg_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats)
    )

    X_train = np.vstack([feats_keep.X, feats_train_neg.X])
    y_train = np.concatenate(
        [np.ones(len(feats_keep.X)), np.zeros(len(feats_train_neg.X))]
    )
    names = feats_keep.names

    # ----- TEST SET (the plant-and-recover evaluation) -----
    feats_hold = build_features(
        hold_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats)
    )
    feats_eval_neg = build_features(
        empty_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats)
    )
    X_test = np.vstack([feats_hold.X, feats_eval_neg.X])
    y_test = np.concatenate(
        [np.ones(len(feats_hold.X)), np.zeros(len(feats_eval_neg.X))]
    )

    results: dict = {
        "seed": seed,
        "n_pois": n,
        "n_holdout": len(hold_xy),
        "n_empty": n_neg,
        "destinations_only": destinations_only,
    }

    def _auc(cols: list[int]) -> float:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train[:, cols])
        Xte = sc.transform(X_test[:, cols])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y_train)
        prob = clf.predict_proba(Xte)[:, 1]
        return float(roc_auc_score(y_test, prob))

    # BASELINE: building density only
    bld_cols = [names.index("bld_outer"), names.index("bld_inner")]
    results["auc_baseline_building"] = _auc(bld_cols)

    # CONTEXT: all features
    results["auc_context_full"] = _auc(list(range(len(names))))

    # CONTEXT minus buildings (does non-building context carry signal?)
    nonbld_cols = [
        i for i, nm in enumerate(names) if not nm.startswith("bld_")
    ]
    results["auc_context_nonbuilding"] = _auc(nonbld_cols)

    # store coefs of full model for interpretability
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    results["coefs"] = dict(zip(names, [round(float(c), 3) for c in clf.coef_[0]]))

    return results


def main() -> None:
    data = fetch(BBOX_HEIDELBERG)
    parsed = parse_elements(data)
    print(
        f"\nbbox {BBOX_HEIDELBERG}: {len(parsed['pois'])} POIs, "
        f"{len(parsed['buildings'])} buildings, {len(parsed['roads'])} roads\n"
    )

    for label, dest_only in [("ALL POIs", False), ("DESTINATIONS only", True)]:
        print(f"===== {label} =====")
        rows = [run_seed(parsed, seed, destinations_only=dest_only) for seed in (0, 1, 2)]
        for r in rows:
            print(
                f"  seed {r['seed']}: n_poi={r['n_pois']} n_hold={r['n_holdout']} "
                f"n_empty={r['n_empty']} | "
                f"baseline(bld)={r['auc_baseline_building']:.3f} "
                f"context(full)={r['auc_context_full']:.3f} "
                f"context(non-bld)={r['auc_context_nonbuilding']:.3f}"
            )
        base = np.mean([r["auc_baseline_building"] for r in rows])
        ctx = np.mean([r["auc_context_full"] for r in rows])
        nonb = np.mean([r["auc_context_nonbuilding"] for r in rows])
        print(
            f"  MEAN: baseline(bld)={base:.3f}  context(full)={ctx:.3f}  "
            f"context(non-bld)={nonb:.3f}  | lift(full-base)={ctx - base:+.3f}"
        )
        print(f"  full-model coefs (last seed): {rows[-1]['coefs']}\n")


if __name__ == "__main__":
    main()
