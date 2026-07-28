"""Qualitative inspection of plant-and-recover: concrete recovered/missed cases,
a spatial-null sanity check, and (optionally) an undermapped-area look.

FAST EXPLORATORY SPIKE — not production code.

Regenerate:
    uv run \
        python -m artifacts_scripts.osm_spike.inspect_cases
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from artifacts_scripts.osm_spike.fetch_osm import (
    BBOX_HEIDELBERG,
    fetch,
    parse_elements,
)
from artifacts_scripts.osm_spike.plant_recover import (
    FURNITURE,
    build_features,
    sample_empty_locations,
    to_xy,
)


def _prep(parsed, destinations_only=True, seed=0):
    rng = np.random.default_rng(seed)
    pois = parsed["pois"]
    if destinations_only:
        pois = [p for p in pois if p["category"] not in FURNITURE]
    buildings, roads = parsed["buildings"], parsed["roads"]
    lat0 = float(np.mean([p["lat"] for p in pois]))
    lon0 = float(np.mean([p["lon"] for p in pois]))
    poi_xy = to_xy(np.array([p["lat"] for p in pois]), np.array([p["lon"] for p in pois]), lat0, lon0)
    bld_xy = to_xy(np.array([b["lat"] for b in buildings]), np.array([b["lon"] for b in buildings]), lat0, lon0)
    road_xy = to_xy(np.array([r["lat"] for r in roads]), np.array([r["lon"] for r in roads]), lat0, lon0)
    cats = sorted({p["category"] for p in pois})
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    poi_cat_idx = np.array([cat_to_idx[p["category"]] for p in pois])
    return rng, pois, poi_xy, bld_xy, road_xy, poi_cat_idx, cats, lat0, lon0


def spatial_null(parsed, seed=0):
    """Sanity check: replace held-out POIs with the SAME NUMBER of random
    near-building empties as 'fake positives'. A valid harness should drop to
    AUC~0.5 (the model can't recover locations that aren't really POIs)."""
    rng, pois, poi_xy, bld_xy, road_xy, poi_cat_idx, cats, *_ = _prep(parsed, True, seed)
    n = len(pois)
    perm = rng.permutation(n)
    n_hold = int(round(0.30 * n))
    keep_idx = perm[n_hold:]
    keep_xy, keep_cat = poi_xy[keep_idx], poi_cat_idx[keep_idx]
    bounds = (poi_xy[:, 0].min(), poi_xy[:, 1].min(), poi_xy[:, 0].max(), poi_xy[:, 1].max())

    train_neg = sample_empty_locations(len(keep_xy), poi_xy, bld_xy, bounds, rng)
    fk = build_features(keep_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    fn = build_features(train_neg, bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    Xtr = np.vstack([fk.X, fn.X]); ytr = np.concatenate([np.ones(len(fk.X)), np.zeros(len(fn.X))])

    # FAKE positives: random near-building empties (NOT real POIs)
    fake_pos = sample_empty_locations(n_hold, poi_xy, bld_xy, bounds, rng)
    eval_neg = sample_empty_locations(n_hold, poi_xy, bld_xy, bounds, rng)
    m = min(len(fake_pos), len(eval_neg))
    ff = build_features(fake_pos[:m], bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    fe = build_features(eval_neg[:m], bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    Xte = np.vstack([ff.X, fe.X]); yte = np.concatenate([np.ones(m), np.zeros(m)])

    sc = StandardScaler(); clf = LogisticRegression(max_iter=1000)
    clf.fit(sc.fit_transform(Xtr), ytr)
    prob = clf.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, prob))


def concrete_cases(parsed, seed=0, k=4):
    """Train on kept POIs, score held-out vs empty, list the best-recovered
    held-out POIs (high score) and worst-missed (low score), with tags."""
    rng, pois, poi_xy, bld_xy, road_xy, poi_cat_idx, cats, lat0, lon0 = _prep(parsed, True, seed)
    n = len(pois)
    perm = rng.permutation(n)
    n_hold = int(round(0.30 * n))
    hold_idx, keep_idx = perm[:n_hold], perm[n_hold:]
    keep_xy, keep_cat = poi_xy[keep_idx], poi_cat_idx[keep_idx]
    hold_xy = poi_xy[hold_idx]
    bounds = (poi_xy[:, 0].min(), poi_xy[:, 1].min(), poi_xy[:, 0].max(), poi_xy[:, 1].max())

    train_neg = sample_empty_locations(len(keep_xy), poi_xy, bld_xy, bounds, rng)
    fk = build_features(keep_xy, bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    fn = build_features(train_neg, bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    Xtr = np.vstack([fk.X, fn.X]); ytr = np.concatenate([np.ones(len(fk.X)), np.zeros(len(fn.X))])

    empty_xy = sample_empty_locations(n_hold, poi_xy, bld_xy, bounds, rng)
    m = min(len(empty_xy), len(hold_xy))
    fh = build_features(hold_xy[:m], bld_xy, road_xy, keep_xy, keep_cat, len(cats))
    fe = build_features(empty_xy[:m], bld_xy, road_xy, keep_xy, keep_cat, len(cats))

    sc = StandardScaler(); clf = LogisticRegression(max_iter=1000)
    clf.fit(sc.fit_transform(Xtr), ytr)
    hold_prob = clf.predict_proba(sc.transform(fh.X))[:, 1]
    empty_prob = clf.predict_proba(sc.transform(fe.X))[:, 1]

    auc = roc_auc_score(
        np.concatenate([np.ones(m), np.zeros(m)]),
        np.concatenate([hold_prob, empty_prob]),
    )

    hidx_sorted = np.argsort(-hold_prob)  # high score first
    pois_hold = [pois[i] for i in hold_idx[:m]]

    def describe(local_i):
        p = pois_hold[local_i]
        nm = p["tags"].get("name", "(unnamed)")
        return (
            f"score={hold_prob[local_i]:.3f}  {p['category']:>14}  "
            f"'{nm}'  @({p['lat']:.5f},{p['lon']:.5f})  "
            f"https://www.openstreetmap.org/{p['type']}/{p['id']}"
        )

    print(f"  (seed {seed}, destinations-only, test AUC={auc:.3f}, empty median={np.median(empty_prob):.3f})")
    print(f"  --- TOP {k} RECOVERED held-out POIs (context says: clearly should be here) ---")
    for li in hidx_sorted[:k]:
        print("   ", describe(li))
    print(f"  --- BOTTOM {k} MISSED held-out POIs (context gave low expectation) ---")
    for li in hidx_sorted[-k:]:
        print("   ", describe(li))


# A known UNDER-mapped area for the qualitative secondary check.
# Central Heidelberg is OSM gold-standard; for contrast we want a place where
# the built environment is dense but POI mapping is sparse. Peripheral / non-
# European city centers are typically POI-thin relative to their building stock.
# Default: a ~2 km^2 bbox over central Tirana, Albania — dense urban fabric,
# historically thin OSM POI coverage. (Verify/adjust coords if empty.)
BBOX_UNDERMAPPED = (41.320, 19.815, 41.335, 19.835)  # central Tirana, AL


def coverage_summary(parsed, label):
    n_poi = len(parsed["pois"])
    n_dest = sum(1 for p in parsed["pois"] if p["category"] not in FURNITURE)
    n_bld = len(parsed["buildings"])
    n_road = len(parsed["roads"])
    poi_per_bld = n_poi / max(n_bld, 1)
    dest_per_bld = n_dest / max(n_bld, 1)
    print(
        f"  {label:>26}: {n_poi:5d} POIs ({n_dest} dest) | {n_bld:5d} bld | "
        f"{n_road:5d} road | POI/bld={poi_per_bld:.3f} dest/bld={dest_per_bld:.3f}"
    )
    return dict(n_poi=n_poi, n_dest=n_dest, n_bld=n_bld, n_road=n_road,
               poi_per_bld=poi_per_bld, dest_per_bld=dest_per_bld)


def main():
    data = fetch(BBOX_HEIDELBERG)
    parsed = parse_elements(data)
    print(f"\nbbox {BBOX_HEIDELBERG}: {len(parsed['pois'])} POIs (destinations: "
          f"{sum(1 for p in parsed['pois'] if p['category'] not in FURNITURE)})\n")

    print("===== SPATIAL NULL (fake positives = random empties; should be ~0.5) =====")
    nulls = [spatial_null(parsed, s) for s in (0, 1, 2)]
    print(f"  null AUCs: {[round(x,3) for x in nulls]}  mean={np.mean(nulls):.3f}\n")

    print("===== CONCRETE CASES =====")
    concrete_cases(parsed, seed=0, k=4)

    print("\n===== SECONDARY: well-mapped vs under-mapped coverage contrast =====")
    coverage_summary(parsed, "Heidelberg (control)")
    try:
        u = parse_elements(fetch(BBOX_UNDERMAPPED))
        if u["pois"]:
            coverage_summary(u, "Tirana (under-mapped)")
        else:
            print("  Tirana: no POIs returned (adjust bbox).")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip undermapped] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
