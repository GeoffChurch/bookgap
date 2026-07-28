"""DECISIVE salvage test: does the OSM building context-expectation model find
NATURAL BLOCK-SCALE missingness (contiguous unmapped fabric), judged polygon-to-
polygon against an external footprint oracle?

FAST EXPLORATORY SPIKE — not production code. Successor to external_oracle.py.

WHY THIS EXISTS
---------------
external_oracle.py tested the context model ("a building is expected here given
surrounding roads/buildings/POIs") against MS Global ML Building Footprints in
central Tirana and it FAILED: strict isolated-gap AUC = 0.41 (worse than random).
Diagnosis — two compounding problems:

  1. WRONG MISSINGNESS REGIME. Tirana's real missing buildings were salt-and-
     pepper singletons (52/63 components were size-1). A context model is
     STRUCTURALLY BLIND to a single missing building: the surroundings look
     identical whether or not that one building is mapped. The model can only
     work on BLOCK-SCALE holes — contiguous unmapped fabric where the absent
     buildings collectively change the local statistics.

  2. GRID ARTIFACT. The 50 m grid + MS/OSM registration offset made buildings a
     few meters across a cell boundary register as "missing", inflating the AUC.

This script fixes BOTH:
  (a) FIND a tile that actually has block-scale holes — peri-urban / informal /
      mapping-frontier areas where OSM roads are mapped but buildings are
      partially absent and the absences CLUSTER. We probe several candidate
      bboxes and MEASURE the component-size distribution to prove the regime.
  (b) POLYGON-LEVEL matching: a real missing building is an external footprint
      whose centroid has NO OSM building centroid within tolerance tau (no grid).

METHOD
------
  * OSM layer: way["building"] centroids (+ POI-ways carrying building), roads,
    POIs — via the Overpass cache (fetch_osm.py).
  * External layer: MS Global ML Building Footprints centroids, clipped to bbox,
    raw .gz DELETED after clip (disk hygiene). Multi-part QuadKeys are ALL
    downloaded (external_oracle.py dropped extra parts — fixed here).
  * MISSING set: external centroid with nearest-OSM-building distance > tau.
  * BLOCK-SCALE structure: cluster the MISSING external centroids by single-
    linkage (connect two missing buildings if within link_m) and report the
    component-size distribution. THIS is the polygon-level analog of the grid's
    4-connected components and the direct contrast with Tirana's 52/63 singletons.
  * EVAL: candidate locations = external centroids (real) + sampled empties (no
    external, no OSM). Restrict to NO-OSM candidates near built fabric (>=1 road
    within road_ctx_m) so there is context to predict from. Positive = external
    present. Rank by the context model; compare to a road-density-only baseline
    and random. Report AUC, precision@k, base rate, tau-sensitivity, a well-
    mapped specificity control, and concrete confirmed examples.

The honest comparison is MCAR 0.92 (plant_recover, random deletion) vs this real
block-scale number.

CAVEATS (printed): MS footprints are ML-derived (2026-02 vintage): own false
+/-, vintage mismatch, registration offset vs OSM. If a hole has NO mapped roads,
the model has NO context and SHOULD fail — we report whether the chosen tile's
holes have road context.

Regenerate:
    uv run \
        python -m artifacts_scripts.osm_spike.salvage_blockscale
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from artifacts_scripts.osm_spike.fetch_osm import (
    BBOX_HEIDELBERG,
    fetch,
    parse_elements,
)
from artifacts_scripts.osm_spike.external_oracle import (
    CACHE_DIR,
    MS_LINKS_CACHE,
    MS_LINKS_URL,
    UA,
    _load_ms_links,
    _poly_centroid,
    bbox_quadkeys,
)
from artifacts_scripts.osm_spike.plant_recover import radius_counts, to_xy

# ----------------------------------------------------------------------------
# Candidate bboxes (south, west, north, east). Peri-urban edges, ~2-4 km^2.
# Mapping-frontier areas: OSM roads mapped, buildings partially absent. We probe
# a handful and pick one satisfying ALL THREE criteria (roads present, 20-70%
# external footprints unmatched in OSM, and the missing set CLUSTERS).
# ----------------------------------------------------------------------------
CANDIDATES: dict[str, tuple[float, float, float, float]] = {
    # NOTE: Kinshasa (DRC) was probed and DROPPED — MS Global ML footprints have
    # a coverage HOLE over eastern Kinshasa (QuadKey 300010321 MS data stops at
    # lat -4.21 / lon 15.41; our bboxes had 0 MS centroids) while OSM is densely
    # HOT-mapped (~31k buildings). That is the inverse regime (OSM dense, oracle
    # absent) and cannot test missingness. The cities below all have confirmed
    # dense MS coverage (verified: Dar QuadKey 300112113 has 666k centroids).
    #
    # Dar es Salaam, TZ — informal peri-urban expansion, dense MS coverage
    "dar_kimara": (-6.800, 39.160, -6.775, 39.190),
    "dar_kinondoni": (-6.770, 39.200, -6.750, 39.230),
    "dar_kigamboni": (-6.840, 39.300, -6.815, 39.330),
    "dar_gongolamboto": (-6.840, 39.200, -6.815, 39.230),
    # Lagos, NG — mainland peri-urban
    "lagos_alimosho": (6.580, 3.250, 6.605, 3.285),
    "lagos_ikorodu": (6.600, 3.490, 6.625, 3.520),
    "lagos_ayobo": (6.610, 3.270, 6.635, 3.300),
    # Luanda, AO — southern expansion
    "luanda_viana": (-8.920, 13.350, -8.895, 13.385),
    "luanda_cacuaco": (-8.770, 13.380, -8.745, 13.410),
    # Antananarivo, MG — peri-urban
    "tana_north": (-18.870, 47.510, -18.850, 47.540),
    "tana_south": (-18.940, 47.520, -18.915, 47.550),
}

# Geometry / matching parameters
TAU_DEFAULT = 13.0          # m: external centroid is "missing" if no OSM bld within
TAU_LIST = (10.0, 13.0, 18.0)
LINK_M = 30.0               # m: two missing buildings are linked if within this
ROAD_CTX_M = 120.0          # m: candidate must have >=1 road within (has context)
EMPTY_MIN_EXT_M = 18.0      # m: an "empty" must have no external bld within
EMPTY_MIN_OSM_M = 18.0      # m: ... and no OSM bld within
N_EMPTY_MULT = 3           # sample this many empties per positive (capped)


# ----------------------------------------------------------------------------
# Microsoft footprints — multi-part-aware fetch (fixes external_oracle drop bug)
# ----------------------------------------------------------------------------
def fetch_ms_buildings_multipart(
    bbox: tuple[float, float, float, float], label: str, *, force: bool = False
) -> list[tuple[float, float]]:
    """MS building centroids (lat, lon) clipped to bbox. Downloads ALL parts of
    every covering zoom-9 QuadKey, clips, caches the small clipped set, deletes
    raw .gz. external_oracle.fetch_ms_buildings kept only the last URL per
    QuadKey — for dense Global-South tiles with multiple parts that silently
    dropped most buildings, so we re-implement the part-aware loop here.
    """
    s, w, n, e = bbox
    clip_cache = CACHE_DIR / f"ms_blk_{label}.json"
    if clip_cache.exists() and not force:
        print(f"[ms cache hit] {clip_cache.name}", file=sys.stderr)
        return [tuple(p) for p in json.loads(clip_cache.read_text())]

    qks = bbox_quadkeys(bbox, 9)
    links = _load_ms_links()
    urls_for_qk: dict[str, list[str]] = defaultdict(list)
    for row in links:
        if row["QuadKey"] in qks:
            urls_for_qk[row["QuadKey"]].append(row["Url"])
    missing = qks - set(urls_for_qk)
    if missing:
        print(f"[ms warn] no MS tile for quadkeys {sorted(missing)} (label {label})",
              file=sys.stderr)

    centroids: list[tuple[float, float]] = []
    for qk in sorted(urls_for_qk):
        for pi, url in enumerate(urls_for_qk[qk]):
            raw_gz = CACHE_DIR / f"_rawblk_{label}_{qk}_{pi}.csv.gz"
            try:
                print(f"[ms] dl qk={qk} part={pi} -> {raw_gz.name}", file=sys.stderr)
                with requests.get(url, timeout=300, headers=UA, stream=True) as r:
                    r.raise_for_status()
                    with raw_gz.open("wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                n_total, n_in = 0, 0
                with gzip.open(raw_gz, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        n_total += 1
                        try:
                            feat = json.loads(line)
                            geom = feat["geometry"]
                            if geom["type"] != "Polygon":
                                continue
                            c = _poly_centroid(geom["coordinates"])
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
                        if c is None:
                            continue
                        lon, lat = c
                        if w <= lon <= e and s <= lat <= n:
                            centroids.append((lat, lon))
                            n_in += 1
                print(f"[ms]   qk={qk} part={pi}: {n_total} scanned, +{n_in} in bbox "
                      f"({len(centroids)} total)", file=sys.stderr)
            finally:
                if raw_gz.exists():
                    sz = raw_gz.stat().st_size
                    raw_gz.unlink()
                    print(f"[ms]   deleted raw {raw_gz.name} ({sz/1e6:.1f} MB)",
                          file=sys.stderr)

    clip_cache.write_text(json.dumps(centroids))
    print(f"[ms] cached {len(centroids)} clipped centroids -> {clip_cache.name}",
          file=sys.stderr)
    return centroids


# ----------------------------------------------------------------------------
# Layers (local meters frame) — polygon-level, no grid
# ----------------------------------------------------------------------------
@dataclass
class Layers:
    osm_bld_xy: np.ndarray
    road_xy: np.ndarray
    poi_xy: np.ndarray
    ext_xy: np.ndarray
    lat0: float
    lon0: float
    bbox: tuple[float, float, float, float]
    m_lat: float
    m_lon: float


def build_layers(bbox, parsed, ms_latlon) -> Layers:
    s, w, n, e = bbox
    lat0 = 0.5 * (s + n)
    lon0 = 0.5 * (w + e)
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat0))

    bld = list(parsed["buildings"])
    poi_bld = [p for p in parsed["pois"]
               if p["type"] == "way" and "building" in p["tags"]]
    osm_bld = bld + poi_bld

    def _xy(items):
        if not items:
            return np.zeros((0, 2))
        return to_xy(np.array([x["lat"] for x in items]),
                     np.array([x["lon"] for x in items]), lat0, lon0)

    osm_bld_xy = _xy(osm_bld)
    road_xy = _xy(parsed["roads"])
    poi_xy = _xy(parsed["pois"])
    if ms_latlon:
        ext_xy = to_xy(np.array([p[0] for p in ms_latlon]),
                       np.array([p[1] for p in ms_latlon]), lat0, lon0)
    else:
        ext_xy = np.zeros((0, 2))
    return Layers(osm_bld_xy, road_xy, poi_xy, ext_xy, lat0, lon0, bbox, m_lat, m_lon)


def nearest_dist(query_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    """Distance (m) from each query point to the nearest source point.
    Chunked dense computation (consistent with the spike's numpy style; corpora
    here are a few thousand points so this is fine)."""
    if len(query_xy) == 0:
        return np.zeros(0)
    if len(source_xy) == 0:
        return np.full(len(query_xy), np.inf)
    out = np.empty(len(query_xy))
    chunk = 2000
    sx = source_xy[:, 0]
    sy = source_xy[:, 1]
    for i in range(0, len(query_xy), chunk):
        q = query_xy[i:i + chunk]
        d2 = (q[:, 0:1] - sx[None, :]) ** 2 + (q[:, 1:2] - sy[None, :]) ** 2
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def cluster_components(points_xy: np.ndarray, link_m: float) -> list[int]:
    """Single-linkage connected components: two points are linked if within
    link_m. Returns component sizes (descending). Union-find over the link graph.
    This is the polygon-level analog of grid 4-connectivity — it measures whether
    the MISSING buildings form contiguous fabric (block-scale) or scatter.
    """
    n = len(points_xy)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    r2 = link_m * link_m
    px = points_xy[:, 0]
    py = points_xy[:, 1]
    chunk = 1500
    for i in range(0, n, chunk):
        q = points_xy[i:i + chunk]
        d2 = (q[:, 0:1] - px[None, :]) ** 2 + (q[:, 1:2] - py[None, :]) ** 2
        within = d2 <= r2
        for li in range(q.shape[0]):
            gi = i + li
            for gj in np.where(within[li])[0]:
                if gj > gi:
                    union(gi, int(gj))
    sizes: dict[int, int] = defaultdict(int)
    for a in range(n):
        sizes[find(a)] += 1
    return sorted(sizes.values(), reverse=True)


def context_features(query_xy: np.ndarray, layers: Layers, *,
                     inner: float = 120.0, outer: float = 300.0,
                     self_exclude: float = 18.0) -> np.ndarray:
    """Context-only building-expectation features: surrounding roads, OTHER OSM
    buildings, POIs. Self-exclusion removes OSM buildings within self_exclude of
    the query so a query sitting on an OSM building doesn't trivially see it."""
    bld_out = radius_counts(query_xy, layers.osm_bld_xy, outer)
    bld_in = radius_counts(query_xy, layers.osm_bld_xy, inner)
    bld_self = radius_counts(query_xy, layers.osm_bld_xy, self_exclude)
    bld_out = np.maximum(bld_out - bld_self, 0)
    bld_in = np.maximum(bld_in - bld_self, 0)
    road_out = radius_counts(query_xy, layers.road_xy, outer)
    road_in = radius_counts(query_xy, layers.road_xy, inner)
    poi_out = radius_counts(query_xy, layers.poi_xy, outer)
    poi_in = radius_counts(query_xy, layers.poi_xy, inner)
    return np.column_stack(
        [bld_out, bld_in, road_out, road_in, poi_out, poi_in]).astype(np.float64)


def sample_empties(layers: Layers, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample empty locations: no external bld within EMPTY_MIN_EXT_M and no OSM
    bld within EMPTY_MIN_OSM_M, drawn uniformly over the bbox extent in meters.
    No 'near building' requirement here — the eval mask (road context) is applied
    later uniformly to positives and empties alike."""
    s, w, n_, e = layers.bbox
    xmin = (w - layers.lon0) * layers.m_lon
    xmax = (e - layers.lon0) * layers.m_lon
    ymin = (s - layers.lat0) * layers.m_lat
    ymax = (n_ - layers.lat0) * layers.m_lat
    out = []
    tries = 0
    max_tries = n * 400
    while len(out) < n and tries < max_tries:
        tries += 1
        cand = np.array([[rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]])
        if len(layers.ext_xy) and nearest_dist(cand, layers.ext_xy)[0] < EMPTY_MIN_EXT_M:
            continue
        if len(layers.osm_bld_xy) and nearest_dist(cand, layers.osm_bld_xy)[0] < EMPTY_MIN_OSM_M:
            continue
        out.append(cand[0])
    if len(out) < n:
        print(f"[warn] sampled only {len(out)}/{n} empties in {tries} tries",
              file=sys.stderr)
    return np.array(out) if out else np.zeros((0, 2))


# ----------------------------------------------------------------------------
# Probe: does a bbox satisfy the three criteria?
# ----------------------------------------------------------------------------
@dataclass
class Probe:
    label: str
    bbox: tuple
    n_osm_bld: int
    n_road: int
    n_ext: int
    n_missing: int          # external footprints with no OSM bld within tau
    frac_missing: float     # n_missing / n_ext
    comp_sizes: list        # descending, of the missing set @ LINK_M
    n_comp: int
    n_singleton: int
    frac_missing_with_road_ctx: float  # of missing, fraction with road within ROAD_CTX_M
    largest_comp: int
    qualifies: bool
    reason: str = ""


def probe_bbox(label: str, bbox, *, tau: float = TAU_DEFAULT) -> Probe | None:
    try:
        parsed = parse_elements(fetch(bbox))
    except Exception as e:  # noqa: BLE001
        print(f"[probe skip] {label}: OSM fetch failed: {e}", file=sys.stderr)
        return None
    try:
        ms = fetch_ms_buildings_multipart(bbox, label)
    except Exception as e:  # noqa: BLE001
        print(f"[probe skip] {label}: MS fetch failed: {e}", file=sys.stderr)
        return None
    layers = build_layers(bbox, parsed, ms)
    n_ext = len(layers.ext_xy)
    n_osm = len(layers.osm_bld_xy)
    n_road = len(layers.road_xy)
    if n_ext == 0:
        return Probe(label, bbox, n_osm, n_road, 0, 0, float("nan"), [], 0, 0,
                     float("nan"), 0, False, "no external footprints")
    d_ext_to_osm = nearest_dist(layers.ext_xy, layers.osm_bld_xy)
    missing_mask = d_ext_to_osm > tau
    miss_xy = layers.ext_xy[missing_mask]
    n_missing = int(missing_mask.sum())
    frac_missing = n_missing / n_ext
    comp_sizes = cluster_components(miss_xy, LINK_M)
    n_comp = len(comp_sizes)
    n_singleton = sum(1 for s in comp_sizes if s == 1)
    largest = comp_sizes[0] if comp_sizes else 0
    # road context on the missing set
    if n_missing:
        road_near_missing = radius_counts(miss_xy, layers.road_xy, ROAD_CTX_M)
        frac_road_ctx = float((road_near_missing >= 1).mean())
    else:
        frac_road_ctx = float("nan")
    # qualification: roads mapped, 20-70% missing, clustered (largest comp big &
    # not singleton-dominated), and the missing fabric has road context.
    reasons = []
    if n_road < 40:
        reasons.append(f"few roads({n_road})")
    if not (0.20 <= frac_missing <= 0.70):
        reasons.append(f"frac_missing={frac_missing:.2f} outside [.2,.7]")
    if largest < 12:
        reasons.append(f"largest_comp={largest}<12 (not block-scale)")
    if n_comp and (n_singleton / n_comp) > 0.75:
        reasons.append(f"singleton-dominated({n_singleton}/{n_comp})")
    if not (frac_road_ctx >= 0.5):
        reasons.append(f"only {frac_road_ctx:.2f} of missing have road ctx")
    qualifies = not reasons
    return Probe(label, bbox, n_osm, n_road, n_ext, n_missing, frac_missing,
                 comp_sizes, n_comp, n_singleton, frac_road_ctx, largest,
                 qualifies, "; ".join(reasons))


def _print_probe(p: Probe):
    print(f"\n--- PROBE {p.label}  bbox={p.bbox} ---")
    print(f"    OSM: bld={p.n_osm_bld} road={p.n_road} | MS ext bld={p.n_ext}")
    if p.n_ext == 0:
        print(f"    DISQUALIFIED: {p.reason}")
        return
    big = [s for s in p.comp_sizes if s >= 5]
    print(f"    MISSING (no OSM bld within {TAU_DEFAULT:.0f} m): {p.n_missing} "
          f"({100*p.frac_missing:.0f}% of ext)")
    print(f"    component sizes @ link {LINK_M:.0f} m: {p.n_comp} comps, "
          f"singletons={p.n_singleton}, >=5-clusters={len(big)} "
          f"(top {p.comp_sizes[:8]})")
    print(f"    largest block={p.largest_comp}; missing-with-road-context="
          f"{100*p.frac_missing_with_road_ctx:.0f}%")
    print(f"    QUALIFIES={p.qualifies}" + (f"  [{p.reason}]" if p.reason else ""))


# ----------------------------------------------------------------------------
# Full polygon-level evaluation on the chosen tile
# ----------------------------------------------------------------------------
@dataclass
class EvalResult:
    label: str
    bbox: tuple
    tau: float
    n_osm_bld: int
    n_ext: int
    n_missing: int
    frac_missing: float
    comp_sizes: list
    n_comp: int
    n_singleton: int
    # eval
    n_eval: int
    n_pos: int
    base_rate: float
    auc_context: float
    auc_road: float
    prec_at_k: dict
    examples: list                  # (lat, lon, score, nearest_osm_m)
    frac_missing_with_road_ctx: float
    tau_sensitivity: dict = field(default_factory=dict)  # tau -> (n_missing, frac, auc)


def run_eval(label: str, bbox, *, tau: float = TAU_DEFAULT, seed: int = 0,
             k_list=(10, 25, 50), tau_list=TAU_LIST) -> EvalResult:
    parsed = parse_elements(fetch(bbox))
    ms = fetch_ms_buildings_multipart(bbox, label)
    layers = build_layers(bbox, parsed, ms)
    rng = np.random.default_rng(seed)

    n_ext = len(layers.ext_xy)
    n_osm = len(layers.osm_bld_xy)

    # ---- Train context model on OSM alone (positives = OSM buildings; negatives
    #      = empty locations near built fabric, no OSM building). ----
    near_built = (radius_counts(layers.osm_bld_xy, layers.road_xy, 150.0) >= 1) | True
    # train positives: OSM buildings themselves
    pos_xy = layers.osm_bld_xy
    # train negatives: empties (no OSM bld nearby, but near a road = buildable)
    n_train_neg = max(50, len(pos_xy))
    neg_xy = sample_empties(layers, n_train_neg, rng)
    # keep only negatives with road context (buildable space), to match eval
    if len(neg_xy):
        nb = radius_counts(neg_xy, layers.road_xy, ROAD_CTX_M) >= 1
        neg_xy = neg_xy[nb]
    if len(pos_xy) == 0 or len(neg_xy) == 0:
        raise RuntimeError(f"{label}: degenerate train (pos={len(pos_xy)} neg={len(neg_xy)})")

    feat_pos = context_features(pos_xy, layers)
    feat_neg = context_features(neg_xy, layers)
    X_train = np.vstack([feat_pos, feat_neg])
    y_train = np.concatenate([np.ones(len(feat_pos)), np.zeros(len(feat_neg))])

    def _fit(cols):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train[:, cols])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y_train)
        return sc, clf, cols

    full_cols = list(range(X_train.shape[1]))
    road_cols = [2, 3]
    m_full = _fit(full_cols)
    m_road = _fit(road_cols)

    def _score(model, q_xy):
        sc, clf, cols = model
        fx = context_features(q_xy, layers)
        return clf.predict_proba(sc.transform(fx[:, cols]))[:, 1]

    # ---- Build candidate eval set: positives = MISSING external (no OSM within
    #      tau), negatives = empties (no external, no OSM). Restrict ALL to road
    #      context (>=1 road within ROAD_CTX_M). ----
    d_ext_to_osm = nearest_dist(layers.ext_xy, layers.osm_bld_xy)
    missing_mask = d_ext_to_osm > tau
    miss_xy = layers.ext_xy[missing_mask]
    n_missing = int(missing_mask.sum())
    frac_missing = n_missing / n_ext if n_ext else float("nan")

    comp_sizes = cluster_components(miss_xy, LINK_M)
    n_comp = len(comp_sizes)
    n_singleton = sum(1 for s in comp_sizes if s == 1)

    # road context on missing
    if n_missing:
        miss_road = radius_counts(miss_xy, layers.road_xy, ROAD_CTX_M)
        frac_road_ctx = float((miss_road >= 1).mean())
        pos_eval_xy = miss_xy[miss_road >= 1]
    else:
        frac_road_ctx = float("nan")
        pos_eval_xy = miss_xy

    n_neg_eval = min(len(pos_eval_xy) * N_EMPTY_MULT, 2000)
    neg_eval_xy = sample_empties(layers, max(n_neg_eval, 1), rng)
    if len(neg_eval_xy):
        nb = radius_counts(neg_eval_xy, layers.road_xy, ROAD_CTX_M) >= 1
        neg_eval_xy = neg_eval_xy[nb]

    cand_xy = np.vstack([pos_eval_xy, neg_eval_xy]) if len(pos_eval_xy) else neg_eval_xy
    y_eval = np.concatenate([np.ones(len(pos_eval_xy)), np.zeros(len(neg_eval_xy))])
    base_rate = float(y_eval.mean()) if len(y_eval) else float("nan")

    s_ctx = _score(m_full, cand_xy)
    s_road = _score(m_road, cand_xy)

    def _auc(scores):
        if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            return float("nan")
        return float(roc_auc_score(y_eval, scores))

    auc_ctx = _auc(s_ctx)
    auc_road = _auc(s_road)

    order = np.argsort(-s_ctx)
    prec_at_k = {}
    for k in k_list:
        kk = min(k, len(order))
        prec_at_k[k] = float(y_eval[order[:kk]].mean()) if kk else float("nan")

    # concrete examples: top-ranked POSITIVES (model expects bld, OSM empty, MS
    # confirms), with verified nearest-OSM distance
    examples = []
    pos_nearest = nearest_dist(pos_eval_xy, layers.osm_bld_xy) if len(pos_eval_xy) else np.zeros(0)
    for j in order:
        if y_eval[j] == 1:
            x, y = cand_xy[j]
            lat = layers.lat0 + y / layers.m_lat
            lon = layers.lon0 + x / layers.m_lon
            nd = float(pos_nearest[j]) if j < len(pos_nearest) else float("nan")
            examples.append((round(lat, 6), round(lon, 6),
                             round(float(s_ctx[j]), 3), round(nd, 1)))
        if len(examples) >= 3:
            break

    # tau sensitivity: recompute missing set + AUC at each tau (reuse trained model)
    tau_sens = {}
    for t in tau_list:
        mm = d_ext_to_osm > t
        mxy = layers.ext_xy[mm]
        if len(mxy):
            mr = radius_counts(mxy, layers.road_xy, ROAD_CTX_M)
            pxy = mxy[mr >= 1]
        else:
            pxy = mxy
        ne = sample_empties(layers, max(min(len(pxy) * N_EMPTY_MULT, 2000), 1), rng)
        if len(ne):
            ne = ne[radius_counts(ne, layers.road_xy, ROAD_CTX_M) >= 1]
        if not len(pxy) or not len(ne):
            tau_sens[t] = (int(mm.sum()), float(mm.sum()) / n_ext if n_ext else float("nan"),
                           float("nan"))
            continue
        cxy = np.vstack([pxy, ne])
        yv = np.concatenate([np.ones(len(pxy)), np.zeros(len(ne))])
        sv = _score(m_full, cxy)
        a = float(roc_auc_score(yv, sv)) if 0 < yv.sum() < len(yv) else float("nan")
        tau_sens[t] = (int(mm.sum()), float(mm.sum()) / n_ext if n_ext else float("nan"), a)

    return EvalResult(
        label=label, bbox=bbox, tau=tau, n_osm_bld=n_osm, n_ext=n_ext,
        n_missing=n_missing, frac_missing=frac_missing, comp_sizes=comp_sizes,
        n_comp=n_comp, n_singleton=n_singleton, n_eval=len(y_eval),
        n_pos=int(y_eval.sum()), base_rate=base_rate, auc_context=auc_ctx,
        auc_road=auc_road, prec_at_k=prec_at_k, examples=examples,
        frac_missing_with_road_ctx=frac_road_ctx, tau_sensitivity=tau_sens,
    )


def _print_eval(r: EvalResult):
    print(f"\n========== EVAL {r.label}  bbox={r.bbox}  tau={r.tau:.0f} m ==========")
    print(f"  OSM buildings={r.n_osm_bld} | MS external={r.n_ext}")
    print(f"  MISSING (no OSM within {r.tau:.0f} m): {r.n_missing} "
          f"({100*r.frac_missing:.0f}% of external)")
    big = [s for s in r.comp_sizes if s >= 5]
    print(f"  block structure @ link {LINK_M:.0f} m: {r.n_comp} comps, "
          f"singletons={r.n_singleton}, >=5-clusters={len(big)} (top {r.comp_sizes[:10]})")
    print(f"  missing-with-road-context = {100*r.frac_missing_with_road_ctx:.0f}%")
    print(f"  EVAL set: n={r.n_eval} positives={r.n_pos} base_rate={r.base_rate:.3f}")
    print(f"    AUC context model      = {r.auc_context:.3f}")
    print(f"    AUC road-only baseline = {r.auc_road:.3f}")
    print(f"    AUC random             = 0.500")
    pk = "  ".join(f"P@{k}={v:.3f}" for k, v in r.prec_at_k.items())
    print(f"    precision@k (context)  : {pk}  (base rate {r.base_rate:.3f})")
    print(f"  tau sensitivity (tau -> n_missing, frac, AUC):")
    for t, (nm, fr, a) in r.tau_sensitivity.items():
        print(f"      tau={t:>4.0f} m: n_missing={nm:5d} frac={fr:.2f} AUC={a:.3f}")
    print(f"  EXAMPLES (model expects bld, OSM empty within tau, MS confirms):")
    for lat, lon, sc, nd in r.examples:
        print(f"    ({lat}, {lon})  score={sc}  nearestOSM={nd:.1f}m  "
              f"https://www.openstreetmap.org/#map=19/{lat}/{lon}")


def main():
    print("=" * 78)
    print("DECISIVE SALVAGE TEST — block-scale OSM building completeness")
    print("Polygon-level matching (no grid). MCAR plant-recover baseline = AUC ~0.92.")
    print("Prior Tirana strict isolated-gap AUC = 0.41 (singleton-dominated, grid).")
    print("=" * 78)

    # ---- STEP 1: probe candidates, pick a qualifying block-scale tile ----
    print("\n##### STEP 1: probe candidate peri-urban bboxes #####")
    probes: list[Probe] = []
    for label, bbox in CANDIDATES.items():
        p = probe_bbox(label, bbox)
        if p is not None:
            probes.append(p)
            _print_probe(p)

    qualifying = [p for p in probes if p.qualifies]
    print("\n##### STEP 1 SUMMARY #####")
    for p in sorted(probes, key=lambda x: (-(x.largest_comp), x.label)):
        flag = "QUALIFIES" if p.qualifies else "no"
        print(f"  {flag:>10}  {p.label:<22} ext={p.n_ext:5d} "
              f"missing={100*p.frac_missing if p.frac_missing==p.frac_missing else float('nan'):4.0f}% "
              f"largest_block={p.largest_comp:3d} roads={p.n_road}")

    if not qualifying:
        print("\n[!!] COULD NOT LOCATE THE BLOCK-SCALE REGIME among probed bboxes.")
        print("     (Distinct from 'model fails'.) Falling back to the best-available")
        print("     tiles by largest missing component for an illustrative eval.")
        to_eval = sorted(
            (p for p in probes if p.n_ext > 0 and p.largest_comp >= 5),
            key=lambda x: -x.largest_comp)[:3]
    else:
        # Evaluate ALL qualifiers (ordered by block size). The experiment's finding
        # is that detectability varies across qualifying tiles by WHETHER the holes
        # abut already-mapped fabric (so context differs from empty space) — the
        # interior of a very large hole has no nearby OSM building and looks empty.
        # Reporting every qualifier is the honest comparison, not cherry-picking one.
        to_eval = sorted(qualifying, key=lambda x: -x.largest_comp)

    if not to_eval:
        print("\nNo usable tile (no external footprints anywhere). STOP.")
        return

    print(f"\n##### CHOSEN TILES ({len(to_eval)}): "
          f"{', '.join(p.label for p in to_eval)} #####")

    # ---- STEP 2+3: polygon-level eval on each chosen tile ----
    r = None
    for p in to_eval:
        print(f"\n----- evaluating {p.label} (qualifies={p.qualifies}, "
              f"largest_block={p.largest_comp}) -----")
        r = run_eval(p.label, p.bbox)
        _print_eval(r)

    # ---- specificity control: well-mapped Heidelberg, identical pipeline ----
    print("\n##### SPECIFICITY CONTROL: Heidelberg (well-mapped) #####")
    rc = run_eval("heidelberg_control", BBOX_HEIDELBERG)
    _print_eval(rc)

    # ---- CAVEATS ----
    print("\n----- CAVEATS -----")
    print("  - MS footprints are ML-derived (2026-02 vintage): own false +/-,")
    print("    vintage mismatch, and registration offset vs OSM.")
    print(f"  - 'missing' = external centroid with NO OSM bld centroid within tau")
    print(f"    ({r.tau:.0f} m default); tau-sensitivity reported above.")
    print("  - OSM buildings = way['building'] centroids; node-only outbuildings")
    print("    are not counted as footprints.")
    print("  - Eval restricted to road-context locations (>=1 road within")
    print(f"    {ROAD_CTX_M:.0f} m): holes with NO mapped roads have no context and")
    print("    the model SHOULD fail there by construction (reported as %road-ctx).")
    print("  - The honest comparison: MCAR 0.92 vs this real block-scale AUC.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# PROVENANCE — regenerate (do NOT hand-edit any cached outputs):
#   uv run \
#       python -m artifacts_scripts.osm_spike.salvage_blockscale
#
# External ground truth: Microsoft Global ML Building Footprints
#   (repo microsoft/GlobalMLBuildingFootprints; dataset-links.csv -> per-QuadKey
#    gzipped GeoJSONL on Azure blob; ALL parts per QuadKey downloaded). Raw tiles
#   are downloaded, clipped to bbox, and DELETED; only small clipped centroid
#   JSONs + the links CSV are cached under artifacts_scripts/osm_spike/cache/.
# OSM context via Overpass (fetch_osm.py), cached per-bbox.
# ---------------------------------------------------------------------------
