"""External-oracle evaluation: does the OSM context-expectation model find
NATURALLY missing buildings (real, organic, clustered missingness) — not just
synthetic MCAR deletions?

FAST EXPLORATORY SPIKE — not production code.

Prior spike (plant_recover.py) showed the context-only "a feature is expected
here" model recovers RANDOM held-out POIs at AUC~0.92. That is MCAR (missing
completely at random) — the easy case. This script tests the harder, honest
question: does the same context model coincide with buildings OSM is GENUINELY
missing, judged by an EXTERNAL ground-truth dataset (Microsoft Global ML
Building Footprints, ML-derived from satellite, repo
microsoft/GlobalMLBuildingFootprints)?

METHOD (grid rasterization — sidesteps fiddly polygon matching):
  1. Pick an under-mapped bbox (Tirana, AL) + a well-mapped control (Heidelberg, DE).
  2. OSM layer: building-footprint centroids (way["building"], incl. those that
     also carry amenity/shop tags), roads, POIs — from the prior Overpass cache.
  3. External layer: MS footprint centroids for the bbox. We resolve the single
     zoom-9 Bing QuadKey covering the bbox, download ONLY that tile's gzipped
     GeoJSONL, stream-parse centroids, CLIP to the bbox, cache the small clipped
     set as JSON, and DELETE the raw multi-MB download (disk hygiene).
  4. Rasterize bbox to ~CELL_M m cells. For each cell:
        osm_present = any OSM building centroid in cell
        ext_present = any MS  building centroid in cell
        REAL MISSING = ext_present AND NOT osm_present
  5. Expectation model: trained on OSM CONTEXT ONLY to predict building presence
     from the surroundings (roads + OTHER buildings + POIs), with a self-exclusion
     radius so a cell never sees its own building. Positives = real OSM building
     cells; negatives = near-built empty cells. This is the "a building is
     expected here" model, fit on OSM alone — it never sees the external data.
  6. EVALUATE on the held-out question: among cells with NO OSM building, rank by
     model expectation; POSITIVE label = external building present there.
     Report ranking AUC vs random (0.5), precision@k, and the base rate.

Specificity control: the same pipeline on well-mapped Heidelberg — the
external-vs-OSM missing set should be small, and the model should NOT
manufacture many false gaps there.

CAVEATS (reported in output): external footprints have their own errors,
vintage (MS tiles here are 2026-02), and registration offset; ML footprints
have false positives; grid resolution / alignment tolerance matter. The honest
expectation is a meaningfully-lower-but-still-strong AUC than the 0.92 MCAR
number — clustered real missingness is harder than random deletion.

Regenerate:
    uv run \
        python -m artifacts_scripts.osm_spike.external_oracle
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import sys
from dataclasses import dataclass
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
from artifacts_scripts.osm_spike.inspect_cases import BBOX_UNDERMAPPED  # Tirana
from artifacts_scripts.osm_spike.plant_recover import (
    radius_counts,
    to_xy,
)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MS_LINKS_URL = (
    "https://minedbuildings.z5.web.core.windows.net/"
    "global-buildings/dataset-links.csv"
)
MS_LINKS_CACHE = CACHE_DIR / "ms_dataset_links.csv"

CELL_M = 50.0  # grid cell size in meters
UA = {"User-Agent": "bookgap-osm-spike/0.1 (research probe)"}


# ----------------------------------------------------------------------------
# Bing QuadKey tile math (zoom 9, the MS tiling level)
# ----------------------------------------------------------------------------
def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    sin = math.sin(math.radians(lat))
    x = (lon + 180.0) / 360.0
    y = 0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)
    n = 2 ** z
    tx = int(min(n - 1, max(0, math.floor(x * n))))
    ty = int(min(n - 1, max(0, math.floor(y * n))))
    return tx, ty


def _tile_to_quadkey(tx: int, ty: int, z: int) -> str:
    qk = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tx & mask:
            digit += 1
        if ty & mask:
            digit += 2
        qk += str(digit)
    return qk


def bbox_quadkeys(bbox: tuple[float, float, float, float], z: int = 9) -> set[str]:
    s, w, n, e = bbox
    txw, tyn = _lonlat_to_tile(w, n, z)
    txe, tys = _lonlat_to_tile(e, s, z)
    qks = set()
    for tx in range(min(txw, txe), max(txw, txe) + 1):
        for ty in range(min(tyn, tys), max(tyn, tys) + 1):
            qks.add(_tile_to_quadkey(tx, ty, z))
    return qks


# ----------------------------------------------------------------------------
# Microsoft Global ML Building Footprints — fetch, clip, delete-raw
# ----------------------------------------------------------------------------
def _load_ms_links() -> list[dict]:
    if not MS_LINKS_CACHE.exists():
        print(f"[ms] fetching dataset-links.csv -> {MS_LINKS_CACHE.name}", file=sys.stderr)
        r = requests.get(MS_LINKS_URL, timeout=120, headers=UA)
        r.raise_for_status()
        MS_LINKS_CACHE.write_text(r.text)
    return list(csv.DictReader(MS_LINKS_CACHE.open()))


def _poly_centroid(coords: list) -> tuple[float, float] | None:
    """Centroid of a GeoJSON Polygon's outer ring (mean of vertices).

    Good enough for the 'which cell does this building fall in' decision; we are
    not computing area, just a representative point.
    """
    try:
        ring = coords[0]
    except (IndexError, TypeError):
        return None
    if not ring:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))  # (lon, lat)


def fetch_ms_buildings(
    bbox: tuple[float, float, float, float], label: str, *, force: bool = False
) -> list[tuple[float, float]]:
    """Return MS building centroids (lat, lon) clipped to bbox.

    Downloads only the zoom-9 tile(s) covering bbox, clips, caches the small
    clipped set, and DELETES the raw .gz download.
    """
    s, w, n, e = bbox
    clip_cache = CACHE_DIR / f"ms_buildings_{label}.json"
    if clip_cache.exists() and not force:
        print(f"[ms cache hit] {clip_cache.name}", file=sys.stderr)
        return [tuple(p) for p in json.loads(clip_cache.read_text())]

    qks = bbox_quadkeys(bbox, 9)
    links = _load_ms_links()
    # Build (location,quadkey)->url; we match on quadkey alone (locations are
    # disjoint regions, the qk is globally unique at z9).
    url_for_qk: dict[str, str] = {}
    for row in links:
        if row["QuadKey"] in qks:
            url_for_qk[row["QuadKey"]] = row["Url"]
    missing = qks - set(url_for_qk)
    if missing:
        print(f"[ms warn] no MS tile for quadkeys {sorted(missing)} (label {label})", file=sys.stderr)

    centroids: list[tuple[float, float]] = []
    for qk, url in url_for_qk.items():
        raw_gz = CACHE_DIR / f"_raw_{label}_{qk}.csv.gz"
        try:
            print(f"[ms] download tile qk={qk} -> {raw_gz.name}", file=sys.stderr)
            with requests.get(url, timeout=300, headers=UA, stream=True) as r:
                r.raise_for_status()
                with raw_gz.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            # stream-parse the gzipped GeoJSONL, clipping as we go
            n_total = 0
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
            print(
                f"[ms] tile qk={qk}: {n_total} features scanned, "
                f"{len(centroids)} inside bbox so far",
                file=sys.stderr,
            )
        finally:
            # disk hygiene: delete the raw multi-MB download no matter what
            if raw_gz.exists():
                sz = raw_gz.stat().st_size
                raw_gz.unlink()
                print(f"[ms] deleted raw {raw_gz.name} ({sz/1e6:.1f} MB)", file=sys.stderr)

    clip_cache.write_text(json.dumps(centroids))
    print(f"[ms] cached {len(centroids)} clipped centroids -> {clip_cache.name}", file=sys.stderr)
    return centroids


# ----------------------------------------------------------------------------
# Grid + feature plumbing
# ----------------------------------------------------------------------------
@dataclass
class Layers:
    osm_bld_xy: np.ndarray   # OSM building footprint centroids (meters)
    road_xy: np.ndarray
    poi_xy: np.ndarray
    ext_xy: np.ndarray       # external (MS) building centroids (meters)
    lat0: float
    lon0: float
    bbox: tuple[float, float, float, float]


def build_layers(bbox, parsed, ms_latlon) -> Layers:
    """Project all layers to a common local meters frame centered on the bbox."""
    s, w, n, e = bbox
    lat0 = 0.5 * (s + n)
    lon0 = 0.5 * (w + e)

    # OSM building footprints: pure building-ways + POI-ways that carry building.
    bld = [b for b in parsed["buildings"]]  # building-ways (way center)
    poi_bld = [p for p in parsed["pois"] if p["type"] == "way" and "building" in p["tags"]]
    osm_bld = bld + poi_bld

    def _xy(items):
        if not items:
            return np.zeros((0, 2))
        return to_xy(
            np.array([x["lat"] for x in items]),
            np.array([x["lon"] for x in items]),
            lat0, lon0,
        )

    osm_bld_xy = _xy(osm_bld)
    road_xy = _xy(parsed["roads"])
    poi_xy = _xy(parsed["pois"])

    if ms_latlon:
        ext_xy = to_xy(
            np.array([p[0] for p in ms_latlon]),
            np.array([p[1] for p in ms_latlon]),
            lat0, lon0,
        )
    else:
        ext_xy = np.zeros((0, 2))

    return Layers(osm_bld_xy, road_xy, poi_xy, ext_xy, lat0, lon0, bbox)


def make_grid(layers: Layers, cell_m: float = CELL_M):
    """Cell-center grid spanning the bbox in the local meters frame."""
    s, w, n, e = layers.bbox
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(layers.lat0))
    # bbox extent in meters relative to (lat0, lon0)
    xmin = (w - layers.lon0) * m_lon
    xmax = (e - layers.lon0) * m_lon
    ymin = (s - layers.lat0) * m_lat
    ymax = (n - layers.lat0) * m_lat
    nx = max(1, int(round((xmax - xmin) / cell_m)))
    ny = max(1, int(round((ymax - ymin) / cell_m)))
    xs = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    ys = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    gx, gy = np.meshgrid(xs, ys)
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    return centers, (xmin, ymin, xmax, ymax), (nx, ny)


def cell_has_point(centers, points_xy, cell_m):
    """For each cell center, is there a point within cell_m/2 in BOTH axes?

    (Chebyshev half-cell test — a point is 'in' the cell whose center is nearest
    along each axis.) Returns bool array len(centers).
    """
    if len(points_xy) == 0:
        return np.zeros(len(centers), dtype=bool)
    half = cell_m / 2.0
    out = np.zeros(len(centers), dtype=bool)
    chunk = 4000
    px = points_xy[:, 0]
    py = points_xy[:, 1]
    for i in range(0, len(centers), chunk):
        c = centers[i : i + chunk]
        dx = np.abs(c[:, 0:1] - px[None, :])
        dy = np.abs(c[:, 1:2] - py[None, :])
        inside = (dx <= half) & (dy <= half)
        out[i : i + chunk] = inside.any(axis=1)
    return out


def context_features(
    query_xy: np.ndarray,
    layers: Layers,
    *,
    inner: float = 120.0,
    outer: float = 300.0,
    self_exclude: float = CELL_M,  # subtract OSM buildings in the query's own cell
) -> np.ndarray:
    """Context-only features for building expectation: surrounding roads, OTHER
    OSM buildings, and POIs. Self-exclusion removes OSM buildings within the
    query cell so a cell can't see its own building (mirrors plant_recover).
    """
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
        [bld_out, bld_in, road_out, road_in, poi_out, poi_in]
    ).astype(np.float64)


# ----------------------------------------------------------------------------
# The evaluation
# ----------------------------------------------------------------------------
@dataclass
class OracleResult:
    label: str
    bbox: tuple
    n_osm_bld: int
    n_ext_bld: int
    n_cells: int
    n_osm_cells: int
    n_ext_cells: int
    n_real_missing_cells: int   # ext present AND no OSM building
    base_rate: float            # among no-OSM cells, frac with ext building
    auc_context: float
    auc_baseline_road: float
    prec_at_k: dict             # k -> precision among top-k model-ranked no-OSM cells
    examples: list              # concrete (lat,lon,score) where model expects + ext confirms
    # STRICT view: restrict the eval to ISOLATED cells whose nearest OSM building
    # is > isolate_m away — true clustered gaps, not sub-cell registration offset.
    isolate_m: float
    n_strict_eval: int
    n_strict_pos: int
    strict_base_rate: float
    auc_context_strict: float
    strict_examples: list       # (lat,lon,score,nearest_osm_m)
    # spatial structure of real-missing cells (4-connected components)
    n_components: int
    n_singleton_components: int
    component_sizes: list       # descending


def run_oracle(
    label: str,
    bbox,
    *,
    seed: int = 0,
    cell_m: float = CELL_M,
    k_list=(10, 25, 50),
) -> OracleResult:
    parsed = parse_elements(fetch(bbox))
    ms = fetch_ms_buildings(bbox, label)
    layers = build_layers(bbox, parsed, ms)
    centers, bounds, (nx, ny) = make_grid(layers, cell_m)

    osm_cell = cell_has_point(centers, layers.osm_bld_xy, cell_m)
    ext_cell = cell_has_point(centers, layers.ext_xy, cell_m)
    real_missing = ext_cell & (~osm_cell)

    rng = np.random.default_rng(seed)

    # ---- Train the context-only building-expectation model on OSM alone ----
    # Positives: cells WITH an OSM building. Negatives: cells near built fabric
    # (>=1 road OR building within 150 m) but WITHOUT an OSM building. This makes
    # the negative set a fair 'built-up but empty' challenge rather than fields.
    near_built = (radius_counts(centers, layers.road_xy, 150.0) >= 1) | (
        radius_counts(centers, layers.osm_bld_xy, 150.0) >= 1
    )
    pos_idx = np.where(osm_cell)[0]
    neg_pool = np.where((~osm_cell) & near_built)[0]
    if len(pos_idx) == 0 or len(neg_pool) == 0:
        raise RuntimeError(f"{label}: degenerate train set (pos={len(pos_idx)} neg={len(neg_pool)})")
    n_neg = min(len(neg_pool), len(pos_idx))
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
    train_idx = np.concatenate([pos_idx, neg_idx])
    y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    feats_all = context_features(centers, layers)

    def _fit_score(cols):
        sc = StandardScaler()
        Xtr = sc.fit_transform(feats_all[train_idx][:, cols])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y_train)
        return clf.predict_proba(sc.transform(feats_all[:, cols]))[:, 1]

    # feature order: bld_out, bld_in, road_out, road_in, poi_out, poi_in
    score_ctx = _fit_score(list(range(feats_all.shape[1])))
    score_road = _fit_score([2, 3])  # road-only baseline (a structural confound)

    # ---- EVALUATE on no-OSM-building cells: rank by model, label = ext present.
    eval_mask = ~osm_cell
    # restrict the evaluation to plausibly-buildable space (near built fabric):
    # otherwise the bbox edge of fields/water trivially inflates the AUC.
    eval_mask = eval_mask & near_built
    ye = ext_cell[eval_mask].astype(int)
    se_ctx = score_ctx[eval_mask]
    se_road = score_road[eval_mask]
    centers_eval = centers[eval_mask]

    base_rate = float(ye.mean()) if len(ye) else float("nan")

    def _auc(scores):
        if ye.sum() == 0 or ye.sum() == len(ye):
            return float("nan")
        return float(roc_auc_score(ye, scores))

    auc_ctx = _auc(se_ctx)
    auc_road = _auc(se_road)

    # precision@k for the context model
    order = np.argsort(-se_ctx)
    prec_at_k = {}
    for k in k_list:
        kk = min(k, len(order))
        if kk == 0:
            prec_at_k[k] = float("nan")
        else:
            prec_at_k[k] = float(ye[order[:kk]].mean())

    # concrete examples: top model-ranked no-OSM cells that ext CONFIRMS
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(layers.lat0))
    examples = []
    for j in order:
        if ye[j] == 1:
            x, y = centers_eval[j]
            lat = layers.lat0 + y / m_lat
            lon = layers.lon0 + x / m_lon
            examples.append((round(lat, 6), round(lon, 6), round(float(se_ctx[j]), 3)))
        if len(examples) >= 3:
            break

    # ---- STRICT view: isolated gaps only (nearest OSM building > isolate_m). ----
    # The as-is grid AUC is partly driven by no-OSM cells that nonetheless sit a
    # few meters from an OSM building (MS/OSM registration + 50 m cell granularity).
    # Those are not true omissions. Here we keep ONLY cells whose nearest OSM
    # building is far away, so positives are genuine clustered gaps and negatives
    # are genuine isolated empty space — the honest test of clustered missingness.
    isolate_m = 50.0
    if len(layers.osm_bld_xy):
        nearest_osm = np.empty(len(centers))
        chunk = 4000
        for i in range(0, len(centers), chunk):
            c = centers[i : i + chunk]
            d = np.sqrt(
                (c[:, 0:1] - layers.osm_bld_xy[None, :, 0]) ** 2
                + (c[:, 1:2] - layers.osm_bld_xy[None, :, 1]) ** 2
            )
            nearest_osm[i : i + chunk] = d.min(axis=1)
    else:
        nearest_osm = np.full(len(centers), np.inf)
    strict_mask = (~osm_cell) & near_built & (nearest_osm > isolate_m)
    ys = ext_cell[strict_mask].astype(int)
    ss = score_ctx[strict_mask]
    cs = centers[strict_mask]
    dns = nearest_osm[strict_mask]
    if len(ys) and 0 < ys.sum() < len(ys):
        auc_strict = float(roc_auc_score(ys, ss))
    else:
        auc_strict = float("nan")
    strict_base = float(ys.mean()) if len(ys) else float("nan")
    strict_examples = []
    if len(ss):
        so = np.argsort(-ss)
        for j in so:
            if ys[j] == 1:
                lat = layers.lat0 + cs[j, 1] / m_lat
                lon = layers.lon0 + cs[j, 0] / m_lon
                strict_examples.append(
                    (round(lat, 6), round(lon, 6), round(float(ss[j]), 3), round(float(dns[j]), 0))
                )
            if len(strict_examples) >= 3:
                break

    # ---- spatial structure of real-missing cells (4-connected components). ----
    # Distinguishes scattered single-building omissions (singletons) from
    # block-scale clustered holes (large components) — they imply different
    # detectability for a context model.
    try:
        from scipy import ndimage  # noqa: PLC0415 — optional, spike-local

        rm_grid = real_missing.reshape(ny, nx)
        lab, ncomp = ndimage.label(rm_grid)
        comp_sizes = sorted(
            (int(s) for s in ndimage.sum(rm_grid, lab, range(1, ncomp + 1))),
            reverse=True,
        )
        n_singletons = sum(1 for s in comp_sizes if s == 1)
    except Exception:  # noqa: BLE001 — scipy optional
        ncomp, comp_sizes, n_singletons = -1, [], -1

    osm_bld_total = int(len(layers.osm_bld_xy))
    ext_bld_total = int(len(layers.ext_xy))

    return OracleResult(
        label=label,
        bbox=bbox,
        n_osm_bld=osm_bld_total,
        n_ext_bld=ext_bld_total,
        n_cells=len(centers),
        n_osm_cells=int(osm_cell.sum()),
        n_ext_cells=int(ext_cell.sum()),
        n_real_missing_cells=int(real_missing.sum()),
        base_rate=base_rate,
        auc_context=auc_ctx,
        auc_baseline_road=auc_road,
        prec_at_k=prec_at_k,
        examples=examples,
        isolate_m=isolate_m,
        n_strict_eval=int(len(ys)),
        n_strict_pos=int(ys.sum()) if len(ys) else 0,
        strict_base_rate=strict_base,
        auc_context_strict=auc_strict,
        strict_examples=strict_examples,
        n_components=ncomp,
        n_singleton_components=n_singletons,
        component_sizes=comp_sizes,
    )


def _print_result(r: OracleResult):
    print(f"\n========== {r.label}  bbox={r.bbox} ==========")
    print(f"  OSM building footprints (in bbox): {r.n_osm_bld}")
    print(f"  External (MS) building footprints: {r.n_ext_bld}")
    print(f"  grid: {r.n_cells} cells @ {CELL_M:.0f} m")
    print(f"    cells w/ OSM building : {r.n_osm_cells}")
    print(f"    cells w/ MS  building : {r.n_ext_cells}")
    print(f"    REAL-MISSING cells (MS present, no OSM): {r.n_real_missing_cells}")
    print(f"  EVAL (no-OSM, near-built cells): base_rate(MS present)={r.base_rate:.3f}")
    print(f"    AUC context model     = {r.auc_context:.3f}")
    print(f"    AUC road-only baseline= {r.auc_baseline_road:.3f}")
    print(f"    AUC random            = 0.500")
    pk = "  ".join(f"P@{k}={v:.3f}" for k, v in r.prec_at_k.items())
    print(f"    precision@k (context): {pk}   (lift over base = P@k - {r.base_rate:.3f})")
    print(f"  EXAMPLES (model expects bld, OSM empty, MS confirms): lat,lon,score")
    for lat, lon, sc in r.examples:
        print(f"    ({lat}, {lon})  score={sc}  "
              f"https://www.openstreetmap.org/#map=19/{lat}/{lon}")
    # ---- strict (isolated-gap) view: the honest clustered-missingness test ----
    print(f"  STRICT eval (nearest OSM bld > {r.isolate_m:.0f} m — true isolated gaps):")
    print(f"    n_eval={r.n_strict_eval} positives(MS)={r.n_strict_pos} "
          f"base_rate={r.strict_base_rate:.3f}")
    print(f"    AUC context (strict) = {r.auc_context_strict:.3f}   (vs random 0.500)")
    for lat, lon, sc, dnm in r.strict_examples:
        print(f"      ({lat}, {lon})  score={sc}  nearestOSM={dnm:.0f}m  "
              f"https://www.openstreetmap.org/#map=19/{lat}/{lon}")
    # ---- spatial structure of the missingness ----
    if r.n_components >= 0:
        big = [s for s in r.component_sizes if s >= 3]
        print(f"  MISSINGNESS STRUCTURE: {r.n_real_missing_cells} cells -> "
              f"{r.n_components} components; singletons={r.n_singleton_components}; "
              f">=3-cell clusters={len(big)} (sizes {big[:8]})")


def main():
    print("External-oracle test: does the OSM context-expectation model find")
    print("NATURALLY missing buildings (vs MS Global ML Building Footprints)?")
    print(f"grid cell = {CELL_M:.0f} m;  MCAR plant-recover baseline (prior) = AUC ~0.92")

    results = []
    for label, bbox in [
        ("Tirana (under-mapped)", BBOX_UNDERMAPPED),
        ("Heidelberg (control)", BBOX_HEIDELBERG),
    ]:
        try:
            r = run_oracle(label, bbox)
            results.append(r)
            _print_result(r)
        except Exception as e:  # noqa: BLE001 — spike: report and continue
            print(f"\n[ERROR] {label}: {e}", file=sys.stderr)
            raise

    print("\n----- CAVEATS -----")
    print("  - MS footprints are ML-derived (2026-02 vintage here): own false")
    print("    positives/negatives, vintage mismatch, and registration offset vs OSM.")
    print(f"  - {CELL_M:.0f} m cells + half-cell Chebyshev matching: a building straddling")
    print("    a cell edge can register in an adjacent cell; alignment tolerance matters.")
    print("  - 'No OSM building' uses way['building'] centroids; tiny outbuildings")
    print("    mapped only as nodes are not counted as footprints.")
    print("  - Eval restricted to near-built cells to avoid trivial field/water AUC.")
    print("  - STRICT view is the load-bearing honest test: positives are MS")
    print("    buildings > 50 m from ANY OSM building (true clustered gaps); the")
    print("    as-is grid AUC is inflated by sub-cell MS/OSM registration offset.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# PROVENANCE — regenerate (do not hand-edit cached outputs):
#   uv run \
#       python -m artifacts_scripts.osm_spike.external_oracle
#
# External ground truth: Microsoft Global ML Building Footprints
#   (repo microsoft/GlobalMLBuildingFootprints; dataset-links.csv -> per-QuadKey
#    gzipped GeoJSONL on Azure blob). Raw tiles are downloaded, clipped to the
#   bbox, and DELETED; only the small clipped centroid JSONs and the links CSV
#   are cached under artifacts_scripts/osm_spike/cache/.
# OSM context comes from the prior Overpass cache (fetch_osm.py).
# ---------------------------------------------------------------------------
