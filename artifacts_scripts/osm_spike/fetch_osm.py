"""Fetch OSM data via Overpass API and cache to local JSON.

FAST EXPLORATORY SPIKE — not production code.

Fetches POIs (amenity/shop), building footprints, and roads within a small bbox
and caches the raw Overpass JSON to a local file for reuse.

Regenerate:
    uv run \
        python -m artifacts_scripts.osm_spike.fetch_osm
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# (south, west, north, east)
BBOX_HEIDELBERG = (49.405, 8.680, 49.420, 8.710)  # central Heidelberg, ~2 km^2 well-mapped control


def _bbox_str(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return f"{s},{w},{n},{e}"


def build_query(bbox: tuple[float, float, float, float]) -> str:
    """Overpass QL: POIs (amenity OR shop), buildings, roads in bbox."""
    bb = _bbox_str(bbox)
    return f"""
[out:json][timeout:120];
(
  node["amenity"]({bb});
  way["amenity"]({bb});
  node["shop"]({bb});
  way["shop"]({bb});
  way["building"]({bb});
  way["highway"]({bb});
);
out center tags;
""".strip()


def fetch(bbox: tuple[float, float, float, float], *, force: bool = False) -> dict:
    """Fetch (cached) Overpass JSON for a bbox."""
    query = build_query(bbox)
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"overpass_{key}.json"

    if cache_file.exists() and not force:
        print(f"[cache hit] {cache_file.name} for bbox {bbox}", file=sys.stderr)
        return json.loads(cache_file.read_text())

    last_err = None
    for endpoint in ENDPOINTS:
        for attempt in range(3):
            try:
                print(f"[fetch] {endpoint} attempt {attempt + 1} bbox {bbox}", file=sys.stderr)
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=180,
                    headers={"User-Agent": "bookgap-osm-spike/0.1 (research probe)"},
                )
                if resp.status_code in (429, 504):
                    wait = 10 * (attempt + 1)
                    print(f"[backoff] HTTP {resp.status_code}, sleeping {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                cache_file.write_text(json.dumps(data))
                n = len(data.get("elements", []))
                print(f"[ok] {n} elements -> {cache_file.name}", file=sys.stderr)
                return data
            except Exception as e:  # noqa: BLE001 — spike: tolerate any transport error
                last_err = e
                print(f"[err] {e}", file=sys.stderr)
                time.sleep(5)
    raise RuntimeError(f"All endpoints failed: {last_err}")


def parse_elements(data: dict) -> dict[str, list[dict]]:
    """Split elements into pois / buildings / roads with (lat, lon) coords.

    A way uses its 'center' (from `out center`). POIs that are both amenity/shop
    AND building keep their POI classification.
    """
    pois, buildings, roads = [], [], []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:  # way
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]
        if lat is None or lon is None:
            continue
        rec = {"id": el["id"], "type": el["type"], "lat": lat, "lon": lon, "tags": tags}

        is_poi = "amenity" in tags or "shop" in tags
        if is_poi:
            rec["category"] = tags.get("amenity") or tags.get("shop")
            pois.append(rec)
        elif "building" in tags:
            buildings.append(rec)
        elif "highway" in tags:
            roads.append(rec)
    return {"pois": pois, "buildings": buildings, "roads": roads}


if __name__ == "__main__":
    data = fetch(BBOX_HEIDELBERG)
    parsed = parse_elements(data)
    print(
        f"Heidelberg bbox {BBOX_HEIDELBERG}: "
        f"{len(parsed['pois'])} POIs, "
        f"{len(parsed['buildings'])} buildings, "
        f"{len(parsed['roads'])} roads"
    )
