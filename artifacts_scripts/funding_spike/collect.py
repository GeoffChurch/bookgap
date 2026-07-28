"""Funding-WASTE spike: collect NIH RePORTER R01 grants (FY2013-2016),
their linked publications, and iCite citation counts. Read-only, cached.

Inverse-gap framing: gap score Delta = log(structural) - log(operational).
Funding WASTE = operational HIGH (dollars awarded) + structural LOW (research
footprint: linked pubs, citations, RCR). We want to know if "funded but
structurally invisible" survives the obvious confounds (grant age, mechanism),
analogous to the size-confound the gap-detection work controls for.

This module ONLY collects + caches. Analysis lives in analyze.py.

Regenerate:
    uv run \
        python -m artifacts_scripts.funding_spike.collect
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

PROJECTS_URL = "https://api.reporter.nih.gov/v2/projects/search"
PUBS_URL = "https://api.reporter.nih.gov/v2/publications/search"
ICITE_URL = "https://icite.od.nih.gov/api/pubs"

FISCAL_YEARS = [2013, 2014, 2015, 2016]
ACTIVITY_CODES = ["R01"]
# Modest sample: cap distinct core projects we deep-dive (pubs+citations).
MAX_CORE_PROJECTS = 3000
# Rows to pull per FY. RePORTER deep-pagination caps at offset+limit<=15000, and
# per-FY R01 counts are ~25k, so we take a dollar-NEUTRAL slice (sorted by
# core_project_num) up to this cap. Crucially we do NOT sort/truncate by award:
# the smallest grants are exactly the waste-signal candidates we must keep.
ROWS_PER_FY_CAP = 15000
PAGE = 500  # RePORTER max ~500
RANDOM_SEED = 1234  # for reproducible deep-dive subsample
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bookgap-funding-spike/0.1 (read-only research)"})


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _post(url, body, tries=5):
    for i in range(tries):
        try:
            r = SESSION.post(url, json=body, timeout=90)
            if r.status_code == 429:
                wait = 2.0 * (i + 1)
                _log(f"  429, backoff {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 1.5 * (i + 1)
            _log(f"  POST err {e!r}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"POST {url} failed after {tries} tries")


def _get(url, params, tries=5):
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=90)
            if r.status_code == 429:
                wait = 2.0 * (i + 1)
                _log(f"  429, backoff {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 1.5 * (i + 1)
            _log(f"  GET err {e!r}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {tries} tries")


# ---------------------------------------------------------------- projects ----
def fetch_projects() -> list[dict]:
    """All R01 fiscal-year rows for FISCAL_YEARS. One row per (grant, FY).
    award_amount is per-FY-row, so we keep every row and aggregate later."""
    cache_f = CACHE / "projects_rows.json"
    if cache_f.exists():
        rows = json.loads(cache_f.read_text())
        _log(f"projects: loaded {len(rows)} cached rows")
        return rows

    fields = [
        "CoreProjectNum", "AwardAmount", "FiscalYear", "ActivityCode",
        "ProjectTitle", "ProjectNum", "ProjectStartDate", "ProjectEndDate",
        "AwardNoticeDate",
    ]
    rows: list[dict] = []
    seen_core: set[str] = set()
    for fy in FISCAL_YEARS:
        offset = 0
        total = None
        while True:
            body = {
                "criteria": {"fiscal_years": [fy], "activity_codes": ACTIVITY_CODES},
                "offset": offset, "limit": PAGE,
                "include_fields": fields,
                # Dollar-NEUTRAL stable sort. We keep the full dollar range so the
                # smallest (most-likely-underperforming) grants survive.
                "sort_field": "core_project_num", "sort_order": "asc",
            }
            j = _post(PROJECTS_URL, body)
            meta = j.get("meta", {})
            if total is None:
                total = meta.get("total")
                _log(f"FY{fy}: total rows={total}")
            res = j.get("results", [])
            if not res:
                break
            for k in res:
                rows.append({
                    "core_project_num": k.get("core_project_num"),
                    "award_amount": k.get("award_amount"),
                    "fiscal_year": k.get("fiscal_year"),
                    "activity_code": k.get("activity_code"),
                    "project_title": k.get("project_title"),
                    "project_num": k.get("project_num"),
                    "project_start_date": k.get("project_start_date"),
                    "project_end_date": k.get("project_end_date"),
                    "award_notice_date": k.get("award_notice_date"),
                })
                seen_core.add(k.get("core_project_num"))
            offset += PAGE
            if offset >= ROWS_PER_FY_CAP:
                _log(f"FY{fy}: reached {ROWS_PER_FY_CAP} pagination cap")
                break
            if offset >= (total or 0):
                break
            time.sleep(0.15)
    cache_f.write_text(json.dumps(rows))
    _log(f"projects: fetched {len(rows)} rows, {len(seen_core)} distinct core projects")
    return rows


def aggregate_grants(rows: list[dict]) -> dict[str, dict]:
    """Aggregate fiscal-year rows -> one record per core_project_num."""
    g: dict[str, dict] = {}
    for r in rows:
        cpn = r["core_project_num"]
        if cpn is None:
            continue
        rec = g.setdefault(cpn, {
            "core_project_num": cpn,
            "total_award": 0.0,
            "fiscal_years": set(),
            "title": r.get("project_title"),
            "start_date": r.get("project_start_date"),
            "end_date": r.get("project_end_date"),
            "project_nums": set(),
            "activity_code": r.get("activity_code"),
        })
        amt = r.get("award_amount") or 0
        rec["total_award"] += amt
        if r.get("fiscal_year"):
            rec["fiscal_years"].add(r["fiscal_year"])
        if r.get("project_num"):
            rec["project_nums"].add(r["project_num"])
        # earliest start / latest end across rows
        if r.get("project_start_date") and (rec["start_date"] is None or r["project_start_date"] < rec["start_date"]):
            rec["start_date"] = r["project_start_date"]
        if r.get("project_end_date") and (rec["end_date"] is None or r["project_end_date"] > rec["end_date"]):
            rec["end_date"] = r["project_end_date"]
    # finalize
    for rec in g.values():
        rec["fiscal_years"] = sorted(rec["fiscal_years"])
        rec["first_fy"] = min(rec["fiscal_years"]) if rec["fiscal_years"] else None
        rec["n_fy_rows"] = len(rec["fiscal_years"])
        # application type: derive from project_num prefix digit (1=new,2=renewal,
        # 3=supplement,5=noncompeting continuation,7=transfer). A core project
        # that ever appears as type 3 (admin supplement) is flagged.
        types = set()
        for pn in rec["project_nums"]:
            # project_num like '5R01DK089201-03'
            lead = pn[0] if pn and pn[0].isdigit() else None
            if lead:
                types.add(lead)
        rec["appl_types"] = sorted(types)
        rec["ever_supplement"] = "3" in types
        rec["project_nums"] = sorted(rec["project_nums"])
    return g


# ------------------------------------------------------------ publications ----
def fetch_pubs(core_nums: list[str]) -> dict[str, list[int]]:
    """core_project_num -> list of PMIDs. Cached as a single JSON map.
    publications/search accepts a LIST of core_project_nums per call."""
    cache_f = CACHE / "pubs_by_core.json"
    cached: dict[str, list[int]] = {}
    if cache_f.exists():
        cached = json.loads(cache_f.read_text())
    todo = [c for c in core_nums if c not in cached]
    _log(f"pubs: {len(cached)} cached, {len(todo)} to fetch")
    BATCH = 50  # core projects per query
    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        # paginate within the batch (a batch can have many pubs)
        per_core: dict[str, list[int]] = {c: [] for c in batch}
        offset = 0
        while True:
            body = {"criteria": {"core_project_nums": batch},
                    "offset": offset, "limit": PAGE,
                    "sort_field": "coreproject", "sort_order": "desc"}
            j = _post(PUBS_URL, body)
            res = j.get("results", [])
            if not res:
                break
            for p in res:
                cp = p.get("coreproject")
                pm = p.get("pmid")
                if cp in per_core and pm is not None:
                    per_core[cp].append(int(pm))
            total = j.get("meta", {}).get("total", 0)
            offset += PAGE
            if offset >= total:
                break
            time.sleep(0.15)
        for c, pmids in per_core.items():
            cached[c] = sorted(set(pmids))
        if (bi // BATCH) % 5 == 0:
            cache_f.write_text(json.dumps(cached))
            _log(f"pubs: {bi + len(batch)}/{len(todo)} cores done")
        time.sleep(0.15)
    cache_f.write_text(json.dumps(cached))
    _log(f"pubs: total {len(cached)} cores, "
         f"{sum(len(v) for v in cached.values())} pmid links")
    return cached


# ------------------------------------------------------------------ icite ----
def fetch_icite(pmids: list[int]) -> dict[int, dict]:
    """PMID -> {citation_count, relative_citation_ratio, year}. Cached."""
    cache_f = CACHE / "icite_by_pmid.json"
    cached: dict[str, dict] = {}
    if cache_f.exists():
        cached = json.loads(cache_f.read_text())
    uniq = sorted({int(p) for p in pmids})
    todo = [p for p in uniq if str(p) not in cached]
    _log(f"icite: {len(cached)} cached, {len(todo)} to fetch (of {len(uniq)} uniq)")
    BATCH = 200
    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        params = {"pmids": ",".join(str(p) for p in batch)}
        j = _get(ICITE_URL, params)
        for d in j.get("data", []) or []:
            pm = d.get("pmid")
            if pm is None:
                continue
            cached[str(pm)] = {
                "citation_count": d.get("citation_count"),
                "relative_citation_ratio": d.get("relative_citation_ratio"),
                "year": d.get("year"),
                "is_research_article": d.get("is_research_article"),
            }
        if (bi // BATCH) % 5 == 0:
            cache_f.write_text(json.dumps(cached))
            _log(f"icite: {bi + len(batch)}/{len(todo)} pmids done")
        time.sleep(0.2)
    cache_f.write_text(json.dumps(cached))
    _log(f"icite: total {len(cached)} pmids cached")
    return {int(k): v for k, v in cached.items()}


# ------------------------------------------------------------------ build ----
def build_dataset() -> Path:
    rows = fetch_projects()
    grants = aggregate_grants(rows)
    _log(f"aggregated {len(grants)} distinct core projects")

    # Deep-dive a dollar-NEUTRAL random subsample so dollars AND output vary
    # naturally (a top-by-award selection would compress the dollar range and
    # bias the regression). Reproducible via RANDOM_SEED. Only keep grants with
    # positive total award.
    import random
    pool = [r for r in grants.values() if r["total_award"] and r["total_award"] > 0]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)
    selected = pool[:MAX_CORE_PROJECTS]
    _log(f"selected random {len(selected)} of {len(pool)} grants for deep-dive")

    core_nums = [r["core_project_num"] for r in selected]
    pubs = fetch_pubs(core_nums)

    all_pmids: list[int] = []
    for c in core_nums:
        all_pmids.extend(pubs.get(c, []))
    icite = fetch_icite(all_pmids)

    # assemble per-grant output record
    out = []
    for r in selected:
        pmids = pubs.get(r["core_project_num"], [])
        n_pub = len(pmids)
        cites = [icite.get(p, {}).get("citation_count") or 0 for p in pmids]
        rcrs = [icite.get(p, {}).get("relative_citation_ratio") for p in pmids]
        rcrs = [x for x in rcrs if x is not None]
        years = [icite.get(p, {}).get("year") for p in pmids if icite.get(p, {}).get("year")]
        out.append({
            "core_project_num": r["core_project_num"],
            "title": r["title"],
            "total_award": r["total_award"],
            "fiscal_years": r["fiscal_years"],
            "first_fy": r["first_fy"],
            "n_fy_rows": r["n_fy_rows"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "appl_types": r["appl_types"],
            "ever_supplement": r["ever_supplement"],
            "n_pub": n_pub,
            "sum_citations": int(sum(cites)),
            "sum_rcr": float(sum(rcrs)) if rcrs else 0.0,
            "max_pub_year": max(years) if years else None,
            "min_pub_year": min(years) if years else None,
        })

    out_f = CACHE / "dataset.json"
    out_f.write_text(json.dumps(out, indent=2))
    _log(f"wrote {out_f} with {len(out)} grants")
    return out_f


if __name__ == "__main__":
    build_dataset()


# ---------------------------------------------------------------------------
# PROVENANCE: regenerate (do not hand-edit cache/dataset.json) with
#   uv run \
#       python -m artifacts_scripts.funding_spike.collect
# Read-only NIH RePORTER + iCite collector. Caches under cache/. See analyze.py
# for the validity analysis that consumes cache/dataset.json.
# ---------------------------------------------------------------------------
