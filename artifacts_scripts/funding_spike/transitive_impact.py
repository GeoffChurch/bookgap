"""Funding-WASTE spike, FOLLOW-UP: TRANSITIVE (forward-Katz) output measure.

The original spike (collect.py + analyze.py) measured each grant's research
output three DIRECT ways -- publication count, summed direct citations
(iCite citation_count), summed RCR -- and found NO high-dollar/low-output
"waste" population (raw dollars->output positive; age-controlled partial
non-negative; flattest direct measure was summed citations at +0.03 n.s.).

This module adds the ONE output direction the original skipped: a TRANSITIVE,
PageRank-style score where a paper is credited for being cited *by* impactful
work -- influence flowing through the forward citation graph. It then re-runs
the IDENTICAL analysis (raw Pearson + age-controlled partial, same controls)
so the transitive measure sits in one table alongside pubs / citations / RCR.

REUSE: this reads the SAME cache/dataset.json grant set, dollars, age, FY,
duration, and the SAME cache/pubs_by_core.json linked-PMID lists. It does NOT
re-sample. It only ADDS an output column.

------------------------------------------------------------------- algorithm
Edge source: iCite `cited_by` (PMIDs citing a paper) -- same data source as the
original spike, cheapest path. The original cache stored only 4 iCite fields
(no cited_by), so we re-fetch FULL iCite records for the seed PMIDs and for the
hop-1 PMIDs, caching cited_by lists under cache/cited_by/.

Transitive score per grant (truncated forward Katz, 2 hops, beta=0.5):

    score(g) = |H0| + beta * |H1| + beta^2 * |H2|

  H0 = the grant's own papers (seeds)
  H1 = distinct papers citing H0       (1 forward hop)
  H2 = distinct papers citing H1        (2 forward hops)

Using distinct REACHABLE SETS (a paper counts once, at its shortest hop) -- the
standard bounded-Katz / forward-influence formulation. It is genuinely
transitive: an indirectly-citing paper (2 hops) contributes via the beta^2
term. Seeded per grant, aggregated to one number per grant.

BOUNDS (all logged; silent truncation is not acceptable):
  * FANOUT_CAP citers kept per paper (deterministic: smallest PMIDs first).
    Bounds giant hubs (one seed has 13,587 citers).
  * 2-hop truncation (beta^2 term); beta = BETA.
  * HOP1_UNIVERSE_CAP on the global DISTINCT hop-1 PMID set expanded for hop-2.
    Selected deterministically (smallest PMIDs first). Grants whose hop-1 papers
    fall outside the expanded universe still get their H2 contribution from
    whatever IS expanded; the shortfall is logged per-grant and in aggregate.

Regenerate:
    uv run \
        python -m artifacts_scripts.funding_spike.transitive_impact
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from scipy import stats

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
EDGE_DIR = CACHE / "cited_by"
EDGE_DIR.mkdir(exist_ok=True, parents=True)

ICITE_URL = "https://icite.od.nih.gov/api/pubs"

# ---- transitive-score knobs (all logged in the report) ----
BETA = 0.5                  # Katz decay per hop
FANOUT_CAP = 100            # max citers kept per paper (deterministic)
HOP1_UNIVERSE_CAP = 150_000 # max distinct hop-1 PMIDs expanded for hop-2
ICITE_BATCH = 200
DATA_REF_YEAR = 2026        # match analyze.py
# Full 2-hop over all 3000 grants needs ~3M citer fetches (hours, hammers iCite).
# Subsample grants for a tractable PROPER 2-hop measure; None = all grants.
N_GRANT_SUBSAMPLE = 250
SUBSAMPLE_SEED = 20260615

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bookgap-funding-spike/0.1 (read-only research)"})


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _get(url, params, tries=6):
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


# ---------------------------------------------------------- cited_by fetch ----
def _shard_path(pmid: int) -> Path:
    # 256 shards by pmid % 256 so no single JSON gets huge; each shard maps
    # str(pmid) -> capped sorted list of citing PMIDs.
    return EDGE_DIR / f"cb_{pmid % 256:03d}.json"


def _load_all_shards() -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for f in sorted(EDGE_DIR.glob("cb_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in d.items():
            out[int(k)] = v
    return out


def fetch_cited_by(pmids: list[int], label: str) -> dict[int, list[int]]:
    """For each pmid, fetch iCite cited_by, cap at FANOUT_CAP (smallest PMIDs
    first, deterministic), and persist into mod-256 shards. Returns the map for
    the requested pmids (loading already-cached ones from shards)."""
    uniq = sorted({int(p) for p in pmids})
    # load existing shards once
    have = _load_all_shards()
    todo = [p for p in uniq if p not in have]
    _log(f"[{label}] cited_by: {len(uniq):,} requested, "
         f"{len(uniq) - len(todo):,} cached, {len(todo):,} to fetch")

    # group todo by shard so we batch writes
    dirty: dict[int, dict[str, list[int]]] = {}
    n_capped = 0
    fetched = 0
    for bi in range(0, len(todo), ICITE_BATCH):
        batch = todo[bi:bi + ICITE_BATCH]
        j = _get(ICITE_URL, {"pmids": ",".join(str(p) for p in batch)})
        returned = {int(d["pmid"]): d for d in (j.get("data") or []) if d.get("pmid") is not None}
        for p in batch:
            rec = returned.get(p)
            cb = rec.get("cited_by") if rec else None
            if not cb:
                lst: list[int] = []
            else:
                lst = sorted(int(x) for x in cb)
                if len(lst) > FANOUT_CAP:
                    lst = lst[:FANOUT_CAP]
                    n_capped += 1
            sh = p % 256
            dirty.setdefault(sh, {})[str(p)] = lst
            have[p] = lst
            fetched += 1
        if (bi // ICITE_BATCH) % 25 == 0:
            _flush_dirty(dirty)
            dirty = {}
            _log(f"[{label}] cited_by: {bi + len(batch):,}/{len(todo):,} fetched "
                 f"({n_capped:,} hit FANOUT_CAP={FANOUT_CAP})")
        time.sleep(0.2)
    _flush_dirty(dirty)
    _log(f"[{label}] cited_by: done. fetched {fetched:,}, {n_capped:,} capped at {FANOUT_CAP}")
    return {p: have.get(p, []) for p in uniq}


def _flush_dirty(dirty: dict[int, dict[str, list[int]]]):
    for sh, add in dirty.items():
        f = EDGE_DIR / f"cb_{sh:03d}.json"
        cur = {}
        if f.exists():
            try:
                cur = json.loads(f.read_text())
            except Exception:
                cur = {}
        cur.update(add)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur))
        tmp.replace(f)


# --------------------------------------------------------- build transitive ---
def build_transitive() -> tuple[dict, dict]:
    """Returns (per-grant transitive scores, caps/coverage log)."""
    dataset = json.loads((CACHE / "dataset.json").read_text())
    pubs = json.loads((CACHE / "pubs_by_core.json").read_text())

    # seeds per grant (H0)
    grant_seeds = {r["core_project_num"]: [int(x) for x in pubs.get(r["core_project_num"], [])]
                   for r in dataset}
    if N_GRANT_SUBSAMPLE and len(grant_seeds) > N_GRANT_SUBSAMPLE:
        rng = random.Random(SUBSAMPLE_SEED)
        keep = set(rng.sample(sorted(grant_seeds), N_GRANT_SUBSAMPLE))
        grant_seeds = {k: v for k, v in grant_seeds.items() if k in keep}
        _log(f"SUBSAMPLE: {len(grant_seeds)} grants (seed={SUBSAMPLE_SEED}) for tractable full 2-hop")
    all_seeds = sorted({s for ss in grant_seeds.values() for s in ss})
    _log(f"grants={len(grant_seeds)}  unique seed PMIDs (H0)={len(all_seeds):,}")

    # ---- fetch cited_by for all seeds -> gives every grant's H1 ----
    seed_cb = fetch_cited_by(all_seeds, "seeds")

    # global distinct hop-1 universe
    hop1_all: set[int] = set()
    for s in all_seeds:
        hop1_all.update(seed_cb.get(s, []))
    hop1_all.discard(0)
    _log(f"distinct hop-1 PMIDs (H1 universe, post-FANOUT_CAP)={len(hop1_all):,}")

    # ---- cap the hop-1 universe we EXPAND for hop-2 (RANDOM, not smallest-PMID:
    #      smallest-first biases toward old papers and would systematically drop
    #      recent citers) ----
    if len(hop1_all) > HOP1_UNIVERSE_CAP:
        rng = random.Random(SUBSAMPLE_SEED + 1)
        expanded = set(rng.sample(sorted(hop1_all), HOP1_UNIVERSE_CAP))
    else:
        expanded = set(hop1_all)
    n_truncated_universe = len(hop1_all) - len(expanded)
    _log(f"hop-1 universe cap={HOP1_UNIVERSE_CAP:,}: expanding {len(expanded):,}, "
         f"truncating {n_truncated_universe:,} for hop-2 (random sample)")

    # ---- fetch cited_by for the expanded hop-1 set -> gives H2 ----
    hop1_cb = fetch_cited_by(sorted(expanded), "hop1") if expanded else {}

    # ---- per-grant Katz score on distinct reachable sets ----
    scores: dict[str, dict] = {}
    grants_with_h2_shortfall = 0
    for cpn, seeds in grant_seeds.items():
        H0 = set(seeds)
        H1: set[int] = set()
        for s in seeds:
            H1.update(seed_cb.get(s, []))
        H1.discard(0)
        H1 -= H0  # shortest-hop dedupe: a seed that also cites a seed counts at H0

        # which of this grant's H1 papers were expanded for hop-2?
        h1_expandable = H1 & expanded
        n_h1_unexpanded = len(H1) - len(h1_expandable)

        H2: set[int] = set()
        for p in h1_expandable:
            H2.update(hop1_cb.get(p, []))
        H2.discard(0)
        H2 -= H0
        H2 -= H1  # shortest-hop dedupe

        if n_h1_unexpanded > 0:
            grants_with_h2_shortfall += 1

        katz = len(H0) + BETA * len(H1) + (BETA ** 2) * len(H2)
        scores[cpn] = {
            "h0": len(H0), "h1": len(H1), "h2": len(H2),
            "h1_unexpanded": n_h1_unexpanded,
            "transitive_katz": float(katz),
        }

    caps_log = {
        "beta": BETA,
        "n_grant_subsample": N_GRANT_SUBSAMPLE,
        "subsample_seed": SUBSAMPLE_SEED,
        "fanout_cap_per_paper": FANOUT_CAP,
        "hop1_universe_cap": HOP1_UNIVERSE_CAP,
        "n_seed_pmids_H0": len(all_seeds),
        "n_distinct_hop1_PMIDs": len(hop1_all),
        "n_hop1_expanded_for_hop2": len(expanded),
        "n_hop1_truncated_from_universe": n_truncated_universe,
        "frac_hop1_universe_expanded": (len(expanded) / max(1, len(hop1_all))),
        "n_grants_with_any_h2_shortfall": grants_with_h2_shortfall,
        "edge_source": "iCite cited_by (forward citation graph)",
        "hops": 2,
        "dedupe": "distinct reachable sets, shortest-hop (paper counted once)",
    }
    _log(f"caps/coverage: {json.dumps(caps_log, indent=2)}")
    (CACHE / "transitive_scores.json").write_text(json.dumps(scores, indent=2))
    return scores, caps_log


# ----------------------------------------------------------------- analysis ---
def _start_year(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "")).year
    except Exception:
        return None


def partial_corr(x, y, z):
    """Identical to analyze.py.partial_corr."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    r, p = stats.pearsonr(rx, ry)
    return r, p, rx, ry


def analyze(scores: dict, caps_log: dict):
    rows = json.loads((CACHE / "dataset.json").read_text())
    n_all = len(rows)

    # derive grant age EXACTLY as analyze.py does
    for r in rows:
        sy = _start_year(r.get("start_date")) or r.get("first_fy")
        r["start_year"] = sy
        r["age"] = (DATA_REF_YEAR - sy) if sy else None
        sc = scores.get(r["core_project_num"], {})
        r["transitive_katz"] = sc.get("transitive_katz", 0.0)
        r["_h0"] = sc.get("h0", 0)
        r["_h1"] = sc.get("h1", 0)
        r["_h2"] = sc.get("h2", 0)

    # SAME filter as analyze.py: usable start year + positive dollars; AND
    # restrict to the grant subsample we computed transitive scores for, so all
    # four output measures are compared on the IDENTICAL n.
    rows = [r for r in rows if r["age"] is not None and r["total_award"] and r["total_award"] > 0
            and r["core_project_num"] in scores]
    n_used = len(rows)

    dollars = np.array([r["total_award"] for r in rows], float)
    n_pub = np.array([r["n_pub"] for r in rows], float)
    cites = np.array([r["sum_citations"] for r in rows], float)
    rcr = np.array([r["sum_rcr"] for r in rows], float)
    trans = np.array([r["transitive_katz"] for r in rows], float)
    age = np.array([r["age"] for r in rows], float)
    first_fy = np.array([r["first_fy"] for r in rows], float)
    n_fy = np.array([r["n_fy_rows"] for r in rows], float)

    ld = np.log(dollars + 1)
    controls = [age, first_fy, n_fy]  # SAME controls as analyze.py

    outputs = {
        "n_pub": n_pub,
        "sum_citations": cites,
        "sum_rcr": rcr,
        "transitive_katz": trans,
    }

    table = {}
    for name, y in outputs.items():
        ly = np.log(y + 1)
        r_raw, p_raw = stats.pearsonr(ld, ly)
        r_par, p_par, _, _ = partial_corr(ld, ly, controls)
        table[name] = {
            "raw_pearson_logdollar_logoutput": [float(r_raw), float(p_raw)],
            "partial_corr_dollar_output_given_age_fy_dur": [float(r_par), float(p_par)],
            "frac_zero_output": float((y == 0).mean()),
            "n_zero_output": int((y == 0).sum()),
        }

    # high-$ / low-transitive face validity: lowest transitive among the top
    # dollar tercile (so "high $" is real), then the simplest flag: biggest $
    # among zero-transitive grants.
    dollar_hi = dollars >= np.percentile(dollars, 66.7)
    idx_hi = np.where(dollar_hi)[0]
    idx_hi_sorted = idx_hi[np.argsort(trans[idx_hi])]  # lowest transitive first

    zero_trans_idx = [i for i in range(len(rows)) if trans[i] == 0]
    zero_trans_idx.sort(key=lambda i: -dollars[i])

    def brief(i):
        r = rows[i]
        return {
            "core_project_num": r["core_project_num"],
            "total_award": int(r["total_award"]),
            "first_fy": r["first_fy"],
            "age": int(r["age"]),
            "n_fy_rows": r["n_fy_rows"],
            "n_pub": int(r["n_pub"]),
            "sum_citations": int(r["sum_citations"]),
            "sum_rcr": round(r["sum_rcr"], 2),
            "transitive_katz": round(r["transitive_katz"], 2),
            "h0_h1_h2": [r["_h0"], r["_h1"], r["_h2"]],
            "title": (r["title"] or "")[:110],
        }

    report = {
        "n_all_grants": n_all,
        "n_used_grants": n_used,
        "transitive_caps_log": caps_log,
        "controls": ["age", "first_fy", "n_fy_rows"],
        "ref_year": DATA_REF_YEAR,
        "comparison_table": table,
        "transitive_summary": {
            "min": float(trans.min()), "median": float(np.median(trans)),
            "mean": float(trans.mean()), "max": float(trans.max()),
        },
        "high_dollar_low_transitive_top": [brief(i) for i in idx_hi_sorted[:8]],
        "top_zero_transitive_by_dollars": [brief(i) for i in zero_trans_idx[:8]],
    }
    out_f = CACHE / "transitive_analysis.json"
    out_f.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    scores, caps_log = build_transitive()
    analyze(scores, caps_log)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# PROVENANCE: regenerate (do not hand-edit cache/transitive_*.json) with
#   uv run \
#       python -m artifacts_scripts.funding_spike.transitive_impact
# Reuses cache/dataset.json + cache/pubs_by_core.json from collect.py (SAME
# grant set / dollars / age / FY / duration / linked PMIDs). Adds ONLY a
# transitive forward-Katz output measure (iCite cited_by edges, 2 hops,
# beta=0.5, fan-out cap 100, hop-1 universe cap 150k random, grant subsample 250
# for tractability -- all logged in cache/transitive_analysis.json). Caches
# citation edges under cache/cited_by/ (mod-256 shards). Re-runs the identical
# raw-Pearson + age-controlled-partial analysis from analyze.py so the
# transitive measure sits alongside pubs / citations / RCR.
# ---------------------------------------------------------------------------
