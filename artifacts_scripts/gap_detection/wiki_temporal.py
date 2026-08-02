"""Track 2 wiki temporal holdout — shared machinery.

Phase 0: arrival-density pre-check on the 400k store (page_ids are
creation-ordered; calibrate id->date with a small exact-API sample).
Phase 1: creation dates — exact API first-revision timestamps for a stratified
calibration+validation sample, monotone (isotonic) id->date proxy, validated.
Phase 2: leakage-free T-corpus — every fitted object refit on <=T docs only.
Phase 3: predictors at T, arrivals embedded via the frozen T-transform,
fill-rate vs growth-matched nulls, precision@K vs realized-change instruments.

Context hygiene: processes article text internally; emits ONLY titles, counts,
aggregates, metric numbers.

Network: Wikimedia action API only (creation-date fetching), concurrency <= 20,
descriptive User-Agent, exponential backoff, every response cached to
cache/wiki/creation_dates/ so reruns are offline.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WIKI_PARQUET = REPO / "cache/wiki/store/wiki_400k/articles.parquet"
CREATION_CACHE = REPO / "cache/wiki/creation_dates"
OUT_DIR = REPO / "artifacts/gap_detection_eval"

# Wikipedia's User-Agent policy asks for a way to contact whoever is running the
# bot, so this carries a contact when one is configured rather than dropping the
# idea. Read from the environment: a literal address identified the author.
# https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
_CONTACT = os.environ.get("CONTACT_EMAIL", "")
USER_AGENT = (
    "bookgap-gap-detection-research/0.1 "
    f"(temporal-holdout creation-date calibration{f'; {_CONTACT}' if _CONTACT else ''}) "
    "python-requests"
)
API_URL = "https://en.wikipedia.org/w/api.php"
MAX_CONCURRENCY = 16


def load_store_ids() -> tuple[np.ndarray, np.ndarray]:
    """(page_ids, titles) for the full 400k store; no abstracts touched."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(WIKI_PARQUET, columns=["page_id", "title"])
    ids = tbl.column("page_id").to_numpy().astype(np.int64)
    titles = np.asarray(tbl.column("title").to_pylist(), dtype=object)
    return ids, titles


# ------------------------------------------------------------- creation-date API client
def _cache_path(page_id: int) -> Path:
    """One JSON file per page id, sharded to keep directories small."""
    shard = f"{int(page_id) % 256:02x}"
    return CREATION_CACHE / shard / f"{int(page_id)}.json"


def cached_creation(page_id: int) -> dict | None:
    p = _cache_path(page_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _write_cache(page_id: int, payload: dict) -> None:
    p = _cache_path(page_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def fetch_creation_dates(page_ids: list[int], log=print) -> dict[int, dict]:
    """First-revision timestamps via the action API (one page per call).

    Returns {page_id: record}; record fields:
      ts (ISO string or None), missing (bool), redirect (bool), title (str|None)
    Every API response is cached to disk; cached entries are never refetched.
    """
    import asyncio

    import aiohttp

    todo = [int(p) for p in page_ids if cached_creation(int(p)) is None]
    out: dict[int, dict] = {
        int(p): cached_creation(int(p))
        for p in page_ids
        if cached_creation(int(p)) is not None
    }
    if not todo:
        return out
    log(f"creation-dates: {len(out)} cached, fetching {len(todo)}")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def fetch_one(session: aiohttp.ClientSession, pid: int) -> tuple[int, dict]:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "pageids": str(pid),
            "prop": "revisions|info",
            "rvlimit": "1",
            "rvdir": "newer",
            "rvprop": "timestamp",
        }
        backoff = 1.0
        async with sem:
            for _attempt in range(7):
                try:
                    async with session.get(API_URL, params=params) as resp:
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 60.0)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                page = (data.get("query", {}).get("pages") or [{}])[0]
                rec = {
                    "page_id": pid,
                    "missing": bool(page.get("missing", False)),
                    "redirect": bool(page.get("redirect", False)),
                    "title": page.get("title"),
                    "ts": None,
                    "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                revs = page.get("revisions") or []
                if revs:
                    rec["ts"] = revs[0].get("timestamp")
                return pid, rec
            return pid, {"page_id": pid, "missing": None, "redirect": None,
                         "title": None, "ts": None, "error": "max_retries"}

    async def run() -> None:
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {"User-Agent": USER_AGENT}
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)
        async with aiohttp.ClientSession(
            timeout=timeout, headers=headers, connector=connector
        ) as session:
            done = 0
            for s in range(0, len(todo), 200):
                chunk = todo[s : s + 200]
                results = await asyncio.gather(*(fetch_one(session, p) for p in chunk))
                for pid, rec in results:
                    if rec.get("error") != "max_retries":
                        _write_cache(pid, rec)
                    out[pid] = rec
                done += len(chunk)
                if done % 1000 < 200:
                    log(f"  fetched {done}/{len(todo)}")

    asyncio.run(run())
    return out


def ts_to_days(ts: str) -> float:
    """ISO timestamp -> days since 2001-01-01 (Wikipedia epoch)."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0


def days_to_iso(days: float) -> str:
    epoch = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()
    return datetime.fromtimestamp(epoch + days * 86400.0, tz=timezone.utc).date().isoformat()


def year_to_days(year: int, month: int = 1, day: int = 1) -> float:
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0


# ------------------------------------------------------------- id->date proxy
def fit_id_date_proxy(ids: np.ndarray, days: np.ndarray):
    """Monotone (isotonic) id -> creation-days curve.

    Returns a callable ids -> predicted days. Outside the calibration range the
    curve clamps to the boundary values (callers should densify near boundaries
    of interest instead of extrapolating).
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(ids.astype(np.float64), days.astype(np.float64))

    def predict(q: np.ndarray) -> np.ndarray:
        return iso.predict(np.asarray(q, dtype=np.float64))

    return predict


def invert_date_to_id(predict, target_days: float, id_lo: int, id_hi: int) -> int:
    """Smallest id whose predicted date exceeds target_days (binary search)."""
    lo, hi = int(id_lo), int(id_hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if float(predict(np.array([mid]))[0]) >= target_days:
            hi = mid
        else:
            lo = mid + 1
    return lo


def proxy_error_report(predict, ids: np.ndarray, days: np.ndarray) -> dict:
    """Median/p90 |error| in days on a held-out set."""
    pred = predict(ids)
    err = np.abs(pred - days)
    return {
        "n": int(ids.size),
        "median_abs_err_days": float(np.median(err)),
        "p90_abs_err_days": float(np.quantile(err, 0.90)),
        "max_abs_err_days": float(err.max()) if err.size else float("nan"),
    }
