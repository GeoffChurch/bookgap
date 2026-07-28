"""Track 2 OA phase — OpenAlex data layer (the wiki-store swap).

Mirrors `wiki_temporal.py`/`data.py`'s role: provides the works, dates,
abstracts and — the OA-specific win — the DATED CITATION GRAPH, so the rest of
the Track-2 pipeline (`t_corpus.py`, `t2_outcomes.py`, the detectors) runs
unchanged.

Universe (LOCKED): OpenAlex `primary_topic.id:T10181` (NLP Techniques).

Acquisition: cursor paging on the OpenAlex polite pool — every request carries
`mailto=geoffrey.a.churchill@gmail.com` and a descriptive User-Agent, exponential
backoff on 429/5xx, and EVERY page cached to `cache/oa/nlp_t10181/` so reruns are
fully offline. Daily cap 100K calls; this needs ~1,250.

Dated citation graph (the leak-free object wiki could not give): each work's
`referenced_works ∩ universe` are outgoing edges (citing→cited). An edge is
"present at T" iff the CITING work's publication_date ≤ T. Because papers cite
only earlier work, thresholding on the citing date reconstructs the graph exactly
as it stood at T, with NO future leakage. `build_dated_graph` returns the T-graph
and the T+Δ-graph.

Context hygiene: abstracts are decoded INSIDE this module (positions→tokens) and
returned as plain strings to in-process callers ONLY; nothing here prints raw
abstract text. Emitted/logged values are titles, counts, aggregates, numbers.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import gzip
import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
OA_CACHE = REPO / "cache/oa/nlp_t10181"
OUT_DIR = REPO / "artifacts/gap_detection_eval"
CACHE_DIR = OUT_DIR / "cache"

API_URL = "https://api.openalex.org/works"
MAILTO = "geoffrey.a.churchill@gmail.com"
USER_AGENT = (
    "bookgap-gap-detection-research/0.1 "
    f"(https://github.com/gchurchill; mailto:{MAILTO}) "
    "oa-nlp-temporal-holdout cursor-paging"
)
UNIVERSE_TOPIC = "T10181"
FALLBACK_TOPICS = ("T10181", "T10028", "T10201")  # NLP + Topic Modeling + Speech
SELECT_FIELDS = (
    "id,publication_date,abstract_inverted_index,referenced_works,"
    "primary_topic,cited_by_count"
)
PER_PAGE = 200
PUB_FROM = "2008-01-01"
PUB_TO = "2019-12-31"

EPOCH = date(2008, 1, 1)  # OA-phase day epoch (universe lower bound)


# ----------------------------------------------------------------- id / date helpers
def short_id(oa_id: str | None) -> str | None:
    """`https://openalex.org/W123` (or `openalex.org/W123`, `W123`) -> `W123`."""
    if not oa_id:
        return None
    return oa_id.rsplit("/", 1)[-1]


def date_to_days(d: str | date) -> float:
    """ISO date -> days since the OA epoch (2008-01-01)."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    return float((d - EPOCH).days)


def days_to_iso(days: float) -> str:
    return (datetime(2008, 1, 1, tzinfo=timezone.utc)
            + __import__("datetime").timedelta(days=float(days))).date().isoformat()


def year_to_days(year: int, month: int = 1, day: int = 1) -> float:
    return float((date(year, month, day) - EPOCH).days)


# ----------------------------------------------------------------- cursor-paging client
def _cache_path(topic_filter: str, page_idx: int) -> Path:
    """One gzipped-JSON file per page. Sharded by a hash of the filter so the
    fallback (wider) universe caches separately from the locked one."""
    h = hashlib.sha1(topic_filter.encode()).hexdigest()[:8]
    sub = OA_CACHE / f"{topic_filter.split(':')[-1].replace('|', '_')}_{h}"
    return sub / f"page_{page_idx:05d}.json.gz"


def _write_page(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def _read_page(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fetch_universe(
    topics: tuple[str, ...] = (UNIVERSE_TOPIC,),
    pub_from: str = PUB_FROM,
    pub_to: str = PUB_TO,
    log=print,
    max_pages: int | None = None,
) -> str:
    """Cursor-page the whole T10181 (or fallback) universe to disk; return the
    cache subdir key. Resumes from the last cached page; only NEW pages hit the
    network. Each page stores the raw `meta`+`results` payload.

    The topic filter is `primary_topic.id:T10181` (or `T10181|T10028|T10201`).
    """
    import requests

    topic_filter = "primary_topic.id:" + "|".join(topics)
    filt = f"{topic_filter},from_publication_date:{pub_from},to_publication_date:{pub_to}"
    sub_key = topic_filter

    # Resume: find the highest contiguous cached page and its next_cursor.
    cursor = "*"
    page_idx = 0
    n_cached = 0
    while True:
        cached = _read_page(_cache_path(sub_key, page_idx))
        if cached is None:
            break
        nxt = cached.get("meta", {}).get("next_cursor")
        n_cached += len(cached.get("results", []))
        page_idx += 1
        cursor = nxt
        if cursor is None:
            log(f"universe fully cached: {page_idx} pages, {n_cached:,} works")
            return sub_key
    if page_idx:
        log(f"resuming from cached page {page_idx} ({n_cached:,} works cached)")

    headers = {"User-Agent": USER_AGENT}
    session = requests.Session()
    session.headers.update(headers)
    fetched = 0
    while cursor is not None:
        if max_pages is not None and page_idx >= max_pages:
            log(f"max_pages={max_pages} reached; stopping (partial)")
            break
        params = {
            "filter": filt, "select": SELECT_FIELDS,
            "per-page": str(PER_PAGE), "cursor": cursor, "mailto": MAILTO,
        }
        backoff = 1.0
        data = None
        for _attempt in range(8):
            try:
                resp = session.get(API_URL, params=params, timeout=60)
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 64.0)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:  # network / decode
                log(f"  page {page_idx} retry after error: {type(e).__name__}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 64.0)
        if data is None:
            raise RuntimeError(f"fetch_universe: page {page_idx} failed after retries")
        _write_page(_cache_path(sub_key, page_idx), data)
        got = len(data.get("results", []))
        fetched += got
        n_cached += got
        nxt = data.get("meta", {}).get("next_cursor")
        if page_idx == 0:
            log(f"universe meta.count = {data['meta']['count']:,} "
                f"(per_page {data['meta'].get('per_page')})")
        page_idx += 1
        cursor = nxt
        if page_idx % 25 == 0:
            log(f"  fetched {n_cached:,} works ({page_idx} pages); polite-pool pacing")
        time.sleep(0.05)  # gentle pacing on the polite pool
    log(f"universe acquired: {page_idx} pages, {n_cached:,} works "
        f"({fetched:,} newly fetched)")
    return sub_key


def iter_works(sub_key: str):
    """Yield raw work dicts from all cached pages of a universe (offline)."""
    page_idx = 0
    while True:
        cached = _read_page(_cache_path(sub_key, page_idx))
        if cached is None:
            break
        for w in cached.get("results", []):
            yield w
        page_idx += 1


# ----------------------------------------------------------------- language filter
# A small minority (~5%) of T10181 abstracts are non-English (mostly French). With
# stop_words="english" their function words (des, la, les, …) survive TF-IDF and
# form a perfectly-coherent LANGUAGE cluster that the coherence-max fg selector
# grabs — a language artifact, not a topical NLP gap. We keep the corpus
# English-only (documented approximation): a doc is non-English iff its
# non-English-stopword mass exceeds its English-stopword mass. Cheap, deterministic,
# no extra network field needed.
import re as _re

_EN_STOP = frozenset(
    "the and of to in a is for with that this we are on as by an be at from our can "
    "which these those it its has have was were will not but their they them there "
    "such between using based both into more most than then also each other some "
    "while where when what how all any".split()
)
_NONEN_STOP = frozenset(
    # French + a few Spanish/German/Portuguese/Italian function words (the common
    # non-English families in the slice). All absent from technical English prose.
    "des la les et le une dans nous en pour sur du est par qui que un au aux avec ce "
    "cette nos plus sont ne pas se son sa ses leur cet "  # French
    "los las del una para con como por entre sobre este esta sus "  # Spanish
    "der die und das den dem ein eine ist mit auf fuer von zu im "  # German
    "da do na no uma para com que dos das "  # Portuguese
    "il lo gli della delle nella sono per con".split()  # Italian
)
_TOKEN_RE = _re.compile(r"[a-zA-Zàâçéèêëîïôûùüÿñáíóúäöüãõ]+")


def is_english(text: str) -> bool:
    """True iff the abstract looks English (English-stopword mass >= non-English).

    Empty text -> True (a missing abstract is handled by the char filter, not here).
    """
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return True
    n = len(toks)
    en = sum(w in _EN_STOP for w in toks) / n
    non = sum(w in _NONEN_STOP for w in toks) / n
    return en >= non


# ----------------------------------------------------------------- abstract decode
def decode_abstract(inv: dict | None) -> str:
    """abstract_inverted_index {token: [positions]} -> reconstructed text.

    Returns "" when absent. Robust to duplicate positions / gaps (unlike the
    strict ingest resolver): later writes win, gaps fill with a single space.
    NEVER printed — returned to in-process vectorizers only.
    """
    if not inv:
        return ""
    pos_to_word: dict[int, str] = {}
    max_pos = -1
    for word, positions in inv.items():
        for p in positions:
            pos_to_word[p] = word
            if p > max_pos:
                max_pos = p
    if max_pos < 0:
        return ""
    return " ".join(pos_to_word.get(i, "") for i in range(max_pos + 1)).strip()


# ----------------------------------------------------------------- corpus assembly
def load_works_table(sub_key: str, log=print) -> dict:
    """Materialize parallel arrays over the universe (one row per work).

    Returns a dict of numpy arrays:
      wid (object W-ids), days (float, NaN if no date), has_date (bool),
      abstract_chars (int), cited_by (int), n_refs_universe (int placeholder 0;
      filled by build_dated_graph), row_of (W-id -> row index dict).
    Abstracts are decoded lazily by the caller via `decode_for_rows` to bound
    memory (the texts can be large); only lengths are precomputed here.
    """
    wids: list[str] = []
    days: list[float] = []
    abs_chars: list[int] = []
    cited: list[int] = []
    primary_ok: list[bool] = []
    eng_ok: list[bool] = []
    n_seen = 0
    for w in iter_works(sub_key):
        n_seen += 1
        wid = short_id(w.get("id"))
        if wid is None:
            continue
        wids.append(wid)
        pd = w.get("publication_date")
        days.append(date_to_days(pd) if pd else float("nan"))
        inv = w.get("abstract_inverted_index")
        # decode once here (length + language); texts are NOT retained (hygiene/memory)
        txt = decode_abstract(inv) if inv else ""
        abs_chars.append(len(txt))
        eng_ok.append(is_english(txt))
        cited.append(int(w.get("cited_by_count") or 0))
        pt = (w.get("primary_topic") or {}).get("id")
        primary_ok.append(short_id(pt) == UNIVERSE_TOPIC)
    wid_arr = np.asarray(wids, dtype=object)
    row_of = {w: i for i, w in enumerate(wid_arr)}
    out = {
        "wid": wid_arr,
        "days": np.asarray(days, dtype=np.float64),
        "abstract_chars": np.asarray(abs_chars, dtype=np.int64),
        "english_ok": np.asarray(eng_ok, dtype=bool),
        "cited_by": np.asarray(cited, dtype=np.int64),
        "primary_is_t10181": np.asarray(primary_ok, dtype=bool),
        "row_of": row_of,
        "n_seen": n_seen,
    }
    abstracted = out["abstract_chars"] > 0
    log(f"works table: {wid_arr.size:,} works "
        f"({int(out['days'].size - np.isnan(out['days']).sum()):,} dated, "
        f"{int(abstracted.sum()):,} abstracted, "
        f"{int((~out['english_ok'] & abstracted).sum()):,} non-English abstracts dropped, "
        f"{int(out['primary_is_t10181'].sum()):,} primary=T10181)")
    return out


def decode_for_rows(sub_key: str, rows: np.ndarray, wid: np.ndarray) -> list[str]:
    """Decode abstracts for the given universe rows, in row order.

    Builds a W-id -> text map by streaming pages once, then gathers. Memory is
    bounded by the requested row subset (texts only kept for those rows).
    """
    want = {str(wid[r]) for r in rows}
    text_of: dict[str, str] = {}
    for w in iter_works(sub_key):
        sid = short_id(w.get("id"))
        if sid in want:
            text_of[sid] = decode_abstract(w.get("abstract_inverted_index"))
    return [text_of.get(str(wid[r]), "") for r in rows]


# ----------------------------------------------------------------- dated citation graph
def build_dated_graph(
    sub_key: str, row_of: dict[str, int], wid: np.ndarray, days: np.ndarray,
    log=print,
) -> dict:
    """Intra-universe citation edges with citing-side dates (the leak-free object).

    For each citing work c, every r in (referenced_works(c) ∩ universe) is an
    edge c -> r. The edge's "time" is c's publication_date (papers cite only
    earlier work). Returns COO arrays once; callers threshold by citing date to
    get the graph at any T:

        present_at(T) := citing_days <= T

    Output dict: src, dst (row indices), citing_days (float, per edge),
    n_works, plus a helper `at(t_days)` returning a CSR adjacency (citing->cited)
    restricted to edges present at t_days (and to nodes dated <= t_days on the
    citing side; cited side may be undated but is an earlier paper by assumption).
    """
    n = wid.size
    src: list[int] = []
    dst: list[int] = []
    cdays: list[float] = []
    n_with_refs = 0
    n_edge_raw = 0
    for w in iter_works(sub_key):
        c = short_id(w.get("id"))
        ci = row_of.get(c)
        if ci is None:
            continue
        refs = w.get("referenced_works") or []
        if refs:
            n_with_refs += 1
        cd = days[ci]
        for r in refs:
            rid = short_id(r)
            ri = row_of.get(rid)
            if ri is None:
                continue  # reference outside the universe
            n_edge_raw += 1
            src.append(ci)
            dst.append(ri)
            cdays.append(cd)
    src_a = np.asarray(src, dtype=np.int64)
    dst_a = np.asarray(dst, dtype=np.int64)
    cd_a = np.asarray(cdays, dtype=np.float64)
    log(f"dated graph: {n:,} nodes, {src_a.size:,} intra-universe edges "
        f"({n_with_refs:,} works carry >=1 in-universe reference)")

    def at(t_days: float) -> sp.csr_matrix:
        m = np.isfinite(cd_a) & (cd_a <= t_days)
        return sp.csr_matrix(
            (np.ones(int(m.sum()), dtype=np.float32), (src_a[m], dst_a[m])),
            shape=(n, n),
        )

    return {"src": src_a, "dst": dst_a, "citing_days": cd_a,
            "n_works": n, "n_with_refs": n_with_refs, "at": at}


def graph_stats(adj: sp.csr_matrix, dated_node: np.ndarray) -> dict:
    """Node/edge/density stats restricted to the dated-at-T node set.

    `dated_node` (bool over all universe rows) marks nodes that EXIST at the
    snapshot (publication_date <= T). Edges are already date-filtered by `at`."""
    n_nodes = int(dated_node.sum())
    sub = adj[dated_node][:, dated_node]
    n_edges = int(sub.nnz)
    deg_out = np.asarray(sub.sum(axis=1)).ravel()
    deg_in = np.asarray(sub.sum(axis=0)).ravel()
    n_nonisolated = int(((deg_out + deg_in) > 0).sum())
    dens = n_edges / max(n_nodes * (n_nodes - 1), 1)
    return {
        "n_nodes_dated": n_nodes,
        "n_edges": n_edges,
        "n_nonisolated": n_nonisolated,
        "density": round(float(dens), 8),
        "mean_out_degree": round(float(deg_out.mean()) if n_nodes else 0.0, 3),
        "max_in_degree": int(deg_in.max()) if n_nodes else 0,
    }


if __name__ == "__main__":  # tiny offline smoke (no network) over whatever is cached
    import sys

    key = "primary_topic.id:T10181"
    n = sum(1 for _ in iter_works(key))
    print(f"cached works for {key}: {n:,}", file=sys.stderr)
