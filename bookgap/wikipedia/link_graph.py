# bookgap/wikipedia/link_graph.py
"""Cirrus-native Wikipedia article->article link graph.

The CirrusSearch *content* dump (``enwiki-<date>-cirrussearch-content.json.gz``)
carries the link graph natively: every namespace-0 (article) document has an
``outgoing_link`` list of linked *titles* and a ``redirect`` list of
``{namespace, title}`` entries that redirect TO that page. This module streams
that single dump and builds a directed article->article adjacency, internally
coherent with the GIST embeddings built from the SAME dump (no cross-dump
``page_id``/date mismatch, no bookgen dependency).

Title-format note
-----------------
``outgoing_link`` titles use **underscores** (``Natural_language_processing``)
while the document ``title`` and ``redirect`` titles use **spaces**
(``Natural language processing``). We normalize underscores -> spaces
everywhere so the three sources share one key space.

Namespace filtering
-------------------
The content dump contains ONLY namespace-0 documents, so the
``{title -> page_id}`` map only ever holds article titles. Resolving an
``outgoing_link`` against that map therefore drops any non-article link
(``Template:``/``Category:``/``Wikipedia:``/...) automatically — an
unresolved title is simply skipped. We additionally short-circuit on a
known MediaWiki namespace prefix to avoid wasting a dict lookup on the
~40% of out-links that are non-article; this is a speed optimization, not
a correctness requirement (titles like ``Spider-Man: ...`` or ``2001: A
Space Odyssey`` keep their colon because their prefix is not a namespace).

Provenance is pinned to the dump date (default ``20251229``), matching the
embedding store's manifest.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray

MAIN_NAMESPACE = 0
CIRRUS_DUMP_DATE = "20251229"

# Canonical MediaWiki namespace prefixes + the common aliases/talk variants
# that appear in enwiki out-links. A link whose leading ``<prefix>:`` token is
# in this set is a non-article link and is skipped before any title lookup.
# (Matching is case-insensitive; underscores already normalized to spaces.)
_NON_ARTICLE_PREFIXES = frozenset(
    p.lower()
    for p in (
        "Media", "Special",
        "Talk",
        "User", "User talk",
        "Wikipedia", "Wikipedia talk", "WP", "Project", "Project talk",
        "File", "File talk", "Image", "Image talk",
        "MediaWiki", "MediaWiki talk",
        "Template", "Template talk",
        "Help", "Help talk",
        "Category", "Category talk",
        "Portal", "Portal talk",
        "Draft", "Draft talk",
        "TimedText", "TimedText talk",
        "Module", "Module talk",
        "Gadget", "Gadget talk",
        "Gadget definition", "Gadget definition talk",
        "Book", "Book talk",
        "Education Program", "Education Program talk",
    )
)


def _norm_title(title: str) -> str:
    """Normalize a title to the shared key space: underscores -> spaces, strip.

    enwiki treats the first character as case-insensitive in principle, but the
    cirrus dump's own ``title``/``redirect`` fields and the ``outgoing_link``
    titles agree on first-letter casing for blue links, so we do NOT case-fold
    (case-folding would over-merge distinct titles like ``PH`` vs ``Ph``).
    """
    return title.replace("_", " ").strip()


def _is_non_article_link(norm_title: str) -> bool:
    """True if a (space-normalized) out-link title is a non-article namespace.

    Fast pre-filter only: a ``False`` here does not assert the title IS an
    article — final say belongs to the title/redirect resolution.
    """
    colon = norm_title.find(":")
    if colon <= 0:
        return False
    return norm_title[:colon].lower() in _NON_ARTICLE_PREFIXES


@dataclass(frozen=True)
class LinkGraph:
    """Directed article->article link graph aligned to a caller-provided index.

    ``adj`` is a ``scipy.sparse`` CSR matrix where ``adj[i, j] == 1`` means the
    article at row ``i`` links to the article at row ``j``. Rows/cols are aligned
    to ``id_list`` (``id_list[r]`` is the ``page_id`` of row ``r``), which is the
    caller-provided embedded-set ordering — so the graph lines up row-for-row
    with a ``WikipediaCorpus``/embedding matrix.
    """

    adj: sp.csr_matrix
    id_list: list[int]
    dump_date: str
    n_links_total: int          # resolved in-set edges kept (== adj.nnz pre-dedup)
    n_links_dropped: int        # out-links that did not resolve into the set


def _iter_content_docs(path: str | Path) -> Iterable[dict]:
    """Yield each namespace-0 document dict from a cirrus content dump.

    The dump alternates an action line ``{"index": {"_id": ...}}`` with the
    document line. We attach the action-line ``_id`` as ``__action_id`` so the
    caller can prefer it over the doc's own ``page_id`` (they agree in practice).
    Truncated trailing line on a partial download -> clean stop.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        pending_id: Optional[int] = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break
            action = obj.get("index")
            if action is not None:
                raw = action.get("_id")
                pending_id = int(raw) if raw is not None else None
                continue
            if obj.get("namespace") != MAIN_NAMESPACE:
                pending_id = None
                continue
            obj["__action_id"] = pending_id
            pending_id = None
            yield obj


def build_title_maps(
    dump_path: str | Path,
    limit: Optional[int] = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """One streaming pass: build ``title -> page_id`` and ``redirect -> page_id``.

    Returns ``(title_to_pid, redirect_to_pid)`` with both keys space-normalized.
    ``redirect_to_pid`` maps each redirect *source* title to the canonical
    article's ``page_id`` (the page that declares the redirect in its
    ``redirect`` field). Canonical titles win on collision: a redirect that
    happens to share a string with a real article title is overwritten by the
    article title at the end.

    ``limit`` caps the number of articles scanned (for bounded tests).
    """
    title_to_pid: dict[str, int] = {}
    redirect_to_pid: dict[str, int] = {}
    n = 0
    for doc in _iter_content_docs(dump_path):
        pid = doc.get("__action_id")
        if pid is None:
            pid = doc.get("page_id")
        if pid is None:
            continue
        pid = int(pid)
        title = doc.get("title")
        if title:
            title_to_pid[_norm_title(title)] = pid
        for rd in doc.get("redirect") or ():
            # Only NS0 redirects can resolve to an article target.
            if rd.get("namespace") not in (None, MAIN_NAMESPACE):
                continue
            rtitle = rd.get("title")
            if rtitle:
                redirect_to_pid[_norm_title(rtitle)] = pid
        n += 1
        if limit is not None and n >= limit:
            break
    # Canonical article titles take precedence over any same-string redirect.
    for t, pid in title_to_pid.items():
        if t in redirect_to_pid:
            redirect_to_pid[t] = pid
    return title_to_pid, redirect_to_pid


def load_or_build_title_maps(
    dump_path: str | Path,
    cache_path: str | Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(title_to_pid, redirect_to_pid)``, building from the dump once.

    The title/redirect maps are the single unavoidable *full-dump* pass (every
    article contributes its title + redirects), so we persist them and never
    re-stream the 43 GB gzip on subsequent runs.

    Cache format: a parquet at ``cache_path`` with columns
    ``key:str, page_id:int64, kind:str`` where ``kind`` is ``"title"`` for the
    canonical ``title_to_pid`` entries and ``"redirect"`` for the
    ``redirect_to_pid`` entries. Reconstructing the two dicts is a single linear
    scan; no dump access.

    On a cache miss, :func:`build_title_maps` streams the dump once, the result
    is written to ``cache_path`` (parent dirs created), and returned.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        tbl = pq.read_table(cache_path, columns=["key", "page_id", "kind"])
        keys = tbl.column("key").to_pylist()
        pids = tbl.column("page_id").to_pylist()
        kinds = tbl.column("kind").to_pylist()
        title_to_pid: dict[str, int] = {}
        redirect_to_pid: dict[str, int] = {}
        for k, p, kind in zip(keys, pids, kinds):
            if kind == "title":
                title_to_pid[k] = int(p)
            else:
                redirect_to_pid[k] = int(p)
        return title_to_pid, redirect_to_pid

    title_to_pid, redirect_to_pid = build_title_maps(dump_path)

    keys = list(title_to_pid.keys()) + list(redirect_to_pid.keys())
    pids = list(title_to_pid.values()) + list(redirect_to_pid.values())
    kinds = ["title"] * len(title_to_pid) + ["redirect"] * len(redirect_to_pid)
    table = pa.table(
        {
            "key": pa.array(keys, type=pa.string()),
            "page_id": pa.array(pids, type=pa.int64()),
            "kind": pa.array(kinds, type=pa.string()),
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp sibling then rename so a crash mid-write never leaves a
    # half-written cache that a later run would load as authoritative.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(cache_path)
    return title_to_pid, redirect_to_pid


def build_link_graph(
    dump_path: str | Path,
    id_list: list[int],
    *,
    title_to_pid: Optional[dict[str, int]] = None,
    redirect_to_pid: Optional[dict[str, int]] = None,
    limit: Optional[int] = None,
    dump_date: str = CIRRUS_DUMP_DATE,
) -> LinkGraph:
    """Build the directed article->article CSR over the ``id_list`` row order.

    Streams the cirrus dump twice (once for the title/redirect maps unless they
    are supplied, once for edges). For each article whose ``page_id`` is in
    ``id_list``, every ``outgoing_link`` is normalized, non-article-namespace
    links short-circuited, then resolved through ``title_to_pid`` (falling back
    to ``redirect_to_pid``). Edges whose target ``page_id`` is also in
    ``id_list`` are kept; everything else is counted as dropped.

    Args:
        dump_path: path to the cirrus content ``.json.gz``.
        id_list: the embedded set's ``page_id``s, in the desired row order
            (typically a ``WikipediaCorpus`` page-id ordering or a subset).
        title_to_pid / redirect_to_pid: precomputed maps (skip pass 1 if given).
        limit: cap articles scanned in BOTH passes (bounded tests only). Note a
            ``limit`` on the edge pass means out-links from later articles are
            never read, so use it only for small coherent slices.
        dump_date: provenance tag stored on the result.

    Returns:
        :class:`LinkGraph` with a deduplicated 0/1 CSR adjacency.
    """
    id_list = [int(p) for p in id_list]
    pid_to_row = {pid: r for r, pid in enumerate(id_list)}
    n = len(id_list)

    if title_to_pid is None or redirect_to_pid is None:
        title_to_pid, redirect_to_pid = build_title_maps(dump_path, limit=limit)

    rows: list[int] = []
    cols: list[int] = []
    n_dropped = 0

    scanned = 0
    for doc in _iter_content_docs(dump_path):
        scanned += 1
        if limit is not None and scanned > limit:
            break
        pid = doc.get("__action_id")
        if pid is None:
            pid = doc.get("page_id")
        if pid is None:
            continue
        src_row = pid_to_row.get(int(pid))
        if src_row is None:
            continue  # source article not in the embedded set
        for raw_link in doc.get("outgoing_link") or ():
            nt = _norm_title(raw_link)
            if not nt or _is_non_article_link(nt):
                continue
            tgt_pid = title_to_pid.get(nt)
            if tgt_pid is None:
                tgt_pid = redirect_to_pid.get(nt)
            if tgt_pid is None:
                continue  # links to a page outside our title universe
            tgt_row = pid_to_row.get(tgt_pid)
            if tgt_row is None:
                n_dropped += 1  # resolved, but target not in the embedded set
                continue
            if tgt_row == src_row:
                continue  # drop self-loops
            rows.append(src_row)
            cols.append(tgt_row)

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix(
        (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n, n),
    )
    # Collapse any duplicate (i, j) entries to a single 1.0.
    adj.sum_duplicates()
    adj.data[:] = 1.0

    return LinkGraph(
        adj=adj,
        id_list=id_list,
        dump_date=dump_date,
        n_links_total=int(adj.nnz),
        n_links_dropped=int(n_dropped),
    )
