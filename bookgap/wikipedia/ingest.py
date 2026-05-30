# bookgap/wikipedia/ingest.py
"""Streaming ingest of a MediaWiki CirrusSearch *content* dump.

The CirrusSearch content dump is an Elasticsearch bulk NDJSON file with
**alternating lines**: an action line ``{"index":{"_id":"<page_id>", ...}}``
followed by the document line
``{"namespace":0,"title":...,"opening_text":...,"text":...,...}``.

We keep only ``namespace == 0`` documents (main/article namespace) and
extract ``(page_id, title, abstract)`` where ``abstract`` is the lead
paragraph (``opening_text``).

Abstract fallback: when ``opening_text`` is null/empty we fall back to the
first ~600 characters of ``text`` (CirrusSearch's flattened plain-text body),
trimmed at a word boundary. Documents with no usable text at all are skipped.

The dump may be a partial download; gzip is a streaming format so a truncated
``.gz`` is a valid prefix and ``iter_articles`` will read as many complete
records as are present, then stop (a trailing partial line is ignored).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

MAIN_NAMESPACE = 0
# Max chars of `text` used when opening_text is missing. Roughly a lead-sized
# slice; trimmed at a word boundary so we don't cut mid-token.
_TEXT_FALLBACK_CHARS = 600

# Parquet schema for the materialized articles table. page_id is int64
# (Wikipedia page ids comfortably fit), title/abstract are utf8.
ARTICLES_SCHEMA = pa.schema(
    [
        pa.field("page_id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
    ]
)


def _abstract_from_doc(doc: dict) -> Optional[str]:
    """Return the article abstract, or None if no usable text exists.

    Prefers ``opening_text`` (the lead/summary). Falls back to a leading
    slice of ``text`` trimmed at a word boundary.
    """
    opening = doc.get("opening_text")
    if opening and opening.strip():
        return opening.strip()
    text = doc.get("text")
    if text and text.strip():
        snippet = text.strip()[:_TEXT_FALLBACK_CHARS]
        # Trim back to the last whitespace to avoid a half-word, unless that
        # would discard almost everything.
        cut = snippet.rfind(" ")
        if cut > _TEXT_FALLBACK_CHARS // 2:
            snippet = snippet[:cut]
        return snippet.strip()
    return None


def iter_articles(path: str | Path, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream ``{"page_id","title","abstract"}`` dicts from a Cirrus dump.

    Only ``namespace == 0`` documents with a usable abstract are yielded.
    ``limit`` caps the number of *yielded* articles (not lines read), which is
    what you want for bounded test slices off a still-downloading prefix.
    """
    path = Path(path)
    yielded = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        pending_id: Optional[int] = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Truncated trailing line on a partial dump — stop cleanly.
                break
            action = obj.get("index")
            if action is not None:
                # Action line: remember the _id for the document that follows.
                raw_id = action.get("_id")
                pending_id = int(raw_id) if raw_id is not None else None
                continue
            # Document line.
            if obj.get("namespace") != MAIN_NAMESPACE:
                pending_id = None
                continue
            abstract = _abstract_from_doc(obj)
            if abstract is None:
                pending_id = None
                continue
            # Prefer the action-line _id; fall back to the doc's page_id.
            page_id = pending_id if pending_id is not None else obj.get("page_id")
            pending_id = None
            if page_id is None:
                continue
            title = obj.get("title") or ""
            yield {"page_id": int(page_id), "title": title, "abstract": abstract}
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def write_articles_parquet(
    src_dump: str | Path,
    out_path: str | Path,
    limit: Optional[int] = None,
    batch_rows: int = 50_000,
) -> int:
    """Materialize a ``(page_id, title, abstract)`` parquet from a Cirrus dump.

    Streams in row batches so memory stays bounded for the full ~7M-article
    dump. Returns the number of articles written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    page_ids: list[int] = []
    titles: list[str] = []
    abstracts: list[str] = []
    n_written = 0

    def flush() -> None:
        nonlocal writer, n_written
        if not page_ids:
            return
        table = pa.table(
            {"page_id": page_ids, "title": titles, "abstract": abstracts},
            schema=ARTICLES_SCHEMA,
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, ARTICLES_SCHEMA)
        writer.write_table(table)
        n_written += len(page_ids)
        page_ids.clear()
        titles.clear()
        abstracts.clear()

    try:
        for art in iter_articles(src_dump, limit=limit):
            page_ids.append(art["page_id"])
            titles.append(art["title"])
            abstracts.append(art["abstract"])
            if len(page_ids) >= batch_rows:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        # No rows at all — still write an empty, schema-correct parquet.
        pq.write_table(pa.table({"page_id": [], "title": [], "abstract": []}, schema=ARTICLES_SCHEMA), out_path)
    return n_written
