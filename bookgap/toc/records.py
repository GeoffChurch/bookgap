"""Book records for ToC derivation, read via bookgap's sanctioned corpus_dataset.

corpus_dataset is imported lazily inside read_book_records so importing this
module (or the pure _records_from_rows helper) never pulls bookgap's heavy
embedder/torch/faiss import chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class BookRecord:
    openalex_id: str
    title: str
    abstract: str
    primary_topic_id: str | None
    primary_subfield_id: str | None
    language: str | None


_COLUMNS = ["id", "display_name", "abstract",
            "primary_topic_id", "primary_subfield_id", "language"]


def _records_from_rows(rows: Iterable[Mapping[str, Any]], language: str | None) -> Iterator[BookRecord]:
    for r in rows:
        ab = r.get("abstract")
        if not ab or not ab.strip():
            continue
        lang = r.get("language")
        if language is not None and lang != language:
            continue
        yield BookRecord(
            openalex_id=r["id"],
            title=r.get("display_name") or "",
            abstract=ab,
            primary_topic_id=r.get("primary_topic_id"),
            primary_subfield_id=r.get("primary_subfield_id"),
            language=lang,
        )


def read_book_records(
    snapshot_date: str, *, domain: str = "3", language: str | None = "en",
) -> Iterator[BookRecord]:
    """Stream domain book records via corpus_dataset (lazy heavy import)."""
    from bookgap.openalex_ingest import corpus_dataset  # noqa: PLC0415 (deliberate lazy import)
    from bookgap.util import Filter

    dset = corpus_dataset(Filter(domain=domain, work_types=("book",)), snapshot_date)
    tbl = dset.to_table(columns=_COLUMNS)
    return _records_from_rows(tbl.to_pylist(), language)
