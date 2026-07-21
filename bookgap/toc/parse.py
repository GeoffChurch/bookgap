"""Turn BookRecords into neutral openalex-toc/v1 ToCs: run every profile, keep
the richest passing parse per record, dedup by normalized title."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from bookgap.toc.profiles import PROFILES, TocNodeRaw
from bookgap.toc.records import BookRecord

_SCHEMA = "openalex-toc/v1"


@dataclass(frozen=True)
class NeutralToc:
    openalex_id: str
    title: str
    primary_topic_id: str | None
    primary_subfield_id: str | None
    language: str | None
    snapshot_date: str
    parser_profile: str
    nodes: tuple[TocNodeRaw, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "openalex_id": self.openalex_id,
            "title": self.title,
            "primary_topic_id": self.primary_topic_id,
            "primary_subfield_id": self.primary_subfield_id,
            "language": self.language,
            "snapshot_date": self.snapshot_date,
            "parser_profile": self.parser_profile,
            "nodes": [_node_to_dict(n) for n in self.nodes],
        }


def _node_to_dict(n: TocNodeRaw) -> dict[str, Any]:
    return {"title": n.title, "kind": n.kind, "children": [_node_to_dict(c) for c in n.children]}


def _norm_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def derive_tocs(records: Iterable[BookRecord], snapshot_date: str) -> list[NeutralToc]:
    best_by_key: dict[str, NeutralToc] = {}
    for rec in records:
        best = None
        for profile in PROFILES.values():
            parsed = profile(rec.abstract)
            if parsed and (best is None or len(parsed.nodes) > len(best.nodes)):
                best = parsed
        if best is None or len(best.nodes) < 3:
            continue
        nt = NeutralToc(
            openalex_id=rec.openalex_id, title=rec.title,
            primary_topic_id=rec.primary_topic_id, primary_subfield_id=rec.primary_subfield_id,
            language=rec.language, snapshot_date=snapshot_date,
            parser_profile=best.profile, nodes=best.nodes,
        )
        key = _norm_key(rec.title)
        cur = best_by_key.get(key)
        if cur is None or len(nt.nodes) > len(cur.nodes):
            best_by_key[key] = nt
    return list(best_by_key.values())
