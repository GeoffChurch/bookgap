"""CLI: read domain book records -> derive ToCs -> write neutral JSON corpus.

The real corpus_dataset read happens only in main() via read_book_records; the
pure write_corpus/argparser are unit-tested without touching the cache.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bookgap.toc.parse import NeutralToc, derive_tocs
from bookgap.toc.profiles import PROFILES
from bookgap.toc.records import read_book_records

_DEFAULT_SNAPSHOT = "2025-11-06"


def write_corpus(tocs: Iterable[NeutralToc], out_dir: Path) -> dict[str, Any]:
    """Write the corpus as one JSONL (one NeutralToc per line) + manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_profile: Counter[str] = Counter()
    by_topic: Counter[str] = Counter()
    total = 0
    with (out_dir / "tocs.jsonl").open("w", encoding="utf-8") as fh:
        for nt in tocs:
            fh.write(json.dumps(nt.to_dict(), ensure_ascii=False) + "\n")
            by_profile[nt.parser_profile] += 1
            by_topic[nt.primary_topic_id or "unknown"] += 1
            total += 1
    manifest = {"total": total, "by_profile": dict(by_profile), "by_topic": dict(by_topic)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Harvest OpenAlex book ToCs -> neutral JSON corpus.")
    ap.add_argument("--domain", default="3")
    ap.add_argument("--snapshot", default=_DEFAULT_SNAPSHOT)
    ap.add_argument("--language", default="en", help="'none' disables the language filter")
    ap.add_argument("--profiles", default="chapter-n,arabic-list,roman-list")
    ap.add_argument("--out", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    ns = build_argparser().parse_args(argv)
    unknown = [p for p in ns.profiles.split(",") if p not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profiles: {unknown}; known: {sorted(PROFILES)}")
    language = None if ns.language == "none" else ns.language
    records = read_book_records(ns.snapshot, domain=ns.domain, language=language)
    tocs = derive_tocs(records, ns.snapshot)
    manifest = write_corpus(tocs, Path(ns.out))
    print(f"harvested {manifest['total']} ToCs -> {ns.out}")
    print(f"  by profile: {manifest['by_profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
