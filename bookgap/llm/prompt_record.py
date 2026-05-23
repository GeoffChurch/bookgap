"""Frozen-dataclass schemas + JSON read/write for the two durable
per-(topic, gap) artifact families:

- ``ContextRecord``: backend-agnostic gap analysis written by
  ``bookgap.llm.artifacts.context_export`` once per gap.
- ``PromptRecord``: the exact structured input handed to a backend for
  one proposal-generation call. Written by the orchestrator. The
  rendered system/user prompt strings are NOT stored — call
  ``bookgap.llm.prompt_render.render_prompts(record)`` to recompute.

Filename convention for both: ``gap_{gap_idx}_topic_{topic_id}.json``
matching the ``custom_id`` already used in ``ollama_results.jsonl``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CONTEXT_SCHEMA_VERSION = 2
PROMPT_SCHEMA_VERSION = 1


def record_filename(gap_idx: int, topic_id: str) -> str:
    return f"gap_{gap_idx}_topic_{topic_id}.json"


@dataclass(frozen=True)
class ArticleEntry:
    openalex_id: str
    display_name: str
    abstract: str


@dataclass(frozen=True)
class BookEntry:
    openalex_id: str
    display_name: str
    abstract: str
    articles_covered: int
    fraction_covered: float


@dataclass(frozen=True)
class GapStats:
    gap_size: int
    gap_size_percentile: float
    gap_zscore: float
    shapiro_log10_pval: float


@dataclass(frozen=True)
class FurthestArticle:
    openalex_id: str
    display_name: str
    abstract: str
    sqdist: float


@dataclass(frozen=True)
class ProposalCoverage:
    """How much of the gap the proposal would cover if added as a book."""
    articles_covered: int
    fraction_covered: float


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class LocalProjection2D:
    """Azimuthal equidistant projection from the seed article, with
    angular axes chosen by a green-vs-total GEP. Radii are exact ND
    distances from the seed; the gap circle drawn at radius
    ``gap_radius`` contains exactly the in-gap points.
    """
    gap_radius: float
    seed: Point2D              # always (0.0, 0.0); included for self-containedness
    proposal: Point2D
    gap_articles: list[Point2D]
    # Parallel mask to gap_articles: True if the article flips to the
    # proposal's Voronoi cell once the proposal is added as a book.
    gap_articles_flipped_to_proposal: list[bool]
    books: list[Point2D]


@dataclass(frozen=True)
class BookCoverage:
    openalex_id: str
    display_name: str
    abstract: str
    articles_covered: int
    fraction_covered: float
    articles_covered_after_proposal: int  # NEW, required


@dataclass(frozen=True)
class ContextRecord:
    topic_id: str
    topic_name: str
    gap_idx: int
    gap_stats: GapStats
    topic_totals_articles: int
    topic_totals_books: int
    articles_by_centrality: list[ArticleEntry]
    furthest_article: FurthestArticle | None
    covering_books: list[BookCoverage]
    proposal_coverage: ProposalCoverage             # NEW, required
    local_projection_2d: LocalProjection2D          # NEW, required
    schema_version: int = CONTEXT_SCHEMA_VERSION


@dataclass(frozen=True)
class PromptRecord:
    custom_id: str
    topic_id: str
    topic_name: str
    gap_idx: int
    requested_num_prompt_articles: int
    requested_num_prompt_books: int
    sampled_articles: list[ArticleEntry]
    covering_books: list[BookEntry]
    backend: str
    model: str
    generation_params: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PROMPT_SCHEMA_VERSION


def write_context_record(dirpath: Path, rec: ContextRecord) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / record_filename(rec.gap_idx, rec.topic_id)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, indent=2, ensure_ascii=False)
    return out


def write_prompt_record(dirpath: Path, rec: PromptRecord) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / record_filename(rec.gap_idx, rec.topic_id)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, indent=2, ensure_ascii=False)
    return out


def read_context_record(path: Path) -> ContextRecord:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["schema_version"] != CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"ContextRecord schema_version {d['schema_version']} "
            f"!= {CONTEXT_SCHEMA_VERSION} (file: {path})"
        )
    return ContextRecord(
        schema_version=d["schema_version"],
        topic_id=d["topic_id"],
        topic_name=d["topic_name"],
        gap_idx=d["gap_idx"],
        gap_stats=GapStats(**d["gap_stats"]),
        topic_totals_articles=d["topic_totals_articles"],
        topic_totals_books=d["topic_totals_books"],
        articles_by_centrality=[ArticleEntry(**a)
                                for a in d["articles_by_centrality"]],
        furthest_article=(FurthestArticle(**d["furthest_article"])
                          if d["furthest_article"] is not None else None),
        covering_books=[BookCoverage(**b) for b in d["covering_books"]],
        proposal_coverage=ProposalCoverage(**d["proposal_coverage"]),
        local_projection_2d=_deserialize_local_projection_2d(
            d["local_projection_2d"]
        ),
    )


def _deserialize_local_projection_2d(d: dict) -> LocalProjection2D:
    return LocalProjection2D(
        gap_radius=d["gap_radius"],
        seed=Point2D(**d["seed"]),
        proposal=Point2D(**d["proposal"]),
        gap_articles=[Point2D(**p) for p in d["gap_articles"]],
        gap_articles_flipped_to_proposal=list(
            d["gap_articles_flipped_to_proposal"]
        ),
        books=[Point2D(**p) for p in d["books"]],
    )


def read_prompt_record(path: Path) -> PromptRecord:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["schema_version"] != PROMPT_SCHEMA_VERSION:
        raise ValueError(
            f"PromptRecord schema_version {d['schema_version']} "
            f"!= {PROMPT_SCHEMA_VERSION} (file: {path})"
        )
    return PromptRecord(
        schema_version=d["schema_version"],
        custom_id=d["custom_id"],
        topic_id=d["topic_id"],
        topic_name=d["topic_name"],
        gap_idx=d["gap_idx"],
        requested_num_prompt_articles=d["requested_num_prompt_articles"],
        requested_num_prompt_books=d["requested_num_prompt_books"],
        sampled_articles=[ArticleEntry(**a) for a in d["sampled_articles"]],
        covering_books=[BookEntry(**b) for b in d["covering_books"]],
        backend=d["backend"],
        model=d["model"],
        generation_params=d["generation_params"],
    )
