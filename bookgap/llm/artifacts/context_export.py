"""Per-(topic, gap) context export: writes one v2 ContextRecord JSON per
gap into ``<output_dir>/contexts/``.

v2 requires per-gap proposal-coverage analysis (the proposal's projected
embedding, its book-set companions, and the gap article embeddings). The
schema captures the full gap composition (all articles ranked by
centrality, all covering books ranked by descending coverage), plus the
post-proposal coverage counts and a 2D AEP projection for visualization.

Filename: ``gap_{gap_idx}_topic_{topic_id}.json`` matching the
``custom_id`` already used in ``ollama_results.jsonl``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bookgap.llm.artifacts.common import group_gap_articles_by_covering_books
from bookgap.llm.coverage_analysis import (
    compute_post_proposal_coverage,
    compute_local_projection_2d,
)
from bookgap.llm.data_structures import GapWithContext
from bookgap.llm.prompt_record import (
    ArticleEntry, BookCoverage, ContextRecord, FurthestArticle, GapStats,
    LocalProjection2D, Point2D, ProposalCoverage,
    write_context_record,
)
from bookgap.util import LOGGER


def save_gap_context(
    gap_with_context: GapWithContext,
    output_dir: Path,
    *,
    proposal_analysis: dict[str, Any],
) -> Path:
    """Build a v2 ContextRecord and write it to
    ``<output_dir>/contexts/gap_{gap_idx}_topic_{topic_id}.json``.

    ``proposal_analysis`` keys (all required):
      * ``gap_article_embeddings``: NDArray (gap_size, D) of in-gap article
        embeddings, in the SAME order as
        ``gap.get_dists_idxs().idxs[:gap_size]``.
      * ``book_embeddings``: NDArray (N_covering_books, D), in the SAME
        order as the covering_books list emitted by this function
        (sorted by descending coverage with book_idx as tie-breaker).
      * ``proposal_embedding``: NDArray (D,)
      * ``seed_embedding``: NDArray (D,) — the seed article embedding
        (gap center).
      * ``gap_radius``: float — the K-ball radius r.
    """
    contexts_dir = output_dir / "contexts"
    record = _build_context_record(gap_with_context, proposal_analysis)
    return write_context_record(contexts_dir, record)


def generate_all_contexts(
    gaps: list[GapWithContext],
    output_dir: Path,
    *,
    proposal_analyses: dict[tuple[str, int], dict[str, Any]],
) -> list[Path]:
    """Write a v2 ContextRecord JSON for every gap. Raises ``KeyError``
    if any gap is missing from ``proposal_analyses``.
    """
    paths: list[Path] = []
    for g in gaps:
        key = (g.topic_id, g.gap_idx)
        if key not in proposal_analyses:
            raise KeyError(
                f"proposal_analyses missing entry for {key}; v2 contexts "
                f"require per-gap proposal coverage data"
            )
        try:
            paths.append(save_gap_context(
                g, output_dir, proposal_analysis=proposal_analyses[key],
            ))
        except ValueError as exc:
            LOGGER.warning(
                "Skipping context for gap %d topic %s: %s",
                g.gap_idx, g.topic_id, exc,
            )
    return paths


def _build_context_record(
    g: GapWithContext, analysis: dict[str, Any],
) -> ContextRecord:
    gap = g.gap
    gap_dists_idxs = gap.get_dists_idxs()
    gap_size = g.gap_size
    if gap_size == 0:
        raise ValueError(f"Gap {g.gap_idx} has no articles")
    article_works = g.articles.works()
    book_works = g.books.works()

    articles_by_centrality = [
        _article_entry(article_works[int(idx)])
        for idx in gap_dists_idxs.idxs[:gap_size]
    ]
    furthest: FurthestArticle | None = None
    if gap_size > 1:
        furthest_pos = int(np.argmax(gap_dists_idxs.sqdists))
        furthest_idx = int(gap_dists_idxs.idxs[furthest_pos])
        w = article_works[furthest_idx]
        furthest = FurthestArticle(
            openalex_id=str(_get(w, "id", "")),
            display_name=str(_get(w, "display_name", "")),
            abstract=str(_get(w, "abstract", "")),
            sqdist=float(gap_dists_idxs.sqdists[furthest_pos]),
        )

    raw_coverage = group_gap_articles_by_covering_books(gap)
    book_coverage: dict[int, list[int]] = {
        int(b_idx): [int(a_idx) for a_idx in covered]
        for b_idx, covered in raw_coverage.items()
    }
    sorted_book_items = sorted(
        book_coverage.items(), key=lambda item: (-len(item[1]), item[0]),
    )

    # Compute post-proposal coverage. The book ordering of
    # ``analysis["book_embeddings"]`` must match the order of
    # ``sorted_book_items``.
    gap_article_embeddings = np.asarray(analysis["gap_article_embeddings"])
    book_embeddings = np.asarray(analysis["book_embeddings"])
    proposal_embedding = np.asarray(analysis["proposal_embedding"])
    seed_embedding = np.asarray(analysis["seed_embedding"])
    gap_radius = float(analysis["gap_radius"])

    post = compute_post_proposal_coverage(
        gap_article_embeddings=gap_article_embeddings,
        book_embeddings=book_embeddings,
        proposal_embedding=proposal_embedding,
    )

    covering_books: list[BookCoverage] = []
    for (b_idx, covered), after_count in zip(
        sorted_book_items, post.per_book_articles_covered_after, strict=True,
    ):
        w = book_works[b_idx]
        covering_books.append(BookCoverage(
            openalex_id=str(_get(w, "id", "")),
            display_name=str(_get(w, "display_name", "")),
            abstract=str(_get(w, "abstract", "")),
            articles_covered=len(covered),
            fraction_covered=(len(covered) / gap_size) if gap_size > 0 else 0.0,
            articles_covered_after_proposal=int(after_count),
        ))

    proposal_coverage = ProposalCoverage(
        articles_covered=post.proposal_articles_covered,
        fraction_covered=post.proposal_fraction_covered,
    )

    # Compute the AEP projection. The green mask: per-article flip
    # indicator. Recompute since ``post`` gives only aggregate counts.
    augmented = np.vstack([book_embeddings, proposal_embedding[None, :]])
    diffs = gap_article_embeddings[:, None, :] - augmented[None, :, :]
    sq = np.einsum("nbd,nbd->nb", diffs, diffs)
    nearest = np.argmin(sq, axis=1)
    green_mask = nearest == book_embeddings.shape[0]

    proj = compute_local_projection_2d(
        seed_embedding=seed_embedding,
        gap_article_embeddings=gap_article_embeddings,
        green_mask=green_mask,
        book_embeddings=book_embeddings,
        proposal_embedding=proposal_embedding,
    )

    local_projection_2d = LocalProjection2D(
        gap_radius=gap_radius,
        seed=Point2D(x=0.0, y=0.0),
        proposal=Point2D(x=float(proj.proposal_2d[0]), y=float(proj.proposal_2d[1])),
        gap_articles=[Point2D(x=float(p[0]), y=float(p[1])) for p in proj.gap_articles_2d],
        gap_articles_flipped_to_proposal=[bool(m) for m in green_mask],
        books=[Point2D(x=float(p[0]), y=float(p[1])) for p in proj.books_2d],
    )

    return ContextRecord(
        topic_id=g.topic_id,
        topic_name=g.topic_name,
        gap_idx=g.gap_idx,
        gap_stats=GapStats(
            gap_size=gap_size,
            gap_size_percentile=float(g.gap_size_percentile),
            gap_zscore=float(g.gap_zscore) if not np.isnan(g.gap_zscore) else float("nan"),
            shapiro_log10_pval=float(g.shapiro_log10_pval) if not np.isnan(g.shapiro_log10_pval) else float("nan"),
        ),
        topic_totals_articles=len(article_works),
        topic_totals_books=len(book_works),
        articles_by_centrality=articles_by_centrality,
        furthest_article=furthest,
        covering_books=covering_books,
        proposal_coverage=proposal_coverage,
        local_projection_2d=local_projection_2d,
    )


def _article_entry(w: Any) -> ArticleEntry:
    return ArticleEntry(
        openalex_id=str(_get(w, "id", "")),
        display_name=str(_get(w, "display_name", "")),
        abstract=str(_get(w, "abstract", "")),
    )


def _get(w: Any, key: str, default: Any) -> Any:
    return w.get(key, default) if isinstance(w, dict) else getattr(w, key, default)
