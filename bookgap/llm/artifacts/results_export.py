"""
Simple results CSV export for Step 2.2.

Generates a proposals-centric CSV that joins each proposal with basic gap
metadata (topic, ids, gap_idx, counts, gap_size).
"""
from collections import deque
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookgap.cloud_index import compute_proposal_rankings_for_gap
from bookgap.configs.config import ExperimentConfig
from bookgap.llm.identifiers import GapProposalId
from bookgap.llm.protocols import ParsedProposal, ProposalContext, ProposalCounts
from bookgap.llm.data_structures import GapWithContext
from bookgap.util import LOGGER


def _enum_name(value: Any) -> Any:
    """Return enum.name when available, otherwise the value itself."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    return name if name is not None else value


def _config_row(config: ExperimentConfig) -> dict[str, Any]:
    """Flatten experiment configuration into a single-row mapping."""

    gap = getattr(config, "gap", None)
    llm = getattr(config, "llm", None)
    data = getattr(config, "data", None)
    batch = getattr(config, "batch", None)
    embedder = getattr(config, "embedder", None)
    output = getattr(config, "output", None)

    export_formats = getattr(output, "export_formats", ()) if output else ()
    deduplicator = getattr(data, "deduplicator", None) if data else None
    export_format_names = [
        str(_enum_name(fmt)) for fmt in export_formats if _enum_name(fmt) is not None
    ]

    row: dict[str, Any] = {
        # Gap configuration
        "config_gap_cell_type": _enum_name(getattr(gap, "cell_type", None)),
        "config_gap_radius_fraction": getattr(gap, "radius_fraction", None),
        "config_gap_drop_k": getattr(gap, "drop_k", None),
        "config_gap_hnsw_m": getattr(gap, "hnsw_m", None),
        "config_gap_hnsw_efc": getattr(gap, "hnsw_efc", None),
        "config_gap_hnsw_efs": getattr(gap, "hnsw_efs", None),
        "config_gap_denoiser": _enum_name(getattr(gap, "denoiser", None)),
        "config_gap_index_type": _enum_name(getattr(gap, "index_type", None)),
        # LLM configuration
        "config_llm_target": getattr(llm, "_target_", None),
        "config_llm_model": _enum_name(getattr(llm, "model", None)),
        "config_llm_num_articles": getattr(llm, "num_articles", None),
        "config_llm_num_books": getattr(llm, "num_books", None),
        "config_llm_temperature": getattr(llm, "temperature", None),
        "config_llm_max_tokens": getattr(llm, "max_tokens", None),
        "config_llm_use_structured_output": getattr(
            llm, "use_structured_output", None
        ),
        # Data configuration
        "config_data_max_works": getattr(data, "max_works", None),
        "config_data_deduplicator_target": (
            deduplicator._target_ if deduplicator is not None else None
        ),
        "config_data_deduplicator_min_sqdist": (
            deduplicator.min_sqdist if deduplicator is not None else None
        ),
        # Batch configuration
        "config_batch_num_gaps": getattr(batch, "num_gaps", None),
        "config_batch_completion_window": getattr(batch, "completion_window", None),
        "config_batch_generate_plots": getattr(batch, "generate_plots", None),
        "config_batch_save_prompts": getattr(batch, "save_prompts", None),
        "config_batch_save_batch_file": getattr(batch, "save_batch_file", None),
        # Embedder configuration
        "config_embedder_model_name": _enum_name(
            getattr(embedder, "model_name", None)
        ),
        # Output configuration
        "config_output_export_formats": ",".join(export_format_names),
        "config_output_create_subdirectories": getattr(
            output, "create_subdirectories", None
        ),
        "config_output_include_metadata": getattr(output, "include_metadata", None),
        "config_output_use_timestamp": getattr(output, "use_timestamp", None),
    }

    return row

def _gap_row(
    g: GapWithContext,
    config: ExperimentConfig,
    counts_map: dict[tuple[str, int], ProposalCounts],
) -> dict[str, Any]:
    """
    Return gap metadata as a plain dict in insertion order (Python 3.7+).

    Includes both requested and actual prompt counts by looking up the
    ProposalCounts for the given (topic_id, gap_idx) pair.
    """
    gap_size = g.gap_size
    num_books = len(g.books.embeddings())
    num_articles = len(g.articles.embeddings())
    counts = counts_map.get((g.topic_id, g.gap_idx))
    actual_articles = counts.num_articles_for_llm if counts else 0
    actual_books = counts.num_books_for_llm if counts else 0
    return {
        "topic_name": g.topic_name,
        "topic_id": g.topic_id,
        "denoiser": config.cloud.denoiser,
        "drop_k": config.cloud.drop_k,
        "radius_fraction": config.gap.radius_fraction,
        "cell_type": config.gap.cell_type.name,
        "gap_idx": g.gap_idx,
        "num_books": num_books,
        "num_articles": num_articles,
        "gap_size": gap_size,
        "gap_size_percentile": g.gap_size_percentile,
        "gap_zscore": g.gap_zscore,
        "shapiro_log10_pval": g.shapiro_log10_pval,
        "requested_num_prompt_articles": config.llm.num_articles,
        "requested_num_prompt_books": config.llm.num_books,
        "actual_num_prompt_articles": actual_articles,
        "actual_num_prompt_books": actual_books,
    }


def _sanitize_text_field(value: str) -> str:
    """Replace control characters with safe escaped sequences for CSV output."""

    def _substitute(match: re.Match[str]) -> str:
        char = match.group(0)
        return {"\n": r"\n", "\r": r"\r", "\t": r"\t"}.get(char, f"\\x{ord(char):02x}")

    return re.sub(pattern=r"[\x00-\x1f\x7f-\x9f]", repl=_substitute, string=value)


def _build_counts_map(
    contexts: list[ProposalContext],
) -> dict[tuple[str, int], ProposalCounts]:
    """Create a lookup from (topic_id, gap_idx) to ProposalCounts."""
    return {(ctx.topic_id, ctx.gap_idx): ctx.counts for ctx in contexts}

def _compute_proposal_rankings_by_gap(
    gaps: list[GapWithContext],
    proposals_grouped: dict[tuple[str, int], list[ParsedProposal]],
) -> dict[tuple[str, int], deque[tuple[int, int]]]:
    """Compute ranking stats for proposals grouped by their gap."""

    gap_map = {(gap.topic_id, gap.gap_idx): gap for gap in gaps}
    rankings: dict[tuple[str, int], deque[tuple[int, int]]] = {}

    for key, group in proposals_grouped.items():
        gap = gap_map.get(key)
        if gap is None:
            LOGGER.warning(
                "No gap context found for topic_id=%s gap_idx=%s; assigning default ranks",
                key[0],
                key[1],
            )
            rankings[key] = deque([(-1, -1)] * len(group))
            continue

        dists_idxs = gap.gap.get_dists_idxs()
        if len(dists_idxs.idxs) == 0:
            LOGGER.warning(
                "Gap context for topic_id=%s gap_idx=%s has no article indices; assigning default ranks",
                key[0],
                key[1],
            )
            rankings[key] = deque([(-1, -1)] * len(group))
            continue

        center_idx = int(dists_idxs.idxs[0])
        article_embeddings = gap.articles.embeddings()
        if center_idx < 0 or center_idx >= len(article_embeddings):
            LOGGER.warning(
                "Gap center index %s out of bounds for topic_id=%s gap_idx=%s; assigning default ranks",
                center_idx,
                key[0],
                key[1],
            )
            rankings[key] = deque([(-1, -1)] * len(group))
            continue

        proposal_embeddings = [
            np.asarray(proposal.embedding, dtype=np.float32) for proposal in group
        ]
        if not proposal_embeddings:
            rankings[key] = deque()
            continue

        proposal_embeddings_array = np.stack(proposal_embeddings, axis=0)
        center_embedding = np.asarray(
            article_embeddings[center_idx : center_idx + 1],
            dtype=np.float32,
        )
        ranking_result = compute_proposal_rankings_for_gap(
            gap.books,
            gap.articles,
            center_embedding,
            proposal_embeddings_array,
        )
        rankings[key] = deque(
            zip(
                ranking_result.book_ranks.astype(int).tolist(),
                ranking_result.article_ranks.astype(int).tolist(),
            )
        )

    return rankings


def _proposal_row(
    parsed: GapProposalId,
    proposal: ParsedProposal,
    ranking: tuple[int, int] | None,
    gap_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    book_rank, article_rank = ranking if ranking is not None else (-1, -1)
    num_books = int(gap_meta.get("num_books", 0)) if gap_meta else 0
    num_articles = int(gap_meta.get("num_articles", 0)) if gap_meta else 0
    book_fraction = (
        book_rank / num_books if book_rank >= 0 and num_books > 0 else -1.0
    )
    article_fraction = (
        article_rank / num_articles if article_rank >= 0 and num_articles > 0 else -1.0
    )
    return {
        "topic_id": parsed.topic_id,
        "gap_idx": parsed.gap_idx,
        "proposal_book_rank": int(book_rank),
        "proposal_book_rank_fraction": book_fraction,
        "proposal_article_rank": int(article_rank),
        "proposal_article_rank_fraction": article_fraction,
        "proposal_title": _sanitize_text_field(proposal.proposal.title),
        "proposal_abstract": _sanitize_text_field(proposal.proposal.abstract),
    }


def generate_gaps_analysis_csv(
    gaps: list[GapWithContext],
    proposals: list[ParsedProposal],
    output_dir: Path,
    config: ExperimentConfig,
    contexts: list[ProposalContext],
    *,
    proposals_grouped: dict[tuple[str, int], list[ParsedProposal]],
    ordered_proposals: list[tuple[ParsedProposal, GapProposalId]],
) -> Path:
    """
    Create a minimal CSV joining proposals with basic gap metadata.

    Rows are proposals; columns include proposal text, custom_id, and gap info.
    """
    LOGGER.info(
        f"Generating simple gaps_analysis_results.csv for {len(proposals)} proposals over {len(gaps)} gaps"
    )

    # Build DataFrames
    counts_map = _build_counts_map(contexts)
    gap_rows = [_gap_row(g, config, counts_map) for g in gaps]
    gap_df = pd.DataFrame(gap_rows)
    gap_meta_map = {(row["topic_id"], row["gap_idx"]): row for row in gap_rows}
    rankings_by_gap = _compute_proposal_rankings_by_gap(gaps, proposals_grouped)

    prop_rows: list[dict[str, Any]] = []
    for proposal, parsed in ordered_proposals:
        ranking_deque = rankings_by_gap.get((parsed.topic_id, parsed.gap_idx))
        ranking_tuple = ranking_deque.popleft() if ranking_deque else None
        gap_meta = gap_meta_map.get((parsed.topic_id, parsed.gap_idx))
        prop_rows.append(_proposal_row(parsed, proposal, ranking_tuple, gap_meta))

    proposals_df = pd.DataFrame(prop_rows)

    if gap_df.empty:
        LOGGER.warning("Gap DataFrame is empty; CSV will contain only proposal columns")

    # Merge proposals with gap metadata (left: proposals)
    key_cols = ["topic_id", "gap_idx"]
    merged = proposals_df.merge(gap_df, on=key_cols, how="left")

    # Broadcast full experiment configuration across all rows
    config_row = _config_row(config)
    for key, value in config_row.items():
        merged[key] = value

    # Reorder columns to ensure proposal_title and proposal_abstract are last
    LAST_COLUMNS = ["proposal_title", "proposal_abstract"]
    column_order = [col for col in merged.columns if col not in LAST_COLUMNS] + LAST_COLUMNS
    merged = merged[column_order]

    # Ensure output dir exists and write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gaps_analysis_results.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8")
    LOGGER.info(f"Wrote CSV with {len(merged)} rows to {csv_path}")
    return csv_path
