"""
Simple results CSV export for Step 2.2.

Generates a proposals-centric CSV that joins each proposal with basic gap
metadata (topic, ids, gap_idx, counts, gap_size).
"""
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from bookgap.configs.config import ExperimentConfig
from bookgap.llm.identifiers import parse_gap_proposal_id
from bookgap.llm.protocols import ParsedProposal, ProposalContext, ProposalCounts
from bookgap.llm.data_structures import GapWithContext
from bookgap.util import LOGGER

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
        "denoiser": config.gap.denoiser.name,
        "drop_k": config.gap.drop_k,
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
    contexts: List[ProposalContext],
) -> dict[tuple[str, int], ProposalCounts]:
    """Create a lookup from (topic_id, gap_idx) to ProposalCounts."""
    counts_map: dict[tuple[str, int], ProposalCounts] = {}
    for ctx in contexts:
        key = (ctx.topic_id, ctx.gap_idx)
        if key in counts_map:
            LOGGER.warning(
                "Duplicate ProposalContext detected for topic_id=%s gap_idx=%s; overwriting previous counts",
                ctx.topic_id,
                ctx.gap_idx,
            )
        counts_map[key] = ctx.counts
    return counts_map


def _proposal_row(p: ParsedProposal) -> Dict[str, Any] | None:
    parsed = parse_gap_proposal_id(p.custom_id)
    return {
        "topic_id": parsed.topic_id,
        "gap_idx": parsed.gap_idx,
        "proposal_title": _sanitize_text_field(p.proposal.title),
        "proposal_abstract": _sanitize_text_field(p.proposal.abstract),
    }


def generate_gaps_analysis_csv(
    gaps: List[GapWithContext],
    proposals: List[ParsedProposal],
    output_dir: Path,
    config: ExperimentConfig,
    contexts: List[ProposalContext],
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
    prop_rows = [r for p in proposals if (r := _proposal_row(p)) is not None]
    proposals_df = pd.DataFrame(prop_rows)

    if gap_df.empty:
        LOGGER.warning("Gap DataFrame is empty; CSV will contain only proposal columns")

    # Merge proposals with gap metadata (left: proposals)
    key_cols = ["topic_id", "gap_idx"]
    merged = proposals_df.merge(gap_df, on=key_cols, how="left")

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
