"""Identifier helpers for gap-based proposals.

This module centralizes construction and parsing of the simple custom IDs used
by the batch-first LLM pipeline. Keeping this logic in one place ensures the
ID format stays consistent between request generation and downstream exports.
"""
from dataclasses import dataclass

from bookgap.util import LOGGER


@dataclass(frozen=True)
class GapProposalId:
    """Typed representation of a proposal custom ID."""

    gap_idx: int
    topic_id: str


def build_gap_proposal_id(*, gap_idx: int, topic_id: str) -> str:
    """Return the canonical custom ID for a proposal.

    Format: ``gap_{gap_idx}_topic_{topic_id}``
    """
    if gap_idx < 0:
        raise ValueError(f"gap_idx must be >= 0, got {gap_idx}")

    if not topic_id:
        raise ValueError("topic_id cannot be empty")

    if topic_id != topic_id.strip():
        raise ValueError("topic_id cannot have leading or trailing whitespace")


    return f"gap_{gap_idx}_topic_{topic_id}"


def parse_gap_proposal_id(custom_id: str) -> GapProposalId:
    """Parse a custom ID produced by :func:`build_gap_proposal_id`.

    Raises ``ValueError`` if the string does not match the expected format.
    """
    if not custom_id:
        raise ValueError("custom_id cannot be empty")
    
    if custom_id != custom_id.strip():
        raise ValueError("custom_id cannot have leading or trailing whitespace")

    parts = custom_id.strip().split("_")
    if len(parts) < 4 or parts[0] != "gap" or parts[2] != "topic":
        raise ValueError(
            "custom_id must have format 'gap_{idx}_topic_{topic_id}', got: "
            f"{custom_id}"
        )

    gap_idx = int(parts[1])
    topic_id = "_".join(parts[3:])  # allow underscores in topic ID

    if topic_id == "":
        raise ValueError("topic_id component cannot be empty")

    return GapProposalId(gap_idx=gap_idx, topic_id=topic_id)
