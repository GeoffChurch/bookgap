"""
Context builder for LLM proposal generation.

This module handles the shared logic for sampling articles and books from gaps,
ensuring deterministic and reproducible selection across different generators.
"""

import random
import hashlib
from typing import TYPE_CHECKING

from bookgap.cloud import SubCloud
from bookgap.config_manager import AnalysisConfiguration
from .protocols import ProposalContext

if TYPE_CHECKING:
    from bookgap.gap_finder import GapProtocol


def build_proposal_context(
    gap: "GapProtocol",
    gap_idx: int,
    topic_name: str,
    topic_id: str,
    analysis_config: AnalysisConfiguration,
    books: SubCloud,
    articles: SubCloud
) -> ProposalContext:
    """Build a proposal context from gap analysis results.
    
    Uses deterministic sampling based on topic_id and gap_idx to ensure
    reproducible results across runs.
    
    Args:
        gap: The gap analysis result
        gap_idx: Index of this gap (0-based)
        topic_name: Human-readable topic name
        topic_id: Unique topic identifier
        analysis_config: Analysis configuration including LLM settings
        books: SubCloud of books for this topic
        articles: SubCloud of articles for this topic
        
    Returns:
        ProposalContext ready for use by any generator
        
    Raises:
        ValueError: If gap contains no articles or critical data is missing
    """
    gap_size = len(gap.dists_idxs.idxs)
    if gap_size == 0:
        raise ValueError(f"Gap {gap_idx} for topic {topic_id} contains no articles")
    
    # Deterministic sampling: seed RNG with topic_id and gap_idx
    seed_string = f"{topic_id}_{gap_idx}"
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
    seed = int(seed_hash[:8], 16)  # Use first 8 hex chars as integer seed
    rng = random.Random(seed)
    
    # Get work metadata
    article_works = articles.works()
    book_works = books.works()
    
    # Sample articles from the gap (up to configured limit)
    requested_articles = analysis_config.llm_config.requested_num_prompt_articles
    sample_size = min(requested_articles, gap_size)
    sampled_indices = rng.sample(range(gap_size), sample_size)
    sampled_articles = [article_works[gap.dists_idxs.idxs[i]] for i in sampled_indices]
    
    # Validate sampled articles have required fields
    for i, article in enumerate(sampled_articles):
        if not article.get('display_name') or not article.get('abstract'):
            raise ValueError(f"Sampled article {i} missing required fields (display_name, abstract)")
    
    # Analyze book coverage and select top covering books
    # Import here to avoid circular import
    from bookgap.gap_finder import group_gap_articles_by_covering_books
    
    covering_books = []
    book_coverage = group_gap_articles_by_covering_books(gap)
    
    if book_coverage:
        # Sort books by coverage (most covering articles first)
        sorted_books = sorted(book_coverage.items(), key=lambda x: len(x[1]), reverse=True)
        requested_books = analysis_config.llm_config.requested_num_prompt_books
        
        for book_idx, _ in sorted_books[:requested_books]:
            book = book_works[book_idx]
            # Validate book has required fields
            if not book.get('display_name') or not book.get('abstract'):
                continue  # Skip books with missing fields rather than failing
            covering_books.append(book)
    
    # Record actual counts used
    num_articles_for_llm = len(sampled_articles)
    num_books_for_llm = len(covering_books)
    counts = {
        "num_articles_for_llm": num_articles_for_llm,
        "num_books_for_llm": num_books_for_llm
    }
    
    return ProposalContext(
        topic_name=topic_name,
        topic_id=topic_id,
        gap_idx=gap_idx,
        requested_num_prompt_articles=requested_articles,
        requested_num_prompt_books=analysis_config.llm_config.requested_num_prompt_books,
        sampled_articles=sampled_articles,
        covering_books=covering_books,
        counts=counts
    )
