
import numpy as np
from bookgap.gap_types import DistsAndIdxs
from bookgap.llm.artifacts.data_preparation import GapProtocol


def get_book_to_articles_mapping(gap: GapProtocol) -> dict[np.intp, DistsAndIdxs]:
    """Get mapping from book indices to their covered articles with distances."""
    return gap.get_book_to_articles_mapping()


def group_gap_articles_by_covering_books(gap: GapProtocol) -> dict[np.intp, list[np.intp]]:
    """Group gap articles by their covering books (simplified version without distances)."""
    book_to_articles = get_book_to_articles_mapping(gap)
    return {book_idx: dists_idxs.idxs.tolist() for book_idx, dists_idxs in book_to_articles.items()}
