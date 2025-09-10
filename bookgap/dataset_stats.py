#!/usr/bin/env python3
"""
Report the number of articles and books in the AI subfield cloud.
"""
from pathlib import Path
from bookgap.cloud import Cloud
from bookgap.util import Filter, FilterPair, SubField, WorkType, setup_logging, LOGGER
from bookgap.parse_args import get_explicit_args, parse_args
from bookgap.index import HNSW, IndexType
from bookgap.deduplicators import Deduplicator
from typing import Optional

def main(
    max_works: Optional[int],
    deduplicator: Optional[Deduplicator],
    index_type: IndexType,
    hnsw_m: int,
    hnsw_efc: int,
    hnsw_efs: int,
    denoiser: str,
    drop_k: int,
    radius_fraction: float,  # Not used here, but for CLI compatibility
):
    # Set up output directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Standard cloud configuration (AI subfield, no topic filter)
    if index_type == IndexType.HNSW:
        index_info = HNSW(hnsw_m=hnsw_m, hnsw_efc=hnsw_efc, hnsw_efs=hnsw_efs)
    else:
        index_info = index_type.value()
    ai_filter_pair = FilterPair(Filter(subfield=SubField.AI.value), Filter())
    ai_cloud = Cloud(
        filter_pair=ai_filter_pair,
        index_info=index_info,
        max_works=max_works,
        deduplicator=deduplicator,
        normalize_embeddings=True,
        denoiser=denoiser,
        drop_k=drop_k
    )

    # Count books and articles
    works = ai_cloud.works()
    num_books = sum(1 for w in works if w.get('type') == WorkType.BOOK.value)
    num_articles = sum(1 for w in works if w.get('type') == WorkType.ARTICLE.value)
    total = len(works)
    print(f"Total works in AI subfield: {total}")
    print(f"Number of books: {num_books}")
    print(f"Number of articles: {num_articles}")
    print(f"Denoiser: {denoiser}")
    print(f"Drop_k: {drop_k}")
    print(f"Index type: {index_type}")
    print(f"Deduplicator: {deduplicator}")
    print(f"Max works: {max_works}")

if __name__ == "__main__":
    from bookgap.script_utils import run_script_main
    run_script_main(main, __file__)
