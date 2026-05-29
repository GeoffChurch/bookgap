"""Seed a curriculum from a gap-detection run's in-gap articles.

See ``docs/backlog/subjective-curriculum-eval.md`` (Recipe C). Reads the full
in-gap article set from a run's ``contexts/gap_{i}_topic_{T}.json``, seeds the
curriculum pipeline's ``target_work_ids`` with them (uniform mass), and runs the
default ``domain=3`` (Physical Sciences) curriculum build + keyword extraction +
summary + plot. With only target (gap) seeds and no source seeds, the default
``BIDIRECTIONAL_PAGE_RANK`` degrades to forward-only personalized PageRank, so
relevance mass flows to the gap's prerequisites (the curriculum on-ramp).

Mirrors ``artifacts_scripts/curriculum/main.py``'s ``main()`` body exactly (it just
swaps Hydra's CLI cfg for a programmatically-built one carrying thousands of seed
IDs, which would overflow argv). Run ONE topic per process — at domain scale the
cloud load is ~60 GB, so orchestrate multiple topics with an external sequential
loop, not in-process.

Example::

    conda run -p .conda python -m artifacts_scripts.curriculum.seed_from_gap \\
        --topic-id T11273
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from bookgap.util import LOGGER, setup_logging

from .build_curriculum import run_curriculum_building
from .config import CurriculumConfig, get_curriculum_output_dir
from .output_summary import generate_curriculum_summary
from .main import run_keyword_extraction, run_plotting

DEFAULT_RUN_DIR = Path("outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10")


def load_gap_article_ids(
    run_dir: Path, topic_id: str, gap_idx: int = 0, top_k: Optional[int] = None
) -> tuple[list[str], str]:
    """Return (in-gap OpenAlex IDs sorted by centrality, topic_name).

    Reads the full membership from the context JSON's ``articles_by_centrality``.
    ``top_k`` keeps only the K most-central articles (default: all).
    """
    ctx_path = Path(run_dir) / "contexts" / f"gap_{gap_idx}_topic_{topic_id}.json"
    ctx = json.loads(ctx_path.read_text())
    ids = [a["openalex_id"] for a in ctx["articles_by_centrality"]]
    if top_k is not None:
        ids = ids[:top_k]
    return ids, ctx.get("topic_name", topic_id)


def load_covering_book_ids(run_dir: Path, topic_id: str, gap_idx: int = 0) -> list[str]:
    """Return the gap's covering-book OpenAlex IDs (established foundations).

    Used as ``source_work_ids`` so bidirectional PageRank's product highlights the
    corridor between established foundations and the gap frontier.
    """
    ctx_path = Path(run_dir) / "contexts" / f"gap_{gap_idx}_topic_{topic_id}.json"
    ctx = json.loads(ctx_path.read_text())
    return [b["openalex_id"] for b in ctx.get("covering_books", [])]


def build_curriculum_for_gap(
    run_dir: Path,
    topic_id: str,
    *,
    gap_idx: int = 0,
    top_k: Optional[int] = None,
    desired_n_topics: int = 15,
    output_dir: Optional[str] = None,
    source_from_books: bool = False,
    background_normalize: bool = False,
    background_exponent: float = 0.5,
    background_alpha: Optional[float] = None,
    similarity: bool = False,
    similarity_mix_weight: Optional[float] = None,
    similarity_knn_k: Optional[int] = None,
) -> Path:
    """Build one curriculum seeded by a gap's in-gap articles. Returns output dir.

    If ``source_from_books`` is set, the gap's covering books are used as
    ``source_work_ids``, so bidirectional PageRank's ``product`` highlights the
    corridor between established foundations and the gap frontier (experiment:
    does this suppress off-topic intrusions?).
    """
    ids, topic_name = load_gap_article_ids(run_dir, topic_id, gap_idx, top_k)
    if not ids:
        raise ValueError(f"No in-gap articles found for {topic_id} gap {gap_idx}")

    source_ids = load_covering_book_ids(run_dir, topic_id, gap_idx) if source_from_books else []

    name = output_dir or f"gap_curriculum_{topic_id}"

    # Build the cfg the way Hydra would (structured node == ConfigStore's), then
    # override the handful of fields we care about. Cloud scope is left at the
    # default (domain=3, Physical Sciences). rank_before_clustering is pinned True
    # so clustering only ever sees ranking_top_k works, never the full ~42M corpus.
    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.target_work_ids = ids
    if source_ids:
        cfg.source_work_ids = source_ids
    cfg.output_dir = name
    cfg.desired_n_topics = desired_n_topics
    cfg.rank_before_clustering = True
    cfg.ranking_top_k = 5000
    if background_normalize:
        cfg.background_normalize = True
        cfg.background_exponent = background_exponent
        if background_alpha is not None:
            cfg.background_alpha = background_alpha
    if similarity:
        cfg.similarity.enabled = True
        if similarity_mix_weight is not None:
            cfg.similarity.mix_weight = similarity_mix_weight
        if similarity_knn_k is not None:
            cfg.similarity.knn_k = similarity_knn_k
    # method stays BIDIRECTIONAL_PAGE_RANK: with source seeds it runs the product
    # (foundations<->gap corridor); without, it degrades to forward-only PR.

    out = get_curriculum_output_dir(name, auto_derived=False)
    setup_logging(
        log_name="pipeline",
        log_level="DEBUG",
        include_time=False,
        log_path=out / "pipeline.log",
        explicit_args=None,
    )
    LOGGER.info("=" * 60)
    LOGGER.info("SEED-FROM-GAP CURRICULUM")
    LOGGER.info("=" * 60)
    LOGGER.info(
        "Topic %s (%s) gap %d: seeding %d in-gap articles (top_k=%s); %d source books",
        topic_id, topic_name, gap_idx, len(ids), top_k, len(source_ids),
    )

    # Replicate main()'s body: build -> keywords -> summary -> plot.
    cloud, embeddings_for_clustering, filtered_indices = run_curriculum_building(
        cfg, output_dir=out
    )
    topic_keywords = run_keyword_extraction(cfg.keywords, out, cloud, filtered_indices)
    generate_curriculum_summary(out, cloud, filtered_indices, topic_keywords)
    if cfg.plot.enabled:
        run_plotting(cfg.plot, out, embeddings_for_clustering, topic_keywords)

    LOGGER.info("Curriculum complete for %s -> %s", topic_id, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seed a curriculum from a gap's in-gap articles (one topic/process)."
    )
    ap.add_argument("--topic-id", required=True)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--gap-idx", type=int, default=0)
    ap.add_argument(
        "--top-k", type=int, default=None,
        help="Seed only the top-K most-central in-gap articles (default: all).",
    )
    ap.add_argument("--desired-n-topics", type=int, default=15)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--source-from-books", action="store_true",
        help="Use the gap's covering books as source_work_ids (bidirectional PR "
             "product: foundations<->gap corridor).",
    )
    ap.add_argument(
        "--background-normalize", action="store_true",
        help="Rank top-k selection by background-normalized score p^a*log(p/g).",
    )
    ap.add_argument("--background-exponent", type=float, default=0.5)
    ap.add_argument(
        "--background-alpha", type=float, default=None,
        help="Background-PR damping; default None = match foreground pagerank.alpha.",
    )
    ap.add_argument(
        "--similarity", action="store_true",
        help="Blend an embedding-kNN graph into the citation graph (semantic backbone).",
    )
    ap.add_argument("--similarity-mix-weight", type=float, default=None,
                    help="Convex blend weight for the similarity graph (default cfg 0.1).")
    ap.add_argument("--similarity-knn-k", type=int, default=None,
                    help="kNN neighbors per node for the similarity graph (default cfg 5).")
    args = ap.parse_args()

    out = build_curriculum_for_gap(
        Path(args.run_dir),
        args.topic_id,
        gap_idx=args.gap_idx,
        top_k=args.top_k,
        desired_n_topics=args.desired_n_topics,
        output_dir=args.output_dir,
        source_from_books=args.source_from_books,
        background_normalize=args.background_normalize,
        background_exponent=args.background_exponent,
        background_alpha=args.background_alpha,
        similarity=args.similarity,
        similarity_mix_weight=args.similarity_mix_weight,
        similarity_knn_k=args.similarity_knn_k,
    )
    print(f"DONE {args.topic_id} -> {out}")


if __name__ == "__main__":
    main()
