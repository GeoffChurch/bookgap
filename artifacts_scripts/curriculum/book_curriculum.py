"""Track A of the real-book ToC reconstruction eval: build OUR curriculum from a
book's OWN gap.

Reads a ``book_own_gap.py`` output JSON (the in-ball article ids = the book's own
gap, computed by treating the book as an article and finding its nearest non-self
book), seeds the curriculum pipeline's ``target_work_ids`` with them, and runs the
build at the SAME scope the gap was computed at. The result is "our curriculum"
for that book's gap, to be compared against the book's real ToC (Track B).

Mirrors ``seed_from_gap.build_curriculum_for_gap``'s build body, but the seed is a
book's own gap rather than a detector-chosen gap. Run ONE per process.

NOTE: the reference book is NOT yet held out of the cloud (a refinement). The seed
is articles, not the book, and the book is one node among the scope's works, so the
leakage is minor for validation; flagged for the clean run.

Example::

    .conda/bin/python -m artifacts_scripts.curriculum.book_curriculum \\
        --gap-json artifacts/book_gaps/W2898513995_cyberbullying_ai.json
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
from .main import run_keyword_extraction, run_plotting
from .output_summary import generate_curriculum_summary


def build_curriculum_from_book_gap(
    gap_json: Path,
    *,
    output_dir: Optional[str] = None,
    subfield: Optional[str] = None,
    field: Optional[str] = None,
    desired_n_topics: int = 15,
    boundary: bool = False,
) -> Path:
    rec = json.loads(Path(gap_json).read_text())
    bk = rec.get("book", rec)  # single-book mode nests under "book"; fall back to flat
    ids = bk["gap_article_ids"]
    if not ids:
        raise ValueError(f"empty gap_article_ids in {gap_json}")
    sf = subfield or rec.get("scope_subfield")
    fld = field or rec.get("scope_field")
    book_id = bk.get("book_id", "unknown")
    name = output_dir or f"book_curriculum_{book_id.rsplit('/', 1)[-1]}"

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.target_work_ids = ids
    cfg.output_dir = name
    cfg.desired_n_topics = desired_n_topics
    cfg.rank_before_clustering = True
    cfg.ranking_top_k = 5000
    if sf:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(sf)
    if fld:
        cfg.experiment.cloud.filter_pair.pre.field = str(fld)
    if boundary:
        cfg.boundary_selection = True
        cfg.background_exponent = 0.5

    out = get_curriculum_output_dir(name, auto_derived=False)
    setup_logging(log_name="pipeline", log_level="DEBUG", include_time=False,
                  log_path=out / "pipeline.log", explicit_args=None)
    LOGGER.info("=" * 60)
    LOGGER.info("BOOK-GAP CURRICULUM (Track A)")
    LOGGER.info("=" * 60)
    LOGGER.info("Book %s: seeding %d in-gap articles; subfield=%s; boundary=%s",
                book_id, len(ids), sf, boundary)

    cloud, embeddings_for_clustering, filtered_indices = run_curriculum_building(cfg, output_dir=out)
    topic_keywords = run_keyword_extraction(cfg.keywords, out, cloud, filtered_indices)
    generate_curriculum_summary(out, cloud, filtered_indices, topic_keywords)
    if cfg.plot.enabled:
        run_plotting(cfg.plot, out, embeddings_for_clustering, topic_keywords)
    LOGGER.info("Curriculum complete for %s -> %s", book_id, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A: curriculum from a book's own gap.")
    ap.add_argument("--gap-json", required=True, type=Path,
                    help="book_own_gap.py output JSON (single-book mode, has gap_article_ids)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--subfield", default=None, help="override scope subfield (default: from gap JSON)")
    ap.add_argument("--field", default=None, help="override scope field (default: from gap JSON)")
    ap.add_argument("--desired-n-topics", type=int, default=15)
    ap.add_argument("--boundary", action="store_true", help="use the boundary-background selector")
    args = ap.parse_args()
    out = build_curriculum_from_book_gap(
        args.gap_json, output_dir=args.output_dir, subfield=args.subfield, field=args.field,
        desired_n_topics=args.desired_n_topics, boundary=args.boundary,
    )
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
