#!/usr/bin/env python3
"""One-off: re-extract a built curriculum run-dir's keywords at a new top_k.

Reuses the existing dendrogram / clustering in the run-dir (does NOT rebuild the
curriculum). Instantiates the CS-field (field=17) cloud, reconstructs
``filtered_indices`` from the run-dir's ``filtered_work_ids.json`` (mapping each
work id to its cloud index via ``cloud.work_ids()``), then calls
``run_keyword_extraction`` followed by ``generate_curriculum_summary`` so the
run-dir's ``topic_keywords.json`` AND ``summary.md`` are rewritten with the new
keyword count via the SAME phrase-mining path the live pipeline uses.

Usage::

    PYTHONPATH=. .conda/bin/python -m artifacts_scripts.curriculum.reextract_keywords_topk \\
        --run-dir artifacts/curriculum/book_curriculum_W4235013513_mcs --top-k 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.main import run_keyword_extraction
from artifacts_scripts.curriculum.output_summary import generate_curriculum_summary
from bookgap.util import LOGGER, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--dynamic-floor", type=float, default=0.0,
                    help="Dynamic top-k: keep keywords scoring >= this fraction of "
                         "the topic's top score (0=off; ~0.05 trims ~0-score padding).")
    ap.add_argument("--soft-topics", action="store_true",
                    help="Fuzzy keyword extraction: each work contributes to every topic with "
                         "soft responsibility >= floor (weighted mass*r), dissolving degenerate chapters.")
    ap.add_argument("--soft-mode", choices=["embedding", "citation"], default="embedding",
                    help="soft responsibility source: embedding softmax over centroids, or "
                         "citation label-propagation (absorbing-walk arm; citation-coherent).")
    ap.add_argument("--soft-alpha", type=float, default=0.5,
                    help="For --soft-mode citation: citation-diffusion weight (1-alpha clamps to hard label).")
    ap.add_argument("--soft-perplexity", type=float, default=1.5,
                    help="Target mean topics-per-work for the soft-membership softmax temperature.")
    ap.add_argument("--soft-floor", type=float, default=0.05,
                    help="Min responsibility for a work to enter a topic's soft keyword pool.")
    ap.add_argument("--field", type=str, default="17",
                    help="OpenAlex field id for the cloud scope (CS = 17).")
    args = ap.parse_args()

    setup_logging(log_name="reextract_keywords", log_level="INFO",
                  include_time=False)

    run_dir = args.run_dir

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cfg.keywords.top_k = int(args.top_k)
    cfg.keywords.dynamic_topk_floor = float(args.dynamic_floor)
    cfg.keywords.soft_topics = bool(args.soft_topics)
    cfg.keywords.soft_mode = str(args.soft_mode)
    cfg.keywords.soft_alpha = float(args.soft_alpha)
    cfg.keywords.soft_perplexity = float(args.soft_perplexity)
    cfg.keywords.soft_floor = float(args.soft_floor)
    cloud = instantiate(cfg.experiment.cloud)

    # Reconstruct filtered_indices: map each filtered work id -> cloud index.
    filtered_ids = json.loads((run_dir / "filtered_work_ids.json").read_text())
    work_ids = cloud.work_ids()
    id2idx = {w: i for i, w in enumerate(work_ids)}
    missing = [w for w in filtered_ids if w not in id2idx]
    if missing:
        raise SystemExit(
            f"{len(missing)} filtered work ids not found in cloud "
            f"(cloud scope mismatch?). First few: {missing[:3]}")
    filtered_indices = np.array([id2idx[w] for w in filtered_ids], dtype=np.intp)
    LOGGER.info("Reconstructed filtered_indices: %d works (cloud has %d)",
                len(filtered_indices), len(work_ids))

    # Rewrites topic_keywords.json via the phrase-mining chain.
    topic_keywords = run_keyword_extraction(
        cfg.keywords, run_dir, cloud, filtered_indices,
    )

    # Regenerate summary.md (parseable format) with the new keyword count.
    generate_curriculum_summary(run_dir, cloud, filtered_indices, topic_keywords)
    LOGGER.info("Rewrote %s and %s",
                run_dir / "topic_keywords.json", run_dir / "summary.md")


if __name__ == "__main__":
    main()
