#!/usr/bin/env python3
"""A0 baseline seeds for the close-the-loop ablation: count-matched RANDOM-region
"gaps".

The detected-gap arm (A1) seeds a curriculum with a book's own-gap articles
(book_own_gap.py). To tell apart "the gap detector landed on the book's material"
from "any curriculum of this size covers the book", we need a density floor: build
the SAME-sized curriculum at a random location and show it does NOT reconstruct the
book. This emits N such random-region seed JSONs, each in book_own_gap's single-book
schema (so book_curriculum.py consumes them unchanged).

Construction (count-matched to the real gap): pick a uniform-random ARTICLE center,
take its ``--match-size`` nearest articles (cosine = dot, embeddings are L2-normed)
as the region. Same cardinality as A1's seed, a random ambient location; the ball
RADIUS floats with local density, so this samples the field's typical density at a
matched article count. We exclude any article that is in the real gap (negligible in
practice) and record the overlap as a diagnostic.

Usage:
  .conda/bin/python -m artifacts_scripts.curriculum.random_region_gap \
      --like-gap artifacts/book_gaps/mcs_gap.json --n-seeds 3 \
      --out-prefix artifacts/book_gaps/cl_A0/mcs_rand
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like-gap", required=True, type=Path,
                    help="a book_own_gap.py output JSON; we copy its scope and match its gap size")
    ap.add_argument("--match-size", type=int, default=0,
                    help="region article count (default 0 = len(real gap_article_ids))")
    ap.add_argument("--n-seeds", type=int, default=3, help="number of random regions to emit")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (center i uses seed+i)")
    ap.add_argument("--out-prefix", required=True, type=Path,
                    help="seed s is written to <out-prefix>_s.json")
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_name="random_region_gap", log_level="INFO", include_time=False,
                  log_path=args.out_prefix.with_suffix(".log"))

    rec = json.loads(args.like_gap.read_text())
    bk = rec.get("book", rec)
    real_gap = set(bk.get("gap_article_ids", []))
    n_match = args.match_size or len(real_gap)
    subfield = rec.get("scope_subfield")
    field = rec.get("scope_field")
    LOGGER.info("matching gap %s: size=%d, subfield=%s field=%s", args.like_gap.name,
                n_match, subfield, field)

    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
    if field:
        cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    art_ids = articles.work_ids()
    a_emb = np.asarray(articles.embeddings(), dtype=np.float32)
    a_titles = articles.work_titles()
    n_art = len(art_ids)
    real_rows = np.array([i for i, a in enumerate(art_ids) if a in real_gap], dtype=np.int64)
    LOGGER.info("scope: %d articles; real-gap rows resolved in scope: %d", n_art, len(real_rows))

    rng = np.random.default_rng(args.seed)
    for s in range(args.n_seeds):
        # Independent uniform-random center per seed (reproducible: seed = base + s).
        c = int(np.random.default_rng(args.seed + s).integers(0, n_art))
        sims = a_emb @ a_emb[c]                      # cosine (rows L2-normed)
        order = np.argsort(sims)[::-1]               # nearest first (includes center)
        # Drop any real-gap article so A0 can't accidentally seed from the target.
        keep = [int(j) for j in order if j not in set(real_rows.tolist())]
        region = keep[: n_match]
        region_ids = [art_ids[j] for j in region]
        overlap = len(set(region_ids) & real_gap)
        nn_sim = float(sims[region[-1]])             # sim at the ball boundary
        LOGGER.info("seed %d: center=%s | region=%d | boundary cos=%.3f | overlap w/ real gap=%d",
                    s, a_titles[c][:50], len(region_ids), nn_sim, overlap)
        out = args.out_prefix.parent / f"{args.out_prefix.name}_{s}.json"
        json.dump({
            "scope_subfield": subfield, "scope_field": field, "method": "random_region",
            "match_size": n_match, "center_seed": args.seed + s,
            "n_articles_in_scope": n_art,
            "book": {
                "book": f"random_region_seed{s}", "book_id": f"random_region_seed{s}",
                "gap_size": len(region_ids), "boundary_cos": nn_sim,
                "overlap_with_real_gap": overlap,
                "sample_titles": [a_titles[j] for j in region[:25]],
                "gap_article_ids": region_ids},
        }, open(out, "w"), indent=1)
        LOGGER.info("wrote %s", out)


if __name__ == "__main__":
    main()
