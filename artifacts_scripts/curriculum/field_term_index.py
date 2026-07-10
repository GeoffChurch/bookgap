"""Cached field-wide work×term index for keyword-concept seeding + the book-deficit gap detector.

docs/backlog/curriculum-keyword-concept-seeding.md. Build ONCE over the whole field (joblib-cached
by field+snapshot+params), reuse instantly. Gives K2W candidates the full field (not an embedding-
nearest prefilter that funnels into the pool's dominant basin) and supplies the field DF that makes
the lift "distinctive vs the whole field."

Uses sklearn CountVectorizer (C-level, one pass, min_df pruning) over each work's title+abstract,
unigrams+bigrams, binary. Returns (X_csr [works×terms, binary], vocab [terms], field_df [per-term
doc count]). W2K = column-sums over a work-set / |set| vs field_df; K2W = weighted column lookups
(do `.tocsc()` once in the caller for fast term-column access).

Build (memory-heavy ~ tens of GB on a 6M-work field; run as a tracked job):
  python -m bookgap.jobs run --prefix field_index -- \\
    uv run python -m artifacts_scripts.curriculum.field_term_index --field 17 --build
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.util import LOGGER, MEMORY, setup_logging

_STOP = "the a an of and or to in for on with is are be by we our this that as at from using".split()


@MEMORY.cache
def build_field_term_index(field: str, snapshot_date: str, min_df: int = 20,
                           max_features: int = 1_000_000):
    """Field-wide binary work×term (1+2gram) CSR + vocab + per-term field DF. Cached by args.

    ``snapshot_date`` is a cache key only (the cloud's own snapshot is used); pass the cloud's
    snapshot so a re-ingest invalidates the cache.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    titles = cloud.work_titles(); abstracts = cloud.work_abstracts()
    LOGGER.info("field %s: vectorizing %d works (1+2gram, min_df=%d, max_features=%d)",
                field, len(titles), min_df, max_features)
    docs = (f"{t}\n{a or ''}" for t, a in zip(titles, abstracts))
    cv = CountVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_features=max_features,
        token_pattern=r"(?u)\b[a-z0-9]{3,}\b", lowercase=True,
        stop_words=_STOP, binary=True, dtype=np.int8,
    )
    X = cv.fit_transform(docs).tocsr()                       # (n_works, |vocab|) binary
    vocab = cv.get_feature_names_out()
    field_df = np.asarray(X.sum(0)).ravel().astype(np.int64)
    LOGGER.info("built: X %s, nnz=%d (%.1f terms/work), vocab=%d",
                X.shape, X.nnz, X.nnz / max(X.shape[0], 1), len(vocab))
    return X, vocab, field_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="17")
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--build", action="store_true", help="trigger the cached build + print stats")
    args = ap.parse_args()
    setup_logging(log_name="field_index", log_level="INFO", include_time=False)
    X, vocab, field_df = build_field_term_index(
        args.field, args.snapshot_date, args.min_df, args.max_features)
    print(f"field {args.field}: X {X.shape}, nnz={X.nnz:,}, vocab={len(vocab):,}, "
          f"cached under bookgap MEMORY (joblib) — reused instantly on next call.")


if __name__ == "__main__":
    main()
