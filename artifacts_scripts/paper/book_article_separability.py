"""How linearly separable books are from articles in each embedder's space, before and after its denoiser.

The paper's embedder comparison argues that Gemma encodes more of the difference
between books and articles than GIST does, and that LPP removes more of it from
Gemma. This measures that directly: the cross-validated balanced accuracy and AUC
of a logistic regression telling books from articles, on each run's raw
embeddings and on the same embeddings after the run's denoiser.

The works are the same for every run: those that survive every run's semantic
deduplication. Within each topic, every such book is paired with an equal number
of that topic's articles drawn at random, so a classifier cannot separate the
classes by topic mix. Raw spaces differ in dimension (384 for GIST, 768 for
Gemma), which gives the classifier more to work with in the larger one; the
denoised spaces share a dimension, so their rows compare like for like.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.book_article_separability \\
        --seed 0 --folds 5 --inverse-regularization 1.0 \\
        --output paper/artifacts/book_article_separability.md RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from artifacts_scripts.common import friendly_embedder_name, markdown_provenance
from bookgap.cloud import (
    Cloud,
    _embeddings_store_path,
    corpus_dataset,
    ensure_cloud_data,
    get_cloud_projection,
    read_embeddings_rows,
)

# A convergence cap, not a tuning knob: ConvergenceWarning is raised as an error
# below, so a fit that hits it fails instead of reporting an unconverged score.
MAX_ITER = 5000


@dataclass(frozen=True)
class Retained:
    """One run's deduplicated works: corpus ids, and positions into its embeddings store."""
    cloud: Cloud
    store: Path
    position: dict[str, int]   # OpenAlex id -> row in the run's embeddings store


@dataclass(frozen=True)
class Score:
    embedder: str
    space: str
    dim: int
    n: int
    balanced_accuracy: float
    balanced_accuracy_sd: float
    auc: float


def retained_works(run_dir: Path) -> Retained:
    cloud = instantiate(OmegaConf.load(run_dir / "experiment_config.yaml").cloud)
    corpus_fp, sorted_idxs, retained = ensure_cloud_data(cloud)
    ids = corpus_dataset(cloud.filter_pair.pre, cloud.snapshot_date).to_table(columns=["id"]).column("id").to_pylist()
    position = {ids[int(sorted_idxs[int(p)])]: int(p) for p in retained}
    return Retained(cloud, _embeddings_store_path(corpus_fp, cloud.embedder), position)


def topic_matched_sample(types: NDArray, topics: NDArray, *, rng: np.random.Generator) -> NDArray[np.int64]:
    """Indices pairing, within each topic, every book with an article drawn at random,
    or every article with a book when the topic has fewer articles than books."""
    picked: list[int] = []
    for topic in sorted(set(topics)):
        books = np.flatnonzero((topics == topic) & (types == "book"))
        articles = np.flatnonzero((topics == topic) & (types == "article"))
        k = min(len(books), len(articles))
        picked += list(rng.choice(books, k, replace=False)) + list(rng.choice(articles, k, replace=False))
    return np.sort(np.array(picked, dtype=np.int64))


def separability(X: NDArray, y: NDArray, *, folds: int, seed: int, inverse_regularization: float) -> tuple[float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cv = cross_validate(
            LogisticRegression(C=inverse_regularization, max_iter=MAX_ITER), X, y,
            cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
            scoring=["balanced_accuracy", "roc_auc"],
        )
    return cv["test_balanced_accuracy"].mean(), cv["test_balanced_accuracy"].std(), cv["test_roc_auc"].mean()


def render(scores: list[Score], *, folds: int) -> str:
    lines = [
        "# Book/article separability by embedder",
        "",
        f"Logistic regression telling books from articles, {folds}-fold cross-validated, on the works every run "
        "retained after deduplication, with each topic's books paired with as many of its articles. "
        "Balanced accuracy is the mean over folds, with the standard deviation across folds.",
        "",
        "| embedder | space | dim | n | balanced accuracy | AUC |",
        "|---|---|---:|---:|---|---:|",
    ]
    lines += [f"| {s.embedder} | {s.space} | {s.dim} | {s.n} | {s.balanced_accuracy:.3f} ± {s.balanced_accuracy_sd:.3f} | {s.auc:.3f} |"
              for s in scores]
    return "\n".join(lines) + "\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--folds", type=int, required=True)
    ap.add_argument("--inverse-regularization", type=float, required=True, help="logistic regression's C")
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runs = [retained_works(d) for d in args.run_dirs]
    shared = sorted(set.intersection(*(set(r.position) for r in runs)))
    first = runs[0].cloud
    table = corpus_dataset(first.filter_pair.pre, first.snapshot_date).to_table(columns=["id", "type", "primary_topic_id"])
    meta = {i: (t, p) for i, t, p in zip(*(table.column(c).to_pylist() for c in ("id", "type", "primary_topic_id")))}
    types = np.array([meta[i][0] for i in shared], dtype=object)
    topics = np.array([meta[i][1] for i in shared], dtype=object)
    pick = topic_matched_sample(types, topics, rng=np.random.default_rng(args.seed))
    y = (types[pick] == "book").astype(int)

    scores: list[Score] = []
    for run in runs:
        rows = np.array([run.position[shared[i]] for i in pick], dtype=np.int64)
        raw = np.asarray(read_embeddings_rows(run.store, rows), dtype=np.float32)
        denoised = np.asarray(get_cloud_projection(run.cloud).apply(raw), dtype=np.float32)
        name = friendly_embedder_name(run.cloud.embedder.model_name)
        for space, X in (("raw", raw), (str(run.cloud.denoiser), denoised)):
            acc, sd, auc = separability(X, y, folds=args.folds, seed=args.seed,
                                        inverse_regularization=args.inverse_regularization)
            scores.append(Score(name, space, X.shape[1], len(y), acc, sd, auc))
            logging.info("%s %s: balanced accuracy %.3f, AUC %.3f", name, space, acc, auc)

    text = render(scores, folds=args.folds) + markdown_provenance(__spec__.name)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
