"""How linearly separable books are from articles in each embedder's space, before and after its denoiser.

The paper's embedder comparison argues that Gemma encodes more of the difference
between books and articles than GIST does, and that LPP removes more of it from
Gemma. This measures that directly.

For each run and each space (the run's raw embeddings, and the same embeddings
after its denoiser), ONE logistic regression is fit to tell books from articles,
under k-fold cross-validation over a single pooled sample. Its held-out
predictions are scored two ways:

- pooled: balanced accuracy and AUC over each held-out fold, averaged over folds.
  Topics weigh by their book counts, and one decision threshold serves them all.
- per topic: balanced accuracy and AUC within each topic, from the same held-out
  predictions, averaged over topics with equal weight. Only topics with at least
  ``--min-topic-pairs`` book/article pairs count, since a topic with a handful of
  books gives a near-arbitrary score.

Every score is repeated for each of ``--seeds``, each seed drawing its own
articles and folds, and reported as the mean and standard deviation over them, so
no number rests on one draw of articles.

The sample is the same for every run: the works that survive every run's
semantic deduplication, with every book in a topic paired with an equal number
of that topic's articles drawn at random, so a work's topic says nothing about
its class. Because the works are shared, topics also compare run against run:
the report counts the topics in which each later run is more separable than the
first.

Raw spaces differ in dimension (384 for GIST, 768 for Gemma), which gives the
classifier more to work with in the larger one; the denoised spaces share a
dimension, so their rows compare like for like.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.book_article_separability \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --folds 5 --inverse-regularization 1.0 --min-topic-pairs 10 \\
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
class Heldout:
    """One classifier's cross-validated results: per-fold scores, and every work's held-out prediction."""
    fold_balanced_accuracy: NDArray
    fold_auc: NDArray
    probability: NDArray   # P(book), from the fold that held the work out
    predicted: NDArray     # the classifier's label at its own threshold


@dataclass(frozen=True)
class Score:
    embedder: str
    space: str
    dim: int
    n: int
    balanced_accuracy: float
    balanced_accuracy_sd: float
    auc: float
    topic_scores: dict[str, tuple[float, float]]   # topic -> (balanced accuracy, AUC)

    @property
    def topic_balanced_accuracy(self) -> float:
        return float(np.mean([b for b, _ in self.topic_scores.values()]))

    @property
    def topic_auc(self) -> float:
        return float(np.mean([a for _, a in self.topic_scores.values()]))


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


def cross_validated(X: NDArray, y: NDArray, *, folds: int, seed: int, inverse_regularization: float) -> Heldout:
    fold_bacc, fold_auc = [], []
    probability, predicted = np.empty(len(y)), np.empty(len(y), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            model = LogisticRegression(C=inverse_regularization, max_iter=MAX_ITER).fit(X[train], y[train])
            probability[test] = model.predict_proba(X[test])[:, 1]
            predicted[test] = model.predict(X[test])
            fold_bacc.append(balanced_accuracy_score(y[test], predicted[test]))
            fold_auc.append(roc_auc_score(y[test], probability[test]))
    return Heldout(np.array(fold_bacc), np.array(fold_auc), probability, predicted)


def per_topic_scores(y: NDArray, heldout: Heldout, topics: NDArray, *, min_pairs: int) -> dict[str, tuple[float, float]]:
    """Balanced accuracy and AUC within each topic with at least ``min_pairs`` book/article pairs."""
    out = {}
    for topic in sorted(set(topics)):
        mine = topics == topic
        if y[mine].sum() >= min_pairs:
            out[topic] = (balanced_accuracy_score(y[mine], heldout.predicted[mine]),
                          roc_auc_score(y[mine], heldout.probability[mine]))
    return out


def _spread(values: list[float]) -> str:
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{np.mean(values):.3f} ± {sd:.3f}"


def render(by_seed: list[list[Score]], *, seeds: list[int], folds: int, min_pairs: int) -> str:
    """One row per embedder and space: each score's mean ± standard deviation over the
    seeds, every seed drawing its own articles and folds."""
    lines = [
        "# Book/article separability by embedder",
        "",
        f"One logistic regression per embedder and space, telling books from articles under {folds}-fold "
        "cross-validation on a single pooled sample: the works every run retained after deduplication, with each "
        "topic's books paired with as many of its articles. Pooled scores are over each held-out fold, averaged "
        f"over folds; per-topic scores are the same held-out predictions scored within each topic with at least "
        f"{min_pairs} pairs, averaged over topics. Each score is the mean ± standard deviation over {len(seeds)} "
        f"seeds ({', '.join(map(str, seeds))}), each drawing its own articles and folds.",
        "",
        "| embedder | space | dim | n | pooled balanced accuracy | pooled AUC | topics | per-topic balanced accuracy | per-topic AUC |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for j, s in enumerate(by_seed[0]):
        rows = [scores[j] for scores in by_seed]
        topics = sorted({len(r.topic_scores) for r in rows})
        lines.append(
            f"| {s.embedder} | {s.space} | {s.dim} | {s.n} | {_spread([r.balanced_accuracy for r in rows])} | "
            f"{_spread([r.auc for r in rows])} | {'–'.join(map(str, topics))} | "
            f"{_spread([r.topic_balanced_accuracy for r in rows])} | {_spread([r.topic_auc for r in rows])} |")
    first = by_seed[0][0].embedder
    pairs = [(i, j) for i, a in enumerate(by_seed[0]) if a.embedder == first
             for j, b in enumerate(by_seed[0]) if b.embedder != first and (a.space == "raw") == (b.space == "raw")]
    if pairs:
        lines += ["", f"Per topic, against {first} (topics scored in both), over the seeds:", ""]
        for i, j in pairs:
            counts = []
            for scores in by_seed:
                a, b = scores[i], scores[j]
                shared = sorted(set(a.topic_scores) & set(b.topic_scores))
                counts.append((sum(b.topic_scores[t][1] > a.topic_scores[t][1] for t in shared), len(shared)))
            higher = sorted({h for h, _ in counts})
            of = sorted({n for _, n in counts})
            lines.append(f"- {by_seed[0][j].embedder} {by_seed[0][j].space} has the higher AUC than {first} "
                         f"{by_seed[0][i].space} in {'–'.join(map(str, higher))} of {'–'.join(map(str, of))} topics.")
    return "\n".join(lines) + "\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="one sample of articles and folds per seed")
    ap.add_argument("--folds", type=int, required=True)
    ap.add_argument("--inverse-regularization", type=float, required=True, help="logistic regression's C")
    ap.add_argument("--min-topic-pairs", type=int, required=True, help="smallest topic the per-topic scores include")
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

    by_seed: list[list[Score]] = []
    for seed in args.seeds:
        pick = topic_matched_sample(types, topics, rng=np.random.default_rng(seed))
        y = (types[pick] == "book").astype(int)
        scores: list[Score] = []
        for run in runs:
            rows = np.array([run.position[shared[i]] for i in pick], dtype=np.int64)
            raw = np.asarray(read_embeddings_rows(run.store, rows), dtype=np.float32)
            denoised = np.asarray(get_cloud_projection(run.cloud).apply(raw), dtype=np.float32)
            name = friendly_embedder_name(run.cloud.embedder.model_name)
            for space, X in (("raw", raw), (str(run.cloud.denoiser), denoised)):
                heldout = cross_validated(X, y, folds=args.folds, seed=seed,
                                          inverse_regularization=args.inverse_regularization)
                score = Score(name, space, X.shape[1], len(y), heldout.fold_balanced_accuracy.mean(),
                              heldout.fold_balanced_accuracy.std(), heldout.fold_auc.mean(),
                              per_topic_scores(y, heldout, topics[pick], min_pairs=args.min_topic_pairs))
                scores.append(score)
                logging.info("seed %d, %s %s: pooled balanced accuracy %.3f, per-topic AUC %.3f", seed, name, space,
                             score.balanced_accuracy, score.topic_auc)
        by_seed.append(scores)

    text = render(by_seed, seeds=args.seeds, folds=args.folds, min_pairs=args.min_topic_pairs) + markdown_provenance(__spec__.name)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
