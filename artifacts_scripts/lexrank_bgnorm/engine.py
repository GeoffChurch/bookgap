"""LexRank / bg-norm machinery on the bipartite sentence-word Gramian walk.

Implements the construction from ``docs/backlog/curriculum-applications.md``
("The bipartite sentence-word construction"):

    B[s, w]  = tf of word w in sentence s            (sparse CSR)
    P        = RowNorm(B) @ RowNorm(B^T)              (two-step stochastic walk)
    p        = PageRank(P, uniform teleport, a=0.85)  == plain LexRank centrality

and the Fightin'-Words-corrected selection score from ``curriculum/core/rank.py``:

    score = p^a * log((p + eps) / (g + eps)),  a = 0.5, eps = 1e-12  (canonical)

Single-document summarization has no seed set, so the background ``g`` is the
experiment's design decision. Two variants, both "vs corpus background":

- **g1** — length-normalized mean GLOBAL word frequency of the sentence
  (corpus unigram model over all articles' bodies), renormalized to a
  distribution over the article's sentences so that p and g live on the same
  scale (mirrors ``rank.py`` where g is a PageRank distribution).
- **g2** — PageRank mass on the Gramian where the word-hop is weighted by
  GLOBAL word frequencies instead of within-article tf
  (B_bg[s, w] = 1[w in s] * f_global(w)): the sentence's expected centrality
  under generic language.

Centrality is computed by the tracked, frozen ``curriculum.core.pagerank
.compute_pagerank`` (never-densify discipline: B, B^T and the S x S Gramian
product all stay scipy CSR; per-article S is a few hundred, so materializing
the SPARSE product is safe — the corpus-scale alternative would be factored
power iteration v <- RowNorm(B)(RowNorm(B^T) v)).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import normalize as sk_normalize

import curriculum.core.pagerank as _cpr
from curriculum.core.pagerank import compute_pagerank

# Silence the tqdm progress bar inside compute_pagerank for batch runs (runtime
# patch only; no tracked file is modified). _NoTqdm matches the call surface.


class _NoTqdm:
    def __init__(self, *a, **k):  # noqa: D401
        self.total = k.get("total")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def update(self, *a):
        return None

    def set_postfix(self, *a, **k):
        return None


def silence_pagerank_progress() -> None:
    _cpr._tqdm = _NoTqdm


BG_EXPONENT = 0.5   # canonical Fightin'-Words exponent (curriculum/config.py)
BG_EPS = 1e-12      # Laplace floor (curriculum/config.py)
PR_ALPHA = 0.85
PR_TOL = 1e-10
PR_MAX_ITER = 200


# --------------------------------------------------------------------------- #
# B matrix and the two-step walk
# --------------------------------------------------------------------------- #

def build_sentence_word_matrix(
    sent_tokens: Sequence[Sequence[str]],
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """Sparse tf incidence ``B[s, w]`` over an article's sentences x words."""
    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for s, toks in enumerate(sent_tokens):
        for t in toks:
            w = vocab.setdefault(t, len(vocab))
            rows.append(s)
            cols.append(w)
    n_s, n_w = len(sent_tokens), max(len(vocab), 1)
    B = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_s, n_w), dtype=np.float64
    )
    B.sum_duplicates()
    return B, vocab


def gramian_walk(B: sp.csr_matrix) -> sp.csr_matrix:
    """Two-step stochastic walk ``P = RowNorm(B) @ RowNorm(B^T)`` (sparse CSR).

    Rows of empty sentences stay all-zero; ``compute_pagerank`` treats them as
    sinks (their mass teleports), so no special-casing is needed.
    """
    P1 = sk_normalize(B, norm="l1", axis=1)               # sentence -> word
    P2 = sk_normalize(B.T.tocsr(), norm="l1", axis=1)     # word -> sentence
    return (P1 @ P2).tocsr()


def centrality(P: sp.csr_matrix) -> NDArray[np.float64]:
    """Uniform-teleport PageRank on a row-stochastic walk (plain LexRank ``p``)."""
    n = P.shape[0]
    seeds = np.full(n, 1.0 / n, dtype=np.float64)
    return compute_pagerank(
        P, seeds, alpha=PR_ALPHA, tol=PR_TOL, max_iter=PR_MAX_ITER, normalize=False
    )


# --------------------------------------------------------------------------- #
# backgrounds and the bg-norm selection score
# --------------------------------------------------------------------------- #

def bgnorm_score(
    p: NDArray[np.float64],
    g: NDArray[np.float64],
    exponent: float = BG_EXPONENT,
    eps: float = BG_EPS,
) -> NDArray[np.float64]:
    """``p^a * log((p+eps)/(g+eps))`` — mirrors ``curriculum/core/rank.py`` exactly."""
    p_fg = np.maximum(p, 0.0)
    return np.power(p_fg, exponent) * np.log((p_fg + eps) / (g + eps))


def g1_global_frequency(
    sent_tokens: Sequence[Sequence[str]],
    global_freq: Mapping[str, float],
) -> NDArray[np.float64]:
    """g1: per-sentence mean global unigram frequency, renormalized over sentences.

    Sentences with no content tokens get g = 0 (they are excluded from ranked
    selection upstream, see ``rankable_mask``).
    """
    raw = np.zeros(len(sent_tokens), dtype=np.float64)
    for i, toks in enumerate(sent_tokens):
        if toks:
            raw[i] = float(np.mean([global_freq.get(t, 0.0) for t in toks]))
    total = raw.sum()
    return raw / total if total > 0 else raw


def g2_generic_walk(
    B: sp.csr_matrix,
    vocab: Mapping[str, int],
    global_freq: Mapping[str, float],
) -> NDArray[np.float64]:
    """g2: PageRank on the Gramian whose word-hop weight is the GLOBAL frequency.

    ``B_bg[s, w] = 1[w in s] * f_global(w)`` — the sentence->word hop sends mass
    to words in proportion to how common they are in the whole corpus (instead
    of within-article tf), so mass concentrates on sentences reachable through
    generic vocabulary: expected centrality under generic language.
    """
    f = np.zeros(B.shape[1], dtype=np.float64)
    for tok, j in vocab.items():
        f[j] = global_freq.get(tok, 0.0)
    B_bin = B.copy()
    B_bin.data = np.ones_like(B_bin.data)
    B_bg = B_bin.multiply(f[np.newaxis, :]).tocsr()
    return centrality(gramian_walk(B_bg))


def rankable_mask(sent_tokens: Sequence[Sequence[str]]) -> NDArray[np.bool_]:
    """Sentences eligible for RANKED selection: at least one content token.

    A zero-token sentence is a degenerate node (teleport-only mass, g = 0):
    under the bg-norm log-lift it would score spuriously high, and under plain
    LexRank spuriously low-but-nonzero. Excluding them from every ranked method
    (lexrank, bg-norm, degree) keeps the comparison clean; lead-3 and random
    keep all sentences (they do not consult content).
    """
    return np.array([len(t) > 0 for t in sent_tokens], dtype=bool)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

def tfidf_degree(B: sp.csr_matrix) -> NDArray[np.float64]:
    """Degree baseline: sum of cosine similarities over tf-idf sentence vectors.

    idf is within-article: ``log(S / df_w)`` (plus 1 to keep ubiquitous words
    nonzero). degree_s = sum_{s'} cos(v_s, v_s') - 1 computed by two sparse
    matvecs (U @ (U^T @ 1)); never densified.
    """
    n_s = B.shape[0]
    df = np.asarray((B > 0).sum(axis=0)).ravel()
    idf = np.log(n_s / np.maximum(df, 1.0)) + 1.0
    U = sk_normalize(B.multiply(idf[np.newaxis, :]).tocsr(), norm="l2", axis=1)
    deg = np.asarray(U @ (U.T @ np.ones(n_s)))
    has_norm = np.asarray((U.power(2)).sum(axis=1)).ravel() > 0
    deg = deg - has_norm.astype(np.float64)  # remove self-similarity where defined
    return np.maximum(deg, 0.0)


# --------------------------------------------------------------------------- #
# budget-controlled selection
# --------------------------------------------------------------------------- #

def select_by_budget(
    ranking: Sequence[int],
    sent_lengths: Sequence[int],
    budget: int,
) -> list[int]:
    """Take sentences in ``ranking`` order until total length reaches ``budget``.

    The sentence that crosses the budget IS included (the standard
    fill-to-budget protocol; ROUGE F1 penalizes overshoot symmetrically across
    methods). Zero-length sentences are skipped. Returns document order.
    """
    chosen: list[int] = []
    total = 0
    for idx in ranking:
        if sent_lengths[idx] <= 0:
            continue
        chosen.append(idx)
        total += sent_lengths[idx]
        if total >= budget:
            break
    return sorted(chosen)


def ranking_from_scores(
    scores: NDArray[np.float64],
    mask: Optional[NDArray[np.bool_]] = None,
) -> list[int]:
    """Indices by descending score (stable; masked-out sentences excluded)."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    if mask is None:
        return [int(i) for i in order]
    return [int(i) for i in order if mask[i]]
