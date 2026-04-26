"""Prototype: NPMI-based bigram mining + external background for keywords.

Two extensions to the production keyword extraction, evaluated side-by-side:

1. NPMI-thresholded bigram phrase mining. Bigrams whose components are both
   stopword-free and whose NPMI crosses ``npmi_threshold`` get merged in-place
   (``"bell theorem"`` -> ``"bell_theorem"``), and the resulting mixed-vocab
   corpus is fed to the existing log-odds machinery unchanged.

2. External background corpus. Instead of computing each topic's background
   from "all *other* curriculum topics" (the production default — self-
   referential when the curriculum lives in one domain), sample N random
   non-curriculum works from the broader OpenAlex parquet cache and use
   those as the background. The Fightin' Words z-score then asks "what's
   distinctive vs. broader academic writing" rather than "vs. other curriculum
   topics." Domain words (``theory``, ``quantum``, ``first``) and
   journal-website boilerplate (``articles_citing``, ``reflect_usage``)
   that are uniformly distributed across the corpus get penalised
   automatically.

The probe runs both the original (curriculum-internal background) and the new
(external background) variants on the same curriculum and prints
side-by-side keywords per topic.

Usage:
    python -m artifacts_scripts.curriculum.bigram_keywords_probe \\
        artifacts/curriculum/<run_dir>

Run as a one-off probe; not wired into the curriculum pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow.dataset as ds

from artifacts_scripts.curriculum.keyword_extraction import (
    STOPWORDS,
    DocWeighting,
    TopicWeighting,
    compute_log_odds_ratio_docs,
    filter_stopwords,
    tokenize,
)


def _load_texts_for_filtered_ids(
    filtered_ids: List[str],
    parquet_root: Path,
) -> Dict[str, str]:
    """Pull title+abstract for each work ID from the per-scope parquet cache."""
    ids_set = set(filtered_ids)
    id_to_text: Dict[str, str] = {}
    files = list(parquet_root.glob("**/*.parquet"))
    print(f"  scanning {len(files)} parquet files...")
    for pf in files:
        try:
            d = ds.dataset(pf)
            t = d.to_table(columns=["id", "display_name", "abstract"]).to_pylist()
            for row in t:
                wid = row["id"]
                if wid in ids_set and wid not in id_to_text:
                    title = row.get("display_name") or ""
                    abstract = row.get("abstract") or ""
                    id_to_text[wid] = f"{title}\n{abstract}"
        except Exception:
            pass
        if len(id_to_text) >= len(ids_set):
            break
    return id_to_text


def _shannon_entropy(counts: Counter) -> float:
    """Shannon entropy in bits of a Counter's relative frequencies. 0 if empty."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H


def _compute_bigram_boundary_entropy(
    doc_tokens: List[List[str]],
) -> Dict[Tuple[str, str], Tuple[float, float, int, int]]:
    """For each adjacent bigram (a,b) in the corpus, compute left/right boundary
    entropy: the Shannon entropy (bits) over the *neighboring* tokens that
    surround occurrences of "a b".

    - left_be((a,b))  = entropy of P(W | "_ a b") — what *precedes* the bigram
    - right_be((a,b)) = entropy of P(W | "a b _") — what *follows* the bigram

    Boundary BOS/EOS contributions are counted with a sentinel symbol so a
    phrase that's frequently sentence-initial / -final still gets entropy
    credit for that variability.

    A phrase like "bell theorem" should have HIGH boundary entropy on both
    sides (varied surrounding context), while a boilerplate sequence like
    "articles citing" (always part of "Reuse ... articles citing this article")
    has LOW boundary entropy on both sides.

    Returns: dict mapping (a, b) -> (left_be, right_be, n_left_obs, n_right_obs).
    """
    BOS = "<BOS>"
    EOS = "<EOS>"
    left_ctx: Dict[Tuple[str, str], Counter] = {}
    right_ctx: Dict[Tuple[str, str], Counter] = {}
    for doc in doc_tokens:
        n = len(doc)
        for i in range(n - 1):
            bg = (doc[i], doc[i + 1])
            if bg not in left_ctx:
                left_ctx[bg] = Counter()
                right_ctx[bg] = Counter()
            left_ctx[bg][doc[i - 1] if i > 0 else BOS] += 1
            right_ctx[bg][doc[i + 2] if i + 2 < n else EOS] += 1

    out: Dict[Tuple[str, str], Tuple[float, float, int, int]] = {}
    for bg in left_ctx:
        l = _shannon_entropy(left_ctx[bg])
        r = _shannon_entropy(right_ctx[bg])
        out[bg] = (l, r, sum(left_ctx[bg].values()), sum(right_ctx[bg].values()))
    return out


def _mine_bigrams_be_only(
    doc_tokens: List[List[str]],
    *,
    min_boundary_be: float = 1.5,
    min_count: int = 3,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Mine bigrams using ONLY branching entropy of the surrounding context
    (option 1 of the design space): a real phrase has clean boundaries on
    both sides, regardless of its internal stickiness.

    Score = ``min(left_be, right_be)`` — the harder boundary is the limiting
    one. Bigrams below ``min_boundary_be`` on either side, or with global
    count below ``min_count``, are dropped. Stopword-bearing candidates
    rejected as in NPMI mining.

    A small count floor is unavoidable here — a once-only bigram has
    boundary entropy 0 by construction (only one observation), so the
    threshold alone can't distinguish noise from low-count signal. NPMI's
    ``score = NPMI · count`` recipe is unavailable here because there's
    no well-defined "frequency-weighted entropy"; the score is intrinsic
    to the boundary distribution.

    Returns: list of (score, a, b, count) sorted high-to-low.
    """
    bg_be = _compute_bigram_boundary_entropy(doc_tokens)
    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), (l, r, n_left, _n_right) in bg_be.items():
        if a in stopwords or b in stopwords:
            continue
        if n_left < min_count:
            continue
        score = min(l, r)
        if score < min_boundary_be:
            continue
        scored.append((score, a, b, n_left))
    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _bigram_npmi_be_records(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    min_count: int = 3,
    stopwords: Set[str] = STOPWORDS,
) -> Tuple[
    List[Tuple[Tuple[str, str], int, float, float, float]],
    float,
]:
    """Build per-candidate records (bigram, count, npmi, l_be, r_be).

    Also returns the corpus marginal unigram entropy ``H_W`` in bits,
    used by the alpha · H_W threshold strategy. Lifted out of the
    individual mining functions so the threshold-strategy recipes can
    share the heavy lifting.
    """
    uni: Counter[str] = Counter()
    bi: Counter[Tuple[str, str]] = Counter()
    n_uni = 0
    n_bi = 0
    for doc in doc_tokens:
        if not doc:
            continue
        for tok in doc:
            uni[tok] += 1
            n_uni += 1
        for a, b in zip(doc, doc[1:]):
            bi[(a, b)] += 1
            n_bi += 1
    if n_uni == 0:
        return [], 0.0
    # H_W: marginal context entropy = entropy of unigram distribution in bits
    H_W = 0.0
    for c in uni.values():
        if c > 0:
            p = c / n_uni
            H_W -= p * math.log2(p)
    if n_bi == 0:
        return [], H_W
    bg_be = _compute_bigram_boundary_entropy(doc_tokens)
    records: List[Tuple[Tuple[str, str], int, float, float, float]] = []
    for (a, b), cab in bi.items():
        if cab < min_count:
            continue
        if a in stopwords or b in stopwords:
            continue
        p_ab = cab / n_bi
        p_a = uni[a] / n_uni
        p_b = uni[b] / n_uni
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            continue
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= 0:
            continue
        npmi = pmi / denom
        if npmi < npmi_threshold:
            continue
        l_be, r_be, _, _ = bg_be.get((a, b), (0.0, 0.0, 0, 0))
        records.append(((a, b), cab, npmi, l_be, r_be))
    return records, H_W


def _mine_quantile_be(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    be_quantile: float = 0.5,
    min_count: int = 3,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Recipe D: quantile-based BE threshold.

    Compute ``min(left_be, right_be)`` for every candidate; threshold at
    the ``be_quantile`` percentile of the *candidate-distribution* BE
    values. Candidates above the quantile pass; below fail.

    The threshold is data-driven: it adapts to whatever BE distribution
    the corpus produces. ``be_quantile=0.5`` keeps the upper-half of
    candidates by BE; ``0.25`` keeps the upper three-quarters.

    Returns list of (score = NPMI · count · BE, a, b, count).
    """
    records, _ = _bigram_npmi_be_records(
        doc_tokens, npmi_threshold=npmi_threshold,
        min_count=min_count, stopwords=stopwords,
    )
    if not records:
        return []
    be_values = [min(l, r) for _, _, _, l, r in records]
    threshold = float(np.quantile(be_values, be_quantile))
    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), cab, npmi, l, r in records:
        be = min(l, r)
        if be < threshold:
            continue
        score = npmi * cab * be
        scored.append((score, a, b, cab))
    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _mine_marginal_be(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    alpha: float = 0.4,
    min_count: int = 3,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Recipe E: marginal-entropy reference threshold.

    Threshold BE at ``alpha · H_W`` where H_W is the corpus-marginal
    unigram entropy. Intuition: "the bigram's surrounding context is at
    least ``alpha``-as-varied as the corpus's typical context distribution."

    H_W is computed from the same corpus and adapts to corpus size /
    vocabulary diversity automatically; ``alpha`` is the only hand-set
    parameter and has a clean interpretation.

    Returns list of (score = NPMI · count · BE, a, b, count).
    """
    records, H_W = _bigram_npmi_be_records(
        doc_tokens, npmi_threshold=npmi_threshold,
        min_count=min_count, stopwords=stopwords,
    )
    if not records:
        return []
    threshold = alpha * H_W
    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), cab, npmi, l, r in records:
        be = min(l, r)
        if be < threshold:
            continue
        score = npmi * cab * be
        scored.append((score, a, b, cab))
    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _mine_normalized_be(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    min_norm_be: float = 0.5,
    min_count: int = 3,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Recipe F: count-normalised BE.

    Normalise BE by ``log2(count + 1)`` so it lies in roughly ``[0, 1]``:
    1 = "every observation has a distinct neighbour" (maximally varied),
    0 = "all observations share one neighbour" (fixed sequence). This
    handles the asymmetry that low-count bigrams are mechanically capped
    at low absolute BE — a 5-occurrence bigram has max BE ≈ 2.3 bits
    even with perfectly uniform context. The normalisation is the
    entropy analogue of NPMI's ``PMI / -log P(w)`` trick.

    Threshold ``min_norm_be`` is in ``[0, 1]`` and has a clean
    interpretation: 0.5 means "at least sqrt(count) distinct neighbours."

    Returns list of (score = NPMI · count · normalized_BE, a, b, count).
    """
    records, _ = _bigram_npmi_be_records(
        doc_tokens, npmi_threshold=npmi_threshold,
        min_count=min_count, stopwords=stopwords,
    )
    if not records:
        return []
    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), cab, npmi, l, r in records:
        be = min(l, r)
        denom = math.log2(cab + 1)
        if denom <= 0:
            continue
        norm_be = be / denom
        if norm_be < min_norm_be:
            continue
        score = npmi * cab * norm_be
        scored.append((score, a, b, cab))
    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _mine_bigrams_npmi_be(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    min_boundary_be: float = 0.5,
    min_count: int = 3,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Mine bigrams using NPMI · count · min(left_be, right_be) (option 2).

    Combines all three signals multiplicatively: a candidate must (i) stick
    together more than chance (NPMI), (ii) appear often enough to matter
    (count), and (iii) have varied surrounding context (boundary entropy)
    — clearing all three to score highly. Boilerplate aggregates clear
    NPMI · count but fail boundary entropy; chance pairs clear boundary
    entropy but fail NPMI.

    Returns: list of (score, a, b, count) sorted high-to-low.
    """
    # Compute NPMI counts in one pass (mirrors _mine_bigrams).
    uni: Counter[str] = Counter()
    bi: Counter[Tuple[str, str]] = Counter()
    n_uni = 0
    n_bi = 0
    for doc in doc_tokens:
        if not doc:
            continue
        for tok in doc:
            uni[tok] += 1
            n_uni += 1
        for a, b in zip(doc, doc[1:]):
            bi[(a, b)] += 1
            n_bi += 1
    bg_be = _compute_bigram_boundary_entropy(doc_tokens)

    if n_uni == 0 or n_bi == 0:
        return []

    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), cab in bi.items():
        if cab < min_count:
            continue
        if a in stopwords or b in stopwords:
            continue
        p_ab = cab / n_bi
        p_a = uni[a] / n_uni
        p_b = uni[b] / n_uni
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            continue
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= 0:
            continue
        npmi = pmi / denom
        if npmi < npmi_threshold:
            continue
        l_be, r_be, _, _ = bg_be.get((a, b), (0.0, 0.0, 0, 0))
        be_score = min(l_be, r_be)
        if be_score < min_boundary_be:
            continue
        score = npmi * cab * be_score
        scored.append((score, a, b, cab))
    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _mine_bigrams(
    doc_tokens: List[List[str]],
    *,
    count_floor: int = 5,
    npmi_threshold: float = 0.4,
    max_bigrams: int = 5000,
    stopwords: Set[str] = STOPWORDS,
) -> List[Tuple[float, str, str, int]]:
    """Score adjacent-token bigrams by NPMI; keep above threshold.

    Stopwords are excluded as bigram members (a bigram with a stopword on
    either side is filtered out post-scoring), but they are kept in the
    raw stream for bigram counting so we don't merge across them
    (e.g. "field of theory" -> "field theory" is rejected because
    we count "field of" and "of theory", not "field theory").

    Returns: list of (npmi, a, b, count) sorted high-to-low, capped at
    ``max_bigrams``.
    """
    uni: Counter[str] = Counter()
    bi: Counter[Tuple[str, str]] = Counter()
    n_uni = 0
    n_bi = 0
    for doc in doc_tokens:
        if not doc:
            continue
        for tok in doc:
            uni[tok] += 1
            n_uni += 1
        for a, b in zip(doc, doc[1:]):
            bi[(a, b)] += 1
            n_bi += 1

    if n_uni == 0 or n_bi == 0:
        return []

    scored: List[Tuple[float, str, str, int]] = []
    for (a, b), cab in bi.items():
        if cab < count_floor:
            continue
        if a in stopwords or b in stopwords:
            continue
        p_ab = cab / n_bi
        p_a = uni[a] / n_uni
        p_b = uni[b] / n_uni
        if p_a == 0 or p_b == 0 or p_ab == 0:
            continue
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= 0:
            continue
        npmi = pmi / denom
        if npmi >= npmi_threshold:
            scored.append((npmi, a, b, cab))

    scored.sort(reverse=True)
    return scored[:max_bigrams]


def _sample_external_docs(
    parquet_root: Path,
    exclude_ids: Set[str],
    n_docs: int = 5000,
    seed: int = 42,
) -> List[str]:
    """Sample non-curriculum docs uniformly across the parquet cache.

    Walks parquet files in shuffled order, reads each, shuffles its rows,
    skips any row whose work ID is in ``exclude_ids``, and accumulates
    title+abstract texts until ``n_docs`` is reached. Not perfectly
    uniform across the population (small parquets are over-represented
    by row), but good enough as a "broader academic writing" baseline.
    """
    import random
    rng = random.Random(seed)
    files = list(parquet_root.glob("**/*.parquet"))
    rng.shuffle(files)
    docs: List[str] = []
    for pf in files:
        if len(docs) >= n_docs:
            break
        try:
            d = ds.dataset(pf)
            t = d.to_table(columns=["id", "display_name", "abstract"]).to_pylist()
            rng.shuffle(t)
            for row in t:
                if row["id"] in exclude_ids:
                    continue
                title = row.get("display_name") or ""
                abstract = row.get("abstract") or ""
                if not title and not abstract:
                    continue
                docs.append(f"{title}\n{abstract}")
                if len(docs) >= n_docs:
                    break
        except Exception:
            pass
    return docs


def _topic_docs_to_counts(docs: List[List[str]]) -> Tuple[Dict[str, float], float]:
    """EQUAL doc-weighting: each non-empty doc contributes unit mass via
    normalised within-doc term frequencies. Matches the production
    ``DocWeighting.EQUAL`` semantics."""
    counts: Dict[str, float] = {}
    total = 0.0
    for tokens in docs:
        if not tokens:
            continue
        per_doc = Counter(tokens)
        doc_len = float(sum(per_doc.values()))
        if doc_len <= 0:
            continue
        total += 1.0
        for w, c in per_doc.items():
            counts[w] = counts.get(w, 0.0) + (float(c) / doc_len)
    return counts, total


def _topic_docs_to_relevance_counts(
    docs: List[List[str]], weights: List[float],
) -> Tuple[Dict[str, float], float]:
    """RELEVANCE foreground weighting: each non-empty doc contributes
    mass proportional to its (normalised) relevance weight via
    normalised term frequencies. Matches production semantics."""
    nonempty = [(d, w) for d, w in zip(docs, weights) if d]
    if not nonempty:
        return {}, 0.0
    n_nonempty = float(len(nonempty))
    weight_sum = sum(w for _, w in nonempty)
    scale = n_nonempty / weight_sum if weight_sum > 0 else 1.0
    counts: Dict[str, float] = {}
    total = 0.0
    for tokens, weight in nonempty:
        per_doc = Counter(tokens)
        doc_len = float(sum(per_doc.values()))
        if doc_len <= 0:
            continue
        w = float(weight) * scale
        total += w
        for word, c in per_doc.items():
            counts[word] = counts.get(word, 0.0) + (float(c) / doc_len) * w
    return counts, total


def _fw_marginal_background(
    foreground_topic_docs: List[List[List[str]]],
    foreground_doc_weights: List[List[float]],
    *,
    prior_weight: float = 0.01,
    top_k: int = 12,
    weight_by_freq: bool = False,
) -> List[List[Tuple[str, float]]]:
    """Fightin' Words against the curriculum-wide marginal as background.

    Same z-score machinery as the production version, but the background
    is the marginal P(w) across *all* topics (including this one) rather
    than ``global - this topic``. The test becomes a *unary* one: "is
    P(w|t) higher than the corpus-prior P(w)?"

    Conceptually subsumes both the curriculum-internal and external-
    background views: there's no "vs. what corpus" hyperparameter — the
    marginal is whatever the foreground union supplies. Domain words
    common at similar rates in every topic get delta ≈ 0 (since
    P(w|t) ≈ P(w)); topic-unique words get delta ≈ -log(P(t)). The
    variance term still penalises low-count noise.

    Foreground per-topic uses RELEVANCE weighting; the marginal is built
    from the same RELEVANCE-weighted pooled counts so the test is
    self-consistent in weighting.

    ``weight_by_freq=True`` post-multiplies the z-score by the topic's
    relevance-weighted count for the word (``y_in``). This is the
    pointwise contribution to the topic-keyword mutual information:
    standardised effect size scaled by empirical mass. It crushes
    once-in-corpus tokens (count ~1) regardless of their per-occurrence
    z-score, addressing the singleton-noise failure mode where
    low-count words pass the variance filter on small topics.
    """
    n_topics = len(foreground_topic_docs)

    fg_counts: List[Dict[str, float]] = []
    fg_totals: List[float] = []
    for tidx in range(n_topics):
        c, t = _topic_docs_to_relevance_counts(
            foreground_topic_docs[tidx], foreground_doc_weights[tidx],
        )
        fg_counts.append(c)
        fg_totals.append(t)

    # Pool foreground into a single marginal: sum the RELEVANCE-weighted
    # per-topic counts and totals. This is the curriculum-wide P(w).
    marginal_counts: Dict[str, float] = {}
    marginal_total = 0.0
    for tidx in range(n_topics):
        for w, c in fg_counts[tidx].items():
            marginal_counts[w] = marginal_counts.get(w, 0.0) + c
        marginal_total += fg_totals[tidx]

    vocab: Set[str] = set(marginal_counts.keys())
    vocab_size = len(vocab)
    if vocab_size == 0:
        return [[] for _ in range(n_topics)]
    alpha_0 = prior_weight * vocab_size

    results: List[List[Tuple[str, float]]] = []
    for tidx in range(n_topics):
        counts_in = fg_counts[tidx]
        total_in = fg_totals[tidx]
        scores: Dict[str, float] = {}
        for word in vocab:
            y_in = float(counts_in.get(word, 0.0))
            # Background = full marginal (NOT marginal minus this topic).
            y_out = float(marginal_counts.get(word, 0.0))

            y_in_smooth = y_in + prior_weight
            y_out_smooth = y_out + prior_weight
            n_in_smooth = total_in + alpha_0
            n_out_smooth = marginal_total + alpha_0

            denom_in = max(n_in_smooth - y_in_smooth, 1e-12)
            denom_out = max(n_out_smooth - y_out_smooth, 1e-12)

            log_odds_in = math.log(y_in_smooth / denom_in)
            log_odds_out = math.log(y_out_smooth / denom_out)
            delta = log_odds_in - log_odds_out

            variance = 1.0 / y_in_smooth + 1.0 / y_out_smooth
            z = delta / math.sqrt(variance)
            if weight_by_freq:
                # Multiply by relevance-weighted topic count: rank by
                # (effect size / sd) * mass = pointwise contribution to
                # topic-keyword mutual information.
                scores[word] = float(z * y_in)
            else:
                scores[word] = float(z)

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results.append(sorted_scores)
    return results


def _fw_external_background(
    foreground_topic_docs: List[List[List[str]]],
    foreground_doc_weights: List[List[float]],
    external_docs: List[List[str]],
    *,
    prior_weight: float = 0.01,
    top_k: int = 12,
) -> List[List[Tuple[str, float]]]:
    """Fightin' Words with an external (non-curriculum) background pool.

    Foreground per topic: RELEVANCE-weighted normalised term frequencies.
    Background: a single EQUAL-weighted pool of external docs. Unlike the
    production version, the background is NOT topic-minus-self — it's a
    fixed external corpus, so domain words common to the whole curriculum
    no longer cancel out via self-reference.
    """
    n_topics = len(foreground_topic_docs)

    fg_counts: List[Dict[str, float]] = []
    fg_totals: List[float] = []
    for tidx in range(n_topics):
        c, t = _topic_docs_to_relevance_counts(
            foreground_topic_docs[tidx], foreground_doc_weights[tidx],
        )
        fg_counts.append(c)
        fg_totals.append(t)

    bg_counts, bg_total = _topic_docs_to_counts(external_docs)

    vocab: Set[str] = set(bg_counts.keys())
    for c in fg_counts:
        vocab.update(c.keys())
    vocab_size = len(vocab)
    if vocab_size == 0:
        return [[] for _ in range(n_topics)]
    alpha_0 = prior_weight * vocab_size

    results: List[List[Tuple[str, float]]] = []
    for tidx in range(n_topics):
        counts_in = fg_counts[tidx]
        total_in = fg_totals[tidx]
        scores: Dict[str, float] = {}
        for word in vocab:
            y_in = float(counts_in.get(word, 0.0))
            y_out = float(bg_counts.get(word, 0.0))

            y_in_smooth = y_in + prior_weight
            y_out_smooth = y_out + prior_weight
            n_in_smooth = total_in + alpha_0
            n_out_smooth = bg_total + alpha_0

            denom_in = max(n_in_smooth - y_in_smooth, 1e-12)
            denom_out = max(n_out_smooth - y_out_smooth, 1e-12)

            log_odds_in = math.log(y_in_smooth / denom_in)
            log_odds_out = math.log(y_out_smooth / denom_out)
            delta = log_odds_in - log_odds_out

            variance = 1.0 / y_in_smooth + 1.0 / y_out_smooth
            z = delta / math.sqrt(variance)
            scores[word] = float(z)

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results.append(sorted_scores)
    return results


def _apply_bigrams(
    doc_tokens: List[List[str]],
    bigrams: Set[Tuple[str, str]],
) -> List[List[str]]:
    """Greedy left-to-right bigram merge on each doc.

    Merged bigrams use ``"_"`` as the join char so they survive downstream
    stopword filtering (no underscore is in any stopword) and remain
    recognisable as multi-word terms in the output.
    """
    out: List[List[str]] = []
    for doc in doc_tokens:
        merged: List[str] = []
        i = 0
        n = len(doc)
        while i < n:
            if i + 1 < n and (doc[i], doc[i + 1]) in bigrams:
                merged.append(f"{doc[i]}_{doc[i + 1]}")
                i += 2
            else:
                merged.append(doc[i])
                i += 1
        out.append(merged)
    return out


def run_probe(
    run_dir: Path,
    *,
    npmi_threshold: float = 0.4,
    count_floor: int = 5,
    max_bigrams: int = 5000,
    top_k_keywords: int = 12,
    external_background_n: int = 5000,
) -> None:
    print(f"=== Bigram keyword probe on {run_dir} ===")
    with open(run_dir / "filtered_work_ids.json") as f:
        filtered_ids: List[str] = json.load(f)
    point_to_cluster: np.ndarray = np.load(run_dir / "point_to_cluster.npy")
    point_masses: np.ndarray = np.load(run_dir / "point_masses.npy")
    n_topics = int(point_to_cluster.max()) + 1
    print(f"  top-k: {len(filtered_ids)} works; topics: {n_topics}")

    parquet_root = Path("cache/works_parquet")
    print("  loading text for top-k work IDs...")
    id_to_text = _load_texts_for_filtered_ids(filtered_ids, parquet_root)
    print(f"  found texts: {len(id_to_text)} / {len(filtered_ids)}")
    texts = [id_to_text.get(wid, "") for wid in filtered_ids]

    print("  tokenizing (markup-strip, numeric-filter, no stopwords yet)...")
    raw_doc_tokens: List[List[str]] = [tokenize(t) for t in texts]

    print(
        f"  mining bigrams: count_floor={count_floor}, "
        f"npmi_threshold={npmi_threshold}, max_bigrams={max_bigrams}..."
    )
    scored = _mine_bigrams(
        raw_doc_tokens,
        count_floor=count_floor,
        npmi_threshold=npmi_threshold,
        max_bigrams=max_bigrams,
    )
    print(f"  bigrams above threshold: {len(scored)}")
    print("  top-30 bigrams by NPMI:")
    for npmi, a, b, c in scored[:30]:
        print(f"    {a}_{b:<25} npmi={npmi:.3f}  count={c}")

    # Apply merging
    bigram_set = {(a, b) for _, a, b, _ in scored}
    merged_doc_tokens = _apply_bigrams(raw_doc_tokens, bigram_set)
    # Now drop unigram stopwords (bigrams contain underscore so they pass).
    merged_doc_tokens = [filter_stopwords(d) for d in merged_doc_tokens]

    # Group by topic, weight by relevance
    topic_doc_tokens: List[List[List[str]]] = [[] for _ in range(n_topics)]
    topic_doc_weights: List[List[float]] = [[] for _ in range(n_topics)]
    for work_idx, tokens in enumerate(merged_doc_tokens):
        if not tokens:
            continue
        topic_idx = int(point_to_cluster[work_idx])
        if topic_idx < 0 or topic_idx >= n_topics:
            continue
        topic_doc_tokens[topic_idx].append(tokens)
        topic_doc_weights[topic_idx].append(float(point_masses[work_idx]))

    print("  running Fightin' Words (curriculum-internal background)...")
    kw_internal = compute_log_odds_ratio_docs(
        topic_doc_tokens,
        top_k=top_k_keywords,
        prior_weight=0.01,
        doc_weighting=DocWeighting.RELEVANCE,
        topic_weighting=TopicWeighting.EQUAL,
        topic_doc_weights=topic_doc_weights,
    )

    print("  running Fightin' Words (vs. global marginal)...")
    kw_marginal = _fw_marginal_background(
        topic_doc_tokens,
        topic_doc_weights,
        prior_weight=0.01,
        top_k=top_k_keywords,
    )

    print("  running Fightin' Words (vs. global marginal, freq-weighted)...")
    kw_marginal_fw = _fw_marginal_background(
        topic_doc_tokens,
        topic_doc_weights,
        prior_weight=0.01,
        top_k=top_k_keywords,
        weight_by_freq=True,
    )

    print(f"  sampling {external_background_n} external (non-curriculum) docs...")
    exclude_ids = set(filtered_ids)
    ext_texts = _sample_external_docs(
        parquet_root, exclude_ids, n_docs=external_background_n,
    )
    print(f"  external docs sampled: {len(ext_texts)}")
    print("  tokenizing + bigramming external docs (same vocab)...")
    ext_raw_tokens = [tokenize(t) for t in ext_texts]
    ext_merged = _apply_bigrams(ext_raw_tokens, bigram_set)
    ext_merged = [filter_stopwords(d) for d in ext_merged]

    print("  running Fightin' Words (external background)...")
    kw_external = _fw_external_background(
        topic_doc_tokens,
        topic_doc_weights,
        ext_merged,
        prior_weight=0.01,
        top_k=top_k_keywords,
    )

    print("\n=== Side-by-side keywords per topic ===")
    for tidx in range(n_topics):
        n_docs = len(topic_doc_tokens[tidx])
        terms_int = [w for w, _ in kw_internal[tidx]]
        terms_mar = [w for w, _ in kw_marginal[tidx]]
        terms_mfw = [w for w, _ in kw_marginal_fw[tidx]]
        terms_ext = [w for w, _ in kw_external[tidx]]
        print(f"\nTopic {tidx:2d} ({n_docs} docs)")
        print(f"  internal-bg:        {', '.join(terms_int)}")
        print(f"  marginal-bg:        {', '.join(terms_mar)}")
        print(f"  marginal-bg + freq: {', '.join(terms_mfw)}")
        print(f"  external-bg:        {', '.join(terms_ext)}")

    # Cross-topic overlap diagnostic (the user's "intra-curricular diversity"
    # objective): for each variant, count how often a keyword shows up in
    # multiple topics' top-K. Lower = more diverse.
    def _crowding_score(per_topic_kws: List[List[Tuple[str, float]]]) -> Tuple[float, int]:
        """Return (avg_topics_per_word, n_words_in_>=3_topics)."""
        word_topic_count: Counter[str] = Counter()
        for kws in per_topic_kws:
            for w, _ in kws:
                word_topic_count[w] += 1
        if not word_topic_count:
            return (0.0, 0)
        avg = sum(word_topic_count.values()) / len(word_topic_count)
        n_crowded = sum(1 for c in word_topic_count.values() if c >= 3)
        return (avg, n_crowded)

    print("\n=== Cross-topic crowding (lower is better) ===")
    for name, kws in [
        ("internal-bg       ", kw_internal),
        ("marginal-bg       ", kw_marginal),
        ("marginal-bg + freq", kw_marginal_fw),
        ("external-bg       ", kw_external),
    ]:
        avg, crowded = _crowding_score(kws)
        print(f"  {name}: avg topics/keyword = {avg:.3f}, "
              f"keywords in 3+ topics = {crowded}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--npmi-threshold", type=float, default=0.4)
    parser.add_argument("--count-floor", type=int, default=5)
    parser.add_argument("--max-bigrams", type=int, default=5000)
    parser.add_argument("--top-k-keywords", type=int, default=12)
    parser.add_argument("--external-background-n", type=int, default=5000)
    args = parser.parse_args()
    run_probe(
        args.run_dir,
        npmi_threshold=args.npmi_threshold,
        count_floor=args.count_floor,
        max_bigrams=args.max_bigrams,
        top_k_keywords=args.top_k_keywords,
        external_background_n=args.external_background_n,
    )


if __name__ == "__main__":
    main()
