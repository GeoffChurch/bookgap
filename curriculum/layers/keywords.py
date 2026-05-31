"""Topic keyword extraction layer (log-odds + lattice phrase mining).

Re-exports the pure keyword-extraction primitives (``_keyword_extraction``,
lifted verbatim from ``artifacts_scripts/curriculum/keyword_extraction.py`` — it
was already numpy/stdlib-only) and provides :func:`keywords_for_topics`, the
per-topic orchestrator lifted from ``wiki_curriculum_demo.keywords_for_topics``
so the parity gate produces identical keyword tables.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from curriculum.layers._keyword_extraction import (
    BackgroundMode,
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    extract_topic_keywords,
    filter_stopwords,
    mine_phrases_lattice,
    parse_with_phrases,
    tokenize,
)

__all__ = [
    "BackgroundMode",
    "DocWeighting",
    "KeywordMethod",
    "TopicWeighting",
    "extract_topic_keywords",
    "filter_stopwords",
    "mine_phrases_lattice",
    "parse_with_phrases",
    "tokenize",
    "keywords_for_topics",
]


def keywords_for_topics(
    texts: List[str],
    point_to_cluster: NDArray[np.intp],
    masses: NDArray[np.floating],
    n_topics: int,
    *,
    samples_per_topic: int = 80,
    top_k: int = 12,
    bigrams: bool = True,
    bigram_iterations: int = 6,
    npmi_threshold: float = 0.4,
    min_boundary_be: float = 0.5,
    count_floor: int = 3,
    max_phrases: int = 5000,
) -> List[List[Tuple[str, float]]]:
    """Per-topic keywords via lattice phrase mining + log-odds.

    Byte-for-byte the recipe from ``wiki_curriculum_demo.keywords_for_topics``:
    sample the top-``samples_per_topic`` member docs per topic (by mass),
    iteratively mine multi-word phrases over the whole corpus, re-parse, filter
    stopwords, then run marginal-background log-odds (Fightin' Words) with
    ``weight_by_freq=True``.
    """
    topic_raw: List[List[List[str]]] = []
    for t in range(n_topics):
        idxs = np.where(point_to_cluster == t)[0]
        if idxs.size == 0:
            topic_raw.append([])
            continue
        order = np.argsort(masses[idxs])[::-1]
        pick = idxs[order[:samples_per_topic]]
        docs = []
        for i in pick:
            toks = tokenize(texts[int(i)])
            if toks:
                docs.append(toks)
        topic_raw.append(docs)

    flat = [doc for docs in topic_raw for doc in docs]
    parsed = flat
    if bigrams:
        phrase_vocab: set = set()
        for _ in range(bigram_iterations):
            new = mine_phrases_lattice(
                parsed,
                npmi_threshold=npmi_threshold,
                min_boundary_be=min_boundary_be,
                count_floor=count_floor,
                max_phrases=max_phrases,
            )
            new -= phrase_vocab
            if not new:
                break
            phrase_vocab |= new
            parsed = [parse_with_phrases(doc, phrase_vocab) for doc in flat]

    merged: List[List[List[str]]] = []
    off = 0
    for docs in topic_raw:
        merged.append(parsed[off:off + len(docs)])
        off += len(docs)

    topic_doc_tokens: List[List[List[str]]] = []
    for docs in merged:
        out = []
        for toks in docs:
            toks = filter_stopwords(toks)
            if toks:
                out.append(toks)
        topic_doc_tokens.append(out)

    return extract_topic_keywords(
        topic_doc_tokens,
        method=KeywordMethod.LOG_ODDS,
        top_k=top_k,
        doc_weighting=DocWeighting.EQUAL,
        topic_weighting=TopicWeighting.EQUAL,
        background_mode=BackgroundMode.MARGINAL,
        weight_by_freq=True,
    )
