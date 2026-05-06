"""Shared keyword + nearest-book helpers for gap-method scripts.

Lifted from ``gap_method_comparison_table.py`` so the Phase 1
book-existence filter can compute the same per-gap keywords and
top-books without duplicating the orchestration. Behavior is
intentionally identical to the previous inline pipeline; the
comparison table now imports from here.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from artifacts_scripts.curriculum.keyword_extraction import (
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


def _gap_to_raw_tokens(
    members: list[int],
    titles: list[str],
    abstracts: list[str],
) -> list[list[str]]:
    """Per-article tokenize (no stopword filter — bigram mining needs raw tokens)."""
    out = []
    for m in members:
        if m >= len(titles):
            continue
        text_parts = []
        if titles[m]:
            text_parts.append(titles[m])
        if abstracts[m]:
            text_parts.append(abstracts[m])
        text = " ".join(text_parts)
        toks = tokenize(text)
        if toks:
            out.append(toks)
    return out


def compute_gap_keywords(
    gaps: list[list[int]],
    titles: list[str],
    abstracts: list[str],
    top_k: int = 5,
    bigram_iterations: int = 2,
    verbose: bool = True,
) -> list[list[tuple[str, float]]]:
    """Compute distinguishing keywords per gap using the production
    log-odds + lattice-bigram pipeline. Joint phrase mining across all
    gaps so phrase vocab is shared; per-gap pooled-marginal background
    for the keyword score itself.

    Returns one list of (keyword, score) tuples per input gap.
    """
    raw_topic_tokens = [_gap_to_raw_tokens(members, titles, abstracts) for members in gaps]

    flat_raw = [doc for topic_docs in raw_topic_tokens for doc in topic_docs]
    if verbose:
        print(f"Mining phrases ({bigram_iterations} iters, lattice longest-match) "
              f"from {len(flat_raw)} docs...")
    phrase_vocab: set = set()
    parsed = flat_raw
    for it in range(bigram_iterations):
        t0 = time.time()
        new_phrases = mine_phrases_lattice(parsed, npmi_threshold=0.5, min_boundary_be=0.5)
        t_mine = time.time() - t0
        if not new_phrases:
            if verbose:
                print(f"  iter {it+1}: 0 new phrases ({t_mine:.1f}s); stopping")
            break
        prev = len(phrase_vocab)
        phrase_vocab |= new_phrases
        added = len(phrase_vocab) - prev
        t1 = time.time()
        parsed = [parse_with_phrases(doc, phrase_vocab) for doc in flat_raw]
        if verbose:
            print(f"  iter {it+1}: +{added} phrases ({t_mine:.1f}s + {time.time()-t1:.1f}s); "
                  f"total {len(phrase_vocab)}")

    merged_per_topic: list[list[list[str]]] = []
    offset = 0
    for topic_docs in raw_topic_tokens:
        n = len(topic_docs)
        merged_per_topic.append(parsed[offset:offset + n])
        offset += n

    topic_doc_tokens: list[list[list[str]]] = []
    for topic_docs in merged_per_topic:
        kept = []
        for tokens in topic_docs:
            tokens = filter_stopwords(tokens)
            if tokens:
                kept.append(tokens)
        topic_doc_tokens.append(kept)

    return extract_topic_keywords(
        topic_doc_tokens,
        method=KeywordMethod.LOG_ODDS,
        top_k=top_k,
        doc_weighting=DocWeighting.EQUAL,
        topic_weighting=TopicWeighting.POOLED,
        background_mode=BackgroundMode.MARGINAL,
        weight_by_freq=True,
    )


def compute_top_books(
    members: list[int],
    emb: np.ndarray,
    book_index,
    book_global_idx: np.ndarray,
    all_titles: np.ndarray,
    k: int = 3,
) -> list[tuple[int, str, int]]:
    """Return up to k (book_global_idx, title, vote_count) tuples,
    sorted by descending vote count. Empty list if no members.
    """
    if not members:
        return []
    gap_emb = np.ascontiguousarray(emb[members])
    _, I_b = book_index.search(gap_emb, k=1)
    nearest_books_global = book_global_idx[I_b[:, 0]]
    cnt = Counter(nearest_books_global.tolist())
    out: list[tuple[int, str, int]] = []
    for bi, c in cnt.most_common(k):
        title = all_titles[bi] if bi < len(all_titles) else None
        title = str(title) if title else "(no title)"
        out.append((int(bi), title, int(c)))
    return out
