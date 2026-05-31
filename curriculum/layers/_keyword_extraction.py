"""Keyword extraction utilities for topic summarization.

Provides methods for extracting distinctive keywords from topic documents:
- TF-IDF scoring
- Log-odds ratio (Fightin' Words style) scoring
"""

from __future__ import annotations

import math
import re
from collections import Counter
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Common English stopwords
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "of", "in", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "could", "ought",
    "as", "until", "while", "because", "although", "also", "both", "either", "neither",
    "however", "therefore", "thus", "hence", "since", "yet", "still", "already",
    # Academic/paper-specific stopwords
    "paper", "study", "research", "results", "analysis", "method", "methods", "approach",
    "model", "models", "data", "based", "using", "used", "show", "shows", "shown",
    "present", "presents", "presented", "propose", "proposes", "proposed", "work",
    "new", "novel", "et", "al", "fig", "figure", "table", "section", "chapter",
    # Bibliographic / journal noise
    "page", "pages", "vol", "volume", "issue", "pp", "journal", "monthly", "quarterly",
    # XML / MathML / HTML markup residue. The pre-tokenization regexes strip
    # most markup, but some tokens (xmlns attribute names, the "mml" inside
    # `<mml:math xmlns:mml=...>` after tag stripping, URL fragments embedded
    # in attributes) survive. These appear ONLY in markup-heavy abstracts
    # (Phys Rev / APS / IEEE), so without explicit filtering they get
    # disproportionately high z-scores.
    "mml", "xmlns", "xlink", "xsi", "xsl", "xhtml", "dtd", "rng", "doctype",
    "http", "https", "www", "org", "com", "html", "xml",
    # LaTeX commands that survive pre-stripping (e.g. when authors omit the
    # backslash or write `ensuremath` as a literal word). Belt-and-suspenders
    # for whatever the strip regex misses. Greek letters are deliberately
    # NOT listed: the LaTeX-strip regex removes \alpha/\beta/etc. wholesale,
    # and in plain prose "alpha decay" / "beta version" carry real meaning.
    "ensuremath", "mathrm", "mathbb", "mathcal", "mathit", "mathsf", "mathbf",
    "textrm", "textit", "textbf", "frac", "sqrt", "infty", "cdot",
    "langle", "rangle", "rightarrow", "leftarrow",
    "xrightarrow", "xleftarrow",
}


class KeywordMethod(str, Enum):
    TFIDF = "tfidf"
    LOG_ODDS = "log_odds"


class DocWeighting(str, Enum):
    TOKEN = "token"
    EQUAL = "equal"
    RELEVANCE = "relevance"  # Weight each doc by its CircuitRank/PageRank mass


class TopicWeighting(str, Enum):
    POOLED = "pooled"
    EQUAL = "equal"


class BackgroundMode(str, Enum):
    """How to construct the comparison background for log-odds keyword extraction.

    The choice answers the question "vs. what?" that Fightin' Words' two-sample
    log-odds test poses for each topic.

    - MARGINAL (default): background is the curriculum-wide marginal P(w),
      built by pooling foreground counts across all topics (this topic
      included). The test is unary: "is P(w|t) higher than the corpus-prior
      P(w)?" Domain words common at similar rates in every topic naturally
      get delta ≈ 0 and demote themselves; topic-unique words get delta
      ≈ -log(P(t)). No "vs. what corpus" hyperparameter — the corpus
      supplies its own prior.

    - SUBTRACTIVE: legacy two-sample test of "this topic vs. all OTHER
      topics" — per-topic background is EQUAL-weighted, then this topic's
      contribution is subtracted from the global pool. Self-referential
      when the curriculum is mono-domain (the "other topics" share most
      domain vocabulary), which can boost domain words.
    """
    MARGINAL = "marginal"
    SUBTRACTIVE = "subtractive"

def _topic_docs_to_fractional_counts(
    topic_docs: List[List[str]],
    doc_weighting: DocWeighting,
    doc_weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], float]:
    """Convert a topic's documents into (counts, total) for downstream scoring.

    - doc_weighting="token": standard bag-of-tokens counts (longer docs weigh more)
    - doc_weighting="equal": each document contributes unit mass via normalized term
      frequencies, so document length does not affect weight
    - doc_weighting="relevance": each document contributes mass proportional to its
      relevance weight (normalized term frequencies scaled by doc_weights)
      
    Args:
        topic_docs: List of token lists, one per document
        doc_weighting: How to weight documents
        doc_weights: Per-document weights (required for RELEVANCE, ignored otherwise).
            Will be normalized to sum to len(topic_docs) to match EQUAL scale.
    """
    if doc_weighting not in {DocWeighting.TOKEN, DocWeighting.EQUAL, DocWeighting.RELEVANCE}:
        raise ValueError(f"Invalid doc_weighting: {doc_weighting}")

    if doc_weighting == DocWeighting.RELEVANCE:
        if doc_weights is None:
            raise ValueError("doc_weights required for RELEVANCE weighting")
        if len(doc_weights) != len(topic_docs):
            raise ValueError(
                f"doc_weights length ({len(doc_weights)}) must match "
                f"topic_docs length ({len(topic_docs)})"
            )

    if doc_weighting == DocWeighting.TOKEN:
        c: Counter[str] = Counter()
        for tokens in topic_docs:
            c.update(tokens)
        total = float(sum(c.values()))
        return {k: float(v) for k, v in c.items()}, total

    if doc_weighting == DocWeighting.EQUAL:
        # Each document contributes unit mass via normalized term frequencies
        counts: Dict[str, float] = {}
        total_docs = 0.0
        for tokens in topic_docs:
            if not tokens:
                continue
            per_doc = Counter(tokens)
            doc_len = float(sum(per_doc.values()))
            if doc_len <= 0:
                continue
            total_docs += 1.0
            for word, cnt in per_doc.items():
                counts[word] = counts.get(word, 0.0) + (float(cnt) / doc_len)
        return counts, total_docs

    # RELEVANCE weighting: normalize weights then scale each document's contribution
    # Normalize weights to sum to number of non-empty docs (matching EQUAL's scale)
    n_nonempty = sum(1 for tokens in topic_docs if tokens)
    if n_nonempty == 0:
        return {}, 0.0
    
    # doc_weights is guaranteed non-None here due to validation above
    weights = doc_weights  # type narrowing for type checker
    assert weights is not None  # for type checker
    
    # Normalize weights to sum to n_nonempty (same scale as EQUAL)
    weight_sum = sum(w for w, tokens in zip(weights, topic_docs) if tokens)
    if weight_sum <= 0:
        # Fallback to equal weighting if all weights are zero
        scale = 1.0
    else:
        scale = float(n_nonempty) / weight_sum
    
    counts = {}
    total_mass = 0.0
    for tokens, weight in zip(topic_docs, weights):
        if not tokens:
            continue
        per_doc = Counter(tokens)
        doc_len = float(sum(per_doc.values()))
        if doc_len <= 0:
            continue
        # Normalized weight for this document
        w = float(weight) * scale
        total_mass += w
        for word, cnt in per_doc.items():
            # Normalized term frequency scaled by document weight
            counts[word] = counts.get(word, 0.0) + (float(cnt) / doc_len) * w
    
    return counts, total_mass

_XML_TAG_RE = re.compile(r'<[^>]+>')
_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+\*?')
_URL_RE = re.compile(r'https?://\S+|www\.\S+\b')
_NUMERIC_RE = re.compile(r'^\d+$')


def _strip_markup(text: str) -> str:
    """Remove XML/HTML tags, LaTeX commands, and URLs.

    OpenAlex abstracts in physics journals (Phys Rev, APS, IEEE) ship with
    embedded MathML (``<mml:math xmlns:mml="...">``), LaTeX commands
    (``\\ensuremath{\\mu}``, ``\\mathrm{C}``), and URL fragments. The base
    ``re.split(r'[^a-zA-Z0-9]+')`` tokenizer turns these into noise tokens
    (``mml``, ``xmlns``, ``http``, ``ensuremath``, ``mathrm``) that
    Fightin' Words then ranks highly because markup density genuinely
    varies by topic. Strip them up-front so the ranking sees prose only.

    Order matters: XML tags first, since attribute values often contain
    URLs (``xmlns:mml="http://..."``) that the URL regex would otherwise
    greedily consume — eating past the closing ``>`` and breaking tag
    matching.
    """
    text = _XML_TAG_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _LATEX_CMD_RE.sub(' ', text)
    return text


def tokenize(text: str) -> List[str]:
    """Lowercase, strip markup, split on non-alphanumeric, filter junk.

    Drops XML/HTML tags, LaTeX commands, and URLs before splitting (see
    ``_strip_markup``), then filters tokens shorter than 3 chars and
    pure-numeric tokens (page numbers, years, etc.).

    Args:
        text: Input text string.

    Returns:
        List of lowercase tokens.
    """
    text = _strip_markup(text)
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if len(t) >= 3 and not _NUMERIC_RE.match(t)]


def filter_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    """Remove stopwords from token list.

    Args:
        tokens: List of tokens.
        stopwords: Set of stopwords to remove (uses default if None).

    Returns:
        Filtered token list.
    """
    if stopwords is None:
        stopwords = STOPWORDS
    return [t for t in tokens if t not in stopwords]


# Bigram-merge join character. Underscore is in no stopword and gets through
# the tokenizer's non-alphanumeric splitter unchanged, so merged bigrams
# survive ``filter_stopwords`` cleanly.
BIGRAM_JOIN = "_"


def mine_bigrams(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    max_bigrams: int = 5000,
    weight_by_freq: bool = True,
    stopwords: Optional[Set[str]] = None,
) -> Set[Tuple[str, str]]:
    """Mine high-NPMI adjacent-token bigrams from a corpus.

    NPMI(a,b) = log(P(a,b) / (P(a)P(b))) / -log(P(a,b)) is bounded in
    [-1, 1] and rewards genuine collocations independent of frequency.
    With ``weight_by_freq=True`` (default), bigrams are ranked by
    NPMI · count — the per-bigram contribution to mutual information
    in the unigram-pair sense — so low-count noise bigrams (which can
    score NPMI≈1 if both members appear only once together) get
    crushed without needing an explicit count floor.

    Stopword-bearing bigrams (either side stopworded) are dropped:
    keeps phrasing like "field theory" but drops "of the". Pass
    ``stopwords`` explicitly to override the default set.

    Args:
        doc_tokens: Per-document tokens, after ``tokenize`` but BEFORE
            ``filter_stopwords`` (so cross-stopword pairs aren't
            silently merged into phantom bigrams).
        npmi_threshold: Minimum NPMI to consider a candidate.
        max_bigrams: Cap on bigrams returned.
        weight_by_freq: Rank by NPMI · count (default; principled, no
            count floor needed) vs. raw NPMI.
        stopwords: Stopword set. ``None`` uses the module default.

    Returns:
        Set of ``(a, b)`` adjacent-token tuples to merge.
    """
    if stopwords is None:
        stopwords = STOPWORDS

    uni_counts: Counter[str] = Counter()
    bi_counts: Counter[Tuple[str, str]] = Counter()
    n_uni = 0
    n_bi = 0
    for doc in doc_tokens:
        if not doc:
            continue
        for tok in doc:
            uni_counts[tok] += 1
            n_uni += 1
        for a, b in zip(doc, doc[1:]):
            bi_counts[(a, b)] += 1
            n_bi += 1

    if n_uni == 0 or n_bi == 0:
        return set()

    scored: List[Tuple[float, Tuple[str, str]]] = []
    for (a, b), cab in bi_counts.items():
        if a in stopwords or b in stopwords:
            continue
        p_ab = cab / n_bi
        p_a = uni_counts[a] / n_uni
        p_b = uni_counts[b] / n_uni
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            continue
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= 0:
            continue
        npmi = pmi / denom
        if npmi < npmi_threshold:
            continue
        score = npmi * cab if weight_by_freq else npmi
        scored.append((score, (a, b)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return {bigram for _, bigram in scored[:max_bigrams]}


def apply_bigrams(
    tokens: List[str],
    bigrams: Set[Tuple[str, str]],
) -> List[str]:
    """Greedy left-to-right bigram merge in a single token sequence.

    Adjacent token pairs in ``bigrams`` are joined with ``BIGRAM_JOIN``
    (e.g. ``"bell theorem"`` -> ``"bell_theorem"``); the merged token
    consumes both positions, so each token participates in at most one
    bigram. Tokens not part of any merge are passed through untouched.
    """
    if not bigrams or not tokens:
        return list(tokens)
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and (tokens[i], tokens[i + 1]) in bigrams:
            out.append(f"{tokens[i]}{BIGRAM_JOIN}{tokens[i + 1]}")
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def _shannon_entropy_bits(counts: Counter) -> float:
    """Shannon entropy in bits of a Counter's relative frequencies."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


_BOS = "<BOS>"
_EOS = "<EOS>"


def _bigram_boundary_entropy(
    doc_tokens: List[List[str]],
) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """For each adjacent bigram (a, b) in the corpus, compute ``(left_be,
    right_be, count)``: Shannon entropy in bits over the *neighbouring*
    tokens around each occurrence of ``a b``.

    A real phrase like ``bell theorem`` has HIGH boundary entropy on both
    sides (varied surrounding context); a boilerplate sequence like
    ``articles citing`` (always part of "Reuse...articles citing this
    article") has LOW boundary entropy on both sides. The two signals
    distinguish genuine phrases from fixed-sequence aggregates that
    NPMI alone can't tell apart.

    BOS / EOS are tracked with sentinel tokens so phrases that frequently
    appear at sentence-edge get entropy credit for that variability.
    """
    left_ctx: Dict[Tuple[str, str], Counter] = {}
    right_ctx: Dict[Tuple[str, str], Counter] = {}
    for doc in doc_tokens:
        n = len(doc)
        for i in range(n - 1):
            bg = (doc[i], doc[i + 1])
            if bg not in left_ctx:
                left_ctx[bg] = Counter()
                right_ctx[bg] = Counter()
            left_ctx[bg][doc[i - 1] if i > 0 else _BOS] += 1
            right_ctx[bg][doc[i + 2] if i + 2 < n else _EOS] += 1
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for bg in left_ctx:
        out[bg] = (
            _shannon_entropy_bits(left_ctx[bg]),
            _shannon_entropy_bits(right_ctx[bg]),
            sum(left_ctx[bg].values()),
        )
    return out


def mine_phrases_lattice(
    doc_tokens: List[List[str]],
    *,
    npmi_threshold: float = 0.4,
    min_boundary_be: float = 0.5,
    be_quantile: float = 0.0,
    normalize_be: bool = False,
    count_floor: int = 3,
    max_phrases: int = 5000,
    stopwords: Optional[Set[str]] = None,
) -> Set[str]:
    """Mine adjacent-token bigrams using the combined-score recipe:

        score = NPMI(a,b) · count(a,b) · min(left_be, right_be)

    Each signal must clear its threshold (NPMI ≥ ``npmi_threshold``,
    count ≥ ``count_floor``, BE ≥ both ``min_boundary_be`` *and* the
    ``be_quantile``-th percentile of the candidate BE distribution).
    Stopword-bearing candidates rejected. Returns a set of joined
    strings (``"a_b"``) suitable for use with ``parse_with_phrases``.

    BE thresholds:
    - ``min_boundary_be`` is an absolute bit-count floor (default 0.5).
      Stable across iterations of the lattice loop, where the BE
      distribution shifts as more candidates get consumed by merges.
    - ``be_quantile`` is a data-driven complementary filter (default
      0.0 = disabled): the ``q``-th percentile of the candidate BE
      values. Useful as a *softer* per-corpus knob (e.g. 0.5 = keep
      the upper half), but in the iterative loop the distribution
      shifts each pass — quantile alone tends to admit boilerplate
      mega-grams in iters 2+ where the median BE drops. Combine with
      ``min_boundary_be`` as a hard floor.
    - ``normalize_be=True`` (default False) replaces the BE factor in
      the score with ``BE / log2(count + 1)`` — bounded in roughly
      ``[0, 1]``, the entropy analogue of NPMI's ``PMI / -log P(w)``
      normalisation. Each candidate is scaled by its own count rather
      than against the population, which is intrinsically iteration-
      stable: a candidate's normalised score doesn't depend on the
      overall distribution shifting between iterations. The threshold
      filters still apply to the *un-normalised* BE; this flag changes
      only the score function used for ranking.

    The combined score crushes both (i) chance pairs (low NPMI) and (ii)
    fixed-sequence boilerplate (low boundary entropy), even when each
    looks plausible under one signal alone. Probe data on the curriculum
    corpus shows it admits roughly 1/14 known boilerplate items in the
    top-200 versus 9/14 for NPMI · count alone. See
    ``artifacts_scripts/curriculum/be_phrase_probe.py``.

    For iterative use: pair with ``parse_with_phrases`` and re-parse the
    raw token sequences on each iteration with the *cumulative* phrase
    vocabulary. Re-parsing implements the semtok-style "exclusive
    entropy" effect — counts attributable to longer phrases never appear
    in shorter-phrase context distributions, because the longest-match
    parser consumes those positions on every iteration.
    """
    if stopwords is None:
        stopwords = STOPWORDS

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
        return set()

    bg_be = _bigram_boundary_entropy(doc_tokens)

    # Pass 1: collect all candidates that pass NPMI / count / stopword
    # filters; record their BE values for the quantile computation.
    candidates: List[Tuple[Tuple[str, str], int, float, float]] = []
    for (a, b), cab in bi.items():
        if cab < count_floor:
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
        l_be, r_be, _ = bg_be.get((a, b), (0.0, 0.0, 0))
        be = min(l_be, r_be)
        candidates.append(((a, b), cab, npmi, be))

    if not candidates:
        return set()

    # BE threshold: maximum of the absolute floor (``min_boundary_be``)
    # and the data-driven quantile (``be_quantile`` percentile of the
    # candidate BE distribution). The absolute floor stabilises behaviour
    # across iterations of the lattice loop; the quantile lets callers
    # opt into a stricter data-driven cut.
    quantile_floor = 0.0
    if be_quantile > 0.0:
        be_values = [be for _, _, _, be in candidates]
        quantile_floor = float(np.quantile(be_values, be_quantile))
    be_threshold = max(min_boundary_be, quantile_floor)

    scored: List[Tuple[float, str]] = []
    for (a, b), cab, npmi, be in candidates:
        if be < be_threshold:
            continue
        if normalize_be:
            # F-recipe: ``BE / log2(count + 1)`` ∈ [0, 1] roughly. The
            # per-candidate normalisation is intrinsically iteration-
            # stable — each candidate's score depends only on its own
            # count, not on the population distribution.
            be_factor = be / math.log2(cab + 1)
        else:
            be_factor = be
        score = npmi * cab * be_factor
        # Joining ``a`` and ``b`` with BIGRAM_JOIN produces the merged
        # surface form. If ``a`` or ``b`` is itself an already-merged
        # phrase (an underscore-joined string from a prior iteration),
        # the join correctly produces the longer-phrase string —
        # ``"einstein_podolsky"`` + ``"_"`` + ``"rosen"`` ==
        # ``"einstein_podolsky_rosen"``.
        merged = f"{a}{BIGRAM_JOIN}{b}"
        scored.append((score, merged))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {merged for _, merged in scored[:max_phrases]}


def parse_with_phrases(
    tokens: List[str],
    phrases: Iterable[str],
) -> List[str]:
    """Longest-match tokenisation of ``tokens`` against a phrase vocabulary.

    Each phrase is a joined string (``"bell_theorem"``,
    ``"einstein_podolsky_rosen"``); ``parse_with_phrases`` recovers the
    underlying-token tuple by splitting on ``BIGRAM_JOIN``, then matches
    the longest available phrase at each scan position. Each merged
    phrase consumes its constituent positions, so iterating mining +
    parsing produces a strict superset of the previous iteration's
    vocabulary — and computes boundary entropy on a corpus where
    already-merged phrases do not appear as separate adjacencies
    (the semtok-style "exclusive entropy" effect, achieved by re-parsing
    rather than explicit count subtraction).
    """
    phrase_set = set(phrases) if not isinstance(phrases, set) else phrases
    if not phrase_set or not tokens:
        return list(tokens)

    # Build trie keyed on underlying tokens. Sentinel ``__END__`` marks
    # nodes where a complete phrase ends.
    trie: Dict[str, Any] = {}
    for phrase_str in phrase_set:
        node = trie
        for tok in phrase_str.split(BIGRAM_JOIN):
            node = node.setdefault(tok, {})
        node["__END__"] = phrase_str

    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        node = trie
        longest_match: Optional[str] = None
        longest_len = 0
        for j in range(i, n):
            nxt = node.get(tokens[j])
            if nxt is None:
                break
            node = nxt
            end = node.get("__END__")
            if end is not None:
                longest_match = end
                longest_len = j - i + 1
        if longest_match is not None:
            out.append(longest_match)
            i += longest_len
        else:
            out.append(tokens[i])
            i += 1
    return out


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency (normalized by document length).
    
    Args:
        tokens: List of tokens in a document.
        
    Returns:
        Dictionary mapping token to frequency.
    """
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {token: count / total for token, count in counts.items()}


def compute_idf(
    topic_vocabularies: List[Set[str]],
    vocabulary: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute inverse document frequency across topics.
    
    Args:
        topic_vocabularies: List of vocabulary sets, one per topic.
        vocabulary: Optional set of words to compute IDF for (uses union if None).
        
    Returns:
        Dictionary mapping word to IDF score.
    """
    n_topics = len(topic_vocabularies)
    
    if vocabulary is None:
        vocabulary = set().union(*topic_vocabularies)
    
    idf: Dict[str, float] = {}
    for word in vocabulary:
        # Count topics containing this word
        doc_freq = sum(1 for vocab in topic_vocabularies if word in vocab)
        # IDF with smoothing to avoid division by zero
        idf[word] = np.log((n_topics + 1) / (doc_freq + 1)) + 1
    
    return idf


def compute_tfidf_scores(
    topic_tokens: List[List[str]],
    top_k: int = 20,
) -> List[List[Tuple[str, float]]]:
    """Compute TF-IDF scores for each topic.
    
    Args:
        topic_tokens: List of token lists, one per topic.
        top_k: Number of top keywords to return per topic.
        
    Returns:
        List of lists of (word, score) tuples for each topic.
    """
    n_topics = len(topic_tokens)
    
    # Compute TF for each topic
    topic_tfs = [compute_tf(tokens) for tokens in topic_tokens]
    
    # Compute vocabulary for each topic
    topic_vocabularies = [set(tf.keys()) for tf in topic_tfs]
    
    # Compute IDF across topics
    idf = compute_idf(topic_vocabularies)
    
    # Compute TF-IDF and extract top keywords
    results: List[List[Tuple[str, float]]] = []
    
    for topic_idx in range(n_topics):
        tf = topic_tfs[topic_idx]
        
        # Compute TF-IDF scores
        tfidf_scores: Dict[str, float] = {}
        for word, tf_score in tf.items():
            tfidf_scores[word] = tf_score * idf.get(word, 1.0)
        
        # Sort by score and take top_k
        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_words[:top_k])
    
    return results


def compute_tfidf_scores_docs(
    topic_doc_tokens: List[List[List[str]]],
    top_k: int = 20,
    doc_weighting: DocWeighting = DocWeighting.EQUAL,
    idf_scope: str = "auto",
) -> List[List[Tuple[str, float]]]:
    """Compute TF-IDF scores using per-document tokens.

    This supports document-length-invariant weighting via doc_weighting="equal".

    idf_scope:
    - "auto": Automatically selects "topics" for equal doc weighting (recommended),
      "documents" for token weighting
    - "documents": IDF computed over all sampled documents across all topics
      with WEIGHTED document frequency (weights match doc_weighting scheme)
    - "topics": IDF computed over topics (aggregated vocabularies)
    """
    if idf_scope not in {"auto", "documents", "topics"}:
        raise ValueError(f"Unknown idf_scope: {idf_scope}")
    
    # Auto-select optimal IDF scope based on doc_weighting
    if idf_scope == "auto":
        # With equal doc weighting, topics are the right granularity for IDF
        # With token weighting, documents provide finer-grained discrimination
        idf_scope = "topics" if doc_weighting == DocWeighting.EQUAL else "documents"

    n_topics = len(topic_doc_tokens)

    # Build per-topic counts/totals according to requested weighting
    topic_counts: List[Dict[str, float]] = []
    topic_totals: List[float] = []
    topic_vocabularies: List[Set[str]] = []
    for docs in topic_doc_tokens:
        counts, total = _topic_docs_to_fractional_counts(docs, doc_weighting=doc_weighting)
        topic_counts.append(counts)
        topic_totals.append(total)
        topic_vocabularies.append(set(counts.keys()))

    # Compute IDF
    idf: Dict[str, float]
    if idf_scope == "topics":
        idf = compute_idf(topic_vocabularies)
    else:
        # Document-level IDF with WEIGHTED document frequency
        # This matches the doc_weighting scheme used for TF computation
        df: Dict[str, float] = {}
        n_docs = 0.0
        
        if doc_weighting == DocWeighting.TOKEN:
            # Standard document frequency: count documents containing each word
            for docs in topic_doc_tokens:
                for tokens in docs:
                    if not tokens:
                        continue
                    n_docs += 1.0
                    for w in set(tokens):
                        df[w] = df.get(w, 0.0) + 1.0
        else:  # DocWeighting.EQUAL
            # Weighted document frequency: each doc contributes based on term's
            # within-document frequency (normalized by doc length)
            # This ensures common words that appear in every doc with high frequency
            # get properly penalized even when using equal doc weighting
            for docs in topic_doc_tokens:
                for tokens in docs:
                    if not tokens:
                        continue
                    doc_len = float(len(tokens))
                    if doc_len <= 0:
                        continue
                    n_docs += 1.0
                    # Weight each word's contribution by its normalized frequency in this doc
                    word_counts = Counter(tokens)
                    for w, cnt in word_counts.items():
                        # Add the word's fractional presence in this document
                        df[w] = df.get(w, 0.0) + (float(cnt) / doc_len)
        
        idf = {}
        for w, d in df.items():
            # Use fractional df in IDF calculation
            idf[w] = float(np.log((n_docs + 1.0) / (d + 1.0)) + 1.0)

    results: List[List[Tuple[str, float]]] = []
    for topic_idx in range(n_topics):
        counts = topic_counts[topic_idx]
        total = topic_totals[topic_idx]
        if total <= 0:
            results.append([])
            continue

        tfidf_scores: Dict[str, float] = {}
        for word, word_cnt in counts.items():
            # Skip stopwords
            if word in STOPWORDS:
                continue
            tf = float(word_cnt) / float(total)
            tfidf_scores[word] = tf * idf.get(word, 1.0)

        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_words[:top_k])

    return results


def compute_log_odds_ratio(
    topic_tokens: List[List[str]],
    top_k: int = 20,
    prior_weight: float = 0.01,
) -> List[List[Tuple[str, float]]]:
    """Compute Fightin' Words z-scores for each topic.
    
    Implements the method from Monroe, Colaresi, & Quinn (2008) "Fightin' Words:
    Lexical Feature Selection and Evaluation for Identifying the Content of 
    Political Conflict".
    
    The key insight is to compute a z-score by dividing the log-odds difference
    by its variance. This naturally penalizes rare words (which have high variance)
    without needing explicit stopword lists.
    
    For common words like "the", although they may appear slightly more in one topic,
    the difference is not statistically significant relative to the variance.
    For rare words, even if they only appear in one topic, the high variance
    means they won't get extreme scores unless they appear enough times.
    
    Args:
        topic_tokens: List of token lists, one per topic.
        top_k: Number of top keywords to return per topic.
        prior_weight: Symmetric Dirichlet prior (alpha). Smaller values = less smoothing.
            Default 0.01 works well per the original paper.
        
    Returns:
        List of lists of (word, z-score) tuples for each topic.
    """
    n_topics = len(topic_tokens)
    
    # Build global vocabulary and count
    global_counts: Counter[str] = Counter()
    topic_counts: List[Counter[str]] = []
    
    for tokens in topic_tokens:
        counts = Counter(tokens)
        topic_counts.append(counts)
        global_counts.update(counts)
    
    vocabulary = set(global_counts.keys())
    vocab_size = len(vocabulary)
    
    if vocab_size == 0:
        return [[] for _ in range(n_topics)]
    
    # Total tokens per topic and globally
    topic_totals = [sum(counts.values()) for counts in topic_counts]
    global_total = sum(global_counts.values())
    
    # Total prior mass
    alpha_0 = prior_weight * vocab_size
    
    results: List[List[Tuple[str, float]]] = []
    
    for topic_idx in range(n_topics):
        counts_in = topic_counts[topic_idx]
        total_in = topic_totals[topic_idx]
        
        # Background: all other topics combined
        total_out = global_total - total_in
        
        scores: Dict[str, float] = {}
        
        for word in vocabulary:
            y_in = counts_in.get(word, 0)  # count in topic
            y_out = global_counts[word] - y_in  # count in background
            
            # Smoothed counts with Dirichlet prior
            y_in_smooth = y_in + prior_weight
            y_out_smooth = y_out + prior_weight
            
            # Smoothed totals
            n_in_smooth = total_in + alpha_0
            n_out_smooth = total_out + alpha_0
            
            # Log-odds for topic vs background (Equation 22 in Monroe et al.)
            # delta = log(y_in / (n_in - y_in)) - log(y_out / (n_out - y_out))
            # Simplified: log((y_in * (n_out - y_out)) / (y_out * (n_in - y_in)))
            log_odds_in = np.log(y_in_smooth / (n_in_smooth - y_in_smooth))
            log_odds_out = np.log(y_out_smooth / (n_out_smooth - y_out_smooth))
            delta = log_odds_in - log_odds_out
            
            # Variance of the log-odds ratio (Equation 23 in Monroe et al.)
            # Based on the delta method approximation
            variance = 1.0 / y_in_smooth + 1.0 / y_out_smooth
            
            # Z-score: delta normalized by standard deviation
            # This is the key - it penalizes rare words automatically
            z_score = delta / np.sqrt(variance)
            
            scores[word] = z_score
        
        # Sort by score and take top_k
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_words[:top_k])
    
    return results


def compute_log_odds_ratio_docs(
    topic_doc_tokens: List[List[List[str]]],
    top_k: int = 20,
    prior_weight: float = 0.01,
    doc_weighting: DocWeighting = DocWeighting.EQUAL,
    topic_weighting: TopicWeighting = TopicWeighting.EQUAL,
    topic_doc_weights: Optional[List[List[float]]] = None,
    background_mode: BackgroundMode = BackgroundMode.MARGINAL,
    weight_by_freq: bool = True,
) -> List[List[Tuple[str, float]]]:
    """Compute Fightin' Words z-scores using per-document tokens.

    With doc_weighting="equal", each document contributes unit mass via normalized
    within-document frequencies, so longer documents do not dominate.

    With doc_weighting="relevance", each document contributes mass proportional to
    its relevance weight.

    Args:
        topic_doc_tokens: Topics -> documents -> tokens.
        top_k: Number of top keywords to return per topic.
        prior_weight: Symmetric Dirichlet prior (alpha).
        doc_weighting: How to weight documents within a topic.
        topic_weighting: How to weight topics in background construction
            (only meaningful when ``background_mode == SUBTRACTIVE``).
        topic_doc_weights: Per-document weights for each topic. Required if
            doc_weighting=RELEVANCE. Shape: topics -> documents -> weight.
        background_mode: How to build the comparison background. See the
            ``BackgroundMode`` docstring. Default ``MARGINAL`` is the
            unary "vs. corpus prior" formulation; legacy production used
            ``SUBTRACTIVE``.
        weight_by_freq: If True (default), post-multiply each word's z-score
            by its topic-internal count ``y_in``. This is the per-word
            contribution to the topic-keyword mutual information
            (≈ P(w,t) · PMI(w,t)): standardised effect size scaled by
            empirical mass. Crushes once-in-corpus tokens that pass the
            variance filter on small (singleton) topics. Set False to
            recover pure standardised z-scores.
    """
    n_topics = len(topic_doc_tokens)

    if topic_weighting not in {TopicWeighting.POOLED, TopicWeighting.EQUAL}:
        raise ValueError(f"Invalid topic_weighting: {topic_weighting}")
    if background_mode not in {BackgroundMode.MARGINAL, BackgroundMode.SUBTRACTIVE}:
        raise ValueError(f"Invalid background_mode: {background_mode}")

    if doc_weighting == DocWeighting.RELEVANCE and topic_doc_weights is None:
        raise ValueError("topic_doc_weights required for RELEVANCE doc_weighting")

    # Build foreground counts (used for topic-specific scores).
    foreground_counts: List[Dict[str, float]] = []
    foreground_totals: List[float] = []
    for topic_idx, docs in enumerate(topic_doc_tokens):
        if doc_weighting == DocWeighting.RELEVANCE:
            weights = topic_doc_weights[topic_idx] if topic_doc_weights else None
            counts, total = _topic_docs_to_fractional_counts(
                docs, doc_weighting=doc_weighting, doc_weights=weights,
            )
        else:
            counts, total = _topic_docs_to_fractional_counts(
                docs, doc_weighting=doc_weighting,
            )
        foreground_counts.append(counts)
        foreground_totals.append(float(total))

    vocabulary: Set[str] = set()
    for c in foreground_counts:
        vocabulary.update(c.keys())

    # Build background counts.
    if background_mode == BackgroundMode.MARGINAL:
        # Marginal: pool the foreground counts directly. Same doc_weighting as
        # foreground for self-consistency. ``topic_weighting=EQUAL`` rescales
        # each topic to unit mass before pooling so large topics don't dominate.
        global_counts: Dict[str, float] = {}
        global_total = 0.0
        for tidx in range(n_topics):
            t_counts = foreground_counts[tidx]
            t_total = foreground_totals[tidx]
            if topic_weighting == TopicWeighting.EQUAL and t_total > 0:
                scale = 1.0 / t_total
                for w, cval in t_counts.items():
                    global_counts[w] = global_counts.get(w, 0.0) + cval * scale
                global_total += 1.0
            else:
                for w, cval in t_counts.items():
                    global_counts[w] = global_counts.get(w, 0.0) + cval
                global_total += t_total
        # In MARGINAL mode there is no per-topic background array; every topic
        # compares against the same pooled marginal.
        background_counts_per_topic: Optional[List[Dict[str, float]]] = None
        background_totals_per_topic: Optional[List[float]] = None
    else:
        # SUBTRACTIVE: per-topic background with EQUAL doc-weighting (legacy).
        background_weighting = (
            DocWeighting.EQUAL
            if doc_weighting == DocWeighting.RELEVANCE
            else doc_weighting
        )
        background_counts_per_topic = []
        background_totals_per_topic = []
        global_counts = {}
        global_total = 0.0
        for docs in topic_doc_tokens:
            bg_counts, bg_total = _topic_docs_to_fractional_counts(
                docs, doc_weighting=background_weighting,
            )
            if topic_weighting == TopicWeighting.EQUAL and bg_total > 0:
                scale = 1.0 / float(bg_total)
                bg_counts = {w: float(cv) * scale for w, cv in bg_counts.items()}
                bg_total = 1.0
            background_counts_per_topic.append(bg_counts)
            background_totals_per_topic.append(float(bg_total))
            for w, cv in bg_counts.items():
                global_counts[w] = global_counts.get(w, 0.0) + float(cv)
            global_total += float(bg_total)
            vocabulary.update(bg_counts.keys())

    if not vocabulary:
        return [[] for _ in range(n_topics)]

    alpha_0 = prior_weight * len(vocabulary)

    results: List[List[Tuple[str, float]]] = []
    for topic_idx in range(n_topics):
        counts_in = foreground_counts[topic_idx]
        total_in = float(foreground_totals[topic_idx])

        # Empty topic (no docs sampled — e.g. when clustering produces fewer
        # real communities than ``desired_n_topics`` and the truncation pads
        # with size-0 placeholders): return an empty keyword list rather
        # than the deterministic-garbage tokens that fall out of FW
        # scoring against zero foreground mass.
        if total_in <= 0:
            results.append([])
            continue

        if background_mode == BackgroundMode.MARGINAL:
            total_out = global_total
        else:
            assert background_totals_per_topic is not None  # set in SUBTRACTIVE branch
            total_out_topic = background_totals_per_topic[topic_idx]
            total_out = global_total - total_out_topic

        scores: Dict[str, float] = {}
        for word in vocabulary:
            y_in = float(counts_in.get(word, 0.0))
            if background_mode == BackgroundMode.MARGINAL:
                y_out = float(global_counts.get(word, 0.0))
            else:
                assert background_counts_per_topic is not None  # set in SUBTRACTIVE branch
                counts_out_topic = background_counts_per_topic[topic_idx]
                y_out = (
                    float(global_counts.get(word, 0.0))
                    - float(counts_out_topic.get(word, 0.0))
                )

            y_in_smooth = y_in + prior_weight
            y_out_smooth = y_out + prior_weight

            n_in_smooth = total_in + alpha_0
            n_out_smooth = total_out + alpha_0

            denom_in = max(n_in_smooth - y_in_smooth, 1e-12)
            denom_out = max(n_out_smooth - y_out_smooth, 1e-12)

            log_odds_in = np.log(y_in_smooth / denom_in)
            log_odds_out = np.log(y_out_smooth / denom_out)
            delta = log_odds_in - log_odds_out

            variance = 1.0 / y_in_smooth + 1.0 / y_out_smooth
            z_score = delta / np.sqrt(variance)

            if weight_by_freq:
                # Per-word contribution to topic-keyword mutual information:
                # standardised effect size scaled by empirical mass. Low-count
                # words get crushed even when their per-occurrence z is high.
                z_score = z_score * y_in

            scores[word] = float(z_score)

        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results.append(sorted_words[:top_k])

    return results


def extract_topic_keywords(
    topic_doc_tokens: List[List[List[str]]],
    method: KeywordMethod = KeywordMethod.LOG_ODDS,
    top_k: int = 20,
    doc_weighting: DocWeighting = DocWeighting.EQUAL,
    idf_scope: str = "auto",
    prior_weight: float = 0.01,
    topic_weighting: TopicWeighting = TopicWeighting.POOLED,
    topic_doc_weights: Optional[List[List[float]]] = None,
    background_mode: BackgroundMode = BackgroundMode.MARGINAL,
    weight_by_freq: bool = True,
) -> List[List[Tuple[str, float]]]:
    """Extract distinctive keywords for each topic.

    Args:
        topic_doc_tokens: Topics -> documents -> tokens.
        method: Extraction method.
        top_k: Number of top keywords to return per topic.
        doc_weighting: How to weight documents within a topic.
        idf_scope: For TF-IDF, compute IDF across "documents", "topics", or "auto" (recommended).
        prior_weight: For log-odds, symmetric Dirichlet prior (alpha).
        topic_weighting: For log-odds, how to weight topics in background construction.
        topic_doc_weights: Per-document relevance weights for each topic. Required if
            doc_weighting=RELEVANCE. Shape: topics -> documents -> weight.
        background_mode: For log-odds, how to construct the background.
            See ``BackgroundMode``. Default ``MARGINAL`` (unary "vs. corpus
            prior" formulation).
        weight_by_freq: For log-odds, multiply z-scores by topic-internal
            count to rank by per-word MI contribution. Default True.

    Returns:
        List of lists of (word, score) tuples for each topic.
    """
    if method == KeywordMethod.TFIDF:
        # TF-IDF doesn't support relevance weighting currently
        if doc_weighting == DocWeighting.RELEVANCE:
            raise ValueError("TF-IDF does not support RELEVANCE doc_weighting yet")
        return compute_tfidf_scores_docs(
            topic_doc_tokens,
            top_k=top_k,
            doc_weighting=doc_weighting,
            idf_scope=idf_scope,
        )
    if method == KeywordMethod.LOG_ODDS:
        return compute_log_odds_ratio_docs(
            topic_doc_tokens,
            top_k=top_k,
            prior_weight=float(prior_weight),
            doc_weighting=doc_weighting,
            topic_weighting=topic_weighting,
            topic_doc_weights=topic_doc_weights,
            background_mode=background_mode,
            weight_by_freq=weight_by_freq,
        )
    raise ValueError(f"Unknown keyword extraction method: {method}")
