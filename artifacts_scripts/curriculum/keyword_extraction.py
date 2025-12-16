"""Keyword extraction utilities for topic summarization.

Provides methods for extracting distinctive keywords from topic documents:
- TF-IDF scoring
- Log-odds ratio (Fightin' Words style) scoring
"""

from __future__ import annotations

import re
from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

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
}


class KeywordMethod(str, Enum):
    TFIDF = "tfidf"
    LOG_ODDS = "log_odds"


class DocWeighting(str, Enum):
    TOKEN = "token"
    EQUAL = "equal"


class TopicWeighting(str, Enum):
    POOLED = "pooled"
    EQUAL = "equal"

def _topic_docs_to_fractional_counts(
    topic_docs: List[List[str]],
    doc_weighting: DocWeighting,
) -> Tuple[Dict[str, float], float]:
    """Convert a topic's documents into (counts, total) for downstream scoring.

    - doc_weighting="token": standard bag-of-tokens counts (longer docs weigh more)
    - doc_weighting="equal": each document contributes unit mass via normalized term
      frequencies, so document length does not affect weight
    """
    if doc_weighting not in {DocWeighting.TOKEN, DocWeighting.EQUAL}:
        raise ValueError(f"Invalid doc_weighting: {doc_weighting}")

    if doc_weighting == DocWeighting.TOKEN:
        c = Counter()
        for tokens in topic_docs:
            c.update(tokens)
        total = float(sum(c.values()))
        return {k: float(v) for k, v in c.items()}, total

    # equal document weighting
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

def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric, filter short tokens.
    
    Args:
        text: Input text string.
        
    Returns:
        List of lowercase tokens.
    """
    # Lowercase and split on non-alphanumeric
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    # Filter empty and very short tokens
    return [t for t in tokens if len(t) >= 3]


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
        
        idf: Dict[str, float] = {}
        for w, d in df.items():
            # Use fractional df in IDF calculation
            idf[w] = np.log((n_docs + 1.0) / (d + 1.0)) + 1.0

    results: List[List[Tuple[str, float]]] = []
    for topic_idx in range(n_topics):
        counts = topic_counts[topic_idx]
        total = topic_totals[topic_idx]
        if total <= 0:
            results.append([])
            continue

        tfidf_scores: Dict[str, float] = {}
        for word, cnt in counts.items():
            # Skip stopwords
            if word in STOPWORDS:
                continue
            tf = float(cnt) / float(total)
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
) -> List[List[Tuple[str, float]]]:
    """Compute Fightin' Words z-scores using per-document tokens.

    With doc_weighting="equal", each document contributes unit mass via normalized
    within-document frequencies, so longer documents do not dominate.
    """
    n_topics = len(topic_doc_tokens)

    if topic_weighting not in {TopicWeighting.POOLED, TopicWeighting.EQUAL}:
        raise ValueError(f"Invalid topic_weighting: {topic_weighting}")

    topic_counts: List[Dict[str, float]] = []
    topic_totals: List[float] = []
    global_counts: Dict[str, float] = {}
    vocabulary: Set[str] = set()

    for docs in topic_doc_tokens:
        counts, total = _topic_docs_to_fractional_counts(docs, doc_weighting=doc_weighting)

        # Optional topic-level normalization: make each topic contribute equal total mass.
        # This affects ONLY the background construction (ambient distribution).
        if topic_weighting == TopicWeighting.EQUAL and total > 0:
            scale = 1.0 / float(total)
            counts = {w: float(c) * scale for w, c in counts.items()}
            total = 1.0

        topic_counts.append(counts)
        topic_totals.append(float(total))
        for w, c in counts.items():
            global_counts[w] = global_counts.get(w, 0.0) + float(c)
        vocabulary.update(counts.keys())

    vocab_size = len(vocabulary)
    if vocab_size == 0:
        return [[] for _ in range(n_topics)]

    global_total = float(sum(global_counts.values()))
    alpha_0 = prior_weight * vocab_size

    results: List[List[Tuple[str, float]]] = []
    for topic_idx in range(n_topics):
        counts_in = topic_counts[topic_idx]
        total_in = float(topic_totals[topic_idx])
        total_out = global_total - total_in

        scores: Dict[str, float] = {}
        for word in vocabulary:
            y_in = float(counts_in.get(word, 0.0))
            y_out = float(global_counts.get(word, 0.0)) - y_in

            y_in_smooth = y_in + prior_weight
            y_out_smooth = y_out + prior_weight

            n_in_smooth = total_in + alpha_0
            n_out_smooth = total_out + alpha_0

            # Guard against numerical edge cases when totals are tiny
            denom_in = max(n_in_smooth - y_in_smooth, 1e-12)
            denom_out = max(n_out_smooth - y_out_smooth, 1e-12)

            log_odds_in = np.log(y_in_smooth / denom_in)
            log_odds_out = np.log(y_out_smooth / denom_out)
            delta = log_odds_in - log_odds_out

            variance = 1.0 / y_in_smooth + 1.0 / y_out_smooth
            z_score = delta / np.sqrt(variance)
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
        
    Returns:
        List of lists of (word, score) tuples for each topic.
    """
    if method == KeywordMethod.TFIDF:
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
        )
    raise ValueError(f"Unknown keyword extraction method: {method}")
