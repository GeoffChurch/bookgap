"""Vendored ROUGE-1/2/L F1 (the ``rouge_score`` package is absent from the env).

Standard n-gram-overlap ROUGE-N (clipped counts) and LCS-based ROUGE-L over
token sequences. Divergences from the official package, documented per house
rule:

- **No Porter stemmer** (official ``use_stemmer=True`` runs Porter on tokens);
  scores here correspond to ``rouge_score`` with ``use_stemmer=False``.
- Tokenization is the same as ``rouge_score``'s default: lowercase, keep
  ``[a-z0-9]+`` runs (callers pass pre-tokenized sequences from
  ``textproc.word_tokens``).
- ROUGE-L is the flat-sequence LCS (``rougeL``), not the sentence-level union
  variant (``rougeLsum``).

Unit-tested against hand-computed toy cases in ``tests/test_rouge.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple, Sequence


class RougeScore(NamedTuple):
    precision: float
    recall: float
    f1: float


def _f1(overlap: int, hyp_total: int, ref_total: int) -> RougeScore:
    p = overlap / hyp_total if hyp_total else 0.0
    r = overlap / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return RougeScore(p, r, f)


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(ref: Sequence[str], hyp: Sequence[str], n: int) -> RougeScore:
    """ROUGE-N with clipped n-gram counts (the standard definition)."""
    ref_counts = _ngrams(ref, n)
    hyp_counts = _ngrams(hyp, n)
    overlap = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
    return _f1(overlap, sum(hyp_counts.values()), sum(ref_counts.values()))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest-common-subsequence length, O(len(a)*len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(ref: Sequence[str], hyp: Sequence[str]) -> RougeScore:
    """Flat-sequence ROUGE-L F1 (LCS precision vs hyp length, recall vs ref length)."""
    return _f1(lcs_length(ref, hyp), len(hyp), len(ref))


def score_all(ref: Sequence[str], hyp: Sequence[str]) -> dict[str, float]:
    """F1 for rouge1/rouge2/rougeL as a flat dict (the eval's scoring unit)."""
    return {
        "rouge1": rouge_n(ref, hyp, 1).f1,
        "rouge2": rouge_n(ref, hyp, 2).f1,
        "rougeL": rouge_l(ref, hyp).f1,
    }
