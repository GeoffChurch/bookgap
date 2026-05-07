"""Tests for Phase 1 book-existence filter helpers.

Covers the pure-logic pieces (tokenization, Jaccard overlap, the
book-existence signal classification). Full-pipeline correctness is
validated empirically against T10181.
"""
import pytest

from artifacts_scripts._keyword_helpers import (
    compute_book_existence_signal,
    jaccard_set,
    tokenize_for_overlap,
)


class TestTokenizeForOverlap:
    def test_lowercases_and_splits_on_non_word(self):
        assert tokenize_for_overlap("Word-Sense Disambiguation") == {"word", "sense", "disambiguation"}

    def test_drops_stopwords(self):
        assert tokenize_for_overlap("A Pocket Guide to Vocabulary") == {"pocket", "guide", "vocabulary"}

    def test_drops_short_tokens(self):
        # 1-char tokens dropped (punctuation residue).
        assert "a" not in tokenize_for_overlap("a b cat")
        assert "cat" in tokenize_for_overlap("a b cat")

    def test_empty_input(self):
        assert tokenize_for_overlap("") == set()
        assert tokenize_for_overlap(None) == set()


class TestJaccardSet:
    def test_identical_sets_one(self):
        assert jaccard_set({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets_zero(self):
        assert jaccard_set({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        # |A ∩ B| = 1, |A ∪ B| = 3 → 1/3
        assert jaccard_set({"a", "b"}, {"a", "c"}) == pytest.approx(1.0 / 3.0)

    def test_both_empty_zero(self):
        assert jaccard_set(set(), set()) == 0.0


class TestComputeBookExistenceSignal:
    def test_high_concentration_aligned_book_flags_book_exists(self):
        # 50 members, 30 voting same book whose title shares keywords.
        out = compute_book_existence_signal(
            gap_size=50,
            top_book_votes=30,
            top_book_title="Word-Sense Disambiguation",
            top_keywords=["word_sense_disambiguation", "wsd", "sense", "disambiguation", "ambiguity"],
            vote_concentration_threshold=0.25,
            title_overlap_threshold=0.2,
            templated_concentration_threshold=0.5,
        )
        assert out["vote_concentration"] == pytest.approx(0.6)
        assert out["title_keyword_overlap"] > 0.2
        assert out["book_exists_flag"] is True
        assert out["templated_leak_flag"] is False

    def test_high_concentration_random_book_flags_templated_leak(self):
        out = compute_book_existence_signal(
            gap_size=50,
            top_book_votes=40,
            top_book_title="A Pocket Guide to Vocabulary",
            top_keywords=["n3c", "concept", "standardized", "ontology", "snomed"],
        )
        assert out["vote_concentration"] == pytest.approx(0.8)
        assert out["title_keyword_overlap"] == 0.0
        assert out["book_exists_flag"] is False
        assert out["templated_leak_flag"] is True

    def test_low_concentration_no_flag(self):
        out = compute_book_existence_signal(
            gap_size=100,
            top_book_votes=10,
            top_book_title="Statistical Machine Translation",
            top_keywords=["translation", "statistical", "alignment", "decoding", "bleu"],
        )
        assert out["vote_concentration"] == pytest.approx(0.1)
        assert out["book_exists_flag"] is False
        assert out["templated_leak_flag"] is False

    def test_no_top_book(self):
        out = compute_book_existence_signal(
            gap_size=20,
            top_book_votes=0,
            top_book_title=None,
            top_keywords=["a", "b", "c"],
        )
        assert out["vote_concentration"] == 0.0
        assert out["title_keyword_overlap"] == 0.0
        assert out["book_exists_flag"] is False
        assert out["templated_leak_flag"] is False
