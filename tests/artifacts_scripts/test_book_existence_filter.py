"""Tests for Phase 1 book-existence filter helpers.

Covers the pure-logic pieces (tokenization, Jaccard overlap, the
book-existence signal classification). Full-pipeline correctness is
validated empirically against T10181.
"""

import pytest

# Imports artifacts_scripts.keyword_helpers, which imports the private sibling
# `curriculum` (see pyproject.toml). Transitive, so easy to miss: this test does
# not mention curriculum anywhere.
pytest.importorskip("curriculum", reason="sibling curriculum checkout not installed")

import json

import pytest

from artifacts_scripts.keyword_helpers import (
    compute_book_existence_signal,
    jaccard_set,
    tokenize_for_overlap,
)
from artifacts_scripts.gap_book_existence_filter import _row_members, _row_size


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


class TestRowMembers:
    def test_legacy_article_indices_json(self):
        row = {"article_indices_json": json.dumps([1, 2, 3])}
        assert _row_members(row) == [1, 2, 3]

    def test_member_indices_json_filters_to_articles(self):
        row = {
            "member_indices_json": json.dumps([
                {"idx": 1, "type": "article"},
                {"idx": 7, "type": "book"},
                {"idx": 12, "type": "article"},
            ])
        }
        assert _row_members(row) == [1, 12]

    def test_member_indices_json_all_books_returns_empty(self):
        row = {
            "member_indices_json": json.dumps([
                {"idx": 5, "type": "book"},
                {"idx": 9, "type": "book"},
            ])
        }
        assert _row_members(row) == []

    def test_prefers_legacy_when_both_present(self):
        # Plan §Task 9 keeps both columns populated for backward-compat.
        # Order in _MEMBER_COLUMNS is (article_indices_json, member_indices_json),
        # so the legacy column wins.
        row = {
            "article_indices_json": json.dumps([1, 2]),
            "member_indices_json": json.dumps([
                {"idx": 1, "type": "article"},
                {"idx": 2, "type": "article"},
                {"idx": 99, "type": "book"},
            ]),
        }
        # Legacy column already excludes books, so both paths should produce [1, 2].
        assert _row_members(row) == [1, 2]

    def test_empty_string_column_raises(self):
        # An empty cell isn't a valid input; the helper should escalate
        # rather than silently returning [].
        row = {"article_indices_json": ""}
        with pytest.raises(KeyError):
            _row_members(row)

    def test_no_member_column_raises(self):
        row = {"some_other_col": "foo"}
        with pytest.raises(KeyError):
            _row_members(row)


class TestRowSize:
    def test_reads_final_size(self):
        row = {"final_size": "42"}
        assert _row_size(row, [1, 2, 3]) == 42

    def test_reads_gap_size(self):
        row = {"gap_size": "100"}
        assert _row_size(row, []) == 100

    def test_reads_total_size(self):
        row = {"total_size": "7"}
        assert _row_size(row, [10, 11, 12, 13]) == 7

    def test_falls_back_to_member_count_when_no_size_column(self):
        row = {"some_other_col": "foo"}
        assert _row_size(row, [1, 2, 3]) == 3

    def test_falls_back_when_size_column_empty(self):
        row = {"final_size": ""}
        assert _row_size(row, [1, 2, 3]) == 3
