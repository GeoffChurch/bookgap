"""Tests for the columnar fast path in SubCloud retention."""
from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from bookgap.subcloud import SubCloud, get_subcloud_retained_idxs
from bookgap.util import Filter, WorkType
from bookgap.filters import TemporalFilter, _extract_year


class _MockCloud:
    def __init__(self, pub_dates, types, works=None):
        self._pub_dates = pub_dates
        self._types = types
        if works is None:
            works = []
            for d, t in zip(pub_dates, types):
                w: dict[str, Any] = {"id": f"W{len(works)}"}
                if d is not None:
                    w["publication_date"] = d
                # _read_corpus_dicts coerces None -> "UNKNOWN" in cloud.py.
                w["type"] = t if t is not None else "UNKNOWN"
                works.append(w)
        self._works = works

    def work_publication_dates(self): return self._pub_dates
    def work_types(self): return self._types
    def works(self): return self._works
    def work_ids(self): return [w["id"] for w in self._works]
    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other


@pytest.fixture
def small_corpus():
    pub_dates = [
        "2010-01-01", "2010-12-31", "2011-06-15", "2011-06-15", "2009-05-01",
        "2009", None, "", "2010-03-03", "2010", "2011-01-01", "2008-07-07",
        "2012-02-02", "2012-02-02", "2010-10-10", "2010-11-11", "2007-09-09",
        "2013-01-15", "2013-01-15", "2010-05-05",
    ]
    types = [
        "book", "article", "book", "preprint", "book", "article", "article",
        "book", "article", "book", "book-chapter", "article", "book",
        "preprint", None, "article", "book", "article", "book", "article",
    ]
    return _MockCloud(pub_dates, types)


def _dict_iter_mask(cloud, f):
    return np.array([f.check_work(w) for w in cloud.works()], dtype=bool)


def test_base_filter_compute_columnar_mask_returns_none():
    f = Filter(topic="T10181", work_types=("book",))
    assert f.compute_columnar_mask(cloud=object()) is None


@pytest.mark.parametrize("date_str,expected", [
    (None, None), ("", None), ("2010-01-01", 2010),
    ("2010-12-31", 2010), ("2010", 2010), ("1999-06", 1999),
])
def test_extract_year_well_formed(date_str, expected):
    assert _extract_year(date_str) == expected


def test_extract_year_malformed_raises():
    with pytest.raises(ValueError):
        _extract_year("not-a-date")


@pytest.mark.parametrize("filter_factory,label", [
    (lambda: TemporalFilter(max_year=2010, work_type=WorkType.BOOK), "max_year+book"),
    (lambda: TemporalFilter(max_year=2010, work_type=WorkType.ARTICLE), "max_year+article"),
    (lambda: TemporalFilter(max_year=2010, work_type=WorkType.PREPRINT), "max_year+preprint"),
    (lambda: TemporalFilter(exact_year=2011, work_type=WorkType.BOOK), "exact_year+book"),
    (lambda: TemporalFilter(exact_year=2011, work_type=WorkType.ARTICLE), "exact_year+article"),
    (lambda: TemporalFilter(max_year=2012), "year_only_no_type"),
    (lambda: TemporalFilter(work_type=WorkType.BOOK), "type_only_no_year"),
    (lambda: TemporalFilter(), "empty_temporal_filter"),
])
def test_columnar_mask_matches_dict_iter(small_corpus, filter_factory, label):
    f = filter_factory()
    expected = _dict_iter_mask(small_corpus, f)
    actual = f.compute_columnar_mask(small_corpus)
    assert actual is not None, f"[{label}] expected mask, got None"
    np.testing.assert_array_equal(np.asarray(actual, dtype=bool), expected)


def test_columnar_mask_rejects_missing_and_empty_dates(small_corpus):
    f = TemporalFilter(max_year=9999)
    mask = np.asarray(f.compute_columnar_mask(small_corpus), dtype=bool)
    assert not mask[6] and not mask[7]
    assert np.delete(mask, [6, 7]).all()


def test_columnar_mask_article_preprint_interchangeable(small_corpus):
    f = TemporalFilter(work_type=WorkType.ARTICLE)
    mask = np.asarray(f.compute_columnar_mask(small_corpus), dtype=bool)
    np.testing.assert_array_equal(mask, _dict_iter_mask(small_corpus, f))
    assert mask[3] and mask[13]


def test_columnar_mask_book_chapter_not_matched_by_book(small_corpus):
    f = TemporalFilter(work_type=WorkType.BOOK)
    mask = np.asarray(f.compute_columnar_mask(small_corpus), dtype=bool)
    assert not mask[10]


def test_columnar_mask_none_type_rejected(small_corpus):
    for wt in (WorkType.BOOK, WorkType.ARTICLE, WorkType.PREPRINT):
        f = TemporalFilter(work_type=wt)
        mask = np.asarray(f.compute_columnar_mask(small_corpus), dtype=bool)
        assert not mask[14]
        np.testing.assert_array_equal(mask, _dict_iter_mask(small_corpus, f))


@pytest.mark.parametrize("kwargs", [
    {"subfield": "1702"}, {"topic": "T10181"},
    {"work_types": ("book",)}, {"domain": "3"}, {"field": "17"},
])
def test_columnar_mask_falls_back_for_parent_filter_constraints(small_corpus, kwargs):
    f = TemporalFilter(max_year=2010, work_type=WorkType.BOOK, **kwargs)
    assert f.compute_columnar_mask(small_corpus) is None


def _bare(subcloud):
    """Bypass lru_cache + joblib MEMORY.cache."""
    return get_subcloud_retained_idxs.__wrapped__.func(subcloud)


def test_get_subcloud_retained_idxs_takes_fast_path(small_corpus):
    f = TemporalFilter(max_year=2010, work_type=WorkType.BOOK)
    original = small_corpus.works
    spy = MagicMock(wraps=original)
    small_corpus.works = spy
    sc = SubCloud(parent=small_corpus, post_filter=f)
    idxs = _bare(sc)
    spy.assert_not_called()
    expected = np.where(np.array([f.check_work(w) for w in original()], dtype=bool))[0].astype(np.uint64)
    np.testing.assert_array_equal(idxs, expected)
    assert idxs.dtype == np.uint64


def test_get_subcloud_retained_idxs_falls_back_when_mask_is_none(small_corpus):
    f = Filter(work_types=("book",))
    original = small_corpus.works
    spy = MagicMock(wraps=original)
    small_corpus.works = spy
    sc = SubCloud(parent=small_corpus, post_filter=f)
    idxs = _bare(sc)
    spy.assert_called()
    expected = np.array([i for i, w in enumerate(original()) if f.check_work(w)], dtype=np.uint64)
    np.testing.assert_array_equal(idxs, expected)
    assert idxs.dtype == np.uint64
