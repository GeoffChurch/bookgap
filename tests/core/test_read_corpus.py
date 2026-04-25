"""Tests for the chunked corpus-column / corpus-dicts readers.

These replace a naive ``to_table() + take(all positions)`` implementation
that crashed with ``ArrowInvalid: offset overflow`` on full-corpus
reads once the combined string-column bytes exceeded pyarrow's 32-bit
StringArray offset limit (~2 GB). The chunked versions scan
Dataset batches and ``take`` per-batch, so each intermediate output
stays well under that limit.

Tests here verify correctness on small multi-batch datasets: matching
the naive fancy-index semantics, preserving caller-supplied position
order, and raising ``IndexError`` on out-of-range input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from bookgap.cloud import _read_corpus_column, _read_corpus_dicts


def _make_corpus(tmp_path: Path, n_per_file: int, n_files: int) -> ds.Dataset:
    """Create a multi-file dataset whose rows span several batches.

    Row k gets id ``W{k}`` and content derived from k, so the chunked
    reader's output can be checked against the identity mapping.
    """
    for f in range(n_files):
        start = f * n_per_file
        end = start + n_per_file
        table = pa.table({
            "id": [f"W{k}" for k in range(start, end)],
            "display_name": [f"title {k}" for k in range(start, end)],
            "abstract": [f"abstract {k}" for k in range(start, end)],
            "type": ["article"] * n_per_file,
            "publication_date": ["2024-01-01"] * n_per_file,
            "created_date": ["2024-01-01"] * n_per_file,
            "updated_date": ["2024-01-01"] * n_per_file,
            "language": ["en"] * n_per_file,
            "topic_label": [f"topic-{k % 3}" for k in range(start, end)],
            "primary_topic_id": [f"T{k % 3}" for k in range(start, end)],
            "primary_subfield_id": [f"SF{k % 2}" for k in range(start, end)],
            "referenced_works": [[] for _ in range(n_per_file)],
        })
        pq.write_table(table, tmp_path / f"shard_{f}.parquet")
    return ds.dataset(str(tmp_path), format="parquet")


@pytest.fixture
def small_corpus(tmp_path):
    # 3 files × 100 rows = 300 rows. chunk_size=50 forces multi-batch-per-file.
    return _make_corpus(tmp_path, n_per_file=100, n_files=3)


def test_read_column_matches_naive_fancy_index(small_corpus):
    positions = np.array([0, 50, 99, 100, 150, 250, 299], dtype=np.uint64)
    got = _read_corpus_column(small_corpus, "id", positions, chunk_size=50)
    assert got == [f"W{p}" for p in positions.tolist()]


def test_read_column_preserves_unsorted_order(small_corpus):
    positions = np.array([250, 5, 199, 42, 0], dtype=np.uint64)
    got = _read_corpus_column(small_corpus, "id", positions, chunk_size=50)
    assert got == [f"W{p}" for p in positions.tolist()]


def test_read_column_empty_positions(small_corpus):
    got = _read_corpus_column(small_corpus, "id", np.array([], dtype=np.uint64))
    assert got == []


def test_read_column_out_of_range_raises(small_corpus):
    positions = np.array([0, 300], dtype=np.uint64)  # 300 == len
    with pytest.raises(IndexError, match="out of range"):
        _read_corpus_column(small_corpus, "id", positions, chunk_size=50)


def test_read_column_every_row_covered(small_corpus):
    positions = np.arange(300, dtype=np.uint64)
    got = _read_corpus_column(small_corpus, "id", positions, chunk_size=37)
    assert got == [f"W{k}" for k in range(300)]


def test_read_dicts_matches_naive_and_preserves_order(small_corpus):
    positions = np.array([250, 5, 199, 42, 0], dtype=np.uint64)
    got = _read_corpus_dicts(small_corpus, positions, chunk_size=50)
    assert [w["id"] for w in got] == [f"W{p}" for p in positions.tolist()]
    # Spot-check a single row's full dict shape
    w = got[0]
    assert w == {
        "id": "W250",
        "display_name": "title 250",
        "abstract": "abstract 250",
        "type": "article",
        "publication_date": "2024-01-01",
        "created_date": "2024-01-01",
        "updated_date": "2024-01-01",
        "language": "en",
        "topic": "topic-1",
        "primary_topic": {"id": "T1", "subfield": {"id": "SF0"}},
        "referenced_works": [],
    }


def test_read_dicts_none_coalesces_to_defaults(tmp_path):
    """display_name/abstract/type None → "" / "" / "UNKNOWN"; None refs → []."""
    table = pa.table({
        "id": ["W0", "W1"],
        "display_name": [None, "t"],
        "abstract": [None, "a"],
        "type": [None, "book"],
        "publication_date": [None, "2024-01-01"],
        "created_date": [None, None],
        "updated_date": [None, None],
        "language": [None, "en"],
        "topic_label": [None, "tl"],
        "primary_topic_id": [None, "T1"],
        "primary_subfield_id": [None, "SF0"],
        "referenced_works": [None, ["W2"]],
    })
    pq.write_table(table, tmp_path / "x.parquet")
    corpus = ds.dataset(str(tmp_path), format="parquet")
    got = _read_corpus_dicts(corpus, np.array([0, 1], dtype=np.uint64))
    assert got[0]["display_name"] == ""
    assert got[0]["abstract"] == ""
    assert got[0]["type"] == "UNKNOWN"
    assert got[0]["referenced_works"] == []
    assert got[1]["type"] == "book"
    assert got[1]["referenced_works"] == ["W2"]


def test_read_column_many_duplicates(small_corpus):
    """Caller may request the same position repeatedly; output preserves
    that repetition."""
    positions = np.array([5, 5, 5, 100, 100], dtype=np.uint64)
    got = _read_corpus_column(small_corpus, "id", positions, chunk_size=50)
    assert got == ["W5", "W5", "W5", "W100", "W100"]
