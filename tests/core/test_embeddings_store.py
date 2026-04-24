"""Tests for the embeddings-parquet read/write helpers.

:func:`read_embeddings_rows` exists to replace the old
"load-everything-then-fancy-index" pattern that simultaneously held a
64 GB source array and a 62 GB filtered copy during post-dedup, which
OOM-killed the curriculum pipeline on a 250 GB box. These tests verify
its correctness (matches the naive pattern's output) on small data.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bookgap.cloud import (
    read_embeddings_rows,
    read_embeddings_store,
    write_embeddings_store,
)


@pytest.fixture
def small_parquet(tmp_path):
    d = 8
    n = 200
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    path = tmp_path / "e.parquet"
    write_embeddings_store(path, emb)
    return path, emb


def test_read_all_rows_matches_read_embeddings_store(small_parquet):
    path, emb = small_parquet
    rows = read_embeddings_rows(path)
    assert rows.shape == emb.shape
    np.testing.assert_array_equal(rows, emb)
    np.testing.assert_array_equal(rows, read_embeddings_store(path))


def test_read_subset_matches_fancy_index(small_parquet):
    path, emb = small_parquet
    retained = np.array([0, 5, 17, 42, 99, 150, 199], dtype=np.uint64)
    got = read_embeddings_rows(path, retained)
    assert got.shape == (len(retained), emb.shape[1])
    np.testing.assert_array_equal(got, emb[retained])


def test_read_subset_empty_returns_0xD(small_parquet):
    path, _ = small_parquet
    got = read_embeddings_rows(path, np.array([], dtype=np.uint64))
    assert got.shape == (0, 8)


def test_requires_ascending_order(small_parquet):
    path, _ = small_parquet
    unsorted = np.array([5, 2, 99, 42, 1], dtype=np.uint64)
    with pytest.raises(ValueError, match="ascending"):
        read_embeddings_rows(path, unsorted)


def test_single_position(small_parquet):
    path, emb = small_parquet
    for idx in (0, 50, 199):
        got = read_embeddings_rows(path, np.array([idx], dtype=np.uint64))
        np.testing.assert_array_equal(got[0], emb[idx])


def test_multi_row_group_streaming(tmp_path):
    """Multi-row-group parquet. Exercises the streaming boundary logic."""
    d = 4
    n = 10_000
    emb = np.arange(n * d, dtype=np.float32).reshape(n, d)
    path = tmp_path / "multi.parquet"
    flat = pa.array(emb.reshape(-1), type=pa.float32())
    emb_col = pa.FixedSizeListArray.from_arrays(flat, list_size=d)
    table = pa.table({
        "row_id": pa.array(range(n), type=pa.int64()),
        "embeddings": emb_col,
    })
    table = table.replace_schema_metadata({b"bookgap:embedding_dim": str(d).encode()})
    pq.write_table(table, path, compression="zstd", row_group_size=1000)
    pf = pq.ParquetFile(path)
    assert pf.num_row_groups == 10

    retained = np.array([0, 999, 1000, 1001, 4999, 5000, 9999], dtype=np.uint64)
    got = read_embeddings_rows(path, retained)
    np.testing.assert_array_equal(got, emb[retained])


def test_roundtrip_full_read_then_filter_equivalent(small_parquet):
    """Regression test for the use case that OOM-killed production."""
    path, emb = small_parquet
    retained = np.sort(
        np.random.default_rng(1).choice(200, size=73, replace=False).astype(np.uint64)
    )
    streaming_result = read_embeddings_rows(path, retained)
    naive_result = read_embeddings_store(path)[retained]
    np.testing.assert_array_equal(streaming_result, naive_result)
