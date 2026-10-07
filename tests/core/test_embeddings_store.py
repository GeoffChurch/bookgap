"""Tests for the embeddings-store read/write helpers.

The store is an uncompressed float32 ``.npy`` array read through a memory map.
These tests check that reads return exactly what was written: every row, any
subset in the order requested, and the empty cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from bookgap.cloud import (
    read_embeddings_rows,
    read_embeddings_store,
    write_embeddings_store,
)


@pytest.fixture
def small_store(tmp_path):
    d = 8
    n = 200
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    path = tmp_path / "embeddings.npy"
    write_embeddings_store(path, emb)
    return path, emb


def test_read_all_rows_matches_written(small_store):
    path, emb = small_store
    rows = read_embeddings_rows(path)
    assert rows.dtype == np.float32
    np.testing.assert_array_equal(rows, emb)
    np.testing.assert_array_equal(read_embeddings_store(path), emb)


def test_read_store_is_a_read_only_memory_map(small_store):
    path, _ = small_store
    store = read_embeddings_store(path)
    assert isinstance(store, np.memmap)
    with pytest.raises(ValueError):
        store[0, 0] = 1.0


def test_read_subset_matches_fancy_index(small_store):
    path, emb = small_store
    retained = np.array([0, 5, 17, 42, 99, 150, 199], dtype=np.uint64)
    got = read_embeddings_rows(path, retained)
    assert got.shape == (len(retained), emb.shape[1])
    np.testing.assert_array_equal(got, emb[retained])


def test_read_subset_keeps_the_order_given(small_store):
    path, emb = small_store
    unsorted = np.array([5, 2, 99, 42, 1], dtype=np.uint64)
    np.testing.assert_array_equal(read_embeddings_rows(path, unsorted), emb[unsorted])


def test_read_subset_empty_returns_0xD(small_store):
    path, _ = small_store
    got = read_embeddings_rows(path, np.array([], dtype=np.uint64))
    assert got.shape == (0, 8)


def test_single_position(small_store):
    path, emb = small_store
    for idx in (0, 50, 199):
        got = read_embeddings_rows(path, np.array([idx], dtype=np.uint64))
        np.testing.assert_array_equal(got[0], emb[idx])


def test_subset_is_a_copy_not_a_view(small_store):
    path, emb = small_store
    got = read_embeddings_rows(path, np.array([3, 4], dtype=np.uint64))
    got[:] = 0.0
    np.testing.assert_array_equal(read_embeddings_rows(path, np.array([3, 4], dtype=np.uint64)), emb[[3, 4]])


def test_empty_store_roundtrip(tmp_path):
    path = tmp_path / "embeddings.npy"
    write_embeddings_store(path, np.empty((0, 8), dtype=np.float32))
    assert read_embeddings_rows(path).shape == (0, 8)
    assert read_embeddings_rows(path, np.array([], dtype=np.uint64)).shape == (0, 8)


def test_write_is_atomic_and_leaves_no_temp_file(small_store):
    path, _ = small_store
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
