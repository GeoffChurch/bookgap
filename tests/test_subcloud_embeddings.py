"""Byte-identity tests for the row-selective ``SubCloud.embeddings()``.

``SubCloud.embeddings()`` used to materialise the *entire* parent cloud's
embedding array (58+ GiB at domain scope) just to fancy-index a tiny subset
out of it, OOM-killing on small subclouds. It now reads only its own rows
from the embeddings parquet via :func:`read_embeddings_rows`.

These tests assert the new result is byte-identical (values, dtype, shape,
AND row order) to the old ``parent.embeddings()[get_subcloud_retained_idxs]``
path. They drive the REAL ``SubCloud.embeddings()`` code (composition of
parent parquet positions with the subcloud's retained indices, the ascending
requirement, and the row-wise denoise) against a synthetic parquet; only the
three module-level collaborators it looks up
(``cloud_embeddings_parquet_positions``, ``get_subcloud_retained_idxs``,
``denoise``) are stubbed, so the new logic itself runs unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import bookgap.subcloud as subcloud_mod
from bookgap.cloud import (
    read_embeddings_rows,
    write_embeddings_store,
)
from bookgap.cloud_denoiser import ProjectiveDenoiserProjection
from bookgap.subcloud import SubCloud


@dataclass(frozen=True)
class _FakeParent:
    """Stand-in for a Cloud parent.

    Only needs to be hashable (frozen dataclass) so it can flow through the
    real ``SubCloud`` frozen dataclass and be used as an identity. All of its
    behaviour reaches ``SubCloud.embeddings()`` through the stubbed module
    functions, never through methods on this object.
    """

    tag: str
    normalize: bool


def _make_parquet(tmp_path, n_total, d, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_total, d)).astype(np.float32)
    path = tmp_path / "embeddings.parquet"
    write_embeddings_store(path, emb)
    return path, emb


def _make_projection(d, normalize, seed=1):
    rng = np.random.default_rng(seed)
    # A genuine projective denoiser payload: per-row mean-centre + matmul.
    fit_mean = rng.standard_normal(d).astype(np.float32)
    projection_matrix = rng.standard_normal((d, d - 2)).astype(np.float32)
    proj = ProjectiveDenoiserProjection(
        projection_matrix=projection_matrix, fit_mean=fit_mean
    )

    def denoise(parent, embeddings):
        out = proj.apply(embeddings)
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / norms
        return out

    return denoise


def _install(monkeypatch, emb_path, parent_positions, sub_idxs, denoise_fn):
    """Wire the three module-level collaborators SubCloud.embeddings() uses."""
    monkeypatch.setattr(
        subcloud_mod,
        "cloud_embeddings_parquet_positions",
        lambda parent: (emb_path, np.asarray(parent_positions, dtype=np.uint64)),
    )
    monkeypatch.setattr(
        subcloud_mod,
        "get_subcloud_retained_idxs",
        lambda sc: np.asarray(sub_idxs, dtype=np.uint64),
    )
    monkeypatch.setattr(subcloud_mod, "denoise", denoise_fn)


def _old_reference(emb_path, parent_positions, sub_idxs, denoise_fn, parent):
    """The previous full-parent path: parent.embeddings()[sub_idxs]."""
    parent_positions = np.asarray(parent_positions, dtype=np.uint64)
    sub_idxs = np.asarray(sub_idxs, dtype=np.uint64)
    parent_raw = read_embeddings_rows(emb_path, parent_positions)
    parent_embeddings = denoise_fn(parent, parent_raw)
    return parent_embeddings[sub_idxs]


@pytest.mark.parametrize("normalize", [False, True])
def test_subcloud_embeddings_byte_identical_ascending(tmp_path, monkeypatch, normalize):
    d = 8
    n_total = 200
    emb_path, _ = _make_parquet(tmp_path, n_total, d, seed=0)
    # Parent retains a strict ascending subset of parquet rows (post-dedup /
    # post-filter shape). The subcloud then keeps an ascending subset of THOSE
    # rows -> composed positions stay ascending (the common no-permute case).
    parent_positions = np.array(
        [0, 3, 7, 10, 11, 19, 42, 80, 99, 150, 199], dtype=np.uint64
    )
    sub_idxs = np.array([0, 2, 3, 6, 10], dtype=np.uint64)  # ascending
    denoise_fn = _make_projection(d, normalize)
    parent = _FakeParent(tag="p", normalize=normalize)

    old = _old_reference(emb_path, parent_positions, sub_idxs, denoise_fn, parent)
    _install(monkeypatch, emb_path, parent_positions, sub_idxs, denoise_fn)
    sc = SubCloud(parent=parent, post_filter=object())  # post_filter unused (stubbed)
    new = sc.embeddings()

    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert np.array_equal(new, old)


def test_subcloud_embeddings_handles_nonascending_subset(tmp_path, monkeypatch):
    """Defensive path: if retained idxs are NOT ascending, row order is preserved.

    Real ``get_subcloud_retained_idxs`` returns ascending idxs, but the new
    code sorts + inverse-permutes to be safe. This test forces a scrambled
    ``sub_idxs`` and asserts the output still matches the naive fancy-index in
    the exact requested order.
    """
    d = 6
    n_total = 120
    emb_path, _ = _make_parquet(tmp_path, n_total, d, seed=5)
    parent_positions = np.array(
        [2, 5, 9, 13, 27, 50, 64, 88, 100, 119], dtype=np.uint64
    )
    # Scrambled order into the parent's retained set -> composed parquet
    # positions are non-ascending, exercising the sort + inverse branch.
    sub_idxs = np.array([7, 1, 9, 3, 0, 5], dtype=np.uint64)
    denoise_fn = _make_projection(d, normalize=True, seed=3)
    parent = _FakeParent(tag="q", normalize=True)

    old = _old_reference(emb_path, parent_positions, sub_idxs, denoise_fn, parent)
    _install(monkeypatch, emb_path, parent_positions, sub_idxs, denoise_fn)
    sc = SubCloud(parent=parent, post_filter=object())
    new = sc.embeddings()

    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert np.array_equal(new, old)


def test_subcloud_embeddings_multi_row_group(tmp_path, monkeypatch):
    """Across parquet row-group boundaries the streamed read still matches."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = 4
    n_total = 10_000
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((n_total, d)).astype(np.float32)
    path = tmp_path / "multi.parquet"
    flat = pa.array(emb.reshape(-1), type=pa.float32())
    emb_col = pa.FixedSizeListArray.from_arrays(flat, list_size=d)
    table = pa.table({"embeddings": emb_col})
    table = table.replace_schema_metadata({b"bookgap:embedding_dim": str(d).encode()})
    pq.write_table(table, path, compression="zstd", row_group_size=1000)

    parent_positions = np.array(
        [0, 999, 1000, 1001, 4999, 5000, 5001, 9998, 9999], dtype=np.uint64
    )
    sub_idxs = np.array([0, 1, 4, 6, 8], dtype=np.uint64)
    denoise_fn = _make_projection(d, normalize=False, seed=9)
    parent = _FakeParent(tag="m", normalize=False)

    old = _old_reference(path, parent_positions, sub_idxs, denoise_fn, parent)
    _install(monkeypatch, path, parent_positions, sub_idxs, denoise_fn)
    sc = SubCloud(parent=parent, post_filter=object())
    new = sc.embeddings()

    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert np.array_equal(new, old)


def test_subcloud_embeddings_empty_subset(tmp_path, monkeypatch):
    d = 8
    emb_path, _ = _make_parquet(tmp_path, 50, d, seed=2)
    parent_positions = np.array([1, 4, 9, 20, 49], dtype=np.uint64)
    sub_idxs = np.array([], dtype=np.uint64)
    denoise_fn = _make_projection(d, normalize=True, seed=4)
    parent = _FakeParent(tag="e", normalize=True)

    old = _old_reference(emb_path, parent_positions, sub_idxs, denoise_fn, parent)
    _install(monkeypatch, emb_path, parent_positions, sub_idxs, denoise_fn)
    sc = SubCloud(parent=parent, post_filter=object())
    new = sc.embeddings()

    assert new.shape == old.shape
    assert np.array_equal(new, old)
