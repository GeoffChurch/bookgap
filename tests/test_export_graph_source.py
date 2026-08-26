"""Unit tests for artifacts_scripts/export_graph_source.py.

Exercises ``assemble_arrays`` against a stubbed Cloud (duck-typed: only the
five accessor methods it calls, no cache/corpus access) and confirms the
result round-trips through curriculum's ``save_graph_npz`` / ``load_graph_npz``.
No real Cloud/cache access here — that's T7's capture-script integration.
"""
from __future__ import annotations


import pytest

# The npz writer is a CONTRACT shared with the engine, and it now lives in bookgen's
# `artifacts` package (curriculum/ dissolved into engine/ + arena/). Not a declared
# dependency here, so these tests skip rather than error on a clone without it.
pytest.importorskip("artifacts", reason="bookgen `artifacts` package not importable")

import numpy as np
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf

from artifacts_scripts.export_graph_source import assemble_arrays, build_meta
from artifacts.sources.graph_npz import load_graph_npz, save_graph_npz


class _StubCloud:
    """Duck-typed stand-in for bookgap.cloud.Cloud's array-accessor surface."""

    def __init__(self, *, ids, titles, abstracts, emb, adj, name="stub-cloud", embedder=None):
        self._ids = ids
        self._titles = titles
        self._abstracts = abstracts
        self._emb = emb
        self._adj = adj
        self._name = name
        self.embedder = embedder if embedder is not None else _StubEmbedder()

    def work_ids(self):
        return list(self._ids)

    def work_titles(self):
        return list(self._titles)

    def work_abstracts(self):
        return list(self._abstracts)

    def embeddings(self):
        return self._emb

    def citation_graph(self, normalize):
        assert normalize is False
        return self._adj

    def name(self):
        return self._name


class _StubEmbedder:
    model_name = "stub-embedder-v0"


def _make_stub(n=4, dim=3):
    ids = [f"W{i}" for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    abstracts = [f"Abstract {i}." for i in range(n)]
    emb = np.arange(n * dim, dtype=np.float64).reshape(n, dim)  # non-float32 on purpose
    # A tiny citation ring: i cites i+1 (mod n).
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    adj = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(n, n))
    return _StubCloud(ids=ids, titles=titles, abstracts=abstracts, emb=emb, adj=adj)


def test_assemble_arrays_shapes_and_types():
    cloud = _make_stub(n=4, dim=3)
    ids, adj, titles, abstracts, emb = assemble_arrays(cloud)

    assert ids == ["W0", "W1", "W2", "W3"]
    assert titles == ["Title 0", "Title 1", "Title 2", "Title 3"]
    assert abstracts == ["Abstract 0.", "Abstract 1.", "Abstract 2.", "Abstract 3."]
    assert isinstance(adj, sp.csr_matrix)
    assert adj.shape == (4, 4)
    assert adj[0, 1] == 1.0
    assert emb.shape == (4, 3)
    assert emb.dtype == np.float32


def test_assemble_arrays_rejects_mismatched_lengths():
    cloud = _make_stub(n=4, dim=3)
    cloud._titles = cloud._titles[:-1]  # drop one title -> length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        assemble_arrays(cloud)


def test_assemble_arrays_save_load_roundtrip(tmp_path):
    cloud = _make_stub(n=5, dim=4)
    ids, adj, titles, abstracts, emb = assemble_arrays(cloud)

    out_path = tmp_path / "graph.npz"
    save_graph_npz(out_path, ids=ids, adj=adj, titles=titles, abstracts=abstracts, emb=emb)

    loaded = load_graph_npz(out_path, seed_ids=[ids[0], ids[2]])

    assert loaded.ids() == ids
    assert loaded.node_text() == [f"{t}. {a}" for t, a in zip(titles, abstracts)]
    np.testing.assert_array_equal(loaded.node_embeddings(), emb)
    assert loaded.node_embeddings().dtype == np.float32
    np.testing.assert_array_equal(
        loaded.adjacency().toarray(), adj.toarray()
    )
    np.testing.assert_array_equal(loaded.seed_nodes(), np.array([0, 2], dtype=np.int64))


def test_assemble_arrays_coerces_none_titles_and_abstracts_to_empty_string():
    """A corpus row can carry a genuinely missing title/abstract as None
    (not ""). assemble_arrays must coerce it to "" -- otherwise
    save_graph_npz's object-array dtype accepts None as an element, and
    downstream str()-ing on load renders it as the literal text "None"."""
    cloud = _make_stub(n=4, dim=3)
    cloud._titles[1] = None
    cloud._abstracts[2] = None

    ids, adj, titles, abstracts, emb = assemble_arrays(cloud)

    assert titles == ["Title 0", "", "Title 2", "Title 3"]
    assert abstracts == ["Abstract 0.", "Abstract 1.", "", "Abstract 3."]
    assert all(t is not None for t in titles)
    assert all(a is not None for a in abstracts)


def test_assemble_arrays_no_transpose_citation_convention():
    """adj[i, j] from citation_graph already means 'i cites/points to j' —
    assemble_arrays must pass it through unchanged (no transpose)."""
    cloud = _make_stub(n=3, dim=2)
    _, adj, _, _, _ = assemble_arrays(cloud)
    # Stub built i -> (i+1) % n; confirm that exact edge set survived.
    got_edges = set(zip(*adj.nonzero()))
    assert got_edges == {(0, 1), (1, 2), (2, 0)}


def test_build_meta_records_identity_and_passthrough(monkeypatch):
    cloud = _make_stub(n=3, dim=2)
    ids, adj, _, _, _ = assemble_arrays(cloud)

    # build_meta calls bookgap.cloud.ensure_cloud_data(cloud) for the corpus
    # fingerprint; stub it out since our _StubCloud isn't a real Cloud.
    import artifacts_scripts.export_graph_source as mod

    monkeypatch.setattr(
        mod, "ensure_cloud_data", lambda cloud: ("fake-fingerprint", None, None)
    )

    cfg = OmegaConf.create({"cloud": {"denoiser": "NONE"}})
    meta = build_meta(
        cloud,
        cfg,
        ids=ids,
        adj=adj,
        target_work_ids=["W0", "W99"],
        config_name="standard",
        overrides=["cloud.denoiser=NONE"],
    )

    assert meta["cloud_name"] == "stub-cloud"
    assert meta["cloud_fingerprint"] == "fake-fingerprint"
    assert meta["config_name"] == "standard"
    assert meta["config_overrides"] == ["cloud.denoiser=NONE"]
    assert meta["n_nodes"] == 3
    assert meta["n_edges"] == int(adj.nnz)
    assert meta["embedder_name"] == "stub-embedder-v0"
    # Passthrough, not resolved: W99 isn't in `ids` and that's fine here.
    assert meta["target_work_ids"] == ["W0", "W99"]
