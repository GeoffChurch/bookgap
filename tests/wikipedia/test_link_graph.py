# tests/wikipedia/test_link_graph.py
"""Tests for the cirrus-native Wikipedia link graph (bookgap/wikipedia/link_graph).

These read a small prefix of the (possibly partial) CirrusSearch content dump
and assert the article->article graph has plausible structure and that a known
hub resolves to expected targets. Skipped when the dump is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from bookgap.util import CACHE_DIR
from bookgap.wikipedia.link_graph import (
    _is_non_article_link,
    _norm_title,
    build_link_graph,
    build_title_maps,
    load_or_build_title_maps,
)

DUMP = CACHE_DIR / "wiki" / "dumps" / "enwiki-20251229-cirrussearch-content.json.gz"
N_SAMPLE = 30_000  # a few thousand+ articles; enough for hubs to resolve

pytestmark = pytest.mark.skipif(not DUMP.exists(), reason="cirrus dump not present")


def test_norm_title_and_prefix_filter():
    # underscores -> spaces; whitespace stripped
    assert _norm_title("Natural_language_processing") == "Natural language processing"
    assert _norm_title("  A_B  ") == "A B"
    # namespace prefixes are dropped; legitimate colons in titles are kept
    assert _is_non_article_link("Template:Foo")
    assert _is_non_article_link("Category:Bar")
    assert _is_non_article_link("Wikipedia:Protection policy")
    assert not _is_non_article_link("Spider-Man: No Way Home")
    assert not _is_non_article_link("2001: A Space Odyssey")
    assert not _is_non_article_link("Natural language processing")


@pytest.fixture(scope="module")
def maps():
    return build_title_maps(DUMP, limit=N_SAMPLE)


def test_title_maps_have_expected_entries(maps):
    title_to_pid, redirect_to_pid = maps
    assert len(title_to_pid) > 1000
    # All keys space-normalized (no underscores), values positive ints.
    assert all("_" not in t for t in list(title_to_pid)[:5000])
    assert all(isinstance(p, int) and p > 0 for p in list(title_to_pid.values())[:5000])
    # A famous early-alphabet article that is certainly in the first 30k.
    assert "Anarchism" in title_to_pid
    # Canonical titles win over same-string redirects.
    for t in list(title_to_pid)[:2000]:
        if t in redirect_to_pid:
            assert redirect_to_pid[t] == title_to_pid[t]


def test_link_graph_structure(maps):
    title_to_pid, redirect_to_pid = maps
    id_list = list(title_to_pid.values())
    lg = build_link_graph(
        DUMP,
        id_list,
        title_to_pid=title_to_pid,
        redirect_to_pid=redirect_to_pid,
        limit=N_SAMPLE,
    )
    adj = lg.adj
    n = adj.shape[0]
    assert n == len(id_list)
    assert adj.dtype == np.float32
    # Binary adjacency, no self-loops.
    assert set(np.unique(adj.data).tolist()) <= {1.0}
    assert adj.diagonal().sum() == 0
    # Plausible connectivity: a 30k-article in-set slice links densely enough
    # that mean out-degree is comfortably > 1 and most rows have an edge.
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    assert out_deg.mean() > 2.0
    assert (out_deg > 0).mean() > 0.5
    assert lg.dump_date == "20251229"


def test_title_maps_cache_roundtrip(maps, tmp_path):
    """build -> persist -> load reconstructs identical (title, redirect) maps.

    Exercises the persist/load halves of ``load_or_build_title_maps`` without a
    full-dump build: we seed the cache from the bounded ``maps`` fixture by
    writing it through the public loader (cache-miss path is the full dump, so
    we instead write the parquet directly via a known-good build), then assert a
    second load returns dicts equal to the originals.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    title_to_pid, redirect_to_pid = maps
    cache_path = tmp_path / "title_maps.parquet"

    # Persist exactly as load_or_build_title_maps would on a cache miss.
    keys = list(title_to_pid.keys()) + list(redirect_to_pid.keys())
    pids = list(title_to_pid.values()) + list(redirect_to_pid.values())
    kinds = ["title"] * len(title_to_pid) + ["redirect"] * len(redirect_to_pid)
    pq.write_table(
        pa.table(
            {
                "key": pa.array(keys, type=pa.string()),
                "page_id": pa.array(pids, type=pa.int64()),
                "kind": pa.array(kinds, type=pa.string()),
            }
        ),
        cache_path,
    )

    # Cache hit: must NOT touch the dump and must reconstruct the maps exactly.
    t2, r2 = load_or_build_title_maps(DUMP, cache_path)
    assert t2 == title_to_pid
    assert r2 == redirect_to_pid
    # page_ids round-trip as plain python ints (not numpy), keys as str.
    assert all(isinstance(p, int) for p in list(t2.values())[:100])
    assert all(isinstance(k, str) for k in list(t2)[:100])


def test_known_article_links_to_expected_targets(maps):
    """A known hub links to expected targets after redirect resolution."""
    title_to_pid, redirect_to_pid = maps
    # 'Anarchism' is article 12 (first content doc); it links to many concrete
    # anarchism subtopics. Build a focused graph over a set that includes it and
    # a handful of its expected targets, all present in the 30k prefix.
    anchor = "Anarchism"
    expected = [
        "Anarcho-syndicalism",
        "Individualist anarchism",
        "Anarchist communism",
    ]
    present_expected = [t for t in expected if t in title_to_pid]
    assert present_expected, "expected anarchism targets not in sample prefix"

    keep_titles = [anchor] + present_expected
    id_list = [title_to_pid[t] for t in keep_titles]
    lg = build_link_graph(
        DUMP,
        id_list,
        title_to_pid=title_to_pid,
        redirect_to_pid=redirect_to_pid,
        limit=N_SAMPLE,
    )
    row = {pid: r for r, pid in enumerate(lg.id_list)}
    src = row[title_to_pid[anchor]]
    out_neighbors = set(lg.adj.indices[lg.adj.indptr[src]:lg.adj.indptr[src + 1]].tolist())
    for t in present_expected:
        tgt = row[title_to_pid[t]]
        assert tgt in out_neighbors, f"Anarchism should link to {t}"
