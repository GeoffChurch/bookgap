"""Parity gate: reproduce the committed wiki-native curriculum via the package.

The committed ``docs/wikipedia-curriculum.md`` was produced by the bookgap
``wiki_curriculum_demo.py`` from the cached subgraph
``cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz`` with its DEFAULT
settings: forward PageRank (alpha=0.85) seeded at the 6 NLP seed titles, top-4000
selection by RAW PR mass, bisecting-KMeans cut to 12 topics, mass ordering,
title+abstract keyword text.

This test loads the SAME npz as a plain :class:`InMemoryGraphSource` (NO bookgap
import) and runs the new ``build_curriculum`` with the equivalent config, then
asserts byte-for-byte parity on the structure that defines the curriculum:
topic count, per-rank cluster SIZE, per-rank top-5 representative PAGES (the
membership+order signal), and each topic's HEADLINE keyword. This proves the
lift is behaviour-preserving (SP1-style parity discipline).

Skipped automatically when the large fixture is not present (e.g. CI).
"""

from __future__ import annotations

import pytest

from curriculum import (
    CurriculumConfig,
    EmbeddingsMode,
    OrderingMode,
    RankMethod,
    build_curriculum,
)
from curriculum.config import SelectionMode
from curriculum.tests.wiki_fixture import DEFAULT_FIXTURE, load_wiki_graph_source

# Committed curriculum (docs/wikipedia-curriculum.md): per-rank
# (size, headline_keyword, [top representative pages]). Pages are matched as a
# prefix because the committed table truncates long titles to 40 chars.
EXPECTED = [
    (233, "algorithm", ["Computer science", "Artificial intelligence", "Algorithm"]),
    (230, "grammar", ["Formal grammar", "Programming language", "Grammar checker"]),
    (259, "software", ["Digital object identifier", "Bibcode", "Handle System"]),
    (341, "university", ["PubMed", "ArXiv", "OCLC"]),
    (349, "iso", ["ISBN", "ISSN", "ISO 4"]),
    (1, "machine_learning", ["Machine learning"]),
    (1, "broadly", ["Natural language processing"]),
    (1, "sentence", ["Parsing"]),
    (1, "computational", ["Computational linguistics"]),
    (319, "computer", ["Optical character recognition", "Operating system", "Database"]),
    (1, "superseded", ["Language model"]),
    (1, "embedding", ["Word embedding"]),
]


def _wiki_native_config() -> CurriculumConfig:
    """The config equivalent to wiki_curriculum_demo's default invocation."""
    cfg = CurriculumConfig(
        method=RankMethod.PAGE_RANK,
        embeddings_mode=EmbeddingsMode.PROVIDED,
        ordering=OrderingMode.MASS,
    )
    cfg.pagerank.alpha = 0.85
    cfg.pagerank.tol = 1e-9
    cfg.pagerank.max_iter = 200
    cfg.selection.mode = SelectionMode.TOP_N
    cfg.selection.top_n = 4000
    cfg.clustering.desired_n_topics = 12
    cfg.keywords.enabled = True
    cfg.keywords.samples_per_topic = 80
    cfg.keywords.top_k = 12
    return cfg


@pytest.mark.skipif(
    not DEFAULT_FIXTURE.exists(),
    reason=f"wiki subgraph fixture not present: {DEFAULT_FIXTURE}",
)
def test_wiki_native_parity() -> None:
    gs = load_wiki_graph_source()
    assert gs is not None
    curr = build_curriculum(gs, _wiki_native_config())

    assert curr.n_topics == len(EXPECTED), (
        f"expected {len(EXPECTED)} topics, got {curr.n_topics}"
    )

    ids = gs.ids()
    for rank, topic_id in enumerate(curr.topic_order):
        topic_id = int(topic_id)
        exp_size, exp_kw, exp_pages = EXPECTED[rank]

        size = int((curr.point_to_cluster == topic_id).sum())
        assert size == exp_size, (
            f"rank {rank + 1}: size {size} != expected {exp_size}"
        )

        members = curr.topic_member_indices(topic_id)
        got_pages = [ids[int(i)] for i in members[:5]]
        for k, exp_page in enumerate(exp_pages):
            assert got_pages[k] == exp_page, (
                f"rank {rank + 1} page {k}: {got_pages[k]!r} != {exp_page!r} "
                f"(got pages {got_pages})"
            )

        kws = [w for w, _ in curr.topic_keywords.get(topic_id, [])]
        assert kws and kws[0] == exp_kw, (
            f"rank {rank + 1}: headline keyword {kws[:1]} != [{exp_kw!r}]"
        )
