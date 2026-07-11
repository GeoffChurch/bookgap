"""Tests for bookgap.util.

``TestBuildCitationAdjacencyFromIds`` was relocated here from
``tests/curriculum/test_pagerank.py`` during Phase-3 Task 10's coverage
triage: ``build_citation_adjacency_from_ids`` is bookgap's own utility
(used by ``bookgap/cloud.py``, ``bookgap/cloud_denoiser.py``,
``bookgap/tf_gep_embedder.py``, and several ``artifacts_scripts``), not an
engine internal -- it just happened to share a test file with PageRank
engine-internals tests. Everything else in that file either duplicated
coverage already in the curriculum engine's own test suite or was ported
there as a genuine gap; see the Task 10 report for the full triage.
"""

import pytest

from bookgap.util import build_citation_adjacency_from_ids


@pytest.fixture
def simple_work_ids():
    """Create simple work IDs."""
    return ["W0", "W1", "W2", "W3", "W4"]


@pytest.fixture
def simple_work_references():
    """Create simple references with citation structure.

    Citation graph:
    0 -> 1, 2
    1 -> 2
    2 -> (none)
    3 -> 0, 2
    4 -> 3

    So 2 is highly cited (authority), 4 cites least (depends on others).
    """
    return [
        ["W1", "W2"],  # W0 cites W1, W2
        ["W2"],        # W1 cites W2
        [],            # W2 cites nothing
        ["W0", "W2"],  # W3 cites W0, W2
        ["W3"],        # W4 cites W3
    ]


class TestBuildCitationAdjacencyFromIds:
    """Tests for build_citation_adjacency_from_ids."""

    def test_basic_structure(self, simple_work_ids, simple_work_references):
        """Test basic adjacency matrix structure."""
        adj, id_to_idx = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )

        # Shape should be n x n
        assert adj.shape == (5, 5)

        # Check id_to_idx mapping
        assert id_to_idx["W0"] == 0
        assert id_to_idx["W4"] == 4

    def test_self_loops(self, simple_work_ids, simple_work_references):
        """Test self-loops are added when requested."""
        adj_no_loops, _ = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )
        adj_with_loops, _ = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=True
        )

        # With self-loops should have more non-zeros
        assert adj_with_loops.nnz > adj_no_loops.nnz

        # Diagonal should be non-zero with self-loops
        diag = adj_with_loops.diagonal()
        assert (diag > 0).all()

    def test_citation_edges(self, simple_work_ids, simple_work_references):
        """Test citation edges are correct."""
        adj, id_to_idx = build_citation_adjacency_from_ids(
            simple_work_ids, simple_work_references, add_self_loops=False
        )

        # W0 cites W1 and W2
        assert adj[0, 1] == 1.0
        assert adj[0, 2] == 1.0

        # W2 cites nothing (except possibly self)
        assert adj[2, 0] == 0.0
        assert adj[2, 1] == 0.0
        assert adj[2, 3] == 0.0
        assert adj[2, 4] == 0.0

    def test_external_references_ignored(self):
        """Test that references to external works are ignored."""
        work_ids = ["W0", "W1"]
        work_references = [["W1", "external_id"], []]

        adj, _ = build_citation_adjacency_from_ids(
            work_ids, work_references, add_self_loops=False
        )

        # Only W0->W1 edge, not W0->external
        assert adj.nnz == 1
        assert adj[0, 1] == 1.0
