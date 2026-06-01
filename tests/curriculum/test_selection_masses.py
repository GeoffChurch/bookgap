"""Tests for the top-k selection-score computation in build_curriculum.

Covers the three selection modes: raw PR mass (default), global
background-normalization, and the field-local *boundary background*.
"""

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.curriculum.build_curriculum import compute_selection_masses
from artifacts_scripts.curriculum.config import CurriculumConfig, RankMethod
from artifacts_scripts.curriculum.pagerank import compute_pagerank


def _toy_graph():
    """Foundation (2) vs hub (3): equal forward p, but hub has off-topic citers."""
    n = 6
    edges = [(0, 2), (1, 2), (0, 3), (1, 3), (4, 3), (5, 3)]
    r, c = zip(*edges)
    adj = sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))
    seeds = np.zeros(n)
    seeds[[0, 1]] = 0.5
    p = np.asarray(compute_pagerank(adj, seeds, alpha=0.85, normalize=True)).reshape(-1)
    return adj, p


def test_default_returns_raw_masses():
    adj, p = _toy_graph()
    cfg = CurriculumConfig()
    cfg.method = RankMethod.PAGE_RANK
    out = compute_selection_masses(cfg, adj, p)
    np.testing.assert_allclose(out, p)


def test_boundary_selection_demotes_hub():
    adj, p = _toy_graph()
    cfg = CurriculumConfig()
    cfg.method = RankMethod.PAGE_RANK
    cfg.boundary_selection = True
    out = compute_selection_masses(cfg, adj, p)
    # Equal forward mass on foundation (2) and hub (3), but boundary background
    # inflates g_local on the hub -> foundation must outrank the hub.
    assert np.isclose(p[2], p[3])
    assert out[2] > out[3]


def test_boundary_selection_ignored_for_non_pagerank_method():
    adj, p = _toy_graph()
    cfg = CurriculumConfig()
    cfg.method = RankMethod.CIRCUIT_RANK
    cfg.boundary_selection = True
    out = compute_selection_masses(cfg, adj, p)
    np.testing.assert_allclose(out, p)
