"""Litmus test for the boundary-background curriculum selection helper.

The toy graph has a shared "foundation" node (cited only by the on-topic
seeds) and a "hub" node (cited by the on-topic seeds AND by two off-topic
works). Forward seeded PageRank gives both the same mass; the boundary
background must inflate ``g_local`` on the hub (via its off-topic citers)
so the selection score demotes the hub and keeps the foundation.
"""

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.curriculum.pagerank import compute_pagerank, boundary_background


def test_boundary_demotes_hub_keeps_foundation():
    n = 6
    edges = [(0, 2), (1, 2), (0, 3), (1, 3), (4, 3), (5, 3)]  # i->j = i cites j; foundation=2, hub=3
    r, c = zip(*edges)
    adj = sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))
    seeds = np.zeros(n)
    seeds[[0, 1]] = 0.5
    p = np.asarray(compute_pagerank(adj, seeds, alpha=0.85, normalize=True)).reshape(-1)
    g = boundary_background(adj, p, alpha=0.85, tol=1e-9, max_iter=100)
    eps = 1e-12
    score = np.power(np.maximum(p, 0.0), 0.5) * np.log((p + eps) / (g + eps))
    assert score[2] > score[3]  # foundation kept over hub  (note p[2]==p[3])
