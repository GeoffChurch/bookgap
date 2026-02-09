import numpy as np
import scipy.sparse as sp
import pytest
from artifacts_scripts.curriculum.pagerank import compute_bidirectional_pagerank

def test_bidirectional_pagerank_basic():
    # Simple line graph: 0 -> 1 -> 2 -> 3 -> 4
    # 0 is most recent (citer), 4 is most foundational (cited)
    n = 5
    data = np.ones(4)
    row = np.array([0, 1, 2, 3])
    col = np.array([1, 2, 3, 4])
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    
    # Target is recent (0), Source is foundational (4)
    target_seeds = np.zeros(n)
    target_seeds[0] = 1.0
    source_seeds = np.zeros(n)
    source_seeds[4] = 1.0
    
    relevance = compute_bidirectional_pagerank(adj, source_seeds, target_seeds, alpha=0.9)
    
    # All nodes should have some relevance as they are on the only path
    assert relevance.shape == (n,)
    assert np.all(relevance > 0)
    assert pytest.approx(relevance.sum()) == 1.0

def test_bidirectional_pagerank_branching():
    # Graph: 0 -> 1 -> 2
    #        0 -> 3 -> 2
    #        0 -> 4 -> 5 (dead end for 0->2 path)
    # 0 is recent, 2 is foundational
    n = 6
    row = np.array([0, 1, 0, 3, 0, 4])
    col = np.array([1, 2, 3, 2, 4, 5])
    data = np.ones(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    
    target_seeds = np.zeros(n)
    target_seeds[0] = 1.0
    source_seeds = np.zeros(n)
    source_seeds[2] = 1.0
    
    relevance = compute_bidirectional_pagerank(adj, source_seeds, target_seeds, alpha=0.9)
    
    # Path is 0-1-2 and 0-3-2. Node 4 and 5 are not on any path from 0 to 2.
    path_nodes = [0, 1, 2, 3]
    off_path_nodes = [4, 5]
    
    for p in path_nodes:
        for o in off_path_nodes:
            assert relevance[p] > relevance[o]

def test_bidirectional_pagerank_no_path():
    # Disconnected graph: 0 -> 1, 2 -> 3
    n = 4
    adj = sp.csr_matrix(([1, 1], ([0, 2], [1, 3])), shape=(n, n))
    
    # Target is 0, Source is 3. No path exists.
    target_seeds = np.zeros(n)
    target_seeds[0] = 1.0
    source_seeds = np.zeros(n)
    source_seeds[3] = 1.0
    
    relevance = compute_bidirectional_pagerank(adj, source_seeds, target_seeds, alpha=0.9)
    
    # Without a path, relevance should be 0 everywhere because the forward
    # and reverse passes never intersect.
    assert relevance.shape == (n,)
    assert np.all(relevance == 0)
    assert relevance.sum() == 0
