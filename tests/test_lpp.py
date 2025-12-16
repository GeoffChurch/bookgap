import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from artifacts_scripts.curriculum.dimensionality_reduction import project_to_2d, DimensionalityReductionMethod


class DummyConfig:
    def __init__(self, center=True, n_components=2):
        self.center = center
        self.n_components = n_components


def test_lpp_basic_clustering():
    # 4 nodes, two tight clusters (0,1) and (2,3)
    X = np.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ])

    # Adjacency: strong within-cluster links, plus self-loops
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    A = sp.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=float)

    # Row-normalize
    A = normalize(A, norm='l1', axis=1)

    cfg = DummyConfig(center=True, n_components=2)
    emb2d, reducer = project_to_2d(X, labels=None, method=DimensionalityReductionMethod.LPP, lpp_config=cfg, citation_graph=A)

    assert emb2d.shape == (4, 2)

    # distances within clusters should be smaller than between clusters
    from scipy.spatial.distance import pdist, squareform
    dmat = squareform(pdist(emb2d))
    within = (dmat[0,1] + dmat[2,3]) / 2.0
    between = (dmat[0,2] + dmat[0,3] + dmat[1,2] + dmat[1,3]) / 4.0
    assert within < between


def test_lpp_requires_normalized_rows():
    X = np.eye(3)
    # Non-normalized adjacency
    A = sp.csr_matrix([[1, 2, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cfg = DummyConfig()
    emb2d, reducer = project_to_2d(X, method=DimensionalityReductionMethod.LPP, lpp_config=cfg, citation_graph=A)
    assert emb2d.shape[0] == X.shape[0]
