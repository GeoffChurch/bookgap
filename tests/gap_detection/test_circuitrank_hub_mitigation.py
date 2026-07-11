import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.circuitrank import build_conductance_adjacency, transform_node_masses


def _make_graph_for_weight_tests() -> sp.csr_matrix:
    """Graph with a clear high-indegree hub.

    Nodes:
      0 cites hub(4) and leaf(2)
      1 cites hub(4) and leaf(3)
      5 cites hub(4)

    So:
      outdeg(0)=2, outdeg(1)=2, outdeg(5)=1
      indeg(4)=3, indeg(2)=1, indeg(3)=1
    """
    n = 6
    rows = [0, 0, 1, 1, 5]
    cols = [4, 2, 4, 3, 4]
    data = [1, 1, 1, 1, 1]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_build_conductance_adjacency_pagerank_normalized_weights():
    adj = _make_graph_for_weight_tests()
    eff = build_conductance_adjacency(adj, normalize=True, edge_weighting="pagerank")

    # Weight for edge v->u is stored at [u, v] in eff.
    assert eff[4, 0] == 0.5  # 0 has outdeg 2
    assert eff[2, 0] == 0.5
    assert eff[4, 1] == 0.5  # 1 has outdeg 2
    assert eff[3, 1] == 0.5
    assert eff[4, 5] == 1.0  # 5 has outdeg 1


def test_build_conductance_adjacency_symmetric_no_renorm():
    adj = _make_graph_for_weight_tests()
    eff = build_conductance_adjacency(
        adj,
        normalize=True,  # ignored for non-pagerank modes
        edge_weighting="symmetric",
        degree_epsilon=0.0,
    )

    w_hub_from_out2 = 1.0 / np.sqrt(2.0 * 3.0)
    w_leaf_from_out2 = 1.0 / np.sqrt(2.0 * 1.0)
    w_hub_from_out1 = 1.0 / np.sqrt(1.0 * 3.0)

    np.testing.assert_allclose(eff[4, 0], w_hub_from_out2, atol=0, rtol=0)
    np.testing.assert_allclose(eff[2, 0], w_leaf_from_out2, atol=0, rtol=0)
    np.testing.assert_allclose(eff[4, 1], w_hub_from_out2, atol=0, rtol=0)
    np.testing.assert_allclose(eff[3, 1], w_leaf_from_out2, atol=0, rtol=0)
    np.testing.assert_allclose(eff[4, 5], w_hub_from_out1, atol=0, rtol=0)


def test_build_conductance_adjacency_indegree_discount_with_renorm():
    adj = _make_graph_for_weight_tests()
    eff = build_conductance_adjacency(
        adj,
        normalize=True,
        edge_weighting="indegree_discount_renorm",
        indegree_discount_gamma=0.5,
        degree_epsilon=0.0,
    )

    # For citer 0: weights proportional to 1/sqrt(indeg)
    # hub indeg=3 => 1/sqrt(3), leaf indeg=1 => 1
    a = 1.0 / np.sqrt(3.0)
    b = 1.0
    hub_w = a / (a + b)
    leaf_w = b / (a + b)

    np.testing.assert_allclose(eff[4, 0], hub_w, atol=0, rtol=0)
    np.testing.assert_allclose(eff[2, 0], leaf_w, atol=0, rtol=0)
    np.testing.assert_allclose(eff[4, 1], hub_w, atol=0, rtol=0)
    np.testing.assert_allclose(eff[3, 1], leaf_w, atol=0, rtol=0)

    # Citer 5 has only one outgoing edge, so renorm yields weight 1.
    np.testing.assert_allclose(eff[4, 5], 1.0, atol=0, rtol=0)


def test_transform_node_masses_sqrt_with_renormalize():
    masses = np.array([0.0, 4.0, 9.0], dtype=np.float64)
    out = transform_node_masses(masses, transform="sqrt", renormalize=True)

    # sqrt -> [0,2,3], then renorm to preserve sum=13.
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float64)
    expected *= 13.0 / expected.sum()
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)
