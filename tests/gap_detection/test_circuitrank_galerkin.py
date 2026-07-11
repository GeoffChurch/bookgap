import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.circuitrank import (
    compute_graph_potential,
    compute_graph_potential_galerkin,
)


def test_galerkin_matches_node_space_with_identity_basis():
    # Simple 2-node graph: 0 cites 1
    adj = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
    )

    embeddings = np.eye(2, dtype=np.float64)
    current_sources = np.array([1.0, -1.0], dtype=np.float64)

    phi_node = compute_graph_potential(
        adj,
        current_sources=current_sources,
        leakage=0.1,
        normalize=False,
        edge_weighting="pagerank",
    )

    phi_galerkin = compute_graph_potential_galerkin(
        adj,
        embeddings,
        current_sources=current_sources,
        leakage=0.1,
        normalize=False,
        edge_weighting="pagerank",
    )

    assert np.allclose(phi_node, phi_galerkin, rtol=1e-6, atol=1e-8)
