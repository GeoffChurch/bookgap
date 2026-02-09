import numpy as np
import scipy.sparse as sp

from artifacts_scripts.curriculum.similarity_graph import (
    build_similarity_adjacency_from_knn,
    mix_adjacency_matrices,
)


def test_similarity_adjacency_row_normalized_no_self():
    # Three nodes, each with two neighbors (including self)
    neighbor_ids = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
        dtype=np.int64,
    )
    # Squared distances for normalized vectors: 0 -> cos=1, 2 -> cos=0
    neighbor_sq_dists = np.array(
        [
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )

    adj = build_similarity_adjacency_from_knn(
        neighbor_ids,
        neighbor_sq_dists,
        n_nodes=3,
        sharpness=4.0,
        row_normalize=True,
        include_self=False,
    )

    # No self-loops expected
    assert adj[0, 0] == 0
    assert adj[1, 1] == 0
    assert adj[2, 2] == 0

    # Row sums should be 1 when normalized
    row_sums = np.asarray(adj.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 1.0)


def test_mix_adjacency_convex_and_global_scale():
    adj_cite = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
    )
    adj_sim = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
    )

    mixed_convex = mix_adjacency_matrices(adj_cite, adj_sim, mix_weight=0.5, mix_mode="convex")
    assert np.isclose(mixed_convex[0, 1], 1.5)

    mixed_scaled = mix_adjacency_matrices(
        adj_cite, adj_sim, mix_weight=0.5, mix_mode="additive_global_scale"
    )
    # sum_cite=1, sum_sim=2 => scale=0.5 => add 0.5*0.5*2 = 0.5
    assert np.isclose(mixed_scaled[0, 1], 1.5)
