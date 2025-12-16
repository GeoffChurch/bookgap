import numpy as np
from pathlib import Path

import artifacts_scripts.curriculum.plot_curriculum as mod
from artifacts_scripts.curriculum.plot_curriculum import plot_curriculum


class DummyReducer:
    explained_variance_ratio_ = np.array([0.1089, 0.0543])


def test_plot_curriculum_plain(tmp_path, monkeypatch):
    # Prepare fake curriculum data
    frontier_nodes = np.array([0, 1, 2], dtype=np.int64)
    leaf_to_cluster = np.array([0, 0, 1, 1, 2, -1], dtype=np.int64)
    centroids = np.random.RandomState(0).randn(3, 4).astype(np.float64)
    linearization_order = np.array([0, 1, 2], dtype=np.int64)

    def fake_load(curriculum_dir, embeddings):
        return frontier_nodes, leaf_to_cluster, centroids, linearization_order

    def fake_project(arr, labels, method, umap_config, lpp_config, citation_graph=None):
        # Return a simple 2D projection and dummy reducer
        if arr.shape[1] > 2:
            proj = np.random.RandomState(1).randn(arr.shape[0], 2)
        else:
            proj = arr.copy()
        return proj, DummyReducer()

    monkeypatch.setattr(mod, "load_curriculum_data", fake_load)
    monkeypatch.setattr(mod, "project_to_2d", fake_project)

    out = tmp_path / "curriculum_plot.png"
    plot_curriculum(
        curriculum_dir=tmp_path,
        embeddings=np.random.randn(6, 4),
        output_path=out,
        show_labels=False,
    )

    assert out.exists()


def test_plot_curriculum_background(tmp_path, monkeypatch):
    frontier_nodes = np.array([0, 1], dtype=np.int64)
    leaf_to_cluster = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    centroids = np.random.RandomState(0).randn(2, 4).astype(np.float64)
    linearization_order = np.array([0, 1], dtype=np.int64)

    def fake_load(curriculum_dir, embeddings):
        return frontier_nodes, leaf_to_cluster, centroids, linearization_order

    def fake_project(arr, labels, method, umap_config, lpp_config, citation_graph=None):
        proj = np.random.RandomState(2).randn(arr.shape[0], 2)
        return proj, DummyReducer()

    monkeypatch.setattr(mod, "load_curriculum_data", fake_load)
    monkeypatch.setattr(mod, "project_to_2d", fake_project)

    out = tmp_path / "curriculum_plot_with_background.png"
    plot_curriculum(
        curriculum_dir=tmp_path,
        embeddings=np.random.randn(5, 4),
        output_path=out,
        show_labels=False,
        background=True,
        max_points_per_topic=2,
    )

    assert out.exists()
