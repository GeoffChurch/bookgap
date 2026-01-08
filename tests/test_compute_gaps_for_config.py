import numpy as np
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
import pytest
import bookgap.gaps

from bookgap.llm.data_structures import GapWithContext
from bookgap.types import CellType
from bookgap.util import Filter

# Note: We use _compute_gaps_impl directly in tests to bypass caching
# (which would try to pickle Mock objects in test configs)

# Get the original unwrapped functions before the test starts
_original_compute_gaps_impl = bookgap.gaps._compute_gaps_impl.__wrapped__  # type: ignore[attr-defined]
_original_compute_gaps_for_topic = bookgap.gaps._compute_gaps_for_topic.__wrapped__  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _uncached_compute_gaps_impl(monkeypatch):
    """Replace the cached _compute_gaps_impl with an uncached version for testing.
    
    This allows tests to use Mock objects without hitting pickle errors from joblib caching.
    """
    # Use the unwrapped functions directly to bypass joblib hashing of Mock inputs
    monkeypatch.setattr(bookgap.gaps, '_compute_gaps_impl', _original_compute_gaps_impl)
    monkeypatch.setattr(bookgap.gaps, '_compute_gaps_for_topic', _original_compute_gaps_for_topic)
    yield


class _TopicsDict:
    def __init__(self, display_name: str):
        self.display_name = display_name
    def __getitem__(self, key: str) -> dict[str, Any]:
        # Minimal payload used by compute_gaps_for_config
        return {"display_name": self.display_name, "subfield": {"id": "https://openalex.org/S1711"}}


class _SubCloudStub:
    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
    def embeddings(self) -> np.ndarray:
        return self._embeddings


def _make_cloud_mock(merged_filter: Filter) -> Mock:
    """Create a mock Cloud object with the given merged filter."""
    mock_cloud = Mock(spec=['filter_pair'])
    mock_cloud.filter_pair.merged.return_value = merged_filter
    return mock_cloud


def _make_args(
    cloud: Any,
    num_gaps: int = 2,
    radius_fraction: float = 0.5,
    cell_type_algo: Any = None,
) -> tuple[Any, float, Any, int]:
    """Create arguments for _compute_gaps_impl.
    
    Args:
        cloud: Mock Cloud object
        num_gaps: Number of gaps to find
        radius_fraction: Radius fraction for gap detection
        cell_type_algo: Mock algorithm object (will be wrapped in CellType)
    
    Returns:
        Tuple of (cloud, radius_fraction, cell_type, num_gaps) for _compute_gaps_impl
    """
    if cell_type_algo is None:
        cell_type_algo = Mock()
    
    # Create a mock CellType that has a .value property returning our mock algorithm
    mock_cell_type = Mock()
    mock_cell_type.value = cell_type_algo
    
    return cloud, radius_fraction, mock_cell_type, num_gaps


@patch("bookgap.gaps.instantiate")
@patch("bookgap.gaps.get_subclouds_for_topic")
@patch("bookgap.gaps.pyalex.Topics", return_value=_TopicsDict("Machine Learning"))
@patch("bookgap.gaps.resolve_filter_upward", return_value=Filter(field="17", subfield="1711", topic="T10828"))
def test_compute_gaps_for_config_single_topic_success(mock_resolve, mock_topics, mock_get_subclouds, mock_instantiate):
    # Cloud with merged filter specifying a single topic
    merged_filter = Filter(field="17", subfield="1711", topic="T10828", work_types=("article", "book"))
    mock_cloud = _make_cloud_mock(merged_filter)

    # Subclouds with embeddings
    books = _SubCloudStub(np.eye(3, dtype=np.float32))
    articles = _SubCloudStub(np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], dtype=np.float32))
    mock_get_subclouds.return_value = (books, articles)

    # Configure gap algorithm mocks
    # First call (book gap sizes): returns list of gaps
    book_gap1 = Mock(); book_gap1.get_dists_idxs.return_value = SimpleNamespace(idxs=[0, 1])
    book_gap2 = Mock(); book_gap2.get_dists_idxs.return_value = SimpleNamespace(idxs=[0, 1, 2])
    # Second call (article query gaps): iterable of gaps limited by num_gaps
    art_gap1 = Mock(); art_gap1.get_dists_idxs.return_value = SimpleNamespace(idxs=[0, 1, 2])
    art_gap2 = Mock(); art_gap2.get_dists_idxs.return_value = SimpleNamespace(idxs=[0, 1])

    # Create a mock algorithm with configured side_effect
    mock_cell_type_algo = Mock()
    mock_cell_type_algo.compute_gap_sizes_for_embeddings.return_value = [1, 2, 3]
    # find_gaps should yield article gaps (book gaps are handled via compute_gap_sizes_for_embeddings)
    mock_cell_type_algo.find_gaps.side_effect = [iter([art_gap1, art_gap2])]
    
    cloud, radius_fraction, cell_type, num_gaps = _make_args(mock_cloud, num_gaps=2, cell_type_algo=mock_cell_type_algo)
    gaps = bookgap.gaps._compute_gaps_impl(cloud, radius_fraction, cell_type, num_gaps)

    assert isinstance(gaps, list)
    assert len(gaps) == 2
    assert all(isinstance(g, GapWithContext) for g in gaps)

    # Validate first gap content
    g0 = gaps[0]
    assert g0.topic_name == "Machine Learning"
    assert g0.topic_id == "T10828"
    assert g0.gap_idx == 0
    assert g0.book_gap_sizes == [1, 2, 3]
    assert g0.gap_size == 3  # from art_gap1

    # Validate second gap content
    g1 = gaps[1]
    assert g1.gap_idx == 1
    assert g1.gap_size == 2  # from art_gap2


@patch("bookgap.gaps.instantiate")
@patch("bookgap.gaps.get_subclouds_for_topic")
@patch("bookgap.gaps.pyalex.Topics", return_value=_TopicsDict("AI"))
@patch("bookgap.gaps.resolve_filter_upward", return_value=Filter(field="17", subfield="1711", topic="T10828"))
def test_compute_gaps_for_config_no_articles(mock_resolve, mock_topics, mock_get_subclouds, mock_instantiate):
    # Single topic path again
    merged_filter = Filter(field="17", subfield="1711", topic="T10828", work_types=("article", "book"))
    mock_cloud = _make_cloud_mock(merged_filter)

    # No articles; should skip topic and return empty list
    books = _SubCloudStub(np.eye(1, dtype=np.float32))
    articles = _SubCloudStub(np.empty((0, 3), dtype=np.float32))
    mock_get_subclouds.return_value = (books, articles)

    mock_cell_type_algo = Mock()
    mock_cell_type_algo.compute_gap_sizes_for_embeddings.return_value = [1]
    cloud, radius_fraction, cell_type, num_gaps = _make_args(mock_cloud, num_gaps=2, cell_type_algo=mock_cell_type_algo)
    gaps = bookgap.gaps._compute_gaps_impl(cloud, radius_fraction, cell_type, num_gaps)
    assert gaps == []


@patch("bookgap.gaps.instantiate")
@patch("bookgap.gaps.get_field_topics", return_value=["T123", "T456"])
@patch("bookgap.gaps.get_subfield_topics")
@patch("bookgap.gaps.pyalex.Topics", side_effect=[_TopicsDict("Topic A"), _TopicsDict("Topic B")])
@patch("bookgap.gaps.get_subclouds_for_topic")
@patch("bookgap.gaps.resolve_filter_upward", return_value=Filter(field="17"))
def test_compute_gaps_for_config_field_only(mock_resolve, mock_get_subclouds, mock_topics, mock_get_subfield_topics, mock_get_field_topics, mock_instantiate):
    # Field-only filter resolved upward
    merged_filter = Filter(field="17", subfield=None, topic=None, work_types=("article", "book"))
    mock_cloud = _make_cloud_mock(merged_filter)

    # Subclouds
    books = _SubCloudStub(np.eye(2, dtype=np.float32))
    articles = _SubCloudStub(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    mock_get_subclouds.side_effect = [(books, articles), (books, articles)]

    # Gap algo: two topics, each yields one article gap
    art_gap_a = Mock(); art_gap_a.get_dists_idxs.return_value = SimpleNamespace(idxs=[0])
    art_gap_b = Mock(); art_gap_b.get_dists_idxs.return_value = SimpleNamespace(idxs=[0, 1])

    mock_cell_type_algo = Mock()
    mock_cell_type_algo.compute_gap_sizes_for_embeddings.return_value = [1, 1]
    mock_cell_type_algo.find_gaps.side_effect = [
        iter([art_gap_a]),  # topic T123
        iter([art_gap_b]),  # topic T456
    ]
    
    cloud, radius_fraction, cell_type, num_gaps = _make_args(mock_cloud, num_gaps=1, cell_type_algo=mock_cell_type_algo)
    gaps = bookgap.gaps._compute_gaps_impl(cloud, radius_fraction, cell_type, num_gaps)
    assert len(gaps) == 2
    assert [g.topic_id for g in gaps] == ["T123", "T456"]
    assert [g.gap_size for g in gaps] == [1, 2]
