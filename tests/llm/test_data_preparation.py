import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from bookgap.llm.artifacts.data_preparation import GapVisualizationDataPreparer
from bookgap.visualization import ProjectionMethod

# Minimal stand-ins to avoid heavy FAISS deps in unit tests

@dataclass(frozen=True)
class DummyDistsIdxs:
    sqdists: np.ndarray
    idxs: np.ndarray

@dataclass(frozen=True)
class DummySubCloud:
    _embeddings: np.ndarray
    def embeddings(self):
        return self._embeddings
    def index(self):
        # Build a simple exact L2 index each call to avoid storing non-picklable state
        d = self._embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        x = np.ascontiguousarray(self._embeddings.astype(np.float32))
        index.add(x) # type: ignore
        return index

class DummyGap:
    def __init__(self, gap_indices: List[int], sqdists: Optional[List[float]] = None, covering_books: Optional[List[int]] = None):
        self._idxs = np.array(gap_indices, dtype=int)
        if sqdists is None:
            # First is center (0), then increasing
            self._sqdists = np.array([0.0] + [float(i+1) for i in range(len(gap_indices)-1)], dtype=float)
        else:
            self._sqdists = np.array(sqdists, dtype=float)
        self._covering_books = list(covering_books or [])
    def get_dists_idxs(self) -> DummyDistsIdxs:
        return DummyDistsIdxs(sqdists=self._sqdists, idxs=self._idxs)
    def get_covering_book_indices(self):
        return self._covering_books


def make_clouds(n_articles: int = 10, n_books: int = 3, dim: int = 4, seed: int = 0) -> Tuple[DummySubCloud, DummySubCloud]:
    rng = np.random.default_rng(seed)
    articles = rng.normal(size=(n_articles, dim)).astype(np.float32)
    books = rng.normal(size=(n_books, dim)).astype(np.float32)
    return DummySubCloud(books), DummySubCloud(articles)


def test_prepare_standard_gap_data_center_position_and_basic_assembly(monkeypatch):
    # Arrange: small deterministic clouds
    books, articles = make_clouds(n_articles=8, n_books=2, dim=5, seed=42)
    # Gap includes three article indices, first is center
    gap_indices = [2, 5, 7]
    gap = DummyGap(gap_indices, sqdists=[0.0, 1.0, 2.0], covering_books=[0])

    # Build GapWithContext-like lightweight object
    class GWCLike:
        def __init__(self, gap, topic_name, topic_id, gap_idx, books, articles):
            self.gap = gap
            self.topic_name = topic_name
            self.topic_id = topic_id
            self.gap_idx = gap_idx
            self.books = books
            self.articles = articles
    gwc = GWCLike(gap=gap, topic_name="Topic", topic_id="T1", gap_idx=0, books=books, articles=articles)

    # Monkeypatch Embedder to avoid heavy model load
    class DummyEmbedder:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def embed_texts(self, pairs, batch_size: int = 1):
            # Return zeros of appropriate shape
            return np.zeros((len(pairs), articles.embeddings().shape[1]), dtype=np.float32)

    import bookgap.llm.artifacts.data_preparation as dp
    monkeypatch.setattr(dp, 'Embedder', DummyEmbedder)

    # Act
    preparer = GapVisualizationDataPreparer()
    gwc_any: Any = gwc
    data = preparer.prepare_standard_gap_data(gwc_any, proposals=[], method=ProjectionMethod.pca())

    # Assert: center_position should point to the first gap article added
    assert data.center_position is not None
    assert data.center_position in data.gap_article_positions
    # First gap article added corresponds to center (idx 2)
    # Since base list begins with all gap articles in order, center should be the first pos
    assert data.center_position == data.gap_article_positions[0]

    # Books present include covering book(s)
    assert len(data.book_positions) >= 1


def test_neighbor_search_skip_flags(monkeypatch):
    # Arrange deterministic small clouds and gap
    books, articles = make_clouds(n_articles=6, n_books=2, dim=3, seed=7)
    gap_indices = [1, 3]
    gap = DummyGap(gap_indices, covering_books=[1])

    # Build minimal GWC
    class GWCLike:
        def __init__(self, gap, topic_name, topic_id, gap_idx, books, articles):
            self.gap = gap
            self.topic_name = topic_name
            self.topic_id = topic_id
            self.gap_idx = gap_idx
            self.books = books
            self.articles = articles
    gwc = GWCLike(gap=gap, topic_name="Topic", topic_id="T1", gap_idx=0, books=books, articles=articles)

    # Stub Embedder (not used when proposals=[] but harmless)
    class DummyEmbedder:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def embed_texts(self, pairs, batch_size: int = 1):
            return np.zeros((len(pairs), articles.embeddings().shape[1]), dtype=np.float32)
    import bookgap.llm.artifacts.data_preparation as dp
    monkeypatch.setattr(dp, 'Embedder', DummyEmbedder)

    # Act
    preparer = GapVisualizationDataPreparer()
    gwc_any: Any = gwc
    data = preparer.prepare_standard_gap_data(gwc_any, proposals=[], method=ProjectionMethod.pca())

    # Assert: skip flags mean no non-gap article neighbors added
    assert set(data.article_positions) == set(data.gap_article_positions)
    # But books should include at least the covering book and possibly neighbors
    assert len(data.book_positions) >= 1
