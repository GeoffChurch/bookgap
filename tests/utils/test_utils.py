"""
Test utilities for creating mock clouds and subclouds for testing purposes.

This module consolidates the common patterns for test cloud creation found across test files.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any


class MockIndex:
    """Mock FAISS index for testing."""
    def __init__(self, embeddings: NDArray):
        self.embeddings = embeddings
        
    def search(self, query_embeddings: NDArray, k: int):
        """Mock search that returns reasonable nearest neighbors."""
        n_query = len(query_embeddings)
        
        # Compute actual distances for more realistic results
        distances = []
        indices = []
        
        for query_vec in query_embeddings:
            # Calculate squared distances to all points
            diffs = self.embeddings - query_vec
            sq_dists = np.sum(diffs * diffs, axis=1)
            
            # Get k nearest neighbors
            nearest_idx = np.argsort(sq_dists)[:k]
            nearest_dists = sq_dists[nearest_idx]
            
            distances.append(nearest_dists)
            indices.append(nearest_idx)
        
        # Convert to the expected format
        D = np.array(distances, dtype=np.float32)
        I = np.array(indices, dtype=np.int64)
        
        # Pad if necessary to ensure consistent shape
        if D.shape[1] < k:
            pad_width = k - D.shape[1]
            D = np.pad(D, ((0, 0), (0, pad_width)), constant_values=float('inf'))
            I = np.pad(I, ((0, 0), (0, pad_width)), constant_values=-1)
        
        return D, I


class MockSubCloud:
    """
    Mock SubCloud-like object that can be used in tests.
    
    Provides the basic interface needed for testing without requiring
    full SubCloud/Cloud infrastructure.
    """
    
    def __init__(self, embeddings: NDArray, works_data: List[Dict[str, Any]], name: str):
        """
        Create a mock SubCloud for testing.
        
        Args:
            embeddings: Array of embeddings, shape (N, D)
            works_data: List of work dictionaries with 'id' and 'display_name'
            name: Name for the cloud (for debugging/logging)
        """
        self._embeddings = embeddings.astype(np.float32)
        self._works = works_data
        self._name = name
        self._index = self._create_mock_index()
    
    def _create_mock_index(self) -> MockIndex:
        """Create a mock FAISS index for testing."""
        return MockIndex(self._embeddings)
    
    def embeddings(self) -> NDArray:
        """Return the embeddings array."""
        return self._embeddings
        
    def works(self) -> List[Dict[str, Any]]:
        """Return the works list."""
        return self._works
        
    def index(self):
        """Return the mock index."""
        return self._index
        
    @property
    def name(self) -> str:
        """Return the cloud name."""
        return self._name
        
    def __len__(self) -> int:
        """Return the number of items in the cloud."""
        return len(self._embeddings)


def create_mock_subcloud(embeddings: NDArray, works_data: List[Dict[str, Any]], name: str) -> MockSubCloud:
    """
    Helper function to create a mock SubCloud for testing.
    
    Args:
        embeddings: Array of embeddings, shape (N, D)  
        works_data: List of work dictionaries with 'id' and 'display_name'
        name: Name for the cloud
        
    Returns:
        MockSubCloud instance
    """
    return MockSubCloud(embeddings, works_data, name)


def create_simple_mock_subcloud(embeddings: NDArray, name: str) -> MockSubCloud:
    """
    Helper to create a simple mock SubCloud with auto-generated work data.
    
    Args:
        embeddings: Array of embeddings, shape (N, D)
        name: Name for the cloud
        
    Returns:
        MockSubCloud instance with auto-generated works
    """
    works_data = [
        {'id': f'{name}_{i}', 'display_name': f'{name.title()} {i}'}
        for i in range(len(embeddings))
    ]
    return create_mock_subcloud(embeddings, works_data, name)


def create_test_embeddings_2d(points_list: List[List[float]]) -> NDArray:
    """
    Create a 2D embeddings array from a list of points for testing.
    
    Args:
        points_list: List of 2D points, e.g. [[1.0, 0.0], [0.0, 1.0]]
        
    Returns:
        NumPy array of shape (N, 2) with dtype float32
    """
    return np.array(points_list, dtype=np.float32)


# Standard test patterns
def create_standard_test_clouds():
    """
    Create standard book and article clouds for testing gap analysis.
    
    Returns:
        Tuple of (books_cloud, articles_cloud) with standard test data
    """
    book_embeddings = create_test_embeddings_2d([
        [1.0, 0.0],  # Book 0
        [0.0, 1.0]   # Book 1  
    ])
    
    article_embeddings = create_test_embeddings_2d([
        [2.0, 0.0],  # Article 0 - closest to Book 0
        [0.0, 2.0],  # Article 1 - closest to Book 1
        [1.0, 1.0]   # Article 2 - equidistant from both books
    ])
    
    books = create_simple_mock_subcloud(book_embeddings, 'test_books')
    articles = create_simple_mock_subcloud(article_embeddings, 'test_articles')
    
    return books, articles
