"""Integration tests for the corpus dataset → sort → embed → filter → read pipeline."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from pathlib import Path
from numpy.testing import assert_allclose

from bookgap.openalex_ingest import (
    _write_scope_parquet,
    _scope_parquet_path,
    corpus_dataset,
    ResolvedScope,
)
from bookgap.cloud import (
    _sorted_corpus_idxs,
    _post_filter_mask,
    _read_corpus_column,
    write_embeddings_store,
    read_embeddings_store,
)
from bookgap.util import Filter, WorkType


def _make_scope_works(scope: ResolvedScope, n: int, start_id: int = 0) -> list[dict]:
    """Create works for a scope with predictable IDs and dates."""
    works = []
    for i in range(n):
        gid = start_id + i
        works.append({
            "id": f"https://openalex.org/W{gid}",
            "display_name": f"Title {gid}",
            "abstract": f"Abstract for work {gid}",
            "type": scope.work_type,
            "publication_date": f"2024-{(gid % 12) + 1:02d}-{(gid % 28) + 1:02d}",
            "created_date": "2024-01-01",
            "updated_date": "2024-01-01",
            "language": "en",
            "topic": scope.topic,
            "field_id": scope.field,
            "subfield_id": scope.subfield,
            "topic_id": scope.topic,
            "primary_topic": {
                "id": f"https://openalex.org/{scope.topic}",
                "subfield": {"id": f"https://openalex.org/subfields/{scope.subfield}"},
            },
            "referenced_works": [f"https://openalex.org/W{start_id + j}" for j in range(i)],
        })
    return works


@pytest.fixture
def multi_scope_corpus(tmp_path, monkeypatch):
    """Create a multi-scope corpus with known data."""
    import bookgap.openalex_ingest as ingest_mod
    import bookgap.cloud as cloud_mod

    monkeypatch.setattr(ingest_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cloud_mod, "CACHE_DIR", tmp_path)

    # Clear lru_cache on corpus_dataset since we changed CACHE_DIR
    corpus_dataset.cache_clear()

    scopes = [
        ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="book"),
        ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="article"),
        ResolvedScope(field="17", subfield="1702", topic="T10021", work_type="article"),
    ]

    # Write scope Parquet files with known content
    # Scope 0: 5 books  (IDs W0-W4)
    # Scope 1: 10 articles (IDs W100-W109)
    # Scope 2: 8 articles  (IDs W200-W207)
    _write_scope_parquet(_scope_parquet_path(scopes[0]), _make_scope_works(scopes[0], 5, start_id=0))
    _write_scope_parquet(_scope_parquet_path(scopes[1]), _make_scope_works(scopes[1], 10, start_id=100))
    _write_scope_parquet(_scope_parquet_path(scopes[2]), _make_scope_works(scopes[2], 8, start_id=200))

    # Monkeypatch scope resolution to return our test scopes
    monkeypatch.setattr(ingest_mod, "_resolve_filter_scopes", lambda f, snapshot_date: scopes)

    work_filter = Filter(subfield="1702", work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value))

    yield {
        "scopes": scopes,
        "filter": work_filter,
        "snapshot_date": "TEST",
        "tmp_path": tmp_path,
        "total_works": 23,  # 5 + 10 + 8
    }

    # Clean up lru_cache so we don't pollute other tests
    corpus_dataset.cache_clear()


class TestCorpusDatasetRowOrder:
    def test_multi_scope_row_count(self, multi_scope_corpus):
        """corpus_dataset should contain all works from all scopes."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        assert corpus.count_rows() == multi_scope_corpus["total_works"]

    def test_row_order_matches_scope_concatenation(self, multi_scope_corpus):
        """Rows should appear in scope order: scope0 then scope1 then scope2."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        ids = corpus.to_table(columns=["id"]).column("id").to_pylist()

        # First 5 should be W0-W4 (books from scope 0)
        assert ids[0] == "https://openalex.org/W0"
        assert ids[4] == "https://openalex.org/W4"
        # Next 10 should be W100-W109 (articles from scope 1)
        assert ids[5] == "https://openalex.org/W100"
        assert ids[14] == "https://openalex.org/W109"
        # Last 8 should be W200-W207 (articles from scope 2)
        assert ids[15] == "https://openalex.org/W200"
        assert ids[22] == "https://openalex.org/W207"

    def test_column_selective_read(self, multi_scope_corpus):
        """Reading specific columns should not load others."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        table = corpus.to_table(columns=["id", "type"])
        assert table.num_columns == 2
        assert "abstract" not in table.schema.names


class TestSortedCorpusIdxs:
    def test_sorts_by_date(self, multi_scope_corpus):
        """_sorted_corpus_idxs should reorder works by publication_date."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        # All 23 works should be included
        assert len(sorted_idxs) == multi_scope_corpus["total_works"]

        # Verify dates are non-decreasing in sorted order
        all_dates = corpus.to_table(columns=["publication_date"]).column("publication_date").to_pylist()
        sorted_dates = [all_dates[i] for i in sorted_idxs]
        assert sorted_dates == sorted(sorted_dates)

    def test_result_is_subset_of_corpus_rows(self, multi_scope_corpus):
        """sorted_idxs must be valid physical row indices into the corpus."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)
        n = corpus.count_rows()
        assert all(0 <= int(idx) < n for idx in sorted_idxs)

    def test_max_works_truncates(self, multi_scope_corpus):
        """max_works should return only the earliest N works."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=10)
        assert len(sorted_idxs) == 10

    def test_max_works_returns_earliest(self, multi_scope_corpus):
        """The truncated set should contain the N earliest dates, not arbitrary ones."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        full_idxs = _sorted_corpus_idxs(corpus, max_works=None)
        trunc_idxs = _sorted_corpus_idxs(corpus, max_works=10)

        # Truncated should equal the first 10 of the full sorted list
        np.testing.assert_array_equal(trunc_idxs, full_idxs[:10])

    def test_deterministic_for_tied_dates(self, multi_scope_corpus):
        """Multiple calls should return identical results even with tied dates."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        idxs1 = _sorted_corpus_idxs(corpus, max_works=None)
        idxs2 = _sorted_corpus_idxs(corpus, max_works=None)
        np.testing.assert_array_equal(idxs1, idxs2)

    def test_no_duplicate_indices(self, multi_scope_corpus):
        """Each physical row should appear at most once in sorted_idxs."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)
        assert len(sorted_idxs) == len(set(sorted_idxs.tolist()))


class TestPostFilterMask:
    def test_filter_by_work_type_books(self, multi_scope_corpus):
        """Post-filter should correctly retain only books."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        book_filter = Filter(work_types=(WorkType.BOOK.value,))
        mask = _post_filter_mask(corpus, book_filter, sorted_idxs)

        # Exactly 5 books in the corpus
        assert mask.sum() == 5

        # The retained rows should all have type == "book"
        all_types = corpus.to_table(columns=["type"]).column("type").to_pylist()
        for pos in np.where(mask)[0]:
            corpus_row = int(sorted_idxs[pos])
            assert all_types[corpus_row] == "book"

    def test_filter_by_work_type_articles(self, multi_scope_corpus):
        """Post-filter for articles should retain all 18 articles."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        article_filter = Filter(work_types=(WorkType.ARTICLE.value,))
        mask = _post_filter_mask(corpus, article_filter, sorted_idxs)

        # 10 articles from scope 1 + 8 from scope 2 = 18
        assert mask.sum() == 18

    def test_empty_filter_passes_all(self, multi_scope_corpus):
        """Filter() with no constraints should pass all works."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        empty_filter = Filter()
        mask = _post_filter_mask(corpus, empty_filter, sorted_idxs)
        assert mask.all()
        assert len(mask) == len(sorted_idxs)

    def test_mask_length_matches_input_positions(self, multi_scope_corpus):
        """The mask must have exactly as many entries as corpus_row_idxs."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=15)

        book_filter = Filter(work_types=(WorkType.BOOK.value,))
        mask = _post_filter_mask(corpus, book_filter, sorted_idxs)
        assert len(mask) == len(sorted_idxs)

    def test_mask_dtype_is_bool(self, multi_scope_corpus):
        """Mask should be a boolean numpy array."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        mask = _post_filter_mask(corpus, Filter(), sorted_idxs)
        assert mask.dtype == bool


class TestReadCorpusColumn:
    def test_returns_correct_values_at_positions(self, multi_scope_corpus):
        """_read_corpus_column should return values at the specified positions."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        # Physical rows 0, 5, 15 are the first works from scopes 0, 1, 2
        positions = np.array([0, 5, 15], dtype=np.uint64)
        ids = _read_corpus_column(corpus, "id", positions)
        assert ids == [
            "https://openalex.org/W0",
            "https://openalex.org/W100",
            "https://openalex.org/W200",
        ]

    def test_single_element_position(self, multi_scope_corpus):
        """Should work correctly with a single-element position array."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        positions = np.array([0], dtype=np.uint64)
        ids = _read_corpus_column(corpus, "id", positions)
        assert len(ids) == 1
        assert ids[0] == "https://openalex.org/W0"

    def test_type_column(self, multi_scope_corpus):
        """Reading the 'type' column should return correct work types."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        # Row 0 is a book (scope 0), row 5 is an article (scope 1)
        positions = np.array([0, 5], dtype=np.uint64)
        types = _read_corpus_column(corpus, "type", positions)
        assert types[0] == "book"
        assert types[1] == "article"

    def test_positions_can_be_non_contiguous(self, multi_scope_corpus):
        """Positions do not need to be contiguous."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        positions = np.array([0, 2, 4, 22], dtype=np.uint64)
        ids = _read_corpus_column(corpus, "id", positions)
        assert ids[0] == "https://openalex.org/W0"
        assert ids[1] == "https://openalex.org/W2"
        assert ids[2] == "https://openalex.org/W4"
        assert ids[3] == "https://openalex.org/W207"


class TestIndexChainConsistency:
    """End-to-end consistency tests across the full index chain."""

    def test_sorted_idxs_index_into_corpus(self, multi_scope_corpus):
        """sorted_idxs values must be valid row indices for the corpus dataset."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        # Use sorted_idxs to read IDs and verify they are valid OpenAlex URLs
        ids = _read_corpus_column(corpus, "id", sorted_idxs)
        assert all(id_.startswith("https://openalex.org/W") for id_ in ids)

    def test_post_filter_positions_index_into_sorted_idxs(self, multi_scope_corpus):
        """retained_positions from post_filter index into sorted_idxs."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        book_filter = Filter(work_types=(WorkType.BOOK.value,))
        mask = _post_filter_mask(corpus, book_filter, sorted_idxs)
        retained_positions = np.where(mask)[0].astype(np.uint64)

        # retained_positions are indices into sorted_idxs
        retained_corpus_rows = sorted_idxs[retained_positions]

        # Reading at retained_corpus_rows should give only books
        types = _read_corpus_column(corpus, "type", retained_corpus_rows)
        assert all(t == "book" for t in types)

    def test_sorted_order_preserved_through_chain(self, multi_scope_corpus):
        """Works accessed via sorted_idxs should be in ascending date order."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        dates = _read_corpus_column(corpus, "publication_date", sorted_idxs)
        assert dates == sorted(dates)

    def test_book_retained_positions_are_sorted(self, multi_scope_corpus):
        """Even after post-filtering, the remaining works should be in date order."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)

        book_filter = Filter(work_types=(WorkType.BOOK.value,))
        mask = _post_filter_mask(corpus, book_filter, sorted_idxs)
        retained_positions = np.where(mask)[0].astype(np.uint64)
        retained_rows = sorted_idxs[retained_positions]

        dates = _read_corpus_column(corpus, "publication_date", retained_rows)
        assert dates == sorted(dates)

    def test_embeddings_align_with_sorted_idxs(self, multi_scope_corpus):
        """Embeddings written for sorted_idxs should index 1:1 with those positions."""
        corpus = corpus_dataset(multi_scope_corpus["filter"], multi_scope_corpus["snapshot_date"])
        sorted_idxs = _sorted_corpus_idxs(corpus, max_works=None)
        n = len(sorted_idxs)

        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(n, 8)).astype(np.float32)

        emb_path = multi_scope_corpus["tmp_path"] / "chain_emb" / "embeddings.npy"
        write_embeddings_store(emb_path, embeddings)
        loaded = read_embeddings_store(emb_path)

        # After post-filtering to books, select embedding rows by retained_positions
        book_filter = Filter(work_types=(WorkType.BOOK.value,))
        mask = _post_filter_mask(corpus, book_filter, sorted_idxs)
        retained_positions = np.where(mask)[0].astype(np.uint64)

        book_embeddings = loaded[retained_positions]
        assert book_embeddings.shape == (5, 8)
        assert_allclose(book_embeddings, embeddings[retained_positions], atol=1e-7)


class TestEmbeddingsStoreRoundtrip:
    def test_write_read_preserves_shape_and_values(self, multi_scope_corpus):
        """Embeddings written to the store should round-trip exactly."""
        rng = np.random.default_rng(42)
        path = multi_scope_corpus["tmp_path"] / "emb" / "embeddings.npy"
        embeddings = rng.normal(size=(23, 16)).astype(np.float32)

        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)

        assert loaded.shape == embeddings.shape
        assert loaded.dtype == np.float32
        assert_allclose(loaded, embeddings, atol=1e-7)

    def test_subset_indexing(self, multi_scope_corpus):
        """Indexing into loaded embeddings by retained_positions should work correctly."""
        rng = np.random.default_rng(42)
        path = multi_scope_corpus["tmp_path"] / "emb2" / "embeddings.npy"
        embeddings = rng.normal(size=(23, 16)).astype(np.float32)

        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)

        # Simulate retained_positions selecting a subset
        retained = np.array([0, 5, 10, 15, 20], dtype=np.uint64)
        subset = loaded[retained]
        assert subset.shape == (5, 16)
        assert_allclose(subset, embeddings[retained], atol=1e-7)

    def test_large_dimensionality(self, multi_scope_corpus):
        """Embeddings with large embedding dimensions should round-trip correctly."""
        rng = np.random.default_rng(99)
        path = multi_scope_corpus["tmp_path"] / "emb_large" / "embeddings.npy"
        embeddings = rng.normal(size=(10, 384)).astype(np.float32)

        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)

        assert loaded.shape == (10, 384)
        assert_allclose(loaded, embeddings, atol=1e-7)

    def test_single_row(self, multi_scope_corpus):
        """A single-row embeddings store should round-trip correctly."""
        rng = np.random.default_rng(1)
        path = multi_scope_corpus["tmp_path"] / "emb_single" / "embeddings.npy"
        embeddings = rng.normal(size=(1, 16)).astype(np.float32)

        write_embeddings_store(path, embeddings)
        loaded = read_embeddings_store(path)

        assert loaded.shape == (1, 16)
        assert_allclose(loaded, embeddings, atol=1e-7)
