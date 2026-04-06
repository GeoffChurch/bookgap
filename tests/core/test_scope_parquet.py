from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np
import pytest
from pathlib import Path

from bookgap.openalex_ingest import _write_scope_parquet, _scope_parquet_path, ResolvedScope, fetch_works, corpus_dataset, fetch_all_scopes
from bookgap.util import Filter, WorkType


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    import bookgap.openalex_ingest as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    return tmp_path


def _sample_works(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"https://openalex.org/W{i}",
            "display_name": f"Title {i}",
            "abstract": f"Abstract text for work {i} with some words",
            "type": "article" if i % 2 == 0 else "book",
            "publication_date": f"2024-01-{i+1:02d}",
            "created_date": f"2024-01-{i+1:02d}",
            "updated_date": f"2024-01-{i+1:02d}",
            "language": "en",
            "topic": "T10020",
            "field_id": "17",
            "subfield_id": "1702",
            "topic_id": "T10020",
            "primary_topic": {"id": "T10020", "subfield": {"id": "1702"}},
            "referenced_works": [f"https://openalex.org/W{j}" for j in range(i)],
        }
        for i in range(n)
    ]


class TestWriteScopeParquet:
    def test_roundtrip(self, tmp_cache):
        works = _sample_works(10)
        path = tmp_cache / "test.parquet"
        _write_scope_parquet(path, works)
        assert path.exists()
        table = pq.read_table(path)
        assert len(table) == 10
        assert "id" in table.schema.names
        assert "abstract" in table.schema.names
        assert "referenced_works" in table.schema.names
        assert "embeddings" not in table.schema.names

    def test_column_selective_read(self, tmp_cache):
        works = _sample_works(10)
        path = tmp_cache / "test.parquet"
        _write_scope_parquet(path, works)
        table = pq.read_table(path, columns=["id", "referenced_works"])
        assert table.num_columns == 2
        assert "abstract" not in table.schema.names

    def test_empty_works(self, tmp_cache):
        path = tmp_cache / "empty.parquet"
        _write_scope_parquet(path, [])
        table = pq.read_table(path)
        assert len(table) == 0

    def test_dataset_return(self, tmp_cache):
        works = _sample_works(5)
        path = tmp_cache / "test.parquet"
        _write_scope_parquet(path, works)
        dataset = ds.dataset(path, format="parquet")
        assert isinstance(dataset, ds.Dataset)
        assert dataset.count_rows() == 5
        table = dataset.to_table(columns=["id"])
        assert table.num_columns == 1


class TestScopeParquetPath:
    def test_path_structure(self, tmp_cache):
        scope = ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="article")
        path = _scope_parquet_path(scope)
        assert path.name == "article.parquet"
        assert "17" in str(path) and "1702" in str(path) and "T10020" in str(path)


class TestFetchWorksParquet:
    def test_returns_dataset(self, tmp_cache):
        scope = ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="article")
        works = _sample_works(5)
        path = _scope_parquet_path(scope)
        _write_scope_parquet(path, works)
        result = fetch_works(scope)
        assert isinstance(result, ds.Dataset)
        assert result.count_rows() == 5

    def test_idempotent(self, tmp_cache):
        scope = ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="book")
        path = _scope_parquet_path(scope)
        _write_scope_parquet(path, _sample_works(3))
        d1 = fetch_works(scope)
        d2 = fetch_works(scope)
        assert d1.count_rows() == d2.count_rows() == 3


class TestCorpusDataset:
    def test_unions_scopes(self, tmp_cache, monkeypatch):
        scopes = [
            ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="book"),
            ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="article"),
        ]
        _write_scope_parquet(_scope_parquet_path(scopes[0]), _sample_works(3))
        _write_scope_parquet(_scope_parquet_path(scopes[1]), _sample_works(7))
        import bookgap.openalex_ingest as mod
        monkeypatch.setattr(mod, "_resolve_filter_scopes", lambda f: scopes)
        filt = Filter(subfield="1702", work_types=(WorkType.BOOK.value, WorkType.ARTICLE.value))
        dataset = corpus_dataset(filt)
        assert isinstance(dataset, ds.Dataset)
        assert dataset.count_rows() == 10

    def test_column_selection(self, tmp_cache, monkeypatch):
        scope = ResolvedScope(field="17", subfield="1702", topic="T10020", work_type="article")
        _write_scope_parquet(_scope_parquet_path(scope), _sample_works(5))
        import bookgap.openalex_ingest as mod
        monkeypatch.setattr(mod, "_resolve_filter_scopes", lambda f: [scope])
        filt = Filter(subfield="1702", work_types=(WorkType.ARTICLE.value,))
        dataset = corpus_dataset(filt)
        table = dataset.to_table(columns=["id", "referenced_works"])
        assert table.num_columns == 2
        assert len(table) == 5
