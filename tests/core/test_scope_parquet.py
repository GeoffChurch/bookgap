from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np
import pytest
from pathlib import Path

from bookgap.openalex_ingest import _write_scope_parquet, _scope_parquet_path, ResolvedScope


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
