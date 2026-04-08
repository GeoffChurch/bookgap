from __future__ import annotations

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bookgap.openalex_ingest import _write_scope_parquet, ResolvedScope


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
            "abstract": f"Abstract text for work {i}",
            "type": "article",
            "publication_date": f"2024-01-{i+1:02d}",
            "created_date": f"2024-01-{i+1:02d}",
            "updated_date": f"2024-01-{i+1:02d}",
            "language": "en",
            "topic": "T10020",
            "field_id": "17",
            "subfield_id": "1702",
            "topic_id": "T10020",
            "primary_topic": {"id": "T10020", "subfield": {"id": "1702"}},
            "referenced_works": [],
        }
        for i in range(n)
    ]


class TestScopeStatus:
    def test_not_fetched(self, tmp_cache):
        from bookgap.scope_ops import get_scope_status
        scope = ResolvedScope(field="17", subfield="1702", topic="T10181", work_type="article")
        status = get_scope_status(scope)
        assert status["fetched"] is False
        assert status["work_count"] == 0
        assert status["gist_embedded"] is False
        assert status["gemma_embedded"] is False

    def test_fetched(self, tmp_cache):
        from bookgap.scope_ops import get_scope_status
        scope = ResolvedScope(field="17", subfield="1702", topic="T10181", work_type="article")
        path = tmp_cache / "works_parquet" / "17" / "1702" / "T10181" / "article.parquet"
        _write_scope_parquet(path, _sample_works(10))

        status = get_scope_status(scope)
        assert status["fetched"] is True
        assert status["work_count"] == 10
        assert status["gist_embedded"] is False

    def test_embedded(self, tmp_cache):
        from bookgap.scope_ops import get_scope_status
        scope = ResolvedScope(field="17", subfield="1702", topic="T10181", work_type="article")
        path = tmp_cache / "works_parquet" / "17" / "1702" / "T10181" / "article.parquet"
        _write_scope_parquet(path, _sample_works(5))
        emb_dir = tmp_cache / "embeddings" / "avsolatorio" / "GIST-small-Embedding-v0" / "17" / "1702" / "T10181" / "article"
        emb_dir.mkdir(parents=True)
        np.save(emb_dir / "batch_0_embeddings.npy", np.zeros((5, 384)))

        status = get_scope_status(scope)
        assert status["fetched"] is True
        assert status["gist_embedded"] is True
        assert status["gemma_embedded"] is False


class TestHierarchy:
    def test_get_domain_fields(self):
        from bookgap.scope_ops import get_domain_fields
        mock_domain_data = {
            "id": "https://openalex.org/domains/3",
            "display_name": "Physical Sciences",
            "fields": [
                {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
                {"id": "https://openalex.org/fields/22", "display_name": "Engineering"},
            ],
        }
        with patch("pyalex.Domains") as mock_domains:
            mock_domains.return_value.__getitem__ = MagicMock(return_value=mock_domain_data)
            fields = get_domain_fields("3")
        assert len(fields) == 2
        assert fields[0] == ("17", "Computer Science")
        assert fields[1] == ("22", "Engineering")

    def test_get_field_subfields(self):
        from bookgap.scope_ops import get_field_subfields
        mock_field_data = {
            "id": "https://openalex.org/fields/17",
            "display_name": "Computer Science",
            "subfields": [
                {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
                {"id": "https://openalex.org/subfields/1709", "display_name": "Human-Computer Interaction"},
            ],
        }
        with patch("pyalex.Fields") as mock_fields:
            mock_fields.return_value.__getitem__ = MagicMock(return_value=mock_field_data)
            subfields = get_field_subfields("17")
        assert len(subfields) == 2
        assert subfields[0] == ("1702", "Artificial Intelligence")

    def test_get_subfield_topics(self):
        from bookgap.scope_ops import get_subfield_topics
        mock_pages = [
            [
                {"id": "https://openalex.org/topics/T10181", "display_name": "Natural Language Processing"},
                {"id": "https://openalex.org/topics/T10191", "display_name": "Robotics and Automation"},
            ]
        ]
        with patch("pyalex.Topics") as mock_topics:
            mock_filter = MagicMock()
            mock_filter.paginate.return_value = mock_pages
            mock_topics.return_value.filter.return_value = mock_filter
            topics = get_subfield_topics("1702")
        assert len(topics) == 2
        assert ("T10181", "Natural Language Processing") in topics
