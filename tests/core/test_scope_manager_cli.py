from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def cli_runner():
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def cli_app():
    from scope_manager import _build_cli
    return _build_cli()


class TestListCommand:
    def test_list_domains_table(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["list", "domains"])
        assert result.exit_code == 0
        assert "Physical Sciences" in result.output

    def test_list_domains_json(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["--json", "list", "domains"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["id"] == "3"
        assert data[0]["name"] == "Physical Sciences"

    def test_list_fields_requires_parent(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["list", "fields"])
        assert result.exit_code != 0

    @patch("scope_manager.get_domain_fields")
    def test_list_fields_json(self, mock_get, cli_runner, cli_app):
        mock_get.return_value = [("17", "Computer Science"), ("22", "Engineering")]
        result = cli_runner.invoke(cli_app, ["--json", "list", "fields", "3"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["id"] == "17"

    @patch("scope_manager.get_field_subfields")
    def test_list_subfields_json(self, mock_get, cli_runner, cli_app):
        mock_get.return_value = [("1702", "Artificial Intelligence")]
        result = cli_runner.invoke(cli_app, ["--json", "list", "subfields", "17"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["name"] == "Artificial Intelligence"

    @patch("scope_manager.get_subfield_topics")
    def test_list_topics_json(self, mock_get, cli_runner, cli_app):
        mock_get.return_value = [("T10181", "NLP")]
        result = cli_runner.invoke(cli_app, ["--json", "list", "topics", "1702"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["id"] == "T10181"

    def test_list_invalid_entity(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["list", "galaxies"])
        assert result.exit_code != 0


class TestStatusCommand:
    @patch("scope_manager.get_scope_status")
    def test_single_scope_json(self, mock_status, cli_runner, cli_app):
        mock_status.return_value = {
            "fetched": True,
            "work_count": 42,
            "gist_embedded": True,
            "gemma_embedded": False,
        }
        result = cli_runner.invoke(cli_app, ["--json", "status", "17/1702/T10181/article"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["fetched"] is True
        assert data["work_count"] == 42

    @patch("scope_manager.get_fast_aggregate_status")
    def test_aggregate_status_json(self, mock_agg, cli_runner, cli_app):
        mock_agg.return_value = {
            "article": {"fetched": 5, "total": 10, "gist_count": 3, "gemma_count": 1},
        }
        result = cli_runner.invoke(cli_app, ["--json", "status", "17/1702"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["article"]["fetched"] == 5


class TestFetchCommand:
    @patch("scope_manager.launch_worker")
    def test_fetch_success_json(self, mock_launch, cli_runner, cli_app):
        mock_launch.return_value = "fetch_T10181_article_123"
        result = cli_runner.invoke(cli_app, ["--json", "fetch", "17/1702/T10181/article"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["job_id"] == "fetch_T10181_article_123"

    @patch("scope_manager.launch_worker")
    def test_fetch_duplicate(self, mock_launch, cli_runner, cli_app):
        mock_launch.return_value = None
        result = cli_runner.invoke(cli_app, ["--json", "fetch", "17/1702/T10181/article"])
        assert result.exit_code != 0


class TestEmbedCommand:
    @patch("scope_manager.launch_worker")
    def test_embed_success_json(self, mock_launch, cli_runner, cli_app):
        mock_launch.return_value = "embed_T10181_article_GIST_123"
        result = cli_runner.invoke(cli_app, ["--json", "embed", "17/1702/T10181/article", "--model", "gist"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["job_id"] == "embed_T10181_article_GIST_123"


class TestJobsCommand:
    @patch("scope_manager.discover_jobs")
    def test_jobs_list_json(self, mock_discover, cli_runner, cli_app):
        mock_discover.return_value = [
            {"_job_id": "fetch_1", "action": "fetch", "scope": "17/1702/T10181/article",
             "model": None, "status": "running", "started_at": "2026-04-08T10:00:00"},
        ]
        result = cli_runner.invoke(cli_app, ["--json", "jobs"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["status"] == "running"

    @patch("scope_manager.dismiss_finished_jobs")
    def test_jobs_dismiss_json(self, mock_dismiss, cli_runner, cli_app):
        mock_dismiss.return_value = 3
        result = cli_runner.invoke(cli_app, ["--json", "jobs", "dismiss"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["dismissed"] == 3
