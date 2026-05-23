"""Tests for the LLM-backend dispatch in `workflow_functions`.

Exercises both the historical OpenAI Batch API path and the new
synchronous local-Ollama path through a small in-memory fixture. The
goal is shape parity: both backends must produce a
:class:`BatchResults` whose `proposals` are :class:`ParsedProposal`
objects with embeddings ready for the shared downstream-artifact code.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pytest
from omegaconf import OmegaConf

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.gap_types import DistsAndIdxs
from bookgap.llm.data_structures import BatchResults, GapWithContext
from bookgap.llm.generators.run_state import RunStateManager, RunStep
from bookgap.llm.generators.workflow_functions import (
    _ollama_record_to_jsonl_dict,
    _parse_ollama_results_content,
    _read_completed_custom_ids,
    _resolve_backend,
    _run_ollama_local_path,
    _stream_one_proposal_metrics,
    run_batch_workflow,
)
from bookgap.llm.identifiers import build_gap_proposal_id
from bookgap.llm.protocols import (
    GapProtocol,
    LLMBackend,
    OpenAIModel,
    ParsedProposal,
    Proposal,
    ProposalContext,
    ProposalCounts,
)
from bookgap.subcloud import SubCloud


# ---------------------------------------------------------------------------
# LLMConfig + LLMBackend basic wiring
# ---------------------------------------------------------------------------

class TestLLMBackendEnum:
    def test_values(self) -> None:
        assert LLMBackend.OPENAI_BATCH.value == "openai_batch"
        assert LLMBackend.OLLAMA_LOCAL.value == "ollama_local"

    def test_default_llm_config_uses_openai_batch(self) -> None:
        cfg = LLMConfig()
        assert cfg.backend is LLMBackend.OPENAI_BATCH
        # Hydra _target_ unchanged so existing Hydra-driven instantiation
        # of the OpenAI generator is byte-identical.
        assert cfg._target_ == (
            "bookgap.llm.generators.openai_batch.OpenAIBatchGenerator"
        )
        assert cfg.ollama_model == "gemma4:31b"

    def test_can_select_ollama_via_config(self) -> None:
        cfg = LLMConfig(backend=LLMBackend.OLLAMA_LOCAL)
        assert cfg.backend is LLMBackend.OLLAMA_LOCAL


class TestResolveBackend:
    def _cfg_with_backend(self, raw: Any) -> ExperimentConfig:
        return cast(
            ExperimentConfig, SimpleNamespace(llm=SimpleNamespace(backend=raw)),
        )

    def test_enum_passthrough(self) -> None:
        cfg = self._cfg_with_backend(LLMBackend.OLLAMA_LOCAL)
        assert _resolve_backend(cfg) is LLMBackend.OLLAMA_LOCAL

    def test_string_value_resolves(self) -> None:
        cfg = self._cfg_with_backend("ollama_local")
        assert _resolve_backend(cfg) is LLMBackend.OLLAMA_LOCAL

    def test_string_name_resolves(self) -> None:
        # OmegaConf sometimes surfaces the enum name (uppercase) when
        # loaded from YAML overrides.
        cfg = self._cfg_with_backend("OLLAMA_LOCAL")
        assert _resolve_backend(cfg) is LLMBackend.OLLAMA_LOCAL

    def test_missing_backend_defaults_to_openai_batch(self) -> None:
        # no `backend` attribute on `llm`
        cfg = cast(ExperimentConfig, SimpleNamespace(llm=SimpleNamespace()))
        assert _resolve_backend(cfg) is LLMBackend.OPENAI_BATCH

    def test_unknown_string_raises(self) -> None:
        cfg = self._cfg_with_backend("never_heard_of_it")
        with pytest.raises(ValueError):
            _resolve_backend(cfg)


# ---------------------------------------------------------------------------
# Ollama record persistence helpers
# ---------------------------------------------------------------------------

def _make_context(topic_id: str = "topic-a", gap_idx: int = 0) -> ProposalContext:
    return ProposalContext(
        topic_name="Quantum Computing",
        topic_id=topic_id,
        gap_idx=gap_idx,
        requested_num_prompt_articles=1,
        requested_num_prompt_books=0,
        sampled_articles=[],
        covering_books=[],
        counts=ProposalCounts(num_articles_for_llm=0, num_books_for_llm=0),
    )


class TestOllamaPersistenceHelpers:
    def test_record_serialisation_roundtrip(self) -> None:
        from bookgap.llm.generators.ollama_local import OllamaProposalRecord
        from bookgap.llm.protocols import PromptPair

        proposal = Proposal(title="T", abstract="A")
        record = OllamaProposalRecord(
            proposal=proposal,
            raw_content='{"title": "T", "abstract": "A"}',
            thinking=None,
            prompts=PromptPair(system_prompt="sys", user_prompt="user"),
            context=_make_context(),
            metadata={"model": "test:1b"},
        )
        d = _ollama_record_to_jsonl_dict(record, custom_id="abc")
        assert d["custom_id"] == "abc"
        assert d["proposal"] == {"title": "T", "abstract": "A"}
        assert d["raw_content"] == '{"title": "T", "abstract": "A"}'
        assert d["metadata"] == {"model": "test:1b"}
        # round-trip
        line = json.dumps(d)
        assert json.loads(line) == d

    def test_record_serialisation_with_none_proposal(self) -> None:
        from bookgap.llm.generators.ollama_local import OllamaProposalRecord
        from bookgap.llm.protocols import PromptPair

        record = OllamaProposalRecord(
            proposal=None,
            raw_content="not valid json",
            thinking=None,
            prompts=PromptPair(system_prompt="s", user_prompt="u"),
            context=_make_context(),
            metadata={},
        )
        d = _ollama_record_to_jsonl_dict(record, custom_id="cid")
        assert d["proposal"] is None
        assert d["raw_content"] == "not valid json"

    def test_read_completed_custom_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ollama_results.jsonl"
            # empty/missing returns empty set
            assert _read_completed_custom_ids(p) == set()

            p.write_text(
                json.dumps({"custom_id": "a", "proposal": None}) + "\n"
                + json.dumps({"custom_id": "b", "proposal": {"title": "x", "abstract": "y"}}) + "\n"
                + "\n"
                + "garbage line that's not json\n",
                encoding="utf-8",
            )
            assert _read_completed_custom_ids(p) == {"a", "b"}


# ---------------------------------------------------------------------------
# Backend dispatch end-to-end (both backends share downstream artifacts)
# ---------------------------------------------------------------------------

class _StubSubCloud:
    """Minimal SubCloud for downstream-artifact code in `run_batch_workflow`."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings

    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def index(self):
        idx = faiss.IndexFlatL2(self._embeddings.shape[1])
        vectors = np.ascontiguousarray(self._embeddings, dtype=np.float32)
        cast(Any, idx).add(vectors)
        return idx


class _StubGap:
    def __init__(self, center_idx: int, ambient: SubCloud, landmark: SubCloud) -> None:
        self._dists_idxs = DistsAndIdxs(
            sqdists=np.array([0.0], dtype=np.float32),
            idxs=np.array([center_idx], dtype=np.intp),
        )
        self._ambient = ambient
        self._landmark = landmark

    def get_dists_idxs(self) -> DistsAndIdxs:
        return self._dists_idxs

    def get_ambient(self) -> SubCloud:
        return self._ambient

    def get_landmark(self) -> SubCloud:
        return self._landmark

    def get_covering_book_indices(self):  # pragma: no cover - unused
        return []

    def get_book_to_articles_mapping(self):  # pragma: no cover - unused
        return {}


class _StubAlgorithm:
    """Provides the `compute_gap_sizes_for_embeddings` hook used by
    `_compute_proposal_gap_stats`."""

    def compute_gap_sizes_for_embeddings(self, *, embeddings, **_kwargs):
        return np.full(embeddings.shape[0], 2, dtype=np.int32)


def _make_mock_config(*, backend: LLMBackend, num_gaps: int = 1) -> Any:
    """Build a SimpleNamespace ExperimentConfig stub matching the shape
    expected by `run_batch_workflow` and its helpers."""
    return cast(
        ExperimentConfig,
        SimpleNamespace(
            batch=SimpleNamespace(num_gaps=num_gaps, completion_window="24h"),
            gap=SimpleNamespace(
                radius_fraction=0.5,
                cell_type=SimpleNamespace(value=_StubAlgorithm()),
            ),
            cloud=SimpleNamespace(
                embedder=SimpleNamespace(model_name="test-model"),
                denoiser="NONE",
                drop_k=0,
            ),
            llm=SimpleNamespace(
                backend=backend,
                num_articles=1,
                num_books=1,
                ollama_model="test:1b",
                temperature=0.0,
                ollama_num_ctx=128,
                ollama_num_predict=128,
                model=OpenAIModel.GPT_4O_MINI,
                max_tokens=128,
                use_structured_output=True,
            ),
        ),
    )


def _make_prompt_record(context: ProposalContext) -> "PromptRecord":
    """Build a PromptRecord parallel to a ProposalContext.

    Tests that drive `_run_ollama_local_path` (or `_run_openai_batch_path`)
    directly must pass `prompt_records` parallel to `contexts`; the
    orchestrator does this via `build_prompt_records` in production.
    """
    from bookgap.llm.identifiers import build_gap_proposal_id
    from bookgap.llm.prompt_record import PromptRecord

    return PromptRecord(
        custom_id=build_gap_proposal_id(
            gap_idx=context.gap_idx, topic_id=context.topic_id,
        ),
        topic_id=context.topic_id,
        topic_name=context.topic_name,
        gap_idx=context.gap_idx,
        requested_num_prompt_articles=context.requested_num_prompt_articles,
        requested_num_prompt_books=context.requested_num_prompt_books,
        sampled_articles=[],
        covering_books=[],
        backend="ollama_local",
        model="test:1b",
        generation_params={"temperature": 0.2},
    )


def _make_gap(custom_id_topic: str, gap_idx: int) -> tuple[GapWithContext, ProposalContext]:
    books = cast(SubCloud, _StubSubCloud(np.eye(3, dtype=np.float32)))
    articles = cast(
        SubCloud,
        _StubSubCloud(
            np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], dtype=np.float32)
        ),
    )
    gap = GapWithContext(
        gap=cast(GapProtocol, _StubGap(center_idx=gap_idx, ambient=articles, landmark=books)),
        topic_name="Topic",
        topic_id=custom_id_topic,
        gap_idx=gap_idx,
        books=books,
        articles=articles,
        book_gap_sizes=[1, 2, 3],
        gap_size=1,
        gap_size_percentile=0.5,
        gap_zscore=0.0,
        shapiro_log10_pval=0.0,
    )
    context = _make_context(topic_id=custom_id_topic, gap_idx=gap_idx)
    return gap, context


class TestOpenAIDispatch:
    """Verify the OpenAI Batch path is still taken when backend is OPENAI_BATCH."""

    @patch("bookgap.llm.generators.workflow_functions.generate_all_contexts")
    @patch("bookgap.llm.generators.workflow_functions.generate_gap_plots_for_results")
    @patch("bookgap.llm.generators.workflow_functions.generate_gaps_analysis_csv")
    @patch("bookgap.llm.generators.workflow_functions.save_batch_input_jsonl")
    @patch("bookgap.llm.generators.workflow_functions.submit_batch_job")
    @patch("bookgap.llm.generators.workflow_functions.monitor_and_download_results")
    @patch("bookgap.llm.generators.workflow_functions.generate_batch_requests")
    @patch("bookgap.llm.generators.workflow_functions.build_contexts_for_gaps")
    @patch("bookgap.llm.generators.workflow_functions.compute_gaps_for_config")
    def test_openai_path_taken(
        self,
        mock_compute_gaps,
        mock_build_contexts,
        mock_generate_requests,
        mock_monitor,
        mock_submit,
        mock_save_jsonl,
        mock_csv,
        mock_plots,
        mock_gen_contexts,
    ) -> None:
        gap, context = _make_gap("topic-a", 0)
        mock_compute_gaps.return_value = [gap]
        mock_build_contexts.return_value = [context]
        mock_generate_requests.return_value = [Mock()]
        mock_submit.return_value = "batch-xyz"
        mock_gen_contexts.return_value = []
        mock_plots.return_value = []
        mock_csv.return_value = Path("results.csv")

        proposal = ParsedProposal(
            proposal=Proposal(title="T", abstract="A"),
            custom_id=build_gap_proposal_id(gap_idx=0, topic_id="topic-a"),
            embedding=np.zeros(3, dtype=np.float32),
        )
        mock_monitor.return_value = BatchResults(
            proposals=[proposal],
            metadata=cast(ExperimentConfig, OmegaConf.create({"batch": {"num_gaps": 1}})),
        )

        config = _make_mock_config(backend=LLMBackend.OPENAI_BATCH)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            state = RunStateManager(output_dir)
            result = run_batch_workflow(config, output_dir, state)

        # The OpenAI batch lifecycle was invoked.
        mock_submit.assert_called_once()
        mock_monitor.assert_called_once()
        assert state.state.step == RunStep.COMPLETED
        # And the result has the proposal that monitor returned.
        assert [p.custom_id for p in result.proposals] == [proposal.custom_id]


class TestOllamaDispatch:
    """Verify the Ollama path is taken when backend is OLLAMA_LOCAL and that
    no OpenAI Batch entry points are invoked."""

    def _fake_chat_response(self, title: str, abstract: str):
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps({"title": title, "abstract": abstract}),
                thinking=None,
            ),
            eval_count=10,
            prompt_eval_count=5,
            total_duration=1_000_000,
            eval_duration=900_000,
            done_reason="stop",
        )

    @patch("bookgap.llm.generators.workflow_functions.instantiate")
    @patch("bookgap.llm.generators.workflow_functions.Embedder")
    @patch("bookgap.llm.generators.workflow_functions.generate_all_contexts")
    @patch("bookgap.llm.generators.workflow_functions.generate_gap_plots_for_results")
    @patch("bookgap.llm.generators.workflow_functions.generate_gaps_analysis_csv")
    @patch("bookgap.llm.generators.workflow_functions.submit_batch_job")
    @patch("bookgap.llm.generators.workflow_functions.monitor_and_download_results")
    @patch("bookgap.llm.generators.workflow_functions._compute_proposal_embeddings")
    @patch("bookgap.llm.generators.workflow_functions.build_contexts_for_gaps")
    @patch("bookgap.llm.generators.workflow_functions.compute_gaps_for_config")
    def test_ollama_path_taken(
        self,
        mock_compute_gaps,
        mock_build_contexts,
        mock_embed,
        mock_monitor,
        mock_submit,
        mock_csv,
        mock_plots,
        mock_gen_contexts,
        mock_embedder_cls,
        mock_instantiate,
    ) -> None:
        gap1, ctx1 = _make_gap("topic-a", 0)
        gap2, ctx2 = _make_gap("topic-b", 1)
        mock_compute_gaps.return_value = [gap1, gap2]
        mock_build_contexts.return_value = [ctx1, ctx2]
        mock_gen_contexts.return_value = []
        mock_plots.return_value = []
        mock_csv.return_value = Path("results.csv")
        mock_embed.return_value = np.zeros((2, 3), dtype=np.float32)
        # The streaming-metrics path needs a hot embedder + an instantiated
        # cloud projection; here we patch them out with identity stubs.
        TestStreamingCoverageProgress._patch_hot_embedder(mock_embedder_cls)
        mock_instantiate.return_value = (
            TestStreamingCoverageProgress._identity_cloud_mock()
        )

        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        with patch("bookgap.llm.generators.ollama_local.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = [
                self._fake_chat_response("T1", "A1"),
                self._fake_chat_response("T2", "A2"),
            ]
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                state = RunStateManager(output_dir)
                result = run_batch_workflow(config, output_dir, state)

                # The Ollama backend wrote per-context records as it went.
                results_path = output_dir / "ollama_results.jsonl"
                assert results_path.exists()
                lines = [
                    json.loads(line) for line in
                    results_path.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                assert len(lines) == 2
                custom_ids = {entry["custom_id"] for entry in lines}
                assert custom_ids == {
                    build_gap_proposal_id(gap_idx=0, topic_id="topic-a"),
                    build_gap_proposal_id(gap_idx=1, topic_id="topic-b"),
                }
                assert lines[0]["proposal"] == {"title": "T1", "abstract": "A1"}

        # OpenAI batch entry points were never invoked.
        mock_submit.assert_not_called()
        mock_monitor.assert_not_called()

        # Result has the shape the downstream pipeline needs.
        assert isinstance(result, BatchResults)
        assert len(result.proposals) == 2
        for proposal in result.proposals:
            assert isinstance(proposal, ParsedProposal)
            assert proposal.embedding.shape == (3,)
            assert proposal.embedding.dtype == np.float32

        # The Ollama-backed run still reaches the COMPLETED step.
        assert state.state.step == RunStep.COMPLETED
        # The recorded batch_job_id is the synthetic ollama:<model> tag.
        assert state.state.batch_job_id == "ollama:test:1b"


class TestOllamaResumability:
    """Verify pre-existing on-disk records skip the corresponding
    `ollama.chat` calls so a SIGKILL'd run can resume."""

    @patch("bookgap.llm.generators.workflow_functions.instantiate")
    @patch("bookgap.llm.generators.workflow_functions.Embedder")
    @patch("bookgap.llm.generators.workflow_functions._compute_proposal_embeddings")
    def test_resume_skips_completed(
        self, mock_embed, mock_embedder_cls, mock_instantiate,
    ) -> None:
        mock_embed.return_value = np.zeros((2, 3), dtype=np.float32)
        # Patch the hot embedder + cloud projection used by the streaming
        # metrics path so the test only exercises resume-skip behaviour.
        TestStreamingCoverageProgress._patch_hot_embedder(mock_embedder_cls)
        mock_instantiate.return_value = (
            TestStreamingCoverageProgress._identity_cloud_mock()
        )

        gap1, ctx1 = _make_gap("topic-a", 0)
        gap2, ctx2 = _make_gap("topic-b", 1)
        contexts = [ctx1, ctx2]
        prompt_records = [_make_prompt_record(c) for c in contexts]
        gaps = [gap1, gap2]

        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            # Pre-populate ollama_results.jsonl with the first context's
            # record so the resume path should only call ollama.chat once
            # (for the second context).
            existing_cid = build_gap_proposal_id(gap_idx=0, topic_id="topic-a")
            results_path = output_dir / "ollama_results.jsonl"
            results_path.write_text(
                json.dumps({
                    "custom_id": existing_cid,
                    "proposal": {"title": "Old", "abstract": "Old"},
                    "raw_content": "{...}",
                    "thinking": None,
                    "metadata": {"model": "test:1b"},
                }) + "\n",
                encoding="utf-8",
            )

            state = RunStateManager(output_dir)
            with patch("bookgap.llm.generators.ollama_local.ollama") as mock_ollama:
                mock_ollama.chat.return_value = SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"title": "T2", "abstract": "A2"}),
                        thinking=None,
                    ),
                    eval_count=1,
                    prompt_eval_count=1,
                    total_duration=1,
                    eval_duration=1,
                    done_reason="stop",
                )

                result = _run_ollama_local_path(
                    config=config,
                    output_dir=output_dir,
                    state_manager=state,
                    gaps=gaps,
                    contexts=contexts,
                    prompt_records=prompt_records,
                    context_paths=[],
                )

                # Only one ollama.chat call: the not-yet-completed context.
                assert mock_ollama.chat.call_count == 1

            # Final results file has both records (existing + new).
            lines = [
                json.loads(line) for line in
                results_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            assert len(lines) == 2
            assert {entry["custom_id"] for entry in lines} == {
                existing_cid,
                build_gap_proposal_id(gap_idx=1, topic_id="topic-b"),
            }
            # And both end up in the parsed BatchResults.
            assert len(result.proposals) == 2


class TestStreamingCoverageProgress:
    """The Ollama path emits a `coverage_progress.csv` incrementally so
    operators can `tail -F` headline metrics during a multi-hour run."""

    def _fake_chat_response(self, title: str, abstract: str):
        return SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps({"title": title, "abstract": abstract}),
                thinking=None,
            ),
            eval_count=10,
            prompt_eval_count=5,
            total_duration=1_000_000,
            eval_duration=900_000,
            done_reason="stop",
        )

    @staticmethod
    def _identity_cloud_mock() -> Mock:
        """A Mock cloud whose denoiser_projection is the identity and
        whose normalize_embeddings is False (so the streaming pass uses
        raw embeddings unchanged)."""
        projection = Mock()
        projection.apply.side_effect = lambda x: x
        cloud = Mock()
        cloud.denoiser_projection.return_value = projection
        cloud.normalize_embeddings = False
        return cloud

    @staticmethod
    def _patch_hot_embedder(mock_embedder_cls, embedding_dim: int = 3):
        """Configure the patched Embedder class so its context-manager
        yields an embedder whose ``embed_texts`` returns a deterministic
        unit-direction embedding per call."""

        class _FakeEmbedder:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def embed_texts(self, pairs):
                self.calls += 1
                # Return a single unit vector that does NOT match any gap
                # article exactly; the exact value doesn't matter for
                # schema/row-count tests.
                return np.zeros((len(pairs), embedding_dim), dtype=np.float32)

        instance = _FakeEmbedder()
        mock_embedder_cls.return_value = instance
        return instance

    @patch("bookgap.llm.generators.workflow_functions.instantiate")
    @patch("bookgap.llm.generators.workflow_functions.Embedder")
    @patch("bookgap.llm.generators.workflow_functions._compute_proposal_embeddings")
    def test_streaming_csv_grows_incrementally(
        self,
        mock_embed_post,
        mock_embedder_cls,
        mock_instantiate,
    ) -> None:
        """Each ollama.chat call yields a row in coverage_progress.csv
        before the next call begins. After all calls, the CSV holds N
        data rows + 1 header row."""
        mock_embed_post.return_value = np.zeros((2, 3), dtype=np.float32)
        self._patch_hot_embedder(mock_embedder_cls)
        mock_instantiate.return_value = self._identity_cloud_mock()

        gap1, ctx1 = _make_gap("topic-a", 0)
        gap2, ctx2 = _make_gap("topic-b", 1)
        contexts = [ctx1, ctx2]
        prompt_records = [_make_prompt_record(c) for c in contexts]
        gaps = [gap1, gap2]

        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        # Capture CSV state between each ollama.chat call so we can prove
        # rows arrive incrementally, not in one post-loop dump.
        csv_snapshots: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            csv_path = output_dir / "coverage_progress.csv"

            def capture_after_chat(*args, **kwargs):
                # The streaming row for THIS proposal is written AFTER
                # ollama.chat returns and before the next chat starts.
                # We capture the file state at the moment chat returns;
                # the streaming write happens immediately after.
                resp = self._fake_chat_response(
                    f"T{len(csv_snapshots)+1}", f"A{len(csv_snapshots)+1}",
                )
                # Read the CSV state BEFORE the streaming write for this
                # iteration (so first snapshot may show header-only or
                # be absent before the first write).
                csv_snapshots.append(
                    csv_path.read_text(encoding="utf-8")
                    if csv_path.exists() else ""
                )
                return resp

            state = RunStateManager(output_dir)
            with patch(
                "bookgap.llm.generators.ollama_local.ollama"
            ) as mock_ollama:
                mock_ollama.chat.side_effect = capture_after_chat

                _run_ollama_local_path(
                    config=config,
                    output_dir=output_dir,
                    state_manager=state,
                    gaps=gaps,
                    contexts=contexts,
                    prompt_records=prompt_records,
                    context_paths=[],
                )

            # After the loop, the CSV must exist with header + 2 data rows.
            assert csv_path.exists(), "coverage_progress.csv must be written"
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 3, f"expected 1 header + 2 data, got {lines!r}"

            header = lines[0].split(",")
            assert header == [
                "topic_id",
                "topic_name",
                "gap_size",
                "coverage_pct",
                "proposal_gap_percentile",
                "proposal_title",
            ]

            # Two data rows, one per topic.
            data_rows = [line.split(",") for line in lines[1:]]
            topic_ids = {row[0] for row in data_rows}
            assert topic_ids == {"topic-a", "topic-b"}

            # Incremental writes: at the moment of the second chat call,
            # the first row was already on disk (header + 1 data row).
            # csv_snapshots[1] is what we saw before the second streaming
            # write happened.
            assert len(csv_snapshots) == 2
            after_first_chat = csv_snapshots[1].splitlines()
            assert len(after_first_chat) == 2, (
                f"Expected header + 1 row before second chat completes, "
                f"got {csv_snapshots[1]!r}"
            )

    @patch("bookgap.llm.generators.workflow_functions.instantiate")
    @patch("bookgap.llm.generators.workflow_functions.Embedder")
    @patch("bookgap.llm.generators.workflow_functions._compute_proposal_embeddings")
    def test_streaming_csv_includes_resumed_records(
        self,
        mock_embed_post,
        mock_embedder_cls,
        mock_instantiate,
    ) -> None:
        """When the run resumes from a non-empty ollama_results.jsonl,
        the streaming CSV must include rows for the pre-existing records.
        Otherwise the streaming CSV becomes a partial view of the run."""
        mock_embed_post.return_value = np.zeros((2, 3), dtype=np.float32)
        self._patch_hot_embedder(mock_embedder_cls)
        mock_instantiate.return_value = self._identity_cloud_mock()

        gap1, ctx1 = _make_gap("topic-a", 0)
        gap2, ctx2 = _make_gap("topic-b", 1)
        contexts = [ctx1, ctx2]
        prompt_records = [_make_prompt_record(c) for c in contexts]
        gaps = [gap1, gap2]

        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            # Pre-populate ollama_results.jsonl with a record for
            # topic-a (the run was interrupted after writing it).
            existing_cid = build_gap_proposal_id(
                gap_idx=0, topic_id="topic-a",
            )
            results_path = output_dir / "ollama_results.jsonl"
            results_path.write_text(
                json.dumps({
                    "custom_id": existing_cid,
                    "proposal": {"title": "Resumed", "abstract": "Resumed"},
                    "raw_content": "{...}",
                    "thinking": None,
                    "metadata": {"model": "test:1b"},
                }) + "\n",
                encoding="utf-8",
            )

            state = RunStateManager(output_dir)
            with patch(
                "bookgap.llm.generators.ollama_local.ollama"
            ) as mock_ollama:
                mock_ollama.chat.return_value = self._fake_chat_response(
                    "T2", "A2",
                )

                _run_ollama_local_path(
                    config=config,
                    output_dir=output_dir,
                    state_manager=state,
                    gaps=gaps,
                    contexts=contexts,
                    prompt_records=prompt_records,
                    context_paths=[],
                )

                # The new ollama call was only made for the un-completed
                # context (topic-b).
                assert mock_ollama.chat.call_count == 1

            csv_path = output_dir / "coverage_progress.csv"
            assert csv_path.exists()
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 3, (
                f"Expected header + 2 rows (1 resumed + 1 new), got {lines!r}"
            )

            data_rows = [line.split(",") for line in lines[1:]]
            topic_ids_in_csv = {row[0] for row in data_rows}
            assert topic_ids_in_csv == {"topic-a", "topic-b"}

            # The resumed row's proposal title is the one from the
            # JSONL, not from a fresh ollama call.
            row_by_topic = {row[0]: row for row in data_rows}
            assert row_by_topic["topic-a"][5] == "Resumed"
            assert row_by_topic["topic-b"][5] == "T2"

    @patch("bookgap.llm.generators.workflow_functions.instantiate")
    @patch("bookgap.llm.generators.workflow_functions.Embedder")
    @patch("bookgap.llm.generators.workflow_functions._compute_proposal_embeddings")
    def test_streaming_csv_skips_failed_parses(
        self,
        mock_embed_post,
        mock_embedder_cls,
        mock_instantiate,
    ) -> None:
        """If ``generator.generate`` returns a record whose ``proposal``
        is None (the model emitted nonsense JSON even after retry), the
        streaming pass logs and skips the row for that topic."""
        mock_embed_post.return_value = np.zeros((1, 3), dtype=np.float32)
        self._patch_hot_embedder(mock_embedder_cls)
        mock_instantiate.return_value = self._identity_cloud_mock()

        gap1, ctx1 = _make_gap("topic-a", 0)
        gap2, ctx2 = _make_gap("topic-b", 1)
        contexts = [ctx1, ctx2]
        prompt_records = [_make_prompt_record(c) for c in contexts]
        gaps = [gap1, gap2]

        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            state = RunStateManager(output_dir)
            with patch(
                "bookgap.llm.generators.ollama_local.ollama"
            ) as mock_ollama:
                # First call returns valid JSON; second call returns
                # nonsense that the strict + retry parser cannot parse.
                bad_response = SimpleNamespace(
                    message=SimpleNamespace(
                        content="not valid json at all",
                        thinking=None,
                    ),
                    eval_count=10,
                    prompt_eval_count=5,
                    total_duration=1_000_000,
                    eval_duration=900_000,
                    done_reason="stop",
                )
                mock_ollama.chat.side_effect = [
                    self._fake_chat_response("Good", "Good"),
                    bad_response,
                    bad_response,  # retry
                ]

                _run_ollama_local_path(
                    config=config,
                    output_dir=output_dir,
                    state_manager=state,
                    gaps=gaps,
                    contexts=contexts,
                    prompt_records=prompt_records,
                    context_paths=[],
                )

            csv_path = output_dir / "coverage_progress.csv"
            assert csv_path.exists()
            lines = csv_path.read_text(encoding="utf-8").splitlines()
            # Header + 1 row for the good proposal only.
            assert len(lines) == 2, (
                f"Expected header + 1 row (failed parse skipped), got {lines!r}"
            )
            data_row = lines[1].split(",")
            assert data_row[0] == "topic-a"
            assert data_row[5] == "Good"


class TestStreamOneProposalMetrics:
    """Unit test for the per-iteration streaming helper."""

    def test_returns_row_dict_with_correct_keys(self) -> None:
        """The helper returns a dict with the streaming-row fields."""
        gap, _ctx = _make_gap("topic-x", 0)

        # An identity projection: streaming math operates on raw vectors.
        projection = SimpleNamespace(apply=lambda x: x)

        class _FakeEmbedder:
            def embed_texts(self, pairs):
                # One row matching the first article exactly so
                # coverage is well-defined and coverage_pct = 100%.
                return np.array(
                    [gap.articles.embeddings()[0]], dtype=np.float32,
                )

        proposal = Proposal(title="My Title", abstract="My Abstract")
        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)

        row = _stream_one_proposal_metrics(
            gap=gap,
            proposal=proposal,
            embedder=_FakeEmbedder(),
            projection=projection,
            normalize_embeddings=False,
            config=config,
        )

        assert row is not None
        assert set(row.keys()) == {
            "topic_id",
            "topic_name",
            "gap_size",
            "coverage_pct",
            "proposal_gap_percentile",
            "proposal_title",
        }
        assert row["topic_id"] == "topic-x"
        assert row["proposal_title"] == "My Title"
        assert row["gap_size"] == 1


class TestParseOllamaResults:
    """Direct tests for the Ollama-JSONL → ParsedProposal parser."""

    def test_handles_none_proposal_records(self) -> None:
        config = _make_mock_config(backend=LLMBackend.OLLAMA_LOCAL)
        raw = (
            json.dumps({
                "custom_id": "good",
                "proposal": {"title": "T", "abstract": "A"},
                "raw_content": "...",
                "thinking": None,
                "metadata": {},
            }) + "\n"
            + json.dumps({
                "custom_id": "bad",
                "proposal": None,
                "raw_content": "model emitted nonsense",
                "thinking": None,
                "metadata": {},
            }) + "\n"
        )
        with patch(
            "bookgap.llm.generators.workflow_functions._compute_proposal_embeddings"
        ) as mock_embed:
            mock_embed.return_value = np.zeros((1, 3), dtype=np.float32)
            parsed, stats = _parse_ollama_results_content(raw, config)

        assert len(parsed) == 1
        assert parsed[0].custom_id == "good"
        assert stats["parsed_count"] == 1
        assert stats["parse_failure_count"] == 1
        assert stats["error_count"] == 0
