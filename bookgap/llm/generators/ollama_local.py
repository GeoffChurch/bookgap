"""Ollama-based proposal generator for local LLM inference.

Parallels :class:`OpenAIBatchGenerator` but uses a synchronous local call
to an Ollama model (e.g. ``gemma4:31b``) instead of the OpenAI Batch API.
Prompt construction is the orchestrator's responsibility: callers pass
an already-rendered :class:`PromptPair` (typically from
:func:`bookgap.llm.prompt_render.render_prompts`). This keeps the prompt
format DRY across backends and lets the persisted ``PromptRecord`` serve
as the single source of truth for what was actually sent.

Deviations from the OpenAI batch path:

- One-at-a-time synchronous calls (no batch); suitable for offline eval
  runs but not for the production pipeline.
- Optional structured-output enforcement: when ``use_structured_output``
  is True (the default), the :class:`Proposal` JSON schema is passed as
  the chat request's ``format`` so the model is constrained to
  a single ``{title, abstract}`` object — making JSON-array outputs and
  free-form prose impossible. The orchestrator-rendered user prompt
  also carries a JSON-output instruction (see ``prompt_render``) so
  unstructured runs still get the same hint via prose.
- Ollama models that produce a ``thinking`` block (e.g. gemma4) have
  their thoughts captured separately for transparency but only the
  final ``content`` is parsed as the proposal.
- A prompt never reaches the model truncated. Every request sets
  ``truncate: false``, so Ollama refuses a prompt longer than ``num_ctx``
  instead of silently dropping its start, and a reply whose prompt plus
  output overflowed ``num_ctx`` (Ollama then shifts the context, dropping
  the start of the prompt mid-generation) is refused here. Both raise
  :class:`ContextOverflowError` and stop the run: the cause is the
  configured window, which a retry cannot change, and a proposal written
  without the whole prompt is not a proposal for that prompt.

Use case: the K-ball-vs-HDBSCAN comparison experiment that needs many
proposals generated cheaply, or any other offline eval where keeping
generations local matters more than throughput.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import ollama

from ..protocols import (
    PromptPair,
    Proposal,
    ProposalContext,
)


class ContextOverflowError(RuntimeError):
    """A chat request did not fit in Ollama's context window (``num_ctx``)."""


def _ollama_url(path: str) -> str:
    # The variable and default the ollama client itself uses.
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return f"{host if '://' in host else 'http://' + host}{path}"


def _chat(body: dict[str, Any]) -> ollama.ChatResponse:
    """POST /api/chat. Not ``ollama.chat``: the client cannot send ``truncate``."""
    response = httpx.post(_ollama_url("/api/chat"), json=body, timeout=None)
    if response.status_code == 400 and "exceed_context_size_error" in response.text:
        raise ContextOverflowError(
            f"Ollama refused a prompt longer than num_ctx={body['options']['num_ctx']}; "
            f"raise llm.ollama_num_ctx. Server said: {response.text}"
        )
    response.raise_for_status()
    return ollama.ChatResponse.model_validate(response.json())


@dataclass(frozen=True)
class OllamaProposalRecord:
    """Container for one Ollama-generated proposal + its provenance.

    Distinct from :class:`ProposalResult` (which assumes file-on-disk
    artifacts written by the OpenAI batch flow). Used in-memory by the
    experiment runner so callers decide where to persist.
    """

    proposal: Proposal | None  # None if JSON parsing failed
    raw_content: str           # the model's literal `content` field
    thinking: str | None       # the model's `thinking` field (gemma4-style)
    prompts: PromptPair        # exact prompts sent
    context: ProposalContext   # exact context that fed the prompts
    metadata: dict[str, Any] = field(default_factory=dict)


class OllamaProposalGenerator:
    """Synchronous Ollama proposal generator.

    Prompt construction is the orchestrator's responsibility: callers
    pass an already-rendered :class:`PromptPair` (typically from
    :func:`bookgap.llm.prompt_render.render_prompts`). This keeps prompt
    format DRY across backends and lets the persisted ``PromptRecord``
    serve as the single source of truth for what was actually sent.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        use_structured_output: bool = True,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.use_structured_output = use_structured_output
        self._response_format = (
            Proposal.model_json_schema() if use_structured_output else None
        )
        self.max_retries = max_retries

    def generate(
        self, context: ProposalContext, prompts: PromptPair,
    ) -> OllamaProposalRecord:
        """Generate one proposal via Ollama. ``prompts`` is rendered by
        the orchestrator using the shared ``render_prompts`` function.

        On strict-parse failure (anything other than a single JSON
        object, optionally fenced), we re-send the chat request up to
        ``max_retries`` more times — non-zero temperature gives the
        model another draw at a clean response. If all attempts fail
        strict parsing, we leave ``proposal=None`` (the offline recovery
        utility :func:`_try_parse_proposal_lenient` can still extract a
        proposal from the persisted ``raw_content`` if the operator
        opts in). Every prior attempt's raw output and thinking text is
        preserved under ``metadata["prior_attempts"]`` for forensics.
        """
        response = self._chat_once(prompts)
        raw = response.message.content or ""
        thinking = getattr(response.message, "thinking", None)
        proposal = _try_parse_proposal(raw)
        metadata = self._build_metadata(response)

        prior_attempts: list[dict[str, str | None]] = []
        while proposal is None and len(prior_attempts) < self.max_retries:
            prior_attempts.append({"raw_content": raw, "thinking": thinking})
            response = self._chat_once(prompts)
            raw = response.message.content or ""
            thinking = getattr(response.message, "thinking", None)
            proposal = _try_parse_proposal(raw)
            metadata.update(self._build_metadata(response))

        if prior_attempts:
            metadata["retried"] = True
            metadata["prior_attempts"] = prior_attempts

        return OllamaProposalRecord(
            proposal=proposal,
            raw_content=raw,
            thinking=thinking,
            prompts=prompts,
            context=context,
            metadata=metadata,
        )

    def _chat_once(self, prompts: PromptPair):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts.system_prompt},
                {"role": "user", "content": prompts.user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
            "stream": False,
            "truncate": False,
        }
        if self._response_format is not None:
            body["format"] = self._response_format
        response = _chat(body)
        used = response.prompt_eval_count + response.eval_count
        if used > self.num_ctx:
            raise ContextOverflowError(
                f"prompt ({response.prompt_eval_count} tokens) plus output "
                f"({response.eval_count}) overflowed num_ctx={self.num_ctx}, so Ollama "
                "shifted the context and dropped the start of the prompt; raise "
                "llm.ollama_num_ctx or lower llm.ollama_num_predict"
            )
        return response

    def _build_metadata(self, response) -> dict[str, Any]:
        return {
            "model": self.model,
            "implementation": "ollama_local",
            "num_ctx": self.num_ctx,
            "eval_count": response.eval_count,
            "prompt_eval_count": response.prompt_eval_count,
            "total_duration_ns": response.total_duration,
            "eval_duration_ns": response.eval_duration,
            "done_reason": response.done_reason,
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


def _try_parse_proposal(raw: str) -> Proposal | None:
    """Strict JSON → Proposal parsing.

    Accepts only a single JSON object, optionally wrapped in **one**
    enclosing markdown code fence. Rejects lists, multi-block outputs,
    and any output with stray commentary before/after the JSON. The
    intent is to fail loudly when the model deviates from the
    ``{title, abstract}`` contract so that :meth:`OllamaProposalGenerator.generate`
    can do a one-shot retry. For offline recovery from past JSONL files
    that contain such deviations, use :func:`_try_parse_proposal_lenient`.
    """
    raw = raw.strip()
    m = _MARKDOWN_FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return Proposal.model_validate(parsed)
    except Exception:
        return None


def _try_parse_proposal_lenient(raw: str) -> Proposal | None:
    """Lenient JSON → Proposal parsing, for offline recovery only.

    Scans ``raw`` for every top-level balanced JSON block (``{...}``
    or ``[...]``), parses each, and returns the first ``Proposal``
    that validates. For list blocks, returns the first list item that
    validates. Strings (including escaped quotes) are skipped so
    brackets inside string literals don't confuse the depth counter.

    NOT used during fresh generation — that path uses
    :func:`_try_parse_proposal` + one retry. This is reserved for
    recovering data from JSONL files written before the strict + retry
    machinery was in place.
    """
    raw = raw.strip()
    for block in _iter_balanced_json_blocks(raw):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        try:
            return Proposal.model_validate(parsed)
        except Exception:
            pass
        if isinstance(parsed, list):
            for item in parsed:
                try:
                    return Proposal.model_validate(item)
                except Exception:
                    continue
    return None


def _iter_balanced_json_blocks(text: str):
    """Yield every top-level balanced ``{...}`` or ``[...]`` substring.

    Tracks bracket depth (counting both ``{`` and ``[``) while respecting
    JSON string literals — brackets inside a `"..."` string are ignored.
    Backslash escapes inside strings are honored so ``"a\\"b"`` is one string.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif c in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1
