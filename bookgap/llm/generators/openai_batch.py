"""OpenAI Batch API request builder.

Prompt construction is the orchestrator's responsibility — this class
just wraps a (ProposalContext, PromptPair) pair into the OpenAI batch
API's JSONL request format.
"""
import os

from openai.lib._pydantic import to_strict_json_schema

from ..protocols import (
    BatchRequestData,
    ChatCompletionBody,
    ChatMessage,
    OpenAIModel,
    Proposal,
    ProposalContext,
    PromptPair,
)
from ..identifiers import build_gap_proposal_id


class OpenAIBatchGenerator:
    """Builds OpenAI batch-API request bodies from rendered prompts."""

    def __init__(
        self,
        *,
        model: OpenAIModel,
        temperature: float,
        max_tokens: int,
        use_structured_output: bool,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_structured_output = use_structured_output
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def build_requests(
        self, pairs: list[tuple[ProposalContext, PromptPair]],
    ) -> list[BatchRequestData]:
        return [self._make_request(ctx, prompts) for ctx, prompts in pairs]

    def _make_request(
        self, context: ProposalContext, prompts: PromptPair,
    ) -> BatchRequestData:
        messages = [
            ChatMessage(role="system", content=prompts.system_prompt),
            ChatMessage(role="user", content=prompts.user_prompt),
        ]
        temperature = self.temperature if self.model.supports_temperature else None
        response_format = None
        if self.use_structured_output:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "schema": to_strict_json_schema(Proposal),
                    "strict": True,
                },
            }
        body = ChatCompletionBody(
            model=self.model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        custom_id = build_gap_proposal_id(
            gap_idx=context.gap_idx, topic_id=context.topic_id,
        )
        return BatchRequestData(custom_id=custom_id, body=body)
