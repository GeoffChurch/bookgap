"""OpenAI Batch API generator for textbook proposals.

Clean implementation using system/user message format and simple custom IDs.
No dependency on legacy AnalysisConfiguration or CustomIdManager.
"""

import os

from openai.lib._pydantic import to_strict_json_schema

from bookgap.configs.config import ExperimentConfig
from bookgap.llm.providers import ProviderType
from ..protocols import (
    ProposalContext, BatchRequestData, ChatMessage, ChatCompletionBody,
    ProposalGenerationError, create_proposal_model, PromptPair
)
from .batch_generator import BatchProposalGenerator


class OpenAIBatchGenerator(BatchProposalGenerator):
    """OpenAI Batch API generator for textbook proposals.
    
    Clean implementation using system/user message format and simple custom IDs.
    No dependency on legacy AnalysisConfiguration or CustomIdManager.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize OpenAI batch generator with configuration validation.
        
        Args:
            config: Complete experiment configuration
            
        Raises:
            ProposalGenerationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate provider type
        if config.llm.provider_type != ProviderType.OPENAI:
            raise ProposalGenerationError(
                f"OpenAI generator requires provider_type=ProviderType.OPENAI, got '{config.llm.provider_type}'",
                context={"provider": str(config.llm.provider_type)}
            )
        
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ProposalGenerationError(
                "OPENAI_API_KEY environment variable is required",
                context={"provider": "openai"}
            )
        
        # Store model configuration
        self.model_enum = config.llm.model_name  # Already an OpenAIModel enum
        self.model = self.model_enum.value  # Get string value for API
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.use_structured = config.llm.use_structured_output
        
    def _build_prompt(self, context: ProposalContext) -> PromptPair:
        """Build system and user prompts separately for native message format.
        
        Args:
            context: The proposal context containing research data
            
        Returns:
            PromptPair with system_prompt and user_prompt
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context)
        return PromptPair(system_prompt=system_prompt, user_prompt=user_prompt)
    
    def _create_request_from_prompt(self, prompts: PromptPair, context: ProposalContext) -> BatchRequestData:
        """Create OpenAI batch API request from system/user prompts.
        
        Args:
            prompts: PromptPair containing system_prompt and user_prompt
            context: The original proposal context for custom ID generation
            
        Returns:
            BatchRequestData ready for OpenAI batch API submission
        """
        
        messages = [
            ChatMessage(role="system", content=prompts.system_prompt),
            ChatMessage(role="user", content=prompts.user_prompt)
        ]
        
        # Build request body with explicit parameters
        temperature = self.temperature if self.model_enum.supports_temperature else None
        
        # Build structured output schema if enabled
        response_format = None
        if self.use_structured:
            proposal_model = create_proposal_model(
                chapters_min=self.config.llm.chapters_min,
                chapters_max=self.config.llm.chapters_max
            )
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "schema": to_strict_json_schema(proposal_model),
                    "strict": True
                }
            }
        
        body = ChatCompletionBody(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            response_format=response_format
        )
        
        # Simple custom ID (no embedded configuration)
        custom_id = f"gap_{context.gap_idx}_topic_{context.topic_id}"
        
        return BatchRequestData(
            custom_id=custom_id,
            body=body
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with role definition.
        
        Returns:
            System prompt string defining the AI's role
        """
        return "You are a publishing expert and academic researcher specializing in textbook development."
    
    def _build_user_prompt(self, context: ProposalContext) -> str:
        """Build user prompt from context with clean formatting.
        
        Args:
            context: The proposal context containing research data
            
        Returns:
            Formatted user prompt string
        """
        prompt_parts = []
        
        # Introduction
        prompt_parts.append(
            "Based on the research articles"
        )
        if context.covering_books:
            prompt_parts.append(" and related books")
        prompt_parts.append(
            " provided below, please propose a title, abstract, and table of contents "
            "for a textbook that would prepare students to understand research in this area.\n\n"
        )
        
        # Sample articles section
        prompt_parts.append("## Sample Articles\n\n")
        for i, article in enumerate(context.sampled_articles, 1):
            prompt_parts.append(f"**Article {i}**: {article.display_name}\n")
            prompt_parts.append(f"\t*Abstract*: {article.abstract}\n\n")
        
        # Related books section (if any)
        if context.covering_books:
            prompt_parts.append("## Related Existing Books\n\n")
            for i, book in enumerate(context.covering_books, 1):
                prompt_parts.append(f"**Book {i}**: {book.display_name}\n")
                prompt_parts.append(f"\t*Abstract*: {book.abstract}\n\n")
        
        # Guidelines
        prompt_parts.append("## Guidelines\n\n")
        min_chapters = self.config.llm.chapters_min
        max_chapters = self.config.llm.chapters_max
        prompt_parts.append(f"- Include {min_chapters}-{max_chapters} chapters.\n")
        prompt_parts.append("- Build up systematically from basic to advanced concepts.\n")
        prompt_parts.append("- Each chapter should have a clear title and description of concepts covered.\n")
        prompt_parts.append("- Focus on preparing students for understanding current research in this field.\n")
        
        return "".join(prompt_parts)
