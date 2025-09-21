"""OpenAI Batch API generator for textbook proposals.

Clean implementation using system/user message format and simple custom IDs.
No dependency on legacy AnalysisConfiguration or CustomIdManager.
"""

import os

from openai.lib._pydantic import to_strict_json_schema

from ..protocols import (
    ProposalContext, BatchRequestData, ChatMessage, ChatCompletionBody,
    ProposalGenerationError, Proposal, PromptPair, OpenAIModel
)
from .batch_generator import BatchProposalGenerator


class OpenAIBatchGenerator(BatchProposalGenerator):
    """OpenAI Batch API generator for textbook proposals.
    
    Uses system/user message format and simple custom IDs.
    """
    
    def __init__(self, 
                 model: OpenAIModel,
                 num_articles: int,
                 num_books: int, 
                 temperature: float,
                 max_tokens: int,
                 use_structured_output: bool):
        """Initialize OpenAI batch generator with individual parameters.
        
        Args:
            model: OpenAI model to use
            num_articles: Number of gap articles per prompt
            num_books: Number of covering books per prompt
            temperature: Sampling temperature
            max_tokens: Token limit for responses
            use_structured_output: Whether to use structured JSON output
            
        Raises:
            ProposalGenerationError: If configuration is invalid
        """
        
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ProposalGenerationError(
                "OPENAI_API_KEY environment variable is required",
                context={"provider": "openai"}
            )
        
        # Store individual parameters
        self.model = model
        self.num_articles = num_articles
        self.num_books = num_books
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_structured_output = use_structured_output
        
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
        temperature = self.temperature if self.model.supports_temperature else None
        
        # Build structured output schema if enabled
        response_format = None
        if self.use_structured_output:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_proposal",
                    "schema": to_strict_json_schema(Proposal),
                    "strict": True
                }
            }
        
        body = ChatCompletionBody(
            model=self.model.value,
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
        """Build system prompt with hardcoded chapter count guidance.
        
        Returns:
            System prompt string defining the AI's role
        """
        return "You are an academic researcher and educator."
    
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
            " provided below, please propose a title, abstract, and table of contents"
            " for a textbook that would prepare students to understand research in this area.\n\n"
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
        
        return "".join(prompt_parts)
