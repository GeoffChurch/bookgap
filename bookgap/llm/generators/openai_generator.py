"""
OpenAI Structured Outputs generator for textbook proposals.

This module implements the OpenAIProposalGenerator using OpenAI's Structured Outputs
feature with Pydantic models to ensure guaranteed JSON schema compliance.
Works in both direct API and Batch API modes.
"""

import json
import os

from openai import OpenAI

from bookgap.config_manager import AnalysisConfiguration
from bookgap.gap_directory_manager import GapDirectoryManager
from bookgap.util import create_unicode_safe_filename, LOGGER
from ..protocols import ProposalContext, ProposalResult, Proposal, ProposalCounts, ProposalMetadata


class OpenAIProposalGenerator:
    """Generator using OpenAI Structured Outputs for guaranteed JSON schema compliance."""
    
    def __init__(self, analysis_config: AnalysisConfiguration):
        """Initialize the OpenAI generator with configuration validation.
        
        Args:
            analysis_config: Analysis configuration containing LLM settings
        """
        self.analysis_config = analysis_config
        self.llm_config = analysis_config.llm_config
        
        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, context: ProposalContext) -> ProposalResult:
        """Generate a textbook proposal using OpenAI Structured Outputs.
        
        Args:
            context: The proposal context containing gap analysis and configuration
            
        Returns:
            ProposalResult with generated proposal and metadata
            
        Raises:
            ProposalGenerationError: If generation fails for any reason
        """
        # Build the prompt
        prompt = self._build_prompt(context)
        
        # Get output directory
        prompts_llm_dir = GapDirectoryManager.get_prompts_llm_dir(
            self.analysis_config, context.gap_idx
        )
        
        # Generate file paths
        base_name = f"{create_unicode_safe_filename(context.topic_name)}_id={context.topic_id}"
        prompt_path = prompts_llm_dir / f"{base_name}_prompt.txt"
        json_path = prompts_llm_dir / f"{base_name}_proposal.json"
        raw_response_path = prompts_llm_dir / f"{base_name}_response_raw.json"
        
        # Write the prompt for audit
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Call OpenAI API with structured outputs
        LOGGER.info(f"Calling OpenAI API for gap {context.gap_idx} with structured outputs")
        
        response = self.client.beta.chat.completions.parse(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a publishing expert and academic researcher."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format=Proposal,
            max_completion_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature if not self.llm_config.model.startswith("gpt-5") else None
        )
        
        # Extract the parsed proposal
        proposal = response.choices[0].message.parsed
        
        if proposal is None:
            raise ValueError("OpenAI returned None for parsed proposal")
        
        # Validate chapter count is within bounds
        num_chapters = len(proposal.chapters)
        if not (self.llm_config.chapters_min <= num_chapters <= self.llm_config.chapters_max):
            raise ValueError(f"Chapter count {num_chapters} outside bounds [{self.llm_config.chapters_min}, {self.llm_config.chapters_max}]")
        
        # Write validated JSON proposal
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(proposal.model_dump(), f, indent=2, ensure_ascii=False)
        
        # Write raw response for audit
        with open(raw_response_path, 'w', encoding='utf-8') as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
        
        usage = response.usage.model_dump() if response.usage else None
        
        LOGGER.info(f"Generated proposal for gap {context.gap_idx} with {num_chapters} chapters")
        
        return ProposalResult(
            json_path=json_path,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            counts=ProposalCounts(
                num_articles_for_llm=context.counts.num_articles_for_llm,
                num_books_for_llm=context.counts.num_books_for_llm
            ),
            metadata=ProposalMetadata(
                model=self.llm_config.model,
                implementation="structured",
                request_id=response.id,
                usage=usage
            )
        )
    
    def _build_prompt(self, context: ProposalContext) -> str:
        """Build the prompt for the LLM.
        
        For structured outputs, we don't need the JSON format specification
        since the schema is enforced by the response_format parameter.
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
            title = article.display_name
            abstract = article.abstract
            prompt_parts.append(f"**Article {i}**: {title}\n")
            prompt_parts.append(f"\t*Abstract*: {abstract}\n\n")
        
        # Related books section (if any)
        if context.covering_books:
            prompt_parts.append("## Related Existing Books\n\n")
            for i, book in enumerate(context.covering_books, 1):
                title = book.display_name
                abstract = book.abstract
                prompt_parts.append(f"**Book {i}**: {title}\n")
                prompt_parts.append(f"\t*Abstract*: {abstract}\n\n")
        
        # Guidelines
        prompt_parts.append("## Guidelines\n\n")
        min_chapters = self.llm_config.chapters_min
        max_chapters = self.llm_config.chapters_max
        prompt_parts.append(f"- Include {min_chapters}-{max_chapters} chapters.\n")
        prompt_parts.append("- Build up systematically from basic to advanced concepts.\n")
        prompt_parts.append("- Each chapter should have a clear title and description of concepts covered.\n")
        
        return "".join(prompt_parts)
