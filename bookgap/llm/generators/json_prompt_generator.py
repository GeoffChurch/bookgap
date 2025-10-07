"""
JSON prompt-based generator for textbook proposals.

This module implements the JSONPromptProposalGenerator using verbose prompts
that strongly emphasize JSON-only output, then validates the response with
strict JSON parsing and Pydantic validation.
"""

import json
import os
import re

from openai import OpenAI
from pydantic import ValidationError

from bookgap.config_manager import AnalysisConfiguration
from bookgap.gap_directory_manager import GapDirectoryManager
from bookgap.util import create_unicode_safe_filename, LOGGER
from ..protocols import ProposalContext, ProposalResult, ProposalCounts, ProposalMetadata, Proposal


class JSONPromptProposalGenerator:
    """Generator using verbose prompting with strict JSON parsing and validation."""
    
    def __init__(self, analysis_config: AnalysisConfiguration):
        """Initialize the JSON prompt generator with configuration validation.
        
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
        """Generate a textbook proposal using verbose JSON prompting.
        
        Args:
            context: The proposal context containing gap analysis and configuration
            
        Returns:
            ProposalResult with generated proposal and metadata
        """
        # Build the prompt with JSON format specification
        prompt = self._build_prompt_with_json_format(context)
        
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
        
        # Call OpenAI API without structured outputs
        LOGGER.info(f"Calling OpenAI API for gap {context.gap_idx} with JSON prompting")
        
        response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a publishing expert and academic researcher. You must respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        
        # Extract the raw response text
        raw_content = response.choices[0].message.content
        
        if not raw_content:
            raise ValueError("OpenAI returned empty response content")
        
        # Write raw response for audit (store as string, not object)
        with open(raw_response_path, 'w', encoding='utf-8') as f:
            json.dump({"raw_text": raw_content}, f, indent=2, ensure_ascii=False)
        
        # Clean and parse JSON
        cleaned_json = self._clean_json_content(raw_content)
        proposal_dict = json.loads(cleaned_json)
        
        # Validate with Pydantic
        proposal = Proposal.model_validate(proposal_dict)
        
        # Validate chapter count is within bounds
        num_chapters = len(proposal.chapters)
        if not (self.llm_config.chapters_min <= num_chapters <= self.llm_config.chapters_max):
            raise ValueError(f"Chapter count {num_chapters} outside bounds [{self.llm_config.chapters_min}, {self.llm_config.chapters_max}]")
        
        # Write validated JSON proposal
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(proposal.model_dump(), f, indent=2, ensure_ascii=False)
        
        # Prepare metadata
        usage = response.usage.model_dump() if response.usage else None
        request_id = response.id
        
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
                implementation="prompt",
                request_id=request_id,
                usage=usage
            )
        )
    
    def _build_prompt_with_json_format(self, context: ProposalContext) -> str:
        """Build the prompt with explicit JSON format requirements.
        
        This version includes the "Required Output Format" section to strongly
        emphasize JSON-only output.
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
        
        # Required Output Format section (critical for JSON prompting)
        prompt_parts.append("## Required Output Format\n\n")
        prompt_parts.append("Format your response as valid JSON with EXACTLY the following structure. ")
        prompt_parts.append("Do not add any text before or after this JSON:\n\n")
        prompt_parts.append("{\n")
        prompt_parts.append('  "title": "<Proposed Title>",\n')
        prompt_parts.append('  "abstract": "<Abstract>",\n')
        prompt_parts.append('  "chapters": [\n')
        prompt_parts.append('    {\n')
        prompt_parts.append('      "chapter_title": "<Chapter Name>",\n')
        prompt_parts.append('      "chapter_description": "<Description of concepts covered (1-2 sentences)>"\n')
        prompt_parts.append('    },\n')
        prompt_parts.append('    {\n')
        prompt_parts.append('      "chapter_title": "<Chapter Name>",\n')
        prompt_parts.append('      "chapter_description": "<Description of concepts covered (1-2 sentences)>"\n')
        prompt_parts.append('    }\n')
        min_chapters = self.llm_config.chapters_min
        max_chapters = self.llm_config.chapters_max
        prompt_parts.append(f'    // ... continue for {min_chapters}-{max_chapters} chapters\n')
        prompt_parts.append('  ]\n')
        prompt_parts.append('}\n\n')
        
        # Guidelines
        prompt_parts.append("## Guidelines\n\n")
        prompt_parts.append("- Use the EXACT JSON structure shown above.\n")
        prompt_parts.append("- Replace ALL angle-bracketed placeholders (e.g., <Proposed Title>) with your content.\n")
        prompt_parts.append(f"- Include {min_chapters}-{max_chapters} chapters in the chapters array.\n")
        prompt_parts.append("- Build up systematically from basic to advanced concepts.\n")
        prompt_parts.append("- CRITICAL: Respond with ONLY valid JSON. No additional text before or after.\n")
        
        return "".join(prompt_parts)
    
    def _clean_json_content(self, raw_content: str) -> str:
        """Clean the raw response to extract valid JSON.
        
        This method attempts to extract JSON from the response, handling cases
        where the LLM may have added extra text despite instructions.
        """
        # Strip whitespace
        content = raw_content.strip()
        
        # Try to find JSON block
        # Look for content between first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            raise ValueError("No JSON object found in response")
        
        # Extract JSON portion
        json_content = content[start_idx:end_idx + 1]
        
        return json_content
