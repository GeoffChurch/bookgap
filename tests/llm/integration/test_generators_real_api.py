"""
Integration tests for LLM proposal generators with real OpenAI API.

These tests require OPENAI_API_KEY to be set and make actual API calls.
They are marked with @pytest.mark.integration and excluded by default.
Results are saved in organized directories for inspection.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nextbook.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from nextbook.llm.llm_implementation import Implementation
from nextbook.types import CellType
from nextbook.llm.protocols import ProposalContext
from nextbook.llm.generators.openai_generator import OpenAIProposalGenerator
from nextbook.llm.generators.json_prompt_generator import JSONPromptProposalGenerator


# Sample data for testing
SAMPLE_ARTICLES = [
    {
        'display_name': 'Deep Learning for Computer Vision',
        'abstract': 'This paper presents novel approaches to computer vision using deep neural networks.'
    },
    {
        'display_name': 'Attention Mechanisms in Neural Networks',
        'abstract': 'We explore attention mechanisms and their applications in various neural network architectures.'
    }
]

SAMPLE_BOOKS = [
    {
        'display_name': 'Pattern Recognition and Machine Learning',
        'abstract': 'A comprehensive introduction to pattern recognition and machine learning techniques.'
    }
]

SAMPLE_CONTEXT = ProposalContext(
    topic_name="Machine Learning",
    topic_id="ml_001",
    gap_idx=0,
    requested_num_prompt_articles=2,
    requested_num_prompt_books=1,
    sampled_articles=SAMPLE_ARTICLES,
    covering_books=SAMPLE_BOOKS,
    counts={
        "num_articles_for_llm": 2,
        "num_books_for_llm": 1
    }
)


def create_test_config() -> AnalysisConfiguration:
    """Create a test analysis configuration."""
    gap_config = GapConfiguration(
        denoiser="NONE",  # Use valid denoiser name
        drop_k=0,
        radius_fraction=0.5,
        cell_type=CellType.KBALL
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
        provider="openai",
        model="gpt-5",
        temperature=0.2,
        max_tokens=2000,  # Should be sufficient for 3-4 chapters
        chapters_min=3,
        chapters_max=4,
        requested_num_prompt_articles=2,
        requested_num_prompt_books=1
    )
    
    return AnalysisConfiguration(
        gap_config=gap_config,
        llm_config=llm_config
    )


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
class TestGeneratorsWithRealAPI:
    """Integration tests with real OpenAI API (requires API key)."""
    
    def test_openai_generator_real_api(self):
        """Test OpenAI generator with real API call."""
        config = create_test_config()
        generator = OpenAIProposalGenerator(config)
        
        # Create test results directory in organized location
        test_results_dir = Path(__file__).parent / "results" / "openai"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with patch('nextbook.llm.generators.openai_generator.GapDirectoryManager.get_prompts_llm_dir', return_value=test_results_dir):
                result = generator.generate(SAMPLE_CONTEXT)
                
                # Verify result structure
                assert result.json_path.exists()
                assert result.prompt_path.exists()
                if result.raw_response_path:  # May be None for structured outputs
                    assert result.raw_response_path.exists()
                assert result.counts.num_articles_for_llm == SAMPLE_CONTEXT.counts["num_articles_for_llm"]
                assert result.counts.num_books_for_llm == SAMPLE_CONTEXT.counts["num_books_for_llm"]
                assert result.metadata.provider == "openai"
                assert result.metadata.implementation == "structured"
                
                # Verify JSON content
                with open(result.json_path) as f:
                    proposal_data = json.load(f)
                    assert "title" in proposal_data
                    assert "abstract" in proposal_data
                    assert "chapters" in proposal_data
                    assert 3 <= len(proposal_data["chapters"]) <= 4  # Match our test configuration
                    
            # Test passed - move to success location
            success_dir = test_results_dir.parent / "openai_last_success"
            if success_dir.exists():
                import shutil
                shutil.rmtree(success_dir)
            test_results_dir.rename(success_dir)
                
        except Exception as e:
            # Test failed - move to failure location for inspection
            failure_dir = test_results_dir.parent / "openai_last_failure"
            if failure_dir.exists():
                import shutil
                shutil.rmtree(failure_dir)
            test_results_dir.rename(failure_dir)
            print(f"\nIntegration test failed. Results saved in: {failure_dir}")
            raise  # Re-raise the original exception
    
    def test_json_prompt_generator_real_api(self):
        """Test JSON prompt generator with real API call."""
        gap_config = GapConfiguration(
            denoiser="NONE",
            drop_k=0,
            radius_fraction=0.5,
            cell_type=CellType.KBALL
        )
        
        llm_config = LLMConfiguration(
            implementation=Implementation.PROMPT,
            provider="openai",
            model="gpt-4",
            temperature=0.2,
            max_tokens=2000,
            chapters_min=10,
            chapters_max=15,
            requested_num_prompt_articles=2,
            requested_num_prompt_books=1
        )
        
        config = AnalysisConfiguration.from_components(gap_config, llm_config)
        generator = JSONPromptProposalGenerator(config)
        
        # Create test results directory in organized location
        test_results_dir = Path(__file__).parent / "results" / "json_prompt"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with patch('nextbook.llm.generators.json_prompt_generator.GapDirectoryManager.get_prompts_llm_dir', return_value=test_results_dir):
                result = generator.generate(SAMPLE_CONTEXT)
                
                # Verify result structure
                assert result.json_path.exists()
                assert result.prompt_path.exists()
                if result.raw_response_path:  # May be None
                    assert result.raw_response_path.exists()
                assert result.counts.num_articles_for_llm == SAMPLE_CONTEXT.counts["num_articles_for_llm"]
                assert result.counts.num_books_for_llm == SAMPLE_CONTEXT.counts["num_books_for_llm"]
                assert result.metadata.provider == "openai"
                assert result.metadata.implementation == "prompt"
                
                # Verify JSON content
                with open(result.json_path) as f:
                    proposal_data = json.load(f)
                    assert "title" in proposal_data
                    assert "abstract" in proposal_data
                    assert "chapters" in proposal_data
                    assert len(proposal_data["chapters"]) >= 10
                    
            # Test passed - move to success location
            success_dir = test_results_dir.parent / "json_prompt_last_success"
            if success_dir.exists():
                import shutil
                shutil.rmtree(success_dir)
            test_results_dir.rename(success_dir)
                
        except Exception as e:
            # Test failed - move to failure location for inspection
            failure_dir = test_results_dir.parent / "json_prompt_last_failure"
            if failure_dir.exists():
                import shutil
                shutil.rmtree(failure_dir)
            test_results_dir.rename(failure_dir)
            print(f"\nJSON prompt integration test failed. Results saved in: {failure_dir}")
            raise  # Re-raise the original exception
