"""
Tests for the LLM proposal generators.

These tests validate both OpenAI generator implementations but require
OPENAI_API_KEY to be set. They are designed to work with actual API calls
to ensure real-world compatibility.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from bookgap.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from bookgap.llm.llm_implementation import Implementation
from bookgap.types import CellType
from bookgap.llm.protocols import ProposalContext, ProposalGenerationError
from bookgap.llm.generators.openai_generator import OpenAIProposalGenerator
from bookgap.llm.generators.json_prompt_generator import JSONPromptProposalGenerator


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
    counts={"num_articles_for_llm": 2, "num_books_for_llm": 1}
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


def create_test_config_with_invalid_provider() -> AnalysisConfiguration:
    """Create a test config with invalid provider."""
    gap_config = GapConfiguration(
        denoiser="NONE",
        drop_k=0,
        radius_fraction=0.5,
        cell_type=CellType.KBALL
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
        provider="invalid",  # Invalid provider
        model="gpt-4",
        temperature=0.2,
        max_tokens=2000,
        chapters_min=10,
        chapters_max=15,
        requested_num_prompt_articles=2,
        requested_num_prompt_books=1
    )
    
    return AnalysisConfiguration(
        gap_config=gap_config,
        llm_config=llm_config
    )


class TestOpenAIProposalGenerator:
    """Tests for OpenAIProposalGenerator."""
    
    def test_init_requires_openai_provider(self):
        """Test that generator requires OpenAI provider."""
        with pytest.raises(ValueError, match="Only.*openai.*provider.*supported"):
            create_test_config_with_invalid_provider()
    
    def test_init_requires_api_key(self):
        """Test that generator requires OPENAI_API_KEY."""
        config = create_test_config()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ProposalGenerationError, match="OPENAI_API_KEY environment variable is required"):
                OpenAIProposalGenerator(config)
    
    def test_init_handles_client_creation_failure(self):
        """Test that generator handles OpenAI client creation failures."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.openai_generator.OpenAI', side_effect=Exception("Client error")):
                with pytest.raises(ProposalGenerationError, match="Failed to initialize OpenAI client"):
                    OpenAIProposalGenerator(config)
    
    def test_build_prompt_structure(self):
        """Test that the prompt is built correctly."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.openai_generator.OpenAI'):
                generator = OpenAIProposalGenerator(config)
                prompt = generator._build_prompt(SAMPLE_CONTEXT)
                
                # Check that prompt contains expected sections
                assert "Sample Articles" in prompt
                assert "Related Existing Books" in prompt
                assert "Guidelines" in prompt
                assert "Deep Learning for Computer Vision" in prompt
                assert "Pattern Recognition and Machine Learning" in prompt
                assert "3-4 chapters" in prompt  # Updated to match create_test_config()
    
    @patch('bookgap.llm.generators.openai_generator.GapDirectoryManager.get_prompts_llm_dir')
    def test_generate_handles_api_failure(self, mock_get_dir):
        """Test that generator handles API failures gracefully."""
        config = create_test_config()
        mock_get_dir.return_value = Path(tempfile.mkdtemp())
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")
            
            with patch('bookgap.llm.generators.openai_generator.OpenAI', return_value=mock_client):
                generator = OpenAIProposalGenerator(config)
                
                with pytest.raises(ProposalGenerationError, match="Unexpected error during proposal generation"):
                    generator.generate(SAMPLE_CONTEXT)
    
    @patch('bookgap.llm.generators.openai_generator.GapDirectoryManager.get_prompts_llm_dir')
    def test_generate_validates_chapter_count(self, mock_get_dir):
        """Test that generator validates chapter count bounds."""
        config = create_test_config()
        mock_get_dir.return_value = Path(tempfile.mkdtemp())
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Mock API response with too few chapters
            mock_proposal = Mock()
            mock_proposal.chapters = [Mock()] * 5  # Only 5 chapters, need 10-15
            mock_proposal.model_dump.return_value = {"title": "Test", "abstract": "Test", "chapters": []}
            
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.parsed = mock_proposal
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.id = "test_id"
            mock_response.model_dump.return_value = {}
            
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            
            with patch('bookgap.llm.generators.openai_generator.OpenAI', return_value=mock_client):
                generator = OpenAIProposalGenerator(config)
                
                with pytest.raises(ProposalGenerationError, match="Chapter count 5 outside bounds"):
                    generator.generate(SAMPLE_CONTEXT)


class TestJSONPromptProposalGenerator:
    """Tests for JSONPromptProposalGenerator."""
    
    def test_init_requires_openai_provider(self):
        """Test that generator requires OpenAI provider."""
        with pytest.raises(ValueError, match="Only.*openai.*provider.*supported"):
            create_test_config_with_invalid_provider()
    
    def test_init_requires_api_key(self):
        """Test that generator requires OPENAI_API_KEY."""
        config = create_test_config()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ProposalGenerationError, match="OPENAI_API_KEY environment variable is required"):
                JSONPromptProposalGenerator(config)
    
    def test_build_prompt_includes_json_format(self):
        """Test that the prompt includes JSON format specification."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI'):
                generator = JSONPromptProposalGenerator(config)
                prompt = generator._build_prompt_with_json_format(SAMPLE_CONTEXT)
                
                # Check that prompt contains JSON format specification
                assert "Required Output Format" in prompt
                assert '"title":' in prompt
                assert '"abstract":' in prompt
                assert '"chapters":' in prompt
                assert "EXACT JSON structure" in prompt
                assert "ONLY valid JSON" in prompt
    
    def test_clean_json_content_removes_comments(self):
        """Test that JSON cleaning removes comments."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI'):
                generator = JSONPromptProposalGenerator(config)
                
                raw_json = '''
                {
                  "title": "Test Title",
                  // This is a comment
                  "abstract": "Test abstract",
                  "chapters": []
                }
                '''
                
                cleaned = generator._clean_json_content(raw_json)
                assert "//" not in cleaned
                assert "Test Title" in cleaned
    
    def test_clean_json_content_extracts_json_from_text(self):
        """Test that JSON cleaning extracts JSON from surrounding text."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI'):
                generator = JSONPromptProposalGenerator(config)
                
                raw_content = '''
                Here is your JSON response:
                
                {
                  "title": "Test Title",
                  "abstract": "Test abstract",
                  "chapters": []
                }
                
                I hope this helps!
                '''
                
                cleaned = generator._clean_json_content(raw_content)
                assert cleaned.strip().startswith('{')
                assert cleaned.strip().endswith('}')
                assert "Here is your JSON" not in cleaned
                assert "I hope this helps" not in cleaned
    
    def test_clean_json_content_handles_missing_json(self):
        """Test that JSON cleaning handles content without JSON."""
        config = create_test_config()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI'):
                generator = JSONPromptProposalGenerator(config)
                
                with pytest.raises(ProposalGenerationError, match="No JSON object found"):
                    generator._clean_json_content("This is just text without JSON")
    
    @patch('bookgap.llm.generators.json_prompt_generator.GapDirectoryManager.get_prompts_llm_dir')
    def test_generate_handles_invalid_json_response(self, mock_get_dir):
        """Test that generator handles invalid JSON responses."""
        config = create_test_config()
        mock_get_dir.return_value = Path(tempfile.mkdtemp())
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Mock API response with invalid JSON
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Invalid JSON content"
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI', return_value=mock_client):
                generator = JSONPromptProposalGenerator(config)
                
                with pytest.raises(ProposalGenerationError, match="No JSON object found"):
                    generator.generate(SAMPLE_CONTEXT)
    
    @patch('bookgap.llm.generators.json_prompt_generator.GapDirectoryManager.get_prompts_llm_dir')
    def test_generate_handles_malformed_json(self, mock_get_dir):
        """Test that generator handles malformed JSON."""
        config = create_test_config()
        mock_get_dir.return_value = Path(tempfile.mkdtemp())
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Mock API response with malformed JSON
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = '{"title": "Test", "abstract":}'  # Invalid JSON
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            
            with patch('bookgap.llm.generators.json_prompt_generator.OpenAI', return_value=mock_client):
                generator = JSONPromptProposalGenerator(config)
                
                with pytest.raises(ProposalGenerationError, match="Failed to parse JSON response"):
                    generator.generate(SAMPLE_CONTEXT)
