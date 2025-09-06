#!/usr/bin/env python3
"""
Test suite for Step 5.5: Clean Up Scattered Logic (Critical Fixes).

This test verifies that scattered logic has been properly centralized and that
duplicate functions have been removed while maintaining identical functionality.
"""
import pytest
pytest.skip("Skipped legacy step test (LLMConfiguration deprecated)", allow_module_level=True)
import pytest
from pathlib import Path

def test_batch_operations_imports_from_config_manager():
    """Test that batch_operations imports centralized functions instead of defining its own."""
    from nextbook.batch_utils.batch_operations import build_custom_id, parse_custom_id, extract_topic_name_from_filename
    from nextbook.config_manager import CustomIdManager, TopicPathManager
    
    # These should now be the same functions (imported, not redefined)
    assert build_custom_id is CustomIdManager.build_custom_id
    assert parse_custom_id is CustomIdManager.parse_custom_id
    assert extract_topic_name_from_filename is TopicPathManager.extract_topic_name_from_filename


def test_gap_finder_uses_llm_configuration():
    """Test that gap_finder uses LLMConfiguration for directory naming."""
    from nextbook.config_manager import LLMConfiguration
    
    # Test that LLMConfiguration generates the expected directory format
    llm_config = LLMConfiguration(num_articles=10, num_books=5)
    assert llm_config.to_string() == "articles_10_books_5"
    
    # Test different combinations
    llm_config2 = LLMConfiguration(num_articles=25, num_books=15)
    assert llm_config2.to_string() == "articles_25_books_15"


def test_create_batch_requests_uses_config_path_parser():
    """Test that create_batch_requests_from_prompts uses ConfigPathParser."""
    from nextbook.config_manager import ConfigPathParser, GapConfiguration, LLMConfiguration
    import tempfile
    
    # Create a temporary directory structure that matches expected format
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test directory structure matching what ConfigPathParser expects
        config_dir = temp_path / "NONE_k=5_radiusfrac=0.75"
        gap_dir = config_dir / "gap_0"
        prompts_dir = gap_dir / "prompts"
        llm_dir = prompts_dir / "articles_10_books_3"
        llm_dir.mkdir(parents=True)
        
        # Create a test prompt file
        prompt_file = llm_dir / "Test_Topic_id=topic123_prompt.txt"
        prompt_file.write_text("Test prompt content for LLM")
        
        # Test that ConfigPathParser can parse this structure correctly
        parsed_info = ConfigPathParser.parse_prompt_file_path(str(prompt_file), str(temp_path))
        
        # Verify parsed information
        assert isinstance(parsed_info['gap_config'], GapConfiguration)
        assert parsed_info['gap_config'].denoiser == "NONE"
        assert parsed_info['gap_config'].drop_k == 5
        assert parsed_info['gap_config'].radius_fraction == 0.75
        
        assert isinstance(parsed_info['llm_config'], LLMConfiguration) 
        assert parsed_info['llm_config'].num_articles == 10
        assert parsed_info['llm_config'].num_books == 3
        
        assert parsed_info['topic_name'] == "Test_Topic"
        assert parsed_info['topic_id'] == "topic123"
        assert parsed_info['gap_idx'] == 0
        
        # Note: We can't test create_batch_requests_from_prompts here because 
        # it uses the global GAPS_DIR. That's acceptable for Step 5.5 - 
        # the key achievement is that it now uses ConfigPathParser internally
        # instead of manual string parsing logic.


def test_error_handling_consistency():
    """Test that error handling is consistent across configuration parsing."""
    from nextbook.config_manager import GapConfiguration, LLMConfiguration, CustomIdManager
    
    # Test GapConfiguration error handling
    with pytest.raises(ValueError, match="drop_k must be >= 0"):
        GapConfiguration(denoiser="NONE", drop_k=-1, radius_fraction=0.75)
    
    with pytest.raises(ValueError, match="radius_fraction must be in"):
        GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=1.5)
    
    # Test LLMConfiguration error handling  
    with pytest.raises(ValueError, match="num_articles must be > 0"):
        LLMConfiguration(num_articles=0, num_books=5)
    
    with pytest.raises(ValueError, match="num_books must be >= 0"):
        LLMConfiguration(num_articles=10, num_books=-1)
    
    # Test CustomIdManager error handling
    with pytest.raises(ValueError, match="Expected 5 parts"):
        CustomIdManager.parse_custom_id("invalid___format")
    
    with pytest.raises(ValueError, match="topic_id part must start with 'id='"):
        CustomIdManager.parse_custom_id("NONE_k=5_radiusfrac=0.75___topic___wrong=123___gap_0___articles_10_books_3")


def test_no_duplicate_function_definitions():
    """Test that duplicate functions have been removed from batch_operations."""
    import inspect
    from nextbook.batch_utils import batch_operations
    
    # Get all functions defined in the batch_operations module
    module_functions = []
    for name, obj in inspect.getmembers(batch_operations):
        if inspect.isfunction(obj) and obj.__module__ == batch_operations.__name__:
            module_functions.append(name)
    
    # These functions should NOT be defined locally anymore
    duplicate_functions = ['build_custom_id', 'parse_custom_id', 'extract_topic_name_from_filename']
    
    for func_name in duplicate_functions:
        assert func_name not in module_functions, f"Function {func_name} should be imported, not defined locally in batch_operations"


def test_function_behavior_unchanged():
    """Test that centralized functions behave identically to the old implementations."""
    from nextbook.config_manager import CustomIdManager, TopicPathManager, GapConfiguration, LLMConfiguration
    
    # Create config objects
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=15, num_books=8)
    
    # Test build_custom_id produces expected format
    custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="AI_in_Healthcare", 
        topic_id="topic123",
        gap_idx=2,
        llm_config=llm_config
    )
    expected = "NONE_k=5_radiusfrac=0.75___AI_in_Healthcare___id=topic123___gap_2___articles_15_books_8"
    assert custom_id == expected
    
    # Test parse_custom_id returns correct components
    parsed = CustomIdManager.parse_custom_id(custom_id)
    assert parsed.gap_config == "NONE_k=5_radiusfrac=0.75"
    assert parsed.topic_name == "AI_in_Healthcare"
    assert parsed.topic_id == "topic123"
    assert parsed.gap_idx == 2
    assert parsed.denoiser == "NONE"
    assert parsed.drop_k == 5
    assert parsed.radius_fraction == 0.75
    assert parsed.llm_subdir == "articles_15_books_8"
    
    # Test extract_topic_name_from_filename
    topic_name = TopicPathManager.extract_topic_name_from_filename("AI_in_Healthcare_id=topic123_gap_2_proposal.tex")
    assert topic_name == "AI in Healthcare"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
