#!/usr/bin/env python3
"""
Test suite for Step 6: Update BatchOperations - Phase 1.

This test verifies that batch operations use centralized configuration management
instead of manual string building and parameter explosion.
"""
import pytest
import tempfile
import json
from pathlib import Path
from nextbook.config_manager import GapConfiguration, LLMConfiguration


def test_get_config_dir_from_custom_id_uses_centralized_system():
    """Test that get_config_dir_from_custom_id uses centralized configuration management."""
    from nextbook.batch_utils.batch_operations import get_config_dir_from_custom_id
    from nextbook.gap_directory_manager import GapDirectoryManager
    
    # Test custom_id parsing and directory resolution
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=10, num_books=3)
    
    # Build a test custom_id
    from nextbook.config_manager import CustomIdManager
    custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="Machine_Learning",
        topic_id="topic123", 
        gap_idx=2,
        llm_config=llm_config
    )
    
    # Test that get_config_dir_from_custom_id returns the correct path
    result_dir = get_config_dir_from_custom_id(custom_id)
    expected_dir = GapDirectoryManager.get_config_dir(gap_config)
    
    assert result_dir == expected_dir
    assert result_dir.name == "NONE_k=5_radiusfrac=0.75"


def test_get_config_dir_from_results_file_integration():
    """Test that get_config_dir_from_results_file works with centralized system."""
    from nextbook.batch_utils.batch_operations import get_config_dir_from_results_file
    from nextbook.gap_directory_manager import GapDirectoryManager
    
    # Create a temporary results file with a custom_id
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=10, num_books=3)
    
    from nextbook.config_manager import CustomIdManager
    custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="AI_Research",
        topic_id="topic456", 
        gap_idx=1,
        llm_config=llm_config
    )
    
    # Create temporary results file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        results_data = {
            "id": "batch_req_123",
            "custom_id": custom_id,
            "response": {"status_code": 200},
            "error": None
        }
        json.dump(results_data, f)
        temp_file = Path(f.name)
    
    try:
        # Test that the function returns the correct config directory
        result_dir = get_config_dir_from_results_file(temp_file)
        expected_dir = GapDirectoryManager.get_config_dir(gap_config)
        
        assert result_dir == expected_dir
        assert result_dir.name == "NONE_k=5_radiusfrac=0.75"
        
    finally:
        temp_file.unlink()  # Clean up


def test_centralized_gap_configuration_string_building():
    """Test that GapConfiguration.to_string() produces correct format."""
    from nextbook.config_manager import GapConfiguration
    
    # Test various configurations
    config1 = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    assert config1.to_string() == "NONE_k=5_radiusfrac=0.75"
    
    config2 = GapConfiguration(denoiser="PCA", drop_k=10, radius_fraction=0.5)
    assert config2.to_string() == "PCA_k=10_radiusfrac=0.50"
    
    config3 = GapConfiguration(denoiser="TSNE", drop_k=0, radius_fraction=1.0)
    assert config3.to_string() == "TSNE_k=0_radiusfrac=1.00"


def test_batch_operations_no_manual_string_building():
    """Test that batch_operations no longer contains manual string building patterns."""
    import inspect
    from nextbook.batch_utils import batch_operations
    
    # Get the source code of the module
    source = inspect.getsource(batch_operations)
    
    # Check that we don't have the old manual string building patterns
    manual_patterns = [
        'f"{denoiser}_k={',  # Old manual gap config building
        '_radiusfrac={',     # Old manual radius fraction building
        'f"articles_{',      # Old manual LLM directory building
        '_books_{',          # Old manual LLM directory building
    ]
    
    for pattern in manual_patterns:
        # Allow the pattern in comments and docstrings, but not in actual code
        lines = source.split('\n')
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            # Skip comments and docstrings
            if (stripped_line.startswith('#') or 
                stripped_line.startswith('"""') or 
                stripped_line.startswith("'''")):
                continue
            
            if pattern in line:
                pytest.fail(f"Found manual string building pattern '{pattern}' on line {line_num}: {line.strip()}")


def test_config_objects_integration():
    """Test that config objects work together seamlessly."""
    from nextbook.config_manager import GapConfiguration, LLMConfiguration, CustomIdManager
    from nextbook.gap_directory_manager import GapDirectoryManager
    
    # Create config objects
    gap_config = GapConfiguration(denoiser="NONE", drop_k=3, radius_fraction=0.8)
    llm_config = LLMConfiguration(num_articles=25, num_books=12)
    
    # Test that they integrate properly with directory management
    config_dir = GapDirectoryManager.get_config_dir(gap_config)
    gap_dir = GapDirectoryManager.get_gap_dir(gap_config, 1)
    llm_dir = GapDirectoryManager.get_prompts_llm_dir(gap_config, 1, llm_config)
    
    assert config_dir.name == "NONE_k=3_radiusfrac=0.80"
    assert gap_dir.name == "gap_1"
    assert llm_dir.name == "articles_25_books_12"
    
    # Test that custom_id generation works
    custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="Deep_Learning",
        topic_id="topic789",
        gap_idx=1,
        llm_config=llm_config
    )
    
    expected_id = "NONE_k=3_radiusfrac=0.80___Deep_Learning___id=topic789___gap_1___articles_25_books_12"
    assert custom_id == expected_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
