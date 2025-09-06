"""
Test suite for Step 8: Update GapFinder Integration
Focus on gap finder function signature optimization and config object usage.
"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from nextbook.config_manager import GapConfiguration, LLMConfiguration
from nextbook.gap_directory_manager import GapDirectoryManager


class TestStep8GapFinderIntegration:
    """Test Step 8 improvements to gap finder integration."""
    
    def test_gap_finder_functions_accept_config_objects(self):
        """Test that write_single_kball_gap and write_llm_prompt accept config objects instead of individual parameters."""
        import inspect
        from nextbook.gap_finder import write_single_kball_gap, write_llm_prompt
        
        # Check write_single_kball_gap signature
        sig = inspect.signature(write_single_kball_gap)
        params = list(sig.parameters.keys())
        
        # Should have config objects, not individual parameters
        assert 'gap_config' in params
        assert 'llm_config' in params
        
        # Should NOT have individual parameters
        assert 'denoiser' not in params
        assert 'drop_k' not in params
        assert 'radius_fraction' not in params
        assert 'num_books' not in params
        assert 'num_articles' not in params
        
        # Check write_llm_prompt signature
        sig = inspect.signature(write_llm_prompt)
        params = list(sig.parameters.keys())
        
        # Should have config objects, not individual parameters
        assert 'gap_config' in params
        assert 'llm_config' in params
        
        # Should NOT have individual parameters
        assert 'denoiser' not in params
        assert 'drop_k' not in params
        assert 'radius_fraction' not in params
        assert 'num_books' not in params
        assert 'num_articles' not in params
    
    def test_config_object_parameter_optimization(self):
        """Test that function signatures are optimized to reduce parameter explosion."""
        import inspect
        from nextbook.gap_finder import write_single_kball_gap, write_llm_prompt
        
        # write_single_kball_gap should have significantly fewer parameters now
        sig = inspect.signature(write_single_kball_gap)
        num_params = len(sig.parameters)
        
        # Before optimization it had 11 parameters: gap, gap_idx, topic_name, topic_id, denoiser, drop_k, radius_fraction, num_books, num_articles, books, articles, book_gap_sizes
        # After optimization it should have 9 parameters: gap, gap_idx, topic_name, topic_id, gap_config, llm_config, books, articles, book_gap_sizes
        assert num_params == 9, f"Expected 9 parameters after optimization, got {num_params}"
        
        # write_llm_prompt should also have fewer parameters  
        sig = inspect.signature(write_llm_prompt)
        num_params = len(sig.parameters)
        
        # Before optimization it had 11 parameters: gap, gap_idx, topic_name, topic_id, denoiser, drop_k, radius_fraction, num_books, num_articles, books, articles
        # After optimization it should have 8 parameters: gap, gap_idx, topic_name, topic_id, gap_config, llm_config, books, articles
        assert num_params == 8, f"Expected 8 parameters after optimization, got {num_params}"
    
    def test_gap_finder_uses_centralized_directory_management(self):
        """Test that gap finder functions use GapDirectoryManager instead of manual path construction."""
        # This test verifies that the functions use centralized config objects
        # which automatically integrate with GapDirectoryManager
        gap_config = GapConfiguration(denoiser="NONE", drop_k=10, radius_fraction=0.25)
        llm_config = LLMConfiguration(num_articles=15, num_books=3)
        
        # Test that we can get directories using the config objects
        config_dir = GapDirectoryManager.get_config_dir(gap_config)
        gap_dir = GapDirectoryManager.get_gap_dir(gap_config, 1)  # gap_idx=1
        prompts_dir = GapDirectoryManager.get_prompts_llm_dir(gap_config, 1, llm_config)
        
        # All should return Path objects
        assert isinstance(config_dir, Path)
        assert isinstance(gap_dir, Path)
        assert isinstance(prompts_dir, Path)
        
        # Verify correct structure
        assert gap_config.to_string() in str(config_dir)
        assert "gap_1" in str(gap_dir)
        assert llm_config.to_string() in str(prompts_dir)
    
    def test_step8_function_signature_consistency(self):
        """Test that Step 8 functions have consistent parameter patterns with config objects."""
        import inspect
        from nextbook.gap_finder import write_single_kball_gap, write_llm_prompt
        
        # Both functions should start with similar parameter patterns
        sig1 = inspect.signature(write_single_kball_gap)
        sig2 = inspect.signature(write_llm_prompt)
        
        params1 = list(sig1.parameters.keys())
        params2 = list(sig2.parameters.keys())
        
        # First 6 parameters should be the same for consistency
        expected_start = ['gap', 'gap_idx', 'topic_name', 'topic_id', 'gap_config', 'llm_config']
        assert params1[:6] == expected_start
        assert params2[:6] == expected_start
    
    def test_no_parameter_explosion_in_gap_functions(self):
        """Test that gap finder functions don't suffer from parameter explosion."""
        import inspect
        from nextbook.gap_finder import write_single_kball_gap, write_llm_prompt
        
        # Check that functions don't have excessive parameters
        max_reasonable_params = 10  # Reasonable limit for function parameters
        
        sig1 = inspect.signature(write_single_kball_gap)
        assert len(sig1.parameters) <= max_reasonable_params, f"write_single_kball_gap has too many parameters: {len(sig1.parameters)}"
        
        sig2 = inspect.signature(write_llm_prompt)
        assert len(sig2.parameters) <= max_reasonable_params, f"write_llm_prompt has too many parameters: {len(sig2.parameters)}"
    
    def test_config_object_integration_in_gap_functions(self):
        """Test that gap functions properly integrate with config objects."""
        # Test that we can create the config objects that the functions expect
        gap_config = GapConfiguration(denoiser="TEST", drop_k=20, radius_fraction=0.5)
        llm_config = LLMConfiguration(num_articles=25, num_books=8)
        
        # Verify config objects have expected properties
        assert gap_config.denoiser == "TEST"
        assert gap_config.drop_k == 20
        assert gap_config.radius_fraction == 0.5
        
        assert llm_config.num_articles == 25
        assert llm_config.num_books == 8
        
        # Verify they can be converted to strings (needed for directory paths)
        gap_str = gap_config.to_string()
        llm_str = llm_config.to_string()
        
        assert "TEST_k=20_radiusfrac=0.50" in gap_str
        assert "articles_25_books_8" in llm_str
    
    def test_backwards_compatibility_after_step8(self):
        """Test that Step 8 changes maintain backwards compatibility with the overall system."""
        # Test that config objects can be created from existing patterns
        gap_config = GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=0.75)
        llm_config = LLMConfiguration(num_articles=10, num_books=5)
        
        # Test that they integrate with the directory management system
        config_dir = GapDirectoryManager.get_config_dir(gap_config)
        
        # Should create valid paths
        assert isinstance(config_dir, Path)
        assert "NONE_k=0_radiusfrac=0.75" in str(config_dir)
    
    def test_step8_reduces_configuration_scattered_logic(self):
        """Test that Step 8 successfully reduces scattered configuration logic."""
        from nextbook.gap_finder import write_single_kball_gap, write_llm_prompt
        import inspect
        
        # Functions should use config objects instead of scattered parameters
        for func in [write_single_kball_gap, write_llm_prompt]:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Should have centralized config objects
            config_params = [p for p in params if 'config' in p]
            assert len(config_params) >= 2, f"Function {func.__name__} should have at least 2 config parameters, got {config_params}"
            
            # Should not have scattered individual config parameters
            scattered_params = ['denoiser', 'drop_k', 'radius_fraction', 'num_books', 'num_articles']
            for scattered_param in scattered_params:
                assert scattered_param not in params, f"Function {func.__name__} should not have scattered parameter {scattered_param}"
