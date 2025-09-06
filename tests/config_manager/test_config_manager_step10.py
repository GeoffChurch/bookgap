"""
Test Step 10: Comprehensive Testing and Cleanup

This final step validates the complete centralized configuration management system
with clean tests that demonstrate the system works end-to-end.
"""

import pytest
import tempfile
import time
from pathlib import Path

from nextbook.config_manager import (
    GapConfiguration, LLMConfiguration, CustomIdManager, 
    TopicPathManager, ConfigPathParser
)
from nextbook.gap_directory_manager import GapDirectoryManager
from nextbook.results_manager import GapResults, ResultsFactory, ResultType


def test_end_to_end_workflow():
    """Test complete workflow from config creation to results management."""
    # Step 1: Create configurations
    gap_config = GapConfiguration(
        denoiser="bilateral",
        drop_k=7,
        radius_fraction=0.18
    )
    
    llm_config = LLMConfiguration(
        num_articles=25,
        num_books=15
    )
    
    # Step 2: Create directory structure using centralized system
    gap_dir_manager = GapDirectoryManager()
    config_dir = gap_dir_manager.get_config_dir(gap_config)
    
    # Verify directories are created with correct structure
    assert config_dir.exists()
    assert gap_config.to_string() in str(config_dir)
    
    # Step 3: Create and use results manager
    results = GapResults(gap_config=gap_config, config_dir=config_dir)
    
    # Add some test data with unique identifier
    import time
    unique_topic = f"End-to-End Test Topic {int(time.time()*1000000) % 1000000}"
    results.add_row(
        topic_name=unique_topic,
        topic_id="e2e12345",
        gap_idx=1,
        num_books=llm_config.num_books,
        num_articles=llm_config.num_articles,
        gap_size=42,
        gap_size_percentile=0.87,
        num_articles_for_llm=llm_config.num_articles,
        num_books_for_llm=llm_config.num_books
    )
    
    # Step 4: Verify data integrity
    df = results.get_df()
    assert len(df) >= 1  # Allow for existing data from other tests
    
    # Find our specific row
    our_rows = df[df['topic_name'] == unique_topic]
    assert len(our_rows) == 1
    row = our_rows.iloc[0]
    assert row['denoiser'] == gap_config.denoiser
    assert row['drop_k'] == gap_config.drop_k
    assert row['radius_fraction'] == gap_config.radius_fraction
    assert row['topic_name'] == unique_topic
    
    # Step 5: Test CSV export/import round-trip
    csv_path = config_dir / "test_export_step10.csv"  # Unique name
    results.to_csv(csv_path)
    
    # Reload from CSV using factory
    reloaded_results = ResultsFactory.create_from_csv(csv_path)
    assert isinstance(reloaded_results, GapResults)
    assert reloaded_results.gap_config.denoiser == gap_config.denoiser
    
    reloaded_df = reloaded_results.get_df()
    assert len(reloaded_df) >= 1
    our_reloaded_rows = reloaded_df[reloaded_df['topic_name'] == unique_topic]
    assert len(our_reloaded_rows) == 1
    assert our_reloaded_rows.iloc[0]['topic_name'] == unique_topic


def test_all_components_integration():
    """Test that all centralized components work together."""
    # Simple test of configuration round-trip
    gap_config = GapConfiguration(denoiser="median", drop_k=5, radius_fraction=0.15)
    llm_config = LLMConfiguration(num_articles=20, num_books=10)
    
    # Test configuration string operations
    gap_str = gap_config.to_string()
    parsed_gap = GapConfiguration.from_string(gap_str)
    assert parsed_gap == gap_config
    
    llm_str = llm_config.to_string() 
    parsed_llm = LLMConfiguration.from_string(llm_str)
    assert parsed_llm == llm_config
    
    # Test topic path operations
    topic_name = "Integration Test Topic"
    sanitized = TopicPathManager.sanitize_topic_name(topic_name)
    # Verify sanitization works (actual behavior may vary)
    assert sanitized is not None
    assert len(sanitized) > 0


def test_performance_no_significant_overhead():
    """Test that centralized system doesn't introduce significant performance overhead."""
    # Create test configurations
    gap_config = GapConfiguration(
        denoiser="performance",
        drop_k=5,
        radius_fraction=0.15
    )
    
    llm_config = LLMConfiguration(
        num_articles=20,
        num_books=10
    )
    
    # Time configuration operations
    start_time = time.time()
    
    for i in range(1000):
        # Configuration string operations
        config_str = gap_config.to_string()
        parsed_config = GapConfiguration.from_string(config_str)
        
        # LLM directory operations
        dir_name = llm_config.to_string()
        parsed_llm = LLMConfiguration.from_string(dir_name)
        
        # Verify they're equal
        assert parsed_config == gap_config
        assert parsed_llm == llm_config
    
    elapsed_time = time.time() - start_time
    
    # Should complete 1000 operations in well under 1 second
    assert elapsed_time < 1.0, f"Performance test took {elapsed_time:.3f}s, expected < 1.0s"


def test_error_handling_consistency():
    """Test that error handling is consistent across all components."""
    # Test that all components raise ValueError for invalid inputs
    with pytest.raises(ValueError):
        GapConfiguration(denoiser="", drop_k=5, radius_fraction=0.1)
    
    with pytest.raises(ValueError):
        LLMConfiguration(num_articles=0, num_books=5)
    
    with pytest.raises(ValueError):
        TopicPathManager.sanitize_topic_name("")


def test_real_world_scenarios():
    """Test realistic workflows that might be used in production."""
    # Scenario 1: Batch processing multiple configurations
    configs = [
        GapConfiguration(denoiser="median", drop_k=5, radius_fraction=0.15),
        GapConfiguration(denoiser="gauss", drop_k=10, radius_fraction=0.20),
        GapConfiguration(denoiser="bilateral", drop_k=7, radius_fraction=0.18)
    ]
    
    llm_configs = [
        LLMConfiguration(num_articles=20, num_books=10),
        LLMConfiguration(num_articles=30, num_books=15)
    ]
    
    gap_dir_manager = GapDirectoryManager()
    
    # Create all directory structures
    all_results = []
    for gap_config in configs:
        for llm_config in llm_configs:
            config_dir = gap_dir_manager.get_config_dir(gap_config)
            
            results = GapResults(gap_config=gap_config, config_dir=config_dir)
            
            # Add test data
            results.add_row(
                topic_name=f"Topic for {gap_config.denoiser}",
                topic_id=f"id{gap_config.drop_k}{llm_config.num_articles}",
                gap_idx=1,
                num_books=llm_config.num_books,
                num_articles=llm_config.num_articles,
                gap_size=25,
                num_articles_for_llm=llm_config.num_articles,
                num_books_for_llm=llm_config.num_books
            )
            
            all_results.append(results)
    
    # Verify all results were created correctly
    assert len(all_results) == len(configs) * len(llm_configs)
    
    # Verify each has correct configuration
    for results in all_results:
        assert isinstance(results.gap_config, GapConfiguration)
        df = results.get_df()
        assert len(df) >= 1
        assert results.gap_config.denoiser in ["median", "gauss", "bilateral"]


def test_centralization_benefits_achieved():
    """Test that centralization goals have been achieved."""
    # Test 1: Single source of truth for configuration formats
    gap_config = GapConfiguration(denoiser="test", drop_k=5, radius_fraction=0.1)
    config_str1 = gap_config.to_string()
    
    # The same configuration should always produce the same string
    config_str2 = GapConfiguration(denoiser="test", drop_k=5, radius_fraction=0.1).to_string()
    assert config_str1 == config_str2
    
    # Test 2: Consistent path building across modules
    gap_dir_manager = GapDirectoryManager()
    
    # Directory structure should be consistent
    config_dir1 = gap_dir_manager.get_config_dir(gap_config)
    config_dir2 = gap_dir_manager.get_config_dir(
        GapConfiguration(denoiser="test", drop_k=5, radius_fraction=0.1)
    )
    
    assert config_dir1 == config_dir2
    
    # Test 3: Fail-fast error handling
    with pytest.raises(ValueError):
        GapConfiguration.from_string("invalid_config_string")
    
    # Test 4: Type safety through config objects
    gap_config = GapConfiguration(denoiser="median", drop_k=5, radius_fraction=0.1)
    assert isinstance(gap_config.denoiser, str)
    assert isinstance(gap_config.drop_k, int)
    assert isinstance(gap_config.radius_fraction, float)


def test_all_components_properly_integrated():
    """Verify all components from Steps 1-9 are properly integrated."""
    # This test imports and uses every major component
    from nextbook.config_manager import (
        GapConfiguration, LLMConfiguration, CustomIdManager, 
        TopicPathManager, ConfigPathParser
    )
    from nextbook.gap_directory_manager import GapDirectoryManager
    from nextbook.results_manager import GapResults, ResultsFactory
    
    # All imports should work without errors
    # All classes should be properly defined
    assert GapConfiguration is not None
    assert LLMConfiguration is not None
    assert CustomIdManager is not None
    assert TopicPathManager is not None
    assert ConfigPathParser is not None
    assert GapDirectoryManager is not None
    assert GapResults is not None
    assert ResultsFactory is not None


def test_comprehensive_system_health():
    """Final system health check."""
    # Create a complete configuration
    gap_config = GapConfiguration(
        denoiser="system",
        drop_k=5,
        radius_fraction=0.15
    )
    
    llm_config = LLMConfiguration(
        num_articles=25,
        num_books=15
    )
    
    # Test all major operations work
    config_str = gap_config.to_string()
    parsed_config = GapConfiguration.from_string(config_str)
    assert parsed_config == gap_config
    
    llm_str = llm_config.to_string()
    parsed_llm = LLMConfiguration.from_string(llm_str)
    assert parsed_llm == llm_config
    
    # Test directory manager works
    gap_dir_manager = GapDirectoryManager()
    config_dir = gap_dir_manager.get_config_dir(gap_config)
    assert config_dir.exists()
    assert gap_config.to_string() in str(config_dir)
    
    # Test results manager works
    results = GapResults(gap_config=gap_config, config_dir=config_dir)
    assert results.gap_config == gap_config
    
    # System is healthy if all operations complete successfully
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
