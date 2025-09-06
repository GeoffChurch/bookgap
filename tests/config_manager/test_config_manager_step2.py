#!/usr/bin/env python3
import pytest

# Skipped: legacy step test referencing deprecated GapConfiguration/LLMConfiguration.
pytest.skip("Skipped legacy test (replaced by AnalysisConfiguration tests)", allow_module_level=True)


def test_custom_id_manager_build_valid():
    """Test building valid custom IDs."""
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=10, num_books=3)
    
    custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="quantum_computing", 
        topic_id="1234",
        gap_idx=7,
        llm_config=llm_config
    )
    
    expected = "NONE_k=5_radiusfrac=0.75___quantum_computing___id=1234___gap_7___articles_10_books_3"
    assert custom_id == expected


def test_custom_id_manager_parse_valid():
    """Test parsing valid custom IDs."""
    custom_id = "NONE_k=5_radiusfrac=0.75___quantum_computing___id=1234___gap_7___articles_10_books_3"
    
    components = CustomIdManager.parse_custom_id(custom_id)
    
    assert components.gap_config == "NONE_k=5_radiusfrac=0.75"
    assert components.topic_name == "quantum_computing"
    assert components.topic_id == "1234"
    assert components.gap_idx == 7
    assert components.denoiser == "NONE"
    assert components.drop_k == 5
    assert components.radius_fraction == 0.75
    assert components.llm_subdir == "articles_10_books_3"


def test_custom_id_manager_round_trip():
    """Test that build -> parse -> build gives same result."""
    gap_config = GapConfiguration(denoiser="LPP", drop_k=10, radius_fraction=0.5)
    llm_config = LLMConfiguration(num_articles=20, num_books=0)
    
    original_custom_id = CustomIdManager.build_custom_id(
        gap_config=gap_config,
        topic_name="machine_learning",
        topic_id="5678", 
        gap_idx=15,
        llm_config=llm_config
    )
    
    # Parse it back
    components = CustomIdManager.parse_custom_id(original_custom_id)
    
    # Build again from components
    rebuilt_gap_config = GapConfiguration(
        denoiser=components.denoiser,
        drop_k=components.drop_k,
        radius_fraction=components.radius_fraction
    )
    rebuilt_llm_config = LLMConfiguration.from_string(components.llm_subdir)
    
    rebuilt_custom_id = CustomIdManager.build_custom_id(
        gap_config=rebuilt_gap_config,
        topic_name=components.topic_name,
        topic_id=components.topic_id,
        gap_idx=components.gap_idx,
        llm_config=rebuilt_llm_config
    )
    
    assert original_custom_id == rebuilt_custom_id


def test_custom_id_manager_build_validation_errors():
    """Test that CustomIdManager.build_custom_id throws appropriate errors."""
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=10, num_books=3)
    
    # Negative gap_idx
    with pytest.raises(ValueError, match="gap_idx must be >= 0"):
        CustomIdManager.build_custom_id(gap_config, "topic", "1234", -1, llm_config)
    
    # Empty topic_name
    with pytest.raises(ValueError, match="topic_name cannot be empty"):
        CustomIdManager.build_custom_id(gap_config, "", "1234", 0, llm_config)
    
    # Empty topic_id
    with pytest.raises(ValueError, match="topic_id cannot be empty"):
        CustomIdManager.build_custom_id(gap_config, "topic", "", 0, llm_config)
    
    # Whitespace-only topic_name
    with pytest.raises(ValueError, match="topic_name cannot be empty"):
        CustomIdManager.build_custom_id(gap_config, "   ", "1234", 0, llm_config)


def test_custom_id_manager_parse_validation_errors():
    """Test that CustomIdManager.parse_custom_id throws appropriate errors."""
    # Empty custom_id
    with pytest.raises(ValueError, match="custom_id cannot be empty"):
        CustomIdManager.parse_custom_id("")
    
    # Wrong number of parts
    with pytest.raises(ValueError, match="Expected 5 parts.*separated by '___'"):
        CustomIdManager.parse_custom_id("only___three___parts")
    
    # Missing id= prefix
    with pytest.raises(ValueError, match="topic_id part must start with 'id='"):
        CustomIdManager.parse_custom_id("NONE_k=5_radiusfrac=0.75___topic___1234___gap_0___articles_10_books_3")
    
    # Missing gap_ prefix 
    with pytest.raises(ValueError, match="gap_info part must start with 'gap_'"):
        CustomIdManager.parse_custom_id("NONE_k=5_radiusfrac=0.75___topic___id=1234___0___articles_10_books_3")
    
    # Invalid gap_idx (non-numeric)
    with pytest.raises(ValueError, match="gap_idx must be an integer after 'gap_'"):
        CustomIdManager.parse_custom_id("NONE_k=5_radiusfrac=0.75___topic___id=1234___gap_abc___articles_10_books_3")
    
    # Invalid gap_config format
    with pytest.raises(ValueError, match="Invalid gap_config in custom_id"):
        CustomIdManager.parse_custom_id("invalid_config___topic___id=1234___gap_0___articles_10_books_3")


def test_custom_id_manager_edge_cases():
    """Test edge cases for CustomIdManager."""
    # Zero gap_idx should work
    gap_config = GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=0.0)
    llm_config = LLMConfiguration(num_articles=1, num_books=0)
    
    custom_id = CustomIdManager.build_custom_id(gap_config, "test", "0", 0, llm_config)
    components = CustomIdManager.parse_custom_id(custom_id)
    
    assert components.gap_idx == 0
    assert components.drop_k == 0
    assert components.radius_fraction == 0.0
    
    # Topic names with underscores should work (they won't interfere with parsing)
    custom_id = CustomIdManager.build_custom_id(gap_config, "topic_with_underscores", "999", 5, llm_config)
    components = CustomIdManager.parse_custom_id(custom_id)
    assert components.topic_name == "topic_with_underscores"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
