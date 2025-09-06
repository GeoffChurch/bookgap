#!/usr/bin/env python3
import pytest

# Skipped: legacy step test file; covered by consolidated tests.
pytest.skip("Skipped legacy step test (TopicPathManager covered elsewhere)", allow_module_level=True)


def test_topic_path_manager_sanitize_topic_name():
    """Test topic name sanitization."""
    # Basic sanitization
    assert TopicPathManager.sanitize_topic_name("Quantum Computing") == "Quantum Computing"
    assert TopicPathManager.sanitize_topic_name("Machine_Learning") == "Machine_Learning"
    
    # Remove problematic characters
    sanitized = TopicPathManager.sanitize_topic_name("AI & ML: Future?")
    assert "AI" in sanitized and "ML" in sanitized and "Future" in sanitized
    assert "&" not in sanitized and ":" not in sanitized and "?" not in sanitized
    
    # Handle whitespace
    assert TopicPathManager.sanitize_topic_name("  Topic Name  ") == "Topic Name"


def test_topic_path_manager_sanitize_validation_errors():
    """Test sanitization validation errors."""
    # Empty topic name
    with pytest.raises(ValueError, match="topic_name cannot be empty"):
        TopicPathManager.sanitize_topic_name("")
    
    # Whitespace-only topic name
    with pytest.raises(ValueError, match="topic_name cannot be empty"):
        TopicPathManager.sanitize_topic_name("   ")
    
    # Topic name that becomes empty after sanitization
    with pytest.raises(ValueError, match="topic_name becomes empty after sanitization"):
        TopicPathManager.sanitize_topic_name("@#$%^&*[]{}|\\/<>?:;+")


def test_topic_path_manager_extract_topic_name_from_filename():
    """Test extracting topic names from filenames."""
    # Standard format
    assert TopicPathManager.extract_topic_name_from_filename("Quantum_Computing_id=1234_gap_5_proposal.tex") == "Quantum Computing"
    
    # Without .tex extension
    assert TopicPathManager.extract_topic_name_from_filename("Machine_Learning_id=5678_gap_2_proposal") == "Machine Learning"
    
    # Without _proposal suffix
    assert TopicPathManager.extract_topic_name_from_filename("AI_Research_id=9999_gap_0.tex") == "AI Research"
    
    # Minimal format
    assert TopicPathManager.extract_topic_name_from_filename("Topic_id=123") == "Topic"


def test_topic_path_manager_extract_topic_name_validation_errors():
    """Test topic name extraction validation errors."""
    # Empty filename
    with pytest.raises(ValueError, match="filename cannot be empty"):
        TopicPathManager.extract_topic_name_from_filename("")
    
    # Missing _id= part
    with pytest.raises(ValueError, match="Expected '_id=' in filename"):
        TopicPathManager.extract_topic_name_from_filename("topic_name_no_id.tex")
    
    # Empty topic name after parsing
    with pytest.raises(ValueError, match="Topic name is empty after parsing filename"):
        TopicPathManager.extract_topic_name_from_filename("_id=123.tex")


def test_topic_path_manager_extract_topic_id_from_filename():
    """Test extracting topic IDs from filenames."""
    # Standard format
    assert TopicPathManager.extract_topic_id_from_filename("Topic_Name_id=1234_gap_5_proposal.tex") == "1234"
    
    # Different ID formats
    assert TopicPathManager.extract_topic_id_from_filename("Topic_id=abc123_gap_0.tex") == "abc123"
    assert TopicPathManager.extract_topic_id_from_filename("Topic_id=999") == "999"
    
    # Multiple underscores after ID
    assert TopicPathManager.extract_topic_id_from_filename("Topic_id=456_gap_1_extra_stuff.tex") == "456"


def test_topic_path_manager_extract_topic_id_validation_errors():
    """Test topic ID extraction validation errors."""
    # Empty filename
    with pytest.raises(ValueError, match="filename cannot be empty"):
        TopicPathManager.extract_topic_id_from_filename("")
    
    # Missing _id= part
    with pytest.raises(ValueError, match="Expected '_id=' in filename"):
        TopicPathManager.extract_topic_id_from_filename("topic_name_no_id.tex")
    
    # Multiple _id= parts
    with pytest.raises(ValueError, match="Expected exactly one '_id=' in filename"):
        TopicPathManager.extract_topic_id_from_filename("topic_id=123_id=456.tex")
    
    # Empty topic ID
    with pytest.raises(ValueError, match="Topic ID is empty after parsing filename"):
        TopicPathManager.extract_topic_id_from_filename("topic_id=_gap_0.tex")


def test_topic_path_manager_build_analysis_filename():
    """Test building analysis filenames."""
    filename = TopicPathManager.build_analysis_filename("Quantum Computing", "1234")
    assert filename == "Quantum Computing_id=1234_analysis.txt"
    
    # With sanitization
    filename = TopicPathManager.build_analysis_filename("AI & ML", "5678")
    assert filename.endswith("_id=5678_analysis.txt")
    assert "AI" in filename and "ML" in filename


def test_topic_path_manager_build_analysis_filename_validation_errors():
    """Test analysis filename building validation errors."""
    # Empty topic_id
    with pytest.raises(ValueError, match="topic_id cannot be empty"):
        TopicPathManager.build_analysis_filename("Topic", "")
    
    # Whitespace-only topic_id
    with pytest.raises(ValueError, match="topic_id cannot be empty"):
        TopicPathManager.build_analysis_filename("Topic", "   ")
    
    # Empty topic_name (caught by sanitization)
    with pytest.raises(ValueError, match="topic_name cannot be empty"):
        TopicPathManager.build_analysis_filename("", "1234")


def test_topic_path_manager_build_prompt_filename():
    """Test building prompt filenames."""
    filename = TopicPathManager.build_prompt_filename("Machine Learning", "9999")
    assert filename == "Machine Learning_id=9999_prompt.txt"


def test_topic_path_manager_build_prompt_filename_validation_errors():
    """Test prompt filename building validation errors."""
    # Empty topic_id
    with pytest.raises(ValueError, match="topic_id cannot be empty"):
        TopicPathManager.build_prompt_filename("Topic", "")


def test_topic_path_manager_build_plot_filename():
    """Test building plot filenames."""
    filename = TopicPathManager.build_plot_filename("Quantum Computing", 5, "PCA", "visualization")
    assert filename == "Quantum Computing_gap_5_pca_visualization.png"
    
    # With sanitization
    filename = TopicPathManager.build_plot_filename("AI & ML", 0, "T-SNE Analysis", "scatter_plot")
    expected_parts = ["gap_0", ".png"]
    for part in expected_parts:
        assert part in filename
    assert "AI" in filename and "ML" in filename


def test_topic_path_manager_build_plot_filename_validation_errors():
    """Test plot filename building validation errors."""
    # Negative gap_idx
    with pytest.raises(ValueError, match="gap_idx must be >= 0"):
        TopicPathManager.build_plot_filename("Topic", -1, "method", "plot_type")
    
    # Empty method
    with pytest.raises(ValueError, match="method cannot be empty"):
        TopicPathManager.build_plot_filename("Topic", 0, "", "plot_type")
    
    # Empty plot_type
    with pytest.raises(ValueError, match="plot_type cannot be empty"):
        TopicPathManager.build_plot_filename("Topic", 0, "method", "")
    
    # Whitespace-only method
    with pytest.raises(ValueError, match="method cannot be empty"):
        TopicPathManager.build_plot_filename("Topic", 0, "   ", "plot_type")


def test_topic_path_manager_round_trip_filename_parsing():
    """Test that filename building and parsing work together."""
    original_topic_name = "Quantum Computing"
    original_topic_id = "1234"
    
    # Build analysis filename and extract components back
    analysis_filename = TopicPathManager.build_analysis_filename(original_topic_name, original_topic_id)
    
    # Remove the _analysis.txt suffix to simulate a filename that could be parsed
    base_filename = analysis_filename.replace("_analysis.txt", "_gap_0_proposal.tex")
    
    extracted_topic_name = TopicPathManager.extract_topic_name_from_filename(base_filename)
    extracted_topic_id = TopicPathManager.extract_topic_id_from_filename(base_filename)
    
    assert extracted_topic_name == original_topic_name
    assert extracted_topic_id == original_topic_id


def test_topic_path_manager_edge_cases():
    """Test edge cases for TopicPathManager."""
    # Single character topic name
    assert TopicPathManager.sanitize_topic_name("A") == "A"
    
    # Topic name with numbers
    filename = TopicPathManager.build_analysis_filename("AI2023", "001")
    assert "AI2023" in filename and "001" in filename
    
    # Very long topic names (should work fine)
    long_name = "This is a very long topic name that might cause issues in some systems"
    sanitized = TopicPathManager.sanitize_topic_name(long_name)
    assert len(sanitized) > 0
    
    # Plot filename with gap_idx = 0
    filename = TopicPathManager.build_plot_filename("Topic", 0, "method", "plot")
    assert "gap_0" in filename


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
