#!/usr/bin/env python3
import pytest

pytest.skip("Skipped legacy step test (ConfigPathParser legacy format)", allow_module_level=True)


def test_config_path_parser_parse_prompt_file_path_valid():
    """Test parsing valid prompt file paths."""
    gaps_base_dir = "/artifacts/gaps"
    prompt_path = "/artifacts/gaps/NONE_k=5_radiusfrac=0.75/gap_3/prompts/articles_10_books_5/Quantum_Computing_id=1234_prompt.txt"
    
    result = ConfigPathParser.parse_prompt_file_path(prompt_path, gaps_base_dir)
    
    assert isinstance(result['gap_config'], GapConfiguration)
    assert result['gap_config'].denoiser == "NONE"
    assert result['gap_config'].drop_k == 5
    assert result['gap_config'].radius_fraction == 0.75
    
    assert isinstance(result['llm_config'], LLMConfiguration)
    assert result['llm_config'].num_articles == 10
    assert result['llm_config'].num_books == 5
    
    assert result['topic_name'] == "Quantum_Computing"
    assert result['topic_id'] == "1234"
    assert result['gap_idx'] == 3
    assert result['filename'] == "Quantum_Computing_id=1234_prompt.txt"
    assert result['gap_config_str'] == "NONE_k=5_radiusfrac=0.75"
    assert result['llm_subdir'] == "articles_10_books_5"


def test_config_path_parser_parse_prompt_file_path_variations():
    """Test parsing prompt file paths with different configurations."""
    gaps_base_dir = "/base/gaps"
    
    # Different denoiser
    prompt_path = "/base/gaps/LPP_k=10_radiusfrac=0.50/gap_0/prompts/articles_20_books_0/AI_Research_id=abc123_prompt.txt"
    result = ConfigPathParser.parse_prompt_file_path(prompt_path, gaps_base_dir)
    
    assert result['gap_config'].denoiser == "LPP"
    assert result['gap_config'].drop_k == 10
    assert result['gap_config'].radius_fraction == 0.50
    assert result['gap_idx'] == 0
    assert result['topic_name'] == "AI_Research"
    assert result['topic_id'] == "abc123"
    assert result['llm_config'].num_books == 0


def test_config_path_parser_parse_prompt_file_path_validation_errors():
    """Test prompt file path parsing validation errors."""
    gaps_base_dir = "/artifacts/gaps"
    
    # Path not in gaps directory
    with pytest.raises(FileNotFoundError, match="Prompt file not in gaps directory"):
        ConfigPathParser.parse_prompt_file_path("/other/path/file.txt", gaps_base_dir)
    
    # Wrong number of path parts
    with pytest.raises(ValueError, match="Expected 5 path parts"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/config/gap_0/file.txt", gaps_base_dir)
    
    # Wrong prompts subdirectory
    with pytest.raises(ValueError, match="Expected 'prompts' subdirectory"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/config/gap_0/wrong/llm/file.txt", gaps_base_dir)
    
    # Invalid gap configuration
    with pytest.raises(ValueError, match="Invalid gap configuration"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/invalid_config/gap_0/prompts/articles_5_books_3/file_id=123_prompt.txt", gaps_base_dir)
    
    # Invalid LLM configuration  
    with pytest.raises(ValueError, match="Invalid LLM configuration"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/NONE_k=5_radiusfrac=0.75/gap_0/prompts/invalid_llm/file_id=123_prompt.txt", gaps_base_dir)
    
    # Invalid gap directory format
    with pytest.raises(ValueError, match="Invalid gap directory format"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/NONE_k=5_radiusfrac=0.75/badgap/prompts/articles_5_books_3/file_id=123_prompt.txt", gaps_base_dir)
    
    # Invalid filename format
    with pytest.raises(ValueError, match="Invalid prompt filename format"):
        ConfigPathParser.parse_prompt_file_path("/artifacts/gaps/NONE_k=5_radiusfrac=0.75/gap_0/prompts/articles_5_books_3/invalid_filename.txt", gaps_base_dir)


def test_config_path_parser_get_config_dir_from_custom_id():
    """Test getting config directory from custom ID."""
    custom_id = "NONE_k=5_radiusfrac=0.75___topic___id=1234___gap_3___articles_10_books_5"
    gaps_base_dir = "/artifacts/gaps"
    
    config_dir = ConfigPathParser.get_config_dir_from_custom_id(custom_id, gaps_base_dir)
    assert config_dir == "/artifacts/gaps/NONE_k=5_radiusfrac=0.75"


def test_config_path_parser_get_config_dir_from_custom_id_validation_errors():
    """Test config directory extraction validation errors."""
    gaps_base_dir = "/artifacts/gaps"
    
    # Invalid custom ID
    with pytest.raises(ValueError, match="Expected 5 parts in custom_id"):
        ConfigPathParser.get_config_dir_from_custom_id("invalid___custom___id", gaps_base_dir)


def test_config_path_parser_get_config_dir_from_results_file():
    """Test getting config directory from results file."""
    # Create a temporary results file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        result_data = {
            "id": "batch_12345", 
            "custom_id": "NONE_k=5_radiusfrac=0.75___topic___id=1234___gap_3___articles_10_books_5",
            "response": {"status_code": 200}
        }
        f.write(json.dumps(result_data) + "\n")
        temp_file = f.name
    
    try:
        gaps_base_dir = "/artifacts/gaps"
        config_dir = ConfigPathParser.get_config_dir_from_results_file(temp_file, gaps_base_dir)
        assert config_dir == "/artifacts/gaps/NONE_k=5_radiusfrac=0.75"
    finally:
        Path(temp_file).unlink()


def test_config_path_parser_get_config_dir_from_results_file_validation_errors():
    """Test results file processing validation errors."""
    gaps_base_dir = "/artifacts/gaps"
    
    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Results file does not exist"):
        ConfigPathParser.get_config_dir_from_results_file("/nonexistent/file.jsonl", gaps_base_dir)
    
    # Empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        empty_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Results file is empty"):
            ConfigPathParser.get_config_dir_from_results_file(empty_file, gaps_base_dir)
    finally:
        Path(empty_file).unlink()
    
    # Invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write("invalid json content")
        invalid_json_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Invalid JSON in results file"):
            ConfigPathParser.get_config_dir_from_results_file(invalid_json_file, gaps_base_dir)
    finally:
        Path(invalid_json_file).unlink()
    
    # Missing custom_id
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        result_data = {"id": "batch_12345", "response": {"status_code": 200}}
        f.write(json.dumps(result_data) + "\n")
        no_custom_id_file = f.name
    
    try:
        with pytest.raises(ValueError, match="No custom_id found in results file"):
            ConfigPathParser.get_config_dir_from_results_file(no_custom_id_file, gaps_base_dir)
    finally:
        Path(no_custom_id_file).unlink()


def test_config_path_parser_validate_directory_structure():
    """Test directory structure validation."""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create expected subdirectories
        (temp_path / "subdir1").mkdir()
        (temp_path / "subdir2").mkdir()
        
        # Valid structure
        result = ConfigPathParser.validate_directory_structure(str(temp_path), ["subdir1", "subdir2"])
        assert result is True
        
        # Missing subdirectory
        with pytest.raises(ValueError, match="Expected subdirectory not found"):
            ConfigPathParser.validate_directory_structure(str(temp_path), ["subdir1", "missing_subdir"])


def test_config_path_parser_validate_directory_structure_validation_errors():
    """Test directory structure validation errors."""
    # Non-existent directory
    with pytest.raises(ValueError, match="Directory does not exist"):
        ConfigPathParser.validate_directory_structure("/nonexistent/path", [])
    
    # Path is not a directory
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Path is not a directory"):
            ConfigPathParser.validate_directory_structure(temp_file, [])
    finally:
        Path(temp_file).unlink()


def test_config_path_parser_integration():
    """Test integration between different ConfigPathParser methods."""
    gaps_base_dir = "/artifacts/gaps"
    
    # Create a custom ID using CustomIdManager
    from nextbook.config_manager import CustomIdManager, GapConfiguration, LLMConfiguration
    
    gap_config = GapConfiguration(denoiser="NONE", drop_k=5, radius_fraction=0.75)
    llm_config = LLMConfiguration(num_articles=10, num_books=5)
    
    custom_id = CustomIdManager.build_custom_id(gap_config, "topic_name", "1234", 3, llm_config)
    
    # Extract config directory from custom ID
    config_dir = ConfigPathParser.get_config_dir_from_custom_id(custom_id, gaps_base_dir)
    
    expected_config_dir = f"{gaps_base_dir}/{gap_config.to_string()}"
    assert config_dir == expected_config_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
