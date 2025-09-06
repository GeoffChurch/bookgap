#!/usr/bin/env python3
import pytest

pytest.skip("Skipped legacy config_manager step test; replaced by AnalysisConfiguration tests.", allow_module_level=True)


def test_gap_configuration_valid():
    """Test valid GapConfiguration creation and string conversion."""
    config = GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=0.75)
    
    assert config.denoiser == "NONE"
    assert config.drop_k == 0
    assert config.radius_fraction == 0.75
    assert config.to_string() == "NONE_k=0_radiusfrac=0.75"
    print("✓ GapConfiguration valid creation and string conversion")


def test_gap_configuration_parsing():
    """Test GapConfiguration parsing from string."""
    config_str = "LPP_k=5_radiusfrac=0.50"
    config = GapConfiguration.from_string(config_str)
    
    assert config.denoiser == "LPP"
    assert config.drop_k == 5
    assert config.radius_fraction == 0.50
    assert config.to_string() == config_str
    print("✓ GapConfiguration parsing from string")


def test_gap_configuration_round_trip():
    """Test that string conversion is reversible."""
    original = GapConfiguration(denoiser="TEST", drop_k=10, radius_fraction=0.33)
    config_str = original.to_string()
    parsed = GapConfiguration.from_string(config_str)
    
    assert parsed == original
    print("✓ GapConfiguration round-trip conversion")


def test_gap_configuration_validation_errors():
    """Test that GapConfiguration throws appropriate errors."""
    # Empty denoiser
    try:
        GapConfiguration(denoiser="", drop_k=0, radius_fraction=0.75)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "denoiser cannot be empty" in str(e)
    
    # Negative drop_k
    try:
        GapConfiguration(denoiser="NONE", drop_k=-1, radius_fraction=0.75)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "drop_k must be >= 0" in str(e)
    
    # Invalid radius_fraction
    try:
        GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "radius_fraction must be in [0.0, 1.0]" in str(e)
    
    print("✓ GapConfiguration validation errors")


def test_gap_configuration_parsing_errors():
    """Test that GapConfiguration.from_string throws appropriate errors."""
    # Too few parts
    try:
        GapConfiguration.from_string("NONE_k=0")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have at least 3 parts" in str(e)
    
    # Invalid drop_k format
    try:
        GapConfiguration.from_string("NONE_drop=0_radiusfrac=0.75")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "drop_k part must start with 'k='" in str(e)
    
    # Invalid drop_k value
    try:
        GapConfiguration.from_string("NONE_k=abc_radiusfrac=0.75")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "drop_k must be an integer" in str(e)
    
    # Invalid radius_fraction format
    try:
        GapConfiguration.from_string("NONE_k=0_radius=0.75")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "radius_fraction part must start with 'radiusfrac='" in str(e)
    
    # Invalid radius_fraction value
    try:
        GapConfiguration.from_string("NONE_k=0_radiusfrac=abc")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "radius_fraction must be a float" in str(e)
    
    print("✓ GapConfiguration parsing errors")


def test_llm_configuration_valid():
    """Test valid LLMConfiguration creation and string conversion."""
    config = LLMConfiguration(num_articles=5, num_books=3)
    
    assert config.num_articles == 5
    assert config.num_books == 3
    assert config.to_string() == "articles_5_books_3"
    print("✓ LLMConfiguration valid creation and string conversion")


def test_llm_configuration_parsing():
    """Test LLMConfiguration parsing from string."""
    llm_str = "articles_20_books_10"
    config = LLMConfiguration.from_string(llm_str)
    
    assert config.num_articles == 20
    assert config.num_books == 10
    assert config.to_string() == llm_str
    print("✓ LLMConfiguration parsing from string")


def test_llm_configuration_round_trip():
    """Test that LLM string conversion is reversible."""
    original = LLMConfiguration(num_articles=15, num_books=0)
    llm_str = original.to_string()
    parsed = LLMConfiguration.from_string(llm_str)
    
    assert parsed == original
    print("✓ LLMConfiguration round-trip conversion")


def test_llm_configuration_validation_errors():
    """Test that LLMConfiguration throws appropriate errors."""
    # Zero articles
    try:
        LLMConfiguration(num_articles=0, num_books=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_articles must be > 0" in str(e)
    
    # Negative articles
    try:
        LLMConfiguration(num_articles=-1, num_books=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_articles must be > 0" in str(e)
    
    # Negative books
    try:
        LLMConfiguration(num_articles=5, num_books=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_books must be >= 0" in str(e)
    
    print("✓ LLMConfiguration validation errors")


def test_llm_configuration_parsing_errors():
    """Test that LLMConfiguration.from_string throws appropriate errors."""
    # Wrong format
    try:
        LLMConfiguration.from_string("art_5_bks_3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match format 'articles_X_books_Y'" in str(e)
    
    # Missing parts
    try:
        LLMConfiguration.from_string("articles_5")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match format 'articles_X_books_Y'" in str(e)
    
    # Non-numeric values
    try:
        LLMConfiguration.from_string("articles_abc_books_3")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Article and book counts must be integers" in str(e)
    
    print("✓ LLMConfiguration parsing errors")


if __name__ == "__main__":
    # Run basic tests manually
    test_gap_configuration_valid()
    test_gap_configuration_parsing()
    test_gap_configuration_round_trip()
    test_gap_configuration_validation_errors()
    test_gap_configuration_parsing_errors()
    test_llm_configuration_valid()
    test_llm_configuration_parsing()
    test_llm_configuration_round_trip()
    test_llm_configuration_validation_errors()
    test_llm_configuration_parsing_errors()
    print("\n🎉 All Step 1 tests passed!")


def test_gap_configuration_valid():
    """Test valid GapConfiguration creation and string conversion."""
    config = GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=0.75)
    
    assert config.denoiser == "NONE"
    assert config.drop_k == 0
    assert config.radius_fraction == 0.75
    assert config.to_string() == "NONE_k=0_radiusfrac=0.75"


def test_gap_configuration_parsing():
    """Test GapConfiguration parsing from string."""
    config_str = "LPP_k=5_radiusfrac=0.50"
    config = GapConfiguration.from_string(config_str)
    
    assert config.denoiser == "LPP"
    assert config.drop_k == 5
    assert config.radius_fraction == 0.50
    assert config.to_string() == config_str


def test_gap_configuration_round_trip():
    """Test that string conversion is reversible."""
    original = GapConfiguration(denoiser="TEST", drop_k=10, radius_fraction=0.33)
    config_str = original.to_string()
    parsed = GapConfiguration.from_string(config_str)
    
    assert parsed == original


def test_gap_configuration_validation_errors():
    """Test that GapConfiguration throws appropriate errors."""
    # Empty denoiser
    with pytest.raises(ValueError, match="denoiser cannot be empty"):
        GapConfiguration(denoiser="", drop_k=0, radius_fraction=0.75)
    
    # Negative drop_k
    with pytest.raises(ValueError, match="drop_k must be >= 0"):
        GapConfiguration(denoiser="NONE", drop_k=-1, radius_fraction=0.75)
    
    # Invalid radius_fraction
    with pytest.raises(ValueError, match="radius_fraction must be in \\[0.0, 1.0\\]"):
        GapConfiguration(denoiser="NONE", drop_k=0, radius_fraction=1.5)


def test_gap_configuration_parsing_errors():
    """Test that GapConfiguration.from_string throws appropriate errors."""
    # Too few parts
    with pytest.raises(ValueError, match="must have at least 3 parts"):
        GapConfiguration.from_string("NONE_k=0")
    
    # Invalid drop_k format
    with pytest.raises(ValueError, match="drop_k part must start with 'k='"):
        GapConfiguration.from_string("NONE_drop=0_radiusfrac=0.75")
    
    # Invalid drop_k value
    with pytest.raises(ValueError, match="drop_k must be an integer"):
        GapConfiguration.from_string("NONE_k=abc_radiusfrac=0.75")
    
    # Invalid radius_fraction format
    with pytest.raises(ValueError, match="radius_fraction part must start with 'radiusfrac='"):
        GapConfiguration.from_string("NONE_k=0_radius=0.75")
    
    # Invalid radius_fraction value
    with pytest.raises(ValueError, match="radius_fraction must be a float"):
        GapConfiguration.from_string("NONE_k=0_radiusfrac=abc")


def test_llm_configuration_valid():
    """Test valid LLMConfiguration creation and string conversion."""
    config = LLMConfiguration(num_articles=5, num_books=3)
    
    assert config.num_articles == 5
    assert config.num_books == 3
    assert config.to_string() == "articles_5_books_3"


def test_llm_configuration_parsing():
    """Test LLMConfiguration parsing from string."""
    llm_str = "articles_20_books_10"
    config = LLMConfiguration.from_string(llm_str)
    
    assert config.num_articles == 20
    assert config.num_books == 10
    assert config.to_string() == llm_str


def test_llm_configuration_round_trip():
    """Test that LLM string conversion is reversible."""
    original = LLMConfiguration(num_articles=15, num_books=0)
    llm_str = original.to_string()
    parsed = LLMConfiguration.from_string(llm_str)
    
    assert parsed == original


def test_llm_configuration_validation_errors():
    """Test that LLMConfiguration throws appropriate errors."""
    # Zero articles
    with pytest.raises(ValueError, match="num_articles must be > 0"):
        LLMConfiguration(num_articles=0, num_books=3)
    
    # Negative articles
    with pytest.raises(ValueError, match="num_articles must be > 0"):
        LLMConfiguration(num_articles=-1, num_books=3)
    
    # Negative books
    with pytest.raises(ValueError, match="num_books must be >= 0"):
        LLMConfiguration(num_articles=5, num_books=-1)


def test_llm_configuration_parsing_errors():
    """Test that LLMConfiguration.from_string throws appropriate errors."""
    # Wrong format
    with pytest.raises(ValueError, match="must match format 'articles_X_books_Y'"):
        LLMConfiguration.from_string("art_5_bks_3")
    
    # Missing parts
    with pytest.raises(ValueError, match="must match format 'articles_X_books_Y'"):
        LLMConfiguration.from_string("articles_5")
    
        # Non-numeric values (regex catches this before int parsing)
        with pytest.raises(ValueError, match="must match format 'articles_X_books_Y'"):
            LLMConfiguration.from_string("articles_abc_books_3")
if __name__ == "__main__":
    # Run basic tests manually if pytest not available
    test_gap_configuration_valid()
    test_gap_configuration_parsing()
    test_gap_configuration_round_trip()
    test_llm_configuration_valid()
    test_llm_configuration_parsing()
    test_llm_configuration_round_trip()
    print("All basic tests passed!")
