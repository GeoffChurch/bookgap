"""Quick integration test for the LLM generators."""

from bookgap.config_manager import AnalysisConfiguration, GapConfiguration, LLMConfiguration
from bookgap.llm.llm_implementation import Implementation, get_generator
from bookgap.types import CellType

# Test generator registration
def test_registration():
    """Test that generators can be registered and retrieved."""
    print("Testing generator registration...")
    
    # Create test config
    gap_config = GapConfiguration(
        denoiser="NONE",
        drop_k=0,
        radius_fraction=0.5,
        cell_type=CellType.KBALL
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
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
    
    # Test getting generators
    try:
        structured_gen = get_generator(Implementation.STRUCTURED, config)
        print(f"✓ Structured generator: {type(structured_gen).__name__}")
    except Exception as e:
        print(f"✗ Structured generator failed: {e}")
    
    try:
        prompt_gen = get_generator(Implementation.PROMPT, config)
        print(f"✓ Prompt generator: {type(prompt_gen).__name__}")
    except Exception as e:
        print(f"✗ Prompt generator failed: {e}")
    
    print("Registration test complete!")

if __name__ == "__main__":
    test_registration()
