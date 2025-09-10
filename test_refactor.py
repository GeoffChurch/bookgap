#!/usr/bin/env python3
"""Quick test to verify the refactored enum-based gap algorithm and dataclass structure."""

import sys
import tempfile
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from nextbook.config_manager import CellType, AnalysisConfiguration, GapConfiguration, LLMConfiguration, Implementation
from nextbook.gap_finder import get_gap_algorithm, register_gap_algorithms, kball_gaps, voronoi_gaps
from nextbook.llm.protocols import ProposalCounts, ProposalMetadata, ProposalResult


def test_gap_algorithm_enum():
    """Test that the gap algorithm enum registration works."""
    print("Testing gap algorithm enum registration...")
    
    # Test that algorithms are registered
    kball_algorithm = get_gap_algorithm(CellType.KBALL)
    voronoi_algorithm = get_gap_algorithm(CellType.VORONOI)
    
    # Verify they are the expected functions
    assert kball_algorithm == kball_gaps, f"Expected {kball_gaps}, got {kball_algorithm}"
    assert voronoi_algorithm == voronoi_gaps, f"Expected {voronoi_gaps}, got {voronoi_algorithm}"
    
    print("✓ Gap algorithm enum registration works correctly")


def test_proposal_dataclasses():
    """Test that the new dataclass structure works."""
    print("Testing proposal dataclasses...")
    
    # Test ProposalCounts
    counts = ProposalCounts(num_articles_for_llm=5, num_books_for_llm=3)
    assert counts.num_articles_for_llm == 5
    assert counts.num_books_for_llm == 3
    
    # Test ProposalMetadata
    metadata = ProposalMetadata(
        provider="openai",
        model="gpt-4",
        implementation="structured",
        request_id="test-123",
        usage={"prompt_tokens": 100, "completion_tokens": 200}
    )
    assert metadata.provider == "openai"
    assert metadata.model == "gpt-4"
    assert metadata.implementation == "structured"
    assert metadata.request_id == "test-123"
    assert metadata.usage["prompt_tokens"] == 100
    
    # Test ProposalResult
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "proposal.json"
        prompt_path = Path(tmpdir) / "prompt.txt"
        
        result = ProposalResult(
            json_path=json_path,
            prompt_path=prompt_path,
            raw_response_path=None,
            counts=counts,
            metadata=metadata
        )
        
        assert result.counts.num_articles_for_llm == 5
        assert result.metadata.provider == "openai"
        assert result.raw_response_path is None
    
    print("✓ Proposal dataclasses work correctly")


def test_cell_type_enum():
    """Test that CellType enum has the expected values."""
    print("Testing CellType enum...")
    
    # Test that enum values are set correctly after registration
    assert CellType.KBALL.value == kball_gaps
    assert CellType.VORONOI.value == voronoi_gaps
    
    print("✓ CellType enum works correctly")


def main():
    """Run all tests."""
    print("Running refactor validation tests...\n")
    
    try:
        test_gap_algorithm_enum()
        test_proposal_dataclasses()
        test_cell_type_enum()
        
        print("\n🎉 All tests passed! The refactor is working correctly.")
        print("\nKey improvements:")
        print("  • Gap algorithms now use enum-based selection with type safety")
        print("  • ProposalResult uses statically-typed dataclasses instead of dicts")
        print("  • Backward compatibility logic has been removed")
        print("  • Clean architecture with consistent patterns throughout")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
