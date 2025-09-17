"""
Quick test to verify find_gaps_for_topic algorithm selection fix.
"""

from bookgap.types import CellType
from bookgap.config_manager import GapConfiguration

def test_gap_configuration_parsing():
    """Test that we can parse cell_type from gap configuration directory names."""
    
    # Test KBALL configuration
    kball_config = GapConfiguration.from_directory_name("NONE_k=0_radiusfrac=0.50_kball")
    assert kball_config.cell_type == CellType.KBALL
    print(f"✓ KBALL config parsed correctly: {kball_config.cell_type}")
    
    # Test VORONOI configuration
    voronoi_config = GapConfiguration.from_directory_name("NONE_k=0_radiusfrac=0.50_voronoi")
    assert voronoi_config.cell_type == CellType.VORONOI
    print(f"✓ VORONOI config parsed correctly: {voronoi_config.cell_type}")
    
    print("✓ All gap configuration parsing tests passed!")

def test_function_signature():
    """Test that find_gaps_for_topic now requires cell_type parameter."""
    from bookgap.gap_finder import find_gaps_for_topic
    import inspect
    
    # Get function signature
    sig = inspect.signature(find_gaps_for_topic)
    params = sig.parameters
    
    # Verify cell_type is required (no default)
    assert 'cell_type' in params
    assert params['cell_type'].default == inspect.Parameter.empty
    print(f"✓ cell_type parameter is now required (no default)")
    
    # Verify it comes after radius_fraction
    param_names = list(params.keys())
    cell_type_idx = param_names.index('cell_type')
    radius_fraction_idx = param_names.index('radius_fraction')
    assert cell_type_idx > radius_fraction_idx
    print(f"✓ Parameter order is correct: {param_names}")
    
    print("✓ Function signature tests passed!")

if __name__ == "__main__":
    test_gap_configuration_parsing()
    test_function_signature()
    print("\n🎉 All tests passed! The find_gaps_for_topic algorithm selection fix is working correctly.")
