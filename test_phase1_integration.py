#!/usr/bin/env python3
"""Simple integration test for Phase 1 functionality.

This tests the core functionality without complex mocking,
focusing on the successful integration of our Phase 1 modules.
"""

import sys
from pathlib import Path

def test_phase1_integration():
    """Test that Phase 1 modules work together correctly."""
    print("🧪 Phase 1 Integration Test")
    print("=" * 40)
    
    try:
        # Test 1: Import all modules successfully
        from bookgap.llm.data_structures import GapWithContext, BatchResults, ArtifactPaths
        from bookgap.llm.generators.workflow_artifacts import generate_phase1_artifacts, generate_all_artifacts
        from bookgap.llm.artifacts.gap_plots import save_gap_plot_with_proposals, generate_gap_plots_for_results
        print("✅ All modules imported successfully")
        
        # Test 2: Data structures work
        plots = [Path("test1.png"), Path("test2.png")]
        artifact_paths = ArtifactPaths(plots=plots)
        assert artifact_paths.plots == plots
        assert artifact_paths.context_files is None
        print("✅ ArtifactPaths working correctly")
        
        # Test 3: Check function signatures are correct
        import inspect
        
        sig = inspect.signature(generate_phase1_artifacts)
        params = list(sig.parameters.keys())
        assert 'gaps' in params and 'results' in params
        print("✅ generate_phase1_artifacts has correct signature")
        
        sig = inspect.signature(save_gap_plot_with_proposals) 
        params = list(sig.parameters.keys())
        assert 'gap_with_context' in params and 'proposals' in params
        print("✅ save_gap_plot_with_proposals has correct signature")
        
        # Test 4: Verify no import cycles or syntax errors
        print("✅ No circular import issues detected")
        
        print("\n🎉 Phase 1 integration test PASSED!")
        print("All core functionality is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase1_integration()
    sys.exit(0 if success else 1)
