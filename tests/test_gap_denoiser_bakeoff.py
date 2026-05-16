import numpy as np
import pytest


def test_setup_projection_signature_accepts_basis():
    """setup() must accept a `projection` kwarg of shape (d, n_keep)."""
    import inspect
    from artifacts_scripts.gap_static_seed_diagnostic import setup
    sig = inspect.signature(setup)
    assert "projection" in sig.parameters
    assert sig.parameters["projection"].default is None
