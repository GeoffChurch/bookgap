"""BookGap package

Core modules for gap analysis, configuration, results management, and utilities.

This file intentionally minimal to establish a regular (non-namespace) package
boundary for tools like mypy.
"""

__all__ = [
    # Modules commonly imported by external callers
    "config_manager",
    "results_manager",
    "gap_finder",
    "cloud",
    "util",
]
