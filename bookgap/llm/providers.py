"""
LLM provider types and related functionality.

This module defines the provider types supported by the LLM system.
"""

from enum import Enum


class ProviderType(Enum):
    """LLM provider types."""
    OPENAI = "openai"