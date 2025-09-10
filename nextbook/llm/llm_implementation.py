#!/usr/bin/env python3
"""
LLM implementation enum with direct class mapping.

This module defines the Implementation enum with direct references to generator classes.
Since this is within the LLM module, it can safely import the generators without 
creating circular dependencies.
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generators.openai_generator import OpenAIProposalGenerator
    from .generators.json_prompt_generator import JSONPromptProposalGenerator


class Implementation(Enum):
    """LLM implementation types with direct class mapping."""
    STRUCTURED = "structured"
    PROMPT = "prompt"


def get_generator(implementation: Implementation, config):
    """Get a generator instance for the given implementation and config.
    
    Args:
        implementation: The implementation type
        config: The analysis configuration
        
    Returns:
        An instance of the appropriate generator class
    """
    if implementation == Implementation.STRUCTURED:
        from .generators.openai_generator import OpenAIProposalGenerator
        return OpenAIProposalGenerator(config)
    elif implementation == Implementation.PROMPT:
        from .generators.json_prompt_generator import JSONPromptProposalGenerator
        return JSONPromptProposalGenerator(config)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
