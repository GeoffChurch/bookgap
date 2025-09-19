"""Abstract base class for batch proposal generators.

Defines the interface for batch-first proposal generation, replacing
individual request patterns with OpenAI batch API integration.
"""

from abc import ABC, abstractmethod
from typing import List

from bookgap.configs.llm.llm_config import LLMConfig
from ..protocols import ProposalContext, BatchRequestData, PromptPair


class BatchProposalGenerator(ABC):
    """Abstract base class for batch proposal generators.
    
    Designed for batch processing of multiple contexts into batch API requests.
    """
    
    def generate_batch_requests(self, contexts: List[ProposalContext]) -> List[BatchRequestData]:
        """Generate batch API requests from proposal contexts.
        
        This is the main entry point for batch generation. Processes multiple
        contexts and returns batch API request structures ready for submission.
        
        Args:
            contexts: List of proposal contexts to generate requests for
            
        Returns:
            List of batch API request data structures
            
        Raises:
            ProposalGenerationError: If batch request generation fails
        """
        requests = []
        for context in contexts:
            prompts = self._build_prompt(context)  # Returns PromptPair
            request = self._create_request_from_prompt(prompts, context)
            requests.append(request)
        
        return requests
    
    @abstractmethod
    def _build_prompt(self, context: ProposalContext) -> PromptPair:
        """Build system and user prompts for a single context.
        
        This method extracts prompt building logic into a clean interface.
        Subclasses implement provider-specific prompt formatting.
        
        Args:
            context: The proposal context containing research data
            
        Returns:
            PromptPair with system_prompt and user_prompt
            
        Raises:
            ProposalGenerationError: If prompt building fails
        """
        ...
    
    @abstractmethod  
    def _create_request_from_prompt(self, prompts: PromptPair, context: ProposalContext) -> BatchRequestData:
        """Create batch API request from system and user prompts.
        
        This method handles provider-specific request formatting including
        message structure, model parameters, and structured output configuration.
        
        Args:
            prompts: PromptPair containing system_prompt and user_prompt
            context: The original proposal context for custom ID generation
            
        Returns:
            Batch API request data structure ready for submission
            
        Raises:
            ProposalGenerationError: If request creation fails
        """
        ...
