"""Workflow functions for batch proposal generation.

Pure functions for the batch processing pipeline using Hydra's instantiation system.
"""

from typing import List

from hydra.utils import instantiate

from bookgap.configs.config import ExperimentConfig
from ..protocols import ProposalContext, BatchRequestData, ProposalGenerationError


def generate_batch_requests(contexts: List[ProposalContext], config: ExperimentConfig) -> List[BatchRequestData]:
    """Generate batch API requests from proposal contexts.
    
    Uses Hydra's _target_ pattern to instantiate the appropriate generator for the provider.
    
    Args:
        contexts: List of proposal contexts
        config: Experiment configuration with LLM settings
        
    Returns:
        List of batch API request data structures
        
    Raises:
        ProposalGenerationError: If generator instantiation or generation fails
        
    Example:
        ```python
        from hydra import compose, initialize
        from bookgap.llm.generators.workflow_functions import generate_batch_requests
        
        # Using Hydra config
        with initialize(config_path="../configs"):
            config = compose(config_name="production")
            requests = generate_batch_requests(contexts, config)
        
        # Manual config (for testing)
        config = ExperimentConfig(
            llm=LLMConfig(
                _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
                provider_type=ProviderType.OPENAI,
                model_name=OpenAIModel.GPT_4O_MINI,
                # ... other required fields
            ),
            # ... other config fields
        )
        requests = generate_batch_requests(contexts, config)
        ```
    """
    if not contexts:
        return []
    
    try:
        # Instantiate generator using Hydra's _target_ pattern
        # The LLM config contains _target_ field pointing to the generator class
        generator = instantiate(config.llm, _config=config)
        
        # Generate batch requests
        return generator.generate_batch_requests(contexts)
        
    except Exception as e:
        # Wrap any errors in ProposalGenerationError with context
        error_context = {
            "num_contexts": len(contexts),
            "provider": str(config.llm.provider_type),
            "model": str(config.llm.model_name),
            "target": config.llm._target_
        }
        
        # If it's already a ProposalGenerationError, preserve the original
        if isinstance(e, ProposalGenerationError):
            # Add workflow context to existing error context
            e.context.update(error_context)
            raise e
        else:
            # Wrap other exceptions
            raise ProposalGenerationError(
                f"Failed to generate batch requests: {str(e)}",
                context=error_context
            ) from e


def validate_batch_requests(requests: List[BatchRequestData]) -> None:
    """Validate that batch requests are properly formatted.
    
    Args:
        requests: List of batch request data structures
        
    Raises:
        ProposalGenerationError: If validation fails
    """
    if not requests:
        raise ProposalGenerationError(
            "No batch requests provided",
            context={"validation_step": "empty_check"}
        )
    
    errors = []
    custom_ids = set()
    
    for i, request in enumerate(requests):
        # Check for duplicate custom IDs
        if request.custom_id in custom_ids:
            errors.append(f"Duplicate custom_id found: {request.custom_id}")
        custom_ids.add(request.custom_id)
        
        # Check custom_id is not empty
        if not request.custom_id or not request.custom_id.strip():
            errors.append(f"Empty custom_id found at request {i}")
        
        # Check model is specified
        if not request.body.model:
            errors.append(f"Request {i}: model is not specified")
        
        # Check messages exist
        if not request.body.messages:
            errors.append(f"Request {i}: no messages provided")
        
        # Check message format
        for j, message in enumerate(request.body.messages):
            if message.role not in ["system", "user", "assistant"]:
                errors.append(f"Request {i}, Message {j}: invalid role '{message.role}'")
    
    if errors:
        raise ProposalGenerationError(
            f"Batch request validation failed: {'; '.join(errors)}",
            context={
                "num_requests": len(requests),
                "num_errors": len(errors),
                "validation_errors": errors
            }
        )


def get_supported_models() -> List[str]:
    """Get list of supported model names.
    
    Returns:
        List of supported OpenAI model names
    """
    from ..protocols import OpenAIModel
    return [model.value for model in OpenAIModel]
