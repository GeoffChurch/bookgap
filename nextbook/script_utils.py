"""
Shared utilities for script execution and main function patterns.

This module consolidates the common patterns found across all nextbook scripts:
- Standard logging setup with explicit args
- Main function execution with parsed arguments
- Script name derivation from file paths
"""
from pathlib import Path
from typing import Any, Callable, Dict

from nextbook.parse_args import get_explicit_args, parse_args
from nextbook.util import setup_logging


def run_script_main(main_func: Callable[..., Any], script_path: str, **extra_kwargs: Any) -> None:
    """
    Standard pattern for running a script's main function.
    
    Consolidates the common pattern:
    ```python
    if __name__ == "__main__":
        setup_logging(log_name=Path(__file__).stem, explicit_args=get_explicit_args())
        main(**vars(parse_args()), **extra_kwargs)
    ```
    
    Args:
        main_func: The main function to execute
        script_path: The __file__ path of the script (for logging name derivation)
        **extra_kwargs: Additional keyword arguments to pass to main function
    """
    setup_logging(log_name=Path(script_path).stem, explicit_args=get_explicit_args())
    main_func(**vars(parse_args()), **extra_kwargs)


def get_script_name(script_path: str) -> str:
    """
    Extract script name from file path for logging purposes.
    
    Args:
        script_path: The __file__ path of the script
        
    Returns:
        Script name stem (filename without extension)
    """
    return Path(script_path).stem
