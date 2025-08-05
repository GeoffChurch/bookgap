"""
Batch utilities for OpenAI Batch API operations.

This package contains utilities for managing OpenAI batch jobs, including:
- Creating and submitting batch requests
- Monitoring batch job status
- Downloading and processing results
- Extracting textbook proposals from batch results
"""

from .batch_operations import (
    monitor_batch_job,
    download_batch_results,
    extract_textbook_proposals,
    clean_latex_content,
    list_batch_jobs,
    get_batch_status,
    process_completed_batch,
    BATCH_DIR
)

__all__ = [
    'monitor_batch_job',
    'download_batch_results', 
    'extract_textbook_proposals',
    'clean_latex_content',
    'list_batch_jobs',
    'get_batch_status',
    'process_completed_batch',
    'BATCH_DIR'
]
