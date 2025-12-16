"""Configuration for denoiser analysis."""

from dataclasses import dataclass, field
from typing import List, Optional

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.util import ARTIFACTS_DIR


@dataclass
class DenoiserAnalysisConfig:
    """Configuration for denoiser analysis.
    
    Analyzes and compares different denoising strategies on the AI cloud,
    computing metrics like SNR score, class separability, and locality preservation.
    """
    
    # List of denoisers to analyze (by name from WorkDenoisers enum)
    # If None, analyze all available denoisers
    denoisers: Optional[List[str]] = None
    """List of denoiser names to analyze. None = analyze all."""
    
    # List of denoisers to skip (blacklist)
    skip_denoisers: List[str] = field(
        default_factory=lambda: [
            "JAD_EQUAL",
            "JAD_HALF_L",
            "JAD_CCP",
            "JAD_LIP",
        ]
    )
    """List of denoiser names to skip. These are excluded even if denoisers=None."""
    
    drop_k: int = 1
    """Number of dimensions to drop during denoising."""
    
    max_works: Optional[int] = None
    """Maximum number of works to load. None = load all."""
    
    # Provide a safe default experiment composed of default sub-configs
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        )
    )
    """Experiment configuration (for embedder model selection)."""
    
    # SNR score weights
    snr_alpha: float = 1.0
    """Weight for between-class scatter (S_B) in SNR noise term."""
    
    snr_beta: float = 1.0
    """Weight for Laplacian scatter (S_L) in SNR noise term."""
    
    output_dir: str = field(default_factory=lambda: str(ARTIFACTS_DIR / "denoiser_analysis"))
    """Output directory for results."""
