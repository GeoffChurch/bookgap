"""
Unified Results Management Framework

This module provides a consolidated framework for managing results across all 
nextbook operations, replacing scattered Results classes with a unified system.

Key Components:
- BaseResults: Abstract base class for all result types
- ResultsConfig: Configuration for different result types  
- ResultsValidator: Data validation framework
- Specific result classes: GapResults, BookArticleResults
- Processing pipeline: ResultsProcessor, ResultsMerger, ResultsExporter
- Management system: ResultsFactory, ResultsManager, ResultsRegistry
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable, Literal
import pandas as pd
import numpy as np
import logging
from enum import Enum

# Import centralized configuration system
from .config_manager import AnalysisConfiguration
from .gap_directory_manager import GapDirectoryManager

LOGGER = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class ResultType(Enum):
    """Enumeration of supported result types."""
    GAP_ANALYSIS = "gap_analysis"
    BOOK_ARTICLE = "book_article"
    DATAFRAME = "dataframe"  # For arbitrary DataFrames like batch results
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Defines a validation rule for result data."""
    field_name: str
    rule_type: str  # 'type', 'range', 'custom'
    rule_params: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    

@dataclass
class ResultsConfig:
    """Configuration for different result types."""
    result_type: ResultType
    key_columns: List[str]
    dependent_columns: List[str]
    required_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    csv_dialect: Dict[str, Any] = field(default_factory=dict)
    export_formats: List[str] = field(default_factory=lambda: ['csv'])
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.required_columns:
            self.required_columns = self.key_columns + self.dependent_columns
        
        # Set default CSV dialect
        if not self.csv_dialect:
            self.csv_dialect = {
                'index': False,
                'float_format': '%.6f'
            }
    
    @property
    def all_columns(self) -> List[str]:
        """Get all columns in the correct order."""
        return self.key_columns + self.dependent_columns


class ResultsValidator:
    """Data validation framework with comprehensive error reporting."""
    
    def __init__(self, config: ResultsConfig):
        self.config = config
        self._validators = self._build_validators()
    
    def _build_validators(self) -> Dict[str, List[Callable]]:
        """Build validation functions from configuration rules."""
        validators = {}
        
        for rule in self.config.validation_rules:
            field_name = rule.field_name
            if field_name not in validators:
                validators[field_name] = []
            
            if rule.rule_type == 'type':
                expected_type = rule.rule_params.get('expected_type')
                if expected_type is not None:
                    validators[field_name].append(
                        self._create_type_validator(expected_type, rule.error_message)
                    )
            elif rule.rule_type == 'range':
                validators[field_name].append(
                    self._create_range_validator(
                        rule.rule_params.get('min_val'), 
                        rule.rule_params.get('max_val'),
                        rule.error_message
                    )
                )
            elif rule.rule_type == 'custom':
                custom_func = rule.rule_params.get('validator_func')
                values = rule.rule_params.get('values')
                if custom_func:
                    validators[field_name].append(custom_func)
                elif values:
                    validators[field_name].append(
                        self._create_values_validator(values, rule.error_message)
                    )
        
        return validators
    
    def _create_type_validator(self, expected_type: Type, error_message: str) -> Callable:
        """Create a type validation function."""
        def validate_type(value):
            if not isinstance(value, expected_type) and not pd.isna(value):
                raise ValidationError(error_message or f"Expected type {expected_type.__name__}, got {type(value).__name__}")
        return validate_type
    
    def _create_range_validator(self, min_val: Optional[float], max_val: Optional[float], error_message: str) -> Callable:
        """Create a range validation function."""
        def validate_range(value):
            if pd.isna(value):
                return
            if min_val is not None and value < min_val:
                raise ValidationError(error_message or f"Value {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                raise ValidationError(error_message or f"Value {value} above maximum {max_val}")
        return validate_range
    
    def _create_values_validator(self, allowed_values: List[Any], error_message: str) -> Callable:
        """Create a values validation function for allowed values."""
        def validate_values(value):
            if pd.isna(value):
                return
            if value not in allowed_values:
                raise ValidationError(error_message or f"Value '{value}' not in allowed values: {allowed_values}")
        return validate_values
    
    def validate_row(self, row: Dict[str, Any]) -> List[str]:
        """Validate a single row of data."""
        errors = []
        
        # Check required columns
        missing_cols = [col for col in self.config.required_columns if col not in row]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Run field-specific validators
        for field_name, validators in self._validators.items():
            if field_name in row:
                try:
                    for validator in validators:
                        validator(row[field_name])
                except ValidationError as e:
                    errors.append(f"Field '{field_name}': {str(e)}")
        
        return errors
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """Validate entire DataFrame."""
        all_errors = {}
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            errors = self.validate_row(row_dict)
            if errors:
                all_errors[idx] = errors
        
        return all_errors


class BaseResults(ABC):
    """Abstract base class defining common interface for all result types."""
    
    def __init__(self, config: ResultsConfig, csv_path: Optional[Path] = None):
        self.config = config
        self.csv_path = csv_path
        self.validator = ResultsValidator(config)
        self._df = pd.DataFrame(columns=config.all_columns)
        
        if csv_path and csv_path.exists():
            self._load_from_csv()
    
    def _load_from_csv(self) -> None:
        """Load data from CSV file."""
        if self.csv_path is None:
            return
            
        try:
            # Use column types if specified - pandas expects specific types
            dtype = None
            if self.config.column_types:
                # Convert string type names to pandas dtypes
                dtype = {}
                for col, col_type in self.config.column_types.items():
                    if col_type in ['string', 'str']:
                        dtype[col] = 'string'
                    elif col_type in ['int64', 'int']:
                        dtype[col] = 'int64'
                    elif col_type in ['float64', 'float']:
                        dtype[col] = 'float64'
                    else:
                        dtype[col] = col_type
            
            self._df = pd.read_csv(self.csv_path, dtype=dtype)
            LOGGER.info(f"Loaded {len(self._df)} rows from {self.csv_path}")
            
            # Validate loaded data
            errors = self.validator.validate_dataframe(self._df)
            if errors:
                LOGGER.warning(f"Validation errors in loaded data: {len(errors)} rows with issues")
                for row_idx, row_errors in list(errors.items())[:5]:  # Show first 5 errors
                    LOGGER.warning(f"Row {row_idx}: {row_errors}")
                    
        except Exception as e:
            LOGGER.error(f"Failed to load CSV from {self.csv_path}: {e}")
            self._df = pd.DataFrame(columns=self.config.all_columns)
    
    @abstractmethod 
    def add_row(self, *args, **kwargs) -> None:
        """Add a row to the results. Implementation depends on result type."""
        pass
    
    def to_csv(self, path: Optional[Path] = None) -> None:
        """Export results to CSV."""
        export_path = path or self.csv_path
        if export_path is None:
            raise ValueError("No CSV path specified")
        
        # Ensure directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export with configured dialect
        self._df.to_csv(export_path, **self.config.csv_dialect)
        LOGGER.info(f"Saved {len(self._df)} rows to {export_path}")
    
    def from_csv(self, path: Path) -> None:
        """Load results from CSV."""
        self.csv_path = path
        self._load_from_csv()
    
    def validate(self) -> Dict[int, List[str]]:
        """Validate all data in the results."""
        return self.validator.validate_dataframe(self._df)
    
    def merge(self, other: 'BaseResults', how: Literal['inner', 'outer', 'left', 'right', 'cross'] = 'outer') -> 'BaseResults':
        """Merge with another results object."""
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot merge different result types: {type(self)} and {type(other)}")
        
        # Create new instance
        merged = self.__class__(self.config)
        merged._df = pd.merge(self._df, other._df, on=self.config.key_columns, how=how, suffixes=('', '_other'))
        
        return merged
    
    def get_df(self) -> pd.DataFrame:
        """Get a copy of the DataFrame."""
        return self._df.copy()
    
    def filter_rows(self, **filters) -> pd.DataFrame:
        """Filter rows based on column values."""
        df_filtered = self._df.copy()
        
        for column, value in filters.items():
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == value]
        
        return df_filtered
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the results."""
        summary = {
            'total_rows': len(self._df),
            'columns': list(self._df.columns),
            'memory_usage': self._df.memory_usage(deep=True).sum(),
            'null_counts': self._df.isnull().sum().to_dict()
        }
        
        # Add numeric column statistics
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = self._df[numeric_cols].describe().to_dict()
        
        return summary
    
    def __len__(self) -> int:
        """Return number of rows."""
        return len(self._df)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(rows={len(self._df)}, type={self.config.result_type.value})"


# Configuration instances for existing result types
GAP_ANALYSIS_CONFIG = ResultsConfig(
    result_type=ResultType.GAP_ANALYSIS,
    key_columns=['denoiser', 'drop_k', 'radius_fraction', 'gap_idx', 'topic_id', 'topic_name'],
    dependent_columns=['num_articles', 'num_books', 'gap_size', 'gap_size_percentile', 'gap_zscore', 'shapiro_log10_pval', 'num_articles_for_llm', 'num_books_for_llm'],
    column_types={
        'denoiser': 'string',
        'drop_k': 'int64', 
        'radius_fraction': 'float64',
        'gap_idx': 'int64',
        'topic_id': 'string',
        'topic_name': 'string',
        'num_articles': 'int64',
        'num_books': 'int64', 
        'gap_size': 'int64',
        'gap_size_percentile': 'float64',
        'gap_zscore': 'float64',
        'shapiro_log10_pval': 'float64',
        'num_articles_for_llm': 'int64',
        'num_books_for_llm': 'int64'
    },
    validation_rules=[
        ValidationRule('num_articles', 'range', {'min_val': 0}, "Number of articles must be non-negative"),
        ValidationRule('num_books', 'range', {'min_val': 0}, "Number of books must be non-negative"),
        ValidationRule('gap_size', 'range', {'min_val': 0}, "Gap size must be non-negative"),
        ValidationRule('radius_fraction', 'range', {'min_val': 0, 'max_val': 1}, "Radius fraction must be between 0 and 1"),
        ValidationRule('num_articles_for_llm', 'range', {'min_val': 0}, "Number of articles for LLM must be non-negative"),
        ValidationRule('num_books_for_llm', 'range', {'min_val': 0}, "Number of books for LLM must be non-negative")
    ]
)

# New configuration with cell_type support for AnalysisConfiguration
CONSOLIDATED_GAP_ANALYSIS_CONFIG = ResultsConfig(
    result_type=ResultType.GAP_ANALYSIS,
    key_columns=['denoiser', 'drop_k', 'radius_fraction', 'cell_type', 'gap_idx', 'topic_id', 'topic_name'],
    dependent_columns=[
    # Actual totals in the topic
    'num_articles',
    'num_books',
    
    'gap_size', 'gap_size_percentile', 'gap_zscore', 'shapiro_log10_pval',
    
    # Requested prompt counts from AnalysisConfiguration
    'requested_num_prompt_articles',
    'requested_num_prompt_books',
    
    # Actual prompt counts used (currently min of requested vs available)
    'actual_num_prompt_articles',
    'actual_num_prompt_books'
    ],
    column_types={
        'denoiser': 'string',
        'drop_k': 'int64', 
        'radius_fraction': 'float64',
        'cell_type': 'string',  # NEW COLUMN
        'gap_idx': 'int64',
        'topic_id': 'string',
        'topic_name': 'string',
        'num_articles': 'int64',
        'num_books': 'int64', 
        'gap_size': 'int64',
        'gap_size_percentile': 'float64',
        'gap_zscore': 'float64',
        'shapiro_log10_pval': 'float64',
    'requested_num_prompt_articles': 'int64',
    'requested_num_prompt_books': 'int64',
    'actual_num_prompt_articles': 'int64',
    'actual_num_prompt_books': 'int64'
    },
    validation_rules=[
        ValidationRule('cell_type', 'custom', {'values': ['kball', 'voronoi']}, "Cell type must be 'kball' or 'voronoi'"),
        ValidationRule('num_articles', 'range', {'min_val': 0}, "Number of articles must be non-negative"),
        ValidationRule('num_books', 'range', {'min_val': 0}, "Number of books must be non-negative"),
        ValidationRule('gap_size', 'range', {'min_val': 0}, "Gap size must be non-negative"),
        ValidationRule('radius_fraction', 'range', {'min_val': 0, 'max_val': 1}, "Radius fraction must be between 0 and 1"),
        ValidationRule('requested_num_prompt_articles', 'range', {'min_val': 0}, "Requested prompt articles must be non-negative"),
        ValidationRule('requested_num_prompt_books', 'range', {'min_val': 0}, "Requested prompt books must be non-negative"),
        ValidationRule('actual_num_prompt_articles', 'range', {'min_val': 0}, "Actual prompt articles must be non-negative"),
        ValidationRule('actual_num_prompt_books', 'range', {'min_val': 0}, "Actual prompt books must be non-negative")
    ]
)

BOOK_ARTICLE_CONFIG = ResultsConfig(
    result_type=ResultType.BOOK_ARTICLE,
    key_columns=['topic_name', 'denoiser', 'drop_k'],
    dependent_columns=[],  # Will be extended based on actual usage
    validation_rules=[
        ValidationRule('drop_k', 'range', {'min_val': 0}, "Drop k must be non-negative")
    ]
)


class GapResults(BaseResults):
    """Results manager for gap analysis with consolidated AnalysisConfiguration."""
    
    def __init__(self, analysis_config: AnalysisConfiguration):
        """
        Initialize gap analysis results manager.
        
        Args:
            analysis_config: AnalysisConfiguration object containing all configuration
        """
        if not isinstance(analysis_config, AnalysisConfiguration):
            raise TypeError(f"GapResults only accepts AnalysisConfiguration, got {type(analysis_config).__name__}")
        
        # Store the analysis configuration 
        self.analysis_config = analysis_config
        
        # Use LLM-specific path to prevent overwrites between configurations
        csv_path = GapDirectoryManager.get_gap_analysis_file(analysis_config)
        
        # Initialize with consolidated gap analysis schema
        super().__init__(CONSOLIDATED_GAP_ANALYSIS_CONFIG, csv_path)
    
    def add_row(self, topic_name: str, topic_id: str, gap_idx: int, 
                gap_size: int, gap_size_percentile: float, 
                gap_zscore: float, shapiro_log10_pval: float,
                num_articles: int, num_books: int,
                actual_num_prompt_articles: int, actual_num_prompt_books: int) -> None:
        """Add a gap analysis row."""
        # Build row from AnalysisConfiguration
        new_row = {
            'denoiser': self.analysis_config.gap_config.denoiser,
            'drop_k': self.analysis_config.gap_config.drop_k,
            'radius_fraction': self.analysis_config.gap_config.radius_fraction,
            'cell_type': self.analysis_config.gap_config.cell_type,
            'gap_idx': gap_idx,
            'topic_id': topic_id,
            'topic_name': topic_name,
            'num_articles': num_articles,
            'num_books': num_books,
            'gap_size': gap_size,
            'gap_size_percentile': gap_size_percentile,
            'gap_zscore': gap_zscore,
            'shapiro_log10_pval': shapiro_log10_pval,
            'requested_num_prompt_articles': self.analysis_config.llm_config.requested_num_prompt_articles,
            'requested_num_prompt_books': self.analysis_config.llm_config.requested_num_prompt_books,
            'actual_num_prompt_articles': actual_num_prompt_articles,
            'actual_num_prompt_books': actual_num_prompt_books
        }
        
        # Validate the row
        errors = self.validator.validate_row(new_row)
        if errors:
            raise ValidationError(f"Row validation failed: {errors}")
        
        new_row_df = pd.DataFrame([new_row])[self.config.all_columns]
        LOGGER.info(f"Adding gap analysis row for topic '{topic_name}' (gap {gap_idx})")
        
        # Check for existing row and update or append
        if len(self._df) > 0:
            mask = pd.Series([True] * len(self._df))
            for col in self.config.key_columns:
                mask &= self._df[col] == new_row[col]
            
            if mask.any():
                LOGGER.warning(f"Row already exists, updating: {new_row['topic_name']}")
                self._df.loc[mask, self.config.dependent_columns] = new_row_df[self.config.dependent_columns].values[0]
            else:
                # Handle empty DataFrame concatenation properly
                if self._df.empty:
                    self._df = new_row_df.copy()
                else:
                    self._df = pd.concat([self._df, new_row_df], ignore_index=True)
        else:
            # Initialize with the first row
            self._df = new_row_df.copy()
        
        # Auto-save if path is set
        if self.csv_path:
            self.to_csv()
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the results as a pandas DataFrame."""
        return self.get_df()
    
    @classmethod
    def load_from_csv(cls, csv_path: Path, config: AnalysisConfiguration) -> 'GapResults':
        """Load GapResults from CSV file with validation.
        
        Args:
            csv_path: Path to CSV file
            config: Configuration object for validation
            
        Returns:
            GapResults instance loaded from CSV
            
        Raises:
            ValueError: If CSV format is incompatible with configuration
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV and check for required columns
        df = pd.read_csv(csv_path)
        
        # Validate CSV format - must have cell_type column for consolidated format
        if 'cell_type' not in df.columns:
            raise ValueError("CSV missing required 'cell_type' column for AnalysisConfiguration")
        
        # Create results instance and load data
        results = cls(config)
        results._df = df
        
        return results
    
    # Gap-specific ranking and statistical methods
    def get_ranked_gaps(self, ranking_method: str = 'gap_size', ascending: bool = False) -> pd.DataFrame:
        """Get gaps ranked by specified method."""
        df = self.get_df()
        
        if ranking_method == 'gap_size':
            return df.sort_values('gap_size', ascending=ascending)
        elif ranking_method == 'gap_score':
            # Custom gap score: weighted combination of gap size and statistical significance
            if 'gap_zscore' in df.columns:
                # Combine gap size percentile and z-score for ranking
                df_ranked = df.copy()
                df_ranked['gap_score'] = (
                    0.6 * df_ranked['gap_size_percentile'].fillna(0) + 
                    0.4 * (df_ranked['gap_zscore'].fillna(0) / df_ranked['gap_zscore'].std())
                )
                return df_ranked.sort_values('gap_score', ascending=ascending)
            else:
                return df.sort_values('gap_size_percentile', ascending=ascending)
        elif ranking_method == 'statistical_significance':
            # Rank by z-score and p-value combination
            df_ranked = df.copy()
            if 'gap_zscore' in df.columns and 'shapiro_log10_pval' in df.columns:
                # Higher z-score and lower p-value indicate more significant gaps
                df_ranked['significance_score'] = (
                    df_ranked['gap_zscore'].fillna(0) - 
                    df_ranked['shapiro_log10_pval'].fillna(0)
                )
                return df_ranked.sort_values('significance_score', ascending=ascending)
            else:
                return df.sort_values('gap_zscore', ascending=ascending)
        else:
            # Default to sorting by the specified column if it exists
            if ranking_method in df.columns:
                return df.sort_values(ranking_method, ascending=ascending)
            else:
                LOGGER.warning(f"Unknown ranking method: {ranking_method}, using gap_size")
                return df.sort_values('gap_size', ascending=ascending)
    
    def get_gap_size_distribution(self) -> Dict[str, Any]:
        """Get statistical distribution of gap sizes."""
        df = self.get_df()
        if df.empty or 'gap_size' not in df.columns:
            return {}
        
        gap_sizes = df['gap_size'].dropna()
        return {
            'count': len(gap_sizes),
            'mean': float(gap_sizes.mean()),
            'std': float(gap_sizes.std()),
            'min': int(gap_sizes.min()),
            'max': int(gap_sizes.max()),
            'median': float(gap_sizes.median()),
            'q25': float(gap_sizes.quantile(0.25)),
            'q75': float(gap_sizes.quantile(0.75)),
            'skewness': gap_sizes.skew(),
            'kurtosis': gap_sizes.kurtosis()
        }
    
    def get_topic_summaries(self) -> pd.DataFrame:
        """Get summary statistics per topic."""
        df = self.get_df()
        if df.empty:
            return pd.DataFrame()
        
        # Group by topic and calculate statistics
        topic_stats = df.groupby(['topic_id', 'topic_name']).agg({
            'gap_size': ['count', 'mean', 'std', 'min', 'max'],
            'gap_size_percentile': ['mean', 'std'],
            'gap_zscore': ['mean', 'std'],
            'num_articles': 'first',  # Should be same for all gaps in topic
            'num_books': 'first'
        }).round(3)
        
        # Flatten column names
        topic_stats.columns = ['_'.join(col).strip() for col in topic_stats.columns]
        topic_stats = topic_stats.reset_index()
        
        return topic_stats
    
    def get_configuration_comparison(self) -> pd.DataFrame:
        """Compare gap statistics across different configurations."""
        df = self.get_df()
        if df.empty:
            return pd.DataFrame()
        
        # Group by configuration parameters
        config_stats = df.groupby(['denoiser', 'drop_k', 'radius_fraction']).agg({
            'gap_size': ['count', 'mean', 'std'],
            'gap_size_percentile': ['mean', 'std'],  
            'gap_zscore': ['mean', 'std'],
            'topic_id': 'nunique'  # Number of unique topics
        }).round(3)
        
        # Flatten and clean column names
        config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns]
        config_stats = config_stats.reset_index()
        config_stats.rename(columns={'topic_id_nunique': 'num_topics'}, inplace=True)
        
        return config_stats
    
    def identify_outlier_gaps(self, method: str = 'zscore', threshold: float = 2.0) -> pd.DataFrame:
        """Identify outlier gaps using statistical methods."""
        df = self.get_df()
        if df.empty or 'gap_size' not in df.columns:
            return pd.DataFrame()
        
        df_outliers = df.copy()
        
        if method == 'zscore' and 'gap_zscore' in df.columns:
            # Use pre-calculated z-scores
            df_outliers['is_outlier'] = abs(df_outliers['gap_zscore']) > threshold
        elif method == 'iqr':
            # Use IQR method
            Q1 = df_outliers['gap_size'].quantile(0.25)
            Q3 = df_outliers['gap_size'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_outliers['is_outlier'] = (
                (df_outliers['gap_size'] < lower_bound) | 
                (df_outliers['gap_size'] > upper_bound)
            )
        else:
            # Simple threshold on gap size
            mean_gap = df_outliers['gap_size'].mean()
            std_gap = df_outliers['gap_size'].std()
            df_outliers['is_outlier'] = abs(df_outliers['gap_size'] - mean_gap) > threshold * std_gap
        
        return df_outliers[df_outliers['is_outlier']].copy()
    
    def export_gap_report(self, output_path: Path) -> None:
        """Export comprehensive gap analysis report."""
        report_lines = []
        
        report_lines.append("GAP ANALYSIS COMPREHENSIVE REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated at: {pd.Timestamp.now()}")
        report_lines.append(f"Configuration: {self.analysis_config.gap_config.denoiser}, drop_k={self.analysis_config.gap_config.drop_k}, radius_fraction={self.analysis_config.gap_config.radius_fraction}")
        report_lines.append(f"Cell Type: {self.analysis_config.gap_config.cell_type}")
        report_lines.append(f"Articles: {self.analysis_config.llm_config.requested_num_prompt_articles}, Books: {self.analysis_config.llm_config.requested_num_prompt_books}")
        report_lines.append("")
        
        # Overall statistics
        df = self.get_df()
        if not df.empty:
            report_lines.append("OVERALL STATISTICS")
            report_lines.append("-" * 25)
            report_lines.append(f"Total gaps found: {len(df)}")
            report_lines.append(f"Unique topics: {df['topic_id'].nunique()}")
            
            # Gap size distribution
            gap_dist = self.get_gap_size_distribution()
            if gap_dist:
                report_lines.append(f"Average gap size: {gap_dist['mean']:.2f} ± {gap_dist['std']:.2f}")
                report_lines.append(f"Gap size range: {gap_dist['min']} - {gap_dist['max']}")
                report_lines.append(f"Median gap size: {gap_dist['median']:.1f}")
            report_lines.append("")
            
            # Top gaps
            top_gaps = self.get_ranked_gaps('gap_score').head(10)
            report_lines.append("TOP 10 GAPS BY SCORE")
            report_lines.append("-" * 25)
            for idx, row in top_gaps.iterrows():
                report_lines.append(f"{row['topic_name']} (ID: {row['topic_id']})")
                report_lines.append(f"  Gap size: {row['gap_size']}, Percentile: {row.get('gap_size_percentile', 'N/A'):.3f}")
                if not pd.isna(row.get('gap_zscore')):
                    report_lines.append(f"  Z-score: {row['gap_zscore']:.3f}")
                report_lines.append("")
            
            # Topic summaries
            topic_summaries = self.get_topic_summaries()
            if not topic_summaries.empty:
                report_lines.append("TOPIC SUMMARIES")
                report_lines.append("-" * 20)
                for idx, row in topic_summaries.iterrows():
                    report_lines.append(f"{row['topic_name']} ({row['gap_size_count']} gaps)")
                    report_lines.append(f"  Avg gap size: {row['gap_size_mean']:.1f} ± {row['gap_size_std']:.1f}")
                    report_lines.append(f"  Articles: {row['num_articles_first']}, Books: {row['num_books_first']}")
                    report_lines.append("")
            
            # Outliers
            outliers = self.identify_outlier_gaps()
            if not outliers.empty:
                report_lines.append(f"OUTLIER GAPS ({len(outliers)} found)")
                report_lines.append("-" * 25)
                for idx, row in outliers.head(5).iterrows():  # Show top 5 outliers
                    report_lines.append(f"{row['topic_name']}: gap_size={row['gap_size']}")
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        LOGGER.info(f"Gap analysis report exported to {output_path}")



class BookArticleResults(BaseResults):
    """Enhanced results class for book-article analysis."""
    
    def __init__(self, csv_path: Optional[Path] = None):
        super().__init__(BOOK_ARTICLE_CONFIG, csv_path)
    
    def add_row(self, **row_data) -> None:
        """Add a book-article analysis row."""
        # Check if key columns are present
        for col in self.config.key_columns:
            if col not in row_data:
                raise ValueError(f"Missing required key column: {col}")
        
        # Validate the row
        errors = self.validator.validate_row(row_data)
        if errors:
            raise ValidationError(f"Row validation failed: {errors}")
        
        # Check for existing row
        key_values = {col: row_data[col] for col in self.config.key_columns}
        if self.has_row(**key_values):
            LOGGER.warning(f"Row already exists for keys: {key_values}")
            return
        
        new_row_df = pd.DataFrame([row_data])
        self._df = pd.concat([self._df, new_row_df], ignore_index=True) if not self._df.empty else new_row_df.copy()
        
        # Auto-save if path is set
        if self.csv_path:
            self.to_csv()
    
    def has_row(self, **keys) -> bool:
        """Check if a row with given key values exists."""
        if self._df.empty:
            return False
        
        mask = pd.Series([True] * len(self._df))
        for col, value in keys.items():
            if col in self._df.columns:
                mask &= self._df[col] == value
        
        return bool(mask.any())
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of book-article analysis."""
        df = self.get_df()
        if df.empty:
            return {}
        
        return {
            'total_analyses': len(df),
            'unique_topics': df['topic_name'].nunique() if 'topic_name' in df.columns else 0,
            'unique_denoisers': df['denoiser'].nunique() if 'denoiser' in df.columns else 0,
            'drop_k_values': sorted(df['drop_k'].unique().tolist()) if 'drop_k' in df.columns else []
        }


class DataFrameResults(BaseResults):
    """Generic results class for arbitrary DataFrames like batch processing results.
    
    This class allows integration of domain-specific DataFrames (like merged batch results)
    into the unified results framework without forcing them into predefined schemas.
    """
    
    def __init__(self, df: pd.DataFrame, name: str = "dataframe_results", 
                 key_columns: Optional[List[str]] = None, csv_path: Optional[Path] = None):
        """
        Initialize with a pre-existing DataFrame.
        
        Args:
            df: The DataFrame to wrap
            name: Name for this results set
            key_columns: Optional list of columns that uniquely identify rows
            csv_path: Optional path for CSV persistence
        """
        # Create a dynamic config based on the DataFrame
        detected_key_columns = key_columns or []
        all_columns = df.columns.tolist()
        dependent_columns = [col for col in all_columns if col not in detected_key_columns]
        
        config = ResultsConfig(
            result_type=ResultType.DATAFRAME,
            key_columns=detected_key_columns,
            dependent_columns=dependent_columns,
            required_columns=all_columns,
            column_types={col: df[col].dtype for col in all_columns}
        )
        
        super().__init__(config, csv_path)
        self.name = name
        self._df = df.copy()
        
        # If CSV path provided, save immediately
        if csv_path:
            self.to_csv(csv_path)
    
    def add_row(self, **row_data) -> None:
        """Add a row to the DataFrame results."""
        if not row_data:
            return
        
        new_row_df = pd.DataFrame([row_data])
        self._df = pd.concat([self._df, new_row_df], ignore_index=True) if not self._df.empty else new_row_df.copy()
        
        # Auto-save if path is set
        if self.csv_path:
            self.to_csv()
    
    def export_multiple_formats(self, base_path: Path, 
                               formats: List[str] = ['csv', 'json', 'parquet']) -> List[Path]:
        """Export to multiple formats using ResultsExporter."""
        return ResultsExporter.export_multiple_formats(self, base_path, formats)
    
    def validate_data(self, operation_name: str = "") -> None:
        """Validate the DataFrame using framework patterns."""
        if self._df.empty:
            raise ValidationError(f"Empty DataFrame in {operation_name or self.name}")
        
        LOGGER.info(f"Validated {operation_name or self.name}: {len(self._df)} rows, {len(self._df.columns)} columns")
    
    def process_numeric_columns(self, significant_figures: int = 4) -> 'DataFrameResults':
        """Process numeric columns using framework patterns."""
        processed_df = self._df.copy()
        float_cols = processed_df.select_dtypes(include=['float']).columns
        
        if len(float_cols) > 0:
            LOGGER.info(f"Processing {len(float_cols)} floating-point columns with {significant_figures} significant figures")
            
            for col in float_cols:
                processed_df[col] = processed_df[col].apply(
                    lambda x: float(f"{x:.{significant_figures}g}") if pd.notnull(x) else x
                )
        
        # Return a new DataFrameResults with processed data
        return DataFrameResults(
            processed_df, 
            name=f"{self.name}_processed",
            key_columns=self.config.key_columns,
            csv_path=self.csv_path
        )
    
    @classmethod
    def from_merge(cls, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                   on: Union[str, List[str]], name: str = "merged_results",
                   **merge_kwargs) -> 'DataFrameResults':
        """Create DataFrameResults from merging two DataFrames."""
        merged_df = left_df.merge(right_df, on=on, **merge_kwargs)
        
        # Use merge keys as key columns
        key_columns = on if isinstance(on, list) else [on]
        
        LOGGER.info(f"Created merged DataFrameResults '{name}': {len(merged_df)} rows from {len(left_df)} + {len(right_df)} input rows")
        
        return cls(merged_df, name=name, key_columns=key_columns)


class ResultsProcessor:
    """Enhanced unified processing pipeline for results with advanced analytics."""
    """Enhanced unified processing pipeline for results with advanced analytics."""
    
    @staticmethod
    def filter_results(results: BaseResults, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply multiple filters to results with advanced filtering options."""
        df = results.get_df()
        
        for column, filter_value in filters.items():
            if column not in df.columns:
                continue
                
            # Handle different filter types
            if isinstance(filter_value, dict):
                # Range filters: {'min': 10, 'max': 100}
                if 'min' in filter_value:
                    df = df[df[column] >= filter_value['min']]
                if 'max' in filter_value:
                    df = df[df[column] <= filter_value['max']]
                # List filters: {'in': [1, 2, 3]}
                if 'in' in filter_value:
                    df = df[df[column].isin(filter_value['in'])]
                # String filters: {'contains': 'substring'}
                if 'contains' in filter_value:
                    df = df[df[column].str.contains(filter_value['contains'], na=False)]
                # Null filters: {'not_null': True}
                if 'not_null' in filter_value and filter_value['not_null']:
                    df = df[df[column].notna()]
            elif isinstance(filter_value, list):
                # Simple list filter - column value must be in list
                df = df[df[column].isin(filter_value)]
            else:
                # Simple equality filter
                df = df[df[column] == filter_value]
        
        return df
    
    @staticmethod
    def sort_results(results: BaseResults, sort_config: Union[List[str], Dict[str, Any]], 
                     ascending: bool = True) -> pd.DataFrame:
        """Enhanced sorting with custom algorithms and multi-criteria support."""
        df = results.get_df()
        
        if isinstance(sort_config, list):
            # Simple multi-column sort
            available_columns = [col for col in sort_config if col in df.columns]
            if available_columns:
                return df.sort_values(available_columns, ascending=ascending)
        
        elif isinstance(sort_config, dict):
            # Advanced sorting configuration
            sort_columns = sort_config.get('columns', [])
            sort_ascending = sort_config.get('ascending', ascending)
            sort_algorithm = sort_config.get('algorithm', 'standard')
            
            available_columns = [col for col in sort_columns if col in df.columns]
            if not available_columns:
                return df
            
            if sort_algorithm == 'standard':
                return df.sort_values(available_columns, ascending=sort_ascending)
            
            elif sort_algorithm == 'rank_based':
                # Sort by rank (percentile) rather than raw values
                df_sorted = df.copy()
                for col in available_columns:
                    if df_sorted[col].dtype in ['int64', 'float64']:
                        df_sorted[f'{col}_rank'] = df_sorted[col].rank(pct=True)
                        rank_cols = [f'{col}_rank' for col in available_columns if f'{col}_rank' in df_sorted.columns]
                        return df_sorted.sort_values(rank_cols, ascending=sort_ascending)
                return df_sorted.sort_values(available_columns, ascending=sort_ascending)
            
            elif sort_algorithm == 'weighted':
                # Weighted sorting based on multiple criteria
                weights = sort_config.get('weights', {})
                if weights and len(available_columns) > 1:
                    df_sorted = df.copy()
                    # Normalize each column to 0-1 range
                    for col in available_columns:
                        if df_sorted[col].dtype in ['int64', 'float64']:
                            col_min, col_max = df_sorted[col].min(), df_sorted[col].max()
                            if col_max != col_min:
                                df_sorted[f'{col}_norm'] = (df_sorted[col] - col_min) / (col_max - col_min)
                            else:
                                df_sorted[f'{col}_norm'] = 0
                    
                    # Calculate weighted score
                    df_sorted['weighted_score'] = 0
                    for col in available_columns:
                        weight = weights.get(col, 1.0)
                        norm_col = f'{col}_norm'
                        if norm_col in df_sorted.columns:
                            df_sorted['weighted_score'] += weight * df_sorted[norm_col]
                    
                    return df_sorted.sort_values('weighted_score', ascending=sort_ascending)
                else:
                    return df.sort_values(available_columns, ascending=sort_ascending)
        
        return df
    
    @staticmethod
    def aggregate_results(results: BaseResults, group_by: List[str], 
                         agg_config: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced aggregation with statistical functions and custom aggregators."""
        df = results.get_df()
        
        if not df.empty and all(col in df.columns for col in group_by):
            # Prepare aggregation functions
            agg_functions = {}
            
            for column, agg_spec in agg_config.items():
                if column not in df.columns:
                    continue
                
                if isinstance(agg_spec, str):
                    # Simple aggregation function
                    agg_functions[column] = agg_spec
                elif isinstance(agg_spec, list):
                    # Multiple aggregation functions
                    agg_functions[column] = agg_spec
                elif isinstance(agg_spec, dict):
                    # Custom aggregation with configuration
                    func_type = agg_spec.get('function', 'mean')
                    
                    if func_type == 'percentile':
                        # Custom percentile function
                        percentile = agg_spec.get('percentile', 50)
                        agg_functions[column] = lambda x: x.quantile(percentile / 100)
                    elif func_type == 'outlier_count':
                        # Count outliers (values beyond specified standard deviations)
                        threshold = agg_spec.get('threshold', 2.0)
                        def count_outliers(series):
                            mean_val = series.mean()
                            std_val = series.std()
                            if std_val == 0:
                                return 0
                            z_scores = abs((series - mean_val) / std_val)
                            return (z_scores > threshold).sum()
                        agg_functions[column] = count_outliers
                    elif func_type == 'range':
                        # Calculate range (max - min)
                        agg_functions[column] = lambda x: x.max() - x.min()
                    elif func_type == 'cv':
                        # Coefficient of variation (std/mean)
                        agg_functions[column] = lambda x: x.std() / x.mean() if x.mean() != 0 else 0
                    else:
                        # Standard aggregation function
                        agg_functions[column] = func_type
            
            if agg_functions:
                result = df.groupby(group_by).agg(agg_functions).reset_index()
                
                # Flatten multi-level column names if present
                if isinstance(result.columns, pd.MultiIndex):
                    result.columns = ['_'.join(col).strip() for col in result.columns.values]
                
                return result
        
        return df
    
    @staticmethod
    def get_top_results(results: BaseResults, metric_column: str, n: int = 10, 
                       ascending: bool = False, groupby: Optional[str] = None) -> pd.DataFrame:
        """Enhanced top results with optional grouping and advanced selection."""
        df = results.get_df()
        
        if metric_column not in df.columns:
            LOGGER.warning(f"Metric column '{metric_column}' not found")
            return df.head(n)
        
        if groupby and groupby in df.columns:
            # Get top N within each group
            if ascending:
                result = df.groupby(groupby).apply(
                    lambda x: x.nsmallest(n, metric_column)
                ).reset_index(drop=True)
            else:
                result = df.groupby(groupby).apply(
                    lambda x: x.nlargest(n, metric_column)
                ).reset_index(drop=True)
            return result
        else:
            # Get overall top N
            if ascending:
                return df.nsmallest(n, metric_column)
            else:
                return df.nlargest(n, metric_column)
    
    @staticmethod
    def calculate_statistical_summary(results: BaseResults, 
                                    columns: Optional[List[str]] = None,
                                    include_distribution: bool = False) -> Dict[str, Any]:
        """Calculate comprehensive statistical summary of results."""
        df = results.get_df()
        
        if df.empty:
            return {}
        
        # Select columns to analyze
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in df.columns and 
                             df[col].dtype in ['int64', 'float64']]
        
        summary = {
            'total_rows': len(df),
            'numeric_columns': len(numeric_columns),
            'column_stats': {}
        }
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Helper function to safely convert to float
            def safe_float(val) -> float:
                if pd.isna(val):
                    return 0.0
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0
            
            col_stats = {
                'count': len(series),
                'mean': safe_float(series.mean()),
                'std': safe_float(series.std()),
                'min': safe_float(series.min()),
                'max': safe_float(series.max()),
                'median': safe_float(series.median()),
                'q25': safe_float(series.quantile(0.25)),
                'q75': safe_float(series.quantile(0.75)),
                'skewness': safe_float(series.skew()),
                'kurtosis': safe_float(series.kurtosis()),
                'null_count': df[col].isnull().sum(),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
            
            if include_distribution:
                # Add distribution information
                try:
                    # Calculate histogram bins
                    hist, bin_edges = np.histogram(series, bins=10)
                    col_stats['histogram'] = {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                    
                    # Calculate percentiles
                    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    col_stats['percentiles'] = {
                        f'p{p}': float(series.quantile(p/100)) for p in percentiles
                    }
                except Exception as e:
                    LOGGER.warning(f"Could not calculate distribution for {col}: {e}")
            
            summary['column_stats'][col] = col_stats
        
        return summary
    
    @staticmethod
    def detect_anomalies(results: BaseResults, columns: List[str], 
                        method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies in specified columns using various methods."""
        df = results.get_df()
        
        if df.empty:
            return pd.DataFrame()
        
        available_columns = [col for col in columns if col in df.columns and 
                           df[col].dtype in ['int64', 'float64']]
        
        if not available_columns:
            return df
        
        anomaly_df = df.copy()
        anomaly_flags = pd.DataFrame(index=df.index)
        
        for col in available_columns:
            series = df[col].dropna()
            
            if method == 'zscore':
                # Z-score method
                z_scores = abs((df[col] - series.mean()) / series.std())
                anomaly_flags[f'{col}_anomaly'] = z_scores > threshold
            
            elif method == 'iqr':
                # Interquartile range method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_flags[f'{col}_anomaly'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                )
            
            elif method == 'percentile':
                # Percentile-based method
                lower_percentile = (100 - threshold) / 2
                upper_percentile = 100 - lower_percentile
                lower_bound = series.quantile(lower_percentile / 100)
                upper_bound = series.quantile(upper_percentile / 100)
                anomaly_flags[f'{col}_anomaly'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                )
        
        # Combine anomaly flags
        anomaly_df['is_anomaly'] = anomaly_flags.any(axis=1)
        anomaly_df['anomaly_score'] = anomaly_flags.sum(axis=1)
        
        # Add individual anomaly flags
        for col in anomaly_flags.columns:
            anomaly_df[col] = anomaly_flags[col]
        
        # Return only anomalous rows
        return anomaly_df[anomaly_df['is_anomaly']].copy()
    
    @staticmethod
    def compare_results(results1: BaseResults, results2: BaseResults, 
                       key_columns: List[str], metric_columns: List[str]) -> pd.DataFrame:
        """Compare two results objects and identify differences."""
        df1 = results1.get_df()
        df2 = results2.get_df()
        
        if df1.empty or df2.empty:
            return pd.DataFrame()
        
        # Merge on key columns
        comparison = pd.merge(
            df1[key_columns + metric_columns], 
            df2[key_columns + metric_columns], 
            on=key_columns, 
            how='outer', 
            suffixes=('_1', '_2')
        )
        
        # Calculate differences for each metric column
        for col in metric_columns:
            col1, col2 = f'{col}_1', f'{col}_2'
            if col1 in comparison.columns and col2 in comparison.columns:
                comparison[f'{col}_diff'] = comparison[col2] - comparison[col1]
                comparison[f'{col}_pct_change'] = (
                    (comparison[col2] - comparison[col1]) / comparison[col1] * 100
                ).round(2)
        
        # Add comparison flags
        comparison['in_first_only'] = comparison[[f'{col}_2' for col in metric_columns]].isnull().all(axis=1)
        comparison['in_second_only'] = comparison[[f'{col}_1' for col in metric_columns]].isnull().all(axis=1)
        comparison['has_differences'] = False
        
        for col in metric_columns:
            diff_col = f'{col}_diff'
            if diff_col in comparison.columns:
                comparison['has_differences'] |= (
                    comparison[diff_col].abs() > 1e-6  # Small tolerance for floating point comparison
                )
        
        return comparison


class ResultsMerger:
    """Combines results from multiple sources."""
    
    @staticmethod
    def merge_multiple(results_list: List[BaseResults], 
                      how: Literal['inner', 'outer', 'left', 'right'] = 'outer') -> Optional[BaseResults]:
        """Merge multiple results objects of the same type."""
        if not results_list:
            return None
        
        if len(results_list) == 1:
            return results_list[0]
        
        # Check all results are same type
        result_type = type(results_list[0])
        if not all(isinstance(r, result_type) for r in results_list):
            raise ValueError("All results must be of the same type for merging")
        
        merged = results_list[0]
        for other in results_list[1:]:
            merged = merged.merge(other, how=how)
        
        return merged
    
    @staticmethod
    def combine_dataframes(results_list: List[BaseResults]) -> pd.DataFrame:
        """Combine DataFrames from multiple results (simple concatenation)."""
        if not results_list:
            return pd.DataFrame()
        
        dataframes = [r.get_df() for r in results_list]
        return pd.concat(dataframes, ignore_index=True)


class ResultsExporter:
    """Flexible export system for results."""
    
    @staticmethod
    def to_json(results: BaseResults, path: Path, **kwargs) -> None:
        """Export results to JSON."""
        df = results.get_df()
        df.to_json(path, **kwargs)
        LOGGER.info(f"Exported {len(df)} rows to JSON: {path}")
    
    @staticmethod
    def to_parquet(results: BaseResults, path: Path, **kwargs) -> None:
        """Export results to Parquet format."""
        df = results.get_df()
        df.to_parquet(path, **kwargs)
        LOGGER.info(f"Exported {len(df)} rows to Parquet: {path}")
    
    @staticmethod
    def to_excel(results: BaseResults, path: Path, **kwargs) -> None:
        """Export results to Excel format."""
        df = results.get_df()
        df.to_excel(path, **kwargs)
        LOGGER.info(f"Exported {len(df)} rows to Excel: {path}")
    
    @staticmethod
    def export_multiple_formats(results: BaseResults, base_path: Path, 
                               formats: List[str] = ['csv', 'json']) -> List[Path]:
        """Export to multiple formats."""
        exported_files = []
        
        for fmt in formats:
            if fmt == 'csv':
                csv_path = base_path.with_suffix('.csv')
                results.to_csv(csv_path)
                exported_files.append(csv_path)
            elif fmt == 'json':
                json_path = base_path.with_suffix('.json')
                ResultsExporter.to_json(results, json_path)
                exported_files.append(json_path)
            elif fmt == 'parquet':
                parquet_path = base_path.with_suffix('.parquet')
                ResultsExporter.to_parquet(results, parquet_path)
                exported_files.append(parquet_path)
            elif fmt == 'excel':
                excel_path = base_path.with_suffix('.xlsx')
                ResultsExporter.to_excel(results, excel_path)
                exported_files.append(excel_path)
        
        return exported_files
    


class ResultsFactory:
    """Factory for creating appropriate result types."""
    
    _result_classes = {
        ResultType.GAP_ANALYSIS: GapResults,
        ResultType.BOOK_ARTICLE: BookArticleResults,
        ResultType.DATAFRAME: DataFrameResults
    }
    
    @classmethod
    def create(cls, result_type: ResultType, **kwargs) -> BaseResults:
        """Create results object of specified type."""
        result_class = cls._result_classes.get(result_type)
        if not result_class:
            raise ValueError(f"Unsupported result type: {result_type}")
        
        # Handle different parameter names for different result types
        if result_type == ResultType.GAP_ANALYSIS:
            if 'analysis_config' not in kwargs:
                raise ValueError("GapResults requires 'analysis_config' parameter")
            
            # GapResults expects config_dir instead of csv_path
            if 'csv_path' in kwargs:
                csv_path = kwargs.pop('csv_path')
                kwargs['config_dir'] = csv_path.parent if csv_path else None
            
            # If num_books is provided (from CSV loading), use it as total_books
            if 'num_books' in kwargs:
                kwargs['total_books'] = kwargs.pop('num_books')
        
        return result_class(**kwargs)
    
    @classmethod
    def create_results(cls, result_type: Union[str, ResultType], **kwargs) -> BaseResults:
        """Create results object - accepts string or ResultType enum."""
        if isinstance(result_type, str):
            result_type = ResultType(result_type.lower())
        return cls.create(result_type, **kwargs)
    
    @classmethod
    def create_from_csv(cls, csv_path: Path, result_type: Optional[ResultType] = None) -> BaseResults:
        """Create results object by loading from CSV."""
        if result_type is None:
            result_type = cls._detect_result_type(csv_path)
        
        return cls.create(result_type, csv_path=csv_path)
    
    @classmethod
    def _detect_result_type(cls, csv_path: Path) -> ResultType:
        """Detect result type from CSV structure."""
        try:
            # Read just the header to analyze columns
            df_sample = pd.read_csv(csv_path, nrows=0)
            columns = set(df_sample.columns)
            
            # Check for gap analysis columns
            gap_columns = set(GAP_ANALYSIS_CONFIG.key_columns)
            if gap_columns.issubset(columns):
                return ResultType.GAP_ANALYSIS
            
            # Check for book-article columns  
            book_article_columns = set(BOOK_ARTICLE_CONFIG.key_columns)
            if book_article_columns.issubset(columns):
                return ResultType.BOOK_ARTICLE
            
            # Default to custom
            return ResultType.CUSTOM
            
        except Exception as e:
            LOGGER.warning(f"Could not detect result type from {csv_path}: {e}")
            return ResultType.CUSTOM


class ResultsRegistry:
    """Registration system for custom result types."""
    
    _custom_types: Dict[str, Type[BaseResults]] = {}
    _custom_configs: Dict[str, ResultsConfig] = {}
    
    @classmethod
    def register_result_type(cls, name: str, result_class: Type[BaseResults], 
                           config: ResultsConfig) -> None:
        """Register a custom result type."""
        cls._custom_types[name] = result_class
        cls._custom_configs[name] = config
        LOGGER.info(f"Registered custom result type: {name}")
    
    @classmethod
    def create_custom(cls, name: str, **kwargs) -> BaseResults:
        """Create custom result type."""
        if name not in cls._custom_types:
            raise ValueError(f"Unknown custom result type: {name}")
        
        return cls._custom_types[name](**kwargs)
    
    @classmethod
    def list_custom_types(cls) -> List[str]:
        """List all registered custom types."""
        return list(cls._custom_types.keys())


class ResultsManager:
    """Central management system for all results operations."""
    
    def __init__(self):
        self._active_results: Dict[str, BaseResults] = {}
        self._cache_enabled = True
        self._cache: Dict[str, Any] = {}
    
    def create_results(self, result_type: ResultType, name: str, **kwargs) -> BaseResults:
        """Create and register a results object."""
        results = ResultsFactory.create(result_type, **kwargs)
        self._active_results[name] = results
        LOGGER.info(f"Created and registered results: {name} ({result_type.value})")
        return results
    
    def load_results(self, name: str, csv_path: Path, 
                    result_type: Optional[ResultType] = None) -> BaseResults:
        """Load results from CSV and register."""
        results = ResultsFactory.create_from_csv(csv_path, result_type)
        self._active_results[name] = results
        LOGGER.info(f"Loaded and registered results: {name} from {csv_path}")
        return results
    
    def get_results(self, name: str) -> Optional[BaseResults]:
        """Get registered results by name."""
        return self._active_results.get(name)
    
    def merge_results(self, names: List[str], output_name: str, 
                     how: Literal['inner', 'outer', 'left', 'right'] = 'outer') -> Optional[BaseResults]:
        """Merge multiple registered results."""
        results_list = [self._active_results[name] for name in names if name in self._active_results]
        if not results_list:
            return None
        
        merged = ResultsMerger.merge_multiple(results_list, how)
        if merged:
            self._active_results[output_name] = merged
            LOGGER.info(f"Merged results {names} -> {output_name}")
        
        return merged
    
    def export_results(self, name: str, base_path: Path, 
                      formats: List[str] = ['csv']) -> List[Path]:
        """Export registered results to multiple formats."""
        results = self.get_results(name)
        if results:
            return ResultsExporter.export_multiple_formats(results, base_path, formats)
        return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all managed results."""
        summary = {
            'num_active_results': len(self._active_results),
            'result_names': list(self._active_results.keys()),
            'result_stats': {}
        }
        
        for name, results in self._active_results.items():
            summary['result_stats'][name] = results.get_summary_stats()
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()
        LOGGER.info("Cleared results cache")
    
    def unregister_results(self, name: str) -> bool:
        """Remove results from management."""
        if name in self._active_results:
            del self._active_results[name]
            LOGGER.info(f"Unregistered results: {name}")
            return True
        return False


# Export all public interfaces
__all__ = [
    # Core Framework
    'BaseResults', 'ResultsConfig', 'ResultsValidator', 'ValidationError', 'ResultType',
    
    # Specific Result Types  
    'GapResults', 'BookArticleResults',
    
    # Processing Pipeline
    'ResultsProcessor', 'ResultsMerger', 'ResultsExporter',
    
    # Factory and Management
    'ResultsFactory', 'ResultsRegistry', 'ResultsManager'
]
