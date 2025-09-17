# FINAL PLAN: Clean Batch-First Architecture with Hydra

## Executive Summary

This plan implements a completely clean, batch-first architecture for LLM proposal generation using Hydra structured configs. We break entirely from the existing `batch_workflow.py` interactive approach to create a modern, maintainable system focused on batch processing.

## Code Removal Strategy (Clean Slate Approach)

### 📁 **MOVE TO REFERENCE (Immediate):**
```bash
mv bookgap/batch_utils/ bookgap/batch_utils_OLD/  # Keep for business logic reference
```
- **Purpose**: Reference for understanding existing business logic and patterns
- **Usage**: Look up specific implementation details during new development
- **Benefit**: Complete design freedom for new implementation

### 🏗️ **BUILD FROM SCRATCH:**
- New batch generator system with clean functional architecture
- Hydra configuration system designed from ground up
- OpenAI batch integration using modern patterns
- Pure functional pipeline without legacy constraints

### 🗑️ **DELETE POST-IMPLEMENTATION:**
- `bookgap/batch_utils_OLD/` - Remove after new system is validated
- `bookgap/config_manager.py` - Replace with Hydra structured configs
- All old configuration bridge code
- Legacy interactive workflow patterns
- String-based provider handling throughout codebase

## Code Removal Strategy (Clean Slate Approach)

### � **MOVE TO REFERENCE (Immediate):**
```bash
mv bookgap/batch_utils/ bookgap/batch_utils_OLD/  # Keep for business logic reference
```
- **Purpose**: Reference for understanding existing business logic and patterns
- **Usage**: Look up specific implementation details during new development
- **Benefit**: Complete design freedom for new implementation

### 🏗️ **BUILD FROM SCRATCH:**
- New batch generator system with clean functional architecture
- Hydra configuration system designed from ground up
- OpenAI batch integration using modern patterns
- Pure functional pipeline without legacy constraints

### �️ **DELETE POST-IMPLEMENTATION:**
- `bookgap/batch_utils_OLD/` - Remove after new system is validated
- All old configuration bridge code
- Legacy interactive workflow patterns
- String-based provider handling throughout codebase

## Core Principles

1. **Batch-First**: Design optimized for batch processing (77 prompts → single batch job)
2. **Clean Slate**: No backward compatibility constraints - optimal architecture
3. **Hydra Configs**: Structured dataclass configurations only (no YAML)
4. **Context-Driven**: Rich context objects instead of file I/O coupling
5. **Linear Pipeline**: Simple execution without complex resumption
6. **Structured Outputs**: Leverage proven batch API structured outputs support

## Architecture Overview

```
New Clean Architecture:
Gap Analysis → Context Building → Batch Request Generation → Single Batch Job → Results Processing
```

## Implementation Structure

```
bookgap/
├── batch_generator.py              # NEW: Clean entry point with Hydra
├── configs/                        # NEW: Hydra structured configs
│   ├── __init__.py
│   ├── config.py                   # Main experiment config
│   ├── gap/
│   │   ├── __init__.py
│   │   ├── kball.py               # Gap detection configs
│   │   └── voronoi.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_structured.py   # LLM provider configs
│   │   └── openai_prompt.py
│   └── experiments/
│       ├── __init__.py
│       ├── default.py             # Preset experiment configs
│       ├── large_scale.py
│       └── debug.py
├── llm/
│   ├── generators/
│   │   ├── batch_generator.py      # NEW: Clean batch generator base
│   │   ├── openai_batch.py         # NEW: OpenAI batch implementation
│   │   └── workflow_functions.py   # NEW: Pure pipeline functions
│   └── protocols.py                # UPDATED: Add batch request types
```

**Type Definitions:**
```python
from bookgap.llm.protocols import ProposalContext, Article, Book, ProposalCounts

@dataclass(frozen=True)
class ChatMessage:
    """OpenAI chat message structure."""
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class ChatCompletionBody:
    """OpenAI chat completion request body."""
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float

@dataclass(frozen=True)
class BatchRequestData:
    """OpenAI batch API request structure."""
    custom_id: str
    method: str  # Always "POST"
    url: str     # Always "/v1/chat/completions" 
    body: ChatCompletionBody

# Research data types for type safety:
@dataclass(frozen=True)
class Article:
    display_name: str
    abstract: str

@dataclass(frozen=True)
class Book:
    display_name: str
    abstract: str

# ProposalContext uses typed dataclasses:
# @dataclass(frozen=True)
# class ProposalContext:
#     topic_name: str
#     topic_id: str
#     gap_idx: int
#     requested_num_prompt_articles: int
#     requested_num_prompt_books: int
#     sampled_articles: list[Article]      # Typed instead of dict
#     covering_books: list[Book]           # Typed instead of dict
#     counts: ProposalCounts               # Typed instead of dict
```

**Custom ID Strategy Options:**

*Current Implementation (per-request config):*
- Format: `"{gap_config}___{topic_name}___id={topic_id}___gap_{gap_idx}"`
- Example: `"NONE_k=5_radiusfrac=0.75_kball___quantum_computing___id=1234___gap_3"`
- Stores full configuration in each custom_id (can be very long)

*Proposed: Batch-Level Metadata (centralized config):*
- Custom IDs: Still identify requests (e.g., `"gap_0_topic_1234"`) but no config data
- Configuration: Store entire ExperimentConfig as JSON in batch metadata
- Handle 512-char limit: Split config JSON into chunks with numeric keys `"0"`, `"1"`, `"2"`, etc.
  - Example: If config JSON is 768 chars: `{"0": cfg_str[:512], "1": cfg_str[512:]}`

**REMOVE ENTIRELY POST-IMPLEMENTATION:**
```
├── batch_utils/                    # DELETE: Entire directory
│   ├── batch_workflow.py           # DELETE: Interactive workflow superseded
│   └── batch_operations.py         # DELETE: Replaced by clean functional approach
├── config_manager.py               # DELETE: Replaced by Hydra ConfigStore system
│   ├── GapConfiguration            # DELETE: Legacy config class (replaced by GapConfig)
│   ├── LLMConfiguration            # DELETE: Legacy config class (replaced by LLMConfig)
│   └── AnalysisConfiguration       # DELETE: Legacy config class (replaced by ExperimentConfig)
└── types.py                        # UPDATED: Clean type definitions
```

## Phase 1: Core Infrastructure (Week 1)

### 1.1 Hydra Structured Configs ✅ **COMPLETE**

Create pure dataclass configuration system with ConfigStore presets:

**Core Configuration Classes:**
- `GapConfig` - Gap detection parameters (all required, no defaults)
  - `cell_type: CellType` (enum: KBALL, VORONOI) 
  - `radius_fraction: float`, `drop_k: int`
  - `denoiser: WorkDenoisers` ✅ **EXISTS** - Import from `bookgap.cloud`
  - `index_type: IndexType` ✅ **EXISTS** - Import from `bookgap.index` (HNSW, FLAT_L2)
  - `hnsw_m: int`, `hnsw_efc: int`, `hnsw_efs: int` - HNSW parameters
- `LLMConfig` - Model settings (all required, no defaults)
  - `provider_type: ProviderType` ✅ **EXISTS** - (OPENAI, etc.)
  - `model_name: str`, `max_tokens: int`, `temperature: float`
  - `chapters_min: int`, `chapters_max: int` - From config_manager.py
  - `num_articles: int`, `num_books: int` - From batch_workflow.py
- `DataConfig` - Data source settings (all required, no defaults)
  - `subfield: SubField`, `max_works: Optional[int]`
  - `work_types: Tuple[WorkType, ...]` ✅ **EXISTS** - Import from `bookgap.util` (BOOK, ARTICLE, etc.)
  - `deduplicator: Optional[Deduplicator]` - Deduplication strategy
  - `work_filters: Tuple[WorkFilter, ...]` ✅ **EXISTS** - Data quality filters
  - `filter_preset: Optional[Filters]` ✅ **EXISTS** - Predefined filter combinations (NLP)
- `BatchConfig` - Batch processing options (all required, no defaults)
  - `num_gaps: int`, `generate_plots: bool`
  - `completion_window: str`
- `OutputConfig` - Output format preferences (all required, no defaults)
  - `export_formats: Tuple[ExportFormat, ...]` ✅ **EXISTS** - (CSV, JSON, PARQUET)
- `ExperimentConfig` - Top-level experiment container (all required, no defaults)

**ConfigStore Registration:** ✅ **COMPLETE**
- `production` - Uses existing codebase defaults (parse_args.py, batch_workflow.py, config_manager.py)
- `debug` - Fast settings for development
- `high_quality` - Thorough analysis with creative LLM settings
- Individual component presets: `gap/production`, `llm/creative`, etc.
- **Design**: No dataclass defaults, all values explicit in ConfigStore presets

**✅ IMPLEMENTATION STATUS:**
- ✅ All config dataclasses created with required parameters only
- ✅ ConfigStore registration with 3 preset experiments + component presets
- ✅ Existing codebase defaults preserved in `production` preset
- ✅ Clean directory structure: `bookgap/configs/` (no `experiments/` directory)
- ✅ Legacy compatibility methods removed (no tech debt)
- ✅ Comprehensive test coverage: 20/20 tests passing
- ✅ ExportFormat enum with CSV, JSON, PARQUET, JSONL
- ✅ Integration with all existing enum types
- ✅ Immutable frozen dataclasses for thread safety

**Usage Examples:**
```bash
python script.py config=production              # Existing codebase defaults
python script.py config=debug                   # Fast development settings  
python script.py config=high_quality           # Thorough analysis
python script.py config=production gap=debug   # Mix component presets
python script.py config=production gap.radius_fraction=0.8  # Override values
```
- Use `_target_` pattern for generator factory instantiation
- Provide preset batch configurations (small_test, production, debug)

**Integration with Existing Arguments (from parse_args.py):**
- **Gap Detection**: `drop_k`, `radius_fraction`, `denoiser`, `index_type`, HNSW parameters
- **Data Processing**: `max_works`, deduplication options (`duplicate_fraction`, `min_sqdist`)
- **Index Configuration**: `hnsw_m`, `hnsw_efc`, `hnsw_efs` for HNSW index tuning
- **Deduplication Strategy**: Support both `SqDistDeduplicator(min_sqdist)` and `FractionalDeduplicator(duplicate_fraction)`

**Required Enums (Implementation Priority):**
- `CellType` ✅ **EXISTS** - Import from `bookgap.types` (KBALL, VORONOI) - Maps directly to algorithm classes
- `Implementation` ✅ **EXISTS** - Import from `bookgap.llm.llm_implementation` (STRUCTURED, PROMPT)
- `WorkDenoisers` ✅ **EXISTS** - Import from `bookgap.cloud` (NONE, PCA, GPCA, REL_AMP, LPP, etc.)
- `IndexType` ✅ **EXISTS** - Import from `bookgap.index` (HNSW, FLAT_L2)
- `WorkType` ✅ **EXISTS** - Import from `bookgap.util` (BOOK, ARTICLE, BOOK_CHAPTER, PREPRINT)
- `Deduplicator` ✅ **EXISTS** - Import from `bookgap.deduplicators` (SqDistDeduplicator, FractionalDeduplicator)
- `SubField` ✅ **EXISTS** - Import from `bookgap.util` (AI) - OpenAlex concept IDs
- `Topic` ✅ **EXISTS** - Import from `bookgap.util` (NLP, ROBOTICS) - Topic filters
- `Filters` ✅ **EXISTS** - Import from `bookgap.util` (NLP) - Predefined filter combinations
- `WorkFilter` ✅ **EXISTS** - Import from `bookgap.cloud` - Data quality filters
- `ResultType` ✅ **EXISTS** - Import from `bookgap.results_manager` (GAP_ANALYSIS, BOOK_ARTICLE, etc.)
- `ProviderType` ✅ **EXISTS** - Import from `bookgap.llm.providers` (OPENAI, future providers)
- `ExportFormat` ✅ **EXISTS** - Import from `bookgap.configs.output.export_formats` (CSV, JSON, PARQUET)

### 1.2 Simplified Generator Architecture ✅ **COMPLETE**

**Core Generator Interface:**
- ✅ `BatchProposalGenerator` - ABC defining the batch generator interface
- ✅ **IMPLEMENTED: PromptPair dataclass** for system/user message return type
- ✅ **IMPLEMENTED: OpenAI model enum** for type-safe model handling

**Type Safety Enhancements:**
- ✅ **IMPLEMENTED: Comprehensive dataclass types** - Replace all dict usage with typed dataclasses
- ✅ `Article` and `Book` dataclasses for research data (no more `.get()` calls)
- ✅ `ProposalCounts` dataclass instead of dict for counts
- ✅ `OpenAIModel` enum for model-specific behavior (temperature support detection)
- ✅ `PromptPair` dataclass for clean system/user prompt handling

**Core Methods:**
- ✅ `generate_batch_requests(contexts: list[ProposalContext], config: ExperimentConfig) -> list[BatchRequestData]` - Main method
- ✅ `_build_prompt(context: ProposalContext, config: ExperimentConfig) -> PromptPair` - Abstract method for system/user prompt construction
- ✅ `_create_request_from_prompt(prompts: PromptPair, context: ProposalContext, config: ExperimentConfig) -> BatchRequestData` - Abstract method for request formatting

### 1.3 OpenAI Batch Implementation ✅ **COMPLETE**

**Core Implementation:**
- ✅ `OpenAIBatchGenerator(BatchProposalGenerator)` - OpenAI-specific batch generator
- ✅ Implements structured outputs support using `openai.lib._pydantic.to_strict_json_schema()`
- ✅ Reuses existing prompt building logic from current generators
- ✅ Builds OpenAI batch API requests with proper formatting and custom IDs

**Key Features:**
- ✅ **IMPLEMENTED: System/user message format** for native batch API compatibility
- ✅ **IMPLEMENTED: OpenAI model enum** for type-safe model handling with temperature support detection
- ✅ **IMPLEMENTED: Typed research data** using `Article`, `Book`, and `ProposalCounts` dataclasses
- ✅ Supports both structured and prompt-based implementations
- ✅ Handles GPT-5 temperature exclusion via model enum properties
- ✅ **IMPLEMENTED: Simple custom IDs** with batch-level metadata storage
- ✅ Creates proper batch API request format with separate system/user messages

**Metadata Storage Strategy:**
- ✅ **IMPLEMENTED: Batch-level metadata with config-free custom_ids**
  - Store `ExperimentConfig` as JSON string in batch metadata
  - Split into 512-character chunks with numeric keys: `"0"`, `"1"`, `"2"`, etc.
  - Example: `{"0": json_config[:512], "1": json_config[512:1024], "2": json_config[1024:]}`
  - Custom IDs use simple format: `"gap_{gap_idx}_topic_{topic_id}"`
  - Retrieve full config by concatenating metadata chunks when processing results

**✅ IMPLEMENTATION STATUS:**
- ✅ All batch request data types defined in `bookgap/llm/protocols.py`
- ✅ OpenAIModel enum with temperature support detection
- ✅ Article, Book, ProposalCounts dataclasses for type safety
- ✅ ProposalContext updated to use typed dataclasses
- ✅ LLMConfig updated to use OpenAIModel enum
- ✅ ConfigStore presets updated to use enum values
- ✅ Comprehensive test coverage: 37/37 tests passing
- ✅ All data structures are immutable (frozen dataclasses)
- ✅ Full integration with existing Hydra configuration system

## Phase 2: Workflow Management (Week 1-2)

### 2.1 Simple Function-Based Workflow

**Direct Function Pipeline:**
- `find_gaps_for_config(config: ExperimentConfig) -> list[Gap]` - Find gaps (cached)
- `build_contexts_for_gaps(gaps: list[Gap], config: ExperimentConfig) -> list[ProposalContext]` - Build contexts (cached) from gap sampling  
- `generate_batch_requests(contexts: list[ProposalContext], config: ExperimentConfig) -> list[BatchRequestData]` - Generate prompts and create batch requests, **save batch JSONL file for inspection**
- `submit_batch_job(batch_requests: list[BatchRequestData], config: ExperimentConfig) -> str` - Submit and return job ID
- `monitor_and_download_results(batch_job_id: str, config: ExperimentConfig) -> BatchResults` - Wait for completion and download

**Single Main Function:**
```python
def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    gaps = find_gaps_for_config(config)
    contexts = build_contexts_for_gaps(gaps, config) 
    requests = generate_batch_requests(contexts, config)
    batch_job_id = submit_batch_job(requests, config)
    return monitor_and_download_results(batch_job_id, config)
```

**No Class Overhead:**
- No workflow manager instantiation
- No state management attributes
- No complex orchestration logic
- Simple function composition

### 2.2 Minimal Entry Point

**Direct Main Function:**
```python
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    print(f"Starting batch workflow with {cfg.batch.num_gaps} gaps...")
    results = run_batch_workflow(cfg)
    print(f"Completed! Generated {len(results.proposals)} proposals.")
```

**No Manager Instantiation:**
- No `BatchWorkflowManager(cfg, output_dir)` creation
- No object lifecycle management
- Direct function call with config
- Hydra handles all output directory management automatically

## Phase 3: Integration & Testing (Week 2)

### 3.1 Replace Existing Components

Replace `batch_operations.py` with clean new architecture:
- Delete old file-based batch operations
- Implement new batch submission and monitoring functions
- Clean integration with new generator interface

### 3.2 Simplified Testing

**Test Structure:**
- `TestPipelineFunctions` - Test each pipeline function independently
- `TestOpenAIIntegration` - Test OpenAI-specific batch generator  
- `TestMainWorkflow` - Test complete workflow integration

**Key Test Coverage:**
- Pure function behavior (find_gaps, build_contexts, generate_batch_requests)
- OpenAI batch generator functionality
- End-to-end workflow execution
- Mock external dependencies (OpenAI API, file system)

## Phase 4: Clean Migration & Code Removal (Week 3)

### 4.1 Complete Code Removal (No Backward Compatibility)

**DELETE ENTIRELY:**
- `bookgap/batch_utils/batch_workflow.py` - Interactive workflow superseded by Hydra CLI
- All interactive workflow functions in other files
- Old file-based resumption logic
- `InteractiveLLMConfiguration` class - replaced by Hydra configs
- String-based provider handling - replaced by `ProviderType` enum
- `bookgap/config_manager.py` - **DELETE ENTIRELY** (CustomIdManager, AnalysisConfiguration, etc.)
- `bookgap/gap_directory_manager.py` - **REPLACE** with Hydra output directory management
- **Individual Generator Architecture - DELETE ENTIRELY:**
  - `bookgap/llm/generators/openai_generator.py` - Individual request generator
  - `bookgap/llm/generators/json_prompt_generator.py` - Prompt-based generator  
  - `bookgap/llm/llm_implementation.py` - Manual factory pattern (`Implementation` enum, `get_generator()`)
  - `bookgap/llm/generators/__init__.py` - Individual generator exports
- **Individual Generator Tests - DELETE ENTIRELY:**
  - `tests/test_llm_generators.py` - Individual generator tests
  - `tests/llm/test_llm_generators.py` - Individual generator unit tests
  - `tests/llm/integration/test_generators_real_api.py` - Individual generator integration tests
  - All tests using `get_generator()` factory pattern
- **Legacy Gap Finder Integration - DELETE:**
  - `gap_finder.py` usage of individual generators (replace with batch workflow)
  - `generate_book_proposal()` function (replaced by batch pipeline)

**REPLACE COMPLETELY:**
- `bookgap/batch_utils/batch_operations.py` - Rewrite with clean functional approach
- Generator file output patterns - use Hydra output directories instead of complex directory structures
- Configuration management - eliminate `AnalysisConfiguration` entirely
- Results processing - streamline for batch-first workflow
- **Gap finding workflow** - Replace individual generator calls with batch pipeline
- Error handling - remove interactive recovery, implement batch-appropriate patterns

**MODIFY WITHOUT COMPATIBILITY:**
- Generator factory pattern - use Hydra `_target_` exclusively
- File path management - replace `GapDirectoryManager` with simple Hydra output patterns
- Custom ID management - eliminate complex parsing, use simple IDs + batch metadata

### 4.2 Simplified Architecture Post-Cleanup

**New Clean Structure:**
```
bookgap/
├── batch_generator.py              # NEW: Single entry point
├── configs/                        # NEW: Hydra structured configs
├── llm/
│   ├── generators/
│   │   ├── batch_generator.py      # NEW: Clean batch base class
│   │   ├── openai_batch.py         # NEW: OpenAI batch implementation
│   │   └── workflow_functions.py   # NEW: Pure pipeline functions
│   └── protocols.py                # UPDATED: Clean interfaces only
└── batch_utils/                    # DELETED: Entire directory removed
```

### 4.3 Configuration System Replacement

**REPLACE ENTIRELY:**
- `AnalysisConfiguration`, `GapConfiguration`, `LLMConfiguration` - Replace with pure Hydra dataclass configs
- `CustomIdManager` and `GapDirectoryManager` - Replace with simple Hydra output directory patterns
- All `.from_components()`, `.from_directory_name()` methods - Replaced by Hydra instantiation
- Configuration validation in `__post_init__` - Move to Hydra validators
- String-based configuration parsing - Use structured configs exclusively
- Complex directory structure management - Use Hydra's automatic output directories

**NEW APPROACH:**
```python
# OLD (to be deleted):
gap_config = GapConfiguration.from_directory_name("NONE_k=0_radiusfrac=0.75_kball")
llm_config = LLMConfiguration.from_subdir_name("llm_structured_5_3")
analysis_config = AnalysisConfiguration.from_components(gap_config, llm_config)

# NEW (Pure Hydra dataclass):
@hydra.main(config_path="configs", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    results = run_batch_workflow(cfg)  # Direct config usage - no bridge needed
```

**ELIMINATE ENTIRELY:**
- All `AnalysisConfiguration` bridge logic
- File-path-based configuration parsing
- Manual configuration validation
- Configuration compatibility methods
- `CustomIdManager` and complex custom ID parsing
- `GapDirectoryManager` complex directory structure management

### 4.4 Generator System Simplification

**REMOVE INTERACTIVE PATTERNS:**
- `JSONPromptProposalGenerator` - Keep for real-time API, but remove interactive features
- `OpenAIProposalGenerator` - Streamline for batch-first design
- All file-path-based output generation - Use Hydra output directories

**STREAMLINE:**
- Single generator interface optimized for batch processing
- Remove real-time/interactive mode switching
- Direct Hydra config injection - no `AnalysisConfiguration` bridge needed

## Workflow Design

**Complete Pipeline Approach:**
- **Gap Finding** (~10-30 seconds with cache) - Fast enough to always run
- **Context Building** (~5-15 seconds with cache) - Fast enough to always run  
- **Prompt Generation** (immediate) - Generate fresh each time
- **Batch Submission** (immediate API call) - Submit immediately

**No Resumption Complexity:**
- No file-based state detection
- No workflow stage management
- No prompt file reloading logic
- Rely on existing heavy caching for performance

### Implementation Approach

**Simplified Linear Workflow:**
```python
def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    gaps = find_gaps_for_config(config)           # ~30s (cached)
    contexts = build_contexts_for_gaps(gaps, config)  # ~15s (cached) 
    requests = generate_batch_requests(contexts, config)  # immediate
    
    batch_job_id = submit_batch_job(requests, config)  # immediate
    return monitor_and_download_results(batch_job_id, config)
```

**Storage Strategy:**
- **Prompts**: Hydra output directory (e.g., `outputs/2025-09-15/10-30-15/prompts/`)
- **Optional Caching**: Config-keyed prompt cache for reuse (`cache/prompts/{config_hash}/`)
- **No State Files**: No batch job IDs, no resumption markers

## Functional Pipeline Architecture

**Pure Functions Only:**
- `find_gaps_for_config()` - Gap finding logic
- `build_contexts_for_gaps()` - Context building logic  
- `generate_batch_requests()` - Prompt generation and request formatting
- `submit_batch_job()` - API calls
- `monitor_and_download_results()` - Result retrieval

**No Classes, No State:**
- No workflow manager classes
- No state management
- No object lifecycle
- Simple function pipeline

**Minimal Architecture:**
- Single main function coordinates everything
- Each step is a pure function call
- Hydra handles all configuration and output management

**Clear data flow:**
- `gaps: list[Gap] = find_gaps_for_config(config)`
- `contexts: list[ProposalContext] = build_contexts_for_gaps(gaps, config)`  
- `requests: list[BatchRequestData] = generate_batch_requests(contexts, config)`
- `batch_job_id: str = submit_batch_job(requests, config)`

**Adding new pipeline stages:**
- Create pure validation function: `validate_prompts(prompts: list[str], config: ExperimentConfig) -> list[str]`
- Insert cleanly into pipeline between generation and submission
- Function composition pattern makes integration seamless

## Outstanding Design Decisions & Potential Issues

### Design Decisions to Finalize

1. **Custom ID vs Batch Metadata Strategy** ⚠️ **DECISION NEEDED**
   - **Current**: Full configuration embedded in each custom_id (long strings)
   - **Proposed**: Config-free custom_ids + batch-level metadata chunks
   - **Implementation**: 
     - Serialize `ExperimentConfig` to JSON string
     - Split into 512-char chunks: `{"0": json_str[:512], "1": json_str[512:], ...}`
     - Custom IDs identify requests without config data (e.g., `"gap_0_topic_1234"`)
   - **Consideration**: Need concatenation logic when reading results

2. **Error Handling Strategy**  
   - **Question**: How to handle partial batch failures (e.g., 3 out of 77 requests fail)?
   - **Options**:
     - Fail entire batch (strict)
     - Continue with successful requests, report failures (flexible)
   - **Recommendation**: Flexible approach with detailed error reporting

## Architecture Details

### Generator Factory Pattern
Use Hydra's `_target_` pattern for clean instantiation in dataclass configs:
```python
@dataclass
class LLMConfig:
    _target_: str  # Required field, set in ConfigStore presets
    model: str = "gpt-4o-mini"
    # ... other fields

# ConfigStore preset example:
cs.store(
    group="llm",
    name="openai",
    node=LLMConfig(
        _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
        model="gpt-4o-mini"
    )
)
```

### Output Directory Structure  
Simplified structure leveraging Hydra's automatic output directories:
- **Primary**: Store in Hydra's automatic output directory for run isolation
- **Optional**: Cache prompts by config hash for reuse across runs
- **No Complex Resumption**: Let caching handle performance, use manual scripts for recovery if needed

### Type System
Use existing enums with proper type annotations:
- **Core Enums**: `CellType`, `Implementation`, `WorkDenoisers`, `IndexType`, `WorkType`
- **Specialized Enums**: `SubField`, `Topic`, `Filters`, `WorkFilter`, `ResultType`, `Deduplicator`
- **✅ COMPLETE**: `ProviderType`, `ExportFormat` enums created
- **Critical**: All configuration fields must use proper enum types, never raw strings

### Context Building
- Reuse existing context building logic initially
- No context serialization - rebuild from configuration and cache on resumption
- Prompt files are the source of truth, caching makes rebuilding fast (~30-60s)

## Implementation Issues & Mitigations

**No Config Bridge Needed**: Use Hydra configs directly throughout - no conversion needed
- Pass `ExperimentConfig` directly to all functions
- No intermediate `AnalysisConfiguration` objects
- Simpler function signatures with single config parameter

**Context Building Dependencies**: Context building depends on many existing functions
- Reuse existing context building logic initially
- Gradually clean up dependencies
- Ensure proper injection of configuration

**OpenAI Client Configuration**: Clean client management with API key detection and proper error handling

**Batch Job Monitoring**: Elegant monitoring without blocking CLI
- Store batch job ID in state
- Check status on re-run
- Provide explicit monitoring mode if needed

## Implementation Readiness

**Suggested Implementation Order:**
1. **Create Hydra configs** (foundation for everything)
2. **Implement pure pipeline functions** (core logic)
3. **Build OpenAI batch generator** (main functionality)  
4. **Create single entry point** (user interface)
5. **DELETE everything in removal list** (clean slate)

**Post-Implementation Codebase Size Reduction:**
- `batch_utils/` directory: **DELETE entirely** (~1500+ lines)
- Configuration complexity: **ELIMINATE** (~500+ lines of bridge code)
- Interactive workflow: **REMOVE** (~800+ lines)
- **Total Reduction**: ~2800+ lines of code eliminated
- **Result**: Clean, focused, maintainable batch-first system

## Usage Examples

**Simple, Clean Commands (Post-Cleanup):**
- `python batch_generator.py` - Basic usage with defaults
- `python batch_generator.py batch.num_gaps=50` - Production run with 50 gaps
- `python batch_generator.py gap=kball_sensitive llm=structured_quality` - Custom presets
- `python batch_generator.py llm.model=gpt-4o batch.num_gaps=77` - Full-scale run

**Multi-Run Experiments:**
- `python batch_generator.py -m gap=kball,voronoi llm.model=gpt-4o-mini,gpt-4o` - Matrix experiments
- `python batch_generator.py -m gap.radius_fraction=0.3,0.5,0.7 llm.temperature=0.2,0.7` - Parameter sweeps

**No More:**
- ❌ Interactive step-by-step workflows
- ❌ File-based resumption prompts  
- ❌ Complex configuration assembly
- ❌ Manual batch job monitoring

**Hydra provides:**
- Automatic output directory management
- Configuration override from command line
- Experiment tracking and reproducibility
- Multi-run support for parameter sweeps
- Clean dataclass configuration
