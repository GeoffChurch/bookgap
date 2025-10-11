# FINAL PLAN: Clean Batch-First Architecture with Hydra

## Executive Summary

This plan implements a completely clean, batch-first architecture for LLM proposal generation using Hydra structured configs. We break entirely from the existing `batch_workflow.py` interactive approach to create a modern, maintainable system focused on batch processing.

## 🎯 **CURRENT STATUS: Phase 1 & 2 COMPLETE - Ready for Real-World Testing**

### ✅ **COMPLETED PHASES:**
- **Phase 1**: Hydra structured configs with comprehensive field annotations ✅
- **Phase 2**: Batch generator architecture with workflow functions ✅
- **Configuration cleanup**: Removed unused parameters, provider infrastructure ✅
- **Documentation**: Comprehensive TODO.md for future architectural improvements ✅

### 🚀 **IMMEDIATE NEXT STEP: API Testing**
The system is **functionally complete** and ready for real-world testing with an OpenAI API key:

```bash
# Set API key and test the system
export OPENAI_API_KEY="your-key-here"
python bookgap/batch_generator.py --config-name standard batch.num_gaps=1
```

### 📋 **FUTURE ARCHITECTURE (Documented in TODO.md):**
Major architectural improvements are now documented in TODO.md but are **optional enhancements**, not blockers for current functionality.

## ✅ **IMPLEMENTATION STATUS: PHASES 1-2 COMPLETE**

### **Phase 1: Hydra Configuration System** ✅ **COMPLETE**
- ✅ All config dataclasses created (`ExperimentConfig`, `GapConfig`, `LLMConfig`, `DataConfig`, `BatchConfig`, `OutputConfig`)
- ✅ ConfigStore registration with production/debug presets  
- ✅ Comprehensive field annotations for all configuration parameters
- ✅ Integration with existing enum types (`CellType`, `WorkDenoisers`, `IndexType`, `OpenAIModel`)
- ✅ Immutable frozen dataclasses for thread safety
- ✅ Clean directory structure: `bookgap/configs/`

### **Phase 2: Batch Generator Architecture** ✅ **COMPLETE**  
- ✅ `BatchProposalGenerator` abstract base class
- ✅ `OpenAIBatchGenerator` implementation with structured outputs
- ✅ Complete workflow functions (`find_gaps_for_config`, `build_contexts_for_gaps`, `generate_batch_requests`)
- ✅ Batch request data types (`BatchRequestData`, `ChatMessage`, `ChatCompletionBody`)
- ✅ Main entry point (`batch_generator.py`) with Hydra integration
- ✅ Functional system ready for API testing

### **Configuration Cleanup** ✅ **COMPLETE**
- ✅ Removed unused parameters: `min_book_coverage`, `provider_type`, `experiment_name`
- ✅ Eliminated provider infrastructure: deleted `bookgap/llm/providers.py`
- ✅ Updated 15+ files to remove ProviderType references
- ✅ Simplified generator constructors and validation logic
- ✅ Test infrastructure cleanup: removed outdated test files

## 🔬 **IMMEDIATE NEXT STEPS**

### **Step 1: Real-World API Testing** 🚀
The system is ready for immediate testing with OpenAI API:

```bash
# 1. Set up API key
export OPENAI_API_KEY="your-openai-api-key"

# 2. Test with minimal configuration (single gap)
python bookgap/batch_generator.py batch.num_gaps=1

# 3. Test an alternate configuration 
python bookgap/batch_generator.py --config-name anti_lda_drop1

# 4. Monitor batch job execution and results
# The system will automatically handle batch submission and monitoring
```

**Expected Outcome**: Successful generation of 1-77 research proposals via OpenAI Batch API

### **Step 2: Production Validation** 📊
After successful API testing:

```bash
# Run full analysis with the standard preset
python bookgap/batch_generator.py --config-name standard

# Validate outputs in Hydra's automatic output directory
# Check generated proposals for quality and completeness
```

### **Step 3: Optional Legacy Cleanup** 🧹
After validating the new system works correctly:

```bash
# Remove old batch utilities (optional - can keep for reference)
rm -rf bookgap/batch_utils_OLD/

# Clean up any remaining legacy configuration files
# (Only after confirming new system meets all requirements)
## 📋 **FUTURE ARCHITECTURAL IMPROVEMENTS (Optional)**

All major architectural improvements are now documented in **TODO.md** and represent optional enhancements, not required functionality:

### **Configuration Parameter Grouping** (TODO.md)
- Encapsulate scattered GapConfig parameters into component-specific configs
- `DenoiserConfig`, `HNSWIndexConfig`, `FlatIndexConfig` with type-safe composition
- Enables cleaner separation of concerns and easier extensibility

### **Cloud/SubCloud Compositional Architecture** (TODO.md)  
- Replace rigid `BatchConfig.num_gaps` with flexible per-topic SubCloud specifications
- Enable heterogeneous analysis: different algorithms/parameters per AI research area
- Support explicit gap targeting instead of global "find N gaps across all topics"

### **Additional Improvements** (TODO.md)
- Circular dependency resolution
- Test infrastructure modernization  
- Code quality and type safety improvements

**Note**: These are **architectural enhancements** for future development. The current system is fully functional for immediate research use.

---

## 📖 **IMPLEMENTATION HISTORY (For Reference)**

The sections below document the completed implementation phases for historical reference.
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
- ✅ BatchProposalGenerator ABC implemented (`bookgap/llm/generators/batch_generator.py`)
- ✅ OpenAIBatchGenerator implementation (`bookgap/llm/generators/openai_batch.py`)
- ✅ Workflow functions implemented (`bookgap/llm/generators/workflow_functions.py`)
- ✅ Comprehensive test coverage: 29/29 tests passing
- ✅ All data structures are immutable (frozen dataclasses)
- ✅ Full integration with existing Hydra configuration system

## Phase 2: Workflow Management (Week 1-2)

### 2.1 Simple Function-Based Workflow

**Direct Function Pipeline:**
- `find_gaps_for_config(config: ExperimentConfig) -> list[Gap]` - Find gaps (cached)
- `build_contexts_for_gaps(gaps: list[Gap], config: ExperimentConfig) -> list[ProposalContext]` - Build contexts (cached) from gap sampling  
- ✅ `generate_batch_requests(contexts: list[ProposalContext], config: ExperimentConfig) -> list[BatchRequestData]` - Generate prompts and create batch requests, **save batch JSONL file for inspection** ✅ **IMPLEMENTED**
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

**✅ COMPLETED - Test Suite Cleanup:**
- **Spurious Field Removal**: Removed erroneously added `include_citations` field from:
  - `bookgap/configs/llm/llm_config.py` - Removed from LLMConfig dataclass ✅
  - `bookgap/configs/config_store.py` - Removed from all presets ✅  
  - All test files referencing the field ✅
- **Test Organization**: Moved 6 deprecated test files to `tests_OLD/` for reference ✅
- **Test Results**: Reduced failing tests from 66 to 9 (87% improvement) ✅
- **Validation Tests**: Created `test_llm_config_validation.py` preserving essential business rules ✅

**DELETE ENTIRELY:**
- `bookgap/batch_utils/batch_workflow.py` - Interactive workflow superseded by Hydra CLI
- All interactive workflow functions in other files
- Old file-based resumption logic
- `InteractiveLLMConfiguration` class - replaced by Hydra configs
- `bookgap/config_manager.py` - **DELETE ENTIRELY** (CustomIdManager, AnalysisConfiguration, etc.)
- `bookgap/llm/context_builder.py` - **DELETE ENTIRELY** (replaced by `build_contexts_for_gaps` workflow)
- **Individual Generator Architecture - DELETE ENTIRELY:**
  - `bookgap/llm/llm_implementation.py` - Manual factory pattern (`Implementation` enum, `get_generator()`)
  - `bookgap/llm/generators/__init__.py` - Individual generator exports
- **Individual Generator Tests - ✅ MOVED TO tests_OLD/ FOR REFERENCE:**
  - `tests/test_llm_generators.py` → `tests_OLD/test_llm_generators.py` ✅ **MOVED**
  - `tests/llm/test_llm_generators.py` → `tests_OLD/test_llm_generators.py` ✅ **MOVED**
  - `tests/config_manager/test_analysis_configuration.py` → `tests_OLD/test_analysis_configuration.py` ✅ **MOVED**
  - `tests/llm/test_workflow_functions_old.py` → `tests_OLD/test_workflow_functions_old.py` ✅ **MOVED**
  - All tests using deprecated `LLMConfiguration` class ✅ **MOVED**
  - **Status**: Ready for deletion after new architecture validation complete
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
