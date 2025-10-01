# ARTIFACTS IMPLEMENTATION PLAN: Simplified Depth-First Approach

## ⚠️ **Implementation Note (September 30, 2025)**

**Stashed Complex Implementation**: A complex artifacts implementation with architectural issues has been stashed for reference:
- **Stash**: `stash@{0}: COMPREHENSIVE: Complex artifacts implementation with architectural issues - includes results_export.py, shared_computation.py, and all modifications`
- **Issues Identified**: 
  - `results_export.py` was 396 lines of over-engineered complexity
  - `shared_computation.py` became a dumping ground for mixed concerns
  - Ad-hoc proposal-to-gap lookup structures duplicated across modules
  - Data structure inconsistencies and poor separation of concerns
- **Recovery**: Use `git stash show stash@{0}` to inspect or `git stash apply stash@{0}` to recover if needed
- **Recommendation**: This stash can be deleted after successful simplified implementation

**Proceeding with**: Clean, simple implementation following the original "hardcode everything initially" principle.

---

## Executive Summary

The current Hydra-based `workflow_functions.py` implements core batch processing but lacks the rich artifacts generation that existed in the original workflow. This plan implements a simplified depth-first approach with **hardcoded settings** to restore all missing functionality quickly, starting with basic gap visualization plots and building to full LaTeX/CSV generation.

## Simplified Implementation Principles

### 🏗️ **Hardcode Everything Initially**
- **Plot settings**: Figure size (12, 10), DPI 300, PNG format, standard colors
- **File formats**: JSON with 2-space indent, CSV with UTF-8 encoding
- **Directory structure**: `outputs/{timestamp}/{plots,contexts,proposals,latex}/`
- **LaTeX templates**: Simple article format with minimal styling

### 🔄 **Leverage Unified Architecture** 
- **Eliminate Code Duplication**: Multiple plotting functions share 80-90% of logic
  - `save_gap_plot_with_proposals()` and `save_gap_books_vs_proposals_plot()` use same pipeline
  - Different modes via `PlotConfig.standard_mode()` vs `PlotConfig.comparison_mode()`
  - Single data preparation pipeline with different visualization strategies
- **Shared Components**: Projection, embedding handling, file saving, error handling
- **Configuration-Driven**: Plot appearance controlled by config rather than separate functions in the original workflow. This plan implements a simplified depth-first approach with **hardcoded settings** to restore all missing functionality quickly, starting with basic gap visualization plots and building to full LaTeX/CSV generation.

**Key Simplification**: All settings (plot sizes, file formats, directory structures) will be hardcoded initially. Hydra configuration can be added later as needed.

## Missing Functionalities from Original Workflow

### 📊 **Gap Analysis & Statistics**
- ✅ Individual gap analysis text files (`*_analysis.txt`)
- ✅ Gap context JSON files (`gap_N_context.json`) 
- ✅ Gap statistics computation (percentile, z-score, Shapiro-Wilk test)
- ✅ **book_gap_sizes computation** - Full statistical parity with legacy implementation
- ❌ Book gap sizes Q-Q plots (`book_gap_sizes_qqplot.png`)
- ❌ Gap size histograms and statistical plots

### 🎨 **Visualization & Plots**
- ❌ Basic gap visualization plots (`*_pca_plot.png`)
- ❌ Gap plots with proposals (`*_visualization_with_proposals.png`)
- ❌ Books vs proposals comparison plots (`*_books_vs_proposals.png`)
- ❌ PCA projection with variance annotations
- ❌ Interactive plot generation during gap finding

### 📋 **Results Export & Tables**
-    Main results CSV (`gaps_analysis_results.csv`) — minimal proposals-centric export
- ❌ Comprehensive proposal ranking and embedding analysis
- ❌ Multiple export formats (CSV, JSON, JSONL, Parquet)
- ❌ Results DataFrame with statistical validation

### 📚 **Proposal Management**
- ❌ Individual proposal JSON files (`gap_N_proposal.json`)
- ❌ Proposal content extraction and parsing
- ❌ LLM response raw storage and processing
- ✅ Custom ID management and metadata tracking (centralized in `identifiers.py`)

### 📄 **LaTeX Generation**
- ❌ Individual proposal LaTeX documents (`*.tex`)
- ❌ LaTeX table generation from CSV results
- ❌ Unicode to LaTeX conversion for special characters
- ❌ Standalone LaTeX document generation

### 🔧 **Legacy Integration Points**
- ❌ Proposal ranking computation via `cloud_index.py`

## Simplified File Structure (No Configuration Files)

```
bookgap/
├── llm/
│   ├── generators/
│   │   └── workflow_functions.py      # ✅ UPDATED: Inlined artifacts generation
│   ├── artifacts/                     # ✅ NEW: Modular artifact generation
│       ├── __init__.py
│       ├── gap_plots.py              # ✅ COMPLETE: Gap visualization (hardcoded settings)
│       ├── data_preparation.py       # ✅ COMPLETE: Unified data preparation pipeline
│       ├── context_export.py         # ✅ COMPLETE: Context JSON + analysis files  
│       ├── results_export.py         # MOSTLY COMPLETE: Proposals-centric CSV export
│       ├── proposal_export.py        # ❌ TODO: Individual proposal files
│       └── latex_export.py           # ❌ TODO: LaTeX generation (hardcoded templates)
│   └── identifiers.py                # ✅ NEW: Canonical gap proposal ID helpers
```

## Simplified Depth-First Implementation Plan

### 🚀 **PHASE 1: Basic Gap Plots (Immediate Deliverable)**
**Goal**: Generate `save_gap_plot_with_proposals` plots at end of Phase 1

#### Step 1.1: Create Foundation Module ✅ **COMPLETED**
**File**: `bookgap/llm/artifacts/gap_plots.py` and `bookgap/llm/artifacts/data_preparation.py`
- ✅ **COMPLETED**: Modern unified plotting architecture with `UnifiedGapPlotter`
- ✅ **COMPLETED**: Standardized data structures (`VisualizationData`, `PlotConfig`, `ProjectionMethod`)
- ✅ **COMPLETED**: Basic `save_gap_plot_with_proposals()` functionality
- ✅ **COMPLETED**: Gap center marking (red star for center article)
- ✅ **COMPLETED**: Complete neighbor network visualization (4-way FAISS searches)
- ✅ **COMPLETED**: Article classification (gap vs non-gap neighboring articles)
- ✅ **COMPLETED**: Full `_collect_visualization_data()` equivalent functionality
- ✅ **COMPLETED**: Skip flags for compatibility with original behavior

**Implementation Details**: Enhanced `GapVisualizationDataPreparer` includes:
- Gap center identification using first gap article position
- Complete 4-way FAISS neighbor search logic with hardcoded skip flags
- Type-safe covering book indices extraction
- Article classification and neighbor context visualization
- All information richness from original `gap_finder.save_gap_plot_with_proposals()`
- Unit tests covering changed functionality with real FAISS indices

#### Step 1.2: Integrate with Workflow ✅ **COMPLETED**
**Implementation**: Inlined directly into `workflow_functions.py`
- ✅ **COMPLETED**: Artifact generation integrated into `run_batch_workflow()`
- ✅ **COMPLETED**: **Early artifacts** - Context files generated immediately after gap finding
- ✅ **COMPLETED**: **Late artifacts** - Plots generated after proposal completion
- ✅ **COMPLETED**: Direct calls to `generate_gap_plots_for_results()` and `generate_all_contexts()`
- ✅ **COMPLETED**: Uses Hydra output directory: `Path(HydraConfig.get().runtime.output_dir)`
- ✅ **COMPLETED**: Removed intermediate `workflow_artifacts.py` orchestration layer

#### Step 1.3: Test & Validate Phase 1 ✅ **COMPLETED**
```bash
# Test command generates gap plots with full feature parity
python bookgap/batch_generator.py config=debug batch.num_gaps=1
# Generates: outputs/{hydra_timestamp}/plots/gap_0_visualization_with_proposals.png
# Confirmed output in: outputs/2025-09-26/18-43-49/
```

**Current Status**: 
- ✅ Modern architecture foundation complete
- ✅ Full plotting functionality with feature parity
- ✅ **Plots have complete information richness** - gap center, neighbor network, article classification
- ✅ **Phase 1 Complete** with full feature parity to original implementation

**Recent Implementation (September 29, 2025)**:
- ✅ **book_gap_sizes Computation**: Successfully implemented computation of book gap sizes for complete parity with legacy gap_finder.write_gaps()
- ✅ **Statistical Integration**: Added book_gap_sizes field to GapWithContext dataclass and integrated into workflow_functions.py
- ✅ **Algorithm Debug**: Fixed assertion logic in kball_algorithm.py for proper is_intra_cloud parameter handling
- ✅ **Context Export**: Updated context_export.py to use real book_gap_sizes data instead of placeholder values
- ✅ **Full Validation**: Generated complete artifact files showing statistical parity (z-score 16.33, percentile 1.0) with legacy implementation
- ✅ **Data Completeness**: Confirmed all JSON context files contain complete article/book metadata with no missing fields
- ✅ **Early Artifact Generation**: Context files now generated immediately after gap finding, providing immediate results even if LLM processing fails

**Recent Implementation (October 1, 2025)**:
- ✅ **Results CSV Delivery**: `generate_gaps_analysis_csv()` writes sanitized, proposals-centric CSVs with prompt counts and gap statistics
- ✅ **Canonical IDs**: `bookgap/llm/identifiers.py` centralizes building/parsing of `gap_{idx}_topic_{topic_id}` custom IDs used across requests and exports
- ✅ **Workflow Embeddings Pass**: Batch results now materialize `ParsedProposal` objects after a single `Embedder` pass via the new `PendingProposal` staging type
- ✅ **ArtifactPaths Update**: Results CSV path is stored alongside plots and contexts for downstream integrations
- ✅ **Targeted Tests**: Added `tests/llm/test_results_export.py` to cover CSV sanitization edge cases

**Workflow Improvement - Early vs Late Artifacts**:
- **Early Artifacts** (after Step 1 - Gap Finding): Context JSON files, analysis text files
  - **Benefit**: Available immediately for analysis even if batch processing fails
  - **Content**: Complete gap statistics, article metadata, covering books analysis
- **Late Artifacts** (after Step 5 - Proposal Generation): Plot files requiring proposal data
  - **Benefit**: Include proposal visualizations and comparative analysis
  - **Content**: Gap plots with proposals, books vs proposals comparisons

---

### 📊 **PHASE 2: Results Export & Context Files**
**Goal**: Restore CSV generation and context file export

#### Step 2.1: Context Export Module ✅ **COMPLETED**
**File**: `bookgap/llm/artifacts/context_export.py`
- ✅ **COMPLETED**: `save_gap_context(gap: GapWithContext, output_dir: Path) -> Dict[str, Path]`
- ✅ **COMPLETED**: Standard JSON with 2-space indentation format
- ✅ **COMPLETED**: Generate `gap_N_context.json` with articles, books, gap statistics
- ✅ **COMPLETED**: Generate `gap_N_analysis.txt` with detailed human-readable analysis
- ✅ **COMPLETED**: Ported functionality from `write_single_kball_gap()` in gap_finder.py
- ✅ **COMPLETED**: **Early generation** - Integrated into workflow immediately after gap finding
- ✅ **COMPLETED**: Comprehensive test suite (4/4 tests passing)
- ✅ **COMPLETED**: **book_gap_sizes implementation** - Full statistical parity with legacy gap_finder
- ✅ **COMPLETED**: **Complete data validation** - All JSON context files contain full article/book metadata
- ✅ **COMPLETED**: **Statistical analysis integration** - Z-scores, percentiles, Shapiro-Wilk tests working
- ✅ **COMPLETED**: **Robust workflow** - Context files available even if subsequent LLM processing fails

#### Step 2.2: Results Export Module
**File**: `bookgap/llm/artifacts/results_export.py`
- ✅ **COMPLETED**: Implemented `generate_gaps_analysis_csv(...)` producing `gaps_analysis_results.csv`
- ✅ **COMPLETED**: Proposals-first schema joins sanitized proposal text with gap metadata (topic, gap stats, prompt counts)
- ✅ **COMPLETED**: Control characters escaped via `_sanitize_text_field` (unit-tested)
- ✅ **COMPLETED**: Lookup map for `ProposalCounts` built from `ProposalContext` to report requested vs actual prompt sizes
- ✅ **COMPLETED**: Minimal Pandas dependency only; columns ordered for human readability
- ✅ **COMPLETED**: Dedicated tests in `tests/llm/test_results_export.py` covering sanitization edge cases
- ✅ **COMPLETED**: Integrated into `run_batch_workflow()` so CSV is generated alongside plots and contexts
- ⚠️ **TODO**: Ranking columns (`proposal_book_rank`, `proposal_article_rank`, fractions) still missing; reference legacy implementation in `bookgap/batch_utils_OLD/batch_operations.py::_merge_gap_analysis_with_proposals()` and `compute_proposal_rankings()` for restoration blueprint

#### Step 2.3: Individual Proposal Files
**File**: `bookgap/llm/artifacts/proposal_export.py`
- Implement `save_individual_proposals(results: BatchResults, output_dir: Path) -> List[Path]`
- **Hardcode directory**: `{hydra_output_dir}/openai_structured_proposals/`
- Generate `gap_N_proposal.json` files
- **Simple error handling**: Log errors, continue processing

---

### 🎨 **PHASE 3: Advanced Plots & Statistics**
**Goal**: Complete visualization suite with statistical plots

#### Step 3.1: Statistical Plots
**Extend**: `bookgap/llm/artifacts/gap_plots.py`
- ✅ **Leverage Unified Pipeline**: Both `save_gap_plot_with_proposals()` and `save_gap_books_vs_proposals_plot()` share 90% functionality:
  - **Shared data preparation**: Gap articles, covering books, neighbor searches, projection pipeline
  - **Shared embedding logic**: Proposal handling, PCA/t-SNE projection, axis labeling
  - **Different coloring strategies**: Standard vs classification-based coloring
- **Implementation approach**: 
  - Extend `PlotConfig` with `.standard_mode()` and `.comparison_mode()` class methods
  - Single `_unified_gap_plot_pipeline()` function handles both modes
  - Classification logic as configurable coloring strategy via `coloring_mode` field

**PlotConfig Method Design**:
```python
@classmethod
def standard_mode(cls, method: str, topic_name: str, gap_idx: int) -> "PlotConfig":
    """Standard gap visualization with full neighborhood context.
    Shows: gap articles (red), gap center (red star), neighbors, books (orange), proposals (purple)"""
    return cls(method_name=method, topic_name=topic_name, gap_idx=gap_idx, coloring_mode="standard")

@classmethod  
def comparison_mode(cls, method: str, topic_name: str, gap_idx: int) -> "PlotConfig":
    """Books vs proposals classification visualization.
    Shows: articles colored by nearest neighbor, covering books, proposals (no neighbors/center)"""
    return cls(method_name=method, topic_name=topic_name, gap_idx=gap_idx, coloring_mode="classification",
               article_color="#ff7f0e", gap_article_color="#1f77b4", proposal_marker="s")
```

**Unified Function Usage**:
```python
def save_gap_plot_with_proposals(gap_with_context, proposals, method="PCA"):
    config = PlotConfig.standard_mode(method, gap_with_context.topic_name, gap_with_context.gap_idx)
    return _unified_gap_plot_pipeline(gap_with_context, proposals, config)

def save_gap_books_vs_proposals_plot(gap_with_context, proposals, method="PCA"):
    config = PlotConfig.comparison_mode(method, gap_with_context.topic_name, gap_with_context.gap_idx)
    return _unified_gap_plot_pipeline(gap_with_context, proposals, config)
```
- Port Q-Q plot generation from `gap_statistics.py`
- **Hardcode all plot settings**: colors, markers, sizes, fonts
- Add gap size histograms and distribution analysis

**Code Elimination**: This unified approach eliminates ~80% of code duplication between plotting functions

---

### 📄 **PHASE 4: LaTeX Generation**
**Goal**: Restore complete LaTeX export functionality

#### Step 4.1: LaTeX Export Module
**File**: `bookgap/llm/artifacts/latex_export.py`
- Port `json_to_latex()` and `create_latex_table_from_csv()` from `batch_operations.py`
- Implement `export_proposals_to_latex(results: BatchResults, output_dir: Path) -> List[Path]`
- **Hardcode LaTeX template**: Simple article format with fixed styling
- **Hardcode Unicode handling**: Use pylatexenc with standard settings

---

### 🗂️ **PHASE 5: Simple Integration**
**Goal**: Wire everything together with minimal complexity

#### Step 5.1: Direct Directory Creation
**No separate module needed** - just create directories directly in artifact functions:
```python
from hydra.core.hydra_config import HydraConfig

# In each artifact function, just do:
output_dir = Path(HydraConfig.get().runtime.output_dir)
plots_dir = output_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
# ... save files directly to plots_dir
```

#### Step 5.2: Simple Integration
**File**: `bookgap/llm/generators/workflow_artifacts.py`
```python
@dataclass(frozen=True)
class ArtifactPaths:
    """Container for all generated artifact file paths."""
    plots: List[Path]
    context_files: List[Path]
    proposal_files: List[Path]
    results_csv: Path
    latex_files: List[Path]

# Example of how artifact generation works (inlined in workflow_functions.py):
def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    # Step 1: Find gaps
    gaps = find_gaps_for_config(config)
    
    # ARTIFACTS: Generate context files immediately after gap finding
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    context_paths = generate_all_contexts(gaps, output_dir)
    
    # Steps 2-5: Build contexts, generate requests, submit batch, monitor results
    # ... proposal generation pipeline ...
    
    # ARTIFACTS: Generate plots after proposals are complete
    plot_paths = generate_gap_plots_for_results(gaps, results.proposals, output_dir)

    # ❌ TODO: Add when implementing remaining steps
    # proposal_paths = save_individual_proposals(results, output_dir)
    # latex_paths = generate_latex_files(results, output_dir)

    csv_path = generate_gaps_analysis_csv(gaps, results.proposals, output_dir, config, contexts)

    artifact_paths = ArtifactPaths(
        plots=plot_paths,
        context_files=context_paths,
        results_csv=csv_path,
    )
    return BatchResults(
        proposals=results.proposals,
        metadata=results.metadata,
        artifact_paths=artifact_paths,
    )
```

## Simplified Implementation Principles

### � **Hardcode Everything Initially**
- **Plot settings**: Figure size (12, 10), DPI 300, PNG format, standard colors
- **File formats**: JSON with 2-space indent, CSV with UTF-8 encoding
- **Directory structure**: `outputs/{timestamp}/{plots,contexts,proposals,latex}/`
- **LaTeX templates**: Simple article format with minimal styling

### 📦 **Minimal Dependencies**
- No new configuration classes or Hydra extensions
- Use existing imports and standard library where possible
- Simple error handling with logging, no complex validation frameworks

### 🔄 **Simplified Integration** ✅ **COMPLETED**
```python
# Actual implementation - inlined into run_batch_workflow()
def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    # Step 1: Find gaps
    gaps = find_gaps_for_config(config)
    
    # ARTIFACTS: Generate context files immediately (early artifacts)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    context_paths = generate_all_contexts(gaps, output_dir)
    LOGGER.info(f"Generated context files in {output_dir}/contexts/")
    
    # Steps 2-5: Proposal generation pipeline...
    # ... build contexts, generate requests, submit batch, monitor results ...

    # ARTIFACTS: Generate plots after proposals are complete (late artifacts)
    plot_paths = generate_gap_plots_for_results(gaps, results.proposals, output_dir)
    LOGGER.info(f"Generated plot files in {output_dir}/plots/")

    # ARTIFACTS: Generate CSV once proposals are embedded
    results_csv = generate_gaps_analysis_csv(
        gaps,
        results.proposals,
        output_dir,
        config,
        contexts,
    )
    LOGGER.info(f"Generated results CSV at {results_csv}")

    artifact_paths = ArtifactPaths(
        plots=plot_paths,
        context_files=context_paths,
        results_csv=results_csv,
    )

    return BatchResults(
        proposals=results.proposals,
        metadata=results.metadata,
        artifact_paths=artifact_paths
    )
```

### 📁 **Simple Directory Structure**
- Use Hydra's managed output directory (automatic timestamping)
- Standard subdirectories with predictable names
- Example: `outputs/2025-09-21/14-30-25/plots/` (Hydra's default format)

### 🏗️ **Better Data Structure Architecture**

**Current Issue**: The plan doesn't address data structure consistency between phases.

**Recommendation**: Create a unified data preparation layer:

```python
# bookgap/llm/artifacts/data_preparation.py (extend existing)
class ArtifactDataPreparer:
    """Unified data preparation for all artifact types."""
    
    def prepare_gap_summary(self, gap: GapWithContext) -> dict:
        """Standard gap summary used by CSV, JSON, and LaTeX exports."""
        
    def prepare_proposal_metadata(self, proposal: ParsedProposal) -> dict:
        """Standard proposal metadata used across all exports."""
```

**Benefits**:
- **Eliminate Code Duplication**: Same gap/proposal formatting logic used across CSV, JSON, LaTeX
- **Consistent Data**: All exports use identical gap metadata and proposal information
- **Single Source of Truth**: Changes to data formatting affect all exports uniformly
- **Testable**: Data preparation logic can be unit tested independently

## Success Criteria by Phase

### ✅ **Phase 1 Status - COMPLETE**: 
- ✅ **Architecture Foundation Complete**: Modern unified plotting system working
- ✅ **Full Functionality**: Gap plots generated in proper Hydra output directory structure
- ✅ **Output Path Management**: No more file pollution - all files go to `outputs/{timestamp}/`
- ✅ **Clean Error Handling**: No fallback behavior - fails fast with clear errors
- ✅ **Feature Parity Complete**: Plots include gap center marking, neighbor network, article classification
- ✅ **Validated Implementation**: Working output confirmed in `outputs/2025-09-26/18-43-49/`
- 🎯 **Next Priority**: Step 2.3 - Individual proposal exports

### ✅ **Phase 2 Status - MOSTLY COMPLETE**: 
- ✅ **Step 2.1 COMPLETE**: Context JSON files: `{hydra_output_dir}/contexts/gap_N_context.json`
- ✅ **Step 2.1 COMPLETE**: Context text files: `{hydra_output_dir}/contexts/gap_N_analysis.txt`
- ✅ **Step 2.1 COMPLETE**: **book_gap_sizes computation** - Statistical parity with legacy implementation achieved
- ✅ **Step 2.1 COMPLETE**: **Data structure enhancement** - GapWithContext now carries book_gap_sizes, gap_size, percentile, z-score, Shapiro metrics
- ✅ **Step 2.1 COMPLETE**: **Workflow integration** - Full artifact generation working in batch workflow
- ✅ **Step 2.2 COMPLETE**: `gaps_analysis_results.csv` written beside plots/contexts (sanitized text, prompt counts, topic metadata)
- ⚠️ **Step 2.2 Follow-up**: Reinstate proposal ranking metrics by porting logic from `bookgap/batch_utils_OLD/batch_operations.py::_merge_gap_analysis_with_proposals()` and `compute_proposal_rankings()`
- ❌ **Step 2.3 TODO**: Individual proposal JSON files: `{hydra_output_dir}/openai_structured_proposals/gap_N_proposal.json`

### ✅ **Phase 3 Success**:
- Books vs proposals plots: `{hydra_output_dir}/plots/gap_N_books_vs_proposals.png`
- Q-Q plots and statistical analysis plots with hardcoded styling

### ✅ **Phase 4 Success**:
- LaTeX files: `{hydra_output_dir}/latex/gap_N_proposal.tex`
- All Unicode converted with pylatexenc standard settings

### ✅ **Phase 5 Success**:
- Complete artifacts in single workflow run using Hydra output directory
- All file types generated with hardcoded settings
- Clean, predictable output structure managed by Hydra

## Simplified Timeline Estimate

- **Phase 1**: 0.5-1 day (just port plotting code with hardcoded settings)
- **Phase 2**: 1-1.5 days (CSV + JSON generation, no complex validation)
- **Phase 3**: 0.5-1 day (port additional plots)
- **Phase 4**: 1-1.5 days (LaTeX with simple templates)
- **Phase 5**: 0.5 day (wire everything together)

**Total**: ~4-6 days for complete restoration of all original functionality

This simplified approach focuses on getting functional artifacts quickly by hardcoding all settings initially. Configuration can be added later as needed. After Phase 1, users can immediately see visual results, providing early validation that the new Hydra-based system is working correctly.
