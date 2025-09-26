# ARTIFACTS IMPLEMENTATION PLAN: Simplified Depth-First Approach

## Executive Summary

The current Hydra-based `workflow_functions.py` implements core batch processing but lacks the rich artifacts generation that e## Simplified Implementation Principles

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
- **Configuration-Driven**: Plot appearance controlled by config rather than separate functionsin the original workflow. This plan implements a simplified depth-first approach with **hardcoded settings** to restore all missing functionality quickly, starting with basic gap visualization plots and building to full LaTeX/CSV generation.

**Key Simplification**: All settings (plot sizes, file formats, directory structures) will be hardcoded initially. Hydra configuration can be added later as needed.

## Missing Functionalities from Original Workflow

### 📊 **Gap Analysis & Statistics**
- ❌ Individual gap analysis text files (`*_analysis.txt`)
- ❌ Gap context JSON files (`gap_N_context.json`) 
- ❌ Gap statistics computation (percentile, z-score, Shapiro-Wilk test)
- ❌ Book gap sizes Q-Q plots (`book_gap_sizes_qqplot.png`)
- ❌ Gap size histograms and statistical plots

### 🎨 **Visualization & Plots**
- ❌ Basic gap visualization plots (`*_pca_plot.png`)
- ❌ Gap plots with proposals (`*_visualization_with_proposals.png`)
- ❌ Books vs proposals comparison plots (`*_books_vs_proposals.png`)
- ❌ PCA projection with variance annotations
- ❌ Interactive plot generation during gap finding

### 📋 **Results Export & Tables**
- ❌ Main results CSV (`gaps_analysis_results.csv`)
- ❌ Comprehensive proposal ranking and embedding analysis
- ❌ Multiple export formats (CSV, JSON, JSONL, Parquet)
- ❌ Results DataFrame with statistical validation

### 📚 **Proposal Management**
- ❌ Individual proposal JSON files (`gap_N_proposal.json`)
- ❌ Proposal content extraction and parsing
- ❌ LLM response raw storage and processing
- ❌ Custom ID management and metadata tracking

### 📄 **LaTeX Generation**
- ❌ Individual proposal LaTeX documents (`*.tex`)
- ❌ LaTeX table generation from CSV results
- ❌ Unicode to LaTeX conversion for special characters
- ❌ Standalone LaTeX document generation

### 🗂️ **File Organization & Output**
- ❌ Structured output directory creation (`artifacts/subfield=1702, work_types=[book,article]/`)
- ❌ Subdirectory organization (`plots/`, `contexts/`, `openai_structured_proposals/`)
- ❌ Batch request JSONL file generation for inspection
- ❌ Automated file naming and timestamp management

### 🔧 **Legacy Integration Points**
- ❌ Integration with existing `GapDirectoryManager` and directory structure
- ❌ Results validation using `DataFrameResults` framework
- ❌ Proposal ranking computation via `cloud_index.py`
- ❌ Plot styling and formatting consistency

## Simplified File Structure (No Configuration Files)

```
bookgap/
├── llm/
│   ├── generators/
│   │   ├── workflow_functions.py      # Current - minimal changes
│   │   └── workflow_artifacts.py     # NEW: Main artifacts orchestration
│   └── artifacts/                     # NEW: Modular artifact generation
│       ├── __init__.py
│       ├── gap_plots.py              # Gap visualization (hardcoded settings)
│       ├── context_export.py         # Context JSON + analysis files  
│       ├── results_export.py         # CSV generation (hardcoded formats)
│       ├── proposal_export.py        # Individual proposal files
│       └── latex_export.py           # LaTeX generation (hardcoded templates)
```

## Simplified Depth-First Implementation Plan

### 🚀 **PHASE 1: Basic Gap Plots (Immediate Deliverable)**
**Goal**: Generate `save_gap_plot_with_proposals` plots at end of Phase 1

#### Step 1.1: Create Foundation Module ✅ **PARTIAL - Foundation Complete, Features Missing**
**File**: `bookgap/llm/artifacts/gap_plots.py` 
- ✅ **COMPLETED**: Modern unified plotting architecture with `UnifiedGapPlotter`
- ✅ **COMPLETED**: Standardized data structures (`VisualizationData`, `PlotConfig`, `ProjectionMethod`)
- ✅ **COMPLETED**: Basic `save_gap_plot_with_proposals()` functionality
- ❌ **MISSING**: Gap center marking (red star for center article)
- ❌ **MISSING**: Complete neighbor network visualization (4-way FAISS searches)
- ❌ **MISSING**: Article classification (gap vs non-gap neighboring articles)
- ❌ **MISSING**: Full `_collect_visualization_data()` equivalent functionality

**Remaining Work**: Enhance `GapVisualizationDataPreparer` to include:
- Gap center identification and special marking
- Complete neighbor search logic from original `_perform_gap_neighbor_searches()`
- Article classification and neighbor context visualization
- All information richness from original `gap_finder.save_gap_plot_with_proposals()`

#### Step 1.2: Integrate with Workflow
**File**: `bookgap/llm/generators/workflow_artifacts.py`
- Create `generate_gap_plots_for_results(gaps: List[GapWithContext], results: BatchResults, output_dir: Path) -> List[Path]`
- Add call to `run_batch_workflow()` in `workflow_functions.py`
- Get Hydra output directory: `from hydra.core.hydra_config import HydraConfig; output_dir = Path(HydraConfig.get().runtime.output_dir)`

#### Step 1.3: Test & Validate Phase 1 ⚠️ **PARTIAL SUCCESS**
```bash
# Test command should generate gap plots with proposals
python bookgap/batch_generator.py config=debug batch.num_gaps=1
# Expected: outputs/{hydra_timestamp}/plots/gap_0_visualization_with_proposals.png
```

**Current Status**: 
- ✅ Modern architecture foundation complete
- ✅ Basic plotting functionality working
- ❌ **Plots are less informative than original** - missing gap center, neighbor network context
- ⚠️ **Phase 1 Foundation Complete** but **Feature Parity Incomplete**

---

### 📊 **PHASE 2: Results Export & Context Files**
**Goal**: Restore CSV generation and context file export

#### Step 2.1: Context Export Module
**File**: `bookgap/llm/artifacts/context_export.py`
- Implement `save_gap_context(gap: GapWithContext, gap_idx: int, output_dir: Path) -> Path`
- **Hardcode format**: Standard JSON with 2-space indentation
- Generate `gap_N_context.json` with articles, books, gap statistics
- Port gap analysis text file generation from `write_single_kball_gap()`
- **Simple paths**: `{hydra_output_dir}/contexts/gap_{idx}_context.json`

#### Step 2.2: Results Export Module
**File**: `bookgap/llm/artifacts/results_export.py`
- Implement `generate_gaps_analysis_csv(gaps: List[GapWithContext], results: BatchResults, output_dir: Path) -> Path`
- Port ranking computation from `batch_operations.py`
- **Hardcode format**: CSV only (no JSON/JSONL/Parquet for now)
- **Simple validation**: Basic checks, no full `DataFrameResults` framework
- **Fixed filename**: `gaps_analysis_results.csv`

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

def generate_all_artifacts(gaps: List[GapWithContext], results: BatchResults) -> ArtifactPaths:
    """Simple orchestration function using Hydra output directory."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Generate all artifacts - each function creates its own directory
    plots = generate_all_plots(gaps, results, output_dir)
    contexts = generate_all_contexts(gaps, output_dir)
    proposals = save_all_proposals(results, output_dir)
    csv_path = generate_results_csv(gaps, results, output_dir)
    latex_files = generate_all_latex(results, output_dir)
    
    return ArtifactPaths(plots, contexts, proposals, csv_path, latex_files)
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

### 🔄 **Simple Integration**
```python
# Updated workflow - just add artifacts at the end
def run_batch_workflow(config: ExperimentConfig) -> BatchResults:
    # ... existing workflow steps 1-5 ...
    
    # Step 6: Generate artifacts (hardcoded, always enabled)
    artifact_paths = generate_all_artifacts(gaps, results)
    
    return BatchResults(
        proposals=results.proposals,
        metadata=config,
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

### ⚠️ **Phase 1 Status - FOUNDATION COMPLETE, FEATURES INCOMPLETE**: 
- ✅ **Architecture Foundation Complete**: Modern unified plotting system working
- ✅ **Basic Functionality**: Gap plots generated in proper Hydra output directory structure
- ✅ **Output Path Management**: No more file pollution - all files go to `outputs/{timestamp}/`
- ✅ **Clean Error Handling**: No fallback behavior - fails fast with clear errors
- ❌ **Feature Parity Incomplete**: Plots missing gap center marking, neighbor network, article classification
- 🎯 **Next Priority**: Enhance `GapVisualizationDataPreparer` to match original `gap_finder` information richness

### ✅ **Phase 2 Success**:
- CSV results file: `{hydra_output_dir}/gaps_analysis_results.csv`
- Context JSON files: `{hydra_output_dir}/contexts/gap_N_context.json`
- Individual proposal JSON files: `{hydra_output_dir}/openai_structured_proposals/gap_N_proposal.json`

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
