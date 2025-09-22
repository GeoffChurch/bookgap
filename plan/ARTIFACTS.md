# ARTIFACTS IMPLEMENTATION PLAN: Simplified Depth-First Approach

## Executive Summary

The current Hydra-based `workflow_functions.py` implements core batch processing but lacks the rich artifacts generation that existed in the original workflow. This plan implements a simplified depth-first approach with **hardcoded settings** to restore all missing functionality quickly, starting with basic gap visualization plots and building to full LaTeX/CSV generation.

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

#### Step 1.1: Create Foundation Module
**File**: `bookgap/llm/artifacts/gap_plots.py`
- Port `save_gap_plot_with_proposals()` from `gap_finder.py`
- Port `_create_gap_plot()` and `_compute_base_projection()` helper functions
- **Hardcode all settings**: figure size (12, 10), DPI 300, PNG format, alpha 0.3
- Update signature: `save_gap_plot_with_proposals(gap: GapWithContext, proposals: List[ParsedProposal], output_dir: Path) -> Path`
- Use passed-in Hydra output directory

#### Step 1.2: Integrate with Workflow
**File**: `bookgap/llm/generators/workflow_artifacts.py`
- Create `generate_gap_plots_for_results(gaps: List[GapWithContext], results: BatchResults, output_dir: Path) -> List[Path]`
- Add call to `run_batch_workflow()` in `workflow_functions.py`
- Get Hydra output directory: `from hydra.core.hydra_config import HydraConfig; output_dir = Path(HydraConfig.get().runtime.output_dir)`

#### Step 1.3: Test & Validate Phase 1
```bash
# Test command should generate gap plots with proposals
python bookgap/batch_generator.py config=debug batch.num_gaps=1
# Expected: outputs/{hydra_timestamp}/plots/gap_0_visualization_with_proposals.png
```

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
- Port `save_gap_books_vs_proposals_plot()` from `gap_finder.py`
- Port Q-Q plot generation from `gap_statistics.py`
- **Hardcode all plot settings**: colors, markers, sizes, fonts
- Add gap size histograms and distribution analysis

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

## Success Criteria by Phase

### ✅ **Phase 1 Success**: 
- `python bookgap/batch_generator.py config=debug batch.num_gaps=1` generates gap plots with proposals
- Plots saved to `{hydra_output_dir}/plots/gap_0_visualization_with_proposals.png`

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
