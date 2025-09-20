# TODO: Architecture Improvements

> **Note**: Items should be removed from this list upon completion, not marked as completed.

## Circular Dependencies and TYPE_CHECKING Workarounds

### Current Issue
We have retained hideous `if TYPE_CHECKING` workarounds to avoid circular dependencies in multiple places:

1. **`bookgap/llm/llm_implementation.py`**: 
   - Uses `TYPE_CHECKING` imports for generator classes
   - `Implementation` enum uses string values instead of direct class mapping like `CellType`
   - Circular dependency from `config_manager → llm_implementation → generators → config_manager`
   
2. **`bookgap/llm/context_builder.py`**: Uses `TYPE_CHECKING` to import `GapProtocol` and string annotation `gap: "GapProtocol"` due to circular import between LLM module and `gap_finder.py`

Both issues stem from tight coupling between modules that should be more loosely connected.

### Potential Solution
Consider making the implementation classes more focused on just generating the data needed for a prompt, rather than importing heavy configuration objects. This could involve:

- **Dependency Injection**: Pass only the specific parameters needed (API keys, model names, etc.) rather than the entire `AnalysisConfiguration` object
- **Configuration Separation**: Split configuration into smaller, more focused pieces that don't create circular dependencies
- **Factory Pattern with Delayed Import**: Use a factory that imports generators only when needed, without circular imports
- **Protocol-Based Design**: Define protocols/interfaces that generators implement, allowing for looser coupling

### Benefits of Refactoring
- Eliminate the `TYPE_CHECKING` workarounds in both `llm_implementation.py` and `context_builder.py`
- More testable code (easier to mock specific parameters)
- Clearer separation of concerns
- Reduced coupling between modules
- Allow proper type annotations without string quotes

## CLI Migration

### Migrate to Hydra Structured Configs
Replace current CLI argument parsing with Hydra structured configs.

### Refactor Configuration Parameter Grouping
**Current Issue**: The `GapConfig` contains low-level implementation details that should be encapsulated within their respective component configurations:

**Current scattered parameters in `GapConfig`:**
```python
# Embedding preprocessing
drop_k: int              # Number of components to drop (denoiser parameter)
denoiser: WorkDenoisers  # Denoising method enum

# Index implementation details  
index_type: IndexType    # HNSW vs FLAT_L2
hnsw_m: int             # HNSW connectivity parameter
hnsw_efc: int           # HNSW construction candidates  
hnsw_efs: int           # HNSW search candidates
```

**Target Design**: Encapsulate parameters within instantiable component configs:

```python
@dataclass(frozen=True)
class DenoiserConfig:
    _target_: str           # e.g., "bookgap.denoisers.PCADenoiser"
    drop_k: int            # Moved from GapConfig
    # Other denoiser-specific parameters as needed

@dataclass(frozen=True) 
class HNSWIndexConfig:
    _target_: str = field(default="bookgap.index.HNSWIndex", init=False)  # Hardcoded, immutable
    # HNSW-specific parameters
    hnsw_m: int
    hnsw_efc: int
    hnsw_efs: int

@dataclass(frozen=True)
class FlatIndexConfig:
    _target_: str = field(default="bookgap.index.FlatIndex", init=False)  # Hardcoded, immutable
    # Flat index has no additional parameters

@dataclass(frozen=True)
class GapConfig:
    # Only gap detection algorithm parameters
    cell_type: CellType
    radius_fraction: float
    
    # Component configurations
    denoiser: DenoiserConfig
    index: Union[HNSWIndexConfig, FlatIndexConfig]  # Type-safe index selection
```

**Benefits:**
- Cleaner separation of concerns (gap detection vs embedding preprocessing vs indexing)
- Type safety for component-specific parameters (only HNSW params with HNSW config)
- Easier to add new denoiser/index types without cluttering `GapConfig`
- Follows Hydra's compositional configuration pattern
- Eliminates invalid parameter combinations (HNSW params can't be used with Flat index)
- Hardcoded `_target_` prevents configuration errors and simplifies usage

**Implementation Notes:**
- Use Hydra's `_target_` pattern for instantiation, but hardcode targets in dataclass definitions
- Create separate config groups: `denoiser/pca`, `denoiser/none`, `index/hnsw`, `index/flat`
- Add shared base class/protocol when multiple index types exist: `IndexConfigProtocol`
- Update `build_cloud()` to accept component instances instead of scattered parameters

**Example Usage After Refactoring:**
```bash
# Current (scattered parameters)
python batch_generator.py gap.denoiser=PCA gap.drop_k=5 gap.index_type=HNSW gap.hnsw_m=32

# Target (compositional with type-safe configs)  
python batch_generator.py gap.denoiser=pca_5_components gap.index=hnsw_high_quality
```

**Future Extensibility:**
When adding new index types, create new specific config classes:
```python
@dataclass(frozen=True)
class IVFIndexConfig:
    _target_: str = field(default="bookgap.index.IVFIndex", init=False)
    nlist: int
    nprobe: int

# All index configs would implement IndexConfigProtocol
```

### Refactor Cloud/SubCloud Architecture to Compositional Configuration
**Current Issue**: The batch system has a rigid, single-configuration approach that limits flexibility:

**Current limitations:**
```python
# BatchConfig controls everything globally  
batch.num_gaps: int  # Per-topic limit (e.g., num_gaps=2 means 2 gaps from each topic)

# GapConfig applies same settings to all gaps
gap.cell_type: CellType        # Same algorithm for all
gap.radius_fraction: float     # Same radius for all
gap.denoiser: WorkDenoisers    # Same preprocessing for all
```

**Target Design**: Make Cloud and SubCloud instantiable configurations with per-topic/per-gap control:

```python
@dataclass(frozen=True)
class KBallCellConfig:
    _target_: str = field(default="bookgap.gap_algorithms.KBallCell", init=False)
    radius_fraction: float

@dataclass(frozen=True)
class VoronoiCellConfig:  
    _target_: str = field(default="bookgap.gap_algorithms.VoronoiCell", init=False)
    # Voronoi-specific parameters

@dataclass(frozen=True)
class SubCloudConfig:
    _target_: str = field(default="bookgap.cloud.SubCloud", init=False)
    topic_id: str                    # Which AI topic to analyze
    cell_config: Union[KBallCellConfig, VoronoiCellConfig]  # Algorithm per SubCloud
    num_gaps: int                    # Number of gaps to extract from this SubCloud
    denoiser: DenoiserConfig         # Preprocessing per SubCloud
    index: Union[HNSWIndexConfig, FlatIndexConfig]  # Index per SubCloud

@dataclass(frozen=True)
class CloudConfig:
    _target_: str = field(default="bookgap.cloud.Cloud", init=False)
    subclouds: List[SubCloudConfig]  # Explicit list of what to analyze

@dataclass(frozen=True)
class ExperimentConfig:
    cloud: CloudConfig               # Replaces both BatchConfig and GapConfig
    llm: LLMConfig
    data: DataConfig  
    output: OutputConfig
```

**Benefits:**
- **Per-topic control**: Different algorithms/parameters for different research areas
- **Flexible gap counts**: `num_gaps=5` for quantum computing, `num_gaps=2` for robotics  
- **Mixed algorithms**: Use KBALL for some topics, Voronoi for others
- **Heterogeneous preprocessing**: Different denoisers optimal for different domains
- **Explicit targeting**: No more "find N gaps across all 77 topics" - specify exactly what you want
- **Compositionality**: Mix and match any combination of algorithms and parameters

**Example Usage After Refactoring:**
```bash
# Current (global settings)
python batch_generator.py batch.num_gaps=10 gap.cell_type=KBALL gap.radius_fraction=0.75

# Target (explicit per-topic control)
python batch_generator.py cloud=quantum_computing_focus
# or
python batch_generator.py cloud.subclouds=[quantum_5gaps_kball,robotics_2gaps_voronoi,nlp_3gaps_kball]
```

**Potential Simplifications:**
- **Remove BatchConfig**: Replaced by CloudConfig.subclouds specification
- **Remove/Simplify GapConfig**: Gap algorithm parameters move to cell configs  
- **Streamline cloud.py**: Make Cloud/SubCloud instantiable via Hydra rather than factory functions
- **Cleaner workflow**: `workflow_functions.py` gets explicit SubCloud instances instead of building them

**Implementation Considerations:**
- Major refactoring of `cloud.py` to support instantiable Cloud/SubCloud classes
- Update `workflow_functions.find_gaps_for_config()` to iterate over explicit subclouds
- Create config presets like `cloud/ai_comprehensive`, `cloud/quantum_focus`, `cloud/debug_single`
- Potential breaking changes to existing gap finder interfaces
- May require rethinking the relationship between topics, subclouds, and gap extraction

## Test and Code Quality Issues

### Fix Pytest Warnings
Address the following test warnings (from running `pytest tests/`):

1. **PytestRemovedIn9Warning**: Update `conftest.py` to use `pathlib.Path` instead of deprecated `py.path.local`
2. **PytestCollectionWarning**: Fix test class collection issues in `TestDenoiserContext` (constructor conflicts)
3. **NumPy DeprecationWarnings**: Fix deprecated array-to-scalar conversions in `denoisers.py` lines 722, 723, 728
4. **Faiss DeprecationWarning**: External library warning (can be suppressed if needed)

## Code Quality and Architecture Issues

### Replace Dict Return with Dataclass in ConfigPathParser
The `parse_prompt_file_path()` method in `ConfigPathParser` currently returns a dictionary, which lacks type safety and IDE support:

**Current State**:
```python
def parse_prompt_file_path(prompt_file_path: str, gaps_base_dir: str) -> dict:
    # Returns dict with keys: gap_config, llm_config, topic_name, topic_id, gap_idx, filename, etc.
```

**Problem**:
- No type safety for return value fields
- No IDE autocompletion for accessing returned data
- Prone to typos in dictionary key names
- Difficult to track what fields are available

**Solution**:
1. **Create `PromptFilePathComponents` dataclass** similar to existing `CustomIdComponents`
2. **Update return type** to `PromptFilePathComponents`
3. **Update callers** to use dataclass field access instead of dictionary keys

**Benefits**:
- Type safety and IDE support
- Clear documentation of available fields
- Consistent with other parsing methods in the class
- Catches field access errors at type-check time

### Fix Gap Classes Type Conformance  
`KballGap` and `VoronoiGap` classes should explicitly implement the `GapProtocol` interface to resolve mypy type issues in `gap_finder.py`:

- **KballGap**: Change `class KballGap:` to `class KballGap(GapProtocol):` 
- **VoronoiGap**: Change `class VoronoiGap:` to `class VoronoiGap(GapProtocol):`

Both classes already implement all required methods, but the explicit protocol declaration will:
- Make the design intent clear
- Help mypy understand the type relationships
- Catch any method signature mismatches at type-check time
- Eliminate type errors when these classes are used in functions expecting `GapProtocol`

This requires importing `GapProtocol` in both algorithm files.

### Consolidate Duplicate DistsAndIdxs Class
The `DistsAndIdxs` dataclass is duplicated identically in both gap algorithm files:

- **`kball_algorithm.py`**: Lines 13-16 define `DistsAndIdxs`
- **`voronoi_algorithm.py`**: Lines 15-18 define identical `DistsAndIdxs`

This should be moved to a shared location, such as:
- Create `bookgap/llm/gap_algorithms/types.py` and import from both algorithm files
- Move to `bookgap/types.py` alongside other core types
- Move to `gap_finder.py` where `GapProtocol` is defined and import from algorithm files

The shared class should maintain the same interface to avoid breaking changes.

### Simplify KballAlgorithm Method Structure
`KballAlgorithm.find_gaps` is an unnecessary wrapper around `KballAlgorithm.kball_gaps`:

```python
@staticmethod
def find_gaps(ambient: SubCloud, landmark: SubCloud, query: SubCloud, 
        radius_fraction: float, k: int, is_intra_cloud: bool):
    return KballAlgorithm.kball_gaps(ambient, landmark, query, radius_fraction, k, is_intra_cloud)
```

This should be simplified by:
- Inlining the `kball_gaps` implementation directly into `find_gaps`
- Removing the `kball_gaps` method entirely
- This matches the pattern used by `VoronoiAlgorithm.find_gaps` which contains the full implementation

### Remove .lower() from Enum Name Usage (PARTIALLY DONE)
The code currently uses `self.analysis_config.gap_config.cell_type.name.lower()` in `results_manager.py` line 435, but this should be changed to use the exact enum name:

**Current**: `'cell_type': self.analysis_config.gap_config.cell_type.name.lower(),`
**Should be**: `'cell_type': self.analysis_config.gap_config.cell_type.name,`

This change eliminates collision risk and uses the exact enum names (`KBALL`, `VORONOI`) rather than lowercase versions. This will require updates to:
- Validation rules that expect lowercase values (`['kball', 'voronoi']` → `['KBALL', 'VORONOI']`)  
- Any code that compares or filters on cell_type values
- CSV files and data processing that expect lowercase enum names

The goal is zero collision risk by using exact enum names throughout.

### Break Up Monolithic Type Files
Rename and split monolithic type definition files to prevent circular dependencies:

**Current problem**: Files like `types.py` and `enums.py` become dumping grounds that everything imports from, creating circular dependency webs.

**Specific changes needed**:
- **`types.py`** → `cell_types.py` or `gap_cell_types.py` (currently only contains `CellType` enum)
- Remove any generic `enums.py` files in favor of domain-specific files
- Consider moving types closer to where they're used (e.g., gap-related types in gap modules)

**Benefits**:
- Reduces circular dependencies by making imports more specific
- Makes the codebase more modular and easier to understand
- Follows the principle of keeping related code together
- Prevents files from becoming catch-all dumping grounds

**Pattern to follow**: Use specific, descriptive names that indicate the domain/purpose rather than generic names like `types.py`, `enums.py`, `utils.py`, etc.

### Remove Duplicate Path Management Code
The file `bookgap/path_management.py` appears to be unused but **is NOT a true duplicate** of `config_manager.py`:

**Key differences**:
- `path_management.py`: Takes raw string parameters (`gap_config_str: str, llm_subdir_str: str`)
- `config_manager.py`: Takes structured objects (`analysis_config: AnalysisConfiguration`)

**Purpose difference**:
- `path_management.py`: Was designed to avoid circular dependencies by using only strings
- `config_manager.py`: Uses full configuration objects for higher-level operations

**Current status**:
- No files import from `path_management.py` 
- `batch_operations.py` imports from `config_manager.py`
- The `path_management.py` approach might actually be better for avoiding circular dependencies

**Decision needed**: 
- Keep `path_management.py` and migrate imports to use it (better dependency structure)
- OR delete it if the functionality isn't needed (current state suggests this)
- Don't assume they're identical - they have different APIs
