# TODO: Architecture Improvements

> **Note**: Items should be removed from this list upon completion, not marked as completed.

## Circular Dependencies and TYPE_CHECKING Workarounds

### Current Issue
We have retained hideous `if TYPE_CHECKING` workarounds to avoid circular dependencies in multiple places:

1. **`nextbook/llm/llm_implementation.py`**: 
   - Uses `TYPE_CHECKING` imports for generator classes
   - `Implementation` enum uses string values instead of direct class mapping like `CellType`
   - Circular dependency from `config_manager → llm_implementation → generators → config_manager`
   
2. **`nextbook/llm/context_builder.py`**: Uses `TYPE_CHECKING` to import `GapProtocol` and string annotation `gap: "GapProtocol"` due to circular import between LLM module and `gap_finder.py`

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

## Test and Code Quality Issues

### Fix Pytest Warnings
Address the following test warnings (from running `pytest tests/`):

1. **PytestRemovedIn9Warning**: Update `conftest.py` to use `pathlib.Path` instead of deprecated `py.path.local`
2. **PytestCollectionWarning**: Fix test class collection issues in `TestDenoiserContext` (constructor conflicts)
3. **NumPy DeprecationWarnings**: Fix deprecated array-to-scalar conversions in `denoisers.py` lines 722, 723, 728
4. **Faiss DeprecationWarning**: External library warning (can be suppressed if needed)

## Code Quality and Architecture Issues

### Fix find_gaps_for_topic Algorithm Selection
The `find_gaps_for_topic` function defaults to `CellType.KBALL` but has two problematic usages in `batch_operations.py`:

1. **Line 1086**: `gaps_iter = find_gaps_for_topic(books, articles, config.gap_config.radius_fraction)` - should pass `config.gap_config.cell_type`
2. **Line 1475**: `topic_gaps = find_gaps_for_topic(books, articles, radius_fraction)` - should determine correct `cell_type` from context

These calls ignore the user's configured algorithm choice and always use K-ball. The function should either:
- Remove the default parameter to force explicit specification
- Use the algorithm from available configuration context

Maybe it should accept the config object directly?

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
- Create `nextbook/llm/gap_algorithms/types.py` and import from both algorithm files
- Move to `nextbook/types.py` alongside other core types
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

### Remove .lower() from Enum Name Usage
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
The file `nextbook/path_management.py` appears to be unused but **is NOT a true duplicate** of `config_manager.py`:

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
