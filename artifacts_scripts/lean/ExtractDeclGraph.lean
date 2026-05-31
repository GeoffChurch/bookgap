/-
ExtractDeclGraph.lean — dump mathlib4's declaration-dependency graph.

Run from the mathlib4 checkout with the prebuilt cache in place:

    BOOKGAP_LEAN_OUT=/abs/out/dir lake env lean ExtractDeclGraph.lean

(The output directory is read from the `BOOKGAP_LEAN_OUT` env var; defaults to
the current directory. We trigger the work from a `#eval` so it runs against
the file's already-imported `Mathlib` environment — no second import.)

It `import Mathlib`, walks every constant in the loaded environment, and for
each declaration collects the constants referenced in its **type and value**
(`ConstantInfo.getUsedConstantsAsSet`, which folds over type + value and, for
inductives/constructors/recursors, adds the structurally-implied names).

Filtering choice
----------------
Compiler-internal / auto-generated declarations (`Name.isInternalDetail`:
names with a `_`-prefixed component, or `match_…`/`proof_…`/`eq_…`/`omega_…`
auxiliaries, or any numeric component) are dropped as NODES. Edges are then
emitted only between surviving (non-internal) nodes, so every premise a proof
references resolves to a real declaration in the node table. This means a
theorem's recorded deps are its non-internal premises; the chain through an
internal `proof_…`/`match_…` helper is collapsed away (we do NOT transitively
re-expand internal helpers, so a dep reachable only via an internal helper is
not added — this keeps the graph to human-meaningful premises). Internal decl
count is reported on stderr for transparency.

Output (two streamed TSV files; gzipped afterwards by the driver)
-----------------------------------------------------------------
  nodes.tsv : idx \t name \t kind \t module \t statement \t doc
  edges.tsv : src_idx \t dst_idx

idx is the line order in nodes.tsv (0-based). `statement` is the
pretty-printed type (the theorem/def signature). Tabs/newlines/backslashes in
text fields are backslash-escaped (\t \n \\) so each record is exactly one line.
-/
import Mathlib
import Lean

open Lean Meta

/-- Backslash-escape so a string is safe as one TSV field on one line. -/
def tsvEscape (s : String) : String :=
  s.foldl (init := "") fun acc c =>
    match c with
    | '\\' => acc ++ "\\\\"
    | '\t' => acc ++ "\\t"
    | '\n' => acc ++ "\\n"
    | '\r' => acc ++ "\\r"
    | _    => acc.push c

def kindStr : ConstantInfo → String
  | .axiomInfo _  => "axiom"
  | .defnInfo _   => "def"
  | .thmInfo _    => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _   => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _   => "ctor"
  | .recInfo _    => "recursor"

def run (outDir : String) : MetaM Unit := do
  let env ← getEnv
  let moduleNames := env.allImportedModuleNames
  -- First pass: assign indices to every NON-internal declaration.
  -- Use a stable ordering (sorted by name) so reruns are reproducible.
  let mut keep : Array Name := #[]
  let mut internalCount : Nat := 0
  let mut totalCount : Nat := 0
  let allConsts := env.constants.toList
  for (name, _ci) in allConsts do
    totalCount := totalCount + 1
    if name.isInternalDetail then
      internalCount := internalCount + 1
    else
      keep := keep.push name
  keep := keep.qsort (fun a b => a.toString < b.toString)
  -- Pair each kept name with its ConstantInfo (single SMap lookup; avoids `size`/`find!`).
  let mut idxOf : Std.HashMap Name Nat := Std.HashMap.emptyWithCapacity keep.size
  for h : i in [0:keep.size] do
    idxOf := idxOf.insert keep[i] i
  IO.eprintln s!"total constants: {totalCount}, kept (non-internal): {keep.size}, dropped internal: {internalCount}"

  let nodesPath := System.FilePath.mk outDir / "nodes.tsv"
  let edgesPath := System.FilePath.mk outDir / "edges.tsv"
  -- Open handles directly so the writing loop stays in `MetaM` (needed for
  -- `ppExpr`). Handle writes are `IO` ops auto-lifted into `MetaM`.
  let nodesH ← IO.FS.Handle.mk nodesPath .write
  let edgesH ← IO.FS.Handle.mk edgesPath .write
  do
    let mut edgeCount : Nat := 0
    let mut done : Nat := 0
    for h : i in [0:keep.size] do
      let name := keep[i]
      let ci := (env.constants.find? name).get!
      -- module
      let moduleStr :=
        match env.getModuleIdxFor? name with
        | some midx =>
          if h : midx.toNat < moduleNames.size then (moduleNames[midx.toNat]).toString else ""
        | none => ""  -- defined in the current (empty) main module; shouldn't happen here
      -- statement: pretty-print the type
      let stmt ← (do
        try
          let fmt ← ppExpr ci.type
          pure (toString fmt)
        catch _ => pure "")
      -- docstring
      let doc ← (findDocString? env name)
      let docStr := doc.getD ""
      let line := s!"{i}\t{tsvEscape name.toString}\t{kindStr ci}\t{tsvEscape moduleStr}\t{tsvEscape stmt}\t{tsvEscape docStr}\n"
      nodesH.putStr line
      -- edges: referenced constants that survived the filter, excluding self-loops
      let deps := ci.getUsedConstantsAsSet
      for dep in deps do
        match idxOf[dep]? with
        | some j =>
          if j != i then
            edgesH.putStr s!"{i}\t{j}\n"
            edgeCount := edgeCount + 1
        | none => pure ()
      done := done + 1
      if done % 20000 == 0 then
        IO.eprintln s!"  progress: {done}/{keep.size} decls, {edgeCount} edges"
    IO.eprintln s!"done: {keep.size} nodes, {edgeCount} edges"

/-- Entry point: runs against the environment imported above. -/
def extract : MetaM Unit := do
  let outDir ← (do
    match (← IO.getEnv "BOOKGAP_LEAN_OUT") with
    | some d => pure d
    | none   => pure ".")
  IO.println s!"extracting decl graph to {outDir}"
  run outDir
  IO.println "extraction complete"

#eval extract
