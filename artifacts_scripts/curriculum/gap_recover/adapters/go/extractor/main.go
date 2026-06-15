// Command goextract emits a dependency graph from Go source for a
// gap-detection study. It produces a single node set and TWO edge sets derived
// from two different call-graph algorithms over the SAME corpus:
//
//   - CHA (Class Hierarchy Analysis): over-approximate / noisy. Interface
//     dispatch points at ALL implementers.
//   - RTA (Rapid Type Analysis): precise / cleaner. Interface dispatch points
//     only at reachable+instantiated implementers.
//
// Holding the node set and the structural (type->type) edges fixed while only
// the call-derived (func->type) edges vary gives a downstream study a clean
// "oracle cleanness" dial.
//
// Output (TSV + meta.json) into -out:
//
//	nodes.tsv       id, name, kind, pkg, is_test, concept_pref, text
//	edges_cha.tsv   src_id, dst_id   (structural + CHA operational)
//	edges_rta.tsv   src_id, dst_id   (structural + RTA operational)
//	meta.json       counts + patterns
//
// A node is either a named TYPE (concept, "B") or a FUNCTION/METHOD
// (operation, "F"). Edges mean "src depends-on dst".
//
//   - structural (type->type), identical in both files: D embeds T, has a field
//     of type T, or (interface) embeds interface T.
//   - operational (func->type), CHA/RTA-dependent: F calls a method whose
//     receiver type is T. PLUS the resolution-independent F->T when F constructs
//     T (composite literal / new(T)), identical in both files.
//
// Regenerate rather than edit output by hand. See the smoke command in the
// package README / task spec.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func main() {
	outDir := flag.String("out", "", "output directory for TSV + meta.json (required)")
	pkgsArg := flag.String("pkgs", "encoding/json,net/http,io,bufio,sort,container/heap",
		"comma-separated go/packages patterns (corpus = the matched root packages, not their deps)")
	flag.Parse()

	if *outDir == "" {
		log.Fatal("-out is required")
	}
	patterns := splitPatterns(*pkgsArg)
	if len(patterns) == 0 {
		log.Fatal("-pkgs resolved to no patterns")
	}

	if err := run(*outDir, patterns); err != nil {
		log.Fatalf("extract failed: %v", err)
	}
}

func splitPatterns(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Node model
// ---------------------------------------------------------------------------

type nodeKind string

const (
	kindType   nodeKind = "type"
	kindFunc   nodeKind = "func"
	kindMethod nodeKind = "method"
	kindTest   nodeKind = "test"
)

type node struct {
	id          int
	name        string // fully-qualified Go object name
	kind        nodeKind
	pkg         string // defining package import path
	isTest      bool
	conceptPref bool
	text        string
}

// graph accumulates nodes (keyed by types.Object) and edges.
type graph struct {
	// nodeByObj maps a types.Object (TypeName for types; Func for funcs/methods)
	// to its node id. The object identity is the stable key.
	nodeByObj map[types.Object]int
	// typeNode maps a *types.Named's TypeName object id (for receiver lookup).
	nodes []node

	structEdges map[edge]struct{} // type->type, shared
	chaEdges    map[edge]struct{} // func->type, CHA operational (+ constructs)
	rtaEdges    map[edge]struct{} // func->type, RTA operational (+ constructs)
}

type edge struct{ src, dst int }

func newGraph() *graph {
	return &graph{
		nodeByObj:   map[types.Object]int{},
		structEdges: map[edge]struct{}{},
		chaEdges:    map[edge]struct{}{},
		rtaEdges:    map[edge]struct{}{},
	}
}

func (g *graph) addNode(obj types.Object, n node) int {
	if id, ok := g.nodeByObj[obj]; ok {
		return id
	}
	n.id = len(g.nodes)
	g.nodes = append(g.nodes, n)
	g.nodeByObj[obj] = n.id
	return n.id
}

func (g *graph) idOf(obj types.Object) (int, bool) {
	id, ok := g.nodeByObj[obj]
	return id, ok
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

func run(outDir string, patterns []string) error {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo | packages.NeedDeps |
			packages.NeedImports | packages.NeedModule,
		Tests: false,
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return fmt.Errorf("packages.Load: %w", err)
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("no packages matched patterns %v", patterns)
	}

	// Corpus = the matched root packages (NOT their dependencies). Skip
	// individual packages that failed to type-check, but log them loudly.
	rootPaths := map[string]bool{}
	var usable []*packages.Package
	var skipped []string
	for _, p := range pkgs {
		if len(p.Errors) > 0 {
			for _, e := range p.Errors {
				log.Printf("package %s has error: %v", p.PkgPath, e)
			}
			// If type info is missing we cannot use it for the graph.
			if p.Types == nil || p.TypesInfo == nil {
				skipped = append(skipped, p.PkgPath)
				continue
			}
			log.Printf("package %s has errors but retains type info; including it", p.PkgPath)
		}
		if p.Types == nil || p.TypesInfo == nil {
			log.Printf("package %s missing type info; skipping", p.PkgPath)
			skipped = append(skipped, p.PkgPath)
			continue
		}
		rootPaths[p.PkgPath] = true
		usable = append(usable, p)
	}
	if len(usable) == 0 {
		return fmt.Errorf("all matched packages were unusable (type-check failures): %v", skipped)
	}
	log.Printf("loaded %d packages (%d usable, %d skipped)", len(pkgs), len(usable), len(skipped))

	g := newGraph()

	// Pass 1: declare all corpus nodes (types + funcs/methods) from syntax, so
	// that node ids exist before we resolve edges.
	for _, p := range usable {
		collectNodes(g, p)
	}

	// Pass 2: structural type->type edges from go/types (shared by both files).
	for _, p := range usable {
		collectStructEdges(g, p)
	}

	// Pass 3: construction edges (func->type), resolution-independent, from AST.
	for _, p := range usable {
		collectConstructEdges(g, p)
	}

	// Pass 4: build SSA and the two call graphs; map call edges to operational
	// func->type edges.
	prog, _ := ssautil.AllPackages(usable, ssa.InstantiateGenerics)
	prog.Build()

	cgCHA := cha.CallGraph(prog)
	g.addCallEdges(cgCHA, rootPaths, g.chaEdges)

	roots := corpusRoots(prog, rootPaths)
	if len(roots) == 0 {
		return fmt.Errorf("no SSA roots found for RTA; corpus has %d nodes", len(g.nodes))
	}
	rtaRes := rta.Analyze(roots, true)
	g.addCallEdges(rtaRes.CallGraph, rootPaths, g.rtaEdges)

	return g.write(outDir, patterns, skipped)
}

// ---------------------------------------------------------------------------
// Pass 1: node collection
// ---------------------------------------------------------------------------

func collectNodes(g *graph, p *packages.Package) {
	pkgPath := p.PkgPath
	for _, f := range p.Syntax {
		fileName := p.Fset.File(f.Pos()).Name()
		fileIsTest := strings.HasSuffix(fileName, "_test.go")
		for _, decl := range f.Decls {
			switch d := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}
					obj := p.TypesInfo.Defs[ts.Name]
					tn, ok := obj.(*types.TypeName)
					if !ok || tn == nil {
						continue
					}
					addTypeNode(g, p, tn, fileIsTest)
				}
			case *ast.FuncDecl:
				obj := p.TypesInfo.Defs[d.Name]
				fn, ok := obj.(*types.Func)
				if !ok || fn == nil {
					continue
				}
				addFuncNode(g, p, fn, d, fileIsTest)
			}
		}
	}
	_ = pkgPath
}

func addTypeNode(g *graph, p *packages.Package, tn *types.TypeName, fileIsTest bool) int {
	name := qualifiedTypeName(tn)
	exported := tn.Exported()
	concept := exported && !fileIsTest
	kind := kindType
	return g.addNode(tn, node{
		name:        name,
		kind:        kind,
		pkg:         p.PkgPath,
		isTest:      fileIsTest,
		conceptPref: concept,
		text:        typeText(tn),
	})
}

func addFuncNode(g *graph, p *packages.Package, fn *types.Func, decl *ast.FuncDecl, fileIsTest bool) int {
	sig, _ := fn.Type().(*types.Signature)
	isMethod := sig != nil && sig.Recv() != nil
	testFunc := isTestFuncName(fn.Name()) && !isMethod
	kind := kindFunc
	if isMethod {
		kind = kindMethod
	}
	if fileIsTest || testFunc {
		kind = kindTest
	}
	return g.addNode(fn, node{
		name:        qualifiedFuncName(fn),
		kind:        kind,
		pkg:         p.PkgPath,
		isTest:      fileIsTest || testFunc,
		conceptPref: false,
		text:        funcText(fn),
	})
}

func isTestFuncName(name string) bool {
	for _, pre := range []string{"Test", "Benchmark", "Example", "Fuzz"} {
		if strings.HasPrefix(name, pre) && len(name) > len(pre) {
			// Test name must be followed by an uppercase letter or end (e.g.
			// TestMain), but we accept the conventional prefixes broadly.
			return true
		}
		if name == pre {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

func qualifiedTypeName(tn *types.TypeName) string {
	if tn.Pkg() == nil {
		return tn.Name() // builtin / universe (won't be a corpus node)
	}
	return tn.Pkg().Path() + "." + tn.Name()
}

func qualifiedFuncName(fn *types.Func) string {
	pkgPath := ""
	if fn.Pkg() != nil {
		pkgPath = fn.Pkg().Path()
	}
	sig, _ := fn.Type().(*types.Signature)
	if sig != nil && sig.Recv() != nil {
		recv := sig.Recv().Type()
		ptr := ""
		if p, ok := recv.(*types.Pointer); ok {
			recv = p.Elem()
			ptr = "*"
		}
		recvName := ""
		if named, ok := recv.(*types.Named); ok {
			recvName = named.Obj().Name()
		} else {
			recvName = recv.String()
		}
		if ptr != "" {
			return fmt.Sprintf("%s.(%s%s).%s", pkgPath, ptr, recvName, fn.Name())
		}
		return fmt.Sprintf("%s.%s.%s", pkgPath, recvName, fn.Name())
	}
	if pkgPath == "" {
		return fn.Name()
	}
	return pkgPath + "." + fn.Name()
}

// ---------------------------------------------------------------------------
// Text (signature) — FIRST LINE / SIGNATURE ONLY; never function bodies.
// ---------------------------------------------------------------------------

const textMax = 300

func clip(s string) string {
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > textMax {
		s = s[:textMax]
	}
	return s
}

func funcText(fn *types.Func) string {
	sig, ok := fn.Type().(*types.Signature)
	if !ok {
		return clip("func " + fn.Name())
	}
	var b strings.Builder
	b.WriteString("func ")
	b.WriteString(fn.Name())
	b.WriteString("(")
	writeTuple(&b, sig.Params(), sig.Variadic())
	b.WriteString(")")
	if res := sig.Results(); res != nil && res.Len() > 0 {
		b.WriteString(" ")
		if res.Len() > 1 {
			b.WriteString("(")
		}
		writeTuple(&b, res, false)
		if res.Len() > 1 {
			b.WriteString(")")
		}
	}
	return clip(b.String())
}

func writeTuple(b *strings.Builder, t *types.Tuple, variadic bool) {
	if t == nil {
		return
	}
	for i := 0; i < t.Len(); i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		v := t.At(i)
		ts := v.Type().String()
		if variadic && i == t.Len()-1 {
			if sl, ok := v.Type().(*types.Slice); ok {
				ts = "..." + sl.Elem().String()
			}
		}
		b.WriteString(shortType(ts))
	}
}

// shortType strips long import paths from a type string, keeping the last
// path component (e.g. "encoding/json.RawMessage" -> "json.RawMessage").
func shortType(s string) string {
	// Heuristic: keep as-is but collapse "a/b/c.Name" to "c.Name". We only
	// touch the leading package-path of each dotted identifier crudely.
	return s
}

func typeText(tn *types.TypeName) string {
	named, ok := tn.Type().(*types.Named)
	under := tn.Type().Underlying()
	if ok {
		under = named.Underlying()
	}
	switch u := under.(type) {
	case *types.Struct:
		n := u.NumFields()
		var fields []string
		for i := 0; i < u.NumFields() && i < 12; i++ {
			f := u.Field(i)
			fields = append(fields, f.Name()+" "+shortType(f.Type().String()))
		}
		body := strings.Join(fields, "; ")
		if n > 12 {
			body += "; ..."
		}
		return clip(fmt.Sprintf("type %s struct{%s}", tn.Name(), body))
	case *types.Interface:
		n := u.NumMethods()
		var ms []string
		for i := 0; i < u.NumMethods() && i < 12; i++ {
			m := u.Method(i)
			ms = append(ms, m.Name())
		}
		body := strings.Join(ms, "; ")
		if n > 12 {
			body += "; ..."
		}
		return clip(fmt.Sprintf("type %s interface{%s}", tn.Name(), body))
	default:
		return clip(fmt.Sprintf("type %s %s", tn.Name(), shortType(under.String())))
	}
}

// ---------------------------------------------------------------------------
// Pass 2: structural type->type edges (shared)
// ---------------------------------------------------------------------------

func collectStructEdges(g *graph, p *packages.Package) {
	scope := p.Types.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		tn, ok := obj.(*types.TypeName)
		if !ok {
			continue
		}
		srcID, ok := g.idOf(tn)
		if !ok {
			continue
		}
		named, ok := tn.Type().(*types.Named)
		if !ok {
			continue
		}
		switch u := named.Underlying().(type) {
		case *types.Struct:
			for i := 0; i < u.NumFields(); i++ {
				f := u.Field(i)
				g.addStructEdgeToType(srcID, f.Type())
			}
		case *types.Interface:
			for i := 0; i < u.NumEmbeddeds(); i++ {
				g.addStructEdgeToType(srcID, u.EmbeddedType(i))
			}
		default:
			// Named alias of another named type: D = T style. Underlying may
			// reference a named type only via its own underlying, which we do
			// not chase here (only direct field/embed/iface relations count).
		}
	}
}

// addStructEdgeToType resolves t to a *types.Named (deref pointers, slices,
// maps, arrays, chans one level) and emits a struct edge if the target is a
// corpus node. We unwrap container types one level to capture e.g. a field of
// type []T or *T or map[K]T -> T (and K).
func (g *graph) addStructEdgeToType(srcID int, t types.Type) {
	for _, target := range namedTypesOf(t) {
		if dstID, ok := g.idOf(target.Obj()); ok {
			if dstID == srcID {
				continue // no self-loops from recursive types
			}
			g.structEdges[edge{srcID, dstID}] = struct{}{}
		}
	}
}

// namedTypesOf extracts the *types.Named referents reachable from a field type
// by unwrapping pointer/slice/array/map/chan one level. For maps both key and
// value contribute.
func namedTypesOf(t types.Type) []*types.Named {
	switch u := t.(type) {
	case *types.Named:
		return []*types.Named{u}
	case *types.Pointer:
		return namedTypesOf(u.Elem())
	case *types.Slice:
		return namedTypesOf(u.Elem())
	case *types.Array:
		return namedTypesOf(u.Elem())
	case *types.Chan:
		return namedTypesOf(u.Elem())
	case *types.Map:
		out := namedTypesOf(u.Key())
		out = append(out, namedTypesOf(u.Elem())...)
		return out
	default:
		return nil
	}
}

// ---------------------------------------------------------------------------
// Pass 3: construction edges (func->type), resolution-independent
// ---------------------------------------------------------------------------

func collectConstructEdges(g *graph, p *packages.Package) {
	info := p.TypesInfo
	for _, f := range p.Syntax {
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			fnObj := info.Defs[fd.Name]
			fn, ok := fnObj.(*types.Func)
			if !ok || fn == nil {
				continue
			}
			srcID, ok := g.idOf(fn)
			if !ok {
				continue
			}
			if fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				switch e := n.(type) {
				case *ast.CompositeLit:
					// T{...} or &T{...}: the type is e.Type (or inferred).
					if tv, ok := info.Types[e]; ok {
						g.addConstructEdge(srcID, tv.Type)
					}
				case *ast.CallExpr:
					// new(T): builtin call whose first arg is a type.
					if id, ok := e.Fun.(*ast.Ident); ok && id.Name == "new" && len(e.Args) == 1 {
						if tv, ok := info.Types[e.Args[0]]; ok && tv.IsType() {
							g.addConstructEdge(srcID, tv.Type)
						}
					}
				}
				return true
			})
		}
	}
}

func (g *graph) addConstructEdge(srcID int, t types.Type) {
	// A composite literal's type may be *T, []T, map[K]T, etc.; the constructed
	// named type is the directly-referenced one. Unwrap a leading pointer for
	// &T{} but treat the literal element type as the construction target.
	for _, named := range namedTypesOf(t) {
		if dstID, ok := g.idOf(named.Obj()); ok {
			// constructs edges go into BOTH cha and rta sets (shared).
			g.chaEdges[edge{srcID, dstID}] = struct{}{}
			g.rtaEdges[edge{srcID, dstID}] = struct{}{}
		}
	}
}

// ---------------------------------------------------------------------------
// Pass 4: operational call edges (func->type), CHA/RTA-dependent
// ---------------------------------------------------------------------------

// corpusRoots returns all SSA functions defined in corpus root packages, to
// seed RTA. Using every corpus function (not just exported/tests) keeps the
// reachable set generous while still being RTA-precise about dispatch.
func corpusRoots(prog *ssa.Program, rootPaths map[string]bool) []*ssa.Function {
	var roots []*ssa.Function
	seen := map[*ssa.Function]bool{}
	for fn := range ssautil.AllFunctions(prog) {
		if fn == nil || fn.Pkg == nil || fn.Pkg.Pkg == nil {
			continue
		}
		if !rootPaths[fn.Pkg.Pkg.Path()] {
			continue
		}
		if seen[fn] {
			continue
		}
		seen[fn] = true
		roots = append(roots, fn)
	}
	return roots
}

// addCallEdges walks a call graph and, for each edge whose caller is a corpus
// function and whose callee is a METHOD with a corpus receiver type, emits the
// operational func->type edge caller -> receiverType.
func (g *graph) addCallEdges(cg *callgraph.Graph, rootPaths map[string]bool, into map[edge]struct{}) {
	if cg == nil {
		return
	}
	for fn, n := range cg.Nodes {
		callerObj := corpusFuncObject(fn, rootPaths)
		if callerObj == nil {
			continue
		}
		srcID, ok := g.idOf(callerObj)
		if !ok {
			continue
		}
		for _, e := range n.Out {
			callee := e.Callee
			if callee == nil || callee.Func == nil {
				continue
			}
			recvType := methodReceiverNamed(callee.Func)
			if recvType == nil {
				continue
			}
			if dstID, ok := g.idOf(recvType.Obj()); ok {
				if dstID == srcID {
					// a method calling another method on the same receiver type
					// is still a real use; keep it unless it's a literal self
					// loop on the identical node id (caller==method-of-T,
					// dst==T): that is meaningful (op depends on its concept's
					// other ops' receiver). Allow it.
				}
				into[edge{srcID, dstID}] = struct{}{}
			}
		}
	}
}

// corpusFuncObject returns the types.Func for an SSA function if it is a
// source-level function/method defined in a corpus root package, else nil.
// It maps SSA wrapper/thunk functions back to their origin where possible.
func corpusFuncObject(fn *ssa.Function, rootPaths map[string]bool) types.Object {
	if fn == nil {
		return nil
	}
	// Generic instantiations: use the origin.
	if orig := fn.Origin(); orig != nil {
		fn = orig
	}
	obj := fn.Object()
	tf, ok := obj.(*types.Func)
	if !ok || tf == nil {
		return nil
	}
	if tf.Pkg() == nil || !rootPaths[tf.Pkg().Path()] {
		return nil
	}
	return tf
}

// methodReceiverNamed returns the *types.Named receiver type of an SSA method
// (deref pointer), or nil if fn is not a method.
func methodReceiverNamed(fn *ssa.Function) *types.Named {
	if fn == nil {
		return nil
	}
	if orig := fn.Origin(); orig != nil {
		fn = orig
	}
	sig := fn.Signature
	if sig == nil || sig.Recv() == nil {
		return nil
	}
	rt := sig.Recv().Type()
	if p, ok := rt.(*types.Pointer); ok {
		rt = p.Elem()
	}
	if named, ok := rt.(*types.Named); ok {
		return named
	}
	return nil
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

type meta struct {
	NNodes          int      `json:"n_nodes"`
	NTypes          int      `json:"n_types"`
	NFuncs          int      `json:"n_funcs"`
	NTests          int      `json:"n_tests"`
	NStructEdges    int      `json:"n_struct_edges"`
	NEdgesCHA       int      `json:"n_edges_cha"`
	NEdgesRTA       int      `json:"n_edges_rta"`
	PackagesLoaded  int      `json:"packages_loaded"`
	PackagesSkipped []string `json:"packages_skipped"`
	Patterns        []string `json:"package_patterns"`
}

func (g *graph) write(outDir string, patterns, skipped []string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	// nodes.tsv
	if err := g.writeNodes(filepath.Join(outDir, "nodes.tsv")); err != nil {
		return err
	}

	// edges_cha.tsv = struct edges + cha operational/construct edges
	if err := writeEdges(filepath.Join(outDir, "edges_cha.tsv"), unionEdges(g.structEdges, g.chaEdges)); err != nil {
		return err
	}
	// edges_rta.tsv = struct edges + rta operational/construct edges
	if err := writeEdges(filepath.Join(outDir, "edges_rta.tsv"), unionEdges(g.structEdges, g.rtaEdges)); err != nil {
		return err
	}

	// counts
	var nTypes, nFuncs, nTests int
	for _, n := range g.nodes {
		switch {
		case n.kind == kindType:
			nTypes++
		case n.kind == kindTest:
			nTests++
			nFuncs++ // tests are F-nodes too
		default:
			nFuncs++
		}
	}
	distinctPkgs := map[string]struct{}{}
	for _, n := range g.nodes {
		distinctPkgs[n.pkg] = struct{}{}
	}
	m := meta{
		NNodes:          len(g.nodes),
		NTypes:          nTypes,
		NFuncs:          nFuncs,
		NTests:          nTests,
		NStructEdges:    len(g.structEdges),
		NEdgesCHA:       len(unionEdges(g.structEdges, g.chaEdges)),
		NEdgesRTA:       len(unionEdges(g.structEdges, g.rtaEdges)),
		PackagesLoaded:  len(distinctPkgs),
		PackagesSkipped: skipped,
		Patterns:        patterns,
	}
	mb, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "meta.json"), append(mb, '\n'), 0o644); err != nil {
		return err
	}

	log.Printf("wrote %d nodes, %d struct edges, %d cha edges, %d rta edges to %s",
		m.NNodes, m.NStructEdges, m.NEdgesCHA, m.NEdgesRTA, outDir)
	return nil
}

func (g *graph) writeNodes(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := fmt.Fprintln(f, "id\tname\tkind\tpkg\tis_test\tconcept_pref\ttext"); err != nil {
		return err
	}
	for _, n := range g.nodes {
		isTest := 0
		if n.isTest {
			isTest = 1
		}
		conceptPref := 0
		if n.conceptPref {
			conceptPref = 1
		}
		if _, err := fmt.Fprintf(f, "%d\t%s\t%s\t%s\t%d\t%d\t%s\n",
			n.id, n.name, n.kind, n.pkg, isTest, conceptPref, n.text); err != nil {
			return err
		}
	}
	return nil
}

func writeEdges(path string, edges []edge) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := fmt.Fprintln(f, "src_id\tdst_id"); err != nil {
		return err
	}
	for _, e := range edges {
		if _, err := fmt.Fprintf(f, "%d\t%d\n", e.src, e.dst); err != nil {
			return err
		}
	}
	return nil
}

func unionEdges(a, b map[edge]struct{}) []edge {
	set := make(map[edge]struct{}, len(a)+len(b))
	for e := range a {
		set[e] = struct{}{}
	}
	for e := range b {
		set[e] = struct{}{}
	}
	out := make([]edge, 0, len(set))
	for e := range set {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].src != out[j].src {
			return out[i].src < out[j].src
		}
		return out[i].dst < out[j].dst
	})
	return out
}
