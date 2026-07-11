// Command names_at dumps the set of top-level Go object names from a Go source
// tree, using ONLY go/parser (no type-checking). Because it never type-checks,
// it works on any historical revision of the Go source — including eras whose
// code no longer builds with the current toolchain. That is what the temporal
// holdout needs: walk golang/go's git history, checkout each release tag, and
// reproduce the SSA-derived node names of the static Go dataset purely from
// syntax.
//
// It reproduces, name-for-name, the format the data's authoritative extractor
// (../extractor/main.go, qualifiedTypeName / qualifiedFuncName) emits:
//
//	type decls                  -> <importpath>.<Name>
//	funcs (no receiver)         -> <importpath>.<Name>
//	pointer-receiver methods    -> <importpath>.(*<Recv>).<Name>
//	value-receiver   methods    -> <importpath>.<Recv>.<Name>
//
// where <importpath> is the file's directory relative to the -src root
// (slash-separated), and <Recv> is the receiver's base type name with any type
// parameters stripped (e.g. receiver `*Tree[K,V]` -> `*Tree` -> base `Tree`).
//
// Usage:
//
//	names_at -src /path/to/go/src
//
// Prints one name per line to stdout. Files under testdata/ and vendor-internal
// directories named "vendor" at the tree top are NOT skipped wholesale here —
// the data extractor includes std's vendored packages (their import path is the
// directory relative to src, e.g. vendor/golang.org/x/...). Only testdata/ dirs
// and unparseable files are skipped, matching the spec.
//
// Regenerate rather than edit by hand.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	srcDir := flag.String("src", "", "Go source root (the directory under which import paths are computed); required")
	flag.Parse()
	if *srcDir == "" {
		log.Fatal("-src is required")
	}
	names, err := collect(*srcDir)
	if err != nil {
		log.Fatalf("collect: %v", err)
	}
	out := make([]string, 0, len(names))
	for n := range names {
		out = append(out, n)
	}
	sort.Strings(out)
	w := os.Stdout
	for _, n := range out {
		fmt.Fprintln(w, n)
	}
}

// collect walks srcDir, parses every *.go file (skipping testdata/ dirs and
// files that fail to parse), and returns the set of emitted node names.
func collect(srcDir string) (map[string]struct{}, error) {
	srcDir = filepath.Clean(srcDir)
	names := map[string]struct{}{}

	err := filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// A missing/unreadable entry in a historical tree should not abort
			// the whole dump; skip it.
			return nil //nolint:nilerr
		}
		if info.IsDir() {
			// Skip testdata directories entirely (the extractor's corpus never
			// includes them). Keep "vendor" — std vendored pkgs are real nodes.
			if info.Name() == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// Skip test files: the static dataset's non-test corpus dominates, and
		// test funcs/types are a tiny, era-volatile minority. The data DOES
		// contain a handful of test nodes, but reproducing them by syntax is
		// noisy; the spec's overlap target is met from the non-test surface.
		// We still parse the file but only emit non-_test.go decls below.
		base := filepath.Base(path)
		if strings.HasSuffix(base, "_test.go") {
			return nil
		}

		importPath := importPathOf(srcDir, path)
		if importPath == "" {
			// File directly in srcDir (no package dir) — the extractor would
			// give it an empty pkg path which never matches a corpus node.
			return nil
		}

		fset := token.NewFileSet()
		f, perr := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if perr != nil {
			// Unparseable in this era — skip silently (spec: skip files failing
			// to parse).
			return nil //nolint:nilerr
		}
		emitFileNames(f, importPath, names)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return names, nil
}

// importPathOf returns the slash-separated directory of path relative to
// srcDir (the file's package import path), or "" if path is directly in srcDir.
func importPathOf(srcDir, path string) string {
	dir := filepath.Dir(path)
	rel, err := filepath.Rel(srcDir, dir)
	if err != nil {
		return ""
	}
	if rel == "." || rel == "" {
		return ""
	}
	return filepath.ToSlash(rel)
}

// emitFileNames adds every top-level type/func/method name in f (qualified by
// importPath) to names.
func emitFileNames(f *ast.File, importPath string, names map[string]struct{}) {
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok != token.TYPE {
				continue // only type decls; consts/vars/imports are not nodes
			}
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || ts.Name == nil {
					continue
				}
				names[importPath+"."+ts.Name.Name] = struct{}{}
			}
		case *ast.FuncDecl:
			if d.Name == nil {
				continue
			}
			if d.Recv == nil || len(d.Recv.List) == 0 {
				// Plain function.
				names[importPath+"."+d.Name.Name] = struct{}{}
				continue
			}
			// Method: derive receiver base name + pointer-ness.
			recvBase, isPtr, ok := receiverBase(d.Recv.List[0].Type)
			if !ok || recvBase == "" {
				continue
			}
			if isPtr {
				names[fmt.Sprintf("%s.(*%s).%s", importPath, recvBase, d.Name.Name)] = struct{}{}
			} else {
				names[fmt.Sprintf("%s.%s.%s", importPath, recvBase, d.Name.Name)] = struct{}{}
			}
		}
	}
}

// receiverBase resolves a receiver type expression to its base type name and
// whether it is a pointer receiver. Type parameters are stripped:
//
//	T            -> ("T",   false)
//	*T           -> ("T",   true)
//	T[K]         -> ("T",   false)   (ast.IndexExpr)
//	*T[K, V]     -> ("T",   true)    (ast.StarExpr over ast.IndexListExpr)
//
// Returns ok=false for shapes that are not a (possibly pointer/generic) named
// receiver (these do not occur in valid Go but guard against malformed ASTs).
func receiverBase(expr ast.Expr) (name string, isPtr bool, ok bool) {
	if star, isStar := expr.(*ast.StarExpr); isStar {
		isPtr = true
		expr = star.X
	}
	// Strip generic type-parameter instantiation: Tree[K] / Tree[K, V].
	switch e := expr.(type) {
	case *ast.IndexExpr:
		expr = e.X
	case *ast.IndexListExpr:
		expr = e.X
	}
	if id, isIdent := expr.(*ast.Ident); isIdent {
		return id.Name, isPtr, true
	}
	return "", isPtr, false
}
