"""Reconstruct fully-qualified decl names present at a git tag by tracking the
namespace stack while scanning Mathlib/*.lean. BYTE-correct blob splitting."""
import subprocess, re, sys, time
import numpy as np

REPO = "/home/gchurchill/src/mathlib4_v4.20.0"

DECL_KW = ("theorem","lemma","def","abbrev","structure","class","inductive",
           "instance","opaque","axiom")
MOD = r"(?:noncomputable\s+|protected\s+|private\s+|nonrec\s+|partial\s+|unsafe\s+|scoped\s+|local\s+|mutual\s+|@\[[^\]]*\]\s*)*"
DECL_RE = re.compile(r"^(?:" + MOD + r")(?:" + "|".join(DECL_KW) + r")\s+(@?\[[^\]]*\]\s*)?([^\s:({\[]+)")
NS_RE = re.compile(r"^namespace\s+(\S+)")
ENDLABEL_RE = re.compile(r"^end\s+(\S+)\s*$")
END_RE = re.compile(r"^end\s*$")
SEC_RE = re.compile(r"^section(?:\s+(\S+))?\s*$")

def collect_tag(tag):
    files = [f for f in subprocess.run(
        ["git","-C",REPO,"ls-tree","-r","--name-only",tag,"Mathlib/"],
        capture_output=True, text=True).stdout.splitlines() if f.endswith(".lean")]
    spec = ("".join(f"{tag}:{f}\n" for f in files)).encode()
    raw = subprocess.run(["git","-C",REPO,"cat-file","--batch"], input=spec,
                         capture_output=True).stdout  # BYTES
    fqns = set()
    hdr = re.compile(rb"^[0-9a-f]{40} blob (\d+)\n", re.M)
    pos = 0; L = len(raw)
    while pos < L:
        m = hdr.match(raw, pos)
        if not m:
            nl = raw.find(b"\n", pos)
            if nl < 0: break
            pos = nl+1; continue
        size = int(m.group(1))
        start = m.end()
        body = raw[start:start+size].decode("utf-8","replace")
        pos = start + size + 1  # trailing newline
        _scan_file(body, fqns)
    return fqns

def _scan_file(body, fqns):
    stack = []  # ('ns',name) | ('sec',name|None)
    bc = 0      # block-comment nesting depth
    for line in body.split("\n"):
        # strip block comments by depth counting on the raw line
        if bc:
            o = line.count("/-"); c = line.count("-/")
            bc += o - c
            if bc > 0:
                continue
            bc = 0
            # rare: code after close on same line — keep scanning the tail
            line = line.split("-/",1)[1] if "-/" in line else ""
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        if "/-" in s:
            o = s.count("/-"); c = s.count("-/")
            if o > c:
                bc = o - c
                continue
            # balanced inline comment: drop it
            s = re.sub(r"/-.*?-/", "", s).strip()
            if not s: continue
        m = NS_RE.match(s)
        if m:
            stack.append(("ns", m.group(1))); continue
        m = ENDLABEL_RE.match(s)
        if m:
            lbl = m.group(1)
            for i in range(len(stack)-1,-1,-1):
                if stack[i][1] == lbl:
                    del stack[i:]; break
            else:
                if stack: stack.pop()
            continue
        if END_RE.match(s):
            for i in range(len(stack)-1,-1,-1):
                if stack[i][0] == "sec":
                    del stack[i]; break
            continue
        m = SEC_RE.match(s)
        if m:
            stack.append(("sec", m.group(1))); continue
        m = DECL_RE.match(s)
        if m:
            nm = m.group(2)
            path = ".".join(n for k,n in stack if k=="ns")
            fqns.add(f"{path}.{nm}" if path else nm)

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv)>1 else "v4.20.0"
    t0=time.time()
    fqns = collect_tag(tag)
    print(f"[{tag}] reconstructed {len(fqns)} FQNs in {time.time()-t0:.1f}s")
    d = np.load("/tmp/lean_gap/data.npz", allow_pickle=True)
    gnames = set(str(n) for n in d["names"])
    inter = gnames & fqns
    print(f"  gated graph names: {len(gnames)}")
    print(f"  matched graph∩reconstructed: {len(inter)} = {100*len(inter)/len(gnames):.1f}% of graph")
    miss = gnames - fqns
    print(f"  graph names NOT reconstructed: {len(miss)} = {100*len(miss)/len(gnames):.1f}%")
    import random; random.seed(0)
    print("  sample missed:", random.sample(sorted(miss), min(15,len(miss))))
