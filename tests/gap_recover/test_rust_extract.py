import numpy as np
from artifacts_scripts.gap_recover.adapters.rust import extract

CRATE = {
    "src/lib.rs": (
        "pub struct Shape { pub area: f64 }\n"
        "pub trait Draw { fn draw(&self); }\n"
        "impl Draw for Shape {\n    fn draw(&self) { let _ = self.area; }\n}\n"
        "pub fn make_shape(a: f64) -> Shape { Shape { area: a } }\n"
        "struct Inner { s: Shape }\n"
    ),
}


def _write(tmp_path):
    for rel, src in CRATE.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return tmp_path


def test_rust_nodes_and_users(tmp_path):
    _write(tmp_path)
    r = extract.extract_repo(tmp_path, repo_name="demo")
    byleaf = {}
    for i, n in enumerate(r.names):
        byleaf.setdefault(n.split("::")[-1], i)
    assert "Shape" in byleaf and r.is_B[byleaf["Shape"]]
    assert "make_shape" in byleaf and r.is_F[byleaf["make_shape"]]
    # make_shape constructs Shape + returns Shape -> Shape has an OPERATIONAL user (a func cites it)
    assert r.operational_users()[byleaf["Shape"]] >= 1
    # struct Inner has a field of type Shape -> Shape has a STRUCTURAL user (a type cites it)
    assert r.structural_users()[byleaf["Shape"]] >= 1
    # impl Draw for Shape -> Draw gains a structural user; Draw is a trait (type/B)
    assert r.is_B[byleaf["Draw"]] and r.structural_users()[byleaf["Draw"]] >= 1
    assert r.concept_pref[byleaf["Shape"]]
