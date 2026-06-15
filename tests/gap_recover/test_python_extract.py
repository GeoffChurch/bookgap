import numpy as np
from artifacts_scripts.curriculum.gap_recover.adapters.python import extract

PKG = {
    "shapes/__init__.py": "",
    "shapes/core.py": (
        "class Shape:\n"
        "    def area(self):\n        return 0\n"
        "class Circle(Shape):\n"
        "    def __init__(self, r):\n        self.r = r\n"
        "    def area(self):\n        return 3.14 * self.r * self.r\n"
        "def make_circle(r):\n    return Circle(r)\n"
    ),
    "shapes/tests/test_core.py": (
        "from shapes.core import Circle, make_circle\n"
        "def test_area():\n    assert make_circle(2).area() > 0\n"
        "def test_ctor():\n    assert Circle(1).r == 1\n"
    ),
}

def _write_pkg(tmp_path):
    for rel, src in PKG.items():
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(src)
    return tmp_path

def test_extract_nodes_and_users(tmp_path):
    _write_pkg(tmp_path)
    r = extract.extract_repo(tmp_path, import_name="shapes")
    idx = {n: i for i, n in enumerate(r.names)}
    assert "shapes.core.Shape" in idx and r.is_B[idx["shapes.core.Shape"]]
    assert "shapes.core.make_circle" in idx and r.is_F[idx["shapes.core.make_circle"]]
    # Circle inherits Shape -> Shape gains a STRUCTURAL user (a class cites it)
    assert r.structural_users()[idx["shapes.core.Shape"]] >= 1
    # make_circle + 2 tests construct/parametrize Circle -> Circle has OPERATIONAL users
    assert r.operational_users()[idx["shapes.core.Circle"]] >= 2
    # tests are marked plantable operations; concept_pref only on public classes (non-test module)
    assert r.is_test[idx["shapes.tests.test_core.test_area"]]
    assert r.concept_pref[idx["shapes.core.Circle"]]
