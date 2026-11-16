"""code_saturne backend: the SALOME box duct generated, run and judged (skipped without
code_saturne, e.g. in CI; set CODE_SATURNE to its bin/code_saturne)."""

import shutil
from pathlib import Path

import pytest

from app import saturne

pytestmark = pytest.mark.skipif(saturne.find_executable() is None or not shutil.which("ideasUnvToFoam"),
                                reason="code_saturne or OpenFOAM (mesh import) not installed")

MESH = Path(__file__).resolve().parent.parent / "examples" / "meshes" / "box_duct_salome.unv"


def test_salome_duct_runs_and_is_trustworthy(tmp_path):
    from app.mesh_import import apply_roles, attach, import_unv, map_roles
    from app.sanitize import sanitize_analysis
    from tests.fixtures import SIMPLE_FOAM_ANALYSIS

    a = dict(SIMPLE_FOAM_ANALYSIS, turb_model="kEpsilon", characteristic_length=0.2,
             boundary_conditions=[{"patch_name": "inlet", "type": "fixedValue",
                                   "values": {"U": [2.0, 0, 0]}}])
    a, _ = sanitize_analysis(a, "Air at 2 m/s through this duct")
    mesh = import_unv(MESH)
    a = attach(a, mesh, apply_roles(mesh, map_roles(mesh.patches)[0]))
    case = saturne.generate(a, MESH, tmp_path / "study", "Air at 2 m/s through this duct")
    saturne.run(case)
    rev = saturne.review(case)
    status = {c.name: c.status for c in rev.checks}
    assert status.get("Mass balance") == "pass", rev.checks
    assert status.get("Request") == "pass", rev.checks
    assert rev.verdict != "unreliable", rev.checks
