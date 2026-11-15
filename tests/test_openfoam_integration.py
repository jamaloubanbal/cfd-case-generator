"""
End-to-end tests: generated cases must actually run in OpenFOAM.

For every supported solver × geometry (× turbulence model) combination this
generates a case, shortens the run, executes the case's own ``Allrun`` script
and asserts that

* ``blockMesh`` (and snappyHexMesh for STL cases) succeeds,
* ``checkMesh`` reports ``Mesh OK``,
* no faces were left unassigned (no ``defaultFaces`` patch),
* the solver runs its iterations / time steps and exits cleanly without NaNs.

These tests are skipped automatically when OpenFOAM is not available.  To run
them, source your OpenFOAM environment first, e.g.::

    source /usr/share/openfoam/etc/bashrc      # Debian/Ubuntu package
    pytest -m openfoam -v

Target version: OpenFOAM v1912 (ESI / openfoam.com).
"""

from __future__ import annotations

import copy
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.generators import GENERATORS
from tests.fixtures import (
    AIRFOIL_ANALYSIS,
    BACKSTEP_ANALYSIS,
    BSF_ANALYSIS,
    CYLINDER_ANALYSIS,
    INTER_FOAM_ANALYSIS,
    PIMPLE_FOAM_ANALYSIS,
    PIPE_ANALYSIS,
    RECT_OBS_ANALYSIS,
    RHO_SIMPLE_FOAM_ANALYSIS,
    SIMPLE_FOAM_ANALYSIS,
    T_JUNCTION_ANALYSIS,
    _STL_ANALYSIS_BASE,
)

pytestmark = [
    pytest.mark.openfoam,
    pytest.mark.skipif(
        not (os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh")),
        reason="OpenFOAM environment not sourced (WM_PROJECT_DIR / blockMesh missing)",
    ),
]

STEADY_ITERATIONS = 30
TRANSIENT_STEPS = 15
TIMEOUT_S = 600

BASE_ANALYSES = {
    "simpleFoam": SIMPLE_FOAM_ANALYSIS,
    "pimpleFoam": PIMPLE_FOAM_ANALYSIS,
    "buoyantSimpleFoam": BSF_ANALYSIS,
    "rhoSimpleFoam": RHO_SIMPLE_FOAM_ANALYSIS,
    "interFoam": INTER_FOAM_ANALYSIS,
}

GEOMETRY_ANALYSES = {
    "t_junction": T_JUNCTION_ANALYSIS,
    "rectangular_obstacle": RECT_OBS_ANALYSIS,
    "backstep": BACKSTEP_ANALYSIS,
    "cylinder": CYLINDER_ANALYSIS,
    "pipe": PIPE_ANALYSIS,
    "airfoil": AIRFOIL_ANALYSIS,
}

# Solvers that can be combined with the shared multi-block geometries
# (buoyantSimpleFoam is cavity-only and interFoam is dambreak-only).
FORCED_FLOW_SOLVERS = ("simpleFoam", "pimpleFoam", "rhoSimpleFoam")
TURB_MODELS = ("laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras")

_GEOMETRY_KEYS = ("geometry_type", "geometry_params", "boundary_conditions",
                  "characteristic_length")


def _case_matrix() -> list[tuple[str, str, dict]]:
    """Build (test-id, solver, analysis) tuples for every supported combination."""
    cases = []
    # Every solver on its default geometry with every turbulence model
    for solver, base in BASE_ANALYSES.items():
        for turb in TURB_MODELS:
            if solver == "interFoam" and turb != "kEpsilon":
                continue  # interFoam is always laminar — turb model is ignored
            a = copy.deepcopy(base)
            a["turb_model"] = turb
            cases.append((f"{solver}-{a['geometry_type']}-{turb}", solver, a))

    # Every forced-flow solver on every multi-block geometry
    for geo, geo_analysis in GEOMETRY_ANALYSES.items():
        for solver in FORCED_FLOW_SOLVERS:
            a = copy.deepcopy(BASE_ANALYSES[solver])
            a.update({k: copy.deepcopy(geo_analysis[k]) for k in _GEOMETRY_KEYS if k in geo_analysis})
            if solver == "rhoSimpleFoam":
                # Compressible solver: use a genuinely compressible inflow (~Ma 0.3)
                for bc in a["boundary_conditions"]:
                    if "U" in bc.get("values", {}):
                        bc["values"]["U"] = [100.0 * float(c) for c in bc["values"]["U"]]
            cases.append((f"{solver}-{geo}-{a['turb_model']}", solver, a))

    # Parallel run (decomposePar → mpirun → reconstructPar)
    if shutil.which("mpirun"):
        a = copy.deepcopy(SIMPLE_FOAM_ANALYSIS)
        a["n_cores"] = 2
        cases.append(("simpleFoam-channel-parallel", "simpleFoam", a))

    # Mesh imported from SALOME (.unv) → ideasUnvToFoam, groups mapped to roles by name
    a = copy.deepcopy(SIMPLE_FOAM_ANALYSIS)
    a.update(turb_model="kOmegaSST", characteristic_length=0.2,
             boundary_conditions=[{"patch_name": "inlet", "type": "fixedValue", "values": {"U": [5.0, 0, 0]}}],
             _unv=str(Path(__file__).resolve().parent.parent / "examples" / "meshes" / "l_duct.unv"))
    cases.append(("simpleFoam-imported_mesh-kOmegaSST", "simpleFoam", a))

    # STL import → surfaceFeatureExtract → snappyHexMesh
    cases.append(("simpleFoam-stl_import-kOmegaSST", "simpleFoam", copy.deepcopy(_STL_ANALYSIS_BASE)))
    return cases


def _shorten_run(case_dir: Path) -> None:
    """Limit the run to a few iterations / time steps so the suite stays fast."""
    cd = case_dir / "system" / "controlDict"
    text = cd.read_text()
    delta_t = float(re.search(r"^deltaT\s+([^;]+);", text, re.M).group(1))
    if re.search(r"^application\s+(pimpleFoam|interFoam);", text, re.M):
        end = TRANSIENT_STEPS * delta_t
    else:
        end = STEADY_ITERATIONS * delta_t
    text = re.sub(r"^endTime\s+[^;]+;", f"endTime         {end:g};", text, flags=re.M)
    # Make sure the shortened run writes a result (reconstructPar needs one)
    text = re.sub(r"^writeControl\s+[^;]+;", "writeControl    runTime;", text, flags=re.M)
    text = re.sub(r"^writeInterval\s+[^;]+;", f"writeInterval   {end:g};", text, flags=re.M)
    cd.write_text(text)


def _log(case_dir: Path, app: str) -> str:
    path = case_dir / f"log.{app}"
    return path.read_text(errors="replace") if path.exists() else ""


@pytest.mark.parametrize(
    "solver, analysis",
    [pytest.param(s, a, id=i) for i, s, a in _case_matrix()],
)
def test_case_runs_in_openfoam(solver: str, analysis: dict, tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    if analysis.get("_unv"):
        from app.mesh_import import apply_roles, attach, import_unv, map_roles
        mesh = import_unv(analysis["_unv"])
        analysis = attach(analysis, mesh, apply_roles(mesh, map_roles(mesh.patches)[0]))
    GENERATORS[solver]().generate(analysis, case_dir)
    _shorten_run(case_dir)

    result = subprocess.run(
        ["./Allrun"], cwd=case_dir, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    logs = {p.name: p.read_text(errors="replace")[-3000:] for p in case_dir.glob("log.*")}
    assert result.returncode == 0, (
        f"Allrun failed (rc={result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\nlogs:\n" + "\n".join(f"--- {k}\n{v}" for k, v in logs.items())
    )

    # Mesh: every face must belong to a declared patch
    boundary = (case_dir / "constant" / "polyMesh" / "boundary").read_text()
    assert "defaultFaces" not in boundary, "blockMesh left unassigned faces (defaultFaces patch)"

    # Mesh quality
    check = subprocess.run(
        ["checkMesh"], cwd=case_dir, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    assert check.returncode == 0, check.stdout[-3000:]
    assert "Mesh OK" in check.stdout, check.stdout[-3000:]

    # Solver: ran, finished, no NaNs
    solver_log = _log(case_dir, solver)
    assert re.search(r"^End\s*$", solver_log, re.M), f"{solver} did not finish:\n{solver_log[-3000:]}"
    steps = re.findall(r"^Time = (\S+)", solver_log, re.M)
    assert steps, f"{solver} did not advance in time"
    assert not re.search(r"\bnan\b", solver_log, re.I), f"{solver} produced NaN:\n{solver_log[-3000:]}"
    for value in re.findall(r"Initial residual = (\S+),", solver_log):
        assert math.isfinite(float(value))
