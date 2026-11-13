"""
Regression tests for bugs found by running generated cases in OpenFOAM.

These run without OpenFOAM (they inspect the generated text) so they guard the
fixes in every environment; ``test_openfoam_integration.py`` checks the same
cases end-to-end when OpenFOAM is available.
"""

import copy
import re
from pathlib import Path

import pytest

from app.generators import (
    BuoyantSimpleFoamGenerator,
    InterFoamGenerator,
    PimpleFoamGenerator,
    SimpleFoamGenerator,
)
from app.generators.base import _compute_grading_ratio, _MAX_CELL_GROWTH
from tests.test_generators import (
    AIRFOIL_ANALYSIS,
    BACKSTEP_ANALYSIS,
    BSF_ANALYSIS,
    CYLINDER_ANALYSIS,
    INTER_FOAM_ANALYSIS,
    PIMPLE_FOAM_ANALYSIS,
    T_JUNCTION_ANALYSIS,
    _generate_stl_case,
)


def _read(case: Path, rel: str) -> str:
    return (case / rel).read_text()


# ── Mesh parsing helpers ─────────────────────────────────────────────────────

def _parse_block_mesh(text: str):
    """Return (vertices, blocks) from a rendered blockMeshDict.

    vertices: list[(x, y, z)];  blocks: list[(vertex_ids, cells, grading_str)]
    """
    text = re.sub(r"//[^\n]*", "", text)
    vsec = re.search(r"^vertices\s*\((.*?)^\);", text, re.S | re.M).group(1)
    vertices = [tuple(float(c) for c in m.split()) for m in re.findall(r"\(([^()]+)\)", vsec)]
    bsec = re.search(r"^blocks\s*\((.*?)^\);", text, re.S | re.M).group(1)
    blocks = []
    for m in re.finditer(r"hex\s*\(([^)]*)\)\s*\(([^)]*)\)\s*simpleGrading\s*\((.*?)\)\s*$", bsec, re.M):
        ids = [int(i) for i in m.group(1).split()]
        cells = [int(c) for c in m.group(2).split()]
        blocks.append((ids, cells, m.group(3)))
    return vertices, blocks


def _signed_area_xy(vertices, ids):
    pts = [vertices[i][:2] for i in ids[:4]]
    return 0.5 * sum(
        pts[k][0] * pts[(k + 1) % 4][1] - pts[(k + 1) % 4][0] * pts[k][1] for k in range(4)
    )


# ── pimpleFoam / interFoam solver dictionaries ───────────────────────────────

@pytest.mark.parametrize("turb", ["kEpsilon", "kOmegaSST", "SpalartAllmaras"])
def test_pimple_has_final_solvers_for_turbulence_fields(tmp_path, turb):
    a = {**PIMPLE_FOAM_ANALYSIS, "turb_model": turb}
    PimpleFoamGenerator().generate(a, tmp_path)
    fv = _read(tmp_path, "system/fvSolution")
    assert '"(k|epsilon|omega|nuTilda)Final"' in fv


def test_interfoam_has_pcorr_solver_and_v1912_schemes(tmp_path):
    InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, tmp_path)
    assert '"pcorr.*"' in _read(tmp_path, "system/fvSolution")
    schemes = _read(tmp_path, "system/fvSchemes")
    assert "div(phirb,alpha)" in schemes
    assert "interfaceCompression" not in schemes  # OpenFOAM.org-only syntax


# ── Mesh topology ────────────────────────────────────────────────────────────

def test_cylinder_blocks_are_not_inside_out(tmp_path):
    SimpleFoamGenerator().generate(CYLINDER_ANALYSIS, tmp_path)
    vertices, blocks = _parse_block_mesh(_read(tmp_path, "system/blockMeshDict"))
    assert len(blocks) == 12
    for ids, _, _ in blocks:
        assert _signed_area_xy(vertices, ids) > 0, f"block {ids} is clockwise (inside-out)"


@pytest.mark.parametrize("analysis", [T_JUNCTION_ANALYSIS, BACKSTEP_ANALYSIS, AIRFOIL_ANALYSIS],
                         ids=["t_junction", "backstep", "airfoil"])
def test_blocks_sharing_a_vertical_face_have_identical_y_grading(tmp_path, analysis):
    """blockMesh aborts when two blocks sharing a face disagree on its grading."""
    a = copy.deepcopy(analysis)
    a["Re"] = 1e6          # high Re → non-uniform wall grading is actually produced
    SimpleFoamGenerator().generate(a, tmp_path)
    vertices, blocks = _parse_block_mesh(_read(tmp_path, "system/blockMeshDict"))

    def y_grading(g: str) -> str:
        # simpleGrading (gx gy gz) where each entry may be a compound "((…)(…))"
        depth, parts, cur = 0, [], ""
        for ch in g.strip():
            if ch == " " and depth == 0:
                if cur:
                    parts.append(cur)
                cur = ""
                continue
            depth += ch == "("
            depth -= ch == ")"
            cur += ch
        parts.append(cur)
        return parts[1]

    for i, (a_ids, _, ga) in enumerate(blocks):
        right_face = {a_ids[1], a_ids[2]}          # i = max edge of the front face
        for j, (b_ids, _, gb) in enumerate(blocks):
            if i != j and {b_ids[0], b_ids[3]} == right_face:
                assert y_grading(ga) == y_grading(gb), (
                    f"blocks {i} and {j} share a face but use y-grading {ga!r} vs {gb!r}"
                )


def test_t_junction_assigns_branch_side_walls(tmp_path):
    SimpleFoamGenerator().generate(T_JUNCTION_ANALYSIS, tmp_path)
    bmd = _read(tmp_path, "system/blockMeshDict")
    assert "(5  15 18 8)" in bmd and "(6  9  19 16)" in bmd


def test_t_junction_branch_inlet_points_into_domain(tmp_path):
    a = copy.deepcopy(T_JUNCTION_ANALYSIS)
    a["boundary_conditions"] = [bc for bc in a["boundary_conditions"] if bc["patch_name"] != "inlet2"]
    SimpleFoamGenerator().generate(a, tmp_path)
    u = _read(tmp_path, "0/U")
    block = re.search(r"inlet2\s*\{(.*?)\}", u, re.S).group(1)
    vec = [float(c) for c in re.search(r"uniform \(([^)]*)\)", block).group(1).split()]
    assert vec[1] < 0 and vec[0] == 0, f"inlet2 velocity {vec} must point down (-y) into the branch"


def test_stl_background_patch_types_match_field_types(tmp_path):
    case = _generate_stl_case(tmp_path)
    bmd = _read(case, "system/blockMeshDict")
    assert "type symmetry;" not in bmd
    assert "type symmetryPlane;" in bmd
    assert "symmetryPlane" in _read(case, "0/U")


# ── Physics / numerics ───────────────────────────────────────────────────────

def test_grading_respects_max_cell_growth():
    # A tiny wall-function target over few cells used to give ratios of ~5e4
    n = 20
    ratio = _compute_grading_ratio(10.0, n, 1e-6)
    assert ratio <= _MAX_CELL_GROWTH ** (n - 1) * (1 + 1e-9)


def test_buoyant_turbulence_initialised_from_buoyant_velocity(tmp_path):
    a = {**BSF_ANALYSIS, "turb_model": "kEpsilon"}
    BuoyantSimpleFoamGenerator().generate(a, tmp_path)
    k = float(re.search(r"internalField\s+uniform\s+(\S+);", _read(tmp_path, "0/k")).group(1))
    # U_b = sqrt(g·ΔT/T·L) ≈ 1.2 m/s → k = 1.5·(0.05·U_b)² ≈ 5.7e-3
    assert 1e-3 < k < 1e-2


# ── Run scripts ──────────────────────────────────────────────────────────────

def test_allrun_works_without_runfunctions(tmp_path):
    SimpleFoamGenerator().generate({**PIMPLE_FOAM_ANALYSIS, "solver": "simpleFoam"}, tmp_path)
    allrun = _read(tmp_path, "Allrun")
    assert 'if [ -n "$WM_PROJECT_DIR" ] && [ -f "$WM_PROJECT_DIR/bin/tools/RunFunctions" ]' in allrun
    assert "runApplication() {" in allrun
