"""Tests for app.postprocess — OpenFOAM ASCII reader and renderer (no OpenFOAM needed).

The fixture is a hand-written 2-cell mesh: two unit cubes side by side along x.
"""

from pathlib import Path

import numpy as np
import pytest

from app.postprocess import FoamCase

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""

#  y
#  3───4───5      front face z = 0 (points 0-5), back face z = 1 (points 6-11)
#  │ 0 │ 1 │
#  0───1───2 → x
POINTS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)]
FACES = [
    (1, 4, 10, 7),                       # 0  internal (owner 0, neighbour 1)
    (0, 6, 9, 3),                        # 1  left   (inlet, owner 0)
    (2, 5, 11, 8),                       # 2  right  (outlet, owner 1)
    (0, 1, 7, 6), (1, 2, 8, 7),          # 3,4 bottom (walls)
    (3, 9, 10, 4), (4, 10, 11, 5),       # 5,6 top    (walls)
    (0, 3, 4, 1), (1, 4, 5, 2),          # 7,8 front  (frontAndBack)
    (6, 7, 10, 9), (7, 8, 11, 10),       # 9,10 back  (frontAndBack)
]
OWNER = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
BOUNDARY = [("inlet", "patch", 1, 1), ("outlet", "patch", 2, 1),
            ("walls", "wall", 3, 4), ("frontAndBack", "empty", 7, 4)]


@pytest.fixture
def case(tmp_path: Path) -> Path:
    mesh = tmp_path / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (mesh / "points").write_text(HEADER.format(cls="vectorField", obj="points")
                                 + f"{len(POINTS)}\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in POINTS) + "\n)\n")
    (mesh / "faces").write_text(HEADER.format(cls="faceList", obj="faces")
                                + f"{len(FACES)}\n(\n" + "\n".join(f"4({' '.join(map(str, f))})" for f in FACES) + "\n)\n")
    (mesh / "owner").write_text(HEADER.format(cls="labelList", obj="owner")
                                + f"{len(OWNER)}\n(\n" + "\n".join(map(str, OWNER)) + "\n)\n")
    (mesh / "neighbour").write_text(HEADER.format(cls="labelList", obj="neighbour") + "1\n(\n1\n)\n")
    (mesh / "boundary").write_text(HEADER.format(cls="polyBoundaryMesh", obj="boundary")
        + f"{len(BOUNDARY)}\n(\n" + "".join(
            f"    {n}\n    {{\n        type {t};\n        nFaces {nf};\n        startFace {s};\n    }}\n"
            for n, t, s, nf in BOUNDARY) + ")\n")
    t = tmp_path / "1"
    t.mkdir()
    (t / "U").write_text(HEADER.format(cls="volVectorField", obj="U") + """
dimensions      [0 1 -1 0 0 0 0];
internalField   nonuniform List<vector>
2
(
(1 0 0)
(2 0.5 0)
)
;
boundaryField
{
    inlet   { type fixedValue; value uniform (3 0 0); }
    outlet  { type zeroGradient; }
    walls   { type noSlip; }
    frontAndBack { type empty; }
}
""")
    (t / "p").write_text(HEADER.format(cls="volScalarField", obj="p") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 7;
boundaryField
{
    inlet   { type zeroGradient; }
    outlet  { type fixedValue; value uniform 0; }
    walls   { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    (tmp_path / "0").mkdir()
    return tmp_path


def test_mesh_topology(case):
    c = FoamCase(case)
    assert c.points.shape == (12, 3)
    assert len(c.faces) == 11
    assert c.n_cells == 2
    assert c.boundary["walls"] == {"type": "wall", "startFace": 3, "nFaces": 4}


def test_cell_centres_and_face_areas(case):
    c = FoamCase(case)
    np.testing.assert_allclose(c.cell_centres, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    np.testing.assert_allclose(c.face_areas[c.patch_faces("inlet")], [[-1, 0, 0]])   # outward
    np.testing.assert_allclose(c.face_areas[0], [1, 0, 0])                           # owner → neighbour


def test_fields_uniform_and_nonuniform(case):
    c = FoamCase(case)
    assert c.latest_time() == "1"
    np.testing.assert_allclose(c.field("U", "1"), [[1, 0, 0], [2, 0.5, 0]])
    np.testing.assert_allclose(c.field("p", "1"), [7, 7])


def test_patch_values_and_zero_gradient_fallback(case):
    c = FoamCase(case)
    np.testing.assert_allclose(c.patch_field("U", "1", "inlet"), [[3, 0, 0]])
    np.testing.assert_allclose(c.patch_field("U", "1", "outlet"), [[2, 0.5, 0]])     # owner cell value
    np.testing.assert_allclose(c.patch_field("p", "1", "outlet"), [0])


def test_render_front_polygons(case):
    pytest.importorskip("matplotlib")
    from app.postprocess.render import front_polygons
    polys, cells = front_polygons(FoamCase(case))
    assert sorted(cells.tolist()) == [0, 1]
    assert all(p.shape == (4, 2) for p in polys)
