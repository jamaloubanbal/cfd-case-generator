"""
Write an OpenFOAM polyMesh as an I-DEAS universal file (.unv), the format SALOME exports and
code_saturne reads natively.

OpenFOAM stores a mesh as faces (owner/neighbour), UNV as cells with an ordered vertex
list, so each cell's vertex order is rebuilt from its faces. Hexahedra (blockMesh) and
prisms (the pipe's wedge cells at the axis) are supported. Every boundary patch becomes a
group of boundary-face elements, so it arrives in code_saturne as a named boundary zone.

(EnSight via foamToEnsight was tried first: code_saturne reads its volume but drops the
boundary parts, leaving every boundary face without a group.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# UNV element types: 115 linear brick, 112 linear wedge (solids); 94 quad, 91 triangle (faces)
_SOLID = {8: 115, 6: 112}
_FACE = {4: 94, 3: 91}


def _cell_vertices(faces: list[np.ndarray], pts: np.ndarray) -> list[int]:
    """Ordered vertices of a hexahedron or prism given its faces, each oriented with the
    normal pointing *into* the cell: base face, then the vertex above each base vertex."""
    sizes = sorted(len(f) for f in faces)
    if sizes == [4] * 6:
        base = faces[0]
    elif sizes == [3, 3, 4, 4, 4]:
        base = next(f for f in faces if len(f) == 3)
    else:
        raise ValueError(f"unsupported cell with face sizes {sizes} (only hexahedra and prisms)")
    base_set = set(base.tolist())
    edges = set()
    for f in faces:
        for a, b in zip(f, np.roll(f, -1)):
            edges.add((int(a), int(b))); edges.add((int(b), int(a)))
    top = []
    for v in base:
        up = [b for (a, b) in edges if a == v and b not in base_set]
        if len(up) != 1:
            raise ValueError("could not match the opposite face of a cell")
        top.append(up[0])
    verts = [int(v) for v in base] + top
    # UNV wants the base normal pointing towards the top face: check, flip if not
    p = pts[verts]
    n = len(base)
    centre_base, centre_top = p[:n].mean(0), p[n:].mean(0)
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    if np.dot(normal, centre_top - centre_base) < 0:
        verts = verts[:n][::-1] + verts[n:][::-1]
    return verts


def polymesh_to_unv(case_dir: str | Path, out: str | Path, skip_patches: tuple[str, ...] = (),
                    rename: dict[str, str] | None = None) -> dict[str, int]:
    """Convert *case_dir*/constant/polyMesh to UNV; returns {group: number of faces}.
    *rename* maps patch names to group names."""
    rename = rename or {}
    from app.postprocess import FoamCase

    case = FoamCase(case_dir)
    pts, faces, owner, neigh = case.points, case.faces, case.owner, case.neighbour
    cells: list[list[np.ndarray]] = [[] for _ in range(case.n_cells)]
    for i, f in enumerate(faces):
        cells[owner[i]].append(f[::-1])               # owner: face normal points outwards
        if i < neigh.size:
            cells[neigh[i]].append(f)                 # neighbour: it points inwards already

    L = ["    -1", "  2411"]
    for nid, (x, y, z) in enumerate(pts, 1):
        L.append(f"{nid:10d}{1:10d}{1:10d}{11:10d}")
        L.append(f"{x:25.16E}{y:25.16E}{z:25.16E}")
    L += ["    -1", "    -1", "  2412"]
    eid = 0
    for cf in cells:
        v = _cell_vertices(cf, pts)
        eid += 1
        L.append(f"{eid:10d}{_SOLID[len(v)]:10d}{1:10d}{1:10d}{7:10d}{len(v):10d}")
        L.append("".join(f"{n + 1:10d}" for n in v))
    groups: dict[str, list[int]] = {}
    for name, b in case.boundary.items():
        if name in skip_patches or not b["nFaces"]:
            continue
        ids = groups.setdefault(rename.get(name, name), [])
        for f in faces[b["startFace"]:b["startFace"] + b["nFaces"]]:
            if len(f) not in _FACE:
                raise ValueError(f"unsupported boundary face with {len(f)} vertices on {name}")
            eid += 1
            ids.append(eid)
            L.append(f"{eid:10d}{_FACE[len(f)]:10d}{2:10d}{1:10d}{7:10d}{len(f):10d}")
            L.append("".join(f"{n + 1:10d}" for n in f))
    L += ["    -1", "    -1", "  2467"]
    for gnum, (name, ids) in enumerate(groups.items(), start=1):
        L.append(f"{gnum:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{len(ids):10d}")
        L.append(name)
        for a in range(0, len(ids), 2):
            L.append("".join(f"{8:10d}{e:10d}{0:10d}{0:10d}" for e in ids[a:a + 2]))
    L.append("    -1")
    Path(out).write_text("\n".join(L) + "\n")
    return {k: len(v) for k, v in groups.items()}
