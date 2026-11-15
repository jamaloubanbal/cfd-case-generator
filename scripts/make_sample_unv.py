"""
Write a sample SALOME-style mesh: a 3-D L-shaped duct as an I-DEAS universal file (.unv).

    python scripts/make_sample_unv.py            # → examples/meshes/l_duct.unv

SALOME exports meshes in this format (File → Export → UNV). The groups use names a SALOME
user might choose (French, as SALOME is developed in France): "Entree" (inlet, facing −x),
"Sortie" (outlet, facing +y) and "Parois" (walls) — so mapping them to boundary roles is a
real test, and the flow does not run along x.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "examples" / "meshes" / "l_duct.unv"

# Two boxes (x0, x1, y0, y1, z0, z1) in metres and their cell counts: an L-shaped duct
# 0.2 m × 0.2 m in cross-section: 1 m along x, then 0.8 m up along y.
BOXES = [((0.0, 1.0, 0.0, 0.2, 0.0, 0.2), (40, 8, 8)),
         ((0.8, 1.0, 0.2, 1.0, 0.0, 0.2), (8, 32, 8))]


def _grid(box, n):
    (x0, x1, y0, y1, z0, z1), (nx, ny, nz) = box, n
    xs = [x0 + (x1 - x0) * i / nx for i in range(nx + 1)]
    ys = [y0 + (y1 - y0) * j / ny for j in range(ny + 1)]
    zs = [z0 + (z1 - z0) * k / nz for k in range(nz + 1)]
    return xs, ys, zs


def build():
    nodes: dict[tuple, int] = {}

    def node(p):
        key = tuple(round(c, 9) for c in p)
        if key not in nodes:
            nodes[key] = len(nodes) + 1
        return nodes[key]

    hexes = []
    for box, n in BOXES:
        xs, ys, zs = _grid(box, n)
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(zs) - 1):
                    c = [(xs[i], ys[j], zs[k]), (xs[i + 1], ys[j], zs[k]), (xs[i + 1], ys[j + 1], zs[k]),
                         (xs[i], ys[j + 1], zs[k]), (xs[i], ys[j], zs[k + 1]), (xs[i + 1], ys[j], zs[k + 1]),
                         (xs[i + 1], ys[j + 1], zs[k + 1]), (xs[i], ys[j + 1], zs[k + 1])]
                    hexes.append([node(p) for p in c])

    # Boundary faces: hex faces that belong to exactly one hex
    local = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    count: dict[frozenset, list] = {}
    for h in hexes:
        for f in local:
            q = [h[i] for i in f]
            count.setdefault(frozenset(q), []).append(q)
    coords = {v: k for k, v in nodes.items()}
    groups = {"Entree": [], "Sortie": [], "Parois": []}
    for faces in count.values():
        if len(faces) != 1:
            continue
        q = faces[0]
        pts = [coords[i] for i in q]
        if all(abs(p[0]) < 1e-9 for p in pts):
            groups["Entree"].append(q)
        elif all(abs(p[1] - 1.0) < 1e-9 for p in pts):
            groups["Sortie"].append(q)
        else:
            groups["Parois"].append(q)
    return nodes, hexes, groups


def write(path: Path) -> None:
    nodes, hexes, groups = build()
    L = ["    -1", "  2411"]
    for (x, y, z), nid in sorted(nodes.items(), key=lambda kv: kv[1]):
        L.append(f"{nid:10d}{1:10d}{1:10d}{11:10d}")
        L.append(f"{x:25.16E}{y:25.16E}{z:25.16E}")
    L += ["    -1", "    -1", "  2412"]
    eid = 0
    group_ids: dict[str, list[int]] = {g: [] for g in groups}
    for h in hexes:                                    # 115: linear solid brick
        eid += 1
        L.append(f"{eid:10d}{115:10d}{1:10d}{1:10d}{7:10d}{8:10d}")
        L.append("".join(f"{n:10d}" for n in h))
    for name, faces in groups.items():                 # 94: linear quadrilateral (boundary face)
        for q in faces:
            eid += 1
            group_ids[name].append(eid)
            L.append(f"{eid:10d}{94:10d}{2:10d}{1:10d}{7:10d}{4:10d}")
            L.append("".join(f"{n:10d}" for n in q))
    L += ["    -1", "    -1", "  2467"]
    for gnum, (name, ids) in enumerate(group_ids.items(), start=1):
        L.append(f"{gnum:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{len(ids):10d}")
        L.append(name)
        for a in range(0, len(ids), 2):
            L.append("".join(f"{8:10d}{e:10d}{0:10d}{0:10d}" for e in ids[a:a + 2]))
    L.append("    -1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    print(f"{path}: {len(nodes)} nodes, {len(hexes)} hexahedra, groups "
          + ", ".join(f"{g} ({len(f)} faces)" for g, f in groups.items()))


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
