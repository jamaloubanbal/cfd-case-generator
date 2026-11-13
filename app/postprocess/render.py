"""
Exact 2-D rendering of OpenFOAM meshes and cell fields with matplotlib.

Each cell is drawn as its front-face polygon (the face on the ``empty`` or
``wedge`` patch with the lowest z), so the picture is the real mesh: holes
such as a cylinder or airfoil stay empty and nothing is interpolated.
"""

from __future__ import annotations

import numpy as np

from .foam import FoamCase


def front_polygons(case: FoamCase) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (polygons in the x–y plane, owning cell index per polygon).

    For axisymmetric wedge meshes the radial coordinate √(y²+z²) is used.
    """
    planes = [n for n, b in case.boundary.items() if b["type"] in ("empty", "wedge")]
    if not planes:
        raise ValueError("not a 2-D case (no empty or wedge patch)")
    is_wedge = any(case.boundary[n]["type"] == "wedge" for n in planes)
    if is_wedge:
        planes = [min((n for n in planes if case.boundary[n]["type"] == "wedge"),
                      key=lambda n: case.face_centres[case.patch_faces(n)][:, 2].mean())]

    pts = case.points
    polys, cells = [], []
    for name in planes:
        sl = case.patch_faces(name)
        fc = case.face_centres[sl]
        z_front = fc[:, 2].min()
        for i, face_idx in enumerate(range(sl.start, sl.stop)):
            if not is_wedge and abs(fc[i, 2] - z_front) > 1e-9 * max(1.0, abs(z_front)):
                continue
            p = pts[case.faces[face_idx]]
            xy = np.column_stack([p[:, 0], np.hypot(p[:, 1], p[:, 2])]) if is_wedge else p[:, :2]
            polys.append(xy)
            cells.append(case.owner[face_idx])
    return polys, np.array(cells)


def plot_cells(ax, case: FoamCase, values: np.ndarray | None = None, *, cmap="viridis",
               edges: bool = False, edge_width: float = 0.15, vmin=None, vmax=None,
               label: str | None = None):
    """Draw the mesh (``values=None``) or a cell field on *ax*; returns the collection."""
    from matplotlib.collections import PolyCollection

    polys, cells = front_polygons(case)
    if values is None:
        coll = PolyCollection(polys, facecolors="white", edgecolors="0.25", linewidths=edge_width)
    else:
        # Edges drawn in the face colour hide anti-aliasing seams between cells
        coll = PolyCollection(polys, array=np.asarray(values)[cells], cmap=cmap,
                              edgecolors="0.3" if edges else "face",
                              linewidths=edge_width if edges else 0.4)
        coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    allp = np.vstack(polys)
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
    ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_aspect("equal")
    if values is not None and label:
        ax.figure.colorbar(coll, ax=ax, label=label, shrink=0.85, pad=0.02)
    return coll
