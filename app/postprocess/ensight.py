"""
code_saturne results (EnSight Gold, C Binary) with the interface of ``FoamCase`` that the
validation benchmarks use: ``latest_time()``, ``field(name, time)``, ``cell_centres``,
``n_cells``. Field names follow OpenFOAM ("U", "p"), and the pressure is returned
divided by the density, i.e. kinematic like OpenFOAM's incompressible solvers.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

_NODES = {"tetra4": 4, "penta6": 6, "hexa8": 8, "pyramid5": 5, "tria3": 3, "quad4": 4}
_NAMES = {"U": ("velocity", 3), "p": ("pressure", 1), "k": ("k", 1), "omega": ("omega", 1),
          "epsilon": ("epsilon", 1), "nut": ("turbvisc", 1)}


class _Reader:
    def __init__(self, data: bytes):
        self.d, self.i = data, 0

    def s(self) -> str:
        v = self.d[self.i:self.i + 80].split(b"\0")[0].decode().strip()
        self.i += 80
        return v

    def ints(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.d, "<i4", n, self.i)
        self.i += 4 * n
        return v

    def floats(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.d, "<f4", n, self.i)
        self.i += 4 * n
        return v


def read_geometry(path: Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Nodes and (element type, 0-based connectivity) blocks of the first part."""
    r = _Reader(Path(path).read_bytes())
    r.s(); r.s(); r.s()
    node_id, elem_id = r.s(), r.s()
    tok = r.s()
    if tok == "extents":
        r.floats(6)
        tok = r.s()
    if tok != "part":
        raise ValueError(f"{path}: unexpected EnSight token {tok!r}")
    r.ints(1); r.s()
    if r.s() != "coordinates":
        raise ValueError(f"{path}: no coordinates block")
    nn = int(r.ints(1)[0])
    if "given" in node_id or "ignore" in node_id:
        r.ints(nn)
    xyz = np.stack([r.floats(nn), r.floats(nn), r.floats(nn)], axis=1).astype(float)
    blocks = []
    while r.i < len(r.d):
        et = r.s()
        if et not in _NODES:
            break
        ne = int(r.ints(1)[0])
        if "given" in elem_id or "ignore" in elem_id:
            r.ints(ne)
        blocks.append((et, r.ints(ne * _NODES[et]).reshape(ne, _NODES[et]) - 1))
    return xyz, blocks


def read_variable(path: Path, blocks: list, ncomp: int) -> np.ndarray:
    r = _Reader(Path(path).read_bytes())
    r.s()
    if r.s() != "part":
        raise ValueError(f"{path}: unexpected layout")
    r.ints(1)
    out = []
    for et, conn in blocks:
        if r.s() != et:
            raise ValueError(f"{path}: element blocks do not match the geometry")
        out.append(np.stack([r.floats(len(conn)) for _ in range(ncomp)], axis=1).astype(float))
    v = np.concatenate(out)
    return v[:, 0] if ncomp == 1 else v


class SaturneResult:
    """Fluid-domain results of a code_saturne run (``RESU/<id>/postprocessing``)."""

    def __init__(self, resu_dir: str | Path, rho: float = 1.0):
        self.path = Path(resu_dir)
        self.post = self.path / "postprocessing"
        self.rho = rho
        self._xyz, self._blocks = read_geometry(self.post / "results_fluid_domain.geo")

    def latest_time(self) -> str:
        files = sorted(self.post.glob("results_fluid_domain.velocity.*"))
        return files[-1].suffix.lstrip(".")

    @cached_property
    def cell_centres(self) -> np.ndarray:
        return np.concatenate([self._xyz[c].mean(axis=1) for _, c in self._blocks])

    @property
    def n_cells(self) -> int:
        return sum(len(c) for _, c in self._blocks)

    def field(self, name: str, time: str) -> np.ndarray:
        stem, ncomp = _NAMES.get(name, (name.lower(), 1))
        v = read_variable(self.post / f"results_fluid_domain.{stem}.{time}", self._blocks, ncomp)
        return v / self.rho if name == "p" else v
