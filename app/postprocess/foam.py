"""
Minimal reader for OpenFOAM ASCII cases: polyMesh, cell centres and fields.

Supports what blockMesh / snappyHexMesh and the solvers write with
``writeFormat ascii`` (the setting used by every generated case):

* ``constant/polyMesh/{points,faces,owner,neighbour,boundary}``
* ``volScalarField`` / ``volVectorField`` internal fields (uniform or
  nonuniform) and patch ``value`` entries.

Example
-------
>>> case = FoamCase("path/to/case")
>>> t = case.latest_time()
>>> U = case.field("U", t)                 # (n_cells, 3)
>>> xc = case.cell_centres                 # (n_cells, 3)
>>> T_wall = case.patch_field("T", t, "hotWall")
"""

from __future__ import annotations

import gzip
import re
from functools import cached_property
from pathlib import Path

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)


def _read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(errors="replace")
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gzip.decompress(gz.read_bytes()).decode(errors="replace")
    raise FileNotFoundError(path)


def _strip_header(text: str) -> str:
    """Drop the FoamFile header and comments."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)


def _list_body(text: str, start: int = 0) -> tuple[int, str, int]:
    """Find ``N ( ... )`` at or after *start*: return (N, body, end_index)."""
    m = re.compile(r"(\d+)\s*\(").search(text, start)
    if m is None:
        raise ValueError("no OpenFOAM list found")
    n = int(m.group(1))
    depth, i = 1, m.end()
    while depth:
        c = text[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    return n, text[m.end():i - 1], i


def _parse_values(spec: str, n_expected: int | None = None) -> np.ndarray | float:
    """Parse ``uniform X`` / ``nonuniform List<T> N(...)`` into numbers."""
    spec = spec.strip()
    if spec.startswith("uniform"):
        nums = [float(v) for v in _NUM_RE.findall(spec[len("uniform"):])]
        value = np.array(nums) if len(nums) > 1 else nums[0]
        if n_expected is None:
            return value
        return np.tile(value, (n_expected, 1)) if np.ndim(value) else np.full(n_expected, value)
    n, body, _ = _list_body(spec)
    nums = np.array([float(v) for v in _NUM_RE.findall(body)])
    if n == 0:
        return np.zeros(0)
    width = len(nums) // n
    return nums.reshape(n, width) if width > 1 else nums


class FoamCase:
    """Lazy reader for one OpenFOAM case directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mesh_dir = self.path / "constant" / "polyMesh"

    # ── Time directories ───────────────────────────────────────────────────
    def times(self) -> list[str]:
        out = []
        for d in self.path.iterdir():
            if d.is_dir():
                try:
                    float(d.name)
                except ValueError:
                    continue
                out.append(d.name)
        return sorted(out, key=float)

    def latest_time(self) -> str:
        times = [t for t in self.times() if float(t) > 0] or self.times()
        if not times:
            raise FileNotFoundError(f"no time directories in {self.path}")
        return times[-1]

    # ── Mesh ───────────────────────────────────────────────────────────────
    @cached_property
    def points(self) -> np.ndarray:
        _, body, _ = _list_body(_strip_header(_read_text(self._mesh_dir / "points")))
        return np.array([float(v) for v in _NUM_RE.findall(body)]).reshape(-1, 3)

    @cached_property
    def faces(self) -> list[np.ndarray]:
        _, body, _ = _list_body(_strip_header(_read_text(self._mesh_dir / "faces")))
        return [np.array(f.split(), dtype=int) for f in re.findall(r"\d+\s*\(([^)]*)\)", body)]

    def _labels(self, name: str) -> np.ndarray:
        _, body, _ = _list_body(_strip_header(_read_text(self._mesh_dir / name)))
        return np.array(body.split(), dtype=int)

    @cached_property
    def owner(self) -> np.ndarray:
        return self._labels("owner")

    @cached_property
    def neighbour(self) -> np.ndarray:
        return self._labels("neighbour")

    @cached_property
    def n_cells(self) -> int:
        return int(max(self.owner.max(), self.neighbour.max() if self.neighbour.size else 0)) + 1

    @cached_property
    def boundary(self) -> dict[str, dict]:
        """Patch name → {"type", "startFace", "nFaces"}."""
        text = _strip_header(_read_text(self._mesh_dir / "boundary"))
        _, body, _ = _list_body(text)
        out = {}
        for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", body):
            entries = dict(re.findall(r"(\w+)\s+([^;]+);", block))
            out[name] = {"type": entries["type"].strip(),
                         "startFace": int(entries["startFace"]),
                         "nFaces": int(entries["nFaces"])}
        return out

    @cached_property
    def face_centres(self) -> np.ndarray:
        pts = self.points
        return np.array([pts[f].mean(axis=0) for f in self.faces])

    @cached_property
    def face_areas(self) -> np.ndarray:
        """Area vectors (pointing out of the owner cell), via the polygon cross-product sum."""
        pts = self.points
        out = np.empty((len(self.faces), 3))
        for i, f in enumerate(self.faces):
            p = pts[f]
            out[i] = 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        return out

    @cached_property
    def cell_centres(self) -> np.ndarray:
        """Cell centres as the mean of each cell's unique vertices (exact for hexes/prisms of
        uniform shape, a close approximation otherwise)."""
        cell_pts: list[set] = [set() for _ in range(self.n_cells)]
        nb = self.neighbour
        for i, f in enumerate(self.faces):
            cell_pts[self.owner[i]].update(f.tolist())
            if i < nb.size:
                cell_pts[nb[i]].update(f.tolist())
        pts = self.points
        return np.array([pts[list(s)].mean(axis=0) for s in cell_pts])

    def patch_faces(self, patch: str) -> slice:
        b = self.boundary[patch]
        return slice(b["startFace"], b["startFace"] + b["nFaces"])

    # ── Fields ─────────────────────────────────────────────────────────────
    def _field_text(self, name: str, time: str) -> str:
        return _strip_header(_read_text(self.path / time / name))

    def field(self, name: str, time: str) -> np.ndarray:
        """Internal field: (n_cells,) for scalars, (n_cells, 3) for vectors."""
        text = self._field_text(name, time)
        m = re.search(r"internalField\s+(.*?);\s*\n\s*boundaryField", text, re.S)
        return _parse_values(m.group(1), self.n_cells)

    def patch_field(self, name: str, time: str, patch: str) -> np.ndarray:
        """Face values on *patch*.

        Uses the patch's ``value`` entry when it is written (fixedValue, calculated, …);
        for zeroGradient-type patches, which store no value, the owner-cell value is
        returned — which is exactly the zero-gradient face value.
        """
        text = self._field_text(name, time)
        bf = text[text.index("boundaryField"):]
        m = re.search(rf"\b{re.escape(patch)}\s*\{{(.*?)\n\s*\}}", bf, re.S)
        n = self.boundary[patch]["nFaces"]
        if m:
            v = re.search(r"\bvalue\s+(.*?);", m.group(1), re.S)
            if v:
                return _parse_values(v.group(1), n)
        return self.field(name, time)[self.owner[self.patch_faces(patch)]]
