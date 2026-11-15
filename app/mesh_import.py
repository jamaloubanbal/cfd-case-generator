"""
Import a mesh made in SALOME (or any tool that writes I-DEAS universal files, .unv).

    mesh = import_unv("my_mesh.unv")                 # converts with OpenFOAM's ideasUnvToFoam
    roles = map_roles(mesh.patches, description, backend="gemini")   # LLM; heuristic fallback
    mesh = apply_roles(mesh, roles)                  # sets OpenFOAM patch types

Boundary groups defined in SALOME become patches.  Each patch needs a role (inlet, outlet,
wall, …) for the boundary conditions; the LLM infers it from the group names and the problem
description, a keyword heuristic covers the no-LLM case, and the user can correct it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.postprocess import FoamCase

def mesh_hint(mesh: "ImportedMesh") -> str:
    """Analysis hint with the mesh's dimensions, so the LLM picks a sensible length scale."""
    x0, x1, y0, y1, z0, z1 = mesh.bbox
    groups = "; ".join(f"'{p.name}' area {p.area:.4g} m² facing ({p.normal[0]:.2f}, {p.normal[1]:.2f}, "
                       f"{p.normal[2]:.2f})" for p in mesh.patches)
    return (f"\n\n[Note: the user supplies their own 3-D mesh (made in SALOME), {mesh.n_cells} cells, "
            f"extent {x1 - x0:.4g} × {y1 - y0:.4g} × {z1 - z0:.4g} m. Boundary groups: {groups}. "
            "Choose the solver, fluid, flow model and inlet speed. Use as characteristic_length the "
            "hydraulic diameter of the inlet for internal flow (≈ square root of the inlet area for a "
            "square duct) or the body size for external flow, not the domain length. Name the inlet "
            "boundary condition 'inlet' and the outlet 'outlet'. Set geometry_type to 'channel' (the "
            "geometry comes from the mesh).]")


ROLES = ("inlet", "outlet", "wall", "symmetry", "empty", "hot_wall", "cold_wall")
_OF_TYPE = {"inlet": "patch", "outlet": "patch", "wall": "wall", "hot_wall": "wall",
            "cold_wall": "wall", "symmetry": "symmetryPlane", "empty": "empty"}
_MACRO_ROLE = {"empty": "frontAndBack"}          # role names used by the BC macros

_CONTROL_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam; startTime 0; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
"""


@dataclass
class Patch:
    name: str
    n_faces: int
    area: float
    normal: tuple[float, float, float]     # mean outward unit normal
    role: str | None = None


@dataclass
class ImportedMesh:
    poly_mesh: Path                        # directory holding the converted constant/polyMesh
    patches: list[Patch]
    n_cells: int
    bbox: tuple[float, float, float, float, float, float]
    notes: list[str] = field(default_factory=list)


def import_unv(unv: str | Path | bytes, name: str = "mesh") -> ImportedMesh:
    """Convert a .unv file to an OpenFOAM polyMesh in a temporary directory and describe it."""
    if not shutil.which("ideasUnvToFoam"):
        raise RuntimeError("OpenFOAM's ideasUnvToFoam was not found: source OpenFOAM's etc/bashrc first.")
    work = Path(tempfile.mkdtemp(prefix="cfdgen_mesh_"))
    (work / "system").mkdir()
    (work / "system" / "controlDict").write_text(_CONTROL_DICT)
    src = work / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.unv"
    if isinstance(unv, bytes):
        src.write_bytes(unv)
    else:
        shutil.copy(unv, src)
    proc = subprocess.run(["ideasUnvToFoam", src.name], cwd=work, capture_output=True, text=True)
    boundary = work / "constant" / "polyMesh" / "boundary"
    if proc.returncode != 0 or not boundary.exists():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise ValueError(f"ideasUnvToFoam could not convert the file:\n{tail}")

    case = FoamCase(work)
    pts = case.points
    patches = []
    for pname, b in case.boundary.items():
        sl = case.patch_faces(pname)
        sf = case.face_areas[sl]
        total = sf.sum(axis=0)
        area = float(np.linalg.norm(sf, axis=1).sum())
        mag = np.linalg.norm(total)
        normal = tuple(float(c) for c in (total / mag if mag > 1e-12 * max(area, 1e-30) else (0.0, 0.0, 0.0)))
        patches.append(Patch(pname, b["nFaces"], area, normal))
    notes = []
    if "defaultFaces" in case.boundary:
        notes.append("Some boundary faces belong to no SALOME group; they were put in 'defaultFaces'. "
                     "Give every boundary a group in SALOME, or assign 'defaultFaces' a role.")
    bbox = (*pts.min(axis=0), *pts.max(axis=0))
    bbox = (bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5])
    return ImportedMesh(work / "constant" / "polyMesh", patches, case.n_cells,
                        tuple(float(v) for v in bbox), notes)


# ── Assigning boundary roles ─────────────────────────────────────────────────

_KEYWORDS = [  # (role, regex on the lower-case group name) — English, French, common CFD names
    ("empty", r"front|back|empty|avant|arriere|frontandback|span"),
    ("symmetry", r"sym|farfield|far_field|slip"),
    ("hot_wall", r"hot|chaud|heat"),
    ("cold_wall", r"cold|froid|cool"),
    ("inlet", r"inlet|inflow|entr(e|é)e|entry|in\b|^in_|_in$|admission|source"),
    ("outlet", r"outlet|outflow|sortie|exit|out\b|^out_|_out$|pressure"),
    ("wall", r"wall|paroi|mur|hull|body|surface|obstacle|cylind|airfoil|blade|solid|bottom|top|side"),
]


def heuristic_roles(patches: list[Patch]) -> dict[str, str]:
    """Roles from group names alone; anything unrecognised becomes a wall."""
    roles = {}
    for p in patches:
        name = p.name.lower()
        roles[p.name] = next((r for r, rx in _KEYWORDS if re.search(rx, name)), "wall")
    return roles


ROLE_SCHEMA = {
    "type": "object",
    "properties": {"patches": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "role": {"type": "string", "enum": list(ROLES)},
                       "reason": {"type": "string"}},
        "required": ["name", "role", "reason"], "additionalProperties": False}}},
    "required": ["patches"],
    "additionalProperties": False,
}

_ROLE_SYSTEM = """\
You assign boundary-condition roles to the boundary patches of a CFD mesh made in SALOME.
Patch names are the user's group names, possibly in French (entrée = inlet, sortie = outlet,
paroi = wall). Each patch has its face count, area and mean outward normal.
Roles: inlet (flow enters), outlet (flow leaves), wall (no-slip solid), symmetry (slip or far
field), empty (front/back planes of a one-cell-thick 2-D mesh), hot_wall / cold_wall (heated or
cooled walls in natural convection). Use the problem description to decide. A flow normally
needs at least one inlet and one outlet unless it is natural convection in a closed cavity.
Return one entry per patch, with the exact patch name.
"""


def llm_roles(patches: list[Patch], description: str, backend: str, model: str | None = None,
              api_key: str | None = None, base_url: str | None = None) -> dict[str, str]:
    from app.llm import ask_json
    lines = [f"- {p.name}: {p.n_faces} faces, area {p.area:.4g} m², outward normal "
             f"({p.normal[0]:.2f}, {p.normal[1]:.2f}, {p.normal[2]:.2f})" for p in patches]
    answer = ask_json(_ROLE_SYSTEM, f"Problem: {description}\n\nPatches:\n" + "\n".join(lines),
                      ROLE_SCHEMA, backend=backend, model=model, api_key=api_key, base_url=base_url)
    names = {p.name for p in patches}
    return {e["name"]: e["role"] for e in answer.get("patches", [])
            if e.get("name") in names and e.get("role") in ROLES}


def map_roles(patches: list[Patch], description: str = "", backend: str | None = None,
              **llm_kwargs) -> tuple[dict[str, str], list[str]]:
    """Roles for every patch: the LLM if *backend* is given, the keyword heuristic otherwise
    (and for any patch the LLM skipped).  Returns (roles, notes)."""
    roles, notes = heuristic_roles(patches), []
    if backend:
        try:
            llm = llm_roles(patches, description, backend, **llm_kwargs)
        except Exception as exc:                    # LLM unavailable: keep the heuristic
            notes.append(f"LLM role mapping failed ({exc}); used the name-based heuristic.")
        else:
            for name, role in llm.items():
                if role != roles[name]:
                    notes.append(f"{name}: LLM chose '{role}' (name heuristic: '{roles[name]}').")
                roles[name] = role
            missing = [p.name for p in patches if p.name not in llm]
            if missing:
                notes.append("LLM gave no role for " + ", ".join(missing) + "; used the heuristic.")
    return roles, notes


def check_roles(roles: dict[str, str], solver: str) -> list[str]:
    """Problems with a role assignment (empty list: usable)."""
    values = list(roles.values())
    problems = []
    if solver == "buoyantSimpleFoam":
        if not ({"hot_wall", "cold_wall"} <= set(values)):
            problems.append("natural convection needs a hot_wall and a cold_wall")
    else:
        for need in ("inlet", "outlet"):
            if need not in values:
                problems.append(f"no {need} patch")
    if values.count("empty") == 1:
        problems.append("'empty' needs both the front and the back plane")
    return problems


def apply_roles(mesh: ImportedMesh, roles: dict[str, str]) -> list[dict]:
    """Write the OpenFOAM patch types into the mesh and return patch descriptors for the
    case generator: {name, of_type, role, normal}."""
    boundary = mesh.poly_mesh / "boundary"
    text = boundary.read_text()
    out = []
    for p in mesh.patches:
        role = roles.get(p.name, "wall")
        of_type = _OF_TYPE[role]
        text = re.sub(rf"(\n\s*{re.escape(p.name)}\s*\{{[^}}]*?\btype\s+)\w+;", rf"\g<1>{of_type};", text)
        text = re.sub(rf"(\n\s*{re.escape(p.name)}\s*\{{[^}}]*?)\n\s*inGroups[^;]*;", r"\1", text)
        out.append({"name": p.name, "of_type": of_type, "role": _MACRO_ROLE.get(role, role),
                    "normal": list(p.normal)})
    boundary.write_text(text)
    return out


def attach(analysis: dict, mesh: ImportedMesh, patches: list[dict]) -> dict:
    """Return a copy of *analysis* that generates its case on the imported mesh."""
    if analysis.get("solver") == "interFoam":
        raise ValueError("Free-surface (interFoam) cases are not supported on imported meshes yet.")
    a = dict(analysis)
    a.update(geometry_type="imported_mesh", _mesh_dir=str(mesh.poly_mesh), _mesh_patches=patches,
             _mesh_bbox=mesh.bbox, _mesh_cells=mesh.n_cells)
    return a
