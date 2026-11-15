"""
Abstract base class for all CFD case generators.
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


# ---------------------------------------------------------------------------
# Wall-grading math helpers
# ---------------------------------------------------------------------------
# All helpers return OpenFOAM simpleGrading values (strings) for one block edge.
#
# OpenFOAM convention:
#   simpleGrading ratio = last_cell_size / first_cell_size
#   ratio > 1  →  cells grow away from the start face  (fine at start = floor grading)
#   ratio < 1  →  cells shrink away from the start face (fine at end  = ceil  grading)
#
# Bidirectional (two-wall) grading uses the compound format:
#   ((fraction_cells fraction_length expansion) …)

# Blasius turbulent flat-plate skin-friction: Cf ≈ _CF_A × Re^_CF_B
_CF_A: float = 0.026
_CF_B: float = -1.0 / 7.0


def _wall_y1(Re, inlet_speed: float, nu: float, turb_model: str) -> float:
    """Target first wall-cell height from Blasius Cf and wall-function target y⁺.

    Returns ``float('inf')`` when grading is not applicable (laminar / zero speed).
    """
    if turb_model == "laminar" or Re is None or float(inlet_speed) < 1e-9 or float(nu) <= 0:
        return float("inf")
    target_yplus = 30.0 if turb_model == "SpalartAllmaras" else 50.0
    Re_eff = max(float(Re), 1_000.0)
    Cf     = _CF_A * Re_eff ** _CF_B
    u_tau  = float(inlet_speed) * math.sqrt(max(Cf, 0.0) / 2.0)
    if u_tau < 1e-12:
        return float("inf")
    return target_yplus * float(nu) / u_tau


# Maximum ratio between the sizes of two adjacent cells (common best practice:
# 1.1–1.25).  Larger growth rates produce high-aspect-ratio, poor-quality cells.
_MAX_CELL_GROWTH: float = 1.2


def _compute_grading_ratio(domain_h: float, n_cells: int, y1: float) -> float:
    """Compute OpenFOAM simpleGrading expansion ratio (last/first) for one wall.

    Uses bisection to find per-cell expansion *e* satisfying::

        y1 × (e^n − 1) / (e − 1) = domain_h

    Returns 1.0 (uniform) if grading would not improve on the uniform cell size.
    """
    if (
        not math.isfinite(y1)
        or y1 <= 0
        or domain_h <= 0
        or n_cells < 2
    ):
        return 1.0

    uniform = domain_h / n_cells
    if y1 >= uniform * 0.999:
        return 1.0  # already fine enough — no benefit from grading

    def _residual(e: float) -> float:
        if abs(e - 1.0) < 1e-10:
            return n_cells * y1 - domain_h
        return y1 * (e ** n_cells - 1.0) / (e - 1.0) - domain_h

    # Mesh-quality guard: the cell-to-cell growth rate is capped.  When the
    # wall-function target y1 cannot be reached with n_cells, we accept a
    # coarser first cell (higher y+) rather than extreme aspect ratios.
    # Searching only up to the cap also keeps e**n_cells finite for large n.
    if _residual(_MAX_CELL_GROWTH) <= 0:
        e = _MAX_CELL_GROWTH
    else:
        lo, hi = 1.0 + 1e-8, _MAX_CELL_GROWTH
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _residual(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        e = 0.5 * (lo + hi)
    ratio = e ** (n_cells - 1)
    if ratio < 1.02:
        return 1.0
    return ratio


def _cells_to_reach(domain_h: float, y1: float, base: int, cap: int = 60) -> int:
    """Cells needed across *domain_h* for the wall cell to reach *y1* at the capped growth rate.

    Never fewer than *base* (the geometry's normal count) nor more than *cap*.  For laminar
    flow (y1 = inf) this returns *base*.
    """
    if not math.isfinite(y1) or y1 <= 0 or domain_h <= 0:
        return base
    g = _MAX_CELL_GROWTH
    needed = math.ceil(math.log(1.0 + domain_h * (g - 1.0) / y1) / math.log(g))
    return int(min(max(base, needed), cap))


def _first_cell_height(domain_h: float, n_cells: int, y1: float, both: bool = False) -> float:
    """Height of the wall-adjacent cell that the grading helpers actually produce.

    With *both* the edge is graded towards both ends (two halves of n_cells/2 cells).
    Because the growth rate is capped, this can be larger than the target *y1*.
    """
    if both:
        domain_h, n_cells = domain_h / 2.0, max(2, n_cells // 2)
    n_cells = max(1, int(n_cells))
    r = _compute_grading_ratio(domain_h, n_cells, y1)
    if n_cells < 2 or abs(r - 1.0) < 0.02:
        return domain_h / n_cells
    e = r ** (1.0 / (n_cells - 1))
    return domain_h * (e - 1.0) / (e ** n_cells - 1.0)


def _floor_grading_str(domain_h: float, n_cells: int, y1: float) -> str:
    """simpleGrading string: fine cells at y = 0 (floor), coarse at top."""
    r = _compute_grading_ratio(domain_h, n_cells, y1)
    return "1" if abs(r - 1.0) < 0.02 else f"{r:.4g}"


def _ceil_grading_str(domain_h: float, n_cells: int, y1: float) -> str:
    """simpleGrading string: fine cells at y = domain_h (ceil), coarse at bottom."""
    r = _compute_grading_ratio(domain_h, n_cells, y1)
    if abs(r - 1.0) < 0.02:
        return "1"
    return f"{1.0 / r:.4g}"


def _both_grading_str(domain_h: float, n_cells: int, y1: float) -> str:
    """Bidirectional compound grading: fine at both y = 0 and y = domain_h."""
    n_half = max(2, n_cells // 2)
    r = _compute_grading_ratio(domain_h / 2.0, n_half, y1)
    if abs(r - 1.0) < 0.02:
        return "1"
    inv_r = 1.0 / r
    return f"((0.5 0.5 {r:.4g})(0.5 0.5 {inv_r:.4g}))"


def _decomposition_split(n_cores: int) -> tuple[int, int, int]:
    """Split *n_cores* into (nx, ny, 1) subdomains for hierarchical decomposition.

    Cases are 2-D (one cell in z), so z is never split.  The larger factor goes
    to x because domains are longest in the streamwise direction.
    """
    n = max(1, int(n_cores))
    ny = max(d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0)
    return (n // ny, ny, 1)


# ---------------------------------------------------------------------------
# Post-processing field list helper
# ---------------------------------------------------------------------------

def _sample_fields(ctx: dict) -> list[str]:
    """Return the list of OpenFOAM field names to include in sampleDict/probesDict.

    The list is built from the solver name and turbulence model so that only
    fields actually present in the case are requested.
    """
    solver = ctx.get("solver_name", "")
    turb   = ctx.get("turb_model", "")
    fields: list[str] = ["U", "p"]
    if solver == "buoyantSimpleFoam":
        fields += ["p_rgh", "T"]
    elif solver == "rhoSimpleFoam":
        fields.append("T")
    elif solver == "interFoam":
        fields.append("alpha.water")
    if solver != "interFoam":
        if turb == "kEpsilon":
            fields += ["k", "epsilon"]
        elif turb == "kOmegaSST":
            fields += ["k", "omega"]
        elif turb == "SpalartAllmaras":
            fields.append("nuTilda")
    return fields


# ---------------------------------------------------------------------------
# Patch catalogue — maps geometry_type → ordered list of patch descriptors.
# Each patch dict has:
#   name     : OpenFOAM patch name
#   of_type  : blockMesh patch type ("patch", "wall", "empty")
#   role     : semantic role used by boundary-condition macros
# ---------------------------------------------------------------------------

GEOMETRY_PATCHES: dict[str, list[dict]] = {
    # Standard rectangular channel
    "channel": [
        {"name": "inlet",        "of_type": "patch", "role": "inlet"},
        {"name": "outlet",       "of_type": "patch", "role": "outlet"},
        {"name": "walls",        "of_type": "wall",  "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # Square cavity (natural convection / buoyantSimpleFoam)
    "cavity": [
        {"name": "hotWall",      "of_type": "wall",  "role": "hot_wall"},
        {"name": "coldWall",     "of_type": "wall",  "role": "cold_wall"},
        {"name": "top",          "of_type": "wall",  "role": "wall"},
        {"name": "bottom",       "of_type": "wall",  "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # T-junction (4 blocks)
    "t_junction": [
        {"name": "inlet1",       "of_type": "patch", "role": "inlet"},
        {"name": "inlet2",       "of_type": "patch", "role": "inlet"},
        {"name": "outlet",       "of_type": "patch", "role": "outlet"},
        {"name": "walls",        "of_type": "wall",  "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # Channel with central rectangular obstacle (8 blocks)
    "rectangular_obstacle": [
        {"name": "inlet",        "of_type": "patch", "role": "inlet"},
        {"name": "outlet",       "of_type": "patch", "role": "outlet"},
        {"name": "top",          "of_type": "wall",  "role": "wall"},
        {"name": "bottom",       "of_type": "wall",  "role": "wall"},
        {"name": "obstacle",     "of_type": "wall",  "role": "obstacle"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # Backward-facing step (3 blocks)
    "backstep": [
        {"name": "inlet",        "of_type": "patch", "role": "inlet"},
        {"name": "outlet",       "of_type": "patch", "role": "outlet"},
        {"name": "walls",        "of_type": "wall",  "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # Free-surface basin (interFoam / dam-break)
    "dambreak": [
        {"name": "walls",        "of_type": "wall",  "role": "wall"},   # closed tank
        {"name": "atmosphere",   "of_type": "patch", "role": "atmosphere"},
        {"name": "frontAndBack", "of_type": "empty", "role": "frontAndBack"},
    ],
    # 2-D cylinder in cross-flow — O-grid / butterfly mesh (12 blocks)
    "cylinder": [
        {"name": "inlet",        "of_type": "patch",    "role": "inlet"},
        {"name": "outlet",       "of_type": "patch",    "role": "outlet"},
        {"name": "top",          "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "bottom",       "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "cylinder",     "of_type": "wall",     "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty",    "role": "frontAndBack"},
    ],
    # Axisymmetric pipe — 5-degree wedge mesh (1 block)
    "pipe": [
        {"name": "inlet",   "of_type": "patch",  "role": "inlet"},
        {"name": "outlet",  "of_type": "patch",  "role": "outlet"},
        {"name": "wall",    "of_type": "wall",   "role": "wall"},
        {"name": "axis",    "of_type": "empty",  "role": "frontAndBack"},
        {"name": "front",   "of_type": "wedge",  "role": "wedge"},
        {"name": "back",    "of_type": "wedge",  "role": "wedge"},
    ],
    # 2-D airfoil in external flow — 8-block C/H-grid with rectangular body hole
    "airfoil": [
        {"name": "inlet",        "of_type": "patch",    "role": "inlet"},
        {"name": "outlet",       "of_type": "patch",    "role": "outlet"},
        {"name": "top",          "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "bottom",       "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "airfoil",      "of_type": "wall",     "role": "wall"},
        {"name": "frontAndBack", "of_type": "empty",    "role": "frontAndBack"},
    ],
    # Arbitrary STL geometry — background mesh only; STL wall patch added dynamically
    # Mesh imported from SALOME (.unv): patches come from the mesh (analysis["_mesh_patches"])
    "imported_mesh": [],
    "stl_import": [
        {"name": "inlet",  "of_type": "patch",    "role": "inlet"},
        {"name": "outlet", "of_type": "patch",    "role": "outlet"},
        {"name": "top",    "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "bottom", "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "front",  "of_type": "symmetryPlane", "role": "symmetry"},
        {"name": "back",   "of_type": "symmetryPlane", "role": "symmetry"},
        # STL wall patch appended in _build_geometry_extra with role='wall'
    ],
}


# Geometries whose blockMeshDict is a single box: domain_x/y and nx/ny map
# directly onto the mesh, so the parameter editor can override them.
SINGLE_BLOCK_GEOMETRIES: tuple[str, ...] = ("channel", "cavity", "dambreak")

# Allowed range for the global mesh refinement factor
MIN_REFINEMENT: float = 0.5
MAX_REFINEMENT: float = 4.0


class BaseCaseGenerator(ABC):
    """
    Abstract base for OpenFOAM case generators.

    Subclasses must implement :meth:`generate` which writes all case
    files into *output_dir* using values from *analysis*.

    Class attributes
    ----------------
    solver_name : str
        The OpenFOAM solver name (e.g. "simpleFoam").
    default_geometry_type : str
        Geometry used when the analysis dict does not specify one.
    """

    solver_name: str = ""
    default_geometry_type: str = "channel"

    def __init__(self) -> None:
        # Resolve templates directory relative to this file's location:
        # app/generators/base.py  ->  ../../templates/
        self.templates_root = Path(__file__).resolve().parent.parent.parent / "templates"
        self._env: Environment | None = None

    def _get_jinja_env(self) -> Environment:
        """Return (and cache) a Jinja2 environment for this solver's templates.

        The loader searches:
          1. templates/<solver_name>/   — solver-specific templates
          2. templates/                  — shared _macros/ directory
        """
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader([
                    str(self.templates_root / self.solver_name),
                    str(self.templates_root),   # exposes _macros/boundary_conditions.j2
                ]),
                undefined=StrictUndefined,
                keep_trailing_newline=True,
            )
        return self._env

    def render(self, template_path: str, context: dict) -> str:
        """
        Render a Jinja2 template relative to this solver's template directory.

        Parameters
        ----------
        template_path : str
            Path relative to ``templates/<solver_name>/``, e.g. ``"0/U"``.
        context : dict
            Variables available inside the template.
        """
        env = self._get_jinja_env()
        tmpl = env.get_template(template_path)
        return tmpl.render(**context)

    def write(self, output_dir: Path, relative_path: str, content: str) -> None:
        """Write *content* to *output_dir* / *relative_path*, creating parents."""
        dest = output_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def write_mesh(self, output_dir: Path, ctx: dict, analysis: dict) -> None:
        """Write system/blockMeshDict, or copy an imported mesh into constant/polyMesh."""
        if ctx["geometry_type"] == "imported_mesh":
            src = analysis.get("_mesh_dir")
            if not src or not Path(src).is_dir():
                raise ValueError("imported_mesh needs analysis['_mesh_dir'] (a converted polyMesh)")
            import shutil
            dest = output_dir / "constant" / "polyMesh"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest)
            return
        self.write(output_dir, "system/blockMeshDict", self.render_geometry_block_mesh(ctx["geometry_type"], ctx))

    def render_geometry_block_mesh(self, geometry_type: str, ctx: dict) -> str:
        """Render blockMeshDict from the shared geometry template library."""
        geo_dir = self.templates_root / "geometries" / geometry_type
        env = Environment(
            loader=FileSystemLoader(str(geo_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        return env.get_template("blockMeshDict").render(**ctx)

    def write_run_scripts(self, output_dir: Path, ctx: dict) -> None:
        """Write Allrun, Allclean, and README.md into *output_dir*.

        ``Allrun`` and ``Allclean`` are rendered and marked executable (+x).
        ``README.md`` is rendered as plain text — not executable.
        All three are rendered via the solver's normal Jinja2 environment so
        that ``{{ solver_name }}`` and other context variables are substituted.
        """
        import os
        import stat as _stat

        for name in ("Allrun", "Allclean"):
            content = self.render(f"_shared/{name}", ctx)
            self.write(output_dir, name, content)
            path = output_dir / name
            path.chmod(path.stat().st_mode | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)

        self.write(output_dir, "system/decomposeParDict",
                   self.render("_shared/decomposeParDict",
                               {**ctx, "decomp_n": _decomposition_split(int(ctx.get("n_cores", 1)))}))

        readme = self.render("_shared/README.md", ctx)
        self.write(output_dir, "README.md", readme)

    def write_snappy_files(self, output_dir: Path, ctx: dict, analysis: dict) -> None:
        """Write snappyHexMesh configuration files into *output_dir*/system/.

        Also copies the uploaded STL into ``constant/triSurface/`` when raw
        bytes are present in *analysis* under the ``_stl_bytes`` key.

        Called by each solver's ``generate()`` when
        ``ctx["geometry_type"] == "stl_import"``.
        """
        stl_bytes = analysis.get("_stl_bytes")
        stl_name = ctx.get("stl_name", "geometry")

        if stl_bytes is not None:
            tris_dir = output_dir / "constant" / "triSurface"
            tris_dir.mkdir(parents=True, exist_ok=True)
            (tris_dir / f"{stl_name}.stl").write_bytes(stl_bytes)

        snappy_dir = self.templates_root / "_shared" / "snappy"
        env = Environment(
            loader=FileSystemLoader(str(snappy_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        for fname in ("snappyHexMeshDict", "surfaceFeatureExtractDict", "meshQualityDict"):
            content = env.get_template(fname).render(**ctx)
            self.write(output_dir, f"system/{fname}", content)

    def write_postprocessing_files(self, output_dir: Path, ctx: dict) -> None:
        """Write ``system/sampleDict`` and ``system/probesDict`` into *output_dir*.

        Both files are immediately usable with::

            postProcess -func sample -time latest
            postProcess -func probes -time latest

        Sample locations and field lists are computed from the context
        (``domain_x/y/z``, ``nx``, ``ny``, ``solver_name``, ``turb_model``).
        """
        dx = float(ctx.get("domain_x", 1.0))
        dy = float(ctx.get("domain_y", 1.0))
        dz = float(ctx.get("domain_z", 0.1))
        nx = int(ctx.get("nx", 50))
        ny = int(ctx.get("ny", 25))

        pp_ctx = dict(ctx)
        pp_ctx.update(
            sample_x_start = round(0.02 * dx, 8),
            sample_x_end   = round(0.98 * dx, 8),
            sample_x_mid   = round(0.50 * dx, 8),
            sample_y_start = round(0.02 * dy, 8),
            sample_y_end   = round(0.98 * dy, 8),
            sample_y_mid   = round(0.50 * dy, 8),
            sample_z_mid   = round(0.50 * dz, 8),
            sample_n_x     = min(nx * 2, 500),
            sample_n_y     = min(ny * 2, 250),
            sample_fields  = _sample_fields(ctx),
            probe_pts      = [
                (round(0.10 * dx, 8), round(0.50 * dy, 8), round(0.50 * dz, 8)),
                (round(0.50 * dx, 8), round(0.50 * dy, 8), round(0.50 * dz, 8)),
                (round(0.90 * dx, 8), round(0.50 * dy, 8), round(0.50 * dz, 8)),
            ],
        )

        pp_dir = self.templates_root / "_shared" / "postProcessing"
        env = Environment(
            loader=FileSystemLoader(str(pp_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        for fname in ("sampleDict", "probesDict"):
            content = env.get_template(fname).render(**pp_ctx)
            self.write(output_dir, f"system/{fname}", content)

    @abstractmethod
    def generate(self, analysis: dict, output_dir: Path) -> None:
        """
        Generate all OpenFOAM case files into *output_dir*.

        Parameters
        ----------
        analysis : dict
            Structured analysis returned by :func:`app.llm.analyze_cfd_problem`.
        output_dir : Path
            Destination directory for the case (will be created if absent).
        """

    # ------------------------------------------------------------------
    # Shared context builders
    # ------------------------------------------------------------------

    def _base_context(self, analysis: dict) -> dict:
        """
        Build a Jinja2 rendering context with values that every solver needs.

        Includes geometry_type and the corresponding patches list so that
        the dynamic boundary-condition macros can iterate over them.
        """
        fluid = analysis.get("fluid", {})
        bcs = analysis.get("boundary_conditions", [])
        schemes = analysis.get("numerical_schemes", {})
        mesh = analysis.get("suggested_mesh", {})

        nu = float(fluid.get("nu") or 1.5e-5)
        rho = float(fluid.get("rho") or 1.225)
        char_vel = analysis.get("characteristic_velocity", 1.0)
        char_len = analysis.get("characteristic_length", 1.0)

        inlet_U = self._get_inlet_velocity(bcs)
        inlet_speed = self._get_inlet_speed(bcs)

        _raw_geo = analysis.get("geometry_type") or self.default_geometry_type
        if _raw_geo not in GEOMETRY_PATCHES:
            _raw_geo = self.default_geometry_type
        geometry_type = _raw_geo
        if geometry_type == "imported_mesh":
            patches = self._imported_patches(analysis.get("_mesh_patches") or [], inlet_speed)
        else:
            patches = self._patches_with_inlet_velocities(
                GEOMETRY_PATCHES[geometry_type], bcs, inlet_speed
            )

        turb_model = analysis.get("turb_model") or "kEpsilon"

        # Turbulence initial values (computed from k/eps; generators may override)
        I = 0.05
        k_ref = 1.5 * (max(inlet_speed, 1e-6) * I) ** 2
        L_mix = max(char_len * 0.07, 1e-8)
        eps_ref = (0.09 ** 0.75) * (k_ref ** 1.5) / L_mix
        omega_inlet = eps_ref / (0.09 * max(k_ref, 1e-30))
        nuTilda_inlet = 3.0 * nu   # SA freestream: ~3–5 × nu

        return {
            "solver_name":    self.solver_name,
            "fluid_name":     fluid.get("name", "fluid"),
            "nu":             nu,
            "rho":            rho,
            "char_vel":       char_vel,
            "char_len":       char_len,
            "inlet_U":        inlet_U,
            "inlet_speed":    inlet_speed,
            "bcs":            bcs,
            "schemes":        schemes,
            "mesh":           mesh,
            "Re":             analysis.get("Re"),
            "is_transient":   analysis.get("is_transient", False),
            "geometry_type":  geometry_type,
            "patches":        patches,
            "turb_model":     turb_model,
            "omega_inlet":    f"{omega_inlet:.6e}",
            "nuTilda_inlet":  f"{nuTilda_inlet:.6e}",
            "warnings":       list(analysis.get("warnings") or []),
            "n_cores":        int(analysis.get("n_cores") or 1),
        }

    def _mesh_overrides(
        self, ctx: dict, analysis: dict,
        domain_x: float, domain_y: float, nx: int, ny: int,
    ) -> tuple[float, float, int, int]:
        """Apply the parameter-editor mesh overrides to the solver's default sizing.

        Overrides live in ``analysis["suggested_mesh"]`` (see ``app.editor``):

        * ``domain_x``, ``domain_y``, ``nx``, ``ny`` — honoured for single-block
          geometries, where they map 1:1 onto the blockMeshDict.  Multi-block
          geometries are sized from ``geometry_params`` instead.
        * ``refinement`` — factor applied to every cell count (all geometries
          except ``stl_import``, whose resolution is set by snappyHexMesh).

        Also stores the factor as ``ctx["_refine"]`` for the multi-block
        branches of :meth:`_build_geometry_extra`.
        """
        mesh = analysis.get("suggested_mesh") or {}
        geometry_type = ctx["geometry_type"]

        if geometry_type in SINGLE_BLOCK_GEOMETRIES:
            def _positive(key, cast, current):
                try:
                    val = cast(mesh.get(key))
                except (TypeError, ValueError):
                    return current
                return val if val > 0 else current

            domain_x = _positive("domain_x", float, domain_x)
            domain_y = _positive("domain_y", float, domain_y)
            nx = _positive("nx", int, nx)
            ny = _positive("ny", int, ny)

        try:
            refine = float(mesh.get("refinement") or 1.0)
        except (TypeError, ValueError):
            refine = 1.0
        refine = min(max(refine, MIN_REFINEMENT), MAX_REFINEMENT)
        if geometry_type in ("stl_import", "imported_mesh"):
            refine = 1.0
        ctx["_refine"] = refine
        if refine != 1.0:
            nx = max(2, round(nx * refine))
            ny = max(2, round(ny * refine))
        return domain_x, domain_y, nx, ny

    def mesh_summary(self, analysis: dict) -> dict:
        """Return the mesh this generator would build for *analysis*.

        Used by the parameter editor and the physics checks so that the values
        shown in the UI are the values that end up in ``blockMeshDict``.
        """
        ctx = self._build_context(analysis)
        return {
            "geometry_type": ctx["geometry_type"],
            "domain_x": float(ctx["domain_x"]),
            "domain_y": float(ctx["domain_y"]),
            "nx": int(ctx["nx"]),
            "ny": int(ctx["ny"]),
            "single_block": ctx["geometry_type"] in SINGLE_BLOCK_GEOMETRIES,
            "y1": float(ctx.get("_y1_actual") or float(ctx["domain_y"]) / int(ctx["ny"])),
        }

    def _add_wall_grading(self, ctx: dict) -> None:
        """
        Compute wall-grading Jinja variables and add them to *ctx* in-place.

        Must be called **after** domain_y, domain_x, ny, nx, nu, inlet_speed,
        Re, and turb_model are all present in *ctx*.

        Variables added
        ---------------
        _y1_target   : float  — target first-cell height [m]
        grading_y_floor : str — refines near y = 0
        grading_y_ceil  : str — refines near y = domain_y
        grading_y_both  : str — refines at both y = 0 and y = domain_y
        grading_x_both  : str — refines at both x = 0 and x = domain_x (cavity)
        """
        nu          = float(ctx.get("nu", 1.5e-5))
        inlet_speed = float(ctx.get("inlet_speed", 1.0))
        Re          = ctx.get("Re")
        turb        = ctx.get("turb_model", "kEpsilon")
        domain_y    = float(ctx.get("domain_y", 1.0))
        domain_x    = float(ctx.get("domain_x", 1.0))
        ny          = int(ctx.get("ny", 50))
        nx          = int(ctx.get("nx", 100))

        y1 = _wall_y1(Re, inlet_speed, nu, turb)

        ctx["_y1_target"]      = y1
        ctx["grading_y_floor"] = _floor_grading_str(domain_y, ny, y1)
        ctx["grading_y_ceil"]  = _ceil_grading_str(domain_y, ny, y1)
        ctx["grading_y_both"]  = _both_grading_str(domain_y, ny, y1)
        ctx["grading_x_both"]  = _both_grading_str(domain_x, nx, y1)
        # First wall-cell height of the main wall (used by the y+ physics check);
        # geometry branches in _build_geometry_extra overwrite it for their walls.
        if ctx.get("geometry_type") == "cavity":
            ctx["_y1_actual"] = _first_cell_height(domain_x, nx, y1, both=True)
        else:
            ctx["_y1_actual"] = _first_cell_height(domain_y, ny, y1, both=True)

    def _build_geometry_extra(self, geometry_type: str, analysis: dict, ctx: dict) -> dict:
        """
        Add geometry-specific mesh dimension variables to *ctx* in-place.

        For 'channel', 'cavity', and 'dambreak' the needed variables
        (domain_x/y/z, nx/ny/nz) are already set by the solver's own
        ``_build_context()``.  For the multi-block geometries this method
        computes the additional split-point coordinates and per-block cell
        counts.

        Also computes all wall-grading variables (grading_y_floor, _ceil, _both,
        grading_x_both, and geometry-specific block grading strings) so that
        blockMeshDict templates can use ``simpleGrading`` with physically
        appropriate expansion ratios.
        """
        # Grading must be computed first so multi-block branches can refine it
        self._add_wall_grading(ctx)

        geo_params = analysis.get("geometry_params") or {}
        char_len = analysis.get("characteristic_length", 1.0)

        y1 = ctx.get("_y1_target", float("inf"))
        f = float(ctx.get("_refine", 1.0))   # global mesh refinement factor

        if geometry_type == "t_junction":
            L  = float(geo_params.get("channel_length") or ctx.get("domain_x") or char_len * 10)
            H  = float(geo_params.get("channel_height") or char_len)
            Lb = float(geo_params.get("branch_height")  or char_len * 4)
            bw = float(geo_params.get("branch_width")   or H)
            # Guarantee the branch fits: channel must be at least 4× branch width
            L  = max(L, bw * 4)
            xl = (L - bw) / 2
            xr = (L + bw) / 2
            dz = round(H * 0.1, 4)
            nx_left   = max(round(20 * f), round(xl / L * 60 * f))
            nx_branch = max(round(10 * f), round(bw / L * 60 * f))
            nx_right  = max(round(20 * f), round((L - xr) / L * 60 * f))
            ny_main   = max(round(15 * f), round(H  / (H + Lb) * 40 * f))
            ny_branch = max(round(15 * f), round(Lb / (H + Lb) * 40 * f))
            ctx.update(
                domain_x=L, domain_y=H, domain_z=dz,
                xl=xl, xr=xr, H_branch=H + Lb,
                nx_left=nx_left, nx_branch=nx_branch, nx_right=nx_right,
                ny_main=ny_main, ny_branch=ny_branch,
                # Block-specific grading.  B0, B1 and B3 share y-z faces, so
                # blockMesh requires identical y-grading on all three — B1 uses
                # the two-sided grading too (its top is the branch opening).
                grading_tj_main=_both_grading_str(H, ny_main, y1),
                _y1_actual=_first_cell_height(H, ny_main, y1, both=True),
                grading_tj_b1=_both_grading_str(H, ny_main, y1),
                # B2 (branch): vertical block, no y-grading needed
                grading_tj_branch="1",
            )

        elif geometry_type == "rectangular_obstacle":
            L   = float(geo_params.get("channel_length")   or ctx.get("domain_x") or char_len * 10)
            H   = float(geo_params.get("channel_height")   or char_len * 4)
            # Clamp obstacle to fit inside the channel (≤80% in each direction)
            ow  = min(float(geo_params.get("obstacle_width")  or char_len), L * 0.4)
            oh  = min(float(geo_params.get("obstacle_height") or char_len), H * 0.8)
            xo1 = min(float(geo_params.get("xo1") or L / 3), L * 0.85 - ow)
            xo2 = xo1 + ow
            yo1 = (H - oh) / 2
            yo2 = yo1 + oh
            dz  = round(H * 0.1, 4)
            nx_left  = max(round(15 * f), round(xo1 / L * 80 * f))
            nx_obs   = max(round(10 * f), round(ow  / L * 80 * f))
            nx_right = max(round(25 * f), round((L - xo2) / L * 80 * f))
            ny_bot   = max(round(10 * f), round(yo1 / H * 40 * f))
            ny_obs   = max(round(10 * f), round(oh  / H * 40 * f))
            ny_top   = max(round(10 * f), round((H - yo2) / H * 40 * f))
            ctx.update(
                domain_x=L, domain_y=H, domain_z=dz,
                xo1=xo1, xo2=xo2, yo1=yo1, yo2=yo2,
                nx_left=nx_left, nx_obs=nx_obs, nx_right=nx_right,
                ny_bot=ny_bot, ny_obs=ny_obs, ny_top=ny_top,
                # Block-specific grading:
                # Bottom row (B0,B1,B2): wall at y=0
                grading_ob_bot=_floor_grading_str(yo1, ny_bot, y1),
                _y1_actual=_first_cell_height(yo1, ny_bot, y1),
                # Middle blocks (B3,B5): no horizontal walls → uniform
                grading_ob_mid="1",
                # Top row (B6,B7,B8): wall at y=domain_y
                grading_ob_top=_ceil_grading_str(H - yo2, ny_top, y1),
            )

        elif geometry_type == "backstep":
            L   = float(geo_params.get("channel_length") or ctx.get("domain_x") or char_len * 10)
            H   = float(geo_params.get("channel_height") or char_len * 2)
            Hs  = float(geo_params.get("step_height")    or H / 2)
            xs  = float(geo_params.get("step_x")         or L * 0.3)
            dz  = round(H * 0.1, 4)
            nx_inlet  = max(round(15 * f), round(xs       / L * 60 * f))
            nx_outlet = max(round(40 * f), round((L - xs) / L * 60 * f))
            ny_top    = max(round(15 * f), round((H - Hs) / H * 30 * f))
            ny_bot    = max(round(10 * f), round(Hs        / H * 30 * f))
            ctx.update(
                domain_x=L, domain_y=H, domain_z=dz,
                xs=xs, Hs=Hs,
                nx_inlet=nx_inlet, nx_outlet=nx_outlet,
                ny_top=ny_top, ny_bot=ny_bot,
                # Block-specific grading:
                # B0 (inlet section, y:[Hs,H]): walls at top (y=H) AND step floor (y=Hs)
                grading_bs_b0=_both_grading_str(H - Hs, ny_top, y1),
                _y1_actual=_first_cell_height(H - Hs, ny_top, y1, both=True),
                # B1a (outlet upper, y:[Hs,H]) shares the x=xs face with B0, so
                # it must use the same y-grading (also resolves the shear layer
                # leaving the step corner).
                grading_bs_b1a=_both_grading_str(H - Hs, ny_top, y1),
                # B1b (outlet lower, y:[0,Hs]): wall at floor (y=0) only
                grading_bs_b1b=_floor_grading_str(Hs, ny_bot, y1),
            )

        elif geometry_type == "cylinder":
            # ----------------------------------------------------------------
            # 12-block O-grid / butterfly mesh around a 2-D cylinder.
            #
            # Domain: 0 ≤ x ≤ L, 0 ≤ y ≤ H (2-D extruded by domain_z)
            # Cylinder centre: (cx, cy),  radius: R
            # Inner square:    half-size ri = 4*R  (always fits 4-D around centre)
            #
            # 20 vertices per z-plane (40 total):
            #   Row y=0       : v0..v3  (domain bottom)
            #   Row y=cy-ri   : v4..v7  (inner sq bottom)
            #   Cyl surf ±45° : v8..v11 (SW, SE, NE, NW)
            #   Row y=cy+ri   : v12..v15 (inner sq top)
            #   Row y=H       : v16..v19 (domain top)
            #   Back-face mirror: v20..v39
            # ----------------------------------------------------------------
            _INV_SQRT2 = 1.0 / math.sqrt(2.0)

            R   = float(geo_params.get("cylinder_radius") or char_len / 2)
            L   = float(geo_params.get("channel_length")  or ctx.get("domain_x") or char_len * 20)
            H   = float(geo_params.get("channel_height")  or ctx.get("domain_y") or char_len * 10)

            # Default centre: 5D upstream, centred in y
            cx_default = max(5.0 * 2 * R, L / 4)
            cy_default = H / 2
            cx  = float(geo_params.get("cx") or cx_default)
            cy  = float(geo_params.get("cy") or cy_default)

            # Inner square half-size (4R) — clamp to fit inside domain with margin
            ri  = 4.0 * R
            ri  = min(ri, cx * 0.85, (L - cx) * 0.85, cy * 0.85, (H - cy) * 0.85)
            ri  = max(ri, R * 1.5)  # must be larger than cylinder

            cyl_r45 = R * _INV_SQRT2   # cylinder vertex offset along ±45°

            dz = round(R * 0.5, 6)    # thin 2-D slab

            # ── Cell counts ──────────────────────────────────────────────
            nx_tot = int(ctx.get("nx", 60))
            ny_tot = int(ctx.get("ny", 30))

            n_up  = max(8,  round((cx - ri) / L * nx_tot))
            n_mid = max(6,  round(2 * ri    / L * nx_tot))
            n_dn  = max(12, round((L - cx - ri) / L * nx_tot))
            n_bot = max(6,  round((cy - ri) / H * ny_tot))
            n_top = max(6,  round((H - cy - ri) / H * ny_tot))
            n_cyl_h = max(6, round(2 * ri / H * ny_tot))
            # ── O-ring radial cells and grading ───────────────────────────
            # Enough radial cells for the wall cell to reach the wall-function
            # target y1 (growth ≤ 1.2); a fixed 12 gave y+ ~ 15 000 at Re = 1e7.
            r_depth = ri - R * _INV_SQRT2   # radial depth of O-ring
            n_r   = _cells_to_reach(r_depth, y1, base=max(4, min(12, n_mid // 2)))
            cyl_grad_ceil  = _ceil_grading_str(r_depth, n_r, y1)
            ctx["_y1_actual"] = _first_cell_height(r_depth, n_r, y1)
            cyl_grad_floor = _floor_grading_str(r_depth, n_r, y1)

            ctx.update(
                domain_x=L, domain_y=H, domain_z=dz,
                nx=nx_tot, ny=ny_tot,
                cyl_cx=cx, cyl_cy=cy,
                cyl_r=R,
                cyl_r45=round(cyl_r45, 8),
                cyl_ri=round(ri, 8),
                n_up=n_up, n_mid=n_mid, n_dn=n_dn,
                n_bot=n_bot, n_top=n_top, n_cyl_h=n_cyl_h,
                n_r=n_r,
                cyl_grad_ceil=cyl_grad_ceil,
                cyl_grad_floor=cyl_grad_floor,
            )

        elif geometry_type == "pipe":
            # ----------------------------------------------------------------
            # 1-block axisymmetric wedge mesh for pipe flow.
            #
            # Wedge half-angle α = 2.5° (total sector = 5°).
            # The axis runs along x; y is the radial direction.
            # Front face at z = -R·sin(α), back face at z = +R·sin(α).
            # Axis vertices (y=z=0) are duplicated — required by blockMesh.
            # ----------------------------------------------------------------
            _HALF_DEG = 2.5
            _alpha    = math.radians(_HALF_DEG)
            _cos_half = math.cos(_alpha)   # ≈ 0.99905
            _sin_half = math.sin(_alpha)   # ≈ 0.04362

            pipe_r = float(geo_params.get("pipe_radius") or char_len / 2)
            pipe_l = float(geo_params.get("pipe_length")
                          or geo_params.get("channel_length")
                          or ctx.get("domain_x")
                          or char_len * 10)

            # Size the mesh from the pipe itself (not the generic channel nx/ny):
            # 30 radial cells, axial cells with an aspect ratio of about 4.
            # The generic sizing gave aspect ratios ~20, which made the GAMG
            # pressure solver need hundreds of sweeps per iteration.
            f = float(ctx.get("_refine", 1.0))
            ny = max(10, round(30 * f))
            nx = max(20, min(round(pipe_l / (4.0 * pipe_r / ny)), round(2000 * f)))
            dz = round(2.0 * pipe_r * _sin_half, 8)

            pipe_grad_ceil = _ceil_grading_str(pipe_r, ny, y1)
            ctx["_y1_actual"] = _first_cell_height(pipe_r, ny, y1)

            ctx.update(
                domain_x=pipe_l,
                domain_y=round(pipe_r, 8),
                domain_z=dz,
                nx=nx, ny=ny,
                pipe_r=round(pipe_r, 8),
                pipe_l=round(pipe_l, 8),
                pipe_cos_half=round(_cos_half, 8),
                pipe_sin_half=round(_sin_half, 8),
                pipe_grad_ceil=pipe_grad_ceil,
            )

        elif geometry_type == "airfoil":
            # ----------------------------------------------------------------
            # 8-block C/H-grid around a 2-D finite-thickness flat-plate airfoil.
            #
            # The airfoil body (chord × thickness) is a rectangular "hole" in the
            # mesh — no cells are generated inside it.  The surrounding 8 blocks are:
            #
            #   B0  upstream-centre  x∈[-R, 0]   y∈[-t/2, +t/2]  (directly in front of LE)
            #   B1  upper-upstream   x∈[-R, 0]   y∈[+t/2, +R]
            #   B2  upper-body       x∈[0, c]    y∈[+t/2, +R]    (suction side)
            #   B3  upper-wake       x∈[c, c+w]  y∈[+t/2, +R]
            #   B4  wake-centre      x∈[c, c+w]  y∈[-t/2, +t/2]  (behind TE)
            #   B5  lower-body       x∈[0, c]    y∈[-R, -t/2]    (pressure side)
            #   B6  lower-upstream   x∈[-R, 0]   y∈[-R, -t/2]
            #   B7  lower-wake       x∈[c, c+w]  y∈[-R, -t/2]
            #
            # Airfoil patches:
            #   LE face       — B0 i=max  (5, 6, 22, 21)
            #   upper surface — B2 j=0    (6, 10, 26, 22)
            #   lower surface — B5 j=max  (5, 9, 25, 21)
            #   TE face       — B4 i=0    (9, 10, 26, 25)
            #
            # Vertex layout (front face, z = 0):
            #
            #   3─────────────────────────────────────────────15
            #   2────────6────────────────10──────────────────14  y = +t/2
            #            ╔════════════════╗                       (airfoil body — no cells)
            #   1────────5────────────────9───────────────────13  y = −t/2
            #   0────────4────────────────8───────────────────12
            #   x=−R_far  x=0(LE)         x=chord(TE)   x=chord+wake_l
            # ----------------------------------------------------------------
            chord   = float(geo_params.get("chord")       or char_len)
            t       = float(geo_params.get("thickness")   or chord * 0.12)
            R_far   = float(geo_params.get("r_far")       or chord * 15.0)
            wake_l  = float(geo_params.get("wake_length") or chord * 10.0)

            t_half = t / 2.0
            dz     = round(chord * 0.1, 6)

            # Absolute x / y coordinates
            x0 = round(-R_far,  8)
            x1 = 0.0
            x2 = round(chord,   8)
            x3 = round(chord + wake_l, 8)
            y_bot = round(-R_far,  8)
            y_lo  = round(-t_half, 8)
            y_hi  = round( t_half, 8)
            y_top = round( R_far,  8)

            # Cell counts
            nx_tot = int(ctx.get("nx", 80))
            ny_tot = int(ctx.get("ny", 40))
            total_x = R_far + chord + wake_l
            n_le    = max(15, round(R_far  / total_x * nx_tot))
            n_chord = max(30, round(chord  / total_x * nx_tot))
            n_wake  = max(20, round(wake_l / total_x * nx_tot))
            n_th    = max(3,  round(t      / (2 * R_far) * ny_tot // 3))
            n_up    = max(15, round((R_far - t_half) / (2 * R_far) * ny_tot))
            n_dn    = max(15, round((R_far - t_half) / (2 * R_far) * ny_tot))

            # Wall grading (j direction):
            #   upper blocks (B1,B2,B3): j=0 is near airfoil wall → floor grading
            #   lower blocks (B5,B6,B7): j=max is near airfoil wall → ceil grading
            af_grad_upper = _floor_grading_str(R_far - t_half, n_up, y1)
            ctx["_y1_actual"] = _first_cell_height(R_far - t_half, n_up, y1)
            af_grad_lower = _ceil_grading_str(R_far - t_half, n_dn, y1)

            ctx.update(
                domain_x=round(x3 - x0, 8),
                domain_y=round(2 * R_far, 8),
                domain_z=dz,
                nx=nx_tot, ny=ny_tot,
                af_x0=x0, af_x1=x1, af_x2=x2, af_x3=x3,
                af_y_bot=y_bot, af_y_lo=y_lo, af_y_hi=y_hi, af_y_top=y_top,
                af_chord=chord, af_thickness=t, af_t_half=t_half,
                af_R_far=R_far, af_wake_l=wake_l,
                n_le=n_le, n_chord=n_chord, n_wake=n_wake,
                n_th=n_th, n_up=n_up, n_dn=n_dn,
                af_grad_upper=af_grad_upper,
                af_grad_lower=af_grad_lower,
            )

        elif geometry_type == "imported_mesh":
            # Sizes from the mesh bounding box (used by sampling dictionaries and checks)
            x0, x1, y0, y1, z0, z1 = analysis.get("_mesh_bbox") or (0, 1, 0, 1, 0, 1)
            n_cells = int(analysis.get("_mesh_cells") or 1000)
            side = max(2, round(n_cells ** (1 / 3)))
            ctx.update(domain_x=round(x1 - x0, 6), domain_y=round(y1 - y0, 6), domain_z=round(z1 - z0, 6),
                       nx=side, ny=side, nz=side, _y1_actual=None)

        elif geometry_type == "stl_import":
            # ----------------------------------------------------------------
            # Background mesh bounds from uploaded STL bounding box.
            # analysis["_stl_bbox"] = (xmin, xmax, ymin, ymax, zmin, zmax)
            # analysis["_stl_name"] = stem of the uploaded file
            # ----------------------------------------------------------------
            bbox = analysis.get("_stl_bbox") or (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            stl_xmin, stl_xmax, stl_ymin, stl_ymax, stl_zmin, stl_zmax = bbox
            stl_name = str(analysis.get("_stl_name") or "geometry")

            stl_dx = max(stl_xmax - stl_xmin, 1e-6)
            stl_dy = max(stl_ymax - stl_ymin, 1e-6)
            stl_dz = max(stl_zmax - stl_zmin, 1e-6)
            max_dim = max(stl_dx, stl_dy, stl_dz)

            # Expansion factors: asymmetric in x (extra wake), symmetric in y/z
            upstream   = float(geo_params.get("upstream")   or 2.0)
            downstream = float(geo_params.get("downstream") or 5.0)
            side       = float(geo_params.get("side")       or 2.0)

            bg_xmin = stl_xmin - upstream * stl_dx
            bg_xmax = stl_xmax + downstream * stl_dx
            bg_ymin = stl_ymin - side * stl_dy
            bg_ymax = stl_ymax + side * stl_dy
            span    = side * max(stl_dz, stl_dx * 0.5)
            bg_zmin = stl_zmin - span
            bg_zmax = stl_zmax + span

            # Background cell size: ~1/10 of max STL dimension
            bg_cell_size = float(geo_params.get("bg_cell_size") or max_dim * 0.1)
            bg_cell_size = max(bg_cell_size, 1e-6)
            bg_nx = max(10, round((bg_xmax - bg_xmin) / bg_cell_size))
            bg_ny = max(10, round((bg_ymax - bg_ymin) / bg_cell_size))
            bg_nz = max(10, round((bg_zmax - bg_zmin) / bg_cell_size))

            # locationInMesh: 5 % in from inlet, centred in y/z
            loc_x = bg_xmin + 0.05 * (bg_xmax - bg_xmin)
            loc_y = (bg_ymin + bg_ymax) / 2.0
            loc_z = (bg_zmin + bg_zmax) / 2.0

            # Snappy refinement / layer parameters
            snappy_refine_min    = int(geo_params.get("refine_min")    or 1)
            snappy_refine_max    = int(geo_params.get("refine_max")    or 2)
            snappy_feature_level = int(geo_params.get("feature_level") or 3)
            _n_layers = geo_params.get("n_layers")
            snappy_n_layers      = int(_n_layers if _n_layers is not None else 3)
            snappy_max_cells     = int(geo_params.get("max_cells")     or 2_000_000)

            # Append the STL wall patch (copy list to avoid mutating global GEOMETRY_PATCHES)
            ctx["patches"] = list(ctx["patches"])
            ctx["patches"].append({"name": stl_name, "of_type": "wall", "role": "wall"})

            ctx.update(
                stl_name=stl_name,
                stl_file=f"{stl_name}.stl",
                bg_xmin=round(bg_xmin, 6), bg_xmax=round(bg_xmax, 6),
                bg_ymin=round(bg_ymin, 6), bg_ymax=round(bg_ymax, 6),
                bg_zmin=round(bg_zmin, 6), bg_zmax=round(bg_zmax, 6),
                bg_nx=bg_nx, bg_ny=bg_ny, bg_nz=bg_nz,
                loc_x=round(loc_x, 6),
                loc_y=round(loc_y, 6),
                loc_z=round(loc_z, 6),
                snappy_refine_min=snappy_refine_min,
                snappy_refine_max=snappy_refine_max,
                # surface cells are the background cells refined refine_max times
                _y1_actual=bg_cell_size / 2 ** snappy_refine_max,
                snappy_feature_level=snappy_feature_level,
                snappy_n_layers=snappy_n_layers,
                snappy_max_cells=snappy_max_cells,
                snappy_add_layers=(snappy_n_layers > 0),
                # Override domain/grid to match background mesh for grading helpers
                domain_x=round(bg_xmax - bg_xmin, 6),
                domain_y=round(bg_ymax - bg_ymin, 6),
                domain_z=round(bg_zmax - bg_zmin, 6),
                nx=bg_nx, ny=bg_ny, nz=bg_nz,
            )

        return ctx

    # ------------------------------------------------------------------
    # Inlet BC helpers
    # ------------------------------------------------------------------

    # Default inflow direction for inlet patches whose normal is not +x.
    # (T-junction branch inlet sits on the top of the domain, flow enters downwards.)
    _INLET_DIRECTION: dict[str, tuple[float, float, float]] = {
        "inlet2": (0.0, -1.0, 0.0),
    }

    @classmethod
    def _patches_with_inlet_velocities(cls, patches: list, bcs: list, inlet_speed: float) -> list:
        """Return a copy of *patches* where every inlet carries its own ``U`` string.

        A boundary condition whose ``patch_name`` matches the patch exactly wins;
        otherwise the inlet speed is applied along the patch's inflow direction.
        Without this, every inlet would get the first inlet's vector — e.g. a
        T-junction branch inlet would receive a velocity tangential to its face
        and no fluid would enter through it.
        """
        by_name = {bc.get("patch_name"): bc for bc in bcs}
        out = []
        for p in patches:
            p = dict(p)
            if p["role"] == "inlet":
                val = (by_name.get(p["name"]) or {}).get("values", {}).get("U")
                if isinstance(val, (list, tuple)) and len(val) >= 1:
                    v = (list(val) + [0, 0, 0])[:3]
                    p["U"] = f"({v[0]} {v[1]} {v[2]})"
                elif p["name"] in cls._INLET_DIRECTION:
                    d = cls._INLET_DIRECTION[p["name"]]
                    p["U"] = f"({d[0] * inlet_speed:g} {d[1] * inlet_speed:g} {d[2] * inlet_speed:g})"
            out.append(p)
        return out

    @staticmethod
    def _imported_patches(mesh_patches: list, inlet_speed: float) -> list:
        """Patches of an imported mesh; inlets blow along their inward normal at *inlet_speed*."""
        out = []
        for p in mesh_patches:
            p = dict(p)
            n = p.get("normal")
            if p["role"] == "inlet" and n and any(abs(c) > 1e-9 for c in n):
                p["U"] = "(" + " ".join(f"{-c * inlet_speed:.6g}" for c in n) + ")"
            out.append(p)
        return out

    @staticmethod
    def _get_inlet_velocity(bcs: list) -> str:
        for bc in bcs:
            if "inlet" in bc.get("patch_name", "").lower():
                val = bc.get("values", {}).get("U")
                if isinstance(val, list):
                    v = (list(val) + [0, 0, 0])[:3]
                    return f"({v[0]} {v[1]} {v[2]})"
                if val is not None:
                    return str(val)
        return "(1 0 0)"

    @staticmethod
    def _get_inlet_speed(bcs: list) -> float:
        import math
        for bc in bcs:
            if "inlet" in bc.get("patch_name", "").lower():
                val = bc.get("values", {}).get("U")
                if isinstance(val, list):
                    return math.sqrt(sum(v**2 for v in val))
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return 1.0
