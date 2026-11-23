"""
Validate and repair an LLM analysis before it reaches the case generators.

Structured outputs guarantee the *shape* of the analysis, not that its numbers
make physical sense (e.g. a viscosity in the wrong units, a cylinder larger
than its channel, a solver paired with a geometry it cannot use).  This module
checks every value the generators consume and either fixes it with an
explicit, user-visible note or raises :class:`AnalysisError` when the analysis
cannot be used at all.

    clean, notes = sanitize_analysis(raw_analysis)
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from app.schema import GEOMETRIES, GEOMETRY_PARAM_KEYS, SOLVERS, TURB_MODELS
from app.naming import is_inlet_name


class AnalysisError(ValueError):
    """The analysis is unusable (e.g. unknown solver) and cannot be repaired."""


# Geometry restrictions per solver
_SOLVER_ONLY_GEOMETRY = {"buoyantSimpleFoam": "cavity", "interFoam": "dambreak"}
_EXCLUSIVE_GEOMETRIES = {"cavity", "dambreak"}

# Plausible physical ranges (SI units)
_NU_RANGE = (1e-8, 1e-2)          # m²/s — from gases at high pressure to heavy oils
_RHO_RANGE = (1e-3, 2.5e4)        # kg/m³
_T_RANGE = (1.0, 5000.0)          # K
_P_RANGE = (1.0, 1e9)             # Pa (absolute)
_SPEED_MAX = 5000.0               # m/s
_LENGTH_RANGE = (1e-6, 1e5)       # m


def _to_float(value: Any) -> float | None:
    """Convert numbers / numeric strings to a finite float, else None."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _in_range(value: Any, lo: float, hi: float) -> float | None:
    f = _to_float(value)
    return f if f is not None and lo <= f <= hi else None


def sanitize_analysis(analysis: Any, description: str | None = None) -> tuple[dict, list[str]]:
    """Return ``(clean_analysis, notes)``.

    *notes* lists every automatic correction in plain English so the UI can
    show the user what was changed.  The input is never mutated.

    *description* is the user's original text; a Reynolds number stated in it takes
    precedence over the model's numbers.

    Raises
    ------
    AnalysisError
        If *analysis* is not a dict or names an unsupported solver.
    """
    if not isinstance(analysis, dict):
        raise AnalysisError("The model did not return a JSON object.")
    a = copy.deepcopy(analysis)
    notes: list[str] = []

    # ── Solver, turbulence model, geometry ───────────────────────────────────
    solver = a.get("solver")
    if solver not in SOLVERS:
        raise AnalysisError(
            f"Unsupported solver {solver!r}. Supported solvers: {', '.join(SOLVERS)}."
        )

    if a.get("turb_model") not in TURB_MODELS:
        notes.append(f"Unknown turbulence model {a.get('turb_model')!r} — using kEpsilon.")
        a["turb_model"] = "kEpsilon"

    # Solver/geometry conflicts.  The cavity and dam-break geometries only make sense for
    # natural convection and free-surface flow, so when one of them is chosen the geometry
    # is the stronger signal and the solver is fixed to match (the LLM evaluation showed
    # models picking the right geometry with the wrong solver more often than the reverse).
    geometry = a.get("geometry_type")
    if geometry not in ("stl_import", "imported_mesh"):          # user-supplied meshes keep their geometry
        geometry_solver = {g: s for s, g in _SOLVER_ONLY_GEOMETRY.items()}.get(geometry)
        # (a cavity with an inlet velocity is forced flow: keep the solver, see below)
        if geometry_solver and solver != geometry_solver and not (
                geometry == "cavity" and _has_inflow(a)):
            notes.append(f"The '{geometry}' geometry is for {geometry_solver}, not {solver} — "
                         f"using {geometry_solver}.")
            solver = a["solver"] = geometry_solver
        elif solver == "buoyantSimpleFoam" and _has_inflow(a):
            notes.append("buoyantSimpleFoam is for natural convection in a closed cavity, but the "
                         "flow has an inlet velocity (forced convection) — using simpleFoam.")
            solver = a["solver"] = "simpleFoam"
        required = _SOLVER_ONLY_GEOMETRY.get(solver)
        if required and geometry != required:
            notes.append(f"{solver} requires the '{required}' geometry (got {geometry!r}).")
            geometry = required
        elif not required and (geometry not in GEOMETRIES or geometry in _EXCLUSIVE_GEOMETRIES):
            notes.append(f"Geometry {geometry!r} is not available for {solver} — using 'channel'.")
            geometry = "channel"
        a["geometry_type"] = geometry

    # ── Fluid properties ─────────────────────────────────────────────────────
    fluid = a.get("fluid") if isinstance(a.get("fluid"), dict) else {}
    name = str(fluid.get("name") or ("water" if solver == "interFoam" else "air"))
    is_liquid = any(w in name.lower() for w in ("water", "oil", "liquid", "glycerin"))
    default_nu, default_rho = (1.0e-6, 1000.0) if is_liquid else (1.5e-5, 1.225)

    nu = _in_range(fluid.get("nu"), *_NU_RANGE)
    if nu is None:
        notes.append(
            f"Kinematic viscosity {fluid.get('nu')!r} is outside {_NU_RANGE[0]:g}–{_NU_RANGE[1]:g} m²/s "
            f"(wrong units?) — using {default_nu:g} m²/s for {name}."
        )
        nu = default_nu
    rho = _in_range(fluid.get("rho"), *_RHO_RANGE)
    if rho is None:
        notes.append(f"Density {fluid.get('rho')!r} is implausible — using {default_rho:g} kg/m³.")
        rho = default_rho
    a["fluid"] = {**fluid, "name": name, "nu": nu, "rho": rho}

    # ── Characteristic scales ────────────────────────────────────────────────
    char_len = _in_range(a.get("characteristic_length"), *_LENGTH_RANGE)
    if char_len is None:
        notes.append(f"Characteristic length {a.get('characteristic_length')!r} is invalid — using 1 m.")
        char_len = 1.0
    a["characteristic_length"] = char_len

    char_vel = _in_range(a.get("characteristic_velocity"), 0.0, _SPEED_MAX)
    a["characteristic_velocity"] = char_vel if char_vel is not None else 1.0

    # ── Boundary conditions ──────────────────────────────────────────────────
    bcs_out = []
    for bc in a.get("boundary_conditions") or []:
        if not isinstance(bc, dict) or not bc.get("patch_name"):
            continue
        values = {}
        for key, val in (bc.get("values") or {}).items():
            if val is None:
                continue
            if key == "U":
                vec = val if isinstance(val, (list, tuple)) else [val]
                comps = [_to_float(c) for c in vec][:3]
                if not comps or any(c is None for c in comps):
                    notes.append(f"Ignored non-numeric velocity {val!r} on patch {bc['patch_name']}.")
                    continue
                comps = (comps + [0.0, 0.0, 0.0])[:3]
                speed = math.sqrt(sum(c * c for c in comps))
                if speed > _SPEED_MAX:
                    scale = _SPEED_MAX / speed
                    comps = [c * scale for c in comps]
                    notes.append(f"Velocity on {bc['patch_name']} capped at {_SPEED_MAX:g} m/s.")
                values["U"] = comps
            elif key == "T":
                t = _in_range(val, *_T_RANGE)
                if t is None:
                    notes.append(f"Ignored implausible temperature {val!r} K on {bc['patch_name']}.")
                    continue
                values["T"] = t
            elif key == "alpha.water":
                f = _to_float(val)
                if f is not None:
                    values["alpha.water"] = min(max(f, 0.0), 1.0)
            else:
                values[key] = val
        bcs_out.append({**bc, "values": values})
    a["boundary_conditions"] = bcs_out

    # ── Reference state (compressible / buoyant) ─────────────────────────────
    for key, rng, label in (("T_ref", _T_RANGE, "temperature"), ("p_ref", _P_RANGE, "pressure")):
        if a.get(key) is not None and _in_range(a.get(key), *rng) is None:
            notes.append(f"Implausible reference {label} {a.get(key)!r} — using the default.")
            a[key] = None
        elif a.get(key) is not None:
            a[key] = float(a[key])

    # ── Inlet velocities must point into the domain ──────────────────────────
    _fix_inlet_directions(a, notes)

    # ── Mach number: a stated Mach sets the velocity; Ma > 0.3 needs a compressible solver
    if not is_liquid:
        solver = _consistent_mach(a, solver, description, notes)

    # ── Reynolds number: consistent with U·L/ν and with the user's request ───
    a["Re"] = _consistent_reynolds(a, solver, char_len, nu, description, notes)
    if solver == "buoyantSimpleFoam":
        _consistent_rayleigh(a, description, nu, notes)

    # ── Flow model must match the regime (only where Re leaves no doubt) ─────
    if a["solver"] in _INCOMPRESSIBLE + ("rhoSimpleFoam",) and a["Re"] is not None:
        if a["Re"] < _LAMINAR_BELOW and a["turb_model"] != "laminar":
            notes.append(f"Re = {a['Re']:.3g} is laminar — using the laminar model instead of "
                         f"{a['turb_model']}.")
            a["turb_model"] = "laminar"
        elif a["Re"] > _TURBULENT_ABOVE and a["turb_model"] == "laminar":
            notes.append(f"Re = {a['Re']:.3g} is turbulent — using the k-ω SST model instead of "
                         "laminar flow.")
            a["turb_model"] = "kOmegaSST"

    # ── Bluff body above the shedding threshold needs a transient solver ─────
    crit = _SHEDDING_RE.get(a.get("geometry_type"))
    # Laminar and RANS alike: the wake of a bluff body sheds vortices at any Re above the
    # threshold, and steady RANS of it oscillated without converging (3 of 30 benchmark
    # descriptions); unsteady RANS resolves the shedding.
    if (crit and a["solver"] == "simpleFoam"
            and a["Re"] is not None and a["Re"] > crit):
        model = "" if a["turb_model"] == "laminar" else f" (unsteady RANS, {a['turb_model']})"
        notes.append(
            f"Re = {a['Re']:.3g} is above the vortex-shedding threshold (Re ≈ {crit:g}) for a "
            f"{a['geometry_type'].replace('_', ' ')}: the wake is unsteady and a steady solver "
            f"cannot converge, so the transient pimpleFoam{model} is used instead of simpleFoam."
        )
        a["solver"] = "pimpleFoam"
        a["is_transient"] = True

    # ── Geometry parameters: positive numbers that fit inside the domain ─────
    a["geometry_params"] = _sanitize_geometry_params(
        a.get("geometry_params"), a.get("geometry_type"), char_len, notes,
    )

    # ── Single-box meshes: domain from the geometry, cell counts from the description ─
    _consistent_domain(a, description, notes)

    a["is_transient"] = bool(a.get("is_transient", solver in ("pimpleFoam", "interFoam")))
    a["warnings"] = [str(w) for w in (a.get("warnings") or []) if w]
    a["unsupported"] = unsupported_requests(a, description)
    return a, notes


# ── Domain and cell counts of single-box meshes ──────────────────────────────

_SINGLE_BOX = ("channel", "cavity", "dambreak")
# "80 × 200", "80x200", "80 by 200" (cell counts, so not followed by a length unit)
_CELLS_PATTERN = re.compile(r"\b(\d{1,5})\s*(?:[×xX*]|by)\s*(\d{1,5})(?:\s*(?:[×xX*]|by)\s*(\d{1,5}))?\b"
                            r"(?!\s*(?:m\b|mm\b|cm\b|m²|m2\b|meters?\b|metres?\b))")


def stated_cells(description: str | None) -> tuple[int, int] | None:
    """(nx, ny) from "a mesh of 80 × 200 cells" in the description (a third count of 1 is allowed)."""
    for sentence in re.split(r"[.;\n]+(?:\s|$)", description or ""):
        if not re.search(r"\b(?:cells?|mesh|grid)\b", sentence, re.I):
            continue
        for m in _CELLS_PATTERN.finditer(sentence):
            nx, ny, nz = int(m.group(1)), int(m.group(2)), m.group(3)
            if nx >= 4 and ny >= 4 and (nz is None or int(nz) == 1):
                return nx, ny
    return None


def _consistent_domain(a: dict, description: str | None, notes: list[str]) -> None:
    """For single-box geometries the mesh block *is* the geometry, so the model's
    ``suggested_mesh`` must agree with its ``geometry_params`` and with cell counts the
    user wrote. (A model once put the requested 1 m × 3 m in geometry_params and
    20 m × 10 m in suggested_mesh; the generator used the latter.)"""
    if a.get("geometry_type") not in _SINGLE_BOX:
        return
    mesh = a["suggested_mesh"] = dict(a.get("suggested_mesh") or {})
    gp = a.get("geometry_params") or {}
    L, H = gp.get("channel_length"), gp.get("channel_height")
    dx, dy = _to_float(mesh.get("domain_x")), _to_float(mesh.get("domain_y"))
    resized = False
    if L and H and (not dx or not dy or abs(dx / L - 1) > 0.05 or abs(dy / H - 1) > 0.05):
        if dx and dy:
            notes.append(f"The mesh was sized {dx:g} m × {dy:g} m but the geometry is "
                         f"{L:g} m × {H:g} m — using the geometry's size.")
        mesh["domain_x"], mesh["domain_y"] = L, H
        resized = bool(dx and dy)

    cells = stated_cells(description)
    nx, ny = _to_float(mesh.get("nx")), _to_float(mesh.get("ny"))
    if cells:
        if (nx, ny) != (float(cells[0]), float(cells[1])):
            if nx and ny:
                notes.append(f"The description asks for {cells[0]} × {cells[1]} cells — using that "
                             f"instead of the model's {int(nx)} × {int(ny)}.")
            mesh["nx"], mesh["ny"] = cells
    elif resized and nx and ny:
        # keep the model's cell count, with near-square cells in the new domain
        n = nx * ny
        L, H = mesh["domain_x"], mesh["domain_y"]
        mesh["nx"] = max(4, round(math.sqrt(n * L / H)))
        mesh["ny"] = max(4, round(n / mesh["nx"]))


# ── Requests the generated case cannot represent ─────────────────────────────

_PLUME = re.compile(r"\b(plumes?|jets?|nozzles?|injections?|injected|injecting|orifices?|holes?|"
                    r"slots?|chimneys?|burners?)\b", re.I)
_TRANSIENT_WORDS = re.compile(r"\b(transient|unsteady|time[- ]dependent|time[- ]accurate|"
                              r"evolution over time|puffing)\b", re.I)
_GEOMETRY_WORDS = {
    "cavity": "a closed box heated or cooled through its side walls",
    "channel": "a straight channel with an inlet on the left and an outlet on the right",
    "dambreak": "a closed tank with a water column",
}


def unsupported_requests(a: dict, description: str | None) -> list[str]:
    """What the description asks for that the generated case does not contain.

    Shown before the run so the user is not left with a case that solves a different
    problem (and a repair agent that cannot fix a modelling mismatch)."""
    text = description or ""
    if not text.strip():
        return []
    out = []
    geometry = a.get("geometry_type")
    own_mesh = geometry in ("stl_import", "imported_mesh", "t_junction")   # (its branch is an inlet)
    m = _PLUME.search(text)
    if m and not own_mesh:
        shape = _GEOMETRY_WORDS.get(geometry, f"the built-in '{geometry}' geometry")
        out.append(f"**Geometry:** the description mentions a {m.group(1).lower()}, but no built-in "
                   f"geometry has one; the case uses {shape} instead. To model it, mesh your domain "
                   "in SALOME with an `inlet` group on the opening and import it.")
    if re.search(r"sutherland", text, re.I):
        out.append("**Viscosity:** Sutherland's law is not supported; the case uses a constant "
                   "viscosity.")
    if re.search(r"\b(LES|DNS|large[- ]eddy|direct numerical)\b", text, re.I):
        out.append(f"**Turbulence:** LES and DNS are not supported; the case uses "
                   f"{a.get('turb_model', 'a RANS model')}.")
    steady = a.get("solver") in ("buoyantSimpleFoam", "rhoSimpleFoam", "simpleFoam")
    if steady and _TRANSIENT_WORDS.search(text):
        out.append(f"**Time dependence:** the description asks for a time-dependent flow but "
                   f"{a['solver']} is steady"
                   + (" (there is no transient buoyant solver yet)." if a["solver"] == "buoyantSimpleFoam"
                      else "."))
    return out


# Geometries whose main inlet is the x = 0 face (flow must enter in +x)
_INLET_PLUS_X = {"channel", "t_junction", "rectangular_obstacle", "backstep", "cylinder",
                 "pipe", "airfoil", "stl_import"}
# Laminar Reynolds number above which the wake sheds vortices periodically
_SHEDDING_RE = {"cylinder": 47.0, "rectangular_obstacle": 50.0}
_RE_TOLERANCE = 0.10
# Unambiguous regimes; between the two the model's choice is kept (transitional flow)
_LAMINAR_BELOW = 1_000.0
_TURBULENT_ABOVE = 10_000.0
_INCOMPRESSIBLE = ("simpleFoam", "pimpleFoam")

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_NUMBER_AFTER = (
    r"\s*(?:=|≈|~|:|of|is|at)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*[eE](?P<exp>[-+]?\d+))?"                                   # 1e6
    r"(?:\s*(?:\^|\*\*)\s*(?P<pow>[-+]?\d+)|(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+))?"   # 10^5, 10⁵
    r"(?:\s*[x×*]\s*10\s*(?:(?:\^|\*\*)\s*(?P<pow10>[-+]?\d+)|(?P<sup10>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)))?"  # 2 × 10⁶
    r"(?:\s*(?P<word>million|thousand|billion|[kKM])\b)?"            # 2 million, 50k
)
_RE_PATTERN = re.compile(r"\b(?:Re(?:_\w+)?|Reynolds(?:\s+number)?)" + _NUMBER_AFTER)
_RA_PATTERN = re.compile(r"\b(?:Ra|Rayleigh(?:\s+number)?)" + _NUMBER_AFTER)
_WORDS = {"thousand": 1e3, "k": 1e3, "K": 1e3, "million": 1e6, "M": 1e6, "billion": 1e9}


_MACH_PATTERN = re.compile(r"\b(?:Mach|Ma)(?:\s+number)?\s*(?:=|≈|~|:|of|is|at)?\s*(\d+(?:\.\d+)?)")
_COMPRESSIBLE_ABOVE = 0.3


def stated_mach(description: str | None) -> float | None:
    """Mach number written in the description ("Mach 0.8", "Ma = 0.5")."""
    m = _MACH_PATTERN.search(description or "")
    value = _to_float(m.group(1)) if m else None
    return value if value and 0 < value < 10 else None


def _consistent_mach(a: dict, solver: str, description: str | None, notes: list[str]) -> str:
    """Gas flows only.  Returns the (possibly changed) solver.

    * A Mach number stated in the description sets the inlet speed (models wrote e.g.
      0.82 m/s or 600 m/s for "Mach 0.8").
    * A steady incompressible solver above Ma ≈ 0.3 is replaced by rhoSimpleFoam.
    """
    T = _to_float(a.get("T_ref")) or 300.0
    a_sound = math.sqrt(1.4 * 287.0 * T)
    speed = _first_inlet_speed(a["boundary_conditions"])
    ma_user = stated_mach(description)
    if ma_user and speed > 0 and abs(speed / a_sound / ma_user - 1) > _RE_TOLERANCE:
        scale = ma_user * a_sound / speed
        for bc in a["boundary_conditions"]:
            if is_inlet_name(bc["patch_name"]) and bc["values"].get("U"):
                bc["values"]["U"] = [c * scale for c in bc["values"]["U"]]
        notes.append(f"The description asks for Mach {ma_user:g} but the inlet speed {speed:.4g} m/s is "
                     f"Mach {speed / a_sound:.3g} at {T:g} K — inlet speed set to {ma_user * a_sound:.4g} m/s.")
        speed = ma_user * a_sound
        a["characteristic_velocity"] = speed
    mach = speed / a_sound if speed > 0 else 0.0
    if mach > _COMPRESSIBLE_ABOVE and solver == "simpleFoam":
        notes.append(f"Ma = {mach:.2f} > {_COMPRESSIBLE_ABOVE}: density changes matter, so the compressible "
                     "rhoSimpleFoam is used instead of simpleFoam.")
        solver = a["solver"] = "rhoSimpleFoam"
    elif mach > _COMPRESSIBLE_ABOVE and solver == "pimpleFoam":
        notes.append(f"Ma = {mach:.2f} > {_COMPRESSIBLE_ABOVE} with an incompressible transient solver: "
                     "results will be inaccurate (no transient compressible solver is available).")
    return solver


def stated_reynolds(description: str | None) -> float | None:
    """Reynolds number written in the user's description, e.g. "Re = 100", "Reynolds number
    of 2,000", "Re = 2 million", "Re = 5e5", "Re = 10^5", "Re = 2 × 10⁶"."""
    return _stated_number(_RE_PATTERN, description)


def stated_rayleigh(description: str | None) -> float | None:
    """Rayleigh number written in the description ("Ra = 1e5", "Rayleigh number of 10^6")."""
    return _stated_number(_RA_PATTERN, description)


def _consistent_rayleigh(a: dict, description: str | None, nu: float, notes: list[str]) -> None:
    """Natural convection: a stated Ra sets the cavity size (the temperatures are kept).
    Models picked e.g. a 1 m cavity with ΔT = 50 K for "Ra = 1e5", which is Ra ≈ 5e9."""
    ra_user = stated_rayleigh(description)
    if not ra_user:
        return
    from app.generators.buoyant_simple_foam import BuoyantSimpleFoamGenerator
    T_hot, T_cold, T_mean = BuoyantSimpleFoamGenerator._extract_temperatures(a)
    dT = abs(T_hot - T_cold)
    if dT <= 0 or nu <= 0:
        return
    alpha = nu / 0.71                                   # thermal diffusivity of air, Pr = 0.71
    L = _to_float(a.get("characteristic_length")) or 1.0
    ra = 9.81 / T_mean * dT * L ** 3 / (nu * alpha)
    if abs(ra / ra_user - 1) <= _RE_TOLERANCE:
        return
    L_new = (ra_user * nu * alpha * T_mean / (9.81 * dT)) ** (1.0 / 3.0)
    notes.append(f"The description asks for Ra = {ra_user:g} but a {L:.3g} m cavity with ΔT = {dT:g} K "
                 f"gives Ra = {ra:.3g} — cavity size set to {L_new:.4g} m.")
    a["characteristic_length"] = round(L_new, 6)
    for key in ("channel_length", "channel_height"):
        if isinstance(a.get("geometry_params"), dict):
            a["geometry_params"].pop(key, None)


def _stated_number(pattern: re.Pattern, description: str | None) -> float | None:
    if not description:
        return None
    m = pattern.search(description)
    if not m:
        return None
    value = _to_float(m.group("num").replace(",", ""))
    if value is None:
        return None
    if m.group("exp"):
        value *= 10.0 ** int(m.group("exp"))
    power = m.group("pow") or (m.group("sup") or "").translate(_SUPERSCRIPT)
    if power:
        value = value ** int(power)
    power10 = m.group("pow10") or (m.group("sup10") or "").translate(_SUPERSCRIPT)
    if power10:
        value *= 10.0 ** int(power10)
    if m.group("word"):
        value *= _WORDS[m.group("word")]
    return value if value > 0 else None


def _fix_inlet_directions(a: dict, notes: list[str]) -> None:
    """Small models sometimes put the inlet velocity in the wrong component (e.g. (0, U, 0))."""
    geometry = a.get("geometry_type")
    if geometry not in _INLET_PLUS_X:
        return
    for bc in a["boundary_conditions"]:
        name, U = bc["patch_name"], bc["values"].get("U")
        if not is_inlet_name(name) or not U:
            continue
        speed = math.sqrt(sum(c * c for c in U))
        if speed == 0:
            continue
        if geometry == "t_junction" and name == "inlet2":          # branch inlet on the top face
            if U[1] > -0.1 * speed:
                bc["values"]["U"] = [0.0, -speed, 0.0]
                notes.append(f"Inlet velocity on {name} {_vec(U)} did not enter the branch — "
                             f"using (0, -{speed:g}, 0).")
        elif U[0] < 0.1 * speed:
            bc["values"]["U"] = [speed, 0.0, 0.0]
            notes.append(f"Inlet velocity on {name} {_vec(U)} did not point into the domain (+x) — "
                         f"using ({speed:g}, 0, 0).")


def _vec(U) -> str:
    return "(" + ", ".join(f"{c:g}" for c in U) + ")"


def _consistent_reynolds(a: dict, solver: str, char_len: float, nu: float,
                         description: str | None, notes: list[str]) -> float | None:
    """Make Re, the inlet velocity, the length and the viscosity agree.

    * The user's description states Re → it wins: the inlet velocities are rescaled
      (incompressible solvers only — for compressible flow that would change the Mach number).
    * Otherwise the velocity wins: the model's Re is replaced by U·L/ν when they disagree.
    """
    re_model = _to_float(a.get("Re"))
    re_model = re_model if re_model and re_model > 0 else None
    speed = _first_inlet_speed(a["boundary_conditions"])
    re_flow = speed * char_len / nu if speed > 0 else None
    if re_flow is None:
        return re_model

    re_user = stated_reynolds(description)
    if re_user and abs(re_flow / re_user - 1) > _RE_TOLERANCE:
        if solver in _INCOMPRESSIBLE:
            scale = re_user / re_flow
            for bc in a["boundary_conditions"]:
                if is_inlet_name(bc["patch_name"]) and bc["values"].get("U"):
                    bc["values"]["U"] = [c * scale for c in bc["values"]["U"]]
            a["characteristic_velocity"] = speed * scale
            notes.append(
                f"The model's inputs give Re = U·L/ν = {re_flow:.3g}, but the description asks for "
                f"Re = {re_user:g}: inlet velocity rescaled from {speed:.4g} to {speed * scale:.4g} m/s "
                f"(L = {char_len:g} m, ν = {nu:g} m²/s)."
            )
            return re_user
        notes.append(f"The description asks for Re = {re_user:g} but U·L/ν = {re_flow:.3g}; the "
                     "velocity is kept (it sets the Mach number) — check the inputs.")
        return re_flow

    if re_model and abs(re_flow / re_model - 1) > _RE_TOLERANCE:
        notes.append(f"The model reported Re = {re_model:.3g}, but its inlet velocity, length and "
                     f"viscosity give U·L/ν = {re_flow:.3g} — using {re_flow:.3g}.")
    return re_flow


def _has_inflow(a: dict) -> bool:
    """True if any inlet patch has a non-zero numeric velocity (raw, before sanitising)."""
    for bc in a.get("boundary_conditions") or []:
        if not isinstance(bc, dict) or not is_inlet_name(bc.get("patch_name")):
            continue
        U = (bc.get("values") or {}).get("U")
        if isinstance(U, (list, tuple)):
            comps = [_to_float(c) for c in U]
            if all(c is not None for c in comps) and any(abs(c) > 1e-9 for c in comps):
                return True
    return False


def _first_inlet_speed(bcs: list) -> float:
    for bc in bcs:
        if is_inlet_name(bc.get("patch_name", "")) and "U" in bc["values"]:
            return math.sqrt(sum(c * c for c in bc["values"]["U"]))
    return 0.0


def _sanitize_geometry_params(raw: Any, geometry: str | None, char_len: float,
                              notes: list[str]) -> dict:
    gp: dict = dict(raw) if isinstance(raw, dict) else {}
    # Keys the generators read beyond the schema (STL / dambreak / snappy tuning)
    for key in list(gp):
        if key in GEOMETRY_PARAM_KEYS or key in ("xo1", "x_water", "y_water"):
            f = _to_float(gp[key])
            if gp[key] is not None and (f is None or f <= 0):
                notes.append(f"Ignored invalid geometry parameter {key}={gp[key]!r}.")
                f = None
            gp[key] = f

    def fix(key: str, limit: float, why: str) -> None:
        if gp.get(key) is not None and gp[key] > limit:
            notes.append(f"{key}={gp[key]:g} m {why} — reduced to {limit:g} m.")
            gp[key] = limit

    if geometry == "cylinder":
        H = gp.get("channel_height")
        L = gp.get("channel_length")
        if H:
            fix("cylinder_radius", 0.2 * H, f"does not fit in a {H:g} m high domain")
        if L and gp.get("cx") is not None:
            fix("cx", 0.6 * L, f"leaves no wake region in a {L:g} m long domain")
    elif geometry == "airfoil":
        chord = gp.get("chord") or char_len
        fix("thickness", 0.4 * chord, f"is too thick for a {chord:g} m chord")
    elif geometry == "backstep":
        H, L = gp.get("channel_height"), gp.get("channel_length")
        if H:
            fix("step_height", 0.8 * H, f"must be smaller than the {H:g} m channel height")
        if L:
            fix("step_x", 0.7 * L, f"leaves no room downstream in a {L:g} m channel")
    elif geometry == "pipe":
        L = gp.get("pipe_length")
        if L:
            fix("pipe_radius", 0.5 * L, f"is larger than half the {L:g} m pipe length")
    return gp
