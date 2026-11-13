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
from typing import Any

from app.schema import GEOMETRIES, GEOMETRY_PARAM_KEYS, SOLVERS, TURB_MODELS


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


def sanitize_analysis(analysis: Any) -> tuple[dict, list[str]]:
    """Return ``(clean_analysis, notes)``.

    *notes* lists every automatic correction in plain English so the UI can
    show the user what was changed.  The input is never mutated.

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

    geometry = a.get("geometry_type")
    if geometry != "stl_import":
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

    # ── Reynolds number (recompute when missing or non-numeric) ──────────────
    re_val = _to_float(a.get("Re"))
    inlet_speed = _first_inlet_speed(a["boundary_conditions"])
    if re_val is None and inlet_speed > 0:
        re_val = inlet_speed * char_len / nu
        if a.get("Re") is not None:
            notes.append(f"Recomputed Re = {re_val:.3g} from inlet speed, length and viscosity.")
    a["Re"] = re_val if re_val is None or re_val > 0 else None

    # ── Geometry parameters: positive numbers that fit inside the domain ─────
    a["geometry_params"] = _sanitize_geometry_params(
        a.get("geometry_params"), a.get("geometry_type"), char_len, notes,
    )

    a["is_transient"] = bool(a.get("is_transient", solver in ("pimpleFoam", "interFoam")))
    a["warnings"] = [str(w) for w in (a.get("warnings") or []) if w]
    return a, notes


def _first_inlet_speed(bcs: list) -> float:
    for bc in bcs:
        if "inlet" in bc.get("patch_name", "").lower() and "U" in bc["values"]:
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
