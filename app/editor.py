"""
Parameter-editor helpers — pure functions, no Streamlit imports.

``apply_overrides(analysis, overrides)`` merges a flat overrides dict
(populated by the Streamlit parameter editor widgets) into a deep copy
of the LLM analysis dict so that generators receive the user-adjusted
values without mutating the original.
"""

import copy
import math


def apply_overrides(analysis: dict, overrides: dict) -> dict:
    """
    Return a deep copy of *analysis* with *overrides* applied.

    Recognised keys in *overrides*
    --------------------------------
    inlet_speed : float
        Scalar inlet speed [m/s].  The direction of the original U vector
        is preserved; only the magnitude is replaced.
    nu : float
        Kinematic viscosity [m²/s].  Also triggers a Re recompute.
    turb_model : str
        One of ``laminar`` | ``kEpsilon`` | ``kOmegaSST`` | ``SpalartAllmaras``.
    nx : int
        Streamwise cell count stored in ``suggested_mesh``.
    ny : int
        Wall-normal cell count stored in ``suggested_mesh``.
    domain_x : float
        Streamwise domain length [m] stored in ``suggested_mesh``.
    domain_y : float
        Wall-normal domain height [m] stored in ``suggested_mesh``.

    Parameters
    ----------
    analysis : dict
        Original LLM analysis dict (not mutated).
    overrides : dict
        Flat dict of values to apply.  Unknown keys are ignored.

    Returns
    -------
    dict
        Modified analysis dict, safe to pass directly to any generator.
    """
    a = copy.deepcopy(analysis)

    # ── Inlet velocity ───────────────────────────────────────────────────────
    speed = overrides.get("inlet_speed")
    if speed is not None and speed > 0:
        bcs = a.setdefault("boundary_conditions", [])
        _patched = False
        for bc in bcs:
            if "inlet" in (bc.get("patch_name") or "").lower():
                val = (bc.get("values") or {}).get("U")
                if isinstance(val, list) and val:
                    mag = math.sqrt(sum(float(v) ** 2 for v in val)) or 1.0
                    bc.setdefault("values", {})["U"] = [
                        float(v) / mag * speed for v in val
                    ]
                else:
                    bc.setdefault("values", {})["U"] = [speed, 0.0, 0.0]
                _patched = True
        if not _patched:
            bcs.append({
                "patch_name": "inlet",
                "type": "fixedValue",
                "values": {"U": [speed, 0.0, 0.0]},
            })

    # ── Kinematic viscosity ──────────────────────────────────────────────────
    nu = overrides.get("nu")
    if nu is not None and nu > 0:
        a.setdefault("fluid", {})["nu"] = float(nu)

    # ── Re: recompute from the now-updated speed and nu ──────────────────────
    _nu   = float((a.get("fluid") or {}).get("nu") or 1.5e-5)
    _bcs  = a.get("boundary_conditions") or []
    _spd  = _get_inlet_speed(_bcs)
    _clen = float(a.get("characteristic_length") or 1.0)
    if _nu > 0 and _spd > 0:
        a["Re"] = _spd * _clen / _nu

    # ── Turbulence model ─────────────────────────────────────────────────────
    turb = overrides.get("turb_model")
    if turb is not None:
        a["turb_model"] = turb

    # ── Mesh parameters (forwarded via suggested_mesh) ───────────────────────
    # BaseCaseGenerator._mesh_overrides() reads these: nx/ny/domain_x/domain_y
    # apply to single-block geometries, "refinement" scales every geometry.
    mesh = a.get("suggested_mesh")
    if not isinstance(mesh, dict):
        mesh = a["suggested_mesh"] = {}
    for key in ("nx", "ny", "domain_x", "domain_y"):
        val = overrides.get(key)
        if val is not None:
            mesh[key] = val
    refinement = overrides.get("mesh_refinement")
    if refinement is not None:
        mesh["refinement"] = float(refinement)

    # ── Parallel cores ───────────────────────────────────────────────────────
    n_cores = overrides.get("n_cores")
    if n_cores is not None:
        a["n_cores"] = max(1, int(n_cores))

    # ── snappyHexMesh geometry_params (used when geometry_type == 'stl_import') ─
    if a.get("geometry_params") is None:
        a["geometry_params"] = {}
    geo_params = a["geometry_params"]
    for _snappy_key in ("refine_min", "refine_max", "feature_level", "n_layers",
                        "max_cells", "upstream", "downstream", "side"):
        val = overrides.get(f"snappy_{_snappy_key}")
        if val is not None:
            geo_params[_snappy_key] = val

    return a


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with validation.py (kept private here to avoid circular deps)
# ─────────────────────────────────────────────────────────────────────────────

def mesh_summary(analysis: dict) -> dict:
    """Domain size and cell counts the selected generator will produce.

    Falls back to the simpleFoam channel sizing when the solver is unknown.
    """
    from app.generators import GENERATORS          # local import: generators are heavier
    from app.generators.base import SINGLE_BLOCK_GEOMETRIES

    gen_cls = GENERATORS.get(analysis.get("solver", ""))
    if gen_cls is not None:
        try:
            return gen_cls().mesh_summary(analysis)
        except Exception:                          # malformed analysis → use fallback
            pass
    char_len = float(analysis.get("characteristic_length") or 1.0)
    mesh = analysis.get("suggested_mesh") or {}
    domain_x = float(mesh.get("domain_x") or 20.0 * char_len)
    domain_y = float(mesh.get("domain_y") or 10.0 * char_len)
    return {
        "geometry_type": analysis.get("geometry_type") or "channel",
        "domain_x": domain_x,
        "domain_y": domain_y,
        "nx": int(mesh.get("nx") or max(50, int(domain_x / char_len * 10))),
        "ny": int(mesh.get("ny") or max(25, int(domain_y / char_len * 10))),
        "single_block": (analysis.get("geometry_type") or "channel") in SINGLE_BLOCK_GEOMETRIES,
    }


def _get_inlet_speed(bcs: list) -> float:
    """Extract scalar inlet speed from a boundary conditions list."""
    for bc in bcs:
        if "inlet" in (bc.get("patch_name") or "").lower():
            val = (bc.get("values") or {}).get("U")
            if isinstance(val, list):
                return math.sqrt(sum(float(v) ** 2 for v in val))
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return 1.0


def defaults_from_analysis(analysis: dict) -> dict:
    """
    Compute default editor values from an analysis dict.

    Returns a flat dict with the same keys as accepted by
    :func:`apply_overrides`.  Used to pre-fill widget ``value=`` arguments.
    """
    fluid    = analysis.get("fluid") or {}
    nu       = float(fluid.get("nu") or 1.5e-5)
    bcs      = analysis.get("boundary_conditions") or []
    speed    = _get_inlet_speed(bcs)
    turb     = analysis.get("turb_model") or "kEpsilon"
    char_len = float(analysis.get("characteristic_length") or 1.0)
    mesh     = analysis.get("suggested_mesh") or {}

    # Ask the solver's generator what it will actually build, so the editor
    # shows the real blockMesh values (they differ per solver and geometry).
    summary  = mesh_summary(analysis)
    domain_x = summary["domain_x"]
    domain_y = summary["domain_y"]
    nx       = summary["nx"]
    ny       = summary["ny"]

    n_cores = int(analysis.get("n_cores") or 1)

    # Snappy defaults (geometry_params, relevant when geometry_type == 'stl_import')
    geo_params = analysis.get("geometry_params") or {}
    _n_layers   = geo_params.get("n_layers")

    return {
        "inlet_speed":          speed,
        "nu":                   nu,
        "turb_model":           turb,
        "nx":                   nx,
        "ny":                   ny,
        "domain_x":             domain_x,
        "domain_y":             domain_y,
        "n_cores":              n_cores,
        "mesh_refinement":      float(mesh.get("refinement") or 1.0),
        "single_block":         summary["single_block"],
        "snappy_refine_min":    int(geo_params.get("refine_min")    or 1),
        "snappy_refine_max":    int(geo_params.get("refine_max")    or 2),
        "snappy_feature_level": int(geo_params.get("feature_level") or 3),
        "snappy_n_layers":      int(_n_layers if _n_layers is not None else 3),
        "snappy_max_cells":     int(geo_params.get("max_cells")     or 2_000_000),
        "snappy_upstream":      float(geo_params.get("upstream")    or 2.0),
        "snappy_downstream":    float(geo_params.get("downstream")  or 5.0),
        "snappy_side":          float(geo_params.get("side")        or 2.0),
    }
