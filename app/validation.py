"""
Physics-based pre-download validation checks.

``run_checks(analysis)`` returns a list of result dicts::

    {
        "level":  "ok" | "info" | "warning" | "error",
        "title":  str,   # short label (shown as bold header)
        "detail": str,   # one-sentence explanation with computed values
    }

All checks are purely computational — no file I/O, no network calls.
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

# Speed of sound in air at ~20 °C, standard pressure  [m/s]
_C_SOUND_AIR: float = 343.0

# Blasius turbulent flat-plate skin-friction correlation:
#   Cf ≈ _CF_A × Re^_CF_B   (valid Re ~ 5×10⁵ – 10⁷)
_CF_A: float = 0.026
_CF_B: float = -1.0 / 7.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_checks(analysis: dict) -> list[dict]:
    """
    Run all physics validation checks on *analysis*.

    Parameters
    ----------
    analysis : dict
        Structured analysis dict as returned by ``analyze_cfd_problem``.

    Returns
    -------
    list[dict]
        Ordered list of check results, each with keys
        ``level``, ``title``, ``detail``.
    """
    checks: list[dict] = []

    # ── Extract common quantities ────────────────────────────────────────────
    solver       = analysis.get("solver") or ""
    turb         = analysis.get("turb_model") or "kEpsilon"
    geometry     = analysis.get("geometry_type") or "channel"
    fluid        = analysis.get("fluid") or {}
    nu           = float(fluid.get("nu") or 1.5e-5)
    rho          = float(fluid.get("rho") or 1.225)
    fluid_name   = (fluid.get("name") or "").lower()
    is_transient = bool(analysis.get("is_transient", False))
    char_len     = float(analysis.get("characteristic_length") or 1.0)

    bcs          = analysis.get("boundary_conditions") or []
    inlet_speed  = _get_inlet_speed(bcs)

    Re = analysis.get("Re")
    if Re is None and nu > 0 and inlet_speed > 1e-9:
        Re = inlet_speed * char_len / nu

    mp = _estimate_mesh(analysis, char_len)

    # ── 1. Reynolds number / flow regime ────────────────────────────────────
    if Re is not None:
        checks.append(_check_reynolds(float(Re), solver, turb, geometry))

    # ── 2. Mach number (gases only; skip buoyant and two-phase) ─────────────
    if (
        _is_gas(rho, nu, fluid_name)
        and solver not in ("interFoam", "buoyantSimpleFoam")
    ):
        checks.append(_check_mach(inlet_speed, solver))

    # ── 3. Wall y⁺ estimate (skip two-phase, buoyancy-driven, zero-speed) ───
    if (
        Re is not None
        and inlet_speed > 1e-9
        and solver not in ("interFoam", "buoyantSimpleFoam")
    ):
        checks.append(_check_yplus(float(Re), inlet_speed, nu, mp, turb))

    # ── 4. Courant number (transient solvers only) ───────────────────────────
    if is_transient:
        checks.append(_check_cfl(inlet_speed, mp))

    # ── 5. Mesh resolution adequacy ─────────────────────────────────────────
    checks.append(_check_resolution(mp, Re))

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Individual check functions — each returns a single result dict
# ─────────────────────────────────────────────────────────────────────────────

def _check_reynolds(Re: float, solver: str, turb: str, geometry: str) -> dict:
    """Validate flow regime and turbulence-model compatibility."""
    re_str = f"Re = {Re:.4g}"

    if Re < 1_000:
        return _r(
            "warning",
            f"Laminar regime ({re_str})",
            f"Re < 1 000: the flow is laminar. A RANS turbulence model will "
            "over-predict diffusion. Consider switching turbulenceProperties "
            "to simulationType laminar.",
        )

    if Re < 4_000:
        return _r(
            "warning",
            f"Transitional regime ({re_str})",
            f"Re is between 1 000 and 4 000: RANS models are unreliable in the "
            "transitional regime. Treat results qualitatively.",
        )

    # Turbulent — check model suitability for separation-prone geometries
    if (
        turb == "kEpsilon"
        and geometry in ("backstep", "rectangular_obstacle")
        and Re > 1e4
    ):
        return _r(
            "info",
            f"Turbulent — model tip ({re_str})",
            f"k-ε tends to overpredict turbulent mixing near flow separation "
            f"({geometry} geometry). k-ω SST usually gives better results here.",
        )

    return _r(
        "ok",
        f"Turbulent regime ({re_str})",
        f"Re is well inside the turbulent range and {turb} is an appropriate model.",
    )


def _check_mach(inlet_speed: float, solver: str) -> dict:
    """Check whether compressibility effects are significant."""
    Ma = inlet_speed / _C_SOUND_AIR
    ma_str = f"Ma = {Ma:.3f}"

    if solver in ("simpleFoam", "pimpleFoam") and Ma > 0.3:
        return _r(
            "error",
            f"Compressible flow ({ma_str})",
            f"Ma > 0.3 but solver is incompressible ({solver}). "
            "Density variations will be significant — switch to rhoSimpleFoam.",
        )

    if solver in ("simpleFoam", "pimpleFoam") and Ma > 0.1:
        err_pct = Ma ** 2 / 2 * 100
        return _r(
            "info",
            f"Mild compressibility ({ma_str})",
            f"Ma is between 0.1 and 0.3: incompressible assumption introduces "
            f"≈ {err_pct:.1f} % density error. Results are still usable but "
            "consider rhoSimpleFoam for higher accuracy.",
        )

    if solver == "rhoSimpleFoam" and Ma < 0.1:
        return _r(
            "info",
            f"Low Mach number ({ma_str})",
            "Ma < 0.1 with rhoSimpleFoam: a compressible solver is unnecessary "
            "here. simpleFoam converges faster at low Mach numbers.",
        )

    return _r(
        "ok",
        f"Mach number OK ({ma_str})",
        f"Ma = {Ma:.3f} < 0.3: incompressible assumption is valid.",
    )


def _check_yplus(
    Re: float,
    inlet_speed: float,
    nu: float,
    mp: dict,
    turb: str,
) -> dict:
    """
    Estimate the wall y⁺ of the first mesh cell and compare with the
    valid range for the chosen wall function.

    Uses the Blasius turbulent flat-plate Cf correlation and assumes
    uniform mesh spacing (no grading).
    """
    domain_y = mp["domain_y"]
    ny       = mp["ny"]

    # First-cell height (uniform spacing — generators default to simpleGrading 1 1 1)
    y1 = domain_y / ny

    # Friction velocity u_τ = U × sqrt(Cf / 2)
    Cf    = _CF_A * (max(Re, 1_000) ** _CF_B)
    u_tau = inlet_speed * math.sqrt(max(Cf, 0.0) / 2.0)

    yplus = u_tau * y1 / nu if nu > 0 else 0.0
    label = f"y⁺ ≈ {yplus:.0f}"

    if turb == "SpalartAllmaras":
        # nutUSpaldingWallFunction: Spalding universal law, valid y⁺ = 1–300
        if yplus > 300:
            return _r(
                "warning",
                f"Wall resolution ({label})",
                f"y⁺ ≈ {yplus:.0f} exceeds 300: nutUSpaldingWallFunction (SA) "
                "becomes inaccurate. Refine the near-wall mesh (add grading or "
                "increase ny).",
            )
        return _r(
            "ok",
            f"Wall resolution ({label})",
            f"y⁺ ≈ {yplus:.0f}: within the valid range for "
            "nutUSpaldingWallFunction (1–300).",
        )

    else:
        # nutkWallFunction (kEpsilon, kOmegaSST): log-law, valid y⁺ = 30–300
        if yplus < 5:
            return _r(
                "info",
                f"Wall resolution ({label})",
                f"y⁺ ≈ {yplus:.0f}: the near-wall mesh is very fine. Standard "
                "wall functions target y⁺ = 30–300. Consider coarsening near "
                "walls or switching to a low-Re turbulence model.",
            )
        if yplus < 30:
            return _r(
                "warning",
                f"Wall resolution ({label})",
                f"y⁺ ≈ {yplus:.0f}: falls in the buffer layer (5 < y⁺ < 30). "
                "nutkWallFunction is inaccurate here. Refine to y⁺ < 5 "
                "(resolve sublayer) or coarsen to y⁺ > 30 (wall function regime).",
            )
        if yplus > 300:
            return _r(
                "warning",
                f"Wall resolution ({label})",
                f"y⁺ ≈ {yplus:.0f}: exceeds the valid range for nutkWallFunction "
                "(30–300). Uniform mesh spacing may be too coarse near walls. "
                "Add wall grading or increase ny.",
            )
        return _r(
            "ok",
            f"Wall resolution ({label})",
            f"y⁺ ≈ {yplus:.0f}: within the valid range for nutkWallFunction (30–300).",
        )


def _check_cfl(inlet_speed: float, mp: dict) -> dict:
    """
    Estimate the Courant number using the same formula as PimpleFoamGenerator:
        Δt = 0.4 × Δx / U_inlet   →   CFL ≈ 0.4

    Because the generator derives Δt from a fixed CFL target, the CFL is
    always close to 0.4. This check makes the expected Δt visible to the user.
    """
    domain_x  = mp["domain_x"]
    nx        = mp["nx"]
    cell_dx   = domain_x / nx
    cfl_target = 0.4
    delta_t   = cfl_target * cell_dx / max(inlet_speed, 1e-9)
    cfl_est   = inlet_speed * delta_t / cell_dx

    return _r(
        "ok",
        f"Courant number (CFL ≈ {cfl_est:.2f})",
        f"Time step Δt ≈ {delta_t:.3g} s targets CFL ≈ {cfl_target}. "
        "PIMPLE outer correctors allow CFL > 1 — adjust nOuterCorrectors in "
        "system/fvSolution if needed.",
    )


def _check_resolution(mp: dict, Re) -> dict:
    """
    Estimate the number of mesh cells covering the boundary layer and flag
    if the resolution is likely inadequate.
    """
    nx     = mp["nx"]
    ny     = mp["ny"]
    ncells = nx * ny          # 2-D case: nz = 1
    label  = f"~{ncells:,} cells (2-D)"

    if Re is not None:
        Re_f = float(Re)
        # Turbulent BL thickness (Prandtl 1/7-power):  δ/L ≈ 0.37 Re^(-1/5)
        bl_frac      = 0.37 * (max(Re_f, 1_000) ** (-0.2))
        cells_in_bl  = bl_frac * ny
        if cells_in_bl < 3:
            return _r(
                "info",
                label,
                f"Estimated boundary layer spans ≈ {cells_in_bl:.1f} cells. "
                "Increase ny or add near-wall grading for better resolution.",
            )

    if ncells < 5_000:
        return _r(
            "info",
            label,
            f"Cell count ({ncells:,}) is modest. Results may lack detail in "
            "high-gradient regions — consider increasing nx or ny.",
        )

    if ncells > 500_000:
        return _r(
            "info",
            label,
            f"~{ncells:,} cells: a 2-D run of this size may be slow. "
            "Consider running in parallel (decomposePar + mpirun -np N).",
        )

    return _r(
        "ok",
        label,
        f"Mesh size ({ncells:,} cells) is appropriate for a 2-D RANS case.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _r(level: str, title: str, detail: str) -> dict:
    """Construct a check result dict."""
    return {"level": level, "title": title, "detail": detail}


def _get_inlet_speed(bcs: list) -> float:
    """Extract the scalar inlet speed from the boundary conditions list."""
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


def _is_gas(rho: float, nu: float, fluid_name: str) -> bool:
    """Return True if the fluid is likely a gas rather than a liquid."""
    liquid_keywords = ("water", "liquid", "oil", "diesel", "petrol", "glycol",
                       "ethanol", "mercury", "glycerin")
    if any(w in fluid_name for w in liquid_keywords):
        return False
    # Liquids have density >> 50 kg/m³
    if rho > 50.0:
        return False
    return True


def _estimate_mesh(analysis: dict, char_len: float) -> dict:
    """
    Replicate the generator's default mesh-sizing logic to get approximate
    domain dimensions and cell counts for use in validation checks.

    Follows the same formulas as SimpleFoamGenerator._build_context().
    """
    mesh = analysis.get("suggested_mesh") or {}

    domain_x = float(mesh.get("domain_x") or 20.0 * char_len)
    domain_y = float(mesh.get("domain_y") or 10.0 * char_len)
    nx       = int(mesh.get("nx")       or max(50, int(domain_x / char_len * 10)))
    ny       = int(mesh.get("ny")       or max(25, int(domain_y / char_len * 10)))

    return {"domain_x": domain_x, "domain_y": domain_y, "nx": nx, "ny": ny}
