"""
BuoyantSimpleFoam case generator.

Generates a steady-state buoyancy-driven flow case (natural convection).
Extra files: 0/T, 0/alphat, constant/radiationProperties, constant/g.
"""

import math
from pathlib import Path

from .base import BaseCaseGenerator


class BuoyantSimpleFoamGenerator(BaseCaseGenerator):
    """Generate a complete buoyantSimpleFoam case directory."""

    solver_name = "buoyantSimpleFoam"
    default_geometry_type = "cavity"

    def generate(self, analysis: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._build_context(analysis)

        # --- 0/ field files ---
        turb = ctx["turb_model"]
        for fname in ("U", "p", "p_rgh", "T", "alphat"):
            self.write(output_dir, f"0/{fname}", self.render(f"0/{fname}", ctx))
        if turb in ("kEpsilon", "kOmegaSST"):
            self.write(output_dir, "0/k", self.render("0/k", ctx))
        if turb == "kEpsilon":
            self.write(output_dir, "0/epsilon", self.render("0/epsilon", ctx))
        elif turb == "kOmegaSST":
            self.write(output_dir, "0/omega", self.render("_shared/0/omega", ctx))
        elif turb == "SpalartAllmaras":
            self.write(output_dir, "0/nuTilda", self.render("_shared/0/nuTilda", ctx))
        self.write(output_dir, "0/nut", self.render("0/nut", ctx))

        # --- system/ ---
        self.write(output_dir, "system/controlDict",   self.render("system/controlDict",   ctx))
        self.write(output_dir, "system/fvSchemes",     self.render("system/fvSchemes",     ctx))
        self.write(output_dir, "system/fvSolution",    self.render("system/fvSolution",    ctx))
        self.write(output_dir, "system/blockMeshDict", self.render_geometry_block_mesh(ctx["geometry_type"], ctx))

        # --- constant/ ---
        self.write(output_dir, "constant/thermophysicalProperties",   self.render("constant/thermophysicalProperties",   ctx))
        self.write(output_dir, "constant/turbulenceProperties", self.render("constant/turbulenceProperties", ctx))
        self.write(output_dir, "constant/radiationProperties",  self.render("constant/radiationProperties",  ctx))
        self.write(output_dir, "constant/g",                    self.render("constant/g",                    ctx))

        # --- run scripts + README ---
        self.write_run_scripts(output_dir, ctx)

        # --- post-processing dicts ---
        self.write_postprocessing_files(output_dir, ctx)

        # --- snappyHexMesh files (only when geometry_type == 'stl_import') ---
        if ctx.get("geometry_type") == "stl_import":
            self.write_snappy_files(output_dir, ctx, analysis)

    # ------------------------------------------------------------------

    # Specific gas constant for air: R0 / mol_weight = 8314 / 28.9 ≈ 287.7 J/(kg·K)
    _R_AIR = 8314.0 / 28.9

    def _build_context(self, analysis: dict) -> dict:
        ctx = self._base_context(analysis)

        fluid = analysis.get("fluid", {})
        nu = float(fluid.get("nu") or 1.5e-5)

        char_len = analysis.get("characteristic_length", 1.0)
        char_vel = analysis.get("characteristic_velocity", 0.1)
        inlet_speed = ctx["inlet_speed"]

        # Extract temperatures first — needed for ideal-gas rho computation below
        T_hot, T_cold, T_initial = self._extract_temperatures(analysis)

        # Compute mu from the ideal-gas EOS (p = rho*R*T) rather than trusting
        # fluid.rho from the LLM, which is often in wrong units (e.g. g/cm³).
        # rho_ref = p_atm / (R_air * T_initial)
        rho = 101325.0 / (self._R_AIR * max(T_initial, 200.0))
        mu = nu * rho

        # Turbulence quantities (use buoyant velocity scale if U ~ 0)
        U_scale = max(inlet_speed, 0.01)
        I = 0.05
        k_inlet = 1.5 * (U_scale * I) ** 2
        L_mix = char_len * 0.07
        eps_inlet = (0.09 ** 0.75) * (k_inlet ** 1.5) / max(L_mix, 1e-8)

        # Domain sizing: square cavity (equal width and height)
        domain_x = round(char_len, 4)
        domain_y = round(char_len, 4)
        domain_z = round(char_len * 0.1, 4)
        nx = max(20, int(char_len / char_len * 50))
        ny = max(20, int(char_len / char_len * 50))
        nz = 1

        omega_inlet = eps_inlet / (0.09 * max(k_inlet, 1e-30))
        nuTilda_inlet = nu * 3.0
        ctx.update(
            nu=f"{nu:.6e}",
            mu=f"{mu:.6e}",
            rho=rho,
            p_ref=101325,
            k_inlet=f"{k_inlet:.6e}",
            eps_inlet=f"{eps_inlet:.6e}",
            omega_inlet=f"{omega_inlet:.6e}",
            nuTilda_inlet=f"{nuTilda_inlet:.6e}",
            T_hot=T_hot,
            T_cold=T_cold,
            T_initial=T_initial,
            domain_x=domain_x,
            domain_y=domain_y,
            domain_z=domain_z,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        self._build_geometry_extra(ctx["geometry_type"], analysis, ctx)
        return ctx

    @staticmethod
    def _extract_temperatures(analysis: dict) -> tuple:
        """Return (T_hot, T_cold, T_initial) from the analysis dict."""
        T_hot = 350.0
        T_cold = 300.0

        for bc in analysis.get("boundary_conditions", []):
            name = bc.get("patch_name", "").lower()
            vals = bc.get("values", {})
            T_val = vals.get("T")
            if T_val is None:
                continue
            try:
                T_val = float(T_val)
            except (TypeError, ValueError):
                continue

            if "hot" in name or "left" in name:
                T_hot = T_val
            elif "cold" in name or "right" in name:
                T_cold = T_val
            elif "inlet" in name:
                T_cold = T_val  # treat inlet temperature as cold reference

        T_initial = 0.5 * (T_hot + T_cold)
        return T_hot, T_cold, T_initial
