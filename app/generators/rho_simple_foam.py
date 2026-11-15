"""
RhoSimpleFoam case generator.

Generates a steady-state compressible RANS case (ideal gas, heRhoThermo).
Suitable for subsonic/transonic flows (Ma > ~0.3) where density variation matters.
Extra files vs simpleFoam: 0/T, 0/alphat, constant/thermophysicalProperties.
"""

import math
from pathlib import Path

from .base import BaseCaseGenerator


class RhoSimpleFoamGenerator(BaseCaseGenerator):
    """Generate a complete rhoSimpleFoam case directory."""

    solver_name = "rhoSimpleFoam"
    default_geometry_type = "channel"

    def generate(self, analysis: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._build_context(analysis)

        # --- 0/ field files ---
        turb = ctx["turb_model"]
        for fname in ("U", "p", "T", "alphat"):
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
        self.write_mesh(output_dir, ctx, analysis)

        # --- constant/ ---
        self.write(output_dir, "constant/thermophysicalProperties",
                   self.render("constant/thermophysicalProperties", ctx))
        self.write(output_dir, "constant/turbulenceProperties",
                   self.render("constant/turbulenceProperties", ctx))
        self.write(output_dir, "constant/fvOptions", self.render("constant/fvOptions", ctx))

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
        inlet_speed = ctx["inlet_speed"]

        # Freestream temperature and pressure (LLM may return null — fall back to defaults)
        T_inlet = float(analysis.get("T_ref") or 300.0)
        p_outlet = float(analysis.get("p_ref") or 101325.0)

        # Compute rho from ideal-gas EOS rather than trusting fluid.rho (often null or wrong units)
        rho = p_outlet / (self._R_AIR * max(T_inlet, 200.0))
        mu = nu * rho

        # Turbulence quantities: 1 % turbulence intensity, mixing length = 7 % chord
        I = 0.01
        k_inlet = 1.5 * (inlet_speed * I) ** 2
        L_mix = char_len * 0.07
        eps_inlet = (0.09 ** 0.75) * (k_inlet ** 1.5) / max(L_mix, 1e-8)

        # Domain: 30L × 20L for external aero (wide far-field)
        domain_x = round(char_len * 30, 4)
        domain_y = round(char_len * 20, 4)
        domain_z = round(char_len * 0.1, 4)
        nx = max(60, int(char_len * 30 / char_len * 2))
        ny = max(40, int(char_len * 20 / char_len * 2))
        nz = 1
        domain_x, domain_y, nx, ny = self._mesh_overrides(ctx, analysis, domain_x, domain_y, nx, ny)

        omega_inlet = eps_inlet / (0.09 * max(k_inlet, 1e-30))
        nuTilda_inlet = nu * 3.0
        ctx.update(
            nu=f"{nu:.6e}",
            mu=f"{mu:.6e}",
            rho=rho,
            T_inlet=T_inlet,
            p_outlet=p_outlet,
            k_inlet=f"{k_inlet:.6e}",
            eps_inlet=f"{eps_inlet:.6e}",
            omega_inlet=f"{omega_inlet:.6e}",
            nuTilda_inlet=f"{nuTilda_inlet:.6e}",
            domain_x=domain_x,
            domain_y=domain_y,
            domain_z=domain_z,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        self._build_geometry_extra(ctx["geometry_type"], analysis, ctx)
        return ctx
