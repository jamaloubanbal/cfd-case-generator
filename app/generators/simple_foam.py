"""
SimpleFoam case generator.

Generates a steady-state incompressible OpenFOAM case using the SIMPLE
algorithm (simpleFoam solver).
"""

import math
from pathlib import Path

from .base import BaseCaseGenerator


class SimpleFoamGenerator(BaseCaseGenerator):
    """Generate a complete simpleFoam case directory."""

    solver_name = "simpleFoam"
    default_geometry_type = "channel"

    def generate(self, analysis: dict, output_dir: Path) -> None:
        """
        Write all case files into *output_dir*.

        Parameters
        ----------
        analysis : dict
            Structured CFD analysis from Claude.
        output_dir : Path
            Root of the case folder to create.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._build_context(analysis)

        # --- 0/ field files ---
        turb = ctx["turb_model"]
        for fname in ("U", "p"):
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
        self.write(output_dir, "constant/transportProperties",  self.render("constant/transportProperties",  ctx))
        self.write(output_dir, "constant/turbulenceProperties", self.render("constant/turbulenceProperties", ctx))

        # --- run scripts + README ---
        self.write_run_scripts(output_dir, ctx)

        # --- post-processing dicts ---
        self.write_postprocessing_files(output_dir, ctx)

        # --- snappyHexMesh files (only when geometry_type == 'stl_import') ---
        if ctx.get("geometry_type") == "stl_import":
            self.write_snappy_files(output_dir, ctx, analysis)

    # ------------------------------------------------------------------

    def _build_context(self, analysis: dict) -> dict:
        ctx = self._base_context(analysis)

        fluid = analysis.get("fluid", {})
        nu = float(fluid.get("nu") or 1.5e-5)
        char_len = analysis.get("characteristic_length", 1.0)
        inlet_speed = ctx["inlet_speed"]

        # Turbulence quantities
        I = 0.05  # 5 % turbulence intensity
        k_inlet = 1.5 * (inlet_speed * I) ** 2
        L_mix = char_len * 0.07
        eps_inlet = (0.09 ** 0.75) * (k_inlet ** 1.5) / L_mix

        # Domain sizing: 20L × 10L × 0.1L
        domain_x = round(20.0 * char_len, 4)
        domain_y = round(10.0 * char_len, 4)
        domain_z = round(0.1  * char_len, 4)

        # Cell count: target ~20 000 cells in 2D
        nx = max(50, int(domain_x / char_len * 10))
        ny = max(25, int(domain_y / char_len * 10))
        nz = 1

        omega_inlet = eps_inlet / (0.09 * max(k_inlet, 1e-30))
        nuTilda_inlet = nu * 3.0
        ctx.update(
            nu=f"{nu:.6e}",
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
