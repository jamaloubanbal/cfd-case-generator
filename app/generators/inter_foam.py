"""
InterFoam case generator.

Generates a two-phase VOF case (water/air) using interFoam.
Extra files: 0/alpha.water, constant/transportProperties (two-phase),
constant/g.
"""

import math
from pathlib import Path

from .base import BaseCaseGenerator
from app.naming import is_inlet_name


class InterFoamGenerator(BaseCaseGenerator):
    """Generate a complete interFoam case directory."""

    solver_name = "interFoam"
    default_geometry_type = "dambreak"

    def generate(self, analysis: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._build_context(analysis)

        # --- 0/ field files ---
        for fname in ("U", "p_rgh", "alpha.water"):
            self.write(output_dir, f"0/{fname}", self.render(f"0/{fname}", ctx))

        # --- system/ ---
        self.write(output_dir, "system/controlDict",    self.render("system/controlDict",    ctx))
        self.write(output_dir, "system/fvSchemes",      self.render("system/fvSchemes",      ctx))
        self.write(output_dir, "system/fvSolution",     self.render("system/fvSolution",     ctx))
        self.write_mesh(output_dir, ctx, analysis)
        self.write(output_dir, "system/setFieldsDict",  self.render("system/setFieldsDict",  ctx))

        # --- constant/ ---
        self.write(output_dir, "constant/transportProperties",  self.render("constant/transportProperties",  ctx))
        self.write(output_dir, "constant/turbulenceProperties", self.render("constant/turbulenceProperties", ctx))
        self.write(output_dir, "constant/g",                    self.render("constant/g",                    ctx))

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
        nu = float(fluid.get("nu") or 1.0e-6)      # water default
        rho = float(fluid.get("rho") or 1000.0)    # water default

        char_len = analysis.get("characteristic_length", 0.3)
        inlet_speed = ctx["inlet_speed"]

        # Domain sizing: typical dam-break basin
        domain_x = round(max(char_len * 4.0, 1.0), 4)
        domain_y = round(max(char_len * 2.0, 0.4), 4)
        domain_z = round(0.05, 4)  # thin 2D slab

        # 1 cm cells, but at most 160 across the basin: fixed 1 cm cells gave 80 000
        # cells for a 1 m column (4 m basin) and runs over an hour. The OpenFOAM
        # damBreak tutorial uses ~2 300 cells (its finer variant ~9 000).
        nx = min(160, max(100, int(domain_x / 0.01)))
        ny = max(40, round(nx * domain_y / domain_x))
        nz = 1
        domain_x, domain_y, nx, ny = self._mesh_overrides(ctx, analysis, domain_x, domain_y, nx, ny)

        # Initial water column (can be overridden via geometry_params);
        # clamped so it always lies inside the basin.
        geo_params = analysis.get("geometry_params") or {}
        x_water = float(geo_params.get("x_water") or domain_x * 0.3)
        y_water = float(geo_params.get("y_water") or domain_y * 0.5)
        x_water = round(min(max(x_water, 0.05 * domain_x), 0.9 * domain_x), 4)
        y_water = round(min(max(y_water, 0.05 * domain_y), 0.9 * domain_y), 4)

        # Time-stepping: gravity wave speed ~ sqrt(g*h)
        g = 9.81
        wave_speed = math.sqrt(g * char_len)
        cell_size = domain_x / nx
        delta_t = self._round_sig(0.3 * cell_size / max(wave_speed, 1e-3), 2)
        end_time = self._round_sig(10.0 * math.sqrt(char_len / g), 2)
        write_interval = max(1, int(end_time / delta_t / 50))

        # Determine alpha.water at inlet
        # Default: pure water inlet (dam-break starts with water in place)
        alpha_water_inlet = 1.0
        for bc in analysis.get("boundary_conditions", []):
            if is_inlet_name(bc.get("patch_name", "")):
                a = bc.get("values", {}).get("alpha.water")
                if a is not None:
                    try:
                        alpha_water_inlet = float(a)
                    except (TypeError, ValueError):
                        pass

        ctx.update(
            nu_water=f"{nu:.6e}",
            rho_water=rho,
            alpha_water_inlet=alpha_water_inlet,
            domain_x=domain_x,
            domain_y=domain_y,
            domain_z=domain_z,
            nx=nx,
            ny=ny,
            nz=nz,
            delta_t=f"{delta_t:.6g}",
            end_time=f"{end_time:.6g}",
            write_interval=write_interval,
            x_water=x_water,
            y_water=y_water,
        )
        self._build_geometry_extra(ctx["geometry_type"], analysis, ctx)
        return ctx

    @staticmethod
    def _round_sig(x: float, sig: int) -> float:
        if x == 0:
            return 0.0
        magnitude = math.floor(math.log10(abs(x)))
        factor = 10 ** (sig - 1 - magnitude)
        return round(x * factor) / factor
