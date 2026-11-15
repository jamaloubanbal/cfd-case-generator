"""
Build the README gallery: run every example in examples/ and render mesh + flow field.

    source /usr/share/openfoam/etc/bashrc
    python scripts/make_gallery.py                       # all examples (~15 min)
    python scripts/make_gallery.py --only dam_break

Writes docs/gallery/<example>.png (and .gif for transient cases) plus gallery.json.
Every image comes from a case produced by the normal generator and run with its own
Allrun; fields are read and drawn cell by cell with app.postprocess (no ParaView).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.generators import GENERATORS  # noqa: E402
from app.postprocess import FoamCase  # noqa: E402
from app.postprocess.render import plot_cells  # noqa: E402
from app.sanitize import sanitize_analysis  # noqa: E402

# Run length, quantity to show and view window per example.
#   end_time: iterations (steady) or seconds (transient)
#   frames:   (interval, count) — write this many evenly spaced results for an animation
GALLERY: dict[str, dict] = {
    "cylinder_vortex_shedding": dict(
        end_time=320.0, n_cores=6, frames=(1.0, 40), quantity="Uy_rel", cmap="RdBu_r",
        sym=True, label="u_y / U∞", mesh_zoom=(0.06, 0.18, 0.09, 0.15),
    ),
    "dam_break": dict(
        end_time=0.8, frames=(0.02, 40), quantity="alpha.water", cmap="Blues",
        label="water volume fraction α", vmin=0.0, vmax=1.0, mesh_zoom=(0.0, 0.3, 0.0, 0.15),
    ),
    "backward_facing_step": dict(
        end_time=3000, refine=2.0, quantity="Ux_rel", cmap="RdBu_r", label="u_x / U_inlet", sym_center=0.0,
        view=(0.2, 1.6, 0.0, 0.2), mesh_zoom=(0.3, 0.6, 0.0, 0.2), contour_zero="Ux",
    ),
    "t_junction_mixing": dict(
        end_time=3000, refine=2.0, quantity="U_mag", cmap="viridis", label="|U| [m/s]",
        view=(0.2, 0.8, 0.0, 0.25), mesh_zoom=(0.4, 0.6, 0.0, 0.1),
    ),
    "compressible_backstep": dict(
        end_time=3000, refine=1.5, quantity="Ma", cmap="magma", label="Mach number",
        view=(0.2, 1.6, 0.0, 0.2), mesh_zoom=(0.3, 0.6, 0.0, 0.2),
    ),
}


def _quantity(case: FoamCase, t: str, q: str, analysis: dict) -> np.ndarray:
    bcs = analysis["boundary_conditions"]
    u_ref = max(np.linalg.norm(bc["values"]["U"]) for bc in bcs if "U" in bc["values"]) or 1.0
    if q == "alpha.water":
        return case.field("alpha.water", t)
    U = case.field("U", t)
    if q == "U_mag":
        return np.linalg.norm(U, axis=1)
    if q == "Ux_rel":
        return U[:, 0] / u_ref
    if q == "Uy_rel":
        return U[:, 1] / u_ref
    if q == "Cp":
        return case.field("p", t) / (0.5 * u_ref**2)       # kinematic p, p∞ = 0 at the outlet
    if q == "Ma":
        return np.linalg.norm(U, axis=1) / np.sqrt(1.4 * 287.0 * case.field("T", t))
    raise ValueError(q)


def _prepare(name: str, cfg: dict, work: Path) -> tuple[Path, dict]:
    data = json.loads((REPO / "examples" / f"{name}.json").read_text())
    analysis, _ = sanitize_analysis(data["analysis"])
    analysis["n_cores"] = cfg.get("n_cores", 1)
    if cfg.get("refine"):                      # the parameter editor's mesh refinement factor
        analysis.setdefault("suggested_mesh", {})["refinement"] = cfg["refine"]
    case_dir = work / name
    shutil.rmtree(case_dir, ignore_errors=True)
    GENERATORS[analysis["solver"]]().generate(analysis, case_dir)

    cd = case_dir / "system" / "controlDict"
    text = cd.read_text()
    end = cfg["end_time"]
    text = re.sub(r"^endTime\s+[^;]+;", f"endTime         {end:g};", text, flags=re.M)
    interval, keep = cfg.get("frames", (end, 1))
    text = re.sub(r"^writeControl\s+[^;]+;", "writeControl    adjustableRunTime;" if "frames" in cfg
                  else "writeControl    runTime;", text, flags=re.M)
    text = re.sub(r"^writeInterval\s+[^;]+;", f"writeInterval   {interval:g};", text, flags=re.M)
    text = re.sub(r"^purgeWrite\s+[^;]+;", f"purgeWrite      {keep if 'frames' in cfg else 2};", text, flags=re.M)
    cd.write_text(text)
    return case_dir, analysis


def _draw(ax_mesh, ax_field, case, t, cfg, analysis, title=None):
    vals = _quantity(case, t, cfg["quantity"], analysis)
    vmin, vmax = cfg.get("vmin"), cfg.get("vmax")
    if cfg.get("sym") or "sym_center" in cfg:
        lim = np.percentile(np.abs(vals - cfg.get("sym_center", 0.0)), 99.5)
        vmin, vmax = -lim, lim
    plot_cells(ax_field, case, vals, cmap=cfg["cmap"], vmin=vmin, vmax=vmax, label=cfg["label"])
    if cfg.get("contour_zero"):
        cc = case.cell_centres
        Ux = case.field("U", t)[:, 0]
        ax_field.tricontour(cc[:, 0], cc[:, 1], Ux, levels=[0.0], colors="k", linewidths=0.8)
    if "view" in cfg:
        x0, x1, y0, y1 = cfg["view"]
        ax_field.set_xlim(x0, x1); ax_field.set_ylim(y0, y1)
    ax_field.set_title(title or cfg["label"], fontsize=10)
    if ax_mesh is not None:
        plot_cells(ax_mesh, case, None, edge_width=0.25)
        x0, x1, y0, y1 = cfg["mesh_zoom"]
        ax_mesh.set_xlim(x0, x1); ax_mesh.set_ylim(y0, y1)
        ax_mesh.set_title("Generated mesh (detail)", fontsize=10)
    for ax in (ax_mesh, ax_field):
        if ax is not None:
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def build(name: str, cfg: dict, work: Path, out: Path) -> dict:
    case_dir, analysis = _prepare(name, cfg, work)
    t0 = time.time()
    proc = subprocess.run(["./Allrun"], cwd=case_dir, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: Allrun failed\n{proc.stdout}\n{proc.stderr}")
    case = FoamCase(case_dir)
    t = case.latest_time()

    heading = (f"{analysis['solver']} · {analysis['geometry_type'].replace('_', ' ')} · "
               f"{analysis['turb_model']} · {case.n_cells:,} cells")
    fig, (am, af) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    _draw(am, af, case, t, cfg, analysis, title=f"{cfg['label']} at t = {float(t):g}"
          if analysis["is_transient"] else cfg["label"])
    fig.suptitle(heading, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=120)
    plt.close(fig)

    if "frames" in cfg:
        _animate(name, case, cfg, analysis, out)
    return {"solver": analysis["solver"], "geometry": analysis["geometry_type"],
            "turbulence": analysis["turb_model"], "cells": case.n_cells,
            "mesh_refinement": cfg.get("refine", 1.0),
            "run_time_s": round(elapsed, 1), "cores": cfg.get("n_cores", 1)}


def _animate(name, case, cfg, analysis, out):
    from matplotlib.animation import FuncAnimation, PillowWriter

    times = [t for t in case.times() if float(t) > 0][-cfg["frames"][1]:]
    fig, ax = plt.subplots(figsize=(9, 3.6) if name != "dam_break" else (8, 4.4))
    vals_all = [_quantity(case, t, cfg["quantity"], analysis) for t in times]
    if cfg.get("sym"):
        lim = np.percentile(np.abs(np.concatenate(vals_all)), 99.5)
        cfg = {**cfg, "vmin": -lim, "vmax": lim, "sym": False}
    coll = plot_cells(ax, case, vals_all[0], cmap=cfg["cmap"], vmin=cfg.get("vmin"),
                      vmax=cfg.get("vmax"), label=cfg["label"])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    from app.postprocess.render import front_polygons
    _, cells = front_polygons(case)

    def update(i):
        coll.set_array(vals_all[i][cells])
        ax.set_title(f"{analysis['solver']} — t = {float(times[i]):.2f} s", fontsize=10)
        return (coll,)

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    fig.tight_layout()
    anim.save(out / f"{name}.gif", writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(GALLERY))
    ap.add_argument("--out", default=str(REPO / "docs" / "gallery"))
    ap.add_argument("--workdir")
    args = ap.parse_args()
    if not shutil.which("blockMesh"):
        print("OpenFOAM not found: source its etc/bashrc first.", file=sys.stderr)
        return 2
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory() if not args.workdir else None
    work = Path(args.workdir or tmp.name); work.mkdir(parents=True, exist_ok=True)

    index_path = out / "gallery.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for name in args.only or list(GALLERY):
        print(f"▶ {name} …", flush=True)
        index[name] = build(name, GALLERY[name], work, out)
        print(f"   {index[name]['cells']:,} cells, {index[name]['run_time_s']:.0f} s", flush=True)
        index_path.write_text(json.dumps(index, indent=2) + "\n")   # saved after every case
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
