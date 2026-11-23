"""
code_saturne backend: SALOME meshes and the built-in geometries, steady incompressible flow.

The same sanitised analysis that drives the OpenFOAM generators becomes a code_saturne
study:

    out/                     study directory
    ├── MESH/<mesh>.unv      the SALOME mesh, read natively (groups → boundary zones)
    └── CASE/DATA/setup.xml  written through code_saturne's own GUI model layer, so the
                             file is always valid for the installed version

``generate`` writes the study, ``run`` executes it (``code_saturne run``) and ``review``
judges the result from code_saturne's outputs: residuals.csv, the boundary mass-flow table
and the field ranges in run_solver.log.

Numerics depend on the mesh. SALOME tetrahedra: steady SIMPLE with first-order upwind for
velocity (centred and SOLU oscillated without converging; on a 66 600-cell duct the core
velocity agreed with OpenFOAM within 1 %). Block-structured hexahedra: second-order SOLU
with a local time step, because upwind's numerical diffusion halved the recirculation
length behind the Re 40 cylinder. The velocity slope test (on by default) is switched off on
these meshes: it locally falls back to first-order upwind wherever the solution is
non-monotonic, i.e. in a vortex street, and lowered the Re 100 Strouhal number by 12 %
(0.1455 vs 0.1652 without it; OpenFOAM 0.1653).

All code_saturne specifics live in this module: the installed 9.2 is a development
version, and its Python model API can change between releases.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

CASE = "CASE"
RUN_ID = "run"
DEFAULT_ITERATIONS = 2000      # the 2-D channel needed ~1 500 to bring pressure below 1e-3
TURBULENCE = {"laminar": "off", "kEpsilon": "k-epsilon", "kOmegaSST": "k-omega-SST",
              "SpalartAllmaras": "Spalart-Allmaras"}
# Boundary roles of the mesh import → code_saturne zone natures
# (2-D front/back planes and the pipe's wedge sides become symmetry, as usual in code_saturne)
NATURE = {"inlet": "inlet", "outlet": "outlet", "wall": "wall", "symmetry": "symmetry",
          "frontAndBack": "symmetry", "wedge": "symmetry", "obstacle": "wall",
          "hot_wall": "wall", "cold_wall": "wall"}
# Built-in geometries whose blockMesh is converted (steady forced flow). Not the pipe: its
# 5° wedge has no counterpart in code_saturne (symmetry sides and the degenerate prisms at
# the axis made the run diverge).
GEOMETRIES = ("channel", "t_junction", "rectangular_obstacle", "backstep", "cylinder", "airfoil")
# Words of code_saturne's selection syntax: a zone (group) cannot be named like them
RESERVED = {"all", "and", "or", "not", "range", "normal", "plane", "box", "cylinder", "sphere",
            "inside", "outside", "x", "y", "z", "group"}


def zone_name(group: str) -> str:
    return f"{group}_zone" if group.lower() in RESERVED else group


# ── Installation ─────────────────────────────────────────────────────────────

def find_executable() -> Path | None:
    """code_saturne launcher: $CODE_SATURNE, then PATH, then ~/code_saturne/*/…/bin."""
    for cand in (os.environ.get("CODE_SATURNE"), shutil.which("code_saturne")):
        if cand and Path(cand).is_file():
            return Path(cand)
    found = sorted(glob.glob(str(Path.home() / "code_saturne" / "**" / "bin" / "code_saturne"),
                             recursive=True))
    return Path(found[-1]) if found else None


def _site_packages(exe: Path) -> str:
    found = glob.glob(str(exe.parent.parent / "lib" / "python3*" / "site-packages"))
    if not found:
        raise RuntimeError(f"code_saturne's Python package not found next to {exe}")
    return found[0]


def _python(exe: Path) -> str:
    out = subprocess.run([str(exe), "config", "--python"], capture_output=True, text=True)
    return out.stdout.strip() or "python3"


# ── Case generation ──────────────────────────────────────────────────────────

# Runs in code_saturne's Python with its package on PYTHONPATH; reads the spec (JSON).
_BUILDER = r'''
import json, sys
spec = json.load(open(sys.argv[1]))
from code_saturne.model.XMLengine import Case
from code_saturne.model.XMLinitialize import XMLinit
from code_saturne.model.SolutionDomainModel import SolutionDomainModel
from code_saturne.model.TurbulenceModel import TurbulenceModel
from code_saturne.model.FluidCharacteristicsModel import FluidCharacteristicsModel
from code_saturne.model.LocalizationModel import LocalizationModel, Zone
from code_saturne.model.Boundary import Boundary
from code_saturne.model.TimeStepModel import TimeStepModel
from code_saturne.model.NumericalParamEquationModel import NumericalParamEquationModel

case = Case(file_name=spec["xml"], module="code_saturne")
XMLinit(case).initialize()
case["xmlfile"] = spec["xml"]
SolutionDomainModel(case).addMesh((spec["mesh"], None))
TurbulenceModel(case).setTurbulenceModel(spec["turbulence"])
fluid = FluidCharacteristicsModel(case)
fluid.setInitialValueDensity(spec["rho"])
fluid.setInitialValueViscosity(spec["mu"])
zones = LocalizationModel("BoundaryZone", case)
for i, z in enumerate(spec["zones"], 1):
    zones.addZone(Zone("BoundaryZone", case=case, label=z["label"], codeNumber=i,
                       localization=z["label"], nature=z["nature"]))
for z in spec["zones"]:
    if z["nature"] == "inlet":
        b = Boundary("inlet", z["label"], case)
        b.setVelocityChoice("norm")
        b.setVelocity(z["speed"])
        if spec["turbulence"] != "off":
            b.setHydraulicDiameter(z["hydraulic_diameter"])
ts = TimeStepModel(case)
num = NumericalParamEquationModel(case)
if spec["transient"]:                           # time-accurate, adaptive (Courant-limited) step
    ts.setTimePassing(1)
    ts.setTimeStep(spec["dt"])
    ts.setMaxCourant(0.8)
    if spec["numerics"] == "upwind_steady":
        num.setBlendingFactor("velocity", 0.0)
    else:
        num.setScheme("velocity", "solu")
        num.setBlendingFactor("velocity", 1.0)
        num.setSlopeTest("velocity", "off")
    ts.setStopCriterion("maximum_time", spec["end_time"])
elif spec["numerics"] == "upwind_steady":       # SALOME tetrahedra
    ts.setTimePassing(-1)                       # steady (SIMPLE)
    num.setBlendingFactor("velocity", 0.0)      # first-order upwind
else:                                           # "solu_local": block-structured hexahedra
    ts.setTimePassing(2)                        # local time step (pseudo-transient)
    ts.setTimeStep(spec["dt"])
    ts.setMaxCourant(2.0)
    num.setScheme("velocity", "solu")           # second-order upwind
    num.setBlendingFactor("velocity", 1.0)
    num.setSlopeTest("velocity", "off")         # see the module docstring
if not spec["transient"]:
    ts.setStopCriterion("iterations", spec["iterations"])
from code_saturne.model.OutputControlModel import OutputControlModel
out = OutputControlModel(case)
# log every few steps (not code_saturne's sparse default) so the web app can follow the run
out.setListingFrequency(5 if spec["transient"] else max(1, spec["iterations"] // 400))
if spec["probes"]:
    for x, y, z in spec["probes"]:
        out.addMonitoringPoint(x, y, z)
    out.setMonitoringPointFormat("CSV")
case.xmlSaveDocument()
'''


def _patch_speed(analysis: dict, patch: str, default: float) -> float:
    """Speed of the boundary condition named like *patch* (e.g. a T-junction's inlet2)."""
    for bc in analysis.get("boundary_conditions") or []:
        U = (bc.get("values") or {}).get("U")
        if bc.get("patch_name") == patch and isinstance(U, list):
            return float(math.sqrt(sum(float(c) ** 2 for c in U)))
    return default


def _inlet_speed(analysis: dict) -> float:
    from app.generators.base import BaseCaseGenerator
    return BaseCaseGenerator._get_inlet_speed(analysis.get("boundary_conditions") or [])


def check_supported(analysis: dict) -> None:
    geo = analysis.get("geometry_type")
    if geo != "imported_mesh" and geo not in GEOMETRIES:
        raise ValueError(f"code_saturne: geometry {geo!r} is not supported "
                         f"(supported: a SALOME mesh, {', '.join(GEOMETRIES)}).")
    if analysis.get("solver") not in ("simpleFoam", "pimpleFoam"):
        kind = {"rhoSimpleFoam": "a compressible flow",
                "buoyantSimpleFoam": "natural convection", "interFoam": "a free-surface flow"}
        raise ValueError(f"This problem is {kind.get(analysis.get('solver'), analysis.get('solver'))} "
                         f"({analysis.get('solver')}); the code_saturne backend handles "
                         "incompressible flow (steady or time-dependent) only for now.")
    if analysis.get("turb_model") not in TURBULENCE:
        raise ValueError(f"code_saturne: unsupported flow model {analysis.get('turb_model')}.")


def _builtin_mesh(analysis: dict, unv: Path) -> list[dict]:
    """Mesh a built-in geometry with the OpenFOAM generator and blockMesh, write it as UNV;
    returns the patch descriptors (name, role) of the groups written."""
    import tempfile

    from app.foam_to_unv import polymesh_to_unv
    from app.generators import GENERATORS
    from app.generators.base import GEOMETRY_PATCHES
    if not shutil.which("blockMesh"):
        raise RuntimeError("built-in geometries are meshed with OpenFOAM's blockMesh: source "
                           "OpenFOAM's etc/bashrc (or use a SALOME mesh with --mesh)")
    with tempfile.TemporaryDirectory() as tmp:
        foam = Path(tmp) / "mesh"
        GENERATORS[analysis["solver"]]().generate(analysis, foam)
        r = subprocess.run(["blockMesh"], cwd=foam, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("blockMesh failed:\n" + r.stdout[-1500:])
        names = {p["name"]: zone_name(p["name"]) for p in GEOMETRY_PATCHES[analysis["geometry_type"]]}
        written = polymesh_to_unv(foam, unv, rename=names)
    return [dict(p, name=names[p["name"]], patch=p["name"])
            for p in GEOMETRY_PATCHES[analysis["geometry_type"]] if names[p["name"]] in written]


def generate(analysis: dict, unv: str | Path | None, out: Path, description: str | None = None,
             iterations: int = DEFAULT_ITERATIONS, exe: Path | None = None,
             probes: list[tuple[float, float, float]] | None = None,
             end_time: float | None = None) -> Path:
    """Write a code_saturne study in *out* for *analysis*: on the SALOME mesh *unv*, or for a
    built-in geometry (unv=None) on its blockMesh mesh converted to UNV.
    Returns the case directory."""
    check_supported(analysis)
    exe = exe or find_executable()
    if exe is None:
        raise RuntimeError("code_saturne not found: set CODE_SATURNE to its bin/code_saturne")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)       # `code_saturne create` needs the parent
    subprocess.run([str(exe), "create", "-s", str(out), "-c", CASE], check=True,
                   capture_output=True, text=True)
    if analysis.get("geometry_type") == "imported_mesh":
        unv = Path(unv)
        shutil.copy(unv, out / "MESH" / unv.name)
        patches = analysis.get("_mesh_patches") or []
        bad = [p["name"] for p in patches if zone_name(p["name"]) != p["name"]]
        if bad:
            raise ValueError(f"mesh group(s) {', '.join(bad)}: reserved words in code_saturne; "
                             "rename them in SALOME")
    else:
        unv = out / "MESH" / f"{analysis['geometry_type']}.unv"
        patches = _builtin_mesh(analysis, unv)

    fluid = analysis.get("fluid") or {}
    rho = float(fluid.get("rho") or 1.2)
    nu = float(fluid.get("nu") or 1.5e-5)
    speed = _inlet_speed(analysis)
    zones = []
    for p in patches:
        z = {"label": p["name"], "nature": NATURE.get(p["role"], "wall")}
        if z["nature"] == "inlet":
            z.update(speed=_patch_speed(analysis, p.get("patch", p["name"]), speed),
                     hydraulic_diameter=float(analysis.get("characteristic_length")
                                              or math.sqrt(p.get("area") or 0.0) or 0.1))
        zones.append(z)
    # Tetrahedra (SALOME): only first-order upwind in steady SIMPLE converged. Hexahedra
    # (blockMesh): upwind's numerical diffusion halved the Re 40 cylinder's wake (L/D 1.07 vs
    # 2.10); SOLU converged with the local time step (1.92), not in steady SIMPLE (stalled).
    numerics = "upwind_steady" if analysis.get("geometry_type") == "imported_mesh" else "solu_local"
    case_dir = out / CASE
    spec = {"xml": str(case_dir / "DATA" / "setup.xml"), "mesh": unv.name, "zones": zones,
            "turbulence": TURBULENCE[analysis["turb_model"]], "rho": rho, "mu": rho * nu,
            "iterations": int(iterations), "numerics": numerics,
            "dt": 0.1 * float(analysis.get("characteristic_length") or 1.0) / max(speed, 1e-9),
            "transient": analysis["solver"] == "pimpleFoam", "probes": [list(p) for p in probes or []]}
    if spec["transient"]:
        # same physical duration and initial step as the OpenFOAM generator
        from app.generators import GENERATORS
        ctx = GENERATORS["pimpleFoam"]()._build_context(analysis)
        spec["end_time"] = float(end_time or ctx["end_time"])
        spec["dt"] = float(ctx["delta_t"])
    spec_file = case_dir / "DATA" / "cfdgen_spec.json"
    spec_file.write_text(json.dumps(spec, indent=1))
    builder = case_dir / "DATA" / "cfdgen_build_setup.py"
    builder.write_text(_BUILDER)
    env = dict(os.environ, PYTHONPATH=_site_packages(exe))
    r = subprocess.run([_python(exe), str(builder), str(spec_file)], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("code_saturne setup.xml generation failed:\n" + r.stderr[-2000:])
    from app.review import save_request
    save_request(case_dir, analysis, description)
    return case_dir


# ── Run ──────────────────────────────────────────────────────────────────────

def run(case_dir: Path, exe: Path | None = None, timeout: int = 3600, n_procs: int = 1,
        on_process=None) -> Path:
    """Run the case (on *n_procs* MPI ranks); returns its result directory (RESU/run).
    *on_process* receives the Popen, e.g. to stop the run from the UI."""
    from app.agent import stop_process
    exe = exe or find_executable()
    resu = Path(case_dir) / "RESU" / RUN_ID
    if resu.exists():
        shutil.rmtree(resu)
    cmd = [str(exe), "run", "--id", RUN_ID] + (["-n", str(n_procs)] if n_procs > 1 else [])
    proc = subprocess.Popen(cmd, cwd=case_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    if on_process:
        on_process(proc)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_process(proc)
        raise
    return resu


# ── Results ──────────────────────────────────────────────────────────────────

def residuals(resu: Path) -> tuple[int, dict[str, float]]:
    """Last iteration and its residual per variable, from residuals.csv."""
    lines = (resu / "residuals.csv").read_text().strip().splitlines()
    names = [n.strip() for n in lines[0].split(",")]
    last = [v.strip() for v in lines[-1].split(",")]
    # first column: the iteration (steady) or the time (time-dependent); count the rows then
    n = int(last[0]) if names[0] == "iteration" else len(lines) - 1
    return n, {k: float(v) for k, v in zip(names[1:], last[1:]) if k != "wall_distance"}


def mass_flows(log: str) -> dict[str, float]:
    """Last 'BOUNDARY MASS FLOW INFORMATION' table: boundary type → mass flow [kg/s]."""
    blocks = log.split("BOUNDARY MASS FLOW INFORMATION")
    if len(blocks) < 2:
        return {}
    out = {}
    for m in re.finditer(r"^([A-Za-z][A-Za-z ]+?)\s+\d+\s+(\d+)\s+([-+\d.eE]+)\s*$", blocks[-1], re.M):
        if int(m.group(2)):
            out[m.group(1).strip()] = float(m.group(3))
    return out


def field_ranges(log: str) -> dict[str, tuple[float, float]]:
    """Last table of cell-field ranges: name → (min, max)."""
    out = {}
    for m in re.finditer(r"^v  (\S.*?)\s{2,}([-+\d.eE]+)\s+([-+\d.eE]+)\s+[-+\d.eE]+\s+[-+\d.eE]+\s*$",
                         log, re.M):
        out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def review(case_dir: Path):
    """Judge a code_saturne result with the same verdicts as the OpenFOAM reviewer."""
    from app.review import Review, _load_request, _stated_speed

    case_dir = Path(case_dir)
    resu = case_dir / "RESU" / RUN_ID
    r = Review(case_dir)
    analysis, description = _load_request(case_dir)
    if not (resu / "residuals.csv").exists():
        r.add("Run", "fail", "No results: the case has not been run.")
        return r
    if (resu / "run_status.failed").exists() or (resu / "error").exists():
        err = (resu / "error").read_text(errors="replace") if (resu / "error").exists() else ""
        msg = next((l.strip() for l in err.splitlines() if l.strip() and not l.startswith("/")), "")
        r.add("Run", "fail", f"code_saturne stopped with an error. {msg}".strip())
        return r
    log = (resu / "run_solver.log").read_text(errors="replace")
    it, res = residuals(resu)
    r.facts.update(solver="code_saturne", iterations=it,
                   final_residuals={k: float(f"{v:.3g}") for k, v in res.items()})
    worst_f, worst = max(res.items(), key=lambda kv: kv[1])
    if analysis.get("solver") == "pimpleFoam":
        # time-dependent: the residuals measure the change over a step, not convergence
        m = re.findall(r"^ *INSTANT\s+([-+\d.eE]+)\s+TIME STEP NUMBER\s+(\d+)", log, re.M)
        t_end = float(m[-1][0]) if m else float("nan")
        it = int(m[-1][1]) if m else it
        r.facts["final_time"] = t_end
        r.add("Convergence", "pass" if worst < 0.1 else "warn",
              f"Time-dependent run reached t = {t_end:.4g} s in {it} steps; largest residual in "
              f"the last step {worst_f} = {worst:.2g}.")
    elif worst < 1e-3:
        r.add("Convergence", "pass", f"{it} iterations; all residuals below 1e-3 (largest {worst_f} = {worst:.2g}).")
    elif worst < 1e-1:
        r.add("Convergence", "warn", f"{it} iterations; {worst_f} residual {worst:.2g}: not fully converged.")
    else:
        r.add("Convergence", "fail", f"Not converged: {worst_f} residual {worst:.2g} after {it} iterations.")

    speed = _inlet_speed(analysis)
    want = _stated_speed(description)
    if want:
        ok = abs(speed / want - 1) < 0.05
        r.add("Request", "pass" if ok else "fail",
              f"Inlet speed {speed:.4g} m/s, description says {want:g} m/s.")
    r.facts["inlet_speed"] = speed

    flows = mass_flows(log)
    inflow = -sum(v for v in flows.values() if v < 0)
    if inflow > 0:
        imbalance = abs(sum(flows.values())) / inflow
        r.facts["mass_imbalance"] = float(f"{imbalance:.3g}")
        r.add("Mass balance", "pass" if imbalance < 0.01 else "warn" if imbalance < 0.05 else "fail",
              f"Net mass flow through the boundaries is {imbalance:.2%} of the inflow "
              f"({inflow:.4g} kg/s in).")

    ranges = field_ranges(log)
    vmag = ranges.get("ǁVelocityǁ") or ranges.get("Velocity magnitude")
    if vmag is None or not all(math.isfinite(x) for x in vmag):
        r.add("Fields", "fail" if vmag is not None else "info", "Velocity range not available or not finite.")
    else:
        r.facts["max_U"] = vmag[1]
        status = "warn" if speed and vmag[1] > 10 * speed else "pass"
        r.add("Fields", status, f"max |U| {vmag[1]:.3g} m/s (inflow {speed:.3g} m/s).")
    return r
