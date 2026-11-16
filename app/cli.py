"""
Command-line interface.

    # From a plain-English description (uses an LLM)
    python -m app.cli describe "Laminar flow past a 2-D cylinder at Re = 100" -o cylinder/ --run
    python -m app.cli describe "…" -o case/ --backend gemini      # free tier, GEMINI_API_KEY
    python -m app.cli describe "…" -o case/ --backend ollama      # local, no key

    # From an analysis JSON (no LLM needed) — e.g. one of examples/*.json
    python -m app.cli generate examples/cylinder_vortex_shedding.json -o cylinder/ --run

    # Run and repair automatically if it fails (rules, plus an LLM for harder failures)
    python -m app.cli generate examples/dam_break.json -o dambreak/ --fix gemini
    python -m app.cli fix my_case/ --backend gemini

    # Your own geometry: a mesh made in SALOME (groups become patches, mapped to roles)
    python -m app.cli describe "Air at 5 m/s through this duct" --mesh duct.unv -o duct/ --backend gemini --run
    python -m app.cli generate analysis.json --mesh duct.unv --role Inflow=inlet -o duct/

    # Judge a case that has run (checks, plus an LLM explanation with --backend)
    python -m app.cli review cylinder/ --backend gemini

    # Physics checks only
    python -m app.cli check examples/backstep_turbulent.json

An input JSON is either an analysis (the object described by ``app/schema.py``)
or ``{"description": ..., "analysis": {...}}`` as in ``examples/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.generators import GENERATORS
from app.sanitize import AnalysisError, sanitize_analysis
from app.validation import run_checks

_ICONS = {"ok": "✔", "info": "ℹ", "warning": "⚠", "error": "✖"}


def _with_mesh(analysis: dict, args, description: str | None) -> dict:
    """Import the SALOME mesh, map its groups to boundary roles and attach it to the analysis."""
    from app.mesh_import import ROLES, apply_roles, attach, check_roles, import_unv, map_roles
    mesh = getattr(args, "imported", None) or import_unv(args.mesh, Path(args.mesh).stem)
    backend = getattr(args, "backend", None) if args.cmd == "describe" else None
    roles, notes = map_roles(mesh.patches, description or "", backend=backend,
                             model=getattr(args, "model", None))
    for item in args.role:
        name, _, role = item.partition("=")
        if name not in roles or role not in ROLES:
            raise ValueError(f"--role {item}: groups are {', '.join(roles)}; roles are {', '.join(ROLES)}")
        roles[name] = role
    print(f"Mesh {args.mesh}: {mesh.n_cells:,} cells. Boundary groups → roles"
          + (" (LLM)" if backend else " (from the names)") + ":")
    for p in mesh.patches:
        print(f"  {p.name:20s} {roles[p.name]:10s} {p.n_faces:>7,} faces")
    for n in mesh.notes + notes:
        print(f"  note: {n}")
    problems = check_roles(roles, analysis["solver"])
    if problems:
        raise ValueError("boundary roles: " + "; ".join(problems) + " — fix them with --role GROUP=ROLE")
    return attach(analysis, mesh, apply_roles(mesh, roles)) | {"_unv": str(args.mesh)}


def _load(path: str) -> tuple[dict, str | None]:
    data = json.loads(Path(path).read_text())
    return data.get("analysis", data), data.get("description")


def _prepare(raw: dict, description: str | None = None) -> dict:
    analysis, notes = sanitize_analysis(raw, description)
    for n in notes:
        print(f"  corrected: {n}")
    return analysis


def _print_checks(analysis: dict) -> int:
    worst = 0
    for c in run_checks(analysis):
        print(f"  {_ICONS[c['level']]} {c['title']}: {c['detail']}")
        worst = max(worst, {"ok": 0, "info": 0, "warning": 1, "error": 2}[c["level"]])
    return worst


def _repair(case_dir: Path, fix: str, model: str | None) -> int:
    """Run the case with the repair agent ('rules' = rule-based fixes only, else an LLM backend)."""
    from app.agent import repair
    if not shutil.which("blockMesh"):
        print("error: OpenFOAM not found — source its etc/bashrc first", file=sys.stderr)
        return 2
    backend = None if fix == "rules" else fix
    print(f"Running with the repair agent ({'rules only' if backend is None else 'rules + ' + backend}) …",
          flush=True)
    result = repair(case_dir, backend=backend, model=model, log=lambda s: print("  " + s, flush=True))
    print(f"{'Done' if result.success else 'Not repaired'} — see {case_dir}/agent_report.md and log.*")
    return 0 if result.success else 1


def _review(case_dir: Path, backend: str | None = None, model: str | None = None):
    """Judge the results of a case that has run (and explain them with *backend*, if given)."""
    from app.review import explain, print_review, review, write_report
    try:
        rev = review(case_dir)
        if backend and backend != "rules":
            try:
                explain(rev, backend, model)
            except Exception as exc:                    # the checks still stand without the LLM
                print(f"  (no LLM explanation: {exc})")
    except Exception as exc:
        print(f"Result review failed: {exc}")
        return None
    print_review(rev)
    print(f"  Report: {write_report(rev)}")
    return rev


def _generate_saturne(analysis: dict, out: Path, run: bool, force: bool, description: str | None,
                      backend: str | None, model: str | None) -> int:
    from app import saturne
    from app.review import explain, print_review, write_report
    if "_unv" not in analysis:
        print("error: --code code_saturne needs a SALOME mesh (--mesh FILE.unv)", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        if not force:
            print(f"error: {out} exists and is not empty (use --force to overwrite)", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    try:
        case_dir = saturne.generate(analysis, analysis["_unv"], out, description)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Generated code_saturne study ({analysis['turb_model']}) in {out}/ — case {case_dir.name}, "
          f"setup.xml written through code_saturne's model layer")
    if not run:
        print(f"Run it with:  cd {case_dir} && code_saturne run")
        return 0
    print("Running code_saturne …", flush=True)
    saturne.run(case_dir)
    rev = saturne.review(case_dir)
    if backend and backend != "rules":
        try:
            explain(rev, backend, model)
        except Exception as exc:
            print(f"  (no LLM explanation: {exc})")
    print_review(rev)
    print(f"  Report: {write_report(rev)}")
    return 0 if rev.verdict != "unreliable" else 1


def _generate(analysis: dict, out: Path, run: bool, force: bool,
              fix: str | None = None, fix_model: str | None = None,
              description: str | None = None, backend: str | None = None,
              model: str | None = None) -> int:
    if out.exists() and any(out.iterdir()):
        if not force:
            print(f"error: {out} exists and is not empty (use --force to overwrite)", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    solver = analysis["solver"]
    GENERATORS[solver]().generate(analysis, out)
    from app.review import save_request
    save_request(out, analysis, description)          # lets `cfdgen review` compare with the request
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"Generated {solver} case ({analysis['geometry_type']}, {analysis['turb_model']}) "
          f"in {out}/ — {n_files} files")
    if fix:
        rc = _repair(out, fix, fix_model)
        if rc == 0:
            _review(out, backend or (fix if fix != "rules" else None), model or fix_model)
        return rc
    if not run:
        print(f"Run it with:  cd {out} && ./Allrun")
        return 0
    if not shutil.which("blockMesh"):
        print("error: OpenFOAM not found — source its etc/bashrc to use --run", file=sys.stderr)
        return 2
    print("Running ./Allrun …", flush=True)
    proc = subprocess.run(["./Allrun"], cwd=out)
    print("Done — logs are in log.*" if proc.returncode == 0 else "Allrun failed — see log.* files")
    if proc.returncode == 0:
        _review(out, backend, model)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cfdgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a case from an analysis JSON")
    g.add_argument("input")
    d = sub.add_parser("describe", help="generate a case from a plain-English description (LLM)")
    d.add_argument("description")
    d.add_argument("--backend", default="claude",
                   choices=["claude", "ollama", "gemini", "groq", "openrouter", "openai-compatible"],
                   help="LLM backend; API keys are read from ANTHROPIC_API_KEY, GEMINI_API_KEY, "
                        "GROQ_API_KEY, OPENROUTER_API_KEY or OPENAI_COMPAT_API_KEY")
    d.add_argument("--timeout", type=int, help="seconds to wait for a cloud provider (default 300; "
                                              "one automatic retry)")
    d.add_argument("--base-url", help="endpoint for --backend openai-compatible (default: LM Studio, "
                                       "http://localhost:1234/v1)")
    d.add_argument("--model", help="model id (default: claude-opus-5-5 for Claude, qwen2.5:7b for Ollama; override with CFD_CLAUDE_MODEL / CFD_OLLAMA_MODEL)")
    d.add_argument("--save-analysis", metavar="FILE", help="write the sanitised analysis JSON here")
    for p in (g, d):
        p.add_argument("-o", "--output", required=True, help="case directory to create")
        p.add_argument("--run", action="store_true", help="run ./Allrun after generating")
        p.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
        p.add_argument("--fix", nargs="?", const="rules", metavar="BACKEND",
                       help="run with the repair agent: rule-based fixes, plus an LLM backend if given "
                            "(e.g. --fix gemini)")
        p.add_argument("--fix-model", help="model for the repair agent's LLM")
        p.add_argument("--mesh", metavar="FILE.unv",
                       help="use a mesh made in SALOME (I-DEAS .unv) instead of a built-in geometry")
        p.add_argument("--code", choices=["openfoam", "code_saturne"], default="openfoam",
                       help="CFD code to generate the case for (code_saturne: SALOME meshes, "
                            "steady incompressible flow; set CODE_SATURNE if it is not on PATH)")
        p.add_argument("--role", action="append", default=[], metavar="GROUP=ROLE",
                       help="set a mesh group's boundary role (inlet, outlet, wall, symmetry, empty, "
                            "hot_wall, cold_wall); repeatable")
    c = sub.add_parser("check", help="print the physics checks for an analysis JSON")
    c.add_argument("input")
    f = sub.add_parser("fix", help="run an existing case and repair it if it fails")
    f.add_argument("case_dir")
    f.add_argument("--backend", help="LLM for fixes the rules cannot make (default: rules only)")
    f.add_argument("--model", help="model id for --backend")
    rv = sub.add_parser("review", help="judge the results of a case that has run")
    rv.add_argument("case_dir")
    rv.add_argument("--backend", help="LLM that explains the result and suggests improvements "
                                      "(default: checks only)")
    rv.add_argument("--model", help="model id for --backend")
    args = ap.parse_args(argv)

    if args.cmd == "review":
        case = Path(args.case_dir)
        if not (case / "system" / "controlDict").exists():
            print(f"error: {case} is not an OpenFOAM case (no system/controlDict)", file=sys.stderr)
            return 2
        rev = _review(case, args.backend, args.model)
        return 0 if rev is not None and rev.verdict != "unreliable" else 1

    if args.cmd == "fix":
        case = Path(args.case_dir)
        if not (case / "system" / "controlDict").exists():
            print(f"error: {case} is not an OpenFOAM case (no system/controlDict)", file=sys.stderr)
            return 2
        return _repair(case, args.backend or "rules", args.model)

    try:
        if args.cmd == "describe":
            from app.llm import analyze_cfd_problem
            print(f"Analysing with {args.backend} …", flush=True)
            from app.mesh_import import import_unv, mesh_hint
            if args.mesh:      # import first so the LLM sees the mesh's dimensions
                args.imported = import_unv(args.mesh, Path(args.mesh).stem)
            prompt = args.description + (mesh_hint(args.imported) if args.mesh else "")
            raw = analyze_cfd_problem(prompt, backend=args.backend, model=args.model,
                                      base_url=args.base_url,   # keys come from the environment
                                      timeout=args.timeout)
            description = args.description
        else:
            raw, description = _load(args.input)
        analysis = _prepare(raw, description)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "mesh", None):
        try:
            analysis = _with_mesh(analysis, args, description)
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print(f"Solver: {analysis['solver']} — {analysis.get('solver_justification', '')}")
    print("Physics checks:")
    worst = _print_checks(analysis)
    if args.cmd == "check":
        return 1 if worst >= 2 else 0
    if args.cmd == "describe" and args.save_analysis:
        Path(args.save_analysis).write_text(json.dumps(analysis, indent=2))
    llm = args.backend if args.cmd == "describe" else None
    if args.code == "code_saturne":
        return _generate_saturne(analysis, Path(args.output), args.run, args.force, description,
                                 llm, getattr(args, "model", None))
    return _generate(analysis, Path(args.output), args.run, args.force, args.fix, args.fix_model,
                     description, llm, getattr(args, "model", None))


if __name__ == "__main__":
    sys.exit(main())
