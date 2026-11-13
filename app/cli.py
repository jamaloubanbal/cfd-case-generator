"""
Command-line interface.

    # From a plain-English description (uses an LLM)
    python -m app.cli describe "Laminar flow past a 2-D cylinder at Re = 100" -o cylinder/ --run
    python -m app.cli describe "…" -o case/ --backend gemini      # free tier, GEMINI_API_KEY
    python -m app.cli describe "…" -o case/ --backend ollama      # local, no key

    # From an analysis JSON (no LLM needed) — e.g. one of examples/*.json
    python -m app.cli generate examples/cylinder_vortex_shedding.json -o cylinder/ --run

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


def _generate(analysis: dict, out: Path, run: bool, force: bool) -> int:
    if out.exists() and any(out.iterdir()):
        if not force:
            print(f"error: {out} exists and is not empty (use --force to overwrite)", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    solver = analysis["solver"]
    GENERATORS[solver]().generate(analysis, out)
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"Generated {solver} case ({analysis['geometry_type']}, {analysis['turb_model']}) "
          f"in {out}/ — {n_files} files")
    if not run:
        print(f"Run it with:  cd {out} && ./Allrun")
        return 0
    if not shutil.which("blockMesh"):
        print("error: OpenFOAM not found — source its etc/bashrc to use --run", file=sys.stderr)
        return 2
    print("Running ./Allrun …", flush=True)
    proc = subprocess.run(["./Allrun"], cwd=out)
    print("Done — logs are in log.*" if proc.returncode == 0 else "Allrun failed — see log.* files")
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
    d.add_argument("--base-url", help="endpoint for --backend openai-compatible (default: LM Studio, "
                                       "http://localhost:1234/v1)")
    d.add_argument("--model", help="model id (default: claude-opus-5-5 for Claude, qwen2.5:7b for Ollama; override with CFD_CLAUDE_MODEL / CFD_OLLAMA_MODEL)")
    d.add_argument("--save-analysis", metavar="FILE", help="write the sanitised analysis JSON here")
    for p in (g, d):
        p.add_argument("-o", "--output", required=True, help="case directory to create")
        p.add_argument("--run", action="store_true", help="run ./Allrun after generating")
        p.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    c = sub.add_parser("check", help="print the physics checks for an analysis JSON")
    c.add_argument("input")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "describe":
            from app.llm import analyze_cfd_problem
            print(f"Analysing with {args.backend} …", flush=True)
            raw = analyze_cfd_problem(args.description, backend=args.backend, model=args.model,
                                      base_url=args.base_url)   # keys come from the environment
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

    print(f"Solver: {analysis['solver']} — {analysis.get('solver_justification', '')}")
    print("Physics checks:")
    worst = _print_checks(analysis)
    if args.cmd == "check":
        return 1 if worst >= 2 else 0
    if args.cmd == "describe" and args.save_analysis:
        Path(args.save_analysis).write_text(json.dumps(analysis, indent=2))
    return _generate(analysis, Path(args.output), args.run, args.force)


if __name__ == "__main__":
    sys.exit(main())
