"""
Fault-injection benchmark for the run-and-fix agent.

A working case is generated, a realistic fault is injected (most are bugs this project
actually had), and the agent must make the case run again.  Modes: rules only, LLM only,
rules + LLM.

    source /usr/share/openfoam/etc/bashrc
    python -m evals.faults --check                       # confirm every fault breaks its case
    python -m evals.faults --mode rules
    python -m evals.faults --mode llm --backend gemini --model gemini-3.1-flash-lite
    python -m evals.faults --mode both --backend ollama --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from app.agent import diagnose, repair, run
from app.generators import GENERATORS
from app.sanitize import sanitize_analysis

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "docs" / "evaluation" / "agent_results.json"
STEADY_ITERS, TRANSIENT_STEPS = 60, 15


def _example(name: str) -> dict:
    data = json.loads((REPO / "examples" / f"{name}.json").read_text())
    return sanitize_analysis(data["analysis"], data["description"])[0]


def _fixture(name: str, **changes) -> dict:
    from tests import fixtures
    a = copy.deepcopy(getattr(fixtures, name))
    a.update(changes)
    return a


def _edit(path: Path, pattern: str, repl: str, flags=re.M | re.S) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"fault injection did not match in {path.name}: {pattern!r}")
    path.write_text(new)


# Each fault: (base analysis, injector(case_dir), what it reproduces)
FAULTS = {
    "pimple_missing_final_solvers": (
        lambda: _fixture("PIMPLE_FOAM_ANALYSIS", turb_model="kEpsilon"),
        lambda c: _edit(c / "system/fvSolution", r'\n    "\(k\|epsilon\|omega\|nuTilda\)Final"\s*\{.*?\}\n', "\n"),
        "pimpleFoam without kFinal/epsilonFinal solver entries (original bug)"),
    "interfoam_missing_pcorr": (
        lambda: _fixture("INTER_FOAM_ANALYSIS"),
        lambda c: _edit(c / "system/fvSolution", r'\n    "pcorr\.\*"\s*\{.*?\}\n', "\n"),
        "interFoam without the pcorr solver (original bug)"),
    "interfoam_missing_phirb_scheme": (
        lambda: _fixture("INTER_FOAM_ANALYSIS"),
        lambda c: _edit(c / "system/fvSchemes", r"^\s*div\(phirb,alpha\)[^\n]*\n", ""),
        "interFoam scheme missing in v1912 (original bug: OpenFOAM.org syntax)"),
    "missing_div_scheme": (
        lambda: _fixture("SIMPLE_FOAM_ANALYSIS", turb_model="kEpsilon"),
        lambda c: _edit(c / "system/fvSchemes", r"^\s*div\(phi,k\)[^\n]*\n", ""),
        "convection scheme for k deleted"),
    "buoyant_unstable_relaxation": (
        lambda: _fixture("BSF_ANALYSIS", turb_model="laminar"),
        lambda c: (_edit(c / "system/fvSolution", r"momentumPredictor\s+no;", "consistent               yes;"),
                   _edit(c / "system/fvSolution", r"p_rgh\s+0\.7;", "p_rgh           0.3;"),
                   _edit(c / "system/fvSolution", r"(\n\s+U\s+)0\.3;", r"\g<1>0.7;"),
                   _edit(c / "system/fvSolution", r"(\n\s+h\s+)0\.3;", r"\g<1>0.5;")),
        "buoyantSimpleFoam with SIMPLEC and weak relaxation: negative temperature (original bug)"),
    "rho_missing_pressure_limits": (
        lambda: _example("compressible_obstacle"),
        lambda c: (_edit(c / "system/fvSolution", r"\s*//[^\n]*\n\s*//[^\n]*\n\s*pMinFactor[^\n]*\n\s*pMaxFactor[^\n]*", ""),
                   _edit(c / "system/fvSolution", r"(\n\s+p\s+)0\.1;", r"\g<1>0.7;"),
                   _edit(c / "system/fvSolution", r"(\n\s+U\s+)0\.3;", r"\g<1>0.9;"),
                   _edit(c / "system/fvSolution", r"(\n\s+h\s+)0\.2;", r"\g<1>0.8;")),
        "Mach 0.5 rhoSimpleFoam, aggressive relaxation and no pressure limits"),
    "patch_type_mismatch": (
        lambda: _fixture("CYLINDER_ANALYSIS"),
        lambda c: _edit(c / "0/U", r"(\n    top\s*\{\s*type\s+)symmetryPlane;", r"\g<1>symmetry;"),
        "field patch type does not match the mesh patch (original STL bug)"),
    "missing_bc_value": (
        lambda: _fixture("SIMPLE_FOAM_ANALYSIS", turb_model="kEpsilon"),
        lambda c: _edit(c / "0/p", r"(\n    outlet\s*\{\s*type\s+fixedValue;)\s*value[^;]*;", r"\1"),
        "fixedValue outlet pressure without a value"),
    "dictionary_syntax_error": (
        lambda: _fixture("SIMPLE_FOAM_ANALYSIS", turb_model="kEpsilon"),
        lambda c: _edit(c / "system/fvSchemes", r"\}\n(\s*gradSchemes)", r"\n\1"),
        "unclosed brace in fvSchemes (hand-editing mistake)"),
    "unknown_solver_keyword": (
        lambda: _fixture("SIMPLE_FOAM_ANALYSIS", turb_model="kEpsilon"),
        lambda c: _edit(c / "system/fvSolution", r"(\n    p\n    \{\n\s+solver\s+)GAMG;", r"\1GAMGG;"),
        "typo in a linear-solver name"),
}


def _shorten(case_dir: Path) -> None:
    cd = case_dir / "system" / "controlDict"
    t = cd.read_text()
    dt = float(re.search(r"^deltaT\s+([^;]+);", t, re.M).group(1))
    transient = re.search(r"^application\s+(pimpleFoam|interFoam);", t, re.M)
    end = (TRANSIENT_STEPS if transient else STEADY_ITERS) * dt
    t = re.sub(r"^endTime\s+[^;]+;", f"endTime         {end:g};", t, flags=re.M)
    t = re.sub(r"^writeControl\s+[^;]+;", "writeControl    runTime;", t, flags=re.M)
    t = re.sub(r"^writeInterval\s+[^;]+;", f"writeInterval   {end:g};", t, flags=re.M)
    cd.write_text(t)


def build(name: str, work: Path) -> Path:
    make_analysis, inject, _ = FAULTS[name]
    analysis = make_analysis()
    case = work / name
    GENERATORS[analysis["solver"]]().generate(analysis, case)
    _shorten(case)
    inject(case)
    return case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only verify that each fault breaks its case")
    ap.add_argument("--mode", choices=["rules", "llm", "both"], default="rules")
    ap.add_argument("--backend")
    ap.add_argument("--model")
    ap.add_argument("--only", nargs="*", choices=sorted(FAULTS))
    args = ap.parse_args(argv)
    if args.mode != "rules" and not args.backend and not args.check:
        ap.error("--backend is required for --mode llm/both")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    key = "rules" if args.mode == "rules" else f"{args.mode}:{args.backend}/{args.model or 'default'}"
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.only or FAULTS:
            case = build(name, Path(tmp))
            if args.check:
                d = diagnose(case, run(case))
                print(f"{name:32s} {'STILL RUNS (bad fault)' if d.ok else d.kind + ': ' + d.message[:90]}")
                continue
            print(f"▶ {name} — {FAULTS[name][2]}")
            t0 = time.time()
            r = repair(case, backend=None if args.mode == "rules" else args.backend, model=args.model,
                       use_rules=args.mode != "llm", log=lambda s: print("   " + s))
            results.setdefault(key, {})[name] = {
                "repaired": r.success, "fix_attempts": len(r.attempts),
                "fixers": [a.fixer for a in r.attempts], "seconds": round(time.time() - t0)}
    if not args.check:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))
        ok = sum(v["repaired"] for v in results[key].values())
        print(f"\n{key}: {ok}/{len(results[key])} faults repaired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
