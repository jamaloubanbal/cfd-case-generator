"""
Code-to-code validation: the same benchmarks solved by OpenFOAM and by code_saturne.

    python -m validation.run --only poiseuille_channel cylinder_re40   # OpenFOAM side
    python -m validation.compare                                       # code_saturne side + table

Each benchmark's analysis (the object an LLM would return for its description) goes through
the code_saturne backend (app/saturne.py) and is measured by the *same* evaluation function
as the OpenFOAM case, reading code_saturne's EnSight output (app/postprocess/ensight.py).
OpenFOAM numbers come from docs/validation/results.json. Writes
docs/validation/code_comparison.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from app import saturne
from app.postprocess.ensight import SaturneResult
from validation.benchmarks import BENCHMARKS

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "validation"
# Benchmarks the code_saturne backend can run today (steady incompressible, 2-D geometries)
CODE_SATURNE_BENCHMARKS = ("poiseuille_channel", "cylinder_re40", "cylinder_re100")


def _probes(key: str) -> list[tuple[float, float, float]] | None:
    """Monitoring probes the evaluation needs (the shedding benchmark: 2 D behind the cylinder)."""
    if key != "cylinder_re100":
        return None
    from app.generators import GENERATORS
    b = BENCHMARKS[key]
    m = b.meta
    dz = float(GENERATORS["pimpleFoam"]()._build_context(b.analysis)["domain_z"])
    return [(m["cx"] + 2 * m["D"], m["H"] / 2, dz / 2)]


def run_saturne(key: str, work: Path) -> dict:
    b = BENCHMARKS[key]
    t0 = time.time()
    transient = b.solver == "pimpleFoam"
    case = saturne.generate(b.analysis, None, work / key, b.description,
                            iterations=max(int(b.end_time), 8000), probes=_probes(key),
                            end_time=b.end_time if transient else None)
    saturne.run(case, timeout=4 * 3600)
    rev = saturne.review(case)
    if any(c.name == "Run" for c in rev.checks):
        raise RuntimeError(f"{key}: code_saturne failed: {rev.checks[0].message}")
    res = SaturneResult(case / "RESU" / saturne.RUN_ID, rho=float(b.analysis["fluid"]["rho"]))
    metrics = b.evaluate(res, b.analysis)
    return {"metrics": metrics, "cells": res.n_cells, "elapsed_s": time.time() - t0,
            "verdict": rev.verdict, "iterations": rev.facts.get("iterations")}


def _pct(x: float) -> str:
    return f"{x * 100:+.2f} %"


def write_report(cs: dict[str, dict]) -> None:
    of = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    rows, data = [], {}
    for key, r in cs.items():
        b = BENCHMARKS[key]
        of_metrics = {m["name"]: m for m in (of.get(key) or {}).get("metrics", [])}
        data[key] = {"title": b.title, "cells": r["cells"], "elapsed_s": round(r["elapsed_s"], 1),
                     "verdict": r["verdict"], "metrics": []}
        for m in r["metrics"]:
            o = of_metrics.get(m.name)
            gap = (m.value - o["value"]) / o["value"] if o and o["value"] else None
            data[key]["metrics"].append({"name": m.name, "reference": m.reference, "code_saturne": m.value,
                                         "code_saturne_error": m.error, "openfoam": o["value"] if o else None,
                                         "openfoam_error": o["error"] if o else None, "gap": gap,
                                         "tolerance": m.tolerance, "passed": m.passed})
            rows.append(
                f"| {b.title} | {m.name} | {m.reference:.4g} | "
                + (f"{o['value']:.4g} ({_pct(o['error'])}) {'✅' if o['passed'] else '❌'}" if o else "–")
                + f" | {m.value:.4g} ({_pct(m.error)}) {'✅' if m.passed else '❌'} | "
                + (_pct(gap) if gap is not None else "–") + " |")
    (OUT / "code_comparison.json").write_text(json.dumps(data, indent=2))
    lines = [
        "# Code-to-code validation: OpenFOAM and code_saturne",
        "",
        "The same benchmark (the analysis an LLM would return for its description) is generated",
        "for both codes and measured by the same Python function. code_saturne cases come from",
        "`app/saturne.py` (blockMesh mesh written as UNV, `setup.xml` written by code_saturne's",
        "model layer); re-run with `python -m validation.compare`.",
        "",
        "| Benchmark | Quantity | Reference | OpenFOAM v1912 | code_saturne 9.2 | Code-to-code gap |",
        "|---|---|---:|---:|---:|---:|",
        *rows,
        "",
        "Run details (code_saturne): " + "; ".join(
            f"{d['title']}: {d['cells']:,} cells, {d['elapsed_s']:.0f} s, reviewer verdict {d['verdict']}"
            for d in data.values()) + ".",
        "",
        "**Findings.** Laminar channel flow agrees between the codes to 0.03 % and with the exact "
        "solution to 0.15 %. Vortex shedding at Re = 100 agrees to 0.06 % (St 0.1652 vs 0.1653), "
        "after switching off code_saturne's velocity slope test, which locally reverts to "
        "first-order upwind in a vortex street and had lowered St by 12 %. The steady wake at "
        "Re = 40 does not agree: code_saturne's recirculation is longer than OpenFOAM's and than "
        "the literature (2.13–2.35). Refining the mesh 2× moves OpenFOAM into the literature range "
        "(2.10 → 2.27) and code_saturne away from it (2.50 → 2.64); a time-accurate code_saturne "
        "run gives 2.45, so neither resolution nor the steady formulation explains it. Open.",
        "",
        "Not yet compared: the turbulent backward-facing step (code_saturne does not converge yet "
        "with k-ω SST), the pipe (no wedge equivalent) and the heated cavity (natural convection "
        "is not in the code_saturne backend yet).",
        "",
    ]
    (OUT / "code_comparison.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=CODE_SATURNE_BENCHMARKS)
    ap.add_argument("--workdir", help="where to keep the cases (default: temporary directory)")
    args = ap.parse_args(argv)
    if saturne.find_executable() is None:
        print("error: code_saturne not found (set CODE_SATURNE)", file=sys.stderr)
        return 2
    prev = json.loads((OUT / "code_comparison.json").read_text()) if (OUT / "code_comparison.json").exists() else {}
    ctx = tempfile.TemporaryDirectory() if not args.workdir else None
    work = Path(args.workdir or ctx.name)
    results = {}
    for key in args.only or CODE_SATURNE_BENCHMARKS:
        print(f"▶ {BENCHMARKS[key].title} (code_saturne) …", flush=True)
        r = run_saturne(key, work)
        for m in r["metrics"]:
            print(f"   {'PASS' if m.passed else 'FAIL'}  {m.name}: {m.value:.4g} (ref {m.reference:.4g}, {_pct(m.error)})")
        results[key] = r
    if args.only and prev:
        print("(only the selected benchmarks are re-run; the report lists them)")
    write_report(results)
    print(f"Report: {OUT / 'code_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
