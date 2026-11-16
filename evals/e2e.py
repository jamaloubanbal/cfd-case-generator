"""
End-to-end benchmark: from a plain-English description to a judged OpenFOAM result.

For every description in the evaluation sets (20 dev + 10 held-out), the model's cached
answer (evals/responses/, no API call) is sanitised, turned into a case, run with the
rule-based repair agent, and judged by the result reviewer (app/review.py):

    python -m evals.e2e                                   # gemini answers, all cases
    python -m evals.e2e --responses ollama__qwen2.5-7b --only cyl_re100 cavity_re1000
    python -m evals.e2e --report                          # rewrite the report from saved results

Outcome per case: trustworthy / questionable / unreliable (the reviewer's verdict), or
rejected (the sanitiser refused the answer), failed (did not run even after repairs),
timeout. Results: docs/evaluation/e2e_results.json and e2e_results.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = REPO / "docs" / "evaluation"
RESULTS = OUT / "e2e_results.json"
OUTCOMES = ("trustworthy", "questionable", "unreliable", "failed", "timeout", "rejected")


def _cases() -> list[dict]:
    out = []
    for set_name, f in (("dev", "cases.json"), ("held-out", "cases_holdout.json")):
        for c in json.loads((ROOT / f).read_text())["cases"]:
            out.append({**c, "set": set_name})
    return out


def run_case(case: dict, raw: dict | None, workdir: str, timeout: int) -> dict:
    """Sanitise → generate → run with the rule-based repair agent → review."""
    import subprocess

    from app.agent import repair
    from app.generators import GENERATORS
    from app.review import review, save_request
    from app.sanitize import AnalysisError, sanitize_analysis

    res = {"id": case["id"], "set": case["set"], "description": case["description"]}
    if not isinstance(raw, dict):
        return {**res, "outcome": "rejected", "issues": ["no usable answer from the model"]}
    try:
        a, _ = sanitize_analysis(raw, case["description"])
    except AnalysisError as exc:
        return {**res, "outcome": "rejected", "issues": [str(exc)]}
    a["n_cores"] = 1                              # cases run side by side
    res.update(solver=a["solver"], geometry=a["geometry_type"], turb=a["turb_model"])
    case_dir = Path(workdir) / case["id"]
    t0 = time.time()
    try:
        GENERATORS[a["solver"]]().generate(a, case_dir)
        save_request(case_dir, a, case["description"])
        r = repair(case_dir, backend=None, timeout=timeout, log=lambda s: None)
    except subprocess.TimeoutExpired:
        return {**res, "outcome": "timeout", "issues": [f"no result within {timeout} s"],
                "seconds": round(time.time() - t0)}
    except Exception as exc:                      # generator error: count it, keep going
        return {**res, "outcome": "failed", "issues": [f"{type(exc).__name__}: {exc}"]}
    res.update(fixes=[at.fixer for at in r.attempts], seconds=round(time.time() - t0))
    if not r.success:
        last = r.attempts[-1].diagnosis.message if r.attempts else "did not run"
        return {**res, "outcome": "failed", "issues": [last[:200]]}
    try:
        rev = review(case_dir)
    except Exception as exc:
        return {**res, "outcome": "failed", "issues": [f"review error: {exc}"]}
    return {**res, "outcome": rev.verdict,
            "issues": [f"{c.name}: {c.message}" for c in rev.checks if c.status in ("warn", "fail")],
            "facts": {k: rev.facts.get(k) for k in ("iterations", "Re", "Ma", "mass_imbalance", "y_plus")
                      if rev.facts.get(k) is not None}}


def report(results: dict) -> str:
    lines = ["# End-to-end benchmark", "",
             "From a plain-English description to a judged OpenFOAM result, with no human in the "
             "loop: the model's answer is sanitised, generated, run with the rule-based repair "
             "agent and judged by the [result reviewer](../GUIDE.md#result-review). "
             "Re-run with `python -m evals.e2e`.", ""]
    for key, rows in results.items():
        rows = list(rows.values())
        n = len(rows)
        count = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
        lines += [f"## {key}", "",
                  f"**{count['trustworthy']} of {n} descriptions give a trustworthy result** "
                  f"({count['questionable']} questionable, {count['unreliable']} unreliable, "
                  f"{count['failed'] + count['timeout']} did not run, {count['rejected']} rejected).", "",
                  "| Outcome | Dev (20) | Held-out (10) |", "|---|---|---|"]
        for o in OUTCOMES:
            d = sum(r["outcome"] == o and r["set"] == "dev" for r in rows)
            h = sum(r["outcome"] == o and r["set"] == "held-out" for r in rows)
            if d or h:
                lines.append(f"| {o} | {d} | {h} |")
        lines += ["", "| Case | Set | Solver · geometry · model | Fixes | Outcome | Issues |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (OUTCOMES.index(r["outcome"]), r["id"])):
            setup = " · ".join(r.get(k, "") for k in ("solver", "geometry", "turb") if r.get(k))
            fixes = ", ".join(r.get("fixes") or []) or "–"
            issues = "<br>".join(i.replace("|", "\\|") for i in r.get("issues") or []) or "–"
            lines.append(f"| `{r['id']}` | {r['set']} | {setup} | {fixes} | **{r['outcome']}** | {issues} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default="gemini__gemini-3.1-flash-lite",
                    help="cached answers in evals/responses/ (file name without .json)")
    ap.add_argument("--only", nargs="*", help="case ids")
    ap.add_argument("--jobs", type=int, default=4, help="cases run in parallel")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per OpenFOAM run")
    ap.add_argument("--report", action="store_true", help="only rewrite the report")
    args = ap.parse_args(argv)

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if not args.report:
        answers = json.loads((ROOT / "responses" / f"{args.responses}.json").read_text())["answers"]
        cases = [c for c in _cases() if not args.only or c["id"] in args.only]
        rows = results.setdefault(args.responses, {})
        with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(args.jobs) as pool:
            futs = {pool.submit(run_case, c, (answers.get(c["id"]) or {}).get("analysis"), tmp,
                                args.timeout): c for c in cases}
            for f in as_completed(futs):
                r = f.result()
                rows[r["id"]] = r
                print(f"{r['id']:28s} {r['outcome']:12s} {'; '.join(r.get('issues') or [])[:150]}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))
    (OUT / "e2e_results.md").write_text(report(results))
    for key, rows in results.items():
        ok = sum(r["outcome"] == "trustworthy" for r in rows.values())
        print(f"{key}: {ok}/{len(rows)} trustworthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
