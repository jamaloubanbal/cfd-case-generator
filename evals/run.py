"""
How accurately does each LLM turn a plain-English description into a CFD setup?

    # Ask a model about every case (answers are cached in evals/responses/)
    python -m evals.run --backend ollama --model qwen2.5:7b
    python -m evals.run --backend gemini --model gemini-3.1-flash-lite --delay 6

    # Re-score all cached answers and write docs/evaluation/ (no model calls)
    python -m evals.run --report

Each answer is scored twice: as the model returned it ("raw") and after the
sanitiser ("sanitised"), which shows how much the validation layer repairs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from app.sanitize import AnalysisError, sanitize_analysis

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
# "dev": used while developing the sanitiser rules; "holdout": written afterwards and
# committed before any model saw them — the unbiased estimate.
SETS = {name: json.loads((ROOT / f).read_text())["cases"]
        for name, f in (("dev", "cases.json"), ("holdout", "cases_holdout.json"))}
CASES = SETS["dev"]
RESPONSES = ROOT / "responses"
FIELDS = ("solver", "geometry", "laminar", "transient", "Re", "nu")
RE_TOLERANCE = 0.15
NU_FACTOR = 2.0


# ── Scoring ──────────────────────────────────────────────────────────────────

def _inlet_speed(a: dict) -> float:
    for bc in a.get("boundary_conditions") or []:
        if not isinstance(bc, dict):
            continue
        U = (bc.get("values") or {}).get("U")
        if "inlet" in str(bc.get("patch_name", "")).lower() and isinstance(U, list):
            try:
                return math.sqrt(sum(float(c) ** 2 for c in U))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def simulated_re(a: dict) -> float | None:
    """Reynolds number the generated case would actually simulate: U·L/ν."""
    try:
        nu = float((a.get("fluid") or {}).get("nu"))
        L = float(a.get("characteristic_length"))
    except (TypeError, ValueError):
        return None
    U = _inlet_speed(a)
    return U * L / nu if nu > 0 and U > 0 else None


def score(a: dict | None, expected: dict) -> dict[str, bool]:
    """Field → correct?  A missing or unusable answer scores False on every field."""
    if not isinstance(a, dict):
        return {f: False for f in expected}
    out = {}
    for field, want in expected.items():
        if field == "solver":
            out[field] = a.get("solver") in want
        elif field == "geometry":
            out[field] = a.get("geometry_type") in want
        elif field == "laminar":
            out[field] = (a.get("turb_model") == "laminar") == want
        elif field == "transient":
            out[field] = bool(a.get("is_transient")) == want
        elif field == "Re":
            re_sim = simulated_re(a)
            targets = want if isinstance(want, list) else [want]
            out[field] = re_sim is not None and any(abs(re_sim / t - 1) <= RE_TOLERANCE for t in targets)
        elif field == "nu":
            try:
                nu = float((a.get("fluid") or {}).get("nu"))
                out[field] = 1 / NU_FACTOR <= nu / want <= NU_FACTOR
            except (TypeError, ValueError, ZeroDivisionError):
                out[field] = False
    return out


def score_case(case: dict, raw: dict | None) -> dict:
    sanitised = None
    if isinstance(raw, dict):
        try:
            sanitised, _ = sanitize_analysis(raw, case["description"])
        except AnalysisError:
            sanitised = None
    exp = case["expected"]
    return {"raw": score(raw, exp), "sanitised": score(sanitised, exp)}


# ── Running a model ──────────────────────────────────────────────────────────

def _response_file(backend: str, model: str) -> Path:
    return RESPONSES / f"{backend}__{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}.json"


def collect(backend: str, model: str, delay: float, retry_errors: bool, only: list[str] | None,
            cases: list[dict]) -> None:
    from app.llm import analyze_cfd_problem

    path = _response_file(backend, model)
    data = json.loads(path.read_text()) if path.exists() else {
        "backend": backend, "model": model, "answers": {}}
    RESPONSES.mkdir(exist_ok=True)
    for case in cases:
        cid = case["id"]
        if only and cid not in only:
            continue
        cached = data["answers"].get(cid)
        if cached is not None and not (retry_errors and "error" in cached):
            continue
        print(f"  {cid} …", end=" ", flush=True)
        t0 = time.time()
        try:
            answer = {"analysis": analyze_cfd_problem(case["description"], backend=backend, model=model)}
        except Exception as exc:                       # record failures; they score as wrong
            msg = str(exc)
            if "No API key" in msg or "rejected the API key" in msg or "Cannot reach" in msg:
                # A setup problem, not a wrong answer: stop without caching anything
                print(f"\nStopped: {msg}", file=sys.stderr)
                raise SystemExit(2) from exc
            answer = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        answer["seconds"] = round(time.time() - t0, 1)
        data["answers"][cid] = answer
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False))   # save after every case
        print("error" if "error" in answer else "ok", f"({answer['seconds']} s)", flush=True)
        time.sleep(delay)


# ── Report ───────────────────────────────────────────────────────────────────

def _pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f} %" if values else "—"


def _score_models(cases: list[dict]) -> list[dict]:
    models = []
    for path in sorted(RESPONSES.glob("*.json")):
        d = json.loads(path.read_text())
        rows = {}
        for case in cases:
            ans = d["answers"].get(case["id"])
            if ans is not None:
                rows[case["id"]] = {**score_case(case, ans.get("analysis")),
                                    "error": ans.get("error"), "seconds": ans.get("seconds")}
        if rows:
            models.append({"backend": d["backend"], "model": d["model"], "cases": rows})
    return models


def _section(title: str, intro: str, cases: list[dict], models: list[dict]) -> list[str]:
    lines = [f"## {title}", "", intro, "",
             "| Model | Cases | Solver | Geometry | Laminar/RANS | Steady/transient | Re (±15 %) | Viscosity | **Fully correct** | Median time |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        for stage in ("raw", "sanitised"):
            per_field = {f: [r[stage][f] for r in m["cases"].values() if f in r[stage]] for f in FIELDS}
            full = [all(r[stage].values()) for r in m["cases"].values()]
            secs = sorted(r["seconds"] for r in m["cases"].values() if r["seconds"] is not None)
            median = f"{secs[len(secs) // 2]:.0f} s" if secs else "—"
            name = f"`{m['model']}` ({m['backend']})" if stage == "raw" else "↳ after sanitiser"
            lines.append(f"| {name} | {len(m['cases'])} | " + " | ".join(_pct(per_field[f]) for f in FIELDS)
                         + f" | **{_pct(full)}** | {median if stage == 'raw' else ''} |")
    lines += ["", "Per case, after the sanitiser (✅ fully correct, ❌ wrong fields):", "",
              "| Case | " + " | ".join(f"`{m['model']}`" for m in models) + " |",
              "|---|" + "---|" * len(models)]
    for case in cases:
        cells = []
        for m in models:
            r = m["cases"].get(case["id"])
            if r is None:
                cells.append("—")
            elif r["error"]:
                cells.append("⚠ error")
            else:
                wrong = [f for f, ok in r["sanitised"].items() if not ok]
                cells.append("✅" if not wrong else "❌ " + ", ".join(wrong))
        lines.append(f"| {case['id']} | " + " | ".join(cells) + " |")
    return lines + [""]


def report(out_dir: Path) -> str:
    dev, holdout = _score_models(SETS["dev"]), _score_models(SETS["holdout"])
    lines = [
        "# LLM evaluation",
        "",
        "How accurately each model turns a plain-English description into a CFD setup. "
        "**Raw**: the model's answer as returned. **Sanitised**: after the validation layer "
        "(`app/sanitize.py`), i.e. what the app actually generates.",
        "",
    ]
    if holdout:
        lines += _section(
            f"Held-out set ({len(SETS['holdout'])} cases): unbiased estimate",
            "Written after the sanitiser rules were developed and committed before any model was "
            "run on them ([`evals/cases_holdout.json`](../../evals/cases_holdout.json)); the sanitiser "
            "was not changed after seeing these results.",
            SETS["holdout"], holdout)
    lines += _section(
        f"Development set ({len(SETS['dev'])} cases)",
        "Used to find weaknesses in the sanitiser ([`evals/cases.json`](../../evals/cases.json)), so "
        "the *after sanitiser* numbers here are optimistic.",
        SETS["dev"], dev)
    lines += ["## Notes", "",
              "- Sanitiser rules developed on the development set: solver/geometry conflicts, laminar "
              "vs RANS by Reynolds number, Ma > 0.3 needs a compressible solver, a stated Mach number "
              "sets the velocity. They are general physics, not tuned to particular sentences.",
              "- Not fixed on purpose: all local models misread the dev-set dam-break description (they "
              "used gravity, 9.81, as an inlet velocity). Catching that would need keyword rules fitted "
              "to test sentences, which would inflate the score without improving the app.",
              "- Local models ran on a laptop GPU (GTX 1650 Ti); cloud models are much faster."]
    text = "\n".join(lines) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(text)
    (out_dir / "results.json").write_text(json.dumps({"holdout": holdout, "dev": dev}, indent=1))
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", help="claude | ollama | gemini | groq | openrouter | openai-compatible")
    ap.add_argument("--model", help="model id")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests (free-tier rate limits)")
    ap.add_argument("--retry-errors", action="store_true", help="ask again for cases that failed before")
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--set", choices=["all", "dev", "holdout"], default="all",
                    help="which cases to ask about (already-cached answers are skipped)")
    ap.add_argument("--report", action="store_true", help="only re-score cached answers and write the report")
    ap.add_argument("--out", default=str(REPO / "docs" / "evaluation"))
    args = ap.parse_args(argv)

    if not args.report:
        if not (args.backend and args.model):
            ap.error("--backend and --model are required (or use --report)")
        cases = SETS["dev"] + SETS["holdout"] if args.set == "all" else SETS[args.set]
        print(f"Evaluating {args.model} ({args.backend}) on {len(cases)} cases")
        collect(args.backend, args.model, args.delay, args.retry_errors, args.only, cases)
    print(report(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
