"""
Run-and-fix agent: run a case, diagnose a failure from the OpenFOAM logs, fix it, rerun.

    result = repair("path/to/case", backend="gemini")       # rules first, then the LLM
    print(result.success, len(result.attempts))

Two layers of fixes, tried in order on every failed run:

1. **Rules** — deterministic fixes for failures whose remedy is known: a missing
   ``<field>Final`` or ``pcorr`` solver entry, a missing divergence scheme, divergence of a
   steady solver (stronger under-relaxation, pressure limits) or of a transient one
   (smaller Courant number).
2. **LLM** — the error, the end of the log and the relevant dictionaries are sent to a
   model, which returns one corrected file.

Safety: the LLM may only rewrite existing dictionaries under ``system/``, ``constant/`` or
``0/`` (never ``Allrun`` or the mesh); content that can execute code (``#codeStream``,
``coded…`` boundary conditions, ``systemCall``) is rejected, and so is a "fix" that
shortens the run (changed ``endTime``).  Every attempt is recorded with a diff in
``agent_report.md``.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MAX_ATTEMPTS = 3
_LOG_TAIL_LINES = 60
_MAX_FILE_CHARS = 12_000
_FORBIDDEN = ("#codeStream", "#calc", "codedFixedValue", "codedMixed", "codedFixedValue",
              "systemCall", "coded;", "#includeFunc", "#includeEtc")


# ── Running and diagnosing ───────────────────────────────────────────────────

@dataclass
class Diagnosis:
    ok: bool
    application: str | None = None     # failing OpenFOAM application, e.g. "simpleFoam"
    kind: str = ""                     # missing_entry | divergence | patch_mismatch | syntax | other
    message: str = ""                  # the FOAM FATAL message (or the reason for failure)
    file: str | None = None            # case-relative file named in the error, if any
    entry: str | None = None           # missing entry name, if any
    log_tail: str = ""


def shorten_run(case_dir: Path, steady_iterations: int = 60, transient_steps: int = 15) -> None:
    """Limit a case to a quick check: a few iterations (steady) or time steps (transient)."""
    cd = case_dir / "system" / "controlDict"
    t = cd.read_text()
    dt = float(re.search(r"^deltaT\s+([^;]+);", t, re.M).group(1))
    transient = re.search(r"^application\s+(pimpleFoam|interFoam);", t, re.M)
    end = (transient_steps if transient else steady_iterations) * dt
    t = re.sub(r"^endTime\s+[^;]+;", f"endTime         {end:g};", t, flags=re.M)
    t = re.sub(r"^writeControl\s+[^;]+;", "writeControl    runTime;", t, flags=re.M)
    t = re.sub(r"^writeInterval\s+[^;]+;", f"writeInterval   {end:g};", t, flags=re.M)
    cd.write_text(t)


def clean(case_dir: Path) -> None:
    """Remove results and logs from a previous run (keeps 0/, constant/, system/)."""
    for p in case_dir.iterdir():
        name = p.name
        is_time = re.fullmatch(r"\d+(\.\d+)?(e[-+]?\d+)?", name) and name != "0"
        if (p.is_dir() and (is_time or name.startswith("processor") or name == "postProcessing")) or \
                (p.is_file() and name.startswith("log.")):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    # Remove the mesh only when the case rebuilds it (blockMesh); an imported mesh must stay
    poly = case_dir / "constant" / "polyMesh"
    if poly.exists() and (case_dir / "system" / "blockMeshDict").exists():
        shutil.rmtree(poly)


def run(case_dir: Path, timeout: int = 1800) -> int:
    clean(case_dir)
    try:
        return subprocess.run(["./Allrun"], cwd=case_dir, capture_output=True, text=True,
                              timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def _fatal_block(text: str) -> str:
    m = re.search(r"--> FOAM FATAL (?:IO )?ERROR:?\s*(.*?)(?:\n\s*From function|\nFOAM (?:exiting|aborting))",
                  text, re.S)
    return " ".join(m.group(1).split()) if m else ""


def diagnose(case_dir: Path, returncode: int) -> Diagnosis:
    logs = sorted(case_dir.glob("log.*"), key=lambda p: p.stat().st_mtime)
    for log in reversed(logs):                          # the last log written failed first
        text = log.read_text(errors="replace")
        tail = "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
        app = log.name.removeprefix("log.")
        fatal = _fatal_block(text)
        # Real crash signatures only: every log's banner says "trapFpe: Floating point
        # exception trapping enabled", which must not count as a crash.
        diverged = re.search(r"Negative initial temperature|sigFpe::sigHandler|"
                             r"Initial residual = (?:-?nan|-?inf)|Maximum number of iterations exceeded",
                             text, re.I)
        if not fatal and not diverged:
            continue
        message = fatal or ("floating-point crash (division by zero or overflow)"
                            if "sigFpe" in diverged.group(0) else diverged.group(0))
        d = Diagnosis(ok=False, application=app, message=message, log_tail=tail)
        fm = re.search(r"file: \S*?/((?:system|constant|0)/[\w.]+)", text)
        if fm:   # "system/fvSolution.solvers" → "system/fvSolution", "0/p.boundaryField…" → "0/p"
            rel = re.sub(r"\.(boundaryField|internalField|dimensions)\b.*$", "", fm.group(1))
            d.file = rel if rel.startswith("0/") else rel.split(".")[0]
        em = re.search(r"(?:Entry|keyword) '([^']+)' (?:is )?(?:not found|undefined)", fatal)
        em = em or re.search(r"Essential entry '([^']+)' missing", fatal)
        if em:
            d.kind, d.entry = "missing_entry", em.group(1)
        elif "inconsistent patch and patchField" in fatal:
            d.kind = "patch_mismatch"
        elif re.search(r"parse|Unexpected|expected|keyword \w+ is undefined|bad input", fatal, re.I):
            d.kind = "syntax"
        elif diverged and not fatal or "Negative initial temperature" in fatal:
            d.kind = "divergence"
        else:
            d.kind = "other"
        return d
    if returncode != 0:
        return Diagnosis(ok=False, kind="other", message=f"Allrun exited with code {returncode}",
                         log_tail="\n".join((logs[-1].read_text(errors='replace') if logs else "").splitlines()[-_LOG_TAIL_LINES:]))
    return Diagnosis(ok=True)


# ── Rule-based fixes ─────────────────────────────────────────────────────────

def _solvers_block(text: str) -> tuple[int, int] | None:
    m = re.search(r"^solvers\s*\{", text, re.M)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        i += 1
    return m.end(), i - 1                                # body between the braces


def _add_to_block(path: Path, block: str, entry: str) -> bool:
    """Append *entry* at the end of the top-level *block* (after the entries it may reference)."""
    text = path.read_text()
    m = re.search(rf"^{block}\s*\{{", text, re.M)
    if not m:
        return False
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        i += 1
    end = i - 1                                        # index of the block's closing brace
    path.write_text(text[:end] + entry + text[end:])
    return True


def _scale_relaxation(text: str, factor: float) -> str:
    """Stronger under-relaxation for a diverging SIMPLE solver.

    Equation factors (U, h, k, …) are multiplied by *factor*; the pressure field factor
    follows the SIMPLE rule of thumb  alpha_p ≈ 1 − alpha_U  (between 0.1 and 0.7).
    """
    rm = re.search(r"^relaxationFactors\s*\{", text, re.M)
    if not rm:
        return text
    head, body = text[:rm.end()], text[rm.end():]
    eq = re.search(r"equations\s*\{(.*?)\}", body, re.S)
    if eq:
        new_eq = re.sub(r"^(\s+[\w\"().|*]+\s+)(\d*\.?\d+)\s*;",
                        lambda m: f"{m.group(1)}{max(0.05, round(float(m.group(2)) * factor, 3)):g};",
                        eq.group(1), flags=re.M)
        body = body[:eq.start(1)] + new_eq + body[eq.end(1):]
        um = re.search(r"^\s+U\s+(\d*\.?\d+)\s*;", new_eq, re.M)
        if um:
            alpha_p = min(0.7, max(0.1, round(1.0 - float(um.group(1)), 2)))
            body = re.sub(r"^(\s+(?:p|p_rgh)\s+)(\d*\.?\d+)\s*;", rf"\g<1>{alpha_p:g};", body,
                          count=1, flags=re.M)
    return head + body


def rule_fix(case_dir: Path, d: Diagnosis) -> list[str]:
    """Apply known fixes for *d*; returns descriptions of what was changed (empty: none)."""
    changes: list[str] = []
    fv, fs, cd = (case_dir / "system" / n for n in ("fvSolution", "fvSchemes", "controlDict"))

    if d.kind == "missing_entry" and d.entry:
        e = d.entry
        if (d.file or "").endswith("fvSolution") and e.endswith("Final"):
            base = e[:-5]
            if base.startswith("pcorr"):
                entry = f'    "{e}"\n    {{\n        solver PCG;\n        preconditioner DIC;\n        tolerance 1e-5;\n        relTol 0;\n    }}\n'
            elif re.search(rf'^\s*"?{re.escape(base)}"?\s*$', fv.read_text(), re.M):
                entry = f'    {e}\n    {{\n        ${base};\n        relTol 0;\n    }}\n'
            else:
                entry = (f'    {e}\n    {{\n        solver smoothSolver;\n        smoother symGaussSeidel;\n'
                         f'        tolerance 1e-8;\n        relTol 0;\n    }}\n')
            if _add_to_block(fv, "solvers", entry):
                changes.append(f"system/fvSolution: added solver entry {e}")
        elif (d.file or "").endswith("fvSolution") and e.startswith("pcorr"):
            entry = f'    "{e}.*"\n    {{\n        solver PCG;\n        preconditioner DIC;\n        tolerance 1e-5;\n        relTol 0;\n    }}\n'
            if _add_to_block(fv, "solvers", entry):
                changes.append(f"system/fvSolution: added solver entry {e}")
        elif (d.file or "").endswith("fvSchemes") and e.startswith("div("):
            scheme = "Gauss linear" if "phirb" in e or "dev" in e else "Gauss upwind"
            if _add_to_block(fs, "divSchemes", f"    {e} {scheme};\n"):
                changes.append(f"system/fvSchemes: added {e} {scheme}")

    elif d.kind == "divergence":
        text = cd.read_text() if cd.exists() else ""
        transient = re.search(r"^application\s+(pimpleFoam|interFoam);", text, re.M)
        if transient:
            new = re.sub(r"^adjustTimeStep\s+\w+;", "adjustTimeStep  yes;", text, flags=re.M)
            new = re.sub(r"^maxCo\s+[\d.eE+-]+;", "maxCo           0.25;", new, flags=re.M)
            new = re.sub(r"^(deltaT\s+)([\d.eE+-]+);", lambda m: f"{m.group(1)}{float(m.group(2)) / 4:g};",
                         new, flags=re.M)
            if new != text:
                cd.write_text(new)
                changes.append("system/controlDict: adjustable time step, maxCo 0.25, deltaT / 4")
        else:
            sol = fv.read_text()
            new = _scale_relaxation(sol, 0.6)
            new = re.sub(r"^(\s*)consistent\s+(yes|true|on);", r"\1consistent      no;", new, flags=re.M)
            if re.search(r"^application\s+rhoSimpleFoam;", text, re.M) and "pMinFactor" not in new:
                new = re.sub(r"^(SIMPLE\s*\{)", r"\1\n    pMinFactor      0.5;\n    pMaxFactor      2.0;",
                             new, count=1, flags=re.M)
            if new != sol:
                fv.write_text(new)
                changes.append("system/fvSolution: relaxation factors × 0.6 (and pressure limits if compressible)")
    return changes


# ── LLM fix ──────────────────────────────────────────────────────────────────

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "file": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["diagnosis", "file", "content"],
    "additionalProperties": False,
}

_FIX_SYSTEM = """\
You are an OpenFOAM v1912 (ESI, openfoam.com) expert. A generated case failed to run. You
get the failing application, the error, the end of its log and the relevant case files.
Find the cause and return the COMPLETE corrected content of exactly ONE file:
- "file": the case-relative path of an existing file under system/, constant/ or 0/
  (not constant/polyMesh/ and not Allrun).
- "content": the full new file, keeping its FoamFile header; change as little as possible.
- "diagnosis": one sentence explaining the cause.
Do not change endTime, do not use #codeStream, #calc, coded boundary conditions or systemCall.
"""


def _relevant_files(case_dir: Path, d: Diagnosis) -> list[str]:
    files = [d.file] if d.file else []
    pm = re.search(r"patch (\w+)", d.message)
    field_m = re.search(r"field (\w+(?:\.\w+)?)", d.message)
    if field_m and (case_dir / "0" / field_m.group(1)).exists():
        files.append(f"0/{field_m.group(1)}")
    if d.kind == "patch_mismatch" or pm:
        files.append("constant/polyMesh/boundary")
    files += ["system/fvSolution", "system/fvSchemes", "system/controlDict"]
    seen, out = set(), []
    for f in files:
        if f and f not in seen and (case_dir / f).exists():
            seen.add(f)
            out.append(f)
    return out[:5]


def _match_sent_file(rel: str, sent: list[str]) -> str:
    """Map a near-miss path to the file the model was shown, if unambiguous.

    Small models often get the diagnosis right but the path wrong, e.g. "0/fvSchemes" for
    "system/fvSchemes" or "0/p.boundaryField.outlet" for "0/p".  Unknown names stay as they
    are (and are then rejected).
    """
    if rel in sent:
        return rel
    name = Path(rel).name
    name = re.sub(r"\.(boundaryField|internalField|dimensions)\b.*$", "", name)
    candidates = [s for s in sent if Path(s).name in (name, name.split(".")[0])]
    return candidates[0] if len(candidates) == 1 else rel


def _reject_reason(case_dir: Path, rel: str, content: str) -> str | None:
    target = (case_dir / rel).resolve()
    if case_dir.resolve() not in target.parents:
        return "path outside the case"
    parts = Path(rel).parts
    if not parts or parts[0] not in ("system", "constant", "0") or "polyMesh" in parts:
        return f"{rel} may not be edited"
    if not target.is_file():
        return f"{rel} does not exist"
    if "FoamFile" not in content or len(content) > 4 * _MAX_FILE_CHARS:
        return "content is not a plausible OpenFOAM dictionary"
    if any(f in content for f in _FORBIDDEN):
        return "content contains a code-execution directive"
    if rel == "system/controlDict":
        old = re.search(r"^endTime\s+([^;]+);", target.read_text(), re.M)
        new = re.search(r"^endTime\s+([^;]+);", content, re.M)
        if not new or (old and old.group(1).strip() != new.group(1).strip()):
            return "changing endTime is not a fix"
    return None


def llm_fix(case_dir: Path, d: Diagnosis, backend: str, model: str | None = None,
            api_key: str | None = None, base_url: str | None = None) -> tuple[list[str], str]:
    """Ask the LLM for one corrected file.  Returns (changes, diagnosis or rejection reason)."""
    from app.llm import ask_json

    parts = [f"Failing application: {d.application}", f"Error: {d.message}",
             f"End of log.{d.application}:\n{d.log_tail}"]
    sent = _relevant_files(case_dir, d)
    for rel in sent:
        text = (case_dir / rel).read_text(errors="replace")[:_MAX_FILE_CHARS]
        parts.append(f"=== {rel} ===\n{text}")
    answer = ask_json(_FIX_SYSTEM, "\n\n".join(parts), FIX_SCHEMA, backend=backend, model=model,
                      api_key=api_key, base_url=base_url)
    rel = _match_sent_file(str(answer.get("file", "")).strip().lstrip("./"), sent)
    content = answer.get("content", "")
    reason = _reject_reason(case_dir, rel, content)
    if reason:
        return [], f"rejected LLM edit ({reason}): {answer.get('diagnosis', '')}"
    path = case_dir / rel
    old = path.read_text()
    if old == content:
        return [], f"LLM proposed no change: {answer.get('diagnosis', '')}"
    path.write_text(content)
    diff = "".join(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                        f"a/{rel}", f"b/{rel}", n=1))
    return [f"{rel} (LLM)\n```diff\n{diff}```"], answer.get("diagnosis", "")


# ── The loop ─────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    diagnosis: Diagnosis
    fixer: str = ""
    changes: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RepairResult:
    success: bool
    attempts: list[Attempt]


def repair(case_dir: str | Path, *, backend: str | None = None, model: str | None = None,
           api_key: str | None = None, base_url: str | None = None, use_rules: bool = True,
           max_attempts: int = MAX_ATTEMPTS, timeout: int = 1800, log=print) -> RepairResult:
    """Run the case; while it fails, apply rule fixes, else an LLM fix (if *backend*), rerun."""
    case_dir = Path(case_dir)
    attempts: list[Attempt] = []
    rules_tried: dict[tuple[str, str | None], int] = {}   # how often a rule was tried per failure kind
    for i in range(max_attempts + 1):
        d = diagnose(case_dir, run(case_dir, timeout))
        if d.ok:
            log(f"✔ case runs{' after ' + str(i) + ' fix(es)' if i else ''}")
            result = RepairResult(True, attempts)
            break
        log(f"✖ {d.application}: {d.kind} — {d.message[:160]}")
        if i == max_attempts:
            result = RepairResult(False, attempts)
            break
        att = Attempt(diagnosis=d)
        # Rules are tried once per failure kind (twice for divergence, whose fix is progressive);
        # if the same failure persists, the LLM gets the next attempt.
        limit = 2 if d.kind == "divergence" else 1
        if use_rules and rules_tried.get((d.kind, d.entry), 0) < limit:
            rules_tried[(d.kind, d.entry)] = rules_tried.get((d.kind, d.entry), 0) + 1
            att.changes, att.fixer = rule_fix(case_dir, d), "rules"
        if not att.changes and backend:
            try:
                att.changes, att.note = llm_fix(case_dir, d, backend, model, api_key, base_url)
                att.fixer = f"LLM ({backend})"
            except Exception as exc:                # LLM unavailable: stop, report why
                att.note, att.fixer = f"LLM call failed: {exc}", f"LLM ({backend})"
        attempts.append(att)
        for c in att.changes:
            log(f"  → {att.fixer}: {c.splitlines()[0]}")
        if not att.changes:
            log(f"  → no fix found ({att.note or 'no rule applies and no LLM backend given'})")
            result = RepairResult(False, attempts)
            break
    _write_report(case_dir, result)
    return result


def _write_report(case_dir: Path, result: RepairResult) -> None:
    lines = ["# Repair agent report", "",
             f"**Result:** {'the case runs' if result.success else 'not repaired'} "
             f"after {len(result.attempts)} fix attempt(s).", ""]
    for n, a in enumerate(result.attempts, 1):
        lines += [f"## Attempt {n}: {a.fixer or '—'}", "",
                  f"- **Failure:** `{a.diagnosis.application}` — {a.diagnosis.kind}: {a.diagnosis.message}",
                  *( [f"- **Diagnosis:** {a.note}"] if a.note else [] ), ""]
        lines += [f"- {c}" for c in a.changes] + [""]
    (case_dir / "agent_report.md").write_text("\n".join(lines))
