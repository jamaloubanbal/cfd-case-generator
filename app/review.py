"""
Result reviewer: judges a case *after* it has run.

A run that finishes without errors can still be wrong: at the wrong inlet speed, not
converged, losing mass, or with a mesh the turbulence model's wall treatment does not
suit. ``review(case_dir)`` reads the results and computes, in Python:

* **request vs simulation**: the speed, Re and Mach in the description against the files;
* **convergence**: final residuals, whether the solver met its tolerances, the Courant number;
* **mass balance**: net flux through the open boundaries;
* **wall y⁺**: from the near-wall cells, against the model's valid range;
* **field sanity**: NaN, velocities far above the inflow, negative k, alpha out of [0, 1].

``explain(review, …)`` then asks an LLM (optional) for a verdict, a plain-language
explanation and concrete suggestions. ``write_report`` saves both to ``review.md``.

The request is read from ``cfdgen_analysis.json``, which the CLI and the web app save in
every generated case (``save_request``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.naming import is_inlet_name

REQUEST_FILE = "cfdgen_analysis.json"
_RANK = {"pass": 0, "info": 0, "warn": 1, "fail": 2}
_VERDICT = {0: "trustworthy", 1: "questionable", 2: "unreliable"}
_TRANSIENT = {"pimpleFoam", "interFoam"}
_SPEED_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:m/s|m\.s-1|m s-1|m·s⁻¹)", re.I)


@dataclass
class Check:
    name: str
    status: str            # pass | info | warn | fail
    message: str


@dataclass
class Review:
    case_dir: Path
    checks: list[Check] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    explanation: dict | None = None        # LLM: verdict, summary, suggestions

    @property
    def verdict(self) -> str:
        worst = max((_RANK[c.status] for c in self.checks), default=0)
        if self.explanation and self.explanation.get("verdict") in _VERDICT.values():
            llm = {v: k for k, v in _VERDICT.items()}[self.explanation["verdict"]]
            worst = max(worst, llm)        # the LLM may be stricter, never more lenient
        return _VERDICT[worst]

    def add(self, name: str, status: str, message: str) -> None:
        self.checks.append(Check(name, status, message))


# ── Saving the request with the case ─────────────────────────────────────────

def save_request(case_dir: Path, analysis: dict, description: str | None) -> None:
    """Store the (sanitised) analysis and the user's description in the case."""
    clean = {k: v for k, v in analysis.items() if not k.startswith("_")}
    data = {"description": description or "", "analysis": clean}
    Path(case_dir, REQUEST_FILE).write_text(json.dumps(data, indent=2, default=str))


def _load_request(case_dir: Path) -> tuple[dict, str]:
    p = case_dir / REQUEST_FILE
    if not p.exists():
        return {}, ""
    data = json.loads(p.read_text())
    return data.get("analysis") or {}, data.get("description") or ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _application(case_dir: Path) -> str:
    m = re.search(r"^\s*application\s+(\w+)\s*;", (case_dir / "system/controlDict").read_text(), re.M)
    return m.group(1) if m else ""


def _nu(case_dir: Path, analysis: dict) -> float | None:
    tp = case_dir / "constant/transportProperties"
    if tp.exists():
        m = re.search(r"^\s*nu\s+(?:\[[^\]]*\]\s*)?([\d.eE+\-]+)\s*;", tp.read_text(), re.M)
        if m:
            return float(m.group(1))
    nu = (analysis.get("fluid") or {}).get("nu")
    return float(nu) if nu else None


def _stated_speed(description: str) -> float | None:
    speeds = [float(s.replace(",", ".")) for s in _SPEED_RE.findall(description or "")]
    return speeds[0] if len(set(speeds)) == 1 else None      # ambiguous with several speeds


def _last_float(pattern: str, text: str) -> float | None:
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None


# ── The review ───────────────────────────────────────────────────────────────

def review(case_dir: str | Path) -> Review:
    """Deterministic review of a case that has run."""
    from app.postprocess import FoamCase
    from app.log_parser import parse_foam_log

    case_dir = Path(case_dir)
    r = Review(case_dir)
    analysis, description = _load_request(case_dir)
    app = _application(case_dir)
    log_path = case_dir / f"log.{app}"
    if not log_path.exists():
        r.add("Run", "fail", f"No solver log (log.{app}): the case has not been run.")
        return r
    log = log_path.read_text(errors="replace")
    case = FoamCase(case_dir)
    t = case.latest_time()
    if float(t) == 0:
        r.add("Run", "fail", "The solver wrote no results (only the 0/ directory exists).")
        return r
    r.facts.update(solver=app, latest_time=float(t), cells=case.n_cells)
    transient = app in _TRANSIENT or bool(analysis.get("is_transient"))

    _check_convergence(r, log, transient, parse_foam_log(log), _idle_components(case, t))
    conv = r.checks[-1]
    if (not transient and conv.name == "Convergence" and conv.status in ("warn", "fail")
            and analysis.get("geometry_type") in ("cylinder", "rectangular_obstacle")):
        conv.message += (" Likely cause: the wake of the bluff body sheds vortices, so the flow has "
                         "no steady solution; use a transient solver (pimpleFoam).")
    inlet_speed = _check_request(r, case, analysis, description, case_dir)
    _check_mass(r, case, t)
    _check_yplus(r, case, t, analysis, case_dir)
    u_ref = inlet_speed or _buoyant_velocity(case, t, analysis) or analysis.get("characteristic_velocity")
    _check_fields(r, case, t, u_ref)
    return r


def _idle_components(case, t: str) -> set[str]:
    """Velocity components that are ~0 everywhere (e.g. Uz on a wedge): their normalised
    residuals are noise and say nothing about convergence."""
    try:
        U = np.atleast_2d(case.field("U", t))
    except Exception:
        return set()
    scale = float(np.abs(U).max()) or 1.0
    return {f"U{c}" for i, c in enumerate("xyz") if float(np.abs(U[:, i]).max()) < 1e-3 * scale}


def _check_convergence(r: Review, log: str, transient: bool, parsed: dict,
                       idle: set[str] = frozenset()) -> None:
    finished = bool(re.search(r"^End\s*$", log, re.M))
    final = {f: v[-1] for f, v in parsed["residuals"].items() if v and f not in idle}
    n_steps = len(parsed["times"]) or parsed["n_iters"]
    r.facts.update(iterations=n_steps,
                   final_residuals={f: float(f"{v:.3g}") for f, v in final.items()})
    if not finished:
        r.add("Convergence", "fail", "The solver log has no final 'End': the run crashed or was stopped.")
        return
    if any(not math.isfinite(v) for v in final.values()):
        r.add("Convergence", "fail", "Residuals became NaN or infinite.")
        return
    worst_f, worst = max(final.items(), key=lambda kv: kv[1], default=("", 0.0))
    if transient:
        co = _last_float(r"Courant Number mean:\s*[\d.eE+\-]+\s*max:\s*([\d.eE+\-]+)", log)
        r.facts["max_courant"] = co
        if co is not None and co > 1.0:
            r.add("Time step", "warn", f"Final Courant number {co:.2g} > 1: time accuracy is poor; "
                                       "lower maxCo or deltaT.")
        else:
            r.add("Time step", "pass", f"Final Courant number {co:.2g}." if co is not None else
                  "Courant number not reported.")
        # Per-step initial residuals measure the change over a step (large in vortex shedding),
        # not convergence; above ~0.1 they signal a time step that is too large.
        r.add("Convergence", "pass" if worst < 0.1 else "warn",
              f"Transient run finished; largest initial residual in the last step {worst_f} = {worst:.2g}"
              + ("." if worst < 0.1 else ": the time step is probably too large (lower maxCo)."))
        return
    if re.search(r"solution converged in", log):
        r.add("Convergence", "pass", f"Steady solution converged in {n_steps} iterations "
                                     f"(residual control met; largest {worst_f} = {worst:.2g}).")
    elif worst < 1e-3:
        r.add("Convergence", "pass", f"Reached endTime with all residuals below 1e-3 "
                                     f"(largest {worst_f} = {worst:.2g}).")
    elif worst < 1e-1:
        r.add("Convergence", "warn", f"Reached endTime without converging: {worst_f} residual "
                                     f"{worst:.2g}. Run longer or check the mesh and schemes.")
    else:
        r.add("Convergence", "fail", f"Not converged: {worst_f} residual {worst:.2g} at the end.")


def _inlet_patches(case) -> list[str]:
    return [p for p, b in case.boundary.items() if b["type"] == "patch" and is_inlet_name(p)]


def _check_request(r: Review, case, analysis: dict, description: str, case_dir: Path) -> float | None:
    """Speed / Re / Mach in the description against what the files actually simulate."""
    from app.sanitize import stated_mach, stated_reynolds

    inlets = _inlet_patches(case)
    if not inlets and analysis.get("geometry_type") == "imported_mesh":
        inlets = [p for p, b in case.boundary.items() if b["type"] == "patch"
                  and "fixedValue" in _patch_type(case_dir, p)]
    speeds = []
    for p in inlets:
        try:
            U = np.atleast_2d(case.patch_field("U", "0", p))
            speeds.append(float(np.linalg.norm(U, axis=1).mean()))
        except Exception:
            continue
    speed = max(speeds) if speeds else None
    if not speed:
        r.add("Request", "info", "No inflow boundary: speed, Re and Mach are not compared.")
        return None
    r.facts["inlet_speed"] = round(speed, 6)
    if not description:
        r.add("Request", "info", f"Inlet speed {speed:.4g} m/s (no description saved with the case).")
        return speed

    problems, ok = [], []
    want = _stated_speed(description)
    if want:
        (ok if abs(speed / want - 1) < 0.05 else problems).append(
            f"inlet speed {speed:.4g} m/s, description says {want:g} m/s")
    nu, L = _nu(case_dir, analysis), analysis.get("characteristic_length")
    if nu and L:
        re_sim = speed * float(L) / nu
        r.facts["Re"] = float(f"{re_sim:.4g}")
        re_want = stated_reynolds(description)
        if re_want:
            (ok if abs(re_sim / re_want - 1) < 0.05 else problems).append(
                f"Re {re_sim:.4g}, description says {re_want:g}")
    T = float(analysis.get("T_ref") or 300.0)
    ma_sim = speed / math.sqrt(1.4 * 287.0 * T)
    r.facts["Ma"] = float(f"{ma_sim:.3g}")
    ma_want = stated_mach(description)
    if ma_want:
        (ok if abs(ma_sim / ma_want - 1) < 0.05 else problems).append(
            f"Mach {ma_sim:.3g}, description says {ma_want:g}")
    if problems:
        r.add("Request", "fail", "The simulation does not match the description: " + "; ".join(problems) + ".")
    elif ok:
        r.add("Request", "pass", "Matches the description: " + "; ".join(ok) + ".")
    else:
        r.add("Request", "info", f"Inlet speed {speed:.4g} m/s; the description states no speed, Re or Mach.")
    return speed


def _patch_type(case_dir: Path, patch: str) -> str:
    text = (case_dir / "0/U").read_text()
    m = re.search(rf"\b{re.escape(patch)}\s*\{{[^}}]*?type\s+(\w+)", text, re.S)
    return m.group(1) if m else ""


def _buoyant_velocity(case, t: str, analysis: dict) -> float | None:
    """Free-fall velocity sqrt(g·ΔT/T·L) for natural convection (no inflow to compare with)."""
    if not (case.path / t / "T").exists():
        return None
    T = np.asarray(case.field("T", t), dtype=float)
    L = float(analysis.get("characteristic_length") or 0) or float(np.ptp(case.points[:, 1]))
    dT, Tm = float(np.ptp(T)), float(T.mean())
    return math.sqrt(9.81 * dT / Tm * L) if dT > 0 and Tm > 0 else None


def _check_mass(r: Review, case, t: str) -> None:
    open_patches = [p for p, b in case.boundary.items() if b["type"] == "patch" and b["nFaces"]]
    if not (case.path / t / "phi").exists() or not open_patches:
        return
    flux = {p: float(np.sum(case.patch_field("phi", t, p))) for p in open_patches}
    inflow = -sum(min(f, 0.0) for f in flux.values())
    if inflow <= 0 or len(open_patches) < 2:
        return             # no through-flow (e.g. a closed tank venting to the atmosphere)
    imbalance = abs(sum(flux.values())) / inflow
    r.facts["mass_imbalance"] = float(f"{imbalance:.3g}")
    msg = f"Net flux through the open boundaries is {imbalance:.2%} of the inflow."
    r.add("Mass balance", "pass" if imbalance < 0.01 else "warn" if imbalance < 0.05 else "fail",
          msg if imbalance < 0.01 else msg + " The solution is not converged or the outlet is reversed.")


def _check_yplus(r: Review, case, t: str, analysis: dict, case_dir: Path) -> None:
    model = analysis.get("turb_model") or ""
    tp = case_dir / "constant/turbulenceProperties"
    if tp.exists() and re.search(r"simulationType\s+laminar", tp.read_text()):
        model = "laminar"
    walls = [p for p, b in case.boundary.items() if b["type"] == "wall" and b["nFaces"]]
    nu = _nu(case_dir, analysis)
    if model == "laminar" or not walls or not nu:
        return
    try:
        U = case.field("U", t)
        k = case.field("k", t) if (case.path / t / "k").exists() else None
        cc, fc, fa, owner = case.cell_centres, case.face_centres, case.face_areas, case.owner
    except Exception:
        return
    yp, area = [], []
    for p in walls:
        s = case.patch_faces(p)
        n = fa[s] / np.linalg.norm(fa[s], axis=1, keepdims=True)
        own = owner[s]
        d = np.abs(np.einsum("ij,ij->i", cc[own] - fc[s], n))
        if k is not None:
            u_tau = 0.09 ** 0.25 * np.sqrt(np.clip(k[own], 0, None))
        else:
            Uw = np.atleast_2d(case.patch_field("U", t, p))
            du = U[own] - Uw
            ut = np.linalg.norm(du - np.einsum("ij,ij->i", du, n)[:, None] * n, axis=1)
            u_tau = np.sqrt(nu * ut / np.maximum(d, 1e-30))
        yp.append(d * u_tau / nu)
        area.append(np.linalg.norm(fa[s], axis=1))
    yp, area = np.concatenate(yp), np.concatenate(area)
    mean, p95 = float(np.average(yp, weights=area)), float(np.percentile(yp, 95))
    r.facts["y_plus"] = {"mean": round(mean, 2), "p95": round(p95, 2)}
    span = f"mean {mean:.3g}, 95th percentile {p95:.3g}"
    if model == "kEpsilon":
        if mean < 11:
            r.add("Wall y⁺", "warn", f"y⁺ {span}: too fine for k-ε wall functions (30–300). "
                                     "Use kOmegaSST or a coarser first cell.")
        elif mean > 300:
            r.add("Wall y⁺", "warn", f"y⁺ {span}: above the wall-function range (30–300); refine near walls.")
        else:
            r.add("Wall y⁺", "pass", f"y⁺ {span}: suitable for k-ε wall functions.")
    else:                      # kOmegaSST, SpalartAllmaras: resolved (≤ 5) or wall functions
        if mean <= 5:
            r.add("Wall y⁺", "pass", f"y⁺ {span}: the viscous sublayer is resolved.")
        elif mean > 300:
            r.add("Wall y⁺", "warn", f"y⁺ {span}: above 300; refine near walls.")
        elif mean < 30:
            r.add("Wall y⁺", "info", f"y⁺ {span}: in the buffer layer; {model}'s automatic wall "
                                     "treatment copes, but y⁺ < 5 or 30–300 is more accurate.")
        else:
            r.add("Wall y⁺", "pass", f"y⁺ {span}: in the wall-function range.")


def _check_fields(r: Review, case, t: str, u_ref: float | None) -> None:
    problems = []
    U = case.field("U", t)
    Umag = np.linalg.norm(np.atleast_2d(U), axis=1)
    if not np.all(np.isfinite(Umag)):
        r.add("Fields", "fail", "The velocity field contains NaN or infinite values.")
        return
    r.facts["max_U"] = float(f"{Umag.max():.4g}")
    if u_ref and Umag.max() > 10 * float(u_ref):
        problems.append(f"max |U| {Umag.max():.3g} m/s is over 10× the reference {float(u_ref):.3g} m/s")
    for name, lo, hi in (("k", 0.0, None), ("alpha.water", -1e-3, 1 + 1e-3)):
        if (case.path / t / name).exists():
            v = np.asarray(case.field(name, t), dtype=float)
            if lo is not None and v.min() < lo:
                problems.append(f"{name} has values down to {v.min():.3g}")
            if hi is not None and v.max() > hi:
                problems.append(f"{name} has values up to {v.max():.3g}")
    if problems:
        r.add("Fields", "warn", "Suspicious values: " + "; ".join(problems) + ".")
    else:
        r.add("Fields", "pass", f"Values are finite and plausible (max |U| {Umag.max():.3g} m/s).")


# ── LLM explanation ──────────────────────────────────────────────────────────

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(_VERDICT.values())},
        "summary": {"type": "string"},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "suggestions"],
    "additionalProperties": False,
}

_EXPLAIN_SYSTEM = (
    "You are a senior CFD engineer reviewing an OpenFOAM result for a user who is not a CFD "
    "expert. You get the user's request, the case settings and automatic checks computed from "
    "the results. Give a verdict (trustworthy, questionable or unreliable), a short plain-language "
    "summary of what the simulation shows and how far it can be trusted (3–5 sentences), and at "
    "most 4 concrete, actionable suggestions (settings to change, with values). Rely on the "
    "checks; do not invent numbers that are not given."
)


def explain(rev: Review, backend: str, model: str | None = None, api_key: str | None = None,
            base_url: str | None = None) -> dict:
    """Ask an LLM to judge and explain the review; stores and returns the explanation."""
    from app.llm import ask_json

    analysis, description = _load_request(rev.case_dir)
    keep = ("solver", "turb_model", "geometry_type", "fluid", "Re", "characteristic_length",
            "characteristic_velocity", "is_transient")
    payload = {
        "request": description,
        "settings": {k: analysis.get(k) for k in keep if k in analysis},
        "results": rev.facts,
        "checks": [{"check": c.name, "status": c.status, "detail": c.message} for c in rev.checks],
    }
    out = ask_json(_EXPLAIN_SYSTEM, json.dumps(payload, indent=1, default=str), EXPLAIN_SCHEMA,
                   backend=backend, model=model, api_key=api_key, base_url=base_url)
    rev.explanation = {"verdict": out.get("verdict"), "summary": str(out.get("summary", "")),
                       "suggestions": [str(s) for s in (out.get("suggestions") or [])][:4]}
    return rev.explanation


# ── Report ───────────────────────────────────────────────────────────────────

_ICON = {"pass": "✔", "info": "ℹ", "warn": "⚠", "fail": "✘"}


def to_markdown(rev: Review) -> str:
    lines = [f"# Result review — {rev.verdict}", "", "| | Check | Result |", "|---|---|---|"]
    lines += [f"| {_ICON[c.status]} | {c.name} | {c.message} |" for c in rev.checks]
    if rev.explanation:
        lines += ["", "## Explanation", "", rev.explanation["summary"]]
        if rev.explanation["suggestions"]:
            lines += ["", "## Suggestions", ""] + [f"- {s}" for s in rev.explanation["suggestions"]]
    lines += ["", "<details><summary>Numbers</summary>", "", "```json",
              json.dumps(rev.facts, indent=2), "```", "</details>", ""]
    return "\n".join(lines)


def write_report(rev: Review) -> Path:
    path = rev.case_dir / "review.md"
    path.write_text(to_markdown(rev))
    return path


def print_review(rev: Review, out=print) -> None:
    out(f"Result review: {rev.verdict.upper()}")
    for c in rev.checks:
        out(f"  {_ICON[c.status]} {c.name}: {c.message}")
    if rev.explanation:
        out("  " + rev.explanation["summary"])
        for s in rev.explanation["suggestions"]:
            out(f"  → {s}")
