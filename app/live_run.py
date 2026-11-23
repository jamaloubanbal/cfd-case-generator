"""
A case run in a background thread, so the web app can show it live.

``RunJob`` runs an OpenFOAM case through the repair agent (or a code_saturne study) and
collects, while it runs, one record per time step from the solver output. Each run attempt
of the repair loop is kept as a *segment*, labelled with the fix the agent applied before the
next attempt, so the monitor can show where the agent stepped in. No Streamlit calls here:
the UI only reads :meth:`RunJob.snapshot`.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.log_parser import LogTail

CONVERGED = 1e-4          # steady: all initial residuals below this


@dataclass
class Segment:
    steps: list[dict]
    fix: str | None = None          # what the agent changed before the next attempt


@dataclass
class Frame:
    """The fields of one written time directory, kept because OpenFOAM purges old ones."""
    time: float
    fields: dict[str, np.ndarray]


_FRAME_FIELDS = ("U", "p", "p_rgh", "T", "alpha.water", "k")
_MAX_FRAMES = 240


@dataclass
class Snapshot:
    status: str                     # running | success | failed | stopped | error
    phase: str
    segments: list[Segment]
    transient: bool
    end: float | None               # endTime (OpenFOAM) / iterations or end time (code_saturne)
    elapsed: float
    messages: list[str]
    error: str | None = None
    failed: bool = False            # the current solver log shows a crash

    @property
    def steps(self) -> list[dict]:
        return self.segments[-1].steps if self.segments else []


def _controls(case_dir: Path) -> tuple[str, bool, float | None]:
    """Solver application, transient?, endTime from system/controlDict."""
    t = (case_dir / "system" / "controlDict").read_text()
    app = re.search(r"^application\s+(\w+);", t, re.M).group(1)
    end = re.search(r"^endTime\s+([^;]+);", t, re.M)
    return app, app in ("pimpleFoam", "interFoam", "pisoFoam"), float(end.group(1)) if end else None


class RunJob:
    def __init__(self, case_dir: Path, work_dir: Path, code: str = "openfoam", *, quick: bool = False,
                 llm: dict | None = None):
        self.case_dir, self.work_dir, self.code, self.quick = Path(case_dir), Path(work_dir), code, quick
        self.llm = llm                   # {"backend", "model", "api_key", "base_url"} or None
        self.status, self.phase, self.error = "running", "starting", None
        self.messages: list[str] = []
        self.segments: list[Segment] = []
        self.result = None               # RepairResult (OpenFOAM)
        self._lock = threading.Lock()
        self._proc = None
        self._stop = False
        self._t0 = time.time()
        self._t1: float | None = None
        self.frames: list[Frame] = []    # flow snapshots of the current attempt (2-D OpenFOAM cases)
        self.flow_mesh = None            # (polygons, owning cell per polygon) of the 2-D mesh
        self.flow_note: str | None = None  # why there is no flow view, if so
        self._frames_done = threading.Event()
        if code == "openfoam":
            self.application, self.transient, self.end = _controls(self.case_dir)
            self._tail = LogTail(self.case_dir / f"log.{self.application}")
        else:
            self.application, self.transient, self.end = "code_saturne", *_saturne_controls(self.case_dir)
            self._tail = None
        self._thread = threading.Thread(target=self._work, daemon=True)

    # ── control ──
    def start(self) -> "RunJob":
        self._thread.start()
        return self

    def stop(self) -> None:
        from app.agent import stop_process
        self._stop = True
        if self._proc is not None:
            stop_process(self._proc)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    # ── worker thread ──
    def _set_proc(self, proc) -> None:
        self._proc = proc
        if self._stop:
            self.stop()

    def _work(self) -> None:
        try:
            self._run_openfoam() if self.code == "openfoam" else self._run_saturne()
        except Exception as exc:                   # shown in the UI, never kills the app
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self._t1 = time.time()

    def _run_openfoam(self) -> None:
        from app.agent import repair

        def on_attempt(d, att) -> None:            # the run's logs still exist here
            with self._lock:
                self._tail.poll(final=True)
                fix = None
                if att is not None:
                    more = f" (+{len(att.changes) - 1} more)" if len(att.changes) > 1 else ""
                    fix = f"{att.fixer}: {att.changes[0].splitlines()[0]}{more}"
                self.segments.append(Segment(self._tail.steps, fix))
                self._tail = LogTail(self.case_dir / f"log.{self.application}")
                if att is not None:                 # a new attempt follows: new mesh, new fields
                    self.frames, self.flow_mesh, self._seen = [], None, set()

        llm = self.llm or {}
        watcher = threading.Thread(target=self._watch_frames, daemon=True)
        watcher.start()
        try:
            self.result = repair(self.case_dir, backend=llm.get("backend"), model=llm.get("model"),
                                 api_key=llm.get("api_key"), base_url=llm.get("base_url"),
                                 log=self.messages.append, on_process=self._set_proc,
                                 on_attempt=on_attempt, stop=lambda: self._stop)
        finally:
            self._frames_done.set()
            watcher.join(timeout=30)
        self.status = "stopped" if self._stop else ("success" if self.result.success else "failed")

    def _run_saturne(self) -> None:
        from app import saturne
        self.messages.append("Running code_saturne…")
        saturne.run(self.case_dir, on_process=self._set_proc)
        with self._lock:
            self.segments = [Segment(_saturne_steps(self.case_dir))]
        if self._stop:
            self.status = "stopped"
        elif not self.segments[0].steps:
            self.status, self.error = "failed", "code_saturne wrote no residuals: see RESU/run/run_solver.log"
        else:
            self.status = "success"
            self.messages.append("✔ code_saturne finished")

    # ── read by the UI ──
    def snapshot(self) -> Snapshot:
        with self._lock:
            if self.code == "openfoam":
                current = self._tail.poll() if self.running else []
                segments = self.segments + ([Segment(list(current))] if current else [])
                phase = self._phase_openfoam(bool(current))
                failed = self._tail.failed
            else:
                failed = False
                segments = self.segments if not self.running else [Segment(_saturne_steps(self.case_dir))]
                phase = "solving" if segments and segments[0].steps else "preparing"
        return Snapshot(self.status if not self.running else "running", phase, segments, self.transient,
                        self.end, (self._t1 or time.time()) - self._t0, list(self.messages), self.error, failed)

    # ── flow snapshots (OpenFOAM) ──
    def _watch_frames(self) -> None:
        self._seen: set[str] = set()
        while not self._frames_done.wait(1.0):
            self._capture_frames()
        self._capture_frames()                      # the last time written

    def _capture_frames(self) -> None:
        from app.postprocess import FoamCase
        from app.postprocess.render import front_polygons
        try:
            case = FoamCase(self.case_dir)
            if self.flow_mesh is None:
                if not (self.case_dir / "constant" / "polyMesh" / "owner").exists():
                    return
                try:
                    self.flow_mesh = front_polygons(case)
                except ValueError:
                    self.flow_note = "The flow view draws 2-D cases only; this mesh is 3-D."
                    self._frames_done.set()
                    return
            for name in case.times():
                if name in self._seen or float(name) == 0:
                    continue
                d = self.case_dir / name
                u = d / "U"
                if not u.exists() or not u.read_bytes()[-200:].rstrip().endswith(b"//"):
                    continue                        # still being written
                fields = {}
                for f in _FRAME_FIELDS:
                    if (d / f).exists():
                        v = case.field(f, name)
                        if np.ndim(v) and len(v) == case.n_cells:
                            fields[f] = np.asarray(v, dtype=np.float32)
                self._seen.add(name)
                with self._lock:
                    self.frames.append(Frame(float(name), fields))
                    self.frames.sort(key=lambda fr: fr.time)
                    if len(self.frames) > _MAX_FRAMES:        # keep the run's whole span, coarser
                        self.frames = self.frames[::2] + ([self.frames[-1]] if len(self.frames) % 2 == 0 else [])
        except Exception:                           # a directory purged or rewritten meanwhile
            return

    def flow(self) -> tuple[object, list[Frame], str | None]:
        """2-D mesh polygons, the frames captured so far, and why there are none (if so)."""
        with self._lock:
            return self.flow_mesh, list(self.frames), self.flow_note

    def _phase_openfoam(self, solving: bool) -> str:
        if not self.running:
            return "finished"
        if solving:
            return f"solving ({self.application})"
        logs = sorted(self.case_dir.glob("log.*"), key=lambda p: p.stat().st_mtime)
        return f"running {logs[-1].name.removeprefix('log.')}" if logs else "preparing the mesh"


# ── code_saturne output ──────────────────────────────────────────────────────

def _saturne_controls(case_dir: Path) -> tuple[bool, float | None]:
    xml = (case_dir / "DATA" / "setup.xml").read_text()
    end = re.search(r"<maximum_time>\s*([^<]+)<", xml)
    if end:
        return True, float(end.group(1))
    it = re.search(r"<iterations>\s*(\d+)\s*<", xml)
    return False, float(it.group(1)) if it else None


_INSTANT_RE = re.compile(r"^ INSTANT\s+(\S+)\s+TIME STEP NUMBER\s+(\d+)", re.M)
_CS_RES_RE = re.compile(r"^c  (\S+)\s+\S+\s+\d+\s+\S+\s+\S+\s+(\S+)\s*$", re.M)
_CS_FLOW_RE = re.compile(r"^(Inlet|Free outlet|Free inlet|Convective inlet)\s+\d+\s+(\d+)\s+(\S+)\s*$", re.M)


def _saturne_steps(case_dir: Path) -> list[dict]:
    """Time-step records from run_solver.log, which code_saturne writes as it runs (its
    residuals.csv is only flushed at the end). The residual is the "Time residual" column, the
    one residuals.csv holds; "mass" is the inflow/outflow imbalance relative to the inflow."""
    from app import saturne
    log = case_dir / "RESU" / saturne.RUN_ID / "run_solver.log"
    try:
        text = log.read_text(errors="replace")
    except FileNotFoundError:
        return []
    transient = "maximum_time" in (case_dir / "DATA" / "setup.xml").read_text()
    marks = list(_INSTANT_RE.finditer(text))
    steps = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        block = text[m.end():nxt.start() if nxt else len(text)]
        if nxt is None and "Time residual" not in block:
            break                                   # block still being written
        res = {}
        for r in _CS_RES_RE.finditer(block):
            try:
                res[r.group(1).lower()] = float(r.group(2))
            except ValueError:
                res[r.group(1).lower()] = float("nan")
        if not res:
            continue
        flows = [float(f.group(3)) for f in _CS_FLOW_RE.finditer(block) if int(f.group(2))]
        inflow = -sum(f for f in flows if f < 0)
        mass = abs(sum(flows)) / inflow if inflow > 0 else None
        steps.append({"time": float(m.group(1)) if transient else int(m.group(2)), "co": None,
                      "cont": None, "exec": None, "mass": mass, "res": res})
    return steps


# ── Is it converging? ────────────────────────────────────────────────────────

@dataclass
class Assessment:
    state: str                      # waiting | converged | converging | stalled | diverging | bounded
    message: str
    eta_steps: float | None = None  # steady: iterations still needed at the current rate


def _worst(step: dict) -> float:
    vals = list(step["res"].values())
    return max(vals) if vals else float("nan")


def assess(steps: list[dict], transient: bool, failed: bool = False) -> Assessment:
    """Trend of the largest initial residual over the recent steps (a straight-line fit of its
    logarithm): converging, stalled or diverging; for transient runs, whether the per-step
    residuals stay bounded and the Courant number reasonable."""
    if failed:
        return Assessment("diverging", "The solver crashed (NaN or floating-point error).")
    if len(steps) < 10:
        return Assessment("waiting", "Too few steps yet to judge the trend.")
    worst = np.array([_worst(s) for s in steps])
    if not np.all(np.isfinite(worst)):
        return Assessment("diverging", "A residual became NaN or infinite.")
    n = max(10, len(steps) // 5)
    y = np.log10(np.clip(worst[-n:], 1e-30, None))
    slope = float(np.polyfit(np.arange(n), y, 1)[0]) * 100          # decades per 100 steps
    level = float(worst[-1])
    if transient:
        co = [s["co"] for s in steps[-n:] if s.get("co") is not None]
        if co and max(co) > 5:
            return Assessment("diverging", f"Courant number {max(co):.1f}: the time step is too large.")
        if level > 0.5 and slope > 0.5:
            return Assessment("diverging", f"Residuals growing ({slope:+.2f} decades / 100 steps).")
        return Assessment("bounded", f"Per-step residuals bounded (largest {level:.1e})"
                          + (f", Courant ≤ {max(co):.2f}." if co else "."))
    if worst[-1] < CONVERGED and all(v < CONVERGED for v in steps[-1]["res"].values()):
        return Assessment("converged", f"All residuals below {CONVERGED:g}.")
    if level > 1 or slope > 0.3:
        return Assessment("diverging", f"Largest residual rising ({slope:+.2f} decades / 100 iterations).")
    if slope < -0.05:
        eta = (math.log10(level) - math.log10(CONVERGED)) / (-slope) * 100
        return Assessment("converging", f"Falling {-slope:.2f} decades / 100 iterations.", eta)
    return Assessment("stalled", f"Residuals flat around {level:.1e}: the flow may be unsteady "
                      "(a transient solver may be needed) or the mesh too coarse.")
