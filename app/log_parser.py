"""
Parse OpenFOAM solver log files and extract residual histories.

Works with logs from all ESI OpenFOAM versions (v1912 – v2512).
No file I/O: accepts the log content as a plain string so it can be used
with both uploaded files and pasted text in Streamlit.
"""

import re
from collections import defaultdict

# Matches both smoothSolver/GAMG/PCG/PBiCGStab "Solving for ..." lines.
# Captures: field name, initial residual value.
_RESIDUAL_RE = re.compile(
    r"Solving for (\w+),\s*Initial residual\s*=\s*([\d.eE+\-]+)"
)

# Time = <value> lines (steady: iteration counter; transient: physical time)
_TIME_RE = re.compile(r"^Time\s*=\s*([\d.eE+\-]+)", re.MULTILINE)

# ExecutionTime lines — used to detect end of a time step
_EXEC_RE = re.compile(r"ExecutionTime\s*=\s*([\d.eE+\-]+)\s*s")


def parse_foam_log(text: str) -> dict:
    """Parse an OpenFOAM solver log and return residual histories.

    Parameters
    ----------
    text : str
        Full content of the solver log file.

    Returns
    -------
    dict with keys:
        ``fields``    : list[str]                 — field names in order of appearance
        ``residuals`` : dict[str, list[float]]    — field → per-iteration initial residuals
        ``times``     : list[float]               — solver time values (may be empty)
        ``n_iters``   : int                       — total iterations (time steps) detected
    """
    residuals: dict[str, list[float]] = defaultdict(list)
    fields_order: list[str] = []
    times: list[float] = []

    for m in _RESIDUAL_RE.finditer(text):
        field, val_str = m.group(1), m.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if field not in residuals:
            fields_order.append(field)
        residuals[field].append(val)

    for m in _TIME_RE.finditer(text):
        try:
            t = float(m.group(1))
            if t > 0:          # skip Time = 0 (initial setup output)
                times.append(t)
        except ValueError:
            pass

    # n_iters = number of time steps completed (length of the longest residual list)
    n_iters = max((len(v) for v in residuals.values()), default=0)

    return {
        "fields":    fields_order,
        "residuals": dict(residuals),
        "times":     times,
        "n_iters":   n_iters,
    }


def is_converged(residuals: dict[str, list[float]], threshold: float = 1e-4) -> bool:
    """Return True if all final residuals are below *threshold*."""
    if not residuals:
        return False
    return all(
        vals and vals[-1] < threshold
        for vals in residuals.values()
    )


# ── Per-time-step parsing (live monitor and log analyser) ────────────────────

_STEP_RE = re.compile(r"^Time = ", re.M)
_COURANT_RE = re.compile(r"^Courant Number mean: \S+ max: ([\d.eE+\-]+)", re.M)
_CONT_RE = re.compile(r"time step continuity errors : sum local = ([\d.eE+\-]+)")
_EXEC_LAST_RE = re.compile(r"^ExecutionTime = ([\d.eE+\-]+) s", re.M)
# the crash handler, not the banner ("Floating point exception trapping enabled (FOAM_SIGFPE)")
_FATAL_RE = re.compile(r"FOAM FATAL|sigFpe::|Initial residual = -?(?:nan|inf)", re.I)


def _first_float(rx: re.Pattern, text: str, last: bool = False) -> float | None:
    found = rx.findall(text)
    if not found:
        return None
    try:
        return float(found[-1] if last else found[0])
    except ValueError:
        return float("nan")


def _parse_step(chunk: str) -> dict | None:
    """One ``Time = …`` block: its time and, per field, the first initial residual of the step."""
    head, _, body = chunk.partition("\n")
    try:
        t = float(head.removeprefix("Time = ").split()[0])
    except (ValueError, IndexError):
        return None
    res: dict[str, float] = {}
    for m in _RESIDUAL_RE.finditer(body):
        if m.group(1) not in res:
            try:
                res[m.group(1)] = float(m.group(2))
            except ValueError:
                res[m.group(1)] = float("nan")
    return {"time": t, "res": res, "cont": _first_float(_CONT_RE, body, last=True),
            "exec": _first_float(_EXEC_LAST_RE, body, last=True),
            "_co_next": _first_float(_COURANT_RE, body, last=True)}


class LogTail:
    """Reads a growing OpenFOAM solver log and turns it into one record per time step:
    ``{"time", "res": {field: initial residual}, "co": max Courant, "cont", "exec"}``.
    Only new bytes are read at each :meth:`poll`, so it stays cheap on long transient logs."""

    def __init__(self, path=None):
        self.path = path
        self.steps: list[dict] = []
        self.failed = False
        self._offset = 0
        self._buf = ""
        self._co = None            # the Courant number is printed before its time step

    def poll(self, final: bool = False) -> list[dict]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                new = f.read()
        except FileNotFoundError:
            return self.steps
        self._offset += len(new)
        return self.feed(new.decode(errors="replace"), final)

    def feed(self, new: str, final: bool = False) -> list[dict]:
        """Parse *new* text; a step is kept back until the next one starts (or *final*)."""
        if _FATAL_RE.search(new):
            self.failed = True
        text = self._buf + new
        starts = [m.start() for m in _STEP_RE.finditer(text)]
        if not starts:
            self._buf = "" if final else text
            return self.steps
        if starts[0] > 0:          # banner and set-up before the first time step
            co = _first_float(_COURANT_RE, text[:starts[0]], last=True)
            self._co = co if co is not None else self._co
        bounds = starts + [len(text)]
        complete = len(starts) if final else len(starts) - 1
        for a, b in zip(bounds[:complete], bounds[1:complete + 1]):
            step = _parse_step(text[a:b])
            if step is None:
                continue
            step["co"] = self._co
            co = step.pop("_co_next")
            self._co = co if co is not None else self._co
            if step["res"]:
                self.steps.append(step)
        self._buf = "" if final else text[starts[-1]:]
        return self.steps


def parse_steps(text: str) -> list[dict]:
    """Per-time-step records of a whole log (see :class:`LogTail`)."""
    return LogTail().feed(text, final=True)
