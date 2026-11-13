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
