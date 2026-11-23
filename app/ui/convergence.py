"""Log analyser: residuals, trend and Courant number of any OpenFOAM solver log (e.g. a run
done on a cluster), with the same charts as the live monitor."""

from __future__ import annotations

import re

import streamlit as st

from app.live_run import Segment, Snapshot
from app.log_parser import _FATAL_RE, parse_steps
from app.ui.live_monitor import render_convergence


def render_convergence_monitor() -> None:
    st.caption("Paste or upload an OpenFOAM solver log to plot its residuals and judge its convergence. "
               "Works with logs from any case, not just those generated here.")
    method = st.radio("Input method", ["Upload file", "Paste text"], horizontal=True, key="log_input_method")
    text = ""
    if method == "Paste text":
        text = st.text_area("Solver log", height=160, key="log_paste",
                            placeholder="Paste the contents of log.simpleFoam / log.pimpleFoam / … here")
    else:
        f = st.file_uploader("Upload solver log", type=["log", "txt", "out"], key="log_file")
        if f is not None:
            text = f.read().decode("utf-8", errors="replace")
    if text.strip():
        _render_log(text)


def _render_log(text: str) -> None:
    steps = parse_steps(text)
    if not steps:
        st.warning("No residual lines found — is this an OpenFOAM solver log? Expected lines "
                   "containing 'Solving for …, Initial residual = …'.", icon="⚠️")
        return
    app = re.search(r"^Exec\s*:\s*(\w+)", text, re.M)
    transient = any(s.get("co") is not None for s in steps) or bool(
        app and app.group(1) in ("pimpleFoam", "interFoam", "pisoFoam"))
    fields = sorted({f for s in steps for f in s["res"]})
    st.caption(f"{app.group(1) + ' · ' if app else ''}{len(steps)} {'time steps' if transient else 'iterations'}"
               f" · fields `{'`, `'.join(fields)}`")
    crashed = bool(_FATAL_RE.search(text))
    elapsed = steps[-1].get("exec") or 0.0
    render_convergence(Snapshot("finished", "finished", [Segment(steps)], transient, None, elapsed, [],
                                failed=crashed))
