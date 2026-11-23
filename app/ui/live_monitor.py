"""Convergence charts and the live run monitor (refreshes itself while the case runs)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.live_run import CONVERGED, Segment, Snapshot, assess

_STATE = {"waiting": ("⏳", "Starting"), "converged": ("🟢", "Converged"), "converging": ("🟢", "Converging"),
          "bounded": ("🟢", "Stable"), "stalled": ("🟡", "Stalled"), "diverging": ("🔴", "Diverging")}
_MAX_POINTS = 1500                       # per field: keeps long transient charts light


def _fmt_duration(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def _frame(segments: list[Segment], transient: bool) -> tuple[pd.DataFrame, list[dict], str]:
    """Long-format residuals, the restart markers and the x-axis title. With several run
    attempts the x axis counts steps across attempts, so the restarts line up one after another."""
    use_time = transient and len(segments) == 1
    rows, markers, offset = [], [], 0
    for k, seg in enumerate(segments):
        stride = max(1, len(seg.steps) // _MAX_POINTS)
        for i, s in enumerate(seg.steps):
            if i % stride and i != len(seg.steps) - 1:
                continue
            x = s["time"] if use_time else offset + i + 1
            for f, r in s["res"].items():
                rows.append({"x": x, "Field": f, "Residual": max(r, 1e-14), "t": s["time"],
                             "Co": s.get("co")})
        offset += len(seg.steps)
        if seg.fix and k < len(segments) - 1:
            markers.append({"x": offset + 0.5, "label": f"restart {k + 1}: {seg.fix[:60]}"})
    title = "Simulated time [s]" if use_time else ("Time step" if transient else "Iteration")
    if len(segments) > 1:
        title += " (all attempts)"
    return pd.DataFrame(rows), markers, title


def residual_chart(segments: list[Segment], transient: bool, height: int = 340):
    df, markers, xtitle = _frame(segments, transient)
    if df.empty:
        return None
    lines = alt.Chart(df).mark_line(strokeWidth=1.6).encode(
        x=alt.X("x:Q", title=xtitle),
        y=alt.Y("Residual:Q", scale=alt.Scale(type="log"), title="Initial residual (log scale)"),
        color=alt.Color("Field:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Field:N"), alt.Tooltip("x:Q", title=xtitle),
                 alt.Tooltip("t:Q", title="solver time"), alt.Tooltip("Residual:Q", format=".3e")],
    )
    layers = [lines]
    if not transient:
        layers.append(alt.Chart(pd.DataFrame({"y": [CONVERGED]})).mark_rule(strokeDash=[6, 4], color="gray")
                      .encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="convergence threshold")]))
    if markers:
        m = pd.DataFrame(markers)
        layers.append(alt.Chart(m).mark_rule(color="#d62728", strokeWidth=2)
                      .encode(x="x:Q", tooltip=[alt.Tooltip("label:N", title="repair agent")]))
        layers.append(alt.Chart(m).mark_text(align="left", dx=4, dy=-6, color="#d62728", fontSize=11)
                      .encode(x="x:Q", y=alt.value(12), text="label:N"))
    return alt.layer(*layers).properties(height=height).interactive(bind_y=False)


_SMALL_COLOUR = {"co": "#ff7f0e", "cont": "#9467bd", "mass": "#2ca02c"}


def _small_chart(steps: list[dict], key: str, title: str, xtitle: str, log: bool = False):
    rows = [{"t": s["time"], "v": max(s[key], 1e-14) if log else s[key]}
            for s in steps if s.get(key) is not None]
    if len(rows) < 2:
        return None
    stride = max(1, len(rows) // _MAX_POINTS)
    df = pd.DataFrame(rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else []))
    y = alt.Y("v:Q", title=title, scale=alt.Scale(type="log") if log else alt.Undefined)
    return alt.Chart(df).mark_line(strokeWidth=1.4, color=_SMALL_COLOUR[key]) \
        .encode(x=alt.X("t:Q", title=xtitle), y=y,
                tooltip=[alt.Tooltip("t:Q", title=xtitle), alt.Tooltip("v:Q", title=title, format=".3g")]) \
        .properties(height=180)


def render_convergence(snap: Snapshot, flow=None) -> None:
    """Status cards, residual chart, Courant / continuity charts of a run (live or finished)."""
    steps = snap.steps
    a = assess(steps, snap.transient, snap.failed)
    icon, word = _STATE[a.state]
    last = steps[-1] if steps else None

    progress = None
    if last and snap.end:
        progress = min(1.0, last["time"] / snap.end)
    eta = None
    if last and snap.end and snap.status == "running" and len(steps) > 2:
        first, ex = steps[0], last.get("exec")
        done = last["time"] - first["time"]
        if ex and done > 0:                      # OpenFOAM: solver CPU time per unit of solver time
            eta = ex / done * (snap.end - last["time"])
        elif progress and progress > 0.01:
            eta = snap.elapsed * (1 - progress) / progress

    c = st.columns(5)
    c[0].metric("Status", f"{icon} {word}" if snap.status == "running" or steps else snap.status)
    c[1].metric("Simulated time" if snap.transient else "Iteration",
                f"{last['time']:.4g} s" if last and snap.transient else (f"{int(last['time'])}" if last else "—"))
    c[2].metric("Progress", f"{progress:.0%}" if progress is not None else "—")
    c[3].metric("Elapsed", _fmt_duration(snap.elapsed))
    c[4].metric("Time left", _fmt_duration(eta) if eta else "—",
                help="Estimated from the solver's speed so far. A steady run may also stop early "
                     "when the residual targets are reached.")
    if snap.status == "running" and progress is not None:
        st.progress(progress, text=snap.phase)
    elif snap.status == "running":
        st.caption(f"⏳ {snap.phase}…")

    if steps:
        hint = a.message
        if a.state == "converging" and a.eta_steps:
            hint += f" At this rate, about {a.eta_steps:,.0f} more iterations to reach {CONVERGED:g}."
        (st.error if a.state == "diverging" else st.warning if a.state == "stalled" else st.caption)(
            f"{icon} {hint}")

    left, right = st.columns(2) if flow else (st.container(), None)
    with left:
        st.markdown("**Residuals**")
        chart = residual_chart(snap.segments, snap.transient)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("The residual chart appears once the solver starts its first iteration.")
    if right is not None:
        with right:
            flow()

    xt = "Simulated time [s]" if snap.transient else "Iteration"
    small = [ch for ch in (_small_chart(steps, "co", "max Courant number", xt),
                           _small_chart(steps, "cont", "continuity error (sum local)", xt, log=True),
                           _small_chart(steps, "mass", "mass imbalance / inflow", xt, log=True)) if ch]
    if small:
        for col, ch in zip(st.columns(len(small)), small):
            col.altair_chart(ch, width="stretch")


@st.fragment(run_every=1.5)
def _live(job_key: str) -> None:
    job = st.session_state.get(job_key)
    if job is None:
        return
    if not job.running:                          # finished: refresh the whole page once (Results tab)
        st.rerun()
    snap = job.snapshot()
    top = st.columns([5, 1])
    top[0].markdown(f"**● Live** · {snap.phase}")
    top[1].button("⏹ Stop", key="btn_stop_run", on_click=job.stop, width="stretch")
    render_convergence(snap, flow=_flow(job, live=True))
    _messages(snap)


def _messages(snap: Snapshot) -> None:
    if snap.messages:
        with st.expander(f"Repair agent ({len(snap.messages)} message{'s' * (len(snap.messages) > 1)})",
                         expanded=len(snap.segments) > 1 or snap.status in ("failed", "error")):
            st.code("\n".join(snap.messages), language=None)


def _flow(job, live: bool):
    """The flow view beside the residuals (OpenFOAM runs; code_saturne writes fields at the end)."""
    if job.code != "openfoam":
        return None
    from app.ui.live_flow import render_flow
    return lambda: render_flow(job, live)


def render_job(job_key: str = "run_job") -> None:
    """The run in *job_key*: live while it runs, then its final state."""
    job = st.session_state.get(job_key)
    if job is None:
        return
    if job.running:
        _live(job_key)
        return
    snap = job.snapshot()
    banner = {"success": (st.success, "✅", "Run finished. The review, field plot and download are in the "
                                             "**Results** tab."),
              "failed": (st.error, "🚨", "The run failed and the agent could not repair it. See the "
                                        "agent's messages below and the **Results** tab."),
              "stopped": (st.warning, "⏹️", "Run stopped."),
              "error": (st.error, "🚨", f"The run could not start: {snap.error}")}[snap.status]
    banner[0](banner[2], icon=banner[1])
    render_convergence(snap, flow=_flow(job, live=False))
    _messages(snap)
