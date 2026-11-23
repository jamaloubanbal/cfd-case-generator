"""Flow view of a run: the latest written field while it runs, a replay slider afterwards."""

from __future__ import annotations

import io

import numpy as np
import streamlit as st

# label → (field, component: None = magnitude for vectors, int = component, colour map)
_VIEWS = {
    "Water fraction": ("alpha.water", None, "Blues"),
    "Temperature": ("T", None, "inferno"),
    "Speed |U|": ("U", None, "viridis"),
    "Cross-flow velocity Uy": ("U", 1, "RdBu_r"),
    "Pressure": ("p", None, "coolwarm"),
    "Pressure (p_rgh)": ("p_rgh", None, "coolwarm"),
    "Turbulent kinetic energy k": ("k", None, "magma"),
}


def _values(frame, view: str) -> np.ndarray:
    name, comp, _ = _VIEWS[view]
    v = frame.fields[name]
    if v.ndim == 2:
        return np.linalg.norm(v, axis=1) if comp is None else v[:, comp]
    return v


@st.cache_data(max_entries=64, show_spinner=False)
def _png(_mesh, mesh_id: int, values: np.ndarray, cmap: str, vmin: float, vmax: float, title: str,
         label: str) -> bytes:
    """Rendered with matplotlib's object API (thread-safe, no pyplot state)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    polys, cells = _mesh
    allp = np.vstack(polys)
    w, h = np.ptp(allp[:, 0]), np.ptp(allp[:, 1])
    fig = Figure(figsize=(8, max(2.2, min(6.5, 8 * h / max(w, 1e-12)) + 0.6)), dpi=100)
    ax = fig.add_subplot()
    coll = PolyCollection(polys, array=values[cells], cmap=cmap, edgecolors="face", linewidths=0.4)
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
    ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.colorbar(coll, ax=ax, label=label, shrink=0.85, pad=0.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def render_flow(job, live: bool) -> None:
    if job.code != "openfoam":
        return
    mesh, frames, note = job.flow()
    st.markdown("**Flow**")
    if note:
        st.caption(note)
        return
    if mesh is None or not frames:
        st.caption("The flow appears here when the solver writes its first result "
                   "(every `writeInterval` in system/controlDict).")
        return
    available = [v for v, (f, _, _) in _VIEWS.items() if f in frames[-1].fields]
    default = next((v for v in ("Water fraction", "Temperature", "Speed |U|") if v in available), available[0])
    if st.session_state.get("flow_view") not in available:
        st.session_state["flow_view"] = default
    view = st.selectbox("Field", available, key="flow_view", label_visibility="collapsed")

    if live or len(frames) == 1:
        frame = frames[-1]
    else:
        i = st.select_slider("Replay", options=list(range(len(frames))), value=len(frames) - 1,
                             format_func=lambda k: f"t = {frames[k].time:g} s", key="flow_frame")
        frame = frames[i]
    vals = _values(frame, view)
    # colour range: fixed over the frames so far (robust percentiles), so a replay does not flicker
    pool = np.concatenate([_values(f, view) for f in frames[-12:]])
    lo, hi = np.percentile(pool, [0.5, 99.5])
    if view == "Cross-flow velocity Uy":
        lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    if view == "Water fraction":
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-12
    name, comp, cmap = _VIEWS[view]
    st.image(_png(mesh, id(mesh), vals, cmap, float(lo), float(hi),
                  f"{view} at t = {frame.time:g} s", view), width="stretch")
    st.caption(f"Frame {frames.index(frame) + 1} of {len(frames)}"
               + (" · updates at each write" if live else " · drag the slider to replay the run"))
