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


def _figure(mesh, values: np.ndarray, cmap: str, vmin: float, vmax: float, title: str, label: str,
            dpi: int = 100):
    """One frame, drawn with matplotlib's object API (thread-safe, no pyplot state)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    polys, cells = mesh
    allp = np.vstack(polys)
    w, h = np.ptp(allp[:, 0]), np.ptp(allp[:, 1])
    fig = Figure(figsize=(8, max(2.2, min(6.5, 8 * h / max(w, 1e-12)) + 0.6)), dpi=dpi)
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
    return fig


@st.cache_data(max_entries=64, show_spinner=False)
def _png(_mesh, mesh_id: int, values: np.ndarray, cmap: str, vmin: float, vmax: float, title: str,
         label: str) -> bytes:
    buf = io.BytesIO()
    _figure(_mesh, values, cmap, vmin, vmax, title, label).savefig(buf, format="png")
    return buf.getvalue()


def _limits(view: str, arrays: list[np.ndarray]) -> tuple[float, float]:
    """Colour range over several frames (robust percentiles), so the picture does not flicker."""
    if view == "Water fraction":
        return 0.0, 1.0
    lo, hi = np.percentile(np.concatenate(arrays), [0.5, 99.5])
    if view == "Cross-flow velocity Uy":
        lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    return float(lo), float(max(hi, lo + 1e-12))


GIF_FRAMES = 60          # at most; frames are taken evenly over the run
GIF_MS = 120             # per frame


def make_gif(mesh, frames: list, view: str, progress=None) -> bytes:
    """Animated GIF of *view* over the run, with one colour range for all frames."""
    from PIL import Image

    idx = np.unique(np.linspace(0, len(frames) - 1, min(GIF_FRAMES, len(frames))).round().astype(int))
    vals = [_values(frames[i], view) for i in idx]
    lo, hi = _limits(view, vals)
    cmap = _VIEWS[view][2]
    images = []
    for k, (i, v) in enumerate(zip(idx, vals)):
        buf = io.BytesIO()
        _figure(mesh, v, cmap, lo, hi, f"{view} at t = {frames[i].time:.4g} s", view, dpi=80) \
            .savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT))
        if progress:
            progress((k + 1) / len(idx))
    out = io.BytesIO()
    images[0].save(out, format="GIF", save_all=True, append_images=images[1:], duration=GIF_MS, loop=0,
                   optimize=True)
    return out.getvalue()


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
                             format_func=lambda k: f"t = {frames[k].time:.4g} s", key="flow_frame")
        frame = frames[i]
    vals = _values(frame, view)
    lo, hi = _limits(view, [_values(f, view) for f in frames[-12:]])
    name, comp, cmap = _VIEWS[view]
    st.image(_png(mesh, id(mesh), vals, cmap, lo, hi, f"{view} at t = {frame.time:.4g} s", view),
             width="stretch")
    st.caption(f"Frame {frames.index(frame) + 1} of {len(frames)}"
               + (" · updates at each write" if live else " · drag the slider to replay the run"))
    if not live and len(frames) > 1:
        _gif_export(job, mesh, frames, view)


def _gif_export(job, mesh, frames: list, view: str) -> None:
    key = (str(job.case_dir), view, len(frames))
    gif = st.session_state.get("flow_gif")
    if gif is None or gif[0] != key:
        n = min(GIF_FRAMES, len(frames))
        if st.button(f"🎞 Make an animation ({n} frames, GIF)", key="btn_make_gif"):
            bar = st.progress(0.0, text="Drawing the frames…")
            data = make_gif(mesh, frames, view, progress=lambda f: bar.progress(f, text="Drawing the frames…"))
            bar.empty()
            st.session_state["flow_gif"] = gif = (key, data)
        else:
            return
    fname = f"{job.case_dir.name}_{_VIEWS[view][0].replace('.', '_')}.gif"
    st.download_button(f"⬇ Download the animation ({len(gif[1]) / 1e6:.1f} MB)", gif[1], file_name=fname,
                       mime="image/gif", key="btn_download_gif")
