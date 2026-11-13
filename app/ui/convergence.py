"""Convergence monitor: plot residuals from any OpenFOAM solver log."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.log_parser import is_converged, parse_foam_log


def render_convergence_monitor() -> None:
    st.divider()
    st.subheader("📈 Convergence Monitor")
    st.caption("Paste or upload an OpenFOAM solver log to plot residuals and check convergence. "
               "Works with logs from any case, not just those generated here.")

    with st.expander("Analyse a solver log", expanded=False):
        method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True,
                          key="log_input_method")
        text = ""
        if method == "Paste text":
            text = st.text_area("Solver log", height=160, key="log_paste",
                                placeholder="Paste the contents of log.simpleFoam / log.pimpleFoam / … here")
        else:
            f = st.file_uploader("Upload solver log", type=["log", "txt", "out"], key="log_file")
            if f is not None:
                text = f.read().decode("utf-8", errors="replace")

        if text.strip():
            _render_residuals(text)


def _render_residuals(text: str) -> None:
    parsed = parse_foam_log(text)
    if not parsed["fields"]:
        st.warning("No residual lines found — is this an OpenFOAM solver log? Expected lines "
                   "containing 'Solving for …, Initial residual = …'.", icon="⚠️")
        return

    st.caption(f"Fields detected: `{'`, `'.join(parsed['fields'])}` · {parsed['n_iters']} iteration(s)")
    df = pd.DataFrame([
        {"Iteration": i, "Residual": r, "Field": field}
        for field in parsed["fields"]
        for i, r in enumerate(parsed["residuals"][field], start=1)
    ])
    chart = (
        alt.Chart(df).mark_line()
        .encode(
            x=alt.X("Iteration:Q"),
            y=alt.Y("Residual:Q", scale=alt.Scale(type="log"), title="Initial residual (log scale)"),
            color=alt.Color("Field:N"),
            tooltip=[alt.Tooltip("Iteration:Q"), alt.Tooltip("Field:N"),
                     alt.Tooltip("Residual:Q", format=".3e")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    if is_converged(parsed["residuals"]):
        st.success("All final residuals < 1 × 10⁻⁴ — case appears converged.", icon="✅")
    else:
        final = {f: v[-1] for f, v in parsed["residuals"].items() if v}
        worst = max(final, key=final.get)
        st.warning(f"Not yet converged. Largest final residual: `{worst}` = {final[worst]:.3e}", icon="⚠️")
