"""
CFD Case Generator — Streamlit entry point.

Run with:
    streamlit run app/main.py

The page has four tabs, built from the sections in ``app/ui/``:
Setup (mesh upload, description, analysis, parameters, physics checks, case download),
Run (start / stop, live convergence monitor, repair agent), Results (review, field, download)
and Log analyser (residuals of any solver log).
"""

import sys
from pathlib import Path

# Make `app.*` importable regardless of the working directory Streamlit uses.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="CFD Case Generator", page_icon="🌊", layout="wide")

from app.ui.analysis_view import render_analysis_summary  # noqa: E402
from app.ui.case_output import render_case_output  # noqa: E402
from app.ui.convergence import render_convergence_monitor  # noqa: E402
from app.ui.parameter_editor import render_parameter_editor  # noqa: E402
from app.ui.physics_checks import render_physics_checks  # noqa: E402
from app.ui.problem_input import render_problem_input, render_stl_upload  # noqa: E402
from app.ui.mesh_upload import render_mesh_upload, render_role_editor  # noqa: E402
from app.ui.run_panel import render_results_tab, render_run_tab  # noqa: E402
from app.ui.sidebar import render_geometry, render_sidebar  # noqa: E402

geometry_slot = render_sidebar()

code = "code_saturne" if st.session_state.get("cfd_code") == "code_saturne" else "OpenFOAM"
st.title("CFD Case Generator")
st.markdown(f"Describe your flow problem in plain English: the AI sets up a **{code}** case, which "
            "you can download or run here, watch converge and have checked.")
st.caption(f"AI: {st.session_state.get('ai_label', '—')}"
           + ("" if st.session_state.get("ai_ready") else " (not ready: see the sidebar)"))

tab_setup, tab_run, tab_results, tab_log = st.tabs(
    ["① Setup", "② Run", "③ Results", "📈 Log analyser"], key="main_tab")

working = None
with tab_setup:
    render_stl_upload()
    render_mesh_upload()
    render_problem_input()
    analysis = st.session_state.get("analysis")
    if analysis:
        render_analysis_summary(analysis, st.session_state.get("analysis_notes", []))
        render_role_editor(analysis)
        working = render_parameter_editor(analysis)
        render_physics_checks(working)
        render_case_output(working)

with tab_run:
    render_run_tab(working)

with tab_results:
    render_results_tab()

with tab_log:
    render_convergence_monitor()

render_geometry(geometry_slot)          # last, so it shows the analysis made in this run
