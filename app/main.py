"""
CFD Case Generator — Streamlit entry point.

Run with:
    streamlit run app/main.py

The page is composed of sections in ``app/ui/``:
sidebar → STL upload → problem description → analysis summary →
parameter editor → physics checks → case generation → convergence monitor.
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
from app.ui.sidebar import render_sidebar  # noqa: E402

render_sidebar()

backend_label = ("Claude" if st.session_state.get("backend", "claude") == "claude"
                 else st.session_state.get("model", "Ollama"))
st.title(f"CFD Case Generator — Powered by {backend_label}")
st.markdown("Describe your CFD problem in natural language and receive a complete, "
            "ready-to-run **OpenFOAM** case folder.")

render_stl_upload()
render_problem_input()

analysis = st.session_state.get("analysis")
if analysis:
    render_analysis_summary(analysis, st.session_state.get("analysis_notes", []))
    working = render_parameter_editor(analysis)
    render_physics_checks(working)
    render_case_output(working)

render_convergence_monitor()
