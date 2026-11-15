"""'Run in OpenFOAM' panel: run the generated case with the repair agent, live in the browser."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from app.agent import repair, shorten_run
from app.utils import zip_directory


def openfoam_available() -> bool:
    return bool(os.environ.get("WM_PROJECT_DIR")) and shutil.which("blockMesh") is not None


def render_run_panel(working: dict) -> None:
    with st.expander("▶ Run in OpenFOAM (with the repair agent)", expanded=False):
        if not openfoam_available():
            st.info("OpenFOAM was not found. To run cases from the browser, start the app from a "
                    "terminal where OpenFOAM is sourced:\n\n"
                    "```bash\nsource /usr/share/openfoam/etc/bashrc 2>/dev/null\n"
                    "streamlit run app/main.py\n```", icon="ℹ️")
            return

        st.caption("Generates the case, runs it and, if it fails, lets the agent diagnose the failure "
                   "from the OpenFOAM logs and repair it: rule-based fixes first, then the LLM.")
        c1, c2 = st.columns(2)
        with c1:
            quick = st.radio("Run length", ["Quick check (≈60 iterations / 15 time steps)", "Full run"],
                             key="run_length", help="A full run can take minutes to hours.") \
                .startswith("Quick")
        with c2:
            backend = st.session_state.get("backend", "claude")
            use_llm = st.checkbox(f"Let the LLM fix what the rules cannot ({backend})", value=True,
                                  key="run_use_llm",
                                  help="Uses the backend and model selected in the sidebar.")
        if st.button("Run case", type="primary", key="btn_run_case"):
            _run(working, quick, backend if use_llm else None)

        result = st.session_state.get("run_result")
        if result:
            _show_result(result)


def _run(working: dict, quick: bool, backend: str | None) -> None:
    from app.ui.case_output import generate_case_dir

    old = st.session_state.pop("run_result", None)
    if old:
        shutil.rmtree(old["dir"], ignore_errors=True)
    work = Path(tempfile.mkdtemp(prefix="cfdgen_run_"))
    case_dir = work / working.get("solver", "case")
    stl = {"bytes": st.session_state.get("stl_bytes"), "name": st.session_state.get("stl_name", "geometry"),
           "bbox": st.session_state.get("stl_bbox")}
    generate_case_dir(working, case_dir, stl)
    if quick:
        shorten_run(case_dir)

    lines: list[str] = []
    with st.status("Running the case…", expanded=True) as status:
        def log(msg: str) -> None:
            lines.append(msg)
            status.write(msg)

        r = repair(case_dir, backend=backend, model=st.session_state.get("model"),
                   api_key=st.session_state.get("api_key") or None,
                   base_url=st.session_state.get("base_url"), log=log)
        status.update(label="The case runs" if r.success else "The case could not be repaired",
                      state="complete" if r.success else "error", expanded=False)
    st.session_state["run_result"] = {"dir": str(work), "case": str(case_dir), "success": r.success,
                                      "attempts": len(r.attempts), "log": lines, "quick": quick}


def _show_result(res: dict) -> None:
    case_dir = Path(res["case"])
    if not case_dir.exists():
        st.session_state.pop("run_result", None)
        return
    n = res["attempts"]
    if res["success"]:
        st.success(("Quick check passed" if res["quick"] else "Run completed")
                   + (f" after {n} automatic fix{'es' if n > 1 else ''}." if n else " without any fix."),
                   icon="✅")
    else:
        st.error(f"Not repaired after {n} attempt(s). See the agent report and the logs below.", icon="🚨")

    report = case_dir / "agent_report.md"
    if n and report.exists():
        with st.expander("Agent report (diagnoses and diffs)", expanded=not res["success"]):
            st.markdown(report.read_text())

    solver_log = next((p for p in sorted(case_dir.glob("log.*"))
                       if p.name.endswith("Foam") and p.name != "log.blockMesh"), None)
    if res["success"]:
        _show_field(case_dir)
    if solver_log is not None:
        from app.ui.convergence import _render_residuals
        st.markdown(f"**Residuals** (`{solver_log.name}`)")
        _render_residuals(solver_log.read_text(errors="replace"))

    st.download_button("Download the case as run (zip)", data=zip_directory(case_dir),
                       file_name=f"{case_dir.name}_run.zip", mime="application/zip", key="btn_download_run")


def _show_field(case_dir: Path) -> None:
    """Velocity magnitude of the latest result, drawn cell by cell (2-D cases)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from app.postprocess import FoamCase
        from app.postprocess.render import plot_cells

        case = FoamCase(case_dir)
        t = case.latest_time()
        U = case.field("U", t)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        plot_cells(ax, case, np.linalg.norm(U, axis=1), cmap="viridis", label="|U| [m/s]")
        ax.set_title(f"Velocity magnitude at t = {float(t):g}", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:                      # 3-D (STL) cases or missing matplotlib
        st.caption(f"Field picture not available for this case ({type(exc).__name__}).")
