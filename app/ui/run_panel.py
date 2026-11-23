"""Run tab (start / stop a run, live monitor) and Results tab (review, field, download)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from app.agent import shorten_run
from app.live_run import RunJob
from app.ui.live_monitor import render_job
from app.utils import zip_directory

JOB = "run_job"


def openfoam_available() -> bool:
    return bool(os.environ.get("WM_PROJECT_DIR")) and shutil.which("blockMesh") is not None


def _code() -> str:
    return "code_saturne" if st.session_state.get("cfd_code") == "code_saturne" else "openfoam"


# ── Run tab ──────────────────────────────────────────────────────────────────

def render_run_tab(working: dict | None) -> None:
    job: RunJob | None = st.session_state.get(JOB)
    running = bool(job and job.running)
    if working is None and not job:
        st.info("Describe your problem in the **Setup** tab first; the case is then run from here.",
                icon="ℹ️")
        return

    if working is not None:
        code = _code()
        if code == "openfoam" and not openfoam_available():
            st.info("OpenFOAM was not found. To run cases from the browser, start the app from a "
                    "terminal where OpenFOAM is sourced:\n\n"
                    "```bash\nsource /usr/share/openfoam/etc/bashrc 2>/dev/null\n"
                    "streamlit run app/main.py\n```", icon="ℹ️")
        elif code == "code_saturne" and _saturne_exe() is None:
            st.info("code_saturne was not found: set CODE_SATURNE to its bin/code_saturne.", icon="ℹ️")
        else:
            _controls(working, code, running)
    err = st.session_state.pop("run_start_error", None)
    if err:
        st.error(f"**The case could not be written:** {err}")
    render_job(JOB)


def _saturne_exe():
    from app import saturne
    return saturne.find_executable()


def _controls(working: dict, code: str, running: bool) -> None:
    from app.ui.case_output import _saturne_unsupported
    if code == "openfoam":
        st.caption("Generates the case and runs it; the residuals update live. If the run fails, the "
                   "repair agent diagnoses the failure from the logs and fixes it (rules first, then the "
                   "LLM), and the chart marks each restart.")
        c1, c2 = st.columns(2)
        quick = c1.radio("Run length", ["Quick check (≈60 iterations / 15 time steps)", "Full run"],
                         key="run_length", help="A full run can take minutes to hours.").startswith("Quick")
        backend = st.session_state.get("backend", "claude")
        use_llm = c2.checkbox(f"Let the LLM fix what the rules cannot ({backend})", value=True,
                              key="run_use_llm", help="Uses the backend and model selected in the sidebar.")
    else:
        st.caption("Generates the code_saturne study and runs it; the residuals update live. Steady runs "
                   "take one to a few minutes.")
        quick, use_llm, backend = False, False, None
    unsupported = _saturne_unsupported(working)
    if unsupported:
        st.warning(unsupported, icon="⚠️")
    mismatch = working.get("unsupported") or []
    if mismatch and not unsupported:
        st.warning(f"This case differs from your description in {len(mismatch)} way"
                   f"{'s' * (len(mismatch) > 1)} (see the **Setup** tab): "
                   + "; ".join(u.split(":**")[0].strip("*") for u in mismatch).lower() + ".", icon="⚠️")
        if not st.checkbox("Run it anyway", key="run_anyway"):
            unsupported = "mismatch"
    if st.button("▶ Run case", type="primary", key="btn_run_case", disabled=running or bool(unsupported),
                 help="A run is in progress." if running else None):
        _start(working, code, quick, backend if use_llm else None)
        st.rerun()


def _start(working: dict, code: str, quick: bool, backend: str | None) -> None:
    from app.ui.case_output import generate_case_dir
    old: RunJob | None = st.session_state.pop(JOB, None)
    if old is not None:
        shutil.rmtree(old.work_dir, ignore_errors=True)
    st.session_state.pop("run_review", None)
    work = Path(tempfile.mkdtemp(prefix="cfdgen_run_"))
    try:
        with st.spinner("Writing the case…"):
            if code == "openfoam":
                case_dir = work / working.get("solver", "case")
                generate_case_dir(working, case_dir, {"bytes": st.session_state.get("stl_bytes"),
                                                      "name": st.session_state.get("stl_name", "geometry"),
                                                      "bbox": st.session_state.get("stl_bbox")})
                if quick:
                    shorten_run(case_dir)
            else:
                from app import saturne
                generate_case_dir(working, work / "code_saturne")
                case_dir = work / "code_saturne" / saturne.CASE
        llm = None
        if backend:
            llm = {"backend": backend, "model": st.session_state.get("model"),
                   "api_key": st.session_state.get("api_key") or None,
                   "base_url": st.session_state.get("base_url")}
        st.session_state[JOB] = RunJob(case_dir, work, code, quick=quick, llm=llm).start()
    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        st.session_state["run_start_error"] = str(exc)


# ── Results tab ──────────────────────────────────────────────────────────────

_REVIEW_ICON = {"pass": "✅", "info": "ℹ️", "warn": "⚠️", "fail": "❌"}


def render_results_tab() -> None:
    job: RunJob | None = st.session_state.get(JOB)
    if job is None:
        st.info("No run yet. Start one in the **Run** tab; its review, field picture and files appear here.",
                icon="ℹ️")
        return
    if job.running:
        st.info("The case is still running: see the **Run** tab. Results appear here when it finishes.",
                icon="⏳")
        return
    if not job.case_dir.exists():
        st.session_state.pop(JOB, None)
        return
    if job.code == "code_saturne":
        _saturne_results(job)
    else:
        _openfoam_results(job)


def _openfoam_results(job: RunJob) -> None:
    case_dir = job.case_dir
    n = len(job.result.attempts) if job.result else 0
    if job.status == "success":
        st.success(("Quick check passed" if job.quick else "Run completed")
                   + (f" after {n} automatic fix{'es' if n > 1 else ''}." if n else " without any fix."),
                   icon="✅")
    elif job.status == "stopped":
        st.warning("The run was stopped before the end: the results below are partial.", icon="⏹️")
    else:
        st.error(f"Not repaired after {n} attempt(s). See the agent report below.", icon="🚨")

    report = case_dir / "agent_report.md"
    if n and report.exists():
        with st.expander("Agent report (diagnoses and diffs)", expanded=job.status != "success"):
            st.markdown(report.read_text())
    if job.status in ("success", "stopped"):
        _show_review(case_dir, job.quick or job.status == "stopped")
        _show_field(case_dir)
    st.download_button("Download the case as run (zip)", data=zip_directory(case_dir),
                       file_name=f"{case_dir.name}_run.zip", mime="application/zip", key="btn_download_run")


def _saturne_results(job: RunJob) -> None:
    from app import saturne
    if job.status == "success":
        _show_review(job.case_dir, False, review_fn=saturne.review)
    elif job.status == "stopped":
        st.warning("The run was stopped before the end.", icon="⏹️")
    else:
        st.error(job.error or "code_saturne failed.", icon="🚨")
    st.download_button("Download the study as run (zip)", data=zip_directory(job.case_dir.parent),
                       file_name="code_saturne_run.zip", mime="application/zip", key="btn_download_cs_run")


def _show_review(case_dir: Path, quick: bool, review_fn=None) -> None:
    """The result reviewer: checks computed from the results, plus an optional LLM explanation."""
    from app.review import explain, review, write_report

    rev = st.session_state.get("run_review")
    if rev is None or rev.case_dir != case_dir:
        try:
            rev = st.session_state["run_review"] = (review_fn or review)(case_dir)
        except Exception as exc:
            st.caption(f"Result review not available ({type(exc).__name__}: {exc}).")
            return
    st.markdown(f"**Result review: {rev.verdict}**"
                + (" — the run was shortened, so convergence is not expected." if quick else ""))
    st.table([{"": _REVIEW_ICON[c.status], "Check": c.name, "Result": c.message} for c in rev.checks])
    backend = st.session_state.get("backend")
    if rev.explanation is None and st.button("Explain the result and suggest improvements (LLM)",
                                             key="btn_review_explain"):
        with st.spinner("Asking the LLM to judge the result…"):
            try:
                explain(rev, backend, model=st.session_state.get("model"),
                        api_key=st.session_state.get("api_key") or None,
                        base_url=st.session_state.get("base_url"))
            except Exception as exc:
                st.error(f"The LLM explanation failed: {exc}")
        write_report(rev)
    if rev.explanation:
        st.info(rev.explanation["summary"])
        for s in rev.explanation["suggestions"]:
            st.markdown(f"- {s}")


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
