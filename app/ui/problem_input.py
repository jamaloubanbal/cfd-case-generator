"""STL upload, problem description and the "Analyse" step."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.llm import analyze_cfd_problem, ollama_is_reachable
from app.sanitize import AnalysisError, sanitize_analysis
from app.stl_parser import stl_bounding_box

_STL_KEYS = ("stl_bytes", "stl_name", "stl_bbox", "stl_filename")

_STL_HINT = (
    "\n\n[Note: the user has uploaded a custom STL geometry. This is an external flow "
    "around a user-provided 3-D body. Choose a solver and boundary conditions appropriate "
    "for external aerodynamics. Set geometry_type to 'channel' — the actual geometry is "
    "meshed with snappyHexMesh.]"
)


def render_stl_upload() -> None:
    """Optional STL upload; stores bytes / name / bounding box in session state."""
    with st.expander("🗂 Upload STL geometry (optional — enables snappyHexMesh workflow)", expanded=False):
        st.caption(
            "Upload a surface STL file to generate a **snappyHexMesh** case around your geometry. "
            "The background mesh bounds are computed from the STL bounding box. Describe the "
            "flow physics below as usual — solver and BCs are still chosen by the LLM."
        )
        stl_file = st.file_uploader(
            "STL file", type=["stl"], key="stl_upload",
            help="Binary or ASCII STL. The file stem is used as the patch name ('body.stl' → 'body').",
        )
        if stl_file is not None:
            data = stl_file.read()
            try:
                bbox = stl_bounding_box(data)
            except ValueError as exc:
                st.error(f"Could not parse STL: {exc}", icon="🚨")
                return
            xmin, xmax, ymin, ymax, zmin, zmax = bbox
            st.session_state.update(
                stl_bytes=data, stl_name=Path(stl_file.name).stem,
                stl_bbox=bbox, stl_filename=stl_file.name,
            )
            st.success(
                f"**{stl_file.name}** loaded — {len(data):,} bytes  \n"
                f"Bounding box: x [{xmin:.4g}, {xmax:.4g}] ({xmax - xmin:.4g} m) · "
                f"y [{ymin:.4g}, {ymax:.4g}] ({ymax - ymin:.4g} m) · "
                f"z [{zmin:.4g}, {zmax:.4g}] ({zmax - zmin:.4g} m)",
                icon="✅",
            )
            st.info(
                "Background mesh extends **2× upstream**, **5× downstream** and **2× on each side** "
                "of the STL. snappyHexMesh refines to the surface (2 levels) and adds 3 prism layers.",
                icon="ℹ️",
            )
        elif "stl_bytes" in st.session_state:
            st.info(f"Using previously uploaded **{st.session_state.get('stl_filename', 'geometry.stl')}**. "
                    "Re-upload to replace, or clear below.")
            if st.button("Clear STL", key="clear_stl"):
                for k in _STL_KEYS:
                    st.session_state.pop(k, None)
                st.rerun()


def render_problem_input() -> None:
    """Description text area + Analyse button; runs and sanitises the analysis."""
    description = st.text_area(
        label="Problem Description",
        placeholder=(
            "Example: Steady-state airflow around a 2D cylinder at Re=200 with a freestream "
            "velocity of 1 m/s. The fluid is air at standard conditions."
        ),
        height=150,
        key="user_description",
    )

    backend = st.session_state.get("backend", "claude")
    if backend == "ollama":
        ready = ollama_is_reachable()
    elif backend == "openai-compatible":
        ready = bool(st.session_state.get("model"))
    else:                                       # Claude, Gemini, Groq, OpenRouter need a key
        ready = bool(st.session_state.get("api_key")) and bool(st.session_state.get("model"))
    help_text = ""
    if not ready:
        help_text = ("Ollama is not reachable. Run: ollama serve" if backend == "ollama"
                     else "Enter the API key and model in the sidebar first.")

    clicked = st.button(
        "Analyse my problem", type="primary", key="btn_analyse",
        disabled=not (description.strip() and ready), help=help_text,
    )
    if clicked and description.strip():
        _run_analysis(description)


def _run_analysis(description: str) -> None:
    has_stl = bool(st.session_state.get("stl_bytes"))
    error = None
    with st.spinner("Analysing your CFD problem..."):
        try:
            raw = analyze_cfd_problem(
                description + (_STL_HINT if has_stl else ""),
                backend=st.session_state.get("backend", "claude"),
                model=st.session_state.get("model"),
                api_key=st.session_state.get("api_key"),
                base_url=st.session_state.get("base_url"),
                ollama_timeout=st.session_state.get("ollama_timeout", 300),
            )
            analysis, notes = sanitize_analysis(raw, description)
        except AnalysisError as exc:
            error = f"**The analysis could not be used:** {exc}"
        except RuntimeError as exc:
            error = f"**API error:** {exc}"
        except ValueError as exc:
            error = f"**Parsing error:** {exc}"
    # Report after the spinner has closed (st.stop() inside it left the spinner on screen)
    if error:
        st.error(error.replace("(--model)", "(Model field in the sidebar)")
                      .replace("--model, the sidebar, ", "the Model field in the sidebar, "))
        st.stop()

    if has_stl:
        analysis["geometry_type"] = "stl_import"

    st.session_state["analysis"] = analysis
    st.session_state["analysis_notes"] = notes
    st.session_state["zip_bytes"] = None
    st.session_state.pop("case_explanation", None)
    # Reset parameter-editor widgets so they pick up the new analysis
    for k in [k for k in st.session_state if k.startswith("pe_")]:
        del st.session_state[k]
