"""Generate the case, offer the download, explain the physics, show next steps."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.generators import GENERATORS
from app.llm import explain_case
from app.utils import zip_directory


def generate_case_zip(working: dict, stl: dict | None = None) -> tuple[bytes, str]:
    """Generate the case for *working* in a temp dir and return ``(zip_bytes, file_name)``."""
    solver = working.get("solver", "")
    generator_cls = GENERATORS.get(solver)
    if generator_cls is None:
        raise ValueError(f"No generator for solver {solver!r}. Supported: {list(GENERATORS)}")

    analysis = dict(working)
    if analysis.get("geometry_type") == "stl_import" and stl:
        analysis.update(_stl_bytes=stl.get("bytes"), _stl_name=stl.get("name", "geometry"),
                        _stl_bbox=stl.get("bbox"))
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir = Path(tmpdir) / solver
        generator_cls().generate(analysis, case_dir)
        return zip_directory(case_dir), f"{solver}_case.zip"


def render_case_output(working: dict) -> None:
    st.divider()
    if st.button("Generate Case Folder", type="primary", key="btn_generate"):
        with st.spinner("Generating case files..."):
            try:
                stl = {"bytes": st.session_state.get("stl_bytes"),
                       "name": st.session_state.get("stl_name", "geometry"),
                       "bbox": st.session_state.get("stl_bbox")}
                zip_bytes, zip_name = generate_case_zip(working, stl)
                st.session_state["zip_bytes"] = zip_bytes
                st.session_state["zip_name"] = zip_name
            except Exception as exc:              # show the user, keep the app alive
                st.session_state["zip_bytes"] = None
                st.error(f"**Generation error:** {exc}")
                with st.expander("Details"):
                    st.exception(exc)

    if not st.session_state.get("zip_bytes"):
        return

    st.success("Case folder generated successfully!")
    st.download_button(
        label="Download case.zip",
        data=st.session_state["zip_bytes"],
        file_name=st.session_state.get("zip_name", "openfoam_case.zip"),
        mime="application/zip",
        key="btn_download",
    )
    _render_explanation(working)
    _render_next_steps(working)


def _render_explanation(working: dict) -> None:
    with st.expander("🔍 Flow physics & what to expect", expanded=False):
        st.caption("Ask the AI to explain the expected flow behaviour, convergence "
                   "characteristics and a post-processing checklist for this case.")
        if st.button("Generate explanation", key="btn_explain"):
            with st.spinner("Generating explanation..."):
                try:
                    st.session_state["case_explanation"] = explain_case(
                        working,
                        st.session_state.get("user_description", ""),
                        backend=st.session_state.get("backend", "claude"),
                        model=st.session_state.get("model"),
                        api_key=st.session_state.get("api_key"),
                        ollama_timeout=st.session_state.get("ollama_timeout", 120),
                    )
                except Exception as exc:
                    st.error(f"Could not generate explanation: {exc}")
        if st.session_state.get("case_explanation"):
            st.markdown(st.session_state["case_explanation"])


def _render_next_steps(working: dict) -> None:
    solver = working.get("solver", "<solver>")
    zip_name = st.session_state.get("zip_name", "case.zip")
    cores = int(working.get("n_cores", 1))
    is_stl = working.get("geometry_type") == "stl_import"
    stl_name = st.session_state.get("stl_name", "geometry")
    mode = f"in parallel on {cores} cores" if cores > 1 else "in serial"

    if is_stl:
        steps = (f"blockMesh\nsurfaceFeatureExtract\nsnappyHexMesh -overwrite\ncheckMesh\n"
                 f"{solver} 2>&1 | tee log.{solver}")
        allrun = f"`Allrun` runs blockMesh → surfaceFeatureExtract → snappyHexMesh → {solver} {mode}."
        tips = (f"- The STL is at `constant/triSurface/{stl_name}.stl` — replace it to test a new geometry.\n"
                "- Tune `refinementSurfaces` levels in `snappyHexMeshDict` for surface resolution.\n"
                "- Run `checkMesh` before solving; fix negative volumes if reported.")
    else:
        extra = "setFields\n" if solver == "interFoam" else ""
        steps = f"blockMesh\ncheckMesh\n{extra}{solver} 2>&1 | tee log.{solver}"
        allrun = f"`Allrun` runs blockMesh{' → setFields' if extra else ''} → {solver} {mode}."
        tips = ("- Lower the relaxation factors in `system/fvSolution` if the run diverges.\n"
                "- Increase `endTime` in `system/controlDict` if the solver stops before converging.")

    st.markdown("")
    with st.expander("What to do next"):
        st.markdown(f"""
**1. Extract and run** (source your OpenFOAM environment first)
```bash
unzip {zip_name} && cd {solver}/
./Allrun
```
{allrun} Logs are written to `log.<application>`. `Allrun` also works on distribution
packages that do not ship OpenFOAM's `RunFunctions`.

**Or step by step:**
```bash
{steps}
```

**2. Monitor convergence** — paste `log.{solver}` into the **Convergence Monitor** below.

**3. Post-process**
```bash
touch case.foam && paraview case.foam
postProcess -func sample -latestTime
```

**4. Clean up:** `./Allclean`

**Tips**
{tips}
""")
