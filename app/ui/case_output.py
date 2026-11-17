"""Generate the case, offer the download, explain the physics, show next steps."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.generators import GENERATORS
from app.llm import explain_case
from app.utils import zip_directory
from app.ui.run_panel import render_run_panel


def generate_case_dir(working: dict, case_dir: Path, stl: dict | None = None) -> None:
    """Generate the case for *working* into *case_dir*."""
    solver = working.get("solver", "")
    generator_cls = GENERATORS.get(solver)
    if generator_cls is None:
        raise ValueError(f"No generator for solver {solver!r}. Supported: {list(GENERATORS)}")
    analysis = dict(working)
    if analysis.get("geometry_type") == "imported_mesh":
        from app.mesh_import import apply_roles, attach, check_roles
        mesh, roles = st.session_state.get("imported_mesh"), st.session_state.get("mesh_roles")
        if mesh is None or roles is None:
            raise ValueError("The imported mesh is no longer loaded: upload it again and re-run the analysis.")
        problems = check_roles(roles, analysis.get("solver", ""))
        if problems:
            raise ValueError("Boundary roles: " + "; ".join(problems) + ". Fix them in the role table.")
        analysis = attach(analysis, mesh, apply_roles(mesh, roles))
    if analysis.get("geometry_type") == "stl_import" and stl:
        analysis.update(_stl_bytes=stl.get("bytes"), _stl_name=stl.get("name", "geometry"),
                        _stl_bbox=stl.get("bbox"))
    if st.session_state.get("cfd_code") == "code_saturne":
        from app import saturne
        unv = None
        if analysis.get("geometry_type") == "imported_mesh":
            unv = case_dir.parent / st.session_state.get("mesh_filename", "mesh.unv")
            unv.parent.mkdir(parents=True, exist_ok=True)
            unv.write_bytes(st.session_state["unv_bytes"])
        saturne.generate(analysis, unv, case_dir, st.session_state.get("user_description"))
        return
    generator_cls().generate(analysis, case_dir)
    from app.review import save_request
    save_request(case_dir, analysis, st.session_state.get("user_description"))


def generate_case_zip(working: dict, stl: dict | None = None) -> tuple[bytes, str]:
    """Generate the case for *working* in a temp dir and return ``(zip_bytes, file_name)``."""
    solver = working.get("solver", "")
    code = "code_saturne" if st.session_state.get("cfd_code") == "code_saturne" else solver
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir = Path(tmpdir) / code
        generate_case_dir(working, case_dir, stl)
        return zip_directory(case_dir), f"{code}_case.zip"


def _saturne_unsupported(working: dict) -> str | None:
    """Why the code_saturne backend cannot generate *working* (None: it can)."""
    if st.session_state.get("cfd_code") != "code_saturne":
        return None
    from app.saturne import check_supported
    try:
        check_supported(working)
    except ValueError as exc:
        return str(exc)
    return None


def render_case_output(working: dict) -> None:
    st.divider()
    unsupported = _saturne_unsupported(working)
    if unsupported:
        st.warning(f"**Not available with code_saturne.** {unsupported} Switch **CFD code** to "
                   "OpenFOAM in the sidebar to generate this case.", icon="⚠️")
    if st.button("Generate Case Folder", type="primary", key="btn_generate",
                 disabled=bool(unsupported)):
        with st.spinner("Generating case files..."):
            try:
                stl = {"bytes": st.session_state.get("stl_bytes"),
                       "name": st.session_state.get("stl_name", "geometry"),
                       "bbox": st.session_state.get("stl_bbox")}
                zip_bytes, zip_name = generate_case_zip(working, stl)
                st.session_state["zip_bytes"] = zip_bytes
                st.session_state["zip_name"] = zip_name
            except ValueError as exc:             # unsupported setup: a message, not a traceback
                st.session_state["zip_bytes"] = None
                st.error(f"**Cannot generate this case:** {exc}")
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
    render_run_panel(working)
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
                        base_url=st.session_state.get("base_url"),
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
