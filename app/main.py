"""
CFD Case Generator — Powered by Claude
Streamlit application entry point.

Run with:
    streamlit run app/main.py
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports work regardless of
# the working directory when Streamlit is invoked.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

# ── Page config must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="CFD Case Generator",
    page_icon="🌊",
    layout="wide",
)

import os as _os

from app.llm import (                            # noqa: E402
    analyze_cfd_problem,
    explain_case,
    ollama_is_reachable,
    list_ollama_models,
)
from app.generators import GENERATORS            # noqa: E402
from app.utils import zip_directory              # noqa: E402
from app.geometry_sketches import get_sketch, get_label  # noqa: E402
from app.validation import run_checks            # noqa: E402
from app.editor import apply_overrides, defaults_from_analysis  # noqa: E402
from app.log_parser import parse_foam_log, is_converged  # noqa: E402
from app.stl_parser import stl_bounding_box              # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("CFD Case Generator")

    # ── Backend selection ─────────────────────────────────────────────────────
    st.markdown("**Backend**")
    backend_choice = st.selectbox(
        "LLM backend",
        options=["Claude API (recommended)", "Ollama (local, free)"],
        label_visibility="collapsed",
    )
    _use_ollama = backend_choice.startswith("Ollama")

    st.divider()

    if not _use_ollama:
        # ── Claude API ────────────────────────────────────────────────────────
        st.markdown("**Claude API**")
        _env_key = _os.environ.get("ANTHROPIC_API_KEY", "")
        if _env_key:
            st.success("API key set via environment variable.", icon="✅")

        api_key_input = st.text_input(
            "Anthropic API Key",
            value=_env_key,
            type="password",
            placeholder="sk-ant-...",
            help="Your key is used only for this session and never stored.",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input
        elif not _env_key:
            st.error("API key not set.", icon="🔴")

        st.session_state["backend"] = "claude"
        st.session_state["model"] = "claude-sonnet-4-6"

    else:
        # ── Ollama ────────────────────────────────────────────────────────────
        st.markdown("**Ollama (local)**")
        _reachable = ollama_is_reachable()

        if _reachable:
            st.success("Ollama is running.", icon="✅")
            _available = list_ollama_models()
            if not _available:
                _available = ["mistral"]
                st.warning("No models found — defaulting to mistral.", icon="⚠️")
        else:
            st.error("Ollama not reachable.", icon="🔴")
            st.caption("`ollama serve`")
            _available = ["mistral"]

        _default_idx = _available.index("mistral") if "mistral" in _available else 0
        selected_model = st.selectbox(
            "Model",
            options=_available,
            index=_default_idx,
        )
        ollama_timeout = st.slider(
            "Timeout (s)",
            min_value=60,
            max_value=600,
            value=300,
            step=30,
            help="How long to wait for the model to respond. Large models need more time.",
        )
        st.caption("Make sure Ollama is running: `ollama serve`")

        st.session_state["backend"] = "ollama"
        st.session_state["model"] = selected_model
        st.session_state["ollama_timeout"] = ollama_timeout
        # Clear any stale Claude key readiness so button logic works
        st.session_state.setdefault("api_key", "")

    # ── Geometry sketch (shown after analysis is complete) ────────────────────
    _analysis_state = st.session_state.get("analysis")
    if _analysis_state:
        _geo_type = _analysis_state.get("geometry_type", "channel")
        _sketch, _sketch_desc = get_sketch(_geo_type)
        _geo_label = get_label(_geo_type)
        st.divider()
        st.markdown(f"**Geometry — {_geo_label}**")
        st.caption(_sketch_desc)
        st.code(_sketch, language=None)

    st.divider()
    st.markdown("**Supported Solvers**")
    st.markdown(
        """
| Solver | Description |
|---|---|
| **simpleFoam** | Steady-state incompressible, single-phase |
| **pimpleFoam** | Transient incompressible (PIMPLE algorithm) |
| **buoyantSimpleFoam** | Steady natural convection / buoyancy |
| **interFoam** | Two-phase VOF (free surface, dam break) |

---
**How it works**

1. Pick a backend and configure credentials above.
2. Describe your CFD problem in plain English.
3. The LLM analyses the problem and selects a solver.
4. Review the summary, then generate a complete OpenFOAM case.
5. Download the `.zip` and run it with `blockMesh && <solver>`.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

_backend_label = "Claude" if st.session_state.get("backend", "claude") == "claude" else st.session_state.get("model", "Ollama")
st.title(f"CFD Case Generator — Powered by {_backend_label}")
st.markdown(
    "Describe your CFD problem in natural language and receive a "
    "complete, ready-to-run **OpenFOAM** case folder."
)

# ── Optional: STL geometry upload ────────────────────────────────────────────

with st.expander("🗂 Upload STL geometry (optional — enables snappyHexMesh workflow)", expanded=False):
    st.caption(
        "Upload a surface STL file to generate a **snappyHexMesh** case around your geometry. "
        "The background mesh bounds are computed automatically from the STL bounding box. "
        "Describe the flow physics below as usual — the solver and BCs are still chosen by the LLM."
    )
    _stl_file = st.file_uploader(
        "STL file",
        type=["stl"],
        key="stl_upload",
        help="Binary or ASCII STL. The file stem is used as the patch name (e.g. 'body.stl' → patch 'body').",
    )
    if _stl_file is not None:
        _stl_bytes = _stl_file.read()
        _stl_stem  = Path(_stl_file.name).stem
        try:
            _bbox = stl_bounding_box(_stl_bytes)
            xmin, xmax, ymin, ymax, zmin, zmax = _bbox
            st.session_state["stl_bytes"]    = _stl_bytes
            st.session_state["stl_name"]     = _stl_stem
            st.session_state["stl_bbox"]     = _bbox
            st.session_state["stl_filename"] = _stl_file.name
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            st.success(
                f"**{_stl_file.name}** loaded — "
                f"{len(_stl_bytes):,} bytes  \n"
                f"Bounding box: "
                f"x [{xmin:.4g}, {xmax:.4g}] ({dx:.4g} m)  ·  "
                f"y [{ymin:.4g}, {ymax:.4g}] ({dy:.4g} m)  ·  "
                f"z [{zmin:.4g}, {zmax:.4g}] ({dz:.4g} m)",
                icon="✅",
            )
            st.info(
                "Background mesh will extend **2× upstream**, **5× downstream**, **2× on each side** of the STL extents. "
                "snappyHexMesh will refine to the surface with 2 refinement levels and add 3 prism layers.",
                icon="ℹ️",
            )
        except ValueError as _e:
            st.error(f"Could not parse STL: {_e}", icon="🚨")
    elif "stl_bytes" in st.session_state:
        # Previously uploaded — show info and offer to clear
        _fn = st.session_state.get("stl_filename", "geometry.stl")
        st.info(f"Using previously uploaded **{_fn}**. Re-upload to replace, or clear below.")
        if st.button("Clear STL", key="clear_stl"):
            for _k in ("stl_bytes", "stl_name", "stl_bbox", "stl_filename"):
                st.session_state.pop(_k, None)
            st.rerun()

# ── Step 1: Problem description input ────────────────────────────────────────

user_description = st.text_area(
    label="Problem Description",
    placeholder=(
        "Example: Steady-state airflow around a 2D cylinder at Re=200 with "
        "a freestream velocity of 1 m/s. The fluid is air at standard conditions."
    ),
    height=150,
    key="user_description",
)

_backend = st.session_state.get("backend", "claude")
_ollama_ok = _backend == "ollama" and ollama_is_reachable()
_claude_ok = _backend == "claude" and bool(st.session_state.get("api_key"))
_ready = _ollama_ok or _claude_ok

_btn_help = ""
if not _ready:
    if _backend == "claude":
        _btn_help = "Enter your Anthropic API key in the sidebar first."
    else:
        _btn_help = "Ollama is not reachable. Run: ollama serve"

analyse_btn = st.button(
    "Analyse my problem",
    type="primary",
    disabled=not (user_description.strip() and _ready),
    help=_btn_help,
)

# ── Trigger analysis ─────────────────────────────────────────────────────────

if analyse_btn and user_description.strip():
    with st.spinner("Analysing your CFD problem..."):
        try:
            # If an STL is uploaded, inject a hint so the LLM chooses external-aero BCs
            _desc_for_llm = user_description
            if st.session_state.get("stl_bytes"):
                _stl_hint = (
                    "\n\n[Note: the user has uploaded a custom STL geometry. "
                    "This is an external flow simulation around a user-provided 3-D body. "
                    "Choose a solver and boundary conditions appropriate for external aerodynamics. "
                    "Set geometry_type to 'channel' — the actual geometry handling is done by snappyHexMesh.]"
                )
                _desc_for_llm = user_description + _stl_hint

            analysis = analyze_cfd_problem(
                _desc_for_llm,
                backend=st.session_state.get("backend", "claude"),
                model=st.session_state.get("model", "claude-sonnet-4-6"),
                api_key=st.session_state.get("api_key"),
                ollama_timeout=st.session_state.get("ollama_timeout", 300),
            )
            # Override geometry_type to stl_import when an STL is present
            if st.session_state.get("stl_bytes"):
                analysis["geometry_type"] = "stl_import"

            st.session_state["analysis"] = analysis
            st.session_state["zip_bytes"] = None  # reset any prior download
            st.session_state.pop("case_explanation", None)  # reset explanation
            # Clear parameter-editor overrides so widgets reset to new LLM values
            for _k in [k for k in st.session_state if k.startswith("pe_")]:
                del st.session_state[_k]
        except RuntimeError as exc:
            st.error(f"**API error:** {exc}")
            st.stop()
        except ValueError as exc:
            st.error(f"**Parsing error:** {exc}")
            st.stop()

# ── Step 2: Display summary ───────────────────────────────────────────────────

if "analysis" in st.session_state and st.session_state["analysis"]:
    analysis: dict = st.session_state["analysis"]

    st.divider()
    st.subheader("Analysis Summary")

    # Solver choice
    solver = analysis.get("solver", "unknown")
    justification = analysis.get("solver_justification", "")
    st.info(f"**Solver:** `{solver}`  \n{justification}")

    col1, col2 = st.columns(2)

    # Fluid properties table
    with col1:
        st.markdown("**Fluid Properties**")
        fluid = analysis.get("fluid", {})
        fluid_table = {
            "Property": ["Name", "Kinematic viscosity ν (m²/s)", "Density ρ (kg/m³)"],
            "Value": [
                fluid.get("name", "—"),
                f"{fluid.get('nu', '—'):.2e}" if isinstance(fluid.get("nu"), (int, float)) else "—",
                f"{fluid.get('rho', '—'):.3g}" if isinstance(fluid.get("rho"), (int, float)) else "—",
            ],
        }
        st.table(fluid_table)

        re = analysis.get("Re")
        if re is not None:
            st.markdown(f"**Reynolds number:** `{re:.4g}`")

        is_transient = analysis.get("is_transient", False)
        st.markdown(f"**Flow regime:** {'Transient' if is_transient else 'Steady-state'}")

    # Suggested mesh info
    with col2:
        mesh = analysis.get("suggested_mesh", {})
        st.markdown("**Suggested Mesh**")
        mesh_table = {
            "Parameter": ["Type", "Approx. cell count", "Domain"],
            "Value": [
                mesh.get("type", "blockMesh"),
                f"{mesh.get('approximate_cell_count', '—'):,}" if isinstance(mesh.get("approximate_cell_count"), int) else "—",
                mesh.get("domain_description", "—"),
            ],
        }
        st.table(mesh_table)

    # Boundary conditions
    st.markdown("**Boundary Conditions**")
    bcs = analysis.get("boundary_conditions", [])
    if bcs:
        bc_rows = []
        for bc in bcs:
            values_str = ", ".join(
                f"{k}={v}" for k, v in bc.get("values", {}).items()
            )
            bc_rows.append({
                "Patch": bc.get("patch_name", "—"),
                "Type": bc.get("type", "—"),
                "Values": values_str or "—",
            })
        st.table(bc_rows)
    else:
        st.markdown("*No boundary conditions extracted.*")

    # Numerical schemes (expandable)
    schemes = analysis.get("numerical_schemes", {})
    if schemes:
        with st.expander("Numerical Schemes"):
            for key, val in schemes.items():
                if key != "justification":
                    st.markdown(f"- **{key}:** `{val}`")
            if "justification" in schemes:
                st.caption(schemes["justification"])

    # LLM-supplied warnings
    warnings = analysis.get("warnings", [])
    if warnings:
        warn_text = "\n".join(f"- {w}" for w in warnings)
        st.warning(f"**Potential issues:**\n{warn_text}")

    # ── Parameter editor ──────────────────────────────────────────────────────

    st.divider()
    _defs   = defaults_from_analysis(analysis)
    _solver = analysis.get("solver", "")

    with st.expander("✏️ Adjust Parameters", expanded=False):
        st.caption(
            "Override LLM defaults before generating. "
            "Physics checks update automatically when you change a value."
        )

        # ── Flow parameters ──────────────────────────────────────────────────
        st.markdown("**Flow**")
        _fc1, _fc2, _fc3 = st.columns(3)

        with _fc1:
            st.number_input(
                "Inlet speed (m/s)",
                min_value=1e-4,
                max_value=1e4,
                value=float(_defs["inlet_speed"]),
                format="%.4g",
                key="pe_inlet_speed",
                help="Magnitude of the inlet velocity vector.",
            )
        with _fc2:
            st.number_input(
                "Kinematic viscosity ν (m²/s)",
                min_value=1e-8,
                max_value=1e-2,
                value=float(_defs["nu"]),
                format="%.3e",
                key="pe_nu",
                help="Air ≈ 1.5 × 10⁻⁵  ·  Water ≈ 1 × 10⁻⁶",
            )
        with _fc3:
            if _solver != "interFoam":
                st.selectbox(
                    "Turbulence model",
                    options=["kEpsilon", "kOmegaSST", "SpalartAllmaras"],
                    index=["kEpsilon", "kOmegaSST", "SpalartAllmaras"].index(
                        _defs["turb_model"]
                        if _defs["turb_model"] in ("kEpsilon", "kOmegaSST", "SpalartAllmaras")
                        else "kEpsilon"
                    ),
                    key="pe_turb_model",
                    help=(
                        "kEpsilon: internal flows · "
                        "kOmegaSST: separation / external aero · "
                        "SpalartAllmaras: attached external aero"
                    ),
                )
            else:
                st.markdown("*interFoam uses laminar model.*")

        # ── Mesh / snappy parameters ─────────────────────────────────────────
        _is_stl = analysis.get("geometry_type") == "stl_import"

        if _is_stl:
            st.markdown("**snappyHexMesh Refinement**")
            st.caption(
                "Background mesh bounds are computed from the STL bounding box "
                "× expansion factors. snappyHexMesh then refines the background mesh "
                "to the surface geometry."
            )

            # Row 1: refinement levels + layers
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            with _sc1:
                st.number_input(
                    "Min refine level",
                    min_value=0, max_value=6, step=1,
                    value=int(_defs["snappy_refine_min"]),
                    key="pe_snappy_refine_min",
                    help="Minimum cell refinement level on the STL surface (0 = background cell size).",
                )
            with _sc2:
                st.number_input(
                    "Max refine level",
                    min_value=1, max_value=8, step=1,
                    value=int(_defs["snappy_refine_max"]),
                    key="pe_snappy_refine_max",
                    help="Maximum cell refinement level on the STL surface (each level halves cell size).",
                )
            with _sc3:
                st.number_input(
                    "Feature edge level",
                    min_value=1, max_value=8, step=1,
                    value=int(_defs["snappy_feature_level"]),
                    key="pe_snappy_feature_level",
                    help="Refinement level along extracted feature edges (sharp corners/ridges).",
                )
            with _sc4:
                st.number_input(
                    "Prism layers",
                    min_value=0, max_value=10, step=1,
                    value=int(_defs["snappy_n_layers"]),
                    key="pe_snappy_n_layers",
                    help="Number of prismatic boundary-layer cells grown on the STL surface. 0 = disabled.",
                )

            # Row 2: max cells + domain expansion factors
            _sc5, _sc6, _sc7, _sc8 = st.columns(4)
            with _sc5:
                st.number_input(
                    "Max cells (×10⁶)",
                    min_value=0.1, max_value=50.0, step=0.5,
                    value=float(_defs["snappy_max_cells"]) / 1e6,
                    format="%.2g",
                    key="pe_snappy_max_cells_m",
                    help="Maximum total cell count after snappy refinement (in millions).",
                )
            with _sc6:
                st.number_input(
                    "Upstream (×L)",
                    min_value=0.5, max_value=20.0, step=0.5,
                    value=float(_defs["snappy_upstream"]),
                    format="%.1f",
                    key="pe_snappy_upstream",
                    help="Background mesh extension upstream of STL, as a multiple of STL length.",
                )
            with _sc7:
                st.number_input(
                    "Downstream (×L)",
                    min_value=0.5, max_value=30.0, step=0.5,
                    value=float(_defs["snappy_downstream"]),
                    format="%.1f",
                    key="pe_snappy_downstream",
                    help="Background mesh extension downstream (wake). 5× is typical for bluff bodies.",
                )
            with _sc8:
                st.number_input(
                    "Side / top (×L)",
                    min_value=0.5, max_value=20.0, step=0.5,
                    value=float(_defs["snappy_side"]),
                    format="%.1f",
                    key="pe_snappy_side",
                    help="Background mesh extension on top, bottom, front, back faces.",
                )
        else:
            st.markdown("**Mesh**")
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)

            with _mc1:
                st.number_input(
                    "nx (streamwise cells)",
                    min_value=10,
                    max_value=1000,
                    step=10,
                    value=int(_defs["nx"]),
                    key="pe_nx",
                    help="Cell count in the streamwise (x) direction.",
                )
            with _mc2:
                st.number_input(
                    "ny (wall-normal cells)",
                    min_value=5,
                    max_value=500,
                    step=5,
                    value=int(_defs["ny"]),
                    key="pe_ny",
                    help="Cell count in the wall-normal (y) direction.",
                )
            with _mc3:
                st.number_input(
                    "Domain length x (m)",
                    min_value=0.01,
                    max_value=10000.0,
                    value=float(_defs["domain_x"]),
                    format="%.4g",
                    key="pe_domain_x",
                    help="Total streamwise extent of the computational domain.",
                )
            with _mc4:
                st.number_input(
                    "Domain height y (m)",
                    min_value=0.01,
                    max_value=10000.0,
                    value=float(_defs["domain_y"]),
                    format="%.4g",
                    key="pe_domain_y",
                    help="Total wall-normal (cross-stream) extent of the domain.",
                )

        # ── Parallel run ─────────────────────────────────────────────────────
        st.markdown("**Parallel**")
        _pc1, _pc2 = st.columns([1, 3])
        with _pc1:
            st.number_input(
                "CPU cores",
                min_value=1,
                max_value=256,
                step=1,
                value=int(_defs.get("n_cores", 1)),
                key="pe_n_cores",
                help=(
                    "1 = serial run (default).  "
                    ">1 = parallel via MPI — generates decomposeParDict "
                    "and updates Allrun to call decomposePar / reconstructPar."
                ),
            )
        with _pc2:
            _nc_preview = int(st.session_state.get("pe_n_cores", 1))
            if _nc_preview > 1:
                st.info(
                    f"Allrun will run: `decomposePar` → `mpirun -np {_nc_preview} {_solver} -parallel` → `reconstructPar`",
                    icon="⚙️",
                )
            else:
                st.caption("Serial: `runApplication blockMesh && runApplication <solver>`")

    # Build working analysis: LLM analysis + any user overrides from pe_* keys
    _overrides: dict = {}
    for _pe_key, _out_key in (
        ("pe_inlet_speed", "inlet_speed"),
        ("pe_nu",          "nu"),
        ("pe_turb_model",  "turb_model"),
        ("pe_nx",          "nx"),
        ("pe_ny",          "ny"),
        ("pe_domain_x",    "domain_x"),
        ("pe_domain_y",    "domain_y"),
        ("pe_n_cores",     "n_cores"),
        # snappy-specific overrides (only active when geometry_type == 'stl_import')
        ("pe_snappy_refine_min",    "snappy_refine_min"),
        ("pe_snappy_refine_max",    "snappy_refine_max"),
        ("pe_snappy_feature_level", "snappy_feature_level"),
        ("pe_snappy_n_layers",      "snappy_n_layers"),
        ("pe_snappy_upstream",      "snappy_upstream"),
        ("pe_snappy_downstream",    "snappy_downstream"),
        ("pe_snappy_side",          "snappy_side"),
    ):
        if _pe_key in st.session_state:
            _overrides[_out_key] = st.session_state[_pe_key]
    # max_cells widget stores value in millions — convert back
    if "pe_snappy_max_cells_m" in st.session_state:
        _overrides["snappy_max_cells"] = int(st.session_state["pe_snappy_max_cells_m"] * 1e6)

    _working = apply_overrides(analysis, _overrides)

    # ── Physics validation panel ──────────────────────────────────────────────

    st.divider()
    st.subheader("Physics Checks")

    _checks = run_checks(_working)
    _n_warn  = sum(1 for c in _checks if c["level"] == "warning")
    _n_err   = sum(1 for c in _checks if c["level"] == "error")
    _n_ok    = sum(1 for c in _checks if c["level"] == "ok")

    # Summary badge
    if _n_err > 0:
        st.caption(f"🚨 {_n_err} error(s) · ⚠️ {_n_warn} warning(s) — review before generating")
    elif _n_warn > 0:
        st.caption(f"⚠️ {_n_warn} warning(s) · {_n_ok} check(s) passed")
    else:
        st.caption(f"✅ All {_n_ok} checks passed")

    # Non-OK checks displayed individually; OK checks collapsed into one row
    _ok_titles: list[str] = []
    for _c in _checks:
        _lvl = _c["level"]
        _msg = f"**{_c['title']}**  \n{_c['detail']}"
        if _lvl == "error":
            st.error(_msg, icon="🚨")
        elif _lvl == "warning":
            st.warning(_msg, icon="⚠️")
        elif _lvl == "info":
            st.info(_msg, icon="ℹ️")
        else:
            _ok_titles.append(_c["title"])

    if _ok_titles:
        st.success("  \n".join(f"✔ {t}" for t in _ok_titles))

    # ── Step 3: Generate case folder ─────────────────────────────────────────

    st.divider()
    generate_btn = st.button("Generate Case Folder", type="primary")

    if generate_btn:
        solver_key = _working.get("solver", "")
        generator_cls = GENERATORS.get(solver_key)

        if generator_cls is None:
            st.error(
                f"No generator found for solver `{solver_key}`. "
                f"Supported: {list(GENERATORS.keys())}"
            )
        else:
            with st.spinner("Generating case files..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        case_dir = Path(tmpdir) / solver_key
                        generator = generator_cls()
                        # Inject STL data when geometry_type is stl_import
                        _gen_analysis = dict(_working)
                        if _gen_analysis.get("geometry_type") == "stl_import":
                            _gen_analysis["_stl_bytes"] = st.session_state.get("stl_bytes")
                            _gen_analysis["_stl_name"]  = st.session_state.get("stl_name", "geometry")
                            _gen_analysis["_stl_bbox"]  = st.session_state.get("stl_bbox")
                        else:
                            _gen_analysis = _working
                        generator.generate(_gen_analysis, case_dir)
                        zip_bytes = zip_directory(case_dir)
                        st.session_state["zip_bytes"] = zip_bytes
                        st.session_state["zip_name"] = f"{solver_key}_case.zip"
                except Exception as exc:
                    st.error(f"**Generation error:** {exc}")
                    raise

    # ── Download button (persists after generation) ───────────────────────────

    if st.session_state.get("zip_bytes"):
        st.success("Case folder generated successfully!")
        st.download_button(
            label="Download case.zip",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state.get("zip_name", "openfoam_case.zip"),
            mime="application/zip",
        )

        # ── LLM flow-physics explanation ──────────────────────────────────────

        with st.expander("🔍 Flow physics & what to expect", expanded=False):
            st.caption(
                "Ask the AI to explain the expected flow behaviour, convergence "
                "characteristics, and post-processing checklist for this case."
            )
            if st.button("Generate explanation", key="btn_explain"):
                with st.spinner("Generating explanation..."):
                    try:
                        _expl = explain_case(
                            _working,
                            st.session_state.get("user_description", ""),
                            backend=st.session_state.get("backend", "claude"),
                            model=st.session_state.get("model", "claude-sonnet-4-6"),
                            api_key=st.session_state.get("api_key"),
                            ollama_timeout=st.session_state.get("ollama_timeout", 120),
                        )
                        st.session_state["case_explanation"] = _expl
                    except Exception as _ex:
                        st.error(f"Could not generate explanation: {_ex}")

            if st.session_state.get("case_explanation"):
                st.markdown(st.session_state["case_explanation"])

        # "What to do next" instructions
        with st.expander("What to do next"):
            _wn_solver  = _working.get("solver", "<solver>")
            _wn_zip     = st.session_state.get("zip_name", "case.zip")
            _wn_cores   = int(_working.get("n_cores", 1))
            _wn_is_stl  = _working.get("geometry_type") == "stl_import"
            _wn_stl_name = st.session_state.get("stl_name", "geometry")

            if _wn_is_stl:
                _wn_stepbystep = f"""\
blockMesh
surfaceFeatureExtract           # extract .eMesh edges from the STL
snappyHexMesh -overwrite        # snap + refine + add BL layers
checkMesh                       # verify mesh quality before solving
{_wn_solver} 2>&1 | tee {_wn_solver}.log"""
                _wn_allrun_note = (
                    "The included `Allrun` script runs `blockMesh` → "
                    "`surfaceFeatureExtract` → `snappyHexMesh -overwrite` → solver "
                    + ("in parallel on " + str(_wn_cores) + " cores." if _wn_cores > 1 else "in serial.")
                )
                _wn_tips = f"""\
- STL file is at `constant/triSurface/{_wn_stl_name}.stl` — replace to test a new geometry.
- Tune `snappyHexMeshDict` → `refinementSurfaces level` to control surface resolution.
- Increase `nSurfaceLayers` if the boundary layer is too coarse.
- Run `checkMesh` before the solver; fix negative volumes if reported.
- Adjust relaxation factors in `system/fvSolution` if divergence occurs."""
            else:
                _wn_stepbystep = f"""\
blockMesh
checkMesh
{_wn_solver} 2>&1 | tee {_wn_solver}.log"""
                _wn_allrun_note = (
                    "The included `Allrun` script runs `blockMesh` then the solver "
                    + ("in parallel on " + str(_wn_cores) + " cores." if _wn_cores > 1 else "in serial.")
                )
                _wn_tips = """\
- Adjust relaxation factors in `system/fvSolution` if divergence occurs.
- Increase `endTime` in `system/controlDict` if the solver stops early."""

            st.markdown(
                f"""
**1. Extract the archive**
```bash
unzip {_wn_zip}
cd {_wn_solver}/
```

**2. Run the case (simplest)**
```bash
chmod +x Allrun Allclean
./Allrun
```
{_wn_allrun_note}

**Or step-by-step:**
```bash
{_wn_stepbystep}
```

**3. Monitor convergence**
```bash
foamLog {_wn_solver}.log     # extracts logs/p_0, logs/Ux_0, …
```
Or paste the log into the **Convergence Monitor** below.

**4. Post-process**
```bash
paraFoam                      # or: touch case.foam && paraview case.foam
```

**5. Clean up**
```bash
./Allclean
```

**Tips**
{_wn_tips}
                """
            )

# ─────────────────────────────────────────────────────────────────────────────
# Convergence Monitor — always visible so users can analyse any log
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Convergence Monitor")
st.caption(
    "Paste or upload an OpenFOAM solver log to plot residuals and check convergence. "
    "Works with logs from any case, not just those generated here."
)

with st.expander("Analyse a solver log", expanded=False):
    _log_method = st.radio(
        "Input method",
        ["Paste text", "Upload file"],
        horizontal=True,
        key="log_input_method",
    )

    _log_text = ""
    if _log_method == "Paste text":
        _log_text = st.text_area(
            "Solver log",
            height=160,
            placeholder="Paste the contents of simpleFoam.log / pimpleFoam.log / … here",
            key="log_paste",
        )
    else:
        _log_file = st.file_uploader(
            "Upload solver log",
            type=["log", "txt", "out"],
            key="log_file",
        )
        if _log_file is not None:
            _log_text = _log_file.read().decode("utf-8", errors="replace")

    if _log_text.strip():
        _parsed = parse_foam_log(_log_text)

        if not _parsed["fields"]:
            st.warning(
                "No residual lines found — is this an OpenFOAM solver log? "
                "Expected lines containing 'Solving for …, Initial residual = …'.",
                icon="⚠️",
            )
        else:
            st.caption(
                f"Fields detected: `{'`, `'.join(_parsed['fields'])}`  ·  "
                f"{_parsed['n_iters']} iteration(s)"
            )

            # Build tidy dataframe
            import pandas as _pd
            import altair as _alt

            _rows = []
            for _field in _parsed["fields"]:
                for _i, _r in enumerate(_parsed["residuals"][_field], start=1):
                    _rows.append({"Iteration": _i, "Residual": _r, "Field": _field})
            _df = _pd.DataFrame(_rows)

            # Log-scale residual chart
            _chart = (
                _alt.Chart(_df)
                .mark_line(point=False)
                .encode(
                    x=_alt.X("Iteration:Q"),
                    y=_alt.Y(
                        "Residual:Q",
                        scale=_alt.Scale(type="log"),
                        title="Initial residual (log scale)",
                    ),
                    color=_alt.Color("Field:N"),
                    tooltip=[
                        _alt.Tooltip("Iteration:Q"),
                        _alt.Tooltip("Field:N"),
                        _alt.Tooltip("Residual:Q", format=".3e"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(_chart, use_container_width=True)

            # Convergence verdict
            if is_converged(_parsed["residuals"]):
                st.success(
                    "All final residuals < 1 × 10⁻⁴ — case appears converged.",
                    icon="✅",
                )
            else:
                _final = {
                    f: vals[-1]
                    for f, vals in _parsed["residuals"].items()
                    if vals
                }
                _worst_field = max(_final, key=_final.get)
                st.warning(
                    f"Not yet converged.  "
                    f"Largest final residual: `{_worst_field}` = {_final[_worst_field]:.3e}",
                    icon="⚠️",
                )
