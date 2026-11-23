"""'Adjust Parameters' panel — returns the analysis with user overrides applied."""

from __future__ import annotations

import streamlit as st

from app.editor import apply_overrides, defaults_from_analysis
from app.generators.base import MAX_REFINEMENT, MIN_REFINEMENT

_TURB = ["laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras"]

# widget key → override key understood by app.editor.apply_overrides
_OVERRIDE_KEYS = {
    "pe_inlet_speed": "inlet_speed",
    "pe_nu": "nu",
    "pe_turb_model": "turb_model",
    "pe_nx": "nx",
    "pe_ny": "ny",
    "pe_domain_x": "domain_x",
    "pe_domain_y": "domain_y",
    "pe_mesh_refinement": "mesh_refinement",
    "pe_n_cores": "n_cores",
    "pe_snappy_refine_min": "snappy_refine_min",
    "pe_snappy_refine_max": "snappy_refine_max",
    "pe_snappy_feature_level": "snappy_feature_level",
    "pe_snappy_n_layers": "snappy_n_layers",
    "pe_snappy_upstream": "snappy_upstream",
    "pe_snappy_downstream": "snappy_downstream",
    "pe_snappy_side": "snappy_side",
}


def render_parameter_editor(analysis: dict) -> dict:
    """Render the editor and return a *working* analysis with overrides applied."""
    st.divider()
    defs = defaults_from_analysis(analysis)
    solver = analysis.get("solver", "")
    is_stl = analysis.get("geometry_type") == "stl_import"

    with st.expander("✏️ Adjust Parameters", expanded=False):
        st.caption("Override the LLM's values before generating. "
                   "Physics checks update automatically when you change a value.")

        st.markdown("**Flow**")
        c1, c2, c3 = st.columns(3)
        with c1:
            speed = float(defs["inlet_speed"] or 0.0)
            if speed > 0:
                st.number_input("Inlet speed (m/s)", min_value=1e-4, max_value=1e4,
                                value=min(max(speed, 1e-4), 1e4), format="%.4g", key="pe_inlet_speed",
                                help="Magnitude of the inlet velocity vector (direction is kept).")
            else:                    # dam break, natural convection, closed cavity: nothing to edit
                st.session_state.pop("pe_inlet_speed", None)
                st.caption("**Inlet speed:** none. This case has no inflow; the flow is driven "
                           "by gravity, buoyancy or a moving wall.")
        with c2:
            st.number_input("Kinematic viscosity ν (m²/s)", min_value=1e-8, max_value=1e-2,
                            value=float(defs["nu"]), format="%.3e", key="pe_nu",
                            help="Air ≈ 1.5 × 10⁻⁵  ·  Water ≈ 1 × 10⁻⁶")
        with c3:
            if solver != "interFoam":
                turb = defs["turb_model"] if defs["turb_model"] in _TURB else "kEpsilon"
                st.selectbox("Turbulence model", options=_TURB, index=_TURB.index(turb),
                             key="pe_turb_model",
                             help="laminar: low Re, no turbulence model · kEpsilon: internal flows · kOmegaSST: separation / external aero · "
                                  "SpalartAllmaras: attached external aero")
            else:
                st.markdown("*interFoam uses a laminar model.*")

        if analysis.get("geometry_type") == "imported_mesh":
            st.markdown("**Mesh**")
            st.caption("Mesh imported from SALOME: change it in SALOME and import it again.")
        elif is_stl:
            _snappy_inputs(defs)
        else:
            _mesh_inputs(defs)

        st.markdown("**Parallel**")
        p1, p2 = st.columns([1, 3])
        with p1:
            st.number_input("CPU cores", min_value=1, max_value=256, step=1,
                            value=int(defs.get("n_cores", 1)), key="pe_n_cores",
                            help="1 = serial. >1 = MPI run: decomposePar → mpirun → reconstructPar.")
        with p2:
            n = int(st.session_state.get("pe_n_cores", 1))
            if n > 1:
                st.info(f"Allrun will run: `decomposePar` → `mpirun -np {n} {solver} -parallel` "
                        "→ `reconstructPar`", icon="⚙️")
            else:
                st.caption("Serial: `blockMesh` → solver")

    overrides = {out: st.session_state[key] for key, out in _OVERRIDE_KEYS.items()
                 if key in st.session_state}
    if "pe_snappy_max_cells_m" in st.session_state:
        overrides["snappy_max_cells"] = int(st.session_state["pe_snappy_max_cells_m"] * 1e6)
    # nx/ny/domain only apply to single-block meshes; drop stale values otherwise
    if not defs["single_block"]:
        for k in ("nx", "ny", "domain_x", "domain_y"):
            overrides.pop(k, None)
    return apply_overrides(analysis, overrides)


def _mesh_inputs(defs: dict) -> None:
    st.markdown("**Mesh**")
    if defs["single_block"]:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.number_input("nx (streamwise cells)", min_value=10, max_value=2000, step=10,
                            value=int(defs["nx"]), key="pe_nx")
        with m2:
            st.number_input("ny (wall-normal cells)", min_value=5, max_value=1000, step=5,
                            value=int(defs["ny"]), key="pe_ny")
        with m3:
            st.number_input("Domain length x (m)", min_value=1e-3, max_value=1e4,
                            value=float(defs["domain_x"]), format="%.4g", key="pe_domain_x")
        with m4:
            st.number_input("Domain height y (m)", min_value=1e-3, max_value=1e4,
                            value=float(defs["domain_y"]), format="%.4g", key="pe_domain_y")
    else:
        st.caption(
            f"This geometry is a multi-block mesh sized from its geometry parameters "
            f"({defs['domain_x']:.4g} m × {defs['domain_y']:.4g} m). Use the refinement factor "
            "to scale the cell count of every block."
        )
        st.slider("Mesh refinement factor", min_value=MIN_REFINEMENT, max_value=MAX_REFINEMENT,
                  value=float(defs.get("mesh_refinement", 1.0)), step=0.25, key="pe_mesh_refinement",
                  help="2 = twice as many cells in each direction (≈4× in 2-D).")


def _snappy_inputs(defs: dict) -> None:
    st.markdown("**snappyHexMesh Refinement**")
    st.caption("The background mesh spans the STL bounding box × the expansion factors; "
               "snappyHexMesh then refines it towards the surface.")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.number_input("Min refine level", min_value=0, max_value=6, step=1,
                        value=int(defs["snappy_refine_min"]), key="pe_snappy_refine_min")
    with s2:
        st.number_input("Max refine level", min_value=1, max_value=8, step=1,
                        value=int(defs["snappy_refine_max"]), key="pe_snappy_refine_max",
                        help="Each level halves the cell size.")
    with s3:
        st.number_input("Feature edge level", min_value=1, max_value=8, step=1,
                        value=int(defs["snappy_feature_level"]), key="pe_snappy_feature_level")
    with s4:
        st.number_input("Prism layers", min_value=0, max_value=10, step=1,
                        value=int(defs["snappy_n_layers"]), key="pe_snappy_n_layers",
                        help="0 disables boundary-layer cells.")
    s5, s6, s7, s8 = st.columns(4)
    with s5:
        st.number_input("Max cells (×10⁶)", min_value=0.1, max_value=50.0, step=0.5,
                        value=float(defs["snappy_max_cells"]) / 1e6, format="%.2g",
                        key="pe_snappy_max_cells_m")
    with s6:
        st.number_input("Upstream (×L)", min_value=0.5, max_value=20.0, step=0.5,
                        value=float(defs["snappy_upstream"]), format="%.1f", key="pe_snappy_upstream")
    with s7:
        st.number_input("Downstream (×L)", min_value=0.5, max_value=30.0, step=0.5,
                        value=float(defs["snappy_downstream"]), format="%.1f", key="pe_snappy_downstream")
    with s8:
        st.number_input("Side / top (×L)", min_value=0.5, max_value=20.0, step=0.5,
                        value=float(defs["snappy_side"]), format="%.1f", key="pe_snappy_side")
