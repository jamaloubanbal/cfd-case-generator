"""Read-only summary of the (sanitised) LLM analysis."""

from __future__ import annotations

import streamlit as st


def _fmt(value, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else "—"


def render_analysis_summary(analysis: dict, notes: list[str]) -> None:
    st.divider()
    st.subheader("Analysis Summary")

    if analysis.get("_model_used"):
        st.caption(f"ℹ️ `{st.session_state.get('model', 'The selected model')}` was unavailable "
                   f"(busy or retired), so `{analysis['_model_used']}` answered instead.")
    st.info(f"**Solver:** `{analysis.get('solver', 'unknown')}`  \n"
            f"{analysis.get('solver_justification', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fluid Properties**")
        fluid = analysis.get("fluid", {})
        st.table({
            "Property": ["Name", "Kinematic viscosity ν (m²/s)", "Density ρ (kg/m³)"],
            "Value": [fluid.get("name", "—"), _fmt(fluid.get("nu"), ".2e"), _fmt(fluid.get("rho"), ".3g")],
        })
        if analysis.get("Re") is not None:
            st.markdown(f"**Reynolds number:** `{analysis['Re']:.4g}`")
        st.markdown(f"**Flow regime:** {'Transient' if analysis.get('is_transient') else 'Steady-state'}")

    with col2:
        mesh = analysis.get("suggested_mesh") or {}
        st.markdown("**Suggested Mesh**")
        cells = mesh.get("approximate_cell_count")
        st.table({
            "Parameter": ["Type", "Approx. cell count", "Domain"],
            "Value": [
                mesh.get("type", "blockMesh"),
                f"{cells:,}" if isinstance(cells, int) else "—",
                mesh.get("domain_description", "—"),
            ],
        })

    st.markdown("**Boundary Conditions**")
    bcs = analysis.get("boundary_conditions", [])
    if bcs:
        st.table([
            {
                "Patch": bc.get("patch_name", "—"),
                "Type": bc.get("type", "—"),
                "Values": ", ".join(f"{k}={v}" for k, v in (bc.get("values") or {}).items()
                                    if v is not None) or "—",
            }
            for bc in bcs
        ])
    else:
        st.markdown("*No boundary conditions extracted.*")

    schemes = analysis.get("numerical_schemes") or {}
    if schemes:
        with st.expander("Numerical Schemes (as suggested by the LLM)"):
            for key, val in schemes.items():
                if key != "justification":
                    st.markdown(f"- **{key}:** `{val}`")
            if "justification" in schemes:
                st.caption(schemes["justification"])
            st.caption("The generated case uses validated, solver-specific schemes from the templates.")

    if notes:
        st.info("**Automatic corrections** — these values from the model were implausible "
                "and have been fixed:\n" + "\n".join(f"- {n}" for n in notes), icon="🛠️")

    warnings = analysis.get("warnings") or []
    if warnings:
        st.warning("**Assumptions & potential issues:**\n" + "\n".join(f"- {w}" for w in warnings))
