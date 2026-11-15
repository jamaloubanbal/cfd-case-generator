"""Import a SALOME mesh (.unv) and edit the boundary roles of its groups."""

from __future__ import annotations

import shutil

import streamlit as st

from app.mesh_import import ROLES, check_roles, import_unv

_KEYS = ("imported_mesh", "mesh_filename", "mesh_roles", "mesh_role_notes")


def render_mesh_upload() -> None:
    with st.expander("🧩 Import a SALOME mesh (.unv): use your own geometry", expanded=False):
        st.caption("Mesh any geometry in SALOME, give each boundary a **group** (e.g. *inlet*, *entrée*, "
                   "*walls*), and export it with *File → Export → UNV*. The groups become boundary "
                   "patches; their roles are chosen by the LLM from the names and your description, "
                   "and you can correct them after the analysis.")
        if not shutil.which("ideasUnvToFoam"):
            st.info("Converting .unv files needs OpenFOAM. Start the app from a terminal where "
                    "OpenFOAM is sourced (`source /usr/share/openfoam/etc/bashrc`).", icon="ℹ️")
            return
        f = st.file_uploader("SALOME mesh (.unv)", type=["unv"], key="unv_upload")
        if f is not None and st.session_state.get("mesh_filename") != f.name:
            with st.spinner("Converting the mesh with ideasUnvToFoam…"):
                try:
                    mesh = import_unv(f.read(), f.name.rsplit(".", 1)[0])
                except (RuntimeError, ValueError) as exc:
                    st.error(f"Could not import the mesh: {exc}", icon="🚨")
                    return
            for k in _KEYS:
                st.session_state.pop(k, None)
            st.session_state.update(imported_mesh=mesh, mesh_filename=f.name)

        mesh = st.session_state.get("imported_mesh")
        if mesh is None:
            return
        x0, x1, y0, y1, z0, z1 = mesh.bbox
        st.success(f"**{st.session_state['mesh_filename']}**: {mesh.n_cells:,} cells, "
                   f"{len(mesh.patches)} boundary groups, extent {x1 - x0:.4g} × {y1 - y0:.4g} × "
                   f"{z1 - z0:.4g} m. Describe the flow below and click *Analyse*.", icon="✅")
        st.table([{"Group": p.name, "Faces": p.n_faces, "Area (m²)": f"{p.area:.4g}",
                   "Outward normal": "(" + ", ".join(f"{c:.2f}" for c in p.normal) + ")"}
                  for p in mesh.patches])
        for n in mesh.notes:
            st.warning(n, icon="⚠️")
        if st.button("Remove mesh", key="clear_mesh"):
            for k in _KEYS:
                st.session_state.pop(k, None)
            st.rerun()


def render_role_editor(analysis: dict) -> None:
    """After the analysis: show the group → role mapping and let the user change it."""
    mesh = st.session_state.get("imported_mesh")
    roles = st.session_state.get("mesh_roles")
    if analysis.get("geometry_type") != "imported_mesh" or mesh is None or roles is None:
        return
    st.markdown("**Boundary roles of the imported mesh** (change a role if it is wrong)")
    rows = [{"Group": p.name, "Faces": p.n_faces, "Role": roles.get(p.name, "wall")} for p in mesh.patches]
    edited = st.data_editor(
        rows, key="role_editor", hide_index=True, disabled=["Group", "Faces"],
        column_config={"Role": st.column_config.SelectboxColumn("Role", options=list(ROLES), required=True)},
    )
    st.session_state["mesh_roles"] = {r["Group"]: r["Role"] for r in edited}
    for n in st.session_state.get("mesh_role_notes", []):
        st.caption(f"ℹ️ {n}")
    for problem in check_roles(st.session_state["mesh_roles"], analysis.get("solver", "")):
        st.warning(f"Boundary roles: {problem}.", icon="⚠️")
