"""Physics validation panel (Re, Mach, y+, CFL, resolution)."""

from __future__ import annotations

import streamlit as st

from app.validation import run_checks


def render_physics_checks(working: dict) -> None:
    st.divider()
    st.subheader("Physics Checks")

    checks = run_checks(working)
    n_warn = sum(c["level"] == "warning" for c in checks)
    n_err = sum(c["level"] == "error" for c in checks)
    n_ok = sum(c["level"] == "ok" for c in checks)

    if n_err:
        st.caption(f"🚨 {n_err} error(s) · ⚠️ {n_warn} warning(s) — review before generating")
    elif n_warn:
        st.caption(f"⚠️ {n_warn} warning(s) · {n_ok} check(s) passed")
    else:
        st.caption(f"✅ All {n_ok} checks passed")

    ok_titles = []
    for c in checks:
        msg = f"**{c['title']}**  \n{c['detail']}"
        if c["level"] == "error":
            st.error(msg, icon="🚨")
        elif c["level"] == "warning":
            st.warning(msg, icon="⚠️")
        elif c["level"] == "info":
            st.info(msg, icon="ℹ️")
        else:
            ok_titles.append(c["title"])
    if ok_titles:
        st.success("  \n".join(f"✔ {t}" for t in ok_titles))
