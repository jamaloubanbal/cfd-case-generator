"""Sidebar: LLM backend selection, credentials, geometry sketch, help."""

from __future__ import annotations

import os

import streamlit as st

from app.geometry_sketches import get_label, get_sketch
from app.llm import DEFAULT_CLAUDE_MODEL, DEFAULT_OLLAMA_MODEL, list_ollama_models, ollama_is_reachable

CLAUDE_MODELS = list(dict.fromkeys([DEFAULT_CLAUDE_MODEL, "claude-opus-5-5",
                                    "claude-sonnet-5-5", "claude-haiku-5-5"]))


def render_sidebar() -> None:
    """Render the sidebar and store backend settings in ``st.session_state``.

    Keys set: ``backend``, ``model``, ``api_key``, ``ollama_timeout``.
    """
    with st.sidebar:
        st.title("CFD Case Generator")

        st.markdown("**Backend**")
        backend_choice = st.selectbox(
            "LLM backend",
            options=["Claude API (recommended)", "Ollama (local, free)"],
            label_visibility="collapsed",
            key="backend_choice",
        )
        st.divider()

        if backend_choice.startswith("Ollama"):
            _ollama_settings()
        else:
            _claude_settings()

        analysis = st.session_state.get("analysis")
        if analysis:
            geo_type = analysis.get("geometry_type", "channel")
            sketch, sketch_desc = get_sketch(geo_type)
            st.divider()
            st.markdown(f"**Geometry — {get_label(geo_type)}**")
            st.caption(sketch_desc)
            st.code(sketch, language=None)

        st.divider()
        st.markdown("**Supported Solvers**")
        st.markdown(
            """
| Solver | Description |
|---|---|
| **simpleFoam** | Steady incompressible, single-phase |
| **pimpleFoam** | Transient incompressible (PIMPLE) |
| **rhoSimpleFoam** | Steady compressible (Ma > 0.3) |
| **buoyantSimpleFoam** | Steady natural convection |
| **interFoam** | Two-phase VOF (free surface) |

---
**How it works**

1. Pick a backend and configure credentials above.
2. Describe your CFD problem in plain English.
3. The LLM selects a solver, BCs and geometry; values are sanity-checked.
4. Review and adjust, then generate a complete OpenFOAM case.
5. Download the `.zip` and run `./Allrun`.
            """
        )


def _claude_settings() -> None:
    st.markdown("**Claude API**")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        st.success("API key set via environment variable.", icon="✅")

    api_key_input = st.text_input(
        "Anthropic API Key",
        value=env_key,
        type="password",
        placeholder="sk-ant-...",
        help="Your key is used only for this session and never stored.",
    )
    st.session_state["api_key"] = api_key_input
    if not api_key_input:
        st.error("API key not set.", icon="🔴")

    model = st.selectbox(
        "Model",
        options=CLAUDE_MODELS,
        index=0,
        help="Set the CFD_CLAUDE_MODEL environment variable to change the default.",
        key="claude_model",
    )
    st.session_state["backend"] = "claude"
    st.session_state["model"] = model


def _ollama_settings() -> None:
    st.markdown("**Ollama (local)**")
    if ollama_is_reachable():
        st.success("Ollama is running.", icon="✅")
        available = list_ollama_models()
        if not available:
            available = [DEFAULT_OLLAMA_MODEL]
            st.warning(f"No models found — run: ollama pull {DEFAULT_OLLAMA_MODEL}", icon="⚠️")
    else:
        st.error("Ollama not reachable.", icon="🔴")
        st.caption("`ollama serve`")
        available = [DEFAULT_OLLAMA_MODEL]

    # Prefer the recommended model (Ollama lists "qwen2.5:7b" as-is, "mistral" as "mistral:latest")
    preferred = [DEFAULT_OLLAMA_MODEL, "qwen2.5:7b", "mistral:latest", "mistral"]
    default_idx = next((available.index(m) for m in preferred if m in available), 0)
    model = st.selectbox("Model", options=available, index=default_idx, key="ollama_model")
    timeout = st.slider(
        "Timeout (s)", min_value=60, max_value=600, value=300, step=30,
        help="How long to wait for the model to respond. Large models need more time.",
    )
    st.caption("Structured output needs Ollama ≥ 0.5. Models ≥ 7B give the best results.")

    st.session_state["backend"] = "ollama"
    st.session_state["model"] = model
    st.session_state["ollama_timeout"] = timeout
    st.session_state.setdefault("api_key", "")
