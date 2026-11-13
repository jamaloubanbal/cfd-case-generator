"""Sidebar: LLM backend selection, credentials, geometry sketch, help."""

from __future__ import annotations

import os

import streamlit as st

from app.geometry_sketches import get_label, get_sketch
from app.llm import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    OPENAI_COMPATIBLE_PROVIDERS,
    list_ollama_models,
    ollama_is_reachable,
)

CLAUDE_MODELS = list(dict.fromkeys([DEFAULT_CLAUDE_MODEL, "claude-opus-5-5",
                                    "claude-sonnet-5-5", "claude-haiku-5-5"]))


def render_sidebar() -> None:
    """Render the sidebar and store backend settings in ``st.session_state``.

    Keys set: ``backend``, ``model``, ``api_key``, ``base_url``, ``ollama_timeout``.
    """
    backends = {"Claude API (recommended)": "claude", "Ollama (local, free)": "ollama",
                **{p["label"]: key for key, p in OPENAI_COMPATIBLE_PROVIDERS.items()}}
    with st.sidebar:
        st.title("CFD Case Generator")

        st.markdown("**Backend**")
        backend_choice = st.selectbox(
            "LLM backend",
            options=list(backends),
            label_visibility="collapsed",
            key="backend_choice",
        )
        st.divider()

        backend = backends[backend_choice]
        st.session_state["base_url"] = None
        if backend == "ollama":
            _ollama_settings()
        elif backend == "claude":
            _claude_settings()
        else:
            _compat_settings(backend)

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


def _compat_settings(backend: str) -> None:
    """Gemini / Groq / OpenRouter / any OpenAI-compatible server."""
    p = OPENAI_COMPATIBLE_PROVIDERS[backend]
    st.markdown(f"**{p['label']}**")
    env_key = os.environ.get(p["key_env"], "")
    if p["key_url"]:
        st.caption(f"Free API key: [{p['key_url']}]({p['key_url']}) — or set `{p['key_env']}`.")
    key = st.text_input("API key", value=env_key, type="password", key=f"key_{backend}",
                        help="Used only for this session and never stored.")
    model = st.text_input("Model", value=p["model"], key=f"model_{backend}",
                          help="Provider model id. Free-tier model names change — check the provider's list.")
    if backend == "openai-compatible":
        st.session_state["base_url"] = st.text_input("Base URL", value=p["base_url"], key="compat_url",
                                                     help="e.g. LM Studio: http://localhost:1234/v1")
    if backend != "openai-compatible" and not key:
        st.error("API key not set.", icon="🔴")
    st.caption("Free tiers have rate limits and may use prompts to improve the provider's models.")
    st.session_state["backend"] = backend
    st.session_state["model"] = model
    st.session_state["api_key"] = key


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
