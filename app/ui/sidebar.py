"""Sidebar: CFD code, AI model and credentials, each with a one-line status; geometry sketch.

Keys set in ``st.session_state``: ``cfd_code``, ``backend``, ``model``, ``api_key``,
``base_url``, ``ollama_timeout``, ``ai_ready`` and ``ai_label``.
"""

from __future__ import annotations

import os
import shutil

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

# provider id → (name, cost / privacy note, key environment variable, where to get a key)
_P = OPENAI_COMPATIBLE_PROVIDERS
PROVIDERS = {
    "claude": ("Claude (Anthropic)", "Paid API.", "ANTHROPIC_API_KEY", "https://console.anthropic.com"),
    "gemini": ("Google Gemini", "Free tier with rate limits; free-tier prompts may be used by Google.",
               _P["gemini"]["key_env"], _P["gemini"]["key_url"]),
    "ollama": ("Ollama (local)", "Runs on this computer: free and private, slower and less accurate.",
               None, None),
    "groq": ("Groq", "Free tier with rate limits.", _P["groq"]["key_env"], _P["groq"]["key_url"]),
    "openrouter": ("OpenRouter", "Free models end in “:free”; rate-limited.",
                   _P["openrouter"]["key_env"], _P["openrouter"]["key_url"]),
    "openai-compatible": ("Other OpenAI-compatible server", "e.g. LM Studio or vLLM.",
                          _P["openai-compatible"]["key_env"], None),
}


def _default_provider() -> int:
    """Preselect the first provider whose API key is already in the environment."""
    ids = list(PROVIDERS)
    for i, pid in enumerate(ids):
        env = PROVIDERS[pid][2]
        if env and os.environ.get(env) and pid != "openai-compatible":
            return i
    return 0


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🌊 CFD Case Generator")
        st.caption("Describe a flow in plain English; get a simulation case, run it and check the result.")
        _cfd_code()
        _ai_model()
        _geometry()
        _about()


# ── 1. CFD code ──────────────────────────────────────────────────────────────

def _cfd_code() -> None:
    from app.saturne import find_executable
    st.markdown("#### CFD code")
    cs = find_executable()
    options = ["OpenFOAM"] + (["code_saturne"] if cs else [])
    choice = st.radio("CFD code", options, key="cfd_code_choice", label_visibility="collapsed",
                      horizontal=True)
    st.session_state["cfd_code"] = "code_saturne" if choice == "code_saturne" else "openfoam"

    if choice == "OpenFOAM":
        if os.environ.get("WM_PROJECT_DIR") and shutil.which("blockMesh"):
            st.caption(f"✅ OpenFOAM {os.environ.get('WM_PROJECT_VERSION', '')} found: cases can run here.")
        else:
            st.caption("⚠️ OpenFOAM not found: cases can be generated and downloaded, not run here "
                       "(start the app from a terminal where OpenFOAM is sourced).")
    else:
        st.caption("✅ code_saturne found. Incompressible flow, steady or time-dependent, on a SALOME "
                   "mesh or the channel, T-junction, obstacle, step, cylinder or airfoil.")
    if not cs:
        st.caption("code_saturne is offered when installed (or set `CODE_SATURNE`).")


# ── 2. AI model ──────────────────────────────────────────────────────────────

def _ai_model() -> None:
    st.markdown("#### AI model")
    ids = list(PROVIDERS)
    pid = st.selectbox("Provider", ids, index=_default_provider(), key="provider_choice",
                       format_func=lambda p: PROVIDERS[p][0])
    name, note, key_env, key_url = PROVIDERS[pid]
    st.session_state.update(backend=pid, base_url=None)

    if pid == "ollama":
        ready, status = _ollama()
    else:
        model = _model_input(pid)
        env_key = os.environ.get(key_env, "") if key_env else ""
        key = st.text_input("API key", type="password", key=f"key_{pid}",
                            placeholder=f"from ${key_env}" if env_key else "paste your key",
                            help="Used for this session only, never stored. Leave empty to use "
                                 f"the {key_env} environment variable." if key_env else None)
        key = key or env_key
        if pid == "openai-compatible":
            st.session_state["base_url"] = st.text_input(
                "Server URL", value=_P[pid]["base_url"], key="compat_url",
                help="The server's OpenAI-compatible endpoint, e.g. LM Studio: http://localhost:1234/v1")
        st.session_state.update(model=model, api_key=key)
        if not model:
            ready, status = False, "🔴 Enter a model name."
        elif key or pid == "openai-compatible":
            src = " (key from the environment)" if key and key == env_key else ""
            ready, status = True, f"✅ Ready: `{model}`{src}"
        else:
            link = f" Get one at [{key_url.removeprefix('https://')}]({key_url})." if key_url else ""
            ready, status = False, f"🔴 No API key.{link}"

    st.session_state.update(ai_ready=ready, ai_label=f"{name} · {st.session_state.get('model') or '—'}")
    st.markdown(status)
    st.caption(note)


def _model_input(pid: str) -> str:
    if pid == "claude":
        return st.selectbox("Model", CLAUDE_MODELS, key="claude_model",
                            help="Default set by the CFD_CLAUDE_MODEL environment variable.")
    return st.text_input("Model", value=_P[pid]["model"], key=f"model_{pid}",
                         help="The provider's model id; free-tier model names change, check its list.").strip()


def _ollama() -> tuple[bool, str]:
    if not ollama_is_reachable():
        st.selectbox("Model", [DEFAULT_OLLAMA_MODEL], key="ollama_model", disabled=True)
        st.session_state.update(model=DEFAULT_OLLAMA_MODEL, api_key="", ollama_timeout=300)
        return False, "🔴 Ollama is not running: start it with `ollama serve`."
    available = list_ollama_models()
    preferred = [DEFAULT_OLLAMA_MODEL, "qwen2.5:7b", "mistral:latest", "mistral"]
    if not available:
        st.selectbox("Model", [DEFAULT_OLLAMA_MODEL], key="ollama_model", disabled=True)
        st.session_state.update(model=DEFAULT_OLLAMA_MODEL, api_key="", ollama_timeout=300)
        return False, f"🔴 No model installed: run `ollama pull {DEFAULT_OLLAMA_MODEL}`."
    idx = next((available.index(m) for m in preferred if m in available), 0)
    model = st.selectbox("Model", available, index=idx, key="ollama_model")
    with st.expander("Advanced"):
        timeout = st.slider("Answer timeout (s)", 60, 600, 300, 30, key="ollama_timeout_s",
                            help="Large local models can take minutes to answer.")
    st.session_state.update(model=model, api_key="", ollama_timeout=timeout)
    return True, f"✅ Ready: `{model}`"


# ── Geometry sketch and help ─────────────────────────────────────────────────

def _geometry() -> None:
    analysis = st.session_state.get("analysis")
    if not analysis:
        return
    geo = analysis.get("geometry_type", "channel")
    sketch, desc = get_sketch(geo)
    with st.expander(f"Geometry: {get_label(geo)}", expanded=True):
        st.caption(desc)
        st.code(sketch, language=None)


def _about() -> None:
    with st.expander("What it can simulate"):
        st.markdown(
            """
**OpenFOAM**
- `simpleFoam`: steady incompressible flow (laminar or RANS)
- `pimpleFoam`: time-dependent incompressible flow
- `buoyantSimpleFoam`: natural convection
- `rhoSimpleFoam`: steady compressible flow, up to about Mach 0.5
- `interFoam`: two-phase flow with a free surface

**code_saturne**: incompressible flow, steady or time-dependent.

**Geometries:** channel, pipe, cavity, cylinder, airfoil, backward-facing step,
obstacle, T-junction, dam break, or your own SALOME (.unv) or STL file.

**Steps:** describe the problem in **Setup**, run it in **Run** (live residuals, automatic
repair), read the verdict in **Results**.
            """
        )
