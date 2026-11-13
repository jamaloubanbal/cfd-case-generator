"""
LLM backend integration for CFD problem analysis.

Supported backends
------------------
- ``claude``  — Anthropic Claude API (cloud, requires ANTHROPIC_API_KEY)
- ``ollama``  — Ollama local inference server (free, no key required)

Both backends receive the JSON schema from :mod:`app.schema` as a structured-
output constraint, so the response is always a parseable object of the right
shape.  Physical plausibility is checked separately by :mod:`app.sanitize`.

Public API
----------
    analyze_cfd_problem(user_description, backend, model, api_key) -> dict
    explain_case(analysis, user_description, backend, model, api_key) -> str
    list_ollama_models(base_url) -> list[str]
    ollama_is_reachable(base_url) -> bool
"""

from __future__ import annotations

import json
import os
import time

import anthropic
import requests

from app.schema import ANALYSIS_SCHEMA

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

#: Claude model used when none is given; override with the CFD_CLAUDE_MODEL env var.
DEFAULT_CLAUDE_MODEL = os.environ.get("CFD_CLAUDE_MODEL", "claude-opus-5-5")

#: Models that accept server-side refusal fallbacks and the ``effort`` setting.
#: For these, a request declined by a safety classifier is transparently
#: re-run on a fallback model instead of failing.
_FALLBACK_MODELS = frozenset({
    "claude-fable-5-2", "claude-fable-5-1",
    "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5",
})
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_OLLAMA_BASE_URL = "http://localhost:11434"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis prompt (shared by both backends; the schema enforces the format)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert CFD engineer specialising in OpenFOAM (ESI v1912). The user describes a
CFD problem in plain language; analyse it and fill in the JSON object defined by the output
schema. Use SI units throughout.

Field meanings:
- fluid.nu: kinematic viscosity [m²/s] (air ≈ 1.5e-5, water ≈ 1e-6); fluid.rho: density [kg/m³].
- boundary_conditions: one entry per patch. values.U is the velocity vector [m/s] as
  [Ux, Uy, Uz]; values.T is temperature [K]; values.p pressure; values."alpha.water" the
  water volume fraction. Use null for fields that do not apply to the patch.
  Include at least an inlet, an outlet and a wall or symmetry patch. For a T-junction use
  patch names inlet1 (main inlet) and inlet2 (branch inlet, flow enters in -y).
- characteristic_length [m] (e.g. cylinder diameter, chord, cavity height) and
  characteristic_velocity [m/s]; Re and Ma based on them (null if not applicable).
- T_ref [K] and p_ref [Pa]: freestream temperature and outlet pressure (required for
  rhoSimpleFoam, null otherwise unless known).
- geometry_params: lengths in metres for the chosen geometry only; null for anything not
  stated or not relevant — the generator then uses a sensible default.
    channel_length/channel_height: domain size. branch_width/branch_height: T-junction branch.
    obstacle_width/obstacle_height: rectangular obstacle. step_height/step_x: backward step.
    cylinder_radius/cx/cy: cylinder radius and centre. pipe_radius/pipe_length: pipe.
    chord/thickness/r_far/wake_length: airfoil (defaults 12 % chord, 15 chords, 10 chords).

Solver choice:
- rhoSimpleFoam: steady compressible flow (Ma above ~0.3, or described as high-speed,
  transonic or supersonic).
- simpleFoam: steady incompressible single-phase flow.
- pimpleFoam: transient incompressible flow (e.g. vortex shedding).
- buoyantSimpleFoam: ONLY when buoyancy from temperature differences is the primary driver
  (natural convection). Forced flow with heat transfer uses simpleFoam. Always uses the
  "cavity" geometry; give hotWall/coldWall T values in boundary_conditions.
- interFoam: two-phase liquid/gas free-surface flow. Always uses the "dambreak" geometry;
  add an atmosphere patch for the open top.

Turbulence model: kEpsilon for internal flows, heat transfer, jets, T-junctions;
kOmegaSST for external aerodynamics, separation, adverse pressure gradients (backstep,
obstacles, rhoSimpleFoam); SpalartAllmaras for attached external aero at moderate Re.
Not SpalartAllmaras for buoyancy-driven flow. interFoam ignores the model (laminar).

Geometry: channel (duct or general external flow without a circular body), cavity,
t_junction, rectangular_obstacle (bluff body in a channel), backstep (sudden expansion),
dambreak, cylinder (2-D circular cylinder in cross-flow), pipe (axisymmetric internal pipe
flow), airfoil (wing section, blade, NACA profile).

When information is missing, make a physically reasonable assumption and record it in
warnings.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Parse JSON from a raw model response, stripping markdown fences if present.

    Raises
    ------
    json.JSONDecodeError
        If the cleaned string is not valid JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    return json.loads(text)


def _response_text(message) -> str:
    """Return the text of a Messages API response (skipping thinking blocks)."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    for block in message.content:              # e.g. test doubles without .type
        if isinstance(getattr(block, "text", None), str):
            return block.text
    raise RuntimeError("Claude returned no text content.")


def _check_stop_reason(message) -> None:
    """Turn refusals and truncation into clear errors instead of bad JSON."""
    reason = getattr(message, "stop_reason", None)
    if reason == "refusal":
        details = getattr(message, "stop_details", None)
        explanation = getattr(details, "explanation", None) or "no explanation given"
        raise RuntimeError(f"Claude declined this request ({explanation}).")
    if reason == "max_tokens":
        raise RuntimeError("Claude's response was cut off (max_tokens). Try a shorter description.")


def _claude_create(client: anthropic.Anthropic, model: str, *, effort: str, **params):
    """Call the Messages API, enabling refusal fallbacks where the model supports them."""
    output_config = dict(params.pop("output_config", {}) or {})
    if model in _FALLBACK_MODELS:
        output_config["effort"] = effort
        return client.beta.messages.create(
            model=model,
            betas=[_FALLBACK_BETA],
            fallbacks="default",
            output_config=output_config,
            **params,
        )
    if output_config:
        params["output_config"] = output_config
    return client.messages.create(model=model, **params)


def _claude_client(api_key: str | None) -> anthropic.Anthropic:
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "No Anthropic API key found. Enter your key in the sidebar "
            "or set the ANTHROPIC_API_KEY environment variable."
        )
    return anthropic.Anthropic(api_key=resolved_key)


# ─────────────────────────────────────────────────────────────────────────────
# Claude backend
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_claude(user_description: str, model: str, api_key: str | None) -> dict:
    """Run the analysis via the Claude API with a JSON-schema-constrained response."""
    client = _claude_client(api_key)
    try:
        message = _claude_create(
            client, model,
            effort="medium",
            max_tokens=16000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_description}],
            output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
        )
    except anthropic.APIError as exc:
        raise RuntimeError(f"Anthropic API error: {exc}") from exc

    _check_stop_reason(message)
    try:
        return _extract_json(_response_text(message))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_parse_with_retry(
    user_description: str,
    model: str,
    base_url: str,
    timeout: int = 300,
    max_attempts: int = 3,
) -> dict:
    """POST to Ollama /api/generate with the analysis schema as ``format``.

    Ollama ≥ 0.5 constrains generation to the schema.  Older servers ignore
    unknown schemas, so invalid JSON is still retried a few times.
    """
    last_error: Exception | None = None
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    prompt = f"{_SYSTEM_PROMPT}\n\nUser problem:\n{user_description}"

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": ANALYSIS_SCHEMA,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return _extract_json(resp.json().get("response", ""))

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {base_url}. Make sure Ollama is running: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {timeout} s (model: {model}). "
                "Increase the timeout in the sidebar or try a smaller/faster model."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(1.0 * attempt)

    raise ValueError(
        f"Failed to parse valid JSON from Ollama ({model}) after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


def analyze_cfd_problem_ollama(
    user_description: str,
    model: str = "mistral",
    base_url: str = _OLLAMA_BASE_URL,
    timeout: int = 300,
) -> dict:
    """Analyse a CFD problem using a locally running Ollama model."""
    return _ollama_parse_with_retry(user_description, model, base_url, timeout=timeout)


def ollama_is_reachable(base_url: str = _OLLAMA_BASE_URL) -> bool:
    """Return True if the Ollama server responds within 2 seconds."""
    try:
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def list_ollama_models(base_url: str = _OLLAMA_BASE_URL) -> list[str]:
    """Return locally available Ollama model names (empty if unreachable)."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if "name" in m]
    except requests.exceptions.RequestException:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Unified public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cfd_problem(
    user_description: str,
    backend: str = "claude",
    model: str | None = None,
    api_key: str | None = None,
    ollama_timeout: int = 300,
) -> dict:
    """
    Analyse a CFD problem using the chosen LLM backend.

    Parameters
    ----------
    user_description : str
        Free-text description of the CFD problem.
    backend : str
        ``"claude"`` (default) or ``"ollama"``.
    model : str | None
        Model identifier.  Defaults to :data:`DEFAULT_CLAUDE_MODEL` for Claude
        and ``"mistral"`` for Ollama.
    api_key : str | None
        Anthropic API key (Claude backend only); falls back to ANTHROPIC_API_KEY.
    ollama_timeout : int
        HTTP timeout in seconds for Ollama requests.

    Returns
    -------
    dict
        Raw analysis matching :data:`app.schema.ANALYSIS_SCHEMA`.  Pass it
        through :func:`app.sanitize.sanitize_analysis` before generating.

    Raises
    ------
    ValueError
        If *backend* is not recognised or the response is not valid JSON.
    RuntimeError
        On API / network errors, refusals or truncated responses.
    """
    if backend == "claude":
        return _analyze_claude(user_description, model or DEFAULT_CLAUDE_MODEL, api_key)
    if backend == "ollama":
        return analyze_cfd_problem_ollama(user_description, model or "mistral", timeout=ollama_timeout)
    raise ValueError(f"Unknown backend {backend!r}. Choose 'claude' or 'ollama'.")


# ─────────────────────────────────────────────────────────────────────────────
# Case explanation (flow-physics narrative)
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_SYSTEM = """\
You are an expert CFD engineer. Given a structured JSON analysis of a CFD problem, write a
concise, practical "What to expect" explanation for the user in plain English, using short
paragraphs with **bold** headers.

Cover exactly these four sections (use the exact headers):

**Flow regime & physics**
The Reynolds number regime, whether the flow is laminar/transitional/turbulent, and the
dominant physical mechanisms (inertia, viscosity, buoyancy, free surface, etc.).

**Expected flow features**
What the results should show: boundary layers, separation, recirculation, vortex shedding,
shocks, free-surface dynamics — whatever is relevant.

**Convergence tips**
2–3 practical tips: which residuals to watch, typical iteration count, whether oscillatory
convergence is expected, relaxation-factor advice.

**Post-processing checklist**
2–3 quantities or plots to inspect (e.g. Cp distribution, velocity profiles, streamlines,
residual history).

Keep the total under 300 words. Do not include JSON or code blocks.
"""


def _explain_user_message(analysis: dict, user_description: str) -> str:
    return (
        f"User description:\n{user_description}\n\n"
        f"Analysis JSON:\n{json.dumps(analysis, indent=2, default=str)}"
    )


def _explain_claude(analysis: dict, user_description: str, model: str, api_key: str | None) -> str:
    """Call Claude to generate a plain-English case explanation."""
    client = _claude_client(api_key)
    message = _claude_create(
        client, model,
        effort="low",
        # Thinking tokens count towards max_tokens on current models, so leave
        # headroom beyond the ~400 words of visible output.
        max_tokens=4096,
        system=_EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": _explain_user_message(analysis, user_description)}],
    )
    _check_stop_reason(message)
    return _response_text(message).strip()


def _explain_ollama(analysis: dict, user_description: str, model: str,
                    base_url: str, timeout: int) -> str:
    """Call Ollama to generate a plain-English case explanation."""
    prompt = f"{_EXPLAIN_SYSTEM}\n\n{_explain_user_message(analysis, user_description)}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def explain_case(
    analysis: dict,
    user_description: str,
    backend: str = "claude",
    model: str | None = None,
    api_key: str | None = None,
    ollama_base_url: str = _OLLAMA_BASE_URL,
    ollama_timeout: int = 120,
) -> str:
    """
    Generate a plain-English "What to expect" explanation for the CFD case.

    Returns
    -------
    str
        Markdown-formatted explanation (no JSON).

    Raises
    ------
    RuntimeError
        On API / network errors.
    """
    if backend == "claude":
        return _explain_claude(analysis, user_description, model or DEFAULT_CLAUDE_MODEL, api_key)
    if backend == "ollama":
        return _explain_ollama(analysis, user_description, model or "mistral",
                               ollama_base_url, ollama_timeout)
    raise ValueError(f"Unknown backend {backend!r}.")
