"""
LLM backend integration for CFD problem analysis.

Supported backends
------------------
- ``claude``  — Anthropic Claude API (cloud, requires ANTHROPIC_API_KEY)
- ``ollama``  — Ollama local inference server (free, no key required)

Public API
----------
    analyze_cfd_problem(user_description, backend, model, api_key) -> dict
    list_ollama_models(base_url) -> list[str]
    ollama_is_reachable(base_url) -> bool
"""

import json
import os
import time
from typing import Any

import anthropic
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Shared system prompt (used by both backends)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert CFD (Computational Fluid Dynamics) engineer specialising in OpenFOAM.
The user will describe a CFD problem in natural language.
Your task is to analyse the problem and return ONLY a valid JSON object — no markdown fences,
no preamble, no explanation outside the JSON.

The JSON must have exactly these keys:

{
  "solver": "<one of: simpleFoam | pimpleFoam | buoyantSimpleFoam | interFoam | rhoSimpleFoam>",
  "solver_justification": "<one sentence explaining the solver choice>",
  "fluid": {
    "name": "<fluid name, e.g. air, water>",
    "nu": <kinematic viscosity in m²/s, float>,
    "rho": <density in kg/m³, float>
  },
  "boundary_conditions": [
    {
      "patch_name": "<patch name>",
      "type": "<BC type, e.g. fixedValue, zeroGradient, noSlip, symmetry>",
      "values": {
        "<field>": "<value or [vx, vy, vz] for vectors>"
      }
    }
  ],
  "numerical_schemes": {
    "ddtSchemes": "<e.g. steadyState or Euler>",
    "gradSchemes": "<e.g. Gauss linear>",
    "divSchemes": "<e.g. Gauss linearUpwind grad(U)>",
    "laplacianSchemes": "<e.g. Gauss linear corrected>",
    "interpolationSchemes": "<e.g. linear>",
    "snGradSchemes": "<e.g. corrected>",
    "justification": "<one sentence>"
  },
  "Re": <Reynolds number as float, or null if not applicable>,
  "is_transient": <true or false>,
  "suggested_mesh": {
    "type": "<blockMesh or snappyHexMesh>",
    "approximate_cell_count": <integer>,
    "domain_description": "<brief description e.g. 20Dx10D channel>"
  },
  "characteristic_length": <characteristic length in metres, float, e.g. cylinder diameter>,
  "characteristic_velocity": <characteristic velocity in m/s, float>,
  "Ma": <Mach number as float, or null if not applicable>,
  "T_ref": <freestream/reference temperature in K, float — required for rhoSimpleFoam>,
  "p_ref": <reference/outlet pressure in Pa, float — required for rhoSimpleFoam>,
  "turb_model": "<one of: kEpsilon | kOmegaSST | SpalartAllmaras>",
  "geometry_type": "<one of: channel | cavity | t_junction | rectangular_obstacle | backstep | dambreak | cylinder | pipe | airfoil>",
  "geometry_params": {
    "channel_length":  <domain length in m, float, or null>,
    "channel_height":  <domain height in m, float, or null>,
    "branch_width":    <branch duct width for t_junction in m, float, or null>,
    "branch_height":   <branch duct length for t_junction in m, float, or null>,
    "obstacle_width":  <obstacle streamwise width for rectangular_obstacle in m, float, or null>,
    "obstacle_height": <obstacle height for rectangular_obstacle in m, float, or null>,
    "step_height":     <step height for backstep in m, float, or null>,
    "step_x":          <streamwise position of step for backstep in m, float, or null>,
    "cylinder_radius": <cylinder radius for cylinder geometry in m, float, or null>,
    "cx":              <cylinder centre x for cylinder geometry in m, float, or null>,
    "cy":              <cylinder centre y for cylinder geometry in m, float, or null>,
    "pipe_radius":     <pipe inner radius for pipe geometry in m, float, or null>,
    "pipe_length":     <pipe length for pipe geometry in m, float, or null>,
    "chord":           <airfoil chord length in m for airfoil geometry, float, or null>,
    "thickness":       <airfoil max thickness in m for airfoil geometry (default 12 % chord), float, or null>,
    "r_far":           <far-field radius in m for airfoil geometry (default 15 × chord), float, or null>,
    "wake_length":     <wake extension downstream of TE in m for airfoil geometry (default 10 × chord), float, or null>
  },
  "warnings": ["<warning string>", ...]
}

Rules:
- solver must be exactly one of the five listed values.
- Choose rhoSimpleFoam when the Mach number is above ~0.3 or when the problem is described
  as compressible, high-speed, transonic, or supersonic and steady-state.
- Choose simpleFoam for steady incompressible flows (Ma < 0.3, no buoyancy, single phase).
- Choose pimpleFoam for transient incompressible flows.
- Choose buoyantSimpleFoam ONLY when gravity and temperature differences are the PRIMARY
  driver of flow (natural convection / buoyancy-driven). If an inlet velocity or pressure
  difference forces the flow, use simpleFoam instead. buoyantSimpleFoam MUST always use
  geometry_type "cavity".
- Choose interFoam for two-phase (liquid-gas) free-surface flows.
- All numeric values must be JSON numbers (not strings).
- If a piece of information is not given, make a physically reasonable assumption and note it in warnings.
- boundary_conditions must include at least: inlet, outlet, and a wall or symmetry patch.
- For interFoam, add a patch for the atmosphere (open top) if relevant.
- For buoyantSimpleFoam, include temperature BCs (hotWall/coldWall T values) in boundary_conditions.
- For rhoSimpleFoam, always include T_ref and p_ref.
- turb_model must be exactly one of the three listed values. Choose based on physics:
  * kEpsilon — default for internal flows, heat transfer, jet mixing, t_junction, dam-break.
  * kOmegaSST — preferred for external aerodynamics, boundary layers, adverse pressure
    gradients, flow separation (rhoSimpleFoam, backstep, rectangular_obstacle).
  * SpalartAllmaras — one transport equation; good for attached external aero at moderate Re.
    Avoid with buoyantSimpleFoam (not suited to buoyancy) or interFoam (use kEpsilon there).
- For interFoam, turb_model is ignored (turbulenceProperties is always laminar).
- geometry_type must be exactly one of the nine listed values. Choose based on the problem:
  * channel — default for duct/channel flow and external aero without circular body (simpleFoam, pimpleFoam, rhoSimpleFoam)
  * cavity — enclosed square/rectangular cavity with heated/cooled walls; REQUIRED for buoyantSimpleFoam
  * t_junction — flow splitting at a T-intersection; two inlets or one inlet + one branch (NOT for buoyantSimpleFoam)
  * rectangular_obstacle — channel with a bluff body / rectangular block in the flow (NOT for buoyantSimpleFoam)
  * backstep — backward-facing step; sudden channel expansion (NOT for buoyantSimpleFoam)
  * dambreak — free-surface basin with open top (interFoam)
  * cylinder — 2-D circular cylinder in cross-flow using O-grid/butterfly mesh; use when the body is circular/cylindrical and the domain is external (NOT for buoyantSimpleFoam or interFoam)
  * pipe — axisymmetric internal pipe flow using a 5° wedge mesh; use when the geometry is a circular pipe or tube and the flow is axisymmetric (NOT for buoyantSimpleFoam or interFoam)
  * airfoil — 2-D airfoil (wing cross-section) in external flow using an 8-block C/H-grid; use when the body is a lifting surface, wing, blade, or NACA profile; kOmegaSST or SpalartAllmaras recommended (NOT for buoyantSimpleFoam or interFoam)
- geometry_params: provide non-null values only for parameters relevant to the chosen geometry.
  Set irrelevant parameters to null.  If a value is not stated, set to null and the generator
  will use a physically reasonable default.
- Respond with ONLY the JSON object. No extra text.
"""

_OLLAMA_BASE_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Parse JSON from a raw model response string.

    Strips markdown code fences if present before parsing.

    Raises
    ------
    json.JSONDecodeError
        If the cleaned string is not valid JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```) and closing fence
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# Claude backend
# ─────────────────────────────────────────────────────────────────────────────

def _claude_parse_with_retry(
    client: anthropic.Anthropic,
    user_description: str,
    model: str,
    max_attempts: int = 3,
) -> dict:
    """Call Claude and retry up to *max_attempts* times if JSON parsing fails."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_description}],
            )
            return _extract_json(message.content[0].text)

        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(1.0 * attempt)
        except anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

    raise ValueError(
        f"Failed to parse valid JSON from Claude after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


def _analyze_claude(
    user_description: str,
    model: str,
    api_key: str | None,
) -> dict:
    """Run analysis via the Anthropic Claude API."""
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "No Anthropic API key found. Enter your key in the sidebar "
            "or set the ANTHROPIC_API_KEY environment variable."
        )
    client = anthropic.Anthropic(api_key=resolved_key)
    return _claude_parse_with_retry(client, user_description, model)


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
    """POST to Ollama /api/generate and retry up to *max_attempts* times."""
    last_error: Exception | None = None
    endpoint = f"{base_url.rstrip('/')}/api/generate"

    # Build the full prompt with system instructions prepended
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"User problem:\n{user_description}"
    )

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return _extract_json(raw)

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {base_url}. "
                "Make sure Ollama is running: ollama serve"
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
    """
    Analyse a CFD problem using a locally running Ollama model.

    Parameters
    ----------
    user_description : str
        Free-text CFD problem description.
    model : str
        Ollama model tag, e.g. ``"mistral"`` or ``"llama3"``.
    base_url : str
        Base URL of the Ollama REST API (default ``http://localhost:11434``).
    timeout : int
        HTTP request timeout in seconds (default 300).

    Returns
    -------
    dict
        Parsed analysis JSON.

    Raises
    ------
    RuntimeError
        If Ollama is unreachable or returns an HTTP error.
    ValueError
        If the model response cannot be parsed as valid JSON after retries.
    """
    return _ollama_parse_with_retry(user_description, model, base_url, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def ollama_is_reachable(base_url: str = _OLLAMA_BASE_URL) -> bool:
    """Return True if the Ollama server responds within 2 seconds."""
    try:
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def list_ollama_models(base_url: str = _OLLAMA_BASE_URL) -> list[str]:
    """
    Return a list of locally available Ollama model names.

    Returns an empty list if Ollama is not reachable.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models if "name" in m]
    except requests.exceptions.RequestException:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Unified public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cfd_problem(
    user_description: str,
    backend: str = "claude",
    model: str = "claude-sonnet-4-6",
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
    model : str
        Model identifier.  For Claude: ``"claude-sonnet-4-6"``.
        For Ollama: e.g. ``"mistral"``, ``"llama3"``.
    api_key : str | None
        Anthropic API key (Claude backend only).  Falls back to the
        ``ANTHROPIC_API_KEY`` environment variable if not provided.
    ollama_timeout : int
        HTTP timeout in seconds for Ollama requests (default 300).

    Returns
    -------
    dict
        Structured analysis with keys: solver, fluid, boundary_conditions,
        numerical_schemes, Re, is_transient, suggested_mesh, warnings, etc.

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    RuntimeError
        On API / network errors.
    """
    if backend == "claude":
        return _analyze_claude(user_description, model, api_key)
    elif backend == "ollama":
        return analyze_cfd_problem_ollama(user_description, model, timeout=ollama_timeout)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'claude' or 'ollama'."
        )


def generate_mesh_script(analysis: dict) -> str:  # noqa: ARG001
    """Reserved for future use."""
    raise NotImplementedError("generate_mesh_script is reserved for a future release.")


# ─────────────────────────────────────────────────────────────────────────────
# Case explanation (flow-physics narrative)
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_SYSTEM = """\
You are an expert CFD engineer.  Given a structured JSON analysis of a CFD problem
you will write a concise, practical "What to expect" explanation for the user.
Write in plain English, using short paragraphs with **bold** headers.

Cover exactly these four sections (use the exact headers):

**Flow regime & physics**
Describe the Reynolds number regime, whether the flow is laminar/transitional/turbulent,
and the dominant physical mechanisms (inertia, viscosity, buoyancy, free-surface, etc.).

**Expected flow features**
Describe what the user should see in the results: boundary layers, separation zones,
recirculation, vortex shedding, shock waves, free-surface dynamics — whatever is relevant.

**Convergence tips**
List 2–3 practical tips: what residual to watch, typical iteration count, whether to
expect oscillatory convergence, relaxation-factor advice.

**Post-processing checklist**
List 2–3 quantities or plots the user should inspect (e.g. Cp distribution, velocity
profiles, streamlines, residual history).

Keep the total response under 300 words.  Do NOT include any JSON or code blocks.
"""


def _explain_claude(
    analysis: dict,
    user_description: str,
    model: str,
    api_key: str | None,
) -> str:
    """Call Claude to generate a plain-English case explanation."""
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError("No Anthropic API key — enter your key in the sidebar.")
    client = anthropic.Anthropic(api_key=resolved_key)
    import json as _json
    user_msg = (
        f"User description:\n{user_description}\n\n"
        f"Analysis JSON:\n{_json.dumps(analysis, indent=2)}"
    )
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text.strip()


def _explain_ollama(
    analysis: dict,
    user_description: str,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Call Ollama to generate a plain-English case explanation."""
    import json as _json
    prompt = (
        f"{_EXPLAIN_SYSTEM}\n\n"
        f"User description:\n{user_description}\n\n"
        f"Analysis JSON:\n{_json.dumps(analysis, indent=2)}"
    )
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        endpoint,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def explain_case(
    analysis: dict,
    user_description: str,
    backend: str = "claude",
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    ollama_base_url: str = _OLLAMA_BASE_URL,
    ollama_timeout: int = 120,
) -> str:
    """
    Generate a plain-English "What to expect" explanation for the CFD case.

    Parameters
    ----------
    analysis : dict
        Structured analysis dict (as returned by :func:`analyze_cfd_problem`).
    user_description : str
        Original free-text problem description from the user.
    backend : str
        ``"claude"`` or ``"ollama"``.
    model : str
        Model identifier.
    api_key : str | None
        Anthropic API key (Claude backend only).
    ollama_base_url : str
        Base URL for Ollama server.
    ollama_timeout : int
        HTTP timeout in seconds for Ollama requests (default 120).

    Returns
    -------
    str
        Markdown-formatted explanation (plain English, no JSON).

    Raises
    ------
    RuntimeError
        On API / network errors.
    """
    if backend == "claude":
        return _explain_claude(analysis, user_description, model, api_key)
    elif backend == "ollama":
        return _explain_ollama(
            analysis, user_description, model, ollama_base_url, ollama_timeout
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}.")
