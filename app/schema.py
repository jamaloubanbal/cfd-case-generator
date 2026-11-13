"""
JSON schema for the LLM's CFD analysis.

The same schema is sent to both backends so the model is *constrained* to
produce a valid object instead of being asked nicely:

* Claude — ``output_config={"format": {"type": "json_schema", "schema": ...}}``
* Ollama — ``"format": <schema>`` (structured outputs, Ollama ≥ 0.5)

Structured outputs do not support numeric bounds (``minimum``/``maximum``), so
physical plausibility is enforced afterwards by :mod:`app.sanitize`.
"""

from __future__ import annotations

SOLVERS: tuple[str, ...] = (
    "simpleFoam", "pimpleFoam", "buoyantSimpleFoam", "interFoam", "rhoSimpleFoam",
)
TURB_MODELS: tuple[str, ...] = ("laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras")
GEOMETRIES: tuple[str, ...] = (
    "channel", "cavity", "t_junction", "rectangular_obstacle", "backstep",
    "dambreak", "cylinder", "pipe", "airfoil",
)

GEOMETRY_PARAM_KEYS: tuple[str, ...] = (
    "channel_length", "channel_height",
    "branch_width", "branch_height",
    "obstacle_width", "obstacle_height",
    "step_height", "step_x",
    "cylinder_radius", "cx", "cy",
    "pipe_radius", "pipe_length",
    "chord", "thickness", "r_far", "wake_length",
)

_NUM = {"type": "number"}
_NULLABLE_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_STR = {"type": "string"}


def _obj(properties: dict) -> dict:
    """Strict object: every property required, no extras (structured-output rule)."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ANALYSIS_SCHEMA: dict = _obj({
    "solver": {"type": "string", "enum": list(SOLVERS)},
    "solver_justification": _STR,
    "fluid": _obj({
        "name": _STR,
        "nu": _NUM,
        "rho": _NUM,
    }),
    "boundary_conditions": {
        "type": "array",
        "items": _obj({
            "patch_name": _STR,
            "type": _STR,
            # Only the fields the generators consume; null when not applicable.
            "values": _obj({
                "U": {"anyOf": [{"type": "array", "items": _NUM}, {"type": "null"}]},
                "p": _NULLABLE_NUM,
                "T": _NULLABLE_NUM,
                "alpha.water": _NULLABLE_NUM,
            }),
        }),
    },
    "numerical_schemes": _obj({
        "ddtSchemes": _STR,
        "gradSchemes": _STR,
        "divSchemes": _STR,
        "laplacianSchemes": _STR,
        "interpolationSchemes": _STR,
        "snGradSchemes": _STR,
        "justification": _STR,
    }),
    "Re": _NULLABLE_NUM,
    "is_transient": {"type": "boolean"},
    "suggested_mesh": _obj({
        "type": {"type": "string", "enum": ["blockMesh", "snappyHexMesh"]},
        "approximate_cell_count": {"type": "integer"},
        "domain_description": _STR,
    }),
    "characteristic_length": _NUM,
    "characteristic_velocity": _NUM,
    "Ma": _NULLABLE_NUM,
    "T_ref": _NULLABLE_NUM,
    "p_ref": _NULLABLE_NUM,
    "turb_model": {"type": "string", "enum": list(TURB_MODELS)},
    "geometry_type": {"type": "string", "enum": list(GEOMETRIES)},
    "geometry_params": _obj({k: _NULLABLE_NUM for k in GEOMETRY_PARAM_KEYS}),
    "warnings": {"type": "array", "items": _STR},
})
