"""Shared analysis fixtures (what the LLM returns) used by the tests in tests/ and tests_extended/."""

import struct as _struct

from app.generators import SimpleFoamGenerator


SIMPLE_FOAM_ANALYSIS = {
    "solver": "simpleFoam",
    "solver_justification": "Steady-state incompressible flow at Re=200.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kEpsilon",
    "geometry_type": "channel",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "inlet",  "type": "fixedValue",   "values": {"U": [1.0, 0.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
        {"patch_name": "walls",  "type": "noSlip",       "values": {}},
    ],
    "numerical_schemes": {
        "ddtSchemes": "steadyState",
        "gradSchemes": "Gauss linear",
        "divSchemes": "Gauss linearUpwind grad(U)",
        "laplacianSchemes": "Gauss linear corrected",
        "interpolationSchemes": "linear",
        "snGradSchemes": "corrected",
        "justification": "Standard steady incompressible schemes.",
    },
    "Re": 200.0,
    "is_transient": False,
    "suggested_mesh": {
        "type": "blockMesh",
        "approximate_cell_count": 20000,
        "domain_description": "20D x 10D channel",
    },
    "characteristic_length": 1.0,
    "characteristic_velocity": 1.0,
    "warnings": [],
}

PIMPLE_FOAM_ANALYSIS = {
    "solver": "pimpleFoam",
    "solver_justification": "Transient incompressible flow.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kEpsilon",
    "geometry_type": "channel",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "inlet",  "type": "fixedValue",   "values": {"U": [1.0, 0.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
        {"patch_name": "walls",  "type": "noSlip",       "values": {}},
    ],
    "numerical_schemes": {
        "ddtSchemes": "Euler",
        "gradSchemes": "Gauss linear",
        "divSchemes": "Gauss linearUpwind grad(U)",
        "laplacianSchemes": "Gauss linear corrected",
        "interpolationSchemes": "linear",
        "snGradSchemes": "corrected",
        "justification": "First-order Euler for transient.",
    },
    "Re": 1000.0,
    "is_transient": True,
    "suggested_mesh": {
        "type": "blockMesh",
        "approximate_cell_count": 50000,
        "domain_description": "30L x 15L channel",
    },
    "characteristic_length": 1.0,
    "characteristic_velocity": 1.0,
    "warnings": [],
}

BSF_ANALYSIS = {
    "solver": "buoyantSimpleFoam",
    "solver_justification": "Steady natural convection in a square cavity.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kEpsilon",
    "geometry_type": "cavity",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "hotWall",  "type": "fixedValue", "values": {"T": 350.0}},
        {"patch_name": "coldWall", "type": "fixedValue", "values": {"T": 300.0}},
        {"patch_name": "inlet",    "type": "fixedValue", "values": {"U": [0.0, 0.0, 0.0]}},
        {"patch_name": "outlet",   "type": "zeroGradient", "values": {}},
    ],
    "numerical_schemes": {
        "ddtSchemes": "steadyState",
        "gradSchemes": "Gauss linear",
        "divSchemes": "Gauss linearUpwind grad(U)",
        "laplacianSchemes": "Gauss linear corrected",
        "interpolationSchemes": "linear",
        "snGradSchemes": "corrected",
        "justification": "Standard buoyant SIMPLE schemes.",
    },
    "Re": None,
    "is_transient": False,
    "suggested_mesh": {
        "type": "blockMesh",
        "approximate_cell_count": 10000,
        "domain_description": "1m x 1m square cavity",
    },
    "characteristic_length": 1.0,
    "characteristic_velocity": 0.1,
    "warnings": [],
}

RHO_SIMPLE_FOAM_ANALYSIS = {
    "solver": "rhoSimpleFoam",
    "solver_justification": "Steady compressible RANS for transonic airfoil at Mach 0.8.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kOmegaSST",
    "geometry_type": "channel",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "inlet",  "type": "fixedValue",   "values": {"U": [272.0, 0.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
        {"patch_name": "walls",  "type": "noSlip",       "values": {}},
    ],
    "numerical_schemes": {
        "ddtSchemes": "steadyState",
        "gradSchemes": "Gauss linear",
        "divSchemes": "Gauss linearUpwind grad(U)",
        "laplacianSchemes": "Gauss linear corrected",
        "interpolationSchemes": "linear",
        "snGradSchemes": "corrected",
        "justification": "Standard compressible SIMPLE schemes.",
    },
    "Re": 1.8e7,
    "is_transient": False,
    "suggested_mesh": {
        "type": "blockMesh",
        "approximate_cell_count": 50000,
        "domain_description": "30c x 20c far-field",
    },
    "characteristic_length": 1.0,
    "characteristic_velocity": 272.0,
    "Ma": 0.8,
    "T_ref": 300.0,
    "p_ref": 101325.0,
    "warnings": [],
}

INTER_FOAM_ANALYSIS = {
    "solver": "interFoam",
    "solver_justification": "Two-phase gravity-driven dam-break flow.",
    "fluid": {"name": "water", "nu": 1.0e-6, "rho": 1000.0},
    "turb_model": "kEpsilon",
    "geometry_type": "dambreak",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "inlet",      "type": "fixedValue",   "values": {"U": [0.0, 0.0, 0.0]}},
        {"patch_name": "outlet",     "type": "zeroGradient", "values": {}},
        {"patch_name": "walls",      "type": "noSlip",       "values": {}},
        {"patch_name": "atmosphere", "type": "totalPressure","values": {}},
    ],
    "numerical_schemes": {
        "ddtSchemes": "Euler",
        "gradSchemes": "Gauss linear",
        "divSchemes": "Gauss linearUpwind grad(U)",
        "laplacianSchemes": "Gauss linear corrected",
        "interpolationSchemes": "linear",
        "snGradSchemes": "corrected",
        "justification": "Standard VOF schemes for interFoam.",
    },
    "Re": None,
    "is_transient": True,
    "suggested_mesh": {
        "type": "blockMesh",
        "approximate_cell_count": 40000,
        "domain_description": "1m x 0.6m basin",
    },
    "characteristic_length": 0.3,
    "characteristic_velocity": 1.72,
    "warnings": [],
}


# Geometry variant: T-junction (simpleFoam)
T_JUNCTION_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "t_junction",
    "geometry_params": {
        "channel_length": 2.0,
        "channel_height": 0.2,
        "branch_height":  0.6,
        "branch_width":   0.2,
    },
    "boundary_conditions": [
        {"patch_name": "inlet1", "type": "fixedValue",   "values": {"U": [1.0, 0.0, 0.0]}},
        {"patch_name": "inlet2", "type": "fixedValue",   "values": {"U": [0.0, -1.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
        {"patch_name": "walls",  "type": "noSlip",       "values": {}},
    ],
}

# Geometry variant: rectangular obstacle (simpleFoam)
RECT_OBS_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "rectangular_obstacle",
    "geometry_params": {
        "channel_length":  2.0,
        "channel_height":  0.5,
        "obstacle_width":  0.1,
        "obstacle_height": 0.1,
        "xo1":             0.6,
    },
}

# Geometry variant: backward-facing step (simpleFoam)
BACKSTEP_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "backstep",
    "geometry_params": {
        "channel_length": 2.0,
        "channel_height": 0.2,
        "step_height":    0.1,
        "step_x":         0.6,
    },
}


def _make_binary_stl(vertices: list[tuple[float, float, float]]) -> bytes:
    """Create a minimal binary STL with one triangle per 3 vertices."""
    n_tri = len(vertices) // 3
    data = b"\x00" * 80  # header
    data += _struct.pack("<I", n_tri)
    for i in range(n_tri):
        data += _struct.pack("<fff", 0.0, 0.0, 1.0)  # normal
        for v in vertices[i * 3 : i * 3 + 3]:
            data += _struct.pack("<fff", *v)
        data += _struct.pack("<H", 0)  # attribute
    return data


# Example STL: a rough 1 × 0.2 × 0.1 m box (12 triangles, 6 faces × 2)
_BOX_VERTS = [
    # Front face (z=0)
    (0, 0, 0), (1, 0, 0), (1, 0.2, 0),
    (0, 0, 0), (1, 0.2, 0), (0, 0.2, 0),
    # Back face (z=0.1)
    (0, 0, 0.1), (1, 0.2, 0.1), (1, 0, 0.1),
    (0, 0, 0.1), (0, 0.2, 0.1), (1, 0.2, 0.1),
    # Bottom (y=0)
    (0, 0, 0), (1, 0, 0.1), (1, 0, 0),
    (0, 0, 0), (0, 0, 0.1), (1, 0, 0.1),
    # Top (y=0.2)
    (0, 0.2, 0), (1, 0.2, 0), (1, 0.2, 0.1),
    (0, 0.2, 0), (1, 0.2, 0.1), (0, 0.2, 0.1),
    # Left (x=0)
    (0, 0, 0), (0, 0.2, 0.1), (0, 0, 0.1),
    (0, 0, 0), (0, 0.2, 0), (0, 0.2, 0.1),
    # Right (x=1)
    (1, 0, 0), (1, 0, 0.1), (1, 0.2, 0.1),
    (1, 0, 0), (1, 0.2, 0.1), (1, 0.2, 0),
]
_BOX_STL_BYTES = _make_binary_stl(_BOX_VERTS)
_BOX_BBOX = (0.0, 1.0, 0.0, 0.2, 0.0, 0.1)  # (xmin,xmax, ymin,ymax, zmin,zmax)

_STL_ANALYSIS_BASE = {
    "solver": "simpleFoam",
    "solver_justification": "External aero around STL body.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kOmegaSST",
    "geometry_type": "stl_import",
    "geometry_params": {},
    "boundary_conditions": [
        {"patch_name": "inlet",  "type": "fixedValue",   "values": {"U": [30.0, 0.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
    ],
    "numerical_schemes": {"ddtSchemes": "steadyState"},
    "Re": 2_000_000.0,
    "is_transient": False,
    "suggested_mesh": {},
    "characteristic_length": 1.0,
    "characteristic_velocity": 30.0,
    "_stl_bytes": _BOX_STL_BYTES,
    "_stl_name": "body",
    "_stl_bbox": _BOX_BBOX,
    "n_cores": 1,
}


def _generate_stl_case(tmp_path, analysis=None):
    """Generate a simpleFoam case with stl_import geometry."""
    if analysis is None:
        analysis = _STL_ANALYSIS_BASE
    gen = SimpleFoamGenerator()
    case_dir = tmp_path / "stl_case"
    gen.generate(analysis, case_dir)
    return case_dir


CYLINDER_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "cylinder",
    "turb_model": "kOmegaSST",
    "characteristic_length": 0.1,   # cylinder diameter = 0.1 m  → R = 0.05 m
    "characteristic_velocity": 5.0,
    "Re": 50000.0,
    "geometry_params": {
        "cylinder_radius": 0.05,
        "channel_length":  2.0,
        "channel_height":  1.0,
        "cx": 0.5,
        "cy": 0.5,
    },
}

PIPE_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "pipe",
    "turb_model": "kEpsilon",
    "characteristic_length": 0.1,   # pipe diameter = 0.1 m  → R = 0.05 m
    "characteristic_velocity": 1.0,
    "Re": 10000.0,
    "geometry_params": {
        "pipe_radius": 0.05,
        "pipe_length": 1.0,
    },
}


AIRFOIL_ANALYSIS = {
    **SIMPLE_FOAM_ANALYSIS,
    "geometry_type": "airfoil",
    "turb_model": "kOmegaSST",
    "characteristic_length": 1.0,      # chord = 1.0 m
    "characteristic_velocity": 30.0,
    "Re": 2_000_000.0,
    "geometry_params": {
        "chord":        1.0,
        "thickness":    0.12,           # 12 % chord → NACA-0012-like
        "r_far":        15.0,
        "wake_length":  10.0,
    },
}
