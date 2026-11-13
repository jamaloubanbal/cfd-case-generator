"""Tests for app.sanitize — repairing implausible LLM analyses."""

import copy

import pytest

from app.generators import GENERATORS
from app.sanitize import AnalysisError, sanitize_analysis
from tests.test_generators import (
    BSF_ANALYSIS,
    CYLINDER_ANALYSIS,
    INTER_FOAM_ANALYSIS,
    SIMPLE_FOAM_ANALYSIS,
)


def _with(base: dict, **changes) -> dict:
    a = copy.deepcopy(base)
    a.update(changes)
    return a


def test_valid_analysis_passes_without_notes():
    clean, notes = sanitize_analysis(SIMPLE_FOAM_ANALYSIS)
    assert notes == []
    assert clean["solver"] == "simpleFoam"
    assert clean["fluid"]["nu"] == pytest.approx(1.5e-5)


def test_input_is_not_mutated():
    a = _with(SIMPLE_FOAM_ANALYSIS, fluid={"name": "air", "nu": "bad", "rho": 1.2})
    before = copy.deepcopy(a)
    sanitize_analysis(a)
    assert a == before


@pytest.mark.parametrize("bad", [None, "simpleFOAM", "icoFoam", 42])
def test_unknown_solver_raises(bad):
    with pytest.raises(AnalysisError, match="Unsupported solver"):
        sanitize_analysis(_with(SIMPLE_FOAM_ANALYSIS, solver=bad))


def test_non_dict_raises():
    with pytest.raises(AnalysisError):
        sanitize_analysis(["not", "a", "dict"])


def test_viscosity_in_wrong_units_is_replaced():
    # e.g. dynamic viscosity in cP mistaken for kinematic viscosity
    clean, notes = sanitize_analysis(_with(SIMPLE_FOAM_ANALYSIS, fluid={"name": "water", "nu": 1.0, "rho": 1000}))
    assert clean["fluid"]["nu"] == pytest.approx(1e-6)
    assert any("viscosity" in n for n in notes)


def test_numeric_strings_are_accepted():
    a = _with(SIMPLE_FOAM_ANALYSIS, fluid={"name": "air", "nu": "1.5e-5", "rho": "1.2"},
              characteristic_length="0.5")
    clean, notes = sanitize_analysis(a)
    assert clean["fluid"]["nu"] == pytest.approx(1.5e-5)
    assert clean["characteristic_length"] == pytest.approx(0.5)
    assert notes == []


def test_buoyant_solver_forced_to_cavity():
    clean, notes = sanitize_analysis(_with(BSF_ANALYSIS, geometry_type="channel"))
    assert clean["geometry_type"] == "cavity"
    assert any("cavity" in n for n in notes)


def test_interfoam_forced_to_dambreak():
    clean, _ = sanitize_analysis(_with(INTER_FOAM_ANALYSIS, geometry_type="cylinder"))
    assert clean["geometry_type"] == "dambreak"


def test_cavity_not_allowed_for_forced_flow():
    clean, notes = sanitize_analysis(_with(SIMPLE_FOAM_ANALYSIS, geometry_type="cavity"))
    assert clean["geometry_type"] == "channel"
    assert notes


def test_unknown_turbulence_model_defaults():
    clean, notes = sanitize_analysis(_with(SIMPLE_FOAM_ANALYSIS, turb_model="LES"))
    assert clean["turb_model"] == "kEpsilon"
    assert notes


def test_velocity_vector_is_padded_and_capped():
    a = _with(SIMPLE_FOAM_ANALYSIS, boundary_conditions=[
        {"patch_name": "inlet", "type": "fixedValue", "values": {"U": [1e6]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {"U": None}},
    ])
    clean, notes = sanitize_analysis(a)
    u = clean["boundary_conditions"][0]["values"]["U"]
    assert len(u) == 3 and u[0] == pytest.approx(5000.0)
    assert "U" not in clean["boundary_conditions"][1]["values"]   # nulls removed
    assert any("capped" in n for n in notes)


def test_implausible_temperature_dropped():
    a = _with(BSF_ANALYSIS, boundary_conditions=[
        {"patch_name": "hotWall", "type": "fixedValue", "values": {"T": -40}},
        {"patch_name": "coldWall", "type": "fixedValue", "values": {"T": 300}},
    ])
    clean, notes = sanitize_analysis(a)
    assert "T" not in clean["boundary_conditions"][0]["values"]
    assert clean["boundary_conditions"][1]["values"]["T"] == 300


def test_reynolds_number_recomputed_when_missing():
    clean, _ = sanitize_analysis(_with(SIMPLE_FOAM_ANALYSIS, Re=None))
    assert clean["Re"] == pytest.approx(1.0 * 1.0 / 1.5e-5)


def test_cylinder_too_large_for_domain_is_shrunk():
    a = _with(CYLINDER_ANALYSIS, geometry_params={
        **CYLINDER_ANALYSIS["geometry_params"], "cylinder_radius": 2.0,
    })
    clean, notes = sanitize_analysis(a)
    assert clean["geometry_params"]["cylinder_radius"] == pytest.approx(0.2)
    assert any("cylinder_radius" in n for n in notes)


def test_negative_geometry_params_become_null():
    a = _with(SIMPLE_FOAM_ANALYSIS, geometry_params={"channel_length": -3, "channel_height": "abc"})
    clean, notes = sanitize_analysis(a)
    assert clean["geometry_params"]["channel_length"] is None
    assert clean["geometry_params"]["channel_height"] is None
    assert len(notes) == 2


def test_sanitized_garbage_still_generates(tmp_path):
    """A badly broken (but recognisable) analysis must still produce a case."""
    garbage = {
        "solver": "simpleFoam",
        "fluid": {"nu": -1, "rho": "heavy"},
        "boundary_conditions": [{"patch_name": "inlet", "values": {"U": "fast"}}, "junk"],
        "characteristic_length": 0,
        "turb_model": "magic",
        "geometry_type": "spaceship",
        "geometry_params": None,
        "warnings": None,
    }
    clean, notes = sanitize_analysis(garbage)
    assert len(notes) >= 5
    GENERATORS[clean["solver"]]().generate(clean, tmp_path)
    assert (tmp_path / "system" / "blockMeshDict").exists()
