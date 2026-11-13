"""
Tests for app.editor — parameter-override pure functions.

No API key, no file I/O, no Streamlit import required.
"""

import math
import pytest

from app.editor import apply_overrides, defaults_from_analysis, _get_inlet_speed


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_analysis(**overrides) -> dict:
    base = {
        "solver": "simpleFoam",
        "turb_model": "kEpsilon",
        "geometry_type": "channel",
        "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
        "boundary_conditions": [
            {"patch_name": "inlet",  "type": "fixedValue",
             "values": {"U": [10.0, 0.0, 0.0]}},
            {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
        ],
        "Re": 6.667e5,
        "is_transient": False,
        "characteristic_length": 1.0,
        "suggested_mesh": {},
        "warnings": [],
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# _get_inlet_speed helper
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInletSpeed:

    def test_vector(self):
        bcs = [{"patch_name": "inlet", "values": {"U": [3.0, 4.0, 0.0]}}]
        assert _get_inlet_speed(bcs) == pytest.approx(5.0)

    def test_scalar(self):
        bcs = [{"patch_name": "inlet", "values": {"U": 7.0}}]
        assert _get_inlet_speed(bcs) == pytest.approx(7.0)

    def test_no_inlet_returns_one(self):
        assert _get_inlet_speed([]) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# defaults_from_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultsFromAnalysis:

    def test_keys_present(self):
        d = defaults_from_analysis(_base_analysis())
        assert {"inlet_speed", "nu", "turb_model", "nx", "ny",
                "domain_x", "domain_y"} <= d.keys()

    def test_speed_from_bcs(self):
        d = defaults_from_analysis(_base_analysis())
        assert d["inlet_speed"] == pytest.approx(10.0)

    def test_nu_from_fluid(self):
        d = defaults_from_analysis(_base_analysis())
        assert d["nu"] == pytest.approx(1.5e-5)

    def test_turb_model(self):
        d = defaults_from_analysis(_base_analysis())
        assert d["turb_model"] == "kEpsilon"

    def test_nx_ny_minimum(self):
        d = defaults_from_analysis(_base_analysis())
        assert d["nx"] >= 50
        assert d["ny"] >= 25

    def test_domain_x_is_20x_char_len(self):
        d = defaults_from_analysis(_base_analysis())
        # char_len = 1.0 → domain_x = 20.0
        assert d["domain_x"] == pytest.approx(20.0)

    def test_uses_suggested_mesh_when_present(self):
        a = _base_analysis()
        a["suggested_mesh"] = {"nx": 80, "ny": 40, "domain_x": 5.0, "domain_y": 2.0}
        d = defaults_from_analysis(a)
        assert d["nx"] == 80
        assert d["ny"] == 40
        assert d["domain_x"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — does not mutate original
# ─────────────────────────────────────────────────────────────────────────────

class TestNoMutation:

    def test_original_not_mutated(self):
        orig = _base_analysis()
        orig_nu = orig["fluid"]["nu"]
        _ = apply_overrides(orig, {"nu": 1e-6})
        assert orig["fluid"]["nu"] == pytest.approx(orig_nu)

    def test_original_bcs_not_mutated(self):
        orig = _base_analysis()
        orig_u = orig["boundary_conditions"][0]["values"]["U"][:]
        _ = apply_overrides(orig, {"inlet_speed": 99.0})
        assert orig["boundary_conditions"][0]["values"]["U"] == orig_u


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — inlet speed
# ─────────────────────────────────────────────────────────────────────────────

class TestInletSpeedOverride:

    def test_speed_magnitude_updated(self):
        a = apply_overrides(_base_analysis(), {"inlet_speed": 5.0})
        u = a["boundary_conditions"][0]["values"]["U"]
        assert math.sqrt(sum(v ** 2 for v in u)) == pytest.approx(5.0)

    def test_direction_preserved(self):
        # Original vector is [10, 0, 0] — direction stays (1, 0, 0)
        a = apply_overrides(_base_analysis(), {"inlet_speed": 3.0})
        u = a["boundary_conditions"][0]["values"]["U"]
        assert u[1] == pytest.approx(0.0)
        assert u[2] == pytest.approx(0.0)
        assert u[0] == pytest.approx(3.0)

    def test_diagonal_direction_preserved(self):
        analysis = _base_analysis()
        analysis["boundary_conditions"][0]["values"]["U"] = [3.0, 4.0, 0.0]
        a = apply_overrides(analysis, {"inlet_speed": 10.0})
        u = a["boundary_conditions"][0]["values"]["U"]
        # Direction (0.6, 0.8, 0) scaled to magnitude 10
        assert math.sqrt(sum(v ** 2 for v in u)) == pytest.approx(10.0)
        assert u[0] == pytest.approx(6.0)
        assert u[1] == pytest.approx(8.0)

    def test_no_inlet_bc_creates_one(self):
        analysis = _base_analysis()
        analysis["boundary_conditions"] = []  # no BCs at all
        a = apply_overrides(analysis, {"inlet_speed": 7.0})
        inlet_bcs = [bc for bc in a["boundary_conditions"]
                     if "inlet" in bc["patch_name"].lower()]
        assert len(inlet_bcs) == 1
        u = inlet_bcs[0]["values"]["U"]
        assert math.sqrt(sum(v ** 2 for v in u)) == pytest.approx(7.0)

    def test_zero_speed_ignored(self):
        # speed=0 should not be applied (guard against divide-by-zero)
        a = apply_overrides(_base_analysis(), {"inlet_speed": 0.0})
        u = a["boundary_conditions"][0]["values"]["U"]
        assert math.sqrt(sum(v ** 2 for v in u)) == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — viscosity and Re recomputation
# ─────────────────────────────────────────────────────────────────────────────

class TestViscosityAndReOverride:

    def test_nu_updated(self):
        a = apply_overrides(_base_analysis(), {"nu": 1e-6})
        assert a["fluid"]["nu"] == pytest.approx(1e-6)

    def test_re_recomputed(self):
        # U=10, char_len=1.0, nu=1e-6 → Re = 10/1e-6 = 1e7
        a = apply_overrides(_base_analysis(), {"nu": 1e-6})
        assert a["Re"] == pytest.approx(1e7, rel=1e-3)

    def test_re_recomputed_with_speed_and_nu(self):
        # U=5, char_len=1.0, nu=5e-5 → Re = 5/5e-5 = 1e5
        a = apply_overrides(
            _base_analysis(),
            {"inlet_speed": 5.0, "nu": 5e-5},
        )
        assert a["Re"] == pytest.approx(1e5, rel=1e-3)

    def test_zero_nu_ignored(self):
        orig_nu = _base_analysis()["fluid"]["nu"]
        a = apply_overrides(_base_analysis(), {"nu": 0.0})
        assert a["fluid"]["nu"] == pytest.approx(orig_nu)


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — turbulence model
# ─────────────────────────────────────────────────────────────────────────────

class TestTurbModelOverride:

    def test_turb_model_replaced(self):
        a = apply_overrides(_base_analysis(), {"turb_model": "kOmegaSST"})
        assert a["turb_model"] == "kOmegaSST"

    def test_sa_applied(self):
        a = apply_overrides(_base_analysis(), {"turb_model": "SpalartAllmaras"})
        assert a["turb_model"] == "SpalartAllmaras"

    def test_no_override_keeps_original(self):
        a = apply_overrides(_base_analysis(), {})
        assert a["turb_model"] == "kEpsilon"


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — mesh parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestMeshOverride:

    def test_nx_stored_in_suggested_mesh(self):
        a = apply_overrides(_base_analysis(), {"nx": 200})
        assert a["suggested_mesh"]["nx"] == 200

    def test_ny_stored_in_suggested_mesh(self):
        a = apply_overrides(_base_analysis(), {"ny": 75})
        assert a["suggested_mesh"]["ny"] == 75

    def test_domain_x_stored(self):
        a = apply_overrides(_base_analysis(), {"domain_x": 15.0})
        assert a["suggested_mesh"]["domain_x"] == pytest.approx(15.0)

    def test_domain_y_stored(self):
        a = apply_overrides(_base_analysis(), {"domain_y": 3.0})
        assert a["suggested_mesh"]["domain_y"] == pytest.approx(3.0)

    def test_existing_mesh_keys_preserved(self):
        analysis = _base_analysis()
        analysis["suggested_mesh"] = {"approximate_cell_count": 50000}
        a = apply_overrides(analysis, {"nx": 100})
        assert a["suggested_mesh"]["approximate_cell_count"] == 50000
        assert a["suggested_mesh"]["nx"] == 100

    def test_all_mesh_params_at_once(self):
        a = apply_overrides(
            _base_analysis(),
            {"nx": 120, "ny": 60, "domain_x": 8.0, "domain_y": 4.0},
        )
        m = a["suggested_mesh"]
        assert m["nx"] == 120
        assert m["ny"] == 60
        assert m["domain_x"] == pytest.approx(8.0)
        assert m["domain_y"] == pytest.approx(4.0)


# ─────────────────────────────────────────────────────────────────────────────
# apply_overrides — empty overrides passthrough
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyOverrides:

    def test_empty_overrides_preserves_solver(self):
        a = apply_overrides(_base_analysis(), {})
        assert a["solver"] == "simpleFoam"

    def test_empty_overrides_preserves_fluid(self):
        orig = _base_analysis()
        a = apply_overrides(orig, {})
        assert a["fluid"]["nu"] == pytest.approx(orig["fluid"]["nu"])

    def test_no_crash_on_minimal_analysis(self):
        apply_overrides({}, {})

    def test_unknown_keys_ignored(self):
        a = apply_overrides(_base_analysis(), {"unknown_param": 999})
        assert a["solver"] == "simpleFoam"
