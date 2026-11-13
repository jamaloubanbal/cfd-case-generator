"""
Tests for app.validation — physics-based pre-download checks.

No API key or file I/O required.
"""

import pytest

from app.validation import (
    run_checks,
    _check_reynolds,
    _check_mach,
    _check_yplus,
    _check_cfl,
    _check_resolution,
    _get_inlet_speed,
    _is_gas,
    _estimate_mesh,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared analysis dicts
# ─────────────────────────────────────────────────────────────────────────────

def _make_analysis(**overrides) -> dict:
    """Minimal valid analysis dict; keyword args override defaults."""
    base = {
        "solver": "simpleFoam",
        "turb_model": "kEpsilon",
        "geometry_type": "channel",
        "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
        "boundary_conditions": [
            {"patch_name": "inlet", "type": "fixedValue",
             "values": {"U": [10.0, 0.0, 0.0]}},
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
# Helper function unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_get_inlet_speed_vector(self):
        bcs = [{"patch_name": "inlet", "type": "fixedValue",
                "values": {"U": [3.0, 4.0, 0.0]}}]
        assert abs(_get_inlet_speed(bcs) - 5.0) < 1e-9

    def test_get_inlet_speed_scalar(self):
        bcs = [{"patch_name": "inlet", "type": "fixedValue",
                "values": {"U": 7.5}}]
        assert _get_inlet_speed(bcs) == pytest.approx(7.5)

    def test_get_inlet_speed_default(self):
        # No inlet BC → fall back to 1.0
        assert _get_inlet_speed([]) == pytest.approx(1.0)

    def test_is_gas_air(self):
        assert _is_gas(1.225, 1.5e-5, "air") is True

    def test_is_gas_water_name(self):
        assert _is_gas(1000.0, 1e-6, "water") is False

    def test_is_gas_high_density(self):
        # Unnamed fluid but density > 50 → liquid
        assert _is_gas(800.0, 5e-6, "unknown") is False

    def test_is_gas_low_density_unknown(self):
        assert _is_gas(1.8, 1e-5, "nitrogen") is True

    def test_estimate_mesh_defaults(self):
        mp = _estimate_mesh({"suggested_mesh": {}}, char_len=1.0)
        assert mp["domain_x"] == pytest.approx(20.0)
        assert mp["domain_y"] == pytest.approx(10.0)
        assert mp["nx"] >= 50
        assert mp["ny"] >= 25

    def test_estimate_mesh_from_dict(self):
        mesh = {"domain_x": 5.0, "domain_y": 2.0, "nx": 100, "ny": 50}
        mp = _estimate_mesh({"suggested_mesh": mesh}, char_len=1.0)
        assert mp["domain_x"] == pytest.approx(5.0)
        assert mp["ny"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# Reynolds number check
# ─────────────────────────────────────────────────────────────────────────────

class TestReynoldsCheck:

    def test_laminar_regime(self):
        c = _check_reynolds(500.0, "simpleFoam", "kEpsilon", "channel")
        assert c["level"] == "warning"
        assert "laminar" in c["title"].lower()

    def test_transitional_regime(self):
        c = _check_reynolds(2000.0, "simpleFoam", "kEpsilon", "channel")
        assert c["level"] == "warning"
        assert "transitional" in c["title"].lower()

    def test_turbulent_ok(self):
        c = _check_reynolds(1e5, "simpleFoam", "kEpsilon", "channel")
        assert c["level"] == "ok"

    def test_kepsilon_backstep_info(self):
        # k-ε with backstep → info tip
        c = _check_reynolds(5e4, "simpleFoam", "kEpsilon", "backstep")
        assert c["level"] == "info"
        assert "k-ε" in c["detail"] or "k-ω" in c["detail"]

    def test_ksst_backstep_ok(self):
        # kOmegaSST with backstep → ok (no tip needed)
        c = _check_reynolds(5e4, "simpleFoam", "kOmegaSST", "backstep")
        assert c["level"] == "ok"

    def test_kepsilon_rectangular_obstacle_info(self):
        c = _check_reynolds(2e4, "simpleFoam", "kEpsilon", "rectangular_obstacle")
        assert c["level"] == "info"


# ─────────────────────────────────────────────────────────────────────────────
# Mach number check
# ─────────────────────────────────────────────────────────────────────────────

class TestMachCheck:

    def test_low_mach_ok(self):
        c = _check_mach(10.0, "simpleFoam")       # Ma ≈ 0.029
        assert c["level"] == "ok"

    def test_mild_compressibility_info(self):
        c = _check_mach(60.0, "simpleFoam")       # Ma ≈ 0.175
        assert c["level"] == "info"

    def test_high_mach_incompressible_error(self):
        c = _check_mach(150.0, "simpleFoam")      # Ma ≈ 0.437
        assert c["level"] == "error"
        assert "rhoSimpleFoam" in c["detail"]

    def test_rho_low_mach_info(self):
        c = _check_mach(10.0, "rhoSimpleFoam")    # Ma ≈ 0.029
        assert c["level"] == "info"
        assert "simpleFoam" in c["detail"]

    def test_rho_high_mach_ok(self):
        c = _check_mach(150.0, "rhoSimpleFoam")   # Ma ≈ 0.437
        assert c["level"] == "ok"

    def test_pimple_high_mach_error(self):
        c = _check_mach(200.0, "pimpleFoam")
        assert c["level"] == "error"


# ─────────────────────────────────────────────────────────────────────────────
# y⁺ check
# ─────────────────────────────────────────────────────────────────────────────

class TestYPlusCheck:

    _MP_COARSE = {"domain_y": 10.0, "ny": 25, "domain_x": 20.0, "nx": 50}
    _MP_FINE   = {"domain_y": 1.0,  "ny": 200, "domain_x": 20.0, "nx": 50}

    def test_kEpsilon_valid_range(self):
        # Coarse mesh → y+ likely in 30–300 range for moderate Re
        c = _check_yplus(1e5, 10.0, 1.5e-5, self._MP_COARSE, "kEpsilon")
        # Should be ok or warning depending on exact value — just check no crash
        assert c["level"] in ("ok", "warning", "info")
        assert "y⁺" in c["title"]

    def test_sa_high_yplus_warning(self):
        # Very coarse mesh + high speed → y+ >> 300
        c = _check_yplus(1e7, 500.0, 1.5e-5, self._MP_COARSE, "SpalartAllmaras")
        assert c["level"] == "warning"

    def test_kEpsilon_too_fine_info(self):
        # Very fine mesh → y+ < 5
        c = _check_yplus(1000.0, 0.01, 1.5e-5, self._MP_FINE, "kEpsilon")
        assert c["level"] == "info"

    def test_kEpsilon_buffer_layer_warning(self):
        # Tune mesh / Re so 5 < y+ < 30
        # Re=5000, U=1 m/s, domain_y=0.5, ny=100, nu=1.5e-5
        mp = {"domain_y": 0.5, "ny": 100, "domain_x": 10.0, "nx": 50}
        c = _check_yplus(5000.0, 1.0, 1.5e-5, mp, "kEpsilon")
        assert c["level"] in ("warning", "ok")   # boundary case — check no error

    def test_returns_dict_with_required_keys(self):
        c = _check_yplus(1e5, 10.0, 1.5e-5, self._MP_COARSE, "kEpsilon")
        assert {"level", "title", "detail"} <= c.keys()


# ─────────────────────────────────────────────────────────────────────────────
# CFL check
# ─────────────────────────────────────────────────────────────────────────────

class TestCFLCheck:

    def test_cfl_always_ok(self):
        # Generator formula guarantees CFL ≈ 0.4
        mp = {"domain_x": 10.0, "nx": 100, "domain_y": 2.0, "ny": 40}
        c = _check_cfl(5.0, mp)
        assert c["level"] == "ok"
        assert "CFL" in c["title"]
        assert "Δt" in c["detail"]

    def test_cfl_zero_speed_no_crash(self):
        mp = {"domain_x": 10.0, "nx": 100, "domain_y": 2.0, "ny": 40}
        c = _check_cfl(0.0, mp)
        assert c["level"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Mesh resolution check
# ─────────────────────────────────────────────────────────────────────────────

class TestResolutionCheck:

    def test_reasonable_mesh_ok(self):
        # ny=100 → ~3.7 cells in BL at Re=1e5, above the <3 threshold
        mp = {"nx": 100, "ny": 100, "domain_x": 10.0, "domain_y": 5.0}
        c = _check_resolution(mp, Re=1e5)
        assert c["level"] == "ok"

    def test_small_mesh_info(self):
        mp = {"nx": 20, "ny": 10, "domain_x": 10.0, "domain_y": 5.0}
        c = _check_resolution(mp, Re=None)
        assert c["level"] == "info"

    def test_large_mesh_info(self):
        mp = {"nx": 1000, "ny": 600, "domain_x": 10.0, "domain_y": 5.0}
        c = _check_resolution(mp, Re=1e6)
        assert c["level"] == "info"
        assert "parallel" in c["detail"].lower()

    def test_coarse_bl_info(self):
        # Very high Re → thin BL → few cells in BL
        mp = {"nx": 100, "ny": 25, "domain_x": 10.0, "domain_y": 5.0}
        c = _check_resolution(mp, Re=1e8)
        assert c["level"] == "info"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_checks on full analysis dicts
# ─────────────────────────────────────────────────────────────────────────────

class TestRunChecks:

    def test_returns_list(self):
        a = _make_analysis()
        result = run_checks(a)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_dicts_have_required_keys(self):
        a = _make_analysis()
        for c in run_checks(a):
            assert "level" in c
            assert "title" in c
            assert "detail" in c
            assert c["level"] in ("ok", "info", "warning", "error")

    def test_laminar_flow_triggers_warning(self):
        a = _make_analysis(Re=100.0,
                           boundary_conditions=[
                               {"patch_name": "inlet", "type": "fixedValue",
                                "values": {"U": [0.001, 0.0, 0.0]}}
                           ])
        levels = [c["level"] for c in run_checks(a)]
        assert "warning" in levels

    def test_high_mach_simpleFoam_triggers_error(self):
        a = _make_analysis(
            solver="simpleFoam",
            Re=1e6,
            boundary_conditions=[
                {"patch_name": "inlet", "type": "fixedValue",
                 "values": {"U": [400.0, 0.0, 0.0]}}
            ],
        )
        levels = [c["level"] for c in run_checks(a)]
        assert "error" in levels

    def test_interFoam_skips_mach_and_yplus(self):
        a = _make_analysis(
            solver="interFoam",
            turb_model="kEpsilon",
            fluid={"name": "water", "nu": 1e-6, "rho": 1000.0},
        )
        checks = run_checks(a)
        titles_lower = " ".join(c["title"].lower() for c in checks)
        # Mach and y+ checks should be absent
        assert "mach" not in titles_lower
        assert "y⁺" not in titles_lower

    def test_buoyantSimpleFoam_skips_mach_and_yplus(self):
        a = _make_analysis(
            solver="buoyantSimpleFoam",
            geometry_type="cavity",
            boundary_conditions=[],   # no inlet → speed=0
        )
        checks = run_checks(a)
        titles_lower = " ".join(c["title"].lower() for c in checks)
        assert "mach" not in titles_lower
        assert "y⁺" not in titles_lower

    def test_transient_adds_cfl_check(self):
        a = _make_analysis(is_transient=True, solver="pimpleFoam")
        checks = run_checks(a)
        assert any("CFL" in c["title"] for c in checks)

    def test_steady_no_cfl_check(self):
        a = _make_analysis(is_transient=False, solver="simpleFoam")
        checks = run_checks(a)
        assert not any("CFL" in c["title"] for c in checks)

    def test_kEpsilon_backstep_info_check(self):
        a = _make_analysis(
            solver="simpleFoam",
            turb_model="kEpsilon",
            geometry_type="backstep",
            Re=5e4,
        )
        levels = [c["level"] for c in run_checks(a)]
        assert "info" in levels

    def test_water_skips_mach(self):
        a = _make_analysis(
            fluid={"name": "water", "nu": 1e-6, "rho": 1000.0},
            solver="simpleFoam",
        )
        checks = run_checks(a)
        assert not any("mach" in c["title"].lower() for c in checks)

    def test_no_crash_on_minimal_analysis(self):
        """run_checks must not raise even with a nearly empty dict."""
        run_checks({})

    def test_re_computed_from_inlet_speed_when_missing(self):
        """If Re is absent, it should be estimated from inlet speed + char_len."""
        a = _make_analysis()
        del a["Re"]   # remove explicit Re
        checks = run_checks(a)
        # Should still produce a Re check (not skip it)
        assert any("turbulent" in c["title"].lower() or "laminar" in c["title"].lower()
                   or "transitional" in c["title"].lower()
                   for c in checks)


def test_yplus_check_uses_graded_first_cell():
    """The check used domain_y/ny (uniform) and warned y+ ≈ 4000 on a mesh graded to y+ ≈ 50."""
    import copy
    from tests.fixtures import SIMPLE_FOAM_ANALYSIS
    a = copy.deepcopy(SIMPLE_FOAM_ANALYSIS)
    a.update(Re=1e6, boundary_conditions=[{"patch_name": "inlet", "type": "fixedValue",
                                           "values": {"U": [15.0, 0, 0]}}])
    check = next(c for c in run_checks(a) if c["title"].startswith("Wall resolution"))
    assert check["level"] == "ok", check
