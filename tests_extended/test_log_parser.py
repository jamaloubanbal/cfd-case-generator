"""
Tests for app/log_parser.py — OpenFOAM solver log parsing.

No API key required; all tests use synthetic log strings.
"""

import pytest
from app.log_parser import parse_foam_log, is_converged


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic log fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Minimal simpleFoam-style log (3 iterations, p + Ux + Uy + k + epsilon)
_SIMPLE_LOG = """\
Create time

Create mesh for time = 0

SIMPLE: convergence criteria
    field p   tolerance 1e-04
    field U   tolerance 1e-04

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 0.1, Final residual = 0.01, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.05, Final residual = 0.005, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.2, Final residual = 0.002, No Iterations 9
smoothSolver:  Solving for k, Initial residual = 0.08, Final residual = 0.008, No Iterations 1
smoothSolver:  Solving for epsilon, Initial residual = 0.09, Final residual = 0.009, No Iterations 1
ExecutionTime = 0.12 s  ClockTime = 0 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.05, Final residual = 0.005, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.02, Final residual = 0.002, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.08, Final residual = 0.0008, No Iterations 7
smoothSolver:  Solving for k, Initial residual = 0.03, Final residual = 0.003, No Iterations 1
smoothSolver:  Solving for epsilon, Initial residual = 0.04, Final residual = 0.004, No Iterations 1
ExecutionTime = 0.23 s  ClockTime = 0 s

Time = 3

smoothSolver:  Solving for Ux, Initial residual = 1e-05, Final residual = 1e-06, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 8e-06, Final residual = 8e-07, No Iterations 3
GAMG:  Solving for p, Initial residual = 2e-05, Final residual = 2e-06, No Iterations 5
smoothSolver:  Solving for k, Initial residual = 5e-06, Final residual = 5e-07, No Iterations 1
smoothSolver:  Solving for epsilon, Initial residual = 6e-06, Final residual = 6e-07, No Iterations 1
ExecutionTime = 0.34 s  ClockTime = 0 s
"""

# Log with no residual lines at all
_EMPTY_LOG = """\
Create time
Create mesh for time = 0
Selecting Newtonian transport model
"""

# Single-field log (only p)
_P_ONLY_LOG = """\
Time = 1
GAMG:  Solving for p, Initial residual = 0.5, Final residual = 0.01, No Iterations 10
ExecutionTime = 0.1 s  ClockTime = 0 s
Time = 2
GAMG:  Solving for p, Initial residual = 1e-05, Final residual = 1e-06, No Iterations 5
ExecutionTime = 0.2 s  ClockTime = 0 s
"""


# ─────────────────────────────────────────────────────────────────────────────
# parse_foam_log tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFoamLog:

    def test_returns_dict_with_required_keys(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert set(result.keys()) >= {"fields", "residuals", "times", "n_iters"}

    def test_fields_detected(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert "Ux" in result["fields"]
        assert "p" in result["fields"]
        assert "k" in result["fields"]
        assert "epsilon" in result["fields"]

    def test_fields_order_preserved(self):
        result = parse_foam_log(_SIMPLE_LOG)
        # Ux appears before p in the log
        assert result["fields"].index("Ux") < result["fields"].index("p")

    def test_residual_counts_match_iterations(self):
        result = parse_foam_log(_SIMPLE_LOG)
        # 3 iterations → 3 residual values per field
        assert len(result["residuals"]["Ux"]) == 3
        assert len(result["residuals"]["p"]) == 3

    def test_residual_values_correct(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert result["residuals"]["Ux"][0] == pytest.approx(0.1)
        assert result["residuals"]["Ux"][1] == pytest.approx(0.05)
        assert result["residuals"]["p"][0] == pytest.approx(0.2)

    def test_n_iters_correct(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert result["n_iters"] == 3

    def test_times_extracted(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert result["times"] == pytest.approx([1.0, 2.0, 3.0])

    def test_empty_log_returns_empty_fields(self):
        result = parse_foam_log(_EMPTY_LOG)
        assert result["fields"] == []
        assert result["residuals"] == {}
        assert result["n_iters"] == 0

    def test_single_field_log(self):
        result = parse_foam_log(_P_ONLY_LOG)
        assert result["fields"] == ["p"]
        assert len(result["residuals"]["p"]) == 2

    def test_scientific_notation_parsed(self):
        result = parse_foam_log(_SIMPLE_LOG)
        # Iteration 3 has 1e-05 for Ux
        assert result["residuals"]["Ux"][2] == pytest.approx(1e-5)

    def test_empty_string_returns_empty(self):
        result = parse_foam_log("")
        assert result["fields"] == []
        assert result["n_iters"] == 0

    def test_residuals_dict_keys_match_fields(self):
        result = parse_foam_log(_SIMPLE_LOG)
        assert set(result["fields"]) == set(result["residuals"].keys())


# ─────────────────────────────────────────────────────────────────────────────
# is_converged tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIsConverged:

    def test_converged_when_all_below_threshold(self):
        residuals = {"p": [0.1, 0.01, 1e-5], "Ux": [0.05, 0.005, 2e-5]}
        assert is_converged(residuals) is True

    def test_not_converged_when_one_above_threshold(self):
        residuals = {"p": [0.1, 0.01, 1e-5], "Ux": [0.05, 0.01, 5e-3]}
        assert is_converged(residuals) is False

    def test_not_converged_for_empty_residuals(self):
        assert is_converged({}) is False

    def test_converged_full_log(self):
        result = parse_foam_log(_SIMPLE_LOG)
        # Final residuals are all ~1e-5 or smaller
        assert is_converged(result["residuals"]) is True

    def test_not_converged_partial_log(self):
        # Only 2 iterations, residuals still large
        log = """\
Time = 1
smoothSolver:  Solving for Ux, Initial residual = 0.1, Final residual = 0.01, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.2, Final residual = 0.02, No Iterations 9
ExecutionTime = 0.1 s  ClockTime = 0 s
"""
        result = parse_foam_log(log)
        assert is_converged(result["residuals"]) is False

    def test_custom_threshold(self):
        residuals = {"p": [0.1, 0.001], "Ux": [0.1, 5e-4]}
        assert is_converged(residuals, threshold=1e-2) is True
        assert is_converged(residuals, threshold=1e-4) is False
