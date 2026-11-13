"""
Tests for the CFD case generators.

Uses a hardcoded mock analysis dict so no Anthropic API key is required.
"""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.generators import (
    GENERATORS,
    SimpleFoamGenerator,
    PimpleFoamGenerator,
    BuoyantSimpleFoamGenerator,
    InterFoamGenerator,
    RhoSimpleFoamGenerator,
)
from app.generators.base import (
    _wall_y1,
    _compute_grading_ratio,
    _floor_grading_str,
    _ceil_grading_str,
    _both_grading_str,
    _sample_fields,
)
from app.utils import zip_directory, count_zip_files


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock analyses
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Generator registry
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorRegistry:
    def test_all_solvers_registered(self):
        expected = {"simpleFoam", "pimpleFoam", "buoyantSimpleFoam", "interFoam", "rhoSimpleFoam"}
        assert set(GENERATORS.keys()) == expected

    def test_generators_are_instantiable(self):
        for cls in GENERATORS.values():
            gen = cls()
            assert gen.solver_name != ""


# ─────────────────────────────────────────────────────────────────────────────
# File tree tests
# ─────────────────────────────────────────────────────────────────────────────

def _generated_files(generator_cls, analysis: dict, tmp_path: Path) -> set:
    """Run the generator and return a set of relative file paths."""
    case_dir = tmp_path / "case"
    gen = generator_cls()
    gen.generate(analysis, case_dir)
    return {str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file()}


class TestSimpleFoamFileTree:
    EXPECTED = {
        "0/U", "0/p", "0/k", "0/epsilon", "0/nut",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict",
        "constant/transportProperties", "constant/turbulenceProperties",
    }

    def test_all_files_created(self, tmp_path):
        files = _generated_files(SimpleFoamGenerator, SIMPLE_FOAM_ANALYSIS, tmp_path)
        assert self.EXPECTED.issubset(files), f"Missing: {self.EXPECTED - files}"

    def test_controldict_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "simpleFoam" in content

    def test_nu_in_transportproperties(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "transportProperties").read_text()
        assert "1.500000e-05" in content

    def test_foamfile_header_present(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        for fname in ("0/U", "system/controlDict", "constant/transportProperties"):
            text = (case_dir / fname).read_text()
            assert "FoamFile" in text, f"FoamFile header missing in {fname}"


class TestPimpleFoamFileTree:
    EXPECTED = {
        "0/U", "0/p", "0/k", "0/epsilon", "0/nut",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict",
        "constant/transportProperties", "constant/turbulenceProperties",
    }

    def test_all_files_created(self, tmp_path):
        files = _generated_files(PimpleFoamGenerator, PIMPLE_FOAM_ANALYSIS, tmp_path)
        assert self.EXPECTED.issubset(files), f"Missing: {self.EXPECTED - files}"

    def test_controldict_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "pimpleFoam" in content

    def test_controldict_has_deltat(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "deltaT" in content

    def test_controldict_has_endtime(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "endTime" in content

    def test_fvschemes_euler_time(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSchemes").read_text()
        assert "Euler" in content


class TestBuoyantSimpleFoamFileTree:
    EXPECTED = {
        "0/U", "0/p", "0/p_rgh", "0/T", "0/alphat", "0/k", "0/epsilon", "0/nut",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict",
        "constant/thermophysicalProperties", "constant/turbulenceProperties",
        "constant/radiationProperties", "constant/g",
    }

    def test_all_files_created(self, tmp_path):
        files = _generated_files(BuoyantSimpleFoamGenerator, BSF_ANALYSIS, tmp_path)
        assert self.EXPECTED.issubset(files), f"Missing: {self.EXPECTED - files}"

    def test_controldict_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "buoyantSimpleFoam" in content

    def test_temperature_bc_in_T(self, tmp_path):
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS, case_dir)
        content = (case_dir / "0" / "T").read_text()
        assert "350" in content  # T_hot
        assert "300" in content  # T_cold

    def test_radiation_properties_present(self, tmp_path):
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "radiationProperties").read_text()
        assert "none" in content

    def test_gravity_file_present(self, tmp_path):
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "g").read_text()
        assert "9.81" in content


class TestRhoSimpleFoamFileTree:
    # RHO_SIMPLE_FOAM_ANALYSIS uses kOmegaSST → omega instead of epsilon
    EXPECTED = {
        "0/U", "0/p", "0/T", "0/k", "0/omega", "0/nut", "0/alphat",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict",
        "constant/thermophysicalProperties", "constant/turbulenceProperties",
    }

    def test_all_files_created(self, tmp_path):
        files = _generated_files(RhoSimpleFoamGenerator, RHO_SIMPLE_FOAM_ANALYSIS, tmp_path)
        assert self.EXPECTED.issubset(files), f"Missing: {self.EXPECTED - files}"

    def test_controldict_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "rhoSimpleFoam" in content

    def test_p_outlet_in_p(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "0" / "p").read_text()
        assert "101325" in content

    def test_T_inlet_in_T(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "0" / "T").read_text()
        assert "300" in content

    def test_thermos_in_thermophysical(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "thermophysicalProperties").read_text()
        assert "heRhoThermo" in content
        assert "perfectGas" in content

    def test_fvsolution_has_rho_relaxation(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSolution").read_text()
        assert "rho" in content

    def test_fvschemes_has_enthalpy_div(self, tmp_path):
        case_dir = tmp_path / "case"
        RhoSimpleFoamGenerator().generate(RHO_SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSchemes").read_text()
        assert "div(phi,h)" in content
        assert "div(phid,p)" in content


class TestInterFoamFileTree:
    EXPECTED = {
        "0/U", "0/p_rgh", "0/alpha.water",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict",
        "constant/transportProperties", "constant/turbulenceProperties", "constant/g",
    }

    def test_all_files_created(self, tmp_path):
        files = _generated_files(InterFoamGenerator, INTER_FOAM_ANALYSIS, tmp_path)
        assert self.EXPECTED.issubset(files), f"Missing: {self.EXPECTED - files}"

    def test_controldict_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "controlDict").read_text()
        assert "interFoam" in content

    def test_alpha_water_field_present(self, tmp_path):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "0" / "alpha.water").read_text()
        assert "alpha.water" in content

    def test_transport_properties_two_phase(self, tmp_path):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "transportProperties").read_text()
        assert "water" in content
        assert "air" in content
        assert "sigma" in content


# ─────────────────────────────────────────────────────────────────────────────
# Zip tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZipGeneration:
    def test_zip_is_nonempty(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        zip_bytes = zip_directory(case_dir)
        assert len(zip_bytes) > 0

    def test_zip_contains_at_least_8_files(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        zip_bytes = zip_directory(case_dir)
        assert count_zip_files(zip_bytes) >= 8

    def test_zip_is_valid_archive(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        zip_bytes = zip_directory(case_dir)
        assert zipfile.is_zipfile(io.BytesIO(zip_bytes))

    def test_interfoam_zip_contains_at_least_8_files(self, tmp_path):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, case_dir)
        zip_bytes = zip_directory(case_dir)
        assert count_zip_files(zip_bytes) >= 8

    def test_buoyant_zip_contains_at_least_12_files(self, tmp_path):
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS, case_dir)
        zip_bytes = zip_directory(case_dir)
        assert count_zip_files(zip_bytes) >= 12


# ─────────────────────────────────────────────────────────────────────────────
# OpenFOAM syntax spot-checks
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenFOAMSyntax:
    """Verify that generated files contain valid OpenFOAM dictionary syntax."""

    # Non-OpenFOAM files that do not carry a FoamFile header
    _NON_FOAM_FILES = {"Allrun", "Allclean", "README.md"}

    @pytest.mark.parametrize("generator_cls,analysis", [
        (SimpleFoamGenerator,         SIMPLE_FOAM_ANALYSIS),
        (PimpleFoamGenerator,         PIMPLE_FOAM_ANALYSIS),
        (BuoyantSimpleFoamGenerator,  BSF_ANALYSIS),
        (InterFoamGenerator,          INTER_FOAM_ANALYSIS),
        (RhoSimpleFoamGenerator,      RHO_SIMPLE_FOAM_ANALYSIS),
    ])
    def test_all_files_have_foamfile_header(self, generator_cls, analysis, tmp_path):
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        for fpath in case_dir.rglob("*"):
            if fpath.is_file() and fpath.name not in self._NON_FOAM_FILES:
                text = fpath.read_text()
                assert "FoamFile" in text, (
                    f"Missing FoamFile header in {fpath.relative_to(case_dir)}"
                )

    @pytest.mark.parametrize("generator_cls,analysis", [
        (SimpleFoamGenerator,         SIMPLE_FOAM_ANALYSIS),
        (PimpleFoamGenerator,         PIMPLE_FOAM_ANALYSIS),
        (BuoyantSimpleFoamGenerator,  BSF_ANALYSIS),
        (InterFoamGenerator,          INTER_FOAM_ANALYSIS),
        (RhoSimpleFoamGenerator,      RHO_SIMPLE_FOAM_ANALYSIS),
    ])
    def test_no_jinja_placeholders_remain(self, generator_cls, analysis, tmp_path):
        """Ensure no un-rendered {{ }} tokens remain in generated files."""
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        for fpath in case_dir.rglob("*"):
            if fpath.is_file():
                text = fpath.read_text()
                assert "{{" not in text and "}}" not in text, (
                    f"Un-rendered Jinja2 placeholder in {fpath.relative_to(case_dir)}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# LLM routing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMRouting:
    """
    Verify that analyze_cfd_problem routes correctly to Claude or Ollama.
    No real API calls are made — both backends are mocked.
    """

    def test_claude_route_calls_anthropic(self):
        """Claude backend must invoke the Anthropic client, not requests."""
        from app.llm import analyze_cfd_problem

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(SIMPLE_FOAM_ANALYSIS))]

        with patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic_cls.return_value = mock_client

            result = analyze_cfd_problem(
                "steady airflow at Re=200",
                backend="claude",
                model="claude-sonnet-4-6",
                api_key="sk-ant-fake",
            )

        mock_client.messages.create.assert_called_once()
        assert result["solver"] == "simpleFoam"

    def test_ollama_route_calls_requests_post(self):
        """Ollama backend must POST to /api/generate, not call Anthropic."""
        from app.llm import analyze_cfd_problem

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps(SIMPLE_FOAM_ANALYSIS)
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.llm.requests.post", return_value=mock_response) as mock_post:
            result = analyze_cfd_problem(
                "steady airflow at Re=200",
                backend="ollama",
                model="mistral",
            )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/api/generate" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["model"] == "mistral"
        assert result["solver"] == "simpleFoam"

    def test_ollama_connection_error_raises_runtime_error(self):
        """A ConnectionError from requests must surface as a clear RuntimeError."""
        import requests as req
        from app.llm import analyze_cfd_problem

        with patch("app.llm.requests.post", side_effect=req.exceptions.ConnectionError):
            with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
                analyze_cfd_problem(
                    "some problem",
                    backend="ollama",
                    model="mistral",
                )

    def test_unknown_backend_raises_value_error(self):
        from app.llm import analyze_cfd_problem

        with pytest.raises(ValueError, match="Unknown backend"):
            analyze_cfd_problem("some problem", backend="gpt4")

    def test_ollama_markdown_fence_stripped(self):
        """Ollama response wrapped in ```json fences must still parse correctly."""
        from app.llm import analyze_cfd_problem

        fenced = f"```json\n{json.dumps(SIMPLE_FOAM_ANALYSIS)}\n```"
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": fenced}
        mock_response.raise_for_status = MagicMock()

        with patch("app.llm.requests.post", return_value=mock_response):
            result = analyze_cfd_problem(
                "steady airflow",
                backend="ollama",
                model="mistral",
            )

        assert result["solver"] == "simpleFoam"

    def test_ollama_is_reachable_returns_bool(self):
        """ollama_is_reachable must return True when server responds."""
        from app.llm import ollama_is_reachable

        mock_resp = MagicMock()
        with patch("app.llm.requests.get", return_value=mock_resp):
            assert ollama_is_reachable() is True

    def test_ollama_is_reachable_false_on_connection_error(self):
        import requests as req
        from app.llm import ollama_is_reachable

        with patch("app.llm.requests.get", side_effect=req.exceptions.ConnectionError):
            assert ollama_is_reachable() is False

    def test_list_ollama_models_parses_tags(self):
        """list_ollama_models must extract model names from /api/tags response."""
        from app.llm import list_ollama_models

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama3:8b"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("app.llm.requests.get", return_value=mock_resp):
            models = list_ollama_models()

        assert models == ["mistral:latest", "llama3:8b"]

    def test_list_ollama_models_empty_on_error(self):
        import requests as req
        from app.llm import list_ollama_models

        with patch("app.llm.requests.get", side_effect=req.exceptions.ConnectionError):
            assert list_ollama_models() == []


# ─────────────────────────────────────────────────────────────────────────────
# Geometry preset tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometryPresets:
    """Verify that geometry presets generate valid blockMeshDict files."""

    @pytest.mark.parametrize("analysis,expected_patch", [
        (T_JUNCTION_ANALYSIS,   "inlet1"),
        (T_JUNCTION_ANALYSIS,   "inlet2"),
        (RECT_OBS_ANALYSIS,     "obstacle"),
        (BACKSTEP_ANALYSIS,     "inlet"),
    ])
    def test_blockmeshdict_contains_patch(self, analysis, expected_patch, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(analysis, case_dir)
        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert expected_patch in content, (
            f"Patch '{expected_patch}' not found in blockMeshDict "
            f"for geometry_type={analysis['geometry_type']}"
        )

    @pytest.mark.parametrize("analysis", [
        T_JUNCTION_ANALYSIS,
        RECT_OBS_ANALYSIS,
        BACKSTEP_ANALYSIS,
    ])
    def test_no_jinja_placeholders_in_blockmesh(self, analysis, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(analysis, case_dir)
        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "{{" not in content and "}}" not in content

    def test_cavity_geometry_used_by_default_for_buoyant(self, tmp_path):
        """BuoyantSimpleFoam without geometry_type uses 'cavity' by default."""
        analysis_no_geo = {k: v for k, v in BSF_ANALYSIS.items() if k != "geometry_type"}
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(analysis_no_geo, case_dir)
        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "hotWall" in content or "coldWall" in content

    def test_dambreak_geometry_used_by_default_for_interfoam(self, tmp_path):
        """InterFoam without geometry_type uses 'dambreak' by default."""
        analysis_no_geo = {k: v for k, v in INTER_FOAM_ANALYSIS.items() if k != "geometry_type"}
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(analysis_no_geo, case_dir)
        content = (case_dir / "system" / "blockMeshDict").read_text()
        assert "atmosphere" in content


# ─────────────────────────────────────────────────────────────────────────────
# Turbulence model tests
# ─────────────────────────────────────────────────────────────────────────────

# kOmegaSST variant
SST_ANALYSIS = {**SIMPLE_FOAM_ANALYSIS, "turb_model": "kOmegaSST"}
# Spalart-Allmaras variant
SA_ANALYSIS = {**SIMPLE_FOAM_ANALYSIS, "turb_model": "SpalartAllmaras"}
# kOmegaSST for pimpleFoam
PIMPLE_SST_ANALYSIS = {**PIMPLE_FOAM_ANALYSIS, "turb_model": "kOmegaSST"}


class TestTurbulenceModels:
    """Verify that kOmegaSST and SpalartAllmaras generate correct field files."""

    def test_komegasst_creates_omega_not_epsilon(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SST_ANALYSIS, case_dir)
        files = {str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file()}
        assert "0/omega" in files
        assert "0/k" in files
        assert "0/epsilon" not in files

    def test_spalartallmaras_creates_nutilda_not_k_epsilon(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SA_ANALYSIS, case_dir)
        files = {str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file()}
        assert "0/nuTilda" in files
        assert "0/k" not in files
        assert "0/epsilon" not in files
        assert "0/omega" not in files

    def test_komegasst_turbulence_properties(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SST_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "turbulenceProperties").read_text()
        assert "kOmegaSST" in content

    def test_spalartallmaras_turbulence_properties(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SA_ANALYSIS, case_dir)
        content = (case_dir / "constant" / "turbulenceProperties").read_text()
        assert "SpalartAllmaras" in content

    def test_komegasst_fvschemes_has_omega_div(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SST_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSchemes").read_text()
        assert "div(phi,omega)" in content
        assert "div(phi,epsilon)" not in content

    def test_spalartallmaras_fvschemes_has_nutilda_div(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SA_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSchemes").read_text()
        assert "div(phi,nuTilda)" in content
        assert "div(phi,k)" not in content

    def test_komegasst_fvsolution_has_omega_solver(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SST_ANALYSIS, case_dir)
        content = (case_dir / "system" / "fvSolution").read_text()
        assert "omega" in content
        assert "epsilon" not in content

    def test_komegasst_nut_uses_nutkwallfunction(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SST_ANALYSIS, case_dir)
        content = (case_dir / "0" / "nut").read_text()
        assert "nutkWallFunction" in content

    def test_spalartallmaras_nut_uses_spalding(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SA_ANALYSIS, case_dir)
        content = (case_dir / "0" / "nut").read_text()
        assert "nutUSpaldingWallFunction" in content

    def test_default_turb_model_is_kepsilon_when_not_specified(self, tmp_path):
        """Analysis without turb_model key falls back to kEpsilon."""
        analysis_no_turb = {k: v for k, v in SIMPLE_FOAM_ANALYSIS.items() if k != "turb_model"}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(analysis_no_turb, case_dir)
        files = {str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file()}
        assert "0/epsilon" in files

    @pytest.mark.parametrize("analysis,expected_model", [
        (SST_ANALYSIS,   "kOmegaSST"),
        (SA_ANALYSIS,    "SpalartAllmaras"),
        (PIMPLE_SST_ANALYSIS, "kOmegaSST"),
    ])
    def test_no_jinja_placeholders_for_turb_variants(self, analysis, expected_model, tmp_path):
        gen = SimpleFoamGenerator() if "pimple" not in str(analysis.get("solver","")) else PimpleFoamGenerator()
        # Use the right generator
        if analysis.get("solver") == "pimpleFoam":
            gen = PimpleFoamGenerator()
        case_dir = tmp_path / "case"
        gen.generate(analysis, case_dir)
        for fpath in case_dir.rglob("*"):
            if fpath.is_file():
                text = fpath.read_text()
                assert "{{" not in text and "}}" not in text, (
                    f"Un-rendered placeholder in {fpath.relative_to(case_dir)}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# decomposeParDict + parallel Allrun tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeParDict:

    def test_decomposepardict_created_serial(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        assert (case_dir / "system" / "decomposeParDict").exists()

    def test_decomposepardict_has_foamfile_header(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "decomposeParDict").read_text()
        assert "FoamFile" in content

    def test_decomposepardict_default_n_cores_is_1(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "system" / "decomposeParDict").read_text()
        assert "numberOfSubdomains  1;" in content

    def test_decomposepardict_n_cores_respected(self, tmp_path):
        parallel_analysis = {**SIMPLE_FOAM_ANALYSIS, "n_cores": 8}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(parallel_analysis, case_dir)
        content = (case_dir / "system" / "decomposeParDict").read_text()
        assert "numberOfSubdomains  8;" in content

    def test_decomposepardict_hierarchical_method(self, tmp_path):
        """hierarchical is always available (scotch needs an optional library)."""
        parallel_analysis = {**SIMPLE_FOAM_ANALYSIS, "n_cores": 8}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(parallel_analysis, case_dir)
        content = (case_dir / "system" / "decomposeParDict").read_text()
        assert "method          hierarchical;" in content
        assert "n           (4 2 1);" in content

    @pytest.mark.parametrize("n, expected", [
        (1, (1, 1, 1)), (2, (2, 1, 1)), (4, (2, 2, 1)), (6, (3, 2, 1)),
        (7, (7, 1, 1)), (8, (4, 2, 1)), (16, (4, 4, 1)),
    ])
    def test_decomposition_split_product(self, n, expected):
        from app.generators.base import _decomposition_split
        split = _decomposition_split(n)
        assert split == expected
        assert split[0] * split[1] * split[2] == n

    def test_decomposepardict_no_jinja_placeholders(self, tmp_path):
        parallel_analysis = {**SIMPLE_FOAM_ANALYSIS, "n_cores": 4}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(parallel_analysis, case_dir)
        content = (case_dir / "system" / "decomposeParDict").read_text()
        assert "{{" not in content and "}}" not in content

    def test_allrun_serial_no_decompose(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "decomposePar" not in content
        assert "runApplication simpleFoam" in content

    def test_allrun_parallel_has_decompose(self, tmp_path):
        parallel_analysis = {**SIMPLE_FOAM_ANALYSIS, "n_cores": 4}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(parallel_analysis, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "decomposePar" in content
        assert "runParallel simpleFoam" in content
        assert "reconstructPar" in content

    def test_allrun_parallel_no_runApplication_solver(self, tmp_path):
        # When parallel, solver runs via runParallel not runApplication
        parallel_analysis = {**SIMPLE_FOAM_ANALYSIS, "n_cores": 4}
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(parallel_analysis, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "runApplication simpleFoam" not in content

    def test_n_cores_propagated_via_apply_overrides(self):
        from app.editor import apply_overrides
        result = apply_overrides(SIMPLE_FOAM_ANALYSIS, {"n_cores": 6})
        assert result["n_cores"] == 6

    def test_n_cores_default_is_1_in_defaults(self):
        from app.editor import defaults_from_analysis
        defaults = defaults_from_analysis(SIMPLE_FOAM_ANALYSIS)
        assert defaults["n_cores"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Allrun / Allclean / README tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScripts:
    """Verify that Allrun, Allclean, and README.md are generated correctly."""

    @pytest.mark.parametrize("generator_cls,analysis,expected_solver", [
        (SimpleFoamGenerator,        SIMPLE_FOAM_ANALYSIS,       "simpleFoam"),
        (PimpleFoamGenerator,        PIMPLE_FOAM_ANALYSIS,       "pimpleFoam"),
        (BuoyantSimpleFoamGenerator, BSF_ANALYSIS,               "buoyantSimpleFoam"),
        (InterFoamGenerator,         INTER_FOAM_ANALYSIS,        "interFoam"),
        (RhoSimpleFoamGenerator,     RHO_SIMPLE_FOAM_ANALYSIS,   "rhoSimpleFoam"),
    ])
    def test_allrun_created(self, generator_cls, analysis, expected_solver, tmp_path):
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        assert (case_dir / "Allrun").exists()

    @pytest.mark.parametrize("generator_cls,analysis,expected_solver", [
        (SimpleFoamGenerator,        SIMPLE_FOAM_ANALYSIS,       "simpleFoam"),
        (PimpleFoamGenerator,        PIMPLE_FOAM_ANALYSIS,       "pimpleFoam"),
        (BuoyantSimpleFoamGenerator, BSF_ANALYSIS,               "buoyantSimpleFoam"),
        (InterFoamGenerator,         INTER_FOAM_ANALYSIS,        "interFoam"),
        (RhoSimpleFoamGenerator,     RHO_SIMPLE_FOAM_ANALYSIS,   "rhoSimpleFoam"),
    ])
    def test_allrun_contains_solver_name(self, generator_cls, analysis, expected_solver, tmp_path):
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert expected_solver in content

    @pytest.mark.parametrize("generator_cls,analysis,expected_solver", [
        (SimpleFoamGenerator,        SIMPLE_FOAM_ANALYSIS,       "simpleFoam"),
        (PimpleFoamGenerator,        PIMPLE_FOAM_ANALYSIS,       "pimpleFoam"),
        (BuoyantSimpleFoamGenerator, BSF_ANALYSIS,               "buoyantSimpleFoam"),
        (InterFoamGenerator,         INTER_FOAM_ANALYSIS,        "interFoam"),
        (RhoSimpleFoamGenerator,     RHO_SIMPLE_FOAM_ANALYSIS,   "rhoSimpleFoam"),
    ])
    def test_allrun_has_shebang(self, generator_cls, analysis, expected_solver, tmp_path):
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert content.startswith("#!/bin/sh")

    @pytest.mark.parametrize("generator_cls,analysis,expected_solver", [
        (SimpleFoamGenerator,        SIMPLE_FOAM_ANALYSIS,       "simpleFoam"),
        (PimpleFoamGenerator,        PIMPLE_FOAM_ANALYSIS,       "pimpleFoam"),
        (BuoyantSimpleFoamGenerator, BSF_ANALYSIS,               "buoyantSimpleFoam"),
        (InterFoamGenerator,         INTER_FOAM_ANALYSIS,        "interFoam"),
        (RhoSimpleFoamGenerator,     RHO_SIMPLE_FOAM_ANALYSIS,   "rhoSimpleFoam"),
    ])
    def test_allrun_is_executable(self, generator_cls, analysis, expected_solver, tmp_path):
        import stat
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        mode = (case_dir / "Allrun").stat().st_mode
        assert mode & stat.S_IXUSR, "Allrun is not user-executable"

    def test_allclean_created(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        assert (case_dir / "Allclean").exists()

    def test_allclean_is_executable(self, tmp_path):
        import stat
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        mode = (case_dir / "Allclean").stat().st_mode
        assert mode & stat.S_IXUSR

    def test_readme_created(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        assert (case_dir / "README.md").exists()

    def test_readme_no_placeholders(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "{{" not in content and "}}" not in content

    def test_readme_contains_solver_name(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "simpleFoam" in content

    def test_readme_contains_domain_dims(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "domain_x" not in content  # template var should be rendered
        assert "20.0" in content or "20" in content  # char_len=1 → domain_x=20

    def test_readme_contains_run_instructions(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "./Allrun" in content
        assert "blockMesh" in content

    def test_readme_interfoam_has_setfields_note(self, tmp_path):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(INTER_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "setFields" in content

    def test_readme_transient_note_for_pimple(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "Transient" in content

    def test_readme_warnings_shown_when_present(self, tmp_path):
        analysis_with_warnings = {
            **SIMPLE_FOAM_ANALYSIS,
            "warnings": ["High Re: consider mesh refinement", "Check convergence"],
        }
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(analysis_with_warnings, case_dir)
        content = (case_dir / "README.md").read_text()
        assert "High Re: consider mesh refinement" in content

    def test_allrun_blockmesh_before_solver(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "Allrun").read_text()
        bm_pos = content.index("blockMesh")
        sf_pos = content.index("simpleFoam")
        assert bm_pos < sf_pos, "blockMesh must appear before simpleFoam in Allrun"


# ─────────────────────────────────────────────────────────────────────────────
# Wall-grading unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWallGradingHelpers:
    """Unit tests for the pure grading math functions in base.py."""

    def test_wall_y1_returns_positive_for_turbulent(self):
        y1 = _wall_y1(Re=1e6, inlet_speed=10.0, nu=1.5e-5, turb_model="kEpsilon")
        assert y1 > 0 and y1 < 0.1  # physically plausible

    def test_wall_y1_returns_inf_for_zero_speed(self):
        y1 = _wall_y1(Re=1e6, inlet_speed=0.0, nu=1.5e-5, turb_model="kEpsilon")
        import math
        assert math.isinf(y1)

    def test_wall_y1_returns_inf_for_none_re(self):
        import math
        y1 = _wall_y1(Re=None, inlet_speed=10.0, nu=1.5e-5, turb_model="kEpsilon")
        assert math.isinf(y1)

    def test_wall_y1_sa_smaller_than_kepsilon(self):
        # SpalartAllmaras targets y+ 30, kEpsilon targets y+ 50 — so SA gives smaller y1
        y1_sa = _wall_y1(Re=1e6, inlet_speed=10.0, nu=1.5e-5, turb_model="SpalartAllmaras")
        y1_ke = _wall_y1(Re=1e6, inlet_speed=10.0, nu=1.5e-5, turb_model="kEpsilon")
        assert y1_sa < y1_ke

    def test_compute_grading_ratio_returns_one_when_uniform_sufficient(self):
        # y1 >= uniform cell size → no grading needed → ratio = 1.0
        domain_h, n = 1.0, 10
        uniform = domain_h / n
        ratio = _compute_grading_ratio(domain_h, n, uniform * 1.5)
        assert ratio == pytest.approx(1.0)

    def test_compute_grading_ratio_greater_than_one_when_grading_needed(self):
        ratio = _compute_grading_ratio(domain_h=1.0, n_cells=50, y1=1e-4)
        assert ratio > 1.0

    def test_compute_grading_ratio_returns_one_for_zero_y1(self):
        ratio = _compute_grading_ratio(1.0, 50, 0.0)
        assert ratio == pytest.approx(1.0)

    def test_compute_grading_ratio_returns_one_for_zero_cells(self):
        ratio = _compute_grading_ratio(1.0, 0, 1e-4)
        assert ratio == pytest.approx(1.0)

    def test_floor_grading_str_returns_string(self):
        s = _floor_grading_str(1.0, 50, 1e-4)
        assert isinstance(s, str) and len(s) > 0

    def test_floor_grading_str_numeric_for_turbulent(self):
        s = _floor_grading_str(1.0, 50, 1e-4)
        # Should be a parseable float > 1
        assert float(s) > 1.0

    def test_ceil_grading_str_less_than_one_for_turbulent(self):
        s = _ceil_grading_str(1.0, 50, 1e-4)
        assert float(s) < 1.0

    def test_floor_and_ceil_are_reciprocals(self):
        floor_s = _floor_grading_str(1.0, 50, 1e-4)
        ceil_s  = _ceil_grading_str(1.0, 50, 1e-4)
        assert float(floor_s) * float(ceil_s) == pytest.approx(1.0, rel=1e-3)

    def test_both_grading_str_compound_format(self):
        s = _both_grading_str(1.0, 50, 1e-4)
        # Compound format: "((0.5 0.5 r)(0.5 0.5 inv_r))"
        assert s.startswith("((") and s.endswith("))")

    def test_uniform_grading_returns_one(self):
        # When y1 is large (inf), all helpers return "1"
        import math
        assert _floor_grading_str(1.0, 50, math.inf) == "1"
        assert _ceil_grading_str(1.0, 50, math.inf)  == "1"
        assert _both_grading_str(1.0, 50, math.inf)  == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: grading variables in blockMeshDict output
# ─────────────────────────────────────────────────────────────────────────────

def _blockmesh(generator, analysis, tmp_path) -> str:
    case_dir = tmp_path / "case"
    generator.generate(analysis, case_dir)
    return (case_dir / "system" / "blockMeshDict").read_text()


class TestGradingInBlockMeshDict:
    """Verify that grading variables are rendered (no raw {{ }}) and that
    turbulent cases use non-trivial simpleGrading values."""

    # ── channel ──────────────────────────────────────────────────────────────

    def test_channel_no_placeholders(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), SIMPLE_FOAM_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    def test_channel_has_simplegr_ading(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), SIMPLE_FOAM_ANALYSIS, tmp_path)
        assert "simpleGrading" in content

    def test_channel_turbulent_grading_not_all_ones(self, tmp_path):
        # High-Re case → grading should be applied (not "1 1 1")
        content = _blockmesh(SimpleFoamGenerator(), SIMPLE_FOAM_ANALYSIS, tmp_path)
        # The y-grading entry may be a float or compound "((…)(…))" — both contain digits
        import re
        # Non-greedy match captures everything between the two outer "1" entries
        match = re.search(r"simpleGrading\s*\(1\s+(.+?)\s+1\)", content)
        assert match is not None, "simpleGrading pattern not found"
        assert match.group(1) != "1", "Expected non-trivial y-grading for turbulent channel"

    # ── cavity ───────────────────────────────────────────────────────────────

    def test_cavity_no_placeholders(self, tmp_path):
        content = _blockmesh(BuoyantSimpleFoamGenerator(), BSF_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    # ── dambreak ─────────────────────────────────────────────────────────────

    def test_dambreak_no_placeholders(self, tmp_path):
        content = _blockmesh(InterFoamGenerator(), INTER_FOAM_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    # ── t_junction ───────────────────────────────────────────────────────────

    def test_t_junction_no_placeholders(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), T_JUNCTION_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    def test_t_junction_has_four_hex_blocks(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), T_JUNCTION_ANALYSIS, tmp_path)
        assert content.count("hex (") == 4

    # ── rectangular_obstacle ─────────────────────────────────────────────────

    def test_rectangular_obstacle_no_placeholders(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), RECT_OBS_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    def test_rectangular_obstacle_has_eight_hex_blocks(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), RECT_OBS_ANALYSIS, tmp_path)
        assert content.count("hex (") == 8

    # ── backstep ─────────────────────────────────────────────────────────────

    def test_backstep_no_placeholders(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), BACKSTEP_ANALYSIS, tmp_path)
        assert "{{" not in content and "}}" not in content

    def test_backstep_has_three_hex_blocks(self, tmp_path):
        content = _blockmesh(SimpleFoamGenerator(), BACKSTEP_ANALYSIS, tmp_path)
        assert content.count("hex (") == 3

    # ── laminar / low-Re → grading should be "1" ────────────────────────────

    def test_low_re_channel_grading_is_uniform(self, tmp_path):
        laminar = {**SIMPLE_FOAM_ANALYSIS, "Re": 500.0,
                   "boundary_conditions": [
                       {"patch_name": "inlet", "type": "fixedValue",
                        "values": {"U": [0.01, 0.0, 0.0]}},
                       {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
                   ]}
        content = _blockmesh(SimpleFoamGenerator(), laminar, tmp_path)
        import re
        match = re.search(r"simpleGrading\s*\(1\s+(.+?)\s+1\)", content)
        assert match is not None
        # For laminar, y1 >= uniform cell size → grading value should be "1"
        assert match.group(1) == "1", (
            f"Expected uniform grading for laminar case, got {match.group(1)!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# setFieldsDict tests (interFoam)
# ─────────────────────────────────────────────────────────────────────────────

class TestSetFieldsDict:
    """Verify system/setFieldsDict is generated correctly for interFoam."""

    def _generate(self, tmp_path, analysis=None):
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(analysis or INTER_FOAM_ANALYSIS, case_dir)
        return case_dir

    # ── File existence and structure ─────────────────────────────────────────

    def test_set_fields_dict_exists(self, tmp_path):
        case_dir = self._generate(tmp_path)
        assert (case_dir / "system" / "setFieldsDict").exists()

    def test_set_fields_dict_has_foamfile_header(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "FoamFile" in content

    def test_set_fields_dict_no_placeholders(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "{{" not in content and "}}" not in content

    def test_set_fields_dict_has_box_to_cell(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "boxToCell" in content

    def test_set_fields_dict_default_alpha_is_zero(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "volScalarFieldValue alpha.water 0" in content

    def test_set_fields_dict_region_alpha_is_one(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "volScalarFieldValue alpha.water 1" in content

    def test_set_fields_dict_object_name(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        assert "object      setFieldsDict;" in content

    # ── Water column dimensions ──────────────────────────────────────────────

    def test_x_water_defaults_to_30_percent_domain(self, tmp_path):
        case_dir = self._generate(tmp_path)
        ctx = InterFoamGenerator()._build_context(INTER_FOAM_ANALYSIS)
        assert pytest.approx(ctx["x_water"], rel=0.01) == ctx["domain_x"] * 0.3

    def test_y_water_defaults_to_50_percent_domain(self, tmp_path):
        ctx = InterFoamGenerator()._build_context(INTER_FOAM_ANALYSIS)
        assert pytest.approx(ctx["y_water"], rel=0.01) == ctx["domain_y"] * 0.5

    def test_x_water_appears_in_box_line(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        ctx = InterFoamGenerator()._build_context(INTER_FOAM_ANALYSIS)
        # The box line should contain the x_water value
        assert str(ctx["x_water"]) in content

    def test_y_water_appears_in_box_line(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "system" / "setFieldsDict").read_text()
        ctx = InterFoamGenerator()._build_context(INTER_FOAM_ANALYSIS)
        assert str(ctx["y_water"]) in content

    def test_x_water_overridable_via_geometry_params(self, tmp_path):
        analysis = {**INTER_FOAM_ANALYSIS, "geometry_params": {"x_water": 0.25}}
        ctx = InterFoamGenerator()._build_context(analysis)
        assert ctx["x_water"] == pytest.approx(0.25)

    def test_y_water_overridable_via_geometry_params(self, tmp_path):
        analysis = {**INTER_FOAM_ANALYSIS, "geometry_params": {"y_water": 0.18}}
        ctx = InterFoamGenerator()._build_context(analysis)
        assert ctx["y_water"] == pytest.approx(0.18)

    # ── Allrun integration ───────────────────────────────────────────────────

    def test_allrun_has_set_fields_for_interfoam(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "Allrun").read_text()
        assert "setFields" in content

    def test_allrun_set_fields_after_blockmesh(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "Allrun").read_text()
        bm_pos = content.index("blockMesh")
        sf_pos = content.index("setFields")
        assert bm_pos < sf_pos, "setFields must appear after blockMesh"

    def test_allrun_set_fields_before_solver(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "Allrun").read_text()
        sf_pos = content.index("setFields")
        if_pos = content.index("interFoam")
        assert sf_pos < if_pos, "setFields must appear before interFoam"

    def test_allrun_no_set_fields_for_simplefoam(self, tmp_path):
        case_dir = tmp_path / "case"
        SimpleFoamGenerator().generate(SIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "setFields" not in content

    def test_allrun_no_set_fields_for_pimplefoam(self, tmp_path):
        case_dir = tmp_path / "case"
        PimpleFoamGenerator().generate(PIMPLE_FOAM_ANALYSIS, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "setFields" not in content

    def test_allrun_interfoam_parallel_still_has_set_fields(self, tmp_path):
        parallel = {**INTER_FOAM_ANALYSIS, "n_cores": 4}
        case_dir = tmp_path / "case"
        InterFoamGenerator().generate(parallel, case_dir)
        content = (case_dir / "Allrun").read_text()
        assert "setFields" in content
        assert "decomposePar" in content
        # setFields must come before decomposePar (parallel decomposition)
        sf_pos = content.index("setFields")
        dp_pos = content.index("decomposePar")
        assert sf_pos < dp_pos, "setFields must appear before decomposePar"

    # ── README note ──────────────────────────────────────────────────────────

    def test_readme_interfoam_note_mentions_set_fields_automatic(self, tmp_path):
        case_dir = self._generate(tmp_path)
        content = (case_dir / "README.md").read_text()
        assert "setFields" in content
        # Should now indicate it's automatic (not that the user has to add it)
        assert "automatically" in content or "Allrun" in content


# =============================================================================
# TestBSFv2512Compatibility — verify all v2512 compressible BSF requirements
# =============================================================================

def _bsf_render(tmp_path, analysis=None):
    """Generate a BSF case and return (fvSchemes_text, fvSolution_text)."""
    a = analysis or BSF_ANALYSIS
    case_dir = tmp_path / "case"
    BuoyantSimpleFoamGenerator().generate(a, case_dir)
    fvs  = (case_dir / "system" / "fvSchemes").read_text()
    fvso = (case_dir / "system" / "fvSolution").read_text()
    return fvs, fvso


class TestBSFv2512Compatibility:
    """Generated BSF cases must contain all entries required by OpenFOAM v2512."""

    # -- fvSchemes: compressible stress divergence ----------------------------

    def test_fvschemes_has_rho_nueff_div_entry(self, tmp_path):
        fvs, _ = _bsf_render(tmp_path)
        assert "div(((rho*nuEff)*dev2(T(grad(U)))))" in fvs

    def test_fvschemes_still_has_incompressible_stress_entry(self, tmp_path):
        """Keep the v1912 form so the case also runs on older versions."""
        fvs, _ = _bsf_render(tmp_path)
        assert "div((nuEff*dev2(T(grad(U)))))" in fvs

    # -- fvSchemes: energy / kinetic energy entries ---------------------------

    def test_fvschemes_has_div_phi_h(self, tmp_path):
        fvs, _ = _bsf_render(tmp_path)
        assert "div(phi,h)" in fvs

    def test_fvschemes_has_div_phi_K(self, tmp_path):
        fvs, _ = _bsf_render(tmp_path)
        assert "div(phi,K)" in fvs

    def test_fvschemes_has_div_phi_Ekp(self, tmp_path):
        fvs, _ = _bsf_render(tmp_path)
        assert "div(phi,Ekp)" in fvs

    # -- fvSolution: h and e solver entries -----------------------------------

    def test_fvsolution_has_h_solver(self, tmp_path):
        _, fvso = _bsf_render(tmp_path)
        assert "    h\n    {" in fvso

    def test_fvsolution_has_e_solver(self, tmp_path):
        _, fvso = _bsf_render(tmp_path)
        assert "    e\n    {" in fvso

    # -- fvSolution: residualControl includes h --------------------------------

    def test_fvsolution_residualcontrol_has_h(self, tmp_path):
        _, fvso = _bsf_render(tmp_path)
        assert "h               1e-5" in fvso

    # -- fvSolution: relaxationFactors includes h ------------------------------

    def test_fvsolution_relaxation_has_h(self, tmp_path):
        _, fvso = _bsf_render(tmp_path)
        assert "h               0.5" in fvso

    # -- mu computation: must use ideal-gas rho, not fluid.rho ----------------

    def test_mu_is_physically_reasonable_for_air(self, tmp_path):
        """mu for air must be in [1e-6, 1e-3] Pa*s even if LLM gives wrong rho."""
        bad_rho_analysis = {
            **BSF_ANALYSIS,
            "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1086.0},
        }
        case_dir = tmp_path / "case"
        BuoyantSimpleFoamGenerator().generate(bad_rho_analysis, case_dir)
        thermo = (case_dir / "constant" / "thermophysicalProperties").read_text()
        import re
        m = re.search(r"mu\s+([\d.e+\-]+)", thermo)
        assert m is not None, "mu not found in thermophysicalProperties"
        mu_val = float(m.group(1))
        assert 1e-6 < mu_val < 1e-3, (
            f"mu={mu_val:.3e} Pa*s is outside physically reasonable range for air"
        )

    def test_mu_is_lower_at_higher_temperature(self, tmp_path):
        """At higher T, rho_ideal is lower so mu = nu*rho is smaller."""
        import re
        hot_analysis = {
            **BSF_ANALYSIS,
            "boundary_conditions": [
                {"patch_name": "hotWall",  "type": "fixedValue", "values": {"T": 600.0}},
                {"patch_name": "coldWall", "type": "fixedValue", "values": {"T": 400.0}},
            ],
        }
        d1 = tmp_path / "hot"
        d2 = tmp_path / "cold"
        BuoyantSimpleFoamGenerator().generate(hot_analysis,  d1)
        BuoyantSimpleFoamGenerator().generate(BSF_ANALYSIS,  d2)
        mu_hot  = float(re.search(r"mu\s+([\d.e+\-]+)",
                                  (d1 / "constant" / "thermophysicalProperties").read_text()).group(1))
        mu_cold = float(re.search(r"mu\s+([\d.e+\-]+)",
                                  (d2 / "constant" / "thermophysicalProperties").read_text()).group(1))
        assert mu_hot < mu_cold, (
            f"Expected mu(hot={mu_hot:.3e}) < mu(cold={mu_cold:.3e}) "
            "for higher temperature (lower ideal-gas density)"
        )

    # -- all turbulence models keep the new entries ---------------------------

    @pytest.mark.parametrize("turb", ["kEpsilon", "kOmegaSST", "SpalartAllmaras"])
    def test_v2512_entries_present_for_all_turb_models(self, tmp_path, turb):
        analysis = {**BSF_ANALYSIS, "turb_model": turb}
        fvs, fvso = _bsf_render(tmp_path, analysis)
        assert "div(phi,K)" in fvs
        assert "div(phi,h)" in fvs
        assert "div(((rho*nuEff)*dev2(T(grad(U)))))" in fvs
        assert "h               1e-5" in fvso


# =============================================================================
# snappyHexMesh / STL import tests
# =============================================================================

import struct as _struct


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


class TestSTLParser:
    """Tests for app.stl_parser."""

    def test_binary_stl_bbox(self):
        from app.stl_parser import stl_bounding_box
        bbox = stl_bounding_box(_BOX_STL_BYTES)
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        # float32 precision ≈ 1e-7; use 1e-6 tolerance
        assert abs(xmin) < 1e-6
        assert abs(xmax - 1.0) < 1e-6
        assert abs(ymin) < 1e-6
        assert abs(ymax - 0.2) < 1e-6
        assert abs(zmin) < 1e-6
        assert abs(zmax - 0.1) < 1e-6

    def test_ascii_stl_bbox(self):
        from app.stl_parser import stl_bounding_box
        ascii_stl = b"""solid test
  facet normal 0 0 1
    outer loop
      vertex 0.5 1.0 2.0
      vertex 3.0 0.0 -1.0
      vertex -1.0 2.5 4.0
    endloop
  endfacet
endsolid test
"""
        bbox = stl_bounding_box(ascii_stl)
        assert bbox[0] == -1.0   # xmin
        assert bbox[1] == 3.0    # xmax
        assert bbox[2] == 0.0    # ymin
        assert bbox[3] == 2.5    # ymax
        assert bbox[4] == -1.0   # zmin
        assert bbox[5] == 4.0    # zmax

    def test_empty_stl_raises(self):
        from app.stl_parser import stl_bounding_box
        with pytest.raises(ValueError):
            stl_bounding_box(b"solid empty\nendsolid empty\n")

    def test_truncated_binary_falls_back_to_ascii(self):
        """A short data blob that looks binary but is truncated should try ASCII."""
        from app.stl_parser import stl_bounding_box
        # build data that passes binary length check but has valid ASCII inside
        ascii_part = b"solid test\n  facet normal 0 0 1\n    outer loop\n      vertex 1 2 3\n      vertex 4 5 6\n      vertex 7 8 9\n    endloop\n  endfacet\nendsolid test\n"
        with pytest.raises(ValueError):
            # No vertices in this nonsense
            stl_bounding_box(b"\x00" * 84)


class TestSnappyHexMeshGeneration:
    """snappyHexMesh files are generated when geometry_type == 'stl_import'."""

    def test_snappy_dict_files_created(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        assert (case / "system" / "snappyHexMeshDict").exists()
        assert (case / "system" / "surfaceFeatureExtractDict").exists()
        assert (case / "system" / "meshQualityDict").exists()

    def test_stl_written_to_triSurface(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        stl_path = case / "constant" / "triSurface" / "body.stl"
        assert stl_path.exists()
        assert stl_path.read_bytes() == _BOX_STL_BYTES

    def test_snappy_dict_references_stl_name(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "snappyHexMeshDict").read_text()
        assert "body.stl" in text
        assert "name body" in text

    def test_location_in_mesh_near_inlet(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "snappyHexMeshDict").read_text()
        import re
        m = re.search(r"locationInMesh\s*\(\s*([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s*\)", text)
        assert m is not None, "locationInMesh not found"
        loc_x = float(m.group(1))
        # background mesh x: from 0.0 - 2*1.0 = -2.0 to 1.0 + 5*1.0 = 6.0
        # loc_x = -2.0 + 0.05 * 8.0 = -1.6
        assert loc_x < 0.0  # near inlet (left of STL)

    def test_surface_feature_extract_references_stl(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "surfaceFeatureExtractDict").read_text()
        assert "body.stl" in text
        assert "body.eMesh" in text

    def test_blockMeshDict_has_stl_import_patches(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "inlet" in text
        assert "outlet" in text
        assert "symmetry" in text

    def test_allrun_has_snappy_step(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        allrun = (case / "Allrun").read_text()
        assert "surfaceFeatureExtract" in allrun
        assert "snappyHexMesh" in allrun

    def test_field_U_has_stl_wall_patch(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        u_text = (case / "0" / "U").read_text()
        # STL wall patch "body" should appear as noSlip
        assert "body" in u_text
        assert "noSlip" in u_text

    def test_field_p_has_stl_wall_patch(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        p_text = (case / "0" / "p").read_text()
        assert "body" in p_text

    def test_background_mesh_bounds_computed_from_stl(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        # STL x: [0,1], upstream 2× → bg_xmin = 0 - 2*1 = -2.0
        assert "-2.0" in text
        # STL x: [0,1], downstream 5× → bg_xmax = 1 + 5*1 = 6.0
        assert "6.0" in text

    def test_no_stl_bytes_no_triSurface(self, tmp_path):
        """When _stl_bytes is None, constant/triSurface is not created."""
        analysis = {**_STL_ANALYSIS_BASE, "_stl_bytes": None}
        case = _generate_stl_case(tmp_path, analysis)
        # snappy dict files still generated
        assert (case / "system" / "snappyHexMeshDict").exists()
        # but triSurface dir is absent (or empty)
        tri_stl = case / "constant" / "triSurface" / "body.stl"
        assert not tri_stl.exists()

    def test_stl_import_with_pimple_foam(self, tmp_path):
        """pimpleFoam also generates snappy files for stl_import."""
        analysis = {
            **_STL_ANALYSIS_BASE,
            "solver": "pimpleFoam",
            "is_transient": True,
            "delta_t": 0.001,
            "end_time": 1.0,
        }
        gen = PimpleFoamGenerator()
        case = tmp_path / "pimple_stl"
        gen.generate(analysis, case)
        assert (case / "system" / "snappyHexMeshDict").exists()
        assert (case / "constant" / "triSurface" / "body.stl").exists()

    def test_stl_import_with_rho_simple_foam(self, tmp_path):
        """rhoSimpleFoam also generates snappy files for stl_import."""
        analysis = {
            **_STL_ANALYSIS_BASE,
            "solver": "rhoSimpleFoam",
            "T_ref": 300.0,
            "p_ref": 101325.0,
        }
        gen = RhoSimpleFoamGenerator()
        case = tmp_path / "rho_stl"
        gen.generate(analysis, case)
        assert (case / "system" / "snappyHexMeshDict").exists()

    def test_mesh_quality_dict_has_max_non_ortho(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "meshQualityDict").read_text()
        assert "maxNonOrtho" in text
        assert "65" in text

    def test_snappy_add_layers_true_by_default(self, tmp_path):
        case = _generate_stl_case(tmp_path)
        text = (case / "system" / "snappyHexMeshDict").read_text()
        assert "addLayers       true" in text

    def test_snappy_n_layers_zero_disables_layers(self, tmp_path):
        analysis = {**_STL_ANALYSIS_BASE, "geometry_params": {"n_layers": 0}}
        case = _generate_stl_case(tmp_path, analysis)
        text = (case / "system" / "snappyHexMeshDict").read_text()
        assert "addLayers       false" in text

    def test_custom_expansion_factors(self, tmp_path):
        analysis = {
            **_STL_ANALYSIS_BASE,
            "geometry_params": {"upstream": 1.0, "downstream": 3.0, "side": 1.0},
        }
        case = _generate_stl_case(tmp_path, analysis)
        text = (case / "system" / "blockMeshDict").read_text()
        # bg_xmin = 0 - 1*1 = -1.0, bg_xmax = 1 + 3*1 = 4.0
        assert "-1.0" in text
        assert "4.0" in text

    def test_geometry_type_not_stl_no_snappy(self, tmp_path):
        """channel geometry does not generate snappy files."""
        gen = SimpleFoamGenerator()
        case = tmp_path / "channel_case"
        gen.generate(SIMPLE_FOAM_ANALYSIS, case)
        assert not (case / "system" / "snappyHexMeshDict").exists()
        assert not (case / "constant" / "triSurface").exists()

class TestSnappySyntax:
    """OpenFOAM file-header and Jinja-placeholder checks for stl_import cases."""

    _NON_FOAM_FILES = {"Allrun", "Allclean", "README.md"}

    @pytest.mark.parametrize("generator_cls,extra", [
        (SimpleFoamGenerator,     {}),
        (PimpleFoamGenerator,     {"is_transient": True, "delta_t": 0.001, "end_time": 1.0}),
        (RhoSimpleFoamGenerator,  {"solver": "rhoSimpleFoam", "T_ref": 300.0, "p_ref": 101325.0}),
    ])
    def test_stl_all_files_have_foamfile_header(self, generator_cls, extra, tmp_path):
        analysis = {**_STL_ANALYSIS_BASE, **extra}
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        for fpath in case_dir.rglob("*"):
            if fpath.is_file() and fpath.name not in self._NON_FOAM_FILES:
                # Skip binary STL
                if fpath.suffix == ".stl":
                    continue
                text = fpath.read_text()
                assert "FoamFile" in text, (
                    f"Missing FoamFile header in {fpath.relative_to(case_dir)}"
                )

    @pytest.mark.parametrize("generator_cls,extra", [
        (SimpleFoamGenerator,     {}),
        (RhoSimpleFoamGenerator,  {"solver": "rhoSimpleFoam", "T_ref": 300.0, "p_ref": 101325.0}),
    ])
    def test_stl_no_jinja_placeholders_remain(self, generator_cls, extra, tmp_path):
        import re
        analysis = {**_STL_ANALYSIS_BASE, **extra}
        case_dir = tmp_path / "case"
        generator_cls().generate(analysis, case_dir)
        for fpath in case_dir.rglob("*"):
            if fpath.is_file() and fpath.suffix not in (".stl", ".eMesh"):
                text = fpath.read_text()
                placeholders = re.findall(r"\{\{.*?\}\}", text)
                assert not placeholders, (
                    f"Un-rendered Jinja placeholders {placeholders} in "
                    f"{fpath.relative_to(case_dir)}"
                )


class TestEditorSnappyDefaults:
    """defaults_from_analysis and apply_overrides handle snappy parameters."""

    def test_defaults_include_snappy_keys(self):
        from app.editor import defaults_from_analysis
        d = defaults_from_analysis(_STL_ANALYSIS_BASE)
        assert "snappy_refine_min"    in d
        assert "snappy_refine_max"    in d
        assert "snappy_feature_level" in d
        assert "snappy_n_layers"      in d
        assert "snappy_max_cells"     in d
        assert "snappy_upstream"      in d
        assert "snappy_downstream"    in d
        assert "snappy_side"          in d

    def test_defaults_have_sensible_values(self):
        from app.editor import defaults_from_analysis
        d = defaults_from_analysis(_STL_ANALYSIS_BASE)
        assert d["snappy_refine_min"] == 1
        assert d["snappy_refine_max"] == 2
        assert d["snappy_feature_level"] == 3
        assert d["snappy_n_layers"] == 3
        assert d["snappy_max_cells"] == 2_000_000
        assert d["snappy_upstream"] == 2.0
        assert d["snappy_downstream"] == 5.0
        assert d["snappy_side"] == 2.0

    def test_apply_overrides_sets_geometry_params(self):
        from app.editor import apply_overrides
        overrides = {
            "snappy_refine_min": 2,
            "snappy_refine_max": 4,
            "snappy_n_layers": 5,
            "snappy_max_cells": 1_000_000,
            "snappy_upstream": 3.0,
            "snappy_downstream": 8.0,
        }
        result = apply_overrides(_STL_ANALYSIS_BASE, overrides)
        gp = result["geometry_params"]
        assert gp["refine_min"] == 2
        assert gp["refine_max"] == 4
        assert gp["n_layers"] == 5
        assert gp["max_cells"] == 1_000_000
        assert gp["upstream"] == 3.0
        assert gp["downstream"] == 8.0

    def test_apply_overrides_zero_layers_works(self):
        """n_layers=0 disables addLayers and must not be converted to default."""
        from app.editor import apply_overrides
        result = apply_overrides(_STL_ANALYSIS_BASE, {"snappy_n_layers": 0})
        assert result["geometry_params"]["n_layers"] == 0

    def test_snappy_overrides_affect_generated_case(self, tmp_path):
        """Overriding refine_max changes the value in snappyHexMeshDict."""
        from app.editor import apply_overrides
        overrides = {"snappy_refine_max": 5, "snappy_n_layers": 0}
        working = apply_overrides(_STL_ANALYSIS_BASE, overrides)
        case = tmp_path / "case"
        SimpleFoamGenerator().generate(working, case)
        snappy_text = (case / "system" / "snappyHexMeshDict").read_text()
        assert "level ( 1 5 )" in snappy_text
        assert "addLayers       false" in snappy_text


class TestGeometrySketchSTL:
    """geometry_sketches.py has correct stl_import entries."""

    def test_stl_import_sketch_exists(self):
        from app.geometry_sketches import GEOMETRY_SKETCHES
        assert "stl_import" in GEOMETRY_SKETCHES

    def test_stl_import_label_is_human_readable(self):
        from app.geometry_sketches import get_label
        label = get_label("stl_import")
        assert "snappy" in label.lower() or "stl" in label.lower()
        assert label != "stl_import"

    def test_stl_import_sketch_contains_key_elements(self):
        from app.geometry_sketches import get_sketch
        sketch, desc = get_sketch("stl_import")
        assert "inlet" in sketch.lower() or "→" in sketch
        assert "outlet" in sketch.lower() or "→" in sketch
        assert "body" in sketch.lower() or "stl" in desc.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing dictionaries (sampleDict + probesDict)
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleFields:
    """_sample_fields() returns correct field lists."""

    def test_simple_foam_kEpsilon(self):
        ctx = {"solver_name": "simpleFoam", "turb_model": "kEpsilon"}
        fields = _sample_fields(ctx)
        assert "U" in fields
        assert "p" in fields
        assert "k" in fields
        assert "epsilon" in fields
        assert "omega" not in fields
        assert "alpha.water" not in fields

    def test_simple_foam_kOmegaSST(self):
        ctx = {"solver_name": "simpleFoam", "turb_model": "kOmegaSST"}
        fields = _sample_fields(ctx)
        assert "k" in fields
        assert "omega" in fields
        assert "epsilon" not in fields

    def test_simple_foam_SpalartAllmaras(self):
        ctx = {"solver_name": "simpleFoam", "turb_model": "SpalartAllmaras"}
        fields = _sample_fields(ctx)
        assert "nuTilda" in fields
        assert "k" not in fields

    def test_buoyant_simple_foam_includes_temperature(self):
        ctx = {"solver_name": "buoyantSimpleFoam", "turb_model": "kEpsilon"}
        fields = _sample_fields(ctx)
        assert "T" in fields
        assert "p_rgh" in fields

    def test_rho_simple_foam_includes_temperature(self):
        ctx = {"solver_name": "rhoSimpleFoam", "turb_model": "kEpsilon"}
        fields = _sample_fields(ctx)
        assert "T" in fields
        assert "p_rgh" not in fields

    def test_inter_foam_includes_alpha(self):
        ctx = {"solver_name": "interFoam", "turb_model": "kEpsilon"}
        fields = _sample_fields(ctx)
        assert "alpha.water" in fields
        assert "k" not in fields
        assert "epsilon" not in fields

    def test_always_includes_U_and_p(self):
        for solver in ("simpleFoam", "pimpleFoam", "rhoSimpleFoam",
                       "buoyantSimpleFoam", "interFoam"):
            ctx = {"solver_name": solver, "turb_model": "kEpsilon"}
            fields = _sample_fields(ctx)
            assert "U" in fields
            assert "p" in fields


class TestPostProcessingDicts:
    """sampleDict and probesDict are written for every generator."""

    @pytest.fixture
    def simple_case(self, tmp_path):
        gen = SimpleFoamGenerator()
        gen.generate(SIMPLE_FOAM_ANALYSIS, tmp_path)
        return tmp_path

    @pytest.fixture
    def pimple_case(self, tmp_path):
        gen = PimpleFoamGenerator()
        gen.generate(PIMPLE_FOAM_ANALYSIS, tmp_path)
        return tmp_path

    @pytest.fixture
    def bsf_case(self, tmp_path):
        gen = BuoyantSimpleFoamGenerator()
        gen.generate(BSF_ANALYSIS, tmp_path)
        return tmp_path

    @pytest.fixture
    def inter_case(self, tmp_path):
        gen = InterFoamGenerator()
        gen.generate(INTER_FOAM_ANALYSIS, tmp_path)
        return tmp_path

    @pytest.fixture
    def rho_case(self, tmp_path):
        gen = RhoSimpleFoamGenerator()
        gen.generate(RHO_SIMPLE_FOAM_ANALYSIS, tmp_path)
        return tmp_path

    # ── File existence ────────────────────────────────────────────────────────

    def test_sample_dict_written_simpleFoam(self, simple_case):
        assert (simple_case / "system" / "sampleDict").exists()

    def test_probes_dict_written_simpleFoam(self, simple_case):
        assert (simple_case / "system" / "probesDict").exists()

    def test_sample_dict_written_pimpleFoam(self, pimple_case):
        assert (pimple_case / "system" / "sampleDict").exists()

    def test_probes_dict_written_pimpleFoam(self, pimple_case):
        assert (pimple_case / "system" / "probesDict").exists()

    def test_sample_dict_written_buoyantSimpleFoam(self, bsf_case):
        assert (bsf_case / "system" / "sampleDict").exists()

    def test_probes_dict_written_buoyantSimpleFoam(self, bsf_case):
        assert (bsf_case / "system" / "probesDict").exists()

    def test_sample_dict_written_interFoam(self, inter_case):
        assert (inter_case / "system" / "sampleDict").exists()

    def test_probes_dict_written_interFoam(self, inter_case):
        assert (inter_case / "system" / "probesDict").exists()

    def test_sample_dict_written_rhoSimpleFoam(self, rho_case):
        assert (rho_case / "system" / "sampleDict").exists()

    def test_probes_dict_written_rhoSimpleFoam(self, rho_case):
        assert (rho_case / "system" / "probesDict").exists()

    # ── Content: sampleDict structure ─────────────────────────────────────────

    def test_sampleDict_has_FoamFile_header(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "FoamFile" in text
        assert "sampleDict" in text

    def test_sampleDict_has_centreline_set(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "centreline" in text

    def test_sampleDict_has_crossSection_set(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "crossSection" in text

    def test_sampleDict_has_interpolationScheme(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "interpolationScheme" in text

    def test_sampleDict_has_setFormat(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "setFormat" in text

    def test_sampleDict_fields_simpleFoam_kEpsilon(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        for field in ("U", "p", "k", "epsilon"):
            assert field in text

    def test_sampleDict_no_jinja_placeholders(self, simple_case):
        text = (simple_case / "system" / "sampleDict").read_text()
        assert "{{" not in text
        assert "{%" not in text

    def test_sampleDict_fields_interFoam(self, inter_case):
        text = (inter_case / "system" / "sampleDict").read_text()
        assert "alpha.water" in text
        assert "k" not in text  # interFoam is laminar

    def test_sampleDict_fields_buoyantSimpleFoam(self, bsf_case):
        text = (bsf_case / "system" / "sampleDict").read_text()
        assert "T" in text
        assert "p_rgh" in text

    def test_sampleDict_fields_rhoSimpleFoam(self, rho_case):
        text = (rho_case / "system" / "sampleDict").read_text()
        assert "T" in text

    # ── Content: probesDict structure ─────────────────────────────────────────

    def test_probesDict_has_FoamFile_header(self, simple_case):
        text = (simple_case / "system" / "probesDict").read_text()
        assert "FoamFile" in text
        assert "probesDict" in text

    def test_probesDict_has_three_probe_points(self, simple_case):
        text = (simple_case / "system" / "probesDict").read_text()
        # Each probe line looks like:  ( x y z )
        import re
        probes = re.findall(r"\(\s*[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s*\)", text)
        assert len(probes) == 3

    def test_probesDict_no_jinja_placeholders(self, simple_case):
        text = (simple_case / "system" / "probesDict").read_text()
        assert "{{" not in text
        assert "{%" not in text

    def test_probesDict_has_libsampling(self, simple_case):
        text = (simple_case / "system" / "probesDict").read_text()
        assert "libsampling" in text

    def test_probesDict_fields_interFoam(self, inter_case):
        text = (inter_case / "system" / "probesDict").read_text()
        assert "alpha.water" in text

    # ── Numeric correctness: sample locations ─────────────────────────────────

    def test_sampleDict_centreline_x_spans_domain(self, simple_case):
        """sampleDict must contain coordinates covering the domain extent (>5 m default)."""
        text = (simple_case / "system" / "sampleDict").read_text()
        import re
        # Match decimal numbers in the full file; domain_x default ~20 m → end x ~19.6
        floats = [float(m) for m in re.findall(r"\b\d+\.\d+\b", text)]
        assert any(f > 5.0 for f in floats), "Expected x-coordinate > 5 m in sampleDict"

    def test_readme_mentions_postprocess_commands(self, simple_case):
        readme = (simple_case / "README.md").read_text()
        assert "postProcess" in readme
        assert "sampleDict" in readme or "sample" in readme


# ─────────────────────────────────────────────────────────────────────────────
# Cylinder and pipe geometry tests
# ─────────────────────────────────────────────────────────────────────────────

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


class TestCylinderPipeMeshes:
    """Verify cylinder (O-grid) and pipe (wedge) blockMeshDict generation."""

    # ── Cylinder ──────────────────────────────────────────────────────────────

    def _cylinder_case(self, tmp_path):
        case_dir = tmp_path / "cylinder"
        SimpleFoamGenerator().generate(CYLINDER_ANALYSIS, case_dir)
        return case_dir

    def test_cylinder_blockmeshdict_created(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        assert (case / "system" / "blockMeshDict").exists()

    def test_cylinder_no_jinja_placeholders(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "{{" not in text and "}}" not in text

    def test_cylinder_has_arc_edges(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "arc" in text, "Expected arc edges for cylinder O-grid"

    def test_cylinder_has_eight_arc_edges(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        arc_count = text.count("\n    arc ")
        assert arc_count == 8, f"Expected 8 arc edges, got {arc_count}"

    def test_cylinder_patch_present(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "cylinder" in text

    def test_cylinder_symmetry_patches(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "symmetryPlane" in text

    def test_cylinder_twelve_blocks(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        # Count hex block entries
        assert text.count("hex (") == 12, "Expected 12 hex blocks for O-grid"

    def test_cylinder_frontandback_empty(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "frontAndBack" in text
        assert "type empty" in text

    def test_cylinder_U_has_symmetryplane_for_top(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "0" / "U").read_text()
        assert "symmetryPlane" in text

    def test_cylinder_U_has_noSlip_for_cylinder_wall(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "0" / "U").read_text()
        assert "noSlip" in text

    def test_cylinder_foamfile_header(self, tmp_path):
        case = self._cylinder_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "FoamFile" in text

    # ── Pipe ──────────────────────────────────────────────────────────────────

    def _pipe_case(self, tmp_path):
        case_dir = tmp_path / "pipe"
        SimpleFoamGenerator().generate(PIPE_ANALYSIS, case_dir)
        return case_dir

    def test_pipe_blockmeshdict_created(self, tmp_path):
        case = self._pipe_case(tmp_path)
        assert (case / "system" / "blockMeshDict").exists()

    def test_pipe_no_jinja_placeholders(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "{{" not in text and "}}" not in text

    def test_pipe_one_block(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert text.count("hex (") == 1, "Expected exactly 1 hex block for wedge pipe"

    def test_pipe_wedge_patches_present(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "front" in text
        assert "back" in text
        assert "wedge" in text

    def test_pipe_axis_patch_empty(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "axis" in text
        assert "type empty" in text

    def test_pipe_wall_patch_present(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "    wall\n" in text

    def test_pipe_U_wedge_bc(self, tmp_path):
        """front and back patches in 0/U must have type wedge."""
        case = self._pipe_case(tmp_path)
        text = (case / "0" / "U").read_text()
        assert "type            wedge" in text

    def test_pipe_p_wedge_bc(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "0" / "p").read_text()
        assert "type            wedge" in text

    def test_pipe_foamfile_header(self, tmp_path):
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "FoamFile" in text

    def test_pipe_eight_vertices(self, tmp_path):
        """Wedge pipe has exactly 8 vertex entries (axis duplicated)."""
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        # Isolate the vertices block: starts at "vertices\n(" keyword (the dict keyword,
        # not word in a comment), ends at first ");" after that.
        import re
        # Match the vertices keyword as a standalone word on its own line
        m = re.search(r"^\s*vertices\s*\(", text, re.MULTILINE)
        assert m, "No vertices block found"
        verts_text = text[m.end():]
        verts_text = verts_text.split(");")[0]
        # Each vertex line starts with ( and contains 3 numeric coords (spaces+signs+dots)
        verts = re.findall(r"^\s*\(\s*[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s*\)",
                           verts_text, re.MULTILINE)
        assert len(verts) == 8, f"Expected 8 vertices for wedge pipe, got {len(verts)}"

    def test_pipe_no_arc_edges(self, tmp_path):
        """Pipe mesh has no arc edges — it is a single straight-sided block."""
        case = self._pipe_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "arc" not in text

    # ── Cross-solver: cylinder with pimpleFoam ────────────────────────────────

    def test_cylinder_pimplefoam(self, tmp_path):
        """Cylinder geometry also works with pimpleFoam (transient)."""
        cyl_pimple = {
            **CYLINDER_ANALYSIS,
            "solver": "pimpleFoam",
            "is_transient": True,
        }
        case_dir = tmp_path / "cyl_pimple"
        PimpleFoamGenerator().generate(cyl_pimple, case_dir)
        text = (case_dir / "system" / "blockMeshDict").read_text()
        assert "arc" in text
        assert "cylinder" in text
        assert "{{" not in text

    def test_pipe_pimplefoam(self, tmp_path):
        """Pipe geometry also works with pimpleFoam."""
        pipe_pimple = {
            **PIPE_ANALYSIS,
            "solver": "pimpleFoam",
            "is_transient": True,
        }
        case_dir = tmp_path / "pipe_pimple"
        PimpleFoamGenerator().generate(pipe_pimple, case_dir)
        text = (case_dir / "system" / "blockMeshDict").read_text()
        assert "wedge" in text
        assert "{{" not in text


# ─────────────────────────────────────────────────────────────────────────────
# Airfoil C/H-grid geometry tests
# ─────────────────────────────────────────────────────────────────────────────

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


class TestAirfoilMesh:
    """Verify airfoil 8-block C/H-grid blockMeshDict generation."""

    def _airfoil_case(self, tmp_path):
        case_dir = tmp_path / "airfoil"
        SimpleFoamGenerator().generate(AIRFOIL_ANALYSIS, case_dir)
        return case_dir

    # ── Basic file existence and template rendering ────────────────────────

    def test_airfoil_blockmeshdict_created(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        assert (case / "system" / "blockMeshDict").exists()

    def test_airfoil_no_jinja_placeholders(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "{{" not in text
        assert "}}" not in text

    def test_airfoil_foamfile_header(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "FoamFile" in text
        assert "blockMeshDict" in text

    # ── Block and vertex counts ────────────────────────────────────────────

    def test_airfoil_eight_blocks(self, tmp_path):
        """Airfoil mesh must have exactly 8 hex blocks."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        import re
        # Find the 'blocks' keyword on its own line (not inside a comment)
        m = re.search(r"^\s*blocks\s*\(", text, re.MULTILINE)
        assert m, "No 'blocks' section found"
        blocks_section = text[m.end():].split(");")[0]
        hex_count = len(re.findall(r"\bhex\b", blocks_section))
        assert hex_count == 8, f"Expected 8 hex blocks, found {hex_count}"

    def test_airfoil_32_vertices(self, tmp_path):
        """Airfoil mesh must have exactly 32 vertices (16 front + 16 back)."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        import re
        # Find the vertices section up to the first ');'
        verts_start = re.search(r"^\s*vertices\s*\(", text, re.MULTILINE)
        assert verts_start, "No vertices section found"
        verts_text = text[verts_start.end():].split(");")[0]
        # Match lines containing exactly 3 numbers (x y z)
        coord_lines = re.findall(
            r"^\s*\(\s*[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s*\)",
            verts_text, re.MULTILINE,
        )
        assert len(coord_lines) == 32, f"Expected 32 vertices, found {len(coord_lines)}"

    # ── Patch definitions ──────────────────────────────────────────────────

    def test_airfoil_patch_inlet(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "inlet" in text

    def test_airfoil_patch_outlet(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "outlet" in text

    def test_airfoil_patch_wall(self, tmp_path):
        """The 'airfoil' wall patch must be present."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "airfoil" in text
        assert "type wall" in text

    def test_airfoil_symmetry_patches(self, tmp_path):
        """Top and bottom must be symmetryPlane patches."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "symmetryPlane" in text
        assert "top" in text
        assert "bottom" in text

    def test_airfoil_empty_patch(self, tmp_path):
        """frontAndBack must be an empty (2-D) patch."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        assert "frontAndBack" in text
        assert "type empty" in text

    # ── Airfoil wall has 4 faces (LE, upper, lower, TE) ───────────────────

    def test_airfoil_wall_four_faces(self, tmp_path):
        """The airfoil wall patch must list exactly 4 face entries."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        import re
        # Find the wall patch (type wall) and extract its faces block
        m = re.search(r"type\s+wall\s*;.*?\bfaces\b\s*\((.*?)\);", text, re.DOTALL)
        assert m, "Could not find wall (airfoil) faces block"
        face_entries = re.findall(r"\(\s*\d+\s+\d+\s+\d+\s+\d+\s*\)", m.group(1))
        assert len(face_entries) == 4, f"Expected 4 airfoil faces, found {len(face_entries)}"

    # ── Boundary-condition files ───────────────────────────────────────────

    def test_airfoil_U_has_noSlip_wall(self, tmp_path):
        """U file must have noSlip BC on the airfoil wall."""
        case = self._airfoil_case(tmp_path)
        u_text = (case / "0" / "U").read_text()
        assert "noSlip" in u_text

    def test_airfoil_U_has_symmetry(self, tmp_path):
        """U file must have symmetry BC on top/bottom patches."""
        case = self._airfoil_case(tmp_path)
        u_text = (case / "0" / "U").read_text()
        assert "symmetry" in u_text.lower()

    def test_airfoil_no_jinja_in_U(self, tmp_path):
        case = self._airfoil_case(tmp_path)
        u_text = (case / "0" / "U").read_text()
        assert "{{" not in u_text

    # ── No arc edges (rectangular LE/TE) ──────────────────────────────────

    def test_airfoil_no_arc_edges(self, tmp_path):
        """Airfoil uses straight edges only (no arc, unlike cylinder)."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        import re
        # arc keyword inside the edges() block — should not appear
        edges_section = text.split("edges")[1].split(");")[0]
        assert not re.search(r"\barc\b", edges_section), "Unexpected arc edge in airfoil mesh"

    # ── Context variables sanity check ────────────────────────────────────

    def test_airfoil_geometry_params_in_blockmesh(self, tmp_path):
        """Far-field radius (15 m), chord (1 m) and wake (10 m) must appear in blockMeshDict."""
        case = self._airfoil_case(tmp_path)
        text = (case / "system" / "blockMeshDict").read_text()
        # Far-field upstream inlet x = -R_far = -15
        assert "-15" in text
        # Chord TE x = 1.0
        assert "1" in text

    # ── Cross-solver: airfoil with pimpleFoam ─────────────────────────────

    def test_airfoil_pimplefoam(self, tmp_path):
        """Airfoil geometry also works with pimpleFoam (transient)."""
        af_pimple = {
            **AIRFOIL_ANALYSIS,
            "solver": "pimpleFoam",
            "is_transient": True,
        }
        case_dir = tmp_path / "af_pimple"
        PimpleFoamGenerator().generate(af_pimple, case_dir)
        text = (case_dir / "system" / "blockMeshDict").read_text()
        assert "airfoil" in text
        assert "{{" not in text

    def test_airfoil_rhoSimpleFoam(self, tmp_path):
        """Airfoil geometry works with rhoSimpleFoam (compressible external aero)."""
        from app.generators import RhoSimpleFoamGenerator
        af_rho = {
            **AIRFOIL_ANALYSIS,
            "solver": "rhoSimpleFoam",
            "is_transient": False,
            "Ma": 0.5,
            "T_ref": 288.15,
            "p_ref": 101325.0,
            "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
        }
        case_dir = tmp_path / "af_rho"
        RhoSimpleFoamGenerator().generate(af_rho, case_dir)
        text = (case_dir / "system" / "blockMeshDict").read_text()
        assert "airfoil" in text
        assert "{{" not in text

    # ── Geometry sketches ─────────────────────────────────────────────────

    def test_airfoil_sketch_exists(self):
        """Airfoil geometry sketch must be present in GEOMETRY_SKETCHES."""
        from app.geometry_sketches import GEOMETRY_SKETCHES, GEOMETRY_LABELS
        assert "airfoil" in GEOMETRY_SKETCHES
        assert "airfoil" in GEOMETRY_LABELS
        sketch, desc = GEOMETRY_SKETCHES["airfoil"]
        assert len(sketch) > 0
        assert "airfoil" in desc.lower() or "grid" in desc.lower()
