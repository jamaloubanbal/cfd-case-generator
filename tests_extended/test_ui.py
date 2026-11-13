"""
Headless Streamlit tests (streamlit.testing.v1.AppTest) — no browser, no API key.

The LLM call is mocked; everything else (sanitising, parameter editor, physics
checks, case generation, zip) runs for real.
"""

import copy
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.fixtures import CYLINDER_ANALYSIS, SIMPLE_FOAM_ANALYSIS  # noqa: E402

MAIN = str(Path(__file__).resolve().parent.parent / "app" / "main.py")
TIMEOUT = 60


def _app() -> AppTest:
    return AppTest.from_file(MAIN, default_timeout=TIMEOUT)


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_app_renders_without_analysis():
    at = _app().run()
    _no_exceptions(at)
    assert "CFD Case Generator" in at.title[0].value
    # Analyse is disabled until an API key and a description are present
    analyse = next(b for b in at.button if b.key == "btn_analyse")
    assert analyse.disabled


def test_full_flow_analyse_generate(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raw = copy.deepcopy(SIMPLE_FOAM_ANALYSIS)
    raw["fluid"]["nu"] = 15.0           # wrong units → must be auto-corrected

    with patch("app.ui.problem_input.analyze_cfd_problem", return_value=raw) as mock_llm:
        at = _app().run()
        at.sidebar.text_input[0].input("sk-ant-test").run()
        at.text_area(key="user_description").input("Air in a 2-D channel at 1 m/s").run()
        next(b for b in at.button if b.key == "btn_analyse").click().run()
        _no_exceptions(at)
        mock_llm.assert_called_once()

        assert at.session_state["analysis"]["fluid"]["nu"] == pytest.approx(1.5e-5)
        assert any("Automatic corrections" in i.value for i in at.info)

        next(b for b in at.button if b.key == "btn_generate").click().run()
        _no_exceptions(at)

    zip_bytes = at.session_state["zip_bytes"]
    names = zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()
    assert any(n.endswith("system/blockMeshDict") for n in names)
    assert any(n.endswith("Allrun") for n in names)


def test_mesh_editor_override_reaches_generated_case():
    at = _app()
    at.session_state["analysis"] = copy.deepcopy(SIMPLE_FOAM_ANALYSIS)
    at.run()
    at.number_input(key="pe_nx").set_value(130).run()
    next(b for b in at.button if b.key == "btn_generate").click().run()
    _no_exceptions(at)
    zf = zipfile.ZipFile(io.BytesIO(at.session_state["zip_bytes"]))
    bmd = zf.read(next(n for n in zf.namelist() if n.endswith("system/blockMeshDict"))).decode()
    assert "(130 " in bmd


def test_multiblock_geometry_offers_refinement_not_nx():
    at = _app()
    at.session_state["analysis"] = copy.deepcopy(CYLINDER_ANALYSIS)
    at.run()
    _no_exceptions(at)
    keys = {w.key for w in at.number_input}
    assert "pe_nx" not in keys
    assert any(s.key == "pe_mesh_refinement" for s in at.slider)
