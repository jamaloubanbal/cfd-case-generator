"""Tests for the cfdgen command-line interface and the shipped examples."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cli import main
from app.sanitize import sanitize_analysis

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.json"))


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_examples_are_clean_and_generate(example, tmp_path):
    data = json.loads(example.read_text())
    assert data["description"]
    _, notes = sanitize_analysis(data["analysis"])
    assert notes == [], f"example needs automatic corrections: {notes}"
    assert main(["generate", str(example), "-o", str(tmp_path / "case")]) == 0
    assert (tmp_path / "case" / "Allrun").exists()


def test_generate_refuses_to_overwrite(tmp_path):
    out = tmp_path / "case"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(["generate", str(EXAMPLES[0]), "-o", str(out)]) == 2
    assert (out / "keep.txt").exists()
    assert main(["generate", str(EXAMPLES[0]), "-o", str(out), "--force"]) == 0


def test_check_command(capsys):
    assert main(["check", str(EXAMPLES[0])]) == 0
    assert "Physics checks" in capsys.readouterr().out


def test_describe_uses_llm_and_saves_analysis(tmp_path):
    analysis = json.loads(EXAMPLES[0].read_text())["analysis"]
    with patch("app.llm.analyze_cfd_problem", return_value=analysis) as mock_llm:
        rc = main(["describe", "some flow", "-o", str(tmp_path / "case"),
                   "--save-analysis", str(tmp_path / "a.json")])
    assert rc == 0
    mock_llm.assert_called_once()
    assert json.loads((tmp_path / "a.json").read_text())["solver"] == analysis["solver"]
