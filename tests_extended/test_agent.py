"""Repair agent: diagnosis from logs and the safety rails on LLM edits (no OpenFOAM needed)."""

from pathlib import Path

import pytest

from app.agent import _reject_reason, diagnose

BANNER = "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
HEADER = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n}\n"


@pytest.fixture
def case(tmp_path: Path) -> Path:
    for d in ("system", "constant", "0"):
        (tmp_path / d).mkdir()
    (tmp_path / "system" / "controlDict").write_text(HEADER + "application simpleFoam;\nendTime 100;\n")
    (tmp_path / "system" / "fvSolution").write_text(HEADER + "solvers\n{\n}\n")
    (tmp_path / "Allrun").write_text("#!/bin/sh\n")
    return tmp_path


def test_successful_log_with_fpe_banner_is_ok(case):
    (case / "log.simpleFoam").write_text(BANNER + "Time = 1\nsmoothSolver: Solving for Ux\nEnd\n")
    assert diagnose(case, 0).ok


def test_missing_entry_names_file_without_dictionary_section(case):
    (case / "log.pimpleFoam").write_text(
        BANNER + "--> FOAM FATAL IO ERROR: \nEntry 'kFinal' not found in dictionary \"x\"\n\n"
        f"file: {case}/system/fvSolution.solvers from line 21 to line 58.\n\n    From function f\n")
    d = diagnose(case, 1)
    assert (d.kind, d.entry, d.file) == ("missing_entry", "kFinal", "system/fvSolution")


@pytest.mark.parametrize("rel, content, reason", [
    ("Allrun", HEADER, "may not be edited"),
    ("../outside", HEADER, "outside the case"),
    ("system/fvSolution", HEADER + "#codeStream { code #{ system(\"rm -rf ~\"); #}; };", "code-execution"),
    ("system/controlDict", HEADER + "application simpleFoam;\nendTime 5;\n", "endTime"),
    ("system/fvSolution", "solvers {}", "plausible"),
])
def test_unsafe_llm_edits_are_rejected(case, rel, content, reason):
    assert reason in (_reject_reason(case, rel, content) or "")


def test_reasonable_llm_edit_is_accepted(case):
    assert _reject_reason(case, "system/fvSolution", HEADER + "solvers\n{\n    p { solver GAMG; }\n}\n") is None


@pytest.mark.parametrize("given, expected", [
    ("0/fvSchemes", "system/fvSchemes"), ("0/p.boundaryField.outlet", "0/p"),
    ("system/fvSolution", "system/fvSolution"), ("constant/unknownDict", "constant/unknownDict"),
])
def test_near_miss_paths_map_to_files_the_model_was_shown(given, expected):
    from app.agent import _match_sent_file
    assert _match_sent_file(given, ["0/p", "system/fvSolution", "system/fvSchemes"]) == expected
