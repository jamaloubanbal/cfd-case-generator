"""The LLM-evaluation scorer: correct answers score fully, the sanitiser's repairs are counted."""

from evals.run import CASES, score_case


def test_scoring_raw_vs_sanitised():
    case = next(c for c in CASES if c["id"] == "cyl_re100")
    good = {"solver": "pimpleFoam", "geometry_type": "cylinder", "turb_model": "laminar",
            "is_transient": True, "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.2},
            "characteristic_length": 0.1,
            "boundary_conditions": [{"patch_name": "inlet", "type": "fixedValue",
                                     "values": {"U": [0.015, 0, 0]}}]}
    # mistral's real mistakes: sideways inlet, U·D/ν ≠ 100, steady solver
    bad = {**good, "solver": "simpleFoam", "is_transient": False, "characteristic_length": 1.0,
           "boundary_conditions": [{"patch_name": "inlet", "type": "fixedValue",
                                    "values": {"U": [0, 1, 0]}}]}
    assert all(score_case(case, good)["raw"].values())
    s = score_case(case, bad)
    assert not s["raw"]["solver"] and not s["raw"]["Re"]
    assert all(s["sanitised"].values())                  # the sanitiser repairs all three
    assert not any(score_case(case, None)["sanitised"].values())
