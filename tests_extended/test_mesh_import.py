"""SALOME mesh import: role mapping from group names (no OpenFOAM needed)."""

from pathlib import Path

import pytest

from app.agent import clean
from app.mesh_import import Patch, check_roles, heuristic_roles


def _p(name):
    return Patch(name, 10, 1.0, (1.0, 0.0, 0.0))


@pytest.mark.parametrize("name, role", [
    ("Entree", "inlet"), ("entrée", "inlet"), ("Inflow", "inlet"), ("Sortie", "outlet"),
    ("outlet_1", "outlet"), ("Parois", "wall"), ("hull", "wall"), ("frontAndBack", "empty"),
    ("farfield", "symmetry"), ("hot", "hot_wall"), ("Froid", "cold_wall"), ("xyz", "wall"),
])
def test_group_names_map_to_roles(name, role):
    assert heuristic_roles([_p(name)])[name] == role


def test_role_checks():
    assert check_roles({"a": "inlet", "b": "outlet", "c": "wall"}, "simpleFoam") == []
    assert "no outlet patch" in check_roles({"a": "inlet", "c": "wall"}, "simpleFoam")
    assert check_roles({"h": "hot_wall", "c": "cold_wall"}, "buoyantSimpleFoam") == []


def test_agent_clean_keeps_an_imported_mesh(tmp_path: Path):
    """The agent's clean() used to delete constant/polyMesh even when nothing rebuilds it."""
    (tmp_path / "constant" / "polyMesh").mkdir(parents=True)
    (tmp_path / "system").mkdir()
    clean(tmp_path)
    assert (tmp_path / "constant" / "polyMesh").exists()
    (tmp_path / "system" / "blockMeshDict").write_text("")
    clean(tmp_path)
    assert not (tmp_path / "constant" / "polyMesh").exists()
