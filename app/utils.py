"""
File writing and zip helpers for the CFD Case Generator.
"""

import io
import os
import zipfile
from pathlib import Path
from app.naming import is_inlet_name


def write_file(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def zip_directory(source_dir: Path) -> bytes:
    """
    Recursively zip *source_dir* and return the archive as raw bytes.

    The zip entries use paths relative to *source_dir* so that when the
    user extracts the archive they get a clean case folder.

    Parameters
    ----------
    source_dir : Path
        Root of the OpenFOAM case folder to archive.

    Returns
    -------
    bytes
        The ZIP file content ready for Streamlit's download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                zf.write(file_path, arcname)
    buf.seek(0)
    return buf.read()


def count_zip_files(zip_bytes: bytes) -> int:
    """Return the number of files inside a zip archive given as bytes."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return len([n for n in zf.namelist() if not n.endswith("/")])


def foam_vector(values: list) -> str:
    """Format a Python list as an OpenFOAM vector string, e.g. '(1 0 0)'."""
    return "(" + " ".join(str(v) for v in values) + ")"


def foam_scalar(value: float) -> str:
    """Format a float for OpenFOAM — uses scientific notation for small values."""
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.6e}"
    return str(value)


def get_bc_value(boundary_conditions: list, patch_name: str, field: str, default: str = "0") -> str:
    """
    Search *boundary_conditions* list for the given patch and field.
    Returns the formatted value string or *default* if not found.
    """
    for bc in boundary_conditions:
        if bc.get("patch_name", "").lower() == patch_name.lower():
            val = bc.get("values", {}).get(field)
            if val is not None:
                if isinstance(val, list):
                    return foam_vector(val)
                return str(val)
    return default


def find_patch(boundary_conditions: list, patch_name: str) -> dict:
    """Return the first BC dict whose patch_name matches (case-insensitive)."""
    for bc in boundary_conditions:
        if bc.get("patch_name", "").lower() == patch_name.lower():
            return bc
    return {}


def get_inlet_velocity(boundary_conditions: list) -> str:
    """Extract the inlet velocity vector from BCs; default to (1 0 0)."""
    for bc in boundary_conditions:
        name = bc.get("patch_name", "").lower()
        if is_inlet_name(name):
            val = bc.get("values", {}).get("U")
            if val is not None:
                if isinstance(val, list):
                    return foam_vector(val)
                return str(val)
    return "(1 0 0)"


def get_inlet_speed(boundary_conditions: list) -> float:
    """Return the magnitude of the inlet velocity (first component if vector)."""
    for bc in boundary_conditions:
        name = bc.get("patch_name", "").lower()
        if is_inlet_name(name):
            val = bc.get("values", {}).get("U")
            if isinstance(val, list):
                import math
                return math.sqrt(sum(v**2 for v in val))
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return 1.0
