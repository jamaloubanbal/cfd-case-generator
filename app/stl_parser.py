"""
Lightweight STL bounding-box extractor.

Supports both binary and ASCII STL formats without external dependencies.
Used by the UI to compute the background mesh bounds for snappyHexMesh cases.
"""

from __future__ import annotations

import struct


def stl_bounding_box(data: bytes) -> tuple[float, float, float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax, zmin, zmax)`` for the given STL bytes.

    Tries binary format first (matches expected file size), then falls back to
    ASCII parsing.  Raises ``ValueError`` when no valid vertices are found.
    """
    if len(data) >= 84:
        n_tri = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + n_tri * 50
        if 0 < n_tri <= 10_000_000 and len(data) >= expected:
            try:
                return _binary_bbox(data, n_tri)
            except struct.error:
                pass
    return _ascii_bbox(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_bbox(
    data: bytes, n_tri: int
) -> tuple[float, float, float, float, float, float]:
    """Parse a binary STL and return its bounding box."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    offset = 84
    for _ in range(n_tri):
        # 12 bytes normal, then 3 × 12 bytes vertices, then 2 bytes attribute
        for v in range(3):
            x, y, z = struct.unpack_from("<fff", data, offset + 12 + v * 12)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        offset += 50
    if not xs:
        raise ValueError("Binary STL has no vertices")
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _ascii_bbox(data: bytes) -> tuple[float, float, float, float, float, float]:
    """Parse an ASCII STL and return its bounding box."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    text = data.decode("utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex "):
            parts = stripped.split()
            if len(parts) >= 4:
                xs.append(float(parts[1]))
                ys.append(float(parts[2]))
                zs.append(float(parts[3]))
    if not xs:
        raise ValueError("No vertices found in ASCII STL")
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
