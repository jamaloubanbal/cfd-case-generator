"""Post-processing of OpenFOAM results in pure Python (numpy).

OpenFOAM's own function objects (forces, probes, wallHeatFlux …) crash in some
v1912 builds, including the Debian/Ubuntu package, with an ``IOstream "sha1"``
error.  Reading the ASCII mesh and fields directly works with any build.
"""

from .foam import FoamCase

__all__ = ["FoamCase"]
