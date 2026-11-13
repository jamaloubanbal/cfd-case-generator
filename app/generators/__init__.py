"""
Generator registry.

Maps solver names (as returned by Claude) to their generator classes.
"""

from .simple_foam import SimpleFoamGenerator
from .pimple_foam import PimpleFoamGenerator
from .buoyant_simple_foam import BuoyantSimpleFoamGenerator
from .inter_foam import InterFoamGenerator
from .rho_simple_foam import RhoSimpleFoamGenerator

GENERATORS = {
    "simpleFoam":         SimpleFoamGenerator,
    "pimpleFoam":         PimpleFoamGenerator,
    "buoyantSimpleFoam":  BuoyantSimpleFoamGenerator,
    "interFoam":          InterFoamGenerator,
    "rhoSimpleFoam":      RhoSimpleFoamGenerator,
}

__all__ = [
    "GENERATORS",
    "SimpleFoamGenerator",
    "PimpleFoamGenerator",
    "BuoyantSimpleFoamGenerator",
    "InterFoamGenerator",
    "RhoSimpleFoamGenerator",
]
