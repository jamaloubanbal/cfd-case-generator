"""Recognising boundary names, shared by the sanitiser, generators, checks and editor."""

import re

# English and French inlet names, as LLMs and SALOME users write them: "inlet", "inlet2",
# "inflow", "entree"/"entrée", "entry", "in", "duct_in", "admission".
_INLET = re.compile(r"inlet|inflow|entr[eé]e|\bentry\b|admission|(^|[_\W])in($|[_\W\d])", re.I)


def is_inlet_name(name: str | None) -> bool:
    """True if a boundary-condition or patch name denotes an inlet."""
    return bool(name) and bool(_INLET.search(str(name)))
