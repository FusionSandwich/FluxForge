"""
FluxForge optional unfolding methods.

This subpackage provides alternative unfolding solvers that wrap
external libraries (PyLops for gamma, PyUnfold for neutron).

These are *optional* cross-check methods, not replacements for the
core FluxForge solvers.
"""

from __future__ import annotations

__all__ = [
    "GammaUnfolderRMLE",
    "NeutronUnfolderIBU",
    "ReactionRates",
    "ResponseBundle",
    "SpectrumFile",
]

# Lazy imports to avoid hard dependency on optional packages
def __getattr__(name: str):
    if name == "GammaUnfolderRMLE":
        from .gamma_rmle import GammaUnfolderRMLE
        return GammaUnfolderRMLE
    if name == "NeutronUnfolderIBU":
        from .neutron_ibu import NeutronUnfolderIBU
        return NeutronUnfolderIBU
    if name in ("ReactionRates", "ResponseBundle", "SpectrumFile"):
        from . import _types
        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
