"""Registry-backed unfolding exports."""

from fluxforge.plugins import bootstrap_builtin_registries
from fluxforge.unfolding.base import (
    estimate_unfolding_uncertainties,
    UnfoldingMethod,
    UnfoldingMethodDefinition,
    UnfoldingResult,
    validate_unfolding_inputs,
)
from fluxforge.unfolding.gravel import (
    GravelUnfolder,
    register_builtin_unfolders,
    unfolding_entries,
)
from fluxforge.unfolding.maxed import MaxedUnfolder
from fluxforge.unfolding.ml_seed import MLSeedUnfolder
from fluxforge.unfolding.rmle import RMLEUnfolder

__all__ = [
    "GravelUnfolder",
    "MaxedUnfolder",
    "MLSeedUnfolder",
    "RMLEUnfolder",
    "UnfoldingMethod",
    "UnfoldingMethodDefinition",
    "UnfoldingResult",
    "estimate_unfolding_uncertainties",
    "register_builtin_unfolders",
    "unfolding_entries",
    "validate_unfolding_inputs",
]

_shared_registries = bootstrap_builtin_registries()
if len(_shared_registries.unfolders) == 0:
    register_builtin_unfolders(_shared_registries)
