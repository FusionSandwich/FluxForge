"""Registry-backed GRAVEL unfolding method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxforge.core.unfolding_diagnostics import summarize_flux_bins
from fluxforge.plugins import PluginRegistries, bootstrap_builtin_registries
from fluxforge.solvers.iterative import gravel as legacy_gravel
from fluxforge.unfolding.base import (
    estimate_unfolding_uncertainties,
    UnfoldingMethod,
    UnfoldingMethodDefinition,
    UnfoldingResult,
    validate_unfolding_inputs,
)


@dataclass(frozen=True)
class GravelUnfolder(UnfoldingMethod):
    """GRAVEL wrapper aligned with the shared unfolding contract."""

    max_iterations: int = 1000
    tolerance: float = 1e-4
    chi2_tolerance: float = 0.01
    floor: float = 1e-100
    relaxation: float = 0.7
    convergence_mode: str = "relative"

    @classmethod
    def definition(cls) -> UnfoldingMethodDefinition:
        return UnfoldingMethodDefinition(
            key="gravel",
            label="GRAVEL",
            summary="Gold/SAND-II style iterative unfolding for reproducible detector-response deconvolution.",
            convergence_metric="chi_squared",
            supports_uncertainties=True,
            can_produce_negative_bins=False,
            method_category="user_selected",
        )

    def unfold(
        self,
        measured: np.ndarray,
        response_matrix: np.ndarray,
        **kwargs,
    ) -> UnfoldingResult:
        """Run the GRAVEL adapter against normalized NumPy inputs."""

        measured_array, response_array, initial_flux, uncertainty_array = (
            validate_unfolding_inputs(
                measured,
                response_matrix,
                initial_flux=kwargs.get("initial_flux"),
                measurement_uncertainty=kwargs.get("measurement_uncertainty"),
            )
        )

        max_iterations = int(kwargs.get("max_iterations", self.max_iterations))
        tolerance = float(kwargs.get("tolerance", self.tolerance))
        chi2_tolerance = float(kwargs.get("chi2_tolerance", self.chi2_tolerance))
        floor = float(kwargs.get("floor", self.floor))
        relaxation = float(kwargs.get("relaxation", self.relaxation))
        convergence_mode = str(
            kwargs.get("convergence_mode", self.convergence_mode)
        )
        seed_with_ml = bool(kwargs.get("seed_with_ml", False))
        confidence_threshold = float(kwargs.get("confidence_threshold", 0.6))
        verbose = bool(kwargs.get("verbose", False))

        seed_result = None
        if seed_with_ml:
            from fluxforge.unfolding.ml_seed import MLSeedUnfolder

            seed_result = MLSeedUnfolder().unfold(
                measured_array,
                response_array,
                initial_flux=initial_flux,
                measurement_uncertainty=uncertainty_array,
                confidence_threshold=confidence_threshold,
            )
            if bool(seed_result.parameters_used.get("accepted", False)):
                initial_flux = np.asarray(seed_result.flux, dtype=float)

        solution = legacy_gravel(
            response_array.tolist(),
            measured_array.tolist(),
            initial_flux=initial_flux.tolist() if initial_flux is not None else None,
            measurement_uncertainty=(
                uncertainty_array.tolist() if uncertainty_array is not None else None
            ),
            max_iters=max_iterations,
            tolerance=tolerance,
            chi2_tolerance=chi2_tolerance,
            floor=floor,
            relaxation=relaxation,
            convergence_mode=convergence_mode,
            verbose=verbose,
        )

        flux = np.asarray(solution.flux, dtype=float)
        predicted_measurements = response_array @ flux
        residuals = measured_array - predicted_measurements
        uncertainties = estimate_unfolding_uncertainties(
            response_array,
            measured=measured_array,
            measurement_uncertainty=uncertainty_array,
        )
        definition = self.definition()
        summary = summarize_flux_bins(
            flux,
            negative_policy="floor_clamped",
            nonnegativity_enforced=True,
        )

        return UnfoldingResult(
            flux=flux,
            uncertainties=uncertainties,
            convergence_history=tuple(solution.chi_squared_history),
            method_used=definition.label,
            parameters_used={
                "max_iterations": max_iterations,
                "tolerance": tolerance,
                "chi2_tolerance": chi2_tolerance,
                "floor": floor,
                "relaxation": relaxation,
                "convergence_mode": convergence_mode,
                "used_initial_flux": initial_flux is not None,
                "used_measurement_uncertainty": uncertainty_array is not None,
                "uncertainty_estimator": "pseudo_inverse",
                "negative_policy": summary["negative_policy"],
                "seed_with_ml": seed_with_ml,
                "seed_accepted": bool(
                    seed_result is not None
                    and seed_result.parameters_used.get("accepted", False)
                ),
                "seed_confidence_score": (
                    float(seed_result.parameters_used.get("confidence_score", 0.0))
                    if seed_result is not None
                    else None
                ),
                "seed_backend": (
                    str(seed_result.parameters_used.get("backend"))
                    if seed_result is not None
                    else None
                ),
            },
            negative_bin_count=int(summary["negative_bin_count"]),
            method_category=definition.method_category,
            standards_locked_by=None,
            predicted_measurements=predicted_measurements,
            residuals=residuals,
            chi_squared=solution.chi_squared,
            iterations=solution.iterations,
            converged=solution.converged,
        )


def register_builtin_unfolders(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the currently available unfolding methods."""

    from fluxforge.unfolding.maxed import MaxedUnfolder
    from fluxforge.unfolding.ml_seed import MLSeedUnfolder
    from fluxforge.unfolding.rmle import RMLEUnfolder

    registries.unfolders.clear()
    definition = GravelUnfolder.definition()
    registries.unfolders.register(
        definition.key,
        GravelUnfolder(),
        description=definition.summary,
        tags=("unfolding", "native", definition.key),
    )
    maxed_definition = MaxedUnfolder.definition()
    registries.unfolders.register(
        maxed_definition.key,
        MaxedUnfolder(),
        description=maxed_definition.summary,
        tags=("unfolding", "native", maxed_definition.key),
    )
    rmle_definition = RMLEUnfolder.definition()
    registries.unfolders.register(
        rmle_definition.key,
        RMLEUnfolder(),
        description=rmle_definition.summary,
        recommended=True,
        tags=("unfolding", "native", rmle_definition.key),
        set_default=True,
    )
    ml_seed_definition = MLSeedUnfolder.definition()
    registries.unfolders.register(
        ml_seed_definition.key,
        MLSeedUnfolder(),
        description=ml_seed_definition.summary,
        tags=("unfolding", "native", ml_seed_definition.key),
    )
    return registries


def unfolding_entries(registries: PluginRegistries | None = None):
    """Return built-in unfolding registry entries."""

    shared = bootstrap_builtin_registries(registries)
    if len(shared.unfolders) == 0:
        register_builtin_unfolders(shared)
    return shared.unfolders.entries()


__all__ = [
    "GravelUnfolder",
    "register_builtin_unfolders",
    "unfolding_entries",
]
