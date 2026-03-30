"""Registry-backed RMLE unfolding method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxforge.solvers.rmle import (
    ParameterSelection,
    PoissonPenalty,
    PoissonRMLEConfig,
    RegularizationType,
    ResponseMatrix,
    SpectrumData,
    poisson_rmle_unfolding,
    rmle_unfolding,
)
from fluxforge.unfolding.base import (
    UnfoldingMethod,
    UnfoldingMethodDefinition,
    UnfoldingResult,
    validate_unfolding_inputs,
)


_REGULARIZATION_MAP = {
    "l2": RegularizationType.TIKHONOV,
    "first_derivative": RegularizationType.TIKHONOV_DERIVATIVE,
    "second_derivative": RegularizationType.TIKHONOV_SECOND,
}

_POISSON_PENALTY_MAP = {
    "l2": PoissonPenalty.L2,
    "first_derivative": PoissonPenalty.SOBLEV_1,
    "second_derivative": PoissonPenalty.SOBLEV_2,
}


@dataclass(frozen=True)
class RMLEUnfolder(UnfoldingMethod):
    """Regularized maximum-likelihood unfolding with automatic lambda support."""

    max_iterations: int = 1000
    tolerance: float = 1e-6
    regularization_strength: float = 1.0
    regularization_type: str = "second_derivative"
    auto_regularization: bool = True
    enforce_positivity: bool = True

    @classmethod
    def definition(cls) -> UnfoldingMethodDefinition:
        return UnfoldingMethodDefinition(
            key="rmle",
            label="RMLE",
            summary="Regularized maximum-likelihood unfolding with automatic lambda selection and visible uncertainty bands.",
            convergence_metric="chi_squared",
            supports_uncertainties=True,
            can_produce_negative_bins=False,
            method_category="recommended_default",
        )

    def unfold(
        self,
        measured: np.ndarray,
        response_matrix: np.ndarray,
        **kwargs,
    ) -> UnfoldingResult:
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
        regularization_strength = float(
            kwargs.get("regularization_strength", self.regularization_strength)
        )
        regularization_key = str(
            kwargs.get("regularization_type", self.regularization_type)
        ).strip().lower()
        auto_regularization = bool(
            kwargs.get("auto_regularization", self.auto_regularization)
        )
        enforce_positivity = bool(
            kwargs.get("enforce_positivity", self.enforce_positivity)
        )
        seed_with_ml = bool(kwargs.get("seed_with_ml", False))
        confidence_threshold = float(kwargs.get("confidence_threshold", 0.6))
        regularization = _REGULARIZATION_MAP.get(
            regularization_key,
            RegularizationType.TIKHONOV_SECOND,
        )
        poisson_penalty = _POISSON_PENALTY_MAP.get(
            regularization_key,
            PoissonPenalty.SOBLEV_2,
        )
        param_selection = (
            ParameterSelection.AUTOMATIC
            if auto_regularization
            else ParameterSelection.FIXED
        )

        spectrum = SpectrumData(
            counts=measured_array,
            uncertainty=(
                uncertainty_array
                if uncertainty_array is not None
                else np.sqrt(np.maximum(measured_array, 1.0))
            ),
        )
        response = ResponseMatrix(matrix=response_array)
        seed_result = None
        seeded_initial_flux = initial_flux
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
                seeded_initial_flux = np.asarray(seed_result.flux, dtype=float)

        resolved_regularization_strength = regularization_strength
        if auto_regularization:
            lambda_probe = rmle_unfolding(
                spectrum=spectrum,
                response=response,
                regularization=regularization,
                reg_param=regularization_strength,
                param_selection=param_selection,
                max_iterations=max_iterations,
                tolerance=tolerance,
                enforce_positivity=enforce_positivity,
            )
            resolved_regularization_strength = max(
                float(lambda_probe.regularization_param),
                1e-6,
            )

        solution = poisson_rmle_unfolding(
            spectrum=spectrum,
            response=response,
            config=PoissonRMLEConfig(
                penalty=poisson_penalty,
                alpha=resolved_regularization_strength,
                background_mode="none",
                max_iterations=max_iterations,
                tolerance=tolerance,
                positivity=enforce_positivity,
                initial_solution=seeded_initial_flux,
            ),
        )
        predicted_measurements = response_array @ np.asarray(solution.solution, dtype=float)
        refold_error = float(
            np.linalg.norm(predicted_measurements - measured_array)
            / max(np.linalg.norm(measured_array), 1e-12)
        )
        auto_regularization_refined = False
        if auto_regularization and refold_error > 0.01:
            for _ in range(6):
                resolved_regularization_strength = max(
                    resolved_regularization_strength * 0.1,
                    1e-8,
                )
                solution = poisson_rmle_unfolding(
                    spectrum=spectrum,
                    response=response,
                    config=PoissonRMLEConfig(
                        penalty=poisson_penalty,
                        alpha=resolved_regularization_strength,
                        background_mode="none",
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                        positivity=enforce_positivity,
                        initial_solution=seeded_initial_flux,
                    ),
                )
                predicted_measurements = response_array @ np.asarray(
                    solution.solution,
                    dtype=float,
                )
                refold_error = float(
                    np.linalg.norm(predicted_measurements - measured_array)
                    / max(np.linalg.norm(measured_array), 1e-12)
                )
                auto_regularization_refined = True
                if refold_error <= 0.01:
                    break

        flux = np.asarray(solution.solution, dtype=float)
        uncertainties = np.asarray(solution.uncertainty, dtype=float)
        predicted_measurements = response_array @ flux
        residuals = (
            np.asarray(solution.residuals, dtype=float)
            if solution.residuals is not None
            else measured_array - predicted_measurements
        )

        return UnfoldingResult(
            flux=flux,
            uncertainties=uncertainties,
            convergence_history=(float(solution.reduced_chi_squared),),
            method_used=self.definition().label,
            parameters_used={
                "max_iterations": max_iterations,
                "tolerance": tolerance,
                "regularization_type": poisson_penalty.value,
                "regularization_strength": float(solution.regularization_param),
                "auto_regularization": auto_regularization,
                "auto_regularization_refined": auto_regularization_refined,
                "enforce_positivity": enforce_positivity,
                "rmle_backend": str(
                    solution.diagnostics.get("solver", "poisson_rmle")
                ),
                "used_measurement_uncertainty": uncertainty_array is not None,
                "used_initial_flux": seeded_initial_flux is not None,
                "parameter_selection": param_selection.value,
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
            method_category=self.definition().method_category,
            predicted_measurements=predicted_measurements,
            residuals=residuals,
            chi_squared=float(solution.reduced_chi_squared),
            iterations=int(solution.n_iterations),
            converged=bool(solution.converged),
        )


__all__ = ["RMLEUnfolder"]
