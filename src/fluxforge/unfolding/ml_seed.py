"""Fast seed-generation unfolder used as a standalone option or warm start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxforge.solvers.rmle import tikhonov_matrix
from fluxforge.unfolding.base import (
    estimate_unfolding_uncertainties,
    UnfoldingMethod,
    UnfoldingMethodDefinition,
    UnfoldingResult,
    validate_unfolding_inputs,
)


def _second_difference_operator(n_groups: int) -> np.ndarray:
    """Return a stable smoothing operator for the current group count."""

    if n_groups <= 2:
        return np.eye(n_groups, dtype=float)
    return np.asarray(tikhonov_matrix(n_groups, order=2), dtype=float)


@dataclass(frozen=True)
class MLSeedUnfolder(UnfoldingMethod):
    """Fast approximate seed model for unfolding workflows."""

    regularization_strength: float = 0.05
    smoothing_strength: float = 0.02
    warm_start_iterations: int = 8
    confidence_threshold: float = 0.6
    floor: float = 1e-12

    @classmethod
    def definition(cls) -> UnfoldingMethodDefinition:
        return UnfoldingMethodDefinition(
            key="ml_seed",
            label="ML Seed",
            summary="Fast seed approximation for standalone review or warm-starting GRAVEL/RMLE.",
            convergence_metric="refold_error",
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
        measured_array, response_array, initial_flux, uncertainty_array = (
            validate_unfolding_inputs(
                measured,
                response_matrix,
                initial_flux=kwargs.get("initial_flux"),
                measurement_uncertainty=kwargs.get("measurement_uncertainty"),
            )
        )

        regularization_strength = float(
            kwargs.get("regularization_strength", self.regularization_strength)
        )
        smoothing_strength = float(
            kwargs.get("smoothing_strength", self.smoothing_strength)
        )
        warm_start_iterations = int(
            kwargs.get("warm_start_iterations", self.warm_start_iterations)
        )
        confidence_threshold = float(
            kwargs.get("confidence_threshold", self.confidence_threshold)
        )
        floor = float(kwargs.get("floor", self.floor))

        n_groups = response_array.shape[1]
        if initial_flux is not None:
            prior = np.maximum(np.asarray(initial_flux, dtype=float), floor)
        else:
            mean_response = max(float(np.mean(response_array)), floor)
            mean_measurement = max(float(np.mean(measured_array)), 1.0)
            prior = np.full(
                n_groups,
                mean_measurement / max(mean_response * n_groups, floor),
                dtype=float,
            )
            prior = np.maximum(prior, floor)

        sigma = (
            np.asarray(uncertainty_array, dtype=float)
            if uncertainty_array is not None
            else np.sqrt(np.maximum(measured_array, 1.0))
        )
        sigma = np.maximum(sigma, floor)
        weighted_response = response_array / sigma.reshape(-1, 1)
        weighted_measurements = measured_array / sigma
        smoothing = _second_difference_operator(n_groups)

        lhs = weighted_response.T @ weighted_response
        lhs = lhs + regularization_strength * np.eye(n_groups, dtype=float)
        lhs = lhs + smoothing_strength * (smoothing.T @ smoothing)
        rhs = weighted_response.T @ weighted_measurements
        rhs = rhs + regularization_strength * prior

        try:
            flux = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            flux = np.linalg.pinv(lhs) @ rhs
        flux = np.maximum(np.asarray(flux, dtype=float), floor)

        convergence_history: list[float] = []
        column_sums = np.maximum(np.sum(response_array, axis=0), floor)
        for _ in range(max(warm_start_iterations, 0)):
            predicted = response_array @ flux
            ratio = measured_array / np.maximum(predicted, floor)
            correction = response_array.T @ ratio
            multiplicative = np.maximum(correction / column_sums, floor)
            flux = np.maximum(flux * np.power(multiplicative, 0.5), floor)
            refold_error = float(
                np.linalg.norm((response_array @ flux) - measured_array)
                / max(np.linalg.norm(measured_array), floor)
            )
            convergence_history.append(refold_error)

        predicted_measurements = response_array @ flux
        residuals = measured_array - predicted_measurements
        chi_squared = float(
            np.dot(residuals / sigma, residuals / sigma) / max(measured_array.size - 1, 1)
        )
        refold_error = float(
            np.linalg.norm(residuals) / max(np.linalg.norm(measured_array), floor)
        )
        confidence_score = float(
            np.clip(
                1.0
                / (
                    1.0
                    + (4.0 * refold_error)
                    + (0.5 * max(chi_squared - 1.0, 0.0))
                ),
                0.0,
                1.0,
            )
        )
        accepted = confidence_score >= confidence_threshold
        uncertainties = estimate_unfolding_uncertainties(
            response_array,
            measured=measured_array,
            measurement_uncertainty=uncertainty_array,
        )

        return UnfoldingResult(
            flux=flux,
            uncertainties=uncertainties,
            convergence_history=tuple(convergence_history),
            method_used=self.definition().label,
            parameters_used={
                "backend": "deterministic_seed_surrogate",
                "regularization_strength": regularization_strength,
                "smoothing_strength": smoothing_strength,
                "warm_start_iterations": warm_start_iterations,
                "confidence_score": confidence_score,
                "confidence_threshold": confidence_threshold,
                "accepted": accepted,
                "used_initial_flux": initial_flux is not None,
                "used_measurement_uncertainty": uncertainty_array is not None,
                "uncertainty_estimator": "pseudo_inverse",
                "refold_error": refold_error,
            },
            method_category=self.definition().method_category,
            predicted_measurements=predicted_measurements,
            residuals=residuals,
            chi_squared=chi_squared,
            iterations=max(warm_start_iterations, 1),
            converged=accepted,
        )


__all__ = ["MLSeedUnfolder"]
