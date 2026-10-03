"""Fast seed-generation unfolder used as a standalone option or warm start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxforge.solvers.rmle import tikhonov_matrix
from fluxforge.unfolding.base import (
    unavailable_uncertainty_metadata,
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
            supports_uncertainties=False,
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

        if any(
            not np.isfinite(value) or value < 0
            for value in (regularization_strength, smoothing_strength)
        ):
            raise ValueError(
                "Seed regularization and smoothing strengths must be finite and nonnegative"
            )
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("floor must be finite and positive (in flux units)")
        sigma = (
            np.asarray(uncertainty_array, dtype=float)
            if uncertainty_array is not None
            else np.sqrt(np.maximum(measured_array, 1.0))
        )
        if np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be strictly positive")
        weighted_response = response_array / sigma[:, None]
        weighted_measurements = measured_array / sigma
        if not np.all(np.isfinite(weighted_response)) or not np.all(
            np.isfinite(weighted_measurements)
        ):
            raise ValueError("Whitened seed inputs must be finite")
        n_groups = response_array.shape[1]
        if initial_flux is not None:
            prior = np.maximum(np.asarray(initial_flux, dtype=float), floor)
        else:
            mean_response = float(np.mean(weighted_response))
            if mean_response <= 0:
                raise ValueError("Seed response must have positive sensitivity")
            mean_measurement = float(np.mean(weighted_measurements))
            prior = np.full(
                n_groups,
                mean_measurement / (mean_response * n_groups),
                dtype=float,
            )
            prior = np.maximum(prior, floor)

        smoothing = _second_difference_operator(n_groups)

        # Solve the augmented system without squaring its condition number or
        # overflowing R.T @ R at large but finite weighted sensitivities.
        augmented_response = np.vstack(
            (
                weighted_response,
                np.sqrt(regularization_strength) * np.eye(n_groups),
                np.sqrt(smoothing_strength) * smoothing,
            )
        )
        augmented_measured = np.concatenate(
            (
                weighted_measurements,
                np.sqrt(regularization_strength) * prior,
                np.zeros(smoothing.shape[0]),
            )
        )
        flux = np.linalg.lstsq(augmented_response, augmented_measured, rcond=None)[0]
        if not np.all(np.isfinite(flux)):
            raise ValueError("Seed solution must be finite")
        flux = np.maximum(np.asarray(flux, dtype=float), floor)

        convergence_history: list[float] = []
        column_sums = np.sum(weighted_response, axis=0)
        supported = column_sums > 0
        weighted_norm = np.hypot.reduce(weighted_measurements)
        for _ in range(max(warm_start_iterations, 0)):
            predicted = weighted_response @ flux
            ratio = np.divide(
                weighted_measurements,
                predicted,
                out=np.zeros_like(predicted),
                where=predicted > 0,
            )
            correction = weighted_response.T @ ratio
            multiplicative = np.divide(
                correction, column_sums, out=np.ones_like(column_sums), where=supported
            )
            flux = np.maximum(flux * np.power(multiplicative, 0.5), floor)
            refold_error = float(
                np.hypot.reduce((weighted_response @ flux) - weighted_measurements)
                / (weighted_norm if weighted_norm > 0 else 1.0)
            )
            convergence_history.append(refold_error)

        predicted_measurements = response_array @ flux
        residuals = measured_array - predicted_measurements
        chi_squared = float(np.dot(residuals / sigma, residuals / sigma))
        if not np.isfinite(chi_squared):
            raise ValueError(
                "Seed residual statistic is not finite at this numerical scale"
            )
        refold_error = float(
            np.hypot.reduce(residuals / sigma)
            / (weighted_norm if weighted_norm > 0 else 1.0)
        )
        confidence_score = float(
            np.clip(
                1.0
                / (
                    1.0
                    + (4.0 * refold_error)
                    + (0.5 * max(chi_squared / measured_array.size - 1.0, 0.0))
                ),
                0.0,
                1.0,
            )
        )
        accepted = confidence_score >= confidence_threshold

        return UnfoldingResult(
            flux=flux,
            uncertainties=None,
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
                **unavailable_uncertainty_metadata(
                    self.definition().label,
                    response_array,
                    converged=accepted,
                    measurement_uncertainty=uncertainty_array,
                ),
                "refold_error": refold_error,
                "refold_error_definition": "norm of error-weighted residual / norm of error-weighted measurements",
                "chi_squared_definition": "sum of squared error-weighted postfit residuals; no degrees-of-freedom correction",
                "confidence_is_heuristic": True,
                "seed_weighting": "measurement_error_whitened",
            },
            method_category=self.definition().method_category,
            predicted_measurements=predicted_measurements,
            residuals=residuals,
            chi_squared=chi_squared,
            iterations=max(warm_start_iterations, 1),
            converged=accepted,
        )


__all__ = ["MLSeedUnfolder"]
