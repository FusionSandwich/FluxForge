"""Registry-backed MAXED unfolding method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from fluxforge.unfolding.base import (
    estimate_unfolding_uncertainties,
    UnfoldingMethod,
    UnfoldingMethodDefinition,
    UnfoldingResult,
    validate_unfolding_inputs,
)


@dataclass(frozen=True)
class MaxedUnfolder(UnfoldingMethod):
    """Maximum-entropy unfolding with positive log-space optimization."""

    max_iterations: int = 400
    entropy_weight: float = 0.02
    floor: float = 1e-100
    optimizer: str = "L-BFGS-B"

    @classmethod
    def definition(cls) -> UnfoldingMethodDefinition:
        return UnfoldingMethodDefinition(
            key="maxed",
            label="MAXED",
            summary="Maximum-entropy unfolding with a positive, prior-guided log-space optimizer.",
            convergence_metric="objective",
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

        max_iterations = int(kwargs.get("max_iterations", self.max_iterations))
        entropy_weight = float(kwargs.get("entropy_weight", self.entropy_weight))
        floor = float(kwargs.get("floor", self.floor))
        optimizer = str(kwargs.get("optimizer", self.optimizer))

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

        objective_history: list[float] = []

        def _objective(log_flux: np.ndarray) -> float:
            flux = np.exp(log_flux)
            predicted = response_array @ flux
            residual = (predicted - measured_array) / sigma
            entropy_penalty = np.sum(
                flux * (np.log(np.maximum(flux, floor)) - np.log(prior)) - flux + prior
            )
            return float(0.5 * np.dot(residual, residual) + entropy_weight * entropy_penalty)

        def _callback(log_flux: np.ndarray) -> None:
            objective_history.append(_objective(log_flux))

        initial_log_flux = np.log(prior)
        objective_history.append(_objective(initial_log_flux))
        optimization = minimize(
            _objective,
            initial_log_flux,
            method=optimizer,
            callback=_callback,
            options={"maxiter": max_iterations},
        )

        flux = np.exp(np.asarray(optimization.x, dtype=float))
        predicted_measurements = response_array @ flux
        residuals = measured_array - predicted_measurements
        chi_squared = float(
            np.dot((predicted_measurements - measured_array) / sigma, (predicted_measurements - measured_array) / sigma)
            / max(measured_array.size - 1, 1)
        )
        uncertainties = estimate_unfolding_uncertainties(
            response_array,
            measured=measured_array,
            measurement_uncertainty=uncertainty_array,
        )

        return UnfoldingResult(
            flux=flux,
            uncertainties=uncertainties,
            convergence_history=tuple(objective_history),
            method_used=self.definition().label,
            parameters_used={
                "max_iterations": max_iterations,
                "entropy_weight": entropy_weight,
                "floor": floor,
                "optimizer": optimizer,
                "used_initial_flux": initial_flux is not None,
                "used_measurement_uncertainty": uncertainty_array is not None,
                "uncertainty_estimator": "pseudo_inverse",
                "success": bool(optimization.success),
                "status": int(optimization.status),
                "message": str(optimization.message),
            },
            method_category=self.definition().method_category,
            predicted_measurements=predicted_measurements,
            residuals=residuals,
            chi_squared=chi_squared,
            iterations=len(objective_history) - 1,
            converged=bool(optimization.success),
        )


__all__ = ["MaxedUnfolder"]
