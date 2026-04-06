"""Core abstractions for the unfolding package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fluxforge.core.unfolding_diagnostics import summarize_flux_bins
from fluxforge.core.unfolding_inputs import require_nonnegative


@dataclass(frozen=True)
class UnfoldingMethodDefinition:
    """Static method definition surfaced through the unfolding registry."""

    key: str
    label: str
    summary: str
    convergence_metric: str
    supports_uncertainties: bool = False
    can_produce_negative_bins: bool = False
    method_category: str = "user_selected"


@dataclass
class UnfoldingResult:
    """Normalized unfolding result returned by registry-backed methods."""

    flux: np.ndarray
    uncertainties: np.ndarray | None = None
    convergence_history: tuple[float, ...] = ()
    method_used: str = ""
    parameters_used: dict[str, Any] = field(default_factory=dict)
    negative_bin_count: int = 0
    method_category: str = "user_selected"
    standards_locked_by: str | None = None
    predicted_measurements: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    residuals: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    chi_squared: float = 0.0
    iterations: int = 0
    converged: bool = False

    def __post_init__(self) -> None:
        self.flux = np.asarray(self.flux, dtype=float).reshape(-1)
        if self.uncertainties is not None:
            self.uncertainties = np.asarray(self.uncertainties, dtype=float).reshape(-1)
        self.predicted_measurements = np.asarray(
            self.predicted_measurements,
            dtype=float,
        ).reshape(-1)
        self.residuals = np.asarray(self.residuals, dtype=float).reshape(-1)
        self.convergence_history = tuple(float(value) for value in self.convergence_history)
        summary = summarize_flux_bins(self.flux)
        self.negative_bin_count = int(summary["negative_bin_count"])
        self.iterations = int(self.iterations)
        self.converged = bool(self.converged)
        self.chi_squared = float(self.chi_squared)


class UnfoldingMethod(ABC):
    """Abstract unfolding method contract used by the plugin registry."""

    @classmethod
    @abstractmethod
    def definition(cls) -> UnfoldingMethodDefinition:
        """Return the static method definition for registry/display use."""

    @abstractmethod
    def unfold(
        self,
        measured: np.ndarray,
        response_matrix: np.ndarray,
        **kwargs: Any,
    ) -> UnfoldingResult:
        """Unfold *measured* with *response_matrix* and return normalized results."""


def validate_unfolding_inputs(
    measured: np.ndarray,
    response_matrix: np.ndarray,
    *,
    initial_flux: np.ndarray | None = None,
    measurement_uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate and normalize common unfolding inputs."""

    measured_array = require_nonnegative("measurements", measured).reshape(-1)
    response_array = require_nonnegative("response", response_matrix)
    if response_array.ndim != 2:
        raise ValueError("response_matrix must be a 2-D array")
    if response_array.shape[0] != measured_array.size:
        raise ValueError(
            "response_matrix row count must match the measured spectrum length"
        )

    initial_array = None
    if initial_flux is not None:
        initial_array = require_nonnegative("initial_flux", initial_flux).reshape(-1)
        if initial_array.size != response_array.shape[1]:
            raise ValueError(
                "initial_flux length must match the response_matrix column count"
            )

    uncertainty_array = None
    if measurement_uncertainty is not None:
        uncertainty_array = require_nonnegative(
            "measurement_uncertainty",
            measurement_uncertainty,
        ).reshape(-1)
        if uncertainty_array.size != measured_array.size:
            raise ValueError(
                "measurement_uncertainty length must match the measured spectrum length"
            )

    return measured_array, response_array, initial_array, uncertainty_array


def estimate_unfolding_uncertainties(
    response_matrix: np.ndarray,
    *,
    measured: np.ndarray | None = None,
    measurement_uncertainty: np.ndarray | None = None,
) -> np.ndarray:
    """Return a stable pseudo-inverse uncertainty estimate for unfolded flux bins."""

    response_array = np.asarray(response_matrix, dtype=float)
    if response_array.ndim != 2:
        raise ValueError("response_matrix must be a 2-D array")

    if measurement_uncertainty is not None:
        sigma = require_nonnegative(
            "measurement_uncertainty",
            measurement_uncertainty,
        ).reshape(-1)
    elif measured is not None:
        sigma = np.sqrt(
            np.maximum(require_nonnegative("measurements", measured).reshape(-1), 1.0)
        )
    else:
        sigma = np.ones(response_array.shape[0], dtype=float)

    if sigma.size != response_array.shape[0]:
        raise ValueError(
            "measurement_uncertainty length must match the response_matrix row count"
        )

    try:
        rt_r = response_array.T @ response_array
        scale = max(float(np.trace(rt_r)), 1.0)
        regularization = (1e-10 * scale / max(rt_r.shape[0], 1)) * np.eye(
            rt_r.shape[0],
            dtype=float,
        )
        sensitivity = np.linalg.inv(rt_r + regularization) @ response_array.T
        variance = np.sum((sensitivity * sigma.reshape(1, -1)) ** 2, axis=1)
    except np.linalg.LinAlgError:
        pseudo_inverse = np.linalg.pinv(response_array)
        variance = np.sum((pseudo_inverse * sigma.reshape(1, -1)) ** 2, axis=1)

    return np.sqrt(np.maximum(variance, 0.0))


__all__ = [
    "UnfoldingMethod",
    "UnfoldingMethodDefinition",
    "UnfoldingResult",
    "estimate_unfolding_uncertainties",
    "validate_unfolding_inputs",
]
