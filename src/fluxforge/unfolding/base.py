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
    predicted_measurements: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
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
        self.convergence_history = tuple(
            float(value) for value in self.convergence_history
        )
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
    """Conditional sigma for unconstrained, full-rank weighted linear least squares.

    This compatibility utility describes that fixed linear estimator only, not
    GRAVEL, MAXED, a constrained estimator or a regularized seed. Response is
    held fixed and independent positive measurement sigmas must be supplied.
    Rank-deficient bin uncertainty is unavailable rather than zero in null space.
    """

    response_array = np.asarray(response_matrix, dtype=float)
    if (
        response_array.ndim != 2
        or not all(response_array.shape)
        or not np.all(np.isfinite(response_array))
    ):
        raise ValueError("response_matrix must be a nonempty finite 2-D array")

    if measurement_uncertainty is not None:
        sigma = require_nonnegative(
            "measurement_uncertainty",
            measurement_uncertainty,
        ).reshape(-1)
    else:
        raise ValueError(
            "Linear uncertainty requires explicit measurement_uncertainty; no count or unit-variance assumption is inferred"
        )

    if sigma.size != response_array.shape[0]:
        raise ValueError(
            "measurement_uncertainty length must match the response_matrix row count"
        )

    if np.any(sigma <= 0):
        raise ValueError(
            "Linear uncertainty requires strictly positive measurement sigmas; exact constraints are unsupported"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = response_array / sigma[:, None]
    if not np.all(np.isfinite(whitened)):
        raise ValueError("Whitened response is not finite")
    scale = float(np.max(np.abs(whitened)))
    if scale == 0:
        raise ValueError("Rank-deficient response has unavailable bin uncertainty")
    _, singular, vt = np.linalg.svd(whitened / scale, full_matrices=False)
    tolerance = np.finfo(float).eps * max(whitened.shape) * singular[0]
    if len(singular) < response_array.shape[1] or np.any(singular <= tolerance):
        raise ValueError("Rank-deficient response has unavailable bin uncertainty")
    sensitivity_factor = (vt.T / singular) / scale
    result = np.linalg.norm(sensitivity_factor, axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("Linear propagated uncertainty is not finite")
    return result


def unavailable_uncertainty_metadata(
    method, response_matrix, *, converged, measurement_uncertainty=None
):
    """Explain unavailable estimator uncertainty without substituting a proxy."""
    response = np.asarray(response_matrix, dtype=float)
    rank = None
    if measurement_uncertainty is not None:
        sigma = np.asarray(measurement_uncertainty, dtype=float).reshape(-1)
        if (
            sigma.shape == (response.shape[0],)
            and np.all(np.isfinite(sigma))
            and np.all(sigma > 0)
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                weighted = response / sigma[:, None]
            if np.all(np.isfinite(weighted)):
                scale = float(np.max(np.abs(weighted))) if weighted.size else 0.0
                rank = int(np.linalg.matrix_rank(weighted / scale)) if scale else 0
    reasons = [
        f"Estimator-specific uncertainty propagation is not implemented for {method}"
    ]
    if rank is not None and rank < response.shape[1]:
        reasons.append(
            "Response is rank deficient; full-bin identifiability is unqualified"
        )
    elif rank is None:
        reasons.append(
            "Error-weighted response rank is unavailable without finite positive measurement sigmas"
        )
    if not converged:
        reasons.append("Solver did not converge; estimator uncertainty is unqualified")
    return {
        "uncertainty_estimator": "unavailable",
        "uncertainty_status": "unavailable",
        "uncertainty_unavailable_reason": "; ".join(reasons),
        "uncertainty_qualified": False,
        "response_rank": rank,
        "response_rank_basis": (
            "measurement-error-weighted response; numerical rank only"
            if rank is not None
            else "unavailable"
        ),
    }


__all__ = [
    "UnfoldingMethod",
    "UnfoldingMethodDefinition",
    "UnfoldingResult",
    "estimate_unfolding_uncertainties",
    "unavailable_uncertainty_metadata",
    "validate_unfolding_inputs",
]
