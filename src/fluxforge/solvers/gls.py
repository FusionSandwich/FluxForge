"""
Generalized Least-Squares (GLS) Spectrum Adjustment.

Implements STAYSL-style spectral adjustment following the GLS formalism.

The GLS adjustment solves the overdetermined system:
    y = R φ + ε

where:
    y = measured reaction rates (N_reactions)
    R = response matrix (N_reactions × N_groups)
    φ = neutron flux spectrum to be adjusted
    ε = measurement error

With prior information:
    φ₀ = prior (calculated) spectrum
    V_φ = prior spectrum covariance
    V_y = measurement covariance

The posterior estimate is:
    φ̂ = φ₀ + V_φ R^T (R V_φ R^T + V_y)^(-1) (y - R φ₀)

With posterior covariance:
    V_φ̂ = V_φ - V_φ R^T (R V_φ R^T + V_y)^(-1) R V_φ

Optional response covariance treatment:
    - Augment V_y with propagated response uncertainty
    - Monte Carlo propagation for full treatment
    - Nuisance parameter approach

References
----------
- ASTM E944: Standard Guide for Application of Neutron Spectrum Adjustment Methods
- STAYSL PNNL User Manual (PNNL-22253)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np

from fluxforge.core.unfolding_diagnostics import merge_flux_diagnostics
from fluxforge.core.unfolding_inputs import (
    require_covariance_matrix,
    require_finite,
    require_nonnegative,
)
from fluxforge.core.linalg import (
    Matrix,
    Vector,
    add_vectors,
    elementwise_clip,
    matmul,
    sub_vectors,
    transpose,
)


class ResponseCovariancePolicy(Enum):
    """Policy for handling response matrix uncertainty."""

    IGNORE = "ignore"  # Ignore response uncertainty (default)
    AUGMENT_VY = (
        "augment_vy"  # Augment measurement covariance with response contribution
    )
    NUISANCE = "nuisance"  # Treat response as nuisance parameters
    MONTE_CARLO = "monte_carlo"  # Full MC propagation


@dataclass
class GLSSolution:
    """
    Result of GLS spectrum adjustment.

    Attributes:
        flux: Adjusted (posterior) neutron flux spectrum
        covariance: Posterior flux covariance matrix
        residuals: (y - R φ₀) pre-adjustment residuals
        chi2: Chi-squared statistic
        reduced_chi2: Chi-squared per degree of freedom
        n_dof: Degrees of freedom
        pull: Normalized residuals (residuals / sqrt(diag(V_innovation)))
        influence: Diagonal of the hat matrix (leverage)
        prior_posterior_change: Relative change from prior to posterior
    """

    flux: Vector
    covariance: Matrix
    residuals: Vector
    chi2: float
    reduced_chi2: float = 0.0
    n_dof: int = 0
    pull: Optional[Vector] = None
    influence: Optional[Vector] = None
    prior_posterior_change: Optional[Vector] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.diagnostics = merge_flux_diagnostics(self.diagnostics, self.flux)

    @property
    def flux_uncertainty(self) -> Vector:
        """Standard deviation of posterior flux."""
        import math

        return [math.sqrt(max(self.covariance[i][i], 0)) for i in range(len(self.flux))]


@dataclass
class GLSConfig:
    """
    Configuration for GLS adjustment.

    Attributes:
        enforce_nonnegativity: Clip negative flux values to zero
        response_cov_policy: How to handle response uncertainty
        response_cov: Response covariance matrix (if provided)
        n_mc_samples: Number of MC samples (if using MC policy)
        conditioning_threshold: Threshold for matrix conditioning
        compute_diagnostics: Whether to compute pull, influence, etc.
    """

    enforce_nonnegativity: bool = True
    response_cov_policy: ResponseCovariancePolicy = ResponseCovariancePolicy.IGNORE
    response_cov: Optional[Matrix] = None
    n_mc_samples: int = 100
    conditioning_threshold: float = 1e-10
    compute_diagnostics: bool = True


def _quadratic_form(vec: Vector, mat: Matrix) -> float:
    """Compute v^T M v."""
    total = 0.0
    for i, v_i in enumerate(vec):
        total += v_i * sum(mat[i][j] * vec[j] for j in range(len(vec)))
    return total


def _diagonal(mat: Matrix) -> Vector:
    """Extract diagonal of matrix."""
    return [mat[i][i] for i in range(len(mat))]


def _compute_pull(residuals: Vector, cov_diag: Vector) -> Vector:
    """Compute normalized residuals (pull)."""
    import math

    pull = []
    for r, v in zip(residuals, cov_diag):
        if v > 0:
            pull.append(r / math.sqrt(v))
        else:
            pull.append(0.0)
    return pull


def _augment_measurement_cov(
    measurement_cov: Matrix,
    response: Matrix,
    prior_flux: Vector,
    response_cov: Matrix,
) -> Matrix:
    """
    Augment measurement covariance with response uncertainty contribution.

    V_y_aug = V_y + diag(R σ_R × φ₀)²

    This is a linearized approximation of response uncertainty propagation.
    """
    import math

    n_reactions = len(measurement_cov)
    n_groups = len(prior_flux)

    # Create augmented covariance
    augmented = [
        [measurement_cov[i][j] for j in range(n_reactions)] for i in range(n_reactions)
    ]

    # For each reaction, compute response uncertainty contribution
    for i in range(n_reactions):
        response_var = 0.0
        for g in range(n_groups):
            # Response covariance indexed by reaction and group
            if len(response_cov) > i and len(response_cov[i]) > g:
                sigma_R_ig = math.sqrt(max(response_cov[i][g], 0))
            else:
                sigma_R_ig = 0.0
            response_var += (sigma_R_ig * prior_flux[g]) ** 2
        augmented[i][i] += response_var

    return augmented


def gls_adjust(
    response: Matrix,
    measurements: Vector,
    measurement_cov: Matrix,
    prior_flux: Vector,
    prior_cov: Matrix,
    enforce_nonnegativity: bool = True,
    response_cov: Optional[Matrix] = None,
    response_cov_policy: ResponseCovariancePolicy = ResponseCovariancePolicy.IGNORE,
    compute_diagnostics: bool = True,
) -> GLSSolution:
    """
    Perform GLS spectral adjustment (STAYSL-style).

    Parameters
    ----------
    response : Matrix
        Response matrix R[i,g] (N_reactions × N_groups)
    measurements : Vector
        Measured reaction rates y (N_reactions)
    measurement_cov : Matrix
        Measurement covariance V_y (N_reactions × N_reactions)
    prior_flux : Vector
        Prior (calculated) spectrum φ₀ (N_groups)
    prior_cov : Matrix
        Prior spectrum covariance V_φ (N_groups × N_groups)
    enforce_nonnegativity : bool
        If True, clip negative flux values to zero
    response_cov : Matrix, optional
        Response uncertainty matrix (N_reactions × N_groups) for variance
    response_cov_policy : ResponseCovariancePolicy
        How to handle response uncertainty

    Returns
    -------
    GLSSolution
        Adjusted spectrum with covariance and diagnostics
    """
    import math

    if not isinstance(response_cov_policy, ResponseCovariancePolicy):
        response_cov_policy = ResponseCovariancePolicy(response_cov_policy)
    if response_cov_policy in (
        ResponseCovariancePolicy.NUISANCE,
        ResponseCovariancePolicy.MONTE_CARLO,
    ):
        raise NotImplementedError(
            "This entrypoint supports IGNORE and AUGMENT_VY only; use "
            "gls_adjust_with_response_cov for Monte Carlo propagation."
        )
    if (
        response_cov_policy == ResponseCovariancePolicy.AUGMENT_VY
        and response_cov is None
    ):
        raise ValueError("AUGMENT_VY requires response_cov element variances")

    response_array = require_nonnegative("response", response)
    if response_array.ndim != 2:
        raise ValueError("response must be a 2-D array")
    measurements_array = require_finite("measurements", measurements).reshape(-1)
    measurement_cov_array = require_covariance_matrix(
        "measurement_cov",
        measurement_cov,
    )
    prior_flux_array = require_nonnegative("prior_flux", prior_flux).reshape(-1)
    prior_cov_array = require_covariance_matrix("prior_cov", prior_cov)

    if response_array.shape != (measurements_array.size, prior_flux_array.size):
        raise ValueError(
            f"Response matrix shape ({response_array.shape[0]}×{response_array.shape[1]}) "
            f"doesn't match measurements ({measurements_array.size}) and prior ({prior_flux_array.size})"
        )
    if measurement_cov_array.shape != (
        measurements_array.size,
        measurements_array.size,
    ):
        raise ValueError("measurement_cov shape must match the measurements length")
    if prior_cov_array.shape != (prior_flux_array.size, prior_flux_array.size):
        raise ValueError("prior_cov shape must match the prior_flux length")

    response = response_array.astype(float).tolist()
    measurements = measurements_array.astype(float).tolist()
    measurement_cov = measurement_cov_array.astype(float).tolist()
    prior_flux = prior_flux_array.astype(float).tolist()
    prior_cov = prior_cov_array.astype(float).tolist()
    if response_cov is not None:
        response_cov_array = require_nonnegative("response_cov", response_cov)
        if response_cov_array.shape != response_array.shape:
            raise ValueError("response_cov shape must match the response shape")
        response_cov = response_cov_array.astype(float).tolist()

    n_reactions = len(measurements)
    n_groups = len(prior_flux)

    # Handle response covariance if provided
    working_measurement_cov = measurement_cov
    if (
        response_cov is not None
        and response_cov_policy == ResponseCovariancePolicy.AUGMENT_VY
    ):
        working_measurement_cov = _augment_measurement_cov(
            measurement_cov, response, prior_flux, response_cov
        )

    # Compute R × V_φ
    rc0 = matmul(response, prior_cov)  # type: ignore[arg-type]

    # Innovation covariance: R V_φ R^T + V_y
    innovation_cov = matmul(rc0, transpose(response))  # type: ignore[arg-type]
    for i in range(len(innovation_cov)):
        for j in range(len(innovation_cov)):
            innovation_cov[i][j] += working_measurement_cov[i][j]

    # Invert innovation covariance
    innovation_array = np.asarray(innovation_cov, dtype=float)
    innovation_cov_inv = np.linalg.pinv(innovation_array, hermitian=True).tolist()

    # Kalman gain: V_φ R^T (R V_φ R^T + V_y)^(-1)
    gain = matmul(prior_cov, matmul(transpose(response), innovation_cov_inv))  # type: ignore[arg-type]

    # Model prediction: R φ₀
    model_prediction = matmul(response, prior_flux)  # type: ignore[arg-type]

    # Innovation (residuals): y - R φ₀
    residuals = sub_vectors(measurements, model_prediction)  # type: ignore[arg-type]
    projected_residual = (
        innovation_array @ np.asarray(innovation_cov_inv) @ np.asarray(residuals)
    )
    if not np.allclose(
        projected_residual,
        residuals,
        rtol=1e-9,
        atol=1e-12 * max(1.0, np.linalg.norm(residuals)),
    ):
        raise ValueError("Measurements contradict a zero-variance constraint")

    # Update: V_φ R^T (R V_φ R^T + V_y)^(-1) (y - R φ₀)
    update = matmul(gain, residuals)  # type: ignore[arg-type]

    # Posterior flux: φ̂ = φ₀ + update
    phi_hat = add_vectors(prior_flux, update)
    clipped_bins = [i for i, value in enumerate(phi_hat) if value < 0.0]
    if enforce_nonnegativity:
        phi_hat = elementwise_clip(phi_hat, 0.0)

    # Posterior covariance: V_φ̂ = V_φ - V_φ R^T (R V_φ R^T + V_y)^(-1) R V_φ
    posterior_cov = matmul(prior_cov, matmul(transpose(response), matmul(innovation_cov_inv, rc0)))  # type: ignore[arg-type]
    for i in range(len(prior_cov)):
        for j in range(len(prior_cov)):
            posterior_cov[i][j] = prior_cov[i][j] - posterior_cov[i][j]

    # Chi-squared: (y - R φ₀)^T (R V_φ R^T + V_y)^(-1) (y - R φ₀)
    chi2 = _quadratic_form(residuals, innovation_cov_inv)

    # Degrees of freedom
    # Prior-predictive innovation statistic, not a post-fit residual statistic.
    n_dof = int(np.linalg.matrix_rank(innovation_array))
    reduced_chi2 = chi2 / n_dof if n_dof > 0 else float("nan")

    # Diagnostics
    pull = None
    influence = None
    prior_posterior_change = None
    diagnostics: Dict[str, Any] = {}

    if compute_diagnostics:
        # Pull (normalized residuals)
        innovation_diag = _diagonal(innovation_cov)
        pull = _compute_pull(residuals, innovation_diag)

        # Prior to posterior relative change
        prior_posterior_change = [
            (phi_hat[g] - prior_flux[g]) / max(prior_flux[g], 1e-20)
            for g in range(n_groups)
        ]

        # Measurement-space leverage: diagonal of R K, one value per monitor.
        influence = np.diag(response_array @ np.asarray(gain)).tolist()

        diagnostics = {
            "n_reactions": n_reactions,
            "n_groups": n_groups,
            "condition_number": _estimate_condition(innovation_cov),
            "max_pull": max(abs(p) for p in pull) if pull else 0.0,
            "max_relative_change": max(abs(c) for c in prior_posterior_change),
            "response_cov_policy": response_cov_policy.value,
        }

    diagnostics.update(
        {
            "chi2_convention": "prior_predictive_innovation",
            "degrees_of_freedom_convention": "rank_of_innovation_covariance",
            "posterior_residuals": (
                measurements_array - response_array @ np.asarray(phi_hat)
            ).tolist(),
            "covariance_convention": "unconstrained_linear_gaussian_posterior",
            "clipped_bin_indices": clipped_bins if enforce_nonnegativity else [],
            "constrained_posterior_valid": not (enforce_nonnegativity and clipped_bins),
            "response_uncertainty_approximation": (
                "independent_element_variances_at_prior_flux"
                if response_cov_policy == ResponseCovariancePolicy.AUGMENT_VY
                else "ignored"
            ),
        }
    )

    diagnostics = merge_flux_diagnostics(
        diagnostics,
        phi_hat,
        negative_policy=(
            "clipped_to_zero" if enforce_nonnegativity else "preserved_for_review"
        ),
        nonnegativity_enforced=bool(enforce_nonnegativity),
    )

    return GLSSolution(
        flux=phi_hat,
        covariance=posterior_cov,
        residuals=residuals,
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        n_dof=n_dof,
        pull=pull,
        influence=influence,
        prior_posterior_change=prior_posterior_change,
        diagnostics=diagnostics,
    )


def _estimate_condition(mat: Matrix) -> float:
    """Return the spectral condition number, including off-diagonal coupling."""
    return float(np.linalg.cond(np.asarray(mat, dtype=float)))


def gls_adjust_with_response_cov(
    response: Matrix,
    response_cov: Matrix,
    measurements: Vector,
    measurement_cov: Matrix,
    prior_flux: Vector,
    prior_cov: Matrix,
    n_samples: int = 100,
    enforce_nonnegativity: bool = True,
) -> GLSSolution:
    """
    GLS adjustment with Monte Carlo propagation of response uncertainty.

    This performs multiple GLS adjustments with perturbed response matrices
    to propagate response uncertainty into the final result.

    Parameters
    ----------
    response : Matrix
        Nominal response matrix
    response_cov : Matrix
        Response uncertainty (std dev for each element)
    measurements : Vector
        Measured reaction rates
    measurement_cov : Matrix
        Measurement covariance
    prior_flux : Vector
        Prior spectrum
    prior_cov : Matrix
        Prior covariance
    n_samples : int
        Number of MC samples
    enforce_nonnegativity : bool
        Clip negative values

    Returns
    -------
    GLSSolution
        Adjusted spectrum with MC-propagated uncertainty
    """
    import random
    import math

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2")

    response_array = require_nonnegative("response", response)
    if response_array.ndim != 2:
        raise ValueError("response must be a 2-D array")
    response_cov_array = require_nonnegative("response_cov", response_cov)
    if response_cov_array.shape != response_array.shape:
        raise ValueError("response_cov shape must match the response shape")
    measurements_array = require_finite("measurements", measurements).reshape(-1)
    measurement_cov_array = require_covariance_matrix(
        "measurement_cov",
        measurement_cov,
    )
    prior_flux_array = require_nonnegative("prior_flux", prior_flux).reshape(-1)
    prior_cov_array = require_covariance_matrix("prior_cov", prior_cov)

    response = response_array.astype(float).tolist()
    response_cov = response_cov_array.astype(float).tolist()
    measurements = measurements_array.astype(float).tolist()
    measurement_cov = measurement_cov_array.astype(float).tolist()
    prior_flux = prior_flux_array.astype(float).tolist()
    prior_cov = prior_cov_array.astype(float).tolist()

    n_reactions = len(response)
    n_groups = len(response[0])

    # Collect MC samples of posterior flux
    flux_samples = []
    conditional_covariances = []
    clipped_sample_count = 0

    for _ in range(n_samples):
        # Perturb response matrix
        perturbed_R = [
            [
                max(response[i][g] + random.gauss(0, response_cov[i][g]), 0.0)
                for g in range(n_groups)
            ]
            for i in range(n_reactions)
        ]

        # Run GLS
        result = gls_adjust(
            perturbed_R,
            measurements,
            measurement_cov,
            prior_flux,
            prior_cov,
            enforce_nonnegativity=enforce_nonnegativity,
            compute_diagnostics=False,
        )
        flux_samples.append(result.flux)
        conditional_covariances.append(result.covariance)
        clipped_sample_count += bool(result.diagnostics["clipped_bin_indices"])

    # Compute mean and covariance from samples
    mean_flux = [
        sum(samples[g] for samples in flux_samples) / n_samples for g in range(n_groups)
    ]

    mc_cov = [[0.0 for _ in range(n_groups)] for _ in range(n_groups)]
    for g1 in range(n_groups):
        for g2 in range(n_groups):
            cov_sum = sum(
                (samples[g1] - mean_flux[g1]) * (samples[g2] - mean_flux[g2])
                for samples in flux_samples
            )
            mc_cov[g1][g2] = cov_sum / (n_samples - 1)
            # Total covariance = E[Cov(flux | response)] + Cov(E[flux | response]).
            mc_cov[g1][g2] += (
                sum(cov[g1][g2] for cov in conditional_covariances) / n_samples
            )

    # Run nominal GLS for residuals and chi2
    nominal = gls_adjust(
        response,
        measurements,
        measurement_cov,
        prior_flux,
        prior_cov,
        enforce_nonnegativity=enforce_nonnegativity,
        compute_diagnostics=True,
    )

    # Combine: mean flux, MC covariance, nominal diagnostics
    return GLSSolution(
        flux=mean_flux,
        covariance=mc_cov,
        residuals=nominal.residuals,
        chi2=nominal.chi2,
        reduced_chi2=nominal.reduced_chi2,
        n_dof=nominal.n_dof,
        pull=nominal.pull,
        influence=nominal.influence,
        prior_posterior_change=[
            (mean_flux[g] - prior_flux[g]) / max(prior_flux[g], 1e-20)
            for g in range(n_groups)
        ],
        diagnostics={
            **nominal.diagnostics,
            "response_cov_policy": "monte_carlo",
            "n_mc_samples": n_samples,
            "covariance_convention": "mean_conditional_covariance_plus_between_response_covariance",
            "response_uncertainty_units": "element_standard_deviations",
            "response_uncertainty_approximation": "independent_gaussian_elements_clipped_at_zero",
            "clipped_sample_count": clipped_sample_count,
            "constrained_posterior_valid": clipped_sample_count == 0,
            "diagnostics_reference": "nominal_response",
        },
    )
