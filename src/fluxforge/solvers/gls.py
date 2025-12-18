"""Generalized least-squares spectrum adjustment without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from fluxforge.core.linalg import (
    Matrix,
    Vector,
    add_vectors,
    elementwise_clip,
    matmul,
    pseudo_inverse,
    sub_vectors,
    transpose,
)


@dataclass
class GLSSolution:
    flux: Vector
    covariance: Matrix
    residuals: Vector
    chi2: float


def _quadratic_form(vec: Vector, mat: Matrix) -> float:
    total = 0.0
    for i, v_i in enumerate(vec):
        total += v_i * sum(mat[i][j] * vec[j] for j in range(len(vec)))
    return total


def gls_adjust(
    response: Matrix,
    measurements: Vector,
    measurement_cov: Matrix,
    prior_flux: Vector,
    prior_cov: Matrix,
    enforce_nonnegativity: bool = True,
) -> GLSSolution:
    if len(response[0]) != len(prior_flux):
        raise ValueError("Response matrix and prior flux shape mismatch.")

    rc0 = matmul(response, prior_cov)  # type: ignore[arg-type]
    innovation_cov = matmul(rc0, transpose(response))  # type: ignore[arg-type]
    for i in range(len(innovation_cov)):
        for j in range(len(innovation_cov)):
            innovation_cov[i][j] += measurement_cov[i][j]
    innovation_cov_inv = pseudo_inverse(innovation_cov)

    gain = matmul(prior_cov, matmul(transpose(response), innovation_cov_inv))  # type: ignore[arg-type]
    model_prediction = matmul(response, prior_flux)  # type: ignore[arg-type]
    residuals = sub_vectors(measurements, model_prediction)  # type: ignore[arg-type]
    update = matmul(gain, residuals)  # type: ignore[arg-type]
    phi_hat = add_vectors(prior_flux, update)
    if enforce_nonnegativity:
        phi_hat = elementwise_clip(phi_hat, 0.0)

    posterior_cov = matmul(prior_cov, matmul(transpose(response), matmul(innovation_cov_inv, rc0)))  # type: ignore[arg-type]
    for i in range(len(prior_cov)):
        for j in range(len(prior_cov)):
            posterior_cov[i][j] = prior_cov[i][j] - posterior_cov[i][j]

    chi2 = _quadratic_form(residuals, innovation_cov_inv)
    return GLSSolution(flux=phi_hat, covariance=posterior_cov, residuals=residuals, chi2=chi2)
