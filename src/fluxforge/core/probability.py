"""Strict covariance and normalized Gaussian density primitives."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def covariance_matrix(values, size: int, name: str, *, definite=False):
    """Copy a symmetric PSD covariance; never add a numerical error floor."""
    matrix = np.array(values, dtype=float, copy=True)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    if np.any(np.diag(matrix) < 0):
        raise ValueError(f"{name} must have nonnegative variances")
    # Validate in correlation coordinates so mixed physical units cannot hide
    # an invalid small-variance block behind a large dimensional variance.
    active = np.diag(matrix) > 0
    if np.any(matrix[~active] != 0) or np.any(matrix[:, ~active] != 0):
        raise ValueError(f"{name} has nonzero covariance with a zero-variance input")
    if active.any():
        std = np.sqrt(np.diag(matrix)[active])
        with np.errstate(over="ignore", invalid="ignore"):
            correlation = matrix[np.ix_(active, active)] / std[:, None] / std[None, :]
        if (
            not np.isfinite(correlation).all()
            or np.max(np.abs(correlation)) > 1 + 1e-12
        ):
            raise ValueError(f"{name} must be positive semidefinite")
        if np.max(np.abs(correlation - correlation.T)) > 1e-12:
            raise ValueError(f"{name} must be symmetric")
        symmetric = correlation / 2 + correlation.T / 2
        if np.linalg.eigvalsh(symmetric).min() < -1e-12 * len(std):
            raise ValueError(f"{name} must be positive semidefinite")
    matrix = matrix / 2 + matrix.T / 2
    if definite:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
    matrix.setflags(write=False)
    return matrix


def gaussian_log_density(residual, covariance):
    """Normalized multivariate Gaussian log density, including log determinant."""
    residual = np.asarray(residual, dtype=float)
    factor = np.linalg.cholesky(covariance)
    whitened = solve_triangular(factor, residual, lower=True, check_finite=False)
    return float(
        -0.5
        * (
            whitened @ whitened
            + 2 * np.log(np.diag(factor)).sum()
            + residual.size * np.log(2 * np.pi)
        )
    )
