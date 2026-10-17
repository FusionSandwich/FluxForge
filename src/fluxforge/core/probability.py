"""Strict covariance and normalized Gaussian density primitives."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def covariance_matrix(values, size: int, name: str, *, definite=False):
    """Copy a symmetric PSD covariance; never add a numerical error floor."""
    matrix = np.array(values, dtype=float, copy=True)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    scale = np.max(np.abs(matrix))
    tolerance = 1e-12 * scale
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if np.any(np.diag(matrix) < 0):
        raise ValueError(f"{name} must have nonnegative variances")
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
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
