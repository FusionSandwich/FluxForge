"""Validation helpers for unfolding solvers and workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def require_input(name: str, values: Any) -> np.ndarray:
    """Require that an unfolding input is present and numeric."""

    if values is None:
        raise ValueError(f"The input for {name} must not be None.")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"The input for {name} must not be empty.")
    return array


def require_finite(name: str, values: Any) -> np.ndarray:
    """Require finite numeric values for an unfolding input."""

    array = require_input(name, values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"The items in {name} must be finite.")
    return array


def require_nonnegative(name: str, values: Any) -> np.ndarray:
    """Require a non-negative unfolding input, mirroring pyunfold-style guards."""

    array = require_finite(name, values)
    if np.amin(array) < 0.0:
        raise ValueError(f"The items in {name} must be non-negative.")
    return array


def require_covariance_matrix(name: str, values: Any) -> np.ndarray:
    """Require a finite square covariance matrix with non-negative diagonal."""

    matrix = require_finite(name, values)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"The input for {name} must be a square matrix.")
    if np.any(np.diag(matrix) < 0.0):
        raise ValueError(f"The diagonal of {name} must be non-negative.")
    return matrix


__all__ = [
    "require_covariance_matrix",
    "require_finite",
    "require_input",
    "require_nonnegative",
]
