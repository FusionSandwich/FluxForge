"""Scale-aware validation and spectral factors for possibly noiseless covariance."""

import numpy as np


def covariance_matrix(values, size, name="covariance"):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape {(size, size)}")
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale == 0:
        return matrix.copy()
    normalized = matrix / scale
    tolerance = 20 * max(size, 1) * np.finfo(float).eps
    if np.max(np.abs(normalized - normalized.T)) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    normalized = (normalized + normalized.T) / 2
    eigenvalues, vectors = np.linalg.eigh(normalized)
    tolerance *= max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if np.any(eigenvalues < 0):
        normalized = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
    return normalized * scale


def covariance_factor(matrix):
    """Factor L such that L L^T = C, including singular C; no absolute floor."""
    size = len(matrix)
    matrix = covariance_matrix(matrix, size)
    scale = float(np.max(np.abs(matrix))) if size else 0.0
    if not scale:
        return np.zeros_like(matrix)
    eigenvalues, vectors = np.linalg.eigh(matrix / scale)
    return vectors * (np.sqrt(np.maximum(eigenvalues, 0)) * np.sqrt(scale))
