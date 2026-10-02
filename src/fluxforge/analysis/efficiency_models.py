"""
Efficiency models for HPGe detector calibration.

Includes a semi-empirical model used for HPGe efficiency calibration
with detector window and dead-layer terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from fluxforge.data.attenuation_components import be_window_mu_rho, ge_components_mu_rho


def semi_empirical_efficiency(
    energy_keV: np.ndarray,
    coefficients: Sequence[float],
) -> np.ndarray:
    """
    Semi-empirical HPGe efficiency model.

    Parameters
    ----------
    energy_keV : array-like
        Gamma energy in keV.
    coefficients : sequence of float
        Model parameters. If length 5, uses core parameters:
        [scale, l, alpha, l0, kappa].
        If length 7, includes window/dead-layer terms:
        [scale, l, alpha, l0, kappa, window_g_cm2, dead_g_cm2].

    Returns
    -------
    np.ndarray
        Absolute efficiency at the given energies.
    """
    energy = np.asarray(energy_keV, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)

    if coeffs.size not in (5, 7):
        raise ValueError("Semi-empirical efficiency requires 5 or 7 coefficients.")
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Gamma energy must be finite and positive (keV).")
    if np.any(~np.isfinite(coeffs)) or np.any(coeffs[:4] <= 0) or np.any(coeffs[4:] < 0):
        raise ValueError("Semi-empirical coefficients must be finite and physically nonnegative.")

    scale, length, alpha, length0, kappa = coeffs[:5]
    window = coeffs[5] if coeffs.size == 7 else 0.0
    dead = coeffs[6] if coeffs.size == 7 else 0.0

    tau, sigma, mu_total = ge_components_mu_rho(energy)
    mu_window = be_window_mu_rho(energy)

    window_term = np.exp(-mu_window * window)
    dead_term = np.exp(-mu_total * dead)
    absorption = 1.0 - np.exp(-mu_total * length)
    scatter = tau + sigma * (1.0 - np.exp(-((mu_total * length0) ** alpha))) * kappa

    efficiency = scale * window_term * dead_term * absorption * scatter / mu_total
    return efficiency


def semi_empirical_efficiency_uncertainty(
    energy_keV: np.ndarray,
    coefficients: Sequence[float],
    covariance: np.ndarray,
    step: float = 1e-6,
    parameter_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Propagate coefficient covariance to absolute efficiency uncertainty.
    """
    coeffs = np.asarray(coefficients, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    indices = tuple(range(coeffs.size)) if parameter_indices is None else tuple(parameter_indices)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= coeffs.size for index in indices):
        raise ValueError("Covariance parameter indices must be unique and valid.")
    if cov.ndim != 2 or cov.shape != (len(indices), len(indices)):
        raise ValueError("Covariance matrix size must match coefficient count.")
    if np.any(~np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance must be finite and symmetric.")
    symmetric = (cov + cov.T) / 2
    tolerance = max(float(np.linalg.norm(symmetric, ord=2)), 1.0) * 1e-10
    if np.min(np.linalg.eigvalsh(symmetric)) < -tolerance:
        raise ValueError("Covariance must be positive semidefinite.")

    energy = np.atleast_1d(np.asarray(energy_keV, dtype=float))

    gradients = []
    for i in indices:
        delta = step * (abs(coeffs[i]) if coeffs[i] != 0 else 1.0)
        coeffs_hi = coeffs.copy()
        coeffs_lo = coeffs.copy()
        coeffs_hi[i] += delta
        coeffs_lo[i] = max(coeffs_lo[i] - delta, 0.0)
        f_hi = semi_empirical_efficiency(energy, coeffs_hi)
        f_lo = semi_empirical_efficiency(energy, coeffs_lo)
        gradients.append((f_hi - f_lo) / (coeffs_hi[i] - coeffs_lo[i]))

    grad = np.stack(gradients, axis=1)  # (n_energy, n_params)
    var = np.einsum("ij,jk,ik->i", grad, symmetric, grad)
    if np.any(var < -tolerance * np.sum(grad**2, axis=1)):
        raise ValueError("Covariance predicts negative efficiency variance.")
    return np.sqrt(np.maximum(var, 0.0))


@dataclass
class SemiEmpiricalEfficiency:
    """Convenience wrapper for semi-empirical efficiency calibration."""

    coefficients: Sequence[float]
    covariance: Optional[np.ndarray] = None

    def evaluate(self, energy_keV: np.ndarray) -> np.ndarray:
        return semi_empirical_efficiency(energy_keV, self.coefficients)

    def uncertainty(self, energy_keV: np.ndarray) -> Optional[np.ndarray]:
        if self.covariance is None:
            return None
        return semi_empirical_efficiency_uncertainty(
            energy_keV, self.coefficients, self.covariance
        )
