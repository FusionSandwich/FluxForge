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
) -> np.ndarray:
    """
    Propagate coefficient covariance to absolute efficiency uncertainty.
    """
    coeffs = np.asarray(coefficients, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    if cov.shape[0] != cov.shape[1] or cov.shape[0] != coeffs.size:
        raise ValueError("Covariance matrix size must match coefficient count.")

    base = semi_empirical_efficiency(energy_keV, coeffs)
    energy = np.asarray(energy_keV, dtype=float)

    gradients = []
    for i in range(coeffs.size):
        delta = step * (abs(coeffs[i]) if coeffs[i] != 0 else 1.0)
        coeffs_hi = coeffs.copy()
        coeffs_lo = coeffs.copy()
        coeffs_hi[i] += delta
        coeffs_lo[i] -= delta
        f_hi = semi_empirical_efficiency(energy, coeffs_hi)
        f_lo = semi_empirical_efficiency(energy, coeffs_lo)
        gradients.append((f_hi - f_lo) / (2.0 * delta))

    grad = np.stack(gradients, axis=1)  # (n_energy, n_params)
    var = np.einsum("ij,jk,ik->i", grad, cov, grad)
    return np.sqrt(np.clip(var, 0.0, None))


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
