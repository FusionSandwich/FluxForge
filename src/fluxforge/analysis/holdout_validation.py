"""Holdout prediction for linear-Gaussian unfolding with correlated observations.

For the linear model ``y = A phi + e`` with prior ``phi ~ N(p, C)`` and
observation errors ``e ~ N(0, V)`` over *all* rows (fit and holdout, with
cross-covariance allowed), the fit/holdout rates are jointly Gaussian:

    S = A C A^T + V,   partitioned into fit (f) and holdout (h) blocks.

The flux estimate uses only the fit rows. The holdout predictive
distribution conditioned on the fit data is

    mu    = A_h p + S_hf S_ff^-1 (y_f - A_f p)
    Sigma = S_hh - S_hf S_ff^-1 S_fh

which reduces to ``A_h phi_hat`` and ``A_h C_post A_h^T + V_hh`` when the fit
and holdout errors are independent. Shared calibration, timing or nuclear
data correlations between fit and holdout rows are therefore handled
exactly instead of being refused. Holdout measured values never enter the
flux estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from fluxforge.uncertainty.covariance import covariance_matrix


@dataclass
class HoldoutPrediction:
    fit_flux: np.ndarray
    fit_posterior_covariance: np.ndarray
    holdout_mean: np.ndarray
    holdout_covariance: np.ndarray
    holdout_residuals: np.ndarray
    holdout_standardized_chi2: float
    n_holdout: int


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def predict_holdouts(
    response: np.ndarray,
    rates: np.ndarray,
    prior_flux: np.ndarray,
    prior_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    holdout_index: Sequence[int],
) -> HoldoutPrediction:
    """
    Fit on non-holdout rows and predict holdout rows conditioned on the fit.

    ``observation_covariance`` is the full (n x n) covariance over all rows,
    including any fit/holdout cross terms (observation plus propagated
    response error). Units must be consistent: response maps flux to rates.
    """
    a = np.asarray(response, dtype=float)
    if a.ndim != 2:
        raise ValueError("Response must be a two-dimensional matrix")
    y = np.asarray(rates, dtype=float)
    p = np.asarray(prior_flux, dtype=float)
    c = np.asarray(prior_covariance, dtype=float)
    v = np.asarray(observation_covariance, dtype=float)
    m, n = a.shape
    if y.shape != (m,) or p.shape != (n,) or c.shape != (n, n) or v.shape != (m, m):
        raise ValueError("Inconsistent shapes for response, rates, prior or covariance")
    if not all(np.all(np.isfinite(item)) for item in (a, y, p, c, v)):
        raise ValueError("Response, rates, prior and covariance must be finite")
    c = covariance_matrix(c, n, "prior covariance")
    v = covariance_matrix(v, m, "observation covariance")
    if any(not isinstance(i, (int, np.integer)) for i in holdout_index):
        raise ValueError("holdout_index must contain integer row indices")
    hold = np.array(sorted(int(i) for i in holdout_index), dtype=int)
    if len(set(hold)) != len(hold):
        raise ValueError("holdout_index must not repeat a row")
    if hold.size == 0 or np.any(hold < 0) or np.any(hold >= m):
        raise ValueError("holdout_index must name at least one valid row")
    fit = np.setdiff1d(np.arange(m), hold)
    if fit.size == 0:
        raise ValueError("At least one fit row is required")

    # Scale by the full prior-predictive row variance. A holdout can have no
    # observation noise and still have finite uncertainty from the flux prior.
    predictive_variance = np.einsum("ij,jk,ik->i", a, c, a) + np.diag(v)
    if np.any(predictive_variance < 0) or not np.all(np.isfinite(predictive_variance)):
        raise ValueError("Every row needs nonnegative finite predictive variance")
    fallback = max(
        float(np.max(np.abs(y))), float(np.max(np.abs(a @ p))), np.finfo(float).tiny
    )
    scale = np.where(predictive_variance > 0, np.sqrt(predictive_variance), fallback)
    a_s = a / scale[:, None]
    y_s = y / scale
    v_s = v / scale[:, None] / scale[None, :]

    s = _symmetric(a_s @ c @ a_s.T + v_s)
    s_ff = s[np.ix_(fit, fit)]
    s_hf = s[np.ix_(hold, fit)]
    s_hh = s[np.ix_(hold, hold)]
    innovation = y_s[fit] - a_s[fit] @ p
    s_ff = covariance_matrix(s_ff, len(fit), "fit predictive covariance")

    def spectral(matrix):
        values, vectors = np.linalg.eigh(matrix)
        tolerance = (
            20
            * len(matrix)
            * np.finfo(float).eps
            * max(float(np.max(np.abs(values))), 1.0)
        )
        return values, vectors, values > tolerance

    values, vectors, positive = spectral(s_ff)
    tolerance = (
        100
        * np.finfo(float).eps
        * max(1.0, float(np.linalg.norm(y_s)), float(np.linalg.norm(a_s @ p)))
    )
    if np.linalg.norm(vectors[:, ~positive].T @ innovation) > tolerance:
        raise ValueError(
            "Fit data are incompatible with noiseless covariance directions"
        )

    def solve_ff(rhs: np.ndarray) -> np.ndarray:
        return (vectors[:, positive] / values[positive]) @ (
            vectors[:, positive].T @ rhs
        )

    gain = (c @ a_s[fit].T) @ solve_ff(np.eye(fit.size))
    flux = p + gain @ innovation
    posterior = _symmetric(c - gain @ a_s[fit] @ c)

    mean_s = a_s[hold] @ p + s_hf @ solve_ff(innovation)
    cov_s = _symmetric(s_hh - s_hf @ solve_ff(s_hf.T))
    # The subtraction can leave roundoff on a mathematically zero conditional
    # variance. Judge it against the pre-subtraction predictive scale.
    tolerance_cov = (
        100 * max(m, n) * np.finfo(float).eps * max(float(np.max(np.abs(s_hh))), 1.0)
    )
    if np.min(np.linalg.eigvalsh(cov_s)) < -tolerance_cov:
        raise ValueError("Holdout predictive covariance is not positive semidefinite")
    cv, cq = np.linalg.eigh(cov_s)
    cv = np.maximum(cv, 0.0)
    cov_s = (cq * cv) @ cq.T
    mean = mean_s * scale[hold]
    cov = cov_s * scale[hold][:, None] * scale[hold][None, :]
    residual = y[hold] - mean
    residual_s = residual / scale[hold]
    values, vectors, positive = spectral(cov_s)
    if np.linalg.norm(vectors[:, ~positive].T @ residual_s) > tolerance:
        raise ValueError(
            "Holdout data are incompatible with noiseless covariance directions"
        )
    projected = vectors[:, positive].T @ residual_s
    chi2 = float(np.sum(projected**2 / values[positive]))
    if not np.isfinite(chi2):
        raise ValueError("Holdout predictive covariance is not usable")
    return HoldoutPrediction(flux, posterior, mean, cov, residual, chi2, int(hold.size))


__all__ = ["HoldoutPrediction", "predict_holdouts"]
