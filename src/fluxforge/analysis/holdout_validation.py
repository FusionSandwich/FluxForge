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
    y = np.asarray(rates, dtype=float)
    p = np.asarray(prior_flux, dtype=float)
    c = np.asarray(prior_covariance, dtype=float)
    v = np.asarray(observation_covariance, dtype=float)
    m, n = a.shape
    if y.shape != (m,) or p.shape != (n,) or c.shape != (n, n) or v.shape != (m, m):
        raise ValueError("Inconsistent shapes for response, rates, prior or covariance")
    hold = np.array(sorted(set(int(i) for i in holdout_index)), dtype=int)
    if hold.size == 0 or np.any(hold < 0) or np.any(hold >= m):
        raise ValueError("holdout_index must name at least one valid row")
    fit = np.setdiff1d(np.arange(m), hold)
    if fit.size == 0:
        raise ValueError("At least one fit row is required")

    # Row scaling for conditioning (a pure change of units for each rate row)
    scale = np.sqrt(np.clip(np.diag(v), np.finfo(float).tiny, None))
    a_s = a / scale[:, None]
    y_s = y / scale
    v_s = v / scale[:, None] / scale[None, :]

    s = _symmetric(a_s @ c @ a_s.T + v_s)
    s_ff = s[np.ix_(fit, fit)]
    s_hf = s[np.ix_(hold, fit)]
    s_hh = s[np.ix_(hold, hold)]
    innovation = y_s[fit] - a_s[fit] @ p
    try:
        chol = np.linalg.cholesky(s_ff)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Fit innovation covariance is not positive definite") from exc

    def solve_ff(rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))

    gain = (c @ a_s[fit].T) @ solve_ff(np.eye(fit.size))
    flux = p + gain @ innovation
    posterior = _symmetric(c - gain @ a_s[fit] @ c)

    mean_s = a_s[hold] @ p + s_hf @ solve_ff(innovation)
    cov_s = _symmetric(s_hh - s_hf @ solve_ff(s_hf.T))
    mean = mean_s * scale[hold]
    cov = cov_s * scale[hold][:, None] * scale[hold][None, :]
    residual = y[hold] - mean
    chi2 = float(residual @ np.linalg.solve(cov, residual))
    return HoldoutPrediction(flux, posterior, mean, cov, residual, chi2, int(hold.size))


__all__ = ["HoldoutPrediction", "predict_holdouts"]
