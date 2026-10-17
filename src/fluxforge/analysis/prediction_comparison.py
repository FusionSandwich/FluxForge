"""Calculated/experimental comparisons with declared joint uncertainty."""

import numpy as np

from fluxforge.core.probability import covariance_matrix, gaussian_log_density


def compare_calculated_experimental(calculated, experimental, joint_covariance):
    """Propagate full C/E covariance using a first-order ratio Jacobian.

    Covariance order is [all calculated values, all experimental values].
    Shared normalization/efficiency errors can be represented across both.
    This diagnostic neither treats experimental/vendor values as exact truth nor
    tests independence from training. Ratios near zero denominators need more
    than first-order propagation.
    """
    calculated, experimental = (
        np.asarray(v, dtype=float) for v in (calculated, experimental)
    )
    if (
        calculated.ndim != 1
        or not calculated.size
        or calculated.shape != experimental.shape
        or not np.isfinite(calculated).all()
        or not np.isfinite(experimental).all()
        or np.any(calculated < 0)
        or np.any(experimental <= 0)
    ):
        raise ValueError(
            "require matching nonnegative C and strictly positive E vectors"
        )
    n = calculated.size
    covariance = covariance_matrix(joint_covariance, 2 * n, "joint_covariance")
    jacobian = np.concatenate(
        (np.diag(1 / experimental), np.diag(-calculated / experimental**2)), axis=1
    )
    difference = np.concatenate((np.eye(n), -np.eye(n)), axis=1)
    residual_covariance = difference @ covariance @ difference.T
    residual = calculated - experimental
    log_score = chi_squared = None
    try:
        np.linalg.cholesky(residual_covariance)
        chi_squared = float(residual @ np.linalg.solve(residual_covariance, residual))
        log_score = gaussian_log_density(residual, residual_covariance)
    except np.linalg.LinAlgError:
        pass  # Singular shared-error cancellation is not a usable full density.
    return {
        "calculated_over_experimental": (calculated / experimental).tolist(),
        "ratio_covariance": (jacobian @ covariance @ jacobian.T).tolist(),
        "residual": residual.tolist(),
        "residual_covariance": residual_covariance.tolist(),
        "chi_squared": chi_squared,
        "joint_gaussian_log_density": log_score,
        "ratio_uncertainty_method": "first_order_joint_covariance",
        "physical_qualified": False,
    }
