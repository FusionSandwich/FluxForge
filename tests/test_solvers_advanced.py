"""Tests for fluxforge.solvers.advanced."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.solvers.advanced import (
    LMConfig,
    PPPCorrectionMethod,
    apply_ppp_correction,
    estimate_unknown_uncertainty,
    gls_update_numpy,
    group_averaged_cross_section,
    levenberg_marquardt,
    romberg_integrate,
    spectrum_averaged_cross_section,
)


def test_levenberg_marquardt_fits_linear_model() -> None:
    A = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    x_true = np.array([0.3, -0.2])
    y_data = A @ x_true
    y_cov = np.eye(3) * 1e-4
    x0 = np.zeros(2)

    result = levenberg_marquardt(
        model_func=lambda x: A @ x,
        jacobian_func=lambda x: A,
        y_data=y_data,
        y_cov=y_cov,
        x0=x0,
        config=LMConfig(max_iter=50, tol=1e-12, lambda_init=0.1),
    )

    assert result.converged
    np.testing.assert_allclose(result.x, x_true, rtol=1e-4, atol=1e-6)
    assert result.chi2 >= 0.0
    assert result.covariance.shape == (2, 2)


def test_gls_update_numpy_matches_closed_form_identity_case() -> None:
    response = np.eye(2)
    measurements = np.array([1.0, 2.0])
    measurement_cov = np.eye(2) * 0.25
    prior_flux = np.array([0.0, 0.0])
    prior_cov = np.eye(2)

    phi_post, cov_post, chi2 = gls_update_numpy(
        response=response,
        measurements=measurements,
        measurement_cov=measurement_cov,
        prior_flux=prior_flux,
        prior_cov=prior_cov,
    )

    # For identity response, posterior mean = (1/(1+0.25))*measurement = 0.8*y.
    np.testing.assert_allclose(phi_post, np.array([0.8, 1.6]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(cov_post), np.array([0.2, 0.2]), rtol=1e-12, atol=1e-12)
    assert chi2 > 0.0


def test_gls_update_numpy_sparse_path_runs() -> None:
    response = np.array([[1.0, 0.1], [0.2, 1.0]])
    measurements = np.array([1.2, 0.8])
    measurement_cov = np.eye(2) * 0.05
    prior_flux = np.array([1.0, 1.0])
    prior_cov = np.eye(2) * 0.5

    dense = gls_update_numpy(
        response=response,
        measurements=measurements,
        measurement_cov=measurement_cov,
        prior_flux=prior_flux,
        prior_cov=prior_cov,
        use_sparse=False,
    )
    sparse = gls_update_numpy(
        response=response,
        measurements=measurements,
        measurement_cov=measurement_cov,
        prior_flux=prior_flux,
        prior_cov=prior_cov,
        use_sparse=True,
    )
    np.testing.assert_allclose(dense[0], sparse[0], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(dense[1], sparse[1], rtol=1e-8, atol=1e-8)


def test_romberg_integrate_polynomial() -> None:
    value, err = romberg_integrate(lambda x: x**2, 0.0, 1.0)
    assert abs(value - 1.0 / 3.0) < 1e-9
    assert err >= 0.0


@pytest.mark.parametrize("method", ["romberg", "quad", "trapezoid"])
def test_spectrum_averaged_cross_section_methods(method: str) -> None:
    sigma_avg, unc = spectrum_averaged_cross_section(
        sigma=lambda E: 2.0 * E + 1.0,
        flux=lambda E: 1.0,
        E_low=0.0,
        E_high=2.0,
        method=method,
        n_points=200,
    )
    # Average of 2E+1 over [0,2] is 3.
    assert abs(sigma_avg - 3.0) < 5e-3
    assert unc >= 0.0


def test_spectrum_averaged_cross_section_handles_zero_denominator() -> None:
    sigma_avg, unc = spectrum_averaged_cross_section(
        sigma=lambda E: E,
        flux=lambda E: 0.0,
        E_low=0.0,
        E_high=1.0,
        method="romberg",
    )
    assert sigma_avg == 0.0
    assert np.isinf(unc)


def test_group_averaged_cross_section() -> None:
    sigma_g = group_averaged_cross_section(
        sigma=lambda E: 2.0 * E,
        flux_per_group=np.array([1.0, 1.0]),
        energy_bounds=np.array([0.0, 1.0, 2.0]),
        method="romberg",
    )
    np.testing.assert_allclose(sigma_g, np.array([1.0, 3.0]), rtol=1e-6, atol=1e-6)


def test_apply_ppp_correction_variants() -> None:
    y = np.array([2.0, 4.0])
    V = np.array([[0.04, 0.01], [0.01, 0.16]])

    y_none, V_none = apply_ppp_correction(y, V, PPPCorrectionMethod.NONE)
    np.testing.assert_allclose(y_none, y)
    np.testing.assert_allclose(V_none, V)

    y_log, V_log = apply_ppp_correction(y, V, PPPCorrectionMethod.CHIBA_SMITH)
    np.testing.assert_allclose(y_log, np.log(y))
    np.testing.assert_allclose(V_log[0, 1], V[0, 1] / (y[0] * y[1]))

    y_ratio, V_ratio = apply_ppp_correction(y, V, PPPCorrectionMethod.RATIO)
    np.testing.assert_allclose(y_ratio, y / np.mean(y))
    np.testing.assert_allclose(V_ratio, V / (np.mean(y) ** 2))


def test_estimate_unknown_uncertainty_methods() -> None:
    residuals = np.array([3.0, 3.0])
    covariance = np.eye(2)

    ml = estimate_unknown_uncertainty(residuals, covariance, method="ml")
    birge = estimate_unknown_uncertainty(residuals, covariance, method="birge")
    unknown = estimate_unknown_uncertainty(residuals, covariance, method="not-a-method")

    assert ml > 0.0
    assert birge > 0.0
    assert unknown == 0.0

    # Small residuals should not require extra uncertainty.
    no_extra = estimate_unknown_uncertainty(np.array([0.01, 0.01]), covariance, method="ml")
    assert no_extra == 0.0
