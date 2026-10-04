"""Independent GLS parameter/area oracle for signed Gaussian peak fits."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fluxforge.analysis.peakfit import GaussianPeak, PeakFitResult, fit_single_peak


def synthetic_peak():
    channels = np.arange(41, dtype=float)
    gaussian = np.exp(-0.5 * ((channels - 20) / 2.5) ** 2)
    counts = 50 * gaussian + 0.2 * channels - 15
    covariance = np.eye(41) * 16 + np.ones((41, 41)) * 9
    return channels, counts, covariance


def test_signed_gls_fit_and_area_uncertainty_match_analytic_jacobian():
    channels, counts, covariance = synthetic_peak()
    assert (counts < 0).any()
    fit = fit_single_peak(
        channels,
        counts,
        20,
        fit_width=15,
        initial_sigma=2.5,
        counts_uncertainty=np.sqrt(np.diag(covariance)),
        counts_covariance=csr_matrix(covariance),
    )
    assert fit.success
    assert fit.peak.centroid == pytest.approx(20, abs=1e-5)
    assert fit.peak.sigma == pytest.approx(2.5, abs=1e-5)
    assert fit.peak.area == pytest.approx(50 * 2.5 * np.sqrt(2 * np.pi), rel=1e-6)
    # Analytic derivatives of A exp(-(x-mu)^2/(2 sigma^2)) + slope*x + b.
    x = channels[5:36]
    z = x - 20
    gaussian = np.exp(-0.5 * (z / 2.5) ** 2)
    jacobian = np.column_stack(
        [
            gaussian,
            50 * gaussian * z / 2.5**2,
            50 * gaussian * z**2 / 2.5**3,
            x,
            np.ones(len(x)),
        ]
    )
    window_covariance = covariance[5:36, 5:36]
    parameter_covariance = np.linalg.inv(
        jacobian.T @ np.linalg.solve(window_covariance, jacobian)
    )
    np.testing.assert_allclose(
        fit.covariance, parameter_covariance, rtol=2e-4, atol=1e-5
    )
    gradient = np.array([2.5, 0, 50, 0, 0]) * np.sqrt(2 * np.pi)
    expected_variance = gradient @ parameter_covariance @ gradient
    assert fit.net_counts_uncertainty**2 == pytest.approx(expected_variance, rel=1e-5)
    assert fit.net_counts_uncertainty != pytest.approx(
        fit.peak.area_uncertainty, rel=0.01
    )
    assert fit.chi_squared < 1e-12


def test_uncertainty_scale_changes_fit_error_without_changing_peak_area():
    channels, counts, covariance = synthetic_peak()
    first = fit_single_peak(
        channels,
        counts,
        20,
        fit_width=15,
        counts_uncertainty=np.sqrt(np.diag(covariance)),
    )
    second = fit_single_peak(
        channels,
        counts,
        20,
        fit_width=15,
        counts_uncertainty=5 * np.sqrt(np.diag(covariance)),
    )
    assert first.success and second.success
    assert second.peak.area == pytest.approx(first.peak.area, rel=1e-6)
    assert second.net_counts_uncertainty == pytest.approx(
        5 * first.net_counts_uncertainty, rel=1e-5
    )


@pytest.mark.parametrize(
    "uncertainty", [np.ones(40), np.full(41, np.nan), -np.ones(41)]
)
def test_invalid_fit_uncertainties_fail(uncertainty):
    channels, counts, _ = synthetic_peak()
    with pytest.raises(ValueError, match="counts_uncertainty"):
        fit_single_peak(channels, counts, 20, counts_uncertainty=uncertainty)


def test_invalid_fit_covariance_fails_without_silent_diagonal_fallback():
    channels, counts, covariance = synthetic_peak()
    with pytest.raises(ValueError, match="match the full spectrum"):
        fit_single_peak(channels, counts, 20, counts_covariance=np.eye(40))
    with pytest.raises(ValueError, match="diagonal must match"):
        fit_single_peak(
            channels,
            counts,
            20,
            counts_covariance=covariance,
            counts_uncertainty=np.ones(41),
        )
    covariance[20, 19] += 1
    with pytest.raises(ValueError, match="symmetric"):
        fit_single_peak(channels, counts, 20, counts_covariance=covariance)
    indefinite = np.eye(41)
    indefinite[20, 19] = indefinite[19, 20] = 2
    fit = fit_single_peak(channels, counts, 20, counts_covariance=indefinite)
    assert not fit.success
    assert "positive definite" in fit.message


def test_area_covariance_requires_explicit_parameter_layout():
    peak = GaussianPeak(
        centroid=20, amplitude=50, sigma=2.5, amplitude_unc=3, sigma_unc=0.4
    )
    covariance = np.eye(8)
    fit = PeakFitResult(peak=peak, background=np.zeros(10), covariance=covariance)
    assert fit.net_counts_uncertainty == peak.area_uncertainty
    fit.area_parameter_indices = (4, 6)
    covariance[4, 6] = covariance[6, 4] = -0.5
    expected = 2 * np.pi * (2.5**2 + 50**2 - 2.5 * 50)
    assert fit.net_counts_uncertainty**2 == pytest.approx(expected)


def test_nonfinite_fitted_parameter_covariance_is_not_success(monkeypatch):
    def invalid_fit(*args, **kwargs):
        return np.array(kwargs["p0"]), np.full((5, 5), np.inf)

    monkeypatch.setattr("fluxforge.analysis.peakfit.optimize.curve_fit", invalid_fit)
    channels, counts, _ = synthetic_peak()
    fit = fit_single_peak(channels, counts, 20)
    assert not fit.success
    assert "non-finite" in fit.message
