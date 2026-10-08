"""Independent analytic covariance oracle for all joint Gaussian layouts."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from fluxforge.analysis.peakfit import fit_multiple_peaks


def test_joint_evaluation_budget_exhaustion_returns_failed_components():
    x = np.arange(51, dtype=float)
    y = 200 * np.exp(-0.5 * ((x - 15.3) / 2.2) ** 2)
    y += 150 * np.exp(-0.5 * ((x - 29.6) / 2.8) ** 2) + 20
    fits = fit_multiple_peaks(x, y, [15, 29], fit_width=10, max_evaluations=1)
    assert len(fits) == 2
    assert all(not fit.success for fit in fits)
    assert all("evaluations" in fit.message.lower() for fit in fits)


@pytest.mark.parametrize("budget", [0, -1, 1.5, float("nan"), True])
def test_joint_evaluation_budget_must_be_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        fit_multiple_peaks(np.arange(20), np.ones(20), [5, 12], max_evaluations=budget)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("background", ["linear", "constant"])
def test_joint_signed_fit_matches_analytic_covariance_and_area_gradient(
    shared, background
):
    x = np.arange(51, dtype=float)
    g1 = np.exp(-0.5 * ((x - 15) / 2.5) ** 2)
    g2 = np.exp(-0.5 * ((x - 29) / 2.5) ** 2)
    y = 50 * g1 + 30 * g2 - 12 + (0.1 * x if background == "linear" else 0)
    covariance = 16 * np.eye(len(x)) + 4 * np.ones((len(x), len(x)))
    fits = fit_multiple_peaks(
        x,
        y,
        [15, 29],
        fit_width=10,
        background_model=background,
        share_sigma=shared,
        counts_uncertainty=np.sqrt(np.diag(covariance)),
        counts_covariance=csr_matrix(covariance),
    )
    assert all(f.success for f in fits)
    xx = x[5:40]
    gs = [g1[5:40], g2[5:40]]
    derivs = []
    width_derivs = []
    for amplitude, mu, g in zip([50, 30], [15, 29], gs):
        derivs.extend([g, amplitude * g * (xx - mu) / 2.5**2])
        sigma_derivative = amplitude * g * (xx - mu) ** 2 / 2.5**3
        if shared:
            width_derivs.append(sigma_derivative)
        else:
            derivs.append(sigma_derivative)
    if shared:
        derivs.append(sum(width_derivs))
    if background == "linear":
        derivs.append(xx)
    derivs.append(np.ones(len(xx)))
    jacobian = np.column_stack(derivs)
    expected_covariance = np.linalg.inv(
        jacobian.T @ np.linalg.solve(covariance[5:40, 5:40], jacobian)
    )
    for i, fit in enumerate(fits):
        np.testing.assert_allclose(
            fit.covariance, expected_covariance, rtol=3e-4, atol=1e-5
        )
        assert fit.peak.centroid == pytest.approx([15, 29][i], abs=1e-5)
        assert fit.net_counts == pytest.approx(
            [50, 30][i] * 2.5 * np.sqrt(2 * np.pi), rel=1e-5
        )
        gradient = np.zeros(len(expected_covariance))
        gradient[2 * i if shared else 3 * i] = 2.5 * np.sqrt(2 * np.pi)
        gradient[4 if shared else 3 * i + 2] = [50, 30][i] * np.sqrt(2 * np.pi)
        assert fit.net_counts_uncertainty**2 == pytest.approx(
            gradient @ expected_covariance @ gradient, rel=3e-4
        )
        assert fit.chi_squared < 1e-10


def test_duplicate_components_cannot_report_successful_joint_fit():
    x = np.arange(41, dtype=float)
    y = 50 * np.exp(-0.5 * ((x - 20) / 2.5) ** 2) + 5
    fits = fit_multiple_peaks(x, y, [20, 20], share_sigma=True)
    assert not any(f.success for f in fits)


def test_invalid_joint_count_covariance_cannot_silently_fall_back():
    x = np.arange(41, dtype=float)
    with pytest.raises(ValueError, match="match"):
        fit_multiple_peaks(x, np.ones(41), [15, 25], counts_covariance=np.eye(2))


def test_insufficient_joint_window_cannot_duplicate_single_fit_areas():
    x = np.arange(5, dtype=float)
    fits = fit_multiple_peaks(
        x, np.ones(5) * 100, [1, 2, 3], fit_width=2, share_sigma=True
    )
    assert len(fits) == 3
    assert not any(fit.success for fit in fits)
