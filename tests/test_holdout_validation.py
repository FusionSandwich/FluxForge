"""Holdout prediction with correlated fit/holdout covariance (issue #203)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.holdout_validation import predict_holdouts


def _problem(shared: float = 0.05):
    rng = np.random.default_rng(7)
    a = rng.uniform(0.2, 1.0, (6, 4)) * 1e-24  # cm2-scale response
    p = np.array([4.0, 3.0, 2.0, 1.0]) * 1e12
    c = np.diag((0.5 * p) ** 2)
    rates = a @ p
    independent = np.diag((0.03 * rates) ** 2)
    shared_eff = shared**2 * np.outer(rates, rates)  # one detector efficiency
    return a, p, c, independent + shared_eff, rates


def test_independent_case_reduces_to_posterior_prediction() -> None:
    a, p, c, _, rates = _problem()
    v = np.diag((0.03 * rates) ** 2)
    y = rates * np.array([1.02, 0.97, 1.01, 1.0, 0.99, 1.03])
    out = predict_holdouts(a, y, p, c, v, [4, 5])
    np.testing.assert_allclose(out.holdout_mean, a[[4, 5]] @ out.fit_flux, rtol=1e-10)
    expected = (
        a[[4, 5]] @ out.fit_posterior_covariance @ a[[4, 5]].T
        + v[np.ix_([4, 5], [4, 5])]
    )
    np.testing.assert_allclose(out.holdout_covariance, expected, rtol=1e-8)


def test_correlated_case_matches_direct_joint_gaussian_conditioning() -> None:
    a, p, c, v, rates = _problem()
    y = rates * np.array([1.04, 1.05, 1.03, 1.06, 1.05, 1.02])
    fit, hold = [0, 1, 2, 3], [4, 5]
    s = a @ c @ a.T + v
    innov = y[fit] - a[fit] @ p
    mean = a[hold] @ p + s[np.ix_(hold, fit)] @ np.linalg.solve(
        s[np.ix_(fit, fit)], innov
    )
    cov = s[np.ix_(hold, hold)] - s[np.ix_(hold, fit)] @ np.linalg.solve(
        s[np.ix_(fit, fit)], s[np.ix_(fit, hold)]
    )
    out = predict_holdouts(a, y, p, c, v, hold)
    np.testing.assert_allclose(out.holdout_mean, mean, rtol=1e-9)
    np.testing.assert_allclose(out.holdout_covariance, cov, rtol=1e-7)
    # Shared positive error on the fit rows raises the holdout prediction
    naive = a[hold] @ out.fit_flux
    assert np.all(out.holdout_mean > naive)


def test_holdout_values_never_change_the_fit() -> None:
    a, p, c, v, rates = _problem()
    y1 = rates.copy()
    y2 = rates.copy()
    y2[[4, 5]] *= 3.0
    f1 = predict_holdouts(a, y1, p, c, v, [4, 5]).fit_flux
    f2 = predict_holdouts(a, y2, p, c, v, [4, 5]).fit_flux
    np.testing.assert_array_equal(f1, f2)


def test_standardized_holdout_chi2_is_calibrated_with_shared_errors() -> None:
    a, p, c, v, _ = _problem(shared=0.08)
    rng = np.random.default_rng(11)
    lc, lv = np.linalg.cholesky(c), np.linalg.cholesky(v)
    chi2 = []
    for _ in range(4000):
        truth = p + lc @ rng.standard_normal(4)
        y = a @ truth + lv @ rng.standard_normal(6)
        chi2.append(predict_holdouts(a, y, p, c, v, [4, 5]).holdout_standardized_chi2)
    assert np.mean(chi2) == pytest.approx(2.0, rel=0.06)


def test_noiseless_holdout_keeps_prior_predictive_uncertainty() -> None:
    a = np.array([[1.0], [2.0]])
    y = np.array([1.1, 2.0])
    p = np.array([1.0])
    c = np.array([[1.0]])
    v = np.diag([0.1, 0.0])
    out = predict_holdouts(a, y, p, c, v, [1])
    assert out.holdout_mean[0] == pytest.approx(2.0 * (1.0 + 0.1 / 1.1))
    assert out.holdout_covariance[0, 0] == pytest.approx(4.0 * (1.0 - 1.0 / 1.1))
    assert np.isfinite(out.holdout_standardized_chi2)


def test_deterministic_holdout_requires_compatible_data() -> None:
    args = [
        np.array([[1.0], [0.0]]),
        np.array([1.0, 0.0]),
        np.array([1.0]),
        np.array([[1.0]]),
        np.diag([0.1, 0.0]),
        [1],
    ]
    out = predict_holdouts(*args)
    assert out.holdout_standardized_chi2 == 0.0
    np.testing.assert_array_equal(out.holdout_covariance, [[0.0]])
    args[1] = np.array([1.0, 0.1])
    with pytest.raises(ValueError, match="incompatible with noiseless"):
        predict_holdouts(*args)


def test_holdout_index_rejects_truncation_and_duplicates() -> None:
    a, p, c, v, rates = _problem()
    with pytest.raises(ValueError, match="integer row indices"):
        predict_holdouts(a, rates, p, c, v, [4.9])
    with pytest.raises(ValueError, match="repeat"):
        predict_holdouts(a, rates, p, c, v, [4, 4])
