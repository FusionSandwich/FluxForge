"""Independent posterior and covariance checks for spectrum adjustment."""

import numpy as np
import pytest

from fluxforge.core.unfolding_inputs import require_covariance_matrix
from fluxforge.solvers.gls import (
    ResponseCovariancePolicy,
    gls_adjust,
    gls_adjust_with_response_cov,
)


def test_posterior_matches_precision_space_quadratic_minimum():
    response = np.array([[1.0, 0.25], [0.1, 2.0], [1.0, 1.0]])
    measured = np.array([4.0, 7.0, 6.0])
    prior = np.array([2.0, 3.0])
    cy = np.array([[0.8, 0.1, 0.0], [0.1, 0.5, 0.05], [0.0, 0.05, 0.6]])
    cp = np.array([[1.0, 0.2], [0.2, 2.0]])
    # Independently minimize the sum of prior and observation quadratic penalties.
    wy, wp = np.linalg.inv(cy), np.linalg.inv(cp)
    hessian = wp + response.T @ wy @ response
    expected = np.linalg.solve(hessian, wp @ prior + response.T @ wy @ measured)
    result = gls_adjust(response, measured, cy, prior, cp)
    np.testing.assert_allclose(result.flux, expected, rtol=1e-12)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(hessian), rtol=1e-12)
    assert result.n_dof == 3
    assert len(result.influence) == 3
    np.testing.assert_allclose(
        result.influence, np.diag(response @ np.linalg.solve(hessian, response.T @ wy))
    )


def test_zero_response_uncertainty_keeps_conditional_uncertainty():
    # Product of N(2, 4) and N(4, 1) is N(3.6, 0.8).
    result = gls_adjust_with_response_cov(
        [[1.0]], [[0.0]], [4.0], [[1.0]], [2.0], [[4.0]], n_samples=4
    )
    assert result.flux == pytest.approx([3.6])
    assert result.covariance[0][0] == pytest.approx(0.8)


def test_signed_background_subtracted_observation_is_preserved():
    result = gls_adjust(
        [[1.0]], [-2.0], [[1.0]], [1.0], [[4.0]], enforce_nonnegativity=False
    )
    assert result.flux[0] == pytest.approx(-1.4)
    assert result.residuals == [-3.0]


@pytest.mark.parametrize("samples", [0, 1, -1, 2.5, True])
def test_mc_requires_enough_samples(samples):
    with pytest.raises(ValueError, match="at least 2"):
        gls_adjust_with_response_cov(
            [[1.0]], [[0.0]], [4.0], [[1.0]], [2.0], [[4.0]], n_samples=samples
        )


@pytest.mark.parametrize(
    "covariance, message",
    [
        ([[1.0, 0.2], [0.0, 1.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ],
)
def test_invalid_covariance_is_rejected(covariance, message):
    with pytest.raises(ValueError, match=message):
        require_covariance_matrix("covariance", covariance)


def test_rank_deficient_consistent_observations_are_supported():
    # Two exact copies of one observation provide one independent innovation.
    result = gls_adjust(
        [[1.0], [1.0]], [3.0, 3.0], [[1.0, 1.0], [1.0, 1.0]], [1.0], [[1.0]]
    )
    assert result.flux == pytest.approx([2.0])
    assert result.covariance[0][0] == pytest.approx(0.5)
    assert result.n_dof == 1
    assert result.diagnostics["condition_number"] > 1e12


def test_inconsistent_exact_observation_is_rejected():
    with pytest.raises(ValueError, match="zero-variance"):
        gls_adjust([[1.0]], [3.0], [[0.0]], [1.0], [[0.0]])


@pytest.mark.parametrize(
    "policy", [ResponseCovariancePolicy.NUISANCE, ResponseCovariancePolicy.MONTE_CARLO]
)
def test_unimplemented_policy_cannot_silently_ignore_uncertainty(policy):
    with pytest.raises(NotImplementedError):
        gls_adjust(
            [[1.0]],
            [3.0],
            [[1.0]],
            [1.0],
            [[1.0]],
            response_cov=[[0.1]],
            response_cov_policy=policy,
        )


def test_augmentation_requires_supplied_uncertainty():
    with pytest.raises(ValueError, match="requires response_cov"):
        gls_adjust(
            [[1.0]],
            [3.0],
            [[1.0]],
            [1.0],
            [[1.0]],
            response_cov_policy=ResponseCovariancePolicy.AUGMENT_VY,
        )


def test_clipping_does_not_claim_a_constrained_posterior():
    result = gls_adjust(
        [[1.0, 1.0]], [0.0], [[0.01]], [0.1, 10.0], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert result.flux[0] == 0.0
    assert result.diagnostics["clipped_bin_indices"] == [0]
    assert result.diagnostics["constrained_posterior_valid"] is False


def test_mc_total_covariance_includes_both_terms(monkeypatch):
    perturbations = iter([-0.2, 0.2])
    monkeypatch.setattr("random.gauss", lambda *_: next(perturbations))
    means, variances = [], []
    for r in [0.8, 1.2]:
        precision = 0.25 + r * r
        means.append((0.5 + r * 4.0) / precision)
        variances.append(1.0 / precision)
    result = gls_adjust_with_response_cov(
        [[1.0]], [[0.2]], [4.0], [[1.0]], [2.0], [[4.0]], n_samples=2
    )
    assert result.flux[0] == pytest.approx(np.mean(means))
    assert result.covariance[0][0] == pytest.approx(
        np.mean(variances) + np.var(means, ddof=1)
    )
