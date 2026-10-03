"""Independent second-pass edge checks for ML seed and Gaussian RMLE."""

from unittest.mock import patch
import numpy as np
import pytest
from fluxforge.unfolding import MLSeedUnfolder
from fluxforge.solvers import rmle


@pytest.mark.parametrize("warm", [0, 8])
def test_no_prior_seed_is_invariant_to_individual_row_units(warm):
    response = np.array([[1.0, 2.0], [2.0, 0.5], [0.1, 3.0]])
    observed = np.array([3.0, 7.0, 5.0])
    sigma = np.array([0.1, 0.2, 0.3])
    baseline = MLSeedUnfolder().unfold(
        observed, response, measurement_uncertainty=sigma, warm_start_iterations=warm
    )
    units = np.array([1e-120, 1e120, 1e-20])
    actual = MLSeedUnfolder().unfold(
        observed * units,
        response * units[:, None],
        measurement_uncertainty=sigma * units,
        warm_start_iterations=warm,
    )
    np.testing.assert_allclose(actual.flux, baseline.flux, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        actual.chi_squared, baseline.chi_squared, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        actual.parameters_used["confidence_score"],
        baseline.parameters_used["confidence_score"],
        rtol=1e-12,
        atol=0,
    )
    assert actual.converged == baseline.converged


def test_zero_measurements_are_finite_and_unit_invariant_without_prior():
    units = np.array([1e-100, 1e100])
    for scale in [np.ones(2), units]:
        result = MLSeedUnfolder().unfold(
            np.zeros(2), np.eye(2) * scale[:, None], measurement_uncertainty=scale
        )
        np.testing.assert_allclose(result.flux, [1e-12, 1e-12], rtol=1e-12, atol=0)
        assert np.isfinite(result.chi_squared)
        assert np.isfinite(result.parameters_used["confidence_score"])


def test_unsupported_positive_row_is_preserved_and_rejects_seed():
    response = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    result = MLSeedUnfolder().unfold(
        np.array([3.0, 5.0, 7.0]), response, measurement_uncertainty=np.ones(3)
    )
    assert result.predicted_measurements[1] == 0
    assert result.residuals[1] == 5
    assert not result.converged
    assert result.chi_squared >= 25
    assert np.isfinite(result.flux).all()


def test_unsupported_column_stays_finite_and_rank_is_recorded():
    result = MLSeedUnfolder().unfold(
        np.array([3.0, 6.0]),
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        measurement_uncertainty=np.ones(2),
    )
    assert np.isfinite(result.flux).all()
    assert result.parameters_used["response_rank"] == 1
    assert "rank deficient" in result.parameters_used["uncertainty_unavailable_reason"]
    assert result.uncertainties is None


def test_all_zero_response_without_prior_is_rejected():
    with pytest.raises(ValueError, match="positive sensitivity"):
        MLSeedUnfolder().unfold(
            np.zeros(2), np.zeros((2, 2)), measurement_uncertainty=np.ones(2)
        )


def test_real_bvls_iteration_limit_is_not_reported_as_convergence():
    rng = np.random.default_rng(2)
    response = rng.uniform(0.1, 1.0, (12, 8))
    observed = rng.uniform(0.0, 10.0, 12)
    with patch.object(
        rmle.optimize, "nnls", side_effect=RuntimeError("forced active-set failure")
    ):
        result = rmle.rmle_unfolding(
            rmle.SpectrumData(counts=observed, uncertainty=np.ones(12)),
            rmle.ResponseMatrix(matrix=response),
            regularization=rmle.RegularizationType.NONE,
            max_iterations=1,
            tolerance=1e-12,
        )
    assert not result.converged
    assert np.isfinite(result.solution).all()
    assert np.all(result.solution >= 0)


def test_gaussian_subnormal_measurement_units_preserve_finite_whitened_fit():
    results = []
    for units in [1.0, 1e-320]:
        results.append(
            rmle.rmle_unfolding(
                rmle.SpectrumData(
                    counts=np.array([3.0, 7.0]) * units, uncertainty=np.ones(2) * units
                ),
                rmle.ResponseMatrix(matrix=np.eye(2) * units),
                regularization=rmle.RegularizationType.TIKHONOV,
                reg_param=1.0,
                enforce_positivity=False,
            )
        )
    np.testing.assert_allclose(
        results[1].solution, results[0].solution, rtol=1e-10, atol=0
    )


def test_finite_whitened_seed_inputs_do_not_escape_as_nan_results():
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            result = MLSeedUnfolder().unfold(
                np.array([3.0, 7.0]) * 1e200,
                np.eye(2) * 1e200,
                measurement_uncertainty=np.ones(2),
            )
        except ValueError:
            return  # Controlled numerical rejection is an acceptable contract.
    assert np.isfinite(result.flux).all()
    assert np.isfinite(result.chi_squared)
    assert np.isfinite(result.parameters_used["confidence_score"])
