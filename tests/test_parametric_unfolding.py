"""Analytical and quadrature references independent of the sampling kernel."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm, truncnorm

from fluxforge.solvers.parametric import (
    GaussianResponseModel,
    Parameter,
    ParametricProblem,
    integrate_spectrum,
    parametric_unfold,
    split_rhat,
)


def test_full_energy_and_cross_reaction_covariance():
    response = np.array([[1, 2], [3, 4]])
    factor = np.array([[1, 0], [2, 1], [3, -1], [4, 2]]) * 0.1
    nuclear = factor @ factor.T
    observed = np.array([[1, 0.3], [0.3, 2]])
    model = GaussianResponseModel(response, [3, 6], observed, nuclear)
    flux = np.array([2, 5])
    # An independently constructed linear projection of flattened responses.
    projection = np.kron(np.eye(2), flux.reshape(1, -1))
    expected_covariance = observed + projection @ nuclear @ projection.T
    mean, covariance = model.predict(flux)
    np.testing.assert_allclose(mean, response @ flux)
    np.testing.assert_allclose(covariance, expected_covariance)
    assert model.log_likelihood(flux) == pytest.approx(
        multivariate_normal.logpdf([3, 6], mean=mean, cov=expected_covariance)
    )


def test_response_marginalization_matches_direct_integral():
    model = GaussianResponseModel([[1.0]], [2.4], [[0.16]], [[0.09]])
    expected = quad(
        lambda r: norm.pdf(2.4, loc=2 * r, scale=0.4) * norm.pdf(r, loc=1, scale=0.3),
        -5,
        5,
        epsabs=1e-12,
    )[0]
    assert model.log_likelihood([2]) == pytest.approx(math.log(expected), abs=1e-10)


def test_flux_dependent_log_determinant_is_not_dropped():
    model = GaussianResponseModel([[0]], [0], [[1]], [[1]])
    assert model.log_likelihood([10]) - model.log_likelihood([0]) == pytest.approx(
        -0.5 * math.log(101)
    )


def test_input_arrays_are_copied_and_readonly():
    response, covariance = np.ones((1, 1)), np.ones((1, 1))
    model = GaussianResponseModel(response, [1], covariance)
    response[:] = 99
    covariance[:] = 99
    assert model.response[0, 0] == 1
    assert model.measurement_covariance[0, 0] == 1
    with pytest.raises(ValueError):
        model.response[0, 0] = 2


@pytest.mark.parametrize(
    "covariance",
    [
        [[1, 2], [2, 1]],
        [[1, 1], [0, 1]],
        [[1, 0], [0, -1]],
        [[1, 0], [0, 0]],
        [[1, 0], [0, np.nan]],
    ],
)
def test_invalid_measurement_covariance_rejected(covariance):
    with pytest.raises(ValueError):
        GaussianResponseModel(np.eye(2), [1, 1], covariance)


def test_covariance_validation_is_scale_relative():
    with pytest.raises(ValueError, match="semidefinite"):
        GaussianResponseModel(
            np.eye(2), [1, 1], np.eye(2), np.kron(np.eye(2), [[1, 2], [2, 1]]) * 1e-40
        )
    model = GaussianResponseModel(np.eye(2), [1, 1], np.eye(2) * 1e-40)
    assert np.isfinite(model.log_likelihood([1, 1]))


def test_quadrature_uses_group_integrals_not_midpoint_ordinates():
    result = integrate_spectrum(lambda e, t: t[0] * e**2, [0, 1, 3], [2])
    np.testing.assert_allclose(result, [2 / 3, 52 / 3])


def test_every_parameter_bound_enforced_before_callback():
    def must_not_run(theta):
        raise AssertionError("out-of-support model evaluated")

    problem = ParametricProblem(
        GaussianResponseModel([[1]], [1], [[1]]),
        must_not_run,
        [Parameter("a", 0.1, 1, 0.1), Parameter("b", -2, 2, 0.2)],
        "bounded",
    )
    for theta in ([0.05, 0], [1.1, 0], [0.5, -3], [0.5, 3], [np.nan, 0]):
        assert problem.log_posterior(theta) == -np.inf


def test_custom_prior_is_density_in_declared_coordinates():
    model = GaussianResponseModel([[1]], [1], [[1]])
    problem = ParametricProblem(
        model,
        lambda t: t,
        [Parameter("a", 0, 2, 0.2)],
        "custom",
        log_prior=lambda t: -t[0],
    )
    assert problem.log_posterior([1]) == pytest.approx(model.log_likelihood([1]) - 1)


def test_sampler_matches_truncated_normal_reference_and_repeats_seed():
    model = GaussianResponseModel([[1]], [2], [[0.16]])
    problem = ParametricProblem(
        model, lambda t: t, [Parameter("a", 0, 5, 0.5)], "linear_amplitude"
    )
    result = parametric_unfold(problem, [[0.2], [4]], draws=10000, warmup=1000, seed=42)
    reference = truncnorm(-5, 7.5, loc=2, scale=0.4)
    assert result.flux[0] == pytest.approx(reference.mean(), abs=0.03)
    assert result.flux_samples.var() == pytest.approx(reference.var(), abs=0.02)
    assert result.rhat[0] < 1.02
    assert result.metadata["physical_qualified"] is False
    assert result.metadata["converged"] is None
    first = parametric_unfold(problem, [[1], [3]], draws=10, warmup=10, seed=7)
    second = parametric_unfold(problem, [[1], [3]], draws=10, warmup=10, seed=7)
    np.testing.assert_array_equal(first.parameter_samples, second.parameter_samples)


def test_stuck_chains_do_not_report_successful_rhat():
    assert np.isinf(split_rhat(np.ones((2, 10, 1)))).all()


def test_uncertain_response_posterior_matches_parameter_quadrature():
    model = GaussianResponseModel([[1]], [2], [[0.04]], [[0.09]])
    problem = ParametricProblem(
        model, lambda t: t, [Parameter("flux", 0, 5, 0.6)], "uncertain_response"
    )

    def density(f):
        return norm.pdf(2, loc=f, scale=math.sqrt(0.04 + 0.09 * f * f))

    normalization = quad(density, 0, 5)[0]
    mean = quad(lambda f: f * density(f), 0, 5)[0] / normalization
    result = parametric_unfold(
        problem, [[0.5], [4]], draws=10000, warmup=1000, seed=289
    )
    assert result.flux[0] == pytest.approx(mean, abs=0.06)


@pytest.mark.parametrize("kwargs", [{"draws": 0}, {"thin": 0}, {"warmup": -1}])
def test_invalid_sampling_controls(kwargs):
    problem = ParametricProblem(
        GaussianResponseModel([[1]], [1], [[1]]),
        lambda t: t,
        [Parameter("a", 0, 2, 0.1)],
        "one",
    )
    with pytest.raises(ValueError):
        parametric_unfold(problem, [[1], [1.5]], **kwargs)
