"""GRAVEL/MLEM must not depend on the units of individual rows (issue #194)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.solvers.iterative import gravel, mlem


def _problem():
    rng = np.random.default_rng(1)
    response = rng.uniform(0.1, 1.0, (4, 6))
    response[0, 3:] = 0.0
    response[3, :3] = 0.0
    truth = np.array([3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
    measurements = response @ truth
    # Inconsistent data so row weighting actually matters.
    measurements = measurements * np.array([1.05, 0.97, 1.02, 0.95])
    sigma = np.array([0.03, 0.05, 0.02, 0.04]) * measurements
    return response, measurements, sigma


def _solve(solver, response, measurements, sigma):
    return np.asarray(
        solver(
            response.tolist(),
            measurements.tolist(),
            initial_flux=[1.0] * response.shape[1],
            measurement_uncertainty=None if sigma is None else sigma.tolist(),
            max_iters=4000,
            tolerance=1e-12,
            chi2_tolerance=-1.0,
        ).flux
    )


@pytest.mark.parametrize("solver", [gravel, mlem])
def test_rescaling_one_row_does_not_change_solution(solver) -> None:
    response, measurements, sigma = _problem()
    base = _solve(solver, response, measurements, sigma)
    scaled_response, scaled_measurements = response.copy(), measurements.copy()
    scaled_sigma = sigma.copy()
    scaled_response[0] *= 1e-6
    scaled_measurements[0] *= 1e-6
    scaled_sigma[0] *= 1e-6
    scaled = _solve(solver, scaled_response, scaled_measurements, scaled_sigma)
    np.testing.assert_allclose(scaled, base, rtol=1e-8)


def _reference_gravel(response, measurements, sigma, flux, iterations):
    """UMG/Matzke GRAVEL update with W_ig = (y/sigma)^2 R phi / p."""
    for _ in range(iterations):
        predicted = response @ flux
        weights = (measurements / sigma) ** 2
        w = weights[:, None] * response * flux[None, :] / predicted[:, None]
        flux = flux * np.exp((w * np.log(measurements / predicted)[:, None]).sum(0) / w.sum(0))
    return flux


def test_weighted_gravel_matches_umg_weighting() -> None:
    response, measurements, sigma = _problem()
    observed = gravel(
        response.tolist(),
        measurements.tolist(),
        initial_flux=[1.0] * 6,
        measurement_uncertainty=sigma.tolist(),
        max_iters=50,
        tolerance=0.0,
        chi2_tolerance=-1.0,
        relaxation=1.0,
    )
    expected = _reference_gravel(response, measurements, sigma, np.ones(6), 50)
    np.testing.assert_allclose(observed.flux, expected, rtol=1e-10)
