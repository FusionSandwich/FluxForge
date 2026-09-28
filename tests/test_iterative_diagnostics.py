"""Iterative-solver fit statistics and stopping semantics (issue #195)."""

from __future__ import annotations

import numpy as np

from fluxforge.solvers.iterative import CHI_SQUARED_DEFINITION, gravel, mlem


def test_chi2_target_stop_is_not_reported_as_convergence() -> None:
    response = [[1.0, 0.0], [0.0, 1.0]]
    for solver in (gravel, mlem):
        solution = solver(
            response, [1.5, 3.0], initial_flux=[1.4, 2.9],
            measurement_uncertainty=[0.15, 0.3], max_iters=200,
            tolerance=1e-14, chi2_tolerance=10.0,
        )
        assert solution.stop_reason == "chi2_target"
        assert solution.converged is False


def test_chi2_is_per_measurement_with_stated_definition() -> None:
    response = [[1.0, 0.5, 0.2], [0.1, 1.0, 0.4]]
    solution = gravel(
        response, [2.0, 1.0], initial_flux=[1.0, 1.0, 1.0],
        measurement_uncertainty=[0.2, 0.1], max_iters=3,
        tolerance=0.0,
    )
    assert solution.n_measurements == 2
    assert np.isclose(solution.chi_squared, solution.chi_squared_total / 2)
    assert solution.diagnostics["chi_squared_definition"] == CHI_SQUARED_DEFINITION
    assert solution.stop_reason == "max_iterations"
