from fluxforge.solvers.iterative import gravel, mlem


def test_mlem_identity_converges_to_measurements():
    response = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    measurements = [1.5, 3.0]
    solution = mlem(response, measurements, initial_flux=[0.5, 0.5], max_iters=25, tolerance=1e-9)
    assert solution.converged
    for est, truth in zip(solution.flux, measurements):
        assert abs(est - truth) < 1e-6


def test_gravel_recovers_scale_with_weights():
    response = [
        [1.0, 0.5],
        [0.2, 1.0],
    ]
    true_flux = [2.0, 1.0]
    measurements = [response[0][0] * true_flux[0] + response[0][1] * true_flux[1],
                    response[1][0] * true_flux[0] + response[1][1] * true_flux[1]]
    measurement_uncertainty = [0.05 * m for m in measurements]

    solution = gravel(
        response,
        measurements,
        initial_flux=[1.0, 1.0],
        measurement_uncertainty=measurement_uncertainty,
        max_iters=200,
        tolerance=1e-8,
    )
    assert solution.converged
    for est, truth in zip(solution.flux, true_flux):
        assert abs(est - truth) / truth < 1e-3
