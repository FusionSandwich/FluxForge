from fluxforge.solvers.gls import gls_adjust


def test_gls_identity_case():
    response = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    measurements = [1.0, 2.0, 3.0]
    cy = [
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ]
    prior = [0.0, 0.0, 0.0]
    c0 = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    solution = gls_adjust(response, measurements, cy, prior, c0)
    expected_scale = 1 / 1.01
    for estimate, truth in zip(solution.flux, measurements):
        assert abs(estimate - truth * expected_scale) < 1e-3
    assert solution.chi2 > 0
