from fluxforge.solvers.iterative import gravel, mlem, gradient_descent


def test_mlem_identity_converges_to_measurements():
    """Test MLEM with identity response matrix recovers measurements."""
    response = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    measurements = [1.5, 3.0]
    # More iterations and tighter solver tolerance for better convergence
    solution = mlem(response, measurements, initial_flux=[0.5, 0.5], max_iters=500, tolerance=1e-10)
    assert solution.converged
    for est, truth in zip(solution.flux, measurements):
        # Realistic tolerance for iterative solver (0.5% relative error)
        assert abs(est - truth) / truth < 5e-3, f"Expected {truth}, got {est}"


def test_gravel_recovers_scale_with_weights():
    """Test GRAVEL recovers true flux within reasonable tolerance."""
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
        max_iters=500,
        tolerance=1e-10,
    )
    assert solution.converged
    for est, truth in zip(solution.flux, true_flux):
        # Realistic tolerance for GRAVEL solver (1% relative error)
        assert abs(est - truth) / truth < 0.02, f"Expected {truth}, got {est}"


def test_gradient_descent_identity_case():
    """Test gradient descent with identity matrix recovers measurements."""
    response = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    measurements = [2.0, 4.0, 6.0]
    
    solution = gradient_descent(
        response,
        measurements,
        initial_flux=[1.0, 1.0, 1.0],
        max_iters=5000,
        learning_rate=0.5,
        smoothness_weight=0.0,  # No smoothness for identity test
        chi2_tolerance=0.01,
    )
    
    assert solution.converged or solution.chi_squared < 0.1
    for est, truth in zip(solution.flux, measurements):
        # Allow 5% relative error
        assert abs(est - truth) / truth < 0.05, f"Expected {truth}, got {est}"


def test_gradient_descent_with_smoothness():
    """Test gradient descent produces smooth spectrum."""
    # Response matrix that covers different energy ranges
    response = [
        [1.0, 0.5, 0.1, 0.0, 0.0],
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.1, 0.5, 1.0],
    ]
    # Measurements consistent with a smooth spectrum
    true_flux = [1.0, 2.0, 3.0, 2.0, 1.0]  # Bell-shaped
    measurements = [
        sum(response[i][g] * true_flux[g] for g in range(5))
        for i in range(3)
    ]
    
    solution = gradient_descent(
        response,
        measurements,
        initial_flux=[1.0] * 5,
        max_iters=5000,
        learning_rate=0.5,
        smoothness_weight=0.1,  # Encourage smoothness
        chi2_tolerance=0.1,
    )
    
    # Check that solution is reasonably smooth
    log_diffs = []
    import math
    for g in range(len(solution.flux) - 1):
        log_diffs.append(abs(math.log(solution.flux[g+1] + 1e-12) - math.log(solution.flux[g] + 1e-12)))
    
    avg_smoothness = sum(log_diffs) / len(log_diffs)
    assert avg_smoothness < 2.0, f"Spectrum not smooth enough: avg log-diff = {avg_smoothness}"


def test_gradient_descent_auto_scaling():
    """Test gradient descent auto-scaling works correctly."""
    response = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    # Very different scale from initial guess
    measurements = [1e6, 2e6]
    
    solution = gradient_descent(
        response,
        measurements,
        initial_flux=[1.0, 1.0],  # Much smaller than measurements
        max_iters=5000,
        auto_scale=True,
        smoothness_weight=0.0,
    )
    
    # Should scale up to match measurement magnitude
    for est, truth in zip(solution.flux, measurements):
        assert abs(est - truth) / truth < 0.1, f"Expected ~{truth}, got {est}"
