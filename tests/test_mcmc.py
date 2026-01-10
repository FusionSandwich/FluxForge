"""Tests for MCMC Bayesian solver."""

import pytest
import math

np = pytest.importorskip("numpy")

from fluxforge.solvers.mcmc import (
    mcmc_unfold,
    MCMCSolution,
    mcmc_convergence_diagnostic,
    _log_likelihood,
    _log_prior_smoothness,
    _propose_flux,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def simple_problem():
    """Simple 2-group problem with known solution."""
    # Response matrix: 2 measurements, 2 energy groups
    response = [
        [1.0, 0.5],  # meas 1 = 1.0 * phi_0 + 0.5 * phi_1
        [0.3, 1.0],  # meas 2 = 0.3 * phi_0 + 1.0 * phi_1
    ]
    # True flux: [10, 5]
    # True measurements: [12.5, 8.0]
    measurements = [12.5, 8.0]
    uncertainties = [0.5, 0.4]
    true_flux = [10.0, 5.0]
    return response, measurements, uncertainties, true_flux


@pytest.fixture
def multi_group_problem():
    """5-group problem for testing."""
    # Create diagonal-ish response
    n_meas, n_groups = 4, 5
    response = []
    for i in range(n_meas):
        row = [0.1] * n_groups
        for g in range(n_groups):
            if abs(i - g * (n_meas - 1) / (n_groups - 1)) < 1.5:
                row[g] = 1.0
        response.append(row)
    
    # True flux
    true_flux = [10.0, 8.0, 6.0, 8.0, 10.0]
    
    # Measurements
    measurements = []
    for i in range(n_meas):
        m = sum(response[i][g] * true_flux[g] for g in range(n_groups))
        measurements.append(m)
    
    uncertainties = [m * 0.1 for m in measurements]
    
    return response, measurements, uncertainties, true_flux


# =============================================================================
# MCMCSolution tests
# =============================================================================

class TestMCMCSolution:
    """Test MCMCSolution dataclass."""
    
    def test_dataclass_creation(self):
        """MCMCSolution can be created with required fields."""
        sol = MCMCSolution(
            flux=[1.0, 2.0],
            samples=[[1.0, 2.0], [1.1, 2.1]],
            credible_lower=[0.8, 1.8],
            credible_upper=[1.2, 2.2],
            credible_median=[1.0, 2.0],
            acceptance_rate=0.3,
        )
        assert sol.flux == [1.0, 2.0]
        assert sol.acceptance_rate == 0.3
        
    def test_default_values(self):
        """MCMCSolution has sensible defaults."""
        sol = MCMCSolution(
            flux=[1.0],
            samples=[],
            credible_lower=[0.9],
            credible_upper=[1.1],
            credible_median=[1.0],
            acceptance_rate=0.25,
        )
        assert sol.chi_squared == 0.0
        assert sol.log_posterior_history == []


# =============================================================================
# Likelihood and prior tests
# =============================================================================

class TestLogLikelihood:
    """Test log-likelihood calculation."""
    
    def test_perfect_fit_high_likelihood(self):
        """Perfect fit should have high likelihood."""
        response = [[1.0, 0.0], [0.0, 1.0]]
        flux = [5.0, 3.0]
        measurements = [5.0, 3.0]  # Perfect match
        uncertainties = [0.5, 0.5]
        
        log_lik = _log_likelihood(response, flux, measurements, uncertainties)
        assert log_lik == 0.0  # Perfect fit = 0 chi2 = 0 log-lik term
        
    def test_poor_fit_low_likelihood(self):
        """Poor fit should have lower likelihood."""
        response = [[1.0, 0.0], [0.0, 1.0]]
        flux_good = [5.0, 3.0]
        flux_bad = [10.0, 10.0]
        measurements = [5.0, 3.0]
        uncertainties = [0.5, 0.5]
        
        log_lik_good = _log_likelihood(response, flux_good, measurements, uncertainties)
        log_lik_bad = _log_likelihood(response, flux_bad, measurements, uncertainties)
        
        assert log_lik_good > log_lik_bad


class TestLogPriorSmoothness:
    """Test smoothness prior calculation."""
    
    def test_smooth_spectrum_high_prior(self):
        """Smooth spectrum should have high prior."""
        smooth_flux = [10.0, 10.0, 10.0]  # Constant = smooth
        log_prior = _log_prior_smoothness(smooth_flux, smoothness_weight=1.0)
        assert log_prior == 0.0  # No variation = 0 penalty
        
    def test_rough_spectrum_low_prior(self):
        """Rough spectrum should have lower prior."""
        smooth_flux = [10.0, 10.0, 10.0]
        rough_flux = [10.0, 1.0, 10.0]  # Large jumps
        
        log_prior_smooth = _log_prior_smoothness(smooth_flux, smoothness_weight=1.0)
        log_prior_rough = _log_prior_smoothness(rough_flux, smoothness_weight=1.0)
        
        assert log_prior_smooth > log_prior_rough
        
    def test_negative_flux_rejected(self):
        """Negative flux should return -inf prior."""
        bad_flux = [10.0, -1.0, 5.0]
        log_prior = _log_prior_smoothness(bad_flux)
        assert log_prior == float('-inf')


# =============================================================================
# Proposal function tests
# =============================================================================

class TestProposeFlux:
    """Test flux proposal mechanism."""
    
    def test_proposal_preserves_positivity(self):
        """Proposed flux should be positive."""
        flux = [10.0, 5.0, 1.0]
        for _ in range(100):
            proposal = _propose_flux(flux, step_size=0.2)
            for p in proposal:
                assert p > 0
                
    def test_proposal_changes_values(self):
        """Proposal should produce different values."""
        flux = [10.0, 5.0, 1.0]
        proposals = [_propose_flux(flux, step_size=0.1) for _ in range(10)]
        
        # Should not all be identical
        unique_first = set(p[0] for p in proposals)
        assert len(unique_first) > 1


# =============================================================================
# MCMC solver tests
# =============================================================================

class TestMCMCUnfold:
    """Test MCMC unfolding solver."""
    
    def test_basic_convergence(self, simple_problem):
        """MCMC should converge to reasonable solution."""
        response, measurements, uncertainties, true_flux = simple_problem
        
        result = mcmc_unfold(
            response, measurements,
            measurement_uncertainty=uncertainties,
            n_samples=2000,
            burn_in=500,
            thin=3,
            seed=42,
            verbose=False,
        )
        
        assert isinstance(result, MCMCSolution)
        assert len(result.flux) == 2
        assert len(result.samples) > 0
        assert 0 < result.acceptance_rate < 1
        
    def test_recovers_true_flux(self, simple_problem):
        """MCMC posterior mean should be close to true flux."""
        response, measurements, uncertainties, true_flux = simple_problem
        
        result = mcmc_unfold(
            response, measurements,
            measurement_uncertainty=uncertainties,
            n_samples=5000,
            burn_in=1000,
            thin=5,
            seed=42,
            prior="smoothness",
            smoothness_weight=0.1,
        )
        
        # Check posterior mean is within 30% of true values
        for g, (est, true) in enumerate(zip(result.flux, true_flux)):
            rel_error = abs(est - true) / true
            assert rel_error < 0.3, f"Group {g}: {est:.2f} vs true {true:.2f}"
            
    def test_credible_intervals_contain_mean(self, simple_problem):
        """Credible intervals should contain the posterior mean."""
        response, measurements, uncertainties, true_flux = simple_problem
        
        result = mcmc_unfold(
            response, measurements,
            measurement_uncertainty=uncertainties,
            n_samples=3000,
            burn_in=500,
            thin=3,
            seed=42,
        )
        
        for g in range(len(result.flux)):
            assert result.credible_lower[g] <= result.flux[g] <= result.credible_upper[g]
            
    def test_multi_group(self, multi_group_problem):
        """MCMC should work with multi-group problem."""
        response, measurements, uncertainties, true_flux = multi_group_problem
        
        result = mcmc_unfold(
            response, measurements,
            measurement_uncertainty=uncertainties,
            n_samples=3000,
            burn_in=1000,
            thin=5,
            seed=42,
        )
        
        assert len(result.flux) == 5
        assert result.chi_squared > 0
        
    def test_uniform_prior(self, simple_problem):
        """MCMC should work with uniform prior."""
        response, measurements, uncertainties, _ = simple_problem
        
        result = mcmc_unfold(
            response, measurements,
            measurement_uncertainty=uncertainties,
            n_samples=2000,
            burn_in=500,
            prior="uniform",
            seed=42,
        )
        
        assert len(result.flux) == 2
        assert result.acceptance_rate > 0
        
    def test_reproducibility_with_seed(self, simple_problem):
        """Same seed should give same results."""
        response, measurements, uncertainties, _ = simple_problem
        
        result1 = mcmc_unfold(
            response, measurements,
            n_samples=1000,
            burn_in=200,
            seed=12345,
        )
        
        result2 = mcmc_unfold(
            response, measurements,
            n_samples=1000,
            burn_in=200,
            seed=12345,
        )
        
        assert result1.flux == result2.flux
        assert result1.acceptance_rate == result2.acceptance_rate


# =============================================================================
# Convergence diagnostic tests
# =============================================================================

class TestMCMCConvergenceDiagnostic:
    """Test MCMC convergence diagnostics."""
    
    def test_returns_expected_keys(self):
        """Diagnostic should return expected metrics."""
        samples = [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]] * 10
        
        diag = mcmc_convergence_diagnostic(samples, group=0)
        
        assert 'ess' in diag
        assert 'mean' in diag
        assert 'std' in diag
        assert 'autocorr_1' in diag
        
    def test_ess_positive(self):
        """Effective sample size should be positive."""
        samples = [[1.0 + 0.1 * i] for i in range(50)]
        
        diag = mcmc_convergence_diagnostic(samples, group=0)
        
        assert diag['ess'] > 0
        
    def test_correlated_chain_low_ess(self):
        """Highly correlated chain should have low ESS."""
        # Slowly increasing chain (high autocorrelation)
        correlated = [[float(i)] for i in range(100)]
        
        # Random chain (low autocorrelation)
        import random
        random.seed(42)
        uncorrelated = [[random.gauss(50, 10)] for _ in range(100)]
        
        diag_corr = mcmc_convergence_diagnostic(correlated, group=0)
        diag_uncorr = mcmc_convergence_diagnostic(uncorrelated, group=0)
        
        # Correlated chain should have lower ESS ratio
        ess_ratio_corr = diag_corr['ess'] / len(correlated)
        ess_ratio_uncorr = diag_uncorr['ess'] / len(uncorrelated)
        
        assert ess_ratio_corr < ess_ratio_uncorr
