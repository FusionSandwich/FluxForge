"""
Tests for prior covariance models for GLS spectral adjustment.

Tests the prior_covariance module which provides:
- Diagonal covariance models
- Regional fractional uncertainty models
- Lethargy-correlated covariance models
- Covariance validation utilities
"""

import pytest
import numpy as np
import math

from fluxforge.core.prior_covariance import (
    PriorCovarianceModel,
    ResponseUncertaintyPolicy,
    EnergyRegion,
    STANDARD_REGIONS,
    PriorCovarianceConfig,
    ResponseUncertaintyConfig,
    create_diagonal_prior_covariance,
    create_regional_prior_covariance,
    create_lethargy_correlated_covariance,
    create_mcnp_statistics_covariance,
    validate_covariance_matrix,
    propagate_response_uncertainty_to_vy,
)


# ============================================================================
# Test Standard Energy Regions
# ============================================================================

class TestStandardRegions:
    """Test standard energy region definitions."""
    
    def test_regions_exist(self):
        """Verify standard regions are defined."""
        assert len(STANDARD_REGIONS) >= 3
    
    def test_thermal_region(self):
        """Thermal region exists with ~30% uncertainty."""
        thermal = next((r for r in STANDARD_REGIONS if r.name == "thermal"), None)
        assert thermal is not None
        assert thermal.energy_low_ev == 0.0
        assert thermal.fractional_uncertainty > 0.1
    
    def test_fast_region(self):
        """Fast region exists."""
        fast_regions = [r for r in STANDARD_REGIONS if "fast" in r.name]
        assert len(fast_regions) >= 1


# ============================================================================
# Test Diagonal Covariance
# ============================================================================

class TestDiagonalCovariance:
    """Test diagonal (uncorrelated) prior covariance."""
    
    def test_uniform_uncertainty(self):
        """Uniform fractional uncertainty on diagonal."""
        n = 10
        prior_flux = np.ones(n) * 1e12
        frac_unc = 0.25
        
        V = create_diagonal_prior_covariance(prior_flux, frac_unc)
        
        assert V.shape == (n, n)
        # Check diagonal elements
        expected_var = (frac_unc * 1e12) ** 2
        for i in range(n):
            assert V[i, i] == pytest.approx(expected_var, rel=0.01)
        # Check off-diagonal is zero
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert V[i, j] == 0.0
    
    def test_group_specific_uncertainty(self):
        """Different flux values give different variances."""
        prior_flux = np.array([1e10, 1e11, 1e12, 1e13])
        frac_unc = 0.20
        
        V = create_diagonal_prior_covariance(prior_flux, frac_unc)
        
        # Variances should differ
        assert V[0, 0] < V[1, 1] < V[2, 2] < V[3, 3]
    
    def test_is_symmetric(self):
        """Diagonal matrix is symmetric."""
        V = create_diagonal_prior_covariance(np.ones(5) * 1e12, 0.25)
        assert np.allclose(V, V.T)
    
    def test_is_positive_definite(self):
        """Diagonal matrix with positive entries is positive definite."""
        V = create_diagonal_prior_covariance(np.ones(5) * 1e12, 0.25)
        eigenvalues = np.linalg.eigvalsh(V)
        assert np.all(eigenvalues > 0)


# ============================================================================
# Test Regional Covariance
# ============================================================================

class TestRegionalCovariance:
    """Test regional fractional uncertainty covariance."""
    
    def test_regional_structure(self):
        """Different regions get different uncertainties."""
        n = 100
        prior_flux = np.ones(n) * 1e12
        # Energy bounds spanning all regions
        energy_bounds = np.logspace(-5, 7, n + 1)  # 1e-5 to 1e7 eV
        
        V = create_regional_prior_covariance(prior_flux, energy_bounds)
        
        assert V.shape == (n, n)
    
    def test_custom_regions(self):
        """Custom regions are applied."""
        n = 50
        prior_flux = np.ones(n) * 1e12
        energy_bounds = np.logspace(-3, 5, n + 1)
        
        regions = [
            EnergyRegion("low", 0.0, 1.0, 0.10),
            EnergyRegion("mid", 1.0, 1e3, 0.30),
            EnergyRegion("high", 1e3, 1e6, 0.50),
        ]
        
        V = create_regional_prior_covariance(prior_flux, energy_bounds, regions)
        
        # Variances should vary by region
        diag = np.diag(V)
        assert np.std(diag) > 0  # Not all the same


# ============================================================================
# Test Lethargy-Correlated Covariance
# ============================================================================

class TestLethargyCorrectedCovariance:
    """Test lethargy-correlated covariance model."""
    
    def test_basic_creation(self):
        """Create lethargy-correlated matrix."""
        n = 20
        prior_flux = np.ones(n) * 1e12
        energy_bounds = np.logspace(-3, 7, n + 1)
        
        V = create_lethargy_correlated_covariance(
            prior_flux, energy_bounds,
            fractional_uncertainty=0.25,
            correlation_length=1.0
        )
        
        assert V.shape == (n, n)
    
    def test_correlation_decay(self):
        """Correlation decays with lethargy distance."""
        n = 20
        prior_flux = np.ones(n) * 1e12
        energy_bounds = np.logspace(-3, 7, n + 1)
        
        V = create_lethargy_correlated_covariance(
            prior_flux, energy_bounds,
            fractional_uncertainty=0.25,
            correlation_length=1.0
        )
        
        # Adjacent groups should be more correlated than distant groups
        # Compare correlation (normalized covariance)
        mid = n // 2
        rho_adjacent = V[mid, mid + 1] / np.sqrt(V[mid, mid] * V[mid + 1, mid + 1])
        rho_far = V[0, n - 1] / np.sqrt(V[0, 0] * V[n - 1, n - 1])
        
        assert abs(rho_adjacent) > abs(rho_far)
    
    def test_is_symmetric(self):
        """Correlated matrix is symmetric."""
        n = 10
        prior_flux = np.ones(n) * 1e12
        energy_bounds = np.logspace(-3, 7, n + 1)
        
        V = create_lethargy_correlated_covariance(
            prior_flux, energy_bounds, 0.25, 1.0
        )
        
        assert np.allclose(V, V.T)


# ============================================================================
# Test MCNP Statistics Covariance
# ============================================================================

class TestMCNPStatisticsCovariance:
    """Test MCNP-based covariance creation."""
    
    def test_basic_creation(self):
        """Create from MCNP relative errors."""
        n = 10
        prior_flux = np.ones(n) * 1e12
        rel_errors = np.ones(n) * 0.05  # 5% relative errors
        
        V = create_mcnp_statistics_covariance(prior_flux, rel_errors)
        
        assert V.shape == (n, n)
        # For 5% relative error, sigma = 0.05 * 1e12 = 5e10
        expected_var = (0.05 * 1e12) ** 2
        assert V[0, 0] == pytest.approx(expected_var, rel=0.01)
    
    def test_with_correlation(self):
        """Create with non-zero inter-group correlation."""
        n = 5
        prior_flux = np.ones(n) * 1e12
        rel_errors = np.ones(n) * 0.10
        
        V = create_mcnp_statistics_covariance(prior_flux, rel_errors, correlation=0.3)
        
        # Off-diagonals should be non-zero
        assert V[0, 1] != 0.0


# ============================================================================
# Test Covariance Validation
# ============================================================================

class TestCovarianceValidation:
    """Test covariance matrix validation and fixing."""
    
    def test_valid_covariance(self):
        """Valid covariance should pass."""
        prior_flux = np.ones(10) * 1e12
        V = create_diagonal_prior_covariance(prior_flux, 0.10)
        
        V_fixed, issues = validate_covariance_matrix(V, "test")
        
        assert not issues["asymmetric"]
        assert not issues["not_positive_definite"]
    
    def test_asymmetric_fix(self):
        """Asymmetric matrix should be fixed by symmetrizing."""
        n = 5
        V = np.random.rand(n, n)  # Random, not symmetric
        V_fixed, issues = validate_covariance_matrix(V, "test", fix_issues=True)
        
        # Check that it was identified as asymmetric
        assert issues["asymmetric"] or np.allclose(V, V.T)
        # Fixed version should be symmetric
        assert np.allclose(V_fixed, V_fixed.T)


# ============================================================================
# Test Prior Covariance Config
# ============================================================================

class TestPriorCovarianceConfig:
    """Test PriorCovarianceConfig dataclass and builder."""
    
    def test_default_config(self):
        """Default configuration is valid."""
        config = PriorCovarianceConfig()
        assert config.model_type == PriorCovarianceModel.FRACTIONAL_REGIONAL
        assert config.fractional_uncertainty > 0
    
    def test_build_diagonal(self):
        """Build diagonal covariance from config."""
        config = PriorCovarianceConfig(
            model_type=PriorCovarianceModel.DIAGONAL,
            fractional_uncertainty=0.20
        )
        
        prior_flux = np.ones(10) * 1e12
        energy_bounds = np.logspace(-5, 7, 11)
        
        V = config.build_covariance(prior_flux, energy_bounds)
        
        assert V.shape == (10, 10)
        # Should be diagonal
        for i in range(10):
            for j in range(10):
                if i != j:
                    assert V[i, j] == 0.0
    
    def test_build_regional(self):
        """Build regional covariance from config."""
        config = PriorCovarianceConfig(
            model_type=PriorCovarianceModel.FRACTIONAL_REGIONAL,
            regions=STANDARD_REGIONS
        )
        
        prior_flux = np.ones(50) * 1e12
        energy_bounds = np.logspace(-5, 7, 51)
        
        V = config.build_covariance(prior_flux, energy_bounds)
        
        assert V.shape == (50, 50)
    
    def test_build_lethargy_correlated(self):
        """Build lethargy-correlated covariance from config."""
        config = PriorCovarianceConfig(
            model_type=PriorCovarianceModel.LETHARGY_CORRELATED,
            fractional_uncertainty=0.30,
            correlation_length=2.0
        )
        
        prior_flux = np.ones(20) * 1e12
        energy_bounds = np.logspace(-5, 7, 21)
        
        V = config.build_covariance(prior_flux, energy_bounds)
        
        # Should have non-zero off-diagonals
        assert V[0, 1] != 0.0
    
    def test_user_supplied(self):
        """User-supplied covariance is returned as-is."""
        user_cov = np.eye(10) * 1e20
        config = PriorCovarianceConfig(
            model_type=PriorCovarianceModel.USER_SUPPLIED,
            user_covariance=user_cov
        )
        
        prior_flux = np.ones(10) * 1e12
        energy_bounds = np.logspace(-5, 7, 11)
        
        V = config.build_covariance(prior_flux, energy_bounds)
        
        assert np.allclose(V, user_cov)


# ============================================================================
# Test Response Uncertainty Propagation
# ============================================================================

class TestResponseUncertaintyPropagation:
    """Test propagation of response (cross section) uncertainty."""
    
    def test_propagate_to_vy(self):
        """Response uncertainty augments V_y."""
        n_reactions = 3
        n_groups = 10
        
        R = np.ones((n_reactions, n_groups)) * 1e-24  # Response matrix
        phi = np.ones(n_groups) * 1e12  # Prior flux
        sigma_R = np.ones((n_reactions, n_groups)) * 0.05  # 5% uncertainty
        V_y = np.eye(n_reactions) * 1e-10  # Original measurement covariance
        
        V_aug = propagate_response_uncertainty_to_vy(R, phi, sigma_R, V_y)
        
        # Augmented covariance should be larger
        for i in range(n_reactions):
            assert V_aug[i, i] >= V_y[i, i]


# ============================================================================
# Test Response Uncertainty Config
# ============================================================================

class TestResponseUncertaintyConfig:
    """Test ResponseUncertaintyConfig dataclass."""
    
    def test_default_config(self):
        """Default response uncertainty configuration."""
        config = ResponseUncertaintyConfig()
        assert config.policy == ResponseUncertaintyPolicy.AUGMENT_VY
        assert config.xs_relative_uncertainty > 0
    
    def test_ignore_policy(self):
        """Can set IGNORE policy."""
        config = ResponseUncertaintyConfig(
            policy=ResponseUncertaintyPolicy.IGNORE
        )
        assert config.policy == ResponseUncertaintyPolicy.IGNORE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
