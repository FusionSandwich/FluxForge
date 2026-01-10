"""
Tests for Neutron Corrections Module

Tests for self-shielding and cadmium cover corrections in reactor dosimetry.
"""

import pytest
import numpy as np

from fluxforge.physics.neutron_corrections import (
    SelfShieldingResult,
    CdCoverResult,
    NeutronCorrections,
    calculate_thermal_self_shielding_factor,
    calculate_epithermal_self_shielding_factor,
    calculate_self_shielding,
    calculate_cd_ratio,
    extract_thermal_epithermal_components,
    calculate_all_corrections,
    THERMAL_CROSS_SECTIONS,
    RESONANCE_INTEGRALS,
    ATOMIC_WEIGHTS,
)


# =============================================================================
# Test Physical Constants
# =============================================================================

class TestPhysicalConstants:
    """Tests for physical constants dictionaries."""
    
    def test_thermal_cross_sections_exist(self):
        """Test that thermal cross sections are defined for common reactions."""
        # Note: Keys include full reaction names
        assert any("Co-59" in k for k in THERMAL_CROSS_SECTIONS)
        assert any("Au-197" in k for k in THERMAL_CROSS_SECTIONS)
        
        # Check that dict is not empty
        assert len(THERMAL_CROSS_SECTIONS) > 0
    
    def test_resonance_integrals_exist(self):
        """Test that resonance integrals are defined."""
        assert any("Co-59" in k for k in RESONANCE_INTEGRALS)
        assert any("Au-197" in k for k in RESONANCE_INTEGRALS)
    
    def test_atomic_weights_exist(self):
        """Test that atomic weights are defined."""
        # Keys include isotope notation
        assert any("Co" in k for k in ATOMIC_WEIGHTS)
        assert any("Au" in k for k in ATOMIC_WEIGHTS)
        assert any("Fe" in k for k in ATOMIC_WEIGHTS)


# =============================================================================
# Test Self-Shielding Calculations
# =============================================================================

class TestSelfShielding:
    """Tests for self-shielding factor calculations."""
    
    def test_thermal_self_shielding_thin_sample(self):
        """Test that thin samples have self-shielding factor ~1."""
        # Using the actual API: sigma_0, thickness, n_density, geometry
        f = calculate_thermal_self_shielding_factor(
            sigma_0=100.0,  # barns
            thickness=0.001,  # cm - very thin
            n_density=5.9e22,  # atoms/cm^3 for Au
            geometry="foil",
        )
        
        # For thin samples, f should be close to 1
        assert 0.90 < f <= 1.0
    
    def test_thermal_self_shielding_thick_sample(self):
        """Test that thick samples have self-shielding factor < 1."""
        f = calculate_thermal_self_shielding_factor(
            sigma_0=100.0,
            thickness=0.1,  # thicker sample
            n_density=5.9e22,
            geometry="foil",
        )
        
        # For thick samples, f should be less than 1
        assert 0.0 < f < 0.9
    
    def test_thermal_self_shielding_zero_cross_section(self):
        """Test that zero cross section gives f = 1."""
        f = calculate_thermal_self_shielding_factor(
            sigma_0=0.0,
            thickness=1.0,
            n_density=1e22,
        )
        
        assert f == pytest.approx(1.0, rel=0.01)
    
    def test_calculate_self_shielding_function(self):
        """Test the high-level self-shielding function."""
        # Use actual API parameters
        result = calculate_self_shielding(
            reaction="Co-59(n,g)Co-60",
            thickness=0.01,
            density=8.9,
            geometry="foil",
        )
        
        assert isinstance(result, SelfShieldingResult)
        assert 0.0 < result.G_th <= 1.0
        assert 0.0 < result.G_epi <= 1.0


# =============================================================================
# Test Cadmium Cover Corrections
# =============================================================================

class TestCdCoverCorrections:
    """Tests for cadmium cover ratio and thermal/epithermal extraction."""
    
    def test_cd_ratio_calculation(self):
        """Test Cd ratio calculation from bare and Cd-covered samples."""
        bare_rate = 100.0
        cd_covered_rate = 30.0
        
        R_Cd, R_Cd_unc = calculate_cd_ratio(bare_rate, cd_covered_rate)
        
        # Cd ratio should be bare/covered
        assert R_Cd == pytest.approx(100.0 / 30.0)
    
    def test_cd_ratio_with_uncertainties(self):
        """Test Cd ratio with uncertainty propagation."""
        R_Cd, R_Cd_unc = calculate_cd_ratio(
            activity_bare=100.0,
            activity_cd_covered=25.0,
            uncertainty_bare=5.0,
            uncertainty_covered=2.5,
        )
        
        # Check that it returns the ratio
        assert R_Cd == pytest.approx(4.0)
        # Check uncertainty is non-negative
        assert R_Cd_unc >= 0
    
    def test_thermal_epithermal_extraction(self):
        """Test extraction of thermal and epithermal components."""
        result = extract_thermal_epithermal_components(
            activity_bare=100.0,
            activity_cd_covered=20.0,
            reaction="Co-59(n,g)Co-60",
        )
        
        assert isinstance(result, CdCoverResult)
        
        # Cd ratio should be correct
        assert result.R_Cd == pytest.approx(100.0 / 20.0)
        # Should have thermal and epithermal fractions
        assert result.thermal_fraction >= 0
        assert result.epithermal_fraction >= 0


# =============================================================================
# Test NeutronCorrections - use calculate_all_corrections function
# =============================================================================

class TestNeutronCorrectionsClass:
    """Tests for the NeutronCorrections and calculate_all_corrections."""
    
    def test_calculate_all_corrections(self):
        """Test calculating all corrections."""
        result = calculate_all_corrections(
            reaction="Co-59(n,g)Co-60",
            thickness=0.01,
            density=8.9,
            geometry="foil",
            activity_bare=100.0,
            activity_cd_covered=20.0,
        )
        
        assert isinstance(result, NeutronCorrections)
        assert result.self_shielding is not None
        assert result.cd_cover is not None
        assert result.total_correction > 0
    
    def test_self_shielding_only(self):
        """Test self-shielding without Cd cover."""
        result = calculate_all_corrections(
            reaction="Co-59(n,g)Co-60",
            thickness=0.01,
            density=8.9,
        )
        
        assert result.self_shielding is not None
        assert result.self_shielding.G_th > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
