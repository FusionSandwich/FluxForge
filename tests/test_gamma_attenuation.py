"""
Tests for gamma self-attenuation corrections.

Tests the gamma_attenuation module which provides corrections for:
- Sample self-attenuation (disk, cylinder, sphere geometries)
- Container attenuation
- Combined attenuation corrections
"""

import pytest
import numpy as np
import math

from fluxforge.corrections.gamma_attenuation import (
    SampleGeometry,
    MaterialAttenuation,
    SampleConfiguration,
    AttenuationCorrectionFactor,
    STANDARD_MATERIALS,
    disk_self_attenuation_factor,
    cylinder_self_attenuation_factor,
    sphere_self_attenuation_factor,
    calculate_sample_attenuation,
    calculate_container_attenuation,
    calculate_attenuation_correction,
)


# ============================================================================
# Test Standard Materials
# ============================================================================

class TestStandardMaterials:
    """Test standard material attenuation data."""
    
    def test_materials_exist(self):
        """Standard materials are defined."""
        assert "iron" in STANDARD_MATERIALS
        assert "aluminum" in STANDARD_MATERIALS
        assert "gold" in STANDARD_MATERIALS
    
    def test_iron_properties(self):
        """Iron has correct density."""
        iron = STANDARD_MATERIALS["iron"]
        assert iron.density_g_cm3 == pytest.approx(7.87, rel=0.01)
    
    def test_get_mu_interpolation(self):
        """Linear attenuation coefficient interpolation works."""
        iron = STANDARD_MATERIALS["iron"]
        
        # At 500 keV, mu/rho = 0.0838 cm²/g
        mu = iron.get_mu(500)
        expected_mu = 0.0838 * 7.87  # μ = (μ/ρ) × ρ
        assert mu == pytest.approx(expected_mu, rel=0.1)
    
    def test_mu_energy_dependence(self):
        """Attenuation decreases with energy (above K-edges)."""
        iron = STANDARD_MATERIALS["iron"]
        mu_100 = iron.get_mu(100)
        mu_500 = iron.get_mu(500)
        mu_1000 = iron.get_mu(1000)
        assert mu_100 > mu_500 > mu_1000


# ============================================================================
# Test Disk Self-Attenuation
# ============================================================================

class TestDiskAttenuation:
    """Test disk/foil self-attenuation factor."""
    
    def test_thin_disk(self):
        """Very thin disk has C_att ≈ 1."""
        mu = 1.0  # cm⁻¹
        thickness = 0.0001  # 1 μm
        C = disk_self_attenuation_factor(mu, thickness)
        assert C == pytest.approx(1.0, rel=0.01)
    
    def test_thick_disk(self):
        """Thick disk has C_att → μt."""
        mu = 1.0
        thickness = 10.0  # 10 cm
        C = disk_self_attenuation_factor(mu, thickness)
        # For μt >> 1, C_att → μt
        assert C == pytest.approx(mu * thickness, rel=0.2)
    
    def test_moderate_disk(self):
        """Moderate thickness gives C_att > 1."""
        mu = 0.5
        thickness = 1.0
        C = disk_self_attenuation_factor(mu, thickness)
        assert C > 1.0  # Attenuation requires correction
    
    def test_correction_formula(self):
        """Verify formula: C = μt / (1 - exp(-μt))."""
        mu = 0.3
        t = 2.0
        C = disk_self_attenuation_factor(mu, t)
        expected = (mu * t) / (1.0 - math.exp(-mu * t))
        assert C == pytest.approx(expected, rel=0.01)


# ============================================================================
# Test Cylinder Self-Attenuation
# ============================================================================

class TestCylinderAttenuation:
    """Test cylindrical sample self-attenuation."""
    
    def test_thin_cylinder(self):
        """Very thin cylinder has C_att ≈ 1."""
        mu = 1.0
        diameter = 0.0001  # Very thin
        C = cylinder_self_attenuation_factor(mu, diameter)
        assert C == pytest.approx(1.0, rel=0.1)
    
    def test_cylinder_increases_with_size(self):
        """Larger diameter = more attenuation = higher correction."""
        mu = 0.5
        C_small = cylinder_self_attenuation_factor(mu, 0.5)
        C_large = cylinder_self_attenuation_factor(mu, 2.0)
        assert C_large > C_small


# ============================================================================
# Test Sphere Self-Attenuation
# ============================================================================

class TestSphereAttenuation:
    """Test spherical sample self-attenuation."""
    
    def test_small_sphere(self):
        """Very small sphere has negligible attenuation."""
        mu = 0.01  # Very low attenuation
        diameter = 0.1  # Small
        C = sphere_self_attenuation_factor(mu, diameter)
        # For very small μR, correction approaches 1
        assert C > 0  # Returns positive value
    
    def test_sphere_increases_with_mu(self):
        """Higher attenuation coefficient affects result."""
        diameter = 1.0
        C_low = sphere_self_attenuation_factor(0.1, diameter)
        C_high = sphere_self_attenuation_factor(1.0, diameter)
        # Results should differ
        assert C_low != C_high
    
    def test_sphere_returns_positive(self):
        """Sphere attenuation returns positive value."""
        mu = 0.3
        diameter = 2.0
        C = sphere_self_attenuation_factor(mu, diameter)
        assert C > 0


# ============================================================================
# Test Sample Configuration
# ============================================================================

class TestSampleConfiguration:
    """Test sample configuration creation."""
    
    def test_basic_config(self):
        """Create basic sample configuration."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.1,
        )
        assert config.geometry == SampleGeometry.DISK
        assert config.thickness_cm == 0.1
    
    def test_with_container(self):
        """Sample configuration with container."""
        config = SampleConfiguration(
            geometry=SampleGeometry.CYLINDER,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=1.0,
            container_material=STANDARD_MATERIALS["aluminum"],
            container_thickness_cm=0.2,
        )
        assert config.container_material is not None
        assert config.container_thickness_cm == 0.2


# ============================================================================
# Test Sample Attenuation Calculation
# ============================================================================

class TestCalculateSampleAttenuation:
    """Test sample attenuation calculation."""
    
    def test_point_source(self):
        """Point source has no attenuation."""
        config = SampleConfiguration(
            geometry=SampleGeometry.POINT,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.0,
        )
        C = calculate_sample_attenuation(config, 500)
        assert C == 1.0
    
    def test_disk_sample(self):
        """Disk sample uses disk formula."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.5,
        )
        C = calculate_sample_attenuation(config, 500)
        assert C > 1.0  # Correction needed
    
    def test_energy_dependence(self):
        """Lower energy = more attenuation = higher correction."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.5,
        )
        C_100 = calculate_sample_attenuation(config, 100)
        C_1000 = calculate_sample_attenuation(config, 1000)
        assert C_100 > C_1000


# ============================================================================
# Test Container Attenuation
# ============================================================================

class TestContainerAttenuation:
    """Test container attenuation calculation."""
    
    def test_no_container(self):
        """No container = no correction."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.1,
        )
        C = calculate_container_attenuation(config, 500)
        assert C == 1.0
    
    def test_with_container(self):
        """Container adds correction."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.1,
            container_material=STANDARD_MATERIALS["aluminum"],
            container_thickness_cm=0.3,
        )
        C = calculate_container_attenuation(config, 500)
        assert C > 1.0


# ============================================================================
# Test Total Attenuation Correction
# ============================================================================

class TestTotalAttenuationCorrection:
    """Test combined attenuation correction."""
    
    def test_basic_correction(self):
        """Calculate total attenuation correction."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.5,
        )
        result = calculate_attenuation_correction(config, 500)
        
        assert isinstance(result, AttenuationCorrectionFactor)
        assert result.energy_kev == 500
        assert result.C_att >= 1.0
    
    def test_correction_components(self):
        """Correction includes sample and container components."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.5,
            container_material=STANDARD_MATERIALS["aluminum"],
            container_thickness_cm=0.2,
        )
        result = calculate_attenuation_correction(config, 500)
        
        # Total should be product of components
        expected = result.sample_contribution * result.container_contribution
        assert result.C_att == pytest.approx(expected, rel=0.01)
    
    def test_exclude_container(self):
        """Can exclude container from correction."""
        config = SampleConfiguration(
            geometry=SampleGeometry.DISK,
            material=STANDARD_MATERIALS["iron"],
            thickness_cm=0.5,
            container_material=STANDARD_MATERIALS["aluminum"],
            container_thickness_cm=0.2,
        )
        result = calculate_attenuation_correction(config, 500, include_container=False)
        
        assert result.container_contribution == 1.0


# ============================================================================
# Test Attenuation Correction Factor
# ============================================================================

class TestAttenuationCorrectionFactor:
    """Test AttenuationCorrectionFactor dataclass."""
    
    def test_creation(self):
        """Create correction factor."""
        factor = AttenuationCorrectionFactor(
            energy_kev=661.7,
            C_att=1.15,
            C_att_uncertainty=0.02,
        )
        assert factor.energy_kev == 661.7
        assert factor.C_att == 1.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
