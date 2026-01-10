"""
Unit tests for neutron self-shielding corrections.

Tests the SHIELD-style self-shielding module.
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.corrections.self_shielding import (
    Geometry,
    FluxType,
    MaterialProperties,
    MonitorGeometry,
    SelfShieldingFactor,
    STANDARD_MATERIALS,
    slab_self_shielding_isotropic,
    cylinder_self_shielding_isotropic,
    sphere_self_shielding_isotropic,
)

# Conversion: 1 barn = 1e-24 cm²
BARN_TO_CM2 = 1e-24


class TestSlabSelfShielding:
    """Test slab geometry self-shielding."""
    
    def test_thin_slab(self):
        """Thin slab should have G → 1."""
        sigma_t_cm2 = 10 * BARN_TO_CM2  # 10 barns in cm²
        N = 1e22  # atoms/cm³
        thickness = 1e-5  # cm (very thin)
        
        G = slab_self_shielding_isotropic(sigma_t_cm2, N, thickness)
        
        assert G == pytest.approx(1.0, rel=0.01)
    
    def test_thick_slab(self):
        """Thick slab should have G < 1."""
        sigma_t_cm2 = 1000 * BARN_TO_CM2  # 1000 barns
        N = 1e22  # atoms/cm³
        thickness = 0.1  # cm
        
        G = slab_self_shielding_isotropic(sigma_t_cm2, N, thickness)
        
        assert G < 1.0
        assert G > 0.0
    
    def test_zero_cross_section(self):
        """Zero cross section should give G = 1."""
        G = slab_self_shielding_isotropic(0, 1e22, 0.1)
        
        assert G == pytest.approx(1.0)


class TestCylinderSelfShielding:
    """Test cylinder geometry self-shielding."""
    
    def test_thin_wire(self):
        """Thin wire should have G → 1."""
        sigma_t_cm2 = 10 * BARN_TO_CM2
        N = 1e22
        diameter = 1e-5  # Very thin
        
        G = cylinder_self_shielding_isotropic(sigma_t_cm2, N, diameter)
        
        assert G == pytest.approx(1.0, rel=0.05)
    
    def test_thick_cylinder(self):
        """Thick cylinder should have G < 1."""
        sigma_t_cm2 = 500 * BARN_TO_CM2
        N = 1e22
        diameter = 0.2
        
        G = cylinder_self_shielding_isotropic(sigma_t_cm2, N, diameter)
        
        assert G < 1.0
        assert G > 0.0


class TestSphereSelfShielding:
    """Test sphere geometry self-shielding."""
    
    def test_small_sphere(self):
        """Small sphere should have G → 1."""
        sigma_t_cm2 = 10 * BARN_TO_CM2
        N = 1e22
        diameter = 1e-5
        
        G = sphere_self_shielding_isotropic(sigma_t_cm2, N, diameter)
        
        assert G == pytest.approx(1.0, rel=0.05)
    
    def test_large_sphere(self):
        """Large sphere should have G < 1."""
        sigma_t_cm2 = 500 * BARN_TO_CM2
        N = 1e22
        diameter = 0.5
        
        G = sphere_self_shielding_isotropic(sigma_t_cm2, N, diameter)
        
        assert G < 1.0
        assert G > 0.0


class TestStandardMaterials:
    """Test standard material data."""
    
    def test_gold_exists(self):
        """Gold should be in standard materials."""
        assert "Au" in STANDARD_MATERIALS
        au = STANDARD_MATERIALS["Au"]
        
        assert au.name == "Gold"
        assert au.density_g_cm3 == pytest.approx(19.3, rel=0.01)
        assert au.atomic_mass_amu > 196
    
    def test_cobalt_exists(self):
        """Cobalt should be in standard materials."""
        assert "Co" in STANDARD_MATERIALS
        co = STANDARD_MATERIALS["Co"]
        
        assert co.name == "Cobalt"
        assert co.atomic_mass_amu == pytest.approx(58.93, rel=0.01)
    
    def test_nickel_exists(self):
        """Nickel should be in standard materials."""
        assert "Ni" in STANDARD_MATERIALS


class TestMaterialProperties:
    """Test material properties class."""
    
    def test_number_density_calculation(self):
        """Test automatic number density calculation."""
        mat = MaterialProperties(
            name="Test",
            density_g_cm3=10.0,
            atomic_mass_amu=100.0,
        )
        
        # N = ρ * N_A / A = 10 * 6.02e23 / 100 = 6.02e22
        expected = 10.0 * 6.02214076e23 / 100.0
        assert mat.number_density_per_cm3 == pytest.approx(expected, rel=0.001)


class TestMonitorGeometry:
    """Test monitor geometry class."""
    
    def test_foil_creation(self):
        """Test creating a foil geometry."""
        geom = MonitorGeometry(
            geometry_type=Geometry.SLAB,
            thickness_cm=0.0025,  # 25 μm foil
        )
        
        assert geom.thickness_cm == 0.0025
        assert geom.geometry_type == Geometry.SLAB
    
    def test_wire_creation(self):
        """Test creating a wire geometry."""
        geom = MonitorGeometry(
            geometry_type=Geometry.CYLINDER,
            thickness_cm=0.05,  # 0.5 mm diameter wire
        )
        
        assert geom.thickness_cm == 0.05
        assert geom.geometry_type == Geometry.CYLINDER
    
    def test_characteristic_length(self):
        """Test characteristic length property."""
        geom = MonitorGeometry(
            geometry_type=Geometry.SPHERE,
            thickness_cm=0.1,
        )
        
        assert geom.characteristic_length_cm == 0.1


class TestSelfShieldingFactor:
    """Test self-shielding factor class."""
    
    def test_factor_creation(self):
        """Test creating a factor."""
        factor = SelfShieldingFactor(
            energy_low_ev=1e-3,
            energy_high_ev=0.5,
            G_ss=0.95,
            G_ss_uncertainty=0.01,
        )
        
        assert factor.G_ss == 0.95
        assert factor.energy_low_ev == 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
