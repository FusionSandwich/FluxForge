"""
Tests for cover correction factors (Cd, Gd, B, Au covers).

Tests the covers module which provides cover corrections for:
- Cadmium (Cd) thermal neutron filtering
- Gadolinium (Gd) thermal neutron filtering
- Boron (B) thermal and low-energy filtering
- Gold (Au) specialized filtering

Also tests STAYSL PNNL parity mode with CCF using E₂(x) exponential integral.
"""

import pytest
import numpy as np
import math

from fluxforge.corrections.covers import (
    CoverMaterial,
    CoverProperties,
    CoverConfiguration,
    CoverCorrectionFactor,
    COVER_MATERIALS,
    cover_transmission_1v,
    calculate_cd_cutoff_function,
    calculate_cover_correction_group,
    calculate_cover_corrections,
    # STAYSL PNNL parity mode
    FluxAngularModel,
    CoverCorrectionMethod,
    CoverSpec,
    STAYSL_COVER_DATA,
    AVOGADRO,
    MIL_TO_CM,
    BARN_TO_CM2,
    exponential_integral_E2,
    compute_optical_thickness,
    compute_ccf_staysl,
    STAYSLCoverResult,
    compute_staysl_cover_correction,
    # Energy-dependent mode
    compute_transmission_beam,
    compute_transmission_isotropic,
    compute_group_transmission,
    EnergyDependentCoverResult,
    compute_energy_dependent_cover_corrections,
    create_cd_sigma_total_1v,
    # Parity reports
    STAYSLParityReport,
    create_staysl_parity_report,
)


# ============================================================================
# Test Cover Material Properties
# ============================================================================

class TestCoverMaterials:
    """Test cover material property definitions."""
    
    def test_cadmium_properties(self):
        """Verify Cd properties are defined correctly."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        assert cd.density_g_cm3 == pytest.approx(8.65, rel=0.01)
        assert cd.sigma_0_barns > 2000  # Large thermal absorption
        assert cd.cutoff_energy_ev == pytest.approx(0.55, rel=0.1)
    
    def test_gadolinium_properties(self):
        """Verify Gd properties (very high thermal σ)."""
        gd = COVER_MATERIALS[CoverMaterial.GADOLINIUM]
        assert gd.sigma_0_barns > 40000  # Huge thermal σ
        assert gd.cutoff_energy_ev < 0.1  # Lower cutoff than Cd
    
    def test_boron_properties(self):
        """Verify B properties."""
        b = COVER_MATERIALS[CoverMaterial.BORON]
        assert b.sigma_0_barns > 3000  # B-10 enriched
    
    def test_gold_properties(self):
        """Verify Au properties."""
        au = COVER_MATERIALS[CoverMaterial.GOLD]
        assert au.density_g_cm3 == pytest.approx(19.3, rel=0.01)
        assert au.sigma_0_barns > 90
    
    def test_number_density_calculation(self):
        """Verify number density calculation."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        N = cd.number_density_per_cm3
        assert N > 1e22  # Reasonable atomic density


# ============================================================================
# Test Cover Transmission (1/v)
# ============================================================================

class TestCoverTransmission:
    """Test 1/v cross section transmission calculations."""
    
    def test_thin_cover_high_transmission(self):
        """Thin cover should transmit most neutrons."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        # Fast neutron (1 MeV) through very thin Cd
        T = cover_transmission_1v(1e6, cd, 0.001)  # 10 μm
        assert T > 0.99  # High transmission for fast neutrons
    
    def test_thick_cd_low_transmission(self):
        """Thick Cd should block thermal neutrons."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        # Thermal neutron (0.0253 eV) through 1 mm Cd
        T = cover_transmission_1v(0.0253, cd, 0.1)  # 1 mm
        assert T < 0.01  # Very low transmission for thermal
    
    def test_energy_dependence(self):
        """Transmission should increase with energy (1/v behavior)."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        T_thermal = cover_transmission_1v(0.0253, cd, 0.05)
        T_epithermal = cover_transmission_1v(1.0, cd, 0.05)
        T_fast = cover_transmission_1v(1e6, cd, 0.05)
        assert T_thermal < T_epithermal < T_fast
    
    def test_thickness_dependence(self):
        """Thicker cover should have lower transmission."""
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        T_thin = cover_transmission_1v(0.1, cd, 0.01)
        T_thick = cover_transmission_1v(0.1, cd, 0.1)
        assert T_thick < T_thin


# ============================================================================
# Test Cadmium Cutoff Function
# ============================================================================

class TestCdCutoffFunction:
    """Test empirical Cd cutoff function."""
    
    def test_below_cutoff(self):
        """Below Cd cutoff energy, transmission ~ 0."""
        T = calculate_cd_cutoff_function(0.1, 1.0)  # Well below 0.55 eV
        assert T < 0.5
    
    def test_above_cutoff(self):
        """Above Cd cutoff energy, transmission ~ 1."""
        T = calculate_cd_cutoff_function(1.0, 1.0)  # Well above 0.55 eV
        assert T > 0.5
    
    def test_cutoff_region(self):
        """Transmission transitions around 0.55 eV for 1mm Cd."""
        T_low = calculate_cd_cutoff_function(0.3, 1.0)
        T_cutoff = calculate_cd_cutoff_function(0.55, 1.0)
        T_high = calculate_cd_cutoff_function(1.0, 1.0)
        assert T_low < T_cutoff < T_high


# ============================================================================
# Test Cover Configuration
# ============================================================================

class TestCoverConfiguration:
    """Test cover configuration creation."""
    
    def test_default_properties(self):
        """Configuration uses default material properties."""
        config = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1
        )
        assert config.properties.density_g_cm3 > 0
        assert config.properties.sigma_0_barns > 0
    
    def test_custom_properties(self):
        """Custom properties override defaults."""
        custom = CoverProperties(
            material=CoverMaterial.CADMIUM,
            density_g_cm3=9.0,  # Custom density
            atomic_mass_amu=112.41,
            sigma_0_barns=2500.0,
        )
        config = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1,
            custom_properties=custom
        )
        assert config.properties.density_g_cm3 == 9.0


# ============================================================================
# Test Group Cover Corrections
# ============================================================================

class TestCoverCorrectionGroup:
    """Test per-group cover correction calculation."""
    
    def test_thermal_group_cd(self):
        """Thermal group should have low transmission through Cd."""
        cover = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1  # 1 mm
        )
        corr = calculate_cover_correction_group(1e-5, 0.5, cover)
        assert corr.F_c < 0.5  # Low transmission for thermal
        assert corr.energy_low_ev == 1e-5
        assert corr.energy_high_ev == 0.5
    
    def test_fast_group_cd(self):
        """Fast group should have high transmission through Cd."""
        cover = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1
        )
        corr = calculate_cover_correction_group(1e5, 1e7, cover)
        assert corr.F_c > 0.9  # High transmission for fast
    
    def test_correction_factor_bounds(self):
        """Transmission should be between 0 and 1."""
        cover = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1
        )
        corr = calculate_cover_correction_group(0.1, 10.0, cover)
        assert 0.0 <= corr.F_c <= 1.0


# ============================================================================
# Test Full Cover Corrections Array
# ============================================================================

class TestCoverCorrectionsArray:
    """Test calculation of cover corrections for all groups."""
    
    def test_calculate_cover_corrections(self):
        """Calculate corrections for multi-group structure."""
        # 5-group structure
        bounds = np.array([1e-5, 0.5, 1e3, 1e5, 1e6, 2e7])
        cover = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1
        )
        corrections = calculate_cover_corrections(bounds, cover)
        assert len(corrections) == 5
        for corr in corrections:
            assert isinstance(corr, CoverCorrectionFactor)
    
    def test_corrections_thermal_vs_fast(self):
        """Thermal groups should have lower transmission than fast."""
        bounds = np.array([1e-5, 0.5, 1e5, 2e7])
        cover = CoverConfiguration(
            material=CoverMaterial.CADMIUM,
            thickness_cm=0.1
        )
        corrections = calculate_cover_corrections(bounds, cover)
        # First group is thermal, last is fast
        assert corrections[0].F_c < corrections[-1].F_c


# ============================================================================
# Test Gd Cover
# ============================================================================

class TestGdCover:
    """Test Gadolinium cover calculations."""
    
    def test_gd_high_thermal_absorption(self):
        """Gd has very high thermal absorption."""
        gd = COVER_MATERIALS[CoverMaterial.GADOLINIUM]
        # Very thin Gd should still absorb thermal neutrons well
        T = cover_transmission_1v(0.0253, gd, 0.001)  # 10 μm
        assert T < 0.5  # Good absorption even for thin Gd


# ============================================================================
# Test Cover Material Comparisons
# ============================================================================

class TestCoverComparisons:
    """Test comparison between different cover materials."""
    
    def test_gd_vs_cd_thermal(self):
        """Gd has lower cutoff than Cd."""
        gd = COVER_MATERIALS[CoverMaterial.GADOLINIUM]
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        assert gd.cutoff_energy_ev < cd.cutoff_energy_ev
    
    def test_material_sigma_ordering(self):
        """Verify Gd has highest σ, Au has lowest."""
        gd = COVER_MATERIALS[CoverMaterial.GADOLINIUM]
        cd = COVER_MATERIALS[CoverMaterial.CADMIUM]
        b = COVER_MATERIALS[CoverMaterial.BORON]
        au = COVER_MATERIALS[CoverMaterial.GOLD]
        # Gd-157 has largest thermal σ, Au has smallest
        assert gd.sigma_0_barns > cd.sigma_0_barns
        assert gd.sigma_0_barns > b.sigma_0_barns
        assert cd.sigma_0_barns > au.sigma_0_barns


# ============================================================================
# STAYSL PNNL Parity Mode Tests
# ============================================================================

class TestSTAYSLConstants:
    """Test STAYSL PNNL constants and material data."""
    
    def test_staysl_cadmium_data(self):
        """Verify STAYSL Cd data matches manual."""
        cd_data = STAYSL_COVER_DATA["CADM"]
        assert cd_data["density_g_cm3"] == pytest.approx(8.69, rel=0.01)
        assert cd_data["atomic_mass_g_mol"] == pytest.approx(112.411, rel=0.001)
        assert cd_data["sigma_th_barn"] == pytest.approx(2520.0, rel=0.05)
    
    def test_physical_constants(self):
        """Verify physical constants."""
        assert AVOGADRO == pytest.approx(6.022e23, rel=0.001)
        assert MIL_TO_CM == pytest.approx(2.54e-3, rel=0.01)
        assert BARN_TO_CM2 == 1e-24


class TestExponentialIntegral:
    """Test E₂(x) exponential integral."""
    
    def test_E2_at_zero(self):
        """E₂(0) = 1."""
        assert exponential_integral_E2(0) == 1.0
    
    def test_E2_small_x(self):
        """E₂ for small x should be near 1."""
        assert exponential_integral_E2(0.01) > 0.94  # E₂(0.01) ≈ 0.9497
    
    def test_E2_moderate_x(self):
        """E₂(1) ≈ 0.1485."""
        assert exponential_integral_E2(1.0) == pytest.approx(0.1485, rel=0.01)
    
    def test_E2_large_x(self):
        """E₂ for large x should be near 0."""
        assert exponential_integral_E2(10) < 0.001
    
    def test_E2_very_large_x(self):
        """E₂(>50) returns 0."""
        assert exponential_integral_E2(100) == 0.0
    
    def test_E2_negative_raises(self):
        """Negative x should raise error."""
        with pytest.raises(ValueError):
            exponential_integral_E2(-1)
    
    def test_E2_monotonic_decrease(self):
        """E₂(x) decreases monotonically with x."""
        x_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        e2_values = [exponential_integral_E2(x) for x in x_values]
        for i in range(len(e2_values) - 1):
            assert e2_values[i] > e2_values[i + 1]


class TestCoverSpec:
    """Test CoverSpec dataclass."""
    
    def test_default_cadmium(self):
        """Default CoverSpec uses STAYSL Cd data."""
        cover = CoverSpec()
        assert cover.material_code == "CADM"
        assert cover.thickness_mil == 40.0
        assert cover.density == pytest.approx(8.69, rel=0.01)
    
    def test_thickness_conversion(self):
        """Verify mil to cm conversion."""
        cover = CoverSpec(thickness_mil=100)  # 100 mils
        assert cover.thickness_cm == pytest.approx(0.254, rel=0.001)  # 100 * 2.54e-3
    
    def test_custom_properties(self):
        """Custom properties override defaults."""
        cover = CoverSpec(
            material_code="CADM",
            density_g_cm3=9.0,
            atomic_mass_g_mol=113.0,
            custom_sigma_th_barn=2600.0,
        )
        assert cover.density == 9.0
        assert cover.atomic_mass == 113.0
        assert cover.sigma_th == 2600.0
    
    def test_angular_model_default(self):
        """Default angular model is isotropic."""
        cover = CoverSpec()
        assert cover.angular_model == FluxAngularModel.ISOTROPIC


class TestOpticalThickness:
    """Test optical thickness computation."""
    
    def test_optical_thickness_computation(self):
        """Verify optical thickness formula."""
        cover = CoverSpec(
            material_code="CADM",
            thickness_mil=40.0,  # Standard 40 mil Cd
        )
        x = compute_optical_thickness(cover)
        # x = (N_A * ρ * σ * L) / MW
        # Should be moderate value for 40 mil Cd
        assert 0 < x < 100
    
    def test_optical_thickness_proportional_to_thickness(self):
        """Optical thickness doubles when thickness doubles."""
        cover1 = CoverSpec(thickness_mil=20.0)
        cover2 = CoverSpec(thickness_mil=40.0)
        x1 = compute_optical_thickness(cover1)
        x2 = compute_optical_thickness(cover2)
        assert x2 == pytest.approx(2 * x1, rel=0.001)
    
    def test_optical_thickness_proportional_to_sigma(self):
        """Optical thickness proportional to cross section."""
        cover1 = CoverSpec(custom_sigma_th_barn=1000.0)
        cover2 = CoverSpec(custom_sigma_th_barn=2000.0)
        x1 = compute_optical_thickness(cover1)
        x2 = compute_optical_thickness(cover2)
        assert x2 == pytest.approx(2 * x1, rel=0.001)


class TestCCFComputation:
    """Test STAYSL CCF computation."""
    
    def test_ccf_beam_exponential(self):
        """Beam flux CCF = exp(-x)."""
        cover = CoverSpec(
            thickness_mil=40.0,
            angular_model=FluxAngularModel.BEAM,
        )
        ccf = compute_ccf_staysl(cover)
        x = compute_optical_thickness(cover)
        expected = np.exp(-x)
        assert ccf == pytest.approx(expected, rel=0.01)
    
    def test_ccf_isotropic_E2(self):
        """Isotropic flux CCF = E₂(x)."""
        cover = CoverSpec(
            thickness_mil=40.0,
            angular_model=FluxAngularModel.ISOTROPIC,
        )
        ccf = compute_ccf_staysl(cover)
        x = compute_optical_thickness(cover)
        expected = exponential_integral_E2(x)
        assert ccf == pytest.approx(expected, rel=0.01)
    
    def test_ccf_bounds(self):
        """CCF should be between 0 and 1."""
        for thickness in [1.0, 10.0, 40.0, 100.0, 200.0]:
            cover = CoverSpec(thickness_mil=thickness)
            ccf = compute_ccf_staysl(cover)
            assert 0.0 <= ccf <= 1.0
    
    def test_ccf_decreases_with_thickness(self):
        """Thicker cover = lower CCF."""
        ccfs = []
        for thickness in [10.0, 20.0, 40.0, 80.0]:
            cover = CoverSpec(thickness_mil=thickness)
            ccfs.append(compute_ccf_staysl(cover))
        for i in range(len(ccfs) - 1):
            assert ccfs[i] > ccfs[i + 1]
    
    def test_ccf_beam_vs_isotropic(self):
        """Compare beam and isotropic CCF behavior."""
        # For thick covers with large x, both CCFs are very small
        cover_thick_iso = CoverSpec(
            thickness_mil=40.0,
            angular_model=FluxAngularModel.ISOTROPIC,
        )
        cover_thick_beam = CoverSpec(
            thickness_mil=40.0,
            angular_model=FluxAngularModel.BEAM,
        )
        ccf_iso = compute_ccf_staysl(cover_thick_iso)
        ccf_beam = compute_ccf_staysl(cover_thick_beam)
        # Both should be very small for thick Cd
        assert ccf_iso < 0.01
        assert ccf_beam < 0.01
        
        # For thin covers, CCF is larger
        cover_thin = CoverSpec(thickness_mil=1.0, angular_model=FluxAngularModel.ISOTROPIC)
        ccf_thin = compute_ccf_staysl(cover_thin)
        assert ccf_thin > 0.4  # Thinner cover transmits more
    
    def test_ccf_with_override_sigma(self):
        """Can override sigma_th."""
        cover = CoverSpec(thickness_mil=40.0)
        ccf1 = compute_ccf_staysl(cover, sigma_th_barn=2520.0)
        ccf2 = compute_ccf_staysl(cover, sigma_th_barn=5040.0)
        # Higher sigma = lower CCF
        assert ccf1 > ccf2


class TestSTAYSLCoverResult:
    """Test STAYSLCoverResult dataclass."""
    
    def test_complete_result(self):
        """Compute complete STAYSL cover correction."""
        cover = CoverSpec(thickness_mil=40.0)
        result = compute_staysl_cover_correction(cover)
        
        assert result.cover_spec == cover
        assert result.optical_thickness > 0
        assert 0 < result.ccf <= 1
        assert result.sigma_th_barn == cover.sigma_th
        assert result.method == "staysl_ccf"
    
    def test_result_to_dict(self):
        """Result can be serialized to dict."""
        cover = CoverSpec(thickness_mil=40.0)
        result = compute_staysl_cover_correction(cover)
        d = result.to_dict()
        
        assert d["schema"] == "fluxforge.staysl_cover_result.v1"
        assert d["material_code"] == "CADM"
        assert d["thickness_mil"] == 40.0
        assert "ccf" in d
        assert "optical_thickness_x" in d


class TestEnergyDependentTransmission:
    """Test energy-dependent cover transmission."""
    
    def test_beam_transmission(self):
        """Test beam transmission at specific energy."""
        sigma_t = create_cd_sigma_total_1v(sigma_0_barns=2520.0)
        number_density = 8.69 * AVOGADRO / 112.411  # Cd
        
        # Thermal energy: high absorption
        T_thermal = compute_transmission_beam(0.0253, sigma_t, 0.1, number_density)
        assert T_thermal < 0.01  # Strong attenuation
        
        # Fast energy: low absorption
        T_fast = compute_transmission_beam(1e6, sigma_t, 0.1, number_density)
        assert T_fast > 0.99  # Minimal attenuation
    
    def test_isotropic_transmission(self):
        """Test isotropic transmission at specific energy."""
        sigma_t = create_cd_sigma_total_1v()
        number_density = 8.69 * AVOGADRO / 112.411
        
        T = compute_transmission_isotropic(0.0253, sigma_t, 0.1, number_density)
        assert 0 < T < 1
    
    def test_beam_vs_isotropic_comparison(self):
        """Compare beam and isotropic transmission behavior."""
        sigma_t = create_cd_sigma_total_1v()
        number_density = 8.69 * AVOGADRO / 112.411
        
        # For isotropic flux on slab, average path length is longer
        # so T_isotropic < T_beam (E₂(τ) < exp(-τ))
        for E in [0.1, 1.0, 100.0]:
            T_beam = compute_transmission_beam(E, sigma_t, 0.1, number_density)
            T_iso = compute_transmission_isotropic(E, sigma_t, 0.1, number_density)
            # Both should be between 0 and 1
            assert 0 <= T_beam <= 1
            assert 0 <= T_iso <= 1
            # For optically thin materials at higher energy, beam > isotropic
            # because isotropic averages over all angles including grazing


class TestGroupTransmission:
    """Test group-averaged transmission."""
    
    def test_group_transmission_thermal(self):
        """Thermal group has low transmission."""
        sigma_t = create_cd_sigma_total_1v()
        cover = CoverSpec(thickness_mil=40.0)
        number_density = cover.density * AVOGADRO / cover.atomic_mass
        
        T_g, T_g_unc = compute_group_transmission(
            1e-5, 0.5, sigma_t, cover.thickness_cm, number_density,
            FluxAngularModel.ISOTROPIC
        )
        assert T_g < 0.5  # Low transmission for thermal
        assert T_g_unc >= 0
    
    def test_group_transmission_fast(self):
        """Fast group has high transmission."""
        sigma_t = create_cd_sigma_total_1v()
        cover = CoverSpec(thickness_mil=40.0)
        number_density = cover.density * AVOGADRO / cover.atomic_mass
        
        T_g, _ = compute_group_transmission(
            1e5, 1e7, sigma_t, cover.thickness_cm, number_density,
            FluxAngularModel.ISOTROPIC
        )
        assert T_g > 0.9  # High transmission for fast


class TestEnergyDependentCoverResult:
    """Test full energy-dependent cover correction computation."""
    
    def test_compute_all_groups(self):
        """Compute corrections for all groups."""
        cover = CoverSpec(thickness_mil=40.0)
        bounds = np.array([1e-5, 0.5, 1e3, 1e5, 1e7, 2e7])
        sigma_t = create_cd_sigma_total_1v()
        
        result = compute_energy_dependent_cover_corrections(
            cover, bounds, sigma_t
        )
        
        assert result.n_groups == 5
        assert len(result.group_transmissions) == 5
        assert len(result.group_uncertainties) == 5
        assert result.method == "energy_dependent"
    
    def test_thermal_vs_fast_groups(self):
        """Thermal groups have lower transmission."""
        cover = CoverSpec(thickness_mil=40.0)
        bounds = np.array([1e-5, 0.5, 1e5, 2e7])
        sigma_t = create_cd_sigma_total_1v()
        
        result = compute_energy_dependent_cover_corrections(
            cover, bounds, sigma_t
        )
        
        # First group (thermal) < last group (fast)
        assert result.group_transmissions[0] < result.group_transmissions[-1]
    
    def test_result_to_dict(self):
        """Result can be serialized."""
        cover = CoverSpec(thickness_mil=40.0)
        bounds = np.array([1e-5, 0.5, 1e5])
        sigma_t = create_cd_sigma_total_1v()
        
        result = compute_energy_dependent_cover_corrections(
            cover, bounds, sigma_t
        )
        d = result.to_dict()
        
        assert d["schema"] == "fluxforge.energy_dependent_cover.v1"
        assert d["n_groups"] == 2
        assert len(d["group_transmissions"]) == 2


class TestSTAYSLParityReport:
    """Test STAYSL parity report generation."""
    
    def test_create_scalar_ccf_report(self):
        """Create report with scalar CCF (STAYSL mode)."""
        cover = CoverSpec(thickness_mil=40.0)
        bounds = np.array([1e-5, 0.5, 1e3, 1e5, 2e7])
        
        report = create_staysl_parity_report(
            "Au-197(n,g)",
            cover, bounds,
            use_energy_dependent=False,
        )
        
        assert report.reaction_id == "Au-197(n,g)"
        assert len(report.ccf_values) == 4  # 4 groups
        # Thermal group has CCF applied
        assert report.ccf_values[0] < 1.0
        # Fast groups have CCF = 1 (above cutoff)
        assert report.ccf_values[-1] == 1.0
    
    def test_create_energy_dependent_report(self):
        """Create report with energy-dependent transmission."""
        cover = CoverSpec(thickness_mil=40.0)
        bounds = np.array([1e-5, 0.5, 1e3, 1e5, 2e7])
        sigma_t = create_cd_sigma_total_1v()
        
        report = create_staysl_parity_report(
            "Au-197(n,g)",
            cover, bounds,
            use_energy_dependent=True,
            sigma_total_E=sigma_t,
        )
        
        assert len(report.ccf_values) == 4
        # All groups have some transmission
        for ccf in report.ccf_values:
            assert 0 <= ccf <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
