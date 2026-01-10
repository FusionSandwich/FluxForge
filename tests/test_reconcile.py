"""
Tests for the f/α reconciliation module.

Tests computation of flux parameters from unfolded spectra and
cross-validation with Cd-ratio derived values.
"""

import numpy as np
import pytest

from fluxforge.triga.reconcile import (
    UnfoldedFluxParameters,
    ReconciliationResult,
    ReconciliationStatus,
    compute_f_from_spectrum,
    compute_alpha_from_spectrum,
    compute_flux_parameters_from_spectrum,
    reconcile_flux_parameters,
    quick_f_check,
    E_CD,
)


class TestComputeFFromSpectrum:
    """Tests for compute_f_from_spectrum function."""
    
    def test_simple_two_group(self):
        """Test with simple two-group (thermal/epithermal) spectrum."""
        # Two groups: 0-0.55 eV (thermal), 0.55-1e6 eV (epithermal)
        flux = np.array([1e8, 1e7])  # Thermal 10x epithermal
        energy_bins = np.array([0.0, 0.55, 1e6])  # Boundaries in eV
        
        f, f_unc, phi_th, phi_epi = compute_f_from_spectrum(flux, energy_bins)
        
        # f = (1e8 * 0.55) / (1e7 * ~1e6) = 5.5e7 / 1e13 ≈ 5.5e-6
        # Actually for per-unit-energy flux:
        # phi_th = 1e8 * 0.55 = 5.5e7
        # phi_epi = 1e7 * (1e6 - 0.55) ≈ 1e13
        assert phi_th > 0
        assert phi_epi > 0
        assert f == phi_th / phi_epi
    
    def test_typical_triga_spectrum(self):
        """Test with typical TRIGA spectrum shape."""
        # Create a spectrum with known f value
        # 10 groups from 1e-5 eV to 20 MeV
        energy_bins = np.logspace(-5, 7.3, 11)  # 10 groups
        
        # Create flux that gives f ≈ 20 (typical TRIGA)
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # Maxwellian + 1/E epithermal
        flux = np.zeros(10)
        for i in range(10):
            e = e_centers[i]
            if e < 1.0:  # Thermal region - Maxwellian-like
                flux[i] = 1e12 * np.exp(-e / 0.025)  # kT = 25 meV
            else:  # Epithermal - 1/E
                flux[i] = 1e9 / e
        
        f, f_unc, phi_th, phi_epi = compute_f_from_spectrum(flux, energy_bins)
        
        # Should have reasonable f value
        assert f > 0
        assert phi_th > 0
        assert phi_epi > 0
    
    def test_uncertainty_propagation(self):
        """Test that uncertainties propagate correctly."""
        flux = np.array([1e8, 1e7])
        flux_unc = np.array([1e7, 1e6])  # 10% uncertainties
        energy_bins = np.array([0.0, 0.55, 1e6])
        
        f, f_unc, _, _ = compute_f_from_spectrum(flux, energy_bins, flux_unc)
        
        assert f_unc > 0
        assert f_unc < f  # Relative uncertainty should be < 100%
    
    def test_custom_cd_cutoff(self):
        """Test with custom Cd cutoff energy."""
        flux = np.array([1e8, 5e7, 1e7])
        energy_bins = np.array([0.0, 0.4, 0.8, 1e6])
        
        # With E_Cd = 0.5, second group straddles the boundary
        f1, _, _, _ = compute_f_from_spectrum(flux, energy_bins, e_cd=0.4)
        f2, _, _, _ = compute_f_from_spectrum(flux, energy_bins, e_cd=0.6)
        
        # Lower cutoff should give lower f (less in thermal)
        assert f1 < f2
    
    def test_zero_epithermal(self):
        """Test handling of zero epithermal flux."""
        flux = np.array([1e8])
        energy_bins = np.array([0.0, 0.5])  # All thermal
        
        f, f_unc, phi_th, phi_epi = compute_f_from_spectrum(flux, energy_bins)
        
        assert phi_epi == 0
        assert f == np.inf


class TestComputeAlphaFromSpectrum:
    """Tests for compute_alpha_from_spectrum function."""
    
    def test_pure_1_over_e(self):
        """Test α ≈ 0 for pure 1/E spectrum."""
        # Create pure 1/E spectrum in epithermal region
        energy_bins = np.logspace(0, 5, 21)  # 1 eV to 100 keV
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # φ(E) = const/E means φ×E = const
        flux = 1e10 / e_centers
        
        alpha, alpha_unc = compute_alpha_from_spectrum(flux, energy_bins)
        
        # α should be close to 0
        assert abs(alpha) < 0.1
    
    def test_hard_spectrum(self):
        """Test positive α for harder spectrum."""
        energy_bins = np.logspace(0, 5, 21)
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # φ(E) ∝ E^(-1-α) with α = 0.1 (hard)
        alpha_true = 0.1
        flux = 1e10 / (e_centers**(1 + alpha_true))
        
        alpha, alpha_unc = compute_alpha_from_spectrum(flux, energy_bins)
        
        # Should detect positive α
        assert alpha > 0
        assert abs(alpha - alpha_true) < 0.05  # Within 0.05
    
    def test_soft_spectrum(self):
        """Test negative α for softer spectrum."""
        energy_bins = np.logspace(0, 5, 21)
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # φ(E) ∝ E^(-1-α) with α = -0.1 (soft)
        alpha_true = -0.1
        flux = 1e10 / (e_centers**(1 + alpha_true))
        
        alpha, alpha_unc = compute_alpha_from_spectrum(flux, energy_bins)
        
        # Should detect negative α
        assert alpha < 0
        assert abs(alpha - alpha_true) < 0.05
    
    def test_insufficient_groups(self):
        """Test handling of too few groups in range."""
        energy_bins = np.array([0.01, 0.1])  # Only 1 group
        flux = np.array([1e10])
        
        alpha, alpha_unc = compute_alpha_from_spectrum(flux, energy_bins)
        
        # Should return default
        assert alpha == 0.0


class TestComputeFluxParametersFromSpectrum:
    """Tests for compute_flux_parameters_from_spectrum function."""
    
    def test_complete_parameters(self):
        """Test that all parameters are computed."""
        energy_bins = np.logspace(-5, 7.3, 31)
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # Create realistic spectrum
        flux = np.where(e_centers < 1.0, 1e12 * np.exp(-e_centers/0.025), 1e9/e_centers)
        
        params = compute_flux_parameters_from_spectrum(flux, energy_bins)
        
        assert params.f > 0
        assert params.f_uncertainty >= 0
        assert -0.5 < params.alpha < 0.5
        assert params.phi_thermal > 0
        assert params.phi_epithermal > 0
        assert params.n_groups == 30
        assert params.energy_range == (energy_bins[0], energy_bins[-1])
    
    def test_with_fast_flux(self):
        """Test fast flux computation (>1 MeV)."""
        energy_bins = np.logspace(-5, 7.5, 31)  # Up to ~30 MeV
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # Add fast component
        flux = np.where(e_centers < 1.0, 1e12, 1e9/e_centers)
        
        params = compute_flux_parameters_from_spectrum(flux, energy_bins)
        
        # Should have nonzero fast flux
        assert params.phi_fast > 0


class TestUnfoldedFluxParameters:
    """Tests for UnfoldedFluxParameters dataclass."""
    
    def test_repr(self):
        """Test string representation."""
        params = UnfoldedFluxParameters(
            f=20.5,
            f_uncertainty=2.0,
            alpha=0.05,
            alpha_uncertainty=0.02,
        )
        
        s = repr(params)
        assert "20.5" in s
        assert "0.05" in s


class TestReconcileFluxParameters:
    """Tests for reconcile_flux_parameters function."""
    
    def test_consistent_parameters(self):
        """Test reconciliation with consistent parameters."""
        params = UnfoldedFluxParameters(
            f=20.0,
            f_uncertainty=2.0,
            alpha=0.05,
            alpha_uncertainty=0.02,
        )
        
        result = reconcile_flux_parameters(
            params,
            cdratio_f=19.5,
            cdratio_f_unc=1.5,
            cdratio_alpha=0.04,
            cdratio_alpha_unc=0.015,
        )
        
        assert result.status == ReconciliationStatus.CONSISTENT
        assert result.f_sigma < 2.0
        assert result.alpha_sigma < 2.0
    
    def test_discrepant_f(self):
        """Test reconciliation with discrepant f values."""
        params = UnfoldedFluxParameters(
            f=30.0,
            f_uncertainty=1.0,
            alpha=0.05,
            alpha_uncertainty=0.02,
        )
        
        result = reconcile_flux_parameters(
            params,
            cdratio_f=15.0,  # Very different
            cdratio_f_unc=1.0,
            cdratio_alpha=0.05,
            cdratio_alpha_unc=0.02,
        )
        
        assert result.status == ReconciliationStatus.DISCREPANT
        assert result.f_sigma > 3.0
    
    def test_tension_parameters(self):
        """Test marginal tension case."""
        params = UnfoldedFluxParameters(
            f=25.0,
            f_uncertainty=2.0,
            alpha=0.05,
            alpha_uncertainty=0.02,
        )
        
        result = reconcile_flux_parameters(
            params,
            cdratio_f=20.0,  # 5 apart with ~2.5 combined uncertainty
            cdratio_f_unc=2.0,
            cdratio_alpha=0.05,
            cdratio_alpha_unc=0.02,
        )
        
        # ~2.5 sigma should be tension
        assert result.status in [ReconciliationStatus.TENSION, ReconciliationStatus.CONSISTENT]
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        params = UnfoldedFluxParameters(
            f=40.0,
            f_uncertainty=2.0,
            alpha=0.15,
            alpha_uncertainty=0.02,
        )
        
        result = reconcile_flux_parameters(
            params,
            cdratio_f=20.0,
            cdratio_f_unc=2.0,
            cdratio_alpha=0.02,
            cdratio_alpha_unc=0.02,
        )
        
        assert len(result.recommendations) > 0
        # Should have recommendation about f being high
        assert any("higher" in r.lower() for r in result.recommendations)
    
    def test_summary_generation(self):
        """Test summary string generation."""
        params = UnfoldedFluxParameters(f=20.0, f_uncertainty=2.0, alpha=0.05, alpha_uncertainty=0.02)
        result = reconcile_flux_parameters(params, 19.0, 1.5, 0.04, 0.015)
        
        summary = result.summary()
        assert "f/α RECONCILIATION" in summary
        assert "Unfolded" in summary
        assert "Cd-ratio" in summary
    
    def test_diagnostics_populated(self):
        """Test that diagnostics dictionary is populated."""
        params = UnfoldedFluxParameters(
            f=20.0,
            f_uncertainty=2.0,
            alpha=0.05,
            alpha_uncertainty=0.02,
            phi_thermal=1e13,
            phi_epithermal=5e11,
            n_groups=50,
        )
        
        result = reconcile_flux_parameters(params, 19.0, 1.5, 0.04, 0.015)
        
        assert 'combined_f_uncertainty' in result.diagnostics
        assert 'phi_thermal' in result.diagnostics
        assert result.diagnostics['n_groups'] == 50


class TestQuickFCheck:
    """Tests for quick_f_check function."""
    
    def test_ok_within_tolerance(self):
        """Test check passes when within tolerance."""
        energy_bins = np.array([0.0, E_CD, 1e6])
        # Make f ≈ 10
        flux = np.array([1e8 * E_CD, 1e7 * 1e6])  # Adjust for widths
        flux = np.array([1e12, 1e11])  # Simpler: thermal 10x epithermal integral
        
        # With these values: phi_th = 1e12 * 0.55 = 5.5e11
        #                   phi_epi = 1e11 * (1e6-0.55) ≈ 1e17
        # So f ≈ 5.5e11 / 1e17 = 5.5e-6 (not 10)
        # Let's try differently
        
        energy_bins = np.array([0.0, 0.55, 10.0])  # Narrow epithermal
        flux = np.array([1e10, 1e9])  # Per-unit-energy flux
        # phi_th = 1e10 * 0.55 = 5.5e9
        # phi_epi = 1e9 * 9.45 = 9.45e9
        # f = 5.5/9.45 ≈ 0.58
        
        ok, computed_f, message = quick_f_check(flux, energy_bins, expected_f=0.58, tolerance=0.2)
        
        assert ok
        assert "consistent" in message.lower()
    
    def test_not_ok_outside_tolerance(self):
        """Test check fails when outside tolerance."""
        energy_bins = np.array([0.0, 0.55, 10.0])
        flux = np.array([1e10, 1e9])
        
        ok, computed_f, message = quick_f_check(flux, energy_bins, expected_f=5.0, tolerance=0.2)
        
        assert not ok
        assert "differs" in message.lower()


class TestReconciliationIntegration:
    """Integration tests for full reconciliation workflow."""
    
    def test_full_workflow(self):
        """Test complete reconciliation workflow."""
        # Create realistic spectrum
        np.random.seed(42)
        energy_bins = np.logspace(-5, 7.3, 51)  # 50 groups
        e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        
        # Realistic TRIGA-like spectrum with f ≈ 20, α ≈ 0.02
        flux = np.zeros(50)
        for i, e in enumerate(e_centers):
            if e < 0.5:
                # Thermal: Maxwellian at 300K
                flux[i] = 1e14 * np.sqrt(e) * np.exp(-e / 0.025)
            else:
                # Epithermal: 1/E^(1+α) with α = 0.02
                flux[i] = 1e11 / (e**(1.02))
        
        flux_unc = flux * 0.05  # 5% uncertainty
        
        # Compute parameters from spectrum
        params = compute_flux_parameters_from_spectrum(flux, energy_bins, flux_unc)
        
        # Simulate Cd-ratio values (should be similar)
        cdratio_f = params.f * 1.05  # 5% different
        cdratio_f_unc = cdratio_f * 0.08  # 8% uncertainty
        cdratio_alpha = params.alpha + 0.01  # Slightly different
        cdratio_alpha_unc = 0.02
        
        # Reconcile
        result = reconcile_flux_parameters(
            params,
            cdratio_f, cdratio_f_unc,
            cdratio_alpha, cdratio_alpha_unc,
        )
        
        # Should be consistent (within uncertainties)
        assert result.status in [ReconciliationStatus.CONSISTENT, ReconciliationStatus.TENSION]
        
        # Check all fields populated
        assert result.f_unfolded > 0
        assert result.f_cdratio > 0
        assert result.summary()  # Non-empty
    
    def test_typical_triga_values(self):
        """Test with typical TRIGA reactor values."""
        # Published TRIGA values: f = 15-25, α = -0.05 to 0.05
        params = UnfoldedFluxParameters(
            f=18.5,
            f_uncertainty=1.8,
            alpha=-0.02,
            alpha_uncertainty=0.015,
        )
        
        # Cd-ratio measurement
        result = reconcile_flux_parameters(
            params,
            cdratio_f=20.0,
            cdratio_f_unc=2.5,
            cdratio_alpha=-0.01,
            cdratio_alpha_unc=0.02,
        )
        
        # Typical TRIGA measurements should be consistent
        assert result.status == ReconciliationStatus.CONSISTENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
