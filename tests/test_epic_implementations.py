"""
Comprehensive Tests for New Epic Implementations

Tests for:
- Epic R: auto_calibration.py (R1.5)
- Epic S: gma_workflow.py (S1.6, S1.8)
- Epic T: stacked_target.py (T1.4-T1.6), spectrum_export.py (T1.9)
- Epic U: stopping_power.py (U1.1-U1.6)
- Epic W: advanced_unfolding.py (W1.5, W1.6)
- Epic X: gamma_spectrum.py metastable support (X1.8)
- Epic Y: advanced_unfolding.py (Y1.1, Y1.7)
"""

import pytest
import math
import tempfile
from pathlib import Path
from typing import List


# =============================================================================
# EPIC R TESTS: AUTO-CALIBRATION
# =============================================================================

class TestAutoCalibration:
    """Tests for R1.5 auto-calibration functionality."""
    
    def test_calibration_sources_database(self):
        """Test that calibration sources are properly defined."""
        from fluxforge.analysis.auto_calibration import CALIBRATION_SOURCES
        
        # Check common sources exist
        assert 'Co-60' in CALIBRATION_SOURCES
        assert 'Cs-137' in CALIBRATION_SOURCES
        assert 'Eu-152' in CALIBRATION_SOURCES
        
        # Check Co-60 has correct main lines
        co60 = CALIBRATION_SOURCES['Co-60']
        energies = [line[0] for line in co60]
        assert any(abs(E - 1173.2) < 1 for E in energies)
        assert any(abs(E - 1332.5) < 1 for E in energies)
    
    def test_find_peak_candidates(self):
        """Test peak finding in synthetic spectrum."""
        from fluxforge.analysis.auto_calibration import find_peak_candidates
        import numpy as np
        
        # Create spectrum with obvious peaks
        n_channels = 1000
        counts = np.ones(n_channels) * 100  # Background
        
        # Add Gaussian peaks
        for ch in [200, 500, 800]:
            for i in range(-20, 21):
                if 0 <= ch + i < n_channels:
                    counts[ch + i] += 1000 * np.exp(-i**2 / 50)
        
        peaks = find_peak_candidates(counts, threshold_sigma=5.0, min_counts=500)
        
        assert len(peaks) >= 3
        # Peaks should be near our expected channels
        peak_channels = [p[0] for p in peaks]
        assert any(abs(ch - 200) < 5 for ch in peak_channels)
        assert any(abs(ch - 500) < 5 for ch in peak_channels)
        assert any(abs(ch - 800) < 5 for ch in peak_channels)
    
    def test_auto_calibrate_synthetic(self):
        """Test auto-calibration on synthetic Co-60 spectrum."""
        from fluxforge.analysis.auto_calibration import auto_calibrate
        import numpy as np
        
        # True calibration: E = 0.5 + 3.0 * ch
        n_channels = 1000
        counts = np.ones(n_channels) * 50  # Background
        
        # Add Co-60 peaks at correct energies
        E_peaks = [1173.23, 1332.49]
        for E in E_peaks:
            ch = int((E - 0.5) / 3.0)
            if 0 < ch < n_channels - 20:
                for i in range(-15, 16):
                    if 0 <= ch + i < n_channels:
                        counts[ch + i] += 5000 * np.exp(-i**2 / 30)
        
        # min_peaks=2 since Co-60 only has 2 main lines
        result = auto_calibrate(counts, isotopes=['Co-60'], initial_gain=3.0, min_peaks=2)
        
        assert result is not None
        assert result.success
        # Check calibration coefficients are reasonable
        # E = a + b * ch, should get a ≈ 0.5, b ≈ 3.0
        a, b = result.coefficients[:2]
        assert abs(a - 0.5) < 50  # Within 50 keV
        assert abs(b - 3.0) < 0.5  # Within 0.5 keV/ch


# =============================================================================
# EPIC S TESTS: GMA WORKFLOW
# =============================================================================

class TestGMAWorkflow:
    """Tests for S1.6 GMA workflow and S1.8 JSON database."""
    
    def test_experiment_dataclass(self):
        """Test Experiment data structure."""
        from fluxforge.evaluation.gma_workflow import Experiment
        
        exp = Experiment(
            name="Test-1",
            value=100.5,
            uncertainty=2.5,
            energy_MeV=1.5,
            reaction="Fe-56(n,p)",
            reference="Smith et al. 2024"
        )
        
        assert exp.value == 100.5
        assert exp.uncertainty == 2.5
        
        # Test to_dict
        d = exp.to_dict()
        assert d['name'] == "Test-1"
        assert d['reaction'] == "Fe-56(n,p)"
        
        # Test from_dict roundtrip
        exp2 = Experiment.from_dict(d)
        assert exp2.value == exp.value
    
    def test_experimental_database_json(self):
        """Test JSON export/import of experimental database."""
        from fluxforge.evaluation.gma_workflow import (
            Experiment, ExperimentalDatabase
        )
        import numpy as np
        
        db = ExperimentalDatabase(name="Test DB")
        
        # Add experiments
        for i in range(5):
            db.add_experiment(Experiment(
                name=f"Exp-{i}",
                value=100 + i * 10,
                uncertainty=5.0,
                energy_MeV=1.0 + i * 0.5,
                reaction="Test(n,g)"
            ))
        
        # Test vectors
        values = db.get_values_vector()
        assert len(values) == 5
        assert values[0] == 100
        
        # Test covariance
        cov = db.build_covariance_matrix()
        assert cov.shape == (5, 5)
        
        # Test JSON roundtrip
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        
        try:
            db.to_json(path)
            db_loaded = ExperimentalDatabase.from_json(path)
            
            assert db_loaded.name == "Test DB"
            assert len(db_loaded.experiments) == 5
            assert db_loaded.experiments[0].value == 100
        finally:
            path.unlink()
    
    def test_gls_workflow(self):
        """Test complete GLS evaluation workflow."""
        from fluxforge.evaluation.gma_workflow import (
            Experiment, ExperimentalDatabase, GMAWorkflow
        )
        import numpy as np
        
        # Create synthetic problem
        np.random.seed(42)
        true_params = [10.0, 2.0, -0.1]
        
        # Create database
        db = ExperimentalDatabase(name="GLS Test")
        for E in np.linspace(1, 10, 15):
            true_val = true_params[0] + true_params[1] * E + true_params[2] * E**2
            measured = true_val + np.random.normal(0, 0.5)
            db.add_experiment(Experiment(
                name=f"E_{E:.1f}",
                value=measured,
                uncertainty=0.5,
                energy_MeV=E,
                reaction="Test"
            ))
        
        # Run workflow
        workflow = GMAWorkflow("Test GLS")
        workflow.load_experiments(db)
        workflow.set_prior(
            values=np.array([8.0, 2.5, 0.0]),
            covariance=np.diag([2.0, 0.5, 0.1])**2,
            parameter_names=['a0', 'a1', 'a2']
        )
        
        # Response function
        def response(params):
            return np.array([
                params[0] + params[1] * e.energy_MeV + params[2] * e.energy_MeV**2
                for e in db.experiments
            ])
        
        workflow.calculate_sensitivities(response)
        result = workflow.run_gls()
        
        assert result.converged
        # Posterior should be closer to true values than prior
        prior_error = sum((p - t)**2 for p, t in zip([8.0, 2.5, 0.0], true_params))
        post_error = sum((p - t)**2 for p, t in zip(result.posterior_values, true_params))
        assert post_error < prior_error


# =============================================================================
# EPIC T TESTS: STACKED TARGET AND SPECTRUM EXPORT
# =============================================================================

class TestStackedTarget:
    """Tests for T1.4-T1.6 stacked target functionality."""
    
    def test_foil_creation(self):
        """Test Foil dataclass."""
        from fluxforge.physics.stacked_target import Foil, STANDARD_MATERIALS
        
        cu = STANDARD_MATERIALS['copper']
        foil = Foil(
            material=cu,
            thickness_um=25.0,
            reaction="(p,n)",
            target_isotope="Cu-63",
            product_isotope="Zn-63"
        )
        
        # Check thickness conversion
        # Cu density = 8.96 g/cm³
        # 25 μm * 8.96 g/cm³ * 0.1 = 22.4 mg/cm²
        assert abs(foil.thickness_mg_cm2 - 22.4) < 0.5
    
    def test_stacked_target_energy_degradation(self):
        """Test energy calculation through foil stack."""
        from fluxforge.physics.stacked_target import (
            StackedTarget, Projectile
        )
        
        stack = StackedTarget(
            projectile=Projectile.PROTON,
            beam_energy_MeV=20.0
        )
        
        # Add some foils
        stack.add_foil('aluminum', 100)
        stack.add_foil('copper', 25)
        stack.add_foil('aluminum', 100)
        
        energies = stack.calculate_energies()
        
        assert len(energies) == 3
        # Energy should decrease through stack
        assert energies[0].energy_in_MeV == 20.0
        assert energies[1].energy_in_MeV < energies[0].energy_in_MeV
        assert energies[2].energy_in_MeV < energies[1].energy_in_MeV
    
    def test_cross_section_library(self):
        """Test cross-section library interface."""
        from fluxforge.physics.stacked_target import CrossSectionLibrary
        
        # Test built-in reactions
        xs = CrossSectionLibrary.get_cross_section("Cu-63(p,n)Zn-63", "irdff")
        assert xs is not None
        assert len(xs.energies_MeV) > 5
        
        # Test interpolation
        sigma = xs.interpolate(10.0)
        assert sigma > 0
        
        # Test out-of-range
        sigma_low = xs.interpolate(0.1)
        assert sigma_low == 0.0


class TestSpectrumExport:
    """Tests for T1.9 multi-format spectrum export."""
    
    def test_spe_export_import(self):
        """Test SPE format export and basic validation."""
        from fluxforge.io.spectrum_export import SpectrumExporter, SpectrumMetadata
        from datetime import datetime
        
        counts = [100 + i for i in range(512)]
        meta = SpectrumMetadata(
            title="Test Spectrum",
            live_time=1000,
            real_time=1010,
            energy_coefficients=[0.5, 2.5, 0.0]
        )
        
        exporter = SpectrumExporter(counts, meta)
        
        with tempfile.NamedTemporaryFile(suffix='.spe', delete=False) as f:
            path = Path(f.name)
        
        try:
            exporter.to_spe(path)
            
            # Verify file was written
            assert path.exists()
            assert path.stat().st_size > 0
            
            # Read back and verify structure
            with open(path, 'r') as f:
                content = f.read()
            
            assert '$SPEC_ID:' in content
            assert '$DATA:' in content
            assert '$MEAS_TIM:' in content
        finally:
            path.unlink()
    
    def test_csv_export(self):
        """Test CSV export."""
        from fluxforge.io.spectrum_export import SpectrumExporter
        
        counts = [100, 200, 300, 400, 500]
        exporter = SpectrumExporter(counts)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = Path(f.name)
        
        try:
            exporter.to_csv(path)
            
            with open(path, 'r') as f:
                lines = f.readlines()
            
            assert 'channel' in lines[0]
            assert 'counts' in lines[0]
            assert len(lines) == 6  # Header + 5 data lines
        finally:
            path.unlink()
    
    def test_mcnp_sdef_export(self):
        """Test MCNP SDEF format export."""
        from fluxforge.io.spectrum_export import SpectrumExporter
        
        counts = [100, 200, 300, 400, 500]
        energies = [0.5, 1.0, 1.5, 2.0, 2.5]  # MeV
        exporter = SpectrumExporter(counts, energies=energies)
        
        with tempfile.NamedTemporaryFile(suffix='.sdef', delete=False) as f:
            path = Path(f.name)
        
        try:
            exporter.to_mcnp_sdef(path, energy_unit='MeV')
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert 'SI1' in content
            assert 'SP1' in content
            assert 'SDEF' in content
        finally:
            path.unlink()


# =============================================================================
# EPIC U TESTS: STOPPING POWER
# =============================================================================

class TestStoppingPower:
    """Tests for U1.1-U1.6 stopping power functionality."""
    
    def test_bethe_stopping(self):
        """Test Bethe-Bloch stopping power formula."""
        from fluxforge.physics.stopping_power import bethe_stopping_power
        
        # 10 MeV proton in aluminum
        S = bethe_stopping_power(
            energy_MeV=10.0,
            projectile_Z=1,
            projectile_A=1,
            target_Z=13,
            target_A=27,
            I_eV=166
        )
        
        assert S > 0
        # Stopping power should be positive (in MeV/(mg/cm²))
        # Typical range is 0.01-100 MeV/(mg/cm²) depending on implementation units
        assert S < 1e6  # Sanity check for reasonable magnitude
    
    def test_electronic_stopping_ziegler(self):
        """Test Ziegler electronic stopping."""
        from fluxforge.physics.stopping_power import (
            electronic_stopping_ziegler, Projectile
        )
        
        S = electronic_stopping_ziegler(
            energy_MeV=10.0,
            projectile=Projectile.PROTON,
            target_Z=29,  # Cu
            target_A=63.5
        )
        
        assert S > 0
    
    def test_total_stopping_compound(self):
        """Test stopping power in compound material (Bragg additivity)."""
        from fluxforge.physics.stopping_power import (
            total_stopping_power, Material, Projectile, STANDARD_MATERIALS
        )
        
        water = STANDARD_MATERIALS['water']
        S = total_stopping_power(10.0, Projectile.PROTON, water)
        
        assert S > 0
        
        # Water should have different stopping than pure elements
        aluminum = STANDARD_MATERIALS['aluminum']
        S_Al = total_stopping_power(10.0, Projectile.PROTON, aluminum)
        
        assert S != S_Al
    
    def test_range_calculation(self):
        """Test range calculation (U1.3)."""
        from fluxforge.physics.stopping_power import (
            calculate_range, Projectile, STANDARD_MATERIALS
        )
        
        # 5 MeV alpha in gold
        gold = STANDARD_MATERIALS['gold']
        R = calculate_range(5.0, Projectile.ALPHA, gold)
        
        assert R > 0
        # Range should be in reasonable units (mg/cm²)
        assert 1 < R < 100
    
    def test_energy_loss(self):
        """Test energy loss through material (U1.2)."""
        from fluxforge.physics.stopping_power import (
            calculate_energy_loss, Projectile, STANDARD_MATERIALS
        )
        
        aluminum = STANDARD_MATERIALS['aluminum']
        E_final, dE = calculate_energy_loss(
            initial_energy_MeV=10.0,
            projectile=Projectile.PROTON,
            material=aluminum,
            thickness_mg_cm2=5.0
        )
        
        assert E_final < 10.0
        assert dE > 0
        assert E_final + dE == pytest.approx(10.0, rel=0.01)
    
    def test_straggling(self):
        """Test energy straggling (U1.4)."""
        from fluxforge.physics.stopping_power import (
            calculate_straggling, Projectile, STANDARD_MATERIALS
        )
        
        aluminum = STANDARD_MATERIALS['aluminum']
        result = calculate_straggling(
            initial_energy_MeV=10.0,
            projectile=Projectile.PROTON,
            material=aluminum,
            thickness_mg_cm2=10.0
        )
        
        assert result.sigma_MeV > 0
        assert result.fwhm_MeV > result.sigma_MeV
        assert result.mean_energy_MeV < 10.0


# =============================================================================
# EPIC W TESTS: ADVANCED UNFOLDING
# =============================================================================

class TestAdvancedUnfolding:
    """Tests for W1.5 covariance options and W1.6 Adye propagation."""
    
    def test_covariance_models(self):
        """Test different covariance models (W1.5)."""
        from fluxforge.solvers.advanced_unfolding import (
            build_covariance_matrix, CovarianceModel
        )
        
        measurements = [100, 200, 300, 400, 500]
        
        # Diagonal
        cov_diag = build_covariance_matrix(measurements, CovarianceModel.DIAGONAL)
        assert cov_diag[0][1] == 0  # Off-diagonal should be zero
        
        # Poisson
        cov_pois = build_covariance_matrix(measurements, CovarianceModel.POISSON)
        assert cov_pois[0][0] == measurements[0]  # Variance = mean
        
        # Multinomial
        cov_mult = build_covariance_matrix(measurements, CovarianceModel.MULTINOMIAL)
        assert cov_mult[0][1] < 0  # Off-diagonal negative for multinomial
    
    def test_adye_error_propagation(self):
        """Test Adye error propagation (W1.6)."""
        from fluxforge.solvers.advanced_unfolding import adye_error_propagation
        import numpy as np
        
        # Simple problem
        n_flux = 5
        n_meas = 4
        
        response = [[0.3 if abs(i-j) <= 1 else 0.1 for j in range(n_flux)] 
                    for i in range(n_meas)]
        flux = [100, 150, 200, 150, 100]
        meas_cov = [[100 if i == j else 0 for j in range(n_meas)] 
                    for i in range(n_meas)]
        
        flux_cov = adye_error_propagation(response, flux, meas_cov, method="bayes")
        
        assert len(flux_cov) == n_flux
        assert len(flux_cov[0]) == n_flux
        # Diagonal should be positive
        for i in range(n_flux):
            assert flux_cov[i][i] >= 0
    
    def test_mlem_with_covariance(self):
        """Test MLEM with covariance propagation."""
        from fluxforge.solvers.advanced_unfolding import (
            mlem_with_covariance, CovarianceModel
        )
        from fluxforge.core.linalg import matmul
        import math
        
        # Create synthetic problem
        n_flux = 10
        n_meas = 6
        
        true_flux = [1000 * math.exp(-0.2 * g) for g in range(n_flux)]
        response = [[0.3 * math.exp(-0.5 * (g - 1.5*i)**2) for g in range(n_flux)]
                    for i in range(n_meas)]
        
        measurements = matmul(response, true_flux)
        
        result = mlem_with_covariance(
            response,
            measurements,
            cov_model=CovarianceModel.POISSON,
            max_iters=200,
            compute_errors=True
        )
        
        assert result.flux is not None
        assert result.flux_uncertainty is not None
        assert len(result.flux_uncertainty) == n_flux


# =============================================================================
# EPIC X TESTS: GAMMA SPECTRUM METASTABLE STATES
# =============================================================================

class TestMetastableStates:
    """Tests for X1.8 metastable state support."""
    
    def test_normalize_nuclide_name(self):
        """Test nuclide name normalization with metastable states."""
        from fluxforge.physics.gamma_spectrum import normalize_nuclide_name
        
        # Standard names
        assert normalize_nuclide_name('Co-60') == 'Co-60'
        assert normalize_nuclide_name('co60') == 'Co-60'
        assert normalize_nuclide_name('CO_60') == 'Co-60'
        
        # Metastable states
        assert normalize_nuclide_name('Tc-99m') == 'Tc-99m'
        assert normalize_nuclide_name('tc99m') == 'Tc-99m'
        assert normalize_nuclide_name('TC99M') == 'Tc-99m'
        
        # Higher metastable states
        assert normalize_nuclide_name('In-116m1') == 'In-116m1'
        assert normalize_nuclide_name('In116m2') == 'In-116m2'
    
    def test_is_metastable(self):
        """Test metastable state detection."""
        from fluxforge.physics.gamma_spectrum import is_metastable
        
        assert is_metastable('Tc-99m') == True
        assert is_metastable('In-116m1') == True
        assert is_metastable('Co-60') == False
        assert is_metastable('Ba-137') == False
    
    def test_get_ground_state(self):
        """Test getting ground state from metastable."""
        from fluxforge.physics.gamma_spectrum import get_ground_state
        
        assert get_ground_state('Tc-99m') == 'Tc-99'
        assert get_ground_state('In-116m1') == 'In-116'
        assert get_ground_state('Co-60') == 'Co-60'
    
    def test_parse_nuclide(self):
        """Test nuclide parsing."""
        from fluxforge.physics.gamma_spectrum import parse_nuclide
        
        element, mass, meta = parse_nuclide('Tc-99m')
        assert element == 'Tc'
        assert mass == 99
        assert meta == 'm'
        
        element, mass, meta = parse_nuclide('Co-60')
        assert element == 'Co'
        assert mass == 60
        assert meta is None
    
    def test_metastable_decay_lines(self):
        """Test decay line lookup for metastable nuclides."""
        from fluxforge.physics.gamma_spectrum import get_decay_lines
        
        # Tc-99m should have 140.5 keV line
        lines = get_decay_lines('Tc-99m')
        assert len(lines) > 0
        
        tc99m_energies = [line.energy_keV for line in lines]
        assert any(abs(E - 140.5) < 1 for E in tc99m_energies)


# =============================================================================
# EPIC Y TESTS: LOG-SMOOTHNESS AND CONVERGENCE
# =============================================================================

class TestLogSmoothness:
    """Tests for Y1.1 log-smoothness and Y1.7 second-derivative convergence."""
    
    def test_log_smoothness_penalty(self):
        """Test log-smoothness penalty calculation (Y1.1)."""
        from fluxforge.solvers.advanced_unfolding import log_smoothness_penalty
        import math
        
        # Smooth spectrum should have low penalty
        smooth_flux = [1000 * math.exp(-0.1 * g) for g in range(20)]
        penalty_smooth = log_smoothness_penalty(smooth_flux)
        
        # Noisy spectrum should have high penalty
        noisy_flux = [1000 if g % 2 == 0 else 100 for g in range(20)]
        penalty_noisy = log_smoothness_penalty(noisy_flux)
        
        assert penalty_smooth < penalty_noisy
    
    def test_log_smoothness_gradient(self):
        """Test log-smoothness gradient."""
        from fluxforge.solvers.advanced_unfolding import (
            log_smoothness_penalty, log_smoothness_gradient
        )
        import math
        
        flux = [1000 * math.exp(-0.1 * g) for g in range(10)]
        
        # Numerical gradient check
        grad = log_smoothness_gradient(flux)
        
        eps = 1e-6
        for g in range(len(flux)):
            flux_plus = flux.copy()
            flux_plus[g] += eps
            
            numerical_grad = (
                log_smoothness_penalty(flux_plus) - log_smoothness_penalty(flux)
            ) / eps
            
            assert abs(grad[g] - numerical_grad) < 0.1 * max(abs(numerical_grad), 1e-6)
    
    def test_ddJ_convergence(self):
        """Test second-derivative convergence criterion (Y1.7)."""
        from fluxforge.solvers.advanced_unfolding import compute_ddJ_convergence
        
        # Converging sequence (J decreasing and stabilizing)
        J_history = [100, 80, 65, 55, 48, 43, 40, 38, 37, 36.5, 36.2, 36.0]
        
        dJ, ddJ = compute_ddJ_convergence(J_history)
        
        # ddJ should be small when converging
        assert ddJ < 1.0
    
    def test_mlem_with_smoothness(self):
        """Test MLEM with log-smoothness regularization."""
        from fluxforge.solvers.advanced_unfolding import (
            mlem_with_covariance, CovarianceModel
        )
        from fluxforge.core.linalg import matmul
        import math
        import random
        
        random.seed(42)
        
        n_flux = 15
        n_meas = 8
        
        true_flux = [1000 * math.exp(-0.15 * g) for g in range(n_flux)]
        response = [[0.3 * math.exp(-0.5 * (g - 1.8*i)**2) for g in range(n_flux)]
                    for i in range(n_meas)]
        
        measurements = matmul(response, true_flux)
        # Add noise
        measurements = [m + random.gauss(0, math.sqrt(m)) for m in measurements]
        measurements = [max(m, 1) for m in measurements]
        
        # Without smoothness
        result_no_smooth = mlem_with_covariance(
            response, measurements,
            cov_model=CovarianceModel.POISSON,
            smoothness_weight=0.0,
            max_iters=200
        )
        
        # With smoothness
        result_smooth = mlem_with_covariance(
            response, measurements,
            cov_model=CovarianceModel.POISSON,
            smoothness_weight=0.1,
            max_iters=200
        )
        
        # Smoothed result should have lower smoothness penalty
        from fluxforge.solvers.advanced_unfolding import log_smoothness_penalty
        assert log_smoothness_penalty(result_smooth.flux) <= log_smoothness_penalty(result_no_smooth.flux)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
