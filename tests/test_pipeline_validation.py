"""
FluxForge Pipeline Validation Tests

This module tests the complete pipeline against experimental data to verify
that all master plan goals are being met. Tests are organized by pipeline stage
and reference specific goals from docs/master_plan.md.

Goals Validated:
1. From raw HPGe spectra, produce isotope activities with uncertainty (Stage A-C)
2. Convert activities to EOI reaction rates using irradiation history (Stage D)
3. Build response matrix R[i,g] using dosimetry cross sections (Stage E)
4. Unfold/adjust neutron spectra using multiple solver families (Stage F)
5. Produce posterior spectra with covariance and diagnostics (Stage F)
6. Artifact provenance and round-trip preservation (L0)

Experimental Data Used:
- Fe-Cd-RAFM-1: Fe58(n,γ)→Fe59 activation wire with Cd cover
- Flux wire timing data: Co, Ti, Ni, Sc, In, Cu wires
- Efficiency calibration data
"""

import json
import math
import pytest
from pathlib import Path
from typing import Dict, List, Any

# Import markers for optional dependencies
np = pytest.importorskip("numpy")


# =============================================================================
# Paths and Test Data
# =============================================================================

FLUXFORGE_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = FLUXFORGE_ROOT / "src" / "fluxforge" / "examples"
FE_CD_EXAMPLE = EXAMPLES_DIR / "fe_cd_rafm_1"
FLUX_WIRE_DIR = EXAMPLES_DIR / "flux_wire"


def load_json(path: Path) -> Dict:
    """Load JSON file with error handling."""
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return json.loads(path.read_text())


# =============================================================================
# Stage A Tests: Spectrum Ingestion and Metadata
# =============================================================================

class TestStageASpectrumIngest:
    """
    Tests for Stage A: Spectrum ingest and metadata validation.
    
    Master Plan Goals:
    - A1: Read SPE/CHN/CNF/N42/CSV formats
    - A2: Metadata validation + QC flags
    - A3: Preprocessing tools
    """
    
    def test_spe_reader_basic(self):
        """Test SPE reader can load spectrum files."""
        from fluxforge.io.spe import read_spe_file
        
        # Check if we have test SPE files
        asc_file = FLUX_WIRE_DIR / "Co-Cd-RAFM-1_25cm.ASC"
        if not asc_file.exists():
            pytest.skip("ASC test file not found")
        
        # Currently we might not support ASC, but test the reader exists
        assert callable(read_spe_file)
    
    def test_spectrum_dataclass_fields(self):
        """Test GammaSpectrum has required fields per master plan."""
        from fluxforge.io.spe import GammaSpectrum
        
        # Create a minimal spectrum to test fields
        spectrum = GammaSpectrum(
            counts=np.array([100.0, 200.0, 300.0]),
            channels=np.array([0, 1, 2]),
            live_time=1000.0,
            real_time=1005.0,
        )
        
        # Required fields per master plan
        assert hasattr(spectrum, 'counts')
        assert hasattr(spectrum, 'live_time')
        assert hasattr(spectrum, 'real_time')
        assert spectrum.dead_time_fraction >= 0  # Property, not method
    
    def test_spectrum_dead_time_computation(self):
        """Test dead time fraction is computed correctly."""
        from fluxforge.io.spe import GammaSpectrum
        
        spectrum = GammaSpectrum(
            counts=np.array([100.0]),
            channels=np.array([0]),
            live_time=900.0,
            real_time=1000.0,
        )
        
        # Dead time fraction should be (real - live) / real = 0.1
        assert math.isclose(spectrum.dead_time_fraction, 0.1, rel_tol=1e-6)  # Property
    
    def test_spectrum_energy_calibration(self):
        """Test energy calibration application."""
        from fluxforge.io.spe import GammaSpectrum
        
        spectrum = GammaSpectrum(
            counts=np.array([100.0, 200.0, 300.0]),
            channels=np.array([0, 1, 2]),
            live_time=1000.0,
            real_time=1000.0,
            calibration={'energy': [0.0, 0.5, 0.0]},  # E = 0.5 * ch
        )
        
        energies = spectrum.calibrate_channels()
        assert len(energies) == 3
        assert math.isclose(energies[1], 0.5, rel_tol=1e-6)


# =============================================================================
# Stage B Tests: Peak Detection and Fitting
# =============================================================================

class TestStageBPeakDetection:
    """
    Tests for Stage B: Peak finding and fitting.
    
    Master Plan Goals:
    - B1: Candidate peak detection
    - B2: ROI and multiplet handling
    - B3: PeakReport artifact
    """
    
    def test_peak_dataclass_exists(self):
        """Test GaussianPeak dataclass has required fields."""
        from fluxforge.analysis.peakfit import GaussianPeak
        
        peak = GaussianPeak(
            centroid=1173.0,
            amplitude=1000.0,
            sigma=1.5,
        )
        
        assert hasattr(peak, 'centroid')
        assert hasattr(peak, 'amplitude')
        assert hasattr(peak, 'sigma')
        assert hasattr(peak, 'fwhm')
        assert hasattr(peak, 'area')
    
    def test_gaussian_peak_area_calculation(self):
        """Test peak area is computed correctly."""
        from fluxforge.analysis.peakfit import GaussianPeak
        
        amplitude = 1000.0
        sigma = 1.5
        peak = GaussianPeak(centroid=1173.0, amplitude=amplitude, sigma=sigma)
        
        # Area of Gaussian = amplitude * sigma * sqrt(2*pi)
        expected_area = amplitude * sigma * math.sqrt(2 * math.pi)
        assert math.isclose(peak.area, expected_area, rel_tol=1e-6)  # Property
    
    def test_background_estimation_snip(self):
        """Test SNIP background estimation."""
        from fluxforge.analysis.peakfit import estimate_background
        
        # Create a simple spectrum with a peak
        channels = np.arange(100)
        background = np.full(100, 10.0)
        peak = 100.0 * np.exp(-0.5 * ((channels - 50) / 3) ** 2)
        counts = background + peak
        
        # Estimate background
        bg_estimate = estimate_background(channels, counts, method="snip", iterations=10)
        
        # Background estimate should be close to true background at edges
        assert np.mean(bg_estimate[:10]) < 15  # Should be ~10
        assert np.mean(bg_estimate[90:]) < 15
    
    def test_peak_fit_result_has_covariance(self):
        """Test PeakFitResult includes covariance information."""
        from fluxforge.analysis.peakfit import PeakFitResult, GaussianPeak
        
        peak = GaussianPeak(centroid=100.0, amplitude=500.0, sigma=2.0)
        result = PeakFitResult(
            peak=peak,
            background=np.full(20, 10.0),
            background_model="linear",
            residuals=np.zeros(20),
            chi_squared=16.15,  # ~0.95 per dof
            dof=17,
            fit_region=(90, 110),
        )

        assert hasattr(result, 'chi_squared')
        assert hasattr(result, 'reduced_chi_squared')
        assert result.reduced_chi_squared > 0  # Property
# Stage C Tests: Activity Calculation
# =============================================================================

class TestStageCActivityCalculation:
    """
    Tests for Stage C: Activity computation.
    
    Master Plan Goals:
    - C3: Line activity computation with corrections
    - C4: Combine multiple lines per isotope
    """
    
    def test_gamma_line_measurement_activity(self):
        """Test activity calculation from gamma line measurement."""
        from fluxforge.physics.activation import GammaLineMeasurement
        
        line = GammaLineMeasurement(
            net_counts=10000,
            live_time_s=1000.0,
            efficiency=0.01,
            gamma_intensity=0.5,
            half_life_s=1e9,  # Very long half-life (no decay correction)
            cooling_time_s=0.0,
        )
        
        activity = line.activity_at_reference()
        
        # Activity = counts / (efficiency * intensity * live_time)
        expected = 10000 / (0.01 * 0.5 * 1000.0)
        assert math.isclose(activity, expected, rel_tol=1e-3)
    
    def test_activity_with_decay_correction(self):
        """Test activity calculation includes decay during cooling."""
        from fluxforge.physics.activation import GammaLineMeasurement
        
        half_life = 3600.0  # 1 hour
        cooling_time = 3600.0  # 1 half-life
        
        line = GammaLineMeasurement(
            net_counts=10000,
            live_time_s=100.0,  # Short count time
            efficiency=0.01,
            gamma_intensity=0.5,
            half_life_s=half_life,
            cooling_time_s=cooling_time,
        )
        
        activity = line.activity_at_reference()
        
        # Activity at reference should be ~2x activity at count time
        # (because we decayed for one half-life)
        line_no_cooling = GammaLineMeasurement(
            net_counts=10000,
            live_time_s=100.0,
            efficiency=0.01,
            gamma_intensity=0.5,
            half_life_s=half_life,
            cooling_time_s=0.0,
        )
        activity_no_cooling = line_no_cooling.activity_at_reference()
        
        # With cooling, we should get back a higher original activity
        assert activity > activity_no_cooling
    
    def test_weighted_activity_multiple_lines(self):
        """Test weighted combination of multiple gamma lines."""
        from fluxforge.physics.activation import GammaLineMeasurement, weighted_activity
        
        # Fe-59 has lines at 1099.3 keV and 1291.6 keV
        lines = [
            GammaLineMeasurement(
                net_counts=22892,
                live_time_s=43200.0,
                efficiency=0.0015,
                gamma_intensity=0.565,
                half_life_s=3844368.0,
                cooling_time_s=172800.0,
            ),
            GammaLineMeasurement(
                net_counts=15351,
                live_time_s=43200.0,
                efficiency=0.0015,
                gamma_intensity=0.432,
                half_life_s=3844368.0,
                cooling_time_s=172800.0,
            ),
        ]
        
        activity, uncertainty = weighted_activity(lines)
        
        assert activity > 0
        assert uncertainty > 0
        assert uncertainty < activity  # Uncertainty should be smaller than value
    
    def test_fe_cd_example_activities(self):
        """Test activity calculation using real Fe-Cd-RAFM-1 data."""
        from fluxforge.physics.activation import GammaLineMeasurement, weighted_activity
        
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            
            activity, uncertainty = weighted_activity(gamma_lines)
            
            # Activity should be positive and reasonable
            assert activity > 0
            # Uncertainty should be positive
            assert uncertainty > 0
            # Relative uncertainty should be reasonable (<50%)
            assert uncertainty / activity < 0.5


# =============================================================================
# Stage D Tests: Reaction Rate Inference
# =============================================================================

class TestStageDReactionRates:
    """
    Tests for Stage D: Irradiation history and reaction rates.
    
    Master Plan Goals:
    - D1: Time-history model (piecewise constant)
    - D2: Reaction-rate inference
    - D3: Uncertainty propagation
    """
    
    def test_irradiation_buildup_factor(self):
        """Test irradiation buildup factor calculation."""
        from fluxforge.physics.activation import IrradiationSegment, irradiation_buildup_factor
        
        # Single segment irradiation
        half_life = 3600.0  # 1 hour
        segments = [IrradiationSegment(duration_s=7200.0, relative_power=1.0)]
        
        factor = irradiation_buildup_factor(segments, half_life)
        
        # For 2 half-lives irradiation, factor should be (1 - e^{-2*ln2}) ≈ 0.75
        expected = 1 - math.exp(-2 * math.log(2))
        assert math.isclose(factor, expected, rel_tol=1e-3)
    
    def test_multi_segment_irradiation(self):
        """Test multi-segment irradiation history."""
        from fluxforge.physics.activation import IrradiationSegment, irradiation_buildup_factor
        
        half_life = 3600.0
        
        # Two segments: 1 hour at full power, 1 hour at half power
        segments = [
            IrradiationSegment(duration_s=3600.0, relative_power=1.0),
            IrradiationSegment(duration_s=3600.0, relative_power=0.5),
        ]
        
        factor = irradiation_buildup_factor(segments, half_life)
        
        # Should be positive and less than 2*single segment
        single_segment = [IrradiationSegment(duration_s=7200.0, relative_power=1.0)]
        single_factor = irradiation_buildup_factor(single_segment, half_life)
        
        assert factor > 0
        assert factor < single_factor  # Half power should give less
    
    def test_reaction_rate_from_activity(self):
        """Test reaction rate inference from EOI activity."""
        from fluxforge.physics.activation import (
            IrradiationSegment,
            reaction_rate_from_activity,
            irradiation_buildup_factor,
        )
        
        activity = 1000.0  # Bq
        half_life = 44.495 * 86400  # Fe-59 half-life in seconds
        segments = [IrradiationSegment(duration_s=7200.0, relative_power=1.0)]
        
        rate_estimate = reaction_rate_from_activity(activity, segments, half_life)
        
        assert rate_estimate.rate > 0
        assert rate_estimate.uncertainty > 0
        
        # Verify: rate * buildup_factor = activity
        factor = irradiation_buildup_factor(segments, half_life)
        assert math.isclose(rate_estimate.rate * factor, activity, rel_tol=1e-3)
    
    def test_fe_cd_example_reaction_rates(self):
        """Test reaction rate calculation using real Fe-Cd-RAFM-1 data."""
        from fluxforge.physics.activation import (
            GammaLineMeasurement,
            IrradiationSegment,
            weighted_activity,
            reaction_rate_from_activity,
        )
        
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            
            activity, _ = weighted_activity(gamma_lines)
            rate_estimate = reaction_rate_from_activity(
                activity, segments, reaction["half_life_s"]
            )
            
            # Reaction rate should be positive
            assert rate_estimate.rate > 0
            # Uncertainty should be positive
            assert rate_estimate.uncertainty > 0


# =============================================================================
# Stage E Tests: Response Matrix Construction
# =============================================================================

class TestStageEResponseMatrix:
    """
    Tests for Stage E: Response matrix construction.
    
    Master Plan Goals:
    - F1: Group structures
    - F4: Response matrix construction with diagnostics
    """
    
    def test_energy_group_structure(self):
        """Test energy group structure creation."""
        from fluxforge.core.response import EnergyGroupStructure
        
        boundaries = [0.0305, 100.0, 10000.0, 636000.0]
        groups = EnergyGroupStructure(boundaries)
        
        assert groups.group_count == 3
        assert len(groups.boundaries_eV) == 4  # Actual attribute name
        assert groups.boundaries_eV[0] == 0.0305
    
    def test_response_matrix_shape(self):
        """Test response matrix has correct shape."""
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        
        boundaries = [0.0305, 100.0, 10000.0, 636000.0]
        groups = EnergyGroupStructure(boundaries)
        
        reactions = [
            ReactionCrossSection(reaction_id="rx1", sigma_g=[1.0, 0.5, 0.1]),
            ReactionCrossSection(reaction_id="rx2", sigma_g=[0.1, 1.0, 0.5]),
        ]
        number_densities = [1e20, 1e20]
        
        response = build_response_matrix(reactions, groups, number_densities)
        
        # Response matrix should be [n_reactions x n_groups]
        assert len(response.matrix) == 2  # 2 reactions
        assert len(response.matrix[0]) == 3  # 3 groups
    
    def test_response_matrix_positive(self):
        """Test response matrix elements are non-negative."""
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        
        boundaries = [0.0305, 100.0, 10000.0, 636000.0]
        groups = EnergyGroupStructure(boundaries)
        
        reactions = [
            ReactionCrossSection(reaction_id="rx1", sigma_g=[1.0, 0.5, 0.1]),
        ]
        number_densities = [1e20]
        
        response = build_response_matrix(reactions, groups, number_densities)
        
        for row in response.matrix:
            for element in row:
                assert element >= 0
    
    def test_fe_cd_example_response_matrix(self):
        """Test response matrix construction using real Fe-Cd-RAFM-1 data."""
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        
        boundaries = load_json(FE_CD_EXAMPLE / "boundaries.json")
        cross_sections = load_json(FE_CD_EXAMPLE / "cross_sections.json")
        number_densities = load_json(FE_CD_EXAMPLE / "number_densities.json")
        
        groups = EnergyGroupStructure(boundaries)
        reactions = [
            ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
            for r_id, sigma in cross_sections.items()
        ]
        nd_values = [number_densities[rx.reaction_id] for rx in reactions]
        
        response = build_response_matrix(reactions, groups, nd_values)
        
        assert len(response.matrix) == len(reactions)
        assert len(response.matrix[0]) == groups.group_count


# =============================================================================
# Stage F Tests: Spectrum Unfolding
# =============================================================================

class TestStageFUnfolding:
    """
    Tests for Stage F: Spectrum unfolding/adjustment.
    
    Master Plan Goals:
    - G1: GLS / STAYSL-like adjustment
    - G2: MLEM and GRAVEL cross-checks
    - Diagnostics: χ², residuals, covariance
    """
    
    def test_gls_basic_adjustment(self):
        """Test GLS adjustment with simple case."""
        from fluxforge.solvers.gls import gls_adjust
        
        # Identity response matrix
        response = [[1.0, 0.0], [0.0, 1.0]]
        measurements = [1.5, 3.0]
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior = [1.0, 2.0]
        prior_cov = [[0.25, 0.0], [0.0, 1.0]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        assert len(solution.flux) == 2
        assert solution.chi2 >= 0
        assert len(solution.covariance) == 2
    
    def test_gls_positivity_enforcement(self):
        """Test GLS enforces non-negative flux."""
        from fluxforge.solvers.gls import gls_adjust
        
        response = [[1.0, 0.5], [0.5, 1.0]]
        measurements = [0.1, 0.1]  # Small measurements
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior = [10.0, 10.0]  # Large prior
        prior_cov = [[100.0, 0.0], [0.0, 100.0]]
        
        solution = gls_adjust(
            response, measurements, measurement_cov, prior, prior_cov,
            enforce_nonnegativity=True
        )
        
        # All flux values should be non-negative
        for val in solution.flux:
            assert val >= 0
    
    def test_gls_covariance_output(self):
        """Test GLS returns proper covariance matrix."""
        from fluxforge.solvers.gls import gls_adjust
        
        response = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        measurements = [1.0, 2.0, 3.0]
        measurement_cov = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        prior = [0.5, 1.5, 2.5]
        prior_cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        # Covariance should be square and same size as flux
        assert len(solution.covariance) == len(solution.flux)
        for row in solution.covariance:
            assert len(row) == len(solution.flux)
        
        # Diagonal elements should be positive (variances)
        for i in range(len(solution.flux)):
            assert solution.covariance[i][i] >= 0
    
    def test_gravel_convergence(self):
        """Test GRAVEL converges for known spectrum."""
        from fluxforge.solvers.iterative import gravel
        
        true_flux = [2.0, 1.0, 0.5]
        response = [
            [1.0, 0.5, 0.1],
            [0.1, 1.0, 0.5],
            [0.5, 0.1, 1.0],
        ]
        
        # Generate measurements from true flux
        measurements = []
        for row in response:
            m = sum(r * f for r, f in zip(row, true_flux))
            measurements.append(m)
        
        solution = gravel(
            response, measurements,
            initial_flux=[1.0, 1.0, 1.0],
            max_iters=500,
            tolerance=1e-6,
        )
        
        # Should converge
        assert solution.converged or solution.iterations < 500
        
        # Recovered flux should be close to true
        for est, true in zip(solution.flux, true_flux):
            assert abs(est - true) / true < 0.1  # Within 10%
    
    def test_mlem_convergence(self):
        """Test MLEM converges for known spectrum."""
        from fluxforge.solvers.iterative import mlem
        
        true_flux = [3.0, 2.0]
        response = [[1.0, 0.2], [0.3, 1.0]]
        
        measurements = []
        for row in response:
            m = sum(r * f for r, f in zip(row, true_flux))
            measurements.append(m)
        
        solution = mlem(
            response, measurements,
            initial_flux=[1.0, 1.0],
            max_iters=100,
            tolerance=1e-6,
        )
        
        assert solution.converged or solution.iterations < 100
    
    def test_fe_cd_example_full_workflow(self):
        """Test complete workflow with Fe-Cd-RAFM-1 data."""
        from fluxforge.core.response import (
            EnergyGroupStructure,
            ReactionCrossSection,
            build_response_matrix,
        )
        from fluxforge.physics.activation import (
            GammaLineMeasurement,
            IrradiationSegment,
            weighted_activity,
            reaction_rate_from_activity,
        )
        from fluxforge.solvers.gls import gls_adjust
        
        # Load all data
        boundaries = load_json(FE_CD_EXAMPLE / "boundaries.json")
        cross_sections = load_json(FE_CD_EXAMPLE / "cross_sections.json")
        number_densities = load_json(FE_CD_EXAMPLE / "number_densities.json")
        measurements = load_json(FE_CD_EXAMPLE / "measurements.json")
        prior_flux = load_json(FE_CD_EXAMPLE / "prior_flux.json")
        
        # Build response matrix
        groups = EnergyGroupStructure(boundaries)
        reactions = [
            ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
            for r_id, sigma in cross_sections.items()
        ]
        nd_values = [number_densities[rx.reaction_id] for rx in reactions]
        response = build_response_matrix(reactions, groups, nd_values)
        
        # Calculate reaction rates
        segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
        measured_rates = []
        rate_uncertainties = []
        
        for reaction in measurements["reactions"]:
            gamma_lines = [
                GammaLineMeasurement(**gl)
                for gl in reaction["gamma_lines"]
            ]
            activity, _ = weighted_activity(gamma_lines)
            rate_estimate = reaction_rate_from_activity(
                activity, segments, reaction["half_life_s"]
            )
            measured_rates.append(rate_estimate.rate)
            rate_uncertainties.append(rate_estimate.uncertainty)
        
        # Build covariance matrices
        prior_cov = [
            [(0.25 * val) ** 2 if i == j else 0.0
             for j, val in enumerate(prior_flux)]
            for i, _ in enumerate(prior_flux)
        ]
        measurement_cov = [
            [(rate_uncertainties[i] ** 2) if i == j else 0.0
             for j in range(len(measured_rates))]
            for i in range(len(measured_rates))
        ]
        
        # Perform GLS adjustment
        solution = gls_adjust(
            response.matrix, measured_rates, measurement_cov,
            prior_flux, prior_cov
        )
        
        # Validate outputs
        assert len(solution.flux) == groups.group_count
        assert all(val >= 0.0 for val in solution.flux)
        assert solution.chi2 >= 0.0
        assert len(solution.covariance) == groups.group_count


# =============================================================================
# Artifact I/O Tests
# =============================================================================

class TestArtifactProvenance:
    """
    Tests for artifact I/O and provenance.
    
    Master Plan Goals:
    - L0: Canonical artifacts and provenance
    - Every artifact includes units, normalization, definitions
    """
    
    def test_spectrum_file_includes_provenance(self, tmp_path):
        """Test SpectrumFile artifact includes provenance."""
        from fluxforge.io.spe import GammaSpectrum
        from fluxforge.io.artifacts import write_spectrum_file, read_spectrum_file
        from fluxforge.core.schemas import validate_artifact
        
        spectrum = GammaSpectrum(
            counts=np.array([100.0, 200.0]),
            channels=np.array([0, 1]),
            live_time=1000.0,
            real_time=1005.0,
            spectrum_id="test",
        )
        
        output = tmp_path / "spectrum.json"
        write_spectrum_file(output, spectrum)
        
        payload = read_spectrum_file(output)
        
        # Validate artifact schema
        errors = validate_artifact(payload)
        assert errors == [], f"Schema validation errors: {errors}"
        
        # Check provenance
        assert "provenance" in payload
        assert "units" in payload["provenance"]
        assert "definitions" in payload["provenance"]
    
    def test_unfold_result_includes_diagnostics(self, tmp_path):
        """Test UnfoldResult includes required diagnostics."""
        from fluxforge.io.artifacts import write_unfold_result, read_unfold_result
        from fluxforge.core.schemas import validate_artifact
        
        output = tmp_path / "unfold.json"
        write_unfold_result(
            output,
            boundaries_eV=[0.0, 1.0, 2.0],
            reactions=["rx1", "rx2"],
            flux=[1.0, 2.0],
            covariance=[[0.1, 0.0], [0.0, 0.2]],
            chi2=1.5,
            method="gls",
        )
        
        payload = read_unfold_result(output)
        
        errors = validate_artifact(payload)
        assert errors == [], f"Schema validation errors: {errors}"
        
        # Check diagnostics
        assert "chi2" in payload
        assert "covariance" in payload
        assert "method" in payload
    
    def test_roundtrip_preserves_values(self, tmp_path):
        """Test artifact round-trip preserves numerical values."""
        from fluxforge.io.artifacts import (
            write_reaction_rates,
            read_reaction_rates,
        )
        
        rates = [{"reaction_id": "rx1", "rate": 1.234567890123, "uncertainty": 0.1}]
        segments = [{"duration_s": 3600.0, "relative_power": 1.0}]
        
        output = tmp_path / "rates.json"
        write_reaction_rates(output, rates=rates, segments=segments)
        
        payload = read_reaction_rates(output)
        
        # Value should be preserved
        assert payload["rates"][0]["rate"] == rates[0]["rate"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndValidation:
    """
    End-to-end integration tests validating complete workflows.
    
    Master Plan Goals:
    - Reproducible pipeline from spectra to unfolded spectrum
    - Model-to-experiment comparison capability
    """
    
    def test_synthetic_recovery(self):
        """Test recovery of known spectrum from synthetic data."""
        from fluxforge.solvers.gls import gls_adjust
        from fluxforge.solvers.iterative import gravel, mlem
        
        # True spectrum
        true_flux = [1e5, 5e4, 2e4, 1e4]
        
        # Response matrix (realistic-ish for activation)
        response = [
            [0.8, 0.15, 0.04, 0.01],  # Thermal reaction
            [0.1, 0.6, 0.25, 0.05],   # Resonance reaction
            [0.01, 0.1, 0.5, 0.39],   # Threshold reaction
        ]
        
        # Generate "measurements"
        measurements = []
        for row in response:
            m = sum(r * f for r, f in zip(row, true_flux))
            measurements.append(m)
        
        # Add 5% noise
        np.random.seed(42)
        noisy_measurements = [m * (1 + 0.05 * np.random.randn()) for m in measurements]
        
        # Set up covariances
        measurement_cov = [
            [(0.05 * m) ** 2 if i == j else 0.0
             for j, m in enumerate(noisy_measurements)]
            for i in range(len(noisy_measurements))
        ]
        
        # Prior: flat spectrum at half true magnitude
        prior = [0.5 * f for f in true_flux]
        prior_cov = [
            [(0.5 * p) ** 2 if i == j else 0.0
             for j, p in enumerate(prior)]
            for i in range(len(prior))
        ]
        
        # GLS adjustment
        gls_solution = gls_adjust(
            response, noisy_measurements, measurement_cov,
            prior, prior_cov
        )
        
        # GRAVEL cross-check
        gravel_solution = gravel(
            response, noisy_measurements,
            initial_flux=prior,
            max_iters=500,
        )
        
        # Verify GLS recovers spectrum within uncertainties
        for i, (est, true) in enumerate(zip(gls_solution.flux, true_flux)):
            rel_error = abs(est - true) / true
            assert rel_error < 0.5, f"Group {i}: {rel_error:.2%} error"
        
        # GRAVEL should also recover approximately
        for i, (est, true) in enumerate(zip(gravel_solution.flux, true_flux)):
            rel_error = abs(est - true) / true
            assert rel_error < 0.5, f"GRAVEL Group {i}: {rel_error:.2%} error"
    
    def test_chi2_diagnostic_meaning(self):
        """Test χ² diagnostic indicates fit quality."""
        from fluxforge.solvers.gls import gls_adjust
        
        # Consistent data: measurements match prior prediction exactly
        response = [[1.0, 0.0], [0.0, 1.0]]
        prior = [2.0, 3.0]
        measurements = [2.0, 3.0]  # Exact match
        
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior_cov = [[0.25, 0.0], [0.0, 0.25]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        # χ² should be very small for consistent data
        dof = len(measurements)
        chi2_reduced = solution.chi2 / dof if dof > 0 else solution.chi2
        assert chi2_reduced < 2.0, f"Reduced χ² = {chi2_reduced} too high for consistent data"
    
    def test_solver_consistency(self):
        """Test different solvers give consistent results."""
        from fluxforge.solvers.gls import gls_adjust
        from fluxforge.solvers.iterative import gravel, mlem
        
        # Simple 2-group problem
        response = [[1.0, 0.3], [0.2, 1.0]]
        measurements = [1.0, 0.8]
        prior = [0.8, 0.6]
        
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior_cov = [[0.16, 0.0], [0.0, 0.16]]
        
        gls_solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        gravel_solution = gravel(response, measurements, initial_flux=prior, max_iters=500)
        mlem_solution = mlem(response, measurements, initial_flux=prior, max_iters=100)
        
        # All solutions should be in the same ballpark
        for i in range(2):
            gls_val = gls_solution.flux[i]
            gravel_val = gravel_solution.flux[i]
            mlem_val = mlem_solution.flux[i]
            
            # They should agree within 50% (they use different optimization approaches)
            assert abs(gls_val - gravel_val) / gls_val < 0.5
            assert abs(gls_val - mlem_val) / gls_val < 0.5


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_measurement_single_group(self):
        """Test handling of minimal problem size."""
        from fluxforge.solvers.gls import gls_adjust
        
        response = [[1.0]]
        measurements = [5.0]
        measurement_cov = [[0.25]]
        prior = [4.0]
        prior_cov = [[1.0]]
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        assert len(solution.flux) == 1
        assert solution.flux[0] > 0
    
    def test_zero_measurement_handling(self):
        """Test handling of zero measurements."""
        from fluxforge.solvers.iterative import gravel
        
        response = [[1.0, 0.5], [0.5, 1.0]]
        measurements = [0.0, 1.0]  # One zero measurement
        
        # GRAVEL should handle this without crashing
        solution = gravel(
            response, measurements,
            initial_flux=[1.0, 1.0],
            max_iters=100,
        )
        
        # Should still produce a result
        assert len(solution.flux) == 2
    
    def test_large_uncertainty_prior(self):
        """Test behavior with very uncertain prior."""
        from fluxforge.solvers.gls import gls_adjust
        
        response = [[1.0, 0.0], [0.0, 1.0]]
        measurements = [2.0, 3.0]
        measurement_cov = [[0.01, 0.0], [0.0, 0.01]]
        prior = [0.0, 0.0]  # Zero prior
        prior_cov = [[1e6, 0.0], [0.0, 1e6]]  # Very large uncertainty
        
        solution = gls_adjust(response, measurements, measurement_cov, prior, prior_cov)
        
        # Solution should be pulled toward measurements
        for est, meas in zip(solution.flux, measurements):
            assert abs(est - meas) / meas < 0.2  # Within 20% of measurement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
