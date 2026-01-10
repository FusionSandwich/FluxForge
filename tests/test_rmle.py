"""
Unit tests for RMLE gamma spectrum unfolding.

Tests regularized maximum likelihood estimation solver.
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.solvers.rmle import (
    RegularizationType,
    ParameterSelection,
    PoissonPenalty,
    SpectrumData,
    PeakModel,
    ResponseMatrix,
    UnfoldingResult,
    tikhonov_matrix,
    calculate_chi_squared,
    rmle_unfolding,
    poisson_rmle_unfolding,
    PoissonRMLEConfig,
    select_lambda_lcurve,
    select_lambda_gcv,
    create_gaussian_response_matrix,
    fit_peaks_mle,
    calculate_activity_from_peak,
    RMLEArtifact,
    ActivityResult,
)


class TestSpectrumData:
    """Test spectrum data class."""
    
    def test_creation(self):
        """Test basic spectrum creation."""
        counts = np.random.poisson(100, 1000)
        
        spectrum = SpectrumData(
            counts=counts,
            live_time_s=3600,
        )
        
        assert spectrum.n_channels == 1000
        assert spectrum.live_time_s == 3600
    
    def test_uncertainty_default(self):
        """Default uncertainty should be sqrt(counts)."""
        counts = np.array([100, 400, 900])
        
        spectrum = SpectrumData(counts=counts)
        
        assert spectrum.uncertainty[0] == pytest.approx(10)
        assert spectrum.uncertainty[1] == pytest.approx(20)
        assert spectrum.uncertainty[2] == pytest.approx(30)
    
    def test_count_rate(self):
        """Test count rate calculation."""
        counts = np.array([1000, 2000, 3000])
        
        spectrum = SpectrumData(counts=counts, live_time_s=100)
        
        rate = spectrum.count_rate
        assert rate[0] == pytest.approx(10)
        assert rate[1] == pytest.approx(20)


class TestPeakModel:
    """Test peak model class."""
    
    def test_gaussian_evaluation(self):
        """Test Gaussian peak evaluation."""
        peak = PeakModel(
            centroid=500,
            sigma=5,
            amplitude=1000,
        )
        
        x = np.arange(480, 520)
        y = peak.evaluate(x)
        
        # Peak should be at centroid
        assert np.argmax(y) == 20  # Channel 500
        assert y[20] == pytest.approx(1000)
    
    def test_gaussian_width(self):
        """Test that sigma controls width."""
        peak_narrow = PeakModel(centroid=500, sigma=2, amplitude=1000)
        peak_wide = PeakModel(centroid=500, sigma=10, amplitude=1000)
        
        x = np.arange(480, 520)
        
        y_narrow = peak_narrow.evaluate(x)
        y_wide = peak_wide.evaluate(x)
        
        # Narrow peak should drop off faster
        assert y_narrow[10] < y_wide[10]  # At channel 490


class TestResponseMatrix:
    """Test response matrix class."""
    
    def test_matrix_dimensions(self):
        """Test matrix dimension properties."""
        matrix = np.random.rand(1000, 50)
        
        response = ResponseMatrix(matrix=matrix)
        
        assert response.n_channels == 1000
        assert response.n_energy_bins == 50
    
    def test_column_normalization(self):
        """Test column normalization."""
        matrix = np.array([
            [1, 2],
            [3, 4],
            [6, 4],
        ])
        
        response = ResponseMatrix(matrix=matrix.astype(float))
        normalized = response.normalize_columns()
        
        # Each column should sum to 1
        col_sums = np.sum(normalized.matrix, axis=0)
        assert np.allclose(col_sums, 1.0)


class TestTikhonovMatrix:
    """Test Tikhonov regularization matrices."""
    
    def test_zeroth_order(self):
        """Zeroth order should be identity."""
        L = tikhonov_matrix(5, order=0)
        
        assert L.shape == (5, 5)
        assert np.allclose(L, np.eye(5))
    
    def test_first_order(self):
        """First order should be first derivative operator."""
        L = tikhonov_matrix(5, order=1)
        
        assert L.shape == (4, 5)
        
        # Test that it computes differences
        x = np.array([1, 2, 4, 7, 11])
        diff = L @ x
        expected = np.array([1, 2, 3, 4])
        assert np.allclose(diff, expected)
    
    def test_second_order(self):
        """Second order should be second derivative operator."""
        L = tikhonov_matrix(5, order=2)
        
        assert L.shape == (3, 5)
        
        # Linear function should have zero second derivative
        x = np.array([1, 2, 3, 4, 5])
        result = L @ x
        assert np.allclose(result, 0)


class TestChiSquared:
    """Test chi-squared calculation."""
    
    def test_perfect_fit(self):
        """Perfect fit should have chi² = 0."""
        observed = np.array([100, 200, 300])
        expected = np.array([100, 200, 300])
        uncertainty = np.array([10, 14, 17])
        
        chi2 = calculate_chi_squared(observed, expected, uncertainty)
        
        assert chi2 == pytest.approx(0)
    
    def test_one_sigma_deviation(self):
        """One-sigma deviation per point should give chi² = n."""
        observed = np.array([110, 214, 317])
        expected = np.array([100, 200, 300])
        uncertainty = np.array([10, 14, 17])
        
        chi2 = calculate_chi_squared(observed, expected, uncertainty)
        
        assert chi2 == pytest.approx(3)  # 1 + 1 + 1


class TestGaussianResponseMatrix:
    """Test Gaussian response matrix creation."""
    
    def test_matrix_creation(self):
        """Test creating Gaussian response matrix."""
        def fwhm(E):
            return 2.0 + 0.03 * np.sqrt(E)
        
        response = create_gaussian_response_matrix(
            n_channels=500,
            n_energy_bins=25,
            fwhm_function=fwhm,
            energy_range=(0, 2000),
        )
        
        assert response.n_channels == 500
        assert response.n_energy_bins == 25
    
    def test_peaks_at_correct_positions(self):
        """Peaks should be at correct channel positions."""
        def fwhm(E):
            return 5.0
        
        response = create_gaussian_response_matrix(
            n_channels=100,
            n_energy_bins=10,
            fwhm_function=fwhm,
            energy_range=(0, 1000),
        )
        
        # First energy bin centered at 50 keV should peak around channel 5
        first_col = response.matrix[:, 0]
        peak_channel = np.argmax(first_col)
        assert 0 <= peak_channel <= 10


class TestRMLEUnfolding:
    """Test RMLE unfolding algorithm."""
    
    def test_simple_unfolding(self):
        """Test simple unfolding case."""
        # Create simple test case
        n_channels = 100
        n_bins = 10
        
        # Simple response matrix
        R = np.zeros((n_channels, n_bins))
        for j in range(n_bins):
            center = int((j + 0.5) * n_channels / n_bins)
            R[center-2:center+3, j] = 1 / 5
        
        response = ResponseMatrix(matrix=R)
        
        # True source
        true_source = np.ones(n_bins) * 100
        
        # Generate observed data
        expected = R @ true_source
        observed = np.random.poisson(expected.astype(int)).astype(float)
        
        spectrum = SpectrumData(counts=observed)
        
        # Unfold
        result = rmle_unfolding(
            spectrum=spectrum,
            response=response,
            regularization=RegularizationType.TIKHONOV,
            reg_param=0.1,
        )
        
        assert isinstance(result, UnfoldingResult)
        assert len(result.solution) == n_bins
        assert result.converged
    
    def test_positivity_constraint(self):
        """Test that positivity is enforced."""
        n_channels = 50
        n_bins = 5
        
        R = np.eye(n_bins).repeat(10, axis=0)  # Simple mapping
        response = ResponseMatrix(matrix=R)
        
        counts = np.random.poisson(100, n_channels).astype(float)
        spectrum = SpectrumData(counts=counts)
        
        result = rmle_unfolding(
            spectrum=spectrum,
            response=response,
            enforce_positivity=True,
        )
        
        # All solution values should be non-negative
        assert np.all(result.solution >= 0)


class TestPoissonRMLEUnfolding:
    """Test Poisson-likelihood detector-response unfolding."""

    def test_poisson_unfolding_converges(self):
        """Poisson RMLE should converge on a simple well-posed case."""
        n_channels = 60
        n_bins = 6

        # Simple block response, normalized columns
        R = np.zeros((n_channels, n_bins))
        block = n_channels // n_bins
        for j in range(n_bins):
            lo = j * block
            hi = (j + 1) * block
            R[lo:hi, j] = 1.0
        response = ResponseMatrix(matrix=R).normalize_columns()

        true_mu = np.array([0.0, 50.0, 0.0, 80.0, 0.0, 20.0])
        expected = response.matrix @ true_mu
        # Deterministic integer-ish counts (no noise)
        counts = np.round(expected).astype(float)
        spectrum = SpectrumData(counts=counts)

        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.NONE,
            alpha=0.0,
            background_mode="none",
            max_iterations=400,
            tolerance=1e-9,
            positivity=True,
            guardrail_max_reduced_chi2=1e6,
            mc_samples=0,
        )

        result = poisson_rmle_unfolding(spectrum=spectrum, response=response, config=cfg)

        assert isinstance(result, UnfoldingResult)
        assert result.converged
        assert np.all(result.solution >= 0)
        assert result.diagnostics.get("solver") == "poisson_rmle"
        assert result.diagnostics.get("background_mode") == "none"

        # Should approximately recover the true solution on this separable response
        assert result.solution == pytest.approx(true_mu, rel=0.25, abs=2.0)

    def test_poisson_fallback_triggers_on_guardrail(self):
        """If guardrail is impossible, solver should fall back to LS RMLE."""
        n_channels = 40
        n_bins = 4

        R = np.eye(n_bins).repeat(10, axis=0)
        response = ResponseMatrix(matrix=R.astype(float)).normalize_columns()

        true_mu = np.array([10.0, 20.0, 30.0, 40.0])
        counts = (response.matrix @ true_mu).astype(float)
        spectrum = SpectrumData(counts=counts)

        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.NONE,
            alpha=1.0,
            background_mode="none",
            guardrail_max_reduced_chi2=1e-12,
            max_iterations=50,
            mc_samples=0,
        )

        result = poisson_rmle_unfolding(spectrum=spectrum, response=response, config=cfg)
        assert result.diagnostics.get("poisson_fallback") is True
        assert np.all(result.solution >= 0)

    def test_poisson_mc_uncertainty_nonzero(self):
        """Monte Carlo resampling should produce non-zero uncertainties."""
        n_channels = 60
        n_bins = 6

        R = np.zeros((n_channels, n_bins))
        block = n_channels // n_bins
        for j in range(n_bins):
            lo = j * block
            hi = (j + 1) * block
            R[lo:hi, j] = 1.0
        response = ResponseMatrix(matrix=R).normalize_columns()

        true_mu = np.array([10.0, 0.0, 25.0, 0.0, 40.0, 5.0])
        expected = response.matrix @ true_mu + 1.0
        counts = np.random.default_rng(1).poisson(np.maximum(expected, 0.0)).astype(float)
        spectrum = SpectrumData(counts=counts)

        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.SOBLEV_1,
            alpha=0.2,
            background_mode="constant",
            max_iterations=200,
            tolerance=1e-7,
            guardrail_max_reduced_chi2=1e6,
            mc_samples=5,
            random_seed=0,
        )

        result = poisson_rmle_unfolding(spectrum=spectrum, response=response, config=cfg)
        assert result.converged or result.diagnostics.get("poisson_fallback") is True
        assert np.all(result.uncertainty >= 0)
        assert np.any(result.uncertainty > 0)

    def test_poisson_mc_response_sampler_supported(self):
        """Response-operator resampling should be supported and recorded in diagnostics."""
        n_channels = 60
        n_bins = 6

        R = np.zeros((n_channels, n_bins))
        block = n_channels // n_bins
        for j in range(n_bins):
            lo = j * block
            hi = (j + 1) * block
            R[lo:hi, j] = 1.0
        response = ResponseMatrix(matrix=R).normalize_columns()

        true_mu = np.array([10.0, 0.0, 25.0, 0.0, 40.0, 5.0])
        expected = response.matrix @ true_mu + 2.0
        counts = np.random.default_rng(2).poisson(np.maximum(expected, 0.0)).astype(float)
        spectrum = SpectrumData(counts=counts)

        base = response.matrix.copy()

        def sampler(rng: np.random.Generator) -> ResponseMatrix:
            pert = base + rng.normal(0.0, 0.02, size=base.shape)
            pert = np.clip(pert, 0.0, None)
            return ResponseMatrix(matrix=pert).normalize_columns()

        cfg = PoissonRMLEConfig(
            penalty=PoissonPenalty.SOBLEV_1,
            alpha=0.2,
            background_mode="constant",
            max_iterations=200,
            tolerance=1e-7,
            guardrail_max_reduced_chi2=1e6,
            mc_samples=10,
            random_seed=123,
            response_sampler=sampler,
        )

        result = poisson_rmle_unfolding(spectrum=spectrum, response=response, config=cfg)
        assert result.diagnostics.get("mc_samples") == 10
        assert result.diagnostics.get("mc_response_sampling") is True
        assert np.all(result.uncertainty >= 0)
        assert np.any(result.uncertainty > 0)


class TestLambdaSelection:
    """Test regularization parameter selection."""
    
    def test_lcurve_returns_value(self):
        """L-curve should return a regularization parameter."""
        n_channels = 50
        n_bins = 10
        
        R = np.random.rand(n_channels, n_bins)
        d = np.random.rand(n_channels)
        W = np.eye(n_channels)
        L = np.eye(n_bins)
        
        lam = select_lambda_lcurve(d, R, W, L, n_points=10)
        
        assert lam > 0
    
    def test_gcv_returns_value(self):
        """GCV should return a regularization parameter."""
        n_channels = 50
        n_bins = 10
        
        R = np.random.rand(n_channels, n_bins)
        d = np.random.rand(n_channels)
        W = np.eye(n_channels)
        L = np.eye(n_bins)
        
        lam = select_lambda_gcv(d, R, W, L, n_points=10)
        
        assert lam > 0


class TestPeakFitting:
    """Test peak fitting functionality."""
    
    def test_single_peak_fit(self):
        """Test fitting a single peak."""
        # Create spectrum with one peak
        channels = np.arange(1000)
        continuum = 50 + 0.01 * channels
        peak = 500 * np.exp(-0.5 * ((channels - 500) / 5)**2)
        counts = continuum + peak
        
        # Add noise
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        spectrum = SpectrumData(counts=counts)
        
        def fwhm(E):
            return 12  # ~5 sigma * 2.355
        
        result = fit_peaks_mle(
            spectrum=spectrum,
            peak_positions=[500],
            fwhm_function=fwhm,
            fit_range=(400, 600),
        )
        
        assert result.n_peaks == 1
        # Peak should be found near 500
        assert result.peaks[0].centroid == pytest.approx(500, rel=0.1)


class TestActivityCalculation:
    """Test activity calculation from peaks."""
    
    def test_simple_activity(self):
        """Test basic activity calculation."""
        peak = PeakModel(
            centroid=500,
            sigma=5,
            amplitude=1000,
            amplitude_unc=50,
        )
        
        activity, unc = calculate_activity_from_peak(
            peak=peak,
            efficiency=0.01,
            intensity=0.9,
            live_time_s=3600,
        )
        
        # Activity should be counts / (eff * I * t)
        net_counts = 1000 * 5 * np.sqrt(2 * np.pi)
        expected_activity = net_counts / (0.01 * 0.9 * 3600)
        
        assert activity == pytest.approx(expected_activity, rel=0.01)
        assert unc > 0


class TestRMLEArtifact:
    """Test RMLE artifact class."""
    
    def test_artifact_creation(self):
        """Test artifact creation."""
        counts = np.random.poisson(100, 500)
        spectrum = SpectrumData(counts=counts)
        
        result = UnfoldingResult(
            solution=np.random.rand(10),
            uncertainty=np.random.rand(10) * 0.1,
            chi_squared=25.0,
            converged=True,
        )
        
        artifact = RMLEArtifact(spectrum=spectrum, result=result)
        
        assert artifact.spectrum.n_channels == 500
    
    def test_add_activities(self):
        """Test adding activity results."""
        counts = np.random.poisson(100, 500)
        spectrum = SpectrumData(counts=counts)
        
        result = UnfoldingResult(
            solution=np.random.rand(10),
            uncertainty=np.random.rand(10) * 0.1,
        )
        
        artifact = RMLEArtifact(spectrum=spectrum, result=result)
        
        activity = ActivityResult(
            nuclide="Co-60",
            activity_Bq=1000,
            uncertainty_Bq=50,
            gamma_energy_keV=1332.5,
            peak_counts=5000,
        )
        
        artifact.add_activity(activity)
        
        assert len(artifact.activities) == 1
        assert artifact.activities[0].nuclide == "Co-60"
    
    def test_artifact_serialization(self):
        """Test artifact export to dict."""
        counts = np.random.poisson(100, 500)
        spectrum = SpectrumData(counts=counts, live_time_s=3600)
        
        result = UnfoldingResult(
            solution=np.random.rand(10),
            uncertainty=np.random.rand(10) * 0.1,
            chi_squared=25.0,
            regularization_param=0.1,
            converged=True,
        )
        
        artifact = RMLEArtifact(spectrum=spectrum, result=result)
        
        data = artifact.to_dict()
        
        assert "schema" in data
        assert data["n_channels"] == 500
        assert data["live_time_s"] == 3600
        assert data["chi_squared"] == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
