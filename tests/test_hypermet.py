"""Tests for Hypermet peak model (Theuerkauf-style)."""

import pytest
import numpy as np

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from fluxforge.analysis.peakfit import (
    HypermetPeak,
    GaussianPeak,
    hypermet_function,
    fit_hypermet_peak,
)


# =============================================================================
# HypermetPeak dataclass tests
# =============================================================================

class TestHypermetPeakDataclass:
    """Test the HypermetPeak dataclass properties."""
    
    def test_pure_gaussian(self):
        """Hypermet with no tails = Gaussian."""
        peak = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
        )
        assert peak.has_left_tail is False
        assert peak.has_right_tail is False
        assert peak.has_step is False
        
    def test_with_left_tail(self):
        """Hypermet with left tail enabled."""
        peak = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
            tail_left=4.0,
            tail_left_fraction=0.1,
        )
        assert peak.has_left_tail is True
        assert peak.has_right_tail is False
        
    def test_with_step(self):
        """Hypermet with step function."""
        peak = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
            step_height=0.05,
        )
        assert peak.has_step is True
        
    def test_fwhm_calculation(self):
        """FWHM = 2.355 * sigma."""
        sigma = 2.5
        peak = HypermetPeak(centroid=100.0, amplitude=1000.0, sigma=sigma)
        expected_fwhm = 2.355 * sigma
        assert np.isclose(peak.fwhm, expected_fwhm, rtol=1e-3)
        
    def test_area_pure_gaussian(self):
        """Area for pure Gaussian = amplitude * sigma * sqrt(2*pi)."""
        amplitude = 1000.0
        sigma = 2.0
        peak = HypermetPeak(centroid=100.0, amplitude=amplitude, sigma=sigma)
        expected_area = amplitude * sigma * np.sqrt(2 * np.pi)
        # Allow some tolerance for the approximation
        assert np.isclose(peak.area, expected_area, rtol=0.1)


# =============================================================================
# Hypermet evaluate() method tests
# =============================================================================

class TestHypermetEvaluate:
    """Test the HypermetPeak.evaluate() method."""
    
    def test_evaluate_returns_gaussian_shape(self):
        """Pure Hypermet should return Gaussian shape."""
        peak = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
        )
        x = np.linspace(90, 110, 101)
        y = peak.evaluate(x)
        
        # Maximum should be at centroid
        max_idx = np.argmax(y)
        assert np.isclose(x[max_idx], 100.0, atol=0.5)
        
        # Maximum value should be amplitude
        assert np.isclose(y.max(), 1000.0, rtol=0.01)
        
    def test_evaluate_left_tail_asymmetry(self):
        """Left tail should create asymmetry."""
        peak = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
            tail_left=4.0,
            tail_left_fraction=0.15,
        )
        x = np.linspace(80, 120, 201)
        y = peak.evaluate(x)
        
        # Find values at equal distances from centroid
        y_left = y[x == 95.0][0]  # 5 channels left
        y_right = y[x == 105.0][0]  # 5 channels right
        
        # Left side should be higher due to tail
        assert y_left > y_right
        
    def test_evaluate_step_adds_baseline(self):
        """Step function adds raised baseline on left side."""
        peak_no_step = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
        )
        peak_with_step = HypermetPeak(
            centroid=100.0,
            amplitude=1000.0,
            sigma=2.0,
            step_height=0.02,
        )
        x = np.linspace(80, 120, 201)
        y_no_step = peak_no_step.evaluate(x)
        y_with_step = peak_with_step.evaluate(x)
        
        # Far left should show step contribution
        left_diff = y_with_step[0] - y_no_step[0]
        assert left_diff > 0
        
        # Far right should be nearly same (step -> 0)
        right_diff = y_with_step[-1] - y_no_step[-1]
        assert right_diff < left_diff


# =============================================================================
# Hypermet function tests
# =============================================================================

class TestHypermetFunction:
    """Test the standalone hypermet_function."""
    
    def test_pure_gaussian_matches_numpy(self):
        """Hypermet without tails should match Gaussian."""
        x = np.linspace(90, 110, 101)
        amplitude, centroid, sigma = 1000.0, 100.0, 2.0
        
        # Hypermet with no tails
        y_hyper = hypermet_function(x, amplitude, centroid, sigma, 0, 0, 0)
        
        # Pure Gaussian
        z = (x - centroid) / sigma
        y_gauss = amplitude * np.exp(-0.5 * z**2)
        
        assert np.allclose(y_hyper, y_gauss)
        
    def test_tail_adds_asymmetry(self):
        """Left tail should add counts on left side."""
        x = np.linspace(90, 110, 101)
        
        y_no_tail = hypermet_function(x, 1000, 100, 2, 0, 0, 0)
        y_with_tail = hypermet_function(x, 1000, 100, 2, 4.0, 0.1, 0)
        
        # Left half should have more counts with tail
        left_mask = x < 100
        left_increase = np.sum(y_with_tail[left_mask] - y_no_tail[left_mask])
        assert left_increase > 0
        
    def test_step_adds_baseline(self):
        """Step adds raised baseline on low-energy side."""
        x = np.linspace(70, 130, 201)
        
        y_no_step = hypermet_function(x, 1000, 100, 2, 0, 0, 0)
        y_with_step = hypermet_function(x, 1000, 100, 2, 0, 0, 0.02)
        
        # Far left should show step contribution
        assert y_with_step[0] > y_no_step[0]


# =============================================================================
# Hypermet fitting tests
# =============================================================================

class TestFitHypermetPeak:
    """Test fit_hypermet_peak function."""
    
    def test_fit_pure_gaussian_recovery(self):
        """Fit should recover pure Gaussian parameters."""
        # Create synthetic Gaussian spectrum
        channels = np.arange(200)
        centroid, amplitude, sigma = 100.0, 5000.0, 2.5
        background = 50.0
        
        z = (channels - centroid) / sigma
        counts = amplitude * np.exp(-0.5 * z**2) + background
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        peak, result = fit_hypermet_peak(
            channels, counts,
            peak_channel=100,
            fit_width=15,
            enable_tail=False,
            enable_step=False,
        )
        
        assert result.success
        assert np.isclose(peak.centroid, centroid, atol=0.5)
        assert np.isclose(peak.sigma, sigma, rtol=0.2)
        
    def test_fit_with_tail_enabled(self):
        """Fit with tail enabled should converge."""
        # Create synthetic spectrum using the actual Hypermet function
        channels = np.arange(200)
        centroid, amplitude, sigma = 100.0, 5000.0, 2.5
        background = 50.0
        
        # Use hypermet_function to create realistic test data
        counts = hypermet_function(
            channels.astype(float),
            amplitude, centroid, sigma,
            tail_left=5.0,
            tail_left_frac=0.1,
            step_height=0.0
        ) + background
        
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        peak, result = fit_hypermet_peak(
            channels, counts,
            peak_channel=100,
            fit_width=15,
            enable_tail=True,
            enable_step=False,
        )
        
        assert result.success
        assert peak.has_left_tail
        assert np.isclose(peak.centroid, centroid, atol=1.0)
        
    def test_fit_returns_compatible_result(self):
        """PeakFitResult should be compatible with existing code."""
        channels = np.arange(200)
        centroid, amplitude, sigma = 100.0, 5000.0, 2.5
        background = 50.0
        
        z = (channels - centroid) / sigma
        counts = amplitude * np.exp(-0.5 * z**2) + background
        
        peak, result = fit_hypermet_peak(
            channels, counts,
            peak_channel=100,
            fit_width=15,
        )
        
        # Check PeakFitResult has expected attributes
        assert hasattr(result, 'peak')
        assert hasattr(result, 'chi_squared')
        assert hasattr(result, 'dof')
        assert hasattr(result, 'residuals')
        assert result.background_model == 'hypermet'


# =============================================================================
# Integration with GaussianPeak
# =============================================================================

class TestHypermetGaussianCompatibility:
    """Test that Hypermet results work with Gaussian-based code."""
    
    def test_hypermet_peak_has_gaussian_properties(self):
        """HypermetPeak should have same properties as GaussianPeak."""
        hyper = HypermetPeak(centroid=100, amplitude=1000, sigma=2)
        gauss = GaussianPeak(centroid=100, amplitude=1000, sigma=2)
        
        # Both should have these attributes
        assert hasattr(hyper, 'centroid')
        assert hasattr(hyper, 'amplitude')
        assert hasattr(hyper, 'sigma')
        assert hasattr(hyper, 'fwhm')
        assert hasattr(hyper, 'area')
        
        # FWHM should match
        assert np.isclose(hyper.fwhm, gauss.fwhm)
