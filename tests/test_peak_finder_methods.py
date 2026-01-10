"""
Tests for peak finding algorithms.

Compares all peak finder methods:
- SimplePeakFinder (threshold-based with SNIP background)
- WindowPeakFinder (local window statistics, from peakingduck)
- ChunkedPeakFinder (spectrum subdivision)
- ScipyPeakFinder (scipy with Savitzky-Golay smoothing)
- SegmentedPeakFinder (region-specific thresholds, from rafm_analysis)
- DerivativePeakFinder (first derivative zero-crossing)
- SecondDifferencePeakFinder (discrete Laplacian)
"""

import pytest
import numpy as np


def generate_synthetic_spectrum(
    n_channels: int = 4096,
    peak_positions: list = None,
    peak_amplitudes: list = None,
    peak_widths: list = None,
    background_level: float = 100.0,
    noise_scale: float = 1.0
) -> np.ndarray:
    """Generate a synthetic gamma spectrum for testing."""
    if peak_positions is None:
        peak_positions = [500, 1000, 1500, 2500, 3000, 3500]
    if peak_amplitudes is None:
        peak_amplitudes = [1000, 2000, 500, 1500, 800, 300]
    if peak_widths is None:
        peak_widths = [3, 4, 3, 5, 4, 3]
    
    channels = np.arange(n_channels)
    
    # Background: exponential decay + constant
    background = background_level * (1 + 2 * np.exp(-channels / 1000))
    
    # Add peaks
    spectrum = background.copy()
    for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
        spectrum += amp * np.exp(-0.5 * ((channels - pos) / width) ** 2)
    
    # Add Poisson noise
    if noise_scale > 0:
        spectrum = np.random.poisson(spectrum * noise_scale) / noise_scale
    
    return spectrum.astype(float)


class TestSNIPBackground:
    """Tests for SNIP background estimation."""
    
    def test_snip_removes_peaks(self):
        """SNIP should produce a smooth background below peaks."""
        from fluxforge.analysis.peak_finders import snip_background
        
        spectrum = generate_synthetic_spectrum(noise_scale=0)
        background = snip_background(spectrum, n_iterations=24)
        
        # Background should be everywhere <= spectrum
        assert np.all(background <= spectrum + 1)  # Small tolerance for numerical
        
        # At peak positions, background should be significantly below spectrum
        peak_pos = 1000
        assert spectrum[peak_pos] - background[peak_pos] > 500
    
    def test_snip_flat_background(self):
        """SNIP on flat spectrum should return similar values."""
        from fluxforge.analysis.peak_finders import snip_background
        
        flat = np.ones(1000) * 100
        background = snip_background(flat, n_iterations=10)
        
        # Should be close to 100 everywhere - SNIP on flat should be accurate
        assert np.allclose(background, 100, rtol=0.1)  # Tightened from 0.5


class TestSimplePeakFinder:
    """Tests for SimplePeakFinder."""
    
    def test_finds_peaks(self):
        """Should find peaks above threshold."""
        from fluxforge.analysis.peak_finders import SimplePeakFinder
        
        spectrum = generate_synthetic_spectrum(noise_scale=0.1)
        finder = SimplePeakFinder(threshold_sigma=3.0, min_counts=50)
        peaks = finder.find_peaks(spectrum)
        
        # Should find at least 4 of the 6 peaks
        assert len(peaks) >= 4
        
        # Most peaks should have significant values
        high_value_peaks = [p for p in peaks if p.value >= 100]
        assert len(high_value_peaks) >= 3
    
    def test_unified_api(self):
        """Test that find() and find_peaks() both work."""
        from fluxforge.analysis.peak_finders import SimplePeakFinder
        
        spectrum = generate_synthetic_spectrum()
        finder = SimplePeakFinder()
        
        peaks1 = finder.find(spectrum)
        peaks2 = finder.find_peaks(spectrum)
        
        assert len(peaks1) == len(peaks2)


class TestWindowPeakFinder:
    """Tests for WindowPeakFinder (peakingduck-style)."""
    
    def test_finds_peaks(self):
        """Should find peaks using local window statistics."""
        from fluxforge.analysis.peak_finders import WindowPeakFinder
        
        spectrum = generate_synthetic_spectrum(noise_scale=0.1)
        finder = WindowPeakFinder(threshold_sigma=3.0, window_size=50)
        peaks = finder.find_peaks(spectrum)
        
        assert len(peaks) >= 3
    
    def test_window_size_parameter(self):
        """Window finder should work with different window sizes."""
        from fluxforge.analysis.peak_finders import WindowPeakFinder
        
        # Create clear peaks for reliable detection
        spectrum = generate_synthetic_spectrum(
            peak_positions=[1000, 2000, 3000],
            peak_amplitudes=[3000, 4000, 2500],
            peak_widths=[5, 5, 5],
            noise_scale=0.1
        )
        
        finder = WindowPeakFinder(threshold_sigma=3.0, window_size=60)
        peaks = finder.find_peaks(spectrum)
        
        # Should find at least one peak
        assert len(peaks) >= 1


class TestSegmentedPeakFinder:
    """Tests for SegmentedPeakFinder (rafm_analysis-style)."""
    
    def test_finds_peaks_in_regions(self):
        """Should find peaks with region-specific parameters."""
        from fluxforge.analysis.peak_finders import SegmentedPeakFinder
        
        spectrum = generate_synthetic_spectrum()
        finder = SegmentedPeakFinder(
            splits=[1500, 3000],
            sigmas=[2.0, 2.0, 2.0],
            gaussian_refine=True
        )
        peaks = finder.find_peaks(spectrum)
        
        assert len(peaks) >= 3
    
    def test_gaussian_refinement(self):
        """With Gaussian refinement, centroids should be more accurate."""
        from fluxforge.analysis.peak_finders import SegmentedPeakFinder
        
        # Create spectrum with known peak at exact position
        spectrum = generate_synthetic_spectrum(
            peak_positions=[1000],
            peak_amplitudes=[2000],
            peak_widths=[4],
            noise_scale=0
        )
        
        finder = SegmentedPeakFinder(gaussian_refine=True)
        peaks = finder.find_peaks(spectrum)
        
        # Should find the peak
        assert len(peaks) >= 1
        
        # Find the peak near 1000
        peak_1000 = [p for p in peaks if abs(p.index - 1000) < 20]
        assert len(peak_1000) >= 1
        
        # Centroid should be close to 1000
        if peak_1000[0].centroid is not None:
            assert abs(peak_1000[0].centroid - 1000) < 2


class TestDerivativePeakFinder:
    """Tests for DerivativePeakFinder."""
    
    def test_finds_peaks(self):
        """Should find peaks using derivative analysis."""
        from fluxforge.analysis.peak_finders import DerivativePeakFinder
        
        spectrum = generate_synthetic_spectrum(noise_scale=0.1)
        finder = DerivativePeakFinder(threshold_sigma=3.0)
        peaks = finder.find_peaks(spectrum)
        
        assert len(peaks) >= 2


class TestSecondDifferencePeakFinder:
    """Tests for SecondDifferencePeakFinder."""
    
    def test_finds_peaks(self):
        """Should find peaks using second difference."""
        from fluxforge.analysis.peak_finders import SecondDifferencePeakFinder
        
        spectrum = generate_synthetic_spectrum(noise_scale=0.1)
        finder = SecondDifferencePeakFinder(threshold_sigma=4.0)
        peaks = finder.find_peaks(spectrum)
        
        assert len(peaks) >= 2


class TestPeakFinderFactory:
    """Tests for get_peak_finder factory function."""
    
    def test_all_methods_available(self):
        """All registered methods should be instantiable."""
        from fluxforge.analysis.peak_finders import get_peak_finder, PEAK_FINDER_METHODS
        
        for method in PEAK_FINDER_METHODS:
            finder = get_peak_finder(method)
            assert hasattr(finder, 'find_peaks')
            assert hasattr(finder, 'find')
    
    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        from fluxforge.analysis.peak_finders import get_peak_finder
        
        with pytest.raises(ValueError):
            get_peak_finder('unknown_method')


class TestMultiMethodPeakFinding:
    """Tests for consensus peak finding."""
    
    def test_consensus_peaks(self):
        """Multi-method should find consensus peaks."""
        from fluxforge.analysis.peak_finders import find_peaks_multi_method
        
        spectrum = generate_synthetic_spectrum(noise_scale=0.1)
        peaks = find_peaks_multi_method(
            spectrum,
            methods=['simple', 'window', 'scipy'],
            consensus_threshold=2
        )
        
        # Should find peaks agreed upon by at least 2 methods
        assert len(peaks) >= 2


class TestWeightedMovingAverage:
    """Tests for weighted moving average smoother."""
    
    def test_smoothing(self):
        """Should smooth noisy data."""
        from fluxforge.analysis.peak_finders import weighted_moving_average
        
        # Create noisy data
        x = np.linspace(0, 10, 100)
        noisy = np.sin(x) + 0.3 * np.random.randn(100)
        
        smoothed = weighted_moving_average(noisy, window_size=7)
        
        # Smoothed should have less variance
        assert np.std(smoothed) < np.std(noisy)
    
    def test_preserves_shape(self):
        """Output should have same shape as input."""
        from fluxforge.analysis.peak_finders import weighted_moving_average
        
        data = np.random.randn(500)
        smoothed = weighted_moving_average(data, window_size=11)
        
        assert len(smoothed) == len(data)


class TestPeakRefinement:
    """Tests for peak centroid refinement."""
    
    def test_centroid_refinement(self):
        """Centroid refinement should improve peak positions."""
        from fluxforge.analysis.peak_finders import (
            SimplePeakFinder, refine_peak_centroids
        )
        
        # Create spectrum with peak slightly off-center
        spectrum = generate_synthetic_spectrum(
            peak_positions=[1000],
            peak_amplitudes=[2000],
            peak_widths=[4],
            noise_scale=0
        )
        
        finder = SimplePeakFinder(threshold_sigma=3.0)
        peaks = finder.find_peaks(spectrum)
        refined = refine_peak_centroids(spectrum, peaks, width=5)
        
        # Should have refined centroids
        for p in refined:
            assert p.centroid is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
