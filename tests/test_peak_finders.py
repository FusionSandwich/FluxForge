"""Tests for peak finding algorithms."""

import numpy as np
import pytest
from fluxforge.analysis.peakfit import (
    auto_find_peaks,
    window_peak_finder,
    chunked_peak_finder,
)


def create_synthetic_spectrum(
    n_channels: int = 1024,
    peak_channels: list = None,
    peak_amplitudes: list = None,
    peak_widths: list = None,
    background_level: float = 100.0,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Create synthetic spectrum with known peaks."""
    np.random.seed(seed)
    
    if peak_channels is None:
        peak_channels = [200, 400, 600, 800]
    if peak_amplitudes is None:
        peak_amplitudes = [500, 1000, 750, 300]
    if peak_widths is None:
        peak_widths = [3.0, 4.0, 3.5, 2.5]
    
    channels = np.arange(n_channels)
    
    # Create background (slight slope)
    background = background_level + 0.02 * channels
    
    # Add peaks
    counts = background.copy()
    for ch, amp, width in zip(peak_channels, peak_amplitudes, peak_widths):
        counts += amp * np.exp(-(channels - ch)**2 / (2 * width**2))
    
    # Add Poisson noise
    counts = np.random.poisson(counts.astype(int)).astype(float)
    
    # Add Gaussian noise
    counts += noise_scale * np.sqrt(counts) * np.random.randn(n_channels)
    counts = np.maximum(counts, 0)
    
    return channels, counts, peak_channels


class TestAutoFindPeaks:
    """Tests for the default scipy-based peak finder."""
    
    def test_finds_obvious_peaks(self):
        """Test that obvious peaks are found."""
        channels, counts, true_peaks = create_synthetic_spectrum()
        
        found = auto_find_peaks(channels, counts, threshold=3.0, min_distance=10)
        
        # Should find at least 3 of the 4 peaks
        assert len(found) >= 3
        
        # Check that found peaks are near true peaks
        found_channels = [p[0] for p in found]
        for true_ch in true_peaks[:3]:  # Check first 3 (brightest)
            matches = [f for f in found_channels if abs(f - true_ch) < 15]
            assert len(matches) >= 1, f"Peak at {true_ch} not found"
    
    def test_threshold_sensitivity(self):
        """Test that higher threshold finds fewer peaks."""
        channels, counts, _ = create_synthetic_spectrum()
        
        peaks_low = auto_find_peaks(channels, counts, threshold=2.0)
        peaks_high = auto_find_peaks(channels, counts, threshold=5.0)
        
        assert len(peaks_low) >= len(peaks_high)
    
    def test_min_distance_enforced(self):
        """Test that minimum distance between peaks is respected."""
        channels, counts, _ = create_synthetic_spectrum()
        
        min_dist = 20
        peaks = auto_find_peaks(channels, counts, threshold=2.0, min_distance=min_dist)
        
        if len(peaks) > 1:
            for i in range(len(peaks) - 1):
                assert abs(peaks[i+1][0] - peaks[i][0]) >= min_dist


class TestWindowPeakFinder:
    """Tests for the window-based peak finder."""
    
    def test_finds_peaks_with_window_method(self):
        """Test window peak finder finds peaks."""
        channels, counts, true_peaks = create_synthetic_spectrum()
        
        found = window_peak_finder(
            channels, counts,
            threshold=2.0,
            inner_window=2,
            outer_window=30,
            min_distance=5
        )
        
        # Should find some peaks
        assert len(found) >= 2
        
        # Check format (channel, counts, significance)
        for peak in found:
            assert len(peak) == 3
            assert isinstance(peak[0], int)
            assert peak[2] > 0  # Positive significance
    
    def test_enforce_maximum(self):
        """Test that enforce_maximum removes non-maximum detections."""
        channels, counts, _ = create_synthetic_spectrum()
        
        # With enforce_maximum=True, should get clean peaks
        peaks_enforced = window_peak_finder(
            channels, counts,
            threshold=2.0,
            enforce_maximum=True
        )
        
        # Each peak should be a local maximum
        for ch, val, _ in peaks_enforced:
            idx = np.where(channels == ch)[0][0]
            if idx > 0 and idx < len(counts) - 1:
                assert counts[idx] >= counts[idx-1], f"Peak at {ch} is not local max"
                assert counts[idx] >= counts[idx+1], f"Peak at {ch} is not local max"
    
    def test_varying_window_size(self):
        """Test that larger window gives more stable estimates."""
        channels, counts, _ = create_synthetic_spectrum(noise_scale=2.0)
        
        peaks_small = window_peak_finder(channels, counts, outer_window=10)
        peaks_large = window_peak_finder(channels, counts, outer_window=50)
        
        # Both should find some peaks
        assert len(peaks_small) > 0
        assert len(peaks_large) > 0


class TestChunkedPeakFinder:
    """Tests for the chunked peak finder."""
    
    def test_finds_peaks_in_chunks(self):
        """Test chunked finder finds peaks across spectrum."""
        channels, counts, true_peaks = create_synthetic_spectrum()
        
        found = chunked_peak_finder(channels, counts, threshold=3.0, n_chunks=8)
        
        # Should find peaks
        assert len(found) >= 2
        
        # Check format (channel, significance)
        for peak in found:
            assert len(peak) == 2
            assert isinstance(peak[0], int)
    
    def test_handles_varying_background(self):
        """Test chunked finder handles varying background levels."""
        # Create spectrum with different background in different regions
        channels = np.arange(1024)
        background = np.where(channels < 512, 50.0, 200.0)
        
        # Add peaks in both regions
        counts = background.copy()
        counts += 300 * np.exp(-(channels - 250)**2 / 18)  # Low bg region
        counts += 500 * np.exp(-(channels - 750)**2 / 18)  # High bg region
        
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        found = chunked_peak_finder(channels, counts, threshold=2.5, n_chunks=4)
        
        # Should find peaks in both regions
        low_bg_peaks = [p for p in found if p[0] < 512]
        high_bg_peaks = [p for p in found if p[0] >= 512]
        
        assert len(low_bg_peaks) >= 1, "Should find peak in low background region"
        assert len(high_bg_peaks) >= 1, "Should find peak in high background region"
    
    def test_chunk_count_variation(self):
        """Test that different chunk counts work."""
        channels, counts, _ = create_synthetic_spectrum()
        
        for n_chunks in [4, 8, 16]:
            found = chunked_peak_finder(channels, counts, n_chunks=n_chunks)
            # Should work without error
            assert isinstance(found, list)


class TestAutoFindPeaksWithMethod:
    """Test auto_find_peaks with different finder methods."""
    
    def test_scipy_method(self):
        """Test default scipy method."""
        channels, counts, _ = create_synthetic_spectrum()
        found = auto_find_peaks(channels, counts, finder_method='scipy')
        assert len(found) >= 2
    
    def test_window_method(self):
        """Test window method through auto_find_peaks."""
        channels, counts, _ = create_synthetic_spectrum()
        found = auto_find_peaks(channels, counts, finder_method='window')
        assert len(found) >= 2
        # Should return 2-tuples
        assert len(found[0]) == 2
    
    def test_chunked_method(self):
        """Test chunked method through auto_find_peaks."""
        channels, counts, _ = create_synthetic_spectrum()
        found = auto_find_peaks(channels, counts, finder_method='chunked')
        assert len(found) >= 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_spectrum(self):
        """Test with very small spectrum."""
        channels = np.arange(100)
        counts = 50 + 200 * np.exp(-(channels - 50)**2 / 18)
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        # Should work without error
        found = auto_find_peaks(channels, counts)
        assert isinstance(found, list)
    
    def test_flat_spectrum(self):
        """Test with flat spectrum (no peaks)."""
        channels = np.arange(512)
        counts = np.full(512, 100.0)
        np.random.seed(42)
        counts += np.random.randn(512) * 3
        
        found = auto_find_peaks(channels, counts, threshold=5.0)
        # Should find very few or no peaks
        assert len(found) <= 3
    
    def test_single_peak(self):
        """Test spectrum with single peak."""
        np.random.seed(42)  # Set seed for reproducibility
        channels = np.arange(512)
        counts = 100 + 500 * np.exp(-(channels - 256)**2 / 18)
        counts = np.random.poisson(counts.astype(int)).astype(float)
        
        found = auto_find_peaks(channels, counts, threshold=3.0)
        
        # Should find at least one peak near 256
        assert len(found) >= 1
        # Find the peak closest to 256
        closest_peak = min(found, key=lambda p: abs(p[0] - 256))
        peak_channel = closest_peak[0]
        assert abs(peak_channel - 256) < 15, f"Peak at {peak_channel} not near expected 256"
