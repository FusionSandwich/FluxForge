"""
Advanced Peak Finding Algorithms

Includes multiple peak finding strategies:
- WindowPeakFinder: Local window-based detection with configurable thresholds
- ChunkedPeakFinder: Spectrum subdivision for adaptive thresholding
- ScipyPeakFinder: Scipy-based with Savitzky-Golay smoothing
- SNIPBackground: Statistics-sensitive Non-linear Iterative Peak-clipping

Based on implementations from peakingduck and ROOT's TSpectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal


@dataclass
class PeakInfo:
    """Information about a detected peak."""
    index: int
    value: float
    centroid: Optional[float] = None  # Sub-channel centroid if refined
    fwhm: Optional[float] = None
    area: Optional[float] = None
    significance: Optional[float] = None  # Peak significance (sigma)
    
    def __repr__(self) -> str:
        if self.centroid is not None:
            return f"PeakInfo(channel={self.index}, centroid={self.centroid:.2f}, counts={self.value:.0f})"
        return f"PeakInfo(channel={self.index}, counts={self.value:.0f})"


def snip_background(
    spectrum: np.ndarray,
    n_iterations: int = 24,
    clipping_window: int = 1,
    lls_transform: bool = True
) -> np.ndarray:
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) background estimation.
    
    The SNIP algorithm iteratively clips peaks by comparing each point to
    the average of its neighbors, with optional LLS (Log-Log-Square-root)
    transformation for better handling of Poisson statistics.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum counts
    n_iterations : int
        Number of clipping iterations (default 24)
    clipping_window : int
        Initial window size (increases during iterations)
    lls_transform : bool
        Apply LLS transformation for Poisson data
    
    Returns
    -------
    np.ndarray
        Estimated background
        
    References
    ----------
    C.G. Ryan et al., "SNIP, a statistics-sensitive background treatment for 
    the quantitative analysis of PIXE spectra in geoscience applications", 
    Nuclear Instruments and Methods in Physics Research B, 34 (1988) 396-402.
    """
    n = len(spectrum)
    y = np.copy(spectrum).astype(float)
    
    # LLS transformation: v = log(log(sqrt(y + 1) + 1) + 1)
    if lls_transform:
        y = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    
    background = np.copy(y)
    
    for p in range(1, n_iterations + 1):
        # Window size increases with iteration
        window = clipping_window * p
        
        for i in range(window, n - window):
            # Average of neighbors at distance 'window'
            left = background[i - window]
            right = background[i + window]
            avg = (left + right) / 2.0
            
            # Keep minimum of current value and neighbor average
            background[i] = min(background[i], avg)
    
    # Inverse LLS transformation
    if lls_transform:
        background = (np.exp(np.exp(background) - 1) - 1) ** 2 - 1
        background = np.maximum(background, 0)
    
    return background


def estimate_background_linear(
    spectrum: np.ndarray,
    window_size: int = 50
) -> np.ndarray:
    """
    Simple linear interpolation background estimate.
    
    Uses local minima within sliding windows to estimate background.
    """
    n = len(spectrum)
    background = np.zeros(n)
    
    # Find local minima in windows
    minima_indices = []
    minima_values = []
    
    for i in range(0, n, window_size):
        end = min(i + window_size, n)
        window = spectrum[i:end]
        local_min_idx = np.argmin(window) + i
        minima_indices.append(local_min_idx)
        minima_values.append(spectrum[local_min_idx])
    
    # Interpolate between minima
    background = np.interp(np.arange(n), minima_indices, minima_values)
    
    return background


def savitzky_golay_smooth(
    data: np.ndarray,
    window_size: int = 11,
    order: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    window_size : int
        Must be odd, size of smoothing window
    order : int
        Polynomial order for fitting
    
    Returns
    -------
    np.ndarray
        Smoothed data
    """
    if window_size % 2 == 0:
        window_size += 1
    return signal.savgol_filter(data, window_size, order)


class SimplePeakFinder:
    """
    Simple threshold-based peak finder.
    
    Finds channels where the value exceeds a threshold based on
    the local or global mean.
    
    Unified API:
        finder = SimplePeakFinder(threshold_sigma=5.0)
        peaks = finder.find_peaks(spectrum)  # or finder.find(spectrum)
    """
    
    def __init__(self, threshold: float = 3.0, threshold_sigma: float = None,
                 min_counts: float = 10.0, min_distance: int = 3):
        """
        Parameters
        ----------
        threshold : float
            Number of standard deviations above background (legacy name)
        threshold_sigma : float, optional
            Alias for threshold (preferred name)
        min_counts : float
            Minimum net counts to consider a peak
        min_distance : int
            Minimum distance between peaks in channels
        """
        self.threshold = threshold_sigma if threshold_sigma is not None else threshold
        self.min_counts = min_counts
        self.min_distance = min_distance
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum (alias for find_peaks)."""
        return self.find_peaks(spectrum)
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum."""
        # Estimate background using SNIP
        background = snip_background(spectrum)
        net = spectrum - background
        
        # Estimate noise from Poisson statistics of background
        # Use sqrt(background) as expected standard deviation
        noise_estimate = np.sqrt(np.maximum(background, 1.0))
        
        # Also consider negative fluctuations for noise estimate
        negative_vals = net[net < 0]
        if len(negative_vals) > 10:
            empirical_noise = np.std(negative_vals) * np.sqrt(2)  # Fold negative half
        else:
            empirical_noise = np.mean(noise_estimate)
        
        # Use the larger of the two estimates for robustness
        noise = np.maximum(noise_estimate, empirical_noise)
        
        # Find peaks above threshold
        significance = net / noise
        peak_mask = (significance > self.threshold) & (net > self.min_counts)
        
        # Find peak indices (local maxima above threshold)
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if peak_mask[i] and spectrum[i] >= spectrum[i-1] and spectrum[i] >= spectrum[i+1]:
                peaks.append(PeakInfo(
                    index=i,
                    value=float(spectrum[i]),
                    significance=float(significance[i]),
                    area=float(net[i])  # Approximate area as net counts at peak
                ))
        
        # Filter by minimum distance
        if self.min_distance > 1 and len(peaks) > 1:
            filtered = [peaks[0]]
            for p in peaks[1:]:
                if p.index - filtered[-1].index >= self.min_distance:
                    filtered.append(p)
                elif p.value > filtered[-1].value:
                    # Replace with higher peak
                    filtered[-1] = p
            peaks = filtered
        
        return peaks


class WindowPeakFinder:
    """
    Window-based peak finder with local statistics.
    
    For each channel, computes local mean and standard deviation from
    surrounding channels (excluding an inner window around the point)
    to determine if the point is significantly above background.
    
    Based on peakingduck's WindowPeakFinder implementation.
    
    Unified API:
        finder = WindowPeakFinder(threshold_sigma=4.0, window_size=50)
        peaks = finder.find_peaks(spectrum)  # or finder.find(spectrum)
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        threshold_sigma: float = None,
        window_size: int = None,
        n_inner: int = 0,
        n_outer: int = 40,
        include_point: bool = False,
        enforce_maximum: bool = True,
        min_distance: int = 3
    ):
        """
        Parameters
        ----------
        threshold : float
            Number of sigma above local mean required (legacy)
        threshold_sigma : float, optional
            Alias for threshold (preferred name)
        window_size : int, optional
            Alias for n_outer (preferred name for API consistency)
        n_inner : int
            Inner exclusion window (channels to skip near test point)
        n_outer : int
            Outer window size for statistics
        include_point : bool
            Include test point in statistics
        enforce_maximum : bool
            Require peak to be local maximum
        min_distance : int
            Minimum distance between peaks
        """
        self.threshold = threshold_sigma if threshold_sigma is not None else threshold
        self.n_inner = n_inner
        self.n_outer = window_size if window_size is not None else n_outer
        self.include_point = include_point
        self.enforce_maximum = enforce_maximum
        self.min_distance = min_distance
    
    def _window_stats(
        self,
        data: np.ndarray,
        index: int
    ) -> Tuple[float, float]:
        """Get mean and std of window around index."""
        start = max(0, index - self.n_outer)
        end = min(len(data), index + self.n_outer + 1)
        
        # Get window values, excluding inner region
        values = []
        for i in range(start, end):
            if abs(i - index) > self.n_inner or (i == index and self.include_point):
                values.append(data[i])
        
        if len(values) < 3:
            return data[index], 1.0
        
        values = np.array(values)
        return float(np.mean(values)), float(np.std(values, ddof=1))
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum."""
        peaks = []
        n = len(spectrum)
        
        # Need at least 2*n_outer data points
        if n < 2 * self.n_outer:
            return peaks
        
        for i in range(self.n_outer, n - self.n_outer):
            value = spectrum[i]
            mean, std = self._window_stats(spectrum, i)
            
            local_threshold = mean + std * self.threshold
            
            if value > local_threshold:
                # Check neighboring points also above threshold
                if (spectrum[i+1] >= local_threshold and 
                    spectrum[i-1] >= local_threshold):
                    
                    # Optionally enforce local maximum
                    if self.enforce_maximum:
                        if spectrum[i+1] > value or spectrum[i-1] > value:
                            continue
                    
                    significance = (value - mean) / std if std > 0 else 0.0
                    peaks.append(PeakInfo(
                        index=i,
                        value=float(value),
                        significance=significance
                    ))
        
        # Apply minimum distance filtering
        if self.min_distance > 1 and peaks:
            peaks = merge_nearby_peaks(peaks, self.min_distance)
        
        return peaks
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """
        Alias for find() to provide consistent API.
        
        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum data (counts per channel)
        
        Returns
        -------
        List[PeakInfo]
            Detected peaks
        """
        return self.find(spectrum)


class ChunkedPeakFinder:
    """
    Chunked spectrum peak finder.
    
    Divides spectrum into chunks and applies threshold relative to
    each chunk's statistics, allowing for varying background levels.
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        n_chunks: int = 10,
        base_finder: Optional[SimplePeakFinder] = None
    ):
        """
        Parameters
        ----------
        threshold : float
            Threshold for peak detection
        n_chunks : int
            Number of spectrum subdivisions
        base_finder : SimplePeakFinder, optional
            Peak finder to use per chunk
        """
        self.threshold = threshold
        self.n_chunks = n_chunks
        self.base_finder = base_finder or SimplePeakFinder(threshold)
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum."""
        peaks = []
        
        # Split spectrum into chunks
        chunks = np.array_split(spectrum, self.n_chunks)
        
        offset = 0
        for chunk in chunks:
            chunk_peaks = self.base_finder.find(np.array(chunk))
            
            # Adjust indices for global spectrum position
            for peak in chunk_peaks:
                peaks.append(PeakInfo(
                    index=peak.index + offset,
                    value=peak.value,
                    significance=peak.significance
                ))
            
            offset += len(chunk)
        
        return peaks
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """
        Alias for find() to provide consistent API.
        
        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum data (counts per channel)
        
        Returns
        -------
        List[PeakInfo]
            Detected peaks
        """
        return self.find(spectrum)


class ScipyPeakFinder:
    """
    Scipy-based peak finder with smoothing.
    
    Uses scipy.signal.find_peaks with a Savitzky-Golay smoothed
    threshold curve.
    """
    
    def __init__(
        self,
        threshold_factor: float = 2.0,
        smooth_window: int = 101,
        prominence: Optional[float] = None,
        distance: int = 5
    ):
        """
        Parameters
        ----------
        threshold_factor : float
            Multiply smoothed spectrum by this for threshold
        smooth_window : int
            Savitzky-Golay window size (must be odd)
        prominence : float, optional
            Minimum peak prominence
        distance : int
            Minimum distance between peaks
        """
        self.threshold_factor = threshold_factor
        self.smooth_window = smooth_window
        self.prominence = prominence
        self.distance = distance
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum."""
        # Ensure odd window size
        window = self.smooth_window
        if window % 2 == 0:
            window += 1
        if window >= len(spectrum):
            window = len(spectrum) // 2 * 2 + 1
        
        # Smooth for threshold
        smoothed = savitzky_golay_smooth(spectrum, window, order=2)
        threshold = smoothed * self.threshold_factor
        
        # Find peaks
        kwargs = {
            'height': (threshold, None),
            'distance': self.distance
        }
        if self.prominence is not None:
            kwargs['prominence'] = self.prominence
        
        indices, properties = signal.find_peaks(spectrum, **kwargs)
        
        peaks = []
        for i, idx in enumerate(indices):
            # Calculate significance from height above threshold
            height_above = spectrum[idx] - threshold[idx] if idx < len(threshold) else 0.0
            noise = np.sqrt(np.maximum(threshold[idx], 1.0)) if idx < len(threshold) else 1.0
            significance = height_above / noise if noise > 0 else 0.0
            
            peaks.append(PeakInfo(
                index=int(idx),
                value=float(spectrum[idx]),
                significance=significance
            ))
        
        return peaks
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """
        Alias for find() to provide consistent API.
        
        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum data (counts per channel)
        
        Returns
        -------
        List[PeakInfo]
            Detected peaks
        """
        return self.find(spectrum)


def refine_peak_centroids(
    spectrum: np.ndarray,
    peaks: List[PeakInfo],
    width: int = 3
) -> List[PeakInfo]:
    """
    Refine peak positions using centroid calculation.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum data
    peaks : List[PeakInfo]
        Initial peak positions
    width : int
        Number of channels on each side for centroid
    
    Returns
    -------
    List[PeakInfo]
        Peaks with refined centroids
    """
    refined = []
    n = len(spectrum)
    
    for peak in peaks:
        start = max(0, peak.index - width)
        end = min(n, peak.index + width + 1)
        
        channels = np.arange(start, end)
        counts = spectrum[start:end]
        
        if np.sum(counts) > 0:
            centroid = np.sum(channels * counts) / np.sum(counts)
        else:
            centroid = float(peak.index)
        
        refined.append(PeakInfo(
            index=peak.index,
            value=peak.value,
            centroid=centroid,
            significance=peak.significance
        ))
    
    return refined


def merge_nearby_peaks(
    peaks: List[PeakInfo],
    min_distance: int = 3
) -> List[PeakInfo]:
    """
    Merge peaks that are closer than minimum distance.
    
    Keeps the peak with higher value.
    """
    if not peaks:
        return peaks
    
    # Sort by index
    sorted_peaks = sorted(peaks, key=lambda p: p.index)
    
    merged = [sorted_peaks[0]]
    
    for peak in sorted_peaks[1:]:
        if peak.index - merged[-1].index < min_distance:
            # Keep the higher peak
            if peak.value > merged[-1].value:
                merged[-1] = peak
        else:
            merged.append(peak)
    
    return merged


# =============================================================================
# Additional Peak Finders (from rafm_analysis and peakingduck patterns)
# =============================================================================

def weighted_moving_average(
    data: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Weighted moving average smoothing (from peakingduck).
    
    Weights follow triangular pattern: [1, 2, ..., ceil(n/2), ..., 2, 1]
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    window_size : int
        Size of smoothing window
        
    Returns
    -------
    np.ndarray
        Smoothed data
    """
    import math
    weights = list(range(1, math.ceil(window_size/2)+1)) + list(range(math.floor(window_size/2), 0, -1))
    weights = np.array([w/sum(weights) for w in weights])
    return np.convolve(data, weights, mode='same')


class SegmentedPeakFinder:
    """
    Segmented peak detection with region-specific thresholds.
    
    Divides spectrum into regions (e.g., low/mid/high energy) and applies
    different detection parameters to each, accounting for varying
    background and peak characteristics across the spectrum.
    
    Based on rafm_analysis/batch_compare_spectra.py implementation.
    
    Unified API:
        finder = SegmentedPeakFinder(splits=[2100, 3000])
        peaks = finder.find_peaks(spectrum)  # or finder.find(spectrum)
    """
    
    def __init__(
        self,
        splits: List[int] = None,
        sigmas: List[float] = None,
        region_params: List[dict] = None,
        fit_window: int = 4,
        gaussian_refine: bool = True
    ):
        """
        Parameters
        ----------
        splits : List[int]
            Channel boundaries for regions (default [2100, 3000] for 3 regions)
        sigmas : List[float]
            Gaussian filter sigma for each region (0 = no smoothing)
        region_params : List[dict]
            scipy.signal.find_peaks parameters per region
        fit_window : int
            Half-width for Gaussian fitting refinement
        gaussian_refine : bool
            Whether to refine peaks with Gaussian fits
        """
        self.splits = splits or [2100, 3000]
        self.sigmas = sigmas or [3.0, 2.0, 2.0]
        self.region_params = region_params or [
            {'height': 100, 'prominence': 60, 'distance': 2},
            {'height': 80, 'prominence': 40, 'distance': 2},
            {'height': 80, 'prominence': 30, 'distance': 2},
        ]
        self.fit_window = fit_window
        self.gaussian_refine = gaussian_refine
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks in spectrum."""
        from scipy.ndimage import gaussian_filter1d
        from scipy.optimize import curve_fit
        
        n = len(spectrum)
        peaks = []
        
        # Create region masks
        boundaries = [0] + self.splits + [n]
        
        for i in range(len(boundaries) - 1):
            start_ch = boundaries[i]
            end_ch = boundaries[i + 1]
            
            if start_ch >= end_ch:
                continue
            
            region_data = spectrum[start_ch:end_ch].copy()
            sigma = self.sigmas[i] if i < len(self.sigmas) else 2.0
            params = self.region_params[i] if i < len(self.region_params) else {}
            
            # Apply smoothing if sigma > 0
            if sigma > 0:
                smoothed = gaussian_filter1d(region_data, sigma=sigma, mode='nearest')
            else:
                smoothed = region_data
            
            # Find peaks
            pk_indices, properties = signal.find_peaks(smoothed, **params)
            
            for local_idx in pk_indices:
                global_idx = start_ch + local_idx
                
                # Optional Gaussian refinement
                centroid = float(global_idx)
                fwhm = None
                area = None
                
                if self.gaussian_refine and self.fit_window > 0:
                    lo = max(0, global_idx - self.fit_window)
                    hi = min(n, global_idx + self.fit_window + 1)
                    x_fit = np.arange(lo, hi)
                    y_fit = spectrum[lo:hi]
                    
                    if len(x_fit) >= 4 and np.max(y_fit) > 0:
                        try:
                            def gauss(x, a, mu, sig, c):
                                return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c
                            
                            p0 = [float(spectrum[global_idx]), float(global_idx), 
                                  1.5, float(np.median(y_fit))]
                            popt, _ = curve_fit(gauss, x_fit, y_fit, p0=p0, maxfev=2000)
                            a, mu, sig, c = popt
                            centroid = mu
                            fwhm = 2.355 * abs(sig)
                            area = a * abs(sig) * np.sqrt(2 * np.pi)
                        except (RuntimeError, ValueError):
                            pass
                
                peaks.append(PeakInfo(
                    index=global_idx,
                    value=float(spectrum[global_idx]),
                    centroid=centroid,
                    fwhm=fwhm,
                    area=area
                ))
        
        return sorted(peaks, key=lambda p: p.index)
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Alias for find() to provide consistent API."""
        return self.find(spectrum)


class DerivativePeakFinder:
    """
    Derivative-based peak finder.
    
    Detects peaks by looking for zero-crossings in the first derivative
    where the second derivative is negative (local maxima).
    
    Unified API:
        finder = DerivativePeakFinder(smooth_window=7)
        peaks = finder.find_peaks(spectrum)
    """
    
    def __init__(
        self,
        smooth_window: int = 7,
        threshold_sigma: float = 3.0,
        min_counts: float = 10.0
    ):
        """
        Parameters
        ----------
        smooth_window : int
            Window size for Savitzky-Golay smoothing before differentiation
        threshold_sigma : float
            Minimum significance above noise
        min_counts : float
            Minimum peak height
        """
        self.smooth_window = smooth_window
        self.threshold_sigma = threshold_sigma
        self.min_counts = min_counts
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks using derivative analysis."""
        n = len(spectrum)
        
        # Smooth spectrum
        window = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        if window >= n:
            window = n // 2 * 2 + 1
        
        smoothed = savitzky_golay_smooth(spectrum, window, order=3)
        
        # Compute first and second derivatives
        d1 = np.gradient(smoothed)
        d2 = np.gradient(d1)
        
        # Find zero crossings of first derivative where second derivative is negative
        peaks = []
        for i in range(1, n - 1):
            if d1[i-1] > 0 and d1[i+1] < 0 and d2[i] < 0:
                # Verify it's above threshold
                if spectrum[i] >= self.min_counts:
                    # Estimate significance
                    local_bg = np.median(spectrum[max(0, i-20):min(n, i+20)])
                    noise = np.sqrt(max(local_bg, 1.0))
                    significance = (spectrum[i] - local_bg) / noise
                    
                    if significance >= self.threshold_sigma:
                        peaks.append(PeakInfo(
                            index=i,
                            value=float(spectrum[i]),
                            significance=significance
                        ))
        
        return peaks
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Alias for find()."""
        return self.find(spectrum)


class SecondDifferencePeakFinder:
    """
    Second difference peak finder.
    
    Uses second difference (discrete Laplacian) to detect peaks.
    A peak is detected where the second difference is significantly negative.
    
    This method is robust for detecting sharp peaks against smooth background.
    """
    
    def __init__(
        self,
        threshold_sigma: float = 4.0,
        min_counts: float = 10.0,
        min_distance: int = 3
    ):
        self.threshold_sigma = threshold_sigma
        self.min_counts = min_counts
        self.min_distance = min_distance
    
    def find(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Find peaks using second difference."""
        n = len(spectrum)
        
        # Compute second difference: d2[i] = y[i-1] - 2*y[i] + y[i+1]
        d2 = np.zeros(n)
        d2[1:-1] = spectrum[:-2] - 2*spectrum[1:-1] + spectrum[2:]
        
        # For peaks, d2 should be negative (concave down)
        # Threshold based on noise
        noise = np.sqrt(np.maximum(spectrum, 1.0))
        threshold = -self.threshold_sigma * noise
        
        peaks = []
        for i in range(2, n - 2):
            if d2[i] < threshold[i] and spectrum[i] >= self.min_counts:
                # Check it's a local minimum in d2 (peak in spectrum)
                if d2[i] < d2[i-1] and d2[i] < d2[i+1]:
                    significance = -d2[i] / noise[i]
                    peaks.append(PeakInfo(
                        index=i,
                        value=float(spectrum[i]),
                        significance=significance
                    ))
        
        # Filter by minimum distance
        if self.min_distance > 1:
            peaks = merge_nearby_peaks(peaks, self.min_distance)
        
        return peaks
    
    def find_peaks(self, spectrum: np.ndarray) -> List[PeakInfo]:
        """Alias for find()."""
        return self.find(spectrum)


# =============================================================================
# Unified Peak Finder Interface
# =============================================================================

PEAK_FINDER_METHODS = {
    'simple': SimplePeakFinder,
    'window': WindowPeakFinder,
    'chunked': ChunkedPeakFinder,
    'scipy': ScipyPeakFinder,
    'segmented': SegmentedPeakFinder,
    'derivative': DerivativePeakFinder,
    'second_difference': SecondDifferencePeakFinder,
}


def get_peak_finder(method: str = 'simple', **kwargs):
    """
    Factory function to get a peak finder by method name.
    
    Parameters
    ----------
    method : str
        One of: 'simple', 'window', 'chunked', 'scipy', 'segmented', 
        'derivative', 'second_difference'
    **kwargs
        Arguments passed to the peak finder constructor
    
    Returns
    -------
    Peak finder instance with find() and find_peaks() methods
    
    Examples
    --------
    >>> finder = get_peak_finder('window', threshold_sigma=4.0, window_size=50)
    >>> peaks = finder.find_peaks(spectrum)
    """
    if method not in PEAK_FINDER_METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(PEAK_FINDER_METHODS.keys())}")
    
    return PEAK_FINDER_METHODS[method](**kwargs)


def find_peaks_multi_method(
    spectrum: np.ndarray,
    methods: List[str] = None,
    consensus_threshold: int = 2,
    **kwargs
) -> List[PeakInfo]:
    """
    Find peaks using multiple methods and return consensus peaks.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    methods : List[str]
        Methods to use (default: ['simple', 'window', 'scipy'])
    consensus_threshold : int
        Minimum number of methods that must agree on a peak location
    **kwargs
        Common arguments passed to all finders
    
    Returns
    -------
    List[PeakInfo]
        Peaks detected by at least consensus_threshold methods
    """
    if methods is None:
        methods = ['simple', 'window', 'scipy']
    
    all_peaks = []
    for method in methods:
        try:
            finder = get_peak_finder(method, **kwargs)
            peaks = finder.find_peaks(spectrum)
            for p in peaks:
                all_peaks.append((p.index, p, method))
        except Exception:
            continue
    
    if not all_peaks:
        return []
    
    # Group peaks by location (within 3 channels)
    all_peaks.sort(key=lambda x: x[0])
    
    groups = []
    current_group = [all_peaks[0]]
    
    for item in all_peaks[1:]:
        if item[0] - current_group[-1][0] <= 3:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
    groups.append(current_group)
    
    # Keep peaks with consensus
    consensus_peaks = []
    for group in groups:
        unique_methods = set(item[2] for item in group)
        if len(unique_methods) >= consensus_threshold:
            # Take the peak with highest value
            best = max(group, key=lambda x: x[1].value)
            consensus_peaks.append(best[1])
    
    return consensus_peaks