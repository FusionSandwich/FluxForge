"""
Segmented Peak Detection for Gamma Spectra

Advanced peak detection with region-specific parameters for different
energy ranges. Based on the observation that low-energy, mid-energy,
and high-energy regions of gamma spectra require different detection
thresholds due to varying count rates and peak characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .peakfit import GaussianPeak, PeakFitResult, fit_single_peak, five_point_smooth


# ============================================================================
# Detection Parameters
# ============================================================================

@dataclass
class RegionParams:
    """Peak detection parameters for a spectral region."""
    
    height: float = 50.0      # Minimum peak height
    prominence: float = 30.0  # Minimum peak prominence
    distance: int = 2         # Minimum distance between peaks (channels)
    width: Optional[int] = None  # Minimum peak width
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to scipy.signal.find_peaks kwargs."""
        d = {
            'height': self.height,
            'prominence': self.prominence,
            'distance': self.distance,
        }
        if self.width is not None:
            d['width'] = self.width
        return d


@dataclass
class SegmentedDetectionConfig:
    """
    Configuration for segmented peak detection.
    
    The spectrum is divided into regions by channel number, each with
    its own detection parameters and smoothing sigma.
    """
    
    # Channel boundaries (splits between regions)
    split_channel1: int = 2100  # Low/mid boundary
    split_channel2: int = 3000  # Mid/high boundary
    
    # Smoothing sigma for each region (0 = no smoothing)
    sigma_low: float = 3.0
    sigma_mid: float = 2.0
    sigma_high: float = 2.0
    smooth_method: str = "gaussian"
    
    # Detection parameters for each region
    params_low: RegionParams = field(default_factory=lambda: RegionParams(
        height=100, prominence=60, distance=2
    ))
    params_mid: RegionParams = field(default_factory=lambda: RegionParams(
        height=80, prominence=40, distance=2
    ))
    params_high: RegionParams = field(default_factory=lambda: RegionParams(
        height=80, prominence=30, distance=2
    ))
    
    # Peak fitting
    fit_window: int = 4  # Half-window for Gaussian centroid refinement
    
    # Post-processing
    merge_tolerance_keV: float = 0.6  # Merge peaks closer than this
    min_energy_keV: float = 25.0  # Discard peaks below this energy
    
    @classmethod
    def default(cls) -> 'SegmentedDetectionConfig':
        """Get default configuration."""
        return cls()
    
    @classmethod
    def sensitive(cls) -> 'SegmentedDetectionConfig':
        """More sensitive configuration for weak peaks."""
        return cls(
            params_low=RegionParams(height=50, prominence=30, distance=2),
            params_mid=RegionParams(height=40, prominence=20, distance=2),
            params_high=RegionParams(height=30, prominence=15, distance=2),
        )
    
    @classmethod
    def conservative(cls) -> 'SegmentedDetectionConfig':
        """Conservative configuration to reduce false positives."""
        return cls(
            params_low=RegionParams(height=200, prominence=100, distance=3),
            params_mid=RegionParams(height=150, prominence=80, distance=3),
            params_high=RegionParams(height=100, prominence=50, distance=3),
        )


# ============================================================================
# Peak Detection
# ============================================================================

@dataclass
class DetectedPeak:
    """A peak detected in the spectrum."""
    
    channel: int
    energy_keV: float
    amplitude: float        # Peak height (corrected counts)
    raw_counts: float       # Peak height (raw counts)
    sigma_keV: float = 0.0  # Gaussian sigma from fit
    area: float = 0.0       # Integrated area
    region: str = ""        # 'low', 'mid', or 'high'
    is_fitted: bool = False # Whether centroid was refined
    
    # For report-seeded peaks
    is_report: bool = False
    report_isotope: str = ""
    report_file: str = ""


def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    """Gaussian with constant background."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def _refine_centroid(
    channels: np.ndarray,
    energies: np.ndarray,
    counts: np.ndarray,
    peak_idx: int,
    fit_window: int,
) -> Tuple[float, float, float, float, bool]:
    """
    Refine peak centroid with Gaussian fit.
    
    Returns (energy, amplitude, sigma, area, success)
    """
    n = len(counts)
    lo = max(0, peak_idx - fit_window)
    hi = min(n, peak_idx + fit_window + 1)
    
    x_keV = energies[lo:hi]
    y_loc = counts[lo:hi]
    
    if len(x_keV) < 4:
        return energies[peak_idx], counts[peak_idx], 0.0, 0.0, False
    
    try:
        peak_y = counts[peak_idx]
        initial_mu = x_keV[np.argmax(y_loc)]
        p0 = [peak_y, initial_mu, 1.0, np.median(y_loc)]
        
        popt, _ = curve_fit(
            _gaussian, x_keV, y_loc, p0=p0,
            maxfev=4000,
            bounds=(
                [0, x_keV.min(), 0.1, 0],
                [np.inf, x_keV.max(), 10.0, np.inf]
            )
        )
        
        a, mu, sigma, c = popt
        area = a * sigma * np.sqrt(2 * np.pi)
        return mu, a, sigma, area, True
        
    except (RuntimeError, ValueError):
        return energies[peak_idx], counts[peak_idx], 0.0, 0.0, False


def detect_peaks_segmented(
    channels: np.ndarray,
    energies: np.ndarray,
    counts: np.ndarray,
    raw_counts: Optional[np.ndarray] = None,
    config: Optional[SegmentedDetectionConfig] = None,
) -> List[DetectedPeak]:
    """
    Detect peaks with region-specific parameters.
    
    The spectrum is divided into three regions (low, mid, high energy)
    based on channel number, with different detection thresholds for each.
    
    Parameters
    ----------
    channels : ndarray
        Channel numbers
    energies : ndarray
        Calibrated energies (keV)
    counts : ndarray
        Count values (typically efficiency-corrected)
    raw_counts : ndarray, optional
        Raw counts (before efficiency correction)
    config : SegmentedDetectionConfig, optional
        Detection configuration
    
    Returns
    -------
    list of DetectedPeak
        Detected peaks sorted by energy
    
    Examples
    --------
    >>> config = SegmentedDetectionConfig.sensitive()
    >>> peaks = detect_peaks_segmented(channels, energies, corrected_counts, config=config)
    >>> for p in peaks[:5]:
    ...     print(f"{p.energy_keV:.1f} keV: {p.amplitude:.0f} counts")
    """
    if config is None:
        config = SegmentedDetectionConfig.default()
    
    if raw_counts is None:
        raw_counts = counts
    
    # Define regions
    regions = [
        ('low', channels <= config.split_channel1,
         config.sigma_low, config.params_low),
        ('mid', (channels > config.split_channel1) & (channels <= config.split_channel2),
         config.sigma_mid, config.params_mid),
        ('high', channels > config.split_channel2,
         config.sigma_high, config.params_high),
    ]
    
    peaks = []
    
    for region_name, mask, sigma, params in regions:
        if not mask.any():
            continue
        
        # Get indices for this region
        region_indices = np.where(mask)[0]
        region_counts = counts[mask]
        
        # Apply smoothing for peak detection
        if config.smooth_method == "gaussian":
            smoothed = gaussian_filter1d(region_counts, sigma=sigma) if sigma > 0 else region_counts
        elif config.smooth_method == "five_point":
            smoothed = five_point_smooth(region_counts) if len(region_counts) >= 5 else region_counts
        elif config.smooth_method == "none":
            smoothed = region_counts
        else:
            raise ValueError(f"Unknown smoothing method: {config.smooth_method}")
        
        # Find peaks
        peak_local_idx, _ = find_peaks(smoothed, **params.to_dict())
        
        # Convert to global indices and refine
        for local_idx in peak_local_idx:
            global_idx = region_indices[local_idx]
            
            energy, amplitude, sigma_keV, area, fitted = _refine_centroid(
                channels, energies, counts, global_idx, config.fit_window
            )
            
            peaks.append(DetectedPeak(
                channel=int(channels[global_idx]),
                energy_keV=energy,
                amplitude=amplitude,
                raw_counts=raw_counts[global_idx],
                sigma_keV=sigma_keV,
                area=area,
                region=region_name,
                is_fitted=fitted,
            ))
    
    # Sort by energy
    peaks.sort(key=lambda p: p.energy_keV)
    
    return peaks


def merge_duplicate_peaks(
    peaks: List[DetectedPeak],
    tolerance_keV: float = 0.6,
    prefer_report: bool = True,
) -> List[DetectedPeak]:
    """
    Merge nearby peaks, preferring report-identified peaks.
    
    Parameters
    ----------
    peaks : list of DetectedPeak
        Peaks to merge
    tolerance_keV : float
        Maximum energy difference for merging
    prefer_report : bool
        If True, keep report peaks when merging
    
    Returns
    -------
    list of DetectedPeak
        Merged peaks
    """
    if not peaks:
        return []
    
    # Sort by energy
    sorted_peaks = sorted(peaks, key=lambda p: p.energy_keV)
    
    result = []
    cluster = [sorted_peaks[0]]
    
    def flush_cluster():
        if not cluster:
            return
        
        # Prefer report peak
        if prefer_report:
            report_peaks = [p for p in cluster if p.is_report]
            if report_peaks:
                result.append(report_peaks[0])
                return
        
        # Keep highest amplitude peak
        best = max(cluster, key=lambda p: p.amplitude)
        result.append(best)
    
    for peak in sorted_peaks[1:]:
        if peak.energy_keV - cluster[-1].energy_keV <= tolerance_keV:
            cluster.append(peak)
        else:
            flush_cluster()
            cluster = [peak]
    
    flush_cluster()
    
    return result


def filter_peaks_by_energy(
    peaks: List[DetectedPeak],
    min_energy_keV: float = 25.0,
    max_energy_keV: Optional[float] = None,
) -> List[DetectedPeak]:
    """
    Filter peaks by energy range.
    
    Parameters
    ----------
    peaks : list of DetectedPeak
        Peaks to filter
    min_energy_keV : float
        Minimum energy threshold
    max_energy_keV : float, optional
        Maximum energy threshold
    
    Returns
    -------
    list of DetectedPeak
        Filtered peaks
    """
    result = []
    for p in peaks:
        if not np.isfinite(p.energy_keV):
            continue
        if p.energy_keV < min_energy_keV:
            continue
        if max_energy_keV is not None and p.energy_keV > max_energy_keV:
            continue
        result.append(p)
    return result


# ============================================================================
# Integration with Report Files
# ============================================================================

def create_report_peaks(
    report_entries: List['ReportPeak'],  # from io.genie
    channels: np.ndarray,
    energies: np.ndarray,
    counts: np.ndarray,
) -> List[DetectedPeak]:
    """
    Convert report peak entries to DetectedPeak objects.
    
    Maps report energies to nearest spectrum channels for consistent handling.
    
    Parameters
    ----------
    report_entries : list of ReportPeak
        Peaks from Genie/LabSOCS report parsing
    channels : ndarray
        Spectrum channels
    energies : ndarray
        Spectrum energies
    counts : ndarray
        Spectrum counts
    
    Returns
    -------
    list of DetectedPeak
        Report peaks as DetectedPeak objects
    """
    peaks = []
    
    for entry in report_entries:
        # Find nearest energy bin
        idx = np.abs(energies - entry.energy).argmin()
        
        peaks.append(DetectedPeak(
            channel=int(channels[idx]),
            energy_keV=entry.energy,
            amplitude=counts[idx],
            raw_counts=counts[idx],
            sigma_keV=0.0,
            area=0.0,
            region='',
            is_fitted=False,
            is_report=True,
            report_isotope=entry.isotope,
            report_file=entry.report_file,
        ))
    
    return peaks


def combine_with_report_peaks(
    auto_peaks: List[DetectedPeak],
    report_peaks: List[DetectedPeak],
    merge_tolerance_keV: float = 0.6,
) -> List[DetectedPeak]:
    """
    Combine automatically detected peaks with report-identified peaks.
    
    Report peaks take precedence when there are overlaps.
    
    Parameters
    ----------
    auto_peaks : list of DetectedPeak
        Automatically detected peaks
    report_peaks : list of DetectedPeak
        Peaks from report files
    merge_tolerance_keV : float
        Tolerance for merging
    
    Returns
    -------
    list of DetectedPeak
        Combined and merged peaks
    """
    all_peaks = list(report_peaks) + list(auto_peaks)
    return merge_duplicate_peaks(all_peaks, merge_tolerance_keV, prefer_report=True)
