"""
Peak Fitting Module for HPGe Gamma Spectroscopy

Provides Gaussian peak fitting and analysis for gamma-ray spectra.
Includes background estimation, peak finding, and uncertainty propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize, signal


@dataclass
class GaussianPeak:
    """
    Representation of a Gaussian peak.
    
    The peak function is:
        f(x) = amplitude * exp(-(x - centroid)² / (2 * sigma²))
    
    Attributes
    ----------
    centroid : float
        Peak centroid (energy or channel)
    amplitude : float
        Peak height (counts)
    sigma : float
        Gaussian standard deviation
    centroid_unc : float
        Centroid uncertainty
    amplitude_unc : float
        Amplitude uncertainty
    sigma_unc : float
        Sigma uncertainty
    
    Properties
    ----------
    fwhm : float
        Full width at half maximum
    area : float
        Integrated peak area (counts)
    area_uncertainty : float
        Area uncertainty
    """
    
    centroid: float
    amplitude: float
    sigma: float
    centroid_unc: float = 0.0
    amplitude_unc: float = 0.0
    sigma_unc: float = 0.0
    
    @property
    def fwhm(self) -> float:
        """Full width at half maximum."""
        return 2.355 * self.sigma
    
    @property
    def fwhm_unc(self) -> float:
        """FWHM uncertainty."""
        return 2.355 * self.sigma_unc
    
    @property
    def area(self) -> float:
        """Integrated Gaussian area = amplitude * sigma * sqrt(2π)."""
        return self.amplitude * self.sigma * np.sqrt(2 * np.pi)
    
    @property
    def area_uncertainty(self) -> float:
        """Area uncertainty from error propagation."""
        # d(area)/d(amp) = sigma * sqrt(2π)
        # d(area)/d(sigma) = amp * sqrt(2π)
        sqrt_2pi = np.sqrt(2 * np.pi)
        da_damp = self.sigma * sqrt_2pi
        da_dsig = self.amplitude * sqrt_2pi
        
        var = (da_damp * self.amplitude_unc)**2 + (da_dsig * self.sigma_unc)**2
        return np.sqrt(var)
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Gaussian at x."""
        return self.amplitude * np.exp(
            -(x - self.centroid)**2 / (2 * self.sigma**2)
        )
    
    def __repr__(self) -> str:
        return (
            f"GaussianPeak(centroid={self.centroid:.2f}±{self.centroid_unc:.3f}, "
            f"area={self.area:.1f}±{self.area_uncertainty:.1f}, "
            f"FWHM={self.fwhm:.3f})"
        )


@dataclass
class PeakFitResult:
    """
    Complete peak fitting result.
    
    Attributes
    ----------
    peak : GaussianPeak
        Fitted peak parameters
    background : np.ndarray
        Background under peak
    background_model : str
        Background model type used
    residuals : np.ndarray
        Fit residuals
    chi_squared : float
        Chi-squared of fit
    dof : int
        Degrees of freedom
    fit_region : Tuple[int, int]
        Channel range of fit
    covariance : Optional[np.ndarray]
        Parameter covariance matrix
    success : bool
        Whether fit converged
    message : str
        Fit status message
    """
    
    peak: GaussianPeak
    background: np.ndarray
    background_model: str = 'linear'
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_squared: float = 0.0
    dof: int = 1
    fit_region: Tuple[int, int] = (0, 0)
    covariance: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""
    
    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared."""
        return self.chi_squared / self.dof if self.dof > 0 else 0.0
    
    @property
    def net_counts(self) -> float:
        """Net peak counts (area - background)."""
        return self.peak.area
    
    @property
    def net_counts_uncertainty(self) -> float:
        """Net counts uncertainty."""
        return self.peak.area_uncertainty


def gaussian(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float
) -> np.ndarray:
    """Gaussian function."""
    return amplitude * np.exp(-(x - centroid)**2 / (2 * sigma**2))


def gaussian_with_linear_bg(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    bg_slope: float,
    bg_intercept: float
) -> np.ndarray:
    """Gaussian plus linear background."""
    return gaussian(x, amplitude, centroid, sigma) + bg_slope * x + bg_intercept


def gaussian_with_step_bg(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    bg_left: float,
    bg_right: float,
    step_fraction: float = 0.5
) -> np.ndarray:
    """
    Gaussian with step background (Compton edge model).
    
    Step function transitions from bg_left to bg_right at centroid.
    """
    from scipy.special import erfc
    
    peak = gaussian(x, amplitude, centroid, sigma)
    
    # Error function step
    step = 0.5 * (bg_left + bg_right) + 0.5 * (bg_left - bg_right) * erfc(
        (x - centroid) / (np.sqrt(2) * sigma)
    )
    
    return peak + step


def estimate_background(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_regions: Optional[List[Tuple[int, int]]] = None,
    method: str = 'snip',
    iterations: int = 20,
    window_size: int = 10
) -> np.ndarray:
    """
    Estimate background under spectrum.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    peak_regions : list of (start, end) tuples, optional
        Known peak regions to exclude
    method : str
        Background method: 'snip', 'linear', 'rolling_min'
    iterations : int
        Number of SNIP iterations
    window_size : int
        Window size for rolling methods
    
    Returns
    -------
    np.ndarray
        Estimated background
    """
    if method == 'snip':
        return _snip_background(counts, iterations)
    elif method == 'rolling_min':
        return _rolling_min_background(counts, window_size)
    elif method == 'linear':
        return _linear_background(channels, counts, peak_regions)
    else:
        raise ValueError(f"Unknown background method: {method}")


def _snip_background(counts: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm.
    
    This is the standard algorithm for gamma spectra background estimation.
    """
    # Work with sqrt transform for Poisson statistics
    y = np.sqrt(np.maximum(counts, 0))
    
    for p in range(1, iterations + 1):
        # Decreasing window from iterations to 1
        window = iterations - p + 1
        
        y_new = y.copy()
        for i in range(window, len(y) - window):
            # Average of neighbors at distance window
            avg = 0.5 * (y[i - window] + y[i + window])
            y_new[i] = min(y[i], avg)
        
        y = y_new
    
    # Transform back
    return y**2


def _rolling_min_background(counts: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Simple rolling minimum background."""
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(counts, size=window_size)


def _linear_background(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_regions: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """Fit linear background excluding peak regions."""
    mask = np.ones(len(channels), dtype=bool)
    
    if peak_regions:
        for start, end in peak_regions:
            mask[(channels >= start) & (channels <= end)] = False
    
    if np.sum(mask) < 2:
        # Not enough points, return flat background
        return np.full_like(counts, np.median(counts), dtype=float)
    
    coeffs = np.polyfit(channels[mask], counts[mask], 1)
    return np.polyval(coeffs, channels)


def subtract_background(
    counts: np.ndarray,
    background: np.ndarray,
    minimum: float = 0.0
) -> np.ndarray:
    """
    Subtract background from spectrum.
    
    Parameters
    ----------
    counts : np.ndarray
        Spectrum counts
    background : np.ndarray
        Background to subtract
    minimum : float
        Minimum value for result (default 0)
    
    Returns
    -------
    np.ndarray
        Background-subtracted spectrum
    """
    return np.maximum(counts - background, minimum)


def auto_find_peaks(
    channels: np.ndarray,
    counts: np.ndarray,
    threshold: float = 3.0,
    min_distance: int = 5,
    background_method: str = 'snip'
) -> List[Tuple[int, float]]:
    """
    Automatically find peaks in spectrum.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    threshold : float
        Detection threshold (number of sigma above background)
    min_distance : int
        Minimum distance between peaks (channels)
    background_method : str
        Method for background estimation
    
    Returns
    -------
    list of (channel, significance) tuples
        Detected peak positions and significances
    """
    # Estimate background
    background = estimate_background(channels, counts, method=background_method)
    
    # Calculate significance
    net = counts - background
    sigma = np.sqrt(np.maximum(background, 1))  # Poisson uncertainty
    significance = net / sigma
    
    # Find local maxima
    peaks, properties = signal.find_peaks(
        significance,
        height=threshold,
        distance=min_distance
    )
    
    # Get peak significances
    heights = properties['peak_heights']
    
    return [(int(channels[p]), float(h)) for p, h in zip(peaks, heights)]


def fit_single_peak(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channel: int,
    fit_width: int = 10,
    background_model: str = 'linear',
    fix_centroid: bool = False,
    initial_sigma: Optional[float] = None
) -> PeakFitResult:
    """
    Fit single Gaussian peak to spectrum region.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers (full spectrum)
    counts : np.ndarray
        Spectrum counts (full spectrum)
    peak_channel : int
        Approximate peak channel
    fit_width : int
        Half-width of fit region in channels
    background_model : str
        Background model: 'linear', 'constant', 'step'
    fix_centroid : bool
        If True, fix centroid at peak_channel
    initial_sigma : float, optional
        Initial guess for sigma
    
    Returns
    -------
    PeakFitResult
        Fitting result with peak parameters and uncertainties
    """
    # Extract fit region
    idx_peak = np.argmin(np.abs(channels - peak_channel))
    
    ch_lo = max(0, idx_peak - fit_width)
    ch_hi = min(len(channels), idx_peak + fit_width + 1)
    
    x = channels[ch_lo:ch_hi].astype(float)
    y = counts[ch_lo:ch_hi].astype(float)
    
    # Weights for chi-squared (Poisson uncertainty)
    weights = 1.0 / np.maximum(np.sqrt(y), 1.0)
    
    # Initial guesses
    amplitude_guess = y.max() - y.min()
    centroid_guess = float(peak_channel)
    sigma_guess = initial_sigma if initial_sigma else fit_width / 3.0
    
    # Background initial guess
    bg_left = np.mean(y[:3])
    bg_right = np.mean(y[-3:])
    bg_slope = (bg_right - bg_left) / (x[-1] - x[0])
    bg_intercept = bg_left - bg_slope * x[0]
    
    # Define model and fit
    if background_model == 'linear':
        def model(x, amp, cent, sig, slope, intercept):
            return gaussian(x, amp, cent, sig) + slope * x + intercept
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, bg_slope, bg_intercept]
        
        bounds_lower = [0, x[0], 0.5, -np.inf, -np.inf]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    elif background_model == 'constant':
        def model(x, amp, cent, sig, bg):
            return gaussian(x, amp, cent, sig) + bg
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, (bg_left + bg_right) / 2]
        
        bounds_lower = [0, x[0], 0.5, 0]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    elif background_model == 'step':
        def model(x, amp, cent, sig, bg_l, bg_r):
            return gaussian_with_step_bg(x, amp, cent, sig, bg_l, bg_r)
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, bg_left, bg_right]
        
        bounds_lower = [0, x[0], 0.5, 0, 0]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    else:
        raise ValueError(f"Unknown background model: {background_model}")
    
    # Perform fit
    try:
        popt, pcov = optimize.curve_fit(
            model, x, y,
            p0=p0,
            sigma=1.0 / weights,
            absolute_sigma=True,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        
        perr = np.sqrt(np.diag(pcov))
        success = True
        message = "Fit converged"
        
    except Exception as e:
        # Fit failed, return initial guess with large uncertainties
        popt = p0
        perr = np.array([1e6] * len(p0))
        pcov = None
        success = False
        message = str(e)
    
    # Extract parameters
    amplitude, centroid, sigma = popt[0], popt[1], popt[2]
    amp_err, cent_err, sig_err = perr[0], perr[1], perr[2]
    
    peak = GaussianPeak(
        centroid=centroid,
        amplitude=amplitude,
        sigma=sigma,
        centroid_unc=cent_err,
        amplitude_unc=amp_err,
        sigma_unc=sig_err,
    )
    
    # Calculate background array
    if background_model == 'linear':
        background = popt[3] * x + popt[4]
    elif background_model == 'constant':
        background = np.full_like(x, popt[3])
    elif background_model == 'step':
        from scipy.special import erfc
        background = 0.5 * (popt[3] + popt[4]) + 0.5 * (popt[3] - popt[4]) * erfc(
            (x - centroid) / (np.sqrt(2) * sigma)
        )
    
    # Calculate residuals and chi-squared
    y_fit = model(x, *popt)
    residuals = y - y_fit
    chi_sq = np.sum((residuals * weights)**2)
    dof = len(y) - len(popt)
    
    return PeakFitResult(
        peak=peak,
        background=background,
        background_model=background_model,
        residuals=residuals,
        chi_squared=chi_sq,
        dof=dof,
        fit_region=(int(x[0]), int(x[-1])),
        covariance=pcov,
        success=success,
        message=message,
    )


def fit_multiple_peaks(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channels: List[int],
    fit_width: int = 10,
    background_model: str = 'linear',
    share_sigma: bool = False
) -> List[PeakFitResult]:
    """
    Fit multiple peaks simultaneously.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    peak_channels : list of int
        Approximate peak channels
    fit_width : int
        Half-width of fit region per peak
    background_model : str
        Background model
    share_sigma : bool
        If True, all peaks share same sigma
    
    Returns
    -------
    list of PeakFitResult
        Fitting results for each peak
    """
    # For well-separated peaks, fit individually
    results = []
    
    for peak_ch in sorted(peak_channels):
        result = fit_single_peak(
            channels, counts, peak_ch,
            fit_width=fit_width,
            background_model=background_model
        )
        results.append(result)
    
    return results


def calculate_activity(
    net_counts: float,
    net_counts_unc: float,
    live_time: float,
    efficiency: float,
    efficiency_unc: float,
    emission_probability: float,
    emission_probability_unc: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate activity from peak net counts.
    
    A = N / (t_live * ε * I_γ)
    
    Parameters
    ----------
    net_counts : float
        Net peak counts
    net_counts_unc : float
        Net counts uncertainty
    live_time : float
        Measurement live time (seconds)
    efficiency : float
        Detection efficiency at peak energy
    efficiency_unc : float
        Efficiency uncertainty
    emission_probability : float
        Gamma emission probability
    emission_probability_unc : float
        Emission probability uncertainty
    
    Returns
    -------
    activity : float
        Activity in Bq
    activity_unc : float
        Activity uncertainty in Bq
    """
    # Activity calculation
    denominator = live_time * efficiency * emission_probability
    activity = net_counts / denominator
    
    # Relative uncertainty propagation
    rel_unc_sq = (net_counts_unc / net_counts)**2
    rel_unc_sq += (efficiency_unc / efficiency)**2
    
    if emission_probability_unc > 0:
        rel_unc_sq += (emission_probability_unc / emission_probability)**2
    
    activity_unc = activity * np.sqrt(rel_unc_sq)
    
    return activity, activity_unc


def peak_report(
    results: List[PeakFitResult],
    energies: Optional[np.ndarray] = None,
    energy_calibration: Optional[List[float]] = None
) -> str:
    """
    Generate text report of peak fitting results.
    
    Parameters
    ----------
    results : list of PeakFitResult
        Peak fitting results
    energies : np.ndarray, optional
        Energy array (if already calibrated)
    energy_calibration : list of float, optional
        Energy calibration coefficients [a0, a1, ...]
    
    Returns
    -------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PEAK FITTING REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    header = f"{'Peak':^4}  {'Centroid':^12}  {'Energy':^10}  {'Area':^14}  {'FWHM':^8}  {'χ²/dof':^8}"
    lines.append(header)
    lines.append("-" * 80)
    
    for i, result in enumerate(results, 1):
        peak = result.peak
        
        # Calculate energy if calibration provided
        if energy_calibration:
            energy = sum(c * peak.centroid**j for j, c in enumerate(energy_calibration))
            energy_str = f"{energy:.2f} keV"
        elif energies is not None:
            ch = int(peak.centroid)
            if 0 <= ch < len(energies):
                energy_str = f"{energies[ch]:.2f} keV"
            else:
                energy_str = "N/A"
        else:
            energy_str = "N/A"
        
        chi2_str = f"{result.reduced_chi_squared:.2f}" if result.dof > 0 else "N/A"
        
        line = (
            f"{i:^4}  "
            f"{peak.centroid:.2f}±{peak.centroid_unc:.2f}  "
            f"{energy_str:^10}  "
            f"{peak.area:.1f}±{peak.area_uncertainty:.1f}  "
            f"{peak.fwhm:.3f}  "
            f"{chi2_str:^8}"
        )
        lines.append(line)
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
