"""
Auto-Calibration Module - R1.5 Becquerel Parity

Automatic energy calibration by matching peaks to known isotope lines.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray


# Known calibration isotopes with prominent gamma lines
CALIBRATION_SOURCES: Dict[str, List[Tuple[float, float]]] = {
    # Isotope: [(energy_keV, intensity), ...]
    'Co-60': [(1173.23, 0.9985), (1332.49, 0.9998)],
    'Cs-137': [(661.66, 0.851)],
    'Ba-133': [(81.0, 0.329), (276.4, 0.0716), (302.9, 0.1834), 
               (356.0, 0.6205), (383.8, 0.0894)],
    'Eu-152': [(121.78, 0.286), (244.70, 0.076), (344.28, 0.265),
               (778.90, 0.129), (964.08, 0.146), (1085.87, 0.102),
               (1112.07, 0.136), (1408.01, 0.210)],
    'Na-22': [(511.0, 1.798), (1274.54, 0.9994)],
    'Am-241': [(59.54, 0.359)],
    'K-40': [(1460.83, 0.1066)],
    'Mn-54': [(834.85, 0.9998)],
    'Zn-65': [(1115.55, 0.502)],
    'Co-57': [(122.06, 0.856), (136.47, 0.1068)],
}


@dataclass
class CalibrationMatch:
    """A matched peak-to-line pair."""
    peak_channel: float
    peak_channel_unc: float
    expected_energy: float
    isotope: str
    intensity: float
    residual: float = 0.0
    score: float = 0.0


@dataclass
class AutoCalibrationResult:
    """Result of automatic calibration."""
    coefficients: NDArray
    matches: List[CalibrationMatch]
    r_squared: float
    chi_squared: float
    rms_residual: float
    isotopes_matched: List[str]
    success: bool
    message: str = ""
    
    @property
    def n_matches(self) -> int:
        return len(self.matches)


def find_peak_candidates(
    counts: NDArray,
    threshold_sigma: float = 5.0,
    min_counts: float = 100.0
) -> List[Tuple[int, float]]:
    """
    Find candidate peak channels in spectrum.
    
    Parameters
    ----------
    counts : NDArray
        Spectrum counts
    threshold_sigma : float
        Sigma threshold above background
    min_counts : float
        Minimum counts to consider
    
    Returns
    -------
    list of (channel, peak_counts)
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    from scipy.signal import find_peaks
    
    # Estimate background as running minimum
    background = minimum_filter1d(counts.astype(float), size=50)
    background = maximum_filter1d(background, size=10)
    
    # Net counts above background
    net = counts - background
    sigma = np.sqrt(counts + 1)
    
    # Significance
    significance = net / sigma
    
    # Find peaks using scipy
    peaks, properties = find_peaks(
        significance,
        height=threshold_sigma,
        distance=3,
        prominence=2
    )
    
    # Filter by counts
    valid_peaks = [(p, float(counts[p])) for p in peaks if counts[p] >= min_counts]
    
    # Sort by counts (strongest first)
    valid_peaks.sort(key=lambda x: -x[1])
    
    return valid_peaks


def match_peaks_to_lines(
    peak_channels: List[int],
    known_lines: List[Tuple[float, str, float]],  # (energy, isotope, intensity)
    initial_cal: Tuple[float, float],  # (offset, gain)
    tolerance_keV: float = 5.0
) -> List[CalibrationMatch]:
    """
    Match observed peaks to known calibration lines.
    
    Parameters
    ----------
    peak_channels : list
        Observed peak channel positions
    known_lines : list
        (energy_keV, isotope, intensity) tuples
    initial_cal : tuple
        (offset, gain) for initial energy = offset + gain * channel
    tolerance_keV : float
        Maximum allowed mismatch
    
    Returns
    -------
    list of CalibrationMatch
    """
    offset, gain = initial_cal
    matches = []
    
    for ch in peak_channels:
        estimated_energy = offset + gain * ch
        
        # Find closest known line
        best_match = None
        best_diff = float('inf')
        
        for energy, isotope, intensity in known_lines:
            diff = abs(estimated_energy - energy)
            if diff < best_diff and diff < tolerance_keV:
                best_diff = diff
                best_match = (energy, isotope, intensity)
        
        if best_match:
            matches.append(CalibrationMatch(
                peak_channel=float(ch),
                peak_channel_unc=0.5,  # Default half-channel uncertainty
                expected_energy=best_match[0],
                isotope=best_match[1],
                intensity=best_match[2],
                residual=best_diff,
                score=best_match[2] * (1 - best_diff / tolerance_keV)
            ))
    
    return matches


def auto_calibrate(
    counts: NDArray,
    isotopes: Optional[List[str]] = None,
    initial_gain: float = 0.5,
    initial_offset: float = 0.0,
    degree: int = 1,
    min_peaks: int = 3,
    max_iterations: int = 5,
    tolerance_keV: float = 5.0
) -> AutoCalibrationResult:
    """
    Automatically calibrate spectrum using known isotope lines.
    
    Parameters
    ----------
    counts : NDArray
        Spectrum counts array
    isotopes : list, optional
        List of isotopes present (e.g., ['Co-60', 'Cs-137'])
        If None, tries common calibration sources
    initial_gain : float
        Initial keV/channel estimate
    initial_offset : float
        Initial offset keV
    degree : int
        Polynomial degree (1=linear, 2=quadratic)
    min_peaks : int
        Minimum matched peaks required
    max_iterations : int
        Maximum refinement iterations
    tolerance_keV : float
        Peak-to-line matching tolerance
    
    Returns
    -------
    AutoCalibrationResult
    """
    # Build list of expected lines
    if isotopes is None:
        isotopes = ['Co-60', 'Cs-137', 'Na-22', 'K-40']
    
    known_lines = []
    for iso in isotopes:
        if iso in CALIBRATION_SOURCES:
            for energy, intensity in CALIBRATION_SOURCES[iso]:
                known_lines.append((energy, iso, intensity))
    
    if not known_lines:
        return AutoCalibrationResult(
            coefficients=np.array([initial_offset, initial_gain]),
            matches=[],
            r_squared=0.0,
            chi_squared=float('inf'),
            rms_residual=float('inf'),
            isotopes_matched=[],
            success=False,
            message="No known calibration lines for specified isotopes"
        )
    
    # Sort by energy
    known_lines.sort(key=lambda x: x[0])
    
    # Find peaks in spectrum
    peaks = find_peak_candidates(counts)
    peak_channels = [p[0] for p in peaks[:20]]  # Take top 20 peaks
    
    if len(peak_channels) < min_peaks:
        return AutoCalibrationResult(
            coefficients=np.array([initial_offset, initial_gain]),
            matches=[],
            r_squared=0.0,
            chi_squared=float('inf'),
            rms_residual=float('inf'),
            isotopes_matched=[],
            success=False,
            message=f"Only found {len(peak_channels)} peaks, need at least {min_peaks}"
        )
    
    # Iterative matching and fitting
    current_cal = (initial_offset, initial_gain)
    best_matches = []
    
    for iteration in range(max_iterations):
        # Match peaks to lines
        matches = match_peaks_to_lines(
            peak_channels, known_lines, current_cal, tolerance_keV
        )
        
        if len(matches) < min_peaks:
            continue
        
        # Fit polynomial
        x = np.array([m.peak_channel for m in matches])
        y = np.array([m.expected_energy for m in matches])
        
        coeffs = np.polyfit(x, y, degree)
        
        # Update calibration
        if degree == 1:
            current_cal = (coeffs[1], coeffs[0])
        
        # Calculate residuals
        y_pred = np.polyval(coeffs, x)
        residuals = y - y_pred
        
        for i, m in enumerate(matches):
            m.residual = residuals[i]
        
        # Remove outliers (> 2 keV residual)
        matches = [m for m in matches if abs(m.residual) < 2.0]
        
        if len(matches) >= len(best_matches):
            best_matches = matches
    
    if len(best_matches) < min_peaks:
        return AutoCalibrationResult(
            coefficients=np.array([initial_offset, initial_gain]),
            matches=[],
            r_squared=0.0,
            chi_squared=float('inf'),
            rms_residual=float('inf'),
            isotopes_matched=[],
            success=False,
            message=f"Could only match {len(best_matches)} peaks"
        )
    
    # Final fit
    x = np.array([m.peak_channel for m in best_matches])
    y = np.array([m.expected_energy for m in best_matches])
    
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    residuals = y - y_pred
    
    # Statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals / 0.5)**2)  # Assume 0.5 keV uncertainty
    
    isotopes_matched = list(set(m.isotope for m in best_matches))
    
    return AutoCalibrationResult(
        coefficients=coeffs[::-1],  # Ascending order
        matches=best_matches,
        r_squared=r_squared,
        chi_squared=chi2,
        rms_residual=rms,
        isotopes_matched=isotopes_matched,
        success=True,
        message=f"Matched {len(best_matches)} peaks from {len(isotopes_matched)} isotopes"
    )


def refine_calibration(
    counts: NDArray,
    current_coeffs: NDArray,
    isotopes: List[str],
    degree: int = 2
) -> AutoCalibrationResult:
    """
    Refine an existing calibration with higher-order polynomial.
    
    Parameters
    ----------
    counts : NDArray
        Spectrum counts
    current_coeffs : NDArray
        Current calibration coefficients
    isotopes : list
        Known isotopes in spectrum
    degree : int
        New polynomial degree
    
    Returns
    -------
    AutoCalibrationResult
    """
    # First get initial gain/offset from current coeffs
    if len(current_coeffs) >= 2:
        initial_offset = current_coeffs[0]
        initial_gain = current_coeffs[1]
    else:
        initial_offset = 0.0
        initial_gain = current_coeffs[0] if len(current_coeffs) > 0 else 0.5
    
    return auto_calibrate(
        counts,
        isotopes=isotopes,
        initial_gain=initial_gain,
        initial_offset=initial_offset,
        degree=degree,
        tolerance_keV=3.0
    )


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing auto_calibration module...")
    
    # Create synthetic spectrum with peaks at known energies
    n_channels = 4096
    counts = np.random.poisson(10, n_channels).astype(float)
    
    # Add peaks at calibration energies (assuming 0.5 keV/ch, 0 offset)
    # Cs-137: 661.66 keV -> channel 1323
    # Co-60: 1173.23 keV -> channel 2346
    # Co-60: 1332.49 keV -> channel 2665
    
    def add_peak(ch, amplitude):
        x = np.arange(n_channels)
        counts[:] += amplitude * np.exp(-0.5 * ((x - ch) / 3)**2)
    
    add_peak(1323, 5000)  # Cs-137
    add_peak(2346, 3000)  # Co-60
    add_peak(2665, 3200)  # Co-60
    
    # Test auto-calibration
    result = auto_calibrate(
        counts,
        isotopes=['Co-60', 'Cs-137'],
        initial_gain=0.5,
        initial_offset=0.0
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Matches: {result.n_matches}")
    print(f"R²: {result.r_squared:.4f}")
    print(f"RMS residual: {result.rms_residual:.3f} keV")
    print(f"Coefficients: {result.coefficients}")
    
    if result.success:
        print("\n✅ auto_calibration module tests passed!")
    else:
        print(f"\n⚠️ Auto-calibration returned: {result.message}")
