"""
Flux Wire Analysis Workflow

Complete workflow for analyzing flux wire gamma spectra, including:
- Peak finding and fitting
- Efficiency-corrected activity calculation  
- Comparison between raw and processed results
- Nuclide identification

This module provides tools to process raw gamma spectra and calculate
activities that match commercial analysis software (QuantaGraph).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize

from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.flux_wire import (
    FluxWireData,
    NuclideResult,
    EfficiencyCalibration,
    read_flux_wire,
    read_raw_asc,
    read_processed_txt,
)
from fluxforge.analysis.peak_finders import (
    PeakInfo,
    snip_background,
    WindowPeakFinder,
    refine_peak_centroids,
)
from fluxforge.analysis.peakfit import (
    fit_single_peak,
    fit_multiple_peaks,
    calculate_activity,
    PeakFitResult,
)
from fluxforge.physics.nuclides import NuclideDatabase, get_nuclide_database


# ============================================================================
# Flux Wire Nuclide Library
# ============================================================================

# Standard flux wire nuclides with gamma energies and branching ratios
# Each entry includes the parent element (flux wire material) and reaction type
FLUX_WIRE_NUCLIDES = {
    'Co60': {
        'half_life_s': 166344864.0,  # 5.271 years
        'parent_element': 'Co',
        'reaction': 'Co59(n,g)',
        'gamma_lines': [
            {'energy_keV': 1173.2, 'intensity': 0.9985},
            {'energy_keV': 1332.5, 'intensity': 0.9998},
        ],
    },
    'Sc46': {
        'half_life_s': 7241184.0,  # 83.81 days
        'parent_element': 'Sc',  # Also from Ti via threshold reactions
        'reaction': 'Sc45(n,g)',
        'gamma_lines': [
            {'energy_keV': 889.3, 'intensity': 0.9998},
            {'energy_keV': 1120.5, 'intensity': 0.9999},
        ],
    },
    'Sc47': {
        'half_life_s': 289310.4,  # 3.349 days
        'parent_element': 'Ti',
        'reaction': 'Ti47(n,p)',
        'gamma_lines': [
            {'energy_keV': 159.4, 'intensity': 0.683},
        ],
    },
    'Sc48': {
        'half_life_s': 157320.0,  # 43.7 hours
        'parent_element': 'Ti',
        'reaction': 'Ti48(n,p)',
        'gamma_lines': [
            {'energy_keV': 983.5, 'intensity': 1.001},
            {'energy_keV': 1037.5, 'intensity': 0.976},
            {'energy_keV': 1312.1, 'intensity': 1.001},
        ],
    },
    'Co58': {
        'half_life_s': 6122361.6,  # 70.86 days
        'parent_element': 'Ni',
        'reaction': 'Ni58(n,p)',
        'gamma_lines': [
            {'energy_keV': 810.8, 'intensity': 0.9945},
        ],
    },
    'Cu64': {
        'half_life_s': 45723.6,  # 12.701 hours
        'parent_element': 'Cu',
        'reaction': 'Cu63(n,g)',
        'gamma_lines': [
            {'energy_keV': 1345.8, 'intensity': 0.0047},
        ],
    },
    'In114m': {
        'half_life_s': 4276944.0,  # 49.51 days
        'parent_element': 'In',
        'reaction': 'In113(n,g)',
        'gamma_lines': [
            {'energy_keV': 190.3, 'intensity': 0.1556},
            {'energy_keV': 558.5, 'intensity': 0.044},
            {'energy_keV': 725.2, 'intensity': 0.044},
        ],
    },
    'In115m': {
        'half_life_s': 16146.0,  # 4.485 hours
        'parent_element': 'In',
        'reaction': 'In115(n,n\')',
        'gamma_lines': [
            {'energy_keV': 336.2, 'intensity': 0.458},
        ],
    },
    'Ni57': {
        'half_life_s': 127872.0,  # 35.52 hours
        'parent_element': 'Ni',
        'reaction': 'Ni58(n,2n)',
        'gamma_lines': [
            {'energy_keV': 127.2, 'intensity': 0.167},
            {'energy_keV': 1377.6, 'intensity': 0.817},
            {'energy_keV': 1919.5, 'intensity': 0.123},
        ],
    },
    'Fe59': {
        'half_life_s': 3844800.0,  # 44.5 days
        'parent_element': 'Fe',
        'reaction': 'Fe58(n,g)',
        'gamma_lines': [
            {'energy_keV': 1099.3, 'intensity': 0.565},
            {'energy_keV': 1291.6, 'intensity': 0.432},
        ],
    },
    'Mn54': {
        'half_life_s': 26959200.0,  # 312.03 days
        'parent_element': 'Fe',
        'reaction': 'Fe54(n,p)',
        'gamma_lines': [
            {'energy_keV': 834.8, 'intensity': 0.9998},
        ],
    },
    'Ti51': {
        'half_life_s': 345.6,  # 5.76 minutes
        'parent_element': 'Ti',
        'reaction': 'Ti50(n,g)',
        'gamma_lines': [
            {'energy_keV': 320.1, 'intensity': 0.930},
        ],
    },
}

# Mapping from flux wire element to expected activation products
# Ti wires also produce Sc-46 via Ti46(n,p) threshold reaction
ELEMENT_TO_ISOTOPES = {
    'Co': ['Co60'],
    'Cu': ['Cu64'],
    'In': ['In114m', 'In115m'],
    'Sc': ['Sc46'],
    'Ti': ['Sc46', 'Sc47', 'Sc48', 'Ti51'],  # Threshold reactions
    'Ni': ['Co58', 'Ni57'],  # (n,p) and (n,2n)
    'Fe': ['Fe59', 'Mn54'],  # (n,g) and (n,p)
}


def get_sample_element(sample_name: str) -> Optional[str]:
    """
    Extract the flux wire element from a sample name.
    
    Sample names follow patterns like:
    - "Co-Cd-RAFM-1_25cm" -> Co
    - "Cu-RAFM-1" -> Cu
    - "Ti-RAFM-1_25cm" -> Ti
    - "CU-RAFM-1" -> Cu (case-insensitive)
    
    Parameters
    ----------
    sample_name : str
        Sample identifier string
        
    Returns
    -------
    str or None
        Element symbol, or None if not found
    """
    # Extract first part before any dash or underscore
    # Pattern: Element-Cd-... or Element-RAFM-...
    # Case-insensitive matching
    match = re.match(r'^([A-Za-z]{1,2})(?:-|_)', sample_name)
    if match:
        # Capitalize properly (first letter upper, second lower)
        element_raw = match.group(1)
        element = element_raw[0].upper()
        if len(element_raw) > 1:
            element += element_raw[1].lower()
        # Verify it's a known flux wire element
        if element in ELEMENT_TO_ISOTOPES:
            return element
    return None


def get_expected_isotopes(sample_name: str) -> List[str]:
    """
    Get the expected activation product isotopes for a flux wire sample.
    
    Parameters
    ----------
    sample_name : str
        Sample identifier string
        
    Returns
    -------
    list of str
        Expected isotope names (e.g., ['Co60'])
    """
    element = get_sample_element(sample_name)
    if element:
        return ELEMENT_TO_ISOTOPES.get(element, [])
    return []


@dataclass
class GammaLine:
    """Gamma line definition for nuclide identification."""
    energy_keV: float
    intensity: float  # Branching ratio
    isotope: str
    
    @property
    def activity_factor(self) -> float:
        """Factor to convert counts to activity: 1/intensity."""
        return 1.0 / self.intensity if self.intensity > 0 else 0.0


@dataclass
class IdentifiedPeak:
    """Peak with nuclide identification."""
    
    channel: int
    energy_keV: float
    net_counts: float
    net_counts_unc: float
    gross_counts: float
    background: float
    fwhm: float
    significance: float
    
    # Identification
    isotope: Optional[str] = None
    gamma_line: Optional[GammaLine] = None
    
    # Activity calculation
    efficiency: float = 0.0
    activity_bq: float = 0.0
    activity_unc_bq: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'channel': self.channel,
            'energy_keV': self.energy_keV,
            'net_counts': self.net_counts,
            'net_counts_unc': self.net_counts_unc,
            'gross_counts': self.gross_counts,
            'background': self.background,
            'fwhm': self.fwhm,
            'significance': self.significance,
            'isotope': self.isotope,
            'gamma_energy': self.gamma_line.energy_keV if self.gamma_line else None,
            'branching_ratio': self.gamma_line.intensity if self.gamma_line else None,
            'efficiency': self.efficiency,
            'activity_bq': self.activity_bq,
            'activity_unc_bq': self.activity_unc_bq,
        }


@dataclass
class FluxWireAnalysisResult:
    """Complete flux wire analysis result."""
    
    sample_id: str
    source_file: str
    live_time: float
    real_time: float
    dead_time_pct: float
    
    # Detected peaks with identification
    peaks: List[IdentifiedPeak] = field(default_factory=list)
    
    # Nuclide activities (combined from multiple peaks)
    nuclide_activities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Comparison with reference (if available)
    reference_activities: Dict[str, float] = field(default_factory=dict)
    activity_ratios: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_id': self.sample_id,
            'source_file': self.source_file,
            'live_time': self.live_time,
            'real_time': self.real_time,
            'dead_time_pct': self.dead_time_pct,
            'peaks': [p.to_dict() for p in self.peaks],
            'nuclide_activities': self.nuclide_activities,
            'reference_activities': self.reference_activities,
            'activity_ratios': self.activity_ratios,
        }


def build_gamma_library(
    nuclides: Optional[Dict[str, Dict]] = None,
    isotope_filter: Optional[List[str]] = None,
) -> List[GammaLine]:
    """
    Build gamma line library for nuclide identification.
    
    Parameters
    ----------
    nuclides : dict, optional
        Custom nuclide dictionary. Uses FLUX_WIRE_NUCLIDES if None.
    isotope_filter : list of str, optional
        If provided, only include gamma lines from these isotopes.
        E.g., ['Co60', 'Sc46']
    
    Returns
    -------
    list of GammaLine
        Sorted list of gamma lines
    """
    if nuclides is None:
        nuclides = FLUX_WIRE_NUCLIDES
    
    lines = []
    for isotope, data in nuclides.items():
        # Apply isotope filter if specified
        if isotope_filter is not None and isotope not in isotope_filter:
            continue
            
        for gamma in data.get('gamma_lines', []):
            lines.append(GammaLine(
                energy_keV=gamma['energy_keV'],
                intensity=gamma['intensity'],
                isotope=isotope,
            ))
    
    # Sort by energy
    return sorted(lines, key=lambda x: x.energy_keV)


def identify_peaks(
    peak_energies: np.ndarray,
    gamma_library: List[GammaLine],
    energy_tolerance_keV: float = 2.0
) -> List[Optional[GammaLine]]:
    """
    Identify peaks by matching to gamma library.
    
    Parameters
    ----------
    peak_energies : np.ndarray
        Peak energies in keV
    gamma_library : list of GammaLine
        Reference gamma lines
    energy_tolerance_keV : float
        Maximum energy difference for match
    
    Returns
    -------
    list of GammaLine or None
        Matched gamma line for each peak, or None if no match
    """
    identifications = []
    
    for energy in peak_energies:
        best_match = None
        best_diff = float('inf')
        
        for gamma in gamma_library:
            diff = abs(energy - gamma.energy_keV)
            if diff < energy_tolerance_keV and diff < best_diff:
                best_match = gamma
                best_diff = diff
        
        identifications.append(best_match)
    
    return identifications


def estimate_peak_area(
    spectrum: np.ndarray,
    peak_channel: int,
    background: np.ndarray,
    fwhm_channels: float = 8.0
) -> Tuple[float, float, float]:
    """
    Estimate peak area using simple summation method.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Raw spectrum counts
    peak_channel : int
        Peak centroid channel
    background : np.ndarray
        Background estimate
    fwhm_channels : float
        Full width at half maximum in channels
    
    Returns
    -------
    net_counts : float
        Net peak counts
    net_unc : float
        Net counts uncertainty
    gross_counts : float
        Gross counts in ROI
    """
    # ROI width: ±2*FWHM typically captures 95% of Gaussian
    half_width = int(2.5 * fwhm_channels)
    
    ch_min = max(0, peak_channel - half_width)
    ch_max = min(len(spectrum) - 1, peak_channel + half_width)
    
    # Sum counts in ROI
    gross = float(spectrum[ch_min:ch_max+1].sum())
    bg = float(background[ch_min:ch_max+1].sum())
    
    net = gross - bg
    
    # Uncertainty: sqrt(gross + background) for counting statistics
    net_unc = np.sqrt(gross + bg)
    
    return net, net_unc, gross


def analyze_raw_spectrum(
    spectrum: GammaSpectrum,
    efficiency: Optional[EfficiencyCalibration] = None,
    gamma_library: Optional[List[GammaLine]] = None,
    peak_threshold: float = 5.0,
    min_energy_keV: float = 50.0,
    max_energy_keV: float = 3000.0,
    sample_name: Optional[str] = None,
) -> List[IdentifiedPeak]:
    """
    Analyze raw gamma spectrum to find and identify peaks.
    
    Parameters
    ----------
    spectrum : GammaSpectrum
        Raw spectrum data
    efficiency : EfficiencyCalibration, optional
        Detector efficiency model
    gamma_library : list of GammaLine, optional
        Gamma line library for identification. If sample_name is provided,
        the library will be filtered to only include expected isotopes.
    peak_threshold : float
        Peak detection threshold (sigma above background)
    min_energy_keV, max_energy_keV : float
        Energy range for analysis
    sample_name : str, optional
        Sample identifier (e.g., "Co-Cd-RAFM-1_25cm"). If provided, only
        isotopes expected from that flux wire material will be identified.
    
    Returns
    -------
    list of IdentifiedPeak
        Detected and identified peaks
    """
    # Build or filter gamma library based on sample element
    if gamma_library is None:
        if sample_name:
            expected_isotopes = get_expected_isotopes(sample_name)
            if expected_isotopes:
                gamma_library = build_gamma_library(isotope_filter=expected_isotopes)
            else:
                gamma_library = build_gamma_library()
        else:
            gamma_library = build_gamma_library()
    
    counts = spectrum.counts
    channels = spectrum.channels
    
    # Estimate background using SNIP
    background = snip_background(counts, n_iterations=24)
    
    # Find peaks using window peak finder
    finder = WindowPeakFinder(
        threshold=peak_threshold,
        n_outer=50,
        enforce_maximum=True,
    )
    raw_peaks = finder.find(counts)
    
    identified_peaks = []
    
    for peak in raw_peaks:
        ch = peak.index
        
        # Convert to energy
        if spectrum.energies is not None:
            energy = spectrum.energies[ch]
        else:
            energy = spectrum.channel_to_energy(ch)
        
        # Skip peaks outside energy range
        if energy < min_energy_keV or energy > max_energy_keV:
            continue
        
        # Estimate FWHM in channels (typical HPGe resolution)
        # FWHM ~ 2.0 keV at 661 keV, scales as sqrt(E)
        fwhm_keV = 1.5 + 0.001 * energy
        cal = spectrum.calibration.get('energy', [0, 0.5])
        fwhm_ch = fwhm_keV / cal[1] if len(cal) > 1 and cal[1] > 0 else 4.0
        
        # Estimate peak area
        net, net_unc, gross = estimate_peak_area(
            counts, ch, background, fwhm_ch
        )
        
        # Skip peaks with negative net counts
        if net <= 0:
            continue
        
        # Calculate significance
        significance = net / net_unc if net_unc > 0 else 0
        
        # Skip low-significance peaks
        if significance < peak_threshold:
            continue
        
        # Create identified peak
        id_peak = IdentifiedPeak(
            channel=ch,
            energy_keV=energy,
            net_counts=net,
            net_counts_unc=net_unc,
            gross_counts=gross,
            background=float(background[ch]),
            fwhm=fwhm_keV,
            significance=significance,
        )
        
        # Try to identify nuclide
        matches = identify_peaks(np.array([energy]), gamma_library)
        if matches[0] is not None:
            id_peak.isotope = matches[0].isotope
            id_peak.gamma_line = matches[0]
        
        # Calculate efficiency
        if efficiency is not None:
            id_peak.efficiency = float(efficiency.efficiency(energy))
            
            # Calculate activity if identified
            if id_peak.gamma_line is not None and id_peak.efficiency > 0:
                intensity = id_peak.gamma_line.intensity
                
                activity, activity_unc = calculate_activity(
                    net_counts=net,
                    net_counts_unc=net_unc,
                    live_time=spectrum.live_time,
                    efficiency=id_peak.efficiency,
                    efficiency_unc=0.05 * id_peak.efficiency,  # Assume 5% eff uncertainty
                    emission_probability=intensity,
                    emission_probability_unc=0.01 * intensity,
                )
                
                id_peak.activity_bq = activity
                id_peak.activity_unc_bq = activity_unc
        
        identified_peaks.append(id_peak)
    
    return identified_peaks


def combine_peak_activities(
    peaks: List[IdentifiedPeak]
) -> Dict[str, Dict[str, Any]]:
    """
    Combine activities from multiple peaks of the same nuclide.
    
    Uses weighted average when multiple gamma lines are available.
    
    Parameters
    ----------
    peaks : list of IdentifiedPeak
        Identified peaks with activities
    
    Returns
    -------
    dict
        Nuclide activities with uncertainties
    """
    # Group peaks by nuclide
    nuclide_peaks: Dict[str, List[IdentifiedPeak]] = {}
    for peak in peaks:
        if peak.isotope is None:
            continue
        if peak.isotope not in nuclide_peaks:
            nuclide_peaks[peak.isotope] = []
        nuclide_peaks[peak.isotope].append(peak)
    
    results = {}
    
    for isotope, iso_peaks in nuclide_peaks.items():
        if len(iso_peaks) == 0:
            continue
        
        # Filter peaks with valid activity
        valid_peaks = [p for p in iso_peaks if p.activity_bq > 0 and p.activity_unc_bq > 0]
        
        if len(valid_peaks) == 0:
            continue
        
        if len(valid_peaks) == 1:
            # Single peak: use directly
            p = valid_peaks[0]
            results[isotope] = {
                'activity_bq': p.activity_bq,
                'activity_unc_bq': p.activity_unc_bq,
                'activity_uci': p.activity_bq / 3.7e4,
                'activity_unc_uci': p.activity_unc_bq / 3.7e4,
                'n_peaks': 1,
                'peak_energies': [p.energy_keV],
            }
        else:
            # Multiple peaks: weighted average
            weights = np.array([1.0 / p.activity_unc_bq**2 for p in valid_peaks])
            activities = np.array([p.activity_bq for p in valid_peaks])
            
            weighted_avg = np.sum(weights * activities) / np.sum(weights)
            weighted_unc = 1.0 / np.sqrt(np.sum(weights))
            
            results[isotope] = {
                'activity_bq': weighted_avg,
                'activity_unc_bq': weighted_unc,
                'activity_uci': weighted_avg / 3.7e4,
                'activity_unc_uci': weighted_unc / 3.7e4,
                'n_peaks': len(valid_peaks),
                'peak_energies': [p.energy_keV for p in valid_peaks],
            }
    
    return results


def analyze_flux_wire(
    data: FluxWireData,
    reference_data: Optional[FluxWireData] = None,
    peak_threshold: float = 5.0,
) -> FluxWireAnalysisResult:
    """
    Analyze flux wire data and calculate nuclide activities.
    
    Parameters
    ----------
    data : FluxWireData
        Raw or processed flux wire data
    reference_data : FluxWireData, optional
        Reference processed data for comparison
    peak_threshold : float
        Peak detection threshold (sigma)
    
    Returns
    -------
    FluxWireAnalysisResult
        Complete analysis result
    """
    result = FluxWireAnalysisResult(
        sample_id=data.sample_id,
        source_file=data.source_file,
        live_time=data.live_time,
        real_time=data.real_time,
        dead_time_pct=data.dead_time_pct,
    )
    
    # If raw spectrum available, analyze it
    if data.has_spectrum:
        # Build library filtered by expected isotopes from sample element
        expected_isotopes = get_expected_isotopes(data.sample_id)
        if expected_isotopes:
            gamma_library = build_gamma_library(isotope_filter=expected_isotopes)
        else:
            gamma_library = build_gamma_library()
        
        peaks = analyze_raw_spectrum(
            spectrum=data.spectrum,
            efficiency=data.efficiency,
            gamma_library=gamma_library,
            peak_threshold=peak_threshold,
            sample_name=data.sample_id,
        )
        
        result.peaks = peaks
        result.nuclide_activities = combine_peak_activities(peaks)
    
    # If processed results available, use those as reference
    if reference_data is not None and reference_data.has_nuclides:
        for nuclide in reference_data.nuclides:
            result.reference_activities[nuclide.isotope] = nuclide.activity_bq
    
    # If we have both, calculate ratios
    if result.nuclide_activities and result.reference_activities:
        for isotope, act in result.nuclide_activities.items():
            if isotope in result.reference_activities:
                ref = result.reference_activities[isotope]
                if ref > 0:
                    result.activity_ratios[isotope] = act['activity_bq'] / ref
    
    return result


def compare_raw_vs_processed(
    raw_file: Union[str, Path],
    processed_file: Union[str, Path],
    peak_threshold: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare raw spectrum analysis with commercial processed results.
    
    Parameters
    ----------
    raw_file : str or Path
        Path to raw .ASC file
    processed_file : str or Path
        Path to processed .txt file
    peak_threshold : float
        Peak detection threshold (sigma)
    verbose : bool
        Print comparison results
    
    Returns
    -------
    dict
        Comparison results including activity ratios
    """
    # Load files
    raw_data = read_raw_asc(raw_file)
    processed_data = read_processed_txt(processed_file)
    
    # Copy efficiency from processed to raw
    if processed_data.efficiency is not None:
        raw_data.efficiency = processed_data.efficiency
    
    # Analyze raw spectrum
    result = analyze_flux_wire(
        raw_data,
        reference_data=processed_data,
        peak_threshold=peak_threshold,
    )
    
    comparison = {
        'sample_id': result.sample_id,
        'live_time': result.live_time,
        'n_peaks_found': len(result.peaks),
        'nuclides_analyzed': list(result.nuclide_activities.keys()),
        'reference_nuclides': list(result.reference_activities.keys()),
        'activity_comparison': {},
    }
    
    if verbose:
        print("=" * 80)
        print(f"FLUX WIRE ANALYSIS COMPARISON: {result.sample_id}")
        print("=" * 80)
        print(f"Live time: {result.live_time:.1f} s")
        print(f"Peaks found: {len(result.peaks)}")
        print()
        
        # Show all detected peaks
        print("Detected Peaks:")
        print("-" * 80)
        print(f"{'Energy':>10s} {'Net Counts':>12s} {'Sigma':>8s} {'Isotope':>10s} {'Activity (uCi)':>15s}")
        print("-" * 80)
        
        for peak in result.peaks:
            isotope = peak.isotope or "Unknown"
            activity_uci = peak.activity_bq / 3.7e4 if peak.activity_bq > 0 else 0
            print(f"{peak.energy_keV:10.2f} {peak.net_counts:12.0f} {peak.significance:8.1f} "
                  f"{isotope:>10s} {activity_uci:15.4e}")
        print()
    
    # Compare activities
    for isotope in processed_data.nuclides:
        ref_bq = isotope.activity_bq
        ref_uci = isotope.activity
        
        if isotope.isotope in result.nuclide_activities:
            calc = result.nuclide_activities[isotope.isotope]
            calc_bq = calc['activity_bq']
            calc_uci = calc['activity_uci']
            ratio = calc_bq / ref_bq if ref_bq > 0 else 0
            diff_pct = (ratio - 1.0) * 100
            
            comparison['activity_comparison'][isotope.isotope] = {
                'calculated_bq': calc_bq,
                'calculated_uci': calc_uci,
                'reference_bq': ref_bq,
                'reference_uci': ref_uci,
                'ratio': ratio,
                'diff_pct': diff_pct,
            }
            
            if verbose:
                print(f"{isotope.isotope:>10s}: Calc={calc_uci:.4e} uCi, Ref={ref_uci:.4e} uCi, "
                      f"Ratio={ratio:.3f} ({diff_pct:+.1f}%)")
        else:
            comparison['activity_comparison'][isotope.isotope] = {
                'calculated_bq': None,
                'reference_bq': ref_bq,
                'reference_uci': ref_uci,
                'status': 'not_detected',
            }
            
            if verbose:
                print(f"{isotope.isotope:>10s}: NOT DETECTED (Ref={ref_uci:.4e} uCi)")
    
    return comparison


def batch_analyze_flux_wires(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    peak_threshold: float = 5.0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch compare raw and processed flux wire files.
    
    Parameters
    ----------
    raw_dir : str or Path
        Directory containing raw .ASC files
    processed_dir : str or Path
        Directory containing processed .txt files
    output_file : str or Path, optional
        Path to save results JSON
    peak_threshold : float
        Peak detection threshold
    verbose : bool
        Print progress and results
    
    Returns
    -------
    list of dict
        Comparison results for each file pair
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    results = []
    
    # Find matching file pairs
    raw_files = {f.stem: f for f in raw_dir.glob("*.ASC")}
    processed_files = {f.stem: f for f in processed_dir.glob("*.txt")}
    
    # Match by filename stem
    matched = set(raw_files.keys()) & set(processed_files.keys())
    
    if verbose:
        print(f"Found {len(raw_files)} raw files, {len(processed_files)} processed files")
        print(f"Matched pairs: {len(matched)}")
        print()
    
    for stem in sorted(matched):
        if verbose:
            print(f"\nProcessing: {stem}")
        
        try:
            comparison = compare_raw_vs_processed(
                raw_files[stem],
                processed_files[stem],
                peak_threshold=peak_threshold,
                verbose=verbose,
            )
            results.append(comparison)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results.append({
                'sample_id': stem,
                'error': str(e),
            })
    
    # Save results
    if output_file is not None:
        import json
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_file}")
    
    return results
