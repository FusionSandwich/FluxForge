"""
HPGe Spectrum Processor Integration Module

Provides a complete workflow for processing HPGe gamma spectra:
- SPE file loading
- Energy calibration
- Peak finding and fitting
- Efficiency correction
- Activity calculation
- Comparison with ALARA predictions

This module integrates the io, data, and analysis subpackages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum, read_spe_file, write_spe_file
from fluxforge.data.efficiency import (
    EfficiencyCurve,
    CALIBRATION_SOURCES,
    calculate_efficiency_from_source,
)
from fluxforge.analysis.peakfit import (
    GaussianPeak,
    PeakFitResult,
    fit_single_peak,
    auto_find_peaks,
    calculate_activity,
    estimate_background,
)


@dataclass
class GammaLine:
    """
    Identified gamma line from spectrum analysis.
    
    Attributes
    ----------
    energy : float
        Gamma-ray energy (keV)
    isotope : str
        Source isotope (e.g., 'Co-60', 'Cs-137')
    net_counts : float
        Background-subtracted peak counts
    net_counts_unc : float
        Net counts uncertainty
    activity : float
        Calculated activity (Bq)
    activity_unc : float
        Activity uncertainty (Bq)
    efficiency : float
        Detection efficiency at this energy
    emission_probability : float
        Gamma emission probability
    fit_result : Optional[PeakFitResult]
        Full peak fit result
    """
    
    energy: float
    isotope: str
    net_counts: float
    net_counts_unc: float
    activity: float
    activity_unc: float
    efficiency: float
    emission_probability: float
    fit_result: Optional[PeakFitResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'energy': self.energy,
            'isotope': self.isotope,
            'net_counts': self.net_counts,
            'net_counts_unc': self.net_counts_unc,
            'activity': self.activity,
            'activity_unc': self.activity_unc,
            'efficiency': self.efficiency,
            'emission_probability': self.emission_probability,
        }


@dataclass
class HPGeAnalysisResult:
    """
    Complete HPGe spectrum analysis result.
    
    Attributes
    ----------
    spectrum : GammaSpectrum
        Original spectrum
    gamma_lines : List[GammaLine]
        Identified gamma lines
    efficiency_curve : Optional[EfficiencyCurve]
        Efficiency curve used
    background : np.ndarray
        Estimated background
    analysis_time : datetime
        Analysis timestamp
    metadata : Dict[str, Any]
        Additional metadata
    """
    
    spectrum: GammaSpectrum
    gamma_lines: List[GammaLine] = field(default_factory=list)
    efficiency_curve: Optional[EfficiencyCurve] = None
    background: np.ndarray = field(default_factory=lambda: np.array([]))
    analysis_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_activity(self) -> Tuple[float, float]:
        """Total activity and uncertainty."""
        if not self.gamma_lines:
            return 0.0, 0.0
        
        total = sum(gl.activity for gl in self.gamma_lines)
        unc = np.sqrt(sum(gl.activity_unc**2 for gl in self.gamma_lines))
        return total, unc
    
    def isotope_activities(self) -> Dict[str, Tuple[float, float]]:
        """Get activities grouped by isotope."""
        activities = {}
        
        for gl in self.gamma_lines:
            if gl.isotope not in activities:
                activities[gl.isotope] = []
            activities[gl.isotope].append((gl.activity, gl.activity_unc))
        
        # Average activities for each isotope
        result = {}
        for isotope, values in activities.items():
            acts = [v[0] for v in values]
            uncs = [v[1] for v in values]
            
            # Weighted average
            weights = [1/u**2 if u > 0 else 0 for u in uncs]
            total_weight = sum(weights)
            
            if total_weight > 0:
                avg = sum(a * w for a, w in zip(acts, weights)) / total_weight
                avg_unc = 1.0 / np.sqrt(total_weight)
            else:
                avg = np.mean(acts)
                avg_unc = np.std(acts)
            
            result[isotope] = (avg, avg_unc)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'spectrum_id': self.spectrum.spectrum_id,
            'live_time': self.spectrum.live_time,
            'real_time': self.spectrum.real_time,
            'gamma_lines': [gl.to_dict() for gl in self.gamma_lines],
            'isotope_activities': {
                k: {'activity': v[0], 'uncertainty': v[1]}
                for k, v in self.isotope_activities().items()
            },
            'analysis_time': self.analysis_time.isoformat(),
            'metadata': self.metadata,
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save analysis result to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Gamma Line Libraries
# =============================================================================

# Common gamma lines for activation products
# Format: {isotope: [(energy_keV, emission_probability, uncertainty), ...]}
ACTIVATION_GAMMA_LINES = {
    # Iron/Steel activation products
    'Mn-54': [(834.85, 0.9998, 0.0001)],
    'Mn-56': [(846.76, 0.989, 0.003), (1810.73, 0.272, 0.002)],
    'Fe-59': [(1099.25, 0.565, 0.003), (1291.60, 0.432, 0.002)],
    'Co-57': [(122.06, 0.856, 0.006), (136.47, 0.107, 0.001)],
    'Co-58': [(810.76, 0.9945, 0.0001)],
    'Co-60': [(1173.23, 0.9985, 0.0003), (1332.49, 0.9998, 0.0001)],
    'Cr-51': [(320.08, 0.0986, 0.001)],
    
    # Tungsten activation products
    'W-181': [(6.2, 0.0, 0.0)],  # X-rays only
    'W-185': [],  # Pure beta
    'W-187': [(685.77, 0.314, 0.005), (479.55, 0.218, 0.003)],
    
    # Nickel activation products
    'Ni-57': [(127.16, 0.167, 0.003), (1377.63, 0.817, 0.004)],
    'Ni-63': [],  # Pure beta
    
    # Vanadium activation products
    'V-52': [(1434.08, 0.999, 0.001)],
    'V-48': [(983.52, 0.9998, 0.0001), (1312.10, 0.982, 0.001)],
    
    # Common calibration sources
    'Am-241': [(59.54, 0.359, 0.004)],
    'Ba-133': [(356.01, 0.6205, 0.0019), (80.99, 0.329, 0.003)],
    'Cs-137': [(661.66, 0.851, 0.002)],
    'Eu-152': [
        (121.78, 0.286, 0.003),
        (344.28, 0.266, 0.002),
        (778.90, 0.129, 0.001),
        (964.13, 0.146, 0.001),
        (1112.08, 0.137, 0.001),
        (1408.01, 0.210, 0.002),
    ],
    'Na-22': [(511.00, 1.798, 0.002), (1274.54, 0.9994, 0.0002)],
}


def get_gamma_lines_for_isotope(isotope: str) -> List[Dict[str, float]]:
    """
    Get gamma line data for an isotope.
    
    Returns list of dicts with 'energy', 'intensity', 'uncertainty'.
    """
    if isotope in ACTIVATION_GAMMA_LINES:
        return [
            {'energy': e, 'intensity': i, 'uncertainty': u}
            for e, i, u in ACTIVATION_GAMMA_LINES[isotope]
        ]
    return []


def identify_isotope_from_energy(
    energy: float,
    tolerance: float = 1.0
) -> List[Tuple[str, float, float]]:
    """
    Identify possible isotopes from gamma-ray energy.
    
    Parameters
    ----------
    energy : float
        Gamma-ray energy (keV)
    tolerance : float
        Energy matching tolerance (keV)
    
    Returns
    -------
    list of (isotope, energy, intensity) tuples
        Possible isotope matches
    """
    matches = []
    
    for isotope, lines in ACTIVATION_GAMMA_LINES.items():
        for line_energy, intensity, _ in lines:
            if abs(line_energy - energy) <= tolerance:
                matches.append((isotope, line_energy, intensity))
    
    return matches


# =============================================================================
# HPGe Processor Class
# =============================================================================

class HPGeProcessor:
    """
    Complete HPGe spectrum processing workflow.
    
    Parameters
    ----------
    efficiency_curve : EfficiencyCurve, optional
        Detector efficiency curve
    energy_tolerance : float
        Energy matching tolerance for isotope ID (keV)
    peak_threshold : float
        Peak detection threshold (sigma above background)
    
    Examples
    --------
    >>> processor = HPGeProcessor(efficiency_curve=eff_curve)
    >>> spectrum = read_spe_file("sample.spe")
    >>> result = processor.analyze(spectrum)
    >>> print(result.isotope_activities())
    """
    
    def __init__(
        self,
        efficiency_curve: Optional[EfficiencyCurve] = None,
        energy_tolerance: float = 1.0,
        peak_threshold: float = 3.0
    ):
        self.efficiency_curve = efficiency_curve
        self.energy_tolerance = energy_tolerance
        self.peak_threshold = peak_threshold
        
        # Energy calibration (can be overridden)
        self.energy_calibration: Optional[List[float]] = None
    
    def analyze(
        self,
        spectrum: GammaSpectrum,
        known_isotopes: Optional[List[str]] = None,
        fit_all_peaks: bool = False
    ) -> HPGeAnalysisResult:
        """
        Analyze HPGe spectrum.
        
        Parameters
        ----------
        spectrum : GammaSpectrum
            Spectrum to analyze
        known_isotopes : list of str, optional
            List of expected isotopes (for targeted analysis)
        fit_all_peaks : bool
            If True, fit all detected peaks; else only known lines
        
        Returns
        -------
        HPGeAnalysisResult
            Analysis results
        """
        # Use spectrum calibration or processor calibration
        if spectrum.calibration.get('energy'):
            energy_cal = spectrum.calibration['energy']
        elif self.energy_calibration:
            energy_cal = self.energy_calibration
        else:
            # Default: 1:1 mapping (assume channels = keV)
            energy_cal = [0.0, 1.0]
        
        # Calculate energies
        energies = self._calibrate(spectrum.channels, energy_cal)
        
        # Estimate background
        background = estimate_background(
            spectrum.channels.astype(float),
            spectrum.counts.astype(float),
            method='snip'
        )
        
        gamma_lines = []
        
        if known_isotopes:
            # Targeted analysis: look for specific lines
            for isotope in known_isotopes:
                lines = get_gamma_lines_for_isotope(isotope)
                for line in lines:
                    result = self._analyze_line(
                        spectrum, energies, background,
                        line['energy'], isotope, line['intensity']
                    )
                    if result:
                        gamma_lines.append(result)
        
        if fit_all_peaks or not known_isotopes:
            # Auto peak finding
            peaks = auto_find_peaks(
                spectrum.channels.astype(float),
                spectrum.counts.astype(float),
                threshold=self.peak_threshold
            )
            
            for peak_ch, significance in peaks:
                # Get energy at peak
                if 0 <= peak_ch < len(energies):
                    peak_energy = energies[peak_ch]
                else:
                    continue
                
                # Check if already analyzed
                if any(abs(gl.energy - peak_energy) < self.energy_tolerance
                       for gl in gamma_lines):
                    continue
                
                # Try to identify isotope
                matches = identify_isotope_from_energy(
                    peak_energy, self.energy_tolerance
                )
                
                if matches:
                    isotope, line_energy, intensity = matches[0]
                else:
                    isotope = 'Unknown'
                    line_energy = peak_energy
                    intensity = 1.0
                
                result = self._analyze_line(
                    spectrum, energies, background,
                    peak_energy, isotope, intensity
                )
                if result:
                    gamma_lines.append(result)
        
        return HPGeAnalysisResult(
            spectrum=spectrum,
            gamma_lines=gamma_lines,
            efficiency_curve=self.efficiency_curve,
            background=background,
            metadata={
                'energy_calibration': energy_cal,
                'known_isotopes': known_isotopes,
                'peak_threshold': self.peak_threshold,
            }
        )
    
    def _calibrate(self, channels: np.ndarray, coefficients: List[float]) -> np.ndarray:
        """Apply energy calibration."""
        energies = np.zeros_like(channels, dtype=float)
        for i, coeff in enumerate(coefficients):
            energies += coeff * (channels.astype(float) ** i)
        return energies
    
    def _analyze_line(
        self,
        spectrum: GammaSpectrum,
        energies: np.ndarray,
        background: np.ndarray,
        target_energy: float,
        isotope: str,
        intensity: float
    ) -> Optional[GammaLine]:
        """Analyze a single gamma line."""
        # Find channel for target energy
        ch_idx = np.argmin(np.abs(energies - target_energy))
        
        if ch_idx < 5 or ch_idx > len(spectrum.counts) - 5:
            return None
        
        # Fit peak
        try:
            fit_result = fit_single_peak(
                spectrum.channels.astype(float),
                spectrum.counts.astype(float),
                int(spectrum.channels[ch_idx]),
                fit_width=15,
                background_model='linear'
            )
        except Exception:
            return None
        
        if not fit_result.success:
            return None
        
        # Check if peak is significant
        if fit_result.peak.area < 3 * fit_result.peak.area_uncertainty:
            return None
        
        # Get efficiency
        if self.efficiency_curve:
            efficiency = float(self.efficiency_curve.efficiency(target_energy))
            eff_unc = float(self.efficiency_curve.efficiency_uncertainty(target_energy))
        else:
            efficiency = 1.0
            eff_unc = 0.1
        
        # Calculate activity
        activity, activity_unc = calculate_activity(
            fit_result.peak.area,
            fit_result.peak.area_uncertainty,
            spectrum.live_time,
            efficiency,
            eff_unc * efficiency,  # Convert relative to absolute
            intensity
        )
        
        return GammaLine(
            energy=target_energy,
            isotope=isotope,
            net_counts=fit_result.peak.area,
            net_counts_unc=fit_result.peak.area_uncertainty,
            activity=activity,
            activity_unc=activity_unc,
            efficiency=efficiency,
            emission_probability=intensity,
            fit_result=fit_result,
        )
    
    def compare_with_alara(
        self,
        analysis_result: HPGeAnalysisResult,
        alara_activities: Dict[str, float],
        decay_time: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare measured activities with ALARA predictions.
        
        Parameters
        ----------
        analysis_result : HPGeAnalysisResult
            Measured spectrum analysis
        alara_activities : dict
            ALARA predicted activities {isotope: activity_Bq}
        decay_time : float
            Time since measurement for decay correction
        
        Returns
        -------
        dict
            Comparison results by isotope
        """
        comparison = {}
        
        measured = analysis_result.isotope_activities()
        
        all_isotopes = set(measured.keys()) | set(alara_activities.keys())
        
        for isotope in all_isotopes:
            meas_val, meas_unc = measured.get(isotope, (0.0, 0.0))
            alara_val = alara_activities.get(isotope, 0.0)
            
            if alara_val > 0:
                ratio = meas_val / alara_val
                ratio_unc = ratio * (meas_unc / meas_val) if meas_val > 0 else 0
            else:
                ratio = float('inf') if meas_val > 0 else 0
                ratio_unc = 0
            
            comparison[isotope] = {
                'measured': meas_val,
                'measured_unc': meas_unc,
                'predicted': alara_val,
                'ratio': ratio,
                'ratio_unc': ratio_unc,
                'difference_sigma': abs(meas_val - alara_val) / meas_unc if meas_unc > 0 else 0,
            }
        
        return comparison


# =============================================================================
# Convenience Functions
# =============================================================================

def process_spe_file(
    filepath: Union[str, Path],
    efficiency_curve: Optional[EfficiencyCurve] = None,
    known_isotopes: Optional[List[str]] = None
) -> HPGeAnalysisResult:
    """
    Convenience function to process SPE file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to SPE file
    efficiency_curve : EfficiencyCurve, optional
        Detector efficiency curve
    known_isotopes : list of str, optional
        Expected isotopes
    
    Returns
    -------
    HPGeAnalysisResult
    """
    spectrum = read_spe_file(filepath)
    processor = HPGeProcessor(efficiency_curve=efficiency_curve)
    return processor.analyze(spectrum, known_isotopes=known_isotopes)


def batch_process_spe(
    filepaths: List[Union[str, Path]],
    efficiency_curve: Optional[EfficiencyCurve] = None,
    known_isotopes: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> List[HPGeAnalysisResult]:
    """
    Batch process multiple SPE files.
    
    Parameters
    ----------
    filepaths : list
        Paths to SPE files
    efficiency_curve : EfficiencyCurve, optional
        Detector efficiency curve
    known_isotopes : list of str, optional
        Expected isotopes
    output_dir : str or Path, optional
        Directory to save results
    
    Returns
    -------
    list of HPGeAnalysisResult
    """
    processor = HPGeProcessor(efficiency_curve=efficiency_curve)
    results = []
    
    for fp in filepaths:
        spectrum = read_spe_file(fp)
        result = processor.analyze(spectrum, known_isotopes=known_isotopes)
        results.append(result)
        
        if output_dir:
            output_path = Path(output_dir) / f"{Path(fp).stem}_analysis.json"
            result.save(output_path)
    
    return results
