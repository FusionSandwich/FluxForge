"""
Batch Gamma Spectrum Processing Pipeline

Complete workflow for processing multiple gamma spectra, applying
efficiency corrections, detecting peaks, matching isotopes, and
generating comparison plots and CSV outputs.

Based on the batch_compare_spectra.py workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.genie import (
    read_genie_spectrum,
    parse_asc_filename,
    normalize_timepoint,
    parse_genie_report,
    ReportPeak,
    discover_spectrum_pairs,
    SpectrumPair,
)
from fluxforge.data.efficiency_models import EfficiencyModel, apply_efficiency_correction
from fluxforge.analysis.segmented_detection import (
    SegmentedDetectionConfig,
    DetectedPeak,
    detect_peaks_segmented,
    merge_duplicate_peaks,
    filter_peaks_by_energy,
    create_report_peaks,
    combine_with_report_peaks,
)
from fluxforge.analysis.ensdf_matching import (
    GammaDatabase,
    IsotopeMatch,
    match_peaks_three_tier,
    build_gamma_database,
    build_tier1_isotopes,
    create_matching_databases,
    get_data_source,
)


# ============================================================================
# Batch Processing Configuration
# ============================================================================

@dataclass
class BatchProcessingConfig:
    """Configuration for batch spectrum processing."""
    
    # Directories
    spectra_dir: str = "../spectra_files"
    output_dir: str = "output_plots/batch"
    report_dir: Optional[str] = None  # Genie/LabSOCS report files
    
    # Efficiency model
    efficiency_csv: Optional[str] = None
    efficiency_model_type: str = "log_poly"  # or "log10_poly"
    
    # Detection configuration
    detection_config: SegmentedDetectionConfig = field(
        default_factory=SegmentedDetectionConfig.default
    )
    
    # Matching tolerances (keV)
    tol_tier1: float = 3.0
    tol_tier2: float = 3.0
    tol_tier3: float = 2.0
    top_n_candidates: int = 3
    
    # Elements for tier-2 matching
    tier2_elements: List[str] = field(default_factory=lambda: [
        'Fe', 'Cr', 'Mn', 'Mo', 'V', 'Si', 'P', 'S', 'C', 'Co', 'Ni', 'W',
        'Sb', 'As', 'Ta', 'Tb', 'Al'
    ])
    
    # Tier-1 element sources
    tier1_elements: List[str] = field(default_factory=lambda: [
        'W', 'Cr', 'Fe', 'Ta', 'Co', 'Mn', 'V', 'Al', 'Tb'
    ])
    tier1_min_intensity: float = 10.0
    
    # Plotting
    annotate_top_single: int = 18
    annotate_top_compare: int = 6
    annihilation_tol_keV: float = 3.0
    
    # File patterns
    spectrum_pattern: str = "*.ASC"


# ============================================================================
# Processing Results
# ============================================================================

@dataclass
class SpectrumResult:
    """Result of processing a single spectrum."""
    
    filepath: str
    spectrum: GammaSpectrum
    peaks: List[DetectedPeak]
    matches: List[IsotopeMatch]
    
    # Calibration
    cal_A: float = 0.0
    cal_B: float = 1.0
    cal_C: float = 0.0
    
    # Timing
    live_time: Optional[float] = None
    real_time: Optional[float] = None
    
    # Parsed filename
    sample: str = ""
    letter: str = ""  # 'C' or 'N'
    timepoint: str = ""
    
    # Report info
    n_report_peaks: int = 0
    
    @property
    def energies(self) -> np.ndarray:
        """Calibrated energies."""
        return self.spectrum.energies
    
    @property
    def counts(self) -> np.ndarray:
        """Raw counts."""
        return self.spectrum.counts
    
    @property
    def corrected_counts(self) -> np.ndarray:
        """Efficiency-corrected counts."""
        return getattr(self.spectrum, 'corrected_counts', self.spectrum.counts)


@dataclass
class BatchResult:
    """Result of batch processing."""
    
    results: Dict[str, SpectrumResult]  # keyed by filepath
    pairs: List[SpectrumPair]
    config: BatchProcessingConfig
    
    def get_result(self, filepath: str) -> Optional[SpectrumResult]:
        """Get result by filepath."""
        return self.results.get(filepath)
    
    def get_by_sample_time(self, sample: str, timepoint: str) -> Dict[str, SpectrumResult]:
        """Get C and N results for a sample/timepoint."""
        tp_norm = normalize_timepoint(timepoint)
        matches = {}
        for path, result in self.results.items():
            if result.sample == sample and normalize_timepoint(result.timepoint) == tp_norm:
                matches[result.letter] = result
        return matches


# ============================================================================
# Processing Functions
# ============================================================================

def load_efficiency_model(config: BatchProcessingConfig) -> Optional[EfficiencyModel]:
    """Load efficiency model from configuration."""
    if config.efficiency_csv is None:
        return None
    
    if config.efficiency_model_type == "log10_poly":
        return EfficiencyModel.from_v4_csv(config.efficiency_csv)
    else:
        return EfficiencyModel.from_labsocs_csv(config.efficiency_csv)


def process_single_spectrum(
    filepath: str,
    efficiency_model: Optional[EfficiencyModel],
    tier1_db: Optional[GammaDatabase],
    tier2_db: Optional[GammaDatabase],
    tier3_db: Optional[GammaDatabase],
    config: BatchProcessingConfig,
    report_peaks: Optional[List[ReportPeak]] = None,
) -> SpectrumResult:
    """
    Process a single spectrum file.
    
    Parameters
    ----------
    filepath : str
        Path to spectrum file
    efficiency_model : EfficiencyModel or None
        Detector efficiency model
    tier1_db, tier2_db, tier3_db : GammaDatabase or None
        Matching databases
    config : BatchProcessingConfig
        Processing configuration
    report_peaks : list of ReportPeak, optional
        Pre-identified peaks from report files
    
    Returns
    -------
    SpectrumResult
        Processing result
    """
    # Read spectrum
    spectrum = read_genie_spectrum(filepath)
    
    # Get calibration
    cal = spectrum.calibration.get('energy', [0.0, 1.0, 0.0])
    cal_A = cal[0] if len(cal) > 0 else 0.0
    cal_B = cal[1] if len(cal) > 1 else 1.0
    cal_C = cal[2] if len(cal) > 2 else 0.0
    
    # Apply efficiency correction
    if efficiency_model is not None:
        corrected = apply_efficiency_correction(
            spectrum.counts.astype(float),
            spectrum.energies,
            efficiency_model
        )
        spectrum.corrected_counts = corrected
    else:
        spectrum.corrected_counts = spectrum.counts.astype(float)
    
    # Parse filename
    parsed = parse_asc_filename(filepath)
    sample, letter, timepoint = parsed if parsed else ("", "", "")
    
    # Create report peaks if provided
    if report_peaks:
        report_detected = create_report_peaks(
            report_peaks,
            spectrum.channels,
            spectrum.energies,
            spectrum.corrected_counts,
        )
    else:
        report_detected = []
    
    # Detect peaks
    auto_peaks = detect_peaks_segmented(
        spectrum.channels,
        spectrum.energies,
        spectrum.corrected_counts,
        raw_counts=spectrum.counts,
        config=config.detection_config,
    )
    
    # Combine and deduplicate
    all_peaks = combine_with_report_peaks(
        auto_peaks,
        report_detected,
        merge_tolerance_keV=config.detection_config.merge_tolerance_keV,
    )
    
    # Filter by energy
    all_peaks = filter_peaks_by_energy(
        all_peaks,
        min_energy_keV=config.detection_config.min_energy_keV,
    )
    
    # Match isotopes
    matches = match_peaks_three_tier(
        all_peaks,
        tier1_db,
        tier2_db,
        tier3_db,
        tol_tier1=config.tol_tier1,
        tol_tier2=config.tol_tier2,
        tol_tier3=config.tol_tier3,
        top_n=config.top_n_candidates,
    )
    
    return SpectrumResult(
        filepath=filepath,
        spectrum=spectrum,
        peaks=all_peaks,
        matches=matches,
        cal_A=cal_A,
        cal_B=cal_B,
        cal_C=cal_C,
        live_time=spectrum.live_time,
        real_time=spectrum.real_time,
        sample=sample,
        letter=letter,
        timepoint=timepoint,
        n_report_peaks=len(report_peaks) if report_peaks else 0,
    )


def build_report_map(
    report_dir: str,
    asc_keys: List[Tuple[str, str, str]],
) -> Dict[Tuple[str, str, str], List[ReportPeak]]:
    """
    Build mapping from ASC keys to report peaks.
    
    Parameters
    ----------
    report_dir : str
        Directory containing report files
    asc_keys : list of tuple
        (sample, letter, timepoint) tuples
    
    Returns
    -------
    dict
        Mapping from ASC key to list of ReportPeak
    """
    report_map: Dict[Tuple[str, str, str], List[ReportPeak]] = {}
    
    report_dir = os.path.expanduser(report_dir)
    txt_files = sorted(glob(os.path.join(report_dir, "*.txt")))
    txt_files += sorted(glob(os.path.join(report_dir, "*.TXT")))
    
    if not txt_files:
        return report_map
    
    for rpt_file in txt_files:
        peaks = parse_genie_report(rpt_file)
        if not peaks:
            continue
        
        # Try to match to ASC keys based on filename similarity
        basename = os.path.basename(rpt_file).lower()
        
        best_key = None
        best_score = 0
        
        for key in asc_keys:
            sample, letter, timepoint = key
            score = 0
            
            if sample.lower() in basename:
                score += 3
            if normalize_timepoint(timepoint).lower() in normalize_timepoint(basename):
                score += 3
            if letter.lower() in basename:
                score += 2
            
            if score > best_score:
                best_score = score
                best_key = key
        
        if best_key and best_score >= 5:  # Require reasonable match
            if best_key not in report_map:
                report_map[best_key] = []
            report_map[best_key].extend(peaks)
    
    return report_map


def process_batch(config: BatchProcessingConfig) -> BatchResult:
    """
    Process a batch of spectrum files.
    
    Parameters
    ----------
    config : BatchProcessingConfig
        Processing configuration
    
    Returns
    -------
    BatchResult
        Complete batch results
    """
    # Find spectrum files
    spectra_files = sorted(glob(os.path.join(config.spectra_dir, config.spectrum_pattern)))
    
    if not spectra_files:
        raise ValueError(f"No spectra found in {config.spectra_dir} matching {config.spectrum_pattern}")
    
    # Load efficiency model
    efficiency_model = load_efficiency_model(config)
    
    # Build isotope databases
    tier1_isotopes = build_tier1_isotopes(
        config.tier1_elements,
        min_intensity=config.tier1_min_intensity,
    )
    print(f"Tier-1 isotopes ({get_data_source()}): {tier1_isotopes}")
    
    tier1_db, tier2_db, tier3_db = create_matching_databases(
        tier1_isotopes=tier1_isotopes,
        tier2_elements=config.tier2_elements,
        include_tier3=True,
    )
    
    # Build ASC keys for report matching
    asc_keys = []
    for f in spectra_files:
        parsed = parse_asc_filename(f)
        if parsed:
            asc_keys.append(parsed)
    
    # Build report map if report directory provided
    report_map: Dict[Tuple[str, str, str], List[ReportPeak]] = {}
    if config.report_dir:
        report_map = build_report_map(config.report_dir, asc_keys)
        print(f"Loaded report peaks for {len(report_map)} spectrum files")
    
    # Process each file
    results: Dict[str, SpectrumResult] = {}
    
    for filepath in spectra_files:
        # Get report peaks for this file
        parsed = parse_asc_filename(filepath)
        report_peaks = report_map.get(parsed, []) if parsed else []
        
        try:
            result = process_single_spectrum(
                filepath,
                efficiency_model,
                tier1_db,
                tier2_db,
                tier3_db,
                config,
                report_peaks=report_peaks,
            )
            results[filepath] = result
            print(f"Processed {os.path.basename(filepath)}: {len(result.peaks)} peaks")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Discover pairs
    pairs = discover_spectrum_pairs(config.spectra_dir, config.spectrum_pattern)
    
    return BatchResult(
        results=results,
        pairs=pairs,
        config=config,
    )


# ============================================================================
# Output Functions
# ============================================================================

def results_to_dataframe(batch_result: BatchResult):
    """
    Convert batch results to pandas DataFrame.
    
    Returns DataFrame with columns:
    - File, Sample, Letter, Timepoint
    - Observed_E_keV, Amplitude, Channel
    - Isotope, Gamma_E_keV, Intensity, Tier, Candidates
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for DataFrame output")
    
    rows = []
    
    for filepath, result in batch_result.results.items():
        for peak, match in zip(result.peaks, result.matches):
            rows.append({
                'File': os.path.basename(filepath),
                'Sample': result.sample,
                'Letter': result.letter,
                'Timepoint': result.timepoint,
                'Observed_E_keV': peak.energy_keV,
                'Amplitude': peak.amplitude,
                'Channel': peak.channel,
                'Is_Report': peak.is_report,
                'Report_Isotope': peak.report_isotope,
                'Matched_Isotope': match.matched_isotope,
                'Gamma_E_keV': match.matched_energy,
                'Intensity_%': match.matched_intensity,
                'Delta_keV': match.delta_keV,
                'Tier': match.match_tier,
                'Source': match.match_source,
                'Candidates': match.candidates,
            })
    
    return pd.DataFrame(rows)


def save_results_csv(batch_result: BatchResult, output_path: str):
    """Save batch results to CSV file."""
    df = results_to_dataframe(batch_result)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
