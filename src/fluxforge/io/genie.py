"""
Genie-2000 ASCII Spectrum Parser

Parses Genie-2000 ASCII export files (.ASC, .TXT) which have a different
format than standard SPE files. Includes energy calibration extraction
and efficiency model application.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum


# ============================================================================
# Regex Patterns for Genie-2000 Format
# ============================================================================

ENERGY_CAL_RE = re.compile(r"\b([ABC])\s*=\s*([+-]?\d+\.?\d*E[+-]?\d+)", re.IGNORECASE)
LIVE_TIME_RE = re.compile(r"Elapsed Live Time:\s*([\d.]+)", re.IGNORECASE)
REAL_TIME_RE = re.compile(r"Elapsed Real Time:\s*([\d.]+)", re.IGNORECASE)
START_TIME_RE = re.compile(r"Acquisition Started:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
SAMPLE_ID_RE = re.compile(r"Sample(?:\s+ID)?:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
DETECTOR_RE = re.compile(r"Detector(?:\s+Name)?:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# Filename pattern for RAFM samples: sample-C_timepoint.ASC or sample-N_timepoint.ASC
ASC_NAME_RE = re.compile(r"(.+?)-([CN])_(.+)\.ASC$", re.IGNORECASE)


def parse_genie_header(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse Genie-2000 ASCII file header for calibration and metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Path to Genie-2000 ASCII file
    
    Returns
    -------
    dict
        Header information including:
        - calibration: {'A': float, 'B': float, 'C': float}
        - live_time: float (seconds)
        - real_time: float (seconds)
        - start_time: datetime or None
        - sample_id: str
        - detector_id: str
    """
    header = {
        'calibration': {},
        'live_time': 0.0,
        'real_time': 0.0,
        'start_time': None,
        'sample_id': '',
        'detector_id': '',
    }
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract calibration coefficients A, B, C
    for match in ENERGY_CAL_RE.finditer(content):
        coef = match.group(1).upper()
        value = float(match.group(2))
        header['calibration'][coef] = value
    
    # Extract live time
    match = LIVE_TIME_RE.search(content)
    if match:
        header['live_time'] = float(match.group(1))
    
    # Extract real time
    match = REAL_TIME_RE.search(content)
    if match:
        header['real_time'] = float(match.group(1))
    
    # Extract start time
    match = START_TIME_RE.search(content)
    if match:
        time_str = match.group(1).strip()
        for fmt in [
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y %H:%M:%S %p',
        ]:
            try:
                header['start_time'] = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue
    
    # Extract sample ID
    match = SAMPLE_ID_RE.search(content)
    if match:
        header['sample_id'] = match.group(1).strip()
    
    # Extract detector ID
    match = DETECTOR_RE.search(content)
    if match:
        header['detector_id'] = match.group(1).strip()
    
    return header


def read_genie_spectrum(filepath: Union[str, Path]) -> GammaSpectrum:
    """
    Read Genie-2000 ASCII export file.
    
    The file format has:
    - Header lines with calibration coefficients (A, B, C)
    - Timing information (live time, real time)
    - Channel/Counts data pairs after "Channel" header line
    
    Parameters
    ----------
    filepath : str or Path
        Path to Genie-2000 ASCII file (.ASC or .TXT)
    
    Returns
    -------
    GammaSpectrum
        Parsed spectrum with calibration applied
    
    Examples
    --------
    >>> spectrum = read_genie_spectrum("sample-C_300sEOI.ASC")
    >>> print(f"Live time: {spectrum.live_time} s")
    >>> print(f"Channels: {len(spectrum.counts)}")
    """
    filepath = Path(filepath)
    
    # Parse header
    header = parse_genie_header(filepath)
    
    # Read channel data
    channels = []
    counts = []
    in_data = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Look for start of data section
            if line_stripped.lower().startswith('channel'):
                in_data = True
                continue
            
            if not in_data:
                continue
            
            # Parse channel/count pairs
            parts = line_stripped.split()
            if len(parts) >= 2:
                try:
                    ch = int(parts[0])
                    cnt = int(parts[1])
                    channels.append(ch)
                    counts.append(cnt)
                except ValueError:
                    continue
    
    if not counts:
        raise ValueError(f"No channel data found in {filepath}")
    
    channels = np.array(channels)
    counts = np.array(counts)
    
    # Build calibration coefficients list [A, B, C]
    cal = header['calibration']
    if 'A' in cal and 'B' in cal:
        # E = A + B*ch + C*ch²
        coefficients = [
            cal.get('A', 0.0),
            cal.get('B', 1.0),
            cal.get('C', 0.0),
        ]
    else:
        coefficients = [0.0, 1.0]  # Default: channel = keV
    
    # Calculate energies
    energies = np.zeros_like(channels, dtype=float)
    for i, coef in enumerate(coefficients):
        energies += coef * (channels.astype(float) ** i)
    
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        energies=energies,
        live_time=header['live_time'],
        real_time=header['real_time'],
        start_time=header['start_time'],
        spectrum_id=header['sample_id'] or filepath.stem,
        detector_id=header['detector_id'],
        calibration={'energy': coefficients},
        metadata={
            'format': 'genie2000',
            'source_file': str(filepath),
        },
    )


def parse_asc_filename(filepath: Union[str, Path]) -> Optional[Tuple[str, str, str]]:
    """
    Parse RAFM-style ASC filename to extract sample, letter, and timepoint.
    
    Format: sample-C_timepoint.ASC or sample-N_timepoint.ASC
    
    Parameters
    ----------
    filepath : str or Path
        Path to ASC file
    
    Returns
    -------
    tuple or None
        (sample, letter, timepoint) or None if pattern doesn't match
    
    Examples
    --------
    >>> parse_asc_filename("RAFM3-C_300sEOI.ASC")
    ('RAFM3', 'C', '300sEOI')
    >>> parse_asc_filename("RAFM3-N_15dEOI.ASC")
    ('RAFM3', 'N', '15dEOI')
    """
    basename = os.path.basename(str(filepath))
    match = ASC_NAME_RE.match(basename)
    if match:
        sample, letter, timepoint = match.groups()
        return sample, letter.upper(), timepoint
    return None


def normalize_timepoint(timepoint: str) -> str:
    """
    Normalize timepoint string for matching.
    
    Removes spaces, dashes, underscores for consistent comparison.
    
    Parameters
    ----------
    timepoint : str
        Timepoint string like "300s EOI" or "15d-EOI"
    
    Returns
    -------
    str
        Normalized string like "300sEOI" or "15dEOI"
    """
    t = str(timepoint).strip()
    t = t.replace(' ', '')
    t = t.replace('-', '')
    t = t.replace('_', '')
    return t


# ============================================================================
# Genie/LabSOCS Report File Parsing
# ============================================================================

# Patterns for report file parsing
ID_LINE_RE = re.compile(r"^\s*ID:\s*(.+?)\s*$", re.IGNORECASE)
NUCLIDES_HEADER_RE = re.compile(r"^\s*NUCLIDES ANALYZED\s*$", re.IGNORECASE)
NUCLIDE_LINE_RE = re.compile(
    r"^\s*([A-Za-z]{1,3}\s*\d{1,3}(?:m\d*|m)?)\s+",
    re.IGNORECASE,
)
CENTROID_HEADER_RE = re.compile(r"^\s*CENTROID", re.IGNORECASE)
ROI_HEADER_RE = re.compile(r"^\s*ROI\s+", re.IGNORECASE)
CENTROID_ROW_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+")


@dataclass
class ReportPeak:
    """Peak identified in a Genie/LabSOCS analysis report."""
    
    energy: float  # keV
    isotope: str   # Nuclide label
    net_area: Optional[float] = None
    uncertainty: Optional[float] = None
    report_file: str = ""


def parse_genie_report(filepath: Union[str, Path]) -> List[ReportPeak]:
    """
    Parse Genie/LabSOCS analysis report TXT file.
    
    Extracts pre-identified peaks with their isotope assignments.
    
    Parameters
    ----------
    filepath : str or Path
        Path to report TXT file
    
    Returns
    -------
    list of ReportPeak
        Identified peaks from the report
    """
    filepath = Path(filepath)
    peaks = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        return peaks
    
    # Find NUCLIDES ANALYZED section
    start_idx = None
    for i, line in enumerate(lines):
        if NUCLIDES_HEADER_RE.match(line):
            start_idx = i
            break
    
    if start_idx is None:
        return peaks
    
    current_isotope = None
    in_block = False
    
    for line in lines[start_idx:]:
        line_stripped = line.strip()
        
        if not line_stripped:
            in_block = False
            continue
        
        # Check for nuclide header line
        match = NUCLIDE_LINE_RE.match(line)
        if match:
            current_isotope = match.group(1).replace(' ', '')
            in_block = True
            continue
        
        if not in_block or current_isotope is None:
            continue
        
        # Skip table headers
        if ROI_HEADER_RE.match(line) or CENTROID_HEADER_RE.match(line):
            continue
        
        # Parse centroid energy from data row
        match = CENTROID_ROW_RE.match(line)
        if match:
            try:
                energy = float(match.group(1))
                peaks.append(ReportPeak(
                    energy=energy,
                    isotope=current_isotope,
                    report_file=str(filepath),
                ))
            except ValueError:
                continue
    
    return peaks


def extract_report_id(filepath: Union[str, Path], max_lines: int = 80) -> str:
    """
    Extract the ID line from a report file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to report file
    max_lines : int
        Maximum lines to search
    
    Returns
    -------
    str
        ID string or empty if not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                match = ID_LINE_RE.match(line)
                if match:
                    return match.group(1).strip()
    except OSError:
        pass
    return ""


# ============================================================================
# Batch Processing Utilities
# ============================================================================

@dataclass
class SpectrumPair:
    """Paired C and N spectra for the same sample and timepoint."""
    
    sample: str
    timepoint: str
    c_spectrum: Optional[GammaSpectrum] = None
    n_spectrum: Optional[GammaSpectrum] = None
    c_path: str = ""
    n_path: str = ""


def discover_spectrum_pairs(
    spectra_dir: Union[str, Path],
    pattern: str = "*.ASC"
) -> List[SpectrumPair]:
    """
    Discover C/N spectrum pairs in a directory.
    
    Parameters
    ----------
    spectra_dir : str or Path
        Directory containing spectrum files
    pattern : str
        Glob pattern for spectrum files
    
    Returns
    -------
    list of SpectrumPair
        Matched C/N pairs with their file paths
    """
    from glob import glob
    
    spectra_dir = Path(spectra_dir)
    files = sorted(glob(str(spectra_dir / pattern)))
    
    # Group by (sample, timepoint)
    groups: Dict[Tuple[str, str], Dict[str, str]] = {}
    
    for filepath in files:
        parsed = parse_asc_filename(filepath)
        if parsed is None:
            continue
        
        sample, letter, timepoint = parsed
        key = (sample, normalize_timepoint(timepoint))
        
        if key not in groups:
            groups[key] = {}
        groups[key][letter] = filepath
    
    # Build pairs
    pairs = []
    for (sample, timepoint), paths in groups.items():
        if 'C' in paths and 'N' in paths:
            pairs.append(SpectrumPair(
                sample=sample,
                timepoint=timepoint,
                c_path=paths['C'],
                n_path=paths['N'],
            ))
    
    return pairs


def load_spectrum_pair(pair: SpectrumPair) -> SpectrumPair:
    """
    Load both spectra for a pair.
    
    Parameters
    ----------
    pair : SpectrumPair
        Pair with paths set
    
    Returns
    -------
    SpectrumPair
        Same pair with spectra loaded
    """
    if pair.c_path:
        pair.c_spectrum = read_genie_spectrum(pair.c_path)
    if pair.n_path:
        pair.n_spectrum = read_genie_spectrum(pair.n_path)
    return pair
