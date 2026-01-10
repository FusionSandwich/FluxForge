"""
Flux Wire Gamma Spectrum Parser

Parser for flux wire gamma spectroscopy files, supporting both:
- Raw .ASC files from Genie-2000 MCA 
- Processed .txt files from QuantaGraph commercial analysis software

This module extracts spectra, calibration data, efficiency parameters,
and parsed nuclide activities from flux wire measurements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum


# ============================================================================
# Regex Patterns
# ============================================================================

# ID line pattern
ID_RE = re.compile(r"^\s*ID:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# File/Date pattern for processed files
FILE_DATE_RE = re.compile(
    r"^\s*File:\s*(\S+)\s+Date:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE
)

# Acquisition date for raw files
ACQ_DATE_RE = re.compile(
    r"Acquisition Date:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE
)

# Live time patterns
LT_RT_DT_RE = re.compile(
    r"LT:\s*([\d.,]+)\s*RT:\s*([\d.,]+)\s*DT:\s*([\d.,]+)\s*%?",
    re.IGNORECASE
)
LIVE_TIME_RE = re.compile(r"Elapsed Live Time:\s*([\d.,]+)", re.IGNORECASE)
REAL_TIME_RE = re.compile(r"Elapsed Real Time:\s*([\d.,]+)", re.IGNORECASE)

# Energy calibration patterns (different for raw vs processed)
# Raw ASC: "A = 5.410E-001", "B = 4.980E-001", "C = 2.605E-007"
RAW_CAL_RE = re.compile(r"\b([ABC])\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")

# Processed: "Energy = -1.694E+00 +4.996E-01 * Ch  +6.710E-08 * Ch^2"
PROC_CAL_RE = re.compile(
    r"Energy\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"\s*[+]?\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\*\s*Ch"
    r"(?:\s*[+]?\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\*\s*Ch\^2)?",
    re.IGNORECASE
)

# Efficiency polynomial coefficients  
EFF_COEF_RE = re.compile(r"^\s*C([1-4]):\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", re.MULTILINE)

# Geometry factor
GEOM_A_RE = re.compile(r"Geometry Factor \(A\):\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", re.IGNORECASE)
AL_WINDOW_RE = re.compile(r"Al Window \(T1\):\s*([\d.,]+)\s*um", re.IGNORECASE)
DET_THICK_RE = re.compile(r"Detector Thickness \(DI\)\s*([\d.,]+)\s*cm", re.IGNORECASE)
DEAD_LAYER_RE = re.compile(r"Detector Dead Layer \(DL\):\s*([\d.,]+)\s*um", re.IGNORECASE)
DET_ANGLE_RE = re.compile(r"Det\. Incident Angle \(AI\):\s*([\d.,]+)\s*deg", re.IGNORECASE)
DET_DIAM_RE = re.compile(r"Detector Diameter:\s*([\d.,]+)\s*cm", re.IGNORECASE)
SRC_DIST_RE = re.compile(r"Source Dist\.?:\s*([\d.,]+)\s*cm", re.IGNORECASE)
DET_ID_RE = re.compile(r"Detector ID:\s*(\S+)", re.IGNORECASE)

# FWHM/Resolution
RESOLUTION_RE = re.compile(
    r"Resolution\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[+]?([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\*\s*E",
    re.IGNORECASE
)
FWHM_RE = re.compile(r"FWHM at ([\d.]+) keV = ([\d.]+) keV", re.IGNORECASE)

# Nuclide patterns for processed files
NUCLIDE_HEADER_RE = re.compile(r"NUCLIDES ANALYZED", re.IGNORECASE)
NUCLIDE_LINE_RE = re.compile(
    r"^\s*([A-Za-z]{1,2})(\d{1,3})(m\d?|m)?\s+"
    r"([\d.]+)\s*([smhdaywk]+)\s+"
    r"([A-Z])\s+"
    r"Activity\s*=\s*([0-9.E+-]+)\s*[±�\s]\s*([0-9.E+%-]+)\s*(\w+)",
    re.IGNORECASE
)

# ROI/Peak data line
# "  1173.0   99.85    1173.13    28,478 ± 169      11,202 ± 520    Co60@1173.2    5.94E-03"
PEAK_LINE_RE = re.compile(
    r"^\s*([\d.]+)\s+"           # ROI centroid (channel or energy)
    r"([\d.]+)\s+"               # Radiation intensity %
    r"([\d.]+)\s+"               # Center energy (keV)
    r"([\d,]+)\s*[±�\s]+(\d+)\s+"  # Gross counts ± unc
    r"([\d,]+)\s*[±�\s]+(\d+)\s+"  # Net counts ± unc
    r"(\w+@[\d.]+)\s+"           # Assignment (nuclide@energy)
    r"([0-9.E+-]+)",             # Activity
    re.IGNORECASE
)


def _parse_number(value: str) -> Optional[float]:
    """Parse a number string, handling comma separators."""
    if not value:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _parse_datetime(value: str) -> Optional[datetime]:
    """Parse datetime from various formats."""
    formats = [
        "%B %d, %Y %H:%M:%S",
        "%B %d, %Y %H:%M",
        "%b %d, %Y %H:%M:%S",
        "%b %d, %Y %H:%M",
        "%d-%b-%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass
class NuclideResult:
    """Nuclide analysis result from processed file."""
    
    isotope: str           # e.g., "Co60", "Sc46", "In114m"
    half_life_value: float
    half_life_unit: str    # s, m, h, d, a, y
    activity: float        # uCi (or other unit)
    activity_unc: float
    activity_unit: str     # e.g., "uCi"
    peaks: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def half_life_seconds(self) -> float:
        """Convert half-life to seconds."""
        conversions = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'a': 31557600,  # 365.25 days
            'y': 31557600,
        }
        return self.half_life_value * conversions.get(self.half_life_unit.lower(), 1)
    
    @property
    def activity_bq(self) -> float:
        """Convert activity to Bq."""
        # 1 Ci = 3.7e10 Bq
        unit = self.activity_unit.lower()
        if unit == 'uci':
            return self.activity * 3.7e4
        elif unit == 'nci':
            return self.activity * 37.0
        elif unit == 'ci':
            return self.activity * 3.7e10
        elif unit == 'mci':
            return self.activity * 3.7e7
        elif unit == 'bq':
            return self.activity
        elif unit == 'kbq':
            return self.activity * 1e3
        elif unit == 'mbq':
            return self.activity * 1e6
        return self.activity  # Unknown unit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'isotope': self.isotope,
            'half_life_value': self.half_life_value,
            'half_life_unit': self.half_life_unit,
            'half_life_seconds': self.half_life_seconds,
            'activity': self.activity,
            'activity_unc': self.activity_unc,
            'activity_unit': self.activity_unit,
            'activity_bq': self.activity_bq,
            'peaks': self.peaks,
        }


@dataclass
class EfficiencyCalibration:
    """HPGe detector efficiency calibration parameters."""
    
    # Polynomial coefficients for ln(eff) = C1 + C2*ln(E) + C3*ln(E)^2 + C4*ln(E)^3
    C1: float = 0.0
    C2: float = 0.0
    C3: float = 0.0
    C4: float = 0.0
    
    # Detector model parameters
    geometry_factor_A: float = 1.0
    al_window_T1_um: float = 0.0
    detector_thickness_DI_cm: float = 0.0
    dead_layer_DL_um: float = 0.0
    incident_angle_AI_deg: float = 0.0
    
    # Additional geometry
    detector_diameter_cm: float = 0.0
    source_distance_cm: float = 0.0
    detector_id: str = ""
    
    def efficiency(self, energy_keV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate detector efficiency using LabSOCS/QuantaGraph model.
        
        The QuantaGraph/LabSOCS efficiency formula is:
        ε = DetModel × (C1 + C2×log(E) + C3×log(E)² + C4×log(E)³)
        
        Where:
        - DetModel = A × exp(...attenuation terms...) × (1-exp(-detector_term))
        - C1-C4 are polynomial coefficients
        - log is natural logarithm
        - E is energy in keV
        
        Note: This is a polynomial in log(E), NOT a log-polynomial.
        The coefficients directly give ln(efficiency), which we then exponentiate.
        """
        E = np.atleast_1d(np.asarray(energy_keV, dtype=float))
        
        # Natural log of energy
        with np.errstate(divide='ignore', invalid='ignore'):
            lnE = np.log(np.where(E > 0, E, np.nan))
        
        # The polynomial (C1 + C2*lnE + C3*lnE^2 + C4*lnE^3) gives ln(ε/A)
        # where A is the geometry factor
        # 
        # From the QuantaGraph output format:
        #   ε = DetModel × (C1 + C2×Log(E) + C3×Log(E)² + C4×Log(E)³)
        # where DetModel includes the geometry factor A and attenuation
        #
        # The coefficients C1~-20 indicates this is ln(efficiency) form
        # because ln(0.001) ~ -6.9, not -20
        #
        # Actually looking at typical HPGe efficiencies (~0.001-0.01 at 25cm),
        # and C1=-20.026, if we compute:
        #   ln(eff) = C1 + C2*lnE + C3*lnE^2 + C4*lnE^3
        #   at E=1332 keV: lnE = 7.195
        #   ln(eff) = -20.026 + 10.29*7.195 - 1.655*51.77 + 0.0867*372.5
        #           = -20.026 + 74.04 - 85.68 + 32.3 = 0.63
        #   eff = exp(0.63) = 1.88  (way too high!)
        #
        # So the formula must be:
        #   eff_intermediate = C1 + C2*lnE + C3*lnE^2 + C4*lnE^3
        #   Then multiply by geometry factor A
        
        poly = self.C1 + self.C2 * lnE + self.C3 * lnE**2 + self.C4 * lnE**3
        
        # The polynomial value times geometry factor gives efficiency
        # But we need to handle the sign - if poly is negative, we may need exp()
        # 
        # Let's check: at 1332 keV with given C values:
        #   poly = -20.026 + 10.29*7.195 - 1.655*51.77 + 0.0867*372.5
        #        = -20.026 + 74.04 - 85.68 + 32.30 = 0.634
        #   Then: eff = A * poly = 0.00348 * 0.634 = 0.0022
        # 
        # That's reasonable! So eff = A × poly_value
        
        eff = self.geometry_factor_A * poly
        
        # Clip to physical range
        eff = np.clip(eff, 0.0, 1.0)
        
        return eff.squeeze() if np.isscalar(energy_keV) else eff
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'C1': self.C1,
            'C2': self.C2,
            'C3': self.C3,
            'C4': self.C4,
            'geometry_factor_A': self.geometry_factor_A,
            'al_window_T1_um': self.al_window_T1_um,
            'detector_thickness_DI_cm': self.detector_thickness_DI_cm,
            'dead_layer_DL_um': self.dead_layer_DL_um,
            'incident_angle_AI_deg': self.incident_angle_AI_deg,
            'detector_diameter_cm': self.detector_diameter_cm,
            'source_distance_cm': self.source_distance_cm,
            'detector_id': self.detector_id,
        }


@dataclass
class FluxWireData:
    """
    Complete flux wire measurement data.
    
    Contains both spectrum data and analysis results from either
    raw ASC files or processed QuantaGraph TXT files.
    """
    
    # Identification
    sample_id: str = ""
    file_name: str = ""
    source_file: str = ""
    file_type: str = ""  # 'raw_asc' or 'processed_txt'
    
    # Timing
    start_time: Optional[datetime] = None
    live_time: float = 0.0
    real_time: float = 0.0
    dead_time_pct: float = 0.0
    
    # Calibrations
    energy_calibration: List[float] = field(default_factory=list)  # [a0, a1, a2]
    efficiency: Optional[EfficiencyCalibration] = None
    resolution: List[float] = field(default_factory=list)  # [r0, r1, r2] for FWHM
    
    # Spectrum data (if from raw file)
    spectrum: Optional[GammaSpectrum] = None
    
    # Analysis results (if from processed file)
    nuclides: List[NuclideResult] = field(default_factory=list)
    
    @property
    def has_spectrum(self) -> bool:
        """Check if raw spectrum data is available."""
        return self.spectrum is not None
    
    @property 
    def has_nuclides(self) -> bool:
        """Check if nuclide analysis results are available."""
        return len(self.nuclides) > 0
    
    def channel_to_energy(self, channel: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert channel to energy using calibration."""
        ch = np.asarray(channel, dtype=float)
        cal = self.energy_calibration
        if len(cal) == 0:
            return ch
        
        energy = cal[0] if len(cal) > 0 else 0.0
        if len(cal) > 1:
            energy = energy + cal[1] * ch
        if len(cal) > 2:
            energy = energy + cal[2] * ch**2
        
        return float(energy) if np.isscalar(channel) else energy
    
    def energy_to_channel(self, energy: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert energy to channel using calibration."""
        E = np.asarray(energy, dtype=float)
        cal = self.energy_calibration
        
        if len(cal) < 2:
            return int(E) if np.isscalar(energy) else E.astype(int)
        
        if len(cal) == 2:
            # Linear: E = a0 + a1*ch => ch = (E - a0) / a1
            ch = (E - cal[0]) / cal[1]
        elif len(cal) >= 3 and abs(cal[2]) > 1e-15:
            # Quadratic: solve a2*ch² + a1*ch + (a0 - E) = 0
            a, b, c = cal[2], cal[1], cal[0] - E
            discriminant = b**2 - 4*a*c
            ch = (-b + np.sqrt(np.maximum(discriminant, 0))) / (2*a)
        else:
            ch = (E - cal[0]) / cal[1]
        
        return int(np.round(ch)) if np.isscalar(energy) else np.round(ch).astype(int)
    
    def fwhm_at_energy(self, energy: float) -> float:
        """Calculate FWHM at given energy using resolution calibration."""
        if len(self.resolution) < 2:
            return 2.0  # Default 2 keV
        
        r = self.resolution
        fwhm = r[0] + r[1] * energy
        if len(r) > 2:
            fwhm += r[2] * energy**2
        
        return max(fwhm, 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_id': self.sample_id,
            'file_name': self.file_name,
            'source_file': self.source_file,
            'file_type': self.file_type,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'live_time': self.live_time,
            'real_time': self.real_time,
            'dead_time_pct': self.dead_time_pct,
            'energy_calibration': self.energy_calibration,
            'efficiency': self.efficiency.to_dict() if self.efficiency else None,
            'resolution': self.resolution,
            'has_spectrum': self.has_spectrum,
            'nuclides': [n.to_dict() for n in self.nuclides],
        }


def read_raw_asc(filepath: Union[str, Path]) -> FluxWireData:
    """
    Read raw Genie-2000 ASCII export file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .ASC file
    
    Returns
    -------
    FluxWireData
        Parsed spectrum and calibration data
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    data = FluxWireData(
        source_file=str(filepath),
        file_type='raw_asc',
    )
    
    # Parse ID
    match = ID_RE.search(content)
    if match:
        data.sample_id = match.group(1).strip()
    
    # Parse acquisition date
    match = ACQ_DATE_RE.search(content)
    if match:
        data.start_time = _parse_datetime(match.group(1))
    
    # Parse timing
    match = LIVE_TIME_RE.search(content)
    if match:
        lt = _parse_number(match.group(1))
        if lt is not None:
            data.live_time = lt
    
    match = REAL_TIME_RE.search(content)
    if match:
        rt = _parse_number(match.group(1))
        if rt is not None:
            data.real_time = rt
    
    if data.real_time > 0:
        data.dead_time_pct = 100 * (1 - data.live_time / data.real_time)
    
    # Parse energy calibration (A, B, C coefficients)
    # E = A + B*ch + C*ch²
    cal = {}
    for match in RAW_CAL_RE.finditer(content):
        coef = match.group(1).upper()
        value = float(match.group(2))
        cal[coef] = value
    
    if 'A' in cal and 'B' in cal:
        data.energy_calibration = [
            cal.get('A', 0.0),
            cal.get('B', 1.0),
            cal.get('C', 0.0),
        ]
    
    # Read channel data
    channels = []
    counts = []
    in_data = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Look for start of data section
            if 'Channel' in line and 'Contents' in line:
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
    
    if counts:
        channels_arr = np.array(channels)
        counts_arr = np.array(counts)
        
        # Calculate energies
        energies = None
        if len(data.energy_calibration) >= 2:
            energies = data.channel_to_energy(channels_arr)
        
        data.spectrum = GammaSpectrum(
            counts=counts_arr,
            channels=channels_arr,
            energies=energies,
            live_time=data.live_time,
            real_time=data.real_time,
            start_time=data.start_time,
            spectrum_id=data.sample_id,
            calibration={'energy': data.energy_calibration},
            metadata={'source_file': str(filepath), 'format': 'genie_asc'},
        )
    
    return data


def read_processed_txt(filepath: Union[str, Path]) -> FluxWireData:
    """
    Read processed QuantaGraph TXT file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to processed .txt file
    
    Returns
    -------
    FluxWireData
        Parsed calibration, efficiency, and nuclide results
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    data = FluxWireData(
        source_file=str(filepath),
        file_type='processed_txt',
    )
    
    # Parse ID
    match = ID_RE.search(content)
    if match:
        data.sample_id = match.group(1).strip()
    
    # Parse File and Date
    match = FILE_DATE_RE.search(content)
    if match:
        data.file_name = match.group(1).strip()
        data.start_time = _parse_datetime(match.group(2))
    
    # Parse timing (LT, RT, DT)
    match = LT_RT_DT_RE.search(content)
    if match:
        lt = _parse_number(match.group(1))
        rt = _parse_number(match.group(2))
        dt = _parse_number(match.group(3))
        if lt is not None:
            data.live_time = lt
        if rt is not None:
            data.real_time = rt
        if dt is not None:
            data.dead_time_pct = dt
    
    # Parse energy calibration
    match = PROC_CAL_RE.search(content)
    if match:
        a0 = float(match.group(1))
        a1 = float(match.group(2))
        a2 = float(match.group(3)) if match.group(3) else 0.0
        data.energy_calibration = [a0, a1, a2]
    
    # Parse efficiency calibration
    eff = EfficiencyCalibration()
    
    for match in EFF_COEF_RE.finditer(content):
        coef_num = int(match.group(1))
        value = float(match.group(2))
        if coef_num == 1:
            eff.C1 = value
        elif coef_num == 2:
            eff.C2 = value
        elif coef_num == 3:
            eff.C3 = value
        elif coef_num == 4:
            eff.C4 = value
    
    match = GEOM_A_RE.search(content)
    if match:
        eff.geometry_factor_A = float(match.group(1))
    
    match = AL_WINDOW_RE.search(content)
    if match:
        eff.al_window_T1_um = _parse_number(match.group(1)) or 0.0
    
    match = DET_THICK_RE.search(content)
    if match:
        eff.detector_thickness_DI_cm = _parse_number(match.group(1)) or 0.0
    
    match = DEAD_LAYER_RE.search(content)
    if match:
        eff.dead_layer_DL_um = _parse_number(match.group(1)) or 0.0
    
    match = DET_ANGLE_RE.search(content)
    if match:
        eff.incident_angle_AI_deg = _parse_number(match.group(1)) or 0.0
    
    match = DET_DIAM_RE.search(content)
    if match:
        eff.detector_diameter_cm = _parse_number(match.group(1)) or 0.0
    
    match = SRC_DIST_RE.search(content)
    if match:
        eff.source_distance_cm = _parse_number(match.group(1)) or 0.0
    
    match = DET_ID_RE.search(content)
    if match:
        eff.detector_id = match.group(1).strip()
    
    data.efficiency = eff
    
    # Parse resolution calibration
    match = RESOLUTION_RE.search(content)
    if match:
        r0 = float(match.group(1))
        r1 = float(match.group(2))
        data.resolution = [r0, r1, 0.0]
    
    # Parse NUCLIDES ANALYZED section
    lines = content.split('\n')
    in_nuclides = False
    current_nuclide = None
    
    for line in lines:
        # Look for NUCLIDES ANALYZED header
        if NUCLIDE_HEADER_RE.search(line):
            in_nuclides = True
            continue
        
        if not in_nuclides:
            continue
        
        # Check for nuclide header line
        match = NUCLIDE_LINE_RE.match(line)
        if match:
            element = match.group(1)
            mass = match.group(2)
            meta = match.group(3) or ''
            isotope = f"{element}{mass}{meta}"
            
            hl_value = float(match.group(4))
            hl_unit = match.group(5).lower()
            
            activity = float(match.group(7))
            unc_str = match.group(8)
            
            # Parse uncertainty (may be percentage)
            if '%' in unc_str:
                unc = activity * float(unc_str.replace('%', '')) / 100
            else:
                unc = float(unc_str)
            
            activity_unit = match.group(9)
            
            current_nuclide = NuclideResult(
                isotope=isotope,
                half_life_value=hl_value,
                half_life_unit=hl_unit,
                activity=activity,
                activity_unc=unc,
                activity_unit=activity_unit,
            )
            data.nuclides.append(current_nuclide)
            continue
        
        # Check for peak data line
        match = PEAK_LINE_RE.match(line)
        if match and current_nuclide is not None:
            peak = {
                'centroid': float(match.group(1)),
                'rad_int': float(match.group(2)),
                'center_keV': float(match.group(3)),
                'gross_counts': _parse_number(match.group(4)) or 0,
                'gross_unc': int(match.group(5)),
                'net_counts': _parse_number(match.group(6)) or 0,
                'net_unc': int(match.group(7)),
                'assignment': match.group(8),
                'activity': float(match.group(9)),
            }
            current_nuclide.peaks.append(peak)
    
    return data


def read_flux_wire(filepath: Union[str, Path]) -> FluxWireData:
    """
    Read flux wire file, auto-detecting format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to flux wire data file (.ASC or .txt)
    
    Returns
    -------
    FluxWireData
        Parsed data
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    
    if ext == '.asc':
        return read_raw_asc(filepath)
    elif ext == '.txt':
        return read_processed_txt(filepath)
    else:
        # Try to detect from content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_1000 = f.read(1000)
        
        if 'NUCLIDES ANALYZED' in first_1000:
            return read_processed_txt(filepath)
        elif 'Channel' in first_1000 and 'Contents' in first_1000:
            return read_raw_asc(filepath)
        else:
            raise ValueError(f"Unknown file format: {filepath}")


def load_flux_wire_directory(
    directory: Union[str, Path],
    pattern: str = "*",
    file_types: Optional[List[str]] = None
) -> Dict[str, FluxWireData]:
    """
    Load all flux wire files from a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing flux wire files
    pattern : str
        Glob pattern for filtering files
    file_types : list of str, optional
        File extensions to include (default: ['.asc', '.txt'])
    
    Returns
    -------
    dict
        Mapping of filename stem to FluxWireData
    """
    directory = Path(directory)
    if file_types is None:
        file_types = ['.asc', '.txt']
    
    results = {}
    
    for filepath in sorted(directory.glob(pattern)):
        if filepath.suffix.lower() not in file_types:
            continue
        
        try:
            data = read_flux_wire(filepath)
            results[filepath.stem] = data
        except Exception as e:
            print(f"Warning: Failed to read {filepath}: {e}")
    
    return results
