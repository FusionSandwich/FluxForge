"""
N42 (ANSI N42.42) Gamma Spectrum File Reader

Reads N42.42 XML format files commonly used by:
- RIID (Radioisotope Identification Devices)
- Portable gamma spectrometers
- Homeland security instruments

The N42 format is defined by ANSI N42.42:
"American National Standard Data Format for Radiation Detectors 
Used for Homeland Security"

References:
    ANSI N42.42-2012: Data Format Standard
    IEC 62755: Portable instruments for gamma-ray spectrometry
"""

from __future__ import annotations

import re
import base64
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np


# N42 namespace definitions
N42_NAMESPACES = {
    "n42": "http://physics.nist.gov/N42/2011/N42",
    "n42_2006": "http://physics.nist.gov/Divisions/Div846/Gp4/ANSIN4242/2005/ANSIN4242",
}


@dataclass
class N42Measurement:
    """
    Container for a single N42 measurement/spectrum.
    
    Attributes
    ----------
    spectrum_id : str
        Unique identifier for this spectrum
    counts : np.ndarray
        Channel counts array
    live_time : float
        Live time in seconds
    real_time : float
        Real time in seconds
    start_time : datetime
        Measurement start time
    energy_calibration : Tuple[float, ...]
        Polynomial coefficients (offset, gain, quadratic, ...)
    detector_type : str
        Type of detector (NaI, HPGe, CZT, etc.)
    detector_description : str
        Detector description
    dose_rate : float
        Measured dose rate (if available)
    dose_rate_units : str
        Units for dose rate
    nuclides_identified : List[str]
        List of identified nuclides (if RIID)
    metadata : Dict[str, Any]
        Additional metadata
    """
    spectrum_id: str = ""
    counts: np.ndarray = field(default_factory=lambda: np.array([]))
    live_time: float = 0.0
    real_time: float = 0.0
    start_time: Optional[datetime] = None
    energy_calibration: Tuple[float, ...] = (0.0, 1.0)
    detector_type: str = ""
    detector_description: str = ""
    dose_rate: float = 0.0
    dose_rate_units: str = ""
    nuclides_identified: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.counts)
    
    @property
    def dead_time_fraction(self) -> float:
        """Dead time as fraction."""
        if self.real_time > 0:
            return 1.0 - (self.live_time / self.real_time)
        return 0.0
    
    @property
    def total_counts(self) -> float:
        """Total counts in spectrum."""
        return float(np.sum(self.counts))
    
    @property
    def count_rate(self) -> float:
        """Count rate (cps)."""
        if self.live_time > 0:
            return self.total_counts / self.live_time
        return 0.0
    
    def energy_axis(self) -> np.ndarray:
        """Calculate energy axis from calibration."""
        channels = np.arange(self.n_channels)
        energy = np.zeros_like(channels, dtype=float)
        for i, coef in enumerate(self.energy_calibration):
            energy += coef * (channels ** i)
        return energy
    
    def to_gamma_spectrum(self):
        """Convert to FluxForge GammaSpectrum object."""
        from fluxforge.io.spe import GammaSpectrum
        
        return GammaSpectrum(
            counts=self.counts,
            live_time=self.live_time,
            real_time=self.real_time,
            start_time=self.start_time,
            energy_calibration=self.energy_calibration,
            description=self.detector_description,
        )


@dataclass
class N42Document:
    """
    Container for a complete N42 document.
    
    N42 files can contain multiple measurements, detector information,
    and analysis results.
    """
    measurements: List[N42Measurement] = field(default_factory=list)
    instrument_type: str = ""
    instrument_model: str = ""
    instrument_id: str = ""
    manufacturer: str = ""
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    
    @property
    def n_spectra(self) -> int:
        """Number of spectra in document."""
        return len(self.measurements)
    
    def get_spectrum(self, index: int = 0) -> N42Measurement:
        """Get spectrum by index."""
        return self.measurements[index]


def _find_element(parent: ET.Element, tag: str, namespaces: Dict[str, str]) -> Optional[ET.Element]:
    """Find element with namespace handling."""
    # Try with each namespace
    for prefix, uri in namespaces.items():
        elem = parent.find(f"{{{uri}}}{tag}")
        if elem is not None:
            return elem
    # Try without namespace
    elem = parent.find(tag)
    if elem is not None:
        return elem
    # Try with local-name matching
    for child in parent:
        local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local_name == tag:
            return child
    return None


def _find_all_elements(parent: ET.Element, tag: str, namespaces: Dict[str, str]) -> List[ET.Element]:
    """Find all elements with namespace handling."""
    results = []
    for prefix, uri in namespaces.items():
        results.extend(parent.findall(f"{{{uri}}}{tag}"))
    # Also try without namespace
    results.extend(parent.findall(tag))
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for elem in results:
        elem_id = id(elem)
        if elem_id not in seen:
            seen.add(elem_id)
            unique.append(elem)
    return unique


def _get_text(parent: ET.Element, tag: str, namespaces: Dict[str, str], default: str = "") -> str:
    """Get text content of child element."""
    elem = _find_element(parent, tag, namespaces)
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def _parse_channel_data(data_elem: ET.Element, namespaces: Dict[str, str]) -> np.ndarray:
    """
    Parse channel data from N42 element.
    
    N42 supports multiple encodings:
    - Space-separated integers
    - Base64 encoded binary
    - Compressed formats
    """
    if data_elem is None or data_elem.text is None:
        return np.array([])
    
    # Check for compression attribute
    compression = data_elem.get("compressionCode", "")
    
    # Get raw text
    text = data_elem.text.strip()
    
    if not text:
        return np.array([])
    
    # Check if base64 encoded
    # Base64 typically has no spaces and specific character set
    is_base64 = not any(c.isspace() for c in text[:100]) and len(text) > 100
    
    if is_base64 or compression.lower() in ["none", "base64"]:
        try:
            # Try base64 decode
            binary_data = base64.b64decode(text)
            # Assume 4-byte unsigned integers (common for gamma spectra)
            n_values = len(binary_data) // 4
            counts = struct.unpack(f"<{n_values}I", binary_data[:n_values*4])
            return np.array(counts, dtype=np.float64)
        except Exception:
            pass
    
    # Try space/comma separated integers
    try:
        # Split on whitespace or commas
        values = re.split(r'[\s,]+', text)
        counts = [float(v) for v in values if v]
        return np.array(counts, dtype=np.float64)
    except ValueError:
        pass
    
    return np.array([])


def _parse_calibration(cal_elem: ET.Element, namespaces: Dict[str, str]) -> Tuple[float, ...]:
    """Parse energy calibration from N42 element."""
    if cal_elem is None:
        return (0.0, 1.0)
    
    # Look for polynomial coefficients
    coef_elem = _find_element(cal_elem, "CoefficientValues", namespaces)
    if coef_elem is None:
        coef_elem = _find_element(cal_elem, "Coefficients", namespaces)
    
    if coef_elem is not None and coef_elem.text:
        try:
            values = re.split(r'[\s,]+', coef_elem.text.strip())
            coeffs = [float(v) for v in values if v]
            return tuple(coeffs)
        except ValueError:
            pass
    
    # Try individual coefficient elements
    coeffs = []
    for i in range(5):  # Up to 5th order polynomial
        coef = _get_text(cal_elem, f"Coefficient{i}", namespaces)
        if coef:
            try:
                coeffs.append(float(coef))
            except ValueError:
                break
        else:
            break
    
    if coeffs:
        return tuple(coeffs)
    
    return (0.0, 1.0)


def _parse_measurement(meas_elem: ET.Element, namespaces: Dict[str, str]) -> N42Measurement:
    """Parse a single measurement element."""
    measurement = N42Measurement()
    
    # Get measurement ID
    measurement.spectrum_id = meas_elem.get("id", "")
    
    # Find spectrum element
    spectrum_elem = _find_element(meas_elem, "Spectrum", namespaces)
    if spectrum_elem is None:
        spectrum_elem = _find_element(meas_elem, "ChannelData", namespaces)
    
    if spectrum_elem is not None:
        # Get channel data
        channel_data = _find_element(spectrum_elem, "ChannelData", namespaces)
        if channel_data is None:
            channel_data = spectrum_elem
        measurement.counts = _parse_channel_data(channel_data, namespaces)
        
        # Get live time
        live_time_str = _get_text(spectrum_elem, "LiveTimeDuration", namespaces)
        if not live_time_str:
            live_time_str = _get_text(spectrum_elem, "LiveTime", namespaces)
        if live_time_str:
            measurement.live_time = _parse_duration(live_time_str)
        
        # Get real time
        real_time_str = _get_text(spectrum_elem, "RealTimeDuration", namespaces)
        if not real_time_str:
            real_time_str = _get_text(spectrum_elem, "RealTime", namespaces)
        if real_time_str:
            measurement.real_time = _parse_duration(real_time_str)
    
    # Get start time
    start_time_str = _get_text(meas_elem, "StartDateTime", namespaces)
    if start_time_str:
        measurement.start_time = _parse_datetime(start_time_str)
    
    # Get energy calibration
    cal_elem = _find_element(meas_elem, "EnergyCalibration", namespaces)
    if cal_elem is None:
        cal_elem = _find_element(meas_elem, "Calibration", namespaces)
    measurement.energy_calibration = _parse_calibration(cal_elem, namespaces)
    
    # Get detector information
    det_elem = _find_element(meas_elem, "DetectorInformation", namespaces)
    if det_elem is not None:
        measurement.detector_type = _get_text(det_elem, "DetectorType", namespaces)
        measurement.detector_description = _get_text(det_elem, "DetectorDescription", namespaces)
    
    # Get dose rate
    dose_elem = _find_element(meas_elem, "DoseRate", namespaces)
    if dose_elem is not None and dose_elem.text:
        try:
            measurement.dose_rate = float(dose_elem.text)
            measurement.dose_rate_units = dose_elem.get("units", "")
        except ValueError:
            pass
    
    # Get nuclide identifications
    analysis_elem = _find_element(meas_elem, "AnalysisResults", namespaces)
    if analysis_elem is not None:
        nuclide_elems = _find_all_elements(analysis_elem, "Nuclide", namespaces)
        for nuc_elem in nuclide_elems:
            nuc_name = _get_text(nuc_elem, "NuclideName", namespaces)
            if nuc_name:
                measurement.nuclides_identified.append(nuc_name)
    
    return measurement


def parse_iso8601_duration(duration_str: str) -> Optional[float]:
    """
    Parse ISO 8601 duration string to seconds.
    
    Supports formats like:
    - PT60S (60 seconds)
    - PT5M30S (5 minutes, 30 seconds)
    - PT2H30M (2 hours, 30 minutes)
    - P1DT12H (1 day, 12 hours)
    - P2D (2 days)
    
    Parameters
    ----------
    duration_str : str
        ISO 8601 duration string starting with 'P'
    
    Returns
    -------
    Optional[float]
        Duration in seconds, or None if parsing fails
    
    Examples
    --------
    >>> parse_iso8601_duration("PT60S")
    60.0
    >>> parse_iso8601_duration("P1DT2H30M")
    95400.0
    """
    if not duration_str or not isinstance(duration_str, str):
        return None
    
    duration_str = duration_str.strip()
    
    # Must start with 'P'
    if not duration_str.startswith("P"):
        return None
    
    total_seconds = 0.0
    remaining = duration_str[1:]  # Remove 'P'
    
    try:
        # Check for days before 'T'
        if "T" in remaining:
            date_part, time_part = remaining.split("T", 1)
        else:
            date_part = remaining
            time_part = ""
        
        # Parse date part (days)
        if date_part:
            if "D" in date_part:
                days_str = date_part.replace("D", "")
                if days_str:
                    total_seconds += float(days_str) * 86400
        
        # Parse time part
        if time_part:
            # Hours
            if "H" in time_part:
                hours, time_part = time_part.split("H", 1)
                total_seconds += float(hours) * 3600
            
            # Minutes (but not months which wouldn't appear after T)
            if "M" in time_part:
                minutes, time_part = time_part.split("M", 1)
                total_seconds += float(minutes) * 60
            
            # Seconds
            if "S" in time_part:
                seconds = time_part.replace("S", "")
                if seconds:
                    total_seconds += float(seconds)
        
        return total_seconds
    
    except (ValueError, AttributeError):
        return None


def _parse_duration(duration_str: str) -> float:
    """Parse ISO 8601 duration or simple number."""
    if not duration_str:
        return 0.0
    
    # Try simple float first
    try:
        return float(duration_str)
    except ValueError:
        pass
    
    # Try ISO 8601 duration
    result = parse_iso8601_duration(duration_str)
    if result is not None:
        return result
    
    return 0.0


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ISO 8601 datetime string."""
    if not dt_str:
        return None
    
    # Common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    
    # Remove timezone suffix for parsing
    dt_clean = re.sub(r'[+-]\d{2}:\d{2}$', '', dt_str)
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_clean, fmt)
        except ValueError:
            continue
    
    return None


def read_n42_file(
    filepath: Union[str, Path],
    verbose: bool = False,
) -> N42Document:
    """
    Read an N42.42 XML format file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to N42 file
    verbose : bool
        Print parsing information
    
    Returns
    -------
    N42Document
        Parsed N42 document with all measurements
    
    Examples
    --------
    >>> doc = read_n42_file("spectrum.n42")
    >>> spectrum = doc.get_spectrum()
    >>> print(f"Channels: {spectrum.n_channels}")
    >>> print(f"Live time: {spectrum.live_time} s")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"N42 file not found: {filepath}")
    
    if verbose:
        print(f"Reading N42 file: {filepath}")
    
    # Parse XML
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Detect namespace
    namespaces = N42_NAMESPACES.copy()
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
        namespaces["detected"] = ns[1:-1]
    
    # Create document
    doc = N42Document(file_path=filepath)
    
    # Get instrument information
    instr_elem = _find_element(root, "InstrumentInformation", namespaces)
    if instr_elem is not None:
        doc.instrument_type = _get_text(instr_elem, "InstrumentType", namespaces)
        doc.instrument_model = _get_text(instr_elem, "InstrumentModel", namespaces)
        doc.instrument_id = _get_text(instr_elem, "InstrumentID", namespaces)
        doc.manufacturer = _get_text(instr_elem, "Manufacturer", namespaces)
    
    # Find all measurements
    # N42.42-2011 uses RadMeasurement
    meas_elems = _find_all_elements(root, "RadMeasurement", namespaces)
    
    # N42.42-2006 uses Measurement
    if not meas_elems:
        meas_elems = _find_all_elements(root, "Measurement", namespaces)
    
    # Also check for direct Spectrum elements
    if not meas_elems:
        spectrum_elems = _find_all_elements(root, "Spectrum", namespaces)
        for spec_elem in spectrum_elems:
            # Wrap in a pseudo-measurement element
            meas_elems.append(spec_elem)
    
    for meas_elem in meas_elems:
        measurement = _parse_measurement(meas_elem, namespaces)
        if measurement.n_channels > 0:
            doc.measurements.append(measurement)
    
    if verbose:
        print(f"  Spectra found: {doc.n_spectra}")
        for i, m in enumerate(doc.measurements):
            print(f"    [{i}] {m.n_channels} channels, "
                  f"LT={m.live_time:.1f}s, "
                  f"counts={m.total_counts:.0f}")
    
    return doc


def read_n42_spectrum(
    filepath: Union[str, Path],
    spectrum_index: int = 0,
) -> N42Measurement:
    """
    Read a single spectrum from an N42 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to N42 file
    spectrum_index : int
        Index of spectrum to read (if multiple)
    
    Returns
    -------
    N42Measurement
        The requested spectrum
    """
    doc = read_n42_file(filepath)
    
    if spectrum_index >= len(doc.measurements):
        raise IndexError(f"Spectrum index {spectrum_index} out of range "
                        f"(file has {len(doc.measurements)} spectra)")
    
    return doc.measurements[spectrum_index]


def write_n42_file(
    filepath: Union[str, Path],
    measurements: Union[N42Measurement, List[N42Measurement]],
    instrument_info: Optional[Dict[str, str]] = None,
    version: str = "2011",
) -> None:
    """
    Write measurements to an N42.42 XML file.
    
    Parameters
    ----------
    filepath : str or Path
        Output file path
    measurements : N42Measurement or List[N42Measurement]
        Measurement(s) to write
    instrument_info : Dict, optional
        Instrument information
    version : str
        N42 version ('2006' or '2011')
    """
    filepath = Path(filepath)
    
    if isinstance(measurements, N42Measurement):
        measurements = [measurements]
    
    # Select namespace
    if version == "2011":
        ns = N42_NAMESPACES["n42"]
    else:
        ns = N42_NAMESPACES["n42_2006"]
    
    # Register namespace
    ET.register_namespace("", ns)
    
    # Create root element
    root = ET.Element(f"{{{ns}}}N42InstrumentData")
    
    # Add instrument information
    if instrument_info:
        instr = ET.SubElement(root, f"{{{ns}}}InstrumentInformation")
        for key, value in instrument_info.items():
            elem = ET.SubElement(instr, f"{{{ns}}}{key}")
            elem.text = value
    
    # Add measurements
    for i, meas in enumerate(measurements):
        meas_elem = ET.SubElement(root, f"{{{ns}}}RadMeasurement")
        meas_elem.set("id", meas.spectrum_id or f"Measurement{i}")
        
        # Start time
        if meas.start_time:
            start_elem = ET.SubElement(meas_elem, f"{{{ns}}}StartDateTime")
            start_elem.text = meas.start_time.isoformat()
        
        # Spectrum
        spec_elem = ET.SubElement(meas_elem, f"{{{ns}}}Spectrum")
        
        # Live time
        lt_elem = ET.SubElement(spec_elem, f"{{{ns}}}LiveTimeDuration")
        lt_elem.text = f"PT{meas.live_time}S"
        
        # Real time
        rt_elem = ET.SubElement(spec_elem, f"{{{ns}}}RealTimeDuration")
        rt_elem.text = f"PT{meas.real_time}S"
        
        # Channel data
        data_elem = ET.SubElement(spec_elem, f"{{{ns}}}ChannelData")
        data_elem.text = " ".join(str(int(c)) for c in meas.counts)
        
        # Energy calibration
        if len(meas.energy_calibration) > 1:
            cal_elem = ET.SubElement(meas_elem, f"{{{ns}}}EnergyCalibration")
            coef_elem = ET.SubElement(cal_elem, f"{{{ns}}}CoefficientValues")
            coef_elem.text = " ".join(str(c) for c in meas.energy_calibration)
    
    # Write file
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "N42Measurement",
    "N42Document",
    "read_n42_file",
    "read_n42_spectrum",
    "write_n42_file",
]
