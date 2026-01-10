"""HPGe report/export readers.

Supports:
- Genie/LabSOCS TXT report exports
- CHN binary format (ORTEC/Maestro)
- Generic detector response parsing
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.io.genie import ReportPeak, parse_genie_report
from fluxforge.io.metadata import qc_flags_for_spectrum

FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
ID_RE = re.compile(r"^\s*ID:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FILE_DATE_RE = re.compile(r"^\s*File:\s*(.+?)\s+Date:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
LT_RT_DT_RE = re.compile(
    r"LT:\s*([\d.,]+)\s*RT:\s*([\d.,]+)\s*DT:\s*([\d.,]+)\s*%?",
    re.IGNORECASE,
)
DETECTOR_ID_RE = re.compile(r"Detector\s+ID:\s*(.+)$", re.IGNORECASE)


def _parse_number(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _parse_datetime(value: str) -> Optional[datetime]:
    for fmt in (
        "%B %d, %Y %H:%M:%S",
        "%B %d, %Y %H:%M",
        "%b %d, %Y %H:%M:%S",
        "%b %d, %Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


@dataclass
class HPGeReport:
    """Parsed HPGe report export."""

    report_id: str
    file_name: str = ""
    start_time: Optional[datetime] = None
    live_time: float = 0.0
    real_time: float = 0.0
    dead_time_pct: Optional[float] = None
    detector_id: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    peaks: List[ReportPeak] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    qc_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "file_name": self.file_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "live_time": self.live_time,
            "real_time": self.real_time,
            "dead_time_pct": self.dead_time_pct,
            "detector_id": self.detector_id,
            "calibration": self.calibration,
            "efficiency": self.efficiency,
            "peaks": [peak.__dict__ for peak in self.peaks],
            "metadata": self.metadata,
            "qc_flags": self.qc_flags,
        }


def read_hpge_report(filepath: Union[str, Path]) -> HPGeReport:
    """Read HPGe report export (Genie/LabSOCS TXT)."""
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    report_id = ""
    match = ID_RE.search(content)
    if match:
        report_id = match.group(1).strip()

    file_name = ""
    start_time = None
    match = FILE_DATE_RE.search(content)
    if match:
        file_name = match.group(1).strip()
        start_time = _parse_datetime(match.group(2).strip())

    live_time = 0.0
    real_time = 0.0
    dead_time = None
    match = LT_RT_DT_RE.search(content)
    if match:
        parsed_live = _parse_number(match.group(1))
        parsed_real = _parse_number(match.group(2))
        parsed_dead = _parse_number(match.group(3))
        if parsed_live is not None:
            live_time = parsed_live
        if parsed_real is not None:
            real_time = parsed_real
        if parsed_dead is not None:
            dead_time = parsed_dead

    detector_id = ""
    for line in content.splitlines():
        match = DETECTOR_ID_RE.search(line)
        if match:
            detector_id = match.group(1).strip()
            break

    calibration: Dict[str, Any] = {}
    efficiency: Dict[str, Any] = {}
    for line in content.splitlines():
        if "Energy" in line and "Ch" in line:
            numbers = FLOAT_RE.findall(line)
            if len(numbers) >= 2:
                calibration["energy"] = [float(n) for n in numbers[:3]]
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {"C1", "C2", "C3", "C4", "A", "T1", "DI", "DL"}:
                parsed = _parse_number(value)
                if parsed is not None:
                    efficiency[key] = parsed

    peaks = parse_genie_report(filepath)

    qc_flags = qc_flags_for_spectrum(
        spectrum_id=report_id or filepath.stem,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        calibration=calibration,
        detector_id=detector_id,
    )

    metadata = {
        "source_file": str(filepath),
        "format": "hpge_report",
    }
    if dead_time is not None:
        metadata["dead_time_pct"] = dead_time

    return HPGeReport(
        report_id=report_id or filepath.stem,
        file_name=file_name,
        start_time=start_time,
        live_time=live_time,
        real_time=real_time,
        dead_time_pct=dead_time,
        detector_id=detector_id,
        calibration=calibration,
        efficiency=efficiency,
        peaks=peaks,
        metadata=metadata,
        qc_flags=qc_flags,
    )


# ==============================================================================
# CHN Binary Format Reader (ORTEC/Maestro Compatible)
# ==============================================================================

@dataclass
class CHNSpectrum:
    """
    Parsed CHN binary spectrum file.
    
    CHN is a binary format used by ORTEC/Maestro for storing gamma spectra.
    The format includes header information, calibration, and channel counts.
    
    Attributes:
        counts: Channel counts array
        channels: Channel numbers
        live_time_s: Live time in seconds
        real_time_s: Real time in seconds
        start_time: Acquisition start time
        calibration: Energy calibration coefficients {offset, gain, quadratic}
        detector_id: Detector identifier
        sample_description: Sample description from file
        n_channels: Number of channels
        metadata: Additional metadata
    """
    
    counts: np.ndarray
    channels: np.ndarray
    live_time_s: float
    real_time_s: float
    start_time: Optional[datetime] = None
    calibration: Dict[str, float] = field(default_factory=dict)
    detector_id: str = ""
    sample_description: str = ""
    n_channels: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def dead_time_fraction(self) -> float:
        """Calculate dead time fraction."""
        if self.real_time_s > 0:
            return 1.0 - self.live_time_s / self.real_time_s
        return 0.0
    
    def energy_from_channel(self, channel: np.ndarray) -> np.ndarray:
        """Convert channel to energy using calibration."""
        offset = self.calibration.get("offset", 0.0)
        gain = self.calibration.get("gain", 1.0)
        quad = self.calibration.get("quadratic", 0.0)
        return offset + gain * channel + quad * channel**2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "counts": self.counts.tolist(),
            "channels": self.channels.tolist(),
            "live_time_s": self.live_time_s,
            "real_time_s": self.real_time_s,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "calibration": self.calibration,
            "detector_id": self.detector_id,
            "sample_description": self.sample_description,
            "n_channels": self.n_channels,
            "metadata": self.metadata,
        }


def read_chn_file(filepath: Union[str, Path]) -> CHNSpectrum:
    """
    Read a CHN binary spectrum file.
    
    Supports both 32-bit integer formats common in ORTEC/Maestro systems.
    
    CHN Format Structure (typical):
    - Header (32 bytes): type, MCA number, segment, start time/date
    - Time data (8 bytes): real time, live time (0.02s units)
    - Date/time strings (variable)
    - Channel data: 32-bit integers
    - Calibration data (optional trailer)
    
    Parameters
    ----------
    filepath : str or Path
        Path to CHN file
        
    Returns
    -------
    CHNSpectrum
        Parsed spectrum data
        
    Raises
    ------
    ValueError
        If file cannot be parsed as CHN format
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) < 32:
        raise ValueError(f"File too small for CHN format: {len(data)} bytes")
    
    # Parse header
    # Byte 0: should be -1 (0xFF) for CHN files
    # Bytes 1-2: MCA number (usually 1)
    # Bytes 3-4: Segment number
    # Bytes 5-6: Start time (seconds * 50 / 10)
    # Bytes 7-10: Start date
    
    header_check = struct.unpack('<h', data[0:2])[0]
    
    # Different CHN variants exist; try to detect format
    if header_check == -1:
        # Standard ORTEC format
        return _parse_ortec_chn(data, filepath)
    else:
        # Try Maestro/alternative format
        return _parse_maestro_chn(data, filepath)


def _parse_ortec_chn(data: bytes, filepath: Path) -> CHNSpectrum:
    """Parse ORTEC-style CHN file."""
    
    # Header structure (32 bytes)
    # 0-1: -1 marker
    # 2-3: MCA number
    # 4-5: Segment
    # 6-7: Seconds start (ASCII or binary)
    # 8-9: Real time (0.02s units, 2 bytes)
    # 10-11: Live time (0.02s units, 2 bytes)
    # 12-19: Date string (DDMMMYY or similar)
    # 20-27: Time string (HHMMSS)
    # 28-29: Channel offset
    # 30-31: Number of channels
    
    try:
        # Real time and live time in 0.02s units
        real_time_units = struct.unpack('<H', data[8:10])[0]
        live_time_units = struct.unpack('<H', data[10:12])[0]
        
        # Convert to seconds (0.02 second units)
        real_time_s = real_time_units * 0.02
        live_time_s = live_time_units * 0.02
        
        # If times seem unreasonable, try 4-byte values
        if real_time_s < 0.1 or live_time_s < 0.1:
            real_time_units = struct.unpack('<I', data[8:12])[0]
            live_time_units = struct.unpack('<I', data[12:16])[0]
            real_time_s = real_time_units * 0.02
            live_time_s = live_time_units * 0.02
        
        # Channel offset and count
        channel_offset = struct.unpack('<H', data[28:30])[0]
        n_channels = struct.unpack('<H', data[30:32])[0]
        
        # Validate n_channels
        if n_channels == 0 or n_channels > 32768:
            # Try common values
            file_size = len(data)
            for test_channels in [4096, 8192, 2048, 16384]:
                expected_size = 32 + test_channels * 4
                if abs(file_size - expected_size) < 256:
                    n_channels = test_channels
                    break
            else:
                n_channels = (len(data) - 32) // 4
        
        # Read channel data (32-bit integers)
        data_start = 32
        counts = np.zeros(n_channels, dtype=np.int32)
        for i in range(n_channels):
            offset = data_start + i * 4
            if offset + 4 <= len(data):
                counts[i] = struct.unpack('<I', data[offset:offset + 4])[0]
        
        # Parse date/time if available
        start_time = None
        try:
            date_str = data[12:20].decode('ascii', errors='ignore').strip()
            time_str = data[20:28].decode('ascii', errors='ignore').strip()
            if date_str and time_str:
                start_time = _parse_chn_datetime(date_str, time_str)
        except Exception:
            pass
        
        # Look for calibration in trailer (after channel data)
        calibration: Dict[str, float] = {}
        trailer_start = data_start + n_channels * 4
        if trailer_start + 12 <= len(data):
            try:
                # Calibration coefficients are often stored as floats
                cal_offset = struct.unpack('<f', data[trailer_start:trailer_start + 4])[0]
                cal_gain = struct.unpack('<f', data[trailer_start + 4:trailer_start + 8])[0]
                cal_quad = struct.unpack('<f', data[trailer_start + 8:trailer_start + 12])[0]
                
                # Validate reasonable values
                if -1000 < cal_offset < 1000 and 0 < cal_gain < 100:
                    calibration = {
                        "offset": cal_offset,
                        "gain": cal_gain,
                        "quadratic": cal_quad,
                    }
            except Exception:
                pass
        
        channels = np.arange(channel_offset, channel_offset + n_channels)
        
        return CHNSpectrum(
            counts=counts,
            channels=channels,
            live_time_s=live_time_s,
            real_time_s=real_time_s,
            start_time=start_time,
            calibration=calibration,
            detector_id="",
            sample_description="",
            n_channels=n_channels,
            metadata={
                "source_file": str(filepath),
                "format": "chn_ortec",
            },
        )
        
    except Exception as e:
        raise ValueError(f"Failed to parse ORTEC CHN file: {e}") from e


def _parse_maestro_chn(data: bytes, filepath: Path) -> CHNSpectrum:
    """Parse Maestro-style CHN file."""
    
    # Maestro CHN has slightly different header structure
    # Try to detect based on file structure
    
    # Common structure:
    # 0-1: Version or type indicator
    # 2-3: Number of channels
    # 4-7: Live time (32-bit, units vary)
    # 8-11: Real time (32-bit, units vary)
    
    try:
        # Try version indicator
        version = struct.unpack('<H', data[0:2])[0]
        n_channels = struct.unpack('<H', data[2:4])[0]
        
        # Validate n_channels
        if n_channels == 0 or n_channels > 32768:
            # Calculate from file size assuming 32-bit data
            n_channels = max((len(data) - 64) // 4, 0)
            if n_channels > 32768:
                n_channels = 4096  # Default
        
        # Time values - try different unit interpretations
        live_raw = struct.unpack('<I', data[4:8])[0]
        real_raw = struct.unpack('<I', data[8:12])[0]
        
        # Common unit is 0.02 seconds or milliseconds
        if live_raw > 1_000_000:
            # Likely milliseconds
            live_time_s = live_raw / 1000.0
            real_time_s = real_raw / 1000.0
        elif live_raw > 10_000:
            # Likely 0.02 second units
            live_time_s = live_raw * 0.02
            real_time_s = real_raw * 0.02
        else:
            # Likely seconds
            live_time_s = float(live_raw)
            real_time_s = float(real_raw)
        
        # Read channel data
        data_start = 64  # Common offset for Maestro
        if data_start + n_channels * 4 > len(data):
            data_start = 32
        
        counts = np.zeros(n_channels, dtype=np.int32)
        for i in range(n_channels):
            offset = data_start + i * 4
            if offset + 4 <= len(data):
                counts[i] = struct.unpack('<I', data[offset:offset + 4])[0]
        
        # Check for calibration in header or trailer
        calibration: Dict[str, float] = {}
        
        # Try common calibration locations
        for cal_offset in [12, 16, data_start + n_channels * 4]:
            if cal_offset + 12 <= len(data):
                try:
                    cal_a = struct.unpack('<f', data[cal_offset:cal_offset + 4])[0]
                    cal_b = struct.unpack('<f', data[cal_offset + 4:cal_offset + 8])[0]
                    cal_c = struct.unpack('<f', data[cal_offset + 8:cal_offset + 12])[0]
                    
                    if -1000 < cal_a < 1000 and 0 < cal_b < 100 and -1 < cal_c < 1:
                        calibration = {
                            "offset": cal_a,
                            "gain": cal_b,
                            "quadratic": cal_c,
                        }
                        break
                except Exception:
                    continue
        
        channels = np.arange(n_channels)
        
        return CHNSpectrum(
            counts=counts,
            channels=channels,
            live_time_s=live_time_s,
            real_time_s=real_time_s,
            start_time=None,
            calibration=calibration,
            detector_id="",
            sample_description="",
            n_channels=n_channels,
            metadata={
                "source_file": str(filepath),
                "format": "chn_maestro",
            },
        )
        
    except Exception as e:
        raise ValueError(f"Failed to parse Maestro CHN file: {e}") from e


def _parse_chn_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse CHN date/time strings."""
    
    # Common formats: DDMMMYY, HHMMSS
    date_formats = [
        ("%d%b%y", "%H%M%S"),
        ("%d%b%Y", "%H%M%S"),
        ("%m%d%y", "%H%M%S"),
        ("%Y%m%d", "%H%M%S"),
    ]
    
    for date_fmt, time_fmt in date_formats:
        try:
            combined = f"{date_str} {time_str}"
            combined_fmt = f"{date_fmt} {time_fmt}"
            return datetime.strptime(combined, combined_fmt)
        except ValueError:
            continue
    
    return None


def chn_to_hpge_report(chn_spectrum: CHNSpectrum, report_id: Optional[str] = None) -> HPGeReport:
    """
    Convert CHN spectrum to HPGeReport for unified processing.
    
    Parameters
    ----------
    chn_spectrum : CHNSpectrum
        Parsed CHN file
    report_id : str, optional
        Override report ID
        
    Returns
    -------
    HPGeReport
        Converted report
    """
    source_file = chn_spectrum.metadata.get("source_file", "")
    
    return HPGeReport(
        report_id=report_id or Path(source_file).stem if source_file else "chn_spectrum",
        file_name=source_file,
        start_time=chn_spectrum.start_time,
        live_time=chn_spectrum.live_time_s,
        real_time=chn_spectrum.real_time_s,
        dead_time_pct=chn_spectrum.dead_time_fraction * 100,
        detector_id=chn_spectrum.detector_id,
        calibration=chn_spectrum.calibration,
        efficiency={},
        peaks=[],  # Peaks need to be found separately
        metadata=chn_spectrum.metadata,
        qc_flags=[],
    )


# ==============================================================================
# Unified Spectrum Reader
# ==============================================================================

def read_hpge_spectrum(
    filepath: Union[str, Path],
    format_hint: Optional[str] = None,
) -> Union[HPGeReport, CHNSpectrum]:
    """
    Read HPGe spectrum file with automatic format detection.
    
    Supports:
    - Genie/LabSOCS TXT exports (.txt, .rpt)
    - CHN binary format (.chn)
    - SPE format (delegates to io.spe)
    
    Parameters
    ----------
    filepath : str or Path
        Path to spectrum file
    format_hint : str, optional
        Force specific format: 'txt', 'chn', 'spe'
        
    Returns
    -------
    HPGeReport or CHNSpectrum
        Parsed spectrum data
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if format_hint == 'chn' or suffix == '.chn':
        return read_chn_file(filepath)
    elif format_hint == 'txt' or suffix in ('.txt', '.rpt'):
        return read_hpge_report(filepath)
    elif format_hint == 'spe' or suffix == '.spe':
        # Delegate to SPE reader
        from fluxforge.io.spe import read_spe_file
        return read_spe_file(filepath)
    else:
        # Try to detect format
        try:
            # Try CHN first (binary)
            return read_chn_file(filepath)
        except (ValueError, struct.error):
            pass
        
        # Try text report
        try:
            return read_hpge_report(filepath)
        except Exception:
            pass
        
        raise ValueError(f"Could not determine format of {filepath}")
