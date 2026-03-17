"""
CNF File Reader Module.

Handles reading of Canberra CNF binary spectrum files.

CNF (Canberra Nonfixed Format) is a proprietary binary file format used by 
Canberra/Mirion gamma spectroscopy systems. This module provides basic parsing
capabilities for extracting spectrum data and metadata.

References:
- Based on reverse-engineering and community documentation
- Compatible with Genie 2000 and similar systems

Author: FluxForge Development Team
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from fluxforge.io.spe import GammaSpectrum


# CNF file structure constants
CNF_MAGIC = b'\x00\x00\x00\x00'  # CNF files often start with null bytes
RECORD_HEADER_SIZE = 48


@dataclass
class CNFHeader:
    """
    CNF file header information.
    
    Attributes:
        file_size: Total file size in bytes
        detector_name: Detector identifier string
        sample_id: Sample identification
        start_date: Acquisition start date
        start_time: Acquisition start time
        live_time: Live time in seconds
        real_time: Real time in seconds
        n_channels: Number of spectrum channels
    """
    
    file_size: int = 0
    detector_name: str = ""
    sample_id: str = ""
    start_date: str = ""
    start_time: str = ""
    live_time: float = 0.0
    real_time: float = 0.0
    n_channels: int = 0


@dataclass
class CNFCalibration:
    """
    Energy and shape calibration from CNF file.
    
    Attributes:
        energy_coefficients: Polynomial coefficients for channel->energy
        fwhm_coefficients: FWHM calibration coefficients
        efficiency_coefficients: Efficiency calibration (if present)
    """
    
    energy_coefficients: List[float] = field(default_factory=list)
    fwhm_coefficients: List[float] = field(default_factory=list)
    efficiency_coefficients: List[float] = field(default_factory=list)


@dataclass
class CNFData:
    """
    Complete parsed CNF file data.
    
    Attributes:
        header: File header information
        calibration: Calibration parameters
        spectrum: Channel counts array
        peaks: List of identified peaks (if present)
        metadata: Additional metadata
    """
    
    header: CNFHeader = field(default_factory=CNFHeader)
    calibration: CNFCalibration = field(default_factory=CNFCalibration)
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    peaks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _read_string(data: bytes, offset: int, length: int) -> str:
    """Read a null-terminated string from binary data."""
    raw = data[offset:offset + length]
    try:
        # Remove null bytes and decode
        end = raw.find(b'\x00')
        if end != -1:
            raw = raw[:end]
        return raw.decode('latin-1').strip()
    except (UnicodeDecodeError, AttributeError):
        return ""


def _read_float(data: bytes, offset: int) -> float:
    """Read a 4-byte float (little-endian)."""
    try:
        return struct.unpack('<f', data[offset:offset + 4])[0]
    except struct.error:
        return 0.0


def _read_double(data: bytes, offset: int) -> float:
    """Read an 8-byte double (little-endian)."""
    try:
        return struct.unpack('<d', data[offset:offset + 8])[0]
    except struct.error:
        return 0.0


def _read_uint32(data: bytes, offset: int) -> int:
    """Read a 4-byte unsigned integer (little-endian)."""
    try:
        return struct.unpack('<I', data[offset:offset + 4])[0]
    except struct.error:
        return 0


def _read_uint16(data: bytes, offset: int) -> int:
    """Read a 2-byte unsigned integer (little-endian)."""
    try:
        return struct.unpack('<H', data[offset:offset + 2])[0]
    except struct.error:
        return 0


def _read_uint8(data: bytes, offset: int) -> int:
    """Read a 1-byte unsigned integer."""
    try:
        return data[offset]
    except IndexError:
        return 0


def _read_uint64(data: bytes, offset: int) -> int:
    """Read an 8-byte unsigned integer (little-endian)."""
    try:
        return struct.unpack('<Q', data[offset:offset + 8])[0]
    except struct.error:
        return 0


def _read_pdp11_float(data: bytes, offset: int) -> float:
    """
    Convert PDP11 32-bit float layout to IEEE754 float.

    CNF stores several calibration values in this legacy format.
    """
    w0 = _read_uint16(data, offset)
    w1 = _read_uint16(data, offset + 2)
    packed = struct.pack('<HH', w1, w0)
    try:
        return struct.unpack('<f', packed)[0] / 4.0
    except struct.error:
        return 0.0


def _parse_cnf_sectioned(data: bytes) -> Optional[CNFData]:
    """
    Parse CNF using section headers (closer to Canberra layout).

    This path handles real-world CNF files more reliably than fixed-offset
    heuristics and is used first when section headers are present.
    """
    result = CNFData()
    result.header.file_size = len(data)

    offs_param = None
    offs_chan = None
    detector_name = ""
    sample_desc = ""
    live_time = 0.0
    real_time = 0.0

    i = 0
    while True:
        sec_header = 0x70 + i * 0x30
        i += 1
        if sec_header + 0x0E >= len(data):
            break

        sec_id_header = _read_uint32(data, sec_header)
        if sec_id_header == 0:
            break
        sec_loc = _read_uint32(data, sec_header + 0x0A)
        if sec_loc <= 0 or sec_loc >= len(data):
            continue

        # Parameter section (times, calibration, detector metadata)
        if sec_id_header == 0x00012000:
            offs_param = sec_loc

            offs_calib = offs_param + 0x30 + _read_uint16(data, offs_param + 0x22)
            if offs_calib + 0x50 < len(data):
                a0 = _read_pdp11_float(data, offs_calib + 0x44)
                a1 = _read_pdp11_float(data, offs_calib + 0x48)
                a2 = _read_pdp11_float(data, offs_calib + 0x4C)
                result.calibration.energy_coefficients = [a0, a1, a2]
                detector_name = _read_string(data, offs_calib + 0x30C, 0x06)

            offs_times = offs_param + 0x30 + _read_uint16(data, offs_param + 0x24)
            if offs_times + 0x19 < len(data):
                raw_rt = _read_uint64(data, offs_times + 0x09)
                raw_lt = _read_uint64(data, offs_times + 0x11)
                # CNF stores these in inverted uint64 form.
                real_time = ((~raw_rt) & 0xFFFFFFFFFFFFFFFF) * 1e-7
                live_time = ((~raw_lt) & 0xFFFFFFFFFFFFFFFF) * 1e-7

        # String section
        elif sec_id_header == 0x00012001:
            sample_desc = _read_string(data, sec_loc + 0x036E, 0x100)

        # Channel data section
        elif sec_id_header == 0x00012005:
            offs_chan = sec_loc

    if offs_param is None or offs_chan is None:
        return None

    n_channels = int(_read_uint8(data, offs_param + 0x00BA)) * 256
    if n_channels <= 0:
        n_channels = int(_read_uint16(data, offs_param + 0x00BA))
    if n_channels <= 0:
        return None

    spectrum_offset = offs_chan + 0x200
    byte_count = n_channels * 4
    if spectrum_offset + byte_count > len(data):
        return None

    counts = np.frombuffer(data, dtype='<u4', count=n_channels, offset=spectrum_offset).astype(np.float64)

    # Reject clearly invalid streams and let heuristic parser try.
    if counts.size == 0:
        return None
    if np.nanmax(counts) >= 4.2e9 and np.nanmean(counts) > 1e6:
        return None

    result.spectrum = counts
    result.header.n_channels = n_channels
    result.header.detector_name = detector_name.strip()
    result.header.sample_id = sample_desc.strip()
    if 0 < live_time < 1e8:
        result.header.live_time = live_time
    if 0 < real_time < 1e8:
        result.header.real_time = real_time

    return result


def _find_record(data: bytes, record_id: int) -> Optional[Tuple[int, int]]:
    """
    Find a record in CNF file by ID.
    
    CNF files contain records with headers indicating type and size.
    Returns (offset, size) tuple if found, None otherwise.
    """
    offset = 0
    while offset < len(data) - 8:
        # Read record header
        rec_id = _read_uint16(data, offset)
        rec_size = _read_uint32(data, offset + 2)
        
        if rec_id == record_id:
            return (offset, rec_size)
        
        if rec_size == 0:
            break
        
        offset += rec_size + 48  # Move to next record
        
        # Safety limit
        if offset > len(data):
            break
    
    return None


def parse_cnf_binary(data: bytes) -> CNFData:
    """
    Parse raw CNF binary data.
    
    Parameters
    ----------
    data : bytes
        Raw binary content of CNF file.
        
    Returns
    -------
    CNFData
        Parsed CNF data including spectrum and calibration.
    """
    result = CNFData()
    
    if len(data) < 128:
        return result

    result.header.file_size = len(data)

    # Prefer section-based parsing when available.
    sectioned = _parse_cnf_sectioned(data)
    if sectioned is not None and len(sectioned.spectrum) > 0:
        return sectioned
    
    # CNF structure varies by version, try common patterns
    # Look for spectrum data - usually follows specific pattern
    
    # Method 1: Fixed offset approach (common for older CNF)
    # Spectrum data often starts at offset 0x200 or 0x600
    
    n_channels = 0
    spectrum_offset = 0
    
    # Try to find spectrum marker or use heuristics
    for try_offset in [0x200, 0x600, 0x800, 0x1000]:
        if try_offset + 16 < len(data):
            # Check if this looks like spectrum data
            # (increasing then decreasing pattern typical of spectra)
            test_vals = []
            for i in range(4):
                val = _read_uint32(data, try_offset + i * 4)
                test_vals.append(val)
            
            # Basic sanity check
            if all(0 <= v < 1e9 for v in test_vals):
                spectrum_offset = try_offset
                break
    
    # Determine number of channels
    # Common values: 1024, 2048, 4096, 8192, 16384
    for try_channels in [16384, 8192, 4096, 2048, 1024]:
        if spectrum_offset + try_channels * 4 <= len(data):
            n_channels = try_channels
            break
    
    result.header.n_channels = n_channels
    
    # Read spectrum data
    if n_channels > 0 and spectrum_offset > 0:
        counts = []
        for i in range(n_channels):
            count = _read_uint32(data, spectrum_offset + i * 4)
            counts.append(count)
        result.spectrum = np.array(counts, dtype=np.float64)
    
    # Try to read calibration
    # Energy calibration often at specific offsets
    cal_offsets = [0x30, 0x40, 0x50, 0x100, 0x180]
    
    for cal_off in cal_offsets:
        if cal_off + 24 < len(data):
            a0 = _read_double(data, cal_off)
            a1 = _read_double(data, cal_off + 8)
            a2 = _read_double(data, cal_off + 16)
            
            # Check if these look like reasonable calibration coefficients
            # a0 should be near 0, a1 should be positive (keV/channel ratio)
            if -100 < a0 < 100 and 0 < a1 < 10:
                result.calibration.energy_coefficients = [a0, a1, a2]
                break
    
    # Try to read timing information
    # Live time and real time usually stored as doubles
    for time_off in [0x1A0, 0x1B0, 0x1C0]:
        if time_off + 16 < len(data):
            lt = _read_double(data, time_off)
            rt = _read_double(data, time_off + 8)
            
            if 0 < lt < 1e7 and 0 < rt < 1e7:  # Reasonable time range
                result.header.live_time = lt
                result.header.real_time = rt
                break
    
    return result


def read_cnf_file(filepath: str) -> GammaSpectrum:
    """
    Read a Canberra CNF file.
    
    Parses the binary CNF format and returns a GammaSpectrum object
    compatible with the FluxForge gamma spectroscopy workflow.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .cnf file.
        
    Returns
    -------
    GammaSpectrum
        Parsed spectrum with counts, calibration, and metadata.
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    ValueError
        If file format is not recognized or is empty.
        
    Notes
    -----
    CNF is a proprietary binary format with multiple versions.
    This parser attempts to handle common variants but may not
    work for all CNF files. If parsing fails, consider converting
    to SPE format using Canberra software.
    
    Examples
    --------
    >>> spectrum = read_cnf_file("sample.cnf")
    >>> print(f"Channels: {len(spectrum.counts)}")
    >>> print(f"Live time: {spectrum.live_time} s")
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) < 128:
        raise ValueError(f"File too small to be valid CNF: {filepath}")
    
    # Parse binary data
    cnf = parse_cnf_binary(data)
    
    if len(cnf.spectrum) == 0:
        raise ValueError(
            f"Could not extract spectrum from {filepath}. "
            "File may be corrupt or use an unsupported CNF variant. "
            "Consider converting to SPE format using Canberra software."
        )
    
    # Build calibration dict
    cal = cnf.calibration.energy_coefficients
    if len(cal) >= 2:
        calibration = {
            'a0': cal[0],
            'a1': cal[1],
            'a2': cal[2] if len(cal) >= 3 else 0.0,
        }
    else:
        calibration = {'a0': 0.0, 'a1': 1.0, 'a2': 0.0}  # Default: channel = energy
    
    return GammaSpectrum(
        counts=cnf.spectrum,
        calibration=calibration,
        live_time=cnf.header.live_time if cnf.header.live_time > 0 else 1.0,
        real_time=cnf.header.real_time if cnf.header.real_time > 0 else 1.0,
        detector_id=cnf.header.detector_name or "CNF",
        spectrum_id=cnf.header.sample_id or filepath.stem,
        metadata={
            'format': 'CNF',
            'source_file': str(filepath),
            'n_channels': cnf.header.n_channels,
        },
    )


def can_read_cnf(filepath: str) -> bool:
    """
    Check if a file appears to be a valid CNF file.
    
    Parameters
    ----------
    filepath : str
        Path to file to check.
        
    Returns
    -------
    bool
        True if file appears to be CNF format.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() != '.cnf':
            return False
        
        # Check file size (CNF files are typically > 4KB)
        if path.stat().st_size < 4096:
            return False
        
        # Try to read first bytes
        with open(path, 'rb') as f:
            header = f.read(128)
        
        # Very basic check - CNF files have specific structure
        # but no universal magic number
        return len(header) >= 128
        
    except (OSError, IOError):
        return False
