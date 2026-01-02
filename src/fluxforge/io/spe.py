"""
SPE File Parser for HPGe Gamma Spectroscopy

This module provides standalone SPE file parsing for gamma spectra from
HPGe detectors. It supports both standard SPE format and "$" prefixed format
commonly used by ORTEC MAESTRO and similar software.

This is a standalone implementation that does not require PyNE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class GammaSpectrum:
    """
    Container for gamma spectrum data from HPGe detector.
    
    Attributes
    ----------
    counts : np.ndarray
        Channel counts array
    channels : np.ndarray
        Channel numbers
    energies : Optional[np.ndarray]
        Energy values for each channel (if calibrated)
    live_time : float
        Live time in seconds
    real_time : float
        Real time in seconds
    start_time : Optional[datetime]
        Acquisition start time
    spectrum_id : str
        Spectrum identifier or filename
    detector_id : str
        Detector identifier
    calibration : Dict[str, Any]
        Energy and shape calibration parameters
    metadata : Dict[str, Any]
        Additional metadata from file
        
    Examples
    --------
    >>> from fluxforge.io.spe import read_spe_file
    >>> spectrum = read_spe_file("sample.spe")
    >>> print(f"Live time: {spectrum.live_time} s")
    >>> print(f"Total counts: {spectrum.counts.sum()}")
    """
    
    counts: np.ndarray
    channels: np.ndarray = field(default_factory=lambda: np.array([]))
    energies: Optional[np.ndarray] = None
    live_time: float = 0.0
    real_time: float = 0.0
    start_time: Optional[datetime] = None
    spectrum_id: str = ""
    detector_id: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if len(self.channels) == 0 and len(self.counts) > 0:
            self.channels = np.arange(len(self.counts))
        
        # Apply energy calibration if available
        if self.energies is None and self.calibration:
            self.energies = self.calibrate_channels()
    
    def calibrate_channels(self, coefficients: Optional[List[float]] = None) -> np.ndarray:
        """
        Apply energy calibration to channels.
        
        Parameters
        ----------
        coefficients : list of float, optional
            Polynomial coefficients [a0, a1, a2, ...] where
            E = a0 + a1*ch + a2*ch^2 + ...
            If None, uses self.calibration['energy']
        
        Returns
        -------
        np.ndarray
            Energy values for each channel in keV
        """
        if coefficients is None:
            coefficients = self.calibration.get('energy', [0.0, 1.0])
        
        energies = np.zeros_like(self.channels, dtype=float)
        for i, coeff in enumerate(coefficients):
            energies += coeff * (self.channels ** i)
        
        return energies
    
    def channel_to_energy(self, channel: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert channel number to energy using calibration."""
        coeffs = self.calibration.get('energy', [0.0, 1.0])
        result = sum(c * (channel ** i) for i, c in enumerate(coeffs))
        return result
    
    def energy_to_channel(self, energy: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert energy to channel number using calibration."""
        coeffs = self.calibration.get('energy', [0.0, 1.0])
        
        if len(coeffs) == 2:
            # Linear: E = a0 + a1*ch => ch = (E - a0) / a1
            channel = (energy - coeffs[0]) / coeffs[1]
        elif len(coeffs) == 3:
            # Quadratic: solve a2*ch^2 + a1*ch + (a0 - E) = 0
            a = coeffs[2]
            b = coeffs[1]
            c = coeffs[0] - energy
            discriminant = b**2 - 4*a*c
            channel = (-b + np.sqrt(discriminant)) / (2*a)
        else:
            # Use numerical inversion for higher order
            from scipy import optimize
            def energy_diff(ch):
                return self.channel_to_energy(ch) - energy
            channel = optimize.brentq(energy_diff, 0, len(self.channels))
        
        return int(np.round(channel)) if np.isscalar(channel) else np.round(channel).astype(int)
    
    def counts_in_range(
        self,
        e_min: float,
        e_max: float,
        use_energy: bool = True
    ) -> Tuple[float, float]:
        """
        Get total counts and uncertainty in energy/channel range.
        
        Parameters
        ----------
        e_min, e_max : float
            Range bounds (energy in keV if use_energy=True, else channels)
        use_energy : bool
            If True, interpret bounds as energy; else as channels
        
        Returns
        -------
        counts : float
            Total counts in range
        uncertainty : float
            Poisson uncertainty (sqrt(counts))
        """
        if use_energy:
            ch_min = self.energy_to_channel(e_min)
            ch_max = self.energy_to_channel(e_max)
        else:
            ch_min, ch_max = int(e_min), int(e_max)
        
        mask = (self.channels >= ch_min) & (self.channels <= ch_max)
        total = self.counts[mask].sum()
        
        return float(total), np.sqrt(total)
    
    @property
    def dead_time_fraction(self) -> float:
        """Calculate dead time fraction."""
        if self.real_time > 0:
            return 1.0 - (self.live_time / self.real_time)
        return 0.0
    
    @property
    def count_rate(self) -> float:
        """Calculate total count rate in counts per second."""
        if self.live_time > 0:
            return self.counts.sum() / self.live_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'counts': self.counts.tolist(),
            'channels': self.channels.tolist(),
            'energies': self.energies.tolist() if self.energies is not None else None,
            'live_time': self.live_time,
            'real_time': self.real_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'spectrum_id': self.spectrum_id,
            'detector_id': self.detector_id,
            'calibration': self.calibration,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GammaSpectrum':
        """Create GammaSpectrum from dictionary."""
        return cls(
            counts=np.array(data['counts']),
            channels=np.array(data.get('channels', [])),
            energies=np.array(data['energies']) if data.get('energies') else None,
            live_time=data.get('live_time', 0.0),
            real_time=data.get('real_time', 0.0),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            spectrum_id=data.get('spectrum_id', ''),
            detector_id=data.get('detector_id', ''),
            calibration=data.get('calibration', {}),
            metadata=data.get('metadata', {}),
        )


def read_spe_file(
    filepath: Union[str, Path],
    format_hint: Optional[str] = None
) -> GammaSpectrum:
    """
    Read SPE file and return GammaSpectrum object.
    
    Automatically detects SPE format (standard or "$" prefixed).
    
    Parameters
    ----------
    filepath : str or Path
        Path to SPE file
    format_hint : str, optional
        Force format: 'standard', 'dollar', or None for auto-detect
    
    Returns
    -------
    GammaSpectrum
        Parsed spectrum data
    
    Examples
    --------
    >>> spectrum = read_spe_file("Na22_calibration.spe")
    >>> print(f"Channels: {len(spectrum.counts)}")
    >>> print(f"Live time: {spectrum.live_time} s")
    
    Notes
    -----
    SPE file format sections:
    - $SPEC_ID: Spectrum identifier
    - $DATE_MEA: Measurement date
    - $MEAS_TIM: Live and real time
    - $DATA: Channel counts
    - $ENER_FIT: Energy calibration coefficients
    - $MCA_CAL: Multi-channel analyzer calibration
    - $SHAPE_CAL: Peak shape calibration
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()
    
    # Auto-detect format
    if format_hint is None:
        if content.lstrip().startswith('$'):
            format_hint = 'dollar'
        else:
            format_hint = 'standard'
    
    if format_hint == 'dollar':
        return _parse_dollar_spe(content, str(filepath))
    else:
        return _parse_standard_spe(content, str(filepath))


def _parse_dollar_spe(content: str, filename: str) -> GammaSpectrum:
    """
    Parse "$" prefixed SPE format (ORTEC MAESTRO style).
    
    Format example:
        $SPEC_ID:
        Sample spectrum
        $DATE_MEA:
        10/15/2024 14:30:00
        $MEAS_TIM:
        3600 3650
        $DATA:
        0 8191
        0
        15
        42
        ...
        $ENER_FIT:
        0.000000E+000 5.000000E-001
    """
    sections = {}
    current_section = None
    section_content = []
    
    for line in content.split('\n'):
        line = line.strip()
        
        if line.startswith('$'):
            # Save previous section
            if current_section:
                sections[current_section] = section_content
            
            # Start new section
            current_section = line.rstrip(':')
            section_content = []
        elif current_section:
            section_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = section_content
    
    # Parse spectrum ID
    spectrum_id = ''
    if '$SPEC_ID' in sections:
        spectrum_id = ' '.join(sections['$SPEC_ID']).strip()
    
    # Parse date
    start_time = None
    if '$DATE_MEA' in sections and sections['$DATE_MEA']:
        date_str = sections['$DATE_MEA'][0].strip()
        for fmt in [
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y %H:%M:%S',
        ]:
            try:
                start_time = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
    
    # Parse times
    live_time = 0.0
    real_time = 0.0
    if '$MEAS_TIM' in sections and sections['$MEAS_TIM']:
        times = sections['$MEAS_TIM'][0].split()
        if len(times) >= 2:
            live_time = float(times[0])
            real_time = float(times[1])
        elif len(times) == 1:
            live_time = real_time = float(times[0])
    
    # Parse data
    counts = []
    start_channel = 0
    end_channel = 0
    
    if '$DATA' in sections:
        data_lines = sections['$DATA']
        if data_lines:
            # First line has channel range
            range_parts = data_lines[0].split()
            if len(range_parts) >= 2:
                start_channel = int(range_parts[0])
                end_channel = int(range_parts[1])
            
            # Remaining lines are counts
            for line in data_lines[1:]:
                line = line.strip()
                if line:
                    try:
                        counts.append(float(line))
                    except ValueError:
                        continue
    
    counts = np.array(counts)
    channels = np.arange(start_channel, start_channel + len(counts))
    
    # Parse energy calibration
    calibration = {}
    if '$ENER_FIT' in sections and sections['$ENER_FIT']:
        coeffs = []
        for line in sections['$ENER_FIT']:
            coeffs.extend([float(x) for x in line.split() if x])
        calibration['energy'] = coeffs
    
    if '$MCA_CAL' in sections and sections['$MCA_CAL']:
        # Usually has number of coefficients on first line
        mca_lines = sections['$MCA_CAL']
        if len(mca_lines) > 1:
            coeffs = []
            for line in mca_lines[1:]:
                coeffs.extend([float(x) for x in line.split() if x])
            if 'energy' not in calibration:
                calibration['energy'] = coeffs
    
    if '$SHAPE_CAL' in sections and sections['$SHAPE_CAL']:
        coeffs = []
        for line in sections['$SHAPE_CAL']:
            coeffs.extend([float(x) for x in line.split() if x])
        calibration['shape'] = coeffs
    
    # Collect metadata
    metadata = {
        key.lstrip('$'): ' '.join(val)
        for key, val in sections.items()
        if key not in ['$DATA', '$ENER_FIT', '$MCA_CAL', '$SHAPE_CAL',
                       '$SPEC_ID', '$DATE_MEA', '$MEAS_TIM']
    }
    
    # Parse detector ID if available
    detector_id = ''
    if '$DET_ID' in sections:
        detector_id = ' '.join(sections['$DET_ID']).strip()
    
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        spectrum_id=spectrum_id or Path(filename).stem,
        detector_id=detector_id,
        calibration=calibration,
        metadata=metadata,
    )


def _parse_standard_spe(content: str, filename: str) -> GammaSpectrum:
    """
    Parse standard SPE format (numeric header style).
    
    Format varies but typically:
    - Header lines with counts, times
    - Followed by channel data
    """
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    
    counts = []
    live_time = 0.0
    real_time = 0.0
    calibration = {}
    metadata = {}
    
    in_data = False
    header_complete = False
    
    for i, line in enumerate(lines):
        # Try to parse as a count value
        try:
            val = float(line)
            if header_complete:
                counts.append(val)
            else:
                # Still in header - might be timing info
                if len(counts) == 0:
                    # Could be various header formats
                    pass
            continue
        except ValueError:
            pass
        
        # Check for keywords
        line_lower = line.lower()
        
        if 'live' in line_lower and 'time' in line_lower:
            match = re.search(r'[\d.]+', line)
            if match:
                live_time = float(match.group())
        elif 'real' in line_lower and 'time' in line_lower:
            match = re.search(r'[\d.]+', line)
            if match:
                real_time = float(match.group())
        elif 'calibration' in line_lower or 'energy' in line_lower:
            # Look for coefficients on this or next line
            coeffs = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if coeffs:
                calibration['energy'] = [float(c) for c in coeffs]
        
        # Detect start of data section
        if any(marker in line_lower for marker in ['data:', 'counts:', 'spectrum:']):
            header_complete = True
            in_data = True
    
    # If we didn't find explicit counts, try parsing all numeric lines
    if len(counts) == 0:
        for line in lines:
            try:
                counts.append(float(line))
            except ValueError:
                continue
    
    counts = np.array(counts)
    channels = np.arange(len(counts))
    
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        live_time=live_time,
        real_time=real_time,
        spectrum_id=Path(filename).stem,
        calibration=calibration,
        metadata=metadata,
    )


def write_spe_file(
    spectrum: GammaSpectrum,
    filepath: Union[str, Path],
    format_type: str = 'dollar'
) -> None:
    """
    Write GammaSpectrum to SPE file.
    
    Parameters
    ----------
    spectrum : GammaSpectrum
        Spectrum to write
    filepath : str or Path
        Output file path
    format_type : str
        Format type: 'dollar' for ORTEC-style
    """
    filepath = Path(filepath)
    
    lines = []
    
    # Spectrum ID
    lines.append('$SPEC_ID:')
    lines.append(spectrum.spectrum_id or filepath.stem)
    
    # Detector ID
    if spectrum.detector_id:
        lines.append('$DET_ID:')
        lines.append(spectrum.detector_id)
    
    # Date
    lines.append('$DATE_MEA:')
    if spectrum.start_time:
        lines.append(spectrum.start_time.strftime('%m/%d/%Y %H:%M:%S'))
    else:
        lines.append(datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
    
    # Times
    lines.append('$MEAS_TIM:')
    lines.append(f'{spectrum.live_time:.0f} {spectrum.real_time:.0f}')
    
    # Data
    lines.append('$DATA:')
    start_ch = int(spectrum.channels[0]) if len(spectrum.channels) > 0 else 0
    end_ch = int(spectrum.channels[-1]) if len(spectrum.channels) > 0 else len(spectrum.counts) - 1
    lines.append(f'{start_ch} {end_ch}')
    
    for count in spectrum.counts:
        lines.append(f'{int(count)}')
    
    # Energy calibration
    if 'energy' in spectrum.calibration:
        lines.append('$ENER_FIT:')
        coeffs = spectrum.calibration['energy']
        lines.append(' '.join(f'{c:.6E}' for c in coeffs))
        
        lines.append('$MCA_CAL:')
        lines.append(str(len(coeffs)))
        lines.append(' '.join(f'{c:.6E}' for c in coeffs) + ' keV')
    
    # Shape calibration
    if 'shape' in spectrum.calibration:
        lines.append('$SHAPE_CAL:')
        coeffs = spectrum.calibration['shape']
        lines.append(str(len(coeffs)))
        lines.append(' '.join(f'{c:.6E}' for c in coeffs))
    
    # End marker
    lines.append('$ENDRECORD:')
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def read_multiple_spe(
    filepaths: List[Union[str, Path]],
    sum_spectra: bool = False
) -> Union[List[GammaSpectrum], GammaSpectrum]:
    """
    Read multiple SPE files.
    
    Parameters
    ----------
    filepaths : list of str or Path
        Paths to SPE files
    sum_spectra : bool
        If True, return summed spectrum
    
    Returns
    -------
    list of GammaSpectrum or GammaSpectrum
        Parsed spectra (or summed if sum_spectra=True)
    """
    spectra = [read_spe_file(fp) for fp in filepaths]
    
    if sum_spectra and len(spectra) > 0:
        # Sum counts, add times
        total_counts = np.zeros_like(spectra[0].counts)
        total_live = 0.0
        total_real = 0.0
        
        for sp in spectra:
            if len(sp.counts) == len(total_counts):
                total_counts += sp.counts
            total_live += sp.live_time
            total_real += sp.real_time
        
        return GammaSpectrum(
            counts=total_counts,
            channels=spectra[0].channels.copy(),
            live_time=total_live,
            real_time=total_real,
            spectrum_id='summed_spectrum',
            calibration=spectra[0].calibration.copy(),
            metadata={'source_files': [str(fp) for fp in filepaths]},
        )
    
    return spectra
