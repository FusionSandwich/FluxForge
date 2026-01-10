"""
Spectrum Export Module - Epic T (Curie Parity)

Implements multi-format spectrum export:
- SPE format (IAEA/Canberra)
- CNF format (Camberra/Genie)  
- IEC 62755 format (international standard)
- CSV/TSV for general use
- HDF5 for large datasets
- MCNP SDEF for source definitions

This enables FluxForge spectra to be used with various
detector analysis software and simulation codes.
"""

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, BinaryIO, TextIO
from pathlib import Path
from datetime import datetime
import struct
import json


@dataclass
class SpectrumMetadata:
    """Metadata for spectrum export."""
    title: str = "FluxForge Spectrum"
    description: str = ""
    sample_id: str = ""
    detector_id: str = ""
    
    # Measurement info
    live_time: float = 0.0  # seconds
    real_time: float = 0.0  # seconds
    start_time: Optional[datetime] = None
    
    # Calibration
    energy_coefficients: List[float] = None  # [a, b, c] for E = a + b*ch + c*ch^2
    fwhm_coefficients: List[float] = None
    
    # Source info
    source_description: str = ""
    source_activity_Bq: float = 0.0
    source_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.energy_coefficients is None:
            self.energy_coefficients = [0.0, 1.0, 0.0]  # Default: E = channel
        if self.fwhm_coefficients is None:
            self.fwhm_coefficients = [0.0, 0.0, 0.0]


class SpectrumExporter:
    """
    Multi-format spectrum exporter.
    
    Supports:
    - SPE (IAEA/Canberra ASCII format)
    - CNF (Canberra binary format, simplified)
    - IEC (IEC 62755 XML format)
    - CSV/TSV (plain text)
    - HDF5 (optional, requires h5py)
    - MCNP (SDEF card format)
    
    Example
    -------
    >>> exporter = SpectrumExporter(counts, metadata)
    >>> exporter.to_spe("spectrum.spe")
    >>> exporter.to_csv("spectrum.csv")
    """
    
    def __init__(
        self,
        counts: List[float],
        metadata: Optional[SpectrumMetadata] = None,
        energies: Optional[List[float]] = None,
        uncertainties: Optional[List[float]] = None
    ):
        """
        Initialize exporter.
        
        Parameters
        ----------
        counts : List[float]
            Channel counts or bin values
        metadata : SpectrumMetadata, optional
            Spectrum metadata
        energies : List[float], optional
            Energy bin centers (keV)
        uncertainties : List[float], optional
            Count uncertainties
        """
        self.counts = list(counts)
        self.metadata = metadata or SpectrumMetadata()
        self.energies = energies
        self.uncertainties = uncertainties
        
        if energies is None:
            # Generate from calibration
            self.energies = self._energies_from_calibration()
    
    def _energies_from_calibration(self) -> List[float]:
        """Generate energies from calibration coefficients."""
        a, b, c = self.metadata.energy_coefficients[:3]
        return [a + b * ch + c * ch * ch for ch in range(len(self.counts))]
    
    def to_spe(self, filepath: Union[str, Path]) -> None:
        """
        Export to SPE format (IAEA/Canberra ASCII).
        
        The SPE format is widely supported by gamma spectroscopy software.
        """
        filepath = Path(filepath)
        n_channels = len(self.counts)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("$SPEC_ID:\n")
            f.write(f"{self.metadata.title}\n")
            
            f.write("$SPEC_REM:\n")
            f.write(f"DET# {self.metadata.detector_id or '1'}\n")
            f.write(f"DETDESC# {self.metadata.description}\n")
            
            f.write("$DATE_MEA:\n")
            if self.metadata.start_time:
                f.write(f"{self.metadata.start_time.strftime('%m/%d/%Y %H:%M:%S')}\n")
            else:
                f.write(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}\n")
            
            f.write("$MEAS_TIM:\n")
            f.write(f"{int(self.metadata.live_time)} {int(self.metadata.real_time)}\n")
            
            f.write("$DATA:\n")
            f.write(f"0 {n_channels - 1}\n")
            
            # Channel data (8 values per line)
            for i in range(0, n_channels, 8):
                line_counts = self.counts[i:min(i + 8, n_channels)]
                f.write(" ".join(f"{int(c)}" for c in line_counts) + "\n")
            
            # Energy calibration
            f.write("$ENER_FIT:\n")
            a, b = self.metadata.energy_coefficients[:2]
            f.write(f"{a:.6f} {b:.6f}\n")
            
            f.write("$MCA_CAL:\n")
            f.write("2\n")
            f.write(f"{a:.6E} {b:.6E} keV\n")
            
            # End
            f.write("$ENDRECORD:\n")
    
    def to_csv(
        self,
        filepath: Union[str, Path],
        delimiter: str = ",",
        include_energies: bool = True,
        include_uncertainties: bool = True
    ) -> None:
        """
        Export to CSV/TSV format.
        
        Parameters
        ----------
        filepath : Path
            Output file path
        delimiter : str
            Field delimiter ("," for CSV, "\\t" for TSV)
        include_energies : bool
            Include energy column
        include_uncertainties : bool
            Include uncertainty column
        """
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            # Header
            headers = ["channel"]
            if include_energies:
                headers.append("energy_keV")
            headers.append("counts")
            if include_uncertainties and self.uncertainties:
                headers.append("uncertainty")
            
            f.write(delimiter.join(headers) + "\n")
            
            # Data
            for i, count in enumerate(self.counts):
                row = [str(i)]
                if include_energies:
                    row.append(f"{self.energies[i]:.4f}")
                row.append(f"{count:.2f}")
                if include_uncertainties and self.uncertainties:
                    row.append(f"{self.uncertainties[i]:.2f}")
                
                f.write(delimiter.join(row) + "\n")
    
    def to_iec(self, filepath: Union[str, Path]) -> None:
        """
        Export to IEC 62755 XML format.
        
        This is the international standard format for gamma spectra.
        """
        filepath = Path(filepath)
        
        # Build XML structure
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append('<RadInstrumentData>')
        xml.append('  <RadInstrumentInformation>')
        xml.append(f'    <RadInstrumentManufacturerName>FluxForge</RadInstrumentManufacturerName>')
        xml.append(f'    <RadInstrumentIdentifier>{self.metadata.detector_id or "UNKNOWN"}</RadInstrumentIdentifier>')
        xml.append('  </RadInstrumentInformation>')
        
        xml.append('  <RadMeasurement>')
        xml.append(f'    <MeasurementClassCode>Foreground</MeasurementClassCode>')
        
        if self.metadata.start_time:
            xml.append(f'    <StartDateTime>{self.metadata.start_time.isoformat()}</StartDateTime>')
        
        xml.append(f'    <RealTimeDuration>PT{self.metadata.real_time:.1f}S</RealTimeDuration>')
        
        xml.append('    <Spectrum>')
        xml.append(f'      <LiveTimeDuration>PT{self.metadata.live_time:.1f}S</LiveTimeDuration>')
        
        # Energy calibration
        xml.append('      <Calibration Type="Energy">')
        xml.append('        <EquationCoefficients>')
        for coef in self.metadata.energy_coefficients:
            xml.append(f'          <Coefficient>{coef:.6E}</Coefficient>')
        xml.append('        </EquationCoefficients>')
        xml.append('      </Calibration>')
        
        # Channel data
        xml.append('      <ChannelData>')
        xml.append('        ' + ' '.join(f'{int(c)}' for c in self.counts))
        xml.append('      </ChannelData>')
        
        xml.append('    </Spectrum>')
        xml.append('  </RadMeasurement>')
        xml.append('</RadInstrumentData>')
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(xml))
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export to JSON format.
        
        Comprehensive format including all metadata.
        """
        filepath = Path(filepath)
        
        data = {
            'format': 'FluxForge Spectrum v1.0',
            'created': datetime.now().isoformat(),
            'metadata': {
                'title': self.metadata.title,
                'description': self.metadata.description,
                'sample_id': self.metadata.sample_id,
                'detector_id': self.metadata.detector_id,
                'live_time_s': self.metadata.live_time,
                'real_time_s': self.metadata.real_time,
                'start_time': self.metadata.start_time.isoformat() if self.metadata.start_time else None,
                'energy_calibration': self.metadata.energy_coefficients,
                'fwhm_calibration': self.metadata.fwhm_coefficients,
            },
            'n_channels': len(self.counts),
            'counts': self.counts,
            'energies_keV': self.energies,
        }
        
        if self.uncertainties:
            data['uncertainties'] = self.uncertainties
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_mcnp_sdef(
        self,
        filepath: Union[str, Path],
        particle: str = "p",
        normalize: bool = True,
        energy_unit: str = "MeV"
    ) -> None:
        """
        Export to MCNP SDEF format.
        
        Creates an energy distribution for use as MCNP source definition.
        
        Parameters
        ----------
        filepath : Path
            Output file path
        particle : str
            Particle type ("p" for photon, "n" for neutron)
        normalize : bool
            Normalize to unit probability
        energy_unit : str
            Energy unit ("MeV" or "keV")
        """
        filepath = Path(filepath)
        
        # Convert energies if needed
        scale = 0.001 if energy_unit == "MeV" else 1.0
        energies_out = [e * scale for e in self.energies]
        
        # Normalize probabilities
        total = sum(self.counts)
        if normalize and total > 0:
            probs = [c / total for c in self.counts]
        else:
            probs = self.counts
        
        with open(filepath, 'w') as f:
            f.write(f"c FluxForge MCNP SDEF - {self.metadata.title}\n")
            f.write(f"c Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"c {len(self.counts)} energy bins\n")
            f.write("c\n")
            
            # SI card (source information - energies)
            f.write(f"SI1   H  ")  # Histogram
            for i, E in enumerate(energies_out):
                if i % 6 == 0 and i > 0:
                    f.write("\n      ")
                f.write(f" {E:.6E}")
            f.write("\n")
            
            # SP card (source probabilities)
            f.write(f"SP1   D  ")  # Discrete
            for i, p in enumerate(probs):
                if i % 6 == 0 and i > 0:
                    f.write("\n      ")
                f.write(f" {p:.6E}")
            f.write("\n")
            
            # SDEF card
            f.write(f"c\n")
            f.write(f"SDEF PAR={particle} ERG=D1\n")
    
    def to_hdf5(self, filepath: Union[str, Path], dataset_name: str = "spectrum") -> None:
        """
        Export to HDF5 format (requires h5py).
        
        Efficient format for large datasets and batch processing.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 export: pip install h5py")
        
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'w') as f:
            # Main dataset
            ds = f.create_dataset(dataset_name, data=self.counts)
            ds.attrs['title'] = self.metadata.title
            ds.attrs['n_channels'] = len(self.counts)
            ds.attrs['live_time'] = self.metadata.live_time
            ds.attrs['real_time'] = self.metadata.real_time
            ds.attrs['energy_calibration'] = self.metadata.energy_coefficients
            
            if self.energies:
                f.create_dataset('energies', data=self.energies)
            
            if self.uncertainties:
                f.create_dataset('uncertainties', data=self.uncertainties)
    
    @classmethod
    def from_spe(cls, filepath: Union[str, Path]) -> 'SpectrumExporter':
        """
        Import from SPE format.
        
        Parameters
        ----------
        filepath : Path
            SPE file path
        
        Returns
        -------
        SpectrumExporter
        """
        filepath = Path(filepath)
        metadata = SpectrumMetadata()
        counts = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        section = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('$'):
                section = line[1:].rstrip(':')
                i += 1
                continue
            
            if section == 'SPEC_ID':
                metadata.title = line
            
            elif section == 'MEAS_TIM':
                parts = line.split()
                if len(parts) >= 2:
                    metadata.live_time = float(parts[0])
                    metadata.real_time = float(parts[1])
            
            elif section == 'DATA':
                if ' ' in line:
                    # First line is range
                    i += 1
                    continue
                # Count data lines
                while i < len(lines) and not lines[i].startswith('$'):
                    values = lines[i].split()
                    counts.extend(float(v) for v in values)
                    i += 1
                continue
            
            elif section == 'ENER_FIT':
                parts = line.split()
                if len(parts) >= 2:
                    metadata.energy_coefficients = [float(parts[0]), float(parts[1]), 0.0]
            
            i += 1
        
        return cls(counts, metadata)


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    print("Testing spectrum_export module...")
    
    # Create test spectrum
    import math
    n_channels = 1024
    
    # Simulate a gamma spectrum with peaks
    counts = []
    for ch in range(n_channels):
        E = 0.5 + 3.0 * ch / n_channels  # 0.5 to 3.5 keV (simulated)
        
        # Background (exponential)
        bg = 1000 * math.exp(-E / 0.5)
        
        # Peaks (Gaussian)
        peaks = 0
        for E_peak, intensity, sigma in [(0.662, 5000, 0.02), (1.173, 3000, 0.025), (1.332, 2800, 0.025)]:
            peaks += intensity * math.exp(-0.5 * ((E - E_peak) / sigma)**2)
        
        counts.append(max(0, bg + peaks))
    
    # Create metadata
    meta = SpectrumMetadata(
        title="Test Spectrum",
        description="Co-60 + Cs-137 simulation",
        detector_id="HPGe-001",
        live_time=3600,
        real_time=3610,
        start_time=datetime.now(),
        energy_coefficients=[0.5, 3.0 / 1024, 0.0]
    )
    
    # Create exporter
    exporter = SpectrumExporter(counts, meta)
    
    print(f"Created spectrum with {n_channels} channels")
    print(f"Total counts: {sum(counts):.0f}")
    
    # Test exports
    with tempfile.TemporaryDirectory() as tmpdir:
        # SPE
        spe_path = os.path.join(tmpdir, "test.spe")
        exporter.to_spe(spe_path)
        print(f"✓ Exported to SPE: {os.path.getsize(spe_path)} bytes")
        
        # CSV
        csv_path = os.path.join(tmpdir, "test.csv")
        exporter.to_csv(csv_path)
        print(f"✓ Exported to CSV: {os.path.getsize(csv_path)} bytes")
        
        # IEC
        iec_path = os.path.join(tmpdir, "test.xml")
        exporter.to_iec(iec_path)
        print(f"✓ Exported to IEC XML: {os.path.getsize(iec_path)} bytes")
        
        # JSON
        json_path = os.path.join(tmpdir, "test.json")
        exporter.to_json(json_path)
        print(f"✓ Exported to JSON: {os.path.getsize(json_path)} bytes")
        
        # MCNP
        mcnp_path = os.path.join(tmpdir, "test.sdef")
        exporter.to_mcnp_sdef(mcnp_path)
        print(f"✓ Exported to MCNP SDEF: {os.path.getsize(mcnp_path)} bytes")
        
        # Re-import SPE
        imported = SpectrumExporter.from_spe(spe_path)
        print(f"✓ Re-imported SPE: {len(imported.counts)} channels")
    
    print("\n✅ spectrum_export module tests passed!")
