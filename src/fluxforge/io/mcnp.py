"""MCNP Input/Output Module.

Handles parsing of MCNP input files and reading of tally data (HDF5).
"""

import re
import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def parse_mcnp_input(input_path: str) -> Dict[str, Any]:
    """
    Parse MCNP input file to extract material definitions and densities.
    
    Returns:
        Dict with keys:
        - materials: Dict[int, Dict] (material ID -> properties)
        - cells: Dict[int, Dict] (cell ID -> properties)
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    materials = {}
    current_mat = None
    
    # Simple parser for material cards (mXXXX)
    for raw_line in lines:
        line = raw_line.strip()
        # Remove comments
        if '$' in line:
            line = line.split('$')[0].strip()
            
        if not line:
            continue
            
        # Check for material card
        # Regex explanation:
        # ^[mM] -> starts with m or M
        # (\d+) -> captures the material ID
        # (\s+.*)?$ -> optionally captures whitespace and following content
        m_match = re.match(r'^[mM](\d+)(\s+.*)?$', line)
        if m_match:
            mat_id = int(m_match.group(1))
            content = m_match.group(2).strip() if m_match.group(2) else ""
            current_mat = mat_id
            materials[mat_id] = {'id': mat_id, 'lines': [content] if content else []}
            continue
            
        # Continuation lines (5 spaces or &) - simplified
        # Check indentation on raw_line before stripping
        if current_mat and (raw_line.startswith('     ') or raw_line.strip().endswith('&')):
            materials[current_mat]['lines'].append(line)
            
    # Process material lines
    for mat_id, data in materials.items():
        full_str = ' '.join(data['lines'])
        # Parse isotopes and fractions (simplified)
        components = []
        parts = full_str.split()
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                zaid = parts[i]
                frac = parts[i+1]
                components.append((zaid, frac))
        data['components'] = components
        
    return {'materials': materials}


def read_meshtal_hdf5(h5_path: str, tally_id: int) -> Dict[str, Any]:
    """
    Read mesh tally data from MCNP HDF5 output (meshtal).
    
    Args:
        h5_path: Path to .h5 file (e.g. runtpe.h5 or meshtal.h5)
        tally_id: Tally number to extract
        
    Returns:
        Dict containing:
        - flux: numpy array of flux values
        - error: numpy array of relative errors
        - energy_boundaries: numpy array
        - mesh_bounds: Dict with x, y, z boundaries
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required to read MCNP HDF5 files.")
        
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
    results = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Navigate to tally group
        # Structure varies by MCNP version, this is a common one
        tally_path = f"tallies/tally_{tally_id}"
        if tally_path not in f:
            # Try searching
            found = False
            if 'tallies' in f:
                for key in f['tallies'].keys():
                    if str(tally_id) in key:
                        tally_path = f"tallies/{key}"
                        found = True
                        break
            if not found:
                raise ValueError(f"Tally {tally_id} not found in {h5_path}")
                
        grp = f[tally_path]
        
        # Read flux and error
        # Shape is typically (n_energy, n_time, nx, ny, nz)
        if 'results' in grp:
            # MCNP6.2+ format often puts results in a dataset
            data = grp['results'][:]
            # Assuming flux is first value, error is second (if present)
            # This depends heavily on MCNP version/settings
            results['flux'] = data[:, :, :, :, :, 0] # Extract flux
            if data.shape[-1] > 1:
                results['error'] = data[:, :, :, :, :, 1]
        elif 'flux' in grp:
            results['flux'] = grp['flux'][:]
            if 'relative_error' in grp:
                results['error'] = grp['relative_error'][:]
                
        # Read energy boundaries
        if 'energy_bins' in grp:
            results['energy_boundaries'] = grp['energy_bins'][:]
            
        # Read mesh boundaries
        if 'spatial_bins' in grp:
            # This might need adjustment based on exact structure
            pass
            
    return results


# =============================================================================
# CSV Spectrum Reader (for processed MCNP spectrum exports)
# =============================================================================

@dataclass
class MCNPSpectrum:
    """Container for an MCNP-derived neutron or photon spectrum.

    Attributes:
        energy_low: Lower energy bounds per group (MeV)
        energy_high: Upper energy bounds per group (MeV)
        flux: Group flux values (per source particle unless normalized)
        uncertainty: Relative uncertainty per group (fraction, not %)
        tally_id: MCNP tally identifier
        energy_units: 'eV' or 'MeV'
        normalization: Description of normalization
        metadata: Additional metadata
    """
    energy_low: np.ndarray
    energy_high: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    tally_id: str = ""
    energy_units: str = "MeV"
    normalization: str = "per_source_particle"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.energy_low = np.asarray(self.energy_low, dtype=float)
        self.energy_high = np.asarray(self.energy_high, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.uncertainty = np.asarray(self.uncertainty, dtype=float)

    @property
    def n_groups(self) -> int:
        return len(self.flux)

    @property
    def energy_mid(self) -> np.ndarray:
        return np.sqrt(self.energy_low * self.energy_high)

    @property
    def energy_bounds(self) -> np.ndarray:
        """Return energy boundaries array (n_groups + 1)."""
        return np.concatenate([self.energy_low, [self.energy_high[-1]]])

    def to_eV(self) -> "MCNPSpectrum":
        """Return a copy with energies converted to eV."""
        if self.energy_units == "eV":
            return self
        factor = 1e6  # MeV -> eV
        return MCNPSpectrum(
            energy_low=self.energy_low * factor,
            energy_high=self.energy_high * factor,
            flux=self.flux.copy(),
            uncertainty=self.uncertainty.copy(),
            tally_id=self.tally_id,
            energy_units="eV",
            normalization=self.normalization,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_low": self.energy_low.tolist(),
            "energy_high": self.energy_high.tolist(),
            "flux": self.flux.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "tally_id": self.tally_id,
            "energy_units": self.energy_units,
            "normalization": self.normalization,
            "metadata": self.metadata,
        }


from dataclasses import dataclass, field


def read_mcnp_spectrum_csv(
    filepath: Union[str, Path],
    *,
    energy_units: str = "MeV",
    delimiter: str = ",",
    skip_header: int = 0,
) -> MCNPSpectrum:
    """Read an MCNP spectrum from a CSV file.

    Expected CSV format (no header or header skipped):
        E_low, E_high, flux_zone1, unc_zone1, flux_zone2, unc_zone2, ...

    For single-zone files, columns are: E_low, E_high, flux, unc
    For multi-zone files, returns the volume-weighted or first zone average.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    energy_units : str
        Units of energy columns ('eV' or 'MeV')
    delimiter : str
        CSV delimiter
    skip_header : int
        Number of header lines to skip

    Returns
    -------
    MCNPSpectrum
        Parsed spectrum with flux and uncertainties
    """
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath

    rows: List[List[float]] = []
    with open(filepath, "r") as f:
        # Skip header lines
        for _ in range(skip_header):
            next(f, None)

        for line in f:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(delimiter)
                row = [float(x) for x in parts if x.strip()]
                if row:
                    rows.append(row)
            except ValueError:
                continue  # Skip non-numeric lines

    if not rows:
        raise ValueError(f"No valid data rows found in {filepath}")

    data = np.array(rows)
    n_cols = data.shape[1]

    # Parse based on column count
    energy_low = data[:, 0]
    energy_high = data[:, 1]

    if n_cols == 4:
        # Simple format: E_low, E_high, flux, uncertainty
        flux = data[:, 2]
        uncertainty = data[:, 3]
    elif n_cols >= 6:
        # Multi-zone format: E_low, E_high, flux1, unc1, flux2, unc2, ...
        # Average across zones (simple mean for now)
        n_zones = (n_cols - 2) // 2
        flux_cols = data[:, 2::2]  # Every other column starting at 2
        unc_cols = data[:, 3::2]  # Every other column starting at 3

        # Simple average across zones
        flux = np.mean(flux_cols[:, :n_zones], axis=1)
        # Propagate uncertainties (relative -> absolute -> combine -> relative)
        abs_unc = flux_cols[:, :n_zones] * unc_cols[:, :n_zones]
        uncertainty = np.sqrt(np.sum(abs_unc**2, axis=1)) / (n_zones * np.maximum(flux, 1e-30))
    else:
        raise ValueError(f"Unexpected column count {n_cols} in {filepath}")

    return MCNPSpectrum(
        energy_low=energy_low,
        energy_high=energy_high,
        flux=flux,
        uncertainty=uncertainty,
        tally_id=filepath.stem,
        energy_units=energy_units,
        normalization="per_source_particle",
        metadata={"source_file": str(filepath), "n_groups": len(flux)},
    )


# =============================================================================
# MCTAL Text File Reader
# =============================================================================

@dataclass
class MCTALTally:
    """
    Single tally from an MCTAL file.
    
    Attributes:
        tally_id: Tally number (e.g., 4, 14, 24)
        particle_type: Particle type code (1=neutron, 2=photon, etc.)
        energy_bins: Energy bin boundaries (MeV)
        flux: Tally values
        uncertainty: Relative errors (fraction)
        cells: Cell list if cell tally
        surfaces: Surface list if surface tally
        times: Time bins if time-dependent
        comments: Tally comments from input
    """
    
    tally_id: int
    particle_type: int = 1
    energy_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    flux: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    cells: List[int] = field(default_factory=list)
    surfaces: List[int] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    comments: str = ""
    
    def to_spectrum(self, energy_units: str = "MeV") -> MCNPSpectrum:
        """Convert to MCNPSpectrum format."""
        if len(self.energy_bins) < 2:
            raise ValueError("Tally has insufficient energy bins for spectrum.")
        
        return MCNPSpectrum(
            energy_low=self.energy_bins[:-1],
            energy_high=self.energy_bins[1:],
            flux=self.flux.copy(),
            uncertainty=self.uncertainty.copy(),
            tally_id=str(self.tally_id),
            energy_units=energy_units,
            normalization="per_source_particle",
            metadata={
                "particle_type": self.particle_type,
                "cells": self.cells,
            },
        )


@dataclass
class MCTALFile:
    """
    Parsed MCTAL file containing multiple tallies.
    
    Attributes:
        code: MCNP version code
        version: Version string
        date: Date of run
        time: Time of run
        nps: Number of particles simulated
        histories: Number of histories (random walks)
        tallies: Dict of tally_id -> MCTALTally
        keff: k-effective if criticality problem
        comments: File header comments
    """
    
    code: str = "MCNP"
    version: str = ""
    date: str = ""
    time: str = ""
    nps: int = 0
    histories: int = 0
    tallies: Dict[int, MCTALTally] = field(default_factory=dict)
    keff: Optional[Tuple[float, float]] = None  # (value, std_dev)
    comments: str = ""
    
    def get_tally(self, tally_id: int) -> MCTALTally:
        """Get tally by ID."""
        if tally_id not in self.tallies:
            raise KeyError(f"Tally {tally_id} not found. Available: {list(self.tallies.keys())}")
        return self.tallies[tally_id]
    
    def list_tallies(self) -> List[int]:
        """List all tally IDs."""
        return list(self.tallies.keys())


def read_mctal(filepath: Union[str, Path]) -> MCTALFile:
    """
    Read an MCNP MCTAL file.
    
    The MCTAL file is MCNP's standard ASCII output for tally results.
    This parser handles F4/F5/F6 tallies with energy bins.
    
    Parameters
    ----------
    filepath : str or Path
        Path to MCTAL file.
        
    Returns
    -------
    MCTALFile
        Parsed MCTAL contents with all tallies.
        
    Examples
    --------
    >>> mctal = read_mctal("mctal")
    >>> tally = mctal.get_tally(14)
    >>> spectrum = tally.to_spectrum()
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MCTAL file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    mctal = MCTALFile()
    
    if not lines:
        return mctal
    
    # Parse header line
    # Format: "code version date time"
    header = lines[0].strip().split()
    if header:
        mctal.code = header[0] if len(header) > 0 else ""
        mctal.version = header[1] if len(header) > 1 else ""
        mctal.date = header[2] if len(header) > 2 else ""
        mctal.time = header[3] if len(header) > 3 else ""
    
    # Parse file
    i = 1  # Current line index
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for nps line
        if line.startswith("nps"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mctal.nps = int(parts[1])
                except ValueError:
                    pass
            i += 1
            continue
        
        # Look for tally block
        if line.startswith("tally"):
            # Parse tally header
            parts = line.split()
            if len(parts) >= 2:
                tally_id = int(parts[1])
                particle_type = int(parts[2]) if len(parts) > 2 else 1
                
                tally = MCTALTally(tally_id=tally_id, particle_type=particle_type)
                
                i += 1
                # Parse tally content until next "tally" or end
                i = _parse_tally_block(lines, i, tally)
                
                mctal.tallies[tally_id] = tally
                continue
        
        # Look for kcode results
        if line.startswith("kcode"):
            i += 1
            if i < len(lines):
                kline = lines[i].strip().split()
                if len(kline) >= 2:
                    try:
                        mctal.keff = (float(kline[0]), float(kline[1]))
                    except ValueError:
                        pass
            i += 1
            continue
        
        i += 1
    
    return mctal


def _parse_tally_block(lines: List[str], start_idx: int, tally: MCTALTally) -> int:
    """
    Parse a tally block starting after the tally header.
    
    Returns the index after the tally block ends.
    """
    i = start_idx
    n = len(lines)
    
    # State flags
    in_vals = False
    in_tfc = False
    
    energy_bins = []
    values = []
    errors = []
    cells = []
    
    while i < n:
        line = lines[i].strip()
        
        # Check for next tally or end
        if line.startswith("tally") or line.startswith("kcode"):
            break
        
        # Parse 'f' line (cells/surfaces)
        if line.startswith("f"):
            parts = line.split()
            for p in parts[1:]:
                try:
                    cells.append(int(p))
                except ValueError:
                    pass
            i += 1
            continue
        
        # Parse 'e' line (energy bins)
        if line.startswith("e"):
            parts = line.split()
            for p in parts[1:]:
                try:
                    energy_bins.append(float(p))
                except ValueError:
                    pass
            i += 1
            continue
        
        # Parse 't' line (time bins)
        if line.startswith("t "):
            i += 1
            continue
        
        # Parse 'c' line (cosine bins)
        if line.startswith("c "):
            i += 1
            continue
        
        # Parse 'vals' block
        if line.startswith("vals"):
            in_vals = True
            i += 1
            continue
        
        # Parse 'tfc' block (end of tally data)
        if line.startswith("tfc"):
            in_tfc = True
            i += 1
            # Skip the tfc data
            while i < n:
                if lines[i].strip().startswith("tally") or lines[i].strip().startswith("kcode"):
                    break
                i += 1
            break
        
        # In vals block - read value/error pairs
        if in_vals:
            parts = line.split()
            for j in range(0, len(parts), 2):
                if j + 1 < len(parts):
                    try:
                        values.append(float(parts[j]))
                        errors.append(float(parts[j + 1]))
                    except ValueError:
                        pass
        
        i += 1
    
    # Assign parsed data to tally
    tally.cells = cells
    
    if energy_bins:
        tally.energy_bins = np.array(energy_bins)
    
    if values:
        tally.flux = np.array(values)
        tally.uncertainty = np.array(errors)
    
    return i


def read_mcnp_flux_tally(
    mctal_path: Union[str, Path],
    tally_id: int,
) -> MCNPSpectrum:
    """
    Read a flux tally from MCTAL file and return as spectrum.
    
    Convenience function combining read_mctal() and tally.to_spectrum().
    
    Parameters
    ----------
    mctal_path : str or Path
        Path to MCTAL file.
    tally_id : int
        Tally ID to extract (e.g., 4, 14, 24).
        
    Returns
    -------
    MCNPSpectrum
        Flux spectrum from the specified tally.
    """
    mctal = read_mctal(mctal_path)
    tally = mctal.get_tally(tally_id)
    return tally.to_spectrum()
