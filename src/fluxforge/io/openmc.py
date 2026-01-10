"""
OpenMC Input/Output Module.

Handles reading of OpenMC statepoint files for tally extraction,
with support for flux spectra, energy filters, and uncertainty propagation.

Features:
- Statepoint HDF5 tally extraction
- Flux spectrum extraction with energy bins
- Multi-cell/region support
- Uncertainty handling
- Conversion to FluxForge spectrum format

References:
- OpenMC Documentation: https://docs.openmc.org/
- OpenMC Python API: https://docs.openmc.org/en/stable/pythonapi/

Author: FluxForge Development Team
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import openmc
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False


@dataclass
class OpenMCSpectrum:
    """
    Neutron flux spectrum extracted from OpenMC statepoint.
    
    Attributes:
        energy_bins_ev: Energy bin boundaries in eV (n+1 values for n groups)
        flux: Flux values per energy group
        uncertainty: Standard deviation of flux values
        cell_id: Cell ID where spectrum was tallied (optional)
        score: OpenMC score type (flux, current, etc.)
        n_particles: Number of particles simulated
        metadata: Additional tally metadata
    """
    
    energy_bins_ev: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    cell_id: Optional[int] = None
    score: str = "flux"
    n_particles: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_groups(self) -> int:
        """Number of energy groups."""
        return len(self.flux)
    
    @property
    def group_centers_ev(self) -> np.ndarray:
        """Geometric mean of energy bin boundaries."""
        return np.sqrt(self.energy_bins_ev[:-1] * self.energy_bins_ev[1:])
    
    @property
    def lethargy_widths(self) -> np.ndarray:
        """Lethargy width of each group."""
        return np.log(self.energy_bins_ev[1:] / self.energy_bins_ev[:-1])
    
    @property
    def relative_uncertainty(self) -> np.ndarray:
        """Relative uncertainty (σ/μ)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = self.uncertainty / self.flux
            rel[~np.isfinite(rel)] = 0.0
        return rel
    
    def to_per_unit_lethargy(self) -> OpenMCSpectrum:
        """Convert flux to per-unit-lethargy form."""
        du = self.lethargy_widths
        return OpenMCSpectrum(
            energy_bins_ev=self.energy_bins_ev.copy(),
            flux=self.flux / du,
            uncertainty=self.uncertainty / du,
            cell_id=self.cell_id,
            score=self.score,
            n_particles=self.n_particles,
            metadata=self.metadata.copy(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'energy_bins_ev': self.energy_bins_ev.tolist(),
            'flux': self.flux.tolist(),
            'uncertainty': self.uncertainty.tolist(),
            'cell_id': self.cell_id,
            'score': self.score,
            'n_particles': self.n_particles,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OpenMCSpectrum:
        """Create from dictionary."""
        return cls(
            energy_bins_ev=np.array(data['energy_bins_ev']),
            flux=np.array(data['flux']),
            uncertainty=np.array(data['uncertainty']),
            cell_id=data.get('cell_id'),
            score=data.get('score', 'flux'),
            n_particles=data.get('n_particles', 0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class StatepointInfo:
    """
    Summary information from an OpenMC statepoint file.
    
    Attributes:
        version: OpenMC version
        n_particles: Number of particles simulated
        n_batches: Number of batches
        n_inactive: Number of inactive batches
        k_eff: Effective multiplication factor (if criticality)
        tally_ids: List of tally IDs in the statepoint
        date_time: Date/time string
    """
    
    version: str = ""
    n_particles: int = 0
    n_batches: int = 0
    n_inactive: int = 0
    k_eff: Optional[Tuple[float, float]] = None  # (mean, std_dev)
    tally_ids: List[int] = field(default_factory=list)
    date_time: str = ""


def read_statepoint_info(statepoint_path: Union[str, Path]) -> StatepointInfo:
    """
    Read summary information from an OpenMC statepoint file.
    
    Parameters
    ----------
    statepoint_path : str or Path
        Path to statepoint.h5 file.
        
    Returns
    -------
    StatepointInfo
        Summary information about the simulation.
        
    Raises
    ------
    ImportError
        If h5py is not available.
    FileNotFoundError
        If statepoint file doesn't exist.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required to read OpenMC statepoint files.")
    
    statepoint_path = Path(statepoint_path)
    if not statepoint_path.exists():
        raise FileNotFoundError(f"Statepoint file not found: {statepoint_path}")
    
    info = StatepointInfo()
    
    with h5py.File(statepoint_path, 'r') as f:
        # Read version
        if 'version' in f.attrs:
            version = f.attrs['version']
            if isinstance(version, bytes):
                info.version = version.decode('utf-8')
            elif hasattr(version, '__iter__'):
                info.version = '.'.join(str(v) for v in version)
            else:
                info.version = str(version)
        
        # Read simulation info
        if 'n_particles' in f.attrs:
            info.n_particles = int(f.attrs['n_particles'])
        if 'n_batches' in f.attrs:
            info.n_batches = int(f.attrs['n_batches'])
        if 'n_inactive' in f.attrs:
            info.n_inactive = int(f.attrs['n_inactive'])
        if 'date_and_time' in f.attrs:
            dt = f.attrs['date_and_time']
            if isinstance(dt, bytes):
                info.date_time = dt.decode('utf-8')
            else:
                info.date_time = str(dt)
        
        # Read k-effective if available
        if 'k_combined' in f:
            k_data = f['k_combined'][()]
            if len(k_data) >= 2:
                info.k_eff = (float(k_data[0]), float(k_data[1]))
        
        # Read tally IDs
        if 'tallies' in f:
            for key in f['tallies'].keys():
                if key.startswith('tally '):
                    try:
                        tally_id = int(key.split()[-1])
                        info.tally_ids.append(tally_id)
                    except ValueError:
                        pass
    
    return info


def read_openmc_tally(
    statepoint_path: Union[str, Path],
    tally_id: int,
) -> Dict[str, Any]:
    """
    Read tally data from OpenMC statepoint file (HDF5).
    
    Parameters
    ----------
    statepoint_path : str or Path
        Path to statepoint.h5 file.
    tally_id : int
        ID of the tally to extract.
        
    Returns
    -------
    dict
        Dictionary containing:
        - data: Raw tally data array
        - mean: Mean values (if extractable)
        - std_dev: Standard deviations (if extractable)
        - filters: Filter information
        - nuclides: Nuclide information
        - scores: Score types
        
    Raises
    ------
    ImportError
        If h5py is not available.
    FileNotFoundError
        If statepoint file doesn't exist.
    ValueError
        If tally not found.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required to read OpenMC statepoint files.")
    
    statepoint_path = Path(statepoint_path)
    if not statepoint_path.exists():
        raise FileNotFoundError(f"Statepoint file not found: {statepoint_path}")
    
    results = {}
    
    with h5py.File(statepoint_path, 'r') as f:
        if 'tallies' not in f:
            raise ValueError("No tallies found in statepoint file.")
        
        # Find tally
        tally_key = f"tally {tally_id}"
        tally_group = None
        
        if tally_key in f['tallies']:
            tally_group = f['tallies'][tally_key]
        else:
            # Search by ID
            for key in f['tallies'].keys():
                if key.split()[-1] == str(tally_id):
                    tally_group = f['tallies'][key]
                    break
        
        if tally_group is None:
            raise ValueError(f"Tally ID {tally_id} not found in statepoint.")
        
        # Read results
        if 'results' in tally_group:
            data = tally_group['results'][()]
            results['data'] = data
            
            # OpenMC stores sum and sum_sq, need n_realizations to get mean/std
            # For processed statepoints, it may store mean/std directly
            if 'n_realizations' in tally_group:
                n = int(tally_group['n_realizations'][()])
                if n > 0 and data.shape[-1] >= 2:
                    # data shape: (..., 2) where last dim is [sum, sum_sq]
                    mean = data[..., 0] / n
                    variance = data[..., 1] / n - mean**2
                    variance = np.maximum(variance, 0)  # Ensure non-negative
                    std_dev = np.sqrt(variance / n) if n > 1 else np.zeros_like(mean)
                    results['mean'] = mean
                    results['std_dev'] = std_dev
        
        # Read filter information
        # In OpenMC statepoints, 'filters' in a tally is a dataset with filter IDs
        # The actual filter data is in f['tallies/filters/filter N']
        filters = {}
        if 'filters' in tally_group:
            filter_ids_data = tally_group['filters'][()]
            # Convert to list of filter IDs
            if hasattr(filter_ids_data, '__iter__'):
                filter_ids = [int(fid) for fid in filter_ids_data]
            else:
                filter_ids = [int(filter_ids_data)]
            
            # Read filter details from the tallies/filters group
            if 'filters' in f['tallies']:
                filters_group = f['tallies']['filters']
                for fid in filter_ids:
                    filt_key = f'filter {fid}'
                    if filt_key in filters_group:
                        filt_group = filters_group[filt_key]
                        filt_info = {'id': fid, 'type': ''}
                        
                        if 'type' in filt_group:
                            ftype = filt_group['type'][()]
                            if isinstance(ftype, bytes):
                                ftype = ftype.decode('utf-8')
                            filt_info['type'] = ftype
                        
                        if 'n_bins' in filt_group:
                            filt_info['n_bins'] = int(filt_group['n_bins'][()])
                        
                        if 'bins' in filt_group:
                            filt_info['bins'] = filt_group['bins'][()]
                        
                        filters[fid] = filt_info
        
        results['filters'] = filters
        
        # Read nuclides
        if 'nuclides' in tally_group:
            nuclides = tally_group['nuclides'][()]
            if hasattr(nuclides, '__iter__'):
                results['nuclides'] = [
                    n.decode('utf-8') if isinstance(n, bytes) else str(n)
                    for n in nuclides
                ]
        
        # Read scores
        if 'score_bins' in tally_group:
            scores = tally_group['score_bins'][()]
            if hasattr(scores, '__iter__'):
                results['scores'] = [
                    s.decode('utf-8') if isinstance(s, bytes) else str(s)
                    for s in scores
                ]
    
    return results


def read_openmc_flux_spectrum(
    statepoint_path: Union[str, Path],
    tally_id: int,
    cell_id: Optional[int] = None,
) -> OpenMCSpectrum:
    """
    Extract flux spectrum from OpenMC statepoint.
    
    Assumes tally has an energy filter and flux score.
    
    Parameters
    ----------
    statepoint_path : str or Path
        Path to statepoint.h5 file.
    tally_id : int
        ID of the flux tally.
    cell_id : int, optional
        If tally has cell filter, select this cell.
        
    Returns
    -------
    OpenMCSpectrum
        Extracted flux spectrum with uncertainties.
        
    Raises
    ------
    ValueError
        If tally doesn't have energy filter or flux score.
    """
    if HAS_OPENMC:
        # Use OpenMC API if available (more robust)
        return _read_spectrum_with_openmc_api(statepoint_path, tally_id, cell_id)
    else:
        # Fall back to direct HDF5 parsing
        return _read_spectrum_hdf5_direct(statepoint_path, tally_id, cell_id)


def _read_spectrum_with_openmc_api(
    statepoint_path: Union[str, Path],
    tally_id: int,
    cell_id: Optional[int] = None,
) -> OpenMCSpectrum:
    """Read spectrum using OpenMC Python API."""
    sp = openmc.StatePoint(str(statepoint_path))
    
    tally = sp.get_tally(id=tally_id)
    if tally is None:
        raise ValueError(f"Tally {tally_id} not found in statepoint.")
    
    # Get energy filter
    energy_filter = None
    for f in tally.filters:
        if isinstance(f, openmc.EnergyFilter):
            energy_filter = f
            break
    
    if energy_filter is None:
        raise ValueError(f"Tally {tally_id} has no energy filter.")
    
    energy_bins = energy_filter.bins.flatten()
    
    # Get tally data
    df = tally.get_pandas_dataframe()
    
    # Filter by cell if specified
    if cell_id is not None and 'cell' in df.columns:
        df = df[df['cell'] == cell_id]
    
    # Extract flux data
    flux = df['mean'].values
    std_dev = df['std. dev.'].values
    
    return OpenMCSpectrum(
        energy_bins_ev=energy_bins,
        flux=flux,
        uncertainty=std_dev,
        cell_id=cell_id,
        score='flux',
        n_particles=sp.n_particles if hasattr(sp, 'n_particles') else 0,
        metadata={'tally_id': tally_id},
    )


def _read_spectrum_hdf5_direct(
    statepoint_path: Union[str, Path],
    tally_id: int,
    cell_id: Optional[int] = None,
) -> OpenMCSpectrum:
    """Read spectrum directly from HDF5 without OpenMC API."""
    if not HAS_H5PY:
        raise ImportError("h5py is required to read OpenMC statepoint files.")
    
    statepoint_path = Path(statepoint_path)
    
    with h5py.File(statepoint_path, 'r') as f:
        tally_key = f"tally {tally_id}"
        if tally_key not in f['tallies']:
            raise ValueError(f"Tally {tally_id} not found.")
        
        tally_group = f['tallies'][tally_key]
        
        # Find energy filter
        energy_bins = None
        if 'filters' in tally_group:
            for filt_key in tally_group['filters'].keys():
                filt = tally_group['filters'][filt_key]
                ftype = filt.get('type', [()])
                if isinstance(ftype, h5py.Dataset):
                    ftype = ftype[()]
                if isinstance(ftype, bytes):
                    ftype = ftype.decode('utf-8')
                
                if 'energy' in str(ftype).lower():
                    if 'bins' in filt:
                        energy_bins = filt['bins'][()]
                    break
        
        if energy_bins is None:
            raise ValueError(f"No energy filter found in tally {tally_id}.")
        
        # Get results
        data = tally_group['results'][()]
        n_realizations = 1
        if 'n_realizations' in tally_group:
            n_realizations = int(tally_group['n_realizations'][()])
        
        # Calculate mean and std dev
        if data.shape[-1] >= 2 and n_realizations > 0:
            mean = data[..., 0] / n_realizations
            variance = data[..., 1] / n_realizations - mean**2
            variance = np.maximum(variance, 0)
            std_dev = np.sqrt(variance / max(n_realizations - 1, 1))
        else:
            mean = data.flatten()
            std_dev = np.zeros_like(mean)
        
        # Flatten if needed
        mean = mean.flatten()
        std_dev = std_dev.flatten()
        
        # Ensure correct length
        n_groups = len(energy_bins) - 1
        if len(mean) > n_groups:
            mean = mean[:n_groups]
            std_dev = std_dev[:n_groups]
        
        n_particles = 0
        if 'n_particles' in f.attrs:
            n_particles = int(f.attrs['n_particles'])
        
        return OpenMCSpectrum(
            energy_bins_ev=np.array(energy_bins),
            flux=mean,
            uncertainty=std_dev,
            cell_id=cell_id,
            score='flux',
            n_particles=n_particles,
            metadata={'tally_id': tally_id},
        )
