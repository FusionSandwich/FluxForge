"""
Cross Section Data Module for Dosimetry and Activation Analysis

Provides loaders and containers for nuclear cross section data used in:
- Dosimetry: IRDFF-II, IRDF-2002
- Activation: ENDF, TENDL, FENDL
- Response matrix construction

This module works standalone without requiring PyNE.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate


@dataclass
class CrossSection:
    """
    Container for energy-dependent cross section data.
    
    Attributes
    ----------
    reaction : str
        Reaction identifier, e.g., 'In-115(n,n')In-115m'
    target : str
        Target nuclide
    product : str
        Product nuclide
    mt_number : int
        ENDF MT reaction number
    energies : np.ndarray
        Energy grid in MeV
    values : np.ndarray
        Cross section values in barns
    uncertainties : Optional[np.ndarray]
        Cross section uncertainties (absolute, in barns)
    covariance : Optional[np.ndarray]
        Covariance matrix (if available)
    threshold : float
        Reaction threshold energy in MeV
    source : str
        Data source library name
    metadata : Dict[str, Any]
        Additional metadata
        
    Examples
    --------
    >>> xs = CrossSection.from_endf_file("n-049_In_115.endf", mt=4)
    >>> sigma = xs.evaluate(1.0)  # At 1 MeV
    >>> print(f"σ(1 MeV) = {sigma:.4f} barn")
    """
    
    reaction: str
    target: str
    product: str
    mt_number: int
    energies: np.ndarray
    values: np.ndarray
    uncertainties: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    threshold: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal interpolator
    _interpolator: Optional[Callable] = field(default=None, repr=False)
    
    def evaluate(
        self,
        energy: Union[float, np.ndarray],
        log_interp: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Evaluate cross section at given energy.
        
        Parameters
        ----------
        energy : float or np.ndarray
            Energy in MeV
        log_interp : bool
            Use log-log interpolation (recommended for cross sections)
        
        Returns
        -------
        float or np.ndarray
            Cross section in barns
        """
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        
        # Below threshold
        result = np.zeros_like(energy)
        above_threshold = energy >= self.threshold
        
        if not np.any(above_threshold):
            return result.squeeze()
        
        e_eval = energy[above_threshold]
        
        if log_interp:
            # Log-log interpolation
            log_e = np.log(np.clip(self.energies, 1e-20, None))
            log_xs = np.log(np.clip(self.values, 1e-50, None))
            
            interp = interpolate.interp1d(
                log_e, log_xs,
                kind='linear',
                bounds_error=False,
                fill_value=(log_xs[0], log_xs[-1])
            )
            
            log_result = interp(np.log(np.clip(e_eval, 1e-20, None)))
            result[above_threshold] = np.exp(log_result)
        else:
            # Linear interpolation
            interp = interpolate.interp1d(
                self.energies, self.values,
                kind='linear',
                bounds_error=False,
                fill_value=(self.values[0], self.values[-1])
            )
            result[above_threshold] = interp(e_eval)
        
        # Zero below threshold
        result[energy < self.threshold] = 0.0
        
        # Zero above data range
        result[energy > self.energies.max()] = 0.0
        
        return result.squeeze()
    
    def uncertainty_at(
        self,
        energy: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Get uncertainty at given energy.
        
        Parameters
        ----------
        energy : float or np.ndarray
            Energy in MeV
        
        Returns
        -------
        float or np.ndarray
            Absolute uncertainty in barns
        """
        if self.uncertainties is None:
            # Default 10% relative uncertainty
            return 0.1 * self.evaluate(energy)
        
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        
        interp = interpolate.interp1d(
            self.energies, self.uncertainties,
            kind='linear',
            bounds_error=False,
            fill_value=(self.uncertainties[0], self.uncertainties[-1])
        )
        
        result = interp(energy)
        result[energy < self.threshold] = 0.0
        
        return result.squeeze()
    
    def integrate_over_spectrum(
        self,
        spectrum_energies: np.ndarray,
        spectrum_flux: np.ndarray
    ) -> Tuple[float, float]:
        """
        Integrate cross section weighted by neutron spectrum.
        
        ∫ σ(E) φ(E) dE
        
        Parameters
        ----------
        spectrum_energies : np.ndarray
            Spectrum energy grid (MeV)
        spectrum_flux : np.ndarray
            Flux spectrum (n/cm²/s per MeV or per energy bin)
        
        Returns
        -------
        rate : float
            Reaction rate (reactions/s per target atom per unit flux)
        uncertainty : float
            Rate uncertainty
        """
        # Evaluate cross section on spectrum grid
        sigma = self.evaluate(spectrum_energies)
        
        # Simple trapezoidal integration
        rate = np.trapz(sigma * spectrum_flux, spectrum_energies)
        
        # Uncertainty (assuming uncorrelated uncertainties)
        sigma_unc = self.uncertainty_at(spectrum_energies)
        rate_unc = np.sqrt(np.trapz((sigma_unc * spectrum_flux)**2, spectrum_energies))
        
        return rate, rate_unc
    
    def to_group_structure(
        self,
        group_edges: np.ndarray,
        weighting_spectrum: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collapse cross section to group structure.
        
        Parameters
        ----------
        group_edges : np.ndarray
            Energy group boundaries (MeV), length n_groups + 1
        weighting_spectrum : np.ndarray, optional
            Weighting spectrum for group averaging. If None, use 1/E
        
        Returns
        -------
        group_xs : np.ndarray
            Group-averaged cross sections (barns)
        group_unc : np.ndarray
            Group uncertainties (barns)
        """
        n_groups = len(group_edges) - 1
        group_xs = np.zeros(n_groups)
        group_unc = np.zeros(n_groups)
        
        for g in range(n_groups):
            e_lo = group_edges[g]
            e_hi = group_edges[g + 1]
            
            # Create fine grid within group
            n_points = 50
            e_fine = np.linspace(e_lo, e_hi, n_points)
            
            sigma_fine = self.evaluate(e_fine)
            sigma_unc_fine = self.uncertainty_at(e_fine)
            
            if weighting_spectrum is not None:
                # Interpolate weighting spectrum
                weight_interp = interpolate.interp1d(
                    group_edges[:-1], weighting_spectrum,
                    kind='nearest',
                    bounds_error=False,
                    fill_value=0.0
                )
                weights = weight_interp(e_fine)
            else:
                # 1/E weighting
                weights = 1.0 / e_fine
            
            # Weighted average
            total_weight = np.trapz(weights, e_fine)
            if total_weight > 0:
                group_xs[g] = np.trapz(sigma_fine * weights, e_fine) / total_weight
                group_unc[g] = np.sqrt(
                    np.trapz((sigma_unc_fine * weights)**2, e_fine)
                ) / total_weight
        
        return group_xs, group_unc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'reaction': self.reaction,
            'target': self.target,
            'product': self.product,
            'mt_number': self.mt_number,
            'energies': self.energies.tolist(),
            'values': self.values.tolist(),
            'threshold': self.threshold,
            'source': self.source,
            'metadata': self.metadata,
        }
        if self.uncertainties is not None:
            data['uncertainties'] = self.uncertainties.tolist()
        if self.covariance is not None:
            data['covariance'] = self.covariance.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossSection':
        """Create CrossSection from dictionary."""
        return cls(
            reaction=data['reaction'],
            target=data['target'],
            product=data['product'],
            mt_number=data['mt_number'],
            energies=np.array(data['energies']),
            values=np.array(data['values']),
            uncertainties=np.array(data['uncertainties']) if 'uncertainties' in data else None,
            covariance=np.array(data['covariance']) if 'covariance' in data else None,
            threshold=data.get('threshold', 0.0),
            source=data.get('source', ''),
            metadata=data.get('metadata', {}),
        )


@dataclass
class CrossSectionLibrary:
    """
    Collection of cross sections for dosimetry or activation analysis.
    
    Attributes
    ----------
    name : str
        Library name (e.g., 'IRDFF-II')
    cross_sections : Dict[str, CrossSection]
        Cross sections indexed by reaction identifier
    energy_grid : Optional[np.ndarray]
        Common energy grid if applicable
    metadata : Dict[str, Any]
        Library metadata
    """
    
    name: str
    cross_sections: Dict[str, CrossSection] = field(default_factory=dict)
    energy_grid: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, xs: CrossSection) -> None:
        """Add cross section to library."""
        self.cross_sections[xs.reaction] = xs
    
    def get(self, reaction: str) -> Optional[CrossSection]:
        """Get cross section by reaction identifier."""
        return self.cross_sections.get(reaction)
    
    def list_reactions(self) -> List[str]:
        """List all available reactions."""
        return list(self.cross_sections.keys())
    
    def filter_by_target(self, target: str) -> List[CrossSection]:
        """Get all cross sections for a target nuclide."""
        return [xs for xs in self.cross_sections.values() if xs.target == target]
    
    def filter_by_mt(self, mt: int) -> List[CrossSection]:
        """Get all cross sections with given MT number."""
        return [xs for xs in self.cross_sections.values() if xs.mt_number == mt]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save library to JSON file."""
        data = {
            'name': self.name,
            'cross_sections': {k: v.to_dict() for k, v in self.cross_sections.items()},
            'energy_grid': self.energy_grid.tolist() if self.energy_grid is not None else None,
            'metadata': self.metadata,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CrossSectionLibrary':
        """Load library from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        lib = cls(
            name=data['name'],
            energy_grid=np.array(data['energy_grid']) if data.get('energy_grid') else None,
            metadata=data.get('metadata', {}),
        )
        
        for reaction, xs_data in data.get('cross_sections', {}).items():
            lib.add(CrossSection.from_dict(xs_data))
        
        return lib


# ============================================================================
# IRDFF-II Dosimetry Cross Sections
# ============================================================================

# Standard IRDFF-II reactions for dosimetry
IRDFF_II_REACTIONS = {
    # Threshold reactions
    'In-115(n,n\')In-115m': {'mt': 4, 'threshold': 0.34, 'half_life': 4.486 * 3600},
    'Ni-58(n,p)Co-58': {'mt': 103, 'threshold': 0.0, 'half_life': 70.86 * 86400},
    'Al-27(n,α)Na-24': {'mt': 107, 'threshold': 3.25, 'half_life': 14.96 * 3600},
    'Fe-54(n,p)Mn-54': {'mt': 103, 'threshold': 0.0, 'half_life': 312.2 * 86400},
    'Fe-56(n,p)Mn-56': {'mt': 103, 'threshold': 2.97, 'half_life': 2.58 * 3600},
    'Cu-63(n,α)Co-60': {'mt': 107, 'threshold': 0.0, 'half_life': 1925.28 * 86400},
    'Zr-90(n,2n)Zr-89': {'mt': 16, 'threshold': 12.1, 'half_life': 78.41 * 3600},
    'Nb-93(n,2n)Nb-92m': {'mt': 16, 'threshold': 8.97, 'half_life': 10.15 * 86400},
    
    # Thermal reactions
    'Au-197(n,γ)Au-198': {'mt': 102, 'threshold': 0.0, 'half_life': 2.695 * 86400},
    'Co-59(n,γ)Co-60': {'mt': 102, 'threshold': 0.0, 'half_life': 1925.28 * 86400},
    'Mn-55(n,γ)Mn-56': {'mt': 102, 'threshold': 0.0, 'half_life': 2.58 * 3600},
    
    # Fission reactions
    'U-235(n,f)': {'mt': 18, 'threshold': 0.0, 'half_life': None},
    'U-238(n,f)': {'mt': 18, 'threshold': 1.0, 'half_life': None},
    'Np-237(n,f)': {'mt': 18, 'threshold': 0.0, 'half_life': None},
}


def create_irdff_placeholder_library() -> CrossSectionLibrary:
    """
    Create placeholder IRDFF-II library with approximate cross sections.
    
    This provides representative cross section shapes for development/testing.
    For actual dosimetry, use official IRDFF-II data files.
    
    Returns
    -------
    CrossSectionLibrary
        Library with approximate IRDFF-II cross sections
    """
    lib = CrossSectionLibrary(
        name='IRDFF-II-placeholder',
        metadata={
            'description': 'Placeholder library with approximate shapes',
            'warning': 'Use official IRDFF-II data for actual analysis',
        }
    )
    
    # Common energy grid (MeV)
    energies = np.logspace(-10, 2, 500)  # 1e-10 to 100 MeV
    
    for reaction, info in IRDFF_II_REACTIONS.items():
        # Parse reaction string
        parts = reaction.split('(')
        target = parts[0]
        reaction_type = parts[1].split(')')[0]
        product = parts[1].split(')')[1] if ')' in parts[1] else ''
        
        # Generate approximate cross section shape
        threshold = info['threshold']
        mt = info['mt']
        
        values = np.zeros_like(energies)
        
        if mt == 102:  # (n,γ) - 1/v below, resonance region
            sigma_thermal = 100.0  # Approximate thermal value
            values = sigma_thermal * np.sqrt(0.0253e-6 / np.clip(energies, 1e-12, None))
            # Add resonance structure (simplified)
            values *= (1 + 5 * np.exp(-((energies - 1e-5) / 1e-6)**2))
        
        elif mt in [103, 107]:  # (n,p), (n,α) - threshold reactions
            if threshold > 0:
                above = energies > threshold
                values[above] = 0.1 * (1 - np.exp(-(energies[above] - threshold) / 1.0))
            else:
                values = 0.01 * np.ones_like(energies)
        
        elif mt == 4:  # (n,n') - inelastic
            if threshold > 0:
                above = energies > threshold
                values[above] = 0.3 * (1 - np.exp(-(energies[above] - threshold) / 0.5))
        
        elif mt == 16:  # (n,2n)
            if threshold > 0:
                above = energies > threshold
                values[above] = 1.5 * (1 - np.exp(-(energies[above] - threshold) / 3.0))
        
        elif mt == 18:  # (n,f) - fission
            if threshold > 0:
                above = energies > threshold
                values[above] = 0.5 * (1 + 0.5 * (energies[above] - threshold))
            else:
                values = 500 * np.sqrt(0.0253e-6 / np.clip(energies, 1e-12, None))
                values = np.clip(values, 0, 1000)
        
        xs = CrossSection(
            reaction=reaction,
            target=target,
            product=product,
            mt_number=mt,
            energies=energies.copy(),
            values=values,
            uncertainties=0.1 * values,  # 10% uncertainty
            threshold=threshold,
            source='IRDFF-II-placeholder',
            metadata={'half_life': info['half_life']},
        )
        
        lib.add(xs)
    
    lib.energy_grid = energies
    
    return lib


# ============================================================================
# File Parsers
# ============================================================================

def load_csv_cross_section(
    filepath: Union[str, Path],
    reaction: str,
    target: str,
    product: str,
    mt: int,
    energy_col: int = 0,
    xs_col: int = 1,
    unc_col: Optional[int] = None,
    skiprows: int = 0,
    delimiter: str = ',',
    energy_units: str = 'MeV',
    xs_units: str = 'barn'
) -> CrossSection:
    """
    Load cross section from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    reaction : str
        Reaction identifier
    target, product : str
        Target and product nuclides
    mt : int
        MT reaction number
    energy_col, xs_col : int
        Column indices for energy and cross section
    unc_col : int, optional
        Column index for uncertainty
    skiprows : int
        Number of header rows to skip
    delimiter : str
        Column delimiter
    energy_units : str
        Energy units ('eV', 'keV', 'MeV')
    xs_units : str
        Cross section units ('barn', 'mbarn')
    
    Returns
    -------
    CrossSection
    """
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)
    
    energies = data[:, energy_col]
    values = data[:, xs_col]
    uncertainties = data[:, unc_col] if unc_col is not None else None
    
    # Convert units
    if energy_units == 'eV':
        energies = energies * 1e-6
    elif energy_units == 'keV':
        energies = energies * 1e-3
    
    if xs_units == 'mbarn':
        values = values * 1e-3
        if uncertainties is not None:
            uncertainties = uncertainties * 1e-3
    
    # Find threshold
    nonzero = values > 0
    threshold = energies[nonzero][0] if np.any(nonzero) else 0.0
    
    return CrossSection(
        reaction=reaction,
        target=target,
        product=product,
        mt_number=mt,
        energies=energies,
        values=values,
        uncertainties=uncertainties,
        threshold=threshold,
        source=str(filepath),
    )


def load_tendl_ace(
    filepath: Union[str, Path],
    mt: int = 4
) -> CrossSection:
    """
    Load cross section from TENDL ACE format file.
    
    This is a simplified parser for ACE format. For full support,
    use PyNE or OpenMC data libraries.
    
    Parameters
    ----------
    filepath : str or Path
        Path to ACE file
    mt : int
        MT reaction number to extract
    
    Returns
    -------
    CrossSection
    """
    # ACE format is complex - this is a placeholder
    # For production use, recommend PyNE's ace module or OpenMC
    raise NotImplementedError(
        "Full ACE parsing not implemented. Use PyNE: "
        "from pyne.ace import Library; lib = Library(filepath)"
    )


def load_endf_simple(
    filepath: Union[str, Path],
    mt: int,
    mf: int = 3
) -> CrossSection:
    """
    Simple ENDF file parser for cross section data.
    
    This parses the basic MF=3 cross section format.
    For full ENDF support, use PyNE or FUDGE.
    
    Parameters
    ----------
    filepath : str or Path
        Path to ENDF file
    mt : int
        MT reaction number
    mf : int
        MF file number (default 3 for cross sections)
    
    Returns
    -------
    CrossSection
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse header for ZA and AWR
    za = None
    awr = None
    target = ''
    
    lines = content.split('\n')
    
    # Find MF/MT section
    in_section = False
    section_data = []
    
    for line in lines:
        if len(line) < 75:
            continue
        
        try:
            mat = int(line[66:70])
            mf_line = int(line[70:72])
            mt_line = int(line[72:75])
        except ValueError:
            continue
        
        if mf_line == mf and mt_line == mt:
            in_section = True
            section_data.append(line)
        elif in_section and (mf_line != mf or mt_line != mt):
            break
    
    if not section_data:
        raise ValueError(f"MT={mt}, MF={mf} section not found in {filepath}")
    
    # Parse section (simplified - assumes TAB1 format)
    energies = []
    values = []
    
    # Skip header lines, parse data
    i = 2  # Skip first two header lines
    while i < len(section_data):
        line = section_data[i]
        # ENDF uses 11-character fields
        for j in range(6):
            field = line[j*11:(j+1)*11].strip()
            if field:
                try:
                    val = float(field.replace('+', 'E+').replace('-', 'E-'))
                    if len(energies) == len(values):
                        energies.append(val)
                    else:
                        values.append(val)
                except ValueError:
                    pass
        i += 1
    
    energies = np.array(energies)
    values = np.array(values)
    
    # Energy units in ENDF are eV, convert to MeV
    energies = energies * 1e-6
    
    # Find threshold
    nonzero = values > 0
    threshold = energies[nonzero][0] if np.any(nonzero) else 0.0
    
    return CrossSection(
        reaction=f'MT{mt}',
        target=target,
        product='',
        mt_number=mt,
        energies=energies,
        values=values,
        threshold=threshold,
        source=str(filepath),
    )
