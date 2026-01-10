"""
Unified Nuclear Data Interface for FluxForge
============================================

This module provides a unified interface for accessing nuclear data from
various sources (ENDF/B-VIII.0, IRDFF-II, etc.) with consistent handling of:

- Continuous-energy (CE) and multi-group (MG) cross sections
- Temperature-dependent data with interpolation
- Library-agnostic identifiers and mapping

Key Features:
- O1.1: Unified NuclearData interface for CE and MG cross sections
- O1.4: Multi-temperature support with temperature tags
- O1.5: ENDF ↔ IRDFF bridge utilities for explicit mapping

This prevents library mismatches between transport and activation codes
by providing a single source of truth for nuclear data.

References
----------
- ENDF/B-VIII.0: Nuclear Data Sheets 148 (2018) 1-142
- IRDFF-II: IAEA Technical Report Series No. 452 (2020)

Author: FluxForge Development Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np


class DataType(Enum):
    """Type of nuclear data representation."""
    
    CONTINUOUS_ENERGY = "ce"  # Point-wise energy-dependent
    MULTI_GROUP = "mg"        # Group-averaged values


class DataLibrary(Enum):
    """Nuclear data library identifier."""
    
    ENDF_B_VIII0 = "ENDF/B-VIII.0"
    ENDF_B_VII1 = "ENDF/B-VII.1"
    IRDFF_II = "IRDFF-II"
    JEFF_33 = "JEFF-3.3"
    JENDL_50 = "JENDL-5.0"
    TENDL_2021 = "TENDL-2021"
    CUSTOM = "custom"


@dataclass
class ReactionIdentifier:
    """
    Unique identifier for a nuclear reaction.
    
    Provides library-agnostic identification with mappings.
    
    Attributes:
        target: Target nuclide (e.g., "U-235", "Fe-56")
        mt: ENDF MT reaction number
        product: Product nuclide if applicable
        za: ZA number (1000*Z + A)
        mat: ENDF material number if known
    """
    
    target: str
    mt: int
    product: str = ""
    za: int = 0
    mat: int = 0
    
    @property
    def reaction_string(self) -> str:
        """Standard reaction string (e.g., 'U-235(n,f)')."""
        mt_names = {
            1: "total", 2: "elastic", 4: "inelastic",
            16: "n,2n", 17: "n,3n", 18: "fission",
            102: "n,g", 103: "n,p", 104: "n,d",
            105: "n,t", 106: "n,He3", 107: "n,a",
        }
        rx_name = mt_names.get(self.mt, f"MT{self.mt}")
        return f"{self.target}({rx_name})"
    
    def __hash__(self):
        return hash((self.target, self.mt))
    
    def __eq__(self, other):
        if isinstance(other, ReactionIdentifier):
            return self.target == other.target and self.mt == other.mt
        return False


@dataclass
class TemperatureData:
    """
    Cross section data at a specific temperature.
    
    Attributes:
        temperature_K: Temperature in Kelvin
        energies: Energy grid (eV) for CE data
        values: Cross section values (barns)
        uncertainties: Uncertainties (barns) if available
    """
    
    temperature_K: float
    energies: np.ndarray  # eV
    values: np.ndarray    # barns
    uncertainties: Optional[np.ndarray] = None
    
    def interpolate(self, energy: float) -> float:
        """Interpolate to specific energy."""
        if len(self.energies) == 0:
            return 0.0
        return float(np.interp(energy, self.energies, self.values))


@dataclass
class NuclearData:
    """
    Unified nuclear data container.
    
    Supports both continuous-energy and multi-group representations
    with temperature-dependent data.
    
    Attributes:
        reaction_id: Reaction identifier
        data_type: CE or MG representation
        library: Source library
        temperatures: Dictionary of temperature-specific data
        group_structure: Energy bin boundaries for MG data (eV)
        covariance: Covariance matrix if available
        metadata: Additional metadata
    """
    
    reaction_id: ReactionIdentifier
    data_type: DataType
    library: DataLibrary
    temperatures: Dict[float, TemperatureData] = field(default_factory=dict)
    group_structure: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_temperatures(self) -> List[float]:
        """List available temperatures."""
        return sorted(self.temperatures.keys())
    
    @property
    def n_groups(self) -> int:
        """Number of energy groups for MG data."""
        if self.group_structure is not None:
            return len(self.group_structure) - 1
        return 0
    
    def get_at_temperature(
        self,
        temperature_K: float,
        interpolate: bool = True,
    ) -> Optional[TemperatureData]:
        """
        Get data at specific temperature.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        interpolate : bool
            Whether to interpolate between available temperatures.
            
        Returns
        -------
        TemperatureData or None
            Data at the requested temperature.
        """
        # Exact match
        if temperature_K in self.temperatures:
            return self.temperatures[temperature_K]
        
        # Find nearest
        if not self.temperatures:
            return None
        
        temps = self.available_temperatures
        
        if not interpolate:
            # Return nearest
            nearest = min(temps, key=lambda t: abs(t - temperature_K))
            return self.temperatures[nearest]
        
        # Linear interpolation in sqrt(T) for Doppler broadening
        if temperature_K < temps[0]:
            return self.temperatures[temps[0]]
        if temperature_K > temps[-1]:
            return self.temperatures[temps[-1]]
        
        # Find bracketing temperatures
        for i in range(len(temps) - 1):
            if temps[i] <= temperature_K <= temps[i + 1]:
                t_lo, t_hi = temps[i], temps[i + 1]
                data_lo = self.temperatures[t_lo]
                data_hi = self.temperatures[t_hi]
                
                # Interpolation factor
                f = (np.sqrt(temperature_K) - np.sqrt(t_lo)) / (np.sqrt(t_hi) - np.sqrt(t_lo))
                
                # Interpolate values
                values_interp = (1 - f) * data_lo.values + f * data_hi.values
                
                return TemperatureData(
                    temperature_K=temperature_K,
                    energies=data_lo.energies,
                    values=values_interp,
                )
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'reaction': {
                'target': self.reaction_id.target,
                'mt': self.reaction_id.mt,
                'product': self.reaction_id.product,
            },
            'data_type': self.data_type.value,
            'library': self.library.value,
            'temperatures': {
                str(t): {
                    'energies': d.energies.tolist(),
                    'values': d.values.tolist(),
                }
                for t, d in self.temperatures.items()
            },
            'group_structure': self.group_structure.tolist() if self.group_structure is not None else None,
            'metadata': self.metadata,
        }


class NuclearDataProvider(ABC):
    """Abstract base class for nuclear data providers."""
    
    @abstractmethod
    def get_reaction(
        self,
        target: str,
        mt: int,
        temperature_K: float = 300.0,
    ) -> Optional[NuclearData]:
        """Get data for a specific reaction."""
        pass
    
    @abstractmethod
    def list_reactions(self) -> List[ReactionIdentifier]:
        """List all available reactions."""
        pass
    
    @abstractmethod
    def get_library(self) -> DataLibrary:
        """Get the library this provider represents."""
        pass


# ENDF ↔ IRDFF Mapping (O1.5)

@dataclass
class ReactionMapping:
    """
    Mapping between equivalent reactions in different libraries.
    
    Attributes:
        endf_target: Target nuclide in ENDF
        endf_mt: MT number in ENDF
        irdff_reaction: Reaction name in IRDFF-II
        notes: Any notes about differences
    """
    
    endf_target: str
    endf_mt: int
    irdff_reaction: str
    notes: str = ""


# Standard ENDF to IRDFF-II mappings for dosimetry reactions
ENDF_IRDFF_MAPPINGS: List[ReactionMapping] = [
    ReactionMapping("Au-197", 102, "Au-197(n,g)Au-198", "Primary thermal monitor"),
    ReactionMapping("In-115", 4, "In-115(n,n')In-115m", "Threshold ~0.34 MeV"),
    ReactionMapping("Ni-58", 103, "Ni-58(n,p)Co-58", "Threshold ~0.5 MeV"),
    ReactionMapping("Fe-54", 103, "Fe-54(n,p)Mn-54", "Threshold ~0.5 MeV"),
    ReactionMapping("Al-27", 107, "Al-27(n,a)Na-24", "Threshold ~3.3 MeV"),
    ReactionMapping("Nb-93", 16, "Nb-93(n,2n)Nb-92m", "Threshold ~9 MeV"),
    ReactionMapping("Co-59", 102, "Co-59(n,g)Co-60", "Thermal/epithermal monitor"),
    ReactionMapping("Mn-55", 102, "Mn-55(n,g)Mn-56", "Thermal monitor"),
    ReactionMapping("Ti-46", 103, "Ti-46(n,p)Sc-46", "Threshold ~1.7 MeV"),
    ReactionMapping("Ti-47", 103, "Ti-47(n,p)Sc-47", "Threshold ~0.5 MeV"),
    ReactionMapping("Ti-48", 103, "Ti-48(n,p)Sc-48", "Threshold ~3.3 MeV"),
    ReactionMapping("Fe-56", 103, "Fe-56(n,p)Mn-56", "Threshold ~2.9 MeV"),
    ReactionMapping("Cu-63", 16, "Cu-63(n,2n)Cu-62", "Threshold ~11 MeV"),
    ReactionMapping("Cu-65", 16, "Cu-65(n,2n)Cu-64", "Threshold ~10 MeV"),
    ReactionMapping("Zr-90", 16, "Zr-90(n,2n)Zr-89", "Threshold ~12 MeV"),
    ReactionMapping("S-32", 103, "S-32(n,p)P-32", "Threshold ~0.9 MeV"),
    ReactionMapping("Mg-24", 103, "Mg-24(n,p)Na-24", "Threshold ~5 MeV"),
]


class ENDFIRDFFBridge:
    """
    Bridge between ENDF and IRDFF-II reaction identifiers.
    
    Provides explicit mapping between the two libraries to prevent
    mismatches when using transport code results with activation analysis.
    """
    
    def __init__(self, mappings: Optional[List[ReactionMapping]] = None):
        """Initialize with mappings."""
        self.mappings = mappings or ENDF_IRDFF_MAPPINGS
        
        # Build lookup tables
        self._endf_to_irdff: Dict[Tuple[str, int], str] = {}
        self._irdff_to_endf: Dict[str, Tuple[str, int]] = {}
        
        for m in self.mappings:
            key = (m.endf_target, m.endf_mt)
            self._endf_to_irdff[key] = m.irdff_reaction
            self._irdff_to_endf[m.irdff_reaction] = key
    
    def endf_to_irdff(self, target: str, mt: int) -> Optional[str]:
        """
        Map ENDF reaction to IRDFF-II reaction name.
        
        Parameters
        ----------
        target : str
            Target nuclide (e.g., "Au-197").
        mt : int
            ENDF MT number.
            
        Returns
        -------
        str or None
            IRDFF-II reaction name, or None if no mapping.
        """
        return self._endf_to_irdff.get((target, mt))
    
    def irdff_to_endf(self, reaction: str) -> Optional[Tuple[str, int]]:
        """
        Map IRDFF-II reaction to ENDF (target, MT).
        
        Parameters
        ----------
        reaction : str
            IRDFF-II reaction name (e.g., "Au-197(n,g)Au-198").
            
        Returns
        -------
        tuple or None
            (target, mt) tuple, or None if no mapping.
        """
        return self._irdff_to_endf.get(reaction)
    
    def list_mappings(self) -> List[Dict[str, Any]]:
        """List all available mappings."""
        return [
            {
                'endf_target': m.endf_target,
                'endf_mt': m.endf_mt,
                'irdff_reaction': m.irdff_reaction,
                'notes': m.notes,
            }
            for m in self.mappings
        ]
    
    def validate_consistency(
        self,
        endf_xs: np.ndarray,
        irdff_xs: np.ndarray,
        energies: np.ndarray,
        tolerance: float = 0.1,
    ) -> Tuple[bool, float, str]:
        """
        Check if ENDF and IRDFF cross sections are consistent.
        
        Parameters
        ----------
        endf_xs : np.ndarray
            Cross sections from ENDF evaluation.
        irdff_xs : np.ndarray
            Cross sections from IRDFF-II.
        energies : np.ndarray
            Common energy grid.
        tolerance : float
            Relative tolerance for consistency.
            
        Returns
        -------
        consistent : bool
            Whether data are consistent.
        max_deviation : float
            Maximum relative deviation.
        message : str
            Descriptive message.
        """
        if len(endf_xs) != len(irdff_xs):
            return False, 1.0, "Array lengths differ"
        
        # Compute relative differences where both are nonzero
        mask = (endf_xs > 0) & (irdff_xs > 0)
        if not np.any(mask):
            return True, 0.0, "No overlapping nonzero values"
        
        rel_diff = np.abs(endf_xs[mask] - irdff_xs[mask]) / irdff_xs[mask]
        max_dev = float(np.max(rel_diff))
        
        consistent = max_dev <= tolerance
        
        if consistent:
            message = f"Consistent within {max_dev*100:.1f}% (tolerance: {tolerance*100:.0f}%)"
        else:
            message = f"Deviation of {max_dev*100:.1f}% exceeds tolerance {tolerance*100:.0f}%"
        
        return consistent, max_dev, message


# Multi-temperature utilities (O1.4)

def create_temperature_set(
    base_data: TemperatureData,
    temperatures: List[float],
    broadening_model: str = "sqrt_T",
) -> Dict[float, TemperatureData]:
    """
    Create temperature-dependent data set from base data.
    
    Parameters
    ----------
    base_data : TemperatureData
        Base cross section data (typically at 0K or 300K).
    temperatures : list of float
        Target temperatures in Kelvin.
    broadening_model : str
        Model for Doppler broadening ('sqrt_T' or 'linear').
        
    Returns
    -------
    dict
        Dictionary mapping temperature to TemperatureData.
        
    Note
    ----
    This is a simplified model. For accurate temperature-dependent
    cross sections, use NJOY BROADR module.
    """
    result = {}
    
    for T in temperatures:
        if T == base_data.temperature_K:
            result[T] = base_data
        else:
            # Simplified Doppler broadening approximation
            # Real broadening requires integral over Maxwellian
            result[T] = TemperatureData(
                temperature_K=T,
                energies=base_data.energies.copy(),
                values=base_data.values.copy(),  # Simplified: same values
                uncertainties=base_data.uncertainties.copy() if base_data.uncertainties is not None else None,
            )
    
    return result


def interpolate_temperature(
    data_dict: Dict[float, TemperatureData],
    target_T: float,
) -> Optional[TemperatureData]:
    """
    Interpolate cross section data to target temperature.
    
    Uses sqrt(T) interpolation appropriate for Doppler broadening.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of temperature -> TemperatureData.
    target_T : float
        Target temperature in Kelvin.
        
    Returns
    -------
    TemperatureData or None
        Interpolated data at target temperature.
    """
    if not data_dict:
        return None
    
    temps = sorted(data_dict.keys())
    
    if target_T in temps:
        return data_dict[target_T]
    
    if target_T < temps[0]:
        return data_dict[temps[0]]
    if target_T > temps[-1]:
        return data_dict[temps[-1]]
    
    # Find bracketing temperatures
    for i in range(len(temps) - 1):
        if temps[i] <= target_T <= temps[i + 1]:
            t_lo, t_hi = temps[i], temps[i + 1]
            data_lo = data_dict[t_lo]
            data_hi = data_dict[t_hi]
            
            # sqrt(T) interpolation
            f = (np.sqrt(target_T) - np.sqrt(t_lo)) / (np.sqrt(t_hi) - np.sqrt(t_lo))
            
            values_interp = (1 - f) * data_lo.values + f * data_hi.values
            
            if data_lo.uncertainties is not None and data_hi.uncertainties is not None:
                unc_interp = (1 - f) * data_lo.uncertainties + f * data_hi.uncertainties
            else:
                unc_interp = None
            
            return TemperatureData(
                temperature_K=target_T,
                energies=data_lo.energies,
                values=values_interp,
                uncertainties=unc_interp,
            )
    
    return None


# Convenience factory functions

def create_nuclear_data(
    target: str,
    mt: int,
    energies: np.ndarray,
    values: np.ndarray,
    temperature_K: float = 300.0,
    library: DataLibrary = DataLibrary.CUSTOM,
    data_type: DataType = DataType.CONTINUOUS_ENERGY,
    uncertainties: Optional[np.ndarray] = None,
    product: str = "",
) -> NuclearData:
    """
    Convenience function to create NuclearData object.
    
    Parameters
    ----------
    target : str
        Target nuclide.
    mt : int
        ENDF MT number.
    energies : np.ndarray
        Energy grid (eV).
    values : np.ndarray
        Cross section values (barns).
    temperature_K : float
        Temperature in Kelvin.
    library : DataLibrary
        Source library.
    data_type : DataType
        CE or MG representation.
    uncertainties : np.ndarray, optional
        Uncertainties in barns.
    product : str
        Product nuclide.
        
    Returns
    -------
    NuclearData
        Complete nuclear data object.
    """
    reaction_id = ReactionIdentifier(
        target=target,
        mt=mt,
        product=product,
    )
    
    temp_data = TemperatureData(
        temperature_K=temperature_K,
        energies=energies,
        values=values,
        uncertainties=uncertainties,
    )
    
    return NuclearData(
        reaction_id=reaction_id,
        data_type=data_type,
        library=library,
        temperatures={temperature_K: temp_data},
    )


def create_multigroup_data(
    target: str,
    mt: int,
    group_boundaries: np.ndarray,
    group_values: np.ndarray,
    temperature_K: float = 300.0,
    library: DataLibrary = DataLibrary.CUSTOM,
    covariance: Optional[np.ndarray] = None,
) -> NuclearData:
    """
    Create multi-group nuclear data.
    
    Parameters
    ----------
    target : str
        Target nuclide.
    mt : int
        ENDF MT number.
    group_boundaries : np.ndarray
        Energy bin boundaries (eV), length n_groups + 1.
    group_values : np.ndarray
        Group-averaged cross sections (barns), length n_groups.
    temperature_K : float
        Temperature in Kelvin.
    library : DataLibrary
        Source library.
    covariance : np.ndarray, optional
        Covariance matrix.
        
    Returns
    -------
    NuclearData
        Multi-group nuclear data object.
    """
    reaction_id = ReactionIdentifier(target=target, mt=mt)
    
    # Use bin centers for energy grid
    e_centers = np.sqrt(group_boundaries[:-1] * group_boundaries[1:])
    
    temp_data = TemperatureData(
        temperature_K=temperature_K,
        energies=e_centers,
        values=group_values,
    )
    
    return NuclearData(
        reaction_id=reaction_id,
        data_type=DataType.MULTI_GROUP,
        library=library,
        temperatures={temperature_K: temp_data},
        group_structure=group_boundaries,
        covariance=covariance,
    )
