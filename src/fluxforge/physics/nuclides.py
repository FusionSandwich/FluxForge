"""
Nuclide Database Module

Provides access to nuclear decay data including half-lives, gamma energies,
and emission intensities. Based on ENSDF/decay_2012 data structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class GammaLineData:
    """Data for a single gamma line."""
    energy_eV: float
    intensity: float  # probability per decay
    energy_unc_eV: float = 0.0
    intensity_unc: float = 0.0


@dataclass 
class NuclideData:
    """Complete decay data for a nuclide."""
    name: str
    zai: int  # Z*10000 + A*10 + isomeric_state
    half_life_s: float
    half_life_unc_s: float = 0.0
    gamma_lines: List[GammaLineData] = field(default_factory=list)
    beta_lines: List[Dict] = field(default_factory=list)
    alpha_lines: List[Dict] = field(default_factory=list)
    xray_lines: List[Dict] = field(default_factory=list)
    decay_modes: List[str] = field(default_factory=list)
    
    @property
    def decay_constant(self) -> float:
        """Decay constant λ = ln(2)/t_half."""
        return np.log(2) / self.half_life_s if self.half_life_s > 0 else 0.0
    
    @property
    def mean_lifetime(self) -> float:
        """Mean lifetime τ = 1/λ."""
        return 1.0 / self.decay_constant if self.decay_constant > 0 else float('inf')
    
    def get_gamma_energies(self, min_intensity: float = 0.0) -> np.ndarray:
        """Get array of gamma energies above intensity threshold."""
        return np.array([
            g.energy_eV for g in self.gamma_lines 
            if g.intensity >= min_intensity
        ])
    
    def get_gamma_intensities(self, min_intensity: float = 0.0) -> np.ndarray:
        """Get array of gamma intensities above threshold."""
        return np.array([
            g.intensity for g in self.gamma_lines 
            if g.intensity >= min_intensity
        ])


def parse_zai(zai: int) -> Tuple[int, int, int]:
    """Parse ZAI into (Z, A, isomeric_state)."""
    z = zai // 10000
    a = (zai % 10000) // 10
    i = zai % 10
    return z, a, i


def format_nuclide_name(z: int, a: int, isomer: int = 0) -> str:
    """Format nuclide name from Z, A, isomeric state."""
    ELEMENTS = [
        'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
    ]
    
    if z < 0 or z >= len(ELEMENTS):
        elem = f"Z{z}"
    else:
        elem = ELEMENTS[z]
    
    suffix = 'm' * isomer if isomer > 0 else ''
    return f"{elem}{a}{suffix}"


class NuclideDatabase:
    """
    Database of nuclear decay data.
    
    Provides lookup of half-lives, gamma lines, and other decay properties.
    Can be loaded from JSON files (e.g., decay_2012 format) or populated
    with built-in common nuclides.
    
    Examples
    --------
    >>> db = NuclideDatabase()
    >>> db.load_builtin()
    >>> co60 = db.get("Co60")
    >>> print(f"Co-60 half-life: {co60.half_life_s / 86400:.2f} days")
    >>> print(f"Main gamma lines: {co60.get_gamma_energies() / 1000} keV")
    """
    
    def __init__(self):
        self._nuclides: Dict[str, NuclideData] = {}
        self._zai_index: Dict[int, str] = {}
    
    def __contains__(self, name: str) -> bool:
        """Check if nuclide is in database."""
        return self._normalize_name(name) in self._nuclides
    
    def __len__(self) -> int:
        return len(self._nuclides)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize nuclide name (remove spaces, handle common variants)."""
        name = name.strip().replace(' ', '').replace('-', '')
        # Handle common variants like 'Co-60' -> 'Co60'
        return name
    
    def add(self, nuclide: NuclideData) -> None:
        """Add nuclide to database."""
        key = self._normalize_name(nuclide.name)
        self._nuclides[key] = nuclide
        self._zai_index[nuclide.zai] = key
    
    def get(self, name: str) -> Optional[NuclideData]:
        """Get nuclide data by name."""
        key = self._normalize_name(name)
        return self._nuclides.get(key)
    
    def get_by_zai(self, zai: int) -> Optional[NuclideData]:
        """Get nuclide data by ZAI number."""
        key = self._zai_index.get(zai)
        if key:
            return self._nuclides.get(key)
        return None
    
    def get_half_life(self, name: str) -> Optional[float]:
        """Get half-life in seconds."""
        nuc = self.get(name)
        return nuc.half_life_s if nuc else None
    
    def get_gamma_lines(self, name: str, min_intensity: float = 0.0) -> List[GammaLineData]:
        """Get gamma lines for nuclide above intensity threshold."""
        nuc = self.get(name)
        if not nuc:
            return []
        return [g for g in nuc.gamma_lines if g.intensity >= min_intensity]
    
    def search(self, pattern: str) -> List[str]:
        """Search for nuclides matching pattern."""
        pattern = pattern.lower()
        return [name for name in self._nuclides.keys() if pattern in name.lower()]
    
    def all_nuclides(self) -> List[str]:
        """Return list of all nuclide names."""
        return list(self._nuclides.keys())
    
    def load_json(self, filepath: str) -> None:
        """
        Load nuclide data from JSON file.
        
        Expected format (decay_2012 style):
        {
            "Co60": {
                "halflife": 166344192.0,
                "zai": 270600,
                "gamma": {
                    "lines": {
                        "energies": [1173228.0, 1332492.0],
                        "intensities": [0.9985, 0.9998],
                        ...
                    }
                }
            }
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, nuc_data in data.items():
            gamma_lines = []
            
            if 'gamma' in nuc_data and 'lines' in nuc_data['gamma']:
                lines = nuc_data['gamma']['lines']
                energies = lines.get('energies', [])
                intensities = lines.get('intensities', [])
                energy_uncs = lines.get('energies_unc', [0.0] * len(energies))
                intensity_uncs = lines.get('intensities_unc', [0.0] * len(intensities))
                
                for e, i, eu, iu in zip(energies, intensities, energy_uncs, intensity_uncs):
                    gamma_lines.append(GammaLineData(
                        energy_eV=e,
                        intensity=i,
                        energy_unc_eV=eu,
                        intensity_unc=iu
                    ))
            
            nuclide = NuclideData(
                name=name,
                zai=nuc_data.get('zai', 0),
                half_life_s=nuc_data.get('halflife', 0.0),
                gamma_lines=gamma_lines
            )
            self.add(nuclide)
    
    def load_builtin(self) -> None:
        """Load built-in common nuclides used in activation analysis."""
        
        # Common calibration and activation nuclides
        common_nuclides = [
            # Calibration sources
            NuclideData(
                name="Co60",
                zai=270600,
                half_life_s=166344192.0,  # 5.27 years
                gamma_lines=[
                    GammaLineData(1173228.0, 0.9985),
                    GammaLineData(1332492.0, 0.99983)
                ]
            ),
            NuclideData(
                name="Cs137",
                zai=551370,
                half_life_s=949252800.0,  # 30.08 years
                gamma_lines=[
                    GammaLineData(661657.0, 0.8510)
                ]
            ),
            NuclideData(
                name="Eu152",
                zai=631520,
                half_life_s=426826560.0,  # 13.53 years
                gamma_lines=[
                    GammaLineData(121782.0, 0.2841),
                    GammaLineData(244697.0, 0.0755),
                    GammaLineData(344279.0, 0.2657),
                    GammaLineData(411117.0, 0.02238),
                    GammaLineData(443965.0, 0.03125),
                    GammaLineData(778904.0, 0.1296),
                    GammaLineData(867380.0, 0.04214),
                    GammaLineData(964079.0, 0.1462),
                    GammaLineData(1085837.0, 0.1013),
                    GammaLineData(1112076.0, 0.1340),
                    GammaLineData(1408013.0, 0.2085)
                ]
            ),
            NuclideData(
                name="Eu154",
                zai=631540,
                half_life_s=271555200.0,  # 8.6 years
                gamma_lines=[
                    GammaLineData(123071.0, 0.404),
                    GammaLineData(247929.0, 0.0689),
                    GammaLineData(591755.0, 0.0495),
                    GammaLineData(723301.0, 0.201),
                    GammaLineData(873183.0, 0.122),
                    GammaLineData(996329.0, 0.105),
                    GammaLineData(1004725.0, 0.179),
                    GammaLineData(1274436.0, 0.349)
                ]
            ),
            NuclideData(
                name="Ba133",
                zai=561330,
                half_life_s=332622240.0,  # 10.55 years
                gamma_lines=[
                    GammaLineData(80998.0, 0.329),
                    GammaLineData(276399.0, 0.0716),
                    GammaLineData(302851.0, 0.1834),
                    GammaLineData(356013.0, 0.6205),
                    GammaLineData(383849.0, 0.0894)
                ]
            ),
            NuclideData(
                name="Na22",
                zai=110220,
                half_life_s=82085280.0,  # 2.602 years
                gamma_lines=[
                    GammaLineData(1274537.0, 0.9994),
                    GammaLineData(511000.0, 1.798)  # annihilation (2 photons)
                ]
            ),
            # Activation products - foil materials
            NuclideData(
                name="Au198",
                zai=791980,
                half_life_s=232848.0,  # 2.695 days
                gamma_lines=[
                    GammaLineData(411802.0, 0.9562)
                ]
            ),
            NuclideData(
                name="In116m",
                zai=491161,
                half_life_s=3249.0,  # 54.15 min
                gamma_lines=[
                    GammaLineData(1293558.0, 0.848),
                    GammaLineData(1097326.0, 0.585),
                    GammaLineData(416860.0, 0.277)
                ]
            ),
            NuclideData(
                name="Mn56",
                zai=250560,
                half_life_s=9283.2,  # 2.579 hours
                gamma_lines=[
                    GammaLineData(846764.0, 0.9887),
                    GammaLineData(1810726.0, 0.2719),
                    GammaLineData(2113092.0, 0.1433)
                ]
            ),
            NuclideData(
                name="Fe59",
                zai=260590,
                half_life_s=3844540.8,  # 44.5 days
                gamma_lines=[
                    GammaLineData(1099245.0, 0.5650),
                    GammaLineData(1291596.0, 0.4320)
                ]
            ),
            NuclideData(
                name="Sc46",
                zai=210460,
                half_life_s=7239456.0,  # 83.79 days
                gamma_lines=[
                    GammaLineData(889277.0, 0.99984),
                    GammaLineData(1120545.0, 0.99987)
                ]
            ),
            NuclideData(
                name="Ni65",
                zai=280650,
                half_life_s=9072.0,  # 2.52 hours
                gamma_lines=[
                    GammaLineData(1481840.0, 0.2359),
                    GammaLineData(1115546.0, 0.1543)
                ]
            ),
            NuclideData(
                name="Cu64",
                zai=290640,
                half_life_s=45720.0,  # 12.7 hours
                gamma_lines=[
                    GammaLineData(1345770.0, 0.00475),
                    GammaLineData(511000.0, 0.352)  # annihilation
                ]
            ),
            NuclideData(
                name="Zn65",
                zai=300650,
                half_life_s=21079872.0,  # 244 days
                gamma_lines=[
                    GammaLineData(1115546.0, 0.5004),
                    GammaLineData(511000.0, 0.0284)  # annihilation
                ]
            ),
            NuclideData(
                name="W187",
                zai=741870,
                half_life_s=86760.0,  # 24.1 hours
                gamma_lines=[
                    GammaLineData(685774.0, 0.273),
                    GammaLineData(479550.0, 0.218),
                    GammaLineData(551550.0, 0.0508),
                    GammaLineData(618260.0, 0.0628),
                    GammaLineData(772890.0, 0.0412)
                ]
            ),
            # Short-lived for reactor physics
            NuclideData(
                name="Rh104",
                zai=451040,
                half_life_s=42.3,  # 42.3 seconds
                gamma_lines=[
                    GammaLineData(555810.0, 0.0200),
                    GammaLineData(51400.0, 0.472)
                ]
            ),
            NuclideData(
                name="Dy165",
                zai=661650,
                half_life_s=8388.0,  # 2.33 hours
                gamma_lines=[
                    GammaLineData(94700.0, 0.0358),
                    GammaLineData(361680.0, 0.0084)
                ]
            ),
        ]
        
        for nuc in common_nuclides:
            self.add(nuc)


# Global database instance
_global_db: Optional[NuclideDatabase] = None


def get_nuclide_database() -> NuclideDatabase:
    """Get the global nuclide database, initializing if needed."""
    global _global_db
    if _global_db is None:
        _global_db = NuclideDatabase()
        _global_db.load_builtin()
    return _global_db


def get_half_life(nuclide: str) -> Optional[float]:
    """Convenience function to get half-life from global database."""
    return get_nuclide_database().get_half_life(nuclide)


def get_gamma_lines(nuclide: str, min_intensity: float = 0.0) -> List[GammaLineData]:
    """Convenience function to get gamma lines from global database."""
    return get_nuclide_database().get_gamma_lines(nuclide, min_intensity)
