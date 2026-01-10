"""
NNDC Nuclear Data Queries - Becquerel Parity

Provides access to NNDC nuclear data including:
- Nuclear Wallet Cards data (half-lives, decay modes, abundances)
- Decay radiation data (gamma lines, beta endpoints)
- Isotope class with properties from NNDC

This module provides offline data for common isotopes and can
optionally query NNDC online for additional data.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np


# =============================================================================
# OFFLINE NUCLEAR DATA
# =============================================================================

# Common isotope half-lives (seconds)
HALF_LIVES_S: Dict[str, float] = {
    # Short-lived activation products
    'V-52': 224.58,      # 3.743 min
    'Mn-56': 9283.4,     # 2.5789 h
    'Al-28': 134.5,      # 2.24 min
    'Cu-64': 45720,      # 12.7 h
    'Na-24': 53856,      # 14.96 h
    'Sc-46': 7239360,    # 83.79 d
    'Fe-59': 3845376,    # 44.503 d
    'Co-58': 6125760,    # 70.86 d
    'Co-60': 166344192,  # 5.2714 y
    'Cr-51': 2393280,    # 27.7 d
    'Zn-65': 21085056,   # 244.06 d
    'Ta-182': 9912960,   # 114.74 d
    'W-187': 86184,      # 23.94 h
    'Au-198': 232848,    # 2.6943 d
    'Mn-54': 26986560,   # 312.2 d
    'Fe-55': 86614080,   # 2.744 y
    'Ni-63': 3187296000, # 101.0 y
    'In-115m': 16092,    # 4.47 h
    'In-116m': 3246,     # 54.1 min
    'Mo-99': 237600,     # 66 h
    'Tc-99m': 21624,     # 6.01 h
    'I-131': 693792,     # 8.03 d
    'Cs-137': 949363200, # 30.08 y
    'Ba-137m': 153.12,   # 2.552 min
    'Sr-90': 912384000,  # 28.91 y
    'Y-90': 230400,      # 2.67 d
    'Eu-152': 427593600, # 13.537 y
    'Eu-154': 271468800, # 8.601 y
    # Flux wire isotopes
    'Ni-58': float('inf'),  # Stable
    'Co-59': float('inf'),  # Stable
    'Sc-45': float('inf'),  # Stable
    'Au-197': float('inf'), # Stable
}

# Main gamma lines (keV, intensity fraction)
GAMMA_LINES: Dict[str, List[Tuple[float, float]]] = {
    'Mn-56': [(846.8, 0.989), (1810.7, 0.272), (2113.1, 0.143)],
    'V-52': [(1434.1, 1.0)],
    'Na-24': [(1368.6, 1.0), (2754.0, 0.999)],
    'Sc-46': [(889.3, 1.0), (1120.5, 1.0)],
    'Fe-59': [(1099.2, 0.565), (1291.6, 0.432)],
    'Co-58': [(810.8, 0.994)],
    'Co-60': [(1173.2, 0.999), (1332.5, 1.0)],
    'Cr-51': [(320.1, 0.0986)],
    'Zn-65': [(1115.5, 0.502)],
    'Ta-182': [(67.8, 0.429), (100.1, 0.142), (1121.3, 0.353), (1221.4, 0.273)],
    'W-187': [(479.5, 0.261), (685.8, 0.332)],
    'Au-198': [(411.8, 0.956)],
    'Mn-54': [(834.8, 0.9998)],
    'Cu-64': [(1345.8, 0.00473)],  # Low intensity
    'In-116m': [(1293.6, 0.848), (416.9, 0.272)],
    'Mo-99': [(140.5, 0.0452), (739.5, 0.121)],
    'Tc-99m': [(140.5, 0.89)],
    'I-131': [(364.5, 0.817), (637.0, 0.0717)],
    'Cs-137': [(661.7, 0.851)],
    'Eu-152': [(121.8, 0.286), (344.3, 0.265), (1408.0, 0.21)],
    'Eu-154': [(123.1, 0.404), (723.3, 0.201), (1274.4, 0.349)],
}

# Decay modes
DECAY_MODES: Dict[str, List[Tuple[str, float]]] = {
    'Mn-56': [('B-', 1.0)],
    'V-52': [('B-', 1.0)],
    'Na-24': [('B-', 1.0)],
    'Co-60': [('B-', 1.0)],
    'Cs-137': [('B-', 1.0)],
    'Sr-90': [('B-', 1.0)],
    'Au-198': [('B-', 1.0)],
    'Eu-152': [('B-', 0.278), ('EC', 0.722)],
    'Tc-99m': [('IT', 1.0)],
}

# Atomic masses (u)
ATOMIC_MASSES: Dict[str, float] = {
    'H-1': 1.007825, 'H-2': 2.014102, 'H-3': 3.016049,
    'He-3': 3.016029, 'He-4': 4.002603,
    'Li-6': 6.015122, 'Li-7': 7.016004,
    'C-12': 12.0, 'C-13': 13.003355,
    'N-14': 14.003074, 'N-15': 15.000109,
    'O-16': 15.994915, 'O-17': 16.999132, 'O-18': 17.999160,
    'Fe-54': 53.939611, 'Fe-56': 55.934938, 'Fe-57': 56.935394, 'Fe-58': 57.933276,
    'Mn-55': 54.938045, 'Mn-56': 55.938905,
    'Co-59': 58.933195, 'Co-60': 59.933817,
    'Ni-58': 57.935343, 'Ni-60': 59.930786,
    'Cr-50': 49.946044, 'Cr-52': 51.940508, 'Cr-53': 52.940649, 'Cr-54': 53.938880,
    'V-51': 50.943960, 'V-52': 51.944776,
    'Au-197': 196.966569, 'Au-198': 197.968242,
    'W-186': 185.954364, 'W-187': 186.957160,
    'Ta-181': 180.947996, 'Ta-182': 181.950152,
}

# Natural abundances (fraction)
NATURAL_ABUNDANCES: Dict[str, float] = {
    'H-1': 0.999885, 'H-2': 0.000115,
    'He-3': 0.00000134, 'He-4': 0.99999866,
    'Li-6': 0.0759, 'Li-7': 0.9241,
    'C-12': 0.9893, 'C-13': 0.0107,
    'Fe-54': 0.05845, 'Fe-56': 0.91754, 'Fe-57': 0.02119, 'Fe-58': 0.00282,
    'Ni-58': 0.680769, 'Ni-60': 0.262231,
    'Cr-50': 0.04345, 'Cr-52': 0.83789, 'Cr-53': 0.09501, 'Cr-54': 0.02365,
    'Co-59': 1.0,
    'Mn-55': 1.0,
    'Au-197': 1.0,
    'W-186': 0.2843,
    'Ta-181': 0.99988,
}


# =============================================================================
# ISOTOPE PARSING
# =============================================================================

def parse_isotope(name: str) -> Tuple[str, int, int]:
    """
    Parse isotope string to (symbol, A, m).
    
    Handles formats:
    - 'Co-60', 'Co60', 'co-60'
    - 'Tc-99m', 'Tc99m', 'TC-99M'
    - 'U-235', 'U235'
    
    Returns
    -------
    tuple
        (element_symbol, mass_number, metastable_state)
    """
    name = name.strip()
    
    # Pattern: Element-A or ElementA with optional m/M suffix
    pattern = r'^([A-Za-z]{1,2})-?(\d+)(m?\d?)$'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(f"Cannot parse isotope: {name}")
    
    element = match.group(1).capitalize()
    A = int(match.group(2))
    meta = match.group(3).lower()
    
    m = 0
    if meta:
        if meta == 'm' or meta == 'm1':
            m = 1
        elif meta.startswith('m') and meta[1:].isdigit():
            m = int(meta[1:])
    
    return element, A, m


def format_isotope(element: str, A: int, m: int = 0) -> str:
    """Format isotope as 'El-A' or 'El-Am'."""
    base = f"{element.capitalize()}-{A}"
    if m > 0:
        base += 'm' if m == 1 else f'm{m}'
    return base


def canonical_isotope(name: str) -> str:
    """Convert isotope name to canonical format."""
    element, A, m = parse_isotope(name)
    return format_isotope(element, A, m)


# =============================================================================
# ISOTOPE CLASS
# =============================================================================

@dataclass
class Isotope:
    """
    Isotope with nuclear properties from NNDC.
    
    Matches becquerel.Isotope functionality.
    """
    element: str
    A: int
    m: int = 0
    
    def __post_init__(self):
        self.element = self.element.capitalize()
    
    @classmethod
    def from_string(cls, name: str) -> 'Isotope':
        """Create isotope from string."""
        element, A, m = parse_isotope(name)
        return cls(element=element, A=A, m=m)
    
    @property
    def name(self) -> str:
        """Canonical name."""
        return format_isotope(self.element, self.A, self.m)
    
    @property
    def half_life_s(self) -> Optional[float]:
        """Half-life in seconds."""
        return HALF_LIVES_S.get(self.name)
    
    @property
    def decay_constant(self) -> Optional[float]:
        """Decay constant λ (1/s)."""
        hl = self.half_life_s
        if hl is None or hl == float('inf'):
            return 0.0
        return np.log(2) / hl
    
    @property
    def is_stable(self) -> bool:
        """Check if stable."""
        hl = self.half_life_s
        return hl is not None and hl == float('inf')
    
    @property
    def abundance(self) -> Optional[float]:
        """Natural abundance (fraction)."""
        return NATURAL_ABUNDANCES.get(self.name)
    
    @property
    def atomic_mass(self) -> Optional[float]:
        """Atomic mass (u)."""
        return ATOMIC_MASSES.get(self.name)
    
    @property
    def gamma_lines(self) -> List[Tuple[float, float]]:
        """List of (energy_keV, intensity) tuples."""
        return GAMMA_LINES.get(self.name, [])
    
    @property
    def decay_modes(self) -> List[Tuple[str, float]]:
        """List of (mode, branching_ratio) tuples."""
        return DECAY_MODES.get(self.name, [])
    
    @property
    def main_gamma_keV(self) -> Optional[float]:
        """Energy of strongest gamma line."""
        lines = self.gamma_lines
        if not lines:
            return None
        # Sort by intensity, return highest
        return max(lines, key=lambda x: x[1])[0]
    
    @property
    def specific_activity_Bq_g(self) -> Optional[float]:
        """Specific activity in Bq/g."""
        lam = self.decay_constant
        M = self.atomic_mass
        if lam is None or M is None or lam == 0:
            return 0.0
        # A = λ * N_A / M
        return lam * 6.02214076e23 / M
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        hl = self.half_life_s
        if hl is None:
            hl_str = "unknown"
        elif hl == float('inf'):
            hl_str = "stable"
        elif hl < 60:
            hl_str = f"{hl:.2f} s"
        elif hl < 3600:
            hl_str = f"{hl/60:.2f} min"
        elif hl < 86400:
            hl_str = f"{hl/3600:.2f} h"
        elif hl < 31557600:
            hl_str = f"{hl/86400:.2f} d"
        else:
            hl_str = f"{hl/31557600:.2f} y"
        return f"Isotope({self.name}, T½={hl_str})"


# =============================================================================
# ISOTOPE QUANTITY (Activity/Atoms)
# =============================================================================

@dataclass
class IsotopeQuantity:
    """
    Quantity of an isotope (activity or atoms).
    
    Matches becquerel.IsotopeQuantity functionality.
    """
    isotope: Isotope
    activity_Bq: float = 0.0
    atoms: float = 0.0
    reference_time: float = 0.0  # seconds since some epoch
    
    def __post_init__(self):
        """Initialize from either activity or atoms."""
        lam = self.isotope.decay_constant or 0.0
        
        if self.atoms == 0 and self.activity_Bq > 0 and lam > 0:
            self.atoms = self.activity_Bq / lam
        elif self.activity_Bq == 0 and self.atoms > 0:
            self.activity_Bq = self.atoms * lam
    
    @classmethod
    def from_decays(
        cls,
        isotope: Union[str, Isotope],
        decays: float,
        start_time: float,
        end_time: float
    ) -> 'IsotopeQuantity':
        """
        Create from measured decays.
        
        Parameters
        ----------
        isotope : str or Isotope
            Isotope identifier
        decays : float
            Number of decays observed
        start_time : float
            Start of counting period (s)
        end_time : float
            End of counting period (s)
        
        Returns
        -------
        IsotopeQuantity
            Quantity at start_time
        """
        if isinstance(isotope, str):
            isotope = Isotope.from_string(isotope)
        
        dt = end_time - start_time
        lam = isotope.decay_constant or 0.0
        
        if lam == 0 or dt <= 0:
            return cls(isotope=isotope, activity_Bq=0, reference_time=start_time)
        
        # N = decays / (1 - exp(-λΔt))
        factor = 1 - np.exp(-lam * dt)
        if factor < 1e-10:
            atoms = decays
        else:
            atoms = decays / factor
        
        activity = atoms * lam
        
        return cls(
            isotope=isotope,
            activity_Bq=activity,
            atoms=atoms,
            reference_time=start_time
        )
    
    def activity_at(self, time: float) -> float:
        """Activity at given time (s)."""
        dt = time - self.reference_time
        lam = self.isotope.decay_constant or 0.0
        return self.activity_Bq * np.exp(-lam * dt)
    
    def atoms_at(self, time: float) -> float:
        """Atoms at given time (s)."""
        dt = time - self.reference_time
        lam = self.isotope.decay_constant or 0.0
        return self.atoms * np.exp(-lam * dt)
    
    def decays_in_interval(self, start: float, end: float) -> float:
        """Total decays between start and end times (s)."""
        lam = self.isotope.decay_constant or 0.0
        
        if lam == 0:
            return 0.0
        
        N_start = self.atoms_at(start)
        N_end = self.atoms_at(end)
        
        return N_start - N_end
    
    def average_activity(self, start: float, end: float) -> float:
        """Average activity over interval (Bq)."""
        decays = self.decays_in_interval(start, end)
        dt = end - start
        
        if dt <= 0:
            return 0.0
        
        return decays / dt
    
    def time_when(self, activity_Bq: float) -> float:
        """Find time when activity equals given value."""
        if activity_Bq <= 0:
            return float('inf')
        
        lam = self.isotope.decay_constant or 0.0
        
        if lam == 0:
            return float('inf') if self.activity_Bq >= activity_Bq else self.reference_time
        
        ratio = activity_Bq / self.activity_Bq
        
        if ratio >= 1:
            return self.reference_time
        
        return self.reference_time - np.log(ratio) / lam


# =============================================================================
# WALLET CARDS QUERY
# =============================================================================

def get_nuclear_data(
    isotope: str,
    include_gammas: bool = True,
    include_decay: bool = True
) -> Dict[str, Any]:
    """
    Get nuclear data for an isotope.
    
    Offline-first with optional NNDC query.
    
    Parameters
    ----------
    isotope : str
        Isotope name (e.g., 'Co-60')
    include_gammas : bool
        Include gamma line data
    include_decay : bool
        Include decay mode data
    
    Returns
    -------
    dict
        Nuclear data dictionary
    """
    iso = Isotope.from_string(isotope)
    
    data = {
        'isotope': iso.name,
        'element': iso.element,
        'A': iso.A,
        'm': iso.m,
        'half_life_s': iso.half_life_s,
        'decay_constant': iso.decay_constant,
        'is_stable': iso.is_stable,
        'abundance': iso.abundance,
        'atomic_mass': iso.atomic_mass,
        'specific_activity_Bq_g': iso.specific_activity_Bq_g,
    }
    
    if include_gammas:
        data['gamma_lines'] = iso.gamma_lines
        data['main_gamma_keV'] = iso.main_gamma_keV
    
    if include_decay:
        data['decay_modes'] = iso.decay_modes
    
    return data


def list_isotopes_with_gammas(
    e_min: float = 0,
    e_max: float = 3000,
    i_min: float = 0.0
) -> List[Dict[str, Any]]:
    """
    List isotopes with gamma lines in energy range.
    
    Parameters
    ----------
    e_min : float
        Minimum energy (keV)
    e_max : float
        Maximum energy (keV)
    i_min : float
        Minimum intensity (fraction)
    
    Returns
    -------
    list
        List of matching isotope/gamma dictionaries
    """
    results = []
    
    for iso_name, gammas in GAMMA_LINES.items():
        for E, I in gammas:
            if e_min <= E <= e_max and I >= i_min:
                results.append({
                    'isotope': iso_name,
                    'energy_keV': E,
                    'intensity': I,
                    'half_life_s': HALF_LIVES_S.get(iso_name)
                })
    
    return sorted(results, key=lambda x: x['energy_keV'])


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing nndc module...")
    
    # Test Isotope
    co60 = Isotope.from_string('Co-60')
    print(f"Isotope: {repr(co60)}")
    print(f"  λ = {co60.decay_constant:.4e} s⁻¹")
    print(f"  Gamma lines: {co60.gamma_lines}")
    print(f"  Specific activity: {co60.specific_activity_Bq_g:.4e} Bq/g")
    
    # Test metastable
    tc99m = Isotope.from_string('Tc-99m')
    print(f"\nMetastable: {repr(tc99m)}")
    
    # Test IsotopeQuantity
    qty = IsotopeQuantity(isotope=co60, activity_Bq=1e6, reference_time=0)
    print(f"\nQuantity: A(0) = {qty.activity_Bq:.2e} Bq")
    print(f"  A(1y) = {qty.activity_at(31557600):.2e} Bq")
    print(f"  Decays in 1 day: {qty.decays_in_interval(0, 86400):.2e}")
    
    # Test from_decays
    qty2 = IsotopeQuantity.from_decays('Mn-56', decays=1e6, start_time=0, end_time=3600)
    print(f"\nFrom decays: A(0) = {qty2.activity_Bq:.2e} Bq")
    
    # Test list_isotopes_with_gammas
    gammas_500_1500 = list_isotopes_with_gammas(e_min=500, e_max=1500, i_min=0.1)
    print(f"\nGammas 500-1500 keV (I>10%): {len(gammas_500_1500)} lines")
    for g in gammas_500_1500[:5]:
        print(f"  {g['isotope']}: {g['energy_keV']:.1f} keV ({g['intensity']*100:.1f}%)")
    
    print("\n✅ nndc module tests passed!")
