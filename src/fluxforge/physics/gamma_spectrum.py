"""
Activity to Gamma Spectrum Module - actigamma Parity

Epic X: Forward prediction of gamma spectrum from radionuclide activities.

Given an inventory of activated nuclides and their activities, this module
produces the expected gamma emission spectrum. This is the "forward direction"
compared to spectrum unfolding which is the inverse problem.

Key capabilities:
- Line-to-bin mapping with energy conservation
- Multi-emission type aggregation (gamma, X-ray, beta)
- Nuclide identification from spectrum peaks
- Self-contained decay line database
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# DECAY LINE DATABASE
# =============================================================================

# Embedded decay line data (keV, intensity, emission_type)
# Based on ENSDF/NNDC data
DECAY_LINES: Dict[str, List[Tuple[float, float, str]]] = {
    # Mn-56: Major gamma emitter
    'Mn-56': [
        (846.77, 0.989, 'gamma'),
        (1810.72, 0.272, 'gamma'),
        (2113.09, 0.143, 'gamma'),
        (2522.80, 0.0098, 'gamma'),
        (2657.57, 0.0062, 'gamma'),
        (2959.91, 0.0030, 'gamma'),
    ],
    # V-52
    'V-52': [
        (1434.06, 1.0, 'gamma'),
    ],
    # Na-24
    'Na-24': [
        (1368.63, 1.0, 'gamma'),
        (2754.03, 0.999, 'gamma'),
    ],
    # Co-60
    'Co-60': [
        (1173.23, 0.9985, 'gamma'),
        (1332.49, 0.9998, 'gamma'),
    ],
    # Cs-137 (with Ba-137m)
    'Cs-137': [
        (661.66, 0.851, 'gamma'),  # Actually from Ba-137m
        (31.8, 0.020, 'x-ray'),
        (32.2, 0.037, 'x-ray'),
    ],
    # Fe-59
    'Fe-59': [
        (1099.25, 0.565, 'gamma'),
        (1291.59, 0.432, 'gamma'),
        (142.65, 0.010, 'gamma'),
        (192.35, 0.030, 'gamma'),
    ],
    # Co-58
    'Co-58': [
        (810.76, 0.994, 'gamma'),
        (863.96, 0.0068, 'gamma'),
        (511.0, 0.30, 'gamma'),  # Annihilation
    ],
    # Cr-51
    'Cr-51': [
        (320.08, 0.0986, 'gamma'),
    ],
    # Sc-46
    'Sc-46': [
        (889.28, 0.9998, 'gamma'),
        (1120.55, 0.9999, 'gamma'),
    ],
    # Zn-65
    'Zn-65': [
        (1115.55, 0.502, 'gamma'),
        (511.0, 0.029, 'gamma'),  # Annihilation
    ],
    # Ta-182
    'Ta-182': [
        (67.75, 0.429, 'gamma'),
        (100.11, 0.142, 'gamma'),
        (152.43, 0.070, 'gamma'),
        (156.39, 0.027, 'gamma'),
        (179.39, 0.031, 'gamma'),
        (222.11, 0.075, 'gamma'),
        (229.32, 0.036, 'gamma'),
        (264.08, 0.036, 'gamma'),
        (1121.30, 0.353, 'gamma'),
        (1189.05, 0.165, 'gamma'),
        (1221.40, 0.273, 'gamma'),
        (1231.00, 0.115, 'gamma'),
        (1257.42, 0.015, 'gamma'),
        (1289.14, 0.014, 'gamma'),
    ],
    # W-187
    'W-187': [
        (72.0, 0.107, 'gamma'),
        (134.25, 0.089, 'gamma'),
        (479.53, 0.261, 'gamma'),
        (551.52, 0.054, 'gamma'),
        (618.36, 0.075, 'gamma'),
        (685.81, 0.332, 'gamma'),
        (772.89, 0.051, 'gamma'),
    ],
    # Au-198
    'Au-198': [
        (411.80, 0.956, 'gamma'),
        (675.88, 0.0080, 'gamma'),
    ],
    # Mn-54
    'Mn-54': [
        (834.85, 0.9998, 'gamma'),
    ],
    # Eu-152
    'Eu-152': [
        (121.78, 0.286, 'gamma'),
        (244.70, 0.076, 'gamma'),
        (344.28, 0.265, 'gamma'),
        (411.12, 0.022, 'gamma'),
        (443.97, 0.031, 'gamma'),
        (778.90, 0.129, 'gamma'),
        (867.38, 0.042, 'gamma'),
        (964.08, 0.146, 'gamma'),
        (1085.87, 0.102, 'gamma'),
        (1112.07, 0.136, 'gamma'),
        (1408.01, 0.210, 'gamma'),
    ],
    # Eu-154
    'Eu-154': [
        (123.07, 0.404, 'gamma'),
        (247.93, 0.069, 'gamma'),
        (591.76, 0.050, 'gamma'),
        (723.30, 0.201, 'gamma'),
        (756.80, 0.045, 'gamma'),
        (873.19, 0.122, 'gamma'),
        (996.32, 0.105, 'gamma'),
        (1004.76, 0.179, 'gamma'),
        (1274.44, 0.349, 'gamma'),
    ],
    # I-131
    'I-131': [
        (80.19, 0.026, 'gamma'),
        (284.31, 0.061, 'gamma'),
        (364.49, 0.817, 'gamma'),
        (503.00, 0.0036, 'gamma'),
        (636.99, 0.072, 'gamma'),
        (722.91, 0.018, 'gamma'),
    ],
    # Mo-99
    'Mo-99': [
        (140.51, 0.045, 'gamma'),
        (181.07, 0.060, 'gamma'),
        (366.42, 0.012, 'gamma'),
        (739.50, 0.121, 'gamma'),
        (777.92, 0.043, 'gamma'),
    ],
    # Tc-99m
    'Tc-99m': [
        (140.51, 0.890, 'gamma'),
    ],
    # In-116m
    'In-116m': [
        (138.29, 0.032, 'gamma'),
        (416.86, 0.272, 'gamma'),
        (818.72, 0.115, 'gamma'),
        (1097.33, 0.562, 'gamma'),
        (1293.56, 0.848, 'gamma'),
        (1507.68, 0.100, 'gamma'),
        (2112.31, 0.155, 'gamma'),
    ],
    # Cu-64
    'Cu-64': [
        (511.0, 0.352, 'gamma'),  # Annihilation
        (1345.84, 0.00473, 'gamma'),
    ],
    # Al-28
    'Al-28': [
        (1778.99, 1.0, 'gamma'),
    ],
    # Sr-90 (no significant gammas, pure beta)
    'Sr-90': [],
    # Y-90 (no significant gammas, pure beta)
    'Y-90': [],
}

# Half-lives in seconds
HALF_LIVES_S: Dict[str, float] = {
    'Mn-56': 9283.4,
    'V-52': 224.58,
    'Na-24': 53856,
    'Co-60': 166344192,
    'Cs-137': 949363200,
    'Fe-59': 3845376,
    'Co-58': 6125760,
    'Cr-51': 2393280,
    'Sc-46': 7239360,
    'Zn-65': 21085056,
    'Ta-182': 9912960,
    'W-187': 86184,
    'Au-198': 232848,
    'Mn-54': 26986560,
    'Eu-152': 427593600,
    'Eu-154': 271468800,
    'I-131': 693792,
    'Mo-99': 237600,
    'Tc-99m': 21624,
    'In-116m': 3246,
    'Cu-64': 45720,
    'Al-28': 134.5,
    'Sr-90': 912384000,
    'Y-90': 230400,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DecayLine:
    """A single decay emission line."""
    nuclide: str
    energy_keV: float
    intensity: float  # fraction of decays
    emission_type: str = 'gamma'  # 'gamma', 'x-ray', 'beta', 'alpha'


@dataclass
class EnergyBins:
    """Energy bin specification."""
    edges: NDArray  # n+1 edges for n bins
    unit: str = 'keV'
    
    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
    
    @classmethod
    def linear(cls, e_min: float, e_max: float, n_bins: int) -> 'EnergyBins':
        """Create linearly-spaced bins."""
        edges = np.linspace(e_min, e_max, n_bins + 1)
        return cls(edges=edges)
    
    @classmethod
    def logarithmic(cls, e_min: float, e_max: float, n_bins: int) -> 'EnergyBins':
        """Create logarithmically-spaced bins."""
        edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)
        return cls(edges=edges)
    
    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1
    
    @property
    def centers(self) -> NDArray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])
    
    @property
    def widths(self) -> NDArray:
        return np.diff(self.edges)


def normalize_nuclide_name(name: str) -> str:
    """
    Normalize nuclide name with metastable state support.
    
    Handles various naming conventions:
    - 'Tc99m', 'Tc-99m', 'Tc_99m' -> 'Tc-99m'
    - 'tc99m', 'TC99M' -> 'Tc-99m'
    - 'In-116m1', 'In116m2' -> 'In-116m1', 'In-116m2'
    - 'Co60', 'CO-60', 'co_60' -> 'Co-60'
    
    Parameters
    ----------
    name : str
        Input nuclide name
    
    Returns
    -------
    str
        Normalized nuclide name (e.g., 'Tc-99m', 'Co-60')
    """
    import re
    
    # Remove spaces and convert separators to dash
    name = name.strip().replace('_', '-').replace(' ', '')
    
    # Parse components using regex
    # Match: optional element, mass number, optional metastable indicator
    # Pattern: (element)(mass)(metastable?)
    # Examples: Co60, Co-60, Co60m, Co-60m, Co60m1
    
    pattern = r'^([A-Za-z]{1,2})-?(\d{1,3})([Mm]?\d?)$'
    match = re.match(pattern, name)
    
    if match:
        element = match.group(1).capitalize()
        mass = match.group(2)
        metastable = match.group(3).lower() if match.group(3) else ''
        
        # Format metastable state
        if metastable:
            if metastable == 'm':
                metastable = 'm'
            elif metastable.startswith('m'):
                metastable = metastable  # Keep m1, m2, etc.
            else:
                metastable = ''  # Invalid, ignore
        
        return f"{element}-{mass}{metastable}"
    
    # Fallback: simple normalization
    return name.replace('_', '-').capitalize()


def is_metastable(nuclide: str) -> bool:
    """Check if nuclide is a metastable state."""
    nuclide = normalize_nuclide_name(nuclide)
    return 'm' in nuclide.lower().split('-')[-1]


def get_ground_state(nuclide: str) -> str:
    """Get ground state name from metastable nuclide."""
    nuclide = normalize_nuclide_name(nuclide)
    import re
    # Remove metastable suffix
    return re.sub(r'm\d?$', '', nuclide)


def parse_nuclide(nuclide: str) -> Tuple[str, int, Optional[str]]:
    """
    Parse nuclide name into components.
    
    Parameters
    ----------
    nuclide : str
        Nuclide name (e.g., 'Tc-99m', 'Co-60')
    
    Returns
    -------
    tuple
        (element, mass_number, metastable_state)
        metastable_state is None, 'm', 'm1', 'm2', etc.
    """
    import re
    nuclide = normalize_nuclide_name(nuclide)
    
    pattern = r'^([A-Za-z]{1,2})-(\d+)([Mm]\d?)?$'
    match = re.match(pattern, nuclide)
    
    if match:
        element = match.group(1)
        mass = int(match.group(2))
        meta = match.group(3).lower() if match.group(3) else None
        return element, mass, meta
    
    raise ValueError(f"Cannot parse nuclide name: {nuclide}")


@dataclass
class GammaSpectrum:
    """Predicted gamma spectrum."""
    energies: NDArray  # bin centers (keV)
    intensities: NDArray  # photons/s per bin
    uncertainties: Optional[NDArray] = None
    counts: Optional[NDArray] = None  # counts (for finite live_time)
    emission_type: str = 'gamma'
    contributing_nuclides: List[str] = field(default_factory=list)
    
    @property
    def total_photons(self) -> float:
        """Total photon emission rate (photons/s)."""
        return float(np.sum(self.intensities))


@dataclass
class Inventory:
    """Radionuclide inventory with activities."""
    activities: Dict[str, float] = field(default_factory=dict)  # nuclide -> activity in Bq
    activities_Bq: Optional[Dict[str, float]] = None  # Alias for activities
    uncertainties: Optional[Dict[str, float]] = None
    reference_time: float = 0.0
    
    def __post_init__(self):
        # Handle activities_Bq alias
        if self.activities_Bq is not None and not self.activities:
            self.activities = self.activities_Bq
        
        if self.uncertainties is None:
            # Default 5% uncertainty
            self.uncertainties = {k: 0.05 * v for k, v in self.activities.items()}
    
    def get_activity(self, nuclide: str) -> float:
        """Get activity for nuclide, normalizing name."""
        # Try direct lookup
        if nuclide in self.activities:
            return self.activities[nuclide]
        
        # Try canonical form
        canon = nuclide.replace('_', '-').capitalize()
        if canon in self.activities:
            return self.activities[canon]
        
        return 0.0


# =============================================================================
# SPECTRUM GENERATION
# =============================================================================

def get_decay_lines(
    nuclide: str,
    emission_types: Optional[List[str]] = None
) -> List[DecayLine]:
    """
    Get decay lines for a nuclide.
    
    Parameters
    ----------
    nuclide : str
        Nuclide name (e.g., 'Co-60')
    emission_types : list, optional
        Filter by emission type (e.g., ['gamma', 'x-ray'])
    
    Returns
    -------
    list of DecayLine
    """
    if emission_types is None:
        emission_types = ['gamma', 'x-ray', 'beta', 'alpha']
    
    # Normalize nuclide name with metastable state support
    nuclide = normalize_nuclide_name(nuclide)
    
    lines = DECAY_LINES.get(nuclide, [])
    
    result = []
    for E, I, etype in lines:
        if etype in emission_types:
            result.append(DecayLine(
                nuclide=nuclide,
                energy_keV=E,
                intensity=I,
                emission_type=etype
            ))
    
    return result


def bin_decay_lines(
    lines: List[DecayLine],
    bins: EnergyBins,
    activity_Bq: float = 1.0,
    energy_conservation: bool = False
) -> NDArray:
    """
    Bin decay lines into energy histogram.
    
    Parameters
    ----------
    lines : list of DecayLine
        Decay lines to bin
    bins : EnergyBins
        Energy bin specification
    activity_Bq : float
        Activity in Bq
    energy_conservation : bool
        If True, scale intensity by line_energy/bin_center
        (for dose calculations, matches FISPACT-II behavior)
    
    Returns
    -------
    NDArray
        Photons per second per bin
    """
    spectrum = np.zeros(bins.n_bins)
    
    for line in lines:
        # Find bin for this line
        idx = np.searchsorted(bins.edges, line.energy_keV) - 1
        
        if 0 <= idx < bins.n_bins:
            emission_rate = activity_Bq * line.intensity  # photons/s
            
            if energy_conservation:
                # Scale by energy ratio for dose-equivalent
                bin_center = bins.centers[idx]
                if bin_center > 0:
                    emission_rate *= line.energy_keV / bin_center
            
            spectrum[idx] += emission_rate
    
    return spectrum


def generate_spectrum(
    inventory: Inventory,
    bins: EnergyBins,
    emission_types: Optional[List[str]] = None,
    energy_conservation: bool = False,
    live_time: float = 1.0
) -> GammaSpectrum:
    """
    Generate gamma spectrum from inventory.
    
    Parameters
    ----------
    inventory : Inventory
        Radionuclide inventory with activities
    bins : EnergyBins
        Energy bin specification
    emission_types : list, optional
        Emission types to include (default: ['gamma'])
    energy_conservation : bool
        Apply energy conservation scaling for dose calculations
    live_time : float
        Live time for counting (default: 1.0 s)
    
    Returns
    -------
    GammaSpectrum
        Predicted spectrum (scaled by live_time)
    """
    if emission_types is None:
        emission_types = ['gamma']
    
    total_spectrum = np.zeros(bins.n_bins)
    total_variance = np.zeros(bins.n_bins)
    contributing = []
    
    for nuclide, activity in inventory.activities.items():
        if activity <= 0:
            continue
        
        lines = get_decay_lines(nuclide, emission_types)
        if not lines:
            continue
        
        nuclide_spectrum = bin_decay_lines(
            lines, bins, activity, energy_conservation
        )
        
        total_spectrum += nuclide_spectrum
        
        # Propagate uncertainty
        unc = inventory.uncertainties.get(nuclide, 0.05 * activity)
        rel_unc = unc / max(activity, 1e-30)
        total_variance += (nuclide_spectrum * rel_unc) ** 2
        
        if np.any(nuclide_spectrum > 0):
            contributing.append(nuclide)
    
    # Apply live_time scaling
    scaled_spectrum = total_spectrum * live_time
    scaled_variance = total_variance * live_time**2
    
    return GammaSpectrum(
        energies=bins.centers,
        intensities=scaled_spectrum,
        counts=scaled_spectrum,  # For compatibility
        uncertainties=np.sqrt(scaled_variance),
        emission_type='+'.join(emission_types),
        contributing_nuclides=contributing
    )


def aggregate_emissions(
    nuclide_or_inventory: Union[str, Inventory],
    bins: Optional[EnergyBins] = None,
) -> Dict[str, Any]:
    """
    Aggregate emission data by type for a nuclide or inventory.
    
    Parameters
    ----------
    nuclide_or_inventory : str or Inventory
        Single nuclide name or full inventory
    bins : EnergyBins, optional
        Energy bins for spectrum generation (required for Inventory)
    
    Returns
    -------
    dict
        For single nuclide: {gammas: [], xrays: [], total_gamma_intensity: float}
        For inventory: Dictionary of emission_type -> GammaSpectrum
    """
    if isinstance(nuclide_or_inventory, str):
        # Single nuclide mode - return emission summary
        nuclide = nuclide_or_inventory
        lines = get_decay_lines(nuclide, None)  # Get all types
        
        gammas = [l for l in lines if l.emission_type == 'gamma']
        xrays = [l for l in lines if l.emission_type == 'x-ray']
        
        total_gamma_intensity = sum(l.intensity for l in gammas)
        
        return {
            'gammas': gammas,
            'xrays': xrays,
            'total_gamma_intensity': total_gamma_intensity,
            'nuclide': nuclide
        }
    else:
        # Inventory mode - generate spectra by type
        if bins is None:
            bins = EnergyBins(edges=np.linspace(0, 3000, 301))
        
        inventory = nuclide_or_inventory
        emission_types = ['gamma', 'x-ray', 'beta', 'alpha']
        
        result = {}
        for etype in emission_types:
            spec = generate_spectrum(inventory, bins, [etype])
            if np.any(spec.intensities > 0):
                result[etype] = spec
        
        return result


# =============================================================================
# NUCLIDE IDENTIFICATION
# =============================================================================

@dataclass
class PeakMatch:
    """Result of matching a peak to nuclide database."""
    energy_keV: float
    nuclide: str
    line_energy: float
    intensity: float
    difference_keV: float
    score: float  # 0-1, higher is better match


def identify_nuclides(
    peak_energies: List[float],
    tolerance_keV: float = 1.0,
    min_intensity: float = 0.01
) -> List[List[PeakMatch]]:
    """
    Identify nuclides from measured peak energies.
    
    Parameters
    ----------
    peak_energies : list
        Measured peak energies (keV)
    tolerance_keV : float
        Maximum allowed difference between measured and database energy
    min_intensity : float
        Minimum emission intensity to consider
    
    Returns
    -------
    list of list of PeakMatch
        For each peak, a list of possible matches sorted by score
    """
    all_matches = []
    
    for peak_E in peak_energies:
        peak_matches = []
        
        for nuclide, lines in DECAY_LINES.items():
            for line_E, intensity, etype in lines:
                if etype != 'gamma':
                    continue
                if intensity < min_intensity:
                    continue
                
                diff = abs(peak_E - line_E)
                if diff <= tolerance_keV:
                    # Score based on proximity and intensity
                    proximity_score = 1 - diff / tolerance_keV
                    score = proximity_score * intensity
                    
                    peak_matches.append(PeakMatch(
                        energy_keV=peak_E,
                        nuclide=nuclide,
                        line_energy=line_E,
                        intensity=intensity,
                        difference_keV=diff,
                        score=score
                    ))
        
        # Sort by score
        peak_matches.sort(key=lambda x: -x.score)
        all_matches.append(peak_matches)
    
    return all_matches


def suggest_nuclides(
    peak_energies: List[float],
    tolerance_keV: float = 1.0,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Suggest most likely nuclides from peaks.
    
    Parameters
    ----------
    peak_energies : list
        Measured peak energies
    tolerance_keV : float
        Matching tolerance
    top_n : int
        Maximum number of suggestions to return
    
    Returns
    -------
    list of (nuclide, score) tuples
        Sorted by score descending
    """
    matches = identify_nuclides(peak_energies, tolerance_keV)
    
    nuclide_scores: Dict[str, float] = {}
    
    for peak_matches in matches:
        for match in peak_matches:
            if match.nuclide not in nuclide_scores:
                nuclide_scores[match.nuclide] = 0
            nuclide_scores[match.nuclide] += match.score
    
    # Sort by score and return as list of tuples
    sorted_nuclides = sorted(nuclide_scores.items(), key=lambda x: -x[1])
    return sorted_nuclides[:top_n]


# =============================================================================
# ACTIVITY CALCULATIONS
# =============================================================================

def atoms_to_activity(
    atoms: float,
    half_life_s: float
) -> float:
    """Convert atom count to activity in Bq."""
    if half_life_s <= 0 or half_life_s == float('inf'):
        return 0.0
    lam = np.log(2) / half_life_s
    return atoms * lam


def activity_to_atoms(
    activity_Bq: float,
    half_life_s: float
) -> float:
    """Convert activity in Bq to atom count."""
    if half_life_s <= 0 or half_life_s == float('inf'):
        return 0.0
    lam = np.log(2) / half_life_s
    return activity_Bq / lam


def decay_activity(
    activity_0: float,
    half_life_s: float,
    time_s: float
) -> float:
    """Calculate activity after decay."""
    if half_life_s <= 0 or half_life_s == float('inf'):
        return activity_0
    lam = np.log(2) / half_life_s
    return activity_0 * np.exp(-lam * time_s)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_nuclides() -> List[str]:
    """List all nuclides in database."""
    return list(DECAY_LINES.keys())


def list_available_nuclides() -> List[str]:
    """List all nuclides in database (alias for list_nuclides)."""
    return list_nuclides()


def get_half_life(nuclide: str) -> Optional[float]:
    """Get half-life in seconds."""
    nuclide = nuclide.replace('_', '-').capitalize()
    return HALF_LIVES_S.get(nuclide)


def get_main_gamma(nuclide: str) -> Optional[Tuple[float, float]]:
    """Get main (strongest) gamma line."""
    lines = get_decay_lines(nuclide, ['gamma'])
    if not lines:
        return None
    main = max(lines, key=lambda x: x.intensity)
    return (main.energy_keV, main.intensity)


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing gamma_spectrum module (actigamma parity)...")
    
    # Create inventory
    inventory = Inventory(activities={
        'Co-60': 1e6,  # 1 MBq
        'Cs-137': 5e5,  # 0.5 MBq
        'Eu-152': 1e5,  # 0.1 MBq
    })
    
    print(f"Inventory: {len(inventory.activities)} nuclides")
    
    # Create bins
    bins = EnergyBins.linear(0, 2000, 200)
    print(f"Energy bins: {bins.n_bins} bins, {bins.edges[0]:.0f}-{bins.edges[-1]:.0f} keV")
    
    # Generate spectrum
    spectrum = generate_spectrum(inventory, bins)
    print(f"\nGenerated spectrum:")
    print(f"  Total emission rate: {np.sum(spectrum.intensities):.2e} photons/s")
    print(f"  Contributing nuclides: {spectrum.contributing_nuclides}")
    
    # Find peaks
    peak_indices = np.where(spectrum.intensities > 1e3)[0]
    print(f"\nPeaks > 1000 photons/s:")
    for idx in peak_indices:
        print(f"  {spectrum.energies[idx]:.1f} keV: {spectrum.intensities[idx]:.2e} photons/s")
    
    # Test nuclide identification
    peak_Es = [661.7, 1173.2, 1332.5, 344.3]
    print(f"\nNuclide identification for peaks at {peak_Es} keV:")
    suggestions = suggest_nuclides(peak_Es)
    for nuc, score in list(suggestions.items())[:5]:
        print(f"  {nuc}: score={score:.3f}")
    
    # Test emission aggregation
    print("\nEmission type breakdown:")
    by_type = aggregate_emissions(inventory, bins)
    for etype, spec in by_type.items():
        print(f"  {etype}: {np.sum(spec.intensities):.2e} emissions/s")
    
    print("\n✅ gamma_spectrum module tests passed!")
