"""
Gamma Line Database Module for FluxForge.

Provides access to gamma, x-ray, and other decay radiation line data
for isotope identification and spectroscopy analysis.

Compatible with actigamma's decay_2012 data format and extensible
to support ENDF, IAEA ENSDF, and other nuclear databases.

Based on the actigamma project structure by Thomas Sherwin Sherwin.
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Path to actigamma data (if installed)
ACTIGAMMA_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..", 
    "testing", "actigamma", "actigamma", "data",
    "lines_decay_2012.min.json"
)

# Alternative paths to search for actigamma data
def find_actigamma_data():
    """Search for actigamma data in various locations."""
    search_paths = [
        # Relative to this file going up to project root
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "..", "..",
                     "testing", "actigamma", "actigamma", "data",
                     "lines_decay_2012.min.json"),
        # Common installation paths
        os.path.expanduser("~/.fluxforge/data/lines_decay_2012.min.json"),
        "/usr/share/fluxforge/data/lines_decay_2012.min.json",
        # Try environment variable
        os.path.join(os.environ.get("FLUXFORGE_DATA", ""), "lines_decay_2012.min.json"),
        # Try finding in actigamma package
    ]
    
    for path in search_paths:
        if path and os.path.exists(path):
            return path
    
    # Try to find actigamma via import
    try:
        import actigamma
        actigamma_path = os.path.dirname(actigamma.__file__)
        data_path = os.path.join(actigamma_path, "data", "lines_decay_2012.min.json")
        if os.path.exists(data_path):
            return data_path
    except ImportError:
        pass
    
    return None


# Alternative: FluxForge's own data directory
FLUXFORGE_GAMMA_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gamma_lines.json"
)


@dataclass
class GammaLine:
    """Represents a single gamma or decay line."""
    energy: float  # eV
    energy_unc: float  # eV uncertainty
    intensity: float  # emission probability (0-1 or percentage)
    intensity_unc: float  # uncertainty
    norm: float = 1.0  # normalization factor
    norm_unc: float = 0.0
    
    @property
    def energy_keV(self) -> float:
        """Energy in keV."""
        return self.energy / 1000.0
    
    @property
    def energy_MeV(self) -> float:
        """Energy in MeV."""
        return self.energy / 1e6
    
    def __repr__(self) -> str:
        return f"GammaLine({self.energy_keV:.2f} keV, I={self.intensity:.4f}±{self.intensity_unc:.4f})"


@dataclass
class DecayData:
    """Decay data for a nuclide."""
    nuclide: str
    zai: int  # Z*10000 + A*10 + isomeric state
    halflife: float  # seconds
    gamma_lines: List[GammaLine] = field(default_factory=list)
    xray_lines: List[GammaLine] = field(default_factory=list)
    beta_lines: List[GammaLine] = field(default_factory=list)
    alpha_lines: List[GammaLine] = field(default_factory=list)
    ec_lines: List[GammaLine] = field(default_factory=list)
    electron_lines: List[GammaLine] = field(default_factory=list)
    
    @property
    def halflife_hours(self) -> float:
        """Halflife in hours."""
        return self.halflife / 3600.0
    
    @property
    def halflife_days(self) -> float:
        """Halflife in days."""
        return self.halflife / 86400.0
    
    @property
    def halflife_years(self) -> float:
        """Halflife in years."""
        return self.halflife / 31557600.0  # Julian year
    
    @property
    def decay_constant(self) -> float:
        """Decay constant lambda = ln(2)/halflife (1/s)."""
        return np.log(2) / self.halflife if self.halflife > 0 else 0.0
    
    def get_lines(self, line_type: str = "gamma") -> List[GammaLine]:
        """Get lines of specified type."""
        mapping = {
            "gamma": self.gamma_lines,
            "x-ray": self.xray_lines,
            "xray": self.xray_lines,
            "beta": self.beta_lines,
            "alpha": self.alpha_lines,
            "ec": self.ec_lines,
            "electron": self.electron_lines,
        }
        return mapping.get(line_type.lower(), [])
    
    def energies(self, line_type: str = "gamma") -> np.ndarray:
        """Get energy array for specified line type (eV)."""
        lines = self.get_lines(line_type)
        return np.array([line.energy for line in lines])
    
    def energies_keV(self, line_type: str = "gamma") -> np.ndarray:
        """Get energy array for specified line type (keV)."""
        return self.energies(line_type) / 1000.0
    
    def intensities(self, line_type: str = "gamma") -> np.ndarray:
        """Get intensity array for specified line type."""
        lines = self.get_lines(line_type)
        return np.array([line.intensity * line.norm for line in lines])
    
    def strongest_gamma_lines(self, n: int = 5, min_intensity: float = 0.0) -> List[GammaLine]:
        """Get the n strongest gamma lines above min_intensity."""
        lines = [l for l in self.gamma_lines if l.intensity * l.norm >= min_intensity]
        return sorted(lines, key=lambda x: x.intensity * x.norm, reverse=True)[:n]


class GammaDatabase:
    """
    Gamma line database for isotope identification.
    
    Loads decay data from JSON files (actigamma format) and provides
    query methods for spectroscopy applications.
    
    Example
    -------
    >>> db = GammaDatabase()
    >>> co60 = db.get("Co60")
    >>> print(co60.energies_keV("gamma"))
    [1173.23, 1332.49]
    >>> print(db.find_matches(1332.5, tolerance_keV=1.0))
    [('Co60', GammaLine(1332.49 keV, I=0.9998))]
    """
    
    def __init__(self, datafile: Optional[str] = None):
        """
        Initialize the gamma database.
        
        Parameters
        ----------
        datafile : str, optional
            Path to JSON data file. If not provided, searches for
            actigamma data or FluxForge bundled data.
        """
        self._raw: Dict[str, dict] = {}
        self._nuclides: Dict[str, DecayData] = {}
        
        # Try to find data file
        if datafile is None:
            # Use the search function
            datafile = find_actigamma_data()
            
            if datafile is None and os.path.exists(FLUXFORGE_GAMMA_DATA):
                datafile = FLUXFORGE_GAMMA_DATA
        
        if datafile and os.path.exists(datafile):
            self._load_json(datafile)
    
    def _load_json(self, filepath: str) -> None:
        """Load data from actigamma-format JSON file."""
        with open(filepath, 'r') as f:
            self._raw = json.load(f)
        
        # Parse into DecayData objects
        for name, data in self._raw.items():
            decay = self._parse_nuclide(name, data)
            self._nuclides[name] = decay
    
    def _parse_nuclide(self, name: str, data: dict) -> DecayData:
        """Parse raw JSON data into DecayData object."""
        decay = DecayData(
            nuclide=name,
            zai=data.get("zai", 0),
            halflife=data.get("halflife", 0.0)
        )
        
        # Parse each decay type
        type_mapping = {
            "gamma": "gamma_lines",
            "x-ray": "xray_lines",
            "beta": "beta_lines",
            "alpha": "alpha_lines",
            "ec": "ec_lines",
            "electron": "electron_lines",
        }
        
        for json_key, attr_name in type_mapping.items():
            if json_key in data and "lines" in data[json_key]:
                lines_data = data[json_key]["lines"]
                lines = self._parse_lines(lines_data)
                setattr(decay, attr_name, lines)
        
        return decay
    
    def _parse_lines(self, lines_data: dict) -> List[GammaLine]:
        """Parse lines data from JSON."""
        lines = []
        n_lines = len(lines_data.get("energies", []))
        
        for i in range(n_lines):
            line = GammaLine(
                energy=lines_data["energies"][i],
                energy_unc=lines_data.get("energies_unc", [0.0]*n_lines)[i],
                intensity=lines_data["intensities"][i],
                intensity_unc=lines_data.get("intensities_unc", [0.0]*n_lines)[i],
                norm=lines_data.get("norms", [1.0]*n_lines)[i],
                norm_unc=lines_data.get("norms_unc", [0.0]*n_lines)[i]
            )
            lines.append(line)
        
        return lines
    
    @property
    def nuclides(self) -> List[str]:
        """List of all nuclide names in database."""
        return list(self._nuclides.keys())
    
    @property
    def n_nuclides(self) -> int:
        """Number of nuclides in database."""
        return len(self._nuclides)
    
    def __contains__(self, nuclide: str) -> bool:
        """Check if nuclide is in database."""
        return nuclide in self._nuclides
    
    def __getitem__(self, nuclide: str) -> DecayData:
        """Get decay data for nuclide."""
        return self._nuclides[nuclide]
    
    def get(self, nuclide: str, default: Optional[DecayData] = None) -> Optional[DecayData]:
        """Get decay data for nuclide, or default if not found."""
        return self._nuclides.get(nuclide, default)
    
    def get_by_zai(self, zai: int) -> Optional[DecayData]:
        """Get decay data by ZAI number."""
        for decay in self._nuclides.values():
            if decay.zai == zai:
                return decay
        return None
    
    def nuclides_with_gamma(self) -> List[str]:
        """List nuclides that have gamma lines."""
        return [name for name, decay in self._nuclides.items() 
                if len(decay.gamma_lines) > 0]
    
    def find_matches(
        self,
        energy_keV: float,
        tolerance_keV: float = 1.0,
        line_type: str = "gamma",
        min_intensity: float = 0.0
    ) -> List[Tuple[str, GammaLine]]:
        """
        Find nuclides with lines matching the given energy.
        
        Parameters
        ----------
        energy_keV : float
            Target energy in keV
        tolerance_keV : float
            Energy tolerance window (±)
        line_type : str
            Type of lines to search ("gamma", "x-ray", etc.)
        min_intensity : float
            Minimum intensity threshold
        
        Returns
        -------
        List[Tuple[str, GammaLine]]
            List of (nuclide_name, matching_line) tuples,
            sorted by intensity (strongest first)
        """
        energy_eV = energy_keV * 1000.0
        tolerance_eV = tolerance_keV * 1000.0
        
        matches = []
        for name, decay in self._nuclides.items():
            lines = decay.get_lines(line_type)
            for line in lines:
                if abs(line.energy - energy_eV) <= tolerance_eV:
                    eff_intensity = line.intensity * line.norm
                    if eff_intensity >= min_intensity:
                        matches.append((name, line))
        
        # Sort by intensity (strongest first)
        matches.sort(key=lambda x: x[1].intensity * x[1].norm, reverse=True)
        return matches
    
    def identify_spectrum(
        self,
        peak_energies_keV: List[float],
        tolerance_keV: float = 1.0,
        min_matches: int = 2
    ) -> Dict[str, List[Tuple[float, GammaLine]]]:
        """
        Identify nuclides from a list of peak energies.
        
        Parameters
        ----------
        peak_energies_keV : list
            List of detected peak energies in keV
        tolerance_keV : float
            Energy matching tolerance
        min_matches : int
            Minimum number of matching peaks required
        
        Returns
        -------
        Dict[str, List[Tuple[float, GammaLine]]]
            Dictionary mapping nuclide names to list of
            (peak_energy, matched_line) tuples
        """
        candidates: Dict[str, List[Tuple[float, GammaLine]]] = {}
        
        for energy in peak_energies_keV:
            matches = self.find_matches(energy, tolerance_keV)
            for name, line in matches:
                if name not in candidates:
                    candidates[name] = []
                candidates[name].append((energy, line))
        
        # Filter by minimum matches
        return {name: matches for name, matches in candidates.items() 
                if len(matches) >= min_matches}
    
    def get_calibration_sources(self) -> Dict[str, DecayData]:
        """
        Get common calibration sources.
        
        Returns
        -------
        Dict[str, DecayData]
            Dictionary of common calibration source data
        """
        calibration = ["Co60", "Cs137", "Eu152", "Am241", "Ba133", "Na22", "Mn54"]
        return {name: self._nuclides[name] 
                for name in calibration if name in self._nuclides}
    
    def query(
        self,
        halflife_range: Optional[Tuple[float, float]] = None,
        min_gamma_lines: int = 0,
        energy_range_keV: Optional[Tuple[float, float]] = None
    ) -> List[DecayData]:
        """
        Query database with filters.
        
        Parameters
        ----------
        halflife_range : tuple, optional
            (min, max) halflife in seconds
        min_gamma_lines : int
            Minimum number of gamma lines required
        energy_range_keV : tuple, optional
            (min, max) energy range - nuclide must have at least
            one gamma line in this range
        
        Returns
        -------
        List[DecayData]
            Matching nuclides
        """
        results = []
        
        for decay in self._nuclides.values():
            # Halflife filter
            if halflife_range:
                if not (halflife_range[0] <= decay.halflife <= halflife_range[1]):
                    continue
            
            # Gamma lines count filter
            if len(decay.gamma_lines) < min_gamma_lines:
                continue
            
            # Energy range filter
            if energy_range_keV:
                energies = decay.energies_keV("gamma")
                if len(energies) == 0:
                    continue
                if not any(energy_range_keV[0] <= e <= energy_range_keV[1] 
                          for e in energies):
                    continue
            
            results.append(decay)
        
        return results
    
    def summary(self) -> str:
        """Get database summary string."""
        n_gamma = sum(1 for d in self._nuclides.values() if d.gamma_lines)
        n_xray = sum(1 for d in self._nuclides.values() if d.xray_lines)
        total_gamma = sum(len(d.gamma_lines) for d in self._nuclides.values())
        
        return (f"GammaDatabase: {self.n_nuclides} nuclides, "
                f"{n_gamma} with gamma lines ({total_gamma} total lines), "
                f"{n_xray} with X-ray lines")


# Global default database instance
_default_database: Optional[GammaDatabase] = None


def get_database() -> GammaDatabase:
    """Get or create the default gamma database instance."""
    global _default_database
    if _default_database is None:
        _default_database = GammaDatabase()
    return _default_database


def find_gamma_matches(energy_keV: float, tolerance_keV: float = 1.0) -> List[Tuple[str, GammaLine]]:
    """
    Convenience function to find gamma line matches.
    
    Uses the default database instance.
    """
    return get_database().find_matches(energy_keV, tolerance_keV)


def identify_nuclides(peak_energies_keV: List[float], tolerance_keV: float = 1.0) -> Dict[str, list]:
    """
    Convenience function to identify nuclides from peak list.
    
    Uses the default database instance.
    """
    return get_database().identify_spectrum(peak_energies_keV, tolerance_keV)


# Helper functions for nuclide parsing
def parse_nuclide_name(name: str) -> Tuple[str, int, int]:
    """
    Parse nuclide name like 'Co60' or 'U235m' into (element, mass, isomeric).
    
    Returns
    -------
    Tuple[str, int, int]
        (element_symbol, mass_number, isomeric_state)
    """
    import re
    
    # Match element symbol, mass number, and optional 'm' for isomeric
    match = re.match(r'^([A-Z][a-z]?)(\d+)(m?)$', name)
    if not match:
        raise ValueError(f"Cannot parse nuclide name: {name}")
    
    element = match.group(1)
    mass = int(match.group(2))
    isomeric = 1 if match.group(3) == 'm' else 0
    
    return element, mass, isomeric


def zai_to_name(zai: int) -> str:
    """
    Convert ZAI number to nuclide name.
    
    ZAI = Z*10000 + A*10 + I (isomeric state)
    """
    from .elements import ELEMENT_SYMBOLS
    
    z = zai // 10000
    a = (zai % 10000) // 10
    i = zai % 10
    
    element = ELEMENT_SYMBOLS.get(z, f"Z{z}")
    suffix = "m" if i == 1 else ("m2" if i == 2 else "")
    
    return f"{element}{a}{suffix}"
