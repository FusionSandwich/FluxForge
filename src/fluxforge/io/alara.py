"""
ALARA Input/Output Module

This module provides functions to read and write ALARA format files
for neutron activation analysis workflows.

ALARA (Analytic and Laplacian Adaptive Radioactivity Analysis) is a code
for calculating spatially-dependent decay heat and radioactivity following
neutron irradiation.

Features:
- Read ALARA input files and parse geometry, materials, flux, schedules
- Read ALARA output files and extract activation results
- Write ALARA input files programmatically
- Convert between FluxForge spectrum format and ALARA flux format

References:
- ALARA User's Guide: https://svalinn.github.io/ALARA/
- P.P.H. Wilson, "ALARA: Analytic Laplacian Adaptive Radioactivity Analysis",
  Ph.D. Thesis, University of Wisconsin-Madison, 1999.

Author: FluxForge Development Team
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# Data Classes for ALARA Components
# =============================================================================

@dataclass
class ALARAMixture:
    """ALARA material mixture definition."""
    name: str
    constituents: List[Dict[str, Any]] = field(default_factory=list)
    # Each constituent: {'type': 'element'|'material', 'name': str, 
    #                    'density': float, 'fraction': float}
    
    def add_element(self, name: str, density: float = 1.0, fraction: float = 1.0):
        """Add an element to the mixture."""
        self.constituents.append({
            'type': 'element',
            'name': name,
            'density': density,
            'fraction': fraction,
        })
    
    def add_material(self, name: str, density: float = 1.0, fraction: float = 1.0):
        """Add a pre-defined material to the mixture."""
        self.constituents.append({
            'type': 'material',
            'name': name,
            'density': density,
            'fraction': fraction,
        })
    
    def to_alara(self) -> str:
        """Convert to ALARA input format."""
        lines = [f"mixture {self.name}"]
        for c in self.constituents:
            ctype = c['type']
            cname = c['name']
            dens = c['density']
            frac = c['fraction']
            lines.append(f"    {ctype}  {cname}   {dens}    {frac}")
        lines.append("end")
        return "\n".join(lines)


@dataclass
class ALARASchedule:
    """ALARA irradiation schedule definition."""
    name: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_item(self, duration: str, flux: str, history: str, delay: str = "0 s"):
        """Add an irradiation item to the schedule."""
        self.items.append({
            'duration': duration,
            'flux': flux,
            'history': history,
            'delay': delay,
        })
    
    def to_alara(self) -> str:
        """Convert to ALARA input format."""
        lines = [f"schedule {self.name}"]
        for item in self.items:
            lines.append(f"    {item['duration']}  {item['flux']}  "
                        f"{item['history']}  {item['delay']}")
        lines.append("end")
        return "\n".join(lines)


@dataclass
class ALARAPulseHistory:
    """ALARA pulse history definition."""
    name: str
    pulses: int = 1
    delay: str = "0 s"
    
    def to_alara(self) -> str:
        """Convert to ALARA input format."""
        return f"pulsehistory {self.name}\n    {self.pulses}   {self.delay}\nend"


@dataclass
class ALARAFlux:
    """ALARA flux definition."""
    name: str
    file_path: str
    normalization: float = 1.0
    skip: int = 0
    fmt: str = "default"
    
    def to_alara(self) -> str:
        """Convert to ALARA input format."""
        return f"flux  {self.name}  {self.file_path}  {self.normalization}  {self.skip}  {self.fmt}"


@dataclass
class ALARAZoneResult:
    """Results for a single zone from ALARA output."""
    zone_id: int
    zone_name: str = ""
    mixture_name: str = ""
    mass: float = 0.0
    volume: float = 0.0
    
    # Isotope inventories at each cooling time
    number_density: Dict[str, Dict[str, float]] = field(default_factory=dict)
    specific_activity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decay_heat: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ALARAOutput:
    """Parsed ALARA output file."""
    input_file: str = ""
    version: str = ""
    cooling_times: List[str] = field(default_factory=list)
    zones: List[ALARAZoneResult] = field(default_factory=list)
    
    # Totals
    total_activity: Dict[str, float] = field(default_factory=dict)
    total_heat: Dict[str, float] = field(default_factory=dict)


# Legacy compatibility - keep ALARASettings for backward compatibility
@dataclass
class ALARASettings:
    """Configuration for ALARA simulation (legacy compatibility)."""
    material_name: str
    density: float  # g/cm3
    flux_file: str
    material_lib: str
    element_lib: str
    data_library: str
    cooling_times: List[str]
    irradiation_schedule: List[Tuple[str, str]] = field(default_factory=lambda: [("1h", "flux_1")])
    output_units: str = "Bq cm3"
    truncation: float = 1e-12
    impurity: Tuple[float, float] = (1e-8, 1e-10)
    geometry: str = "rectangular"
    dimension: str = "x"
    dimension_args: List[str] = field(default_factory=lambda: ["0.0", "1 1.0"])


class ALARAInputGenerator:
    """Generates ALARA input files."""
    
    def __init__(self, settings: ALARASettings):
        self.settings = settings
        
    def write(self, output_path: str) -> None:
        """Write ALARA input file."""
        s = self.settings
        mat_name = s.material_name.upper().replace(' ', '_')
        zone_name = s.material_name.lower().replace(' ', '_')
        
        with open(output_path, 'w') as f:
            f.write(f"# ALARA Input generated by FluxForge\n")
            f.write(f"# Material: {s.material_name}\n\n")
            
            # Geometry
            f.write(f"geometry {s.geometry}\n\n")
            f.write(f"dimension {s.dimension}\n")
            for arg in s.dimension_args:
                f.write(f"    {arg}\n")
            f.write("end\n\n")
            
            # Material loading
            f.write("mat_loading\n")
            f.write(f"    zone_1 mix_{zone_name}\n")
            f.write("end\n\n")
            
            # Libraries
            f.write(f"material_lib {s.material_lib}\n")
            f.write(f"element_lib {s.element_lib}\n\n")
            
            # Mixture
            f.write(f"mixture mix_{zone_name}\n")
            f.write(f"    material {mat_name} 1.0 {s.density}\n")
            f.write("end\n\n")
            
            # Flux
            f.write(f"flux flux_1 {s.flux_file} 1.0 0 default\n\n")
            
            # Schedule
            f.write("schedule irradiation\n")
            for duration, flux_id in s.irradiation_schedule:
                f.write(f"    {duration} {flux_id} pulse_once 0 s\n")
            f.write("end\n\n")
            
            # Pulse history
            f.write("pulsehistory pulse_once\n")
            f.write("    1 0 s\n")
            f.write("end\n\n")
            
            # Cooling times
            f.write("cooling\n")
            for ct in s.cooling_times:
                f.write(f"    {ct}\n")
            f.write("end\n\n")
            
            # Data library
            f.write(f"data_library alaralib {s.data_library}\n\n")
            
            # Dump file
            dump_file = os.path.basename(output_path).replace('.inp', '.dump')
            f.write(f"dump_file {dump_file}\n\n")
            
            # Output options
            f.write("output zone\n")
            f.write(f"    units {s.output_units}\n")
            f.write("    constituent\n")
            f.write("    number_density\n")
            f.write("    specific_activity\n")
            f.write("    total_heat\n")
            f.write("    gamma_heat\n")
            f.write("end\n\n")
            
            # Truncation and impurity
            f.write(f"truncation {s.truncation}\n")
            f.write(f"impurity {s.impurity[0]} {s.impurity[1]}\n")


def parse_alara_output(output_text: str) -> Dict[str, Any]:
    """
    Parse ALARA output text.
    
    Returns dictionary with results keyed by cooling time and isotope.
    Structure:
    {
        'cooling_times': ['0s', '1h', ...],
        'isotopes': {
            'Fe55': {
                'activity': [val_t0, val_t1, ...],
                'heat': [val_t0, val_t1, ...]
            },
            ...
        },
        'totals': {
            'activity': [...],
            'heat': [...]
        }
    }
    """
    results = {
        'cooling_times': [],
        'isotopes': {},
        'totals': {'activity': [], 'heat': []}
    }
    
    # This is a simplified parser - a full parser would be more complex
    # and handle the hierarchical structure of ALARA output better.
    # For now, we'll focus on extracting total activity and heat.
    
    lines = output_text.split('\n')
    current_cooling_time = None
    
    # Regex patterns
    cooling_pattern = re.compile(r'Cooling Time:\s+(.+)')
    total_act_pattern = re.compile(r'Total Specific Activity:\s+([0-9.eE+-]+)')
    total_heat_pattern = re.compile(r'Total Decay Heat:\s+([0-9.eE+-]+)')
    
    for line in lines:
        line = line.strip()
        
        m_cool = cooling_pattern.search(line)
        if m_cool:
            current_cooling_time = m_cool.group(1)
            if current_cooling_time not in results['cooling_times']:
                results['cooling_times'].append(current_cooling_time)
            continue
            
        m_act = total_act_pattern.search(line)
        if m_act:
            results['totals']['activity'].append(float(m_act.group(1)))
            continue
            
        m_heat = total_heat_pattern.search(line)
        if m_heat:
            results['totals']['heat'].append(float(m_heat.group(1)))
            continue
            
    return results


# =============================================================================
# ALARA Input File Reading and Writing
# =============================================================================

def read_alara_input(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse an ALARA input file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the ALARA input file.
    
    Returns
    -------
    dict
        Parsed ALARA input structure with keys:
        - geometry_type: str
        - dimensions: dict
        - volumes: list
        - mat_loading: dict
        - mixtures: dict
        - fluxes: dict
        - schedules: dict
        - cooling_times: list
        - truncation: float
    
    Examples
    --------
    >>> alara_input = read_alara_input("sample.alara")
    >>> print(alara_input['geometry_type'])
    'point'
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ALARA input file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    result = {
        'geometry_type': 'point',
        'dimensions': {},
        'volumes': [],
        'mat_loading': {},
        'mixtures': {},
        'fluxes': {},
        'schedules': {},
        'pulse_histories': {},
        'cooling_times': [],
        'material_lib': '',
        'element_lib': '',
        'data_library': ('', ''),
        'truncation': 1e-7,
        'dump_file': '',
    }
    
    # Remove comments
    lines = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('##'):
            continue
        if '#' in line:
            line = line[:line.index('#')]
        lines.append(line)
    
    content = '\n'.join(lines)
    
    # Parse geometry
    geo_match = re.search(r'geometry\s+(\w+)', content)
    if geo_match:
        result['geometry_type'] = geo_match.group(1)
    
    # Parse volumes
    vol_pattern = r'volumes(.*?)end'
    vol_match = re.search(vol_pattern, content, re.DOTALL)
    if vol_match:
        vol_content = vol_match.group(1)
        for line in vol_content.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    vol = float(parts[0])
                    zone = parts[1]
                    result['volumes'].append((vol, zone))
                except ValueError:
                    continue
    
    # Parse mat_loading
    mat_pattern = r'mat_loading(.*?)end'
    mat_match = re.search(mat_pattern, content, re.DOTALL)
    if mat_match:
        mat_content = mat_match.group(1)
        for line in mat_content.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2:
                result['mat_loading'][parts[0]] = parts[1]
    
    # Parse library paths
    matlib_match = re.search(r'material_lib\s+(\S+)', content)
    if matlib_match:
        result['material_lib'] = matlib_match.group(1)
    
    elelib_match = re.search(r'element_lib\s+(\S+)', content)
    if elelib_match:
        result['element_lib'] = elelib_match.group(1)
    
    datalib_match = re.search(r'data_library\s+(\w+)\s+(\S+)', content)
    if datalib_match:
        result['data_library'] = (datalib_match.group(1), datalib_match.group(2))
    
    # Parse mixtures
    mix_pattern = r'mixture\s+(\w+)(.*?)end'
    for match in re.finditer(mix_pattern, content, re.DOTALL):
        mix_name = match.group(1)
        mix_content = match.group(2)
        mixture = ALARAMixture(name=mix_name)
        
        for line in mix_content.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 4:
                ctype = parts[0]
                cname = parts[1]
                try:
                    density = float(parts[2])
                    fraction = float(parts[3])
                    mixture.constituents.append({
                        'type': ctype,
                        'name': cname,
                        'density': density,
                        'fraction': fraction,
                    })
                except ValueError:
                    continue
        
        result['mixtures'][mix_name] = mixture
    
    # Parse fluxes
    flux_pattern = r'flux\s+(\w+)\s+(\S+)\s+([\d.eE+-]+)\s+(\d+)\s+(\w+)'
    for match in re.finditer(flux_pattern, content):
        flux = ALARAFlux(
            name=match.group(1),
            file_path=match.group(2),
            normalization=float(match.group(3)),
            skip=int(match.group(4)),
            fmt=match.group(5),
        )
        result['fluxes'][flux.name] = flux
    
    # Parse cooling times
    cool_pattern = r'cooling(.*?)end'
    cool_match = re.search(cool_pattern, content, re.DOTALL)
    if cool_match:
        cool_content = cool_match.group(1)
        for line in cool_content.strip().split('\n'):
            stripped = line.strip()
            if stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    result['cooling_times'].append(f"{parts[0]} {parts[1]}")
    
    # Parse truncation
    trunc_match = re.search(r'truncation\s+([\d.eE+-]+)', content)
    if trunc_match:
        result['truncation'] = float(trunc_match.group(1))
    
    # Parse dump file
    dump_match = re.search(r'dump_file\s+(\S+)', content)
    if dump_match:
        result['dump_file'] = dump_match.group(1)
    
    return result


def read_alara_output(file_path: Union[str, Path]) -> ALARAOutput:
    """
    Read and parse an ALARA output file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the ALARA output file.
    
    Returns
    -------
    ALARAOutput
        Parsed ALARA output structure with zone results.
    
    Examples
    --------
    >>> output = read_alara_output("sample.out")
    >>> print(output.cooling_times)
    ['shutdown', '0 s', '3 d', '7 d']
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ALARA output file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    output = ALARAOutput()
    
    # Parse version
    version_match = re.search(r'ALARA\s+([\d.]+)', content)
    if version_match:
        output.version = version_match.group(1)
    
    # Find cooling times from header line
    header_pattern = r'isotope\s+(shutdown.*?)$'
    header_match = re.search(header_pattern, content, re.MULTILINE)
    if header_match:
        header_text = header_match.group(1)
        parts = re.split(r'\s{2,}', header_text.strip())
        output.cooling_times = parts
    
    # Parse zone results
    zone_pattern = r'Zone #(\d+):\s+(\w+)\s*\n\s*Mass:\s+([\d.eE+-]+)\s*\n\s*Containing mixture:\s+(\w+)'
    
    for match in re.finditer(zone_pattern, content):
        zone = ALARAZoneResult(
            zone_id=int(match.group(1)),
            zone_name=match.group(2),
            mass=float(match.group(3)),
            mixture_name=match.group(4),
        )
        output.zones.append(zone)
    
    return output


def write_alara_flux(
    flux: np.ndarray,
    energy_edges: np.ndarray,
    file_path: Union[str, Path],
    n_intervals: int = 1,
    normalize: bool = True,
) -> None:
    """
    Write a neutron flux spectrum to ALARA format.
    
    Parameters
    ----------
    flux : np.ndarray
        Flux values for each energy group [n/cm^2/s].
    energy_edges : np.ndarray
        Energy bin edges [eV].
    file_path : str or Path
        Output file path.
    n_intervals : int, optional
        Number of spatial intervals (for mesh tallies). Default 1.
    normalize : bool, optional
        Whether to normalize the flux. Default True.
    
    Notes
    -----
    ALARA expects flux in n/cm^2/s with energy groups from highest to lowest.
    
    Examples
    --------
    >>> flux = np.array([1e10, 2e10, 5e10])
    >>> energy = np.array([0, 1e6, 10e6, 20e6])
    >>> write_alara_flux(flux, energy, "flux.txt")
    """
    file_path = Path(file_path)
    
    n_groups = len(flux)
    
    # ALARA wants groups from high to low energy
    if energy_edges[0] < energy_edges[-1]:
        flux_reversed = flux[::-1]
    else:
        flux_reversed = flux
    
    # Normalize if requested
    if normalize:
        total = np.sum(flux_reversed)
        if total > 0:
            flux_reversed = flux_reversed / total
    
    with open(file_path, 'w') as f:
        f.write(f"{n_groups}\n")
        
        for interval in range(n_intervals):
            for val in flux_reversed:
                f.write(f"{val:.6e}\n")
            
            if interval < n_intervals - 1:
                f.write("\n")


def read_alara_flux(file_path: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """
    Read an ALARA flux file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the ALARA flux file.
    
    Returns
    -------
    n_groups : int
        Number of energy groups.
    flux : np.ndarray
        Flux values.
    
    Examples
    --------
    >>> n_groups, flux = read_alara_flux("flux.txt")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ALARA flux file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    n_groups = None
    flux_values = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if n_groups is None:
            n_groups = int(stripped)
        else:
            try:
                flux_values.append(float(stripped))
            except ValueError:
                continue
    
    return n_groups, np.array(flux_values)


def fluxforge_spectrum_to_alara(
    flux: np.ndarray,
    energy_edges: np.ndarray,
    output_path: Union[str, Path],
    n_intervals: int = 1,
) -> str:
    """
    Convert FluxForge spectrum to ALARA flux format.
    
    Parameters
    ----------
    flux : np.ndarray
        Flux spectrum from FluxForge [n/cm^2/s].
    energy_edges : np.ndarray
        Energy bin edges in eV.
    output_path : str or Path
        Output file path.
    n_intervals : int, optional
        Number of spatial intervals. Default 1.
    
    Returns
    -------
    str
        Path to the written flux file.
    """
    output_path = Path(output_path)
    
    write_alara_flux(flux, energy_edges, output_path, n_intervals, normalize=False)
    
    return str(output_path)


def create_alara_activation_input(
    mixture_name: str,
    elements: Dict[str, float],
    flux_file: str,
    irradiation_time: str,
    cooling_times: List[str],
    output_path: Union[str, Path],
    data_library: str = "",
    material_lib: str = "",
    element_lib: str = "",
    flux_norm: float = 1.0,
    density: float = 7.86,
) -> str:
    """
    Create a complete ALARA input for activation analysis.
    
    Parameters
    ----------
    mixture_name : str
        Name for the material mixture.
    elements : dict
        Element composition as {element: mass_fraction}.
    flux_file : str
        Path to the flux file.
    irradiation_time : str
        Irradiation duration (e.g., "2 h", "30 d").
    cooling_times : list
        List of cooling time strings (e.g., ["0 s", "3 d", "7 d"]).
    output_path : str or Path
        Output path for the ALARA input file.
    data_library : str, optional
        Path to the ALARA data library.
    material_lib : str, optional
        Path to the material library.
    element_lib : str, optional
        Path to the element library.
    flux_norm : float, optional
        Flux normalization factor. Default 1.0.
    density : float, optional
        Material density in g/cc. Default 7.86 (steel).
    
    Returns
    -------
    str
        Path to the created ALARA input file.
    
    Examples
    --------
    >>> elements = {"fe": 0.90, "cr": 0.09, "c": 0.01}
    >>> create_alara_activation_input(
    ...     mixture_name="steel",
    ...     elements=elements,
    ...     flux_file="flux.txt",
    ...     irradiation_time="2 h",
    ...     cooling_times=["0 s", "3 d", "7 d"],
    ...     output_path="activation.alara",
    ... )
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        # Header
        f.write(f"## ALARA Input File - Generated by FluxForge\n")
        f.write(f"## Generated: {datetime.now().isoformat()}\n\n")
        
        # Geometry
        f.write("geometry point\n\n")
        
        # Volumes
        f.write("volumes\n")
        f.write("    1.0    sample_zone\n")
        f.write("end\n\n")
        
        # Material loading
        f.write("mat_loading\n")
        f.write(f"    sample_zone    {mixture_name}\n")
        f.write("end\n\n")
        
        # Library paths
        if material_lib:
            f.write(f"material_lib  {material_lib}\n")
        if element_lib:
            f.write(f"element_lib   {element_lib}\n")
        f.write("\n")
        
        # Mixture
        f.write(f"mixture {mixture_name}\n")
        for elem, frac in elements.items():
            f.write(f"    element  {elem}   {density}    {frac}\n")
        f.write("end\n\n")
        
        # Flux
        f.write(f"flux  irrad_flux  {flux_file}  {flux_norm}  0  default\n\n")
        
        # Schedule
        f.write("schedule irradiation\n")
        f.write(f"    {irradiation_time}  irrad_flux  steady_state  0 s\n")
        f.write("end\n\n")
        
        # Pulse history
        f.write("pulsehistory steady_state\n")
        f.write("    1   0 s\n")
        f.write("end\n\n")
        
        # Dump file
        dump_file = str(output_path.stem) + ".dump"
        f.write(f"dump_file  {dump_file}\n\n")
        
        # Data library
        if data_library:
            f.write(f"data_library alaralib {data_library}\n\n")
        
        # Cooling times
        f.write("cooling\n")
        for ct in cooling_times:
            f.write(f"    {ct}\n")
        f.write("end\n\n")
        
        # Output
        f.write("output zone\n")
        f.write("    units Bq kg\n")
        f.write("    number_density\n")
        f.write("    specific_activity\n")
        f.write("    total_heat\n")
        f.write("end\n\n")
        
        # Truncation
        f.write("truncation 1e-7\n")
    
    return str(output_path)


# =============================================================================
# Voxel-Averaged Statistics (parity with alara_adf_utils)
# =============================================================================

DEFAULT_TIME_GROUPS = {
    '300s': 300,
    '2h': 7200,
    '24h': 86400,
    '4d': 345600,
    '15d': 1296000,
}


def format_time_label(
    seconds: float, 
    preferred_groups: Optional[Dict[str, float]] = None,
    max_group_factor: float = 3.0,
) -> Optional[str]:
    """
    Convert a numeric time in seconds into a compact, human-readable label.
    
    Parameters
    ----------
    seconds : float
        Time in seconds.
    preferred_groups : dict, optional
        Mapping of label to seconds for preferred groupings.
    max_group_factor : float, optional
        Maximum ratio for snapping to preferred groups.
    
    Returns
    -------
    str or None
        Formatted time label like '300s', '24h', '4d', etc.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None

    if seconds <= 0 or seconds < 1.0:
        return 'shutdown'

    if preferred_groups is None:
        preferred_groups = DEFAULT_TIME_GROUPS

    if preferred_groups:
        closest_label = None
        min_ratio = None
        for label, group_seconds in preferred_groups.items():
            if group_seconds <= 0:
                continue
            ratio = abs(np.log10(seconds / group_seconds))
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                closest_label = label

        if min_ratio is not None and min_ratio < np.log10(max_group_factor):
            return closest_label

    for unit, factor in [('y', 365 * 86400), ('d', 86400), ('h', 3600), ('m', 60), ('s', 1)]:
        value = seconds / factor
        if value >= 1:
            if abs(value - round(value)) < 1e-6:
                value = int(round(value))
                return f"{value}{unit}"
            return f"{value:.3g}{unit}"

    return f"{seconds:.3g}s"


def build_zone_stats(
    adf: Any,
    variable: str,
    block: str = 'Zone',
    preferred_groups: Optional[Dict[str, float]] = None,
) -> 'pd.DataFrame':
    """
    Aggregate voxel-wise ALARADFrame data into mean/SEM/relative uncertainty.
    
    Parameters
    ----------
    adf : ALARADFrame
        DataFrame with ALARA results indexed by zone/voxel.
    variable : str
        Variable to extract: 'Specific Activity', 'Total Decay Heat', etc.
    block : str, optional
        Block type filter. Default 'Zone'.
    preferred_groups : dict, optional
        Mapping of label to seconds for time grouping.
    
    Returns
    -------
    pd.DataFrame
        Statistics with columns: run_lbl, nuclide, time, mean, sem, rel_unc.
    """
    import pandas as pd
    
    if adf is None or (hasattr(adf, 'empty') and adf.empty):
        return pd.DataFrame()

    filtered = adf
    
    # Apply block filter if available
    if hasattr(adf, 'BLOCK_ENUM'):
        block_id = adf.BLOCK_ENUM.get(block)
        if block_id is not None and 'block' in filtered.columns:
            filtered = filtered[filtered['block'] == block_id]

    # Apply variable filter if available
    if hasattr(adf, 'VARIABLE_ENUM'):
        variable_id = adf.VARIABLE_ENUM.get(variable)
        if variable_id is not None and 'variable' in filtered.columns:
            filtered = filtered[filtered['variable'] == variable_id]

    if hasattr(filtered, 'empty') and filtered.empty:
        return pd.DataFrame()

    # Aggregate statistics
    stats = (
        filtered
        .groupby(['run_lbl', 'nuclide', 'time'], as_index=False)['value']
        .agg(mean='mean', std='std', count='count')
    )

    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['sem'] = stats['sem'].fillna(0.0)
    stats['rel_unc'] = np.where(stats['mean'] > 0, stats['sem'] / stats['mean'], 0.0)
    stats = stats.rename(columns={'count': 'n_voxels'})

    if preferred_groups is None:
        preferred_groups = DEFAULT_TIME_GROUPS

    stats['time_label'] = stats['time'].apply(
        lambda t: format_time_label(t, preferred_groups=preferred_groups)
    )

    return stats


def build_isotope_table(stats: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Pivot mean/SEM/rel_unc stats into a wide table with mean_* columns.
    
    Parameters
    ----------
    stats : pd.DataFrame
        Output from build_zone_stats().
    
    Returns
    -------
    pd.DataFrame
        Wide table with columns: isotope, mean_300s, sem_300s, rel_unc_300s, ...
    """
    import pandas as pd
    
    if stats is None or stats.empty:
        return pd.DataFrame()

    # Get ordered time labels
    time_order = []
    if not stats.empty:
        ordered = (
            stats[['time', 'time_label']]
            .drop_duplicates()
            .sort_values('time')
        )
        for label in ordered['time_label']:
            if label and label not in time_order:
                time_order.append(label)
    
    frames = []
    for col in ['mean', 'sem', 'rel_unc']:
        pivot = stats.pivot_table(
            index='nuclide',
            columns='time_label',
            values=col,
            aggfunc='first'
        )
        pivot = pivot.reindex(columns=time_order)
        pivot.columns = [f"{col}_{label}" for label in pivot.columns]
        frames.append(pivot)

    table = pd.concat(frames, axis=1).reset_index()
    return table.rename(columns={'nuclide': 'isotope'})
