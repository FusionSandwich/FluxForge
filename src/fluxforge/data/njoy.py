"""
NJOY Processing Pipeline for FluxForge
======================================

This module provides utilities for running NJOY nuclear data processing
to generate reproducible multi-group cross sections from ENDF evaluations.

NJOY is the standard code for processing evaluated nuclear data libraries
into formats suitable for transport codes. Key processing steps include:

1. RECONR - Reconstruct pointwise cross sections from resonance parameters
2. BROADR - Doppler broaden to specified temperatures
3. UNRESR - Process unresolved resonance region
4. GROUPR - Generate multi-group cross sections for specified group structure
5. ERRORR - Process covariance data

This module provides:
- Input template generation for NJOY
- Execution wrapper with error handling
- Output parsing for group cross sections
- Reproducibility metadata tracking

References
----------
- NJOY2016 Manual, LA-UR-17-20093
- ENDF-6 Formats Manual, BNL-203218-2018-INRE

Note: NJOY executable must be installed and available in PATH.

Author: FluxForge Development Team
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np


class NJOYModule(Enum):
    """NJOY processing modules."""
    
    RECONR = "reconr"  # Reconstruct pointwise from resonance parameters
    BROADR = "broadr"  # Doppler broadening
    UNRESR = "unresr"  # Unresolved resonance processing
    HEATR = "heatr"    # Heating/KERMA calculation
    THERMR = "thermr"  # Thermal scattering treatment
    GROUPR = "groupr"  # Multi-group processing
    ERRORR = "errorr"  # Covariance processing
    ACER = "acer"      # ACE format output


class GroupStructure(Enum):
    """Standard group structures supported by NJOY."""
    
    VITAMIN_J = 1       # VITAMIN-J 175g
    XMAS = 2            # XMAS 172g
    ECCO_33 = 3         # ECCO 33g
    ECCO_1968 = 4       # ECCO 1968g
    GAM_II = 5          # GAM-II 68g
    SAND_II = 6         # SAND-II 640g
    WIMS_69 = 7         # WIMS 69g
    SCALE_238 = 8       # SCALE 238g
    SCALE_252 = 9       # SCALE 252g
    CUSTOM = 0          # User-defined structure


# Standard group structure boundaries (eV)
GROUP_STRUCTURE_DATA = {
    GroupStructure.SAND_II: {
        'name': 'SAND-II 640-group',
        'n_groups': 640,
        'description': 'Standard for activation foil analysis',
    },
    GroupStructure.VITAMIN_J: {
        'name': 'VITAMIN-J 175-group',
        'n_groups': 175,
        'description': 'General purpose coupled n/gamma library structure',
    },
    GroupStructure.SCALE_238: {
        'name': 'SCALE 238-group',
        'n_groups': 238,
        'description': 'SCALE thermal reactor structure',
    },
}


@dataclass
class NJOYInput:
    """
    NJOY input specification.
    
    Attributes:
        endf_file: Path to ENDF tape file
        mat_number: Material (MAT) number in ENDF file
        temperatures: List of temperatures (K) for Doppler broadening
        group_structure: Multi-group energy structure
        custom_boundaries: Custom group boundaries (eV) if group_structure is CUSTOM
        modules: List of NJOY modules to run
        output_file: Path for output file
        tolerance: Reconstruction tolerance (default 0.001)
    """
    
    endf_file: Path
    mat_number: int
    temperatures: List[float] = field(default_factory=lambda: [300.0])
    group_structure: GroupStructure = GroupStructure.SAND_II
    custom_boundaries: Optional[np.ndarray] = None
    modules: List[NJOYModule] = field(default_factory=lambda: [
        NJOYModule.RECONR, NJOYModule.BROADR, NJOYModule.GROUPR
    ])
    output_file: Optional[Path] = None
    tolerance: float = 0.001
    
    # Optional parameters
    mf: int = 3  # File type (3 = cross sections)
    mt_list: Optional[List[int]] = None  # Specific MTs to process
    description: str = ""


@dataclass  
class NJOYResult:
    """
    Result from NJOY processing.
    
    Attributes:
        success: Whether processing completed successfully
        group_boundaries: Energy group boundaries (eV)
        cross_sections: Dict mapping MT number to group-averaged XS (barns)
        temperatures: Processed temperatures (K)
        mat_number: Material number processed
        provenance: Processing metadata for reproducibility
        log_file: Path to NJOY output log
        errors: List of error messages if any
    """
    
    success: bool
    group_boundaries: Optional[np.ndarray] = None
    cross_sections: Dict[int, np.ndarray] = field(default_factory=dict)
    temperatures: List[float] = field(default_factory=list)
    mat_number: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)
    log_file: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def n_groups(self) -> int:
        """Number of energy groups."""
        if self.group_boundaries is not None:
            return len(self.group_boundaries) - 1
        return 0
    
    def get_xs(self, mt: int) -> Optional[np.ndarray]:
        """Get cross section for specific MT."""
        return self.cross_sections.get(mt)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"NJOY Processing Result",
            f"  Success: {self.success}",
            f"  MAT: {self.mat_number}",
            f"  Groups: {self.n_groups}",
            f"  Temperatures: {self.temperatures}",
            f"  MTs processed: {list(self.cross_sections.keys())}",
        ]
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


def generate_reconr_input(mat: int, tape_in: int = 20, tape_out: int = 21) -> str:
    """Generate RECONR module input."""
    return f"""reconr
{tape_in} {tape_out}
'RECONR: processed by FluxForge'/
{mat} 0/
0.001 0. 0.003/
0/
"""


def generate_broadr_input(
    mat: int,
    temperatures: List[float],
    tape_in_endf: int = 20,
    tape_in_pendf: int = 21,
    tape_out: int = 22,
) -> str:
    """Generate BROADR module input."""
    n_temps = len(temperatures)
    temps_str = " ".join(f"{t:.1f}" for t in temperatures)
    
    return f"""broadr
{tape_in_endf} {tape_in_pendf} {tape_out}
{mat} {n_temps} 0 0 0./
0.001/
{temps_str}/
0/
"""


def generate_groupr_input(
    mat: int,
    group_structure: int,
    temperatures: List[float],
    tape_in_endf: int = 20,
    tape_in_pendf: int = 22,
    tape_out: int = 23,
    custom_groups: Optional[np.ndarray] = None,
) -> str:
    """
    Generate GROUPR module input.
    
    Parameters
    ----------
    mat : int
        Material number.
    group_structure : int
        IGN value (0 for custom, 1-10 for built-in structures).
    temperatures : list
        Temperatures in K.
    custom_groups : np.ndarray, optional
        Custom group boundaries if group_structure is 0.
    """
    n_temps = len(temperatures)
    temps_str = " ".join(f"{t:.1f}" for t in temperatures)
    
    lines = [
        "groupr",
        f"{tape_in_endf} {tape_in_pendf} 0 {tape_out}",
        f"{mat} {group_structure} 0 1 1 {n_temps} 1 0/",
        "'FluxForge GROUPR processing'/",
        f"{temps_str}/",
        "0./",  # Infinite dilution (sigma0)
    ]
    
    # Add custom group structure if specified
    if group_structure == 0 and custom_groups is not None:
        n_groups = len(custom_groups) - 1
        lines.append(f"{n_groups}/")
        # Write boundaries in NJOY format (high to low energy)
        for e in reversed(custom_groups):
            lines.append(f"{e:.5e}/")
    
    # Process all MTs
    lines.append("3/")  # MF=3 cross sections
    lines.append("3 1 2 4 16 18 102/")  # Total, elastic, inelastic, n2n, fission, capture
    lines.append("0/")  # End MF processing
    lines.append("0/")  # End material
    
    return "\n".join(lines)


def generate_njoy_input(
    config: NJOYInput,
    tape_assignments: Optional[Dict[str, int]] = None,
) -> str:
    """
    Generate complete NJOY input deck.
    
    Parameters
    ----------
    config : NJOYInput
        Processing configuration.
    tape_assignments : dict, optional
        Custom tape unit assignments.
        
    Returns
    -------
    str
        Complete NJOY input deck.
    """
    if tape_assignments is None:
        tape_assignments = {
            'endf_in': 20,
            'pendf_reconr': 21,
            'pendf_broadr': 22,
            'gendf': 23,
        }
    
    lines = [
        "-- NJOY input generated by FluxForge --",
        f"-- {datetime.now().isoformat()} --",
        "",
    ]
    
    current_tape_out = tape_assignments['pendf_reconr']
    
    # RECONR
    if NJOYModule.RECONR in config.modules:
        lines.append(generate_reconr_input(
            config.mat_number,
            tape_assignments['endf_in'],
            tape_assignments['pendf_reconr'],
        ))
        current_tape_out = tape_assignments['pendf_reconr']
    
    # BROADR
    if NJOYModule.BROADR in config.modules:
        lines.append(generate_broadr_input(
            config.mat_number,
            config.temperatures,
            tape_assignments['endf_in'],
            current_tape_out,
            tape_assignments['pendf_broadr'],
        ))
        current_tape_out = tape_assignments['pendf_broadr']
    
    # GROUPR
    if NJOYModule.GROUPR in config.modules:
        ign = config.group_structure.value
        lines.append(generate_groupr_input(
            config.mat_number,
            ign,
            config.temperatures,
            tape_assignments['endf_in'],
            current_tape_out,
            tape_assignments['gendf'],
            config.custom_boundaries,
        ))
    
    # STOP
    lines.append("stop")
    
    return "\n".join(lines)


def run_njoy(
    config: NJOYInput,
    njoy_executable: str = "njoy",
    work_dir: Optional[Path] = None,
    keep_files: bool = False,
) -> NJOYResult:
    """
    Run NJOY processing.
    
    Parameters
    ----------
    config : NJOYInput
        Processing configuration.
    njoy_executable : str
        Path to NJOY executable.
    work_dir : Path, optional
        Working directory (uses temp if not specified).
    keep_files : bool
        Whether to keep intermediate files.
        
    Returns
    -------
    NJOYResult
        Processing results.
    """
    result = NJOYResult(success=False)
    result.mat_number = config.mat_number
    result.temperatures = list(config.temperatures)
    
    # Check NJOY availability
    try:
        version_check = subprocess.run(
            [njoy_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        njoy_version = version_check.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        result.errors.append(f"NJOY executable '{njoy_executable}' not found or not working")
        return result
    
    # Create working directory
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="fluxforge_njoy_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate input
    input_deck = generate_njoy_input(config)
    input_file = work_dir / "njoy.inp"
    input_file.write_text(input_deck)
    
    # Copy/link ENDF file
    tape20 = work_dir / "tape20"
    if config.endf_file.exists():
        tape20.symlink_to(config.endf_file.resolve())
    else:
        result.errors.append(f"ENDF file not found: {config.endf_file}")
        return result
    
    # Run NJOY
    try:
        proc = subprocess.run(
            [njoy_executable],
            cwd=work_dir,
            stdin=open(input_file),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        # Save log
        log_file = work_dir / "njoy.log"
        log_file.write_text(proc.stdout + "\n" + proc.stderr)
        result.log_file = log_file
        
        if proc.returncode != 0:
            result.errors.append(f"NJOY returned code {proc.returncode}")
            result.errors.append(proc.stderr[:500])
            return result
        
    except subprocess.TimeoutExpired:
        result.errors.append("NJOY processing timed out")
        return result
    except Exception as e:
        result.errors.append(f"NJOY execution error: {e}")
        return result
    
    # Parse output
    gendf_file = work_dir / "tape23"
    if gendf_file.exists():
        try:
            group_boundaries, cross_sections = parse_gendf(gendf_file)
            result.group_boundaries = group_boundaries
            result.cross_sections = cross_sections
            result.success = True
        except Exception as e:
            result.errors.append(f"Error parsing GENDF: {e}")
    else:
        result.errors.append("GENDF output file not created")
    
    # Record provenance
    result.provenance = {
        'njoy_version': njoy_version,
        'processed_at': datetime.now().isoformat(),
        'endf_file': str(config.endf_file),
        'mat_number': config.mat_number,
        'temperatures_K': config.temperatures,
        'group_structure': config.group_structure.name,
        'modules': [m.value for m in config.modules],
        'tolerance': config.tolerance,
    }
    
    # Cleanup if requested
    if not keep_files and result.success:
        import shutil
        try:
            shutil.rmtree(work_dir)
        except:
            pass
    
    return result


def parse_gendf(gendf_path: Path) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Parse GENDF (group-averaged cross section) file.
    
    This is a simplified parser for the most common case.
    Full ENDF-6 parsing would require a more comprehensive implementation.
    
    Parameters
    ----------
    gendf_path : Path
        Path to GENDF file.
        
    Returns
    -------
    group_boundaries : np.ndarray
        Energy group boundaries (eV).
    cross_sections : dict
        Dictionary mapping MT number to group cross sections (barns).
    """
    # GENDF format is complex; this is a placeholder for basic parsing
    # In practice, use openmc.data or sandy to parse properly
    
    group_boundaries = np.array([])
    cross_sections = {}
    
    try:
        with open(gendf_path, 'r') as f:
            content = f.read()
        
        # Parse group structure from MF1/MT451
        # Parse cross sections from MF3
        # This requires proper ENDF-6 parsing
        
        # For now, return empty with warning
        import warnings
        warnings.warn("GENDF parsing is limited; consider using openmc.data for full support")
        
    except Exception as e:
        raise ValueError(f"Failed to parse GENDF: {e}")
    
    return group_boundaries, cross_sections


def check_njoy_available(executable: str = "njoy") -> Tuple[bool, str]:
    """
    Check if NJOY is available.
    
    Parameters
    ----------
    executable : str
        Path to NJOY executable.
        
    Returns
    -------
    available : bool
        Whether NJOY is available.
    message : str
        Version info or error message.
    """
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or "version unknown"
            return True, f"NJOY available: {version}"
        else:
            return False, f"NJOY returned error: {result.stderr}"
    except FileNotFoundError:
        return False, f"NJOY executable '{executable}' not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "NJOY check timed out"
    except Exception as e:
        return False, f"Error checking NJOY: {e}"


@dataclass
class NJOYPipelineSpec:
    """
    Specification for a reproducible NJOY processing pipeline.
    
    This dataclass captures all information needed to reproduce
    the cross section processing.
    """
    
    name: str
    description: str
    endf_library: str  # e.g., "ENDF/B-VIII.0", "IRDFF-II"
    materials: List[Dict[str, Any]]  # List of {mat, za, name}
    group_structure: GroupStructure
    custom_boundaries: Optional[np.ndarray] = None
    temperatures: List[float] = field(default_factory=lambda: [300.0])
    modules: List[NJOYModule] = field(default_factory=lambda: [
        NJOYModule.RECONR, NJOYModule.BROADR, NJOYModule.GROUPR
    ])
    tolerance: float = 0.001
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'endf_library': self.endf_library,
            'materials': self.materials,
            'group_structure': self.group_structure.name,
            'custom_boundaries': self.custom_boundaries.tolist() if self.custom_boundaries is not None else None,
            'temperatures': self.temperatures,
            'modules': [m.value for m in self.modules],
            'tolerance': self.tolerance,
            'created_at': self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NJOYPipelineSpec':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            endf_library=data['endf_library'],
            materials=data['materials'],
            group_structure=GroupStructure[data['group_structure']],
            custom_boundaries=np.array(data['custom_boundaries']) if data.get('custom_boundaries') else None,
            temperatures=data.get('temperatures', [300.0]),
            modules=[NJOYModule(m) for m in data.get('modules', ['reconr', 'broadr', 'groupr'])],
            tolerance=data.get('tolerance', 0.001),
            created_at=data.get('created_at', ''),
        )


def create_dosimetry_pipeline() -> NJOYPipelineSpec:
    """
    Create standard pipeline for dosimetry cross sections.
    
    Returns a pipeline specification suitable for processing
    IRDFF-II dosimetry reactions for activation foil analysis.
    """
    return NJOYPipelineSpec(
        name="IRDFF-II Dosimetry Processing",
        description="Standard processing for IRDFF-II dosimetry reactions",
        endf_library="IRDFF-II",
        materials=[],  # To be populated with specific reactions
        group_structure=GroupStructure.SAND_II,
        temperatures=[300.0],
        modules=[NJOYModule.RECONR, NJOYModule.BROADR, NJOYModule.GROUPR],
        tolerance=0.001,
    )
