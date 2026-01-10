"""Shared provenance helpers for FluxForge artifacts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fluxforge import __version__


def hash_bytes(payload: bytes) -> Dict[str, str]:
    """Return standard hashes for a byte payload."""
    sha256 = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()
    return {"sha256": sha256, "md5": md5}


def hash_file(path: Path) -> Dict[str, str]:
    """Hash a file on disk."""
    return hash_bytes(path.read_bytes())


def build_provenance(
    *,
    units: Dict[str, str],
    normalization: Optional[Dict[str, str]] = None,
    definitions: Optional[Dict[str, str]] = None,
    source_hashes: Optional[Dict[str, Dict[str, str]]] = None,
    extra_versions: Optional[Dict[str, str]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a provenance record for serialized artifacts."""
    versions = {"fluxforge": __version__}
    if extra_versions:
        versions.update(extra_versions)
    provenance: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "units": units,
    }
    if normalization:
        provenance["normalization"] = normalization
    if definitions:
        provenance["definitions"] = definitions
    if source_hashes:
        provenance["hashes"] = source_hashes
    if notes:
        provenance["notes"] = notes
    return provenance


# =============================================================================
# Library provenance enforcement (O1.6)
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class NuclearDataLibrary(Enum):
    """Supported nuclear data libraries with version tracking."""
    
    # Cross-section libraries
    IRDFF_II = "IRDFF-II"
    IRDFF_I = "IRDFF-I.0"
    ENDF_B_VIII_0 = "ENDF/B-VIII.0"
    ENDF_B_VII_1 = "ENDF/B-VII.1"
    JEFF_3_3 = "JEFF-3.3"
    JENDL_5 = "JENDL-5"
    
    # Decay data libraries
    ENSDF = "ENSDF"
    DDEP = "DDEP"
    IAEA_DDEP = "IAEA-DDEP"
    
    # Dosimetry-specific
    STAYSL_RXMD = "STAYSL-RXMD"
    
    # Unknown/custom
    UNKNOWN = "unknown"


@dataclass
class LibraryProvenance:
    """Provenance information for a nuclear data library.
    
    Attributes
    ----------
    library : NuclearDataLibrary
        Library identifier
    version : str
        Library version string
    release_date : str
        Release date (YYYY-MM-DD format)
    source_url : str
        URL where library was obtained
    file_hash : str
        SHA-256 hash of source file(s)
    validated : bool
        Whether the library has been validated
    notes : str
        Additional notes
    """
    
    library: NuclearDataLibrary
    version: str = ""
    release_date: str = ""
    source_url: str = ""
    file_hash: str = ""
    validated: bool = False
    notes: str = ""
    
    def is_complete(self) -> bool:
        """Check if provenance information is complete."""
        return (
            self.library != NuclearDataLibrary.UNKNOWN
            and self.version != ""
            and self.file_hash != ""
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "library": self.library.value,
            "version": self.version,
            "release_date": self.release_date,
            "source_url": self.source_url,
            "file_hash": self.file_hash,
            "validated": self.validated,
            "notes": self.notes,
            "is_complete": self.is_complete(),
        }


@dataclass
class ProvenanceBundle:
    """Complete provenance bundle for an analysis.
    
    Tracks provenance of all nuclear data libraries used in an analysis
    to ensure reproducibility and traceability.
    
    Attributes
    ----------
    transport_library : LibraryProvenance
        Transport cross-section library (e.g., ENDF/B-VIII.0)
    dosimetry_library : LibraryProvenance
        Dosimetry cross-sections (e.g., IRDFF-II)
    decay_library : LibraryProvenance
        Decay data library (e.g., ENSDF)
    additional_libraries : list[LibraryProvenance]
        Other libraries used
    analysis_timestamp : str
        When analysis was performed
    fluxforge_version : str
        FluxForge version used
    """
    
    transport_library: LibraryProvenance = field(
        default_factory=lambda: LibraryProvenance(NuclearDataLibrary.UNKNOWN)
    )
    dosimetry_library: LibraryProvenance = field(
        default_factory=lambda: LibraryProvenance(NuclearDataLibrary.UNKNOWN)
    )
    decay_library: LibraryProvenance = field(
        default_factory=lambda: LibraryProvenance(NuclearDataLibrary.UNKNOWN)
    )
    additional_libraries: List[LibraryProvenance] = field(default_factory=list)
    analysis_timestamp: str = ""
    fluxforge_version: str = ""
    
    def __post_init__(self):
        """Set defaults."""
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now(timezone.utc).isoformat()
        if not self.fluxforge_version:
            self.fluxforge_version = __version__
    
    def is_complete(self) -> bool:
        """Check if all required provenance is present."""
        return (
            self.dosimetry_library.is_complete()
            # transport and decay are optional for some analyses
        )
    
    def validate(self, strict: bool = False) -> tuple[bool, List[str]]:
        """Validate provenance bundle.
        
        Parameters
        ----------
        strict : bool
            If True, require all libraries to have complete provenance
            
        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of warning/error messages)
        """
        messages = []
        is_valid = True
        
        # Check dosimetry library (required)
        if not self.dosimetry_library.is_complete():
            messages.append("WARNING: Dosimetry library provenance incomplete")
            if strict:
                is_valid = False
                messages[-1] = messages[-1].replace("WARNING", "ERROR")
        
        # Check transport library
        if self.transport_library.library == NuclearDataLibrary.UNKNOWN:
            messages.append("INFO: Transport library not specified")
        elif not self.transport_library.is_complete():
            messages.append("WARNING: Transport library provenance incomplete")
        
        # Check decay library
        if self.decay_library.library == NuclearDataLibrary.UNKNOWN:
            messages.append("INFO: Decay library not specified")
        elif not self.decay_library.is_complete():
            messages.append("WARNING: Decay library provenance incomplete")
        
        return is_valid, messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_timestamp": self.analysis_timestamp,
            "fluxforge_version": self.fluxforge_version,
            "transport_library": self.transport_library.to_dict(),
            "dosimetry_library": self.dosimetry_library.to_dict(),
            "decay_library": self.decay_library.to_dict(),
            "additional_libraries": [lib.to_dict() for lib in self.additional_libraries],
            "is_complete": self.is_complete(),
        }


class ProvenanceError(Exception):
    """Exception raised when provenance validation fails."""
    pass


def validate_library_provenance(
    bundle: ProvenanceBundle,
    strict: bool = True,
    raise_on_error: bool = True,
) -> tuple[bool, List[str]]:
    """Validate library provenance before output generation.
    
    This function enforces provenance requirements per R2 guardrail:
    outputs without complete provenance are labeled as PROVISIONAL.
    
    Parameters
    ----------
    bundle : ProvenanceBundle
        Provenance bundle to validate
    strict : bool
        If True, require complete provenance for all libraries
    raise_on_error : bool
        If True, raise ProvenanceError on validation failure
        
    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, messages)
        
    Raises
    ------
    ProvenanceError
        If validation fails and raise_on_error is True
    """
    is_valid, messages = bundle.validate(strict=strict)
    
    if not is_valid and raise_on_error:
        raise ProvenanceError(
            f"Provenance validation failed: {'; '.join(messages)}"
        )
    
    return is_valid, messages


def mark_provisional(artifact_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Mark an artifact as provisional due to incomplete provenance.
    
    Parameters
    ----------
    artifact_dict : dict
        Artifact dictionary to mark
        
    Returns
    -------
    dict
        Artifact with provisional flag added
    """
    if "provenance" not in artifact_dict:
        artifact_dict["provenance"] = {}
    
    artifact_dict["provenance"]["provisional"] = True
    artifact_dict["provenance"]["provisional_reason"] = (
        "Library provenance incomplete - results not fully traceable"
    )
    
    return artifact_dict


def create_irdff2_provenance() -> LibraryProvenance:
    """Create standard IRDFF-II provenance record."""
    return LibraryProvenance(
        library=NuclearDataLibrary.IRDFF_II,
        version="2.0",
        release_date="2020-01-01",
        source_url="https://www-nds.iaea.org/IRDFF/",
        validated=True,
        notes="IAEA International Reactor Dosimetry and Fusion File",
    )


def create_endf8_provenance() -> LibraryProvenance:
    """Create standard ENDF/B-VIII.0 provenance record."""
    return LibraryProvenance(
        library=NuclearDataLibrary.ENDF_B_VIII_0,
        version="8.0",
        release_date="2018-02-02",
        source_url="https://www.nndc.bnl.gov/endf-b8.0/",
        validated=True,
        notes="ENDF/B-VIII.0 general purpose library",
    )


def create_ensdf_provenance() -> LibraryProvenance:
    """Create standard ENSDF provenance record."""
    return LibraryProvenance(
        library=NuclearDataLibrary.ENSDF,
        version="2024",
        release_date="2024-01-01",
        source_url="https://www.nndc.bnl.gov/ensdf/",
        validated=True,
        notes="Evaluated Nuclear Structure Data File",
    )
