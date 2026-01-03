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
    if source_hashes:
        provenance["hashes"] = source_hashes
    if notes:
        provenance["notes"] = notes
    return provenance
