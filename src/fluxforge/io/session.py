"""FluxForge `.ffs` session file read/write helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fluxforge.hal import DeviceRegistry
from fluxforge.io.spe import GammaSpectrum


SESSION_FORMAT_VERSION = 1


@dataclass
class FluxForgeSession:
    """Serializable session container for the modern GUI and CLI."""

    spectra: list[GammaSpectrum] = field(default_factory=list)
    active_spectrum_index: int = 0
    source_files: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    device_snapshot: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the session to a JSON-serializable payload."""

        return {
            "format": "fluxforge_session",
            "format_version": SESSION_FORMAT_VERSION,
            "created_at": self.created_at,
            "active_spectrum_index": self.active_spectrum_index,
            "source_files": list(self.source_files),
            "recent_files": list(self.recent_files),
            "device_snapshot": list(self.device_snapshot),
            "metadata": dict(self.metadata),
            "spectra": [spectrum.to_dict() for spectrum in self.spectra],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FluxForgeSession":
        """Build a session object from a persisted payload."""

        spectra = [
            GammaSpectrum.from_dict(record) for record in payload.get("spectra", [])
        ]
        return cls(
            spectra=spectra,
            active_spectrum_index=int(payload.get("active_spectrum_index", 0) or 0),
            source_files=[str(value) for value in payload.get("source_files", [])],
            recent_files=[str(value) for value in payload.get("recent_files", [])],
            device_snapshot=[
                {str(key): str(value) for key, value in dict(item).items()}
                for item in payload.get("device_snapshot", [])
                if isinstance(item, dict)
            ],
            metadata=dict(payload.get("metadata", {})),
            created_at=str(payload.get("created_at") or ""),
        )


def session_from_spectra(
    spectra: Iterable[GammaSpectrum],
    *,
    active_spectrum_index: int = 0,
    source_files: Iterable[str | Path] = (),
    recent_files: Iterable[str | Path] = (),
    device_registry: DeviceRegistry | None = None,
    metadata: dict[str, Any] | None = None,
) -> FluxForgeSession:
    """Create a session from spectra and optional workflow metadata."""

    return FluxForgeSession(
        spectra=list(spectra),
        active_spectrum_index=active_spectrum_index,
        source_files=[str(Path(path)) for path in source_files],
        recent_files=[str(Path(path)) for path in recent_files],
        device_snapshot=(
            device_registry.snapshot() if device_registry is not None else []
        ),
        metadata=dict(metadata or {}),
    )


def write_ffs_session(path: str | Path, session: FluxForgeSession) -> dict[str, Any]:
    """Write a `.ffs` session file."""

    path = Path(path)
    if path.suffix.lower() != ".ffs":
        raise ValueError("FluxForge sessions must use the `.ffs` extension.")
    payload = session.to_dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_ffs_session(path: str | Path) -> FluxForgeSession:
    """Read a `.ffs` session file."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "fluxforge_session":
        raise ValueError(f"{path} is not a FluxForge session file.")
    return FluxForgeSession.from_dict(payload)


__all__ = [
    "FluxForgeSession",
    "SESSION_FORMAT_VERSION",
    "read_ffs_session",
    "session_from_spectra",
    "write_ffs_session",
]
