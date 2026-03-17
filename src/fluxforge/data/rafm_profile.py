"""Bundled RAFM detector/counting presets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RAFMProfile:
    """Bundled RAFM profile metadata."""

    name: str
    description: str
    background_relative_path: Optional[str]
    energy_calibration: List[float]
    efficiency: Dict[str, Any]
    resolution: List[float]

    def resolve_background_path(self, repo_root: Optional[Path] = None) -> Optional[Path]:
        """Resolve the bundled background path relative to the FluxForge repo root."""
        if not self.background_relative_path:
            return None
        base = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
        return (base / self.background_relative_path).resolve()


def _load_profile_payload() -> Dict[str, Any]:
    with resources.files("fluxforge.data").joinpath("rafm_profiles.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def list_rafm_profiles() -> List[str]:
    """List bundled RAFM profile names."""
    return sorted(_load_profile_payload().keys())


def load_rafm_profile(name: str) -> RAFMProfile:
    """Load a bundled RAFM profile by name."""
    payload = _load_profile_payload()
    if name not in payload:
        available = ", ".join(sorted(payload))
        raise ValueError(f"Unknown RAFM profile '{name}'. Available profiles: {available}")
    entry = payload[name]
    return RAFMProfile(
        name=name,
        description=str(entry.get("description", "")),
        background_relative_path=entry.get("background_relative_path"),
        energy_calibration=[float(value) for value in entry.get("energy_calibration", [])],
        efficiency=dict(entry.get("efficiency", {})),
        resolution=[float(value) for value in entry.get("resolution", [])],
    )
