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
    background_resource: Optional[str] = None

    def resolve_background_path(
        self, repo_root: Optional[Path] = None
    ) -> Optional[Path]:
        """Resolve a packaged background, or an explicit repository override.

        Installed wheels keep the selected background beside the profile data.
        The legacy repository-relative locator remains available to callers
        supplying repo_root and to profiles without a packaged resource.
        """
        if repo_root is not None and self.background_relative_path:
            return (repo_root / self.background_relative_path).resolve()
        if self.background_resource:
            resource = resources.files("fluxforge.data").joinpath(self.background_resource)
            if not resource.is_file():
                raise FileNotFoundError(
                    f"Packaged background for profile {self.name!r} is missing: {resource}"
                )
            # Wheel installations expose package data as persistent filesystem files.
            return Path(str(resource)).resolve()
        if not self.background_relative_path:
            return None
        return (Path(__file__).resolve().parents[3] / self.background_relative_path).resolve()


def _load_profile_payload() -> Dict[str, Any]:
    with (
        resources.files("fluxforge.data")
        .joinpath("rafm_profiles.json")
        .open(
            "r",
            encoding="utf-8",
        ) as handle
    ):
        return json.load(handle)


def list_rafm_profiles() -> List[str]:
    """List bundled RAFM profile names."""
    return sorted(_load_profile_payload().keys())


def load_rafm_profile(name: str) -> RAFMProfile:
    """Load a bundled RAFM profile by name."""
    payload = _load_profile_payload()
    if name not in payload:
        available = ", ".join(sorted(payload))
        raise ValueError(
            f"Unknown RAFM profile '{name}'. Available profiles: {available}"
        )
    entry = payload[name]
    return RAFMProfile(
        name=name,
        description=str(entry.get("description", "")),
        background_relative_path=entry.get("background_relative_path"),
        background_resource=entry.get("background_resource"),
        energy_calibration=[
            float(value) for value in entry.get("energy_calibration", [])
        ],
        efficiency=dict(entry.get("efficiency", {})),
        resolution=[float(value) for value in entry.get("resolution", [])],
    )
