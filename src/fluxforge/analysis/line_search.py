"""
Gamma/X-ray line search utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from fluxforge.data.gamma_database import GammaDatabase, GammaLine, DecayData


@dataclass
class LineMatch:
    """Matched decay line."""

    nuclide: str
    energy_keV: float
    intensity: float
    line_type: str


def _normalize_db_key(label: str) -> str:
    return str(label).strip().replace(" ", "").replace("-", "")


def _iter_decay_lines(decay: DecayData, line_type: str) -> Iterable[GammaLine]:
    if line_type in ("gamma", "x-ray", "xray", "beta", "alpha", "ec", "electron"):
        return decay.get_lines(line_type)
    return []


def search_decay_lines(
    energy_keV: float,
    tolerance_keV: float,
    line_type: str = "gamma",
    min_intensity: float = 0.0,
    database: Optional[GammaDatabase] = None,
) -> List[LineMatch]:
    """
    Search for decay lines near a target energy.
    """
    db = database or GammaDatabase()
    matches: List[LineMatch] = []
    for nuclide, decay in db._nuclides.items():
        for line in _iter_decay_lines(decay, line_type):
            line_energy = line.energy_keV
            if (
                abs(line_energy - energy_keV) <= tolerance_keV
                and line.intensity >= min_intensity
            ):
                matches.append(
                    LineMatch(
                        nuclide=decay.nuclide,
                        energy_keV=line_energy,
                        intensity=line.intensity * line.norm,
                        line_type=line_type,
                    )
                )
    return sorted(matches, key=lambda m: (abs(m.energy_keV - energy_keV), -m.intensity))


def list_nuclide_lines(
    nuclide: str,
    line_type: str = "gamma",
    min_intensity: float = 0.0,
    database: Optional[GammaDatabase] = None,
) -> List[LineMatch]:
    """List decay lines for a specific nuclide."""
    db = database or GammaDatabase()
    decay = db.get(_normalize_db_key(nuclide))
    if decay is None:
        return []
    lines = [
        LineMatch(
            nuclide=decay.nuclide,
            energy_keV=line.energy_keV,
            intensity=line.intensity * line.norm,
            line_type=line_type,
        )
        for line in _iter_decay_lines(decay, line_type)
        if line.intensity * line.norm >= min_intensity
    ]
    return sorted(lines, key=lambda m: -m.intensity)
