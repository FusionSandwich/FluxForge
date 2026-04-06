"""
Decay data library loader for offline workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


_UNIT_TO_SECONDS = {
    "us": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def normalize_nuclide_label(label: str) -> str:
    """
    Normalize nuclide labels to the form 'El-AAA' or 'El-AAAm'.
    """
    import re

    raw = str(label).strip().replace(" ", "")
    raw = raw.replace("_", "-")

    patterns = [
        re.compile(r"^(?P<el>[A-Za-z]{1,3})-?(?P<mass>\d{1,3})(?P<meta>m\d*|m)?$"),
        re.compile(r"^(?P<mass>\d{1,3})(?P<meta>m\d*|m)?-?(?P<el>[A-Za-z]{1,3})$"),
    ]

    for pat in patterns:
        match = pat.match(raw)
        if match:
            el = match.group("el").capitalize()
            mass = int(match.group("mass"))
            meta = match.group("meta") or ""
            meta = meta.lower()
            return f"{el}-{mass}{meta}"

    return raw


@dataclass
class DecayDataset:
    """
    In-memory decay dataset.
    """

    half_lives_s: Dict[str, float]
    atomic_masses_g_mol: Dict[str, float]
    progeny: Dict[str, List[str]]
    branching: Dict[str, List[float]]
    modes: Dict[str, List[str]]
    half_life_uncertainties_s: Dict[str, float] = field(default_factory=dict)

    def half_life_s(self, nuclide: str) -> Optional[float]:
        return self.half_lives_s.get(normalize_nuclide_label(nuclide))

    def atomic_mass(self, nuclide: str) -> Optional[float]:
        return self.atomic_masses_g_mol.get(normalize_nuclide_label(nuclide))

    def half_life_uncertainty_s(self, nuclide: str) -> Optional[float]:
        return self.half_life_uncertainties_s.get(normalize_nuclide_label(nuclide))

    def decay_products(self, nuclide: str) -> List[str]:
        return list(self.progeny.get(normalize_nuclide_label(nuclide), []))

    def branching_fractions(self, nuclide: str) -> List[float]:
        return list(self.branching.get(normalize_nuclide_label(nuclide), []))

    def decay_modes(self, nuclide: str) -> List[str]:
        return list(self.modes.get(normalize_nuclide_label(nuclide), []))

    @classmethod
    def from_radioactivedecay_npz(
        cls,
        path: Union[str, Path],
        *,
        half_life_overrides_s: Optional[Dict[str, float]] = None,
        half_life_uncertainties_s: Optional[Dict[str, float]] = None,
    ) -> "DecayDataset":
        """
        Load decay data from radioactivedecay npz bundle.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        nuclides = [normalize_nuclide_label(n) for n in data["nuclides"]]
        masses = data["masses"]
        hldata = data["hldata"]
        progeny = data["progeny"]
        bfs = data["bfs"]
        modes = data["modes"]
        year_conv = float(data["year_conv"])

        half_lives_s: Dict[str, float] = {}
        atomic_masses_g_mol: Dict[str, float] = {}
        progeny_map: Dict[str, List[str]] = {}
        branching_map: Dict[str, List[float]] = {}
        modes_map: Dict[str, List[str]] = {}
        half_life_override_map = {
            normalize_nuclide_label(key): float(value)
            for key, value in (half_life_overrides_s or {}).items()
        }
        half_life_uncertainty_map = {
            normalize_nuclide_label(key): float(value)
            for key, value in (half_life_uncertainties_s or {}).items()
            if value is not None
        }

        for idx, nuclide in enumerate(nuclides):
            atomic_masses_g_mol[nuclide] = float(masses[idx])

            h_val, h_unit, _ = hldata[idx]
            unit = str(h_unit)
            if unit == "y":
                half_life = float(h_val) * year_conv * 86400.0
            else:
                half_life = float(h_val) * _UNIT_TO_SECONDS.get(unit, 1.0)
            half_life = half_life_override_map.get(nuclide, half_life)
            half_lives_s[nuclide] = half_life

            progeny_list = [normalize_nuclide_label(p) for p in progeny[idx]]
            progeny_map[nuclide] = progeny_list
            branching_map[nuclide] = [float(x) for x in bfs[idx]]
            modes_map[nuclide] = [str(x) for x in modes[idx]]

        return cls(
            half_lives_s=half_lives_s,
            atomic_masses_g_mol=atomic_masses_g_mol,
            progeny=progeny_map,
            branching=branching_map,
            modes=modes_map,
            half_life_uncertainties_s=half_life_uncertainty_map,
        )
