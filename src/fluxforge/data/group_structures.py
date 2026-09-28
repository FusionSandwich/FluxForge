"""Exact standard neutron energy group structures.

Boundaries are stored in ``group_structures.json`` (eV, ascending) with the
source of every structure:

* All default structures of ``openmc.mgxs.GROUP_STRUCTURES`` under OpenMC's
  names (CASMO-2 ... ECCO-1968), copied verbatim.
* ``VITAMIN-J-175-NJOY``: the NJOY2016 GROUPR option-17 VITAMIN-J boundaries
  used for ALARA 175-group libraries (aliases ``VITAMIN-J`` and
  ``ALARA-175``). OpenMC's ``VITAMIN-J-175`` is stored with fewer significant
  figures and differs from it at 19 boundaries (<= 3.3e-4 relative, largest
  72,024 eV vs 72,000 eV); use the NJOY structure for ALARA-consistent work.
* ``SAND-II-725`` and ``IRDFF-640`` from the IAEA IRDFF-II ``.egb`` files.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

_DATA_PATH = Path(__file__).with_name("group_structures.json")


@lru_cache(maxsize=1)
def _load() -> Dict:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _normalize(name: str) -> str:
    return re.sub(r"[\s_]+", "-", str(name).strip()).upper()


def _resolve(name: str) -> str:
    data = _load()
    key = _normalize(name)
    by_upper = {k.upper(): k for k in data["structures"]}
    if key in by_upper:
        return by_upper[key]
    aliases = {k.upper(): v for k, v in data["aliases"].items()}
    if key in aliases:
        return aliases[key]
    raise KeyError(
        f"Unknown group structure {name!r}; available: {', '.join(list_group_structures())}"
    )


def list_group_structures(include_aliases: bool = False) -> List[str]:
    """Names of the available structures (optionally with aliases)."""
    data = _load()
    names = list(data["structures"])
    if include_aliases:
        names += list(data["aliases"])
    return names


def get_group_structure(name: str, descending: bool = False) -> np.ndarray:
    """
    Exact boundaries in eV for a named structure.

    ``descending=True`` returns high-to-low order (ALARA/library indexing,
    group 1 highest energy).
    """
    edges = np.array(_load()["structures"][_resolve(name)]["edges_eV"], dtype=float)
    return edges[::-1].copy() if descending else edges


def group_structure_info(name: str) -> Dict[str, object]:
    """Canonical name, group count and source record of a structure."""
    data = _load()
    canonical = _resolve(name)
    entry = data["structures"][canonical]
    return {
        "name": canonical,
        "n_groups": len(entry["edges_eV"]) - 1,
        "source": data["sources"][entry["source"]],
    }


__all__ = ["get_group_structure", "group_structure_info", "list_group_structures"]
