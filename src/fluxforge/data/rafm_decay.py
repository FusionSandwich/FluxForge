"""Authoritative RAFM decay-data subset bundled with FluxForge.

The dataset is a committed subset extracted from the local ``actigamma``
``decay_2012`` library so the RAFM example and flux-wire workflows do not rely
on external repositories at runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_PATH = Path(__file__).with_name("rafm_decay_data.json")


@lru_cache(maxsize=1)
def load_rafm_decay_library() -> Dict[str, Dict[str, Any]]:
    """Load the bundled RAFM decay-data subset."""
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data.pop("_metadata", None)
    return data


def get_rafm_decay_entry(isotope: str) -> Optional[Dict[str, Any]]:
    """Return the committed decay-data entry for an isotope."""
    return load_rafm_decay_library().get(isotope)


def get_rafm_gamma_lines(isotope: str) -> List[Dict[str, float]]:
    """Return gamma lines for an isotope from the bundled RAFM library."""
    entry = get_rafm_decay_entry(isotope)
    if not entry:
        return []
    return [dict(line) for line in entry.get("gamma_lines", [])]


def get_rafm_half_life(isotope: str) -> Optional[float]:
    """Return half-life in seconds for an isotope from the bundled RAFM library."""
    entry = get_rafm_decay_entry(isotope)
    if not entry:
        return None
    return float(entry.get("half_life_seconds", 0.0))
