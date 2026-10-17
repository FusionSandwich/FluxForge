"""Explicit FISPACT inventory JSON import and irradiation-history export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_fispact_activities(path, *, interval_index, required_zai=()):
    """Read activity Bq by exact ZAI, including metastable states.

    interval_index is ZERO-based and explicitly required. Missing nuclides are
    not replaced with zero; required absent ZAI raises KeyError. This reader
    supports the documented inventory_data/nuclides/zai/activity JSON layout,
    not every historical FISPACT export variant.
    """
    if (
        isinstance(interval_index, bool)
        or not isinstance(interval_index, int)
        or interval_index < 0
    ):
        raise ValueError("interval_index must be an explicit nonnegative integer")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("FISPACT inventory root must be an object")
    intervals = data.get("inventory_data")
    if not isinstance(intervals, list) or interval_index >= len(intervals):
        raise ValueError("requested inventory interval is unavailable")
    interval = intervals[interval_index]
    if not isinstance(interval, dict):
        raise ValueError("inventory interval must be an object")
    nuclides = interval.get("nuclides")
    if not isinstance(nuclides, list):
        raise ValueError("inventory interval requires a nuclides list")
    result = {}
    for row in nuclides:
        if not isinstance(row, dict):
            raise ValueError("nuclide record must be an object")
        zai = row.get("zai")
        if isinstance(zai, bool) or not isinstance(zai, int) or zai <= 0:
            raise ValueError("each nuclide requires an exact positive integer ZAI")
        if zai in result:
            raise ValueError(f"duplicate ZAI {zai}")
        activity = row.get("activity")
        if isinstance(activity, bool) or not isinstance(activity, (int, float)):
            raise ValueError("each nuclide requires a numeric activity")
        if not np.isfinite(activity) or activity < 0:
            raise ValueError("activity must be finite and nonnegative (Bq)")
        result[zai] = float(activity)
    for zai in required_zai:
        if zai not in result:
            raise KeyError(f"ZAI {zai} absent from inventory interval {interval_index}")
    return result


def fispact_history_text(flux_cm2_s, durations_s):
    """Render FLUX/TIME/ATOMS cards from scalar flux in particles/cm²/s.

    Total particles/s or current in µA must first be converted using a justified
    beam/target area and charge state. Zero-flux intervals retain decay pauses.
    Each TIME is followed by ATOMS so every interval is solved. This renders
    an inventory-phase history fragment following initial setup/ATOMS, not a
    complete FISPACT input deck. It does not reset the clock with ZERO.
    """
    flux, durations = (np.asarray(v, dtype=float) for v in (flux_cm2_s, durations_s))
    if (
        flux.ndim != 1
        or not flux.size
        or durations.shape != flux.shape
        or not np.isfinite(flux).all()
        or not np.isfinite(durations).all()
        or np.any(flux < 0)
        or np.any(durations <= 0)
    ):
        raise ValueError(
            "flux/duration must be matched nonnegative/positive finite vectors"
        )
    lines = []
    for rate, time in zip(flux, durations):
        lines.extend((f"FLUX {rate:.17g}", f"TIME {time:.17g} SECS", "ATOMS"))
    return "\n".join([*lines, ""])


def write_fispact_history(path, flux_cm2_s, durations_s):
    text = fispact_history_text(flux_cm2_s, durations_s)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return Path(path)
