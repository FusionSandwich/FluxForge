"""Metadata helpers for spectrum readers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def qc_flags_for_spectrum(
    *,
    spectrum_id: str,
    live_time: float,
    real_time: float,
    start_time: Optional[datetime],
    calibration: Optional[Dict[str, Any]] = None,
    detector_id: Optional[str] = None,
) -> List[str]:
    """Build QC flags for missing/inconsistent metadata fields."""
    flags: List[str] = []

    if _missing(spectrum_id):
        flags.append("missing_spectrum_id")

    if live_time <= 0:
        flags.append("missing_live_time")
    if real_time <= 0:
        flags.append("missing_real_time")
    if live_time > 0 and real_time > 0 and live_time > real_time:
        flags.append("live_time_exceeds_real_time")

    if start_time is None:
        flags.append("missing_start_time")

    if calibration is None or not calibration.get("energy"):
        flags.append("missing_energy_calibration")

    if detector_id is not None and _missing(detector_id):
        flags.append("missing_detector_id")

    return flags
