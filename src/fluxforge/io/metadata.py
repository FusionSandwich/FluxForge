"""Metadata helpers for spectrum readers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence


def is_missing_value(value: Any) -> bool:
    """Return True when a metadata value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def append_missing_field_flags(
    *,
    mapping: Mapping[str, Any],
    required_fields: Sequence[str],
    flags: List[str],
    key_prefix: str = "missing_",
) -> None:
    """Append missing-field QC flags for required mapping keys."""
    for field in required_fields:
        if is_missing_value(mapping.get(field)):
            flags.append(f"{key_prefix}{field}")


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
    append_missing_field_flags(
        mapping={"spectrum_id": spectrum_id},
        required_fields=("spectrum_id",),
        flags=flags,
    )

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

    if detector_id is not None and is_missing_value(detector_id):
        flags.append("missing_detector_id")

    return flags
