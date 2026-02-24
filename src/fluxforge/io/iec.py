"""
IEC 62755 ASCII spectrum reader.

Parses IEC text exports that store MCA metadata + channel counts
with an A### line prefix (commonly "A004").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from fluxforge.io.spe import GammaSpectrum


FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class IECSpectrum:
    """Parsed IEC spectrum data."""

    counts: np.ndarray
    channels: np.ndarray
    live_time: float
    real_time: float
    start_time: Optional[datetime] = None
    calibration: List[float] = field(default_factory=list)
    shape: List[float] = field(default_factory=list)
    detector_id: str = ""
    data_source: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


def _strip_prefix(line: str) -> str:
    if len(line) >= 4 and line[0].upper() == "A" and line[1:4].isdigit():
        return line[4:]
    return line


def _extract_numbers(line: str) -> List[float]:
    return [float(val) for val in FLOAT_RE.findall(line)]


def _parse_datetime(tokens: List[str]) -> Optional[datetime]:
    if len(tokens) < 2:
        return None
    candidate = f"{tokens[0]} {tokens[1]}"
    for fmt in ("%d/%m/%y %H:%M:%S", "%m/%d/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def read_iec_file(filepath: Union[str, Path]) -> GammaSpectrum:
    """
    Read an IEC 62755 ASCII spectrum file and return a GammaSpectrum.
    """
    filepath = Path(filepath)
    lines = [line.rstrip("\n") for line in filepath.read_text(encoding="utf-8", errors="ignore").splitlines()]
    stripped = [_strip_prefix(line).strip() for line in lines]

    if len(stripped) < 6:
        raise ValueError(f"IEC file is too short to parse: {filepath}")

    data_source = ""
    detector_id = ""
    live_time = 0.0
    real_time = 0.0
    n_channels = 0
    start_time = None
    energy_coeffs: List[float] = []
    shape_coeffs: List[float] = []

    header_tokens = stripped[0].split()
    if len(header_tokens) >= 2:
        data_source = header_tokens[0]
        detector_id = header_tokens[1]

    timing_nums = _extract_numbers(stripped[1])
    if len(timing_nums) >= 3:
        live_time = float(timing_nums[0])
        real_time = float(timing_nums[1])
        n_channels = int(round(timing_nums[2]))

    start_time = _parse_datetime(stripped[2].split())

    cal_nums = _extract_numbers(stripped[3])
    if len(cal_nums) >= 3:
        energy_coeffs = cal_nums[:3]

    shape_nums = _extract_numbers(stripped[4])
    if len(shape_nums) >= 4:
        shape_coeffs = shape_nums[:4]

    counts: List[int] = []
    started = False
    previous_index = None

    for line in stripped[5:]:
        if not line:
            continue
        tokens = line.split()
        if not tokens:
            continue
        if not started:
            if len(tokens) >= 2 and tokens[0].lstrip("+-").isdigit() and tokens[1].lstrip("+-").isdigit():
                started = True
            else:
                continue
        if len(tokens) < 2:
            continue
        if tokens[0].lstrip("+-").isdigit():
            try:
                idx = int(tokens[0])
            except ValueError:
                idx = None
            if idx is not None:
                if previous_index is not None and idx < previous_index:
                    continue
                previous_index = idx
            for tok in tokens[1:]:
                if tok.lstrip("+-").isdigit():
                    counts.append(int(tok))
                else:
                    try:
                        counts.append(int(float(tok)))
                    except ValueError:
                        continue
        if n_channels and len(counts) >= n_channels:
            counts = counts[:n_channels]
            break

    if not counts:
        raise ValueError(f"No channel data found in IEC file: {filepath}")

    channels = np.arange(len(counts))
    calibration = {"energy": energy_coeffs} if energy_coeffs else {}
    metadata = {
        "format": "IEC",
        "data_source": data_source,
    }
    if shape_coeffs:
        metadata["shape_coeffs"] = ",".join(str(val) for val in shape_coeffs)

    return GammaSpectrum(
        counts=np.array(counts, dtype=float),
        channels=channels,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        spectrum_id=filepath.stem,
        detector_id=detector_id,
        calibration=calibration,
        metadata=metadata,
    )
