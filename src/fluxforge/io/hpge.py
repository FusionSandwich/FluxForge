"""HPGe report/export readers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fluxforge.io.genie import ReportPeak, parse_genie_report
from fluxforge.io.metadata import qc_flags_for_spectrum

FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
ID_RE = re.compile(r"^\s*ID:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FILE_DATE_RE = re.compile(r"^\s*File:\s*(.+?)\s+Date:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
LT_RT_DT_RE = re.compile(
    r"LT:\s*([\d.,]+)\s*RT:\s*([\d.,]+)\s*DT:\s*([\d.,]+)\s*%?",
    re.IGNORECASE,
)
DETECTOR_ID_RE = re.compile(r"Detector\s+ID:\s*(.+)$", re.IGNORECASE)


def _parse_number(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _parse_datetime(value: str) -> Optional[datetime]:
    for fmt in (
        "%B %d, %Y %H:%M:%S",
        "%B %d, %Y %H:%M",
        "%b %d, %Y %H:%M:%S",
        "%b %d, %Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


@dataclass
class HPGeReport:
    """Parsed HPGe report export."""

    report_id: str
    file_name: str = ""
    start_time: Optional[datetime] = None
    live_time: float = 0.0
    real_time: float = 0.0
    dead_time_pct: Optional[float] = None
    detector_id: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    peaks: List[ReportPeak] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    qc_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "file_name": self.file_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "live_time": self.live_time,
            "real_time": self.real_time,
            "dead_time_pct": self.dead_time_pct,
            "detector_id": self.detector_id,
            "calibration": self.calibration,
            "efficiency": self.efficiency,
            "peaks": [peak.__dict__ for peak in self.peaks],
            "metadata": self.metadata,
            "qc_flags": self.qc_flags,
        }


def read_hpge_report(filepath: Union[str, Path]) -> HPGeReport:
    """Read HPGe report export (Genie/LabSOCS TXT)."""
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    report_id = ""
    match = ID_RE.search(content)
    if match:
        report_id = match.group(1).strip()

    file_name = ""
    start_time = None
    match = FILE_DATE_RE.search(content)
    if match:
        file_name = match.group(1).strip()
        start_time = _parse_datetime(match.group(2).strip())

    live_time = 0.0
    real_time = 0.0
    dead_time = None
    match = LT_RT_DT_RE.search(content)
    if match:
        parsed_live = _parse_number(match.group(1))
        parsed_real = _parse_number(match.group(2))
        parsed_dead = _parse_number(match.group(3))
        if parsed_live is not None:
            live_time = parsed_live
        if parsed_real is not None:
            real_time = parsed_real
        if parsed_dead is not None:
            dead_time = parsed_dead

    detector_id = ""
    for line in content.splitlines():
        match = DETECTOR_ID_RE.search(line)
        if match:
            detector_id = match.group(1).strip()
            break

    calibration: Dict[str, Any] = {}
    efficiency: Dict[str, Any] = {}
    for line in content.splitlines():
        if "Energy" in line and "Ch" in line:
            numbers = FLOAT_RE.findall(line)
            if len(numbers) >= 3:
                calibration["energy"] = [float(n) for n in numbers[:3]]
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {"C1", "C2", "C3", "C4", "A", "T1", "DI", "DL"}:
                parsed = _parse_number(value)
                if parsed is not None:
                    efficiency[key] = parsed

    peaks = parse_genie_report(filepath)

    qc_flags = qc_flags_for_spectrum(
        spectrum_id=report_id or filepath.stem,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        calibration=calibration,
        detector_id=detector_id,
    )

    metadata = {
        "source_file": str(filepath),
        "format": "hpge_report",
    }
    if dead_time is not None:
        metadata["dead_time_pct"] = dead_time

    return HPGeReport(
        report_id=report_id or filepath.stem,
        file_name=file_name,
        start_time=start_time,
        live_time=live_time,
        real_time=real_time,
        dead_time_pct=dead_time,
        detector_id=detector_id,
        calibration=calibration,
        efficiency=efficiency,
        peaks=peaks,
        metadata=metadata,
        qc_flags=qc_flags,
    )
