"""CSV readers for FluxForge example exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _parse_datetime(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class EfficiencyExport:
    """LabSOCS-style efficiency CSV export."""

    header: Dict[str, str] = field(default_factory=dict)
    coefficients: Dict[str, float] = field(default_factory=dict)
    table: Dict[str, np.ndarray] = field(default_factory=dict)
    qc_flags: List[str] = field(default_factory=list)


def read_efficiency_export(path: Union[str, Path]) -> EfficiencyExport:
    """Read LabSOCS efficiency CSV export with header row and energy table."""
    path = Path(path)
    qc_flags: List[str] = []
    header: Dict[str, str] = {}
    coefficients: Dict[str, float] = {}
    table: Dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            header_row = next(reader)
            value_row = next(reader)
        except StopIteration:
            raise ValueError(f"Efficiency export {path} is missing header rows.")

        for key, value in zip(header_row, value_row):
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if key in {"C1", "C2", "C3", "C4", "A", "T1", "DI", "DL", "Error"}:
                try:
                    coefficients[key] = float(value)
                except ValueError:
                    qc_flags.append(f"invalid_{key.lower()}")
            else:
                header[key] = value

        table_header: Optional[Sequence[str]] = None
        rows: List[List[float]] = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if table_header is None:
                table_header = [cell.strip() for cell in row]
                continue
            try:
                rows.append([float(cell) for cell in row[: len(table_header)]])
            except ValueError:
                continue

        if table_header and rows:
            values = np.array(rows, dtype=float)
            for idx, name in enumerate(table_header):
                if name:
                    table[name] = values[:, idx]
        else:
            qc_flags.append("missing_efficiency_table")

    for key in ("C1", "C2", "C3", "C4"):
        if key not in coefficients:
            qc_flags.append(f"missing_{key.lower()}")

    return EfficiencyExport(
        header=header,
        coefficients=coefficients,
        table=table,
        qc_flags=qc_flags,
    )


@dataclass
class FluxWireTiming:
    """Timing metadata for flux wire measurements."""

    wire_name: str
    base_name: str
    category: str
    reaction: str
    products: str
    irradiation_start: Optional[datetime]
    irradiation_end: Optional[datetime]
    irradiation_seconds: float
    measurement_time: Optional[datetime]
    cooldown_seconds: float
    cooldown_hours: float
    cooldown_days: float
    qc_flags: List[str] = field(default_factory=list)


def read_flux_wire_timing_csv(path: Union[str, Path]) -> List[FluxWireTiming]:
    """Read flux_wire_timing.csv export."""
    path = Path(path)
    timing: List[FluxWireTiming] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qc_flags: List[str] = []
            for required in (
                "wire_name",
                "base_name",
                "category",
                "reaction",
                "products",
                "irradiation_start",
                "irradiation_end",
                "measurement_time",
            ):
                if not row.get(required):
                    qc_flags.append(f"missing_{required}")

            irradiation_start = _parse_datetime(row.get("irradiation_start", ""))
            irradiation_end = _parse_datetime(row.get("irradiation_end", ""))
            measurement_time = _parse_datetime(row.get("measurement_time", ""))

            def _parse_float(field: str) -> float:
                value = row.get(field, "").strip()
                if not value:
                    qc_flags.append(f"missing_{field}")
                    return 0.0
                try:
                    return float(value)
                except ValueError:
                    qc_flags.append(f"invalid_{field}")
                    return 0.0

            irradiation_seconds = _parse_float("irradiation_seconds")
            cooldown_seconds = _parse_float("cooldown_seconds")
            cooldown_hours = _parse_float("cooldown_hours")
            cooldown_days = _parse_float("cooldown_days")

            if irradiation_start and irradiation_end and irradiation_end < irradiation_start:
                qc_flags.append("irradiation_end_before_start")
            if irradiation_seconds <= 0:
                qc_flags.append("missing_irradiation_duration")

            timing.append(
                FluxWireTiming(
                    wire_name=row.get("wire_name", "").strip(),
                    base_name=row.get("base_name", "").strip(),
                    category=row.get("category", "").strip(),
                    reaction=row.get("reaction", "").strip(),
                    products=row.get("products", "").strip(),
                    irradiation_start=irradiation_start,
                    irradiation_end=irradiation_end,
                    irradiation_seconds=irradiation_seconds,
                    measurement_time=measurement_time,
                    cooldown_seconds=cooldown_seconds,
                    cooldown_hours=cooldown_hours,
                    cooldown_days=cooldown_days,
                    qc_flags=qc_flags,
                )
            )

    return timing
