"""SQLite-backed QA monitor for ASTM-oriented drift tracking."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


def _default_db_path() -> Path:
    root = Path.home() / ".fluxforge"
    root.mkdir(parents=True, exist_ok=True)
    return root / "qa_history.db"


@dataclass(frozen=True)
class QARecord:
    """One QA history point captured from a check-source spectrum."""

    timestamp: datetime
    nuclide: str
    energy_keV: float
    measured_centroid_keV: float
    measured_fwhm_keV: float
    measured_fwhm_channels: float
    net_counts: int
    efficiency: float
    spectrum_file: str
    annotation: str = ""


@dataclass(frozen=True)
class QAStatus:
    """Current drift status for one nuclide/energy combination."""

    nuclide: str
    energy_keV: float
    centroid_drift_keV: float
    fwhm_degradation_pct: float
    efficiency_deviation_pct: float
    status: str
    last_check: datetime


class QAMonitor:
    """SQLite-backed history and drift monitor for QA check sources."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    timestamp TEXT NOT NULL,
                    nuclide TEXT NOT NULL,
                    energy_keV REAL NOT NULL,
                    measured_centroid_keV REAL NOT NULL,
                    measured_fwhm_keV REAL NOT NULL,
                    measured_fwhm_channels REAL NOT NULL,
                    net_counts INTEGER NOT NULL,
                    efficiency REAL NOT NULL,
                    spectrum_file TEXT NOT NULL,
                    annotation TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def record(self, record: QARecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO qa_history (
                    timestamp,
                    nuclide,
                    energy_keV,
                    measured_centroid_keV,
                    measured_fwhm_keV,
                    measured_fwhm_channels,
                    net_counts,
                    efficiency,
                    spectrum_file,
                    annotation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.nuclide,
                    float(record.energy_keV),
                    float(record.measured_centroid_keV),
                    float(record.measured_fwhm_keV),
                    float(record.measured_fwhm_channels),
                    int(record.net_counts),
                    float(record.efficiency),
                    record.spectrum_file,
                    record.annotation,
                ),
            )

    def history(self) -> tuple[QARecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    timestamp,
                    nuclide,
                    energy_keV,
                    measured_centroid_keV,
                    measured_fwhm_keV,
                    measured_fwhm_channels,
                    net_counts,
                    efficiency,
                    spectrum_file,
                    annotation
                FROM qa_history
                ORDER BY timestamp ASC
                """
            ).fetchall()
        return tuple(
            QARecord(
                timestamp=datetime.fromisoformat(row[0]),
                nuclide=row[1],
                energy_keV=float(row[2]),
                measured_centroid_keV=float(row[3]),
                measured_fwhm_keV=float(row[4]),
                measured_fwhm_channels=float(row[5]),
                net_counts=int(row[6]),
                efficiency=float(row[7]),
                spectrum_file=row[8],
                annotation=row[9],
            )
            for row in rows
        )

    def grouped_history(self) -> dict[tuple[str, float], tuple[QARecord, ...]]:
        groups: dict[tuple[str, float], list[QARecord]] = {}
        for record in self.history():
            groups.setdefault((record.nuclide, record.energy_keV), []).append(record)
        return {key: tuple(values) for key, values in groups.items()}

    def status_snapshot(self) -> tuple[QAStatus, ...]:
        statuses: list[QAStatus] = []
        for (nuclide, energy), series in self.grouped_history().items():
            if len(series) < 2:
                baseline = series[0]
                latest = series[-1]
            else:
                baseline = series[0]
                latest = series[-1]
            centroid_drift = latest.measured_centroid_keV - baseline.measured_centroid_keV
            fwhm_degradation = 100.0 * (
                latest.measured_fwhm_keV - baseline.measured_fwhm_keV
            ) / max(baseline.measured_fwhm_keV, 1e-12)
            efficiency_deviation = 100.0 * (
                latest.efficiency - baseline.efficiency
            ) / max(abs(baseline.efficiency), 1e-12)
            status = "green"
            if (
                abs(centroid_drift) > 1.0
                or abs(fwhm_degradation) > 20.0
                or abs(efficiency_deviation) > 10.0
            ):
                status = "red"
            elif (
                abs(centroid_drift) > 0.5
                or abs(fwhm_degradation) > 10.0
                or abs(efficiency_deviation) > 5.0
            ):
                status = "amber"
            statuses.append(
                QAStatus(
                    nuclide=nuclide,
                    energy_keV=energy,
                    centroid_drift_keV=centroid_drift,
                    fwhm_degradation_pct=fwhm_degradation,
                    efficiency_deviation_pct=efficiency_deviation,
                    status=status,
                    last_check=latest.timestamp,
                )
            )
        return tuple(sorted(statuses, key=lambda item: (item.nuclide, item.energy_keV)))

    def seed_demo_history(self) -> None:
        """Populate a small deterministic demo history when the database is empty."""

        if self.history():
            return
        anchor = datetime(2026, 3, 15, 14, 22)
        demo_points = (
            QARecord(
                timestamp=anchor,
                nuclide="Cs-137",
                energy_keV=661.66,
                measured_centroid_keV=661.64,
                measured_fwhm_keV=1.80,
                measured_fwhm_channels=2.3,
                net_counts=12500,
                efficiency=0.081,
                spectrum_file="demo_cs137_001.spe",
                annotation="baseline",
            ),
            QARecord(
                timestamp=anchor + timedelta(days=7),
                nuclide="Cs-137",
                energy_keV=661.66,
                measured_centroid_keV=661.68,
                measured_fwhm_keV=1.82,
                measured_fwhm_channels=2.31,
                net_counts=12150,
                efficiency=0.079,
                spectrum_file="demo_cs137_002.spe",
                annotation="weekly QA",
            ),
            QARecord(
                timestamp=anchor + timedelta(days=7),
                nuclide="Co-60",
                energy_keV=1173.23,
                measured_centroid_keV=1173.12,
                measured_fwhm_keV=2.05,
                measured_fwhm_channels=2.95,
                net_counts=9800,
                efficiency=0.043,
                spectrum_file="demo_co60_002.spe",
                annotation="weekly QA",
            ),
        )
        for record in demo_points:
            self.record(record)


__all__ = ["QAMonitor", "QARecord", "QAStatus"]
