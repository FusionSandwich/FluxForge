"""Tests for CSV reader helper workflows."""

from __future__ import annotations

from fluxforge.io.csv_readers import read_efficiency_export, read_flux_wire_timing_csv


def test_read_efficiency_export_parses_table_and_coefficients(tmp_path) -> None:
    csv_path = tmp_path / "eff.csv"
    csv_path.write_text(
        "C1,C2,C3,C4,Sample\n"
        "1.0,2.0,3.0,4.0,demo\n"
        "Energy,Eff\n"
        "100,0.10\n"
        "200,0.20\n",
        encoding="utf-8",
    )

    parsed = read_efficiency_export(csv_path)

    assert parsed.coefficients["C1"] == 1.0
    assert parsed.coefficients["C4"] == 4.0
    assert parsed.header["Sample"] == "demo"
    assert parsed.table["Energy"].tolist() == [100.0, 200.0]
    assert parsed.table["Eff"].tolist() == [0.10, 0.20]
    assert parsed.qc_flags == []


def test_read_flux_wire_timing_csv_flags_missing_and_invalid_values(tmp_path) -> None:
    csv_path = tmp_path / "timing.csv"
    csv_path.write_text(
        "wire_name,base_name,category,reaction,products,irradiation_start,irradiation_end,measurement_time,irradiation_seconds,cooldown_seconds,cooldown_hours,cooldown_days\n"
        'W1,B1,fast,"(n,g)",W-187,2026-03-01T00:00:00,2026-03-02T00:00:00,2026-03-03T00:00:00,,abc,12,0.5\n',
        encoding="utf-8",
    )

    rows = read_flux_wire_timing_csv(csv_path)
    assert len(rows) == 1

    qc = rows[0].qc_flags
    assert "missing_irradiation_seconds" in qc
    assert "invalid_cooldown_seconds" in qc
    assert "missing_irradiation_duration" in qc
