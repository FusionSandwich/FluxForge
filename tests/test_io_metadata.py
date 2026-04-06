"""Tests for io.metadata helper functions."""

from __future__ import annotations

from datetime import datetime

from fluxforge.io.metadata import (
    append_missing_field_flags,
    is_missing_value,
    qc_flags_for_spectrum,
)


def test_is_missing_value_behavior() -> None:
    assert is_missing_value(None)
    assert is_missing_value("")
    assert is_missing_value("   ")
    assert not is_missing_value("x")
    assert not is_missing_value(0)


def test_append_missing_field_flags_appends_expected_keys() -> None:
    flags: list[str] = []
    append_missing_field_flags(
        mapping={"a": "", "b": "ok", "c": None},
        required_fields=("a", "b", "c", "d"),
        flags=flags,
    )
    assert flags == ["missing_a", "missing_c", "missing_d"]


def test_qc_flags_for_spectrum_regression() -> None:
    flags = qc_flags_for_spectrum(
        spectrum_id="",
        live_time=0.0,
        real_time=0.0,
        start_time=None,
        calibration=None,
        detector_id=" ",
    )

    assert "missing_spectrum_id" in flags
    assert "missing_live_time" in flags
    assert "missing_real_time" in flags
    assert "missing_start_time" in flags
    assert "missing_energy_calibration" in flags
    assert "missing_detector_id" in flags


def test_qc_flags_for_spectrum_valid_case() -> None:
    flags = qc_flags_for_spectrum(
        spectrum_id="spec-1",
        live_time=100.0,
        real_time=120.0,
        start_time=datetime(2026, 3, 17, 12, 0, 0),
        calibration={"energy": [0.0, 1.0]},
        detector_id="DET-1",
    )

    assert flags == []
