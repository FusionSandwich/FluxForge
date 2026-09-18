"""Independent count-area and timing contracts for the default CLI peak path."""

import argparse
import json

import numpy as np
import pytest

from fluxforge.analysis.segmented_detection import (
    _refine_centroid,
    detect_peaks_segmented,
)
from fluxforge.cli.app import cmd_activity_review
from fluxforge.io.artifacts import read_peak_report, write_peak_report


def test_segmented_area_is_raw_counts_independent_of_energy_gain():
    channels = np.arange(128)
    raw = 40 + 200 * np.exp(-0.5 * ((channels - 64) / 2) ** 2)
    expected_area = 200 * 2 * np.sqrt(2 * np.pi)
    results = []
    for gain in (0.25, 2.0):
        peaks = detect_peaks_segmented(
            channels, 10 + gain * channels, raw * 10, raw_counts=raw
        )
        assert len(peaks) == 1
        peak = peaks[0]
        assert peak.is_fitted
        assert peak.area == pytest.approx(expected_area, rel=1e-6)
        assert peak.sigma_keV == pytest.approx(2 * gain, rel=1e-6)
        assert peak.area_uncertainty > 0
        results.append(peak)
    assert results[0].area_uncertainty == pytest.approx(results[1].area_uncertainty)


def test_segmented_area_uncertainty_includes_amplitude_width_covariance(monkeypatch):
    from types import SimpleNamespace
    from fluxforge.analysis import segmented_detection

    covariance = np.diag([4.0, 0.01, 0.25, 1.0])
    covariance[0, 2] = covariance[2, 0] = -0.5
    fit = SimpleNamespace(
        success=True,
        covariance=covariance,
        peak=SimpleNamespace(
            centroid=8.0, amplitude=10.0, sigma=2.0, area=20 * np.sqrt(2 * np.pi)
        ),
    )
    monkeypatch.setattr(segmented_detection, "fit_single_peak", lambda *a, **k: fit)
    channels = np.arange(17)
    result = _refine_centroid(channels, channels * 0.5, np.ones(17), 8, 6)
    expected_variance = 2 * np.pi * (2**2 * 4 + 10**2 * 0.25 + 2 * 2 * 10 * -0.5)
    assert result[5] == pytest.approx(np.sqrt(expected_variance))


def test_peak_report_round_trips_elapsed_counting_time(tmp_path):
    path = tmp_path / "peaks.json"
    write_peak_report(
        path, spectrum_id="timed", live_time_s=80, real_time_s=100, peaks=[]
    )
    payload = read_peak_report(path)
    assert payload["real_time_s"] == 100
    assert payload["live_time_s"] == 80


def test_activity_review_rejects_raw_peak_height_as_area(tmp_path):
    path = tmp_path / "old-peaks.json"
    path.write_text(
        json.dumps(
            {
                "live_time_s": 80,
                "peaks": [
                    {"energy_keV": 100, "raw_counts": 50, "net_counts_uncertainty": 7}
                ],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        peaks_file=path, validate=False, live_time_s=None, energy_tolerance_keV=1
    )
    with pytest.raises(ValueError, match="no net-area counts"):
        cmd_activity_review(args)


@pytest.mark.parametrize("cooling", [None, 30.0])
def test_simple_activity_uses_signed_area_uncertainty_and_elapsed_decay(
    monkeypatch, cooling
):
    from fluxforge.cli import app

    payload = {
        "live_time_s": 80,
        "real_time_s": 100,
        "peaks": [{"energy_keV": 100, "area": -4.0, "area_uncertainty": 3.0}],
    }
    captured = {}
    monkeypatch.setattr(app, "read_peak_report", lambda _: payload)
    monkeypatch.setattr(
        app, "write_line_activities", lambda *a, **k: captured.update(k)
    )
    args = argparse.Namespace(
        peaks_file="unused",
        output="unused",
        validate=False,
        live_time_s=None,
        efficiency=0.2,
        emission_probability=0.5,
        half_life_s=100,
        cooling_time_s=cooling,
        isotope="Co-60",
        reaction_id=None,
    )
    app.cmd_activity(args)
    decay = np.log(2) / 100
    factor = np.exp(decay * (cooling or 0)) / (
        0.2 * 0.5 * 0.8 * (1 - np.exp(-decay * 100)) / decay
    )
    row = captured["lines"][0]
    assert row["activity_Bq"] == pytest.approx(-4 * factor)
    assert row["activity_unc_Bq"] == pytest.approx(3 * factor)
    assert row["activity_reference"] == (
        "count_start" if cooling is None else "end_of_irradiation"
    )


@pytest.mark.parametrize("reference", [None, "count_start"])
def test_rates_reject_activity_without_eoi_reference(monkeypatch, reference):
    from fluxforge.cli import app

    monkeypatch.setattr(
        app,
        "read_line_activities",
        lambda _: {"lines": [{"activity_Bq": 10, "activity_reference": reference}]},
    )
    args = argparse.Namespace(
        lines_file="unused", validate=False, segments_file=None, duration_s=100
    )
    with pytest.raises(ValueError, match="end_of_irradiation"):
        app.cmd_rates(args)
