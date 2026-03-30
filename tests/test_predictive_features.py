from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_dead_time_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.io.spe import GammaSpectrum
from fluxforge.standards.qa_monitor import QARecord


def _spectrum(total_counts: float, *, live_time: float, real_time: float, start_time: datetime):
    counts = np.asarray([total_counts / 4.0] * 4, dtype=float)
    return GammaSpectrum(
        counts=counts,
        channels=np.arange(counts.size, dtype=float),
        calibration={"energy": [0.0, 100.0]},
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        spectrum_id=f"spec-{total_counts}",
    )


def test_count_target_forecast_reports_eta_rate_and_uncertainty():
    anchor = datetime(2026, 3, 30, 9, 0)
    history = (
        _spectrum(1200.0, live_time=100.0, real_time=103.0, start_time=anchor),
        _spectrum(1600.0, live_time=100.0, real_time=106.0, start_time=anchor + timedelta(hours=1)),
        _spectrum(2000.0, live_time=100.0, real_time=110.0, start_time=anchor + timedelta(hours=2)),
    )

    forecast = estimate_count_target_forecast(
        history[-1],
        target_counts=3000.0,
        history_spectra=history,
    )

    assert forecast.current_counts == 2000.0
    assert forecast.count_rate_cps == 20.0
    assert forecast.eta_seconds == 50.0
    assert forecast.eta_uncertainty_seconds is not None
    assert forecast.eta_uncertainty_seconds > 0.0
    assert forecast.trend.slope > 0.0


def test_dead_time_forecast_projects_saturation_risk():
    anchor = datetime(2026, 3, 30, 9, 0)
    history = (
        _spectrum(1000.0, live_time=100.0, real_time=103.0, start_time=anchor),
        _spectrum(1100.0, live_time=100.0, real_time=108.0, start_time=anchor + timedelta(hours=1)),
        _spectrum(1200.0, live_time=100.0, real_time=114.0, start_time=anchor + timedelta(hours=2)),
    )

    forecast = estimate_dead_time_forecast(history, saturation_threshold=0.15)

    assert forecast.current_dead_time_fraction > 0.12
    assert forecast.projected_dead_time_fraction_1h > forecast.current_dead_time_fraction
    assert forecast.eta_to_saturation_seconds is not None
    assert forecast.status in {"amber", "red"}
    assert forecast.trend.slope > 0.0


def test_recalibration_forecast_uses_qa_history_slope():
    anchor = datetime(2026, 3, 1, 12, 0)
    history = (
        QARecord(
            timestamp=anchor,
            nuclide="Cs-137",
            energy_keV=661.66,
            measured_centroid_keV=661.64,
            measured_fwhm_keV=1.80,
            measured_fwhm_channels=2.3,
            net_counts=12000,
            efficiency=0.081,
            spectrum_file="qa-001.spe",
        ),
        QARecord(
            timestamp=anchor + timedelta(days=10),
            nuclide="Cs-137",
            energy_keV=661.66,
            measured_centroid_keV=662.14,
            measured_fwhm_keV=1.88,
            measured_fwhm_channels=2.4,
            net_counts=11800,
            efficiency=0.080,
            spectrum_file="qa-002.spe",
        ),
        QARecord(
            timestamp=anchor + timedelta(days=20),
            nuclide="Cs-137",
            energy_keV=661.66,
            measured_centroid_keV=662.44,
            measured_fwhm_keV=1.96,
            measured_fwhm_channels=2.5,
            net_counts=11750,
            efficiency=0.079,
            spectrum_file="qa-003.spe",
        ),
    )

    forecast = estimate_recalibration_forecast(history)

    assert forecast is not None
    assert forecast.nuclide == "Cs-137"
    assert forecast.predicted_recalibration_at is not None
    assert forecast.days_until_recalibration is not None
    assert 0.0 <= forecast.days_until_recalibration <= 30.0
    assert forecast.trigger_metric in {"centroid drift", "FWHM degradation"}
