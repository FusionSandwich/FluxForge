"""Predictive analytics helpers for offline FluxForge workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from fluxforge.io.spe import GammaSpectrum
from fluxforge.standards.qa_monitor import QARecord


@dataclass(frozen=True)
class LinearTrendFit:
    """Simple linear trend summary with one-sigma slope uncertainty."""

    sample_count: int
    slope: float
    intercept: float
    slope_stderr: float
    r_squared: float


@dataclass(frozen=True)
class CountTargetForecast:
    """Forecast for time-to-target counts under the current ROI rate."""

    roi_bounds_keV: tuple[float, float] | None
    target_counts: float
    current_counts: float
    current_uncertainty: float
    count_rate_cps: float
    count_rate_uncertainty_cps: float
    eta_seconds: float | None
    eta_uncertainty_seconds: float | None
    trend: LinearTrendFit


@dataclass(frozen=True)
class DeadTimeForecast:
    """Trend-based dead-time forecast for the current spectrum history."""

    current_dead_time_fraction: float
    projected_dead_time_fraction_1h: float
    saturation_threshold: float
    eta_to_saturation_seconds: float | None
    status: str
    trend: LinearTrendFit


@dataclass(frozen=True)
class RecalibrationForecast:
    """QA-history forecast for the next recalibration trigger."""

    nuclide: str
    energy_keV: float
    current_fwhm_keV: float
    current_centroid_drift_keV: float
    fwhm_degradation_pct: float
    predicted_recalibration_at: datetime | None
    days_until_recalibration: float | None
    trigger_metric: str
    status: str
    fwhm_trend: LinearTrendFit
    drift_trend: LinearTrendFit


def _fit_linear_trend(xs: Sequence[float], ys: Sequence[float]) -> LinearTrendFit:
    x_values = np.asarray(tuple(xs), dtype=float)
    y_values = np.asarray(tuple(ys), dtype=float)
    sample_count = int(min(x_values.size, y_values.size))
    if sample_count <= 1:
        intercept = float(y_values[0]) if sample_count == 1 else 0.0
        return LinearTrendFit(
            sample_count=sample_count,
            slope=0.0,
            intercept=intercept,
            slope_stderr=0.0,
            r_squared=1.0 if sample_count == 1 else 0.0,
        )

    slope, intercept = np.polyfit(x_values, y_values, 1)
    fitted = slope * x_values + intercept
    residuals = y_values - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    if sample_count > 2:
        x_centered = x_values - np.mean(x_values)
        denom = float(np.sum(x_centered**2))
        slope_stderr = (
            float(np.sqrt((ss_res / max(sample_count - 2, 1)) / denom))
            if denom > 0.0
            else 0.0
        )
    else:
        slope_stderr = 0.0
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    return LinearTrendFit(
        sample_count=sample_count,
        slope=float(slope),
        intercept=float(intercept),
        slope_stderr=float(slope_stderr),
        r_squared=float(r_squared),
    )


def _spectrum_axis_hours(spectra: Sequence[GammaSpectrum]) -> np.ndarray:
    ordered = tuple(spectra)
    start_times = [spectrum.start_time for spectrum in ordered]
    if ordered and all(start_time is not None for start_time in start_times):
        anchor = min(start_times)  # type: ignore[arg-type]
        return np.asarray(
            [
                max(
                    0.0,
                    (start_time - anchor).total_seconds() / 3600.0,  # type: ignore[union-attr]
                )
                for start_time in start_times
            ],
            dtype=float,
        )
    return np.arange(len(ordered), dtype=float)


def estimate_count_target_forecast(
    spectrum: GammaSpectrum,
    *,
    roi_bounds_keV: tuple[float, float] | None = None,
    target_counts: float = 10000.0,
    history_spectra: Sequence[GammaSpectrum] = (),
) -> CountTargetForecast:
    """Estimate time remaining to reach the requested counts target."""

    if roi_bounds_keV is not None:
        current_counts, current_uncertainty = spectrum.counts_in_range(*roi_bounds_keV)
    else:
        current_counts = float(np.sum(np.asarray(spectrum.counts, dtype=float)))
        current_uncertainty = float(
            np.sqrt(np.sum(np.asarray(spectrum.counts_uncertainty, dtype=float) ** 2))
        )
    live_time = max(float(spectrum.live_time), 0.0)
    count_rate = current_counts / live_time if live_time > 0.0 else 0.0
    count_rate_uncertainty = current_uncertainty / live_time if live_time > 0.0 else 0.0
    remaining_counts = max(float(target_counts) - current_counts, 0.0)
    eta_seconds = (
        remaining_counts / count_rate if count_rate > 0.0 and remaining_counts > 0.0 else 0.0
    )
    if count_rate > 0.0 and remaining_counts > 0.0:
        eta_uncertainty = eta_seconds * (count_rate_uncertainty / count_rate)
    else:
        eta_uncertainty = 0.0

    trend_spectra = tuple(history_spectra) or (spectrum,)
    if roi_bounds_keV is not None:
        rates = [
            item.counts_in_range(*roi_bounds_keV)[0] / max(float(item.live_time), 1e-12)
            for item in trend_spectra
        ]
    else:
        rates = [item.count_rate for item in trend_spectra]
    trend = _fit_linear_trend(_spectrum_axis_hours(trend_spectra), rates)

    return CountTargetForecast(
        roi_bounds_keV=roi_bounds_keV,
        target_counts=float(target_counts),
        current_counts=float(current_counts),
        current_uncertainty=float(current_uncertainty),
        count_rate_cps=float(count_rate),
        count_rate_uncertainty_cps=float(count_rate_uncertainty),
        eta_seconds=None if count_rate <= 0.0 else float(eta_seconds),
        eta_uncertainty_seconds=None if count_rate <= 0.0 else float(eta_uncertainty),
        trend=trend,
    )


def estimate_dead_time_forecast(
    spectra: Sequence[GammaSpectrum],
    *,
    saturation_threshold: float = 0.15,
) -> DeadTimeForecast:
    """Project dead-time trend forward and warn when saturation is approaching."""

    ordered = tuple(spectra)
    if not ordered:
        trend = _fit_linear_trend((), ())
        return DeadTimeForecast(
            current_dead_time_fraction=0.0,
            projected_dead_time_fraction_1h=0.0,
            saturation_threshold=float(saturation_threshold),
            eta_to_saturation_seconds=None,
            status="green",
            trend=trend,
        )
    dead_times = [float(item.dead_time_fraction) for item in ordered]
    xs = _spectrum_axis_hours(ordered)
    trend = _fit_linear_trend(xs, dead_times)
    current_dead_time = dead_times[-1]
    projected = current_dead_time + trend.slope * 1.0
    eta_seconds = None
    if current_dead_time >= saturation_threshold:
        eta_seconds = 0.0
        status = "red"
    elif trend.slope > 0.0:
        hours_to_threshold = (saturation_threshold - current_dead_time) / trend.slope
        eta_seconds = max(hours_to_threshold, 0.0) * 3600.0
        if projected >= saturation_threshold:
            status = "red"
        elif projected >= saturation_threshold * 0.8:
            status = "amber"
        else:
            status = "green"
    else:
        status = "green"
    return DeadTimeForecast(
        current_dead_time_fraction=float(current_dead_time),
        projected_dead_time_fraction_1h=float(projected),
        saturation_threshold=float(saturation_threshold),
        eta_to_saturation_seconds=eta_seconds,
        status=status,
        trend=trend,
    )


def estimate_recalibration_forecast(
    qa_records: Sequence[QARecord],
    *,
    centroid_threshold_keV: float = 1.0,
    fwhm_degradation_threshold_pct: float = 20.0,
) -> RecalibrationForecast | None:
    """Forecast the next QA-driven recalibration trigger from history slopes."""

    grouped: dict[tuple[str, float], list[QARecord]] = {}
    for record in qa_records:
        grouped.setdefault((record.nuclide, record.energy_keV), []).append(record)
    if not grouped:
        return None

    best: RecalibrationForecast | None = None
    for (nuclide, energy_keV), series in grouped.items():
        ordered = sorted(series, key=lambda item: item.timestamp)
        baseline = ordered[0]
        latest = ordered[-1]
        xs = np.asarray(
            [
                max(0.0, (record.timestamp - baseline.timestamp).total_seconds() / 86400.0)
                for record in ordered
            ],
            dtype=float,
        )
        fwhm_values = np.asarray(
            [float(record.measured_fwhm_keV) for record in ordered],
            dtype=float,
        )
        drift_values = np.asarray(
            [
                abs(float(record.measured_centroid_keV) - float(baseline.measured_centroid_keV))
                for record in ordered
            ],
            dtype=float,
        )
        fwhm_trend = _fit_linear_trend(xs, fwhm_values)
        drift_trend = _fit_linear_trend(xs, drift_values)

        current_fwhm = float(latest.measured_fwhm_keV)
        current_drift = float(
            abs(latest.measured_centroid_keV - baseline.measured_centroid_keV)
        )
        baseline_fwhm = max(float(baseline.measured_fwhm_keV), 1e-12)
        fwhm_degradation_pct = 100.0 * (current_fwhm - baseline_fwhm) / baseline_fwhm
        fwhm_threshold_value = baseline_fwhm * (1.0 + fwhm_degradation_threshold_pct / 100.0)

        prediction_candidates: list[tuple[datetime | None, float | None, str, str]] = []
        if current_drift >= centroid_threshold_keV:
            prediction_candidates.append((latest.timestamp, 0.0, "centroid drift", "red"))
        elif drift_trend.slope > 0.0:
            days_until_drift = (centroid_threshold_keV - current_drift) / drift_trend.slope
            prediction_candidates.append(
                (
                    latest.timestamp + timedelta(days=max(days_until_drift, 0.0)),
                    max(days_until_drift, 0.0),
                    "centroid drift",
                    "amber" if days_until_drift <= 30.0 else "green",
                )
            )

        if fwhm_degradation_pct >= fwhm_degradation_threshold_pct:
            prediction_candidates.append((latest.timestamp, 0.0, "FWHM degradation", "red"))
        elif fwhm_trend.slope > 0.0:
            days_until_fwhm = (fwhm_threshold_value - current_fwhm) / fwhm_trend.slope
            prediction_candidates.append(
                (
                    latest.timestamp + timedelta(days=max(days_until_fwhm, 0.0)),
                    max(days_until_fwhm, 0.0),
                    "FWHM degradation",
                    "amber" if days_until_fwhm <= 30.0 else "green",
                )
            )

        if prediction_candidates:
            predicted_at, days_remaining, trigger_metric, status = min(
                prediction_candidates,
                key=lambda item: item[0] or datetime.max,
            )
        else:
            predicted_at = None
            days_remaining = None
            trigger_metric = "stable"
            status = "green"

        candidate = RecalibrationForecast(
            nuclide=nuclide,
            energy_keV=float(energy_keV),
            current_fwhm_keV=current_fwhm,
            current_centroid_drift_keV=current_drift,
            fwhm_degradation_pct=float(fwhm_degradation_pct),
            predicted_recalibration_at=predicted_at,
            days_until_recalibration=days_remaining,
            trigger_metric=trigger_metric,
            status=status,
            fwhm_trend=fwhm_trend,
            drift_trend=drift_trend,
        )
        if best is None:
            best = candidate
        elif candidate.predicted_recalibration_at is None:
            continue
        elif best.predicted_recalibration_at is None or (
            candidate.predicted_recalibration_at < best.predicted_recalibration_at
        ):
            best = candidate
    return best


__all__ = [
    "CountTargetForecast",
    "DeadTimeForecast",
    "LinearTrendFit",
    "RecalibrationForecast",
    "estimate_count_target_forecast",
    "estimate_dead_time_forecast",
    "estimate_recalibration_forecast",
]
