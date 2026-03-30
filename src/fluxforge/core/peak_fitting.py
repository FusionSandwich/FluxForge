"""Phase 2 peak-fitting registry and ROI fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from fluxforge.plugins import PluginRegistries, bootstrap_builtin_registries

if TYPE_CHECKING:
    from fluxforge.analysis.peakfit import PeakFitResult


@dataclass(frozen=True)
class PeakFitterDefinition:
    """Registered peak-fitting method."""

    key: str
    label: str
    model_key: str
    background_models: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class InteractivePeakFitResult:
    """ROI-driven fit preview used by the redesigned calibration workspace."""

    fitter_key: str
    fitter_label: str
    background_model: str
    roi_bounds: tuple[float, float]
    channels: np.ndarray
    observed_counts: np.ndarray
    fit_counts: np.ndarray
    background_counts: np.ndarray
    peak_result: PeakFitResult

    @property
    def centroid_channel(self) -> float:
        return float(self.peak_result.peak.centroid)

    @property
    def fwhm_channels(self) -> float:
        return float(self.peak_result.peak.fwhm)

    @property
    def area_counts(self) -> float:
        return float(self.peak_result.net_counts)


def register_builtin_peak_fitters(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in Phase 2 Gaussian and skew fitters."""

    registries.peak_fitters.clear()
    registries.peak_fitters.register(
        "gaussian",
        PeakFitterDefinition(
            key="gaussian",
            label="Gaussian",
            model_key="gaussian",
            background_models=("linear", "constant", "step"),
            summary="Recommended symmetric Gaussian fit for routine HPGe peak work.",
        ),
        description="Recommended Gaussian ROI fitter with live background diagnostics.",
        recommended=True,
        standards_locked=True,
        tags=("phase2", "gaussian", "roi"),
        set_default=True,
    )
    registries.peak_fitters.register(
        "gaussian_skew",
        PeakFitterDefinition(
            key="gaussian_skew",
            label="Skewed Gaussian",
            model_key="gauss_dbl_exp",
            background_models=("linear", "constant"),
            summary="Gaussian core with asymmetric exponential tails for distorted peaks.",
        ),
        description="Optional skewed Gaussian ROI fitter with asymmetric tail terms for tailed or charge-collection-distorted peaks.",
        tags=("phase2", "skew", "roi"),
    )
    return registries


def peak_fitter_entries(registries: PluginRegistries | None = None):
    """Return the shared peak-fitter registry entries."""

    shared = bootstrap_builtin_registries(registries)
    if len(shared.peak_fitters) == 0:
        register_builtin_peak_fitters(shared)
    return shared.peak_fitters.entries()


def fit_roi_peak(
    channels: Sequence[float] | np.ndarray,
    counts: Sequence[float] | np.ndarray,
    roi_bounds: tuple[float, float],
    *,
    fitter_key: str = "gaussian",
    background_model: str = "linear",
    registries: PluginRegistries | None = None,
) -> InteractivePeakFitResult:
    """Fit the peak inside an ROI using a registered fitter."""

    from fluxforge.analysis.peakfit import fit_peak_poisson, fit_single_peak

    shared = bootstrap_builtin_registries(registries)
    if len(shared.peak_fitters) == 0:
        register_builtin_peak_fitters(shared)
    entry = shared.peak_fitters.get_entry(fitter_key)
    definition: PeakFitterDefinition = entry.implementation

    x = np.asarray(channels, dtype=float)
    y = np.asarray(counts, dtype=float)
    lower, upper = sorted((float(roi_bounds[0]), float(roi_bounds[1])))
    mask = (x >= lower) & (x <= upper)
    if np.count_nonzero(mask) < 5:
        raise ValueError("The ROI must span at least 5 channels for a stable fit.")

    x_roi = x[mask]
    y_roi = y[mask]
    fit_background = (
        background_model
        if background_model in definition.background_models
        else definition.background_models[0]
    )
    initial_centroid = float(x_roi[int(np.argmax(y_roi))])
    initial_sigma = max((upper - lower) / 6.0, 1.0)

    if definition.model_key == "gaussian":
        result = fit_single_peak(
            x,
            y,
            int(round(initial_centroid)),
            fit_width=max(int(round((upper - lower) / 2.0)), 3),
            background_model=fit_background,
        )
    else:
        result = fit_peak_poisson(
            x,
            y,
            initial_centroid=initial_centroid,
            initial_sigma=initial_sigma,
            model=definition.model_key,
            background=fit_background,
            fit_range=(lower, upper),
        )

    fit_channels = _fit_channels_from_result(x, result)
    fit_counts = _fit_counts_from_result(x, y, fit_channels, result)
    background_counts = np.asarray(result.background, dtype=float)
    if background_counts.shape != fit_counts.shape:
        background_counts = np.resize(background_counts, fit_counts.shape)

    return InteractivePeakFitResult(
        fitter_key=definition.key,
        fitter_label=definition.label,
        background_model=fit_background,
        roi_bounds=(lower, upper),
        channels=fit_channels,
        observed_counts=_observed_counts_from_channels(x, y, fit_channels),
        fit_counts=fit_counts,
        background_counts=background_counts,
        peak_result=result,
    )


def available_background_models(
    fitter_key: str,
    registries: PluginRegistries | None = None,
) -> tuple[str, ...]:
    """Return allowed background models for a fitter."""

    shared = bootstrap_builtin_registries(registries)
    if len(shared.peak_fitters) == 0:
        register_builtin_peak_fitters(shared)
    definition: PeakFitterDefinition = shared.peak_fitters.get(fitter_key)
    return definition.background_models


def _fit_channels_from_result(
    channels: np.ndarray,
    result: PeakFitResult,
) -> np.ndarray:
    mask = (channels >= result.fit_region[0]) & (channels <= result.fit_region[1])
    fit_channels = np.asarray(channels[mask], dtype=float)
    if fit_channels.size == result.residuals.size:
        return fit_channels
    start_index = int(np.argmin(np.abs(channels - result.fit_region[0])))
    return np.asarray(
        channels[start_index : start_index + result.residuals.size],
        dtype=float,
    )


def _observed_counts_from_channels(
    channels: np.ndarray,
    counts: np.ndarray,
    fit_channels: np.ndarray,
) -> np.ndarray:
    positions = [int(np.argmin(np.abs(channels - channel))) for channel in fit_channels]
    return np.asarray([counts[position] for position in positions], dtype=float)


def _fit_counts_from_result(
    channels: np.ndarray,
    counts: np.ndarray,
    fit_channels: np.ndarray,
    result: PeakFitResult,
) -> np.ndarray:
    observed = _observed_counts_from_channels(
        channels,
        np.asarray(counts, dtype=float),
        fit_channels,
    )
    if observed.shape != result.residuals.shape:
        observed = np.resize(observed, result.residuals.shape)
    return observed - np.asarray(result.residuals, dtype=float)


__all__ = [
    "InteractivePeakFitResult",
    "PeakFitterDefinition",
    "available_background_models",
    "fit_roi_peak",
    "peak_fitter_entries",
    "register_builtin_peak_fitters",
]


_shared_registries = bootstrap_builtin_registries()
if len(_shared_registries.peak_fitters) == 0:
    register_builtin_peak_fitters(_shared_registries)
