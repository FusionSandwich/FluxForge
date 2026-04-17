"""Analysis workspace helpers for the modern Qt shell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint, fit_efficiency_curve
from fluxforge.analysis.efficiency_models import semi_empirical_efficiency
from fluxforge.analysis.peak_finders import (
    PEAK_FINDER_METHODS,
    find_peaks_multi_method,
    get_peak_finder,
)
from fluxforge.analysis.peakfit import auto_find_peaks, estimate_background, fit_multiple_peaks
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.gamma_database import FLUXFORGE_GAMMA_DATA, GammaDatabase
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source
from fluxforge.io.spe import GammaSpectrum
from fluxforge.physics.activation import GammaLineMeasurement
from fluxforge.physics.decay_chain import DecayChain
from fluxforge.ml import MLPeakAnalysisEngine
from fluxforge.plugins import PluginRegistries, bootstrap_builtin_registries


@dataclass(frozen=True)
class PeakCandidate:
    """Auto-detected or manually curated spectrum peak."""

    peak_id: str
    channel: float
    energy_keV: float
    significance: float
    roi_bounds_keV: tuple[float, float]
    net_counts: float
    fit_quality: float
    status: str = "candidate"
    nuclide: str | None = None
    candidate_nuclides: tuple[str, ...] = ()
    reference_lines_keV: tuple[float, ...] = ()
    tags: tuple[str, ...] = ()
    normalized_residuals: tuple[float, ...] = ()
    residual_channels: tuple[float, ...] = ()


@dataclass(frozen=True)
class SpectralPhenomenonEstimate:
    """Estimated location of a common gamma-spectroscopy feature."""

    kind: str
    label: str
    energy_keV: float
    summary: str
    color: str = "#f59e0b"
    relative_height: float | None = None
    estimated_height_counts: float | None = None


@dataclass(frozen=True)
class PeakSearchMethodDefinition:
    """Registered peak-search metadata."""

    key: str
    label: str
    finder_key: str
    summary: str


@dataclass(frozen=True)
class ROIBackgroundMethodDefinition:
    """Registered ROI/background workflow metadata."""

    key: str
    label: str
    summary: str


@dataclass(frozen=True)
class EfficiencyModelDefinition:
    """Registered efficiency-model metadata."""

    key: str
    label: str
    degree: int | None = None
    form: str = "polynomial"
    summary: str = ""


@dataclass(frozen=True)
class EfficiencyCalibrationFitResult:
    """Efficiency fit payload used by the activity surfaces."""

    model_key: str
    model_label: str
    curve: EfficiencyCurve
    residuals: tuple[float, ...]
    rmse: float
    points_used: int


@dataclass(frozen=True)
class ActivityCalculationResult:
    """Activity calculation summary for one peak or nuclide line."""

    nuclide: str
    line_energy_keV: float
    activity_bq: float
    uncertainty_bq: float
    age_corrected_activity_bq: float
    mda_bq: float
    half_life_s: float
    source_age_s: float
    chain_summary: str
    age_corrected_uncertainty_bq: float = 0.0


@dataclass(frozen=True)
class ROIComponentFit:
    """One deconvolved peak component inside an ROI."""

    centroid_channel: float
    centroid_keV: float
    net_counts: float
    net_counts_uncertainty: float
    fwhm_channels: float
    reduced_chi_squared: float


@dataclass(frozen=True)
class ROIAnalysisResult:
    """Explicit ROI/background analysis payload shared by CLI and GUI."""

    label: str
    roi_bounds_keV: tuple[float, float]
    gross_counts: float
    gross_counts_uncertainty: float
    background_counts: float
    background_counts_uncertainty: float
    net_counts: float
    net_counts_uncertainty: float
    centroid_keV: float
    centroid_uncertainty_keV: float
    significance: float
    background_method: str
    peak_search_method: str
    sideband_bounds_keV: tuple[tuple[float, float], tuple[float, float]]
    overlap_components: tuple[ROIComponentFit, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ROISpectrumStatistic:
    """ROI result for one spectrum in a multi-spectrum statistics workflow."""

    label: str
    net_counts: float
    net_counts_uncertainty: float
    centroid_keV: float
    significance: float


@dataclass(frozen=True)
class ROIStatisticsResult:
    """Summary statistics for the same ROI across many spectra."""

    label: str
    roi_bounds_keV: tuple[float, float]
    sample_count: int
    mean_net_counts: float
    stdev_net_counts: float
    relative_std: float
    mean_centroid_keV: float
    stdev_centroid_keV: float
    min_net_counts: float
    max_net_counts: float
    samples: tuple[ROISpectrumStatistic, ...]


@dataclass(frozen=True)
class SurveyPoint:
    """GPS point extracted from a spectrum payload."""

    label: str
    latitude: float
    longitude: float
    source_role: str


@dataclass(frozen=True)
class BayesianNuclideMatchDefinition:
    """Registered nuclide-ID engine metadata."""

    key: str
    label: str
    summary: str


@dataclass(frozen=True)
class ContinuumDriverEstimate:
    """Estimated dominant continuum-driving line for spectral-feature overlays."""

    nuclide: str
    line_energy_keV: float
    score: float
    summary: str


def register_builtin_efficiency_models(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in efficiency models."""

    registries.calibration_models.clear()
    for key, label, degree, form, summary, recommended in (
        (
            "log_poly_2",
            "Log Polynomial (2)",
            2,
            "polynomial",
            "Recommended log-log polynomial curve for routine HPGe efficiency work.",
            True,
        ),
        (
            "log_poly_3",
            "Log Polynomial (3)",
            3,
            "polynomial",
            "Higher-order log polynomial for wider dynamic ranges.",
            False,
        ),
        (
            "gray_functional",
            "Gray Functional",
            3,
            "gray",
            "Gray-style functional form with log-energy terms and inverse-energy tail.",
            False,
        ),
        (
            "semi_empirical_hpge",
            "Semi-Empirical HPGe",
            None,
            "semi_empirical_hpge",
            "Semi-empirical detector-response model with window and dead-layer terms.",
            False,
        ),
    ):
        registries.calibration_models.register(
            key,
            EfficiencyModelDefinition(
                key=key,
                label=label,
                degree=degree,
                form=form,
                summary=summary,
            ),
            description=summary,
            recommended=recommended,
            tags=("analysis", "efficiency"),
            set_default=recommended,
        )
    return registries


def register_builtin_nuclide_id_engines(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in Bayesian library-matching engine."""

    registries.nuclide_id_engines.clear()
    registries.nuclide_id_engines.register(
        "bayesian_lines",
        BayesianNuclideMatchDefinition(
            key="bayesian_lines",
            label="Bayesian Line Matcher",
            summary=(
                "Ranks nuclides by line proximity and intensity-weighted evidence "
                "across the active peak set."
            ),
        ),
        description=(
            "Bayesian line matcher using peak-to-library proximity, intensity priors, "
            "and multi-line evidence accumulation."
        ),
        recommended=True,
        tags=("analysis", "id", "bayesian"),
        set_default=True,
    )
    registries.nuclide_id_engines.register(
        "ml_peak_onnx",
        MLPeakAnalysisEngine(),
        description=MLPeakAnalysisEngine.summary,
        tags=("analysis", "id", "ml", "onnx"),
    )
    return registries


def register_builtin_peak_search_methods(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in peak-search methods used by the modern GUI."""

    registries.peak_search_methods.clear()
    for key, label, finder_key, summary, recommended in (
        (
            "mariscotti",
            "Mariscotti",
            "second_difference",
            "Second-difference peak search tuned for explicit ROI review and classic GSA-style workflows.",
            True,
        ),
        (
            "segmented",
            "Segmented",
            "segmented",
            "Region-aware peak search with optional Gaussian refinement inspired by workstation-style review flows.",
            False,
        ),
        (
            "window",
            "Window Statistics",
            "window",
            "Local-window thresholding for reference-style peak search against changing continua.",
            False,
        ),
        (
            "simple",
            "Simple Threshold",
            "simple",
            "SNIP-backed threshold search for quick first-pass peak discovery.",
            False,
        ),
        (
            "chunked",
            "Chunked",
            "chunked",
            "Chunk-wise adaptive threshold search for spectra with region-varying backgrounds.",
            False,
        ),
        (
            "second_difference",
            "Second Difference",
            "second_difference",
            "FluxForge second-difference search for sharp peaks against smooth continua.",
            False,
        ),
        (
            "derivative",
            "Derivative",
            "derivative",
            "First- and second-derivative zero-crossing search for narrow local maxima.",
            False,
        ),
        (
            "scipy",
            "SciPy Smoothed",
            "scipy",
            "Savitzky-Golay plus `scipy.signal.find_peaks` for stable interactive searching.",
            False,
        ),
        (
            "nasa_peaksearch",
            "NASA Peak Search",
            "scipy",
            "Smoothed SciPy peak search inspired by modular NASA-gamma analysis flows.",
            False,
        ),
        (
            "direct_scipy",
            "Direct SciPy",
            "direct_scipy",
            "Direct `scipy.signal.find_peaks` wrapper for analysts who want raw control.",
            False,
        ),
        (
            "wavelet",
            "Wavelet",
            "wavelet",
            "Continuous-wavelet peak search for broad and narrow peaks in one pass.",
            False,
        ),
        (
            "relative_extrema",
            "Relative Extrema",
            "relative_extrema",
            "Argrelextrema-based local-maxima finder for exploratory review.",
            False,
        ),
        (
            "consensus",
            "Consensus",
            "consensus",
            "Multi-method consensus search requiring agreement between multiple finders.",
            False,
        ),
    ):
        registries.peak_search_methods.register(
            key,
            PeakSearchMethodDefinition(
                key=key,
                label=label,
                finder_key=finder_key,
                summary=summary,
            ),
            description=summary,
            recommended=recommended,
            tags=("analysis", "peaks", "roi"),
            set_default=recommended,
        )
    return registries


def register_builtin_roi_background_methods(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in ROI/background workflows used by 3.16."""

    registries.roi_background_models.clear()
    for key, label, summary, recommended in (
        (
            "roi_sideband",
            "ROI Sideband",
            "Explicit sideband background estimate using analyst-visible left/right continuum windows.",
            True,
        ),
        (
            "snip",
            "SNIP",
            "Statistics-sensitive nonlinear peak clipping for continuum estimation beneath the ROI.",
            False,
        ),
        (
            "linear_minima",
            "Linear Minima",
            "Linear-minima continuum estimate across the ROI while excluding the current peak bounds.",
            False,
        ),
    ):
        registries.roi_background_models.register(
            key,
            ROIBackgroundMethodDefinition(
                key=key,
                label=label,
                summary=summary,
            ),
            description=summary,
            recommended=recommended,
            tags=("analysis", "background", "roi"),
            set_default=recommended,
        )
    return registries


def apply_ml_peak_predictions(
    peaks: Sequence[PeakCandidate],
    predictions: Sequence[object],
    *,
    confidence_threshold: float = 0.45,
) -> tuple[PeakCandidate, ...]:
    """Overlay ML proposal metadata onto the peak table payload."""

    by_peak = {
        str(getattr(prediction, "peak_id")): prediction for prediction in predictions
    }
    updated: list[PeakCandidate] = []
    for peak in peaks:
        prediction = by_peak.get(peak.peak_id)
        if prediction is None:
            updated.append(peak)
            continue
        predicted_nuclide = str(getattr(prediction, "predicted_nuclide", "")).strip()
        confidence = float(getattr(prediction, "confidence", 0.0))
        uncertainty_keV = float(getattr(prediction, "uncertainty_keV", 0.0))
        tags = list(peak.tags)
        tags.append(f"ml:{confidence:.2f}")
        updated.append(
            PeakCandidate(
                peak_id=peak.peak_id,
                channel=peak.channel,
                energy_keV=peak.energy_keV,
                significance=peak.significance,
                roi_bounds_keV=peak.roi_bounds_keV,
                net_counts=peak.net_counts,
                fit_quality=peak.fit_quality,
                status=(
                    "manual"
                    if peak.status == "manual"
                    else "review" if confidence < confidence_threshold else "matched"
                ),
                nuclide=predicted_nuclide or peak.nuclide,
                candidate_nuclides=(
                    (predicted_nuclide,) + tuple(item for item in peak.candidate_nuclides if item != predicted_nuclide)
                    if predicted_nuclide
                    else peak.candidate_nuclides
                ),
                reference_lines_keV=peak.reference_lines_keV,
                tags=tuple(tags),
                normalized_residuals=peak.normalized_residuals,
                residual_channels=peak.residual_channels,
            )
        )
    return tuple(updated)


def estimate_spectral_phenomena(
    photopeak_energy_keV: float,
    *,
    detector_material: str = "hpge",
    spectrum: GammaSpectrum | None = None,
) -> tuple[SpectralPhenomenonEstimate, ...]:
    """Estimate common gamma-spectroscopy artifacts near a selected line."""

    energy = float(photopeak_energy_keV)
    if energy <= 0.0:
        return ()

    electron_rest_keV = 511.0
    phenomena: list[SpectralPhenomenonEstimate] = []

    compton_edge = energy * (1.0 - 1.0 / (1.0 + (2.0 * energy / electron_rest_keV)))
    backscatter_peak = energy / (1.0 + (2.0 * energy / electron_rest_keV))
    phenomena.append(
        SpectralPhenomenonEstimate(
            kind="compton_edge",
            label="Compton Edge",
            energy_keV=compton_edge,
            summary=(
                "Upper bound of the single-scatter Compton continuum for the selected line."
            ),
            color="#f59e0b",
            relative_height=0.28,
        )
    )
    phenomena.append(
        SpectralPhenomenonEstimate(
            kind="backscatter",
            label="Backscatter Peak",
            energy_keV=backscatter_peak,
            summary="Expected backscatter feature from 180° scattering before detector absorption.",
            color="#f97316",
            relative_height=0.22,
        )
    )

    if detector_material.lower() == "hpge":
        for label, escape_energy in (
            ("Ge Kα Escape", energy - 9.87),
            ("Ge Kβ Escape", energy - 10.98),
        ):
            if escape_energy > 0.0:
                phenomena.append(
                    SpectralPhenomenonEstimate(
                        kind="detector_escape",
                        label=label,
                        energy_keV=escape_energy,
                        summary="Detector escape feature estimated for an HPGe crystal.",
                        color="#22c55e",
                        relative_height=0.16,
                    )
                )

    if energy > 2.0 * electron_rest_keV:
        single_escape = energy - electron_rest_keV
        double_escape = energy - (2.0 * electron_rest_keV)
        if single_escape > 0.0:
            phenomena.append(
                SpectralPhenomenonEstimate(
                    kind="single_escape",
                    label="Single Escape",
                    energy_keV=single_escape,
                    summary="Pair-production escape peak after one annihilation photon leaves the detector.",
                    color="#10b981",
                    relative_height=0.2,
                )
            )
        if double_escape > 0.0:
            phenomena.append(
                SpectralPhenomenonEstimate(
                    kind="double_escape",
                    label="Double Escape",
                    energy_keV=double_escape,
                    summary="Pair-production escape peak after both annihilation photons leave the detector.",
                    color="#059669",
                    relative_height=0.14,
                )
            )
        phenomena.append(
            SpectralPhenomenonEstimate(
                kind="annihilation",
                label="Annihilation Line",
                energy_keV=electron_rest_keV,
                summary="511 keV annihilation feature expected when pair production contributes.",
                color="#38bdf8",
                relative_height=0.1,
            )
        )

    if spectrum is not None and len(spectrum.counts):
        energies = np.asarray(spectrum.energies, dtype=float)
        counts = np.asarray(spectrum.counts, dtype=float)
        peak_index = int(np.argmin(np.abs(energies - energy))) if len(energies) else 0
        peak_height = float(counts[peak_index]) if len(counts) else 0.0
        enriched: list[SpectralPhenomenonEstimate] = []
        for item in phenomena:
            rel = item.relative_height if item.relative_height is not None else 0.15
            est_height = max(0.0, peak_height * float(rel))
            enriched.append(
                SpectralPhenomenonEstimate(
                    kind=item.kind,
                    label=item.label,
                    energy_keV=item.energy_keV,
                    summary=item.summary,
                    color=item.color,
                    relative_height=item.relative_height,
                    estimated_height_counts=est_height,
                )
            )
        phenomena = enriched

    return tuple(sorted(phenomena, key=lambda item: (item.energy_keV, item.label)))


def estimate_dominant_continuum_driver(
    spectrum: GammaSpectrum,
    *,
    tolerance_keV: float = 2.0,
    min_intensity: float = 0.02,
    source_id: str = "fluxforge_bundled_gamma",
    custom_path: str | None = None,
) -> ContinuumDriverEstimate | None:
    """Estimate the most likely continuum-driving photopeak from loaded spectrum data."""

    del min_intensity
    peaks = detect_peak_candidates(spectrum)
    if not peaks:
        return None
    matched = bayesian_match_peak_candidates(
        peaks,
        source_id=source_id,
        custom_path=custom_path,
        tolerance_keV=tolerance_keV,
    )
    candidates = [peak for peak in matched if peak.nuclide]
    if not candidates:
        return None

    lead = max(candidates, key=lambda peak: float(peak.significance))
    line_energy_keV = float(lead.energy_keV)
    if lead.reference_lines_keV:
        line_energy_keV = float(
            min(
                lead.reference_lines_keV,
                key=lambda energy: abs(float(energy) - float(lead.energy_keV)),
            )
        )
    delta_keV = abs(line_energy_keV - float(lead.energy_keV))
    score = float(lead.significance) / max(1.0, 1.0 + delta_keV)
    nuclide = str(lead.nuclide or "unknown")
    return ContinuumDriverEstimate(
        nuclide=nuclide,
        line_energy_keV=line_energy_keV,
        score=score,
        summary=(
            f"{nuclide} near {line_energy_keV:.3f} keV "
            f"(peak {lead.energy_keV:.3f} keV, significance {lead.significance:.3f})"
        ),
    )


def detect_peak_candidates(
    spectrum: GammaSpectrum,
    *,
    method: str = "mariscotti",
    threshold: float = 4.0,
    min_distance: int = 18,
    max_peaks: int = 12,
    registries: PluginRegistries | None = None,
) -> tuple[PeakCandidate, ...]:
    """Detect candidate peaks and attach lightweight fit diagnostics."""

    from fluxforge.core.calibration import estimate_local_fwhm_channels
    from fluxforge.core.peak_fitting import fit_roi_peak

    counts = np.asarray(spectrum.counts, dtype=float)
    channels = np.asarray(spectrum.channels, dtype=float)
    if counts.size == 0:
        return ()
    if channels.size == 0:
        channels = np.arange(len(counts), dtype=float)

    peaks = _detect_peak_method_results(
        counts,
        channels=channels,
        method=method,
        threshold=threshold,
        min_distance=min_distance,
        registries=registries,
    )
    ordered = sorted(peaks, key=lambda item: item[1], reverse=True)[:max_peaks]

    candidates: list[PeakCandidate] = []
    for index, (channel, significance) in enumerate(sorted(ordered, key=lambda item: item[0])):
        half_width = max(
            estimate_local_fwhm_channels(counts, int(round(channel))) * 3.0,
            10.0,
        )
        roi_bounds_ch = (
            max(float(channel) - half_width, 0.0),
            min(float(channel) + half_width, float(len(counts) - 1)),
        )
        fit = fit_roi_peak(channels, counts, roi_bounds_ch)
        roi_bounds_keV = (
            float(spectrum.channel_to_energy(roi_bounds_ch[0])),
            float(spectrum.channel_to_energy(roi_bounds_ch[1])),
        )
        normalized_residuals = (
            fit.observed_counts - fit.fit_counts
        ) / np.sqrt(np.clip(fit.fit_counts, 1.0, None))
        energy_keV = float(spectrum.channel_to_energy(fit.centroid_channel))
        candidates.append(
            PeakCandidate(
                peak_id=f"peak-{index + 1}",
                channel=float(fit.centroid_channel),
                energy_keV=energy_keV,
                significance=float(significance),
                roi_bounds_keV=tuple(sorted(roi_bounds_keV)),
                net_counts=float(fit.area_counts),
                fit_quality=float(fit.peak_result.reduced_chi_squared),
                normalized_residuals=tuple(float(value) for value in normalized_residuals),
                residual_channels=tuple(float(value) for value in fit.channels),
            )
        )
    return tuple(candidates)


def _peak_search_definition(
    method: str,
    *,
    registries: PluginRegistries | None = None,
) -> PeakSearchMethodDefinition:
    shared = bootstrap_builtin_registries(registries)
    if len(shared.peak_search_methods) == 0:
        register_builtin_peak_search_methods(shared)
    return shared.peak_search_methods.get(method)


def _detect_peak_method_results(
    counts: np.ndarray,
    *,
    channels: np.ndarray,
    method: str,
    threshold: float,
    min_distance: int,
    registries: PluginRegistries | None = None,
) -> list[tuple[float, float]]:
    definition = _peak_search_definition(method, registries=registries)
    count_array = np.asarray(counts, dtype=float)
    finder_kwargs: dict[str, object] = {
        "threshold_sigma": max(float(threshold), 0.5),
        "min_distance": max(int(min_distance), 1),
    }
    if definition.key in {"nasa_peaksearch", "scipy"}:
        finder_kwargs = {
            "threshold_factor": max(float(threshold) / 3.0, 1.0),
            "smooth_window": min(max(int(min_distance) * 5, 11), 101),
            "distance": max(int(min_distance), 1),
            "prominence": None,
        }
    elif definition.key == "mariscotti":
        finder_kwargs["threshold_sigma"] = max(float(threshold) * 0.9, 3.0)
    elif definition.key == "window":
        finder_kwargs = {
            "threshold_sigma": max(float(threshold), 0.5),
            "window_size": max(int(min_distance) * 4, 32),
            "min_distance": max(int(min_distance), 1),
        }
    elif definition.key == "chunked":
        finder_kwargs = {
            "threshold": max(float(threshold), 0.5),
            "n_chunks": max(min(int(len(count_array) / 512), 16), 4),
        }
    elif definition.key == "segmented":
        finder_kwargs = {"gaussian_refine": True}
    elif definition.key == "direct_scipy":
        finder_kwargs = {
            "distance": max(int(min_distance), 1),
            "height": max(float(np.median(count_array) + np.std(count_array)), 1.0),
        }
    elif definition.key == "wavelet":
        width_max = max(int(min_distance // 2), 4)
        finder_kwargs = {
            "widths": np.arange(1, width_max + 1, dtype=int),
            "min_snr": max(float(threshold) / 3.0, 1.0),
        }
    elif definition.key == "relative_extrema":
        finder_kwargs = {"order": max(int(min_distance // 2), 2)}

    if definition.key == "consensus":
        found = find_peaks_multi_method(
            count_array,
            methods=[
                method_name
                for method_name in ("window", "scipy", "second_difference", "segmented")
                if method_name in PEAK_FINDER_METHODS
            ],
            consensus_threshold=2,
        )
    else:
        finder = get_peak_finder(definition.finder_key, **finder_kwargs)
        found = finder.find_peaks(count_array)
    resolved: list[tuple[float, float]] = []
    for peak in found:
        if getattr(peak, "centroid", None) is not None:
            centroid = float(np.interp(float(peak.centroid), np.arange(len(channels), dtype=float), channels))
        else:
            index = int(np.clip(int(round(float(peak.index))), 0, len(channels) - 1))
            centroid = float(channels[index])
        significance = float(getattr(peak, "significance", 0.0) or 0.0)
        resolved.append((centroid, significance))

    if resolved:
        return resolved

    fallback = auto_find_peaks(
        channels,
        counts,
        threshold=threshold,
        min_distance=min_distance,
    )
    return [(float(channel), float(significance)) for channel, significance in fallback]


def bayesian_match_peak_candidates(
    peaks: Sequence[PeakCandidate],
    *,
    source_id: str = "fluxforge_bundled_gamma",
    custom_path: str | None = None,
    tolerance_keV: float = 2.0,
    overlay_limit: int = 4,
) -> tuple[PeakCandidate, ...]:
    """Assign likely nuclides to the detected peaks using a Bayesian-style score."""

    database = _load_gamma_database(source_id, custom_path=custom_path)
    evidence = _score_database_lines(peaks, database, tolerance_keV=tolerance_keV)
    if not evidence:
        return tuple(peaks)

    ranked_nuclides = [
        nuclide
        for nuclide, score in sorted(
            evidence.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score > 0.0
    ]

    updated: list[PeakCandidate] = []
    for peak in peaks:
        peak_scores: list[tuple[str, float, tuple[float, ...]]] = []
        for nuclide in ranked_nuclides:
            decay = database.get(nuclide)
            if decay is None:
                continue
            strong_lines = tuple(
                round(line.energy_keV, 3)
                for line in decay.strongest_gamma_lines(n=max(overlay_limit, 6))
            )
            if not strong_lines:
                continue
            nearest = min(strong_lines, key=lambda energy: abs(energy - peak.energy_keV))
            delta = abs(nearest - peak.energy_keV)
            if delta > tolerance_keV:
                continue
            score = evidence[nuclide] * math.exp(-0.5 * (delta / max(tolerance_keV, 1e-6)) ** 2)
            peak_scores.append((nuclide, score, strong_lines[:overlay_limit]))

        peak_scores.sort(key=lambda item: item[1], reverse=True)
        if peak_scores:
            best_nuclide, _score, lines = peak_scores[0]
            updated.append(
                PeakCandidate(
                    peak_id=peak.peak_id,
                    channel=peak.channel,
                    energy_keV=peak.energy_keV,
                    significance=peak.significance,
                    roi_bounds_keV=peak.roi_bounds_keV,
                    net_counts=peak.net_counts,
                    fit_quality=peak.fit_quality,
                    status="matched",
                    nuclide=best_nuclide,
                    candidate_nuclides=tuple(item[0] for item in peak_scores[:3]),
                    reference_lines_keV=lines,
                    tags=peak.tags,
                    normalized_residuals=peak.normalized_residuals,
                    residual_channels=peak.residual_channels,
                )
            )
            continue
        updated.append(peak)
    return tuple(updated)


def fit_efficiency_model(
    points: Sequence[EfficiencyPoint],
    *,
    model_key: str = "log_poly_2",
    registries: PluginRegistries | None = None,
) -> EfficiencyCalibrationFitResult:
    """Fit one of the registered efficiency models."""

    shared = bootstrap_builtin_registries(registries)
    if len(shared.calibration_models) == 0:
        register_builtin_efficiency_models(shared)
    definition: EfficiencyModelDefinition = shared.calibration_models.get(model_key)

    if definition.form == "semi_empirical_hpge":
        energies = np.asarray([point.energy_keV for point in points], dtype=float)
        values = np.asarray([point.efficiency()[0] for point in points], dtype=float)
        coefficients = _fit_semi_empirical_coefficients(energies, values)
        curve = EfficiencyCurve(
            model_type="functional",
            parameters={
                "form": "semi_empirical_hpge",
                "coefficients": coefficients,
            },
            energy_range=(float(np.min(energies)), float(np.max(energies))),
        )
        residuals = tuple(float(obs - pred) for obs, pred in zip(values, curve.efficiency(energies)))
        rmse = math.sqrt(float(np.mean(np.square(residuals)))) if residuals else 0.0
        return EfficiencyCalibrationFitResult(
            model_key=definition.key,
            model_label=definition.label,
            curve=curve,
            residuals=residuals,
            rmse=float(rmse),
            points_used=len(points),
        )

    degree = definition.degree if definition.degree is not None else 2
    fit = fit_efficiency_curve(points, degree=degree)
    if definition.form == "gray":
        coefficients = list(fit.coefficients)
        while len(coefficients) < 4:
            coefficients.append(0.0)
        curve = EfficiencyCurve(
            model_type="functional",
            parameters={
                "form": "gray",
                "a": coefficients[0],
                "b": coefficients[1],
                "c": coefficients[2],
                "d": coefficients[3],
            },
            energy_range=fit.curve.energy_range,
        )
        predicted = np.asarray(curve.efficiency([point.energy_keV for point in points]), dtype=float)
        measured = np.asarray([point.efficiency()[0] for point in points], dtype=float)
        residuals = tuple(float(obs - pred) for obs, pred in zip(measured, predicted))
        rmse = math.sqrt(float(np.mean(np.square(residuals)))) if residuals else 0.0
        return EfficiencyCalibrationFitResult(
            model_key=definition.key,
            model_label=definition.label,
            curve=curve,
            residuals=residuals,
            rmse=float(rmse),
            points_used=len(points),
        )

    rmse = math.sqrt(float(np.mean(np.square(fit.residuals)))) if len(fit.residuals) else 0.0
    return EfficiencyCalibrationFitResult(
        model_key=definition.key,
        model_label=definition.label,
        curve=fit.curve,
        residuals=tuple(float(value) for value in fit.residuals),
        rmse=float(rmse),
        points_used=len(points),
    )


def calculate_peak_activity(
    peak: PeakCandidate,
    spectrum: GammaSpectrum,
    *,
    efficiency_curve: EfficiencyCurve,
    gamma_intensity: float,
    half_life_s: float,
    source_age_s: float = 0.0,
    dead_time_fraction: float | None = None,
) -> ActivityCalculationResult:
    """Calculate activity, MDA, and age-corrected activity for one peak."""

    measurement = GammaLineMeasurement(
        net_counts=float(peak.net_counts),
        live_time_s=max(float(spectrum.live_time or 1.0), 1e-6),
        efficiency=float(
            np.asarray(efficiency_curve.efficiency(peak.energy_keV), dtype=float).reshape(-1)[0]
        ),
        gamma_intensity=max(float(gamma_intensity), 1e-6),
        half_life_s=max(float(half_life_s), 1e-6),
        dead_time_fraction=float(
            dead_time_fraction
            if dead_time_fraction is not None
            else getattr(spectrum, "dead_time_fraction", 0.0)
        ),
    )
    activity_bq = measurement.activity_at_reference()
    uncertainty_bq = activity_bq / math.sqrt(max(peak.net_counts, 1.0))
    age_corrected_activity = activity_bq * math.exp(
        math.log(2.0) * source_age_s / max(half_life_s, 1e-6)
    )

    background_counts = max(
        float(np.mean(spectrum.counts)) * max(peak.roi_bounds_keV[1] - peak.roi_bounds_keV[0], 1.0),
        1.0,
    )
    detection_limit_counts = 2.71 + 4.65 * math.sqrt(background_counts)
    mda_bq = detection_limit_counts / max(
        measurement.efficiency * measurement.gamma_intensity * measurement.live_time_s,
        1e-12,
    )

    chain_summary = build_decay_chain_summary(
        peak.nuclide or "unknown",
        half_life_s=half_life_s,
        source_age_s=source_age_s,
    )
    return ActivityCalculationResult(
        nuclide=peak.nuclide or "Unassigned",
        line_energy_keV=float(peak.energy_keV),
        activity_bq=float(activity_bq),
        uncertainty_bq=float(uncertainty_bq),
        age_corrected_activity_bq=float(age_corrected_activity),
        age_corrected_uncertainty_bq=float(
            uncertainty_bq
            * math.exp(math.log(2.0) * source_age_s / max(half_life_s, 1e-6))
        ),
        mda_bq=float(mda_bq),
        half_life_s=float(half_life_s),
        source_age_s=float(source_age_s),
        chain_summary=chain_summary,
    )


def build_decay_chain_summary(
    nuclide: str,
    *,
    half_life_s: float,
    source_age_s: float,
) -> str:
    """Return a short Bateman-based decay summary for the activity panel."""

    daughter = f"{nuclide} daughter"
    chain = DecayChain(
        nuclide,
        nuclide_data={
            nuclide: {"half_life_s": float(half_life_s), "decay_products": {daughter: 1.0}},
            daughter: {"half_life_s": float("inf"), "decay_products": {}},
        },
    )
    result = chain.decay(initial_atoms={nuclide: 1.0}, times=[0.0, float(source_age_s)])
    remaining = float(result.atoms[nuclide][-1])
    daughter_fraction = float(result.atoms[daughter][-1])
    return (
        f"Bateman correction over {source_age_s / 3600.0:.2f} h: "
        f"{nuclide} retains {remaining:.4f} of its EOI inventory and transfers {daughter_fraction:.4f} to the daughter path."
    )


def subtract_background_counts(
    foreground: GammaSpectrum,
    background: GammaSpectrum | None,
    *,
    mode: str = "simple",
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract background counts using the requested analysis mode."""

    foreground_counts = np.asarray(foreground.counts, dtype=float)
    if background is None:
        return foreground_counts.copy()

    background_counts = np.asarray(background.counts, dtype=float)
    if background_counts.shape != foreground_counts.shape:
        background_counts = np.resize(background_counts, foreground_counts.shape)

    if mode == "simple":
        factor = 1.0
    elif mode == "scaled":
        factor = float(scale)
    elif mode == "statistical":
        factor = (
            max(float(foreground.live_time or 1.0), 1.0)
            / max(float(background.live_time or 1.0), 1.0)
        )
    else:
        raise ValueError(f"Unsupported background-subtraction mode: {mode}")
    return np.clip(foreground_counts - factor * background_counts, 0.0, None)


def background_adjusted_spectrum(
    foreground: GammaSpectrum,
    background: GammaSpectrum | None,
    *,
    mode: str = "simple",
    scale: float = 1.0,
) -> GammaSpectrum:
    """Return a spectrum copy with the selected external background workflow applied."""

    adjusted = subtract_background_counts(
        foreground,
        background,
        mode=mode,
        scale=scale,
    )
    factor = _background_subtraction_factor(
        foreground,
        background,
        mode=mode,
        scale=scale,
    )
    foreground_unc = np.asarray(foreground.counts_uncertainty, dtype=float)
    if background is None:
        adjusted_unc = foreground_unc.copy()
    else:
        background_unc = np.asarray(background.counts_uncertainty, dtype=float)
        if background_unc.shape != foreground_unc.shape:
            background_unc = np.resize(background_unc, foreground_unc.shape)
        adjusted_unc = np.sqrt(np.maximum(foreground_unc**2 + (factor * background_unc) ** 2, 0.0))
    metadata = dict(getattr(foreground, "metadata", {}) or {})
    metadata["background_subtraction"] = {
        "mode": mode,
        "scale": float(factor),
        "source": background.spectrum_id if background is not None else None,
    }
    return GammaSpectrum(
        counts=np.asarray(adjusted, dtype=float),
        counts_uncertainty=np.asarray(adjusted_unc, dtype=float),
        channels=np.asarray(foreground.channels, dtype=float),
        energies=(
            np.asarray(foreground.energies, dtype=float)
            if foreground.energies is not None
            else None
        ),
        live_time=float(foreground.live_time),
        real_time=float(foreground.real_time),
        start_time=foreground.start_time,
        spectrum_id=foreground.spectrum_id,
        detector_id=foreground.detector_id,
        calibration=dict(foreground.calibration),
        source_type=foreground.source_type,
        device_id=foreground.device_id,
        device_label=foreground.device_label,
        gps=dict(getattr(foreground, "gps", {}) or {}),
        metadata=metadata,
    )


def analyze_roi_region(
    spectrum: GammaSpectrum,
    *,
    roi_bounds_keV: tuple[float, float],
    label: str = "ROI",
    background_method: str = "roi_sideband",
    peak_search_method: str = "mariscotti",
    sideband_width_keV: float | None = None,
    background_spectrum: GammaSpectrum | None = None,
    background_mode: str = "simple",
    background_scale: float = 1.0,
    decompose_overlaps: bool = False,
    max_components: int = 3,
    registries: PluginRegistries | None = None,
) -> ROIAnalysisResult:
    """Compute explicit ROI/background metrics for one spectrum."""

    shared = bootstrap_builtin_registries(registries)
    if len(shared.roi_background_models) == 0:
        register_builtin_roi_background_methods(shared)
    if len(shared.peak_search_methods) == 0:
        register_builtin_peak_search_methods(shared)
    shared.roi_background_models.get(background_method)
    _peak_search_definition(peak_search_method, registries=shared)

    working = background_adjusted_spectrum(
        spectrum,
        background_spectrum,
        mode=background_mode,
        scale=background_scale,
    )
    lo_keV, hi_keV = sorted((float(roi_bounds_keV[0]), float(roi_bounds_keV[1])))
    roi_lo_ch, roi_hi_ch = _roi_channel_bounds(working, (lo_keV, hi_keV))
    roi_mask = (working.channels >= roi_lo_ch) & (working.channels <= roi_hi_ch)
    if np.count_nonzero(roi_mask) == 0:
        raise ValueError("ROI bounds do not overlap the active spectrum.")

    counts = np.asarray(working.counts, dtype=float)
    counts_unc = np.asarray(working.counts_uncertainty, dtype=float)
    gross_counts = float(np.sum(counts[roi_mask]))
    gross_unc = float(np.sqrt(np.sum(np.square(counts_unc[roi_mask]))))

    background_curve, sideband_bounds = _estimate_roi_background_curve(
        working,
        roi_bounds_keV=(lo_keV, hi_keV),
        background_method=background_method,
        sideband_width_keV=sideband_width_keV,
    )
    background_counts = float(np.sum(background_curve[roi_mask]))
    background_unc = float(np.sqrt(np.sum(np.clip(background_curve[roi_mask], 0.0, None))))
    net_curve = counts - background_curve
    net_counts = float(np.sum(net_curve[roi_mask]))
    net_unc = float(np.sqrt(max(gross_unc**2 + background_unc**2, 0.0)))
    centroid_keV, centroid_unc_keV = _weighted_centroid_keV(
        working,
        net_curve,
        roi_mask,
    )
    significance = float(
        net_counts / max(net_unc, 1.0e-12)
        if net_unc > 0.0
        else 0.0
    )

    notes: list[str] = []
    if background_spectrum is not None:
        notes.append(
            f"External background slot applied with {background_mode} normalization."
        )
    notes.append(f"ROI continuum estimated with {background_method}.")

    overlap_components: tuple[ROIComponentFit, ...] = ()
    if decompose_overlaps:
        overlap_components = _fit_roi_overlap_components(
            working,
            roi_bounds_keV=(lo_keV, hi_keV),
            peak_search_method=peak_search_method,
            max_components=max_components,
            registries=shared,
        )
        if overlap_components:
            notes.append(f"{len(overlap_components)} overlap component(s) fitted.")

    return ROIAnalysisResult(
        label=str(label),
        roi_bounds_keV=(lo_keV, hi_keV),
        gross_counts=gross_counts,
        gross_counts_uncertainty=gross_unc,
        background_counts=background_counts,
        background_counts_uncertainty=background_unc,
        net_counts=net_counts,
        net_counts_uncertainty=net_unc,
        centroid_keV=centroid_keV,
        centroid_uncertainty_keV=centroid_unc_keV,
        significance=significance,
        background_method=background_method,
        peak_search_method=peak_search_method,
        sideband_bounds_keV=sideband_bounds,
        overlap_components=overlap_components,
        notes=tuple(notes),
    )


def compute_roi_statistics(
    spectra: Sequence[tuple[str, GammaSpectrum] | GammaSpectrum],
    *,
    roi_bounds_keV: tuple[float, float],
    label: str = "ROI Statistics",
    background_method: str = "roi_sideband",
    peak_search_method: str = "mariscotti",
    sideband_width_keV: float | None = None,
    registries: PluginRegistries | None = None,
) -> ROIStatisticsResult:
    """Summarize the same ROI across many spectra."""

    samples: list[ROISpectrumStatistic] = []
    for index, item in enumerate(spectra):
        if isinstance(item, tuple):
            sample_label, spectrum = item
        else:
            spectrum = item
            sample_label = getattr(spectrum, "spectrum_id", None) or f"spectrum-{index + 1}"
        result = analyze_roi_region(
            spectrum,
            roi_bounds_keV=roi_bounds_keV,
            label=sample_label,
            background_method=background_method,
            peak_search_method=peak_search_method,
            sideband_width_keV=sideband_width_keV,
            registries=registries,
        )
        samples.append(
            ROISpectrumStatistic(
                label=str(sample_label),
                net_counts=float(result.net_counts),
                net_counts_uncertainty=float(result.net_counts_uncertainty),
                centroid_keV=float(result.centroid_keV),
                significance=float(result.significance),
            )
        )

    if not samples:
        raise ValueError("At least one spectrum is required for ROI statistics.")

    net_values = np.asarray([sample.net_counts for sample in samples], dtype=float)
    centroid_values = np.asarray([sample.centroid_keV for sample in samples], dtype=float)
    sample_count = len(samples)
    stdev_net = float(np.std(net_values, ddof=1)) if sample_count > 1 else 0.0
    stdev_centroid = float(np.std(centroid_values, ddof=1)) if sample_count > 1 else 0.0
    mean_net = float(np.mean(net_values))
    return ROIStatisticsResult(
        label=str(label),
        roi_bounds_keV=tuple(sorted((float(roi_bounds_keV[0]), float(roi_bounds_keV[1])))),
        sample_count=sample_count,
        mean_net_counts=mean_net,
        stdev_net_counts=stdev_net,
        relative_std=float(stdev_net / mean_net) if abs(mean_net) > 1.0e-12 else 0.0,
        mean_centroid_keV=float(np.mean(centroid_values)),
        stdev_centroid_keV=stdev_centroid,
        min_net_counts=float(np.min(net_values)),
        max_net_counts=float(np.max(net_values)),
        samples=tuple(samples),
    )


def _background_subtraction_factor(
    foreground: GammaSpectrum,
    background: GammaSpectrum | None,
    *,
    mode: str,
    scale: float,
) -> float:
    if background is None:
        return 0.0
    if mode == "simple":
        return 1.0
    if mode == "scaled":
        return float(scale)
    if mode == "statistical":
        return (
            max(float(foreground.live_time or 1.0), 1.0)
            / max(float(background.live_time or 1.0), 1.0)
        )
    raise ValueError(f"Unsupported background-subtraction mode: {mode}")


def _roi_channel_bounds(
    spectrum: GammaSpectrum,
    roi_bounds_keV: tuple[float, float],
) -> tuple[int, int]:
    lo_keV, hi_keV = sorted((float(roi_bounds_keV[0]), float(roi_bounds_keV[1])))
    return (
        int(spectrum.energy_to_channel(lo_keV)),
        int(spectrum.energy_to_channel(hi_keV)),
    )


def _estimate_roi_background_curve(
    spectrum: GammaSpectrum,
    *,
    roi_bounds_keV: tuple[float, float],
    background_method: str,
    sideband_width_keV: float | None,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    channels = np.asarray(spectrum.channels, dtype=float)
    counts = np.asarray(spectrum.counts, dtype=float)
    roi_lo_ch, roi_hi_ch = _roi_channel_bounds(spectrum, roi_bounds_keV)
    roi_lo_keV, roi_hi_keV = sorted((float(roi_bounds_keV[0]), float(roi_bounds_keV[1])))
    width_keV = max(roi_hi_keV - roi_lo_keV, 1.0)
    sideband_width_keV = float(sideband_width_keV or max(width_keV * 0.5, 3.0))

    if background_method == "roi_sideband":
        left_keV = (max(0.0, roi_lo_keV - sideband_width_keV), roi_lo_keV)
        right_keV = (roi_hi_keV, roi_hi_keV + sideband_width_keV)
        left_lo_ch, left_hi_ch = _roi_channel_bounds(spectrum, left_keV)
        right_lo_ch, right_hi_ch = _roi_channel_bounds(spectrum, right_keV)
        left_mask = (channels >= left_lo_ch) & (channels <= left_hi_ch)
        right_mask = (channels >= right_lo_ch) & (channels <= right_hi_ch)
        left_rate = float(np.mean(counts[left_mask])) if np.any(left_mask) else float(counts[max(roi_lo_ch - 1, 0)])
        right_rate = float(np.mean(counts[right_mask])) if np.any(right_mask) else float(counts[min(roi_hi_ch, len(counts) - 1)])
        roi_channels = channels[(channels >= roi_lo_ch) & (channels <= roi_hi_ch)]
        if roi_channels.size:
            interp = np.interp(
                roi_channels,
                np.asarray([float(roi_lo_ch), float(roi_hi_ch)], dtype=float),
                np.asarray([left_rate, right_rate], dtype=float),
            )
        else:
            interp = np.array([], dtype=float)
        background_curve = np.zeros_like(counts, dtype=float)
        background_curve[(channels >= roi_lo_ch) & (channels <= roi_hi_ch)] = interp
        return background_curve, (left_keV, right_keV)

    peak_regions = [(int(roi_lo_ch), int(roi_hi_ch))]
    if background_method == "snip":
        curve = estimate_background(
            channels,
            counts,
            method="snip",
        )
    elif background_method == "linear_minima":
        curve = estimate_background(
            channels,
            counts,
            method="linear",
            peak_regions=peak_regions,
        )
    else:
        raise ValueError(f"Unsupported ROI background method: {background_method}")
    empty_bounds = ((roi_lo_keV, roi_lo_keV), (roi_hi_keV, roi_hi_keV))
    return np.asarray(curve, dtype=float), empty_bounds


def _weighted_centroid_keV(
    spectrum: GammaSpectrum,
    net_curve: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[float, float]:
    roi_channels = np.asarray(spectrum.channels[roi_mask], dtype=float)
    roi_weights = np.clip(np.asarray(net_curve[roi_mask], dtype=float), 0.0, None)
    if roi_channels.size == 0:
        return (0.0, 0.0)
    if float(np.sum(roi_weights)) <= 0.0:
        center_channel = float(np.mean(roi_channels))
        center_keV = float(spectrum.channel_to_energy(center_channel))
        return (center_keV, 0.0)
    centroid_channel = float(np.average(roi_channels, weights=roi_weights))
    centroid_keV = float(spectrum.channel_to_energy(centroid_channel))
    variance = float(np.average((roi_channels - centroid_channel) ** 2, weights=roi_weights))
    centroid_unc_keV = float(
        abs(spectrum.channel_to_energy(centroid_channel + math.sqrt(max(variance, 0.0) / max(np.sum(roi_weights), 1.0))))
        - centroid_keV
    )
    return centroid_keV, centroid_unc_keV


def _fit_roi_overlap_components(
    spectrum: GammaSpectrum,
    *,
    roi_bounds_keV: tuple[float, float],
    peak_search_method: str,
    max_components: int,
    registries: PluginRegistries | None = None,
) -> tuple[ROIComponentFit, ...]:
    roi_lo_keV, roi_hi_keV = sorted((float(roi_bounds_keV[0]), float(roi_bounds_keV[1])))
    roi_lo_ch, roi_hi_ch = _roi_channel_bounds(spectrum, (roi_lo_keV, roi_hi_keV))
    channels = np.asarray(spectrum.channels, dtype=float)
    counts = np.asarray(spectrum.counts, dtype=float)
    mask = (channels >= roi_lo_ch) & (channels <= roi_hi_ch)
    if np.count_nonzero(mask) < 5:
        return ()

    local_channels = np.asarray(channels[mask], dtype=float)
    local_counts = np.asarray(counts[mask], dtype=float)
    local_peaks = _detect_peak_method_results(
        local_counts,
        channels=local_channels,
        method=peak_search_method,
        threshold=3.0,
        min_distance=max(int(max((roi_hi_ch - roi_lo_ch) / 6.0, 1.0)), 1),
        registries=registries,
    )
    peak_channels = [int(round(channel)) for channel, _significance in local_peaks[:max_components]]
    minimum_distance = max(int(max((roi_hi_ch - roi_lo_ch) / 8.0, 1.0)), 1)
    if len(peak_channels) < min(2, max_components):
        ranked_indices = np.argsort(local_counts)[::-1]
        for index in ranked_indices:
            candidate_channel = int(round(local_channels[int(index)]))
            if any(abs(candidate_channel - existing) < minimum_distance for existing in peak_channels):
                continue
            if local_counts[int(index)] <= np.median(local_counts):
                continue
            peak_channels.append(candidate_channel)
            if len(peak_channels) >= min(2, max_components):
                break
    if not peak_channels:
        peak_channels = [int(round(local_channels[int(np.argmax(local_counts))]))]
    fit_results = fit_multiple_peaks(
        channels,
        counts,
        peak_channels=peak_channels,
        fit_width=max(int((roi_hi_ch - roi_lo_ch) / 2), 4),
        background_model="linear",
        share_sigma=True,
    )
    components: list[ROIComponentFit] = []
    for result in fit_results[:max_components]:
        components.append(
            ROIComponentFit(
                centroid_channel=float(result.peak.centroid),
                centroid_keV=float(spectrum.channel_to_energy(result.peak.centroid)),
                net_counts=float(result.net_counts),
                net_counts_uncertainty=float(result.net_counts_uncertainty),
                fwhm_channels=float(result.peak.fwhm),
                reduced_chi_squared=float(result.reduced_chi_squared),
            )
        )
    return tuple(components)


def extract_survey_points(
    spectra: Sequence[tuple[str, GammaSpectrum]],
) -> tuple[SurveyPoint, ...]:
    """Extract GPS coordinates from a sequence of spectra."""

    points: list[SurveyPoint] = []
    for role, spectrum in spectra:
        gps = {}
        if isinstance(getattr(spectrum, "gps", None), dict):
            gps.update(spectrum.gps)
        metadata = getattr(spectrum, "metadata", {}) or {}
        if isinstance(metadata, dict) and isinstance(metadata.get("gps"), dict):
            gps.update(metadata["gps"])
        latitude = gps.get("latitude")
        longitude = gps.get("longitude")
        if latitude is None or longitude is None:
            continue
        points.append(
            SurveyPoint(
                label=spectrum.spectrum_id or role.title(),
                latitude=float(latitude),
                longitude=float(longitude),
                source_role=role,
            )
        )
    return tuple(points)


def compute_cascade_sum_lines(
    pinned_nuclides: Sequence[str],
    *,
    source_id: str = "fluxforge_bundled_gamma",
    custom_path: str | None = None,
    limit_per_nuclide: int = 3,
) -> tuple[float, ...]:
    """Return cascade-sum energies for pinned nuclides."""

    database = _load_gamma_database(source_id, custom_path=custom_path)
    energies: set[float] = set()
    for nuclide in pinned_nuclides:
        decay = database.get(nuclide)
        if decay is None:
            continue
        lines = [
            round(line.energy_keV, 3)
            for line in decay.strongest_gamma_lines(n=limit_per_nuclide)
        ]
        for index, energy_a in enumerate(lines):
            for energy_b in lines[index + 1 :]:
                energies.add(round(float(energy_a + energy_b), 3))
    return tuple(sorted(energies))


def _fit_semi_empirical_coefficients(
    energies: np.ndarray,
    values: np.ndarray,
) -> list[float]:
    log_eff = np.log(np.clip(values, 1e-12, 1.0))
    poly = np.polyfit(np.log(energies), log_eff, 2)
    scale = float(np.clip(np.exp(poly[2]), 1e-9, 1.0))
    length = float(np.clip(np.max(values) * 10.0, 0.2, 6.0))
    alpha = 1.15
    length0 = float(np.clip(np.mean(energies) / 1500.0, 0.2, 4.0))
    kappa = float(np.clip(np.exp(poly[1]), 0.1, 3.0))
    coefficients = [scale, length, alpha, length0, kappa]
    predicted = semi_empirical_efficiency(energies, coefficients)
    if np.any(~np.isfinite(predicted)) or np.max(predicted) <= 0:
        coefficients = [1e-3, 2.0, 1.1, 1.0, 0.8]
    return coefficients


def _score_database_lines(
    peaks: Sequence[PeakCandidate],
    database: GammaDatabase,
    *,
    tolerance_keV: float,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for nuclide in database.nuclides:
        decay = database.get(nuclide)
        if decay is None:
            continue
        score = 0.0
        strong_lines = decay.strongest_gamma_lines(n=6)
        if not strong_lines:
            continue
        for peak in peaks:
            nearest = min(
                strong_lines,
                key=lambda line: abs(line.energy_keV - peak.energy_keV),
            )
            delta = abs(nearest.energy_keV - peak.energy_keV)
            if delta > tolerance_keV:
                continue
            intensity = max(nearest.intensity * nearest.norm, 1e-6)
            score += intensity * math.exp(
                -0.5 * (delta / max(tolerance_keV, 1e-6)) ** 2
            )
        if score > 0.0:
            scores[nuclide] = float(score)
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {nuclide: score / total for nuclide, score in scores.items()}


def _load_gamma_database(
    source_id: str,
    *,
    custom_path: str | None = None,
) -> GammaDatabase:
    if source_id == "fluxforge_bundled_gamma":
        data_path = Path(FLUXFORGE_GAMMA_DATA)
        return GammaDatabase(str(data_path if data_path.exists() else ""))
    return load_gamma_identification_source(source_id, custom_path=custom_path)


__all__ = [
    "ActivityCalculationResult",
    "apply_ml_peak_predictions",
    "BayesianNuclideMatchDefinition",
    "ContinuumDriverEstimate",
    "EfficiencyCalibrationFitResult",
    "EfficiencyModelDefinition",
    "PeakCandidate",
    "SpectralPhenomenonEstimate",
    "SurveyPoint",
    "bayesian_match_peak_candidates",
    "build_decay_chain_summary",
    "calculate_peak_activity",
    "compute_cascade_sum_lines",
    "detect_peak_candidates",
    "estimate_dominant_continuum_driver",
    "estimate_spectral_phenomena",
    "extract_survey_points",
    "fit_efficiency_model",
    "register_builtin_efficiency_models",
    "register_builtin_nuclide_id_engines",
    "subtract_background_counts",
]


_shared_registries = bootstrap_builtin_registries()
if len(_shared_registries.calibration_models) == 0:
    register_builtin_efficiency_models(_shared_registries)
if len(_shared_registries.nuclide_id_engines) == 0:
    register_builtin_nuclide_id_engines(_shared_registries)
