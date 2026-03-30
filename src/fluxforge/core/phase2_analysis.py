"""Phase 2 analysis helpers for the modern Qt workspace."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint, fit_efficiency_curve
from fluxforge.analysis.efficiency_models import semi_empirical_efficiency
from fluxforge.analysis.peakfit import auto_find_peaks
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.gamma_database import FLUXFORGE_GAMMA_DATA, GammaDatabase
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source
from fluxforge.io.spe import GammaSpectrum
from fluxforge.physics.activation import GammaLineMeasurement
from fluxforge.physics.decay_chain import DecayChain
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
class EfficiencyModelDefinition:
    """Registered efficiency-model metadata."""

    key: str
    label: str
    degree: int | None = None
    form: str = "polynomial"
    summary: str = ""


@dataclass(frozen=True)
class EfficiencyCalibrationFitResult:
    """Efficiency fit payload used by the Phase 2 activity surfaces."""

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


def register_builtin_efficiency_models(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register the built-in Phase 2 efficiency models."""

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
            tags=("phase2", "efficiency"),
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
        tags=("phase2", "id", "bayesian"),
        set_default=True,
    )
    return registries


def detect_peak_candidates(
    spectrum: GammaSpectrum,
    *,
    threshold: float = 4.0,
    min_distance: int = 18,
    max_peaks: int = 12,
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

    peaks = auto_find_peaks(
        channels,
        counts,
        threshold=threshold,
        min_distance=min_distance,
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

    chain = DecayChain(
        nuclide,
        nuclide_data={
            nuclide: {"half_life_s": float(half_life_s), "decay_products": {"stable": 1.0}},
            "stable": {"half_life_s": float("inf"), "decay_products": {}},
        },
    )
    result = chain.decay(initial_activity={nuclide: 1.0}, times=[0.0, float(source_age_s)])
    remaining = float(result.get_activity(nuclide, time=float(source_age_s)))
    daughter = float(result.get_activity("stable", time=float(source_age_s)))
    return (
        f"Bateman correction over {source_age_s / 3600.0:.2f} h: "
        f"{nuclide} retains {remaining:.4f} of unit activity and transfers {daughter:.4f} to daughter."
    )


def subtract_background_counts(
    foreground: GammaSpectrum,
    background: GammaSpectrum | None,
    *,
    mode: str = "simple",
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract background counts using the requested Phase 2 mode."""

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
    "BayesianNuclideMatchDefinition",
    "EfficiencyCalibrationFitResult",
    "EfficiencyModelDefinition",
    "PeakCandidate",
    "SurveyPoint",
    "bayesian_match_peak_candidates",
    "build_decay_chain_summary",
    "calculate_peak_activity",
    "compute_cascade_sum_lines",
    "detect_peak_candidates",
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
