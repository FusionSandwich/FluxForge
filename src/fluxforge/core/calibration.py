"""Calibration engines for the redesigned FluxForge GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from fluxforge.analysis.detector_calibration import ResolutionCurve


ASTM_E181_ENERGY_LIMIT_KEV = 0.5
ASTM_E181_LOCKED_ORDER = 2
ASTM_E181_LOCK_REASON = (
    "ASTM E181-23 §5.4.2 requires a 2nd-order energy calibration fit "
    "for the locked standards workflow."
)


@dataclass(frozen=True)
class CalibrationOrderResolution:
    """Resolved energy-fit order after standards-mode locking."""

    order: int
    locked_by: str | None = None


@dataclass(frozen=True)
class EnergyCalibrationPoint:
    """One energy-calibration point in the calibration workspace."""

    channel: float
    reference_energy_keV: float
    observed_energy_keV: float | None = None
    uncertainty_keV: float | None = None
    label: str = ""


@dataclass(frozen=True)
class EnergyDeviationPair:
    """Fine-calibration deviation pair applied on top of the polynomial fit."""

    energy_keV: float
    correction_keV: float
    label: str = ""


@dataclass(frozen=True)
class EnergyCalibrationFit:
    """Polynomial energy-calibration fit with residual-first metrics."""

    order: int
    coefficients: tuple[float, ...]
    base_fitted_keV: np.ndarray
    fitted_keV: np.ndarray
    correction_keV: np.ndarray
    residuals_keV: np.ndarray
    uncertainties_keV: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    r_squared: float
    degrees_of_freedom: int
    astm_limit_keV: float
    out_of_tolerance: tuple[bool, ...]
    deviation_pairs: tuple[EnergyDeviationPair, ...] = field(default_factory=tuple)
    locked_by: str | None = None

    @property
    def rms_keV(self) -> float:
        if self.residuals_keV.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.residuals_keV**2)))


@dataclass(frozen=True)
class FWHMCalibrationPoint:
    """One resolution-calibration point."""

    energy_keV: float
    fwhm_keV: float
    uncertainty_keV: float | None = None
    label: str = ""


@dataclass(frozen=True)
class FWHMCalibrationFit:
    """Resolution (FWHM) fit with chi-squared diagnostics."""

    model: str
    coefficients: tuple[float, ...]
    fitted_fwhm_keV: np.ndarray
    residuals_keV: np.ndarray
    uncertainties_keV: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    degrees_of_freedom: int
    curve: ResolutionCurve

    @property
    def rms_keV(self) -> float:
        if self.residuals_keV.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.residuals_keV**2)))


@dataclass(frozen=True)
class QuickCalibrationResult:
    """Linear quick-calibration preview driven by two slider anchors."""

    anchor_channels: tuple[float, float]
    reference_energies_keV: tuple[float, float]
    coefficients: tuple[float, float]

    @property
    def slope_keV_per_channel(self) -> float:
        return float(self.coefficients[1])

    def evaluate(self, channels: Sequence[float] | np.ndarray) -> np.ndarray:
        return evaluate_energy_calibration(self.coefficients, channels)


def standard_requires_astm_e181(standard: str | None) -> bool:
    """Return True when the active standard should lock the calibration workflow."""

    if not standard:
        return False
    normalized = standard.strip().lower().replace("-", " ").replace("_", " ")
    return "astm e181" in normalized


def resolve_energy_calibration_order(
    requested_order: int,
    *,
    standard: str | None = None,
) -> CalibrationOrderResolution:
    """Resolve the requested energy-fit order against standards locks."""

    order = max(1, int(requested_order))
    if standard_requires_astm_e181(standard):
        return CalibrationOrderResolution(
            order=ASTM_E181_LOCKED_ORDER,
            locked_by=ASTM_E181_LOCK_REASON,
        )
    return CalibrationOrderResolution(order=order)


def evaluate_energy_calibration(
    coefficients: Sequence[float],
    channels: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate an energy polynomial stored in ascending coefficient order."""

    values = np.asarray(channels, dtype=float)
    energies = np.zeros_like(values, dtype=float)
    for power, coefficient in enumerate(coefficients):
        energies += float(coefficient) * (values**power)
    return energies


def evaluate_energy_deviation_pairs(
    energies_keV: float | Sequence[float] | np.ndarray,
    pairs: Iterable[EnergyDeviationPair],
) -> np.ndarray:
    """Evaluate piecewise-linear energy corrections from deviation pairs."""

    values = np.asarray(energies_keV, dtype=float)
    pair_list = sorted(
        (
            EnergyDeviationPair(
                energy_keV=float(pair.energy_keV),
                correction_keV=float(pair.correction_keV),
                label=pair.label,
            )
            for pair in pairs
        ),
        key=lambda item: item.energy_keV,
    )
    if not pair_list:
        return np.zeros_like(values, dtype=float)
    if len(pair_list) == 1:
        return np.full_like(values, pair_list[0].correction_keV, dtype=float)

    anchor_energies = np.asarray([pair.energy_keV for pair in pair_list], dtype=float)
    corrections = np.asarray([pair.correction_keV for pair in pair_list], dtype=float)
    return np.interp(
        values,
        anchor_energies,
        corrections,
        left=float(corrections[0]),
        right=float(corrections[-1]),
    )


def apply_energy_deviation_pairs(
    energies_keV: float | Sequence[float] | np.ndarray,
    pairs: Iterable[EnergyDeviationPair],
) -> np.ndarray:
    """Apply deviation-pair corrections to calibrated energies."""

    values = np.asarray(energies_keV, dtype=float)
    return values + evaluate_energy_deviation_pairs(values, pairs)


def energy_calibration_slope(
    coefficients: Sequence[float],
    channel: float,
) -> float:
    """Return dE/dch for an ascending-order energy polynomial."""

    slope = 0.0
    channel_value = float(channel)
    for power, coefficient in enumerate(coefficients[1:], start=1):
        slope += power * float(coefficient) * (channel_value ** (power - 1))
    return slope


def fit_energy_calibration(
    points: Iterable[EnergyCalibrationPoint],
    *,
    order: int = 2,
    standard: str | None = None,
    astm_limit_keV: float = ASTM_E181_ENERGY_LIMIT_KEV,
    default_uncertainty_keV: float | None = None,
    deviation_pairs: Iterable[EnergyDeviationPair] = (),
) -> EnergyCalibrationFit:
    """Fit a weighted energy-calibration polynomial."""

    point_list = list(points)
    resolution = resolve_energy_calibration_order(order, standard=standard)
    if len(point_list) < resolution.order + 1:
        raise ValueError(
            "Not enough calibration points for the requested energy polynomial order."
        )

    channels = np.asarray([point.channel for point in point_list], dtype=float)
    reference_energies = np.asarray(
        [point.reference_energy_keV for point in point_list],
        dtype=float,
    )
    fallback_uncertainty = (
        float(default_uncertainty_keV)
        if default_uncertainty_keV is not None
        else max(astm_limit_keV * 0.5, 0.1)
    )
    uncertainties = np.asarray(
        [
            point.uncertainty_keV
            if point.uncertainty_keV is not None and point.uncertainty_keV > 0.0
            else fallback_uncertainty
            for point in point_list
        ],
        dtype=float,
    )

    coefficients_desc = np.polyfit(
        channels,
        reference_energies,
        deg=resolution.order,
        w=1.0 / np.clip(uncertainties, 1e-6, None),
    )
    coefficients = tuple(float(value) for value in coefficients_desc[::-1])
    normalized_pairs = tuple(
        EnergyDeviationPair(
            energy_keV=float(pair.energy_keV),
            correction_keV=float(pair.correction_keV),
            label=pair.label,
        )
        for pair in deviation_pairs
    )
    base_fitted = evaluate_energy_calibration(coefficients, channels)
    corrections = evaluate_energy_deviation_pairs(base_fitted, normalized_pairs)
    fitted = base_fitted + corrections
    residuals = reference_energies - fitted
    chi_squared = float(
        np.sum((residuals / np.clip(uncertainties, 1e-6, None)) ** 2)
    )
    degrees_of_freedom = len(point_list) - (resolution.order + 1)
    reduced_chi_squared = (
        chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else 0.0
    )
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((reference_energies - np.mean(reference_energies)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    out_of_tolerance = tuple(
        bool(abs(value) > astm_limit_keV + 1e-12) for value in residuals
    )

    return EnergyCalibrationFit(
        order=resolution.order,
        coefficients=coefficients,
        base_fitted_keV=np.asarray(base_fitted, dtype=float),
        fitted_keV=np.asarray(fitted, dtype=float),
        correction_keV=np.asarray(corrections, dtype=float),
        residuals_keV=np.asarray(residuals, dtype=float),
        uncertainties_keV=uncertainties,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        r_squared=r_squared,
        degrees_of_freedom=degrees_of_freedom,
        astm_limit_keV=float(astm_limit_keV),
        out_of_tolerance=out_of_tolerance,
        deviation_pairs=normalized_pairs,
        locked_by=resolution.locked_by,
    )


def fit_quick_slider_calibration(
    anchor_channels: Sequence[float],
    reference_energies_keV: Sequence[float],
) -> QuickCalibrationResult:
    """Create a linear preview calibration from two slider anchors."""

    if len(anchor_channels) != 2 or len(reference_energies_keV) != 2:
        raise ValueError("Quick slider calibration requires exactly two anchors.")
    channel_a, channel_b = (float(anchor_channels[0]), float(anchor_channels[1]))
    energy_a, energy_b = (
        float(reference_energies_keV[0]),
        float(reference_energies_keV[1]),
    )
    if abs(channel_b - channel_a) < 1e-9:
        raise ValueError("Quick slider anchors must occupy distinct channels.")
    slope = (energy_b - energy_a) / (channel_b - channel_a)
    intercept = energy_a - slope * channel_a
    return QuickCalibrationResult(
        anchor_channels=(channel_a, channel_b),
        reference_energies_keV=(energy_a, energy_b),
        coefficients=(intercept, slope),
    )


def fit_fwhm_calibration(
    points: Iterable[FWHMCalibrationPoint],
    *,
    model: str = "sqrt_poly",
    default_relative_uncertainty: float = 0.05,
    minimum_uncertainty_keV: float = 0.03,
) -> FWHMCalibrationFit:
    """Fit the detector resolution curve used by the calibration workspace."""

    from fluxforge.analysis.detector_calibration import fit_resolution_curve

    point_list = list(points)
    parameter_count = 3 if model == "sqrt_poly" else 2
    if len(point_list) < parameter_count:
        raise ValueError("Not enough FWHM points for the requested fit model.")

    energies = np.asarray([point.energy_keV for point in point_list], dtype=float)
    fwhm = np.asarray([point.fwhm_keV for point in point_list], dtype=float)
    uncertainties = np.asarray(
        [
            point.uncertainty_keV
            if point.uncertainty_keV is not None and point.uncertainty_keV > 0.0
            else max(abs(point.fwhm_keV) * default_relative_uncertainty, minimum_uncertainty_keV)
            for point in point_list
        ],
        dtype=float,
    )

    fit = fit_resolution_curve(energies, fwhm, model=model)
    fitted = fit.curve.fwhm(energies)
    residuals = fwhm - fitted
    chi_squared = float(
        np.sum((residuals / np.clip(uncertainties, 1e-6, None)) ** 2)
    )
    degrees_of_freedom = len(point_list) - parameter_count
    reduced_chi_squared = (
        chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else 0.0
    )

    return FWHMCalibrationFit(
        model=model,
        coefficients=tuple(float(value) for value in fit.coefficients),
        fitted_fwhm_keV=np.asarray(fitted, dtype=float),
        residuals_keV=np.asarray(residuals, dtype=float),
        uncertainties_keV=uncertainties,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        curve=fit.curve,
    )


def estimate_local_fwhm_channels(
    counts: Sequence[float] | np.ndarray,
    peak_channel: int,
) -> float:
    """Estimate local FWHM in channels from a half-height walk."""

    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        return 3.0

    channel = int(np.clip(int(round(peak_channel)), 0, values.size - 1))
    peak_height = float(values[channel])
    if peak_height <= 0.0:
        return 3.0

    half_height = peak_height * 0.5
    left = channel
    right = channel
    while left > 0 and values[left] > half_height:
        left -= 1
    while right < values.size - 1 and values[right] > half_height:
        right += 1
    return float(max(right - left, 2))


__all__ = [
    "ASTM_E181_ENERGY_LIMIT_KEV",
    "ASTM_E181_LOCKED_ORDER",
    "ASTM_E181_LOCK_REASON",
    "CalibrationOrderResolution",
    "EnergyDeviationPair",
    "EnergyCalibrationFit",
    "EnergyCalibrationPoint",
    "FWHMCalibrationFit",
    "FWHMCalibrationPoint",
    "QuickCalibrationResult",
    "apply_energy_deviation_pairs",
    "energy_calibration_slope",
    "estimate_local_fwhm_channels",
    "evaluate_energy_calibration",
    "evaluate_energy_deviation_pairs",
    "fit_energy_calibration",
    "fit_fwhm_calibration",
    "fit_quick_slider_calibration",
    "resolve_energy_calibration_order",
    "standard_requires_astm_e181",
]
