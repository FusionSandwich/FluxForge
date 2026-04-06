"""
Detector calibration utilities.

Provides energy-independent helpers for efficiency and resolution fitting
from measured peak data without requiring any external databases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from fluxforge.data.efficiency import EfficiencyCurve, calculate_efficiency_from_source


@dataclass
class EfficiencyPoint:
    """Single efficiency calibration point."""

    energy_keV: float
    net_counts: float
    live_time_s: float
    activity_bq: float
    emission_probability: float
    geometry_factor: float = 1.0
    count_uncertainty: Optional[float] = None
    activity_rel_unc: Optional[float] = None
    probability_uncertainty: Optional[float] = None

    def efficiency(self) -> Tuple[float, float]:
        """Return (efficiency, uncertainty)."""
        eff, unc = calculate_efficiency_from_source(
            measured_counts=self.net_counts,
            live_time=self.live_time_s,
            source_activity=self.activity_bq,
            emission_probability=self.emission_probability,
            geometry_factor=self.geometry_factor,
            count_uncertainty=self.count_uncertainty,
            activity_uncertainty=self.activity_rel_unc,
            probability_uncertainty=self.probability_uncertainty,
        )
        return eff, unc


@dataclass
class EfficiencyFit:
    """Efficiency curve fit result."""

    coefficients: List[float]
    covariance: Optional[np.ndarray]
    curve: EfficiencyCurve
    residuals: np.ndarray


@dataclass
class ResolutionCurve:
    """Resolution (FWHM) curve model."""

    model: str
    coefficients: List[float]

    def fwhm(self, energy_keV: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy_keV, dtype=float)
        if self.model == "linear":
            a0, a1 = self.coefficients
            return a0 + a1 * energy
        if self.model == "sqrt_poly":
            a0, a1, a2 = self.coefficients
            return np.sqrt(np.clip(a0 + a1 * energy + a2 * energy**2, 0.0, None))
        raise ValueError(f"Unknown resolution model: {self.model}")

    def sigma(self, energy_keV: np.ndarray) -> np.ndarray:
        return self.fwhm(energy_keV) / 2.355


@dataclass
class ResolutionFit:
    """Resolution curve fit result."""

    coefficients: List[float]
    curve: ResolutionCurve
    residuals: np.ndarray


def fit_efficiency_curve(
    points: Iterable[EfficiencyPoint],
    degree: int = 2,
    energy_range: Optional[Tuple[float, float]] = None,
    detector_id: str = "",
) -> EfficiencyFit:
    """
    Fit a log-log polynomial efficiency curve.

    ln(eff) = a0 + a1*ln(E) + a2*ln(E)^2 + ...
    """
    energies = []
    efficiencies = []
    uncertainties = []

    for point in points:
        eff, unc = point.efficiency()
        if eff <= 0:
            continue
        energies.append(point.energy_keV)
        efficiencies.append(eff)
        uncertainties.append(max(unc, 1e-12))

    if len(energies) < degree + 1:
        raise ValueError("Not enough calibration points for requested degree.")

    energies_arr = np.array(energies, dtype=float)
    eff_arr = np.array(efficiencies, dtype=float)
    unc_arr = np.array(uncertainties, dtype=float)

    x = np.log(energies_arr)
    y = np.log(eff_arr)
    weights = 1.0 / np.clip(unc_arr / eff_arr, 1e-6, None)

    coeffs_desc, cov = np.polyfit(x, y, degree, w=weights, cov=True)
    coeffs = coeffs_desc[::-1].tolist()

    y_fit = np.polyval(coeffs_desc, x)
    residuals = y - y_fit

    if energy_range is None:
        energy_range = (float(np.min(energies_arr)), float(np.max(energies_arr)))

    curve = EfficiencyCurve.from_polynomial(
        coefficients=coeffs,
        energy_range=energy_range,
        detector_id=detector_id,
    )

    return EfficiencyFit(
        coefficients=coeffs, covariance=cov, curve=curve, residuals=residuals
    )


def fit_resolution_curve(
    energies_keV: Iterable[float],
    fwhm_keV: Iterable[float],
    model: str = "sqrt_poly",
) -> ResolutionFit:
    """
    Fit resolution (FWHM) model to peak widths.

    Supported models:
    - "linear": FWHM = a0 + a1 * E
    - "sqrt_poly": FWHM^2 = a0 + a1 * E + a2 * E^2
    """
    energy = np.asarray(list(energies_keV), dtype=float)
    fwhm = np.asarray(list(fwhm_keV), dtype=float)

    if energy.size != fwhm.size or energy.size < 2:
        raise ValueError("Resolution fit requires matching energy/FWHM arrays.")

    if model == "linear":
        coeffs_desc = np.polyfit(energy, fwhm, 1)
        coeffs = coeffs_desc[::-1].tolist()
        fitted = np.polyval(coeffs_desc, energy)
    elif model == "sqrt_poly":
        target = np.clip(fwhm**2, 0.0, None)
        coeffs_desc = np.polyfit(energy, target, 2)
        coeffs = coeffs_desc[::-1].tolist()
        fitted = np.sqrt(np.clip(np.polyval(coeffs_desc, energy), 0.0, None))
    else:
        raise ValueError(f"Unsupported resolution model: {model}")

    residuals = fwhm - fitted
    curve = ResolutionCurve(model=model, coefficients=coeffs)

    return ResolutionFit(coefficients=coeffs, curve=curve, residuals=residuals)
