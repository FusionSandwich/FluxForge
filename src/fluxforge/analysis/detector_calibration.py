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
    activity_source_id: Optional[str] = None

    def efficiency(self) -> Tuple[float, float]:
        """Return (efficiency, uncertainty)."""
        if self.activity_source_id is not None and (
            not isinstance(self.activity_source_id, str)
            or not self.activity_source_id.strip()
        ):
            raise ValueError("activity_source_id must be a non-empty source identifier.")
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


def efficiency_measurement_covariance(
    points: Iterable[EfficiencyPoint], *, log_space: bool = False
) -> np.ndarray:
    """First-order point covariance, correlating activity errors by source ID.

    Count and line-probability errors remain independent. Lines with the same
    non-empty activity source ID share a fully correlated activity error.
    """
    observed = tuple(points)
    measured = np.asarray([point.efficiency() for point in observed], dtype=float)
    if measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError("Efficiency covariance needs measured calibration points.")
    efficiencies, uncertainties = measured.T
    relative = uncertainties / efficiencies
    covariance = np.diag(relative**2)
    for left, first in enumerate(observed):
        source_id = first.activity_source_id
        if source_id is None:
            continue
        for right in range(left + 1, len(observed)):
            second = observed[right]
            if second.activity_source_id == source_id:
                shared = float(first.activity_rel_unc or 0.0) * float(
                    second.activity_rel_unc or 0.0
                )
                covariance[left, right] = covariance[right, left] = shared
    if not log_space:
        covariance *= np.outer(efficiencies, efficiencies)
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Calibration point covariance needs independent count or line uncertainty."
        ) from exc
    return covariance


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
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("Polynomial degree must be a nonnegative integer.")
    observed = tuple(points)
    energies = []
    efficiencies = []

    for point in observed:
        eff, unc = point.efficiency()
        if not np.isfinite(point.energy_keV) or point.energy_keV <= 0:
            raise ValueError("Calibration energy must be finite and positive (keV).")
        energies.append(point.energy_keV)
        efficiencies.append(eff)

    if len(energies) < degree + 1:
        raise ValueError("Not enough calibration points for requested degree.")

    energies_arr = np.array(energies, dtype=float)
    eff_arr = np.array(efficiencies, dtype=float)

    if len(np.unique(energies_arr)) < degree + 1:
        raise ValueError("Distinct calibration energies are required for this degree.")
    x = np.log(energies_arr)
    y = np.log(eff_arr)
    design = np.column_stack([x**power for power in range(degree + 1)])
    cholesky = np.linalg.cholesky(
        efficiency_measurement_covariance(observed, log_space=True)
    )
    weighted = np.linalg.solve(cholesky, design)
    if np.linalg.matrix_rank(weighted) != degree + 1:
        raise ValueError("Polynomial efficiency basis is rank deficient.")
    coeffs_arr, _, _, _ = np.linalg.lstsq(
        weighted, np.linalg.solve(cholesky, y), rcond=None
    )
    cov = np.linalg.inv(weighted.T @ weighted)
    coeffs = coeffs_arr.tolist()

    y_fit = design @ coeffs_arr
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


def fit_gray_efficiency_curve(
    points: Iterable[EfficiencyPoint],
    detector_id: str = "",
) -> EfficiencyFit:
    """Fit ln(eff) = a + b ln(E) + c ln(E)^2 + d/E, weighted by known errors."""
    observed = list(points)
    if len(observed) < 4:
        raise ValueError("Gray efficiency fit requires at least four points.")
    energies = np.asarray([point.energy_keV for point in observed], dtype=float)
    if np.any(~np.isfinite(energies)) or np.any(energies <= 0):
        raise ValueError("Calibration energy must be finite and positive (keV).")
    measured = np.asarray([point.efficiency() for point in observed], dtype=float)
    efficiencies = measured[:, 0]
    log_energy = np.log(energies)
    design = np.column_stack((np.ones_like(energies), log_energy, log_energy**2, 1 / energies))
    cholesky = np.linalg.cholesky(
        efficiency_measurement_covariance(observed, log_space=True)
    )
    weighted = np.linalg.solve(cholesky, design)
    if np.linalg.matrix_rank(weighted) != 4:
        raise ValueError("Gray efficiency basis is rank deficient; use distinct energies.")
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted, np.linalg.solve(cholesky, np.log(efficiencies)), rcond=None
    )
    covariance = np.linalg.inv(weighted.T @ weighted)
    curve = EfficiencyCurve(
        model_type="functional",
        parameters={"form": "gray", **dict(zip(("a", "b", "c", "d"), coefficients.tolist()))},
        energy_range=(float(np.min(energies)), float(np.max(energies))),
        detector_id=detector_id,
    )
    return EfficiencyFit(
        coefficients=coefficients.tolist(),
        covariance=covariance,
        curve=curve,
        residuals=np.log(efficiencies) - design @ coefficients,
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
