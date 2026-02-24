"""
Lightweight spectroscopy helpers (peak search + simple fits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass
class PeakCandidate:
    """Peak candidate summary."""

    channel: float
    counts: float
    energy_keV: Optional[float] = None


@dataclass
class PeakFitSummary:
    """Gaussian + polynomial baseline fit summary."""

    parameters: List[float]
    energies_keV: List[float]
    fit_counts: List[float]
    baseline_counts: List[float]
    resolution_pct: float


def prominence_peaks(
    channels: np.ndarray,
    counts: np.ndarray,
    prominence: float = 0.01,
    calibration: Optional[callable] = None,
) -> Dict[int, PeakCandidate]:
    """
    Find peaks using SciPy prominence filter.
    """
    peak_idx, _ = find_peaks(counts, prominence=prominence)
    peaks: Dict[int, PeakCandidate] = {}
    for i, idx in enumerate(peak_idx):
        energy = calibration(channels[idx]) if calibration is not None else None
        peaks[i] = PeakCandidate(
            channel=float(channels[idx]),
            counts=float(counts[idx]),
            energy_keV=float(energy) if energy is not None else None,
        )
    return peaks


def _gaussian_unit(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def _poly_baseline(x: np.ndarray, coeffs: List[float]) -> np.ndarray:
    baseline = np.zeros_like(x, dtype=float)
    for power, coef in enumerate(coeffs):
        baseline += coef * (x ** power)
    return baseline


def fit_gaussian_baseline(
    energies_keV: np.ndarray,
    counts: np.ndarray,
    e_min: float,
    e_max: float,
    baseline_order: int = 1,
    max_evals: int = 100000,
) -> PeakFitSummary:
    """
    Fit Gaussian peak with polynomial baseline in a specified energy window.
    """
    if energies_keV is None:
        raise ValueError("Energies are required for Gaussian peak fitting.")

    mask = (energies_keV >= e_min) & (energies_keV <= e_max)
    x = energies_keV[mask].astype(float)
    y = counts[mask].astype(float)

    if x.size < 3:
        raise ValueError("Not enough data points in fit window.")

    def model(x, amplitude, centroid, sigma, *baseline_coeffs):
        return amplitude * _gaussian_unit(x, centroid, sigma) + _poly_baseline(x, list(baseline_coeffs))

    guess = [max(y), 0.5 * (e_min + e_max), 0.1 * (e_max - e_min)]
    guess.extend([0.0] * (baseline_order + 1))

    popt, _ = curve_fit(model, x, y, p0=guess, maxfev=max_evals)

    fit = model(x, *popt)
    baseline = _poly_baseline(x, list(popt[3:]))

    fwhm = 2.0 * popt[2] * np.sqrt(2.0 * np.log(2.0))
    resolution_pct = 100.0 * fwhm / popt[1] if popt[1] != 0 else 0.0

    return PeakFitSummary(
        parameters=list(popt),
        energies_keV=list(x),
        fit_counts=list(fit),
        baseline_counts=list(baseline),
        resolution_pct=float(resolution_pct),
    )
