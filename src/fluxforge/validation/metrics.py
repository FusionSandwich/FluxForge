"""Validation metrics for comparing spectra and derived quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SpectrumComparison:
    truth: np.ndarray
    predicted: np.ndarray

    @property
    def residuals(self) -> np.ndarray:
        return self.predicted - self.truth

    @property
    def ratio(self) -> np.ndarray:
        denom = np.where(np.abs(self.truth) > 0.0, self.truth, np.nan)
        return self.predicted / denom


def spectrum_comparison_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    eps: float = 1e-30,
) -> Dict[str, Any]:
    """Compute basic comparison metrics for two same-shape spectra.

    Metrics are intentionally lightweight and dependency-free beyond NumPy.
    """

    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError("truth and predicted must have the same shape")

    residuals = predicted - truth

    # Guard against division by (near) zero.
    ratio = np.where(np.abs(truth) > eps, predicted / truth, np.nan)

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    finite_ratio = ratio[np.isfinite(ratio)]
    mean_ratio: Optional[float]
    std_ratio: Optional[float]
    if finite_ratio.size:
        mean_ratio = float(np.mean(finite_ratio))
        std_ratio = float(np.std(finite_ratio))
    else:
        mean_ratio = None
        std_ratio = None

    # Correlation can be undefined for constant arrays.
    corr: Optional[float]
    if np.allclose(truth, truth[0]) or np.allclose(predicted, predicted[0]):
        corr = None
    else:
        corr = float(np.corrcoef(truth, predicted)[0, 1])

    return {
        "mae": mae,
        "rmse": rmse,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
        "max_abs_residual": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "corrcoef": corr,
    }
