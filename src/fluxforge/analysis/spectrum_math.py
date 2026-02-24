"""
Spectrum math helpers for GammaSpectrum.

Includes arithmetic and smoothing utilities for offline workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fluxforge.io.spe import GammaSpectrum


@dataclass
class SpectrumPair:
    """Aligned spectrum pair for arithmetic operations."""

    channels: np.ndarray
    left_counts: np.ndarray
    right_counts: np.ndarray
    calibration: dict


def _align_spectra(
    left: GammaSpectrum,
    right: GammaSpectrum,
    rtol: float = 1e-5,
) -> SpectrumPair:
    """
    Align two GammaSpectrum instances on channel grid.
    """
    n_left = len(left.channels)
    n_right = len(right.channels)
    n_min = min(n_left, n_right)

    if not np.allclose(left.channels[:n_min], right.channels[:n_min], rtol=rtol):
        raise ValueError("Spectra have incompatible channel grids.")

    if n_left >= n_right:
        channels = left.channels.copy()
        left_counts = left.counts.copy()
        right_counts = np.pad(right.counts, (0, n_left - n_right), constant_values=0.0)
    else:
        channels = right.channels.copy()
        right_counts = right.counts.copy()
        left_counts = np.pad(left.counts, (0, n_right - n_left), constant_values=0.0)

    calibration = left.calibration if left.calibration == right.calibration else {}

    return SpectrumPair(
        channels=channels,
        left_counts=left_counts,
        right_counts=right_counts,
        calibration=calibration,
    )


def add_spectra(left: GammaSpectrum, right: GammaSpectrum) -> GammaSpectrum:
    """Add two spectra with channel alignment."""
    aligned = _align_spectra(left, right)
    return GammaSpectrum(
        counts=aligned.left_counts + aligned.right_counts,
        channels=aligned.channels,
        live_time=0.0,
        real_time=0.0,
        calibration=aligned.calibration,
        spectrum_id=f"{left.spectrum_id}_plus_{right.spectrum_id}",
        metadata={"operation": "add"},
    )


def subtract_spectra(left: GammaSpectrum, right: GammaSpectrum) -> GammaSpectrum:
    """Subtract two spectra with channel alignment (left - right)."""
    aligned = _align_spectra(left, right)
    return GammaSpectrum(
        counts=aligned.left_counts - aligned.right_counts,
        channels=aligned.channels,
        live_time=0.0,
        real_time=0.0,
        calibration=aligned.calibration,
        spectrum_id=f"{left.spectrum_id}_minus_{right.spectrum_id}",
        metadata={"operation": "subtract"},
    )


def moving_average(
    spectrum: GammaSpectrum,
    width: int,
) -> GammaSpectrum:
    """
    Apply a moving-average smoother.

    The output spectrum is trimmed by `width` channels on each side.
    """
    if width <= 0:
        raise ValueError("width must be a positive integer.")

    counts = np.asarray(spectrum.counts, dtype=float)
    if counts.size < 2 * width + 1:
        raise ValueError("Spectrum too short for requested smoothing width.")

    window = np.ones(2 * width + 1, dtype=float)
    smoothed = np.convolve(counts, window, mode="valid") / window.size
    channels = spectrum.channels[width:-width]

    return GammaSpectrum(
        counts=smoothed,
        channels=channels,
        live_time=0.0,
        real_time=0.0,
        calibration=spectrum.calibration,
        spectrum_id=f"{spectrum.spectrum_id}_smoothed",
        metadata={"operation": "moving_average", "width": width},
    )
