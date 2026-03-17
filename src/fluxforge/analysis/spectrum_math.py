"""
Spectrum math helpers for GammaSpectrum.

Includes arithmetic and smoothing utilities for offline workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import numpy as np

from fluxforge.io.spe import GammaSpectrum


@dataclass
class SpectrumPair:
    """Aligned spectrum pair for arithmetic operations."""

    channels: np.ndarray
    left_counts: np.ndarray
    right_counts: np.ndarray
    left_uncertainty: np.ndarray
    right_uncertainty: np.ndarray
    calibration: dict


def _spectrum_energies(spectrum: GammaSpectrum) -> Optional[np.ndarray]:
    """Return calibrated energies for a spectrum when available."""
    if spectrum.energies is not None:
        return np.asarray(spectrum.energies, dtype=float)
    calibration = spectrum.calibration.get("energy") if spectrum.calibration else None
    if calibration:
        return np.asarray(spectrum.calibrate_channels(coefficients=list(calibration)), dtype=float)
    return None


def _resample_background_to_sample_energy(
    sample: GammaSpectrum,
    background: GammaSpectrum,
    atol_keV: float = 1.0e-3,
    rtol: float = 1.0e-6,
) -> Tuple[GammaSpectrum, bool]:
    """
    Resample a background spectrum onto the sample energy grid when needed.

    Returns the possibly resampled background and a flag indicating whether
    energy-grid alignment was applied.
    """
    sample_energies = _spectrum_energies(sample)
    background_energies = _spectrum_energies(background)
    if sample_energies is None or background_energies is None:
        return background, False

    n_min = min(len(sample_energies), len(background_energies))
    if (
        len(sample_energies) == len(background_energies)
        and np.allclose(sample_energies[:n_min], background_energies[:n_min], atol=atol_keV, rtol=rtol)
    ):
        return background, False

    background_counts = np.asarray(background.counts, dtype=float)
    background_unc = np.asarray(background.counts_uncertainty, dtype=float)
    resampled_counts = np.interp(sample_energies, background_energies, background_counts, left=0.0, right=0.0)
    resampled_variance = np.interp(
        sample_energies,
        background_energies,
        background_unc ** 2,
        left=0.0,
        right=0.0,
    )
    metadata = dict(background.metadata)
    metadata["energy_resampled_to_sample_grid"] = True
    metadata["resampled_from_energy_calibration"] = list(background.calibration.get("energy", []))
    return GammaSpectrum(
        counts=resampled_counts,
        counts_uncertainty=np.sqrt(np.maximum(resampled_variance, 0.0)),
        channels=np.asarray(sample.channels).copy(),
        energies=np.asarray(sample_energies, dtype=float).copy(),
        live_time=float(background.live_time),
        real_time=float(background.real_time),
        start_time=background.start_time,
        spectrum_id=background.spectrum_id,
        detector_id=background.detector_id,
        calibration=dict(sample.calibration),
        metadata=metadata,
    ), True


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
        left_unc = np.pad(
            np.asarray(left.counts_uncertainty, dtype=float),
            (0, max(0, n_left - len(left.counts_uncertainty))),
            constant_values=0.0,
        )[:n_left]
        right_unc = np.pad(
            np.asarray(right.counts_uncertainty, dtype=float),
            (0, n_left - n_right),
            constant_values=0.0,
        )
    else:
        channels = right.channels.copy()
        right_counts = right.counts.copy()
        left_counts = np.pad(left.counts, (0, n_right - n_left), constant_values=0.0)
        right_unc = np.pad(
            np.asarray(right.counts_uncertainty, dtype=float),
            (0, max(0, n_right - len(right.counts_uncertainty))),
            constant_values=0.0,
        )[:n_right]
        left_unc = np.pad(
            np.asarray(left.counts_uncertainty, dtype=float),
            (0, n_right - n_left),
            constant_values=0.0,
        )

    calibration = left.calibration if left.calibration == right.calibration else {}

    return SpectrumPair(
        channels=channels,
        left_counts=left_counts,
        right_counts=right_counts,
        left_uncertainty=left_unc,
        right_uncertainty=right_unc,
        calibration=calibration,
    )


def add_spectra(left: GammaSpectrum, right: GammaSpectrum) -> GammaSpectrum:
    """Add two spectra with channel alignment."""
    aligned = _align_spectra(left, right)
    variance = aligned.left_uncertainty**2 + aligned.right_uncertainty**2
    return GammaSpectrum(
        counts=aligned.left_counts + aligned.right_counts,
        counts_uncertainty=np.sqrt(np.maximum(variance, 0.0)),
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
    variance = aligned.left_uncertainty**2 + aligned.right_uncertainty**2
    return GammaSpectrum(
        counts=aligned.left_counts - aligned.right_counts,
        counts_uncertainty=np.sqrt(np.maximum(variance, 0.0)),
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
    uncertainty = np.sqrt(
        np.convolve(np.asarray(spectrum.counts_uncertainty, dtype=float) ** 2, window, mode="valid")
    ) / window.size

    return GammaSpectrum(
        counts=smoothed,
        counts_uncertainty=uncertainty,
        channels=channels,
        live_time=0.0,
        real_time=0.0,
        calibration=spectrum.calibration,
        spectrum_id=f"{spectrum.spectrum_id}_smoothed",
        metadata={"operation": "moving_average", "width": width},
    )


def _resolve_scale_factor(
    sample: GammaSpectrum,
    background: GammaSpectrum,
    mode: str,
    manual_scale: Optional[float],
) -> float:
    mode = mode.lower()
    if mode == "live":
        denom = float(background.live_time)
        if denom <= 0.0:
            warnings.warn(
                "Background live time is non-positive; using scale factor 1.0.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1.0
        return float(sample.live_time) / denom

    if mode == "real":
        denom = float(background.real_time)
        if denom <= 0.0:
            warnings.warn(
                "Background real time is non-positive; using scale factor 1.0.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1.0
        return float(sample.real_time) / denom

    if mode == "manual":
        if manual_scale is None:
            raise ValueError("manual_scale must be provided when mode='manual'.")
        return float(manual_scale)

    raise ValueError(f"Unknown background scale mode: {mode}")


def subtract_measured_background(
    sample: GammaSpectrum,
    background: Optional[GammaSpectrum],
    mode: str = "live",
    manual_scale: Optional[float] = None,
    negative_policy: str = "hybrid",
    warn_missing: bool = True,
) -> GammaSpectrum:
    """
    Subtract measured background spectrum with uncertainty propagation.

    Parameters
    ----------
    sample : GammaSpectrum
        Sample spectrum.
    background : GammaSpectrum or None
        Measured background spectrum. If None, sample is returned unchanged.
    mode : {'live', 'real', 'manual'}
        Normalization mode for background scaling factor.
    manual_scale : float, optional
        Explicit scale factor for mode='manual'.
    negative_policy : {'hybrid', 'clip', 'preserve'}
        How to handle negative channels after subtraction.
    warn_missing : bool
        Warn when no background spectrum is provided.
    """
    if background is None:
        if warn_missing:
            warnings.warn(
                "Background subtraction enabled but no background spectrum was provided; using raw spectrum.",
                RuntimeWarning,
                stacklevel=2,
            )
        return GammaSpectrum(
            counts=np.asarray(sample.counts, dtype=float).copy(),
            counts_uncertainty=np.asarray(sample.counts_uncertainty, dtype=float).copy(),
            channels=np.asarray(sample.channels, dtype=int).copy(),
            energies=np.asarray(sample.energies, dtype=float).copy() if sample.energies is not None else None,
            live_time=float(sample.live_time),
            real_time=float(sample.real_time),
            start_time=sample.start_time,
            spectrum_id=sample.spectrum_id,
            detector_id=sample.detector_id,
            calibration=dict(sample.calibration),
            metadata=dict(sample.metadata),
        )

    aligned_background, energy_aligned = _resample_background_to_sample_energy(sample, background)
    aligned = _align_spectra(sample, aligned_background)
    scale_factor = _resolve_scale_factor(sample, background, mode=mode, manual_scale=manual_scale)

    net_counts = aligned.left_counts - scale_factor * aligned.right_counts
    variance = aligned.left_uncertainty**2 + (scale_factor**2) * aligned.right_uncertainty**2
    net_uncertainty = np.sqrt(np.maximum(variance, 0.0))

    negative_policy_normalized = negative_policy.lower()
    if negative_policy_normalized not in {"hybrid", "clip", "preserve"}:
        raise ValueError(
            "negative_policy must be one of: 'hybrid', 'clip', 'preserve'."
        )

    negatives = int(np.count_nonzero(net_counts < 0.0))
    clipped = False
    if negative_policy_normalized == "clip":
        if negatives > 0:
            warnings.warn(
                f"Background subtraction produced {negatives} negative bins; clipping to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
        net_counts = np.maximum(net_counts, 0.0)
        clipped = negatives > 0

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "operation": "subtract_measured_background",
            "background_subtraction": {
                "scale_mode": mode.lower(),
                "scale_factor": float(scale_factor),
                "negative_policy": negative_policy_normalized,
                "negative_bins": negatives,
                "clipped": clipped,
                "background_spectrum_id": background.spectrum_id,
                "energy_aligned": energy_aligned,
            },
        }
    )

    return GammaSpectrum(
        counts=net_counts,
        counts_uncertainty=net_uncertainty,
        channels=aligned.channels,
        energies=None,
        live_time=float(sample.live_time),
        real_time=float(sample.real_time),
        start_time=sample.start_time,
        spectrum_id=sample.spectrum_id,
        detector_id=sample.detector_id,
        calibration=dict(sample.calibration),
        metadata=metadata,
    )


def nonnegative_counts_for_algorithm(
    spectrum: GammaSpectrum,
    algorithm_name: str = "algorithm",
    warn: bool = True,
) -> np.ndarray:
    """
    Return non-negative counts for algorithms requiring non-negative inputs.
    """
    counts = np.asarray(spectrum.counts, dtype=float)
    negatives = int(np.count_nonzero(counts < 0.0))
    if negatives > 0 and warn:
        warnings.warn(
            (
                f"{algorithm_name} requires non-negative counts; clipping {negatives} "
                "negative channels generated by background subtraction."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return np.maximum(counts, 0.0)
