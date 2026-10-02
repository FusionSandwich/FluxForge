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
        return np.asarray(
            spectrum.calibrate_channels(coefficients=list(calibration)), dtype=float
        )
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

    for energies, spectrum in (
        (sample_energies, sample),
        (background_energies, background),
    ):
        if (
            energies.shape != spectrum.counts.shape
            or energies.ndim != 1
            or np.any(~np.isfinite(energies))
            or np.any(np.diff(energies) <= 0)
        ):
            raise ValueError(
                "Energy alignment requires finite, increasing channel centers."
            )

    n_min = min(len(sample_energies), len(background_energies))
    if len(sample_energies) == len(background_energies) and np.allclose(
        sample_energies[:n_min], background_energies[:n_min], atol=atol_keV, rtol=rtol
    ):
        if np.array_equal(sample.channels, background.channels):
            return background, False
        return (
            GammaSpectrum(
                counts=np.asarray(background.counts, dtype=float).copy(),
                counts_uncertainty=np.asarray(
                    background.counts_uncertainty, dtype=float
                ).copy(),
                channels=np.asarray(sample.channels).copy(),
                energies=np.asarray(sample_energies, dtype=float).copy(),
                live_time=float(background.live_time),
                real_time=float(background.real_time),
                start_time=background.start_time,
                spectrum_id=background.spectrum_id,
                detector_id=background.detector_id,
                calibration=dict(sample.calibration),
                metadata=dict(background.metadata),
                counts_covariance=background.counts_covariance,
            ),
            True,
        )

    rebin = _histogram_overlap_operator(sample_energies, background_energies)
    resampled_counts = rebin @ np.asarray(background.counts, dtype=float)
    # W C W.T retains the covariance created when output channels share a
    # source count. Its diagonal alone is insufficient for peak/ROI sums.
    resampled_covariance = (
        rebin @ background.count_covariance_matrix() @ rebin.T
    ).tocsr()
    resampled_covariance.eliminate_zeros()
    resampled_variance = resampled_covariance.diagonal()
    metadata = dict(background.metadata)
    metadata["energy_resampled_to_sample_grid"] = True
    metadata["energy_rebin_method"] = "integrated_counts_bin_overlap"
    metadata["energy_bin_edge_policy"] = "midpoint_centers_exterior_half_spacing"
    metadata["energy_coverage_policy"] = "source_target_overlap_only"
    metadata["resampled_from_energy_calibration"] = list(
        background.calibration.get("energy", [])
    )
    return (
        GammaSpectrum(
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
            counts_covariance=resampled_covariance,
        ),
        True,
    )


def _histogram_overlap_operator(target_centers: np.ndarray, source_centers: np.ndarray):
    """Sparse fractional source-bin overlap; no extrapolation or count-height interpolation."""
    from scipy.sparse import coo_matrix

    def edges(centers):
        if len(centers) < 2:
            raise ValueError(
                "Rebinning requires at least two centers to infer bin widths."
            )
        midpoints = (centers[1:] + centers[:-1]) * 0.5
        return np.concatenate(
            (
                [centers[0] - (centers[1] - centers[0]) * 0.5],
                midpoints,
                [centers[-1] + (centers[-1] - centers[-2]) * 0.5],
            )
        )

    shape = (len(target_centers), len(source_centers))
    if not all(shape):
        return coo_matrix(shape, dtype=float).tocsr()
    target, source = edges(target_centers), edges(source_centers)
    rows, columns, fractions = [], [], []
    i = j = 0
    while i < shape[0] and j < shape[1]:
        overlap = min(target[i + 1], source[j + 1]) - max(target[i], source[j])
        if overlap > 0:
            rows.append(i)
            columns.append(j)
            fractions.append(overlap / (source[j + 1] - source[j]))
        if target[i + 1] <= source[j + 1]:
            i += 1
        else:
            j += 1
    return coo_matrix((fractions, (rows, columns)), shape=shape).tocsr()


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


def _resize_count_covariance(spectrum: GammaSpectrum, size: int):
    """Pad/crop a covariance matrix on an already aligned channel grid."""
    from scipy.sparse import csr_matrix

    covariance = spectrum.count_covariance_matrix().tocoo()
    keep = (covariance.row < size) & (covariance.col < size)
    return csr_matrix(
        (covariance.data[keep], (covariance.row[keep], covariance.col[keep])),
        shape=(size, size),
    )


def _sum_count_covariance(left: GammaSpectrum, right: GammaSpectrum, size: int):
    if left.counts_covariance is None and right.counts_covariance is None:
        return None
    return _resize_count_covariance(left, size) + _resize_count_covariance(right, size)


def add_spectra(left: GammaSpectrum, right: GammaSpectrum) -> GammaSpectrum:
    """Add independently acquired spectra with channel alignment."""
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
        counts_covariance=_sum_count_covariance(left, right, len(aligned.channels)),
    )


def subtract_spectra(left: GammaSpectrum, right: GammaSpectrum) -> GammaSpectrum:
    """Subtract independently acquired spectra (left - right)."""
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
        counts_covariance=_sum_count_covariance(left, right, len(aligned.channels)),
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
    from scipy.sparse import coo_matrix

    output_size = len(smoothed)
    rows = np.repeat(np.arange(output_size), window.size)
    columns = (np.arange(output_size)[:, None] + np.arange(window.size)).ravel()
    transform = coo_matrix(
        (np.full(len(rows), 1.0 / window.size), (rows, columns)),
        shape=(output_size, len(counts)),
    ).tocsr()
    covariance = transform @ spectrum.count_covariance_matrix() @ transform.T
    uncertainty = np.sqrt(covariance.diagonal())

    return GammaSpectrum(
        counts=smoothed,
        counts_uncertainty=uncertainty,
        channels=channels,
        live_time=0.0,
        real_time=0.0,
        calibration=spectrum.calibration,
        spectrum_id=f"{spectrum.spectrum_id}_smoothed",
        metadata={"operation": "moving_average", "width": width},
        counts_covariance=covariance,
    )


def _resolve_scale_factor(
    sample: GammaSpectrum,
    background: GammaSpectrum,
    mode: str,
    manual_scale: Optional[float],
) -> float:
    mode = mode.lower()
    if mode == "live":
        numerator = float(sample.live_time)
        denom = float(background.live_time)
        if (
            not np.isfinite(numerator)
            or numerator <= 0.0
            or not np.isfinite(denom)
            or denom <= 0.0
        ):
            raise ValueError(
                "Live-time background scaling requires positive finite sample and background live times."
            )
        return numerator / denom

    if mode == "real":
        numerator = float(sample.real_time)
        denom = float(background.real_time)
        if (
            not np.isfinite(numerator)
            or numerator <= 0.0
            or not np.isfinite(denom)
            or denom <= 0.0
        ):
            raise ValueError(
                "Real-time background scaling requires positive finite sample and background real times."
            )
        return numerator / denom

    if mode == "manual":
        if manual_scale is None:
            raise ValueError("manual_scale must be provided when mode='manual'.")
        scale = float(manual_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("manual_scale must be finite and nonnegative.")
        return scale

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
            counts_uncertainty=np.asarray(
                sample.counts_uncertainty, dtype=float
            ).copy(),
            channels=np.asarray(sample.channels, dtype=int).copy(),
            energies=(
                np.asarray(sample.energies, dtype=float).copy()
                if sample.energies is not None
                else None
            ),
            live_time=float(sample.live_time),
            real_time=float(sample.real_time),
            start_time=sample.start_time,
            spectrum_id=sample.spectrum_id,
            detector_id=sample.detector_id,
            calibration=dict(sample.calibration),
            metadata=dict(sample.metadata),
            counts_covariance=sample.counts_covariance,
        )

    aligned_background, energy_aligned = _resample_background_to_sample_energy(
        sample, background
    )
    aligned = _align_spectra(sample, aligned_background)
    # Background subtraction describes the measured sample, so extra
    # background channels must not extend its count or energy grid.
    sample_channels = len(sample.counts)
    scale_factor = _resolve_scale_factor(
        sample, background, mode=mode, manual_scale=manual_scale
    )

    net_counts = (
        aligned.left_counts[:sample_channels]
        - scale_factor * aligned.right_counts[:sample_channels]
    )
    variance = (
        aligned.left_uncertainty[:sample_channels] ** 2
        + (scale_factor**2) * aligned.right_uncertainty[:sample_channels] ** 2
    )
    net_uncertainty = np.sqrt(np.maximum(variance, 0.0))
    covariance = None
    if (
        sample.counts_covariance is not None
        or aligned_background.counts_covariance is not None
    ):
        background_covariance = _resize_count_covariance(
            aligned_background, sample_channels
        )
        covariance = (
            sample.count_covariance_matrix() + scale_factor**2 * background_covariance
        )
        net_uncertainty = np.sqrt(covariance.diagonal())

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
                "count_covariance_propagated": covariance is not None,
            },
        }
    )

    return GammaSpectrum(
        counts=net_counts,
        counts_uncertainty=net_uncertainty,
        channels=np.asarray(sample.channels).copy(),
        energies=(
            np.asarray(sample.energies, dtype=float).copy()
            if sample.energies is not None
            else None
        ),
        live_time=float(sample.live_time),
        real_time=float(sample.real_time),
        start_time=sample.start_time,
        spectrum_id=sample.spectrum_id,
        detector_id=sample.detector_id,
        calibration=dict(sample.calibration),
        metadata=metadata,
        counts_covariance=covariance,
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
