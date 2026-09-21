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
from fluxforge.analysis.histogram_rebin import rebin_histogram


def spectrum_bin_edges(spectrum: GammaSpectrum) -> np.ndarray:
    """Midpoint edges of recorded energy centers, with half-spacing end bins."""
    centers = _spectrum_energies(spectrum)
    if (centers is None or centers.ndim != 1 or centers.size != len(spectrum.counts)
            or centers.size < 2 or not np.all(np.isfinite(centers))
            or np.any(np.diff(centers) <= 0)):
        raise ValueError("Conservative background rebinning requires at least two finite, strictly increasing energy centers matching counts")
    edges = np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                  centers[:-1] + np.diff(centers) / 2,
                  centers[-1] + (centers[-1] - centers[-2]) / 2]
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Background energy bin edges must be finite and strictly increasing")
    return edges


def _conservative_background(sample, background):
    result = rebin_histogram(
        spectrum_bin_edges(background), background.counts, spectrum_bin_edges(sample),
        source_covariance=background.counts_covariance,
        source_variance=(background.counts_uncertainty ** 2 if background.counts_covariance is None else None),
        coverage="strict",
    )
    payload = background.to_dict()
    payload.update(counts=result.counts.tolist(), counts_uncertainty=np.sqrt(result.covariance.diagonal()).tolist(),
                   counts_covariance=None, channels=sample.channels.tolist(),
                   energies=_spectrum_energies(sample).tolist(), calibration=dict(sample.calibration))
    aligned = GammaSpectrum.from_dict(payload)
    aligned.counts_covariance = result.covariance
    aligned.counts_uncertainty = np.sqrt(result.covariance.diagonal())
    aligned.metadata = dict(background.metadata, rebin={
        "edge_convention": "energy_center_midpoints_half_spacing_endpoints",
        "coverage_policy": "strict", "minimum_coverage": float(result.coverage.min()),
        "discarded_source_counts": result.discarded_source_counts,
        "within_bin_density": "uniform", "source_calibration": dict(background.calibration),
    })
    return aligned, True


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
    """Align histograms conservatively, retaining full counting covariance."""
    sample_energies = _spectrum_energies(sample)
    background_energies = _spectrum_energies(background)
    sample_channels = np.asarray(sample.channels, dtype=float)
    background_channels = np.asarray(background.channels, dtype=float)

    if sample_energies is None and background_energies is None:
        if sample_channels.shape != background_channels.shape or not np.allclose(
            sample_channels, background_channels, atol=0.0, rtol=rtol
        ):
            raise ValueError(
                "Background subtraction requires identical channel grids when "
                "energy calibration is unavailable."
            )
        return background, False

    if sample_energies is None or background_energies is None:
        raise ValueError(
            "Background subtraction requires energy calibration for both spectra "
            "or neither spectrum."
        )

    if not np.all(np.isfinite(sample_energies)) or not np.all(
        np.isfinite(background_energies)
    ):
        raise ValueError("Background subtraction energy grids must be finite.")

    for spectrum, centers in ((sample, sample_energies), (background, background_energies)):
        if centers.ndim != 1 or len(centers) != len(spectrum.counts) or np.any(np.diff(centers) <= 0):
            raise ValueError("Background energy grids must be strictly increasing and match counts")

    if sample_energies.shape == background_energies.shape and np.allclose(
        sample_energies, background_energies, atol=atol_keV, rtol=rtol
    ):
        return background, False

    return _conservative_background(sample, background)


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
    left.require_diagonal("Legacy spectrum addition")
    right.require_diagonal("Legacy spectrum addition")
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
    left.require_diagonal("Legacy spectrum subtraction")
    right.require_diagonal("Legacy spectrum subtraction")
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
    spectrum.require_diagonal("Legacy moving-average smoothing")

    counts = np.asarray(spectrum.counts, dtype=float)
    if counts.size < 2 * width + 1:
        raise ValueError("Spectrum too short for requested smoothing width.")

    window = np.ones(2 * width + 1, dtype=float)
    smoothed = np.convolve(counts, window, mode="valid") / window.size
    channels = spectrum.channels[width:-width]
    uncertainty = (
        np.sqrt(
            np.convolve(
                np.asarray(spectrum.counts_uncertainty, dtype=float) ** 2,
                window,
                mode="valid",
            )
        )
        / window.size
    )

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
    if mode in {"live", "real"}:
        times = []
        for label, spectrum in (("sample", sample), ("background", background)):
            try:
                duration = float(getattr(spectrum, f"{mode}_time"))
            except (TypeError, ValueError):
                duration = float("nan")
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError(
                    f"Background subtraction {label} {mode} time must be finite "
                    "and positive; supply a valid count time or choose an "
                    "explicit manual scale."
                )
            times.append(duration)
        scale_factor = times[0] / times[1]
        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError(
                "Background subtraction scale factor must be finite and positive."
            )
        return scale_factor

    if mode == "manual":
        if manual_scale is None:
            raise ValueError("manual_scale must be provided when mode='manual'.")
        scale_factor = float(manual_scale)
        if not np.isfinite(scale_factor) or scale_factor < 0.0:
            raise ValueError("manual_scale must be finite and non-negative.")
        return scale_factor

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
        Normalization mode for background scaling factor. Automatic modes
        require finite, positive times for both spectra.
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
            counts_covariance=(sample.counts_covariance.copy() if sample.counts_covariance is not None else None),
            counts_uncertainty=np.asarray(
                sample.counts_uncertainty, dtype=float
            ).copy(),
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
            source_type=sample.source_type,
            device_id=sample.device_id,
            device_label=sample.device_label,
            gps=dict(sample.gps),
            metadata=dict(sample.metadata),
        )

    aligned_background, energy_aligned = _resample_background_to_sample_energy(
        sample, background
    )
    aligned = _align_spectra(sample, aligned_background)
    scale_factor = _resolve_scale_factor(
        sample, background, mode=mode, manual_scale=manual_scale
    )

    net_counts = aligned.left_counts - scale_factor * aligned.right_counts
    variance = (
        aligned.left_uncertainty**2 + (scale_factor**2) * aligned.right_uncertainty**2
    )
    covariance = None
    if sample.counts_covariance is not None or aligned_background.counts_covariance is not None:
        covariance = sample.covariance_matrix() + scale_factor**2 * aligned_background.covariance_matrix()
        variance = covariance.diagonal()
    net_uncertainty = np.sqrt(np.maximum(variance, 0.0))

    negative_policy_normalized = negative_policy.lower()
    if negative_policy_normalized not in {"hybrid", "clip", "preserve"}:
        raise ValueError(
            "negative_policy must be one of: 'hybrid', 'clip', 'preserve'."
        )

    negatives = int(np.count_nonzero(net_counts < 0.0))
    clipped = False
    if negative_policy_normalized == "clip":
        if covariance is not None:
            raise ValueError("Clipping is unsupported for correlated counts; preserve signed estimates")
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
                "rebin": aligned_background.metadata.get("rebin"),
                "covariance_assumptions": "independent sample and background; fixed scale and energy bins",
                "shared_background_covariance": "not propagated between separate results",
            },
        }
    )

    return GammaSpectrum(
        counts=net_counts,
        counts_covariance=covariance,
        counts_uncertainty=net_uncertainty,
        channels=aligned.channels,
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
        source_type=sample.source_type,
        device_id=sample.device_id,
        device_label=sample.device_label,
        gps=dict(sample.gps),
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
