"""Spectrum preview and peak-analysis helpers for FluxForge GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import optimize

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional GUI plotting dependency
    Figure = None

from fluxforge.analysis.flux_wire_analysis import (
    _covell_style_local_continuum_counts,
    _gilmore_moving_minimum_counts,
    _standards_tiered_counts,
)
from fluxforge.analysis.peak_finders import find_peaks_multi_method, get_peak_finder
from fluxforge.analysis.peakfit import fit_hypermet_peak, fit_single_peak
from fluxforge.cli import app as cli_app
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.nuclear_data_sources import load_gamma_identification_source
from fluxforge.io.artifacts import read_peak_report, read_spectrum_file
from fluxforge.io.spe import GammaSpectrum
from fluxforge_gui.mpl_helpers import apply_tight_layout, create_offscreen_figure
from fluxforge_gui.models import (
    GuiCalibrationFit,
    GuiCalibrationPoint,
    GuiDiagnosticPlot,
    GuiDiagnosticSeries,
    GuiEfficiencyCalibrationPoint,
    GuiManualRegion,
    GuiPeakCountingResult,
    GuiSpectrumPeak,
    GuiSpectrumPreview,
    GuiSpectrumSeries,
)


def _coerce_path_tokens(raw: str) -> tuple[Path, ...]:
    """Split a multi-path GUI field using semicolons or newlines."""

    tokens = [token.strip() for token in raw.replace("\n", ";").split(";")]
    return tuple(Path(token) for token in tokens if token)


def _estimate_local_fwhm_channels(counts: np.ndarray, peak_channel: int) -> float:
    """Estimate local FWHM in channels using a half-height walk."""

    if peak_channel < 0 or peak_channel >= len(counts):
        return 3.0
    peak_height = float(counts[peak_channel])
    if peak_height <= 0.0:
        return 3.0
    half_height = peak_height * 0.5
    left = int(peak_channel)
    right = int(peak_channel)
    while left > 0 and counts[left] > half_height:
        left -= 1
    while right < len(counts) - 1 and counts[right] > half_height:
        right += 1
    return float(max(right - left, 2))


def _series_to_spectrum(series: GuiSpectrumSeries) -> GammaSpectrum:
    """Convert a GUI series back into a spectrum object for artifact writing."""

    return GammaSpectrum(
        counts=np.asarray(series.counts, dtype=float),
        channels=np.asarray(series.channels, dtype=float),
        energies=np.asarray(series.energies_keV, dtype=float),
        calibration=(
            {"energy": list(series.calibration_coeffs)}
            if series.calibration_coeffs
            else {}
        ),
        spectrum_id=series.label,
    )


def combine_gui_spectrum_series(
    primary: GuiSpectrumSeries,
    secondary: GuiSpectrumSeries,
    *,
    operation: str,
    label: str,
) -> GuiSpectrumSeries:
    """Combine two buffers using arithmetic suitable for multi-spectrum workflows."""

    if len(primary.counts) != len(secondary.counts):
        raise ValueError(
            "Buffer arithmetic requires spectra with the same number of channels."
        )
    if not np.allclose(primary.channels, secondary.channels):
        raise ValueError("Buffer arithmetic requires matching channel grids.")

    left = np.asarray(primary.counts, dtype=float)
    right = np.asarray(secondary.counts, dtype=float)
    op = operation.strip().lower()
    if op == "sum":
        counts = left + right
    elif op == "subtract":
        counts = left - right
    elif op == "average":
        counts = 0.5 * (left + right)
    elif op == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            counts = np.where(np.abs(right) > 1e-12, left / right, 0.0)
    else:
        raise ValueError(f"Unknown buffer arithmetic operation: {operation}")

    return GuiSpectrumSeries(
        label=label,
        channels=np.asarray(primary.channels, dtype=float),
        energies_keV=np.asarray(primary.energies_keV, dtype=float),
        counts=np.asarray(counts, dtype=float),
        calibration_coeffs=tuple(primary.calibration_coeffs),
    )


def fit_gui_energy_calibration(
    points: Iterable[GuiCalibrationPoint], order: int = 1
) -> GuiCalibrationFit:
    """Fit an energy calibration polynomial from GUI-selected points."""

    calibration_points = list(points)
    if len(calibration_points) < max(2, order + 1):
        raise ValueError(
            "Not enough calibration points for the requested polynomial order."
        )
    channels = np.asarray([point.channel for point in calibration_points], dtype=float)
    reference_energies = np.asarray(
        [point.reference_energy_keV for point in calibration_points], dtype=float
    )
    coefficients_desc = np.polyfit(channels, reference_energies, deg=order)
    coefficients = tuple(float(value) for value in coefficients_desc[::-1])
    fitted = np.zeros_like(channels, dtype=float)
    for power, coeff in enumerate(coefficients):
        fitted += coeff * (channels**power)
    residuals = reference_energies - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((reference_energies - np.mean(reference_energies)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return GuiCalibrationFit(
        coefficients=coefficients,
        fitted_keV=fitted,
        residuals_keV=residuals,
        r_squared=r_squared,
    )


def build_calibration_residual_plot(
    points: Iterable[GuiCalibrationPoint], fit: GuiCalibrationFit
) -> GuiDiagnosticPlot:
    """Build a diagnostic residual plot for energy calibration review."""

    point_list = list(points)
    channels = np.asarray([point.channel for point in point_list], dtype=float)
    return GuiDiagnosticPlot(
        title="Calibration residuals",
        x_label="Channel",
        y_label="Residual (keV)",
        series=(
            GuiDiagnosticSeries(
                label="Residuals",
                x=channels,
                y=np.asarray(fit.residuals_keV, dtype=float),
                style="scatter",
                color="#8e44ad",
            ),
        ),
        reference_y=0.0,
    )


def build_peak_count_diagnostic_plot(
    result: GuiPeakCountingResult,
) -> GuiDiagnosticPlot | None:
    """Return a fit/ROI diagnostic plot for a counted peak."""

    if result.diagnostic_channels is None or result.diagnostic_counts is None:
        return None
    series = [
        GuiDiagnosticSeries(
            label="Counts",
            x=np.asarray(result.diagnostic_channels, dtype=float),
            y=np.asarray(result.diagnostic_counts, dtype=float),
            style="line",
            color="#1f77b4",
        )
    ]
    if result.diagnostic_model is not None:
        series.append(
            GuiDiagnosticSeries(
                label="Model",
                x=np.asarray(result.diagnostic_channels, dtype=float),
                y=np.asarray(result.diagnostic_model, dtype=float),
                style="line",
                color="#d62728",
            )
        )
    if result.diagnostic_residuals is not None:
        series.append(
            GuiDiagnosticSeries(
                label="Residuals",
                x=np.asarray(result.diagnostic_channels, dtype=float),
                y=np.asarray(result.diagnostic_residuals, dtype=float),
                style="scatter",
                color="#2ca02c",
            )
        )
    return GuiDiagnosticPlot(
        title=f"Peak diagnostic ({result.method})",
        x_label="Channel",
        y_label="Counts / residual",
        series=tuple(series),
        reference_y=0.0 if result.diagnostic_residuals is not None else None,
    )


def build_efficiency_fit_diagnostic_plot(
    points: Iterable[GuiEfficiencyCalibrationPoint],
    curve: EfficiencyCurve,
) -> GuiDiagnosticPlot:
    """Build an efficiency-fit diagnostic plot with measured points and fitted curve."""

    point_list = list(points)
    energies = np.asarray(
        [point.reference_energy_keV for point in point_list], dtype=float
    )
    efficiencies = np.asarray([point.efficiency for point in point_list], dtype=float)
    grid = np.linspace(float(np.min(energies)), float(np.max(energies)), 200)
    fitted = np.asarray(curve.efficiency(grid), dtype=float)
    return GuiDiagnosticPlot(
        title="Efficiency calibration",
        x_label="Energy (keV)",
        y_label="Efficiency",
        series=(
            GuiDiagnosticSeries(
                label="Measured",
                x=energies,
                y=efficiencies,
                style="scatter",
                color="#ff7f0e",
            ),
            GuiDiagnosticSeries(
                label="Fit", x=grid, y=fitted, style="line", color="#1f77b4"
            ),
        ),
        x_log=True,
        y_log=True,
    )


def parse_gui_constraint_matrix(raw: str, n_peaks: int) -> np.ndarray:
    """Parse a free-form amplitude constraint matrix from GUI text."""

    text = raw.strip()
    if not text:
        return np.eye(n_peaks, dtype=float)
    rows: list[list[float]] = []
    for line in text.replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = [token for token in line.replace(",", " ").split() if token]
        rows.append([float(token) for token in tokens])
    if not rows:
        return np.eye(n_peaks, dtype=float)
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Constraint matrix must be two-dimensional.")
    if matrix.shape[0] != n_peaks:
        raise ValueError(
            f"Constraint matrix must have {n_peaks} row(s), one per fitted peak."
        )
    if matrix.shape[1] < 1:
        raise ValueError("Constraint matrix must have at least one column.")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < min(matrix.shape):
        raise ValueError("Constraint matrix must have full column rank.")
    return matrix


def fit_gui_constrained_multiplet(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channels: list[int],
    *,
    fit_width: int,
    amplitude_constraint_matrix: np.ndarray,
    share_sigma: bool,
    fix_centroids: bool,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """Fit a multiplet using a free-form amplitude constraint matrix."""

    peak_channels = sorted(int(channel) for channel in peak_channels)
    if not peak_channels:
        raise ValueError(
            "At least one peak is required for constrained multiplet fitting."
        )
    if amplitude_constraint_matrix.shape[0] != len(peak_channels):
        raise ValueError("Constraint matrix row count must match the number of peaks.")

    idxs = [int(np.argmin(np.abs(channels - peak_ch))) for peak_ch in peak_channels]
    ch_lo = max(0, min(idxs) - fit_width)
    ch_hi = min(len(channels), max(idxs) + fit_width + 1)
    x = np.asarray(channels[ch_lo:ch_hi], dtype=float)
    y = np.asarray(counts[ch_lo:ch_hi], dtype=float)
    if x.size < len(peak_channels) * 3 + 4:
        raise ValueError("Fit region is too small for constrained multiplet fitting.")

    centroids0 = np.asarray(peak_channels, dtype=float)
    sigma0 = np.full(len(peak_channels), max(1.0, fit_width / 4.0), dtype=float)
    if share_sigma:
        sigma0 = np.array([float(np.mean(sigma0))], dtype=float)
    local_bg = np.array(
        [
            float(np.mean(y[: min(3, len(y))])),
            0.0,
        ]
    )
    observed_amplitudes = []
    for peak_ch in peak_channels:
        idx = int(np.argmin(np.abs(x - peak_ch)))
        observed_amplitudes.append(max(float(y[idx] - np.median(y)), 1.0))
    observed_amplitudes_arr = np.asarray(observed_amplitudes, dtype=float)
    free_amp0, *_ = np.linalg.lstsq(
        amplitude_constraint_matrix, observed_amplitudes_arr, rcond=None
    )
    free_amp0 = np.clip(free_amp0, 1e-3, None)

    params0: list[float] = list(free_amp0)
    if not fix_centroids:
        params0.extend(centroids0.tolist())
    params0.extend(sigma0.tolist())
    params0.extend(local_bg.tolist())

    lower: list[float] = [0.0] * len(free_amp0)
    upper: list[float] = [np.inf] * len(free_amp0)
    if not fix_centroids:
        lower.extend((centroids0 - fit_width).tolist())
        upper.extend((centroids0 + fit_width).tolist())
    if share_sigma:
        lower.append(0.5)
        upper.append(max(10.0, float(fit_width)))
    else:
        lower.extend([0.5] * len(peak_channels))
        upper.extend([max(10.0, float(fit_width))] * len(peak_channels))
    lower.extend([-np.inf, -np.inf])
    upper.extend([np.inf, np.inf])

    def unpack(
        vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        offset = 0
        free = np.asarray(vector[offset : offset + len(free_amp0)], dtype=float)
        offset += len(free_amp0)
        amplitudes = np.maximum(amplitude_constraint_matrix @ free, 0.0)
        if fix_centroids:
            centroids = centroids0.copy()
        else:
            centroids = np.asarray(
                vector[offset : offset + len(peak_channels)], dtype=float
            )
            offset += len(peak_channels)
        if share_sigma:
            sigma_value = float(vector[offset])
            sigmas = np.full(len(peak_channels), sigma_value, dtype=float)
            offset += 1
        else:
            sigmas = np.asarray(
                vector[offset : offset + len(peak_channels)], dtype=float
            )
            offset += len(peak_channels)
        slope = float(vector[offset])
        intercept = float(vector[offset + 1])
        return amplitudes, centroids, sigmas, slope, intercept

    def model(vector: np.ndarray) -> np.ndarray:
        amplitudes, centroids, sigmas, slope, intercept = unpack(vector)
        baseline = slope * x + intercept
        peaks_model = np.zeros_like(x, dtype=float)
        for amplitude, centroid, sigma in zip(amplitudes, centroids, sigmas):
            peaks_model += amplitude * np.exp(-((x - centroid) ** 2) / (2.0 * sigma**2))
        return baseline + peaks_model

    def residuals(vector: np.ndarray) -> np.ndarray:
        weights = 1.0 / np.maximum(np.sqrt(np.maximum(y, 0.0)), 1.0)
        return (model(vector) - y) * weights

    solution = optimize.least_squares(
        residuals,
        np.asarray(params0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=2000,
    )
    fitted = model(solution.x)
    amplitudes, centroids, sigmas, _slope, _intercept = unpack(solution.x)
    results: list[dict[str, float]] = []
    for amplitude, centroid, sigma in zip(amplitudes, centroids, sigmas):
        area = float(amplitude * sigma * np.sqrt(2.0 * np.pi))
        results.append(
            {
                "centroid": float(centroid),
                "amplitude": float(amplitude),
                "sigma": float(sigma),
                "fwhm": float(2.355 * sigma),
                "area": area,
            }
        )
    return results, x, y, fitted


def auto_detect_gui_peaks(
    series: GuiSpectrumSeries,
    *,
    finder_method: str,
    threshold_sigma: float,
    min_distance: int,
    identification_method: str,
    tolerance_keV: float,
    min_matches: int,
    identification_source_id: str = "decay_2012",
    custom_source_path: str | Path | None = None,
    min_intensity: float = 0.0,
) -> tuple[GuiSpectrumPeak, ...]:
    """Auto-detect and label peaks for GUI review."""

    counts = np.asarray(series.counts, dtype=float)
    if finder_method == "consensus":
        detected = find_peaks_multi_method(
            counts,
            methods=["simple", "window", "scipy"],
            consensus_threshold=max(1, min_matches),
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
        )
    else:
        finder_kwargs: dict[str, Any]
        if finder_method in {"simple", "window", "chunked"}:
            finder_kwargs = {
                "threshold_sigma": threshold_sigma,
                "min_distance": min_distance,
            }
        elif finder_method == "scipy":
            finder_kwargs = {
                "threshold_factor": max(1.1, 1.0 + threshold_sigma / 3.0),
                "distance": min_distance,
            }
        elif finder_method == "segmented":
            baseline = float(np.median(counts))
            spread = float(np.std(counts))
            height = max(baseline + threshold_sigma * max(spread, 1.0), baseline + 5.0)
            finder_kwargs = {
                "fit_window": max(4, min_distance),
                "gaussian_refine": True,
                "region_params": [
                    {
                        "height": height,
                        "prominence": max(5.0, 0.25 * height),
                        "distance": max(1, min_distance),
                    },
                    {
                        "height": height,
                        "prominence": max(5.0, 0.25 * height),
                        "distance": max(1, min_distance),
                    },
                    {
                        "height": height,
                        "prominence": max(5.0, 0.25 * height),
                        "distance": max(1, min_distance),
                    },
                ],
            }
        else:
            finder_kwargs = {"min_distance": min_distance}
        finder = get_peak_finder(finder_method, **finder_kwargs)
        detected = finder.find_peaks(counts)

    database = load_gamma_identification_source(
        identification_source_id, custom_path=custom_source_path
    )
    candidate_hits: dict[str, list[tuple[float, Any]]] = {}
    for peak in detected:
        if not (0 <= peak.index < len(series.energies_keV)):
            continue
        energy = float(series.energies_keV[peak.index])
        matches = database.find_matches(
            energy, tolerance_keV=tolerance_keV, min_intensity=min_intensity
        )
        for nuclide, line in matches:
            candidate_hits.setdefault(nuclide, []).append((energy, line))
    consensus_candidates = {
        nuclide: hits
        for nuclide, hits in candidate_hits.items()
        if len(hits) >= max(1, min_matches)
    }
    hybrid_scores = {
        nuclide: float(len(hits))
        + float(sum(line.intensity * line.norm for _, line in hits))
        for nuclide, hits in candidate_hits.items()
    }

    rows: list[GuiSpectrumPeak] = []
    for peak in detected:
        energy = float(series.energies_keV[peak.index])
        label = ""
        matches = database.find_matches(
            energy, tolerance_keV=tolerance_keV, min_intensity=min_intensity
        )
        if identification_method == "line_match" and matches:
            nuclide, line = matches[0]
            label = f"{nuclide} {line.energy_keV:.1f}"
        elif identification_method == "nuclide_consensus":
            best_label = ""
            best_intensity = -1.0
            for nuclide, candidate_matches in consensus_candidates.items():
                for matched_energy, matched_line in candidate_matches:
                    if abs(float(matched_energy) - energy) <= tolerance_keV:
                        intensity = float(matched_line.intensity * matched_line.norm)
                        if intensity > best_intensity:
                            best_intensity = intensity
                            best_label = f"{nuclide} {matched_line.energy_keV:.1f}"
            label = best_label
        elif identification_method == "hybrid_ranked" and matches:
            ranked = sorted(
                matches,
                key=lambda item: (
                    hybrid_scores.get(item[0], 0.0),
                    item[1].intensity * item[1].norm,
                ),
                reverse=True,
            )
            nuclide, line = ranked[0]
            label = f"{nuclide} {line.energy_keV:.1f}"
        rows.append(
            GuiSpectrumPeak(
                energy_keV=energy,
                area=float(peak.area or peak.value),
                channel=int(peak.index),
                label=label,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.energy_keV))


def count_gui_peak(
    series: GuiSpectrumSeries, peak: GuiSpectrumPeak, method: str
) -> GuiPeakCountingResult:
    """Count a selected peak using one of the GUI-exposed counting methods."""

    counts = np.asarray(series.counts, dtype=float)
    peak_channel = int(
        round(
            float(
                peak.channel
                if peak.channel is not None
                else np.interp(peak.energy_keV, series.energies_keV, series.channels)
            )
        )
    )
    peak_channel = max(0, min(len(counts) - 1, peak_channel))
    fwhm_channels = _estimate_local_fwhm_channels(counts, peak_channel)
    uncertainty = np.sqrt(np.maximum(counts, 0.0))
    method_key = method.strip().lower()

    if method_key == "gaussian_fit":
        result = fit_single_peak(
            series.channels,
            counts,
            peak_channel=peak_channel,
            fit_width=int(max(6, round(2.5 * fwhm_channels))),
            background_model="linear",
        )
        lo, hi = result.fit_region
        gross = float(np.sum(counts[int(lo) : int(hi) + 1]))
        roi_x = np.asarray(series.channels[int(lo) : int(hi) + 1], dtype=float)
        roi_counts = np.asarray(counts[int(lo) : int(hi) + 1], dtype=float)
        model = np.asarray(result.background, dtype=float) + np.asarray(
            result.peak.evaluate(roi_x), dtype=float
        )
        return GuiPeakCountingResult(
            method=method_key,
            net_counts=float(result.net_counts),
            net_uncertainty=float(result.net_counts_uncertainty),
            gross_counts=gross,
            gross_uncertainty=float(np.sqrt(max(gross, 0.0))),
            roi_bounds=(int(lo), int(hi)),
            diagnostic_channels=roi_x,
            diagnostic_counts=roi_counts,
            diagnostic_model=model,
            diagnostic_residuals=np.asarray(result.residuals, dtype=float),
        )
    if method_key == "hypermet":
        hypermet_peak, result = fit_hypermet_peak(
            series.channels,
            counts,
            peak_channel=peak_channel,
            fit_width=int(max(8, round(3.0 * fwhm_channels))),
            enable_tail=True,
            enable_step=True,
        )
        lo, hi = result.fit_region
        gross = float(np.sum(counts[int(lo) : int(hi) + 1]))
        roi_x = np.asarray(series.channels[int(lo) : int(hi) + 1], dtype=float)
        roi_counts = np.asarray(counts[int(lo) : int(hi) + 1], dtype=float)
        model = np.asarray(result.background, dtype=float) + np.asarray(
            result.peak.evaluate(roi_x), dtype=float
        )
        return GuiPeakCountingResult(
            method=method_key,
            net_counts=float(hypermet_peak.area),
            net_uncertainty=float(
                max(
                    result.net_counts_uncertainty, np.sqrt(max(hypermet_peak.area, 0.0))
                )
            ),
            gross_counts=gross,
            gross_uncertainty=float(np.sqrt(max(gross, 0.0))),
            roi_bounds=(int(lo), int(hi)),
            diagnostic_channels=roi_x,
            diagnostic_counts=roi_counts,
            diagnostic_model=model,
            diagnostic_residuals=np.asarray(result.residuals, dtype=float),
        )

    if method_key == "covell_local":
        net, unc, gross, _, bounds = _covell_style_local_continuum_counts(
            counts, peak_channel, fwhm_channels, spectrum_uncertainty=uncertainty
        )
    elif method_key == "gilmore_minimum":
        net, unc, gross, _, bounds = _gilmore_moving_minimum_counts(
            counts, peak_channel, fwhm_channels, spectrum_uncertainty=uncertainty
        )
    elif method_key == "iec_tiered":
        net, unc, gross, _ = _standards_tiered_counts(
            raw_counts=counts,
            raw_counts_uncertainty=uncertainty,
            peak_channel=peak_channel,
            fwhm_channels=fwhm_channels,
            group_size=1,
            fit_net=0.0,
            fit_unc=0.0,
        )
        half_width = int(max(3, round(2.0 * fwhm_channels)))
        bounds = (
            max(0, peak_channel - half_width),
            min(len(counts) - 1, peak_channel + half_width),
        )
    else:
        raise ValueError(f"Unknown GUI peak counting method: {method}")
    gross_unc = float(np.sqrt(max(gross, 0.0)))
    roi_x = np.asarray(
        series.channels[int(bounds[0]) : int(bounds[1]) + 1], dtype=float
    )
    roi_counts = np.asarray(counts[int(bounds[0]) : int(bounds[1]) + 1], dtype=float)
    return GuiPeakCountingResult(
        method=method_key,
        net_counts=float(net),
        net_uncertainty=float(unc),
        gross_counts=float(gross),
        gross_uncertainty=gross_unc,
        roi_bounds=(int(bounds[0]), int(bounds[1])),
        diagnostic_channels=roi_x,
        diagnostic_counts=roi_counts,
    )


def _gui_spectrum_energies(spectrum: GammaSpectrum) -> np.ndarray:
    """Return energy values suitable for plotting a spectrum."""

    if spectrum.energies is not None:
        return np.asarray(spectrum.energies, dtype=float)
    if spectrum.calibration:
        return np.asarray(spectrum.calibrate_channels(), dtype=float)
    return np.asarray(spectrum.channels, dtype=float)


def _load_gui_spectrum(path: Path) -> GammaSpectrum:
    """Load either a raw spectrum or a serialized spectrum artifact for the GUI."""

    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        payload = read_spectrum_file(path)
        spectrum_payload = payload.get("spectrum", payload)
        return GammaSpectrum.from_dict(spectrum_payload)
    return cli_app._load_spectrum_from_path(path, validate=False)


def _load_gui_peaks(path: Path) -> tuple[GuiSpectrumPeak, ...]:
    """Load peaks from a FluxForge peak artifact or a legacy lightweight JSON file."""

    payload = read_peak_report(path)
    raw_peaks = payload.get("peaks", [])
    peaks: list[GuiSpectrumPeak] = []
    for item in raw_peaks:
        if isinstance(item, dict):
            energy_keV = item.get("energy_keV")
            if energy_keV is None:
                continue
            area = item.get("area", item.get("net_counts", item.get("raw_counts", 0.0)))
            channel = item.get("channel")
            label = (
                item.get("label") or item.get("isotope") or item.get("nuclide") or ""
            )
            peaks.append(
                GuiSpectrumPeak(
                    energy_keV=float(energy_keV),
                    area=float(area or 0.0),
                    channel=int(channel) if channel is not None else None,
                    label=str(label),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            channel_value = item[0]
            energy_value = item[1]
            area_value = item[2]
            label_value = item[3] if len(item) >= 4 and item[3] is not None else ""
            if energy_value is None:
                continue
            peaks.append(
                GuiSpectrumPeak(
                    energy_keV=float(energy_value),
                    area=float(area_value or 0.0),
                    channel=int(channel_value) if channel_value is not None else None,
                    label=str(label_value),
                )
            )
    peaks.sort(key=lambda item: item.energy_keV)
    return tuple(peaks)


def build_gui_spectrum_preview(
    spectrum_path: str | Path,
    *,
    overlay_paths: Iterable[str | Path] = (),
    peaks_path: str | Path | None = None,
) -> GuiSpectrumPreview:
    """Build preview data for the GUI spectrum viewer tab."""

    primary_path = Path(spectrum_path)
    primary_spectrum = _load_gui_spectrum(primary_path)
    primary = GuiSpectrumSeries(
        label=primary_spectrum.spectrum_id or primary_path.name,
        channels=np.asarray(primary_spectrum.channels, dtype=float),
        energies_keV=_gui_spectrum_energies(primary_spectrum),
        counts=np.asarray(primary_spectrum.counts, dtype=float),
        calibration_coeffs=tuple(
            float(value) for value in primary_spectrum.calibration.get("energy", ())
        ),
    )
    overlays: list[GuiSpectrumSeries] = []
    for overlay_path in overlay_paths:
        overlay = Path(overlay_path)
        spectrum = _load_gui_spectrum(overlay)
        overlays.append(
            GuiSpectrumSeries(
                label=spectrum.spectrum_id or overlay.name,
                channels=np.asarray(spectrum.channels, dtype=float),
                energies_keV=_gui_spectrum_energies(spectrum),
                counts=np.asarray(spectrum.counts, dtype=float),
                calibration_coeffs=tuple(
                    float(value) for value in spectrum.calibration.get("energy", ())
                ),
            )
        )
    peaks = _load_gui_peaks(Path(peaks_path)) if peaks_path else ()
    return GuiSpectrumPreview(
        primary=primary, overlays=tuple(overlays), peaks=tuple(peaks)
    )


def render_gui_spectrum_preview(
    preview: GuiSpectrumPreview,
    *,
    manual_regions: Iterable[GuiManualRegion] = (),
    selected_peak_energy_keV: float | None = None,
    selected_region_label: str | None = None,
    diagnostic_plot: GuiDiagnosticPlot | None = None,
    y_log: bool = False,
    x_min_keV: float | None = None,
    x_max_keV: float | None = None,
    figure: Figure | None = None,
):
    """Render a spectrum preview with optional overlays and peak markers."""

    if Figure is None:
        raise ImportError(
            "matplotlib is required for the FluxForge GUI spectrum preview"
        )

    fig = create_offscreen_figure(figsize=(8.8, 4.8), dpi=100, figure=figure)
    fig.clear()
    if diagnostic_plot is not None:
        gridspec = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.25], hspace=0.18)
        ax = fig.add_subplot(gridspec[0, 0])
        diag_ax = fig.add_subplot(gridspec[1, 0])
    else:
        ax = fig.add_subplot(111)
        diag_ax = None

    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    all_series = (preview.primary, *preview.overlays)
    for index, series in enumerate(all_series):
        ax.plot(
            series.energies_keV,
            series.counts,
            linewidth=1.0 if index == 0 else 0.9,
            alpha=0.95 if index == 0 else 0.72,
            color=palette[index % len(palette)],
            label=series.label,
        )

    for idx, region in enumerate(manual_regions):
        lo_keV, hi_keV = sorted((float(region.left_keV), float(region.right_keV)))
        is_selected = (
            selected_region_label is not None and region.label == selected_region_label
        )
        ax.axvspan(
            lo_keV,
            hi_keV,
            color="#8e44ad",
            alpha=0.18 if is_selected else 0.10,
            zorder=1,
        )
        ax.axvline(
            lo_keV,
            color="#8e44ad",
            linewidth=1.3 if is_selected else 0.8,
            alpha=0.50 if is_selected else 0.30,
        )
        ax.axvline(
            hi_keV,
            color="#8e44ad",
            linewidth=1.3 if is_selected else 0.8,
            alpha=0.50 if is_selected else 0.30,
        )
        ax.text(
            (lo_keV + hi_keV) * 0.5,
            (
                float(np.nanmax(preview.primary.counts)) * 0.82
                if preview.primary.counts.size
                else 1.0
            ),
            region.label or f"ROI {idx + 1}",
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
            color="#6c3483",
        )

    if preview.peaks:
        peak_energies = np.asarray(
            [peak.energy_keV for peak in preview.peaks], dtype=float
        )
        peak_counts = np.interp(
            peak_energies,
            preview.primary.energies_keV,
            preview.primary.counts,
            left=np.nan,
            right=np.nan,
        )
        ax.scatter(
            peak_energies, peak_counts, color="#c44e52", s=20, zorder=5, label="Peaks"
        )
        y_top = (
            float(np.nanmax(preview.primary.counts))
            if preview.primary.counts.size
            else 1.0
        )
        text_y = y_top * (0.92 if y_top > 0.0 else 1.0)
        for peak in preview.peaks[:20]:
            ax.axvline(peak.energy_keV, color="#c44e52", linewidth=0.8, alpha=0.18)
            label = peak.label or f"{peak.energy_keV:.1f} keV"
            ax.text(
                peak.energy_keV,
                text_y,
                label,
                rotation=90,
                ha="center",
                va="top",
                fontsize=7,
                color="#7a1f24",
            )

    if selected_peak_energy_keV is not None:
        ax.axvline(
            float(selected_peak_energy_keV),
            color="#f1c40f",
            linewidth=1.6,
            alpha=0.95,
            linestyle="--",
            label="Selected peak",
        )

    ax.set_title("FluxForge Spectrum Viewer")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if y_log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    if x_min_keV is not None or x_max_keV is not None:
        ax.set_xlim(left=x_min_keV, right=x_max_keV)
    if len(all_series) > 1 or preview.peaks:
        ax.legend(loc="upper right", fontsize=8)
    if diag_ax is not None and diagnostic_plot is not None:
        for series in diagnostic_plot.series:
            if series.style == "scatter":
                diag_ax.scatter(
                    series.x, series.y, s=18, color=series.color, label=series.label
                )
            else:
                diag_ax.plot(
                    series.x,
                    series.y,
                    color=series.color,
                    linewidth=1.1,
                    label=series.label,
                )
        if diagnostic_plot.reference_y is not None:
            diag_ax.axhline(
                diagnostic_plot.reference_y,
                color="#7f8c8d",
                linewidth=0.9,
                linestyle="--",
                alpha=0.75,
            )
        diag_ax.set_title(diagnostic_plot.title, fontsize=9)
        diag_ax.set_xlabel(diagnostic_plot.x_label)
        diag_ax.set_ylabel(diagnostic_plot.y_label)
        diag_ax.grid(True, alpha=0.25, linewidth=0.5)
        if diagnostic_plot.x_log:
            diag_ax.set_xscale("log")
        if diagnostic_plot.y_log:
            diag_ax.set_yscale("log")
        if diagnostic_plot.series:
            diag_ax.legend(loc="best", fontsize=7)
    if diagnostic_plot is not None:
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.94, hspace=0.22)
    else:
        apply_tight_layout(fig)
    return fig, ax


def save_gui_spectrum_preview_image(
    preview: GuiSpectrumPreview,
    output_path: str | Path,
    *,
    manual_regions: Iterable[GuiManualRegion] = (),
    selected_region_label: str | None = None,
    diagnostic_plot: GuiDiagnosticPlot | None = None,
    y_log: bool = False,
    x_min_keV: float | None = None,
    x_max_keV: float | None = None,
) -> Path:
    """Save a GUI-style spectrum preview image for smoke testing and demos."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = render_gui_spectrum_preview(
        preview,
        manual_regions=manual_regions,
        selected_region_label=selected_region_label,
        diagnostic_plot=diagnostic_plot,
        y_log=y_log,
        x_min_keV=x_min_keV,
        x_max_keV=x_max_keV,
    )
    figure.savefig(output, dpi=140)
    return output
