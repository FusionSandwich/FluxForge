#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from ._compare_common import ensure_fluxforge_src
except ImportError:  # Script execution path
    from _compare_common import ensure_fluxforge_src

ROOT = ensure_fluxforge_src(__file__)

from fluxforge.analysis.flux_wire_analysis import (  # noqa: E402
    GammaLine,
    _calibration_slope,
    _covell_style_local_continuum_counts,
    _expand_to_continuum_roi,
    _gilmore_moving_minimum_counts,
    _group_expected_lines_for_fit,
    _prepare_flux_wire_data_with_profile,
    _qg_style_linear_continuum_counts,
    _snip_background_from_signed_counts,
    _standards_tiered_counts,
    apply_qg_report_parity,
    analyze_raw_spectrum_targeted,
)
from fluxforge.analysis.peakfit import (  # noqa: E402
    PeakFitResult,
    expgauss,
    fit_hypermet_peak,
    fit_multiple_peaks,
    fit_single_peak,
    gauss_dbl_exp,
    poisson_neg_log_likelihood,
)
from fluxforge.analysis.spectrum_math import subtract_measured_background  # noqa: E402
from fluxforge._tensorflow_env import configure_tensorflow_cuda_runtime  # noqa: E402
from fluxforge.examples.rafm_workflow import (  # noqa: E402
    default_paths,
    match_peak,
    qg_reference_peaks,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import (
    FluxWireData,
    read_processed_txt,
    read_raw_asc,
)  # noqa: E402


QG_METHOD_NAME = "qg"
QG_METHOD_ALIASES = {"qg", "qg_hybrid", "current_hybrid", "quantumgold", "quantum_gold"}


def normalize_benchmark_method_name(value: str) -> str:
    method = str(value).strip().lower()
    if method in QG_METHOD_ALIASES:
        return QG_METHOD_NAME
    return method


METHODS = [
    QG_METHOD_NAME,
    "tight_continuum",
    "expanded_continuum",
    "covell_local",
    "gilmore_minimum",
    "iec_tiered",
    "maestro_trapezoid",
    "gaussian_fit",
    "hypermet_hybrid",
    "pml_dbl_exp",
    "ml_selector",
]
ML_CANDIDATES = [
    "tight_continuum",
    "expanded_continuum",
    "iec_tiered",
    "gaussian_fit",
    "hypermet_hybrid",
]

USER_VISIBLE_METHOD_CHOICES = [
    QG_METHOD_NAME,
    "quantum_gold",
    *[method for method in METHODS if method != QG_METHOD_NAME],
]


@dataclass
class LineContext:
    reference: Dict[str, Any]
    peak_channel: int
    peak_energy_keV: float
    fwhm_ch: float
    fwhm_keV: float
    slope_keV_per_ch: float
    group_size: int
    fit_net: float
    fit_unc: float
    fit_success: bool
    covell_net: float
    covell_unc: float
    covell_gross: float
    gilmore_net: float
    gilmore_unc: float
    gilmore_gross: float
    iec_net: float
    iec_unc: float
    iec_gross: float
    maestro_net: float
    maestro_gross: float
    hypermet_net: float
    hypermet_unc: float
    hypermet_success: bool
    pml_dbl_exp_net: float
    pml_dbl_exp_unc: float
    pml_dbl_exp_success: bool
    tight_net: float
    tight_unc: float
    tight_gross: float
    expanded_net: float
    expanded_unc: float
    expanded_gross: float
    expanded_bg: float
    current_net: float
    current_unc: float
    current_gross: float
    peak_height: float
    left_edge: float
    right_edge: float
    local_background: float
    local_slope: float
    asymmetry: float
    centroid_shift_ch: float
    tight_to_expanded_ratio: float
    fit_to_tight_ratio: float
    hypermet_to_tight_ratio: float
    neighbor_distance_keV: float


def _safe_rel(pred: float, ref: float) -> Optional[float]:
    if ref <= 0.0:
        return None
    return (pred - ref) / ref


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _line_key(ref_peak: Dict[str, Any]) -> str:
    return f"{ref_peak['isotope']}@{float(ref_peak['energy_keV']):.2f}"


def build_expected_lines(reference_data: FluxWireData) -> List[GammaLine]:
    expected: List[GammaLine] = []
    for row in qg_reference_peaks(reference_data):
        intensity = float(row.get("rad_int_fraction") or 0.0)
        if intensity <= 0.0:
            intensity = max(float(row.get("rad_int_percent") or 0.0) / 100.0, 1.0e-6)
        expected.append(
            GammaLine(
                energy_keV=float(row["energy_keV"]),
                intensity=max(intensity, 1.0e-6),
                isotope=str(row["isotope"]),
            )
        )
    expected.sort(key=lambda line: line.energy_keV)
    return expected


def _sample_intersections(raw_dir: Path, qg_dir: Path) -> List[Tuple[str, Path, Path]]:
    raw_files = {path.stem: path for path in raw_dir.glob("*.ASC")}
    qg_files = {path.stem: path for path in qg_dir.glob("*.txt")}
    common = sorted(set(raw_files) & set(qg_files))
    return [(stem, raw_files[stem], qg_files[stem]) for stem in common]


def _prepare_counts(
    raw_data: FluxWireData,
    background_data: FluxWireData,
    profile_name: str,
) -> Tuple[FluxWireData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    analysis_data = _prepare_flux_wire_data_with_profile(raw_data, profile_name)
    if analysis_data.spectrum is None:
        raise ValueError(f"No raw spectrum for {raw_data.source_file}")
    if background_data.spectrum is None:
        raise ValueError("Background spectrum missing")

    adjusted = subtract_measured_background(
        analysis_data.spectrum,
        background_data.spectrum,
        mode="live",
        negative_policy="hybrid",
        warn_missing=True,
    )
    signed_counts = np.asarray(adjusted.counts, dtype=float)
    raw_counts = np.asarray(analysis_data.spectrum.counts, dtype=float)
    _, counts_for_search, _ = _snip_background_from_signed_counts(
        signed_counts, n_iterations=24
    )
    raw_unc = np.sqrt(np.maximum(raw_counts, 0.0))
    return analysis_data, raw_counts, raw_unc, signed_counts, counts_for_search


def _seed_group(
    analysis_data: FluxWireData,
    counts_for_search: np.ndarray,
    group: Sequence[GammaLine],
) -> List[Tuple[GammaLine, int, float, float, float]]:
    seeds: List[Tuple[GammaLine, int, float, float, float]] = []
    for line in group:
        ch_est = analysis_data.energy_to_channel(line.energy_keV)
        if ch_est < 0 or ch_est >= len(counts_for_search):
            continue
        slope = _calibration_slope(analysis_data.energy_calibration, ch_est)
        fwhm_keV = analysis_data.fwhm_at_energy(line.energy_keV)
        fwhm_ch = max(fwhm_keV / slope, 1.0)
        search_half = int(max(3, round(1.5 * fwhm_ch)))
        ch_lo = max(0, ch_est - search_half)
        ch_hi = min(len(counts_for_search) - 1, ch_est + search_half)
        peak_channel = ch_lo + int(np.argmax(counts_for_search[ch_lo : ch_hi + 1]))
        seeds.append(
            (line, int(peak_channel), float(slope), float(fwhm_keV), float(fwhm_ch))
        )
    return seeds


def _fit_group(
    channels: np.ndarray,
    counts_for_search: np.ndarray,
    seeds: Sequence[Tuple[GammaLine, int, float, float, float]],
) -> List[Optional[PeakFitResult]]:
    if not seeds:
        return []
    fit_peak_channels = [seed[1] for seed in seeds]
    max_fwhm_ch = max(seed[4] for seed in seeds)
    fit_results: List[Optional[PeakFitResult]] = [None] * len(seeds)
    if len(seeds) > 1:
        span = max(fit_peak_channels) - min(fit_peak_channels)
        fit_width = int(max(8, round(0.5 * span + 3.0 * max_fwhm_ch)))
        multiplet_results = fit_multiple_peaks(
            channels=channels,
            counts=counts_for_search,
            peak_channels=fit_peak_channels,
            fit_width=fit_width,
            background_model="linear",
            share_sigma=True,
        )
        if len(multiplet_results) == len(seeds):
            fit_results = multiplet_results
    else:
        fit_width = int(max(6, round(2.5 * seeds[0][4])))
        fit_results[0] = fit_single_peak(
            channels=channels,
            counts=counts_for_search,
            peak_channel=fit_peak_channels[0],
            fit_width=fit_width,
            background_model="linear",
        )
    return fit_results


def _neighbor_distance_keV(
    expected_lines: Sequence[GammaLine], line: GammaLine
) -> float:
    others = [
        abs(other.energy_keV - line.energy_keV)
        for other in expected_lines
        if other is not line
    ]
    return float(min(others)) if others else 1.0e9


def _maestro_trapezoid_counts(
    counts: np.ndarray, c1: int, c2: int
) -> Tuple[float, float, float]:
    c1 = max(0, int(c1))
    c2 = min(len(counts), int(c2))
    if c2 <= c1:
        return 0.0, 0.0, 0.0
    low_start = max(0, c1 - 2)
    low_sum = float(np.sum(counts[low_start:c1])) if c1 > low_start else 0.0
    high_end = min(len(counts), c2 + 2)
    high_sum = float(np.sum(counts[c2:high_end])) if high_end > c2 else 0.0
    width = c2 - c1 + 1
    bg = float((low_sum + high_sum) * (width / 6.0))
    gross = float(np.sum(counts[c1:c2]))
    net = gross - bg
    return net, gross, bg


def _fit_pml_tail_model(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channel: int,
    fwhm_ch: float,
) -> Tuple[float, float, bool]:
    sigma0 = max(float(fwhm_ch) / 2.355, 0.8)
    half_width = max(10, int(round(5.0 * max(fwhm_ch, 1.0))))
    lo = max(0, int(round(peak_channel)) - half_width)
    hi = min(len(counts), int(round(peak_channel)) + half_width + 1)
    x_fit = np.asarray(channels[lo:hi], dtype=float)
    y_fit = np.asarray(counts[lo:hi], dtype=float)
    if x_fit.size < 8:
        return 0.0, 0.0, False

    bg_est = float(0.5 * (np.mean(y_fit[:3]) + np.mean(y_fit[-3:])))
    amp0 = float(max(np.max(y_fit) - bg_est, 1.0))

    def model(params: np.ndarray) -> np.ndarray:
        amp, cent, sig, lr, ls, lc, rr, rs, rc, m, b = params
        sig = max(sig, 0.2)
        lr = max(lr, 0.0)
        rr = max(rr, 0.0)
        ls = max(ls, 1.0e-4)
        rs = max(rs, 1.0e-4)
        lc = max(lc, 0.2)
        rc = max(rc, 0.2)
        return (
            gauss_dbl_exp(x_fit, amp, cent, sig, lr, ls, lc, rr, rs, rc) + m * x_fit + b
        )

    p0 = np.array(
        [
            amp0,
            float(peak_channel),
            sigma0,
            0.08,
            0.02,
            1.0,
            0.02,
            0.02,
            1.0,
            0.0,
            bg_est,
        ],
        dtype=float,
    )

    def objective(params: np.ndarray) -> float:
        y_model = np.maximum(model(params), 1.0e-10)
        return float(poisson_neg_log_likelihood(y_model, y_fit))

    try:
        from scipy import optimize

        result = optimize.minimize(
            objective,
            p0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1.0e-6, "fatol": 1.0e-6},
        )
        if not result.success:
            return 0.0, 0.0, False
        params = np.asarray(result.x, dtype=float)
        peak_only = gauss_dbl_exp(
            x_fit,
            params[0],
            params[1],
            max(params[2], 0.2),
            max(params[3], 0.0),
            max(params[4], 1.0e-4),
            max(params[5], 0.2),
            max(params[6], 0.0),
            max(params[7], 1.0e-4),
            max(params[8], 0.2),
        )
        net = float(np.trapezoid(np.maximum(peak_only, 0.0), x_fit))
        unc = float(math.sqrt(max(net, 0.0)))
        return net, unc, True
    except Exception:
        return 0.0, 0.0, False


def compute_line_contexts(
    raw_data: FluxWireData,
    reference_data: FluxWireData,
    background_data: FluxWireData,
    config: Dict[str, Any],
) -> Dict[str, LineContext]:
    profile_name = str(config["profile_name"])
    analysis_data, raw_counts, raw_unc, _, counts_for_search = _prepare_counts(
        raw_data, background_data, profile_name
    )
    expected_lines = build_expected_lines(reference_data)
    groups = _group_expected_lines_for_fit(expected_lines, analysis_data)
    raw_channels = (
        analysis_data.spectrum.channels
        if analysis_data.spectrum is not None
        else np.arange(len(raw_counts))
    )
    contexts: Dict[str, LineContext] = {}

    for group in groups:
        seeds = _seed_group(analysis_data, counts_for_search, group)
        if not seeds:
            continue
        fit_results = _fit_group(raw_channels, counts_for_search, seeds)

        for (line, peak_channel, slope, fwhm_keV, fwhm_ch), fit in zip(
            seeds, fit_results
        ):
            peak_energy_keV = float(analysis_data.channel_to_energy(peak_channel))
            fit_net = 0.0
            fit_unc = 0.0
            fit_success = False
            centroid_shift_ch = 0.0
            if (
                fit is not None
                and getattr(fit, "success", False)
                and fit.net_counts > 0.0
            ):
                fit_success = True
                fit_net = float(fit.net_counts)
                fit_unc = float(
                    fit.net_counts_uncertainty
                    if fit.net_counts_uncertainty > 0.0
                    else math.sqrt(max(fit_net, 0.0))
                )
                peak_channel = int(round(float(fit.peak.centroid)))
                centroid_shift_ch = float(
                    fit.peak.centroid - analysis_data.energy_to_channel(line.energy_keV)
                )
                fwhm_ch = max(float(fit.peak.fwhm), 1.0)
                peak_energy_keV = float(
                    analysis_data.channel_to_energy(float(fit.peak.centroid))
                )
                fwhm_keV = float(fwhm_ch * slope)

            expanded_lo, expanded_hi = _expand_to_continuum_roi(
                raw_counts,
                int(round(peak_channel)),
                fwhm_ch,
                min_half_width_fwhm=2.0,
                max_half_width_channels=32,
            )
            expanded_net, expanded_unc, expanded_gross, expanded_bg = (
                _qg_style_linear_continuum_counts(
                    raw_counts,
                    expanded_lo,
                    expanded_hi,
                    spectrum_uncertainty=raw_unc,
                )
            )

            tight_hw_net = max(4, int(round(2.5 * fwhm_ch)))
            tight_lo_net = max(0, int(round(peak_channel)) - tight_hw_net)
            tight_hi_net = min(
                len(raw_counts) - 1, int(round(peak_channel)) + tight_hw_net
            )
            tight_net, tight_unc, _, _ = _qg_style_linear_continuum_counts(
                raw_counts,
                tight_lo_net,
                tight_hi_net,
                spectrum_uncertainty=raw_unc,
            )

            tight_hw_gross = max(2, int(round(1.5 * fwhm_ch)))
            tight_lo_gross = max(0, int(round(peak_channel)) - tight_hw_gross)
            tight_hi_gross = min(
                len(raw_counts) - 1, int(round(peak_channel)) + tight_hw_gross
            )
            _, _, tight_gross, _ = _qg_style_linear_continuum_counts(
                raw_counts,
                tight_lo_gross,
                tight_hi_gross,
                spectrum_uncertainty=raw_unc,
            )
            covell_net, covell_unc, covell_gross, _, _ = (
                _covell_style_local_continuum_counts(
                    raw_counts,
                    peak_channel,
                    fwhm_ch,
                    roi_width_fwhm=4.0,
                    background_width_channels=1,
                    background_gap_fwhm=0.0,
                    spectrum_uncertainty=raw_unc,
                )
            )
            gilmore_net, gilmore_unc, gilmore_gross, _, _ = (
                _gilmore_moving_minimum_counts(
                    raw_counts,
                    peak_channel,
                    fwhm_ch,
                    min_half_width_fwhm=2.0,
                    max_half_width_channels=32,
                    spectrum_uncertainty=raw_unc,
                )
            )
            iec_net, iec_unc, iec_gross, _ = _standards_tiered_counts(
                raw_counts=raw_counts,
                raw_counts_uncertainty=raw_unc,
                peak_channel=peak_channel,
                fwhm_channels=fwhm_ch,
                group_size=len(group),
                fit_net=fit_net,
                fit_unc=fit_unc,
                roi_width_fwhm=4.0,
                background_width_channels=1,
                background_gap_fwhm=0.0,
            )
            maestro_net, maestro_gross, _ = _maestro_trapezoid_counts(
                raw_counts,
                tight_lo_net,
                tight_hi_net + 1,
            )

            hypermet_net = 0.0
            hypermet_unc = 0.0
            hypermet_success = False
            pml_dbl_exp_net = 0.0
            pml_dbl_exp_unc = 0.0
            pml_dbl_exp_success = False
            if len(group) == 1:
                try:
                    fit_width = int(max(8, round(3.5 * fwhm_ch)))
                    hypermet_peak, hypermet_result = fit_hypermet_peak(
                        channels=raw_channels,
                        counts=counts_for_search,
                        peak_channel=int(round(peak_channel)),
                        fit_width=fit_width,
                        enable_tail=True,
                        enable_step=True,
                        initial_sigma=max(fwhm_ch / 2.355, 0.8),
                    )
                    if hypermet_result.success and hypermet_peak.area > 0.0:
                        hypermet_success = True
                        hypermet_net = float(hypermet_peak.area)
                        hypermet_unc = float(
                            max(
                                hypermet_result.net_counts_uncertainty,
                                math.sqrt(max(hypermet_net, 0.0)),
                            )
                        )
                except Exception:
                    hypermet_success = False
                pml_dbl_exp_net, pml_dbl_exp_unc, pml_dbl_exp_success = (
                    _fit_pml_tail_model(
                        raw_channels,
                        raw_counts,
                        int(round(peak_channel)),
                        fwhm_ch,
                    )
                )

            if len(group) > 1 and fit_net > 0.0:
                current_net = fit_net
                current_unc = fit_unc
                current_gross = float(tight_gross)
            else:
                if expanded_net > 50000.0:
                    current_net = float(expanded_net)
                    current_unc = float(expanded_unc)
                    current_gross = float(expanded_gross)
                else:
                    current_net = float(tight_net)
                    current_unc = float(tight_unc)
                    current_gross = float(tight_gross)

            local_lo = max(0, int(round(peak_channel - 2 * fwhm_ch)))
            local_hi = min(len(raw_counts) - 1, int(round(peak_channel + 2 * fwhm_ch)))
            local_slice = raw_counts[local_lo : local_hi + 1]
            left_edge = float(raw_counts[tight_lo_net])
            right_edge = float(raw_counts[tight_hi_net])
            local_background = 0.5 * (left_edge + right_edge)
            peak_height = float(raw_counts[int(round(peak_channel))])
            asymmetry = 0.0
            if local_slice.size > 0:
                half = local_slice.size // 2
                left_sum = float(np.sum(local_slice[:half]))
                right_sum = float(np.sum(local_slice[-half:])) if half > 0 else left_sum
                denom = max(left_sum + right_sum, 1.0)
                asymmetry = (right_sum - left_sum) / denom
            local_slope = (right_edge - left_edge) / max(
                float(tight_hi_net - tight_lo_net), 1.0
            )

            ref_match = None
            tol = 1.5
            for ref_peak in qg_reference_peaks(reference_data):
                if ref_peak["isotope"] != line.isotope:
                    continue
                if abs(float(ref_peak["energy_keV"]) - float(line.energy_keV)) <= tol:
                    ref_match = ref_peak
                    break
            if ref_match is None:
                continue

            contexts[_line_key(ref_match)] = LineContext(
                reference=ref_match,
                peak_channel=int(round(peak_channel)),
                peak_energy_keV=float(peak_energy_keV),
                fwhm_ch=float(fwhm_ch),
                fwhm_keV=float(fwhm_keV),
                slope_keV_per_ch=float(slope),
                group_size=len(group),
                fit_net=float(fit_net),
                fit_unc=float(fit_unc),
                fit_success=bool(fit_success),
                covell_net=float(covell_net),
                covell_unc=float(covell_unc),
                covell_gross=float(covell_gross),
                gilmore_net=float(gilmore_net),
                gilmore_unc=float(gilmore_unc),
                gilmore_gross=float(gilmore_gross),
                iec_net=float(iec_net),
                iec_unc=float(iec_unc),
                iec_gross=float(iec_gross),
                maestro_net=float(maestro_net),
                maestro_gross=float(maestro_gross),
                hypermet_net=float(hypermet_net),
                hypermet_unc=float(hypermet_unc),
                hypermet_success=bool(hypermet_success),
                pml_dbl_exp_net=float(pml_dbl_exp_net),
                pml_dbl_exp_unc=float(pml_dbl_exp_unc),
                pml_dbl_exp_success=bool(pml_dbl_exp_success),
                tight_net=float(tight_net),
                tight_unc=float(tight_unc),
                tight_gross=float(tight_gross),
                expanded_net=float(expanded_net),
                expanded_unc=float(expanded_unc),
                expanded_gross=float(expanded_gross),
                expanded_bg=float(expanded_bg),
                current_net=float(current_net),
                current_unc=float(current_unc),
                current_gross=float(current_gross),
                peak_height=float(peak_height),
                left_edge=float(left_edge),
                right_edge=float(right_edge),
                local_background=float(local_background),
                local_slope=float(local_slope),
                asymmetry=float(asymmetry),
                centroid_shift_ch=float(centroid_shift_ch),
                tight_to_expanded_ratio=float(expanded_net / max(tight_net, 1.0)),
                fit_to_tight_ratio=(
                    float(fit_net / max(tight_net, 1.0)) if fit_net > 0.0 else 0.0
                ),
                hypermet_to_tight_ratio=(
                    float(hypermet_net / max(tight_net, 1.0))
                    if hypermet_net > 0.0
                    else 0.0
                ),
                neighbor_distance_keV=float(
                    _neighbor_distance_keV(expected_lines, line)
                ),
            )
    return contexts


def compute_current_method_results(
    raw_data: FluxWireData,
    reference_data: FluxWireData,
    background_data: FluxWireData,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    expected_lines = build_expected_lines(reference_data)
    peaks = analyze_raw_spectrum_targeted(
        data=raw_data,
        expected_lines=expected_lines,
        peak_threshold=float(
            config.get(
                "targeted_peak_significance_sigma",
                config.get("peak_significance_sigma", 3.0),
            )
        ),
        min_energy_keV=float(config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(config.get("max_peak_energy_keV", 3000.0)),
        background_spectrum=background_data.spectrum,
        background_subtract=True,
        profile_name=str(config["profile_name"]),
        roi_width_fwhm=float(config.get("flux_wire_roi_width_fwhm", 4.0)),
        background_width_channels=int(
            config.get("flux_wire_background_width_channels", 1)
        ),
        background_gap_fwhm=float(config.get("flux_wire_background_gap_fwhm", 0.0)),
        comparison_background_model=str(
            config.get("generic_comparison_background_model", "linear")
        ),
        broad_window_max_raw_gross_ratio=float(
            config.get("generic_broad_window_max_raw_gross_ratio", 1.35)
        ),
    )
    apply_qg_report_parity(
        peaks,
        reference_data,
        live_time_s=float(raw_data.live_time),
        efficiency_uncertainty_getter=None,
    )
    results: Dict[str, Dict[str, float]] = {}
    for ref_peak in qg_reference_peaks(reference_data):
        if float(ref_peak["net_counts"]) < float(
            config.get("minimum_qg_net_counts", 1.0)
        ):
            continue
        match, _ = match_peak(ref_peak, peaks, config)
        if match is None:
            continue
        results[_line_key(ref_peak)] = {
            "net": float(
                match.comparison_net_counts
                if match.comparison_net_counts is not None
                else match.net_counts
            ),
            "gross": float(
                match.comparison_gross_counts
                if match.comparison_gross_counts is not None
                else match.gross_counts
            ),
        }
    return results


def build_candidate_predictions(context: LineContext) -> Dict[str, Dict[str, float]]:
    gaussian_net = (
        context.fit_net
        if context.fit_success and context.fit_net > 0.0
        else context.tight_net
    )
    hypermet_net = (
        context.hypermet_net
        if context.hypermet_success and context.hypermet_net > 0.0
        else gaussian_net
    )
    pml_dbl_exp_net = (
        context.pml_dbl_exp_net
        if context.pml_dbl_exp_success and context.pml_dbl_exp_net > 0.0
        else hypermet_net
    )
    return {
        "tight_continuum": {
            "net": float(context.tight_net),
            "gross": float(context.tight_gross),
        },
        "expanded_continuum": {
            "net": float(context.expanded_net),
            "gross": float(context.expanded_gross),
        },
        "covell_local": {
            "net": float(context.covell_net),
            "gross": float(context.covell_gross),
        },
        "gilmore_minimum": {
            "net": float(context.gilmore_net),
            "gross": float(context.gilmore_gross),
        },
        "iec_tiered": {
            "net": float(context.iec_net),
            "gross": float(context.iec_gross),
        },
        "maestro_trapezoid": {
            "net": float(context.maestro_net),
            "gross": float(context.maestro_gross),
        },
        "gaussian_fit": {
            "net": float(gaussian_net),
            "gross": float(context.tight_gross),
        },
        "hypermet_hybrid": {
            "net": float(hypermet_net),
            "gross": float(
                context.tight_gross
                if context.group_size > 1
                else (
                    context.expanded_gross
                    if hypermet_net > 50000.0
                    else context.tight_gross
                )
            ),
        },
        "pml_dbl_exp": {
            "net": float(pml_dbl_exp_net),
            "gross": float(context.tight_gross),
        },
        QG_METHOD_NAME: {
            "net": float(context.current_net),
            "gross": float(context.current_gross),
        },
    }


def build_feature_vector(context: LineContext) -> List[float]:
    return [
        float(context.reference["energy_keV"]),
        float(context.fwhm_keV),
        float(context.fwhm_ch),
        float(math.log1p(max(context.peak_height, 0.0))),
        float(math.log1p(max(context.reference.get("net_counts", 0.0), 0.0))),
        float(math.log1p(max(context.local_background, 0.0))),
        float(context.local_slope),
        float(context.asymmetry),
        float(context.group_size),
        float(context.centroid_shift_ch),
        float(math.log1p(max(context.neighbor_distance_keV, 0.0))),
        float(context.tight_to_expanded_ratio),
        float(context.fit_to_tight_ratio),
        float(context.hypermet_to_tight_ratio),
        float(
            (context.peak_height - context.local_background)
            / max(context.local_background + 1.0, 1.0)
        ),
        float(
            (context.right_edge - context.left_edge)
            / max(context.left_edge + context.right_edge + 1.0, 1.0)
        ),
    ]


def choose_best_candidate_label(context: LineContext) -> str:
    ref_net = float(context.reference.get("net_counts") or 0.0)
    ref_gross = float(context.reference.get("gross_counts") or 0.0)
    candidates = build_candidate_predictions(context)
    best_label = "tight_continuum"
    best_score = float("inf")
    for label in ML_CANDIDATES:
        pred = candidates[label]
        net_rel = abs(_safe_rel(float(pred["net"]), ref_net) or 0.0)
        gross_rel = abs(_safe_rel(float(pred["gross"]), ref_gross) or 0.0)
        multiplet_penalty = (
            0.15
            if context.group_size > 1
            and label in {"tight_continuum", "expanded_continuum"}
            else 0.0
        )
        score = net_rel + 0.35 * gross_rel + multiplet_penalty
        if score < best_score:
            best_score = score
            best_label = label
    return best_label


def train_ml_selector(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    if not rows:
        return predictions

    label_to_index = {label: idx for idx, label in enumerate(ML_CANDIDATES)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    try:
        configure_tensorflow_cuda_runtime()
        import tensorflow as tf
    except Exception:
        majority = choose_majority_label(rows)
        for row in rows:
            predictions[str(row["row_id"])] = majority
        return predictions

    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    samples = sorted({str(row["sample_id"]) for row in rows})
    for sample_id in samples:
        train_rows = [row for row in rows if str(row["sample_id"]) != sample_id]
        test_rows = [row for row in rows if str(row["sample_id"]) == sample_id]
        if not test_rows:
            continue
        if len(train_rows) < 8:
            majority = choose_majority_label(train_rows or rows)
            for row in test_rows:
                predictions[str(row["row_id"])] = majority
            continue

        x_train = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
        y_train = np.asarray(
            [label_to_index[str(row["target_label"])] for row in train_rows],
            dtype=np.int32,
        )
        x_test = np.asarray([row["features"] for row in test_rows], dtype=np.float32)

        mu = x_train.mean(axis=0)
        sigma = x_train.std(axis=0)
        sigma[sigma < 1.0e-6] = 1.0
        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma

        unique_labels = np.unique(y_train)
        if unique_labels.size < 2:
            majority = index_to_label[int(unique_labels[0])]
            for row in test_rows:
                predictions[str(row["row_id"])] = majority
            continue

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(len(ML_CANDIDATES), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=20,
                restore_best_weights=True,
                min_delta=1.0e-4,
            )
        ]
        model.fit(
            x_train,
            y_train,
            epochs=160,
            batch_size=min(8, len(train_rows)),
            verbose=0,
            callbacks=callbacks,
        )
        probs = model.predict(x_test, verbose=0)
        indices = np.argmax(probs, axis=1)
        for row, idx in zip(test_rows, indices):
            predictions[str(row["row_id"])] = index_to_label[int(idx)]
    return predictions


def choose_majority_label(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "tight_continuum"
    counts: Dict[str, int] = {}
    for row in rows:
        label = str(row["target_label"])
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def summarize_method(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method and row["matched"]]
    net_abs = [
        abs(float(row["relative_net_error"]))
        for row in method_rows
        if row["relative_net_error"] is not None
    ]
    gross_abs = [
        abs(float(row["relative_gross_error"]))
        for row in method_rows
        if row["relative_gross_error"] is not None
    ]
    within_net = sum(value <= 0.25 for value in net_abs)
    within_gross = sum(value <= 0.25 for value in gross_abs)
    return {
        "method": method,
        "matched_peaks": len(method_rows),
        "mean_abs_net_error": _mean(net_abs),
        "median_abs_net_error": float(np.median(net_abs)) if net_abs else 0.0,
        "mean_abs_gross_error": _mean(gross_abs),
        "median_abs_gross_error": float(np.median(gross_abs)) if gross_abs else 0.0,
        "net_within_25pct": within_net,
        "gross_within_25pct": within_gross,
    }


def summarize_method_by_sample(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if not row.get("matched"):
            continue
        key = (str(row["sample_id"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (sample_id, method), method_rows in sorted(grouped.items()):
        net_abs = [
            abs(float(row["relative_net_error"]))
            for row in method_rows
            if row["relative_net_error"] is not None
        ]
        gross_abs = [
            abs(float(row["relative_gross_error"]))
            for row in method_rows
            if row["relative_gross_error"] is not None
        ]
        summary_rows.append(
            {
                "sample_id": sample_id,
                "method": method,
                "matched_peaks": len(method_rows),
                "mean_abs_net_error": _mean(net_abs),
                "mean_abs_gross_error": _mean(gross_abs),
            }
        )
    return summary_rows


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(
    path: Path, summary_rows: List[Dict[str, Any]], line_rows: List[Dict[str, Any]]
) -> None:
    lines: List[str] = ["# Peak Count Method Benchmark", ""]
    lines.append("## Overall summary")
    lines.append("")
    lines.append(
        "| Method | Matched peaks | Mean abs net err | Median abs net err | Mean abs gross err | Median abs gross err | Net within 25% | Gross within 25% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(
        summary_rows,
        key=lambda item: (item["mean_abs_net_error"], item["mean_abs_gross_error"]),
    ):
        lines.append(
            f"| {row['method']} | {row['matched_peaks']} | {row['mean_abs_net_error']:.4f} | {row['median_abs_net_error']:.4f} | {row['mean_abs_gross_error']:.4f} | {row['median_abs_gross_error']:.4f} | {row['net_within_25pct']} | {row['gross_within_25pct']} |"
        )

    lines.extend(
        [
            "",
            "## Worst net-count disagreements",
            "",
            "| Sample | Method | Line | QG net | Pred net | Rel err | QG gross | Pred gross | Rel gross err |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        [
            row
            for row in line_rows
            if row["matched"] and row["relative_net_error"] is not None
        ],
        key=lambda item: abs(float(item["relative_net_error"])),
        reverse=True,
    )[:25]
    for row in ranked:
        lines.append(
            f"| {row['sample_id']} | {row['method']} | {row['reference_isotope']} @ {float(row['reference_energy_keV']):.2f} | {float(row['reference_net_counts']):.1f} | {float(row['predicted_net_counts']):.1f} | {float(row['relative_net_error']):+.4f} | {float(row['reference_gross_counts']):.1f} | {float(row['predicted_gross_counts']):.1f} | {float(row['relative_gross_error'] or 0.0):+.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_methods(
    example_root: Path,
    results_root: Path,
    max_spectra: Optional[int] = None,
    selected_methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    selected_method_names = [
        normalize_benchmark_method_name(name) for name in (selected_methods or METHODS)
    ]
    selected_method_names = [
        name for name in METHODS if name in set(selected_method_names)
    ]

    paths = default_paths(example_root)
    config_path = paths.metadata_root / "workflow_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    energy_override = workflow_profile_energy_calibration(config)

    background_data = read_raw_asc(
        paths.background_path,
        energy_calibration_override=energy_override,
        profile_name=str(config["profile_name"]),
    )

    sample_files = _sample_intersections(
        paths.raw_root / "flux_wires", paths.qg_root / "flux_wires"
    )
    if max_spectra is not None:
        sample_files = sample_files[: max(0, int(max_spectra))]

    method_rows: List[Dict[str, Any]] = []
    ml_training_rows: List[Dict[str, Any]] = []
    cached_predictions: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

    for sample_id, raw_path, qg_path in sample_files:
        raw_data = read_raw_asc(
            raw_path,
            energy_calibration_override=energy_override,
            profile_name=str(config["profile_name"]),
        )
        raw_data.sample_id = sample_id
        reference_data = read_processed_txt(
            qg_path, profile_name=str(config["profile_name"])
        )
        contexts = compute_line_contexts(
            raw_data, reference_data, background_data, config
        )
        current_predictions = compute_current_method_results(
            raw_data, reference_data, background_data, config
        )

        for ref_peak in qg_reference_peaks(reference_data):
            if float(ref_peak["net_counts"]) < float(
                config.get("minimum_qg_net_counts", 1.0)
            ):
                continue
            key = _line_key(ref_peak)
            context = contexts.get(key)
            if context is None:
                continue
            candidates = build_candidate_predictions(context)
            if key in current_predictions:
                candidates[QG_METHOD_NAME] = current_predictions[key]
            cached_predictions[(sample_id, key)] = candidates
            ml_training_rows.append(
                {
                    "row_id": f"{sample_id}|{key}",
                    "sample_id": sample_id,
                    "line_key": key,
                    "features": build_feature_vector(context),
                    "target_label": choose_best_candidate_label(context),
                }
            )

    ml_choices = train_ml_selector(ml_training_rows)

    for sample_id, raw_path, qg_path in sample_files:
        reference_data = read_processed_txt(
            qg_path, profile_name=str(config["profile_name"])
        )
        for ref_peak in qg_reference_peaks(reference_data):
            if float(ref_peak["net_counts"]) < float(
                config.get("minimum_qg_net_counts", 1.0)
            ):
                continue
            key = _line_key(ref_peak)
            candidates = cached_predictions.get((sample_id, key))
            if not candidates:
                continue
            ml_choice = ml_choices.get(f"{sample_id}|{key}", "tight_continuum")
            candidates["ml_selector"] = dict(
                candidates.get(ml_choice, candidates["tight_continuum"])
            )
            candidates["ml_selector"]["selected_label"] = ml_choice
            for method in selected_method_names:
                pred = candidates.get(method)
                if pred is None:
                    continue
                predicted_net = float(pred["net"])
                predicted_gross = float(pred["gross"])
                method_rows.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "reference_isotope": str(ref_peak["isotope"]),
                        "reference_energy_keV": float(ref_peak["energy_keV"]),
                        "reference_net_counts": float(ref_peak["net_counts"]),
                        "reference_gross_counts": float(
                            ref_peak.get("gross_counts") or 0.0
                        ),
                        "predicted_net_counts": predicted_net,
                        "predicted_gross_counts": predicted_gross,
                        "relative_net_error": _safe_rel(
                            predicted_net, float(ref_peak["net_counts"])
                        ),
                        "relative_gross_error": _safe_rel(
                            predicted_gross, float(ref_peak.get("gross_counts") or 0.0)
                        ),
                        "matched": True,
                        "ml_selected_label": (
                            pred.get("selected_label")
                            if method == "ml_selector"
                            else ""
                        ),
                    }
                )

    summary_rows = [
        summarize_method(method_rows, method) for method in selected_method_names
    ]
    summary_rows.sort(
        key=lambda row: (row["mean_abs_net_error"], row["mean_abs_gross_error"])
    )
    sample_summary_rows = summarize_method_by_sample(method_rows)

    results_dir = results_root / "method_benchmark"
    write_csv(
        results_dir / "line_comparison.csv",
        method_rows,
        [
            "sample_id",
            "method",
            "reference_isotope",
            "reference_energy_keV",
            "reference_net_counts",
            "reference_gross_counts",
            "predicted_net_counts",
            "predicted_gross_counts",
            "relative_net_error",
            "relative_gross_error",
            "matched",
            "ml_selected_label",
        ],
    )
    write_csv(
        results_dir / "method_summary.csv",
        summary_rows,
        [
            "method",
            "matched_peaks",
            "mean_abs_net_error",
            "median_abs_net_error",
            "mean_abs_gross_error",
            "median_abs_gross_error",
            "net_within_25pct",
            "gross_within_25pct",
        ],
    )
    write_csv(
        results_dir / "sample_method_summary.csv",
        sample_summary_rows,
        [
            "sample_id",
            "method",
            "matched_peaks",
            "mean_abs_net_error",
            "mean_abs_gross_error",
        ],
    )
    write_summary_markdown(results_dir / "method_summary.md", summary_rows, method_rows)

    return {
        "results_dir": str(results_dir),
        "summary_rows": summary_rows,
        "line_rows": method_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple peak-counting methods against QG reference data."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to examples/RAFM_irradiation/results/.",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=None,
        help="Optional cap for debugging or quick runs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        type=normalize_benchmark_method_name,
        choices=USER_VISIBLE_METHOD_CHOICES,
        help=(
            "Optional subset of benchmark methods to include. "
            "Use qg or quantum_gold for the QuantumGold-focused workflow. "
            "Legacy aliases such as qg_hybrid, quantumgold, and current_hybrid are accepted and normalized to qg."
        ),
    )
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parent
    results_root = (args.results_root or (example_root / "results")).resolve()
    report = benchmark_methods(
        example_root,
        results_root,
        max_spectra=args.max_spectra,
        selected_methods=args.methods,
    )
    best = min(
        report["summary_rows"],
        key=lambda row: (row["mean_abs_net_error"], row["mean_abs_gross_error"]),
    )
    print(
        json.dumps(
            {"results_dir": report["results_dir"], "best_method": best}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
