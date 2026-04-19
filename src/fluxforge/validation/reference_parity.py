"""Reference parity runner for algorithm and workflow fixture suites."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from fluxforge.core.activity_review import review_spectrum_activation
from fluxforge.core.analysis_workspace import (
    PeakCandidate,
    background_adjusted_spectrum,
    detect_peak_candidates,
    subtract_background_counts,
)
from fluxforge.core.inventory_timeline import (
    build_inventory_state_from_activity_review,
    build_inventory_state_from_payload,
    compute_inventory_time_evolution,
)
from fluxforge.core.peak_fitting import fit_roi_peak
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.io.reader_factory import read_spectrum_any
from fluxforge.io.spe import GammaSpectrum

REFERENCE_MANIFEST_SCHEMA = "fluxforge.reference_parity_manifest.v1"
ACTIVATION_MANIFEST_SCHEMA = "fluxforge.activation_inventory_fixture.v1"


@dataclass(frozen=True)
class ParityCaseResult:
    """Execution report for one parity fixture manifest."""

    fixture_id: str
    manifest_path: str
    schema: str
    parity_scope: str
    workflow: str
    passed: bool
    compared_outputs: tuple[str, ...]
    mismatches: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "manifest_path": self.manifest_path,
            "schema": self.schema,
            "parity_scope": self.parity_scope,
            "workflow": self.workflow,
            "passed": self.passed,
            "compared_outputs": list(self.compared_outputs),
            "mismatches": list(self.mismatches),
        }


def run_reference_parity_suite(
    *,
    reference_root: Path,
    activation_root: Path | None = None,
    scope: str = "all",
    fixture_id: str | None = None,
    include_activation: bool = True,
) -> dict[str, Any]:
    """Execute reference parity manifests and return a machine-readable summary."""

    scope_value = str(scope or "all").strip().lower()
    if scope_value not in {"all", "algorithm", "workflow"}:
        raise ValueError("scope must be one of: all, algorithm, workflow")

    manifests: list[Path] = []
    reference_case_root = reference_root / "cases"
    manifests.extend(sorted(reference_case_root.glob("**/manifest.json")))
    if include_activation and activation_root is not None:
        manifests.extend(sorted(activation_root.glob("**/manifest.json")))

    if not manifests:
        raise FileNotFoundError("No parity manifest files were found.")

    results: list[ParityCaseResult] = []
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        current_fixture_id = str(manifest.get("fixture_id") or "")
        if fixture_id and fixture_id != current_fixture_id:
            continue

        parity_scope = str(manifest.get("parity_scope") or "workflow").strip().lower()
        if scope_value != "all" and parity_scope != scope_value:
            continue

        try:
            observed_outputs = _execute_manifest(manifest_path, manifest)
            mismatches = _compare_manifest_outputs(manifest_path, manifest, observed_outputs)
            results.append(
                ParityCaseResult(
                    fixture_id=current_fixture_id,
                    manifest_path=str(manifest_path),
                    schema=str(manifest.get("schema") or ""),
                    parity_scope=parity_scope,
                    workflow=str(manifest.get("workflow") or ""),
                    passed=len(mismatches) == 0,
                    compared_outputs=tuple(sorted(observed_outputs.keys())),
                    mismatches=tuple(mismatches),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive failure path
            results.append(
                ParityCaseResult(
                    fixture_id=current_fixture_id,
                    manifest_path=str(manifest_path),
                    schema=str(manifest.get("schema") or ""),
                    parity_scope=parity_scope,
                    workflow=str(manifest.get("workflow") or ""),
                    passed=False,
                    compared_outputs=(),
                    mismatches=(f"execution failed: {type(exc).__name__}: {exc}",),
                )
            )

    if not results:
        raise ValueError("No parity fixtures matched the selected filters.")

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    return {
        "schema": "fluxforge.reference_parity.run.v1",
        "scope": scope_value,
        "fixture_filter": fixture_id,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "results": [result.to_payload() for result in results],
    }


def _execute_manifest(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    schema = str(manifest.get("schema") or "")
    if schema == REFERENCE_MANIFEST_SCHEMA:
        return _execute_reference_manifest(manifest_path.parent, manifest)
    if schema == ACTIVATION_MANIFEST_SCHEMA:
        return _execute_activation_manifest(manifest_path.parent, manifest)
    raise ValueError(f"Unsupported parity manifest schema: {schema}")


def _execute_reference_manifest(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    workflow = str(manifest.get("workflow") or "").strip().lower()
    if workflow == "peak-search-only":
        return _run_peak_search_case(case_dir, manifest)
    if workflow == "overlay-role-workflow":
        return _run_overlay_role_case(case_dir, manifest)
    if workflow == "roi-statistics-workflow":
        return _run_roi_statistics_case(case_dir, manifest)
    if workflow == "spectrum-io-normalization":
        return _run_spectrum_io_case(case_dir, manifest)
    if workflow == "background-subtraction":
        return _run_background_subtraction_case(case_dir, manifest)
    if workflow == "peak-fit-roi":
        return _run_peak_fit_case(case_dir, manifest)
    if workflow in {
        "activity-review -> inventory-review",
        "peaks -> activity-review -> inventory-review",
    }:
        return _run_activity_inventory_case(case_dir, manifest)
    raise ValueError(f"Unsupported reference parity workflow: {workflow}")


def _execute_activation_manifest(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    workflow = str(manifest.get("workflow") or "").strip().lower()
    if workflow == "pure_decay_bateman":
        return _run_activation_decay_case(case_dir, manifest)
    if workflow == "second_irradiation_planner":
        return _run_second_irradiation_case(case_dir, manifest)
    raise ValueError(f"Unsupported activation parity workflow: {workflow}")


def _run_peak_search_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)
    spectrum = _spectrum_from_payload(payload)

    peaks = detect_peak_candidates(
        spectrum,
        method=str(payload.get("method") or "mariscotti"),
        threshold=float(payload.get("threshold") or 4.0),
        min_distance=int(payload.get("min_distance") or 18),
        max_peaks=int(payload.get("max_peaks") or 12),
    )
    energies = sorted(float(peak.energy_keV) for peak in peaks)
    channels = sorted(float(peak.channel) for peak in peaks)
    return {
        "detected_peaks_expected.json": {
            "peak_count": len(peaks),
            "energies_keV": [round(value, 6) for value in energies],
            "channels": [round(value, 3) for value in channels],
            "first_peak_keV": round(energies[0], 6) if energies else None,
        }
    }


def _run_spectrum_io_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)
    sources = payload.get("sources") or []
    if not isinstance(sources, list) or not sources:
        raise ValueError("Spectrum I/O parity input must define a non-empty sources list.")

    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        rel_path = str(source.get("path") or "").strip()
        if not rel_path:
            continue
        source_path = (case_dir / rel_path).resolve()
        spectrum = read_spectrum_any(source_path)
        counts = np.asarray(spectrum.counts, dtype=float)
        probe_channels = [
            int(value)
            for value in (source.get("probe_channels") or [0, 1, 2])
        ]
        energy_probe = [
            _round_float(spectrum.channel_to_energy(channel))
            for channel in probe_channels
        ]

        records.append(
            {
                "label": str(source.get("label") or f"source_{index}"),
                "path": rel_path,
                "format": source_path.suffix.lower(),
                "channel_count": int(counts.size),
                "sum_counts": _round_float(float(np.sum(counts))),
                "max_counts": _round_float(float(np.max(counts))) if counts.size else 0.0,
                "live_time_s": _round_float(float(spectrum.live_time)),
                "real_time_s": _round_float(float(spectrum.real_time)),
                "energy_probe_channels": probe_channels,
                "energy_probe_keV": energy_probe,
            }
        )

    if not records:
        raise ValueError("Spectrum I/O parity case did not produce any records.")

    return {
        "io_parity_expected.json": {
            "schema": "fluxforge.reference_parity.io.v1",
            "record_count": len(records),
            "records": records,
        }
    }


def _run_background_subtraction_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)

    foreground_payload = payload.get("foreground")
    if not isinstance(foreground_payload, dict):
        raise ValueError("Background parity input must define foreground spectrum payload.")
    foreground = _spectrum_from_payload(foreground_payload)

    background_payload = payload.get("background")
    background = (
        _spectrum_from_payload(background_payload)
        if isinstance(background_payload, dict)
        else None
    )

    mode = str(payload.get("mode") or "simple")
    scale = float(payload.get("scale") or 1.0)

    adjusted_counts = subtract_background_counts(
        foreground,
        background,
        mode=mode,
        scale=scale,
    )
    adjusted_spectrum = background_adjusted_spectrum(
        foreground,
        background,
        mode=mode,
        scale=scale,
    )
    meta = (adjusted_spectrum.metadata or {}).get("background_subtraction") or {}

    return {
        "background_subtraction_expected.json": {
            "schema": "fluxforge.reference_parity.background.v1",
            "mode": mode,
            "requested_scale": _round_float(scale),
            "effective_scale": _round_float(float(meta.get("scale") or 0.0)),
            "channel_count": int(adjusted_counts.size),
            "sum_counts": _round_float(float(np.sum(adjusted_counts))),
            "max_counts": _round_float(float(np.max(adjusted_counts))) if adjusted_counts.size else 0.0,
            "first_channels": [_round_float(value) for value in adjusted_counts[:8]],
        }
    }


def _run_overlay_role_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)

    foreground_payload = payload.get("foreground")
    if not isinstance(foreground_payload, dict):
        raise ValueError("Overlay-role parity input must define foreground spectrum payload.")
    foreground = _spectrum_from_payload(foreground_payload)

    background_payload = payload.get("background")
    background = (
        _spectrum_from_payload(background_payload)
        if isinstance(background_payload, dict)
        else None
    )

    overlay_payload = payload.get("overlay")
    overlay = (
        _spectrum_from_payload(overlay_payload)
        if isinstance(overlay_payload, dict)
        else None
    )

    mode = str(payload.get("mode") or "scaled")
    scale = float(payload.get("scale") or 1.0)
    overlay_scale = float(payload.get("overlay_scale") or 1.0)

    adjusted_counts = subtract_background_counts(
        foreground,
        background,
        mode=mode,
        scale=scale,
    )
    display_counts = np.asarray(adjusted_counts, dtype=float)
    overlay_sum_counts = 0.0

    if overlay is not None:
        overlay_counts = np.asarray(overlay.counts, dtype=float)
        overlay_sum_counts = float(np.sum(overlay_counts))
        sample_count = min(display_counts.size, overlay_counts.size)
        if sample_count <= 0:
            raise ValueError("Overlay-role parity input must produce non-empty display channels.")
        display_counts = display_counts[:sample_count] + (
            max(overlay_scale, 0.0) * overlay_counts[:sample_count]
        )

    if display_counts.size == 0:
        raise ValueError("Overlay-role parity input must produce non-empty display channels.")

    foreground_channels = np.asarray(foreground.channels, dtype=float)
    if foreground_channels.size >= display_counts.size:
        display_channels = foreground_channels[: display_counts.size]
    else:
        display_channels = np.arange(display_counts.size, dtype=float)

    display_spectrum = GammaSpectrum(
        counts=display_counts,
        channels=display_channels,
        live_time=float(foreground.live_time),
        real_time=float(foreground.real_time),
        calibration=dict(foreground.calibration or {}),
        spectrum_id=f"{foreground.spectrum_id}_overlay",
    )
    peaks = detect_peak_candidates(
        display_spectrum,
        method=str(payload.get("peak_search_method") or "mariscotti"),
        threshold=float(payload.get("threshold") or 4.0),
        min_distance=int(payload.get("min_distance") or 18),
        max_peaks=int(payload.get("max_peaks") or 12),
    )
    peak_energies = sorted(float(peak.energy_keV) for peak in peaks)

    return {
        "overlay_role_expected.json": {
            "schema": "fluxforge.reference_parity.overlay_role.v1",
            "mode": mode,
            "requested_scale": _round_float(scale),
            "overlay_scale": _round_float(overlay_scale),
            "channel_count": int(display_counts.size),
            "foreground_sum_counts": _round_float(float(np.sum(foreground.counts))),
            "background_sum_counts": _round_float(
                float(np.sum(background.counts)) if background is not None else 0.0
            ),
            "adjusted_sum_counts": _round_float(float(np.sum(adjusted_counts))),
            "overlay_sum_counts": _round_float(overlay_sum_counts),
            "display_sum_counts": _round_float(float(np.sum(display_counts))),
            "peak_count": len(peaks),
            "first_peak_keV": _round_float(peak_energies[0]) if peak_energies else None,
        }
    }


def _run_roi_statistics_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)

    spectrum_payload = payload.get("spectrum")
    if not isinstance(spectrum_payload, dict):
        raise ValueError("ROI statistics parity input must define spectrum payload.")
    spectrum = _spectrum_from_payload(spectrum_payload)

    method = str(payload.get("peak_search_method") or "mariscotti")
    peaks = detect_peak_candidates(
        spectrum,
        method=method,
        threshold=float(payload.get("threshold") or 4.0),
        min_distance=int(payload.get("min_distance") or 18),
        max_peaks=int(payload.get("max_peaks") or 12),
    )
    if not peaks:
        raise ValueError("ROI statistics workflow did not detect any peaks.")

    selected_peak = max(peaks, key=lambda peak: float(peak.significance))
    roi_half_width_keV = max(float(payload.get("roi_half_width_keV") or 1.5), 0.2)
    roi_bounds = (
        float(selected_peak.energy_keV) - roi_half_width_keV,
        float(selected_peak.energy_keV) + roi_half_width_keV,
    )

    channels = np.asarray(spectrum.channels, dtype=float)
    energies_keV = np.asarray(
        [spectrum.channel_to_energy(float(channel)) for channel in channels],
        dtype=float,
    )
    counts = np.asarray(spectrum.counts, dtype=float)
    fit_result = fit_roi_peak(
        energies_keV,
        counts,
        roi_bounds,
        fitter_key=str(payload.get("fitter_key") or "gaussian"),
        background_model=str(payload.get("background_model") or "linear"),
        prior_fwhm_channels=(
            float(payload.get("prior_fwhm_channels"))
            if payload.get("prior_fwhm_channels") is not None
            else None
        ),
    )

    mask = (energies_keV >= roi_bounds[0]) & (energies_keV <= roi_bounds[1])
    gross_counts = float(np.sum(counts[mask])) if np.any(mask) else 0.0

    return {
        "roi_statistics_expected.json": {
            "schema": "fluxforge.reference_parity.roi_statistics.v1",
            "peak_search_method": method,
            "peak_count": len(peaks),
            "selected_peak_keV": _round_float(float(selected_peak.energy_keV)),
            "roi_bounds_keV": [
                _round_float(float(roi_bounds[0])),
                _round_float(float(roi_bounds[1])),
            ],
            "gross_counts": _round_float(gross_counts),
            "centroid_keV": _round_float(fit_result.centroid_channel),
            "fwhm_keV": _round_float(fit_result.fwhm_channels),
            "area_counts": _round_float(fit_result.area_counts),
            "net_counts": _round_float(float(fit_result.peak_result.net_counts)),
            "reduced_chi_squared": _round_float(
                float(fit_result.peak_result.reduced_chi_squared)
            ),
            "success": bool(fit_result.peak_result.success),
        }
    }


def _run_peak_fit_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)

    channels = np.asarray(payload.get("channels") or [], dtype=float)
    counts = np.asarray(payload.get("counts") or [], dtype=float)
    if channels.size == 0 or counts.size == 0:
        raise ValueError("Peak-fit parity input must define non-empty channels and counts.")
    if channels.size != counts.size:
        raise ValueError("Peak-fit parity channels and counts must have the same length.")

    roi = payload.get("roi_bounds") or [0.0, float(channels[-1])]
    if not isinstance(roi, (list, tuple)) or len(roi) != 2:
        raise ValueError("Peak-fit parity input must define roi_bounds with two values.")

    result = fit_roi_peak(
        channels,
        counts,
        (float(roi[0]), float(roi[1])),
        fitter_key=str(payload.get("fitter_key") or "gaussian"),
        background_model=str(payload.get("background_model") or "linear"),
        prior_fwhm_channels=(
            float(payload.get("prior_fwhm_channels"))
            if payload.get("prior_fwhm_channels") is not None
            else None
        ),
    )

    return {
        "peak_fit_expected.json": {
            "schema": "fluxforge.reference_parity.peak_fit.v1",
            "fitter_key": result.fitter_key,
            "background_model": result.background_model,
            "centroid_channel": _round_float(result.centroid_channel),
            "fwhm_channels": _round_float(result.fwhm_channels),
            "area_counts": _round_float(result.area_counts),
            "chi_squared": _round_float(float(result.peak_result.chi_squared)),
            "reduced_chi_squared": _round_float(float(result.peak_result.reduced_chi_squared)),
            "net_counts": _round_float(float(result.peak_result.net_counts)),
            "success": bool(result.peak_result.success),
        }
    }


def _run_activity_inventory_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = _first_input_file(case_dir, manifest)
    payload = _load_json(input_file)

    peaks_payload = payload.get("peaks")
    if peaks_payload is None:
        peaks_payload = payload.get("peaks_report") or payload.get("peak_rows")
    if not isinstance(peaks_payload, list) or not peaks_payload:
        raise ValueError("Activity workflow parity input must include a non-empty peaks list.")

    peaks = _peak_candidates_from_rows(peaks_payload)
    live_time_s = float(payload.get("live_time_s") or 600.0)
    cooling_time_s = float(payload.get("cooling_time_s") or 0.0)
    source_id = str(payload.get("source_id") or "fluxforge_bundled_gamma")
    custom_gamma_path = payload.get("custom_gamma_path")
    if custom_gamma_path:
        custom_gamma_path = str(custom_gamma_path)

    efficiency_curve = _efficiency_curve_from_payload(payload.get("efficiency_curve"))
    review = review_spectrum_activation(
        peaks,
        live_time_s=live_time_s,
        efficiency_curve=efficiency_curve,
        cooling_time_s=cooling_time_s,
        source_id=source_id,
        custom_gamma_path=custom_gamma_path,
        energy_tolerance_keV=float(payload.get("energy_tolerance_keV") or 2.0),
        dead_time_fraction=float(payload.get("dead_time_fraction") or 0.0),
    )

    inventory_state = build_inventory_state_from_activity_review(
        review,
        sample_id=str(payload.get("sample_id") or "fixture_sample"),
    )
    relative_times_s = tuple(
        float(value)
        for value in (payload.get("inventory_relative_times_s") or [0.0, 3600.0, 14400.0])
    )
    inventory = compute_inventory_time_evolution(
        inventory_state,
        relative_times_s=relative_times_s,
        time_origin=str(payload.get("inventory_time_origin") or "eoi"),
        decay_source_id=str(payload.get("decay_source_id") or inventory_state.decay_source_id),
        distance_cm=float(payload.get("distance_cm") or 30.0),
    )

    nuclides = sorted(summary.nuclide for summary in review.isotope_summaries)
    return {
        "isotope_summary_expected.csv": review.isotope_rows(),
        "inventory_timeseries_expected.csv": inventory.time_series_rows(),
        "activity_review_expected.json": {
            "schema": "fluxforge.activity_review.v1",
            "isotope_count": len(review.isotope_summaries),
            "line_count": len(review.line_results),
            "nuclides": nuclides,
        },
        "inventory_review_expected.json": {
            "schema": "fluxforge.inventory_time_evolution.v1",
            "sample_id": inventory.inventory_state.sample_id,
            "time_point_count": len(inventory.relative_times_s),
            "nuclides": sorted(label for label in inventory.activity_series.keys() if label != "Total"),
        },
    }


def _run_activation_decay_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_file = case_dir / str((manifest.get("input_files") or [""])[0])
    payload = _load_json(input_file)
    inventory_state = build_inventory_state_from_payload(
        payload,
        decay_source_id=str(payload.get("decay_source_id") or payload.get("decay_source") or "radioactivedecay_icrp107_kayzero_2023"),
    )
    relative_times_s = tuple(
        float(value)
        for value in (payload.get("relative_times_s") or [0.0, 3600.0, 7200.0, 14400.0])
    )
    result = compute_inventory_time_evolution(
        inventory_state,
        relative_times_s=relative_times_s,
        time_origin=str(payload.get("time_origin") or "eoi"),
        distance_cm=float(payload.get("distance_cm") or 30.0),
    )
    dominant_rows = list(result.reference_rows("eoi"))[: int(payload.get("dominant_top_n") or 5)]
    return {
        "inventory_timeseries_expected.csv": result.time_series_rows(),
        "dominant_contributors_expected.csv": dominant_rows,
    }


def _run_second_irradiation_case(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    input_files = [case_dir / str(item) for item in manifest.get("input_files") or []]
    if len(input_files) < 3:
        raise ValueError("Second-irradiation parity fixture requires three input files.")

    inventory_payload = _load_json(input_files[0])
    first_schedule = _load_json(input_files[1])
    candidate_payload = _load_json(input_files[2])

    inventory_state = build_inventory_state_from_payload(
        inventory_payload,
        decay_source_id=str(inventory_payload.get("decay_source_id") or "radioactivedecay_icrp107_kayzero_2023"),
    )

    first_cooling_s = float(first_schedule.get("first_cooling_time_s") or first_schedule.get("cooling_time_s") or 0.0)
    second_duration_s = float(first_schedule.get("second_irradiation_time_s") or 0.0)
    target_weights = {
        str(key): float(value)
        for key, value in (first_schedule.get("target_weights") or {}).items()
    }

    candidates = candidate_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Second-irradiation candidate file must define a non-empty candidates list.")

    scored: list[dict[str, Any]] = []
    post_rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label") or f"candidate_{index}")
        flux_scale = float(candidate.get("flux_scale") or 1.0)
        second_cooling_s = float(candidate.get("second_cooling_time_s") or 0.0)
        duration_factor = float(candidate.get("duration_factor") or 1.0)

        total_score = 0.0
        rows: list[dict[str, Any]] = []
        for seed in inventory_state.seeds:
            decay_constant = math.log(2.0) / max(float(seed.half_life_s), 1e-12)
            activity_after_first = float(seed.activity_eoi_bq) * math.exp(-decay_constant * first_cooling_s)
            replenishment = float(seed.activity_eoi_bq) * max(flux_scale, 0.0) * max(duration_factor, 0.0)
            if second_duration_s > 0.0:
                saturation = 1.0 - math.exp(-decay_constant * second_duration_s)
            else:
                saturation = 1.0
            activity_after_second = activity_after_first + replenishment * saturation
            activity_measurement = activity_after_second * math.exp(-decay_constant * second_cooling_s)
            weight = target_weights.get(seed.nuclide, 1.0)
            total_score += weight * activity_measurement
            rows.append(
                {
                    "candidate": label,
                    "nuclide": seed.nuclide,
                    "activity_bq": activity_measurement,
                    "weight": weight,
                    "weighted_activity": weight * activity_measurement,
                }
            )

        scored.append(
            {
                "label": label,
                "score": total_score,
                "flux_scale": flux_scale,
                "duration_factor": duration_factor,
                "second_cooling_time_s": second_cooling_s,
            }
        )
        post_rows_by_label[label] = rows

    if not scored:
        raise ValueError("No valid second-irradiation candidates were parsed.")

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = scored[0]
    selected_rows = sorted(
        post_rows_by_label[selected["label"]],
        key=lambda item: (item["nuclide"],),
    )
    return {
        "selected_second_irradiation_expected.json": {
            "selected_label": selected["label"],
            "score": selected["score"],
            "candidate_count": len(scored),
        },
        "post_second_irradiation_inventory_expected.csv": selected_rows,
    }


def _compare_manifest_outputs(
    manifest_path: Path,
    manifest: dict[str, Any],
    observed_outputs: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    tolerances = {
        str(key): float(value)
        for key, value in (manifest.get("tolerances") or {}).items()
    }
    case_dir = manifest_path.parent

    for expected_name in manifest.get("expected_outputs") or []:
        output_name = str(expected_name)
        expected_path = case_dir / output_name
        if not expected_path.exists():
            mismatches.append(f"missing expected artifact: {expected_path}")
            continue
        if output_name not in observed_outputs:
            mismatches.append(f"runner did not produce expected output key: {output_name}")
            continue

        observed = observed_outputs[output_name]
        if expected_path.suffix.lower() == ".json":
            expected = _load_json(expected_path)
            mismatches.extend(
                _compare_json_values(
                    expected,
                    observed,
                    tolerances=tolerances,
                    path=output_name,
                )
            )
            continue

        if expected_path.suffix.lower() == ".csv":
            expected_rows = _load_csv_rows(expected_path)
            observed_rows = _coerce_rows(observed)
            mismatches.extend(
                _compare_csv_rows(
                    expected_rows,
                    observed_rows,
                    tolerances=tolerances,
                    path=output_name,
                )
            )
            continue

        mismatches.append(f"unsupported expected output format: {expected_path}")

    return mismatches


def _compare_json_values(
    expected: Any,
    observed: Any,
    *,
    tolerances: dict[str, float],
    path: str,
) -> list[str]:
    mismatches: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f"{path}: expected object, observed {type(observed).__name__}"]
        for key, value in expected.items():
            if key not in observed:
                mismatches.append(f"{path}.{key}: missing key in observed payload")
                continue
            mismatches.extend(
                _compare_json_values(
                    value,
                    observed[key],
                    tolerances=tolerances,
                    path=f"{path}.{key}",
                )
            )
        return mismatches

    if isinstance(expected, list):
        if not isinstance(observed, list):
            return [f"{path}: expected list, observed {type(observed).__name__}"]
        if len(expected) != len(observed):
            mismatches.append(
                f"{path}: list length mismatch expected {len(expected)} observed {len(observed)}"
            )
            return mismatches
        for index, value in enumerate(expected):
            mismatches.extend(
                _compare_json_values(
                    value,
                    observed[index],
                    tolerances=tolerances,
                    path=f"{path}[{index}]",
                )
            )
        return mismatches

    if _is_number(expected) and _is_number(observed):
        if not _values_close(float(expected), float(observed), path=path, tolerances=tolerances):
            mismatches.append(
                f"{path}: numeric mismatch expected {expected} observed {observed}"
            )
        return mismatches

    if expected != observed:
        mismatches.append(f"{path}: expected {expected!r} observed {observed!r}")
    return mismatches


def _compare_csv_rows(
    expected_rows: list[dict[str, Any]],
    observed_rows: list[dict[str, Any]],
    *,
    tolerances: dict[str, float],
    path: str,
) -> list[str]:
    mismatches: list[str] = []
    if len(observed_rows) < len(expected_rows):
        return [
            f"{path}: row count mismatch expected at least {len(expected_rows)} observed {len(observed_rows)}"
        ]

    for index, expected_row in enumerate(expected_rows):
        observed_row = observed_rows[index]
        for key, expected_value in expected_row.items():
            if key not in observed_row:
                mismatches.append(f"{path}[{index}].{key}: missing column in observed row")
                continue
            observed_value = observed_row[key]
            if _is_number(expected_value) and _is_number(observed_value):
                if not _values_close(
                    float(expected_value),
                    float(observed_value),
                    path=f"{path}.{key}",
                    tolerances=tolerances,
                ):
                    mismatches.append(
                        f"{path}[{index}].{key}: expected {expected_value} observed {observed_value}"
                    )
            elif (
                (expected_value in {"", None} and observed_value in {"", None})
                or str(expected_value) == str(observed_value)
            ):
                continue
            else:
                mismatches.append(
                    f"{path}[{index}].{key}: expected {expected_value!r} observed {observed_value!r}"
                )
    return mismatches


def _values_close(
    expected: float,
    observed: float,
    *,
    path: str,
    tolerances: dict[str, float],
) -> bool:
    absolute_error = abs(observed - expected)
    absolute_tol = _resolve_tolerance(path, tolerances, suffix="_abs")
    if absolute_tol is not None and absolute_error <= absolute_tol:
        return True

    relative_tol = _resolve_tolerance(path, tolerances, suffix="_rel")
    if relative_tol is not None:
        scale = max(abs(expected), 1.0e-12)
        if absolute_error <= relative_tol * scale:
            return True

    if absolute_tol is None and relative_tol is None:
        return absolute_error <= 1.0e-9
    return False


def _resolve_tolerance(path: str, tolerances: dict[str, float], *, suffix: str) -> float | None:
    lowered_path = path.lower()
    selected: float | None = None
    for key, value in tolerances.items():
        if not key.lower().endswith(suffix):
            continue
        token = key[: -len(suffix)].lower()
        if token and token in lowered_path:
            selected = float(value)
            break
    return selected


def _spectrum_from_payload(payload: dict[str, Any]) -> GammaSpectrum:
    counts = np.asarray(payload.get("counts") or [], dtype=float)
    if counts.size == 0:
        raise ValueError("Spectrum payload must define non-empty counts.")
    channels_raw = payload.get("channels")
    if channels_raw is None:
        channels = np.arange(len(counts), dtype=float)
    else:
        channels = np.asarray(channels_raw, dtype=float)
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {"energy": [0.0, 1.0, 0.0]}
    return GammaSpectrum(
        counts=counts,
        channels=channels,
        live_time=float(payload.get("live_time_s") or payload.get("live_time") or 600.0),
        real_time=float(payload.get("real_time_s") or payload.get("real_time") or 600.0),
        calibration=calibration,
        spectrum_id=str(payload.get("spectrum_id") or "parity_fixture"),
    )


def _peak_candidates_from_rows(rows: Sequence[dict[str, Any]]) -> list[PeakCandidate]:
    peaks: list[PeakCandidate] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        energy = float(row.get("energy_keV") or row.get("peak_energy_keV") or 0.0)
        if energy <= 0.0:
            continue
        tolerance = max(float(row.get("roi_half_width_keV") or 1.5), 0.1)
        nuclide = str(row.get("nuclide") or row.get("report_isotope") or "").strip() or None
        peaks.append(
            PeakCandidate(
                peak_id=str(row.get("peak_id") or f"peak-{index}"),
                channel=float(row.get("channel") or index),
                energy_keV=energy,
                significance=float(row.get("significance") or 0.0),
                roi_bounds_keV=(
                    float(row.get("left_keV") or row.get("left_energy_keV") or (energy - tolerance)),
                    float(row.get("right_keV") or row.get("right_energy_keV") or (energy + tolerance)),
                ),
                net_counts=float(
                    row.get("net_counts")
                    or row.get("area")
                    or row.get("raw_counts")
                    or row.get("amplitude")
                    or 0.0
                ),
                fit_quality=float(
                    row.get("fit_quality")
                    or row.get("reduced_chi_squared")
                    or 1.0
                ),
                status="matched" if nuclide else "candidate",
                nuclide=nuclide,
            )
        )
    if not peaks:
        raise ValueError("No peak rows could be converted into parity candidates.")
    return peaks


def _efficiency_curve_from_payload(payload: Any) -> EfficiencyCurve:
    if not isinstance(payload, dict):
        return EfficiencyCurve.from_polynomial([-7.0, 1.3, -0.19], energy_range=(30.0, 3000.0))

    model_type = str(payload.get("model_type") or "polynomial").lower()
    if model_type == "polynomial":
        coefficients = [float(value) for value in payload.get("coefficients") or [-7.0, 1.3, -0.19]]
        energy_range = tuple(float(value) for value in payload.get("energy_range") or (30.0, 3000.0))
        return EfficiencyCurve.from_polynomial(coefficients=coefficients, energy_range=energy_range)

    if model_type == "empirical":
        return EfficiencyCurve.from_calibration_points(
            energies=[float(value) for value in payload.get("energies") or [59.5, 356.0, 661.7, 1173.2]],
            efficiencies=[float(value) for value in payload.get("efficiencies") or [0.12, 0.08, 0.06, 0.03]],
            uncertainties=[float(value) for value in payload.get("uncertainties") or [0.01, 0.01, 0.01, 0.01]],
            interpolation=str(payload.get("interpolation") or "linear"),
        )

    return EfficiencyCurve.from_polynomial([-7.0, 1.3, -0.19], energy_range=(30.0, 3000.0))


def _first_input_file(case_dir: Path, manifest: dict[str, Any]) -> Path:
    input_files = manifest.get("input_files") or []
    if not input_files:
        raise ValueError("Parity manifest must define at least one input file.")
    return case_dir / str(input_files[0])


def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append({key: item[key] for key in item})
        return rows
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _round_float(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "ACTIVATION_MANIFEST_SCHEMA",
    "REFERENCE_MANIFEST_SCHEMA",
    "run_reference_parity_suite",
]
