from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from fluxforge.core.activity_review import (
    ActivityReviewIsotopeSummary,
    ActivityReviewLineResult,
    ActivityReviewResult,
    build_simple_bateman_summary,
)
from fluxforge.core.analysis_workspace import ActivityCalculationResult
from fluxforge.data.isotope_names import format_isotope_name, parse_nndc_isotope_name
from fluxforge.io.spe import GammaSpectrum


ROOT = Path(__file__).resolve().parents[1]
RAFM_EXAMPLE_ROOT = ROOT / "examples" / "RAFM_irradiation"
RAFM_ANALYSIS_ROOT = RAFM_EXAMPLE_ROOT / "results" / "analysis_json"
RAFM_SCHEDULES_PATH = RAFM_EXAMPLE_ROOT / "metadata" / "sample_schedules.json"
RAFM_UNFOLD_MLEM_PATH = RAFM_EXAMPLE_ROOT / "results" / "unfolding" / "mlem.json"
DEFAULT_PHASE6_SAMPLE_ID = "RAFM4-C_15dEOI"


def _normalize_nuclide(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return text
    try:
        element, mass, metastable = parse_nndc_isotope_name(text)
    except ValueError:
        return text
    return format_isotope_name(element, mass, metastable, separator="-")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _infer_half_life_s(
    count_time_activity_bq: float,
    irradiation_time_activity_bq: float,
    cooling_time_s: float,
) -> float:
    count_time = max(float(count_time_activity_bq), 0.0)
    irradiation_time = max(float(irradiation_time_activity_bq), 0.0)
    cooling = max(float(cooling_time_s), 0.0)
    if cooling <= 0.0 or count_time <= 0.0 or irradiation_time <= count_time:
        return max(cooling, 1.0) * 1.0e6
    ratio = irradiation_time / count_time
    if ratio <= 1.0:
        return max(cooling, 1.0) * 1.0e6
    return math.log(2.0) * cooling / math.log(ratio)


def _series_points(
    activity_bq: float,
    uncertainty_bq: float,
    half_life_s: float,
    horizon_s: float,
    *,
    count: int = 64,
) -> tuple[tuple[float, float, float], ...]:
    if count <= 1:
        count = 2
    if half_life_s <= 0.0:
        decay_constant = 0.0
    else:
        decay_constant = math.log(2.0) / half_life_s
    rows = []
    step = float(horizon_s) / float(count - 1)
    for index in range(count):
        time_s = step * index
        factor = math.exp(-decay_constant * time_s) if decay_constant > 0.0 else 1.0
        rows.append(
            (
                float(time_s),
                float(activity_bq) * factor,
                float(uncertainty_bq) * factor,
            )
        )
    return tuple(rows)


def _daughter_series(
    activity_bq: float,
    uncertainty_bq: float,
    half_life_s: float,
    horizon_s: float,
    *,
    count: int = 64,
) -> tuple[tuple[float, float, float], ...]:
    if count <= 1:
        count = 2
    if half_life_s <= 0.0:
        decay_constant = 0.0
    else:
        decay_constant = math.log(2.0) / half_life_s
    rows = []
    step = float(horizon_s) / float(count - 1)
    for index in range(count):
        time_s = step * index
        daughter_fraction = 1.0 - math.exp(-decay_constant * time_s) if decay_constant > 0.0 else 0.0
        rows.append(
            (
                float(time_s),
                float(activity_bq) * daughter_fraction,
                float(uncertainty_bq) * daughter_fraction,
            )
        )
    return tuple(rows)


def _sample_letter(sample_id: str) -> str:
    stem = Path(sample_id).stem
    for token in stem.replace("_", "-").split("-"):
        if len(token) == 1 and token.isalpha():
            return token.upper()
    raise ValueError(f"Could not resolve RAFM sample letter from {sample_id!r}.")


@lru_cache(maxsize=None)
def load_phase6_example_analysis(sample_id: str = DEFAULT_PHASE6_SAMPLE_ID) -> dict[str, Any]:
    path = RAFM_ANALYSIS_ROOT / f"{sample_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_phase6_sample_schedules() -> dict[str, Any]:
    return json.loads(RAFM_SCHEDULES_PATH.read_text(encoding="utf-8"))


def load_phase6_schedule_entry(sample_id: str = DEFAULT_PHASE6_SAMPLE_ID) -> dict[str, Any]:
    payload = load_phase6_sample_schedules()
    sample_letter = _sample_letter(sample_id)
    schedules = payload.get("schedules") or {}
    entry = schedules.get(sample_letter)
    if not isinstance(entry, dict):
        raise KeyError(f"Missing RAFM schedule metadata for sample {sample_id}.")
    return entry


def load_phase6_real_data_paths(sample_id: str = DEFAULT_PHASE6_SAMPLE_ID) -> dict[str, Path]:
    analysis = load_phase6_example_analysis(sample_id)
    counts_name = Path(str(analysis.get("counts_csv") or "")).name
    return {
        "analysis_json": RAFM_ANALYSIS_ROOT / f"{sample_id}.json",
        "counts_csv": RAFM_EXAMPLE_ROOT / "results" / "counts" / counts_name,
        "sample_schedules": RAFM_SCHEDULES_PATH,
        "unfold_result": RAFM_UNFOLD_MLEM_PATH,
    }


def load_phase6_real_optimization_grids(sample_id: str = DEFAULT_PHASE6_SAMPLE_ID) -> dict[str, str]:
    analysis = load_phase6_example_analysis(sample_id)
    schedule = load_phase6_schedule_entry(sample_id)
    phase1 = schedule.get("phase1") or {}
    phase2 = schedule.get("phase2") or {}
    cooling_values = [
        _safe_float(row.get("seconds"))
        for row in list(phase1.get("cooling_times") or []) + list(phase2.get("cooling_times") or [])
        if _safe_float(row.get("seconds")) > 0.0
    ]
    live_time_s = _safe_float((analysis.get("measurement_qc") or [{}])[0].get("live_time_s"))
    if live_time_s <= 0.0:
        live_time_s = _safe_float(analysis.get("timing", {}).get("irradiation_time_s"))
    live_time_s = max(live_time_s, 1.0)
    irradiation_values = [
        _safe_float(phase1.get("irradiation_seconds")),
        _safe_float(phase2.get("irradiation_seconds")),
    ]
    irradiation_values = [value for value in irradiation_values if value > 0.0]
    cooling_values = sorted({int(round(value)) for value in cooling_values if value > 0.0})
    return {
        "irradiation_grid_s": ",".join(str(int(round(value))) for value in irradiation_values),
        "cooldown_grid_s": ",".join(str(value) for value in cooling_values),
        "count_grid_s": str(int(round(live_time_s))),
    }


def load_phase6_real_target_weights(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
    *,
    top_n: int = 3,
) -> dict[str, float]:
    analysis = load_phase6_example_analysis(sample_id)
    isotope_rows = []
    for raw_nuclide, payload in (analysis.get("isotopes") or {}).items():
        if not isinstance(payload, dict):
            continue
        activity = max(
            _safe_float(payload.get("activity_eoi_bq")),
            _safe_float(payload.get("activity_bq")),
        )
        if activity <= 0.0:
            continue
        isotope_rows.append((_normalize_nuclide(raw_nuclide), activity))
    isotope_rows.sort(key=lambda item: item[1], reverse=True)
    selected = isotope_rows[: max(int(top_n), 1)]
    if not selected:
        return {}
    max_activity = max(activity for _nuclide, activity in selected)
    return {
        nuclide: round(activity / max_activity, 6)
        for nuclide, activity in selected
    }


def load_phase6_real_target_weights_text(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
    *,
    top_n: int = 3,
) -> str:
    weights = load_phase6_real_target_weights(sample_id, top_n=top_n)
    return ",".join(f"{nuclide}:{weight:.6f}" for nuclide, weight in weights.items())


def load_phase6_real_activity_review(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
) -> ActivityReviewResult:
    analysis = load_phase6_example_analysis(sample_id)
    timing = analysis.get("timing") or {}
    cooling_time_s = max(_safe_float(timing.get("decay_time_s")), 0.0)
    live_time_s = max(
        _safe_float((analysis.get("measurement_qc") or [{}])[0].get("live_time_s")),
        1.0,
    )
    schedule_entry = load_phase6_schedule_entry(sample_id)
    horizon_candidates = [
        cooling_time_s,
        *[
            _safe_float(row.get("seconds"))
            for row in list((schedule_entry.get("phase1") or {}).get("cooling_times") or [])
            + list((schedule_entry.get("phase2") or {}).get("cooling_times") or [])
            if _safe_float(row.get("seconds")) > 0.0
        ],
    ]
    plot_horizon_s = max(horizon_candidates or [cooling_time_s, 86400.0])

    line_diagnostics = [
        row
        for row in analysis.get("line_diagnostics", []) or []
        if isinstance(row, dict)
    ]
    peaks = [
        row
        for row in analysis.get("peaks", []) or []
        if isinstance(row, dict) and str(row.get("isotope") or "").strip()
    ]
    isotopes = analysis.get("isotopes") or {}

    half_lives_s: dict[str, float] = {}
    peak_groups: dict[str, list[dict[str, Any]]] = {}
    for row in peaks:
        nuclide = _normalize_nuclide(str(row.get("isotope") or ""))
        if not nuclide:
            continue
        peak_groups.setdefault(nuclide, []).append(row)

    isotope_summaries: list[ActivityReviewIsotopeSummary] = []
    for raw_nuclide, payload in isotopes.items():
        if not isinstance(payload, dict):
            continue
        nuclide = _normalize_nuclide(str(raw_nuclide))
        group = peak_groups.get(nuclide, [])
        count_time_activity_bq = _safe_float(payload.get("activity_bq"))
        count_time_uncertainty_bq = _safe_float(payload.get("activity_unc_bq"))
        irradiation_time_activity_bq = max(
            _safe_float(payload.get("activity_eoi_bq")),
            count_time_activity_bq,
        )
        irradiation_time_uncertainty_bq = max(
            _safe_float(payload.get("activity_eoi_unc_bq")),
            count_time_uncertainty_bq,
        )
        half_life_s = _infer_half_life_s(
            count_time_activity_bq,
            irradiation_time_activity_bq,
            cooling_time_s,
        )
        half_lives_s[nuclide] = half_life_s
        peak_energies = tuple(
            sorted(float(value) for value in (payload.get("peak_energies") or []) if value is not None)
        )
        matched_line_energies = []
        for peak in group:
            peak_energy = _safe_float(peak.get("energy_keV"))
            line_energy = peak_energy
            for diagnostic in line_diagnostics:
                raw_iso = _normalize_nuclide(str(diagnostic.get("raw_isotope") or ""))
                if raw_iso != nuclide:
                    continue
                diagnostic_energy = _safe_float(
                    diagnostic.get("raw_energy_keV"),
                    default=_safe_float(diagnostic.get("reference_energy_keV")),
                )
                if abs(diagnostic_energy - peak_energy) > 0.5:
                    continue
                line_energy = _safe_float(
                    diagnostic.get("reference_energy_keV"),
                    default=peak_energy,
                )
                break
            matched_line_energies.append(float(line_energy))
        isotope_summaries.append(
            ActivityReviewIsotopeSummary(
                nuclide=nuclide,
                line_count=int(payload.get("n_peaks") or len(group)),
                peak_energies_keV=peak_energies,
                matched_line_energies_keV=tuple(sorted(matched_line_energies or peak_energies)),
                total_net_counts=float(
                    sum(_safe_float(item.get("net_counts")) for item in group)
                ),
                half_life_s=float(half_life_s),
                count_time_activity_bq=float(count_time_activity_bq),
                count_time_uncertainty_bq=float(count_time_uncertainty_bq),
                irradiation_time_activity_bq=float(irradiation_time_activity_bq),
                irradiation_time_uncertainty_bq=float(irradiation_time_uncertainty_bq),
                cooling_time_s=float(cooling_time_s),
                chain_summary=build_simple_bateman_summary(
                    nuclide,
                    half_life_s=float(half_life_s),
                    cooling_time_s=float(cooling_time_s),
                ),
            )
        )

    line_results: list[ActivityReviewLineResult] = []
    for index, peak in enumerate(peaks, start=1):
        nuclide = _normalize_nuclide(str(peak.get("isotope") or ""))
        if not nuclide:
            continue
        peak_energy_keV = _safe_float(peak.get("energy_keV"))
        matched_line_energy_keV = peak_energy_keV
        emission_probability = 1.0
        emission_probability_uncertainty = 0.0
        for diagnostic in line_diagnostics:
            raw_iso = _normalize_nuclide(str(diagnostic.get("raw_isotope") or ""))
            if raw_iso != nuclide:
                continue
            diagnostic_energy = _safe_float(
                diagnostic.get("raw_energy_keV"),
                default=_safe_float(diagnostic.get("reference_energy_keV")),
            )
            if abs(diagnostic_energy - peak_energy_keV) > 0.5:
                continue
            matched_line_energy_keV = _safe_float(
                diagnostic.get("reference_energy_keV"),
                default=peak_energy_keV,
            )
            emission_probability = max(
                _safe_float(
                    diagnostic.get("raw_branching_ratio"),
                    default=_safe_float(diagnostic.get("reference_rad_int_fraction"), default=1.0),
                ),
                1.0e-12,
            )
            emission_probability_uncertainty = max(
                _safe_float(diagnostic.get("raw_branching_ratio_uncertainty")),
                0.0,
            )
            break
        count_time_activity_bq = _safe_float(peak.get("activity_bq"))
        irradiation_time_activity_bq = max(
            _safe_float(peak.get("eoi_activity_bq")),
            count_time_activity_bq,
        )
        half_life_s = half_lives_s.get(
            nuclide,
            _infer_half_life_s(
                count_time_activity_bq,
                irradiation_time_activity_bq,
                cooling_time_s,
            ),
        )
        line_results.append(
            ActivityReviewLineResult(
                peak_id=f"{sample_id}-peak-{index}",
                nuclide=nuclide,
                peak_energy_keV=float(peak_energy_keV),
                line_energy_keV=float(matched_line_energy_keV),
                line_delta_keV=float(abs(matched_line_energy_keV - peak_energy_keV)),
                net_counts=max(_safe_float(peak.get("net_counts")), 0.0),
                net_counts_uncertainty=max(_safe_float(peak.get("net_counts_unc")), 0.0),
                efficiency=max(_safe_float(peak.get("efficiency"), default=1.0e-12), 1.0e-12),
                efficiency_rel_uncertainty=0.0,
                emission_probability=float(emission_probability),
                emission_probability_uncertainty=float(emission_probability_uncertainty),
                half_life_s=float(half_life_s),
                count_time_activity_bq=float(count_time_activity_bq),
                count_time_uncertainty_bq=max(_safe_float(peak.get("activity_unc_bq")), 0.0),
                irradiation_time_activity_bq=float(irradiation_time_activity_bq),
                irradiation_time_uncertainty_bq=max(
                    _safe_float(peak.get("eoi_activity_unc_bq")),
                    _safe_float(peak.get("activity_unc_bq")),
                ),
                cooling_time_s=float(cooling_time_s),
            )
        )

    decay_plot_data: dict[str, tuple[tuple[float, float, float], ...]] = {}
    bateman_plot_data: dict[str, tuple[tuple[float, float, float], ...]] = {}
    bateman_half_lives_s: dict[str, float] = {}
    for summary in isotope_summaries:
        decay_plot_data[summary.nuclide] = _series_points(
            summary.irradiation_time_activity_bq,
            summary.irradiation_time_uncertainty_bq,
            summary.half_life_s,
            plot_horizon_s,
        )
        parent_label = f"{summary.nuclide} parent"
        daughter_label = f"{summary.nuclide} daughter eq"
        bateman_plot_data[parent_label] = decay_plot_data[summary.nuclide]
        bateman_plot_data[daughter_label] = _daughter_series(
            summary.irradiation_time_activity_bq,
            summary.irradiation_time_uncertainty_bq,
            summary.half_life_s,
            plot_horizon_s,
        )
        bateman_half_lives_s[parent_label] = summary.half_life_s

    return ActivityReviewResult(
        source_id="fluxforge_bundled_gamma",
        custom_gamma_path=None,
        live_time_s=float(live_time_s),
        cooling_time_s=float(cooling_time_s),
        plot_horizon_s=float(plot_horizon_s),
        line_results=tuple(
            sorted(line_results, key=lambda item: (item.nuclide, item.peak_energy_keV))
        ),
        isotope_summaries=tuple(
            sorted(isotope_summaries, key=lambda item: item.irradiation_time_activity_bq, reverse=True)
        ),
        decay_plot_data=decay_plot_data,
        bateman_plot_data=bateman_plot_data,
        half_lives_s=half_lives_s,
        bateman_half_lives_s=bateman_half_lives_s,
    )


def load_phase6_real_activity_review_payload(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
) -> dict[str, Any]:
    analysis = load_phase6_example_analysis(sample_id)
    review = load_phase6_real_activity_review(sample_id)
    payload = review.to_payload()
    payload["spectrum_id"] = str(sample_id)
    payload["sample_group"] = analysis.get("sample_group")
    payload["schedule_source"] = str(RAFM_SCHEDULES_PATH)
    payload["analysis_json_source"] = str(RAFM_ANALYSIS_ROOT / f"{sample_id}.json")
    payload["measurement_time"] = (analysis.get("timing") or {}).get("measurement_time")
    return payload


def load_phase6_real_activity_results(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
) -> tuple[ActivityCalculationResult, ...]:
    review = load_phase6_real_activity_review(sample_id)
    results = []
    for summary in review.isotope_summaries:
        results.append(
            ActivityCalculationResult(
                nuclide=summary.nuclide,
                line_energy_keV=float(
                    summary.matched_line_energies_keV[0]
                    if summary.matched_line_energies_keV
                    else 0.0
                ),
                activity_bq=float(summary.count_time_activity_bq),
                uncertainty_bq=float(summary.count_time_uncertainty_bq),
                age_corrected_activity_bq=float(summary.irradiation_time_activity_bq),
                mda_bq=0.0,
                half_life_s=float(summary.half_life_s),
                source_age_s=float(review.cooling_time_s),
                chain_summary=str(summary.chain_summary),
                age_corrected_uncertainty_bq=float(summary.irradiation_time_uncertainty_bq),
            )
        )
    return tuple(results)


def load_phase6_real_spectrum(
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
) -> GammaSpectrum:
    analysis = load_phase6_example_analysis(sample_id)
    paths = load_phase6_real_data_paths(sample_id)
    channels: list[float] = []
    energies: list[float] = []
    counts: list[float] = []
    counts_uncertainty: list[float] = []
    with paths["counts_csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            channels.append(_safe_float(row.get("channel")))
            energies.append(_safe_float(row.get("energy_keV")))
            counts.append(_safe_float(row.get("raw_counts")))
            counts_uncertainty.append(
                _safe_float(
                    row.get("background_adjusted_unc"),
                    default=math.sqrt(max(_safe_float(row.get("raw_counts")), 0.0)),
                )
            )
    measurement_qc = (analysis.get("measurement_qc") or [{}])[0]
    calibration_coefficients = (
        (analysis.get("analysis_configuration") or {}).get(
            "energy_calibration_coefficients"
        )
        or [0.0, 1.0]
    )
    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        counts_uncertainty=np.asarray(counts_uncertainty, dtype=float),
        channels=np.asarray(channels, dtype=float),
        energies=np.asarray(energies, dtype=float),
        live_time=max(_safe_float(measurement_qc.get("live_time_s")), 1.0),
        real_time=max(
            _safe_float(measurement_qc.get("real_time_s")),
            _safe_float(measurement_qc.get("live_time_s")),
            1.0,
        ),
        spectrum_id=str(sample_id),
        calibration={"energy": list(calibration_coefficients)},
        metadata={
            "sample_group": analysis.get("sample_group"),
            "counts_csv": str(paths["counts_csv"]),
        },
    )


def write_phase6_real_second_irradiation_inputs(
    directory: Path,
    sample_id: str = DEFAULT_PHASE6_SAMPLE_ID,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    schedule_entry = load_phase6_schedule_entry(sample_id)
    phase1 = schedule_entry.get("phase1") or {}
    phase2 = schedule_entry.get("phase2") or {}
    activity_review_payload = load_phase6_real_activity_review_payload(sample_id)
    target_weights = load_phase6_real_target_weights(sample_id)
    inventory_path = directory / f"{sample_id}_activity_review.json"
    inventory_path.write_text(
        json.dumps(activity_review_payload, indent=2),
        encoding="utf-8",
    )

    schedule_payload = {
        "sample_id": sample_id,
        "schedule_source": str(RAFM_SCHEDULES_PATH),
        "first_cooling_time_s": float(activity_review_payload.get("cooling_time_s") or 0.0),
        "second_irradiation_time_s": float(phase2.get("irradiation_seconds") or 0.0),
        "target_weights": target_weights,
        "observed_phase1_windows_s": [
            _safe_float(row.get("seconds"))
            for row in phase1.get("cooling_times") or []
            if _safe_float(row.get("seconds")) > 0.0
        ],
    }
    schedule_path = directory / f"{sample_id}_second_irradiation_schedule.json"
    schedule_path.write_text(json.dumps(schedule_payload, indent=2), encoding="utf-8")

    candidates_payload = {
        "candidates": [
            {
                "label": f"{sample_id}_window_{row.get('label')}",
                "flux_scale": 1.0,
                "duration_factor": 1.0,
                "second_cooling_time_s": float(row.get("seconds") or 0.0),
            }
            for row in phase1.get("cooling_times") or []
            if _safe_float(row.get("seconds")) > 0.0
        ]
    }
    candidates_path = directory / f"{sample_id}_second_irradiation_candidates.json"
    candidates_path.write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    return {
        "inventory": inventory_path,
        "schedule": schedule_path,
        "candidates": candidates_path,
    }


__all__ = [
    "DEFAULT_PHASE6_SAMPLE_ID",
    "RAFM_UNFOLD_MLEM_PATH",
    "load_phase6_example_analysis",
    "load_phase6_real_activity_results",
    "load_phase6_real_activity_review",
    "load_phase6_real_activity_review_payload",
    "load_phase6_real_data_paths",
    "load_phase6_real_optimization_grids",
    "load_phase6_real_spectrum",
    "load_phase6_real_target_weights",
    "load_phase6_real_target_weights_text",
    "load_phase6_sample_schedules",
    "load_phase6_schedule_entry",
    "write_phase6_real_second_irradiation_inputs",
]
