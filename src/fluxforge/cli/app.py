"""Command-line interface for FluxForge using argparse."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

from fluxforge.core.unfolding_inputs import require_nonnegative
from fluxforge.analysis.segmented_detection import (
    SegmentedDetectionConfig,
    detect_peaks_segmented,
)
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.analysis.k0_workflow import (
    CAPABILITY_FLAGS as K0_CAPABILITY_FLAGS,
    aggregate_k0_analysis_bundles,
    analyze_k0_observations,
    build_k0_report_payload,
    build_detector_characterization,
    build_facility_characterization,
    evaluate_k0_qaqc,
    peak_report_to_observations,
    resolve_governed_libraries,
)
from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan
from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan
from fluxforge.analysis.astm_e3376 import analyze_astm_e3376_plan
from fluxforge.analysis.astm_e2005 import analyze_astm_e2005_plan
from fluxforge.analysis.optimization_difom import (
    parse_difom_sweep_payload,
    rank_difom_schedules,
    serialize_difom_ranking,
)
from fluxforge.analysis.optimization_fim import (
    rank_fim_schedules,
    serialize_fim_ranking,
)
from fluxforge.analysis.optimization_mwdcs import (
    parse_mwdcs_sweep_payload,
    rank_mwdcs_schedules,
    serialize_mwdcs_ranking,
)
from fluxforge.analysis.optimization_bassd import (
    parse_bassd_sweep_payload,
    rank_bassd_schedules,
    serialize_bassd_ranking,
)
from fluxforge.analysis.optimization_stbdmr import (
    parse_stbdmr_sweep_payload,
    rank_stbdmr_schedules,
    serialize_stbdmr_ranking,
)
from fluxforge.analysis.optimization_schedule_builder import (
    build_difom_payload_from_activity_review,
    summarize_neutron_spectrum_source,
)
from fluxforge.analysis.isotope_priority import (
    IsotopePriorityWeights,
    rank_isotopes_from_activity_review_payload,
    serialize_isotope_priority_ranking,
)
from fluxforge.analysis.masking_review import (
    rank_line_masking_from_activity_review_payload,
    recommend_alternate_lines,
    serialize_masking_review,
    summarize_masking_isotopes,
)
from fluxforge.core.prior_covariance import PriorCovarianceConfig, PriorCovarianceModel
from fluxforge.core.planning_models import OptimizationScenario
from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    build_response_matrix,
)
from fluxforge.core.schemas import validate_or_raise
from fluxforge.core.activity_review import review_spectrum_activation
from fluxforge.core.analysis_workspace import (
    ContinuumDriverEstimate,
    PeakCandidate,
    analyze_roi_region,
    compute_roi_statistics,
    detect_peak_candidates,
    estimate_dominant_continuum_driver,
    estimate_spectral_phenomena,
    register_builtin_peak_search_methods,
)
from fluxforge.analysis.line_search import list_nuclide_lines
from fluxforge.core.inventory_timeline import (
    DEFAULT_DECAY_SOURCE_ID,
    TIME_ORIGINS,
    build_inventory_state_from_payload,
    build_time_grid,
    compute_inventory_time_evolution,
)
from fluxforge.core.unfolding_diagnostics import merge_flux_diagnostics
from fluxforge.data.efficiency_models import EfficiencyModel
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.kayzero_k0 import (
    import_kayzero_k0_library,
    write_governed_library_json,
    write_import_report_json,
)
from fluxforge.data.nuclear_data_sources import (
    list_nuclear_data_sources,
    list_nuclear_data_sources_by_capability,
    register_user_gamma_source,
    remove_user_gamma_source,
)
from fluxforge.data.rafm_profile import list_rafm_profiles, load_rafm_profile
from fluxforge.examples.rafm_workflow import (
    compare_rafm_completion_results,
    run_qg_benchmark,
    run_rafm_validation,
)
from fluxforge.io.artifacts import (
    read_line_activities,
    read_detector_characterization,
    read_facility_characterization,
    read_ffexp_bundle,
    read_k0_aggregation_bundle,
    read_k0_analysis_bundle,
    read_peak_observation_bundle,
    read_peak_report,
    read_k0_qaqc_bundle,
    read_reaction_rates,
    read_response_bundle,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_detector_characterization,
    write_facility_characterization,
    write_ffexp_bundle,
    write_k0_aggregation_bundle,
    write_k0_analysis_bundle,
    write_k0_qaqc_bundle,
    write_line_activities,
    write_peak_observation_bundle,
    write_peak_report,
    write_reaction_rates,
    write_report_bundle,
    write_response_bundle,
    write_spectrum_file,
    write_unfold_result,
    write_validation_bundle,
)
from fluxforge.io.genie import read_genie_spectrum
from fluxforge.io.spe import GammaSpectrum, read_spe_file
from fluxforge.plots.activation import plot_decay_curves
from fluxforge.physics.activation import (
    IrradiationSegment,
    activation_study_metrics,
    irradiation_buildup_factor,
    reaction_rate_from_activity,
)
from fluxforge.solvers.gls import gls_adjust
from fluxforge.solvers.iterative import gravel, mlem
from fluxforge.unfolding import GravelUnfolder, MLSeedUnfolder, MaxedUnfolder, RMLEUnfolder
from fluxforge.validation import (
    load_phase5_crosswalk,
    render_phase5_crosswalk_markdown,
    run_reference_parity_suite,
    spectrum_comparison_metrics,
    summarize_phase5_crosswalk,
)
from fluxforge.plugins import PluginRegistries
from fluxforge.workflows.irradiation_optimization import (
    build_phase6_support_artifacts,
    parse_second_irradiation_candidates,
    plan_second_irradiation,
    serialize_second_irradiation_plan,
)
from fluxforge.workflows.phase6_ldrd_worked_example import (
    DEFAULT_SAMPLE_ID as PHASE6_LDRD_DEFAULT_SAMPLE_ID,
    default_output_root as phase6_ldrd_default_output_root,
    run_phase6_ldrd_worked_example,
)
from fluxforge.workflows.phase6_ldrd_second_irradiation_decision_repo import (
    default_output_root as phase6_ldrd_second_irradiation_repo_default_output_root,
    run_phase6_ldrd_second_irradiation_decision_repo,
)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_structured_rows(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(
        f"Unsupported structured input format for {path}. Use .json or .csv."
    )


def _extract_structured_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in (
            "rows",
            "items",
            "lines",
            "reactions",
            "peaks",
            "statistics",
            "results",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _peak_search_cli_choices() -> tuple[str, ...]:
    registries = register_builtin_peak_search_methods(PluginRegistries())
    keys = tuple(registries.peak_search_methods.keys())
    if "segmented" in keys:
        return keys
    return ("segmented", *keys)


def _write_dict_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent_dir(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _parse_csv_floats(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        return None
    return [float(value) for value in values]


def _parse_csv_strings(raw: Optional[str]) -> tuple[str, ...]:
    if raw is None:
        return ()
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    return tuple(values)


def _coerce_irradiation_segments(
    *,
    irradiation_time_s: float,
    segments_payload: Any | None,
) -> list[IrradiationSegment]:
    if isinstance(segments_payload, list) and segments_payload:
        segments: list[IrradiationSegment] = []
        for index, item in enumerate(segments_payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Irradiation segment #{index} must be an object with duration_s and optional relative_power."
                )
            duration_s = _safe_float(item.get("duration_s"), default=0.0)
            relative_power = _safe_float(item.get("relative_power"), default=1.0)
            if duration_s <= 0.0:
                continue
            segments.append(
                IrradiationSegment(
                    duration_s=float(duration_s),
                    relative_power=float(relative_power),
                )
            )
        if segments:
            return segments

    duration_s = max(float(irradiation_time_s), 0.0)
    if duration_s <= 0.0:
        return []
    return [IrradiationSegment(duration_s=duration_s, relative_power=1.0)]


def _build_reaction_rate_rows_from_activity_review(
    *,
    isotope_rows: Sequence[dict[str, Any]],
    segments: Sequence[IrradiationSegment],
) -> list[dict[str, Any]]:
    if not segments:
        raise ValueError(
            "Reaction-rate export requires a positive irradiation duration or segment file."
        )

    rows: list[dict[str, Any]] = []
    segment_factor_cache: dict[float, float] = {}

    for row in isotope_rows:
        nuclide = str(row.get("nuclide") or "").strip()
        if not nuclide:
            continue
        activity_eoi = _safe_float(row.get("irradiation_time_activity_Bq"), default=0.0)
        activity_unc = _safe_float(
            row.get("irradiation_time_activity_unc_Bq"),
            default=0.0,
        )
        half_life_s = max(_safe_float(row.get("half_life_s"), default=0.0), 0.0)
        if activity_eoi <= 0.0 or half_life_s <= 0.0:
            continue

        if half_life_s not in segment_factor_cache:
            buildup = irradiation_buildup_factor(segments, half_life_s)
            segment_factor_cache[half_life_s] = max(float(buildup), 1.0e-12)
        factor = segment_factor_cache[half_life_s]

        rate = activity_eoi / factor
        rate_unc = activity_unc / factor
        rows.append(
            {
                "reaction_id": nuclide,
                "nuclide": nuclide,
                "line_count": int(_safe_float(row.get("line_count"), default=0.0)),
                "half_life_s": float(half_life_s),
                "cooling_time_s": float(
                    _safe_float(row.get("cooling_time_s"), default=0.0)
                ),
                "activity_eoi_Bq": float(activity_eoi),
                "activity_eoi_unc_Bq": float(activity_unc),
                "reaction_rate_s": float(rate),
                "reaction_rate_unc_s": float(rate_unc),
                "relative_uncertainty": float(rate_unc / max(rate, 1.0e-12)),
            }
        )

    rows.sort(
        key=lambda item: (
            -_safe_float(item.get("reaction_rate_s")),
            str(item.get("nuclide")),
        )
    )
    return rows


def _candidate_lines_for_recommendation(candidate: Any) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict):
        return []
    lines: list[dict[str, Any]] = []
    base_lines = candidate.get("lines")
    if isinstance(base_lines, list):
        lines.extend(item for item in base_lines if isinstance(item, dict))
    windows = candidate.get("windows")
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict):
                continue
            window_lines = window.get("lines")
            if isinstance(window_lines, list):
                lines.extend(item for item in window_lines if isinstance(item, dict))
    actions = candidate.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_lines = action.get("lines")
            if isinstance(action_lines, list):
                lines.extend(item for item in action_lines if isinstance(item, dict))
    return lines


def _optimization_objective_score(candidate: dict[str, Any]) -> float:
    return _safe_float(
        candidate.get("difom_score"),
        default=_safe_float(
            candidate.get("objective_score"),
            default=_safe_float(
                candidate.get("total_score"),
                default=_safe_float(candidate.get("total_utility")),
            ),
        ),
    )


def _optimization_scenario_from_ranked_candidate(
    objective: str,
    candidate: dict[str, Any],
) -> OptimizationScenario:
    return OptimizationScenario(
        objective=str(objective or "unknown"),
        label=str(candidate.get("label") or ""),
        rank=int(_safe_float(candidate.get("rank"), default=0.0)),
        irradiation_time_s=_safe_float(candidate.get("irradiation_time_s")),
        cooldown_time_s=_safe_float(
            candidate.get("cooldown_time_s"),
            default=_safe_float(
                ((candidate.get("window_scores") or [{}])[0]).get("cooldown_time_s")
            ),
        ),
        count_time_s=_safe_float(
            candidate.get("count_time_s"),
            default=_safe_float(
                ((candidate.get("window_scores") or [{}])[0]).get("count_time_s")
            ),
        ),
        objective_score=_optimization_objective_score(candidate),
    )


def _recommendation_from_optimization_payload(
    optimization_payload: dict[str, Any],
    *,
    optimization_file: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = optimization_payload.get("ranked_candidates") or []
    if not isinstance(ranked, list):
        ranked = []

    objective = str(optimization_payload.get("objective") or "unknown")
    recommendation: dict[str, Any] = {
        "objective": objective,
        "isotopes_of_interest": list(
            optimization_payload.get("isotopes_of_interest") or []
        ),
    }
    if not ranked:
        return recommendation, []

    top = ranked[0] if isinstance(ranked[0], dict) else {}
    top_scenario = _optimization_scenario_from_ranked_candidate(objective, top)
    recommendation.update(
        {
            "recommended_label": top_scenario.label,
            "recommended_rank": int(top_scenario.rank),
            "recommended_irradiation_time_s": float(top_scenario.irradiation_time_s),
            "recommended_cooldown_time_s": float(top_scenario.cooldown_time_s),
            "recommended_count_time_s": float(top_scenario.count_time_s),
            "recommended_objective_score": float(top_scenario.objective_score),
            "recommended_scenario": top_scenario.to_row(),
        }
    )

    if isinstance(top.get("action_scores"), list) and top["action_scores"]:
        recommendation["recommended_expected_dose_uSv"] = float(
            sum(
                _safe_float(item.get("expected_dose_uSv"))
                for item in top["action_scores"]
                if isinstance(item, dict)
            )
        )

    input_payload: dict[str, Any] | None = None
    input_path_raw = optimization_payload.get("input")
    if isinstance(input_path_raw, str) and input_path_raw.strip():
        input_path = Path(input_path_raw)
        if not input_path.is_absolute():
            input_path = optimization_file.parent / input_path
        if input_path.exists():
            try:
                input_payload = _load_json(input_path)
            except Exception:
                input_payload = None

    candidate_lines: list[dict[str, Any]] = []
    if isinstance(input_payload, dict):
        candidates_raw = input_payload.get("candidates") or []
        if isinstance(candidates_raw, list):
            for item in candidates_raw:
                if not isinstance(item, dict):
                    continue
                if str(item.get("label")) != str(top.get("label")):
                    continue
                candidate_lines = _candidate_lines_for_recommendation(item)
                break

    if candidate_lines:
        mask_scores: dict[str, float] = {}
        for line in candidate_lines:
            nuclide = str(line.get("nuclide") or line.get("isotope") or "").strip()
            if not nuclide:
                continue
            interference = _safe_float(line.get("interference_counts"))
            interference += _safe_float(line.get("continuum_counts"))
            background = _safe_float(line.get("background_counts"))
            mask_scores[nuclide] = mask_scores.get(nuclide, 0.0) + max(
                interference + 0.25 * background,
                0.0,
            )
        if mask_scores:
            recommendation["recommended_mask_isotope"] = max(
                mask_scores.items(), key=lambda item: item[1]
            )[0]

    if recommendation.get("isotopes_of_interest"):
        recommendation["recommended_isotope_of_interest"] = recommendation[
            "isotopes_of_interest"
        ][0]

    top_scenario = OptimizationScenario(
        objective=top_scenario.objective,
        label=top_scenario.label,
        rank=top_scenario.rank,
        irradiation_time_s=top_scenario.irradiation_time_s,
        cooldown_time_s=top_scenario.cooldown_time_s,
        count_time_s=top_scenario.count_time_s,
        objective_score=top_scenario.objective_score,
        mask_isotope=(
            str(recommendation.get("recommended_mask_isotope"))
            if recommendation.get("recommended_mask_isotope")
            else None
        ),
        isotope_of_interest=(
            str(recommendation.get("recommended_isotope_of_interest"))
            if recommendation.get("recommended_isotope_of_interest")
            else None
        ),
        expected_dose_uSv=(
            _safe_float(recommendation.get("recommended_expected_dose_uSv"))
            if "recommended_expected_dose_uSv" in recommendation
            else None
        ),
    )
    recommendation["recommended_scenario"] = top_scenario.to_row()

    top_rows: list[dict[str, Any]] = []
    for item in ranked[:10]:
        if not isinstance(item, dict):
            continue
        top_rows.append(
            _optimization_scenario_from_ranked_candidate(objective, item).to_row()
        )

    return recommendation, top_rows


def _normalize_nuclide_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _extract_line_nuclide(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("nuclide") or item.get("isotope") or "").strip()


def _count_candidate_lines(candidate: Any) -> int:
    if not isinstance(candidate, dict):
        return 0
    total = 0
    lines = candidate.get("lines")
    if isinstance(lines, list):
        total += len(lines)
    windows = candidate.get("windows")
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict):
                continue
            window_lines = window.get("lines")
            if isinstance(window_lines, list):
                total += len(window_lines)
    actions = candidate.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_lines = action.get("lines")
            if isinstance(action_lines, list):
                total += len(action_lines)
    return total


def _candidate_has_line_data(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    lines = candidate.get("lines")
    if isinstance(lines, list) and len(lines) > 0:
        return True
    windows = candidate.get("windows")
    if isinstance(windows, list) and len(windows) > 0:
        return True
    actions = candidate.get("actions")
    if isinstance(actions, list) and len(actions) > 0:
        return True
    return False


def _filter_optimization_payload_by_isotopes(
    payload: Any,
    isotopes_of_interest: tuple[str, ...],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Optimization payload must be a JSON object.")

    filtered_payload = copy.deepcopy(payload)
    candidates_raw = filtered_payload.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("Optimization payload requires a non-empty 'candidates' list.")

    before_candidates = len(candidates_raw)
    before_lines = sum(_count_candidate_lines(item) for item in candidates_raw)

    requested = tuple(dict.fromkeys(str(item).strip() for item in isotopes_of_interest if str(item).strip()))
    if len(requested) == 0:
        return filtered_payload, {
            "requested_isotopes": [],
            "candidates_before": before_candidates,
            "candidates_after": before_candidates,
            "line_terms_before": before_lines,
            "line_terms_after": before_lines,
        }

    allowed = {_normalize_nuclide_label(item) for item in requested}

    filtered_candidates: list[Any] = []
    for candidate in candidates_raw:
        if not isinstance(candidate, dict):
            continue
        candidate_copy = dict(candidate)

        base_lines = candidate_copy.get("lines")
        if isinstance(base_lines, list):
            candidate_copy["lines"] = [
                line
                for line in base_lines
                if _normalize_nuclide_label(_extract_line_nuclide(line)) in allowed
            ]

        windows = candidate_copy.get("windows")
        if isinstance(windows, list):
            filtered_windows: list[Dict[str, Any]] = []
            for window in windows:
                if not isinstance(window, dict):
                    continue
                window_copy = dict(window)
                window_lines = window_copy.get("lines")
                if isinstance(window_lines, list):
                    window_copy["lines"] = [
                        line
                        for line in window_lines
                        if _normalize_nuclide_label(_extract_line_nuclide(line)) in allowed
                    ]
                if isinstance(window_copy.get("lines"), list) and len(window_copy["lines"]) > 0:
                    filtered_windows.append(window_copy)
            candidate_copy["windows"] = filtered_windows

        actions = candidate_copy.get("actions")
        if isinstance(actions, list):
            filtered_actions: list[Dict[str, Any]] = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_copy = dict(action)
                action_lines = action_copy.get("lines")
                if isinstance(action_lines, list):
                    action_copy["lines"] = [
                        line
                        for line in action_lines
                        if _normalize_nuclide_label(_extract_line_nuclide(line)) in allowed
                    ]
                if isinstance(action_copy.get("lines"), list) and len(action_copy["lines"]) > 0:
                    filtered_actions.append(action_copy)
            candidate_copy["actions"] = filtered_actions

        if _candidate_has_line_data(candidate_copy):
            filtered_candidates.append(candidate_copy)

    filtered_payload["candidates"] = filtered_candidates
    weights_raw = filtered_payload.get("isotope_weights")
    if isinstance(weights_raw, dict):
        filtered_payload["isotope_weights"] = {
            key: value
            for key, value in weights_raw.items()
            if _normalize_nuclide_label(str(key)) in allowed
        }

    after_lines = sum(_count_candidate_lines(item) for item in filtered_candidates)
    return filtered_payload, {
        "requested_isotopes": list(requested),
        "candidates_before": before_candidates,
        "candidates_after": len(filtered_candidates),
        "line_terms_before": before_lines,
        "line_terms_after": after_lines,
    }


def _write_phase6_support_artifacts(
    *,
    output_path: Path,
    activity_review_payload: Optional[Dict[str, Any]],
    optimization_payload: Optional[Dict[str, Any]],
    isotopes_of_interest: tuple[str, ...],
) -> Dict[str, Path]:
    if activity_review_payload is None:
        return {}

    support = build_phase6_support_artifacts(
        activity_review_payload,
        optimization_payload,
        isotopes_of_interest=isotopes_of_interest,
    )

    outputs = {
        "optimization_grid": _phase6_artifact_path(output_path, "_optimization_grid.csv"),
        "recommended_schedules": _phase6_artifact_path(
            output_path, "_recommended_schedules.csv"
        ),
        "dose_endpoints": _phase6_artifact_path(output_path, "_dose_endpoints.csv"),
        "masking_candidates": _phase6_artifact_path(
            output_path, "_masking_candidates.csv"
        ),
        "inventory_timeseries": _phase6_artifact_path(
            output_path, "_inventory_timeseries.csv"
        ),
        "activities_at_irradiation": _phase6_artifact_path(
            output_path, "_activities_at_irradiation.csv"
        ),
    }

    _write_dict_rows(outputs["optimization_grid"], support["optimization_grid_rows"])
    _write_dict_rows(
        outputs["recommended_schedules"],
        support["recommended_schedule_rows"],
    )
    _write_dict_rows(outputs["dose_endpoints"], support["dose_endpoints_rows"])
    _write_dict_rows(outputs["masking_candidates"], support["masking_candidates_rows"])
    _write_dict_rows(outputs["inventory_timeseries"], support["inventory_timeseries_rows"])
    _write_dict_rows(
        outputs["activities_at_irradiation"],
        support["activities_at_irradiation_rows"],
    )
    return outputs


def _build_ffexp_summary(
    *,
    activity_review_payload: Optional[Dict[str, Any]] = None,
    optimization_payload: Optional[Dict[str, Any]] = None,
    second_irradiation_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    isotope_rows = (
        list(activity_review_payload.get("isotope_summaries") or [])
        if activity_review_payload is not None
        else []
    )
    line_rows = (
        list(activity_review_payload.get("line_results") or [])
        if activity_review_payload is not None
        else []
    )
    ranked_candidates = (
        list(optimization_payload.get("ranked_candidates") or [])
        if optimization_payload is not None
        else []
    )
    summary: Dict[str, Any] = {
        "isotope_count": len(isotope_rows),
        "line_count": len(line_rows),
        "optimization_candidate_count": len(ranked_candidates),
        "peak_fit_complete": bool(line_rows),
        "activity_complete": bool(isotope_rows),
        "naa_quant_complete": bool(isotope_rows),
        "inverse_adjustment_complete": False,
        "comparison_ready": bool(isotope_rows),
    }
    if ranked_candidates:
        top = ranked_candidates[0]
        summary["optimization_objective"] = optimization_payload.get("objective")
        summary["recommended_schedule_label"] = top.get("label")
    if second_irradiation_payload is not None:
        summary["second_irradiation_selected_label"] = (
            second_irradiation_payload.get("selected_candidate") or {}
        ).get("label")
    return summary


def _parse_efficiency_override(raw: Optional[str]) -> Optional[Dict[str, float]]:
    values = _parse_csv_floats(raw)
    if values is None:
        return None
    if len(values) not in {4, 5}:
        raise ValueError(
            "--efficiency-coefficients requires 4 or 5 comma-separated values: "
            "C1,C2,C3,C4[,DetModel]"
        )
    override: Dict[str, float] = {
        "C1": values[0],
        "C2": values[1],
        "C3": values[2],
        "C4": values[3],
    }
    if len(values) == 5:
        override["DetModel"] = values[4]
    return override


def _activity_review_source_choices() -> tuple[str, ...]:
    return tuple(
        record.source_id
        for record in list_nuclear_data_sources_by_capability("peak-identification")
    )


def _inventory_decay_source_choices() -> tuple[str, ...]:
    return tuple(
        record.source_id
        for record in list_nuclear_data_sources_by_capability("inventory-decay")
    )


def _serialize_nuclear_data_source(record) -> dict[str, Any]:
    return {
        "source_id": record.source_id,
        "label": record.label,
        "kind": record.kind,
        "builtin": bool(record.builtin),
        "path_hint": record.path_hint,
        "description": record.description,
        "capabilities": list(record.capabilities),
        "metadata": dict(record.metadata),
    }


def _build_activity_review_efficiency_curve(args: argparse.Namespace) -> EfficiencyCurve:
    rel_uncertainty = max(float(getattr(args, "efficiency_uncertainty", 0.05)), 0.0)
    polynomial = _parse_csv_floats(getattr(args, "efficiency_polynomial", None))
    if polynomial:
        return EfficiencyCurve.from_polynomial(
            coefficients=polynomial,
            energy_range=(1.0, 10000.0),
            uncertainty_model={"type": "constant", "value": rel_uncertainty},
        )

    constant_efficiency = max(float(getattr(args, "efficiency", 1.0)), 1e-12)
    return EfficiencyCurve(
        model_type="empirical",
        parameters={
            "energies": [1.0, 10000.0],
            "efficiencies": [constant_efficiency, constant_efficiency],
            "interpolation": "linear",
        },
        energy_range=(1.0, 10000.0),
        uncertainty_model={"type": "constant", "value": rel_uncertainty},
    )


def _activity_review_artifact_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def _inventory_review_artifact_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def _phase6_artifact_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def _parse_relative_time_points(raw: Optional[str]) -> tuple[float, ...]:
    values = _parse_csv_floats(raw)
    if values is None:
        return ()
    return tuple(float(value) for value in values)


def _close_figure(fig: Any) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def _apply_overrides_to_spectrum(
    spectrum: GammaSpectrum,
    *,
    energy_override: Optional[List[float]],
    efficiency_override: Optional[Dict[str, float]],
) -> GammaSpectrum:
    if energy_override is not None:
        spectrum.calibration["energy"] = [float(c) for c in energy_override]
        spectrum.energies = spectrum.calibrate_channels(spectrum.calibration["energy"])
    if efficiency_override is not None:
        metadata = dict(spectrum.metadata)
        existing_eff = metadata.get("efficiency", {})
        if not isinstance(existing_eff, dict):
            existing_eff = {}
        existing_eff.update({k: float(v) for k, v in efficiency_override.items()})
        metadata["efficiency"] = existing_eff
        spectrum.metadata = metadata
    return spectrum


def _profile_efficiency_override(
    profile_name: Optional[str],
) -> Optional[Dict[str, float]]:
    if not profile_name:
        return None
    profile = load_rafm_profile(profile_name)
    return {
        str(key): float(value)
        for key, value in profile.efficiency.items()
        if isinstance(value, (int, float))
    }


def _profile_background_file(profile_name: Optional[str]) -> Optional[Path]:
    if not profile_name:
        return None
    profile = load_rafm_profile(profile_name)
    background_path = profile.resolve_background_path()
    if background_path is None or not background_path.exists():
        raise FileNotFoundError(
            f"Bundled background spectrum for profile '{profile_name}' was not found: {background_path}"
        )
    return background_path


def _spectrum_energies(spectrum: GammaSpectrum) -> np.ndarray:
    if spectrum.energies is not None:
        return np.asarray(spectrum.energies, dtype=float)
    if spectrum.calibration:
        return np.asarray(spectrum.calibrate_channels(), dtype=float)
    return np.asarray(spectrum.channels, dtype=float)


def _load_spectrum_from_path(
    input_path: Path,
    *,
    validate: bool,
    energy_override: Optional[List[float]] = None,
    efficiency_override: Optional[Dict[str, float]] = None,
) -> GammaSpectrum:
    suffix = input_path.suffix.lower()
    if suffix == ".spe":
        spectrum = read_spe_file(input_path)
    elif suffix in {".asc", ".txt"}:
        spectrum = read_genie_spectrum(
            input_path,
            energy_calibration_override=energy_override,
            efficiency_override=efficiency_override,
        )
    else:
        payload = read_spectrum_file(input_path)
        if validate:
            validate_or_raise(payload)
        spectrum = GammaSpectrum.from_dict(payload["spectrum"])

    return _apply_overrides_to_spectrum(
        spectrum,
        energy_override=energy_override,
        efficiency_override=efficiency_override,
    )


def _normalize_efficiency_coefficients(
    metadata_efficiency: object,
) -> Optional[Dict[str, float]]:
    if not isinstance(metadata_efficiency, dict):
        return None

    key_map = {
        "C1": "C1",
        "C2": "C2",
        "C3": "C3",
        "C4": "C4",
        "DETMODEL": "DetModel",
        "GEOMETRY_FACTOR_A": "DetModel",
        "A": "DetModel",
    }
    normalized: Dict[str, float] = {}
    for raw_key, raw_value in metadata_efficiency.items():
        mapped = key_map.get(str(raw_key).strip().upper())
        if mapped is None:
            continue
        try:
            normalized[mapped] = float(raw_value)
        except (TypeError, ValueError):
            continue

    required = {"C1", "C2", "C3", "C4"}
    if not required.issubset(normalized):
        return None
    normalized.setdefault("DetModel", 1.0)
    return normalized


def _efficiency_model_from_spectrum(
    spectrum: GammaSpectrum,
) -> Optional[EfficiencyModel]:
    coefficients = _normalize_efficiency_coefficients(
        spectrum.metadata.get("efficiency", {})
    )
    if coefficients is None:
        return None
    return EfficiencyModel(
        model_type="log_poly",
        coefficients=coefficients,
        description=f"FluxForge efficiency model for {spectrum.spectrum_id}",
    )


def _export_metadata_lines(
    spectrum: GammaSpectrum,
    *,
    background_file: Optional[Path],
    efficiency_model: Optional[EfficiencyModel] = None,
) -> List[str]:
    lines = [f"spectrum_id={spectrum.spectrum_id}"]
    source_file = spectrum.metadata.get("source_file")
    if source_file:
        lines.append(f"source_file={source_file}")
    if background_file is not None:
        lines.append(f"background_file={background_file}")

    background_info = spectrum.metadata.get("background_subtraction", {})
    if isinstance(background_info, dict):
        for key in (
            "scale_mode",
            "scale_factor",
            "negative_policy",
            "negative_bins",
            "clipped",
            "background_spectrum_id",
        ):
            if key in background_info:
                lines.append(f"{key}={background_info[key]}")

    if efficiency_model is not None:
        for key in ("C1", "C2", "C3", "C4", "DetModel"):
            if key in efficiency_model.coefficients:
                lines.append(f"{key}={efficiency_model.coefficients[key]}")

    return lines


def _write_csv_rows(
    path: Path, metadata_lines: List[str], header: List[str], rows: List[List[float]]
) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in metadata_lines:
            handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _write_background_adjusted_csv(
    output_path: Path,
    spectrum: GammaSpectrum,
    *,
    background_file: Optional[Path],
) -> None:
    energies = _spectrum_energies(spectrum)
    uncertainties = np.asarray(spectrum.counts_uncertainty, dtype=float)
    rows = [
        [int(channel), float(energy), float(count), float(uncertainty)]
        for channel, energy, count, uncertainty in zip(
            spectrum.channels,
            energies,
            spectrum.counts,
            uncertainties,
        )
    ]
    _write_csv_rows(
        output_path,
        _export_metadata_lines(spectrum, background_file=background_file),
        ["channel", "energy_keV", "net_counts", "counts_uncertainty"],
        rows,
    )


def _write_final_corrected_csv(
    output_path: Path,
    spectrum: GammaSpectrum,
    *,
    background_file: Optional[Path],
) -> bool:
    efficiency_model = _efficiency_model_from_spectrum(spectrum)
    if efficiency_model is None:
        warnings.warn(
            (
                "Final corrected export requested but no usable efficiency coefficients were "
                f"available for {spectrum.spectrum_id}; skipping export."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    energies = _spectrum_energies(spectrum)
    efficiencies = np.asarray(efficiency_model.calculate(energies), dtype=float)
    invalid = int(np.count_nonzero(~np.isfinite(efficiencies) | (efficiencies <= 0.0)))
    if invalid > 0:
        warnings.warn(
            (
                f"Final corrected export for {spectrum.spectrum_id} has {invalid} channels with "
                "invalid efficiency values; writing NaN for those rows."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    uncertainties = np.asarray(spectrum.counts_uncertainty, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected_counts = spectrum.counts / efficiencies
        corrected_uncertainty = uncertainties / efficiencies
    corrected_counts = np.where(np.isfinite(corrected_counts), corrected_counts, np.nan)
    corrected_uncertainty = np.where(
        np.isfinite(corrected_uncertainty), corrected_uncertainty, np.nan
    )

    rows = [
        [
            int(channel),
            float(energy),
            float(count),
            float(uncertainty),
            float(efficiency) if np.isfinite(efficiency) else np.nan,
            float(corrected),
            float(corrected_unc),
        ]
        for channel, energy, count, uncertainty, efficiency, corrected, corrected_unc in zip(
            spectrum.channels,
            energies,
            spectrum.counts,
            uncertainties,
            efficiencies,
            corrected_counts,
            corrected_uncertainty,
        )
    ]
    _write_csv_rows(
        output_path,
        _export_metadata_lines(
            spectrum,
            background_file=background_file,
            efficiency_model=efficiency_model,
        ),
        [
            "channel",
            "energy_keV",
            "background_adjusted_counts",
            "background_adjusted_uncertainty",
            "efficiency",
            "efficiency_corrected_counts",
            "efficiency_corrected_uncertainty",
        ],
        rows,
    )
    return True


def _nearest_channel_for_energy(spectrum: GammaSpectrum, energy_keV: float) -> int:
    energies = _spectrum_energies(spectrum)
    return int(np.argmin(np.abs(energies - float(energy_keV))))


def _load_manual_peak_regions(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("peaks", "regions", "manual_peaks"):
                if key in payload and isinstance(payload[key], list):
                    return [dict(row) for row in payload[key]]
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"Unsupported manual peak JSON structure in {path}")

    raise ValueError(
        f"Unsupported manual peak file format for {path}. Use .csv or .json."
    )


def _manual_peak_bounds(
    spectrum: GammaSpectrum,
    region: Dict[str, Any],
) -> Tuple[int, int]:
    left_channel = region.get("left_channel")
    right_channel = region.get("right_channel")
    if left_channel is not None and right_channel is not None:
        lo = int(round(float(left_channel)))
        hi = int(round(float(right_channel)))
    else:
        left_keV = (
            region.get("left_keV")
            or region.get("start_keV")
            or region.get("lo_keV")
            or region.get("xmin_keV")
        )
        right_keV = (
            region.get("right_keV")
            or region.get("end_keV")
            or region.get("hi_keV")
            or region.get("xmax_keV")
        )
        if left_keV is None or right_keV is None:
            raise ValueError(
                "Manual peak entries require either left/right channels or left/right energies."
            )
        lo = _nearest_channel_for_energy(spectrum, float(left_keV))
        hi = _nearest_channel_for_energy(spectrum, float(right_keV))

    lo, hi = sorted((lo, hi))
    lo = max(0, lo)
    hi = min(len(spectrum.counts) - 1, hi)
    if lo > hi:
        raise ValueError(f"Invalid manual peak bounds: {region}")
    return lo, hi


def _manual_peak_rows(
    raw_spectrum: GammaSpectrum,
    analysis_spectrum: GammaSpectrum,
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_counts = np.asarray(raw_spectrum.counts, dtype=float)
    analysis_counts = np.asarray(analysis_spectrum.counts, dtype=float)
    raw_energies = _spectrum_energies(raw_spectrum)
    analysis_energies = _spectrum_energies(analysis_spectrum)

    rows: List[Dict[str, Any]] = []
    for idx, region in enumerate(regions):
        lo, hi = _manual_peak_bounds(raw_spectrum, region)
        peak_slice = slice(lo, hi + 1)
        local_raw = raw_counts[peak_slice]
        local_analysis = analysis_counts[peak_slice]
        peak_offset = int(np.argmax(local_raw)) if local_raw.size else 0
        peak_channel = lo + peak_offset
        gross_counts = float(np.sum(local_raw))
        integrated_counts = float(np.sum(local_analysis))
        gross_unc = float(np.sqrt(max(gross_counts, 0.0)))
        net_unc = float(
            np.sqrt(
                max(
                    float(
                        np.sum(
                            np.asarray(
                                analysis_spectrum.counts_uncertainty[peak_slice],
                                dtype=float,
                            )
                            ** 2
                        )
                    ),
                    0.0,
                )
            )
        )
        label = (
            region.get("label")
            or region.get("name")
            or region.get("isotope")
            or f"manual_peak_{idx + 1}"
        )
        rows.append(
            {
                "channel": int(peak_channel),
                "energy_keV": float(raw_energies[peak_channel]),
                "amplitude": (
                    float(np.max(local_analysis)) if local_analysis.size else 0.0
                ),
                "raw_counts": float(np.max(local_raw)) if local_raw.size else 0.0,
                "sigma_keV": 0.0,
                "area": integrated_counts,
                "region": f"{lo}:{hi}",
                "is_report": bool(
                    region.get("isotope") or region.get("report_isotope")
                ),
                "report_isotope": str(
                    region.get("isotope") or region.get("report_isotope") or ""
                ),
                "report_file": str(region.get("report_file") or ""),
                "label": str(label),
                "manual": True,
                "left_channel": int(lo),
                "right_channel": int(hi),
                "left_energy_keV": float(raw_energies[lo]),
                "right_energy_keV": float(raw_energies[hi]),
                "gross_counts": gross_counts,
                "gross_counts_unc": gross_unc,
                "net_counts": integrated_counts,
                "net_counts_unc": net_unc,
                "background_counts": float(gross_counts - integrated_counts),
                "background_subtracted": bool(
                    analysis_spectrum.metadata.get("background_subtraction")
                ),
                "analysis_peak_energy_keV": float(analysis_energies[peak_channel]),
            }
        )
    return rows


def _resolve_spectral_feature_photopeak(
    args: argparse.Namespace,
    spectrum: GammaSpectrum,
) -> tuple[float, str, ContinuumDriverEstimate | None] | None:
    manual_photopeak = getattr(args, "feature_photopeak_keV", None)
    if manual_photopeak is not None:
        return float(manual_photopeak), "manual photopeak", None

    isotope = str(getattr(args, "feature_isotope", "") or "").strip()
    if isotope:
        target_line = getattr(args, "feature_line_keV", None)
        min_intensity = max(float(getattr(args, "feature_min_intensity", 0.02)), 0.0)
        lines = list_nuclide_lines(isotope, line_type="gamma", min_intensity=min_intensity)
        if not lines:
            raise ValueError(
                f"No gamma lines found for isotope '{isotope}' at min intensity {min_intensity:.3f}."
            )
        if target_line is None:
            selected = lines[0]
        else:
            selected = min(lines, key=lambda line: abs(line.energy_keV - float(target_line)))
        return float(selected.energy_keV), f"{selected.nuclide} library line", None

    manual_line = getattr(args, "feature_line_keV", None)
    if manual_line is not None:
        return float(manual_line), "manual line", None

    if bool(getattr(args, "auto_feature_driver", False)):
        estimate = estimate_dominant_continuum_driver(
            spectrum,
            tolerance_keV=max(float(getattr(args, "feature_tolerance_keV", 2.0)), 0.1),
            min_intensity=max(float(getattr(args, "feature_min_intensity", 0.02)), 0.0),
        )
        if estimate is None:
            raise ValueError(
                "Auto continuum-driver detection did not find a confident line match."
            )
        return float(estimate.line_energy_keV), f"auto {estimate.nuclide}", estimate

    return None


def _build_spectral_feature_payload(
    spectrum: GammaSpectrum,
    *,
    photopeak_energy_keV: float,
    source_label: str,
    detector_material: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_items = estimate_spectral_phenomena(
        photopeak_energy_keV,
        detector_material=detector_material,
        spectrum=spectrum,
    )

    annotations: list[dict[str, Any]] = [
        {
            "kind": "photopeak",
            "label": f"{source_label} photopeak",
            "energy_keV": float(photopeak_energy_keV),
            "color": "#72d6ff",
            "line_style": "-",
            "estimated_height_counts": None,
        }
    ]
    report_features: list[dict[str, Any]] = []
    for item in feature_items:
        annotations.append(
            {
                "kind": item.kind,
                "label": (
                    f"{item.label} ({item.relative_height * 100.0:.0f}%)"
                    if item.relative_height is not None
                    else item.label
                ),
                "energy_keV": float(item.energy_keV),
                "color": item.color,
                "line_style": "--",
                "estimated_height_counts": item.estimated_height_counts,
            }
        )
        report_features.append(
            {
                "kind": item.kind,
                "label": item.label,
                "energy_keV": float(item.energy_keV),
                "relative_height": item.relative_height,
                "estimated_height_counts": item.estimated_height_counts,
                "summary": item.summary,
                "color": item.color,
            }
        )

    report = {
        "photopeak_energy_keV": float(photopeak_energy_keV),
        "source_label": source_label,
        "detector_material": detector_material,
        "features": report_features,
    }
    return annotations, report


def _load_plot_and_analysis_spectra(
    input_path: Path,
    *,
    validate: bool,
    energy_override: Optional[List[float]],
    efficiency_override: Optional[Dict[str, float]],
    background: Optional[GammaSpectrum],
    background_scale_mode: str,
    background_scale_factor: Optional[float],
    use_background_subtracted: bool,
) -> Tuple[GammaSpectrum, GammaSpectrum]:
    raw_spectrum = _load_spectrum_from_path(
        input_path,
        validate=validate,
        energy_override=energy_override,
        efficiency_override=efficiency_override,
    )
    if not use_background_subtracted:
        return raw_spectrum, raw_spectrum
    analysis_spectrum = subtract_measured_background(
        raw_spectrum,
        background,
        mode=background_scale_mode,
        manual_scale=background_scale_factor,
        negative_policy="hybrid",
        warn_missing=True,
    )
    return raw_spectrum, analysis_spectrum


def _ingest_spectrum(
    input_path: Path,
    *,
    validate: bool,
    energy_override: Optional[List[float]],
    efficiency_override: Optional[Dict[str, float]],
    background: Optional[GammaSpectrum],
    background_scale_mode: str,
    background_scale_factor: Optional[float],
) -> GammaSpectrum:
    spectrum = _load_spectrum_from_path(
        input_path,
        validate=validate,
        energy_override=energy_override,
        efficiency_override=efficiency_override,
    )
    return subtract_measured_background(
        spectrum,
        background,
        mode=background_scale_mode,
        manual_scale=background_scale_factor,
        negative_policy="hybrid",
        warn_missing=True,
    )


def _resolve_cli_roi_bounds(
    args: argparse.Namespace,
    spectrum: GammaSpectrum,
) -> tuple[float, float]:
    left_keV = getattr(args, "left_keV", None)
    right_keV = getattr(args, "right_keV", None)
    if left_keV is not None and right_keV is not None:
        return tuple(sorted((float(left_keV), float(right_keV))))

    left_channel = getattr(args, "left_channel", None)
    right_channel = getattr(args, "right_channel", None)
    if left_channel is not None and right_channel is not None:
        return tuple(
            sorted(
                (
                    float(spectrum.channel_to_energy(int(left_channel))),
                    float(spectrum.channel_to_energy(int(right_channel))),
                )
            )
        )
    raise ValueError(
        "ROI bounds require either --left-keV/--right-keV or --left-channel/--right-channel."
    )


def _roi_analysis_payload(result: Any) -> Dict[str, Any]:
    return {
        "label": result.label,
        "roi_bounds_keV": list(result.roi_bounds_keV),
        "gross_counts": result.gross_counts,
        "gross_counts_uncertainty": result.gross_counts_uncertainty,
        "background_counts": result.background_counts,
        "background_counts_uncertainty": result.background_counts_uncertainty,
        "net_counts": result.net_counts,
        "net_counts_uncertainty": result.net_counts_uncertainty,
        "centroid_keV": result.centroid_keV,
        "centroid_uncertainty_keV": result.centroid_uncertainty_keV,
        "significance": result.significance,
        "background_method": result.background_method,
        "peak_search_method": result.peak_search_method,
        "sideband_bounds_keV": [list(bounds) for bounds in result.sideband_bounds_keV],
        "overlap_components": [
            {
                "centroid_channel": component.centroid_channel,
                "centroid_keV": component.centroid_keV,
                "net_counts": component.net_counts,
                "net_counts_uncertainty": component.net_counts_uncertainty,
                "fwhm_channels": component.fwhm_channels,
                "reduced_chi_squared": component.reduced_chi_squared,
            }
            for component in result.overlap_components
        ],
        "notes": list(result.notes),
    }


def _roi_statistics_payload(result: Any) -> Dict[str, Any]:
    return {
        "label": result.label,
        "roi_bounds_keV": list(result.roi_bounds_keV),
        "sample_count": result.sample_count,
        "mean_net_counts": result.mean_net_counts,
        "stdev_net_counts": result.stdev_net_counts,
        "relative_std": result.relative_std,
        "mean_centroid_keV": result.mean_centroid_keV,
        "stdev_centroid_keV": result.stdev_centroid_keV,
        "min_net_counts": result.min_net_counts,
        "max_net_counts": result.max_net_counts,
        "samples": [
            {
                "label": sample.label,
                "net_counts": sample.net_counts,
                "net_counts_uncertainty": sample.net_counts_uncertainty,
                "centroid_keV": sample.centroid_keV,
                "significance": sample.significance,
            }
            for sample in result.samples
        ],
    }


def _print_warning_messages(messages: List[str]) -> None:
    if not messages:
        print("Warnings: none")
        return
    print("Warnings:")
    for message in messages:
        print(f"  - {message}")


def _relative_output_path(
    root_dir: Path, input_dir: Path, input_path: Path, suffix: str
) -> Path:
    relative = input_path.relative_to(input_dir)
    return root_dir / relative.parent / f"{relative.stem}{suffix}"


def _add_validate_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip artifact schema validation.",
    )
    parser.set_defaults(validate=True)


def cmd_ingest(args: argparse.Namespace) -> None:
    input_path = args.input
    energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
    profile_name = getattr(args, "profile", None)
    efficiency_override = _parse_efficiency_override(
        getattr(args, "efficiency_coefficients", None)
    )
    if efficiency_override is None:
        efficiency_override = _profile_efficiency_override(profile_name)
    background_file: Optional[Path] = getattr(args, "background_file", None)
    if background_file is None:
        background_file = _profile_background_file(profile_name)
    background = None
    if background_file is not None:
        background = _load_spectrum_from_path(background_file, validate=args.validate)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectrum = _ingest_spectrum(
            input_path,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
            background=background,
            background_scale_mode=getattr(args, "background_scale_mode", "live"),
            background_scale_factor=getattr(args, "background_scale_factor", None),
        )

        _ensure_parent_dir(args.output)
        write_spectrum_file(args.output, spectrum, source_path=input_path)
        print(f"Wrote spectrum file to {args.output}")

        background_adjusted_output: Optional[Path] = getattr(
            args, "save_background_adjusted", None
        )
        if background_adjusted_output is not None:
            _write_background_adjusted_csv(
                background_adjusted_output,
                spectrum,
                background_file=background_file,
            )
            print(f"Wrote background-adjusted counts to {background_adjusted_output}")

        final_corrected_output: Optional[Path] = getattr(
            args, "save_final_corrected", None
        )
        if final_corrected_output is not None:
            if _write_final_corrected_csv(
                final_corrected_output,
                spectrum,
                background_file=background_file,
            ):
                print(f"Wrote final corrected counts to {final_corrected_output}")

    _print_warning_messages([str(record.message) for record in caught])


def cmd_ingest_batch(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
    profile_name = getattr(args, "profile", None)
    efficiency_override = _parse_efficiency_override(
        getattr(args, "efficiency_coefficients", None)
    )
    if efficiency_override is None:
        efficiency_override = _profile_efficiency_override(profile_name)
    background_file: Optional[Path] = getattr(args, "background_file", None)
    if background_file is None:
        background_file = _profile_background_file(profile_name)
    background = None
    if background_file is not None:
        background = _load_spectrum_from_path(background_file, validate=args.validate)

    input_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".asc", ".txt", ".spe"}
    )
    if not input_files:
        raise ValueError(f"No supported spectrum files found under {input_dir}")

    all_warnings: List[str] = []
    artifact_count = 0
    background_adjusted_count = 0
    final_corrected_count = 0

    for input_path in input_files:
        artifact_path = _relative_output_path(
            args.output_dir, input_dir, input_path, ".json"
        )
        background_adjusted_output = None
        if getattr(args, "background_adjusted_dir", None) is not None:
            background_adjusted_output = _relative_output_path(
                args.background_adjusted_dir,
                input_dir,
                input_path,
                "_background_adjusted.csv",
            )
        final_corrected_output = None
        if getattr(args, "final_corrected_dir", None) is not None:
            final_corrected_output = _relative_output_path(
                args.final_corrected_dir,
                input_dir,
                input_path,
                "_final_corrected.csv",
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spectrum = _ingest_spectrum(
                input_path,
                validate=args.validate,
                energy_override=energy_override,
                efficiency_override=efficiency_override,
                background=background,
                background_scale_mode=getattr(args, "background_scale_mode", "live"),
                background_scale_factor=getattr(args, "background_scale_factor", None),
            )
            _ensure_parent_dir(artifact_path)
            write_spectrum_file(artifact_path, spectrum, source_path=input_path)
            artifact_count += 1
            print(f"Wrote spectrum file to {artifact_path}")

            if background_adjusted_output is not None:
                _write_background_adjusted_csv(
                    background_adjusted_output,
                    spectrum,
                    background_file=background_file,
                )
                background_adjusted_count += 1
                print(
                    f"Wrote background-adjusted counts to {background_adjusted_output}"
                )

            if final_corrected_output is not None:
                if _write_final_corrected_csv(
                    final_corrected_output,
                    spectrum,
                    background_file=background_file,
                ):
                    final_corrected_count += 1
                    print(f"Wrote final corrected counts to {final_corrected_output}")

        for record in caught:
            all_warnings.append(f"{input_path}: {record.message}")

    print(f"Processed {artifact_count} spectra from {input_dir}")
    print(f"Spectrum artifacts directory: {args.output_dir}")
    if getattr(args, "background_adjusted_dir", None) is not None:
        print(
            "Background-adjusted counts directory: "
            f"{args.background_adjusted_dir} ({background_adjusted_count} files)"
        )
    if getattr(args, "final_corrected_dir", None) is not None:
        print(
            "Final corrected counts directory: "
            f"{args.final_corrected_dir} ({final_corrected_count} files)"
        )
    _print_warning_messages(all_warnings)


def cmd_spectrum_plot(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "profile", None)
    energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
    efficiency_override = _parse_efficiency_override(
        getattr(args, "efficiency_coefficients", None)
    )
    if efficiency_override is None:
        efficiency_override = _profile_efficiency_override(profile_name)

    background_file: Optional[Path] = getattr(args, "background_file", None)
    if background_file is None:
        background_file = _profile_background_file(profile_name)

    background = None
    if background_file is not None:
        background = _load_spectrum_from_path(
            background_file,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
        )

    manual_regions: List[Dict[str, Any]] = []
    if getattr(args, "manual_peaks_file", None) is not None:
        manual_regions = _load_manual_peak_regions(args.manual_peaks_file)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_spectrum, analysis_spectrum = _load_plot_and_analysis_spectra(
            args.input,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
            background=background,
            background_scale_mode=getattr(args, "background_scale_mode", "live"),
            background_scale_factor=getattr(args, "background_scale_factor", None),
            use_background_subtracted=bool(
                getattr(args, "background_subtracted", False)
            ),
        )
        spectrum_for_plot = (
            analysis_spectrum if args.background_subtracted else raw_spectrum
        )

        feature_annotations: list[dict[str, Any]] = []
        feature_report_payload: dict[str, Any] | None = None
        feature_line = _resolve_spectral_feature_photopeak(args, spectrum_for_plot)
        if feature_line is not None:
            line_energy_keV, source_label, auto_estimate = feature_line
            detector_material = str(
                getattr(args, "feature_detector_material", "hpge") or "hpge"
            )
            feature_annotations, feature_report_payload = _build_spectral_feature_payload(
                spectrum_for_plot,
                photopeak_energy_keV=line_energy_keV,
                source_label=source_label,
                detector_material=detector_material,
            )
            if auto_estimate is not None:
                feature_report_payload["auto_driver"] = {
                    "nuclide": auto_estimate.nuclide,
                    "line_energy_keV": float(auto_estimate.line_energy_keV),
                    "score": float(auto_estimate.score),
                    "summary": auto_estimate.summary,
                }
                print(
                    "Auto continuum driver: "
                    f"{auto_estimate.nuclide} at {auto_estimate.line_energy_keV:.3f} keV "
                    f"(score {auto_estimate.score:.4g})"
                )
            print(
                "Spectral feature overlays enabled for "
                f"{source_label} at {line_energy_keV:.3f} keV"
            )

        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from fluxforge.plots.spectrum_inspection import plot_gamma_spectrum

        fig, ax = plot_gamma_spectrum(
            spectrum_for_plot,
            title=args.title or spectrum_for_plot.spectrum_id,
            manual_regions=manual_regions,
            x_min_keV=getattr(args, "x_min_keV", None),
            x_max_keV=getattr(args, "x_max_keV", None),
            y_log=bool(getattr(args, "y_log", False)),
            subtitle=(
                "background-subtracted" if args.background_subtracted else "raw counts"
            ),
        )
        if feature_annotations:
            ymax = ax.get_ylim()[1]
            label_y = ymax * 0.96 if np.isfinite(ymax) and ymax > 0 else 1.0
            for item in feature_annotations:
                energy = float(item.get("energy_keV", 0.0) or 0.0)
                if energy <= 0.0:
                    continue
                ax.axvline(
                    energy,
                    color=str(item.get("color") or "#72d6ff"),
                    linewidth=1.0,
                    linestyle=str(item.get("line_style") or "--"),
                    alpha=0.9,
                )
                ax.text(
                    energy,
                    label_y,
                    str(item.get("label") or "feature"),
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=8,
                    color=str(item.get("color") or "#72d6ff"),
                )
        _ensure_parent_dir(args.output)
        fig.savefig(args.output, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Wrote spectrum plot to {args.output}")

        feature_report_output: Optional[Path] = getattr(args, "save_feature_report", None)
        if feature_report_output is not None:
            if feature_report_payload is None:
                raise ValueError(
                    "--save-feature-report requires a selected feature line "
                    "(--feature-photopeak-keV, --feature-line-keV, --feature-isotope, or --auto-feature-driver)."
                )
            _ensure_parent_dir(feature_report_output)
            feature_report_output.write_text(
                json.dumps(feature_report_payload, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote spectral feature report to {feature_report_output}")

        peak_report_output: Optional[Path] = getattr(args, "save_peak_report", None)
        if peak_report_output is not None:
            if not manual_regions:
                raise ValueError("--save-peak-report requires --manual-peaks-file.")
            peaks = _manual_peak_rows(
                raw_spectrum,
                analysis_spectrum if args.background_subtracted else raw_spectrum,
                manual_regions,
            )
            write_peak_report(
                peak_report_output,
                spectrum_id=raw_spectrum.spectrum_id,
                live_time_s=raw_spectrum.live_time,
                peaks=peaks,
                source_path=args.input,
            )
            print(f"Wrote manual peak report to {peak_report_output}")

    _print_warning_messages([str(record.message) for record in caught])


def cmd_peaks(args: argparse.Namespace) -> None:
    if getattr(args, "manual_peaks_file", None) is not None:
        profile_name = getattr(args, "profile", None)
        energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
        efficiency_override = _parse_efficiency_override(
            getattr(args, "efficiency_coefficients", None)
        )
        if efficiency_override is None:
            efficiency_override = _profile_efficiency_override(profile_name)

        background_file: Optional[Path] = getattr(args, "background_file", None)
        if background_file is None:
            background_file = _profile_background_file(profile_name)

        background = None
        if background_file is not None:
            background = _load_spectrum_from_path(
                background_file,
                validate=args.validate,
                energy_override=energy_override,
                efficiency_override=efficiency_override,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw_spectrum, analysis_spectrum = _load_plot_and_analysis_spectra(
                args.spectrum_file,
                validate=args.validate,
                energy_override=energy_override,
                efficiency_override=efficiency_override,
                background=background,
                background_scale_mode=getattr(args, "background_scale_mode", "live"),
                background_scale_factor=getattr(args, "background_scale_factor", None),
                use_background_subtracted=bool(
                    getattr(args, "background_subtracted", False)
                ),
            )
            manual_regions = _load_manual_peak_regions(args.manual_peaks_file)
            peaks = _manual_peak_rows(
                raw_spectrum,
                analysis_spectrum if args.background_subtracted else raw_spectrum,
                manual_regions,
            )
            write_peak_report(
                args.output,
                spectrum_id=raw_spectrum.spectrum_id,
                live_time_s=raw_spectrum.live_time,
                peaks=peaks,
                source_path=args.spectrum_file,
            )
            print(f"Wrote peak report to {args.output}")
        _print_warning_messages([str(record.message) for record in caught])
        return

    spectrum_payload = read_spectrum_file(args.spectrum_file)
    if args.validate:
        validate_or_raise(spectrum_payload)
    spectrum = GammaSpectrum.from_dict(spectrum_payload["spectrum"])
    energies = (
        spectrum.energies
        if spectrum.energies is not None
        else spectrum.channels.astype(float)
    )
    method = str(getattr(args, "method", "segmented"))
    if method != "segmented":
        candidates = detect_peak_candidates(
            spectrum,
            method=method,
            max_peaks=int(getattr(args, "max_peaks", 12)),
        )
        peak_payload = [
            {
                "channel": int(round(peak.channel)),
                "energy_keV": peak.energy_keV,
                "amplitude": peak.net_counts,
                "raw_counts": peak.net_counts,
                "sigma_keV": 0.0,
                "area": peak.net_counts,
                "region": f"{peak.roi_bounds_keV[0]:.3f}:{peak.roi_bounds_keV[1]:.3f}",
                "is_report": bool(peak.nuclide),
                "report_isotope": peak.nuclide or "",
                "report_file": "",
                "label": peak.peak_id,
                "manual": False,
                "left_energy_keV": peak.roi_bounds_keV[0],
                "right_energy_keV": peak.roi_bounds_keV[1],
                "net_counts": peak.net_counts,
                "net_counts_unc": np.sqrt(max(peak.net_counts, 0.0)),
                "significance": peak.significance,
                "fit_quality": peak.fit_quality,
                "peak_search_method": method,
            }
            for peak in candidates
        ]
        write_peak_report(
            args.output,
            spectrum_id=spectrum.spectrum_id,
            live_time_s=spectrum.live_time,
            peaks=peak_payload,
            source_path=args.spectrum_file,
        )
        print(f"Wrote peak report to {args.output}")
        return

    if args.sensitivity == "sensitive":
        config = SegmentedDetectionConfig.sensitive()
    elif args.sensitivity == "conservative":
        config = SegmentedDetectionConfig.conservative()
    else:
        config = SegmentedDetectionConfig()
    config.fit_window = int(args.fit_window)

    peaks = detect_peaks_segmented(
        spectrum.channels,
        energies,
        spectrum.counts,
        config=config,
    )
    peak_payload = [
        {
            "channel": peak.channel,
            "energy_keV": peak.energy_keV,
            "amplitude": peak.amplitude,
            "raw_counts": peak.raw_counts,
            "sigma_keV": peak.sigma_keV,
            "area": peak.area,
            "region": peak.region,
            "is_report": peak.is_report,
            "report_isotope": peak.report_isotope,
            "report_file": peak.report_file,
            "peak_search_method": method,
        }
        for peak in peaks
    ]

    write_peak_report(
        args.output,
        spectrum_id=spectrum.spectrum_id,
        live_time_s=spectrum.live_time,
        peaks=peak_payload,
        source_path=args.spectrum_file,
    )
    print(f"Wrote peak report to {args.output}")


def cmd_roi_analyze(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "profile", None)
    energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
    efficiency_override = _parse_efficiency_override(
        getattr(args, "efficiency_coefficients", None)
    )
    if efficiency_override is None:
        efficiency_override = _profile_efficiency_override(profile_name)

    background_file: Optional[Path] = getattr(args, "background_file", None)
    if background_file is None:
        background_file = _profile_background_file(profile_name)

    background = None
    if background_file is not None:
        background = _load_spectrum_from_path(
            background_file,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_spectrum, _analysis_spectrum = _load_plot_and_analysis_spectra(
            args.input,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
            background=background,
            background_scale_mode=getattr(args, "background_scale_mode", "live"),
            background_scale_factor=getattr(args, "background_scale_factor", None),
            use_background_subtracted=False,
        )
        roi_bounds = _resolve_cli_roi_bounds(args, raw_spectrum)
        result = analyze_roi_region(
            raw_spectrum,
            roi_bounds_keV=roi_bounds,
            label=str(getattr(args, "label", None) or raw_spectrum.spectrum_id or "ROI"),
            background_method=str(getattr(args, "background_method", "roi_sideband")),
            peak_search_method=str(getattr(args, "peak_search_method", "mariscotti")),
            sideband_width_keV=getattr(args, "sideband_width_keV", None),
            background_spectrum=background,
            background_mode={
                "manual": "scaled",
                "live": "statistical",
                "real": "statistical",
            }.get(str(getattr(args, "background_scale_mode", "live")), "statistical"),
            background_scale=float(getattr(args, "background_scale_factor", 1.0) or 1.0),
            decompose_overlaps=bool(getattr(args, "decompose_overlaps", False)),
            max_components=int(getattr(args, "max_components", 3)),
        )

    payload = {
        "schema": "fluxforge.roi_analysis.v1",
        "spectrum_id": raw_spectrum.spectrum_id,
        "analysis": _roi_analysis_payload(result),
    }
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ROI analysis to {args.output}")
    _print_warning_messages([str(record.message) for record in caught])


def cmd_roi_statistics(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "profile", None)
    energy_override = _parse_csv_floats(getattr(args, "energy_calibration", None))
    efficiency_override = _parse_efficiency_override(
        getattr(args, "efficiency_coefficients", None)
    )
    if efficiency_override is None:
        efficiency_override = _profile_efficiency_override(profile_name)

    loaded: list[tuple[str, GammaSpectrum]] = []
    first_spectrum: GammaSpectrum | None = None
    for path in getattr(args, "inputs", []):
        spectrum = _load_spectrum_from_path(
            path,
            validate=args.validate,
            energy_override=energy_override,
            efficiency_override=efficiency_override,
        )
        loaded.append((Path(path).name, spectrum))
        if first_spectrum is None:
            first_spectrum = spectrum
    if first_spectrum is None:
        raise ValueError("At least one ROI statistics input spectrum is required.")

    roi_bounds = _resolve_cli_roi_bounds(args, first_spectrum)
    result = compute_roi_statistics(
        loaded,
        roi_bounds_keV=roi_bounds,
        label=str(getattr(args, "label", None) or "ROI Statistics"),
        background_method=str(getattr(args, "background_method", "roi_sideband")),
        peak_search_method=str(getattr(args, "peak_search_method", "mariscotti")),
        sideband_width_keV=getattr(args, "sideband_width_keV", None),
    )

    payload = {
        "schema": "fluxforge.roi_statistics.v1",
        "statistics": _roi_statistics_payload(result),
    }
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ROI statistics to {args.output}")


def cmd_file_query(args: argparse.Namespace) -> None:
    root = Path(getattr(args, "root", Path("."))).resolve()
    if not root.exists():
        raise ValueError(f"Query root does not exist: {root}")

    patterns = _parse_csv_strings(getattr(args, "patterns", None))
    if not patterns:
        patterns = (
            "**/*.json",
            "**/*.csv",
            "**/*.ASC",
            "**/*.spe",
            "**/*.txt",
        )
    contains = str(getattr(args, "contains", "") or "").strip().lower()
    suffixes = {value.lower() for value in _parse_csv_strings(getattr(args, "suffixes", None))}
    min_size = int(getattr(args, "min_size_bytes", 0) or 0)
    max_size = getattr(args, "max_size_bytes", None)
    limit = max(int(getattr(args, "limit", 2000) or 0), 0)

    modified_after_raw = getattr(args, "modified_after", None)
    modified_after = None
    if modified_after_raw:
        modified_after = datetime.fromisoformat(str(modified_after_raw))

    candidate_paths: Dict[Path, None] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                candidate_paths[path] = None

    rows: List[Dict[str, Any]] = []
    for path in sorted(candidate_paths.keys()):
        rel_path = path.relative_to(root).as_posix()
        if contains and contains not in rel_path.lower():
            continue

        stat = path.stat()
        size_bytes = int(stat.st_size)
        if size_bytes < min_size:
            continue
        if max_size is not None and size_bytes > int(max_size):
            continue

        suffix = path.suffix.lower()
        if suffixes and suffix not in suffixes:
            continue

        modified_time = datetime.fromtimestamp(stat.st_mtime)
        if modified_after is not None and modified_time < modified_after:
            continue

        rows.append(
            {
                "path": rel_path,
                "suffix": suffix,
                "size_bytes": size_bytes,
                "modified_time": modified_time.isoformat(timespec="seconds"),
            }
        )

    if limit > 0:
        rows = rows[:limit]

    payload = {
        "schema": "fluxforge.file_query.v1",
        "root": str(root),
        "patterns": list(patterns),
        "contains": contains or None,
        "suffixes": sorted(suffixes),
        "result_count": len(rows),
        "rows": rows,
    }

    output = Path(args.output)
    _ensure_parent_dir(output)
    output_format = str(getattr(args, "format", "json") or "json").strip().lower()
    if output_format == "csv":
        _write_dict_rows(output, rows)
    else:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote file query results to {output}")


def cmd_batch_compare(args: argparse.Namespace) -> None:
    baseline_payload = _load_structured_rows(Path(args.baseline))
    candidate_payload = _load_structured_rows(Path(args.candidate))
    baseline_rows = _extract_structured_rows(baseline_payload)
    candidate_rows = _extract_structured_rows(candidate_payload)
    if not baseline_rows:
        raise ValueError("Baseline payload did not contain any tabular rows.")
    if not candidate_rows:
        raise ValueError("Candidate payload did not contain any tabular rows.")

    key_fields = _parse_csv_strings(getattr(args, "keys", None))
    if not key_fields:
        raise ValueError("At least one key field is required for batch-compare.")

    def _row_key(row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(str(row.get(field, "")).strip() for field in key_fields)

    baseline_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    candidate_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    baseline_duplicates = 0
    candidate_duplicates = 0
    for row in baseline_rows:
        key = _row_key(row)
        if key in baseline_by_key:
            baseline_duplicates += 1
            continue
        baseline_by_key[key] = row
    for row in candidate_rows:
        key = _row_key(row)
        if key in candidate_by_key:
            candidate_duplicates += 1
            continue
        candidate_by_key[key] = row

    matched_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    baseline_only = sorted(set(baseline_by_key) - set(candidate_by_key))
    candidate_only = sorted(set(candidate_by_key) - set(baseline_by_key))

    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    numeric_fields = _parse_csv_strings(getattr(args, "numeric_fields", None))
    if not numeric_fields:
        shared_fields = set(baseline_rows[0].keys()) & set(candidate_rows[0].keys())
        inferred: List[str] = []
        for field in sorted(shared_fields):
            if field in key_fields:
                continue
            has_numeric_baseline = any(
                _to_float(row.get(field)) is not None for row in baseline_rows
            )
            has_numeric_candidate = any(
                _to_float(row.get(field)) is not None for row in candidate_rows
            )
            if has_numeric_baseline and has_numeric_candidate:
                inferred.append(field)
        numeric_fields = tuple(inferred)

    comparisons: List[Dict[str, Any]] = []
    field_abs_deltas: Dict[str, List[float]] = {field: [] for field in numeric_fields}
    field_abs_rel_deltas: Dict[str, List[float]] = {field: [] for field in numeric_fields}

    for key in matched_keys:
        base_row = baseline_by_key[key]
        cand_row = candidate_by_key[key]
        record: Dict[str, Any] = {field: key[idx] for idx, field in enumerate(key_fields)}
        for field in numeric_fields:
            base_value = _to_float(base_row.get(field))
            cand_value = _to_float(cand_row.get(field))
            record[f"baseline_{field}"] = base_value
            record[f"candidate_{field}"] = cand_value
            if base_value is None or cand_value is None:
                record[f"delta_{field}"] = None
                record[f"relative_delta_{field}"] = None
                continue

            delta = float(cand_value - base_value)
            rel_delta = float(delta / max(abs(base_value), 1.0e-12))
            record[f"delta_{field}"] = delta
            record[f"relative_delta_{field}"] = rel_delta
            field_abs_deltas[field].append(abs(delta))
            field_abs_rel_deltas[field].append(abs(rel_delta))
        comparisons.append(record)

    max_rows = max(int(getattr(args, "max_rows", 2000) or 0), 0)
    if max_rows > 0:
        comparisons = comparisons[:max_rows]

    field_stats: Dict[str, Dict[str, float]] = {}
    for field in numeric_fields:
        abs_deltas = field_abs_deltas.get(field, [])
        abs_rel = field_abs_rel_deltas.get(field, [])
        field_stats[field] = {
            "mean_abs_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            "max_abs_delta": float(np.max(abs_deltas)) if abs_deltas else 0.0,
            "mean_abs_relative_delta": float(np.mean(abs_rel)) if abs_rel else 0.0,
            "max_abs_relative_delta": float(np.max(abs_rel)) if abs_rel else 0.0,
        }

    payload = {
        "schema": "fluxforge.batch_compare.v1",
        "baseline": str(Path(args.baseline)),
        "candidate": str(Path(args.candidate)),
        "key_fields": list(key_fields),
        "numeric_fields": list(numeric_fields),
        "summary": {
            "baseline_rows": len(baseline_rows),
            "candidate_rows": len(candidate_rows),
            "matched_rows": len(matched_keys),
            "baseline_only_rows": len(baseline_only),
            "candidate_only_rows": len(candidate_only),
            "baseline_duplicate_keys": baseline_duplicates,
            "candidate_duplicate_keys": candidate_duplicates,
            "field_stats": field_stats,
        },
        "baseline_only_keys": [
            {field: key[idx] for idx, field in enumerate(key_fields)}
            for key in baseline_only
        ],
        "candidate_only_keys": [
            {field: key[idx] for idx, field in enumerate(key_fields)}
            for key in candidate_only
        ],
        "comparisons": comparisons,
    }

    output = Path(args.output)
    _ensure_parent_dir(output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote batch comparison to {output}")


def cmd_parity_check(args: argparse.Namespace) -> None:
    reference_root = Path(getattr(args, "reference_root", Path("tests/spectra/reference_parity")))
    activation_root_arg = getattr(args, "activation_root", None)
    activation_root = Path(activation_root_arg) if activation_root_arg else None
    output = Path(getattr(args, "output", Path("parity_check.json")))

    payload = run_reference_parity_suite(
        reference_root=reference_root,
        activation_root=activation_root,
        scope=str(getattr(args, "scope", "all") or "all"),
        fixture_id=(str(getattr(args, "fixture_id", "") or "").strip() or None),
        include_activation=bool(getattr(args, "include_activation", True)),
    )

    _ensure_parent_dir(output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload.get("summary") or {}
    print(
        "Parity check complete: "
        f"{summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed "
        f"(total {summary.get('total', 0)})."
    )
    print(f"Wrote parity report to {output}")


def cmd_phase5_crosswalk_report(args: argparse.Namespace) -> None:
    crosswalk_path = Path(
        getattr(
            args,
            "crosswalk",
            Path(".github/project-management/phase5_crosswalk.json"),
        )
    )
    output = Path(
        getattr(args, "output", Path("phase5_crosswalk_report.json"))
    )
    markdown_output = Path(
        getattr(args, "markdown_output", Path("phase5_crosswalk_report.md"))
    )

    payload = load_phase5_crosswalk(crosswalk_path)
    summary = summarize_phase5_crosswalk(payload, workspace_root=Path.cwd())
    report: Dict[str, Any] = {
        "schema": "fluxforge.phase5.crosswalk.report.v1",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "crosswalk_path": str(crosswalk_path),
        "summary": summary,
    }

    if bool(getattr(args, "include_parity_summary", False)):
        reference_root = Path(
            getattr(args, "reference_root", Path("tests/spectra/reference_parity"))
        )
        activation_root = Path(
            getattr(args, "activation_root", Path("tests/activation_inventory/fixtures"))
        )
        parity_payload = run_reference_parity_suite(
            reference_root=reference_root,
            activation_root=activation_root,
            scope="all",
            include_activation=True,
        )
        report["parity_summary"] = parity_payload.get("summary") or {}

    _ensure_parent_dir(output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _ensure_parent_dir(markdown_output)
    markdown_output.write_text(
        render_phase5_crosswalk_markdown(payload, summary),
        encoding="utf-8",
    )

    replay_summary = summary.get("by_replay_state") or {}
    print(
        "Phase 5 crosswalk report complete: "
        f"{summary.get('total_entries', 0)} entries; "
        f"replay-now={replay_summary.get('replay-now', 0)}, "
        f"adapter-required={replay_summary.get('adapter-required', 0)}, "
        f"reference-only={replay_summary.get('reference-only', 0)}"
    )
    print(f"Wrote crosswalk report to {output}")
    print(f"Wrote markdown report to {markdown_output}")


def cmd_gui_acceptance_checklist(args: argparse.Namespace) -> None:
    checklist = Path(
        getattr(
            args,
            "checklist",
            Path("docs/PHASE3_27_RELEASE_CHECKLIST.md"),
        )
    )
    output = Path(getattr(args, "output", Path("gui_acceptance_check.json")))

    artifact_dirs = [
        Path(item)
        for item in (
            getattr(args, "artifact_dir", None)
            or [
                "artifacts/gui_review/phase325_probe",
                "artifacts/gui_review/phase326_probe",
                "artifacts/gui_review/phase327_probe",
            ]
        )
    ]

    checklist_exists = checklist.exists()
    checklist_item_count = 0
    unchecked_items = []
    if checklist_exists:
        lines = checklist.read_text(encoding="utf-8").splitlines()
        for line in lines:
            token = line.strip()
            if token.startswith("- [ ]") or token.startswith("- [x]") or token.startswith("- [X]"):
                checklist_item_count += 1
                if token.startswith("- [ ]"):
                    unchecked_items.append(token[5:].strip())

    artifact_status = []
    for directory in artifact_dirs:
        gallery = directory / "index.html"
        artifact_status.append(
            {
                "path": str(directory),
                "exists": directory.exists(),
                "gallery_exists": gallery.exists(),
            }
        )

    payload = {
        "schema": "fluxforge.gui_acceptance_check.v1",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "checklist": {
            "path": str(checklist),
            "exists": checklist_exists,
            "item_count": checklist_item_count,
            "unchecked_items": unchecked_items,
        },
        "artifacts": artifact_status,
    }
    payload["ready"] = bool(
        checklist_exists
        and checklist_item_count > 0
        and not unchecked_items
        and all(item["gallery_exists"] for item in artifact_status)
    )

    _ensure_parent_dir(output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "GUI acceptance checklist: "
        + ("READY" if payload["ready"] else "NOT READY")
    )
    print(f"Wrote GUI acceptance report to {output}")


def cmd_activity(args: argparse.Namespace) -> None:
    peak_report = read_peak_report(args.peaks_file)
    if args.validate:
        validate_or_raise(peak_report)
    live_time_s = peak_report.get("live_time_s") or args.live_time_s
    if live_time_s is None:
        raise ValueError("Live time is required to compute activities.")

    lines = []
    for idx, peak in enumerate(peak_report["peaks"]):
        net_counts = peak.get("area") or peak.get("raw_counts") or peak.get("amplitude")
        efficiency = args.efficiency
        emission_probability = args.emission_probability
        activity = net_counts / max(
            efficiency * emission_probability * live_time_s, 1e-12
        )
        activity_unc = activity / np.sqrt(max(net_counts, 1e-12))
        isotope = peak.get("report_isotope") or args.isotope or "unknown"
        reaction_id = args.reaction_id or isotope or f"reaction_{idx + 1}"
        line = {
            "energy_keV": peak["energy_keV"],
            "isotope": isotope,
            "reaction_id": reaction_id,
            "net_counts": net_counts,
            "activity_Bq": activity,
            "activity_unc_Bq": activity_unc,
            "efficiency": efficiency,
            "emission_probability": emission_probability,
            "half_life_s": args.half_life_s,
        }
        line.update(
            activation_study_metrics(
                activity_bq=float(activity),
                activity_unc_bq=float(activity_unc),
                half_life_s=float(args.half_life_s),
                isotope=str(isotope) if isotope else None,
                sample_mass_g=getattr(args, "sample_mass_g", None),
            )
        )
        lines.append(line)

    write_line_activities(
        args.output,
        spectrum_id=peak_report.get("spectrum_id", ""),
        lines=lines,
        source_path=args.peaks_file,
    )
    print(f"Wrote line activities to {args.output}")


def cmd_activity_review(args: argparse.Namespace) -> None:
    peak_report = read_peak_report(args.peaks_file)
    if args.validate:
        validate_or_raise(peak_report)

    live_time_s = peak_report.get("live_time_s") or args.live_time_s
    if live_time_s is None:
        raise ValueError("Live time is required to review activities.")

    peaks: list[PeakCandidate] = []
    tolerance_keV = max(float(args.energy_tolerance_keV), 0.1)
    for index, peak in enumerate(peak_report.get("peaks", []) or []):
        if not isinstance(peak, dict):
            continue
        energy_keV = float(peak.get("energy_keV", 0.0) or 0.0)
        net_counts = float(
            peak.get("area") or peak.get("raw_counts") or peak.get("amplitude") or 0.0
        )
        isotope = str(peak.get("report_isotope") or peak.get("isotope") or "").strip() or None
        peaks.append(
            PeakCandidate(
                peak_id=str(peak.get("peak_id") or f"peak-{index + 1}"),
                channel=float(peak.get("channel", index) or index),
                energy_keV=energy_keV,
                significance=float(peak.get("significance", 0.0) or 0.0),
                roi_bounds_keV=(
                    float(peak.get("left_keV", energy_keV - tolerance_keV)),
                    float(peak.get("right_keV", energy_keV + tolerance_keV)),
                ),
                net_counts=net_counts,
                fit_quality=float(
                    peak.get("reduced_chi_squared")
                    or peak.get("fit_quality")
                    or 1.0
                ),
                status="matched" if isotope else "candidate",
                nuclide=isotope,
            )
        )

    review = review_spectrum_activation(
        peaks,
        live_time_s=float(live_time_s),
        efficiency_curve=_build_activity_review_efficiency_curve(args),
        cooling_time_s=float(args.cooling_time_s),
        source_id=str(args.source_id),
        custom_gamma_path=(
            str(args.custom_gamma_path) if getattr(args, "custom_gamma_path", None) else None
        ),
        energy_tolerance_keV=float(args.energy_tolerance_keV),
        dead_time_fraction=float(getattr(args, "dead_time_fraction", 0.0) or 0.0),
        sample_mass_g=getattr(args, "sample_mass_g", None),
    )

    output_path = Path(args.output)
    export_mode = str(getattr(args, "eoi_export", "activities") or "activities").strip().lower()
    if export_mode not in {"activities", "reaction-rates", "both"}:
        raise ValueError(
            "--eoi-export must be one of: activities, reaction-rates, both"
        )

    isotope_csv = Path(
        getattr(args, "isotope_csv_output", None)
        or _activity_review_artifact_path(output_path, "_isotopes.csv")
    )
    line_csv = Path(
        getattr(args, "line_csv_output", None)
        or _activity_review_artifact_path(output_path, "_lines.csv")
    )
    decay_plot = Path(
        getattr(args, "decay_plot", None)
        or _activity_review_artifact_path(output_path, "_decay.png")
    )
    bateman_plot = Path(
        getattr(args, "bateman_plot", None)
        or _activity_review_artifact_path(output_path, "_bateman.png")
    )
    reaction_rate_csv = Path(
        getattr(args, "reaction_rate_csv_output", None)
        or _activity_review_artifact_path(output_path, "_reaction_rates.csv")
    )

    isotope_rows = review.isotope_rows(sample_mass_g=getattr(args, "sample_mass_g", None))
    line_rows = review.line_rows(sample_mass_g=getattr(args, "sample_mass_g", None))

    isotope_artifact = None
    line_artifact = None
    reaction_artifact = None
    reaction_rate_rows: list[dict[str, Any]] = []
    if export_mode in {"activities", "both"}:
        isotope_artifact = _write_csv_table(isotope_csv, isotope_rows)
        line_artifact = _write_csv_table(line_csv, line_rows)

    if export_mode in {"reaction-rates", "both"}:
        raw_segments = None
        segments_file = getattr(args, "irradiation_segments_file", None)
        if segments_file is not None:
            raw_segments = _load_structured_rows(Path(segments_file))
            if isinstance(raw_segments, dict):
                raw_segments = raw_segments.get("segments")
        segments = _coerce_irradiation_segments(
            irradiation_time_s=float(
                getattr(args, "irradiation_time_s", 0.0) or 0.0
            ),
            segments_payload=raw_segments,
        )
        reaction_rate_rows = _build_reaction_rate_rows_from_activity_review(
            isotope_rows=isotope_rows,
            segments=segments,
        )
        reaction_artifact = _write_csv_table(reaction_rate_csv, reaction_rate_rows)

        reaction_rates_output = getattr(args, "reaction_rates_output", None)
        if reaction_rates_output is not None:
            segment_rows = [
                {
                    "duration_s": float(segment.duration_s),
                    "relative_power": float(segment.relative_power),
                }
                for segment in segments
            ]
            write_reaction_rates(
                Path(reaction_rates_output),
                rates=[
                    {
                        "reaction_id": row["reaction_id"],
                        "rate": row["reaction_rate_s"],
                        "uncertainty": row["reaction_rate_unc_s"],
                        "half_life_s": row["half_life_s"],
                    }
                    for row in reaction_rate_rows
                ],
                segments=segment_rows,
                source_path=args.peaks_file,
            )

    decay_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax = plot_decay_curves(
        review.decay_plot_data,
        title="CLI Spectrum Half-Life Decay Review",
        xlabel="Time Since EOI (s)",
        ylabel="Activity (Bq)",
        log_y=True,
        log_x=False,
        half_lives=review.half_lives_s,
        save_path=decay_plot,
    )
    _close_figure(fig)
    bateman_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax = plot_decay_curves(
        review.bateman_plot_data,
        title="CLI Spectrum Bateman Parent/Daughter Review",
        xlabel="Time Since EOI (s)",
        ylabel="EOI-Equivalent Inventory (Bq)",
        log_y=False,
        log_x=False,
        half_lives=review.bateman_half_lives_s,
        save_path=bateman_plot,
    )
    _close_figure(fig)

    payload = review.to_payload(sample_mass_g=getattr(args, "sample_mass_g", None))
    payload["spectrum_id"] = peak_report.get("spectrum_id", "")
    artifacts: dict[str, Any] = {
        "decay_plot": {"path": decay_plot.name, "format": "png"},
        "bateman_plot": {"path": bateman_plot.name, "format": "png"},
    }
    if export_mode in {"activities", "both"}:
        artifacts["isotope_csv"] = isotope_artifact or {
            "path": isotope_csv.name,
            "format": "csv",
        }
        artifacts["line_csv"] = line_artifact or {
            "path": line_csv.name,
            "format": "csv",
        }
    if export_mode in {"reaction-rates", "both"}:
        artifacts["reaction_rates_csv"] = reaction_artifact or {
            "path": reaction_rate_csv.name,
            "format": "csv",
        }
        payload["reaction_rate_rows"] = reaction_rate_rows
    payload["artifacts"] = artifacts
    payload["eoi_export_mode"] = export_mode
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote activity review bundle to {output_path}")


def cmd_inventory_review(args: argparse.Namespace) -> None:
    payload = _load_json(args.activity_review_file)
    inventory_state = build_inventory_state_from_payload(
        payload,
        decay_source_id=str(args.decay_source_id),
    )

    relative_times_s = _parse_relative_time_points(
        getattr(args, "time_points_s", None)
    )
    if not relative_times_s:
        relative_times_s = build_time_grid(
            start_s=float(args.time_start_s),
            stop_s=float(args.time_stop_s),
            count=int(args.time_count),
        )

    result = compute_inventory_time_evolution(
        inventory_state,
        relative_times_s=relative_times_s,
        time_origin=str(args.time_origin),
        decay_source_id=str(args.decay_source_id),
        distance_cm=float(args.distance_cm),
    )

    output_path = Path(args.output)
    time_series_csv = Path(
        getattr(args, "timeseries_csv_output", None)
        or _inventory_review_artifact_path(output_path, "_timeseries.csv")
    )
    eoi_csv = Path(
        getattr(args, "eoi_csv_output", None)
        or _inventory_review_artifact_path(output_path, "_activities_at_irradiation.csv")
    )
    count_start_csv = Path(
        getattr(args, "count_start_csv_output", None)
        or _inventory_review_artifact_path(output_path, "_activities_at_count_start.csv")
    )
    count_end_csv = Path(
        getattr(args, "count_end_csv_output", None)
        or _inventory_review_artifact_path(output_path, "_activities_at_count_end.csv")
    )
    plot_output = Path(
        getattr(args, "plot_output", None)
        or _inventory_review_artifact_path(
            output_path,
            f"_{args.observable}.png",
        )
    )

    time_series_artifact = _write_csv_table(time_series_csv, result.time_series_rows())
    eoi_artifact = _write_csv_table(eoi_csv, list(result.reference_rows("eoi")))
    count_start_artifact = _write_csv_table(
        count_start_csv,
        list(result.reference_rows("count_start")),
    )
    count_end_artifact = _write_csv_table(
        count_end_csv,
        list(result.reference_rows("count_end")),
    )

    plot_output.parent.mkdir(parents=True, exist_ok=True)
    ylabel = {
        "activity": "Activity (Bq)",
        "atoms": "Atoms",
        "mass": "Mass (g)",
        "dose": "Dose Rate (uSv/h)",
    }[str(args.observable)]
    fig, _ax = plot_decay_curves(
        result.plot_data(str(args.observable), top_n=int(args.top_n), include_total=True),
        title=f"Inventory Time Evolution ({str(args.observable).title()})",
        xlabel=f"Time Since {str(args.time_origin).replace('_', ' ').title()} (s)",
        ylabel=ylabel,
        log_y=str(args.observable) in {"activity", "atoms", "dose"},
        log_x=False,
        half_lives=None,
        save_path=plot_output,
    )
    _close_figure(fig)

    output_payload = {
        "schema": "fluxforge.inventory_time_evolution.v1",
        "sample_id": inventory_state.sample_id,
        "activity_review_file": str(args.activity_review_file),
        "gamma_source_id": inventory_state.gamma_source_id,
        "custom_gamma_path": inventory_state.custom_gamma_path,
        "decay_source_id": result.decay_source_id,
        "time_origin": result.time_origin,
        "relative_times_s": list(result.relative_times_s),
        "absolute_times_s": list(result.absolute_times_s),
        "observable": str(args.observable),
        "distance_cm": float(args.distance_cm),
        "notes": list(result.notes),
        "reference_states": {
            name: list(rows)
            for name, rows in result.reference_rows_by_name.items()
        },
        "time_series_rows": result.time_series_rows(),
        "artifacts": {
            "timeseries_csv": (
                time_series_artifact
                or {"path": time_series_csv.name, "format": "csv"}
            ),
            "eoi_csv": eoi_artifact or {"path": eoi_csv.name, "format": "csv"},
            "count_start_csv": (
                count_start_artifact
                or {"path": count_start_csv.name, "format": "csv"}
            ),
            "count_end_csv": (
                count_end_artifact
                or {"path": count_end_csv.name, "format": "csv"}
            ),
            "plot": {"path": plot_output.name, "format": "png"},
        },
    }
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote inventory review bundle to {output_path}")


def cmd_isotope_priority(args: argparse.Namespace) -> None:
    payload = _load_json(args.activity_review_file)
    isotopes_of_interest = _parse_csv_strings(
        getattr(args, "isotopes_of_interest", None)
    )
    weights = IsotopePriorityWeights(
        activity=max(float(getattr(args, "weight_activity", 0.35) or 0.35), 0.0),
        detectability=max(
            float(getattr(args, "weight_detectability", 0.25) or 0.25),
            0.0,
        ),
        confidence=max(float(getattr(args, "weight_confidence", 0.20) or 0.20), 0.0),
        line_support=max(
            float(getattr(args, "weight_line_support", 0.10) or 0.10),
            0.0,
        ),
        dose=max(float(getattr(args, "weight_dose", 0.10) or 0.10), 0.0),
    )
    top_n = int(getattr(args, "top_n", 0) or 0)
    ranked = rank_isotopes_from_activity_review_payload(
        payload,
        weights=weights,
        isotopes_of_interest=isotopes_of_interest,
        top_n=top_n if top_n > 0 else None,
    )

    output_payload = serialize_isotope_priority_ranking(
        ranked,
        source_path=str(args.activity_review_file),
        isotopes_of_interest=isotopes_of_interest,
        weights=weights,
    )
    output_payload["generated_at"] = (
        datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )

    output_path = Path(args.output)
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    csv_output = getattr(args, "csv_output", None)
    if csv_output is not None:
        _write_dict_rows(Path(csv_output), list(output_payload.get("ranked_isotopes") or []))

    print(f"Wrote isotope-priority ranking to {output_path}")


def cmd_masking_review(args: argparse.Namespace) -> None:
    payload = _load_json(args.activity_review_file)
    isotopes_of_interest = _parse_csv_strings(
        getattr(args, "isotopes_of_interest", None)
    )
    top_n = int(getattr(args, "top_n", 0) or 0)

    ranked = rank_line_masking_from_activity_review_payload(
        payload,
        energy_window_keV=float(getattr(args, "energy_window_keV", 3.0) or 3.0),
        isotopes_of_interest=isotopes_of_interest,
        top_n=top_n if top_n > 0 else None,
    )
    isotope_summary_rows = summarize_masking_isotopes(
        ranked,
        top_n=top_n if top_n > 0 else None,
    )
    recommendations = recommend_alternate_lines(
        payload,
        ranked,
        top_n=top_n if top_n > 0 else None,
    )

    output_payload = serialize_masking_review(
        ranked,
        source_path=str(args.activity_review_file),
        isotopes_of_interest=isotopes_of_interest,
        energy_window_keV=float(getattr(args, "energy_window_keV", 3.0) or 3.0),
        isotope_summary_rows=isotope_summary_rows,
        recommendations=recommendations,
    )
    output_payload["generated_at"] = (
        datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )

    output_path = Path(args.output)
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    csv_output = getattr(args, "csv_output", None)
    if csv_output is not None:
        _write_dict_rows(
            Path(csv_output),
            list(output_payload.get("line_masking_results") or []),
        )
    isotope_csv_output = getattr(args, "isotope_csv_output", None)
    if isotope_csv_output is not None:
        _write_dict_rows(
            Path(isotope_csv_output),
            list(output_payload.get("masking_isotope_ranking") or []),
        )
    recommendation_csv_output = getattr(args, "recommendation_csv_output", None)
    if recommendation_csv_output is not None:
        _write_dict_rows(
            Path(recommendation_csv_output),
            list(output_payload.get("alternate_line_recommendations") or []),
        )

    print(f"Wrote masking-review bundle to {output_path}")


def cmd_optimization_sweep(args: argparse.Namespace) -> None:
    objective = str(getattr(args, "objective", "di-fom")).lower()
    payload_source = "input"
    input_path = getattr(args, "input", None)
    neutron_source_summary: dict[str, Any] | None = None
    activity_review_payload: dict[str, Any] | None = None

    if input_path is not None:
        payload = _load_json(Path(input_path))
    else:
        activity_review_file = getattr(args, "activity_review_file", None)
        if activity_review_file is None:
            raise ValueError(
                "optimization-sweep requires --input or --activity-review-file."
            )

        activity_review_payload = _load_json(Path(activity_review_file))
        irradiation_grid_s = _parse_csv_floats(
            getattr(args, "irradiation_grid_s", None)
        ) or [1800.0, 3600.0, 7200.0, 14400.0]
        cooldown_grid_s = _parse_csv_floats(
            getattr(args, "cooldown_grid_s", None)
        ) or [0.0, 1800.0, 7200.0, 21600.0]
        count_grid_s = _parse_csv_floats(
            getattr(args, "count_grid_s", None)
        ) or [300.0, 600.0, 900.0, 1800.0]

        unfold_payload = None
        unfold_file = getattr(args, "unfold_file", None)
        if unfold_file is not None:
            unfold_payload = read_unfold_result(Path(unfold_file))

        neutron_spectrum_file = getattr(args, "neutron_spectrum_file", None)
        neutron_source_summary = summarize_neutron_spectrum_source(
            unfold_payload=unfold_payload,
            neutron_spectrum_csv=(
                None if neutron_spectrum_file is None else Path(neutron_spectrum_file)
            ),
            base_scale=max(float(getattr(args, "flux_scale", 1.0) or 1.0), 0.0),
            reference_integral_flux=max(
                float(getattr(args, "reference_flux_integral", 0.0) or 0.0),
                0.0,
            ),
        )
        payload = build_difom_payload_from_activity_review(
            activity_review_payload,
            irradiation_grid_s=tuple(float(item) for item in irradiation_grid_s),
            cooldown_grid_s=tuple(float(item) for item in cooldown_grid_s),
            count_grid_s=tuple(float(item) for item in count_grid_s),
            reference_irradiation_time_s=max(
                float(getattr(args, "reference_irradiation_time_s", 3600.0) or 3600.0),
                1.0,
            ),
            flux_scale=float(neutron_source_summary.get("applied_flux_scale", 1.0)),
        )
        payload_source = "activity-review"

        generated_candidates_output = getattr(args, "generated_candidates_output", None)
        if generated_candidates_output is not None:
            generated_output = Path(generated_candidates_output)
            _ensure_parent_dir(generated_output)
            generated_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    isotopes_of_interest = _parse_csv_strings(
        getattr(args, "isotopes_of_interest", None)
    )
    payload, isotope_filter_summary = _filter_optimization_payload_by_isotopes(
        payload,
        isotopes_of_interest,
    )
    if int(isotope_filter_summary.get("candidates_after", 0)) <= 0:
        raise ValueError(
            "No schedule candidates remain after applying --isotopes-of-interest."
        )
    isotope_weights: Dict[str, float] = {}

    if objective == "di-fom":
        candidates, isotope_weights = parse_difom_sweep_payload(payload)
        ranked = rank_difom_schedules(candidates, isotope_weights=isotope_weights)
        output_payload = serialize_difom_ranking(ranked)
    elif objective in {"fim-d", "fim-a", "fim-c"}:
        candidates, isotope_weights = parse_difom_sweep_payload(payload)
        target_nuclide = getattr(args, "target_nuclide", None)
        nuisance_variance_fraction = max(
            float(getattr(args, "nuisance_variance_fraction", 0.0) or 0.0),
            0.0,
        )
        fim_regularization = max(
            float(getattr(args, "fim_regularization", 1.0e-6) or 1.0e-6),
            1.0e-12,
        )
        ranked = rank_fim_schedules(
            candidates,
            objective=objective,
            target_nuclide=target_nuclide,
            nuisance_variance_fraction=nuisance_variance_fraction,
            regularization=fim_regularization,
        )
        output_payload = serialize_fim_ranking(ranked, objective=objective)
        output_payload["target_nuclide"] = target_nuclide
        output_payload["nuisance_variance_fraction"] = nuisance_variance_fraction
        output_payload["fim_regularization"] = fim_regularization
    elif objective == "mwdcs":
        raw_offsets = _parse_csv_floats(getattr(args, "mwdcs_window_offsets_s", None))
        window_offsets_s = tuple(raw_offsets) if raw_offsets else (0.0, 7200.0, 86400.0)
        window_count_time_s = max(
            float(getattr(args, "mwdcs_window_count_time_s", 900.0) or 900.0),
            1.0,
        )
        overlap_penalty = max(
            float(getattr(args, "mwdcs_overlap_penalty", 0.0) or 0.0),
            0.0,
        )
        full_spectrum_mode = bool(getattr(args, "mwdcs_full_spectrum_mode", False))
        candidates, isotope_weights = parse_mwdcs_sweep_payload(
            payload,
            default_window_offsets_s=window_offsets_s,
            default_window_count_time_s=window_count_time_s,
        )
        ranked = rank_mwdcs_schedules(
            candidates,
            isotope_weights=isotope_weights,
            full_spectrum_mode=full_spectrum_mode,
            overlap_penalty=overlap_penalty,
        )
        output_payload = serialize_mwdcs_ranking(ranked)
        output_payload["window_offsets_s"] = list(window_offsets_s)
        output_payload["window_count_time_s"] = window_count_time_s
        output_payload["full_spectrum_mode"] = full_spectrum_mode
        output_payload["overlap_penalty"] = overlap_penalty
    elif objective == "bass-d":
        if not bool(getattr(args, "enable_advanced_objectives", False)):
            raise ValueError(
                "Objective 'bass-d' is advanced. Re-run with --enable-advanced-objectives."
            )
        dose_weight = max(
            float(getattr(args, "bassd_dose_weight", 0.02) or 0.02),
            0.0,
        )
        exploration_temperature = max(
            float(getattr(args, "bassd_exploration_temperature", 0.0) or 0.0),
            0.0,
        )
        seed = int(getattr(args, "bassd_seed", 17) or 17)
        candidates, isotope_weights = parse_bassd_sweep_payload(payload)
        ranked = rank_bassd_schedules(
            candidates,
            isotope_weights=isotope_weights,
            dose_weight=dose_weight,
            exploration_temperature=exploration_temperature,
            seed=seed,
        )
        output_payload = serialize_bassd_ranking(ranked)
        output_payload["advanced_objective"] = True
        output_payload["dose_weight"] = dose_weight
        output_payload["exploration_temperature"] = exploration_temperature
        output_payload["seed"] = seed
    elif objective == "stbd-mr":
        if not bool(getattr(args, "enable_advanced_objectives", False)):
            raise ValueError(
                "Objective 'stbd-mr' is advanced. Re-run with --enable-advanced-objectives."
            )
        raw_offsets = _parse_csv_floats(getattr(args, "stbdmr_window_offsets_s", None))
        window_offsets_s = tuple(raw_offsets) if raw_offsets else (0.0, 7200.0, 86400.0)
        window_count_time_s = max(
            float(getattr(args, "stbdmr_window_count_time_s", 900.0) or 900.0),
            1.0,
        )
        masking_regularization = max(
            float(getattr(args, "stbdmr_masking_regularization", 0.1) or 0.1),
            0.0,
        )
        differentiable_graph_mode = bool(
            getattr(args, "stbdmr_differentiable_graph", False)
        )
        graph_temperature = max(
            float(getattr(args, "stbdmr_graph_temperature", 2.0) or 2.0),
            1.0e-6,
        )
        candidates, isotope_weights = parse_stbdmr_sweep_payload(
            payload,
            default_window_offsets_s=window_offsets_s,
            default_window_count_time_s=window_count_time_s,
        )
        ranked = rank_stbdmr_schedules(
            candidates,
            isotope_weights=isotope_weights,
            masking_regularization=masking_regularization,
            differentiable_graph_mode=differentiable_graph_mode,
            graph_temperature=graph_temperature,
        )
        output_payload = serialize_stbdmr_ranking(ranked)
        output_payload["advanced_objective"] = True
        output_payload["window_offsets_s"] = list(window_offsets_s)
        output_payload["window_count_time_s"] = window_count_time_s
        output_payload["masking_regularization"] = masking_regularization
        output_payload["differentiable_graph_mode"] = differentiable_graph_mode
        output_payload["graph_temperature"] = graph_temperature
    else:
        raise ValueError(
            "Unsupported objective. Use one of: di-fom, fim-d, fim-a, fim-c, mwdcs, bass-d, stbd-mr."
        )

    output_payload["input"] = None if input_path is None else str(input_path)
    output_payload["candidate_source"] = payload_source
    if payload_source == "activity-review":
        output_payload["activity_review_file"] = str(getattr(args, "activity_review_file"))
        output_payload["candidate_generation"] = {
            "irradiation_grid_s": _parse_csv_floats(
                getattr(args, "irradiation_grid_s", None)
            )
            or [1800.0, 3600.0, 7200.0, 14400.0],
            "cooldown_grid_s": _parse_csv_floats(getattr(args, "cooldown_grid_s", None))
            or [0.0, 1800.0, 7200.0, 21600.0],
            "count_grid_s": _parse_csv_floats(getattr(args, "count_grid_s", None))
            or [300.0, 600.0, 900.0, 1800.0],
            "reference_irradiation_time_s": float(
                getattr(args, "reference_irradiation_time_s", 3600.0) or 3600.0
            ),
        }
    if neutron_source_summary is not None:
        output_payload["neutron_source"] = neutron_source_summary
    output_payload["isotope_weights"] = isotope_weights
    output_payload["isotopes_of_interest"] = list(isotopes_of_interest)
    output_payload["isotope_filter_summary"] = isotope_filter_summary
    output_payload["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    output_path = Path(args.output)
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    csv_output = getattr(args, "csv_output", None)
    if csv_output is not None:
        rows = []
        for item in output_payload["ranked_candidates"]:
            first_window = (item.get("window_scores") or [{}])[0]
            row = {
                "rank": item["rank"],
                "label": item["label"],
                "irradiation_time_s": item["irradiation_time_s"],
                "cooldown_time_s": item.get(
                    "cooldown_time_s", first_window.get("cooldown_time_s")
                ),
                "count_time_s": item.get(
                    "count_time_s", first_window.get("count_time_s")
                ),
                "objective": objective,
            }
            if objective == "di-fom":
                row["difom_score"] = item["difom_score"]
            elif objective == "mwdcs":
                row["objective_score"] = item["total_score"]
                row["window_count"] = len(item.get("window_scores") or [])
            elif objective == "bass-d":
                row["objective_score"] = item["total_utility"]
                row["action_count"] = len(item.get("action_scores") or [])
            elif objective == "stbd-mr":
                row["objective_score"] = item["total_score"]
                diagnostics = item.get("diagnostics") or {}
                row["graph_density"] = diagnostics.get("graph_density")
                row["masking_penalty"] = diagnostics.get("masking_penalty")
                row["gradient_norm"] = diagnostics.get("gradient_norm")
            else:
                row["objective_score"] = item["objective_score"]
                diagnostics = item.get("matrix_diagnostics") or {}
                row["condition_number"] = diagnostics.get("condition_number")
                row["effective_rank"] = diagnostics.get("effective_rank")
            rows.append(row)
        _write_dict_rows(Path(csv_output), rows)

    if activity_review_payload is not None:
        support_outputs = _write_phase6_support_artifacts(
            output_path=output_path,
            activity_review_payload=activity_review_payload,
            optimization_payload=output_payload,
            isotopes_of_interest=isotopes_of_interest,
        )
        output_payload["support_artifacts"] = {
            key: str(path) for key, path in support_outputs.items()
        }
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"Wrote optimization sweep bundle ({objective}) to {output_path}")


def cmd_second_irradiation_plan(args: argparse.Namespace) -> None:
    inventory_payload = _load_json(Path(args.inventory_file))
    schedule_payload = _load_json(Path(args.schedule_file))
    candidates_payload = _load_json(Path(args.candidates_file))

    inventory_state = build_inventory_state_from_payload(
        inventory_payload,
        decay_source_id=str(
            inventory_payload.get("decay_source_id") or DEFAULT_DECAY_SOURCE_ID
        ),
    )
    candidates = parse_second_irradiation_candidates(candidates_payload)
    plan = plan_second_irradiation(
        inventory_state,
        first_cooling_time_s=float(
            schedule_payload.get("first_cooling_time_s")
            or schedule_payload.get("cooling_time_s")
            or 0.0
        ),
        second_irradiation_time_s=float(
            schedule_payload.get("second_irradiation_time_s") or 0.0
        ),
        target_weights=schedule_payload.get("target_weights") or {},
        candidates=candidates,
    )

    output_payload = serialize_second_irradiation_plan(
        plan,
        inventory_source=str(args.inventory_file),
        schedule_source=str(args.schedule_file),
        candidates_source=str(args.candidates_file),
    )
    output_payload["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    output_path = Path(args.output)
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    selected_csv_output = (
        getattr(args, "csv_output", None)
        or _phase6_artifact_path(output_path, "_selected_inventory.csv")
    )
    _write_dict_rows(Path(selected_csv_output), output_payload["selected_inventory_rows"])
    ranked_csv_output = _phase6_artifact_path(output_path, "_ranked_candidates.csv")
    _write_dict_rows(Path(ranked_csv_output), output_payload["ranked_candidates"])
    output_payload["support_artifacts"] = {
        "selected_inventory": str(selected_csv_output),
        "ranked_candidates": str(ranked_csv_output),
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote second-irradiation plan bundle to {output_path}")


def cmd_ffexp_export(args: argparse.Namespace) -> None:
    activity_review_payload = (
        None
        if getattr(args, "activity_review_file", None) is None
        else _load_json(Path(args.activity_review_file))
    )
    inventory_payload = (
        None
        if getattr(args, "inventory_review_file", None) is None
        else _load_json(Path(args.inventory_review_file))
    )
    masking_payload = (
        None
        if getattr(args, "masking_file", None) is None
        else _load_json(Path(args.masking_file))
    )
    optimization_payload = (
        None
        if getattr(args, "optimization_file", None) is None
        else _load_json(Path(args.optimization_file))
    )
    second_irradiation_payload = (
        None
        if getattr(args, "second_irradiation_file", None) is None
        else _load_json(Path(args.second_irradiation_file))
    )

    support_artifacts = None
    if activity_review_payload is not None:
        support_artifacts = build_phase6_support_artifacts(
            activity_review_payload,
            optimization_payload,
            isotopes_of_interest=tuple(
                str(item)
                for item in ((optimization_payload or {}).get("isotopes_of_interest") or [])
            ),
        )

    if inventory_payload is None and support_artifacts is not None:
        inventory_payload = {
            "schema": "fluxforge.inventory_time_evolution.v1",
            "time_series_rows": support_artifacts["inventory_timeseries_rows"],
            "activities_at_irradiation": support_artifacts["activities_at_irradiation_rows"],
            "activities_at_count_start": support_artifacts["activities_at_count_start_rows"],
            "activities_at_count_end": support_artifacts["activities_at_count_end_rows"],
            "dose_endpoints": support_artifacts["dose_endpoints_rows"],
        }

    if masking_payload is None and support_artifacts is not None:
        masking_payload = {
            "schema": "fluxforge.masking_review.v1",
            "line_masking_results": support_artifacts["masking_candidates_rows"],
            "masking_isotope_ranking": support_artifacts["masking_isotope_rows"],
            "alternate_line_recommendations": support_artifacts["alternate_line_rows"],
        }

    optimization_section = optimization_payload
    if support_artifacts is not None:
        optimization_section = {
            "payload": optimization_payload,
            "optimization_grid": support_artifacts["optimization_grid_rows"],
            "recommended_schedules": support_artifacts["recommended_schedule_rows"],
            "dose_endpoints": support_artifacts["dose_endpoints_rows"],
        }

    summary = _build_ffexp_summary(
        activity_review_payload=activity_review_payload,
        optimization_payload=optimization_payload,
        second_irradiation_payload=second_irradiation_payload,
    )
    metadata = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "activity_review_file": (
            None if getattr(args, "activity_review_file", None) is None else str(args.activity_review_file)
        ),
        "inventory_review_file": (
            None if getattr(args, "inventory_review_file", None) is None else str(args.inventory_review_file)
        ),
        "masking_file": (
            None if getattr(args, "masking_file", None) is None else str(args.masking_file)
        ),
        "optimization_file": (
            None if getattr(args, "optimization_file", None) is None else str(args.optimization_file)
        ),
        "second_irradiation_file": (
            None
            if getattr(args, "second_irradiation_file", None) is None
            else str(args.second_irradiation_file)
        ),
    }
    plot_manifest = {
        "paths": [
            value
            for value in _parse_csv_strings(getattr(args, "plot_paths", None))
            if value
        ]
    }

    output_path = Path(args.output)
    _ensure_parent_dir(output_path)
    write_ffexp_bundle(
        output_path,
        summary=summary,
        metadata=metadata,
        activities=activity_review_payload,
        inventory=inventory_payload,
        masking=masking_payload,
        optimization=optimization_section,
        second_irradiation=second_irradiation_payload,
        plot_manifest=plot_manifest,
        source_path=(
            None
            if getattr(args, "activity_review_file", None) is None
            else Path(args.activity_review_file)
        ),
    )
    print(f"Wrote ffexp benchmark bundle to {output_path}")


def cmd_library_list(args: argparse.Namespace) -> None:
    records = list_nuclear_data_sources()
    capability = str(getattr(args, "capability", "") or "").strip()
    kind = str(getattr(args, "kind", "") or "").strip()
    if capability:
        records = tuple(
            record for record in records if capability in record.capabilities
        )
    if kind:
        records = tuple(record for record in records if record.kind == kind)
    payload = [_serialize_nuclear_data_source(record) for record in records]
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return
    for item in payload:
        capabilities = ",".join(item["capabilities"])
        path_hint = item["path_hint"] or "-"
        print(
            f"{item['source_id']}\t{item['label']}\t{item['kind']}\t"
            f"builtin={item['builtin']}\tcapabilities={capabilities}\tpath={path_hint}"
        )


def cmd_library_register(args: argparse.Namespace) -> None:
    record = register_user_gamma_source(
        args.alias,
        args.locator,
        description=getattr(args, "description", None),
    )
    print(f"Registered {record.source_id} -> {record.path_hint}")


def cmd_library_remove(args: argparse.Namespace) -> None:
    removed = remove_user_gamma_source(str(args.source_id))
    if not removed:
        raise ValueError(f"User library not found: {args.source_id}")
    print(f"Removed {args.source_id}")


def cmd_rates(args: argparse.Namespace) -> None:
    line_payload = read_line_activities(args.lines_file)
    if args.validate:
        validate_or_raise(line_payload)
    segments = None
    if args.segments_file:
        segments = _load_json(args.segments_file)
    if segments is None:
        segments = [{"duration_s": args.duration_s, "relative_power": 1.0}]

    segment_objs = [IrradiationSegment(**seg) for seg in segments]
    rates = []
    for idx, line in enumerate(line_payload["lines"]):
        half_life_s = line.get("half_life_s", args.half_life_s)
        rate_estimate = reaction_rate_from_activity(
            line["activity_Bq"], segment_objs, half_life_s
        )
        reaction_id = (
            line.get("reaction_id") or line.get("isotope") or f"reaction_{idx + 1}"
        )
        rates.append(
            {
                "reaction_id": reaction_id,
                "rate": rate_estimate.rate,
                "uncertainty": rate_estimate.uncertainty,
                "half_life_s": half_life_s,
            }
        )

    write_reaction_rates(
        args.output,
        rates=rates,
        segments=segments,
        source_path=args.lines_file,
    )
    print(f"Wrote reaction rates to {args.output}")


def cmd_astm_e2005(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan_file)
    result = analyze_astm_e2005_plan(plan)
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote ASTM E2005 workflow bundle to {args.output}")


def cmd_astm_e261(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan_file)
    result = analyze_astm_e261_plan(plan)
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote ASTM E261 workflow bundle to {args.output}")


def cmd_astm_e262(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan_file)
    result = analyze_astm_e262_plan(plan)
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote ASTM E262 workflow bundle to {args.output}")


def cmd_astm_e3376(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan_file)
    result = analyze_astm_e3376_plan(plan)
    _ensure_parent_dir(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote ASTM E3376 workflow bundle to {args.output}")


def cmd_rafm_validate(args: argparse.Namespace) -> None:
    summary = run_rafm_validation(
        Path(args.example_root),
        results_root=Path(args.results_root) if args.results_root is not None else None,
        enforce_thresholds=not bool(getattr(args, "no_fail", False)),
        max_spectra=getattr(args, "max_spectra", None),
        flux_wire_counting_method=getattr(args, "flux_wire_counting_method", None),
        generic_targeted_counting_method=getattr(args, "generic_counting_method", None),
    )
    print(f"RAFM validation completed at {summary['results_root']}")
    print("Overall passed: " + ("yes" if bool(summary.get("overall_passed")) else "no"))
    print(f"Matched raw/QG pairs: {summary.get('n_matched_pairs', 0)}")


def cmd_rafm_qg_benchmark(args: argparse.Namespace) -> None:
    summary = run_qg_benchmark(
        Path(args.example_root),
        results_root=Path(args.results_root) if args.results_root is not None else None,
        max_spectra=getattr(args, "max_spectra", None),
    )
    print(f"RAFM QG benchmark completed at {summary['results_root']}")
    print(f"Processed QG spectra: {summary.get('n_qg_processed', 0)}")
    print(f"Reaction rows: {summary.get('n_reaction_rows', 0)}")


def cmd_rafm_compare_branches(args: argparse.Namespace) -> None:
    summary = compare_rafm_completion_results(
        Path(args.raw_results_root),
        Path(args.qg_results_root),
        output_root=Path(args.output_root) if args.output_root is not None else None,
    )
    print(f"RAFM branch comparison completed at {summary['comparison_root']}")
    print(f"Matched reactions: {summary.get('matched_reactions', 0)}")
    print(f"Median |rate rel err|: {summary.get('median_abs_rate_rel_error', 0.0):.6g}")


def cmd_phase6_ldrd_worked_example(args: argparse.Namespace) -> None:
    sample_id = str(
        getattr(args, "sample_id", PHASE6_LDRD_DEFAULT_SAMPLE_ID)
        or PHASE6_LDRD_DEFAULT_SAMPLE_ID
    )
    output_root = Path(
        getattr(args, "output_root", None)
        or phase6_ldrd_default_output_root(sample_id)
    )
    summary_path = run_phase6_ldrd_worked_example(
        sample_id=sample_id,
        output_root=output_root,
    )
    print(f"Wrote Phase 6 LDRD worked example artifacts to {output_root}")
    print(f"Summary: {summary_path}")


def cmd_phase6_ldrd_second_irradiation_repo(args: argparse.Namespace) -> None:
    sample_id = str(
        getattr(args, "sample_id", PHASE6_LDRD_DEFAULT_SAMPLE_ID)
        or PHASE6_LDRD_DEFAULT_SAMPLE_ID
    )
    output_root = Path(
        getattr(args, "output_root", None)
        or phase6_ldrd_second_irradiation_repo_default_output_root(sample_id)
    )
    top_n = max(int(getattr(args, "top_n", 8) or 8), 1)
    outputs = run_phase6_ldrd_second_irradiation_decision_repo(
        sample_id=sample_id,
        output_root=output_root,
        top_n=top_n,
    )
    print(f"Wrote RAFM second-irradiation decision repository to {output_root}")
    print(f"Report: {outputs['report']}")
    for key in ("graphic_schedule", "graphic_isotopes", "graphic_cooldown"):
        graphic = outputs.get(key)
        if graphic is not None:
            print(f"Graphic ({key}): {graphic}")


def cmd_response(args: argparse.Namespace) -> None:
    boundaries = [float(x) for x in _load_json(args.boundaries_file)]
    groups = EnergyGroupStructure(boundaries)
    cross_sections_raw = _load_json(args.cross_section_file)
    number_densities = _load_json(args.number_densities_file)

    reactions = []
    number_density_values: List[float] = []
    for reaction_id, sigma in cross_sections_raw.items():
        reactions.append(
            ReactionCrossSection(
                reaction_id=reaction_id, sigma_g=[float(s) for s in sigma]
            )
        )
        number_density_values.append(float(number_densities[reaction_id]))

    response = build_response_matrix(reactions, groups, number_density_values)
    write_response_bundle(
        args.output,
        matrix=response.matrix,
        reactions=response.reactions,
        boundaries_eV=response.energy_groups.boundaries_eV,
        source_path=args.cross_section_file,
    )
    print(f"Wrote response bundle to {args.output}")


def _zero_covariance(size: int) -> List[List[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _predicted_rates_and_uncertainty(
    response_matrix: List[List[float]],
    flux: List[float],
    covariance: List[List[float]],
) -> tuple[List[float], List[float]]:
    response_np = np.asarray(response_matrix, dtype=float)
    flux_np = np.asarray(flux, dtype=float)
    predicted = response_np @ flux_np

    cov_np = np.asarray(covariance, dtype=float)
    if (
        cov_np.ndim == 2
        and cov_np.shape == (flux_np.size, flux_np.size)
        and np.any(np.abs(cov_np) > 0.0)
    ):
        predicted_cov = response_np @ cov_np @ response_np.T
        predicted_unc = np.sqrt(np.clip(np.diag(predicted_cov), 0.0, None))
    else:
        predicted_unc = np.zeros(response_np.shape[0], dtype=float)
    return predicted.tolist(), predicted_unc.tolist()


def _build_unfold_diagnostics(
    *,
    reactions: List[str],
    response_matrix: List[List[float]],
    measured_rates: List[float],
    rate_uncertainties: List[float],
    prior_flux: List[float],
    prior_cov: List[List[float]],
    flux: List[float],
    covariance: List[List[float]],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(diagnostics)
    payload["reactions"] = [str(item) for item in reactions]
    payload["measured_rates"] = [float(value) for value in measured_rates]
    payload["measured_rate_uncertainties"] = [
        float(value) for value in rate_uncertainties
    ]
    payload["prior_flux"] = [float(value) for value in prior_flux]

    prior_cov_np = np.asarray(prior_cov, dtype=float)
    if prior_cov_np.ndim == 2 and prior_cov_np.shape == (
        len(prior_flux),
        len(prior_flux),
    ):
        payload["prior_flux_uncertainty"] = (
            np.sqrt(np.clip(np.diag(prior_cov_np), 0.0, None)).astype(float).tolist()
        )

    cov_np = np.asarray(covariance, dtype=float)
    if cov_np.ndim == 2 and cov_np.shape == (len(flux), len(flux)):
        payload["flux_uncertainty"] = (
            np.sqrt(np.clip(np.diag(cov_np), 0.0, None)).astype(float).tolist()
        )

    predicted_rates, predicted_unc = _predicted_rates_and_uncertainty(
        response_matrix, flux, covariance
    )
    payload["predicted_rates"] = [float(value) for value in predicted_rates]
    payload["predicted_rate_uncertainties"] = [float(value) for value in predicted_unc]

    residuals = np.asarray(predicted_rates, dtype=float) - np.asarray(
        measured_rates, dtype=float
    )
    sigma = np.asarray(rate_uncertainties, dtype=float)
    pulls = np.divide(residuals, sigma, out=np.zeros_like(residuals), where=sigma > 0.0)
    payload["rate_residuals"] = residuals.astype(float).tolist()
    payload["rate_pulls"] = pulls.astype(float).tolist()
    return merge_flux_diagnostics(payload, flux)


def _solve_unfold_method(
    *,
    method: str,
    response_matrix: List[List[float]],
    measured_rates: List[float],
    rate_uncertainties: List[float],
    measurement_cov: List[List[float]],
    prior_flux: List[float],
    prior_cov: List[List[float]],
    args: argparse.Namespace,
) -> tuple[List[float], List[List[float]], float, str, Dict[str, Any]]:
    method_key = str(method).strip().lower()
    if method_key == "gls":
        solution = gls_adjust(
            response_matrix,
            measured_rates,
            measurement_cov,
            prior_flux,
            prior_cov,
            enforce_nonnegativity=bool(getattr(args, "enforce_nonnegativity", True)),
        )
        diagnostics = {
            **dict(getattr(solution, "diagnostics", {})),
            "reduced_chi2": float(getattr(solution, "reduced_chi2", 0.0)),
            "n_dof": int(getattr(solution, "n_dof", 0)),
        }
        if getattr(solution, "pull", None) is not None:
            diagnostics["pull"] = [float(value) for value in solution.pull]
        if getattr(solution, "prior_posterior_change", None) is not None:
            diagnostics["prior_posterior_change"] = [
                float(value) for value in solution.prior_posterior_change
            ]
        return (
            [float(value) for value in solution.flux],
            [[float(item) for item in row] for row in solution.covariance],
            float(solution.chi2),
            "gls",
            diagnostics,
        )

    if method_key == "gravel":
        solution = GravelUnfolder(
            max_iterations=int(getattr(args, "max_iters", 250)),
            tolerance=float(getattr(args, "tolerance", 1e-4)),
            chi2_tolerance=float(getattr(args, "chi2_tolerance", 0.01)),
            floor=float(getattr(args, "floor", 1e-20)),
            relaxation=float(getattr(args, "relaxation", 0.7)),
        ).unfold(
            np.asarray(measured_rates, dtype=float),
            np.asarray(response_matrix, dtype=float),
            initial_flux=np.asarray(prior_flux, dtype=float),
            measurement_uncertainty=np.asarray(rate_uncertainties, dtype=float),
            convergence_mode=str(getattr(args, "convergence_mode", "relative")),
            seed_with_ml=bool(getattr(args, "use_ml_seed", False)),
            confidence_threshold=float(getattr(args, "ml_seed_threshold", 0.6)),
            verbose=bool(getattr(args, "verbose_solver", False)),
        )
        covariance = np.diag(np.square(np.asarray(solution.uncertainties, dtype=float)))
        diagnostics = {
            **dict(solution.parameters_used),
            "iterations": int(solution.iterations),
            "converged": bool(solution.converged),
            "chi2_history": [float(value) for value in solution.convergence_history],
            "final_residuals": [float(value) for value in solution.residuals],
        }
        return (
            [float(value) for value in solution.flux],
            covariance.tolist(),
            float(solution.chi_squared),
            "gravel",
            diagnostics,
        )

    if method_key == "mlem":
        solution = mlem(
            response_matrix,
            measured_rates,
            initial_flux=prior_flux,
            measurement_uncertainty=rate_uncertainties,
            max_iters=int(getattr(args, "max_iters", 250)),
            tolerance=float(getattr(args, "tolerance", 1e-4)),
            chi2_tolerance=float(getattr(args, "chi2_tolerance", 0.01)),
            floor=float(getattr(args, "floor", 1e-20)),
            relaxation=float(getattr(args, "relaxation", 0.8)),
            convergence_mode=str(getattr(args, "convergence_mode", "relative")),
            verbose=bool(getattr(args, "verbose_solver", False)),
        )
        diagnostics = {
            **dict(getattr(solution, "diagnostics", {})),
            "iterations": int(solution.iterations),
            "converged": bool(solution.converged),
            "chi2_history": [float(value) for value in solution.chi_squared_history],
            "final_residuals": [
                float(value) for value in solution.final_residuals or []
            ],
            "convergence_mode": str(getattr(args, "convergence_mode", "relative")),
        }
        return (
            [float(value) for value in solution.flux],
            _zero_covariance(len(solution.flux)),
            float(solution.chi_squared),
            "mlem",
            diagnostics,
        )

    if method_key == "maxed":
        solution = MaxedUnfolder(
            max_iterations=int(getattr(args, "max_iters", 250)),
        ).unfold(
            np.asarray(measured_rates, dtype=float),
            np.asarray(response_matrix, dtype=float),
            initial_flux=np.asarray(prior_flux, dtype=float),
            measurement_uncertainty=np.asarray(rate_uncertainties, dtype=float),
        )
        covariance = np.diag(np.square(np.asarray(solution.uncertainties, dtype=float)))
        diagnostics = {
            **dict(solution.parameters_used),
            "iterations": int(solution.iterations),
            "converged": bool(solution.converged),
            "chi2_history": [float(value) for value in solution.convergence_history],
            "final_residuals": [float(value) for value in solution.residuals],
        }
        return (
            [float(value) for value in solution.flux],
            covariance.tolist(),
            float(solution.chi_squared),
            "maxed",
            diagnostics,
        )

    if method_key == "ml_seed":
        solution = MLSeedUnfolder().unfold(
            np.asarray(measured_rates, dtype=float),
            np.asarray(response_matrix, dtype=float),
            initial_flux=np.asarray(prior_flux, dtype=float),
            measurement_uncertainty=np.asarray(rate_uncertainties, dtype=float),
            confidence_threshold=float(getattr(args, "ml_seed_threshold", 0.6)),
        )
        covariance = np.diag(np.square(np.asarray(solution.uncertainties, dtype=float)))
        diagnostics = {
            **dict(solution.parameters_used),
            "iterations": int(solution.iterations),
            "converged": bool(solution.converged),
            "chi2_history": [float(value) for value in solution.convergence_history],
            "final_residuals": [float(value) for value in solution.residuals],
        }
        return (
            [float(value) for value in solution.flux],
            covariance.tolist(),
            float(solution.chi_squared),
            "ml_seed",
            diagnostics,
        )

    if method_key == "rmle":
        solution = RMLEUnfolder(
            max_iterations=int(getattr(args, "max_iters", 250)),
            tolerance=float(getattr(args, "tolerance", 1e-6)),
        ).unfold(
            np.asarray(measured_rates, dtype=float),
            np.asarray(response_matrix, dtype=float),
            initial_flux=np.asarray(prior_flux, dtype=float),
            measurement_uncertainty=np.asarray(rate_uncertainties, dtype=float),
            seed_with_ml=bool(getattr(args, "use_ml_seed", False)),
            confidence_threshold=float(getattr(args, "ml_seed_threshold", 0.6)),
        )
        covariance = np.diag(np.square(np.asarray(solution.uncertainties, dtype=float)))
        diagnostics = {
            **dict(solution.parameters_used),
            "iterations": int(solution.iterations),
            "converged": bool(solution.converged),
            "chi2_history": [float(value) for value in solution.convergence_history],
            "final_residuals": [float(value) for value in solution.residuals],
        }
        return (
            [float(value) for value in solution.flux],
            covariance.tolist(),
            float(solution.chi_squared),
            "rmle",
            diagnostics,
        )

    raise ValueError(f"Unsupported unfold method: {method}")


def cmd_unfold(args: argparse.Namespace) -> None:
    response_data = read_response_bundle(args.response_file)
    if args.validate:
        validate_or_raise(response_data)
    response_matrix = require_nonnegative("response", response_data["matrix"]).tolist()
    boundaries = response_data["boundaries_eV"]
    reactions = response_data["reactions"]
    groups = EnergyGroupStructure([float(b) for b in boundaries])

    rates_payload = read_reaction_rates(args.rates_file)
    if args.validate:
        validate_or_raise(rates_payload)
    measured_rates = require_nonnegative(
        "measurements",
        [float(rx["rate"]) for rx in rates_payload["rates"]],
    ).astype(float).tolist()
    rate_uncertainties = require_nonnegative(
        "measurement_uncertainty",
        [float(rx["uncertainty"]) for rx in rates_payload["rates"]],
    ).astype(float).tolist()

    if args.prior_flux_file:
        prior_flux = require_nonnegative(
            "prior_flux",
            [float(v) for v in _load_json(args.prior_flux_file)],
        ).astype(float).tolist()
    else:
        avg_response = sum(sum(row) for row in response_matrix) / max(
            len(response_matrix) * len(response_matrix[0]), 1
        )
        prior_flux = require_nonnegative(
            "prior_flux",
            [
                sum(measured_rates) / max(avg_response, 1e-12)
                for _ in range(groups.group_count)
            ],
        ).astype(float).tolist()

    # Prior covariance model (K8)
    cov_model = PriorCovarianceModel(
        getattr(args, "prior_cov_model", PriorCovarianceModel.DIAGONAL.value)
    )
    prior_cov_config = PriorCovarianceConfig(
        model_type=cov_model,
        fractional_uncertainty=float(args.prior_uncertainty),
        correlation_length=float(getattr(args, "prior_correlation_length", 1.0)),
    )
    prior_cov_np = prior_cov_config.build_covariance(
        np.asarray(prior_flux, dtype=float),
        np.asarray(boundaries, dtype=float),
    )
    prior_cov = prior_cov_np.tolist()
    measurement_cov = [
        [
            (rate_uncertainties[i] ** 2) if i == j else 0.0
            for j in range(len(measured_rates))
        ]
        for i in range(len(measured_rates))
    ]

    flux, covariance, chi2, method, diagnostics = _solve_unfold_method(
        method=getattr(args, "method", "gls"),
        response_matrix=response_matrix,
        measured_rates=measured_rates,
        rate_uncertainties=rate_uncertainties,
        measurement_cov=measurement_cov,
        prior_flux=prior_flux,
        prior_cov=prior_cov,
        args=args,
    )
    diagnostics = _build_unfold_diagnostics(
        reactions=reactions,
        response_matrix=response_matrix,
        measured_rates=measured_rates,
        rate_uncertainties=rate_uncertainties,
        prior_flux=prior_flux,
        prior_cov=prior_cov,
        flux=flux,
        covariance=covariance,
        diagnostics=diagnostics,
    )
    write_unfold_result(
        args.output,
        boundaries_eV=boundaries,
        reactions=reactions,
        flux=flux,
        covariance=covariance,
        chi2=chi2,
        method=method,
        diagnostics=diagnostics,
        source_path=args.rates_file,
    )
    print(f"Saved unfolded spectrum to {args.output} using {method.upper()}")


def cmd_compare(args: argparse.Namespace) -> None:
    unfold_data = read_unfold_result(args.unfold_file)
    if args.validate:
        validate_or_raise(unfold_data)
    predicted_flux = unfold_data["flux"]
    truth_flux = [float(v) for v in _load_json(args.truth_flux_file)]
    residuals = [p - t for p, t in zip(predicted_flux, truth_flux)]
    metrics = spectrum_comparison_metrics(
        np.asarray(truth_flux, dtype=float),
        np.asarray(predicted_flux, dtype=float),
    )
    metrics["chi2"] = unfold_data.get("chi2")
    write_validation_bundle(
        args.output,
        metrics=metrics,
        truth_flux=truth_flux,
        predicted_flux=predicted_flux,
        residuals=residuals,
        source_path=args.unfold_file,
    )
    print(f"Wrote validation bundle to {args.output}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _weighted_mean_and_uncertainty(
    values: List[float], uncertainties: List[float]
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    valid_pairs = [
        (value, unc) for value, unc in zip(values, uncertainties) if unc > 0.0
    ]
    if valid_pairs:
        weights = [1.0 / (unc * unc) for _, unc in valid_pairs]
        weighted_mean = sum(
            value * weight for (value, _), weight in zip(valid_pairs, weights)
        ) / max(sum(weights), 1e-30)
        return float(weighted_mean), float(np.sqrt(1.0 / max(sum(weights), 1e-30)))
    mean_value = float(np.mean(np.asarray(values, dtype=float)))
    if len(values) == 1:
        return mean_value, float(max(uncertainties[0] if uncertainties else 0.0, 0.0))
    spread = float(
        np.std(np.asarray(values, dtype=float), ddof=0) / max(np.sqrt(len(values)), 1.0)
    )
    return mean_value, spread


def _format_report_value(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{precision}g}"
    return str(value)


def _format_key_value_section(title: str, rows: List[tuple[str, Any]]) -> str:
    if not rows:
        return f"{title}\n{'-' * len(title)}\n(none)"
    width = max(len(label) for label, _ in rows)
    body = "\n".join(
        f"{label:<{width}} : {_format_report_value(value)}" for label, value in rows
    )
    return f"{title}\n{'-' * len(title)}\n{body}"


def _format_table_section(title: str, headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return f"{title}\n{'-' * len(title)}\n(none)"
    widths = [len(header) for header in headers]
    normalized_rows: List[List[str]] = []
    for row in rows:
        normalized = [_format_report_value(value) for value in row]
        normalized_rows.append(normalized)
        for idx, cell in enumerate(normalized):
            widths[idx] = max(widths[idx], len(cell))
    header_line = " | ".join(
        f"{header:<{widths[idx]}}" for idx, header in enumerate(headers)
    )
    divider = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(f"{cell:<{widths[idx]}}" for idx, cell in enumerate(row))
        for row in normalized_rows
    )
    return f"{title}\n{'-' * len(title)}\n{header_line}\n{divider}\n{body}"


def _write_csv_table(
    path: Path, rows: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    _ensure_parent_dir(path)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {"path": path.name, "row_count": len(rows), "format": "csv"}


def discover_validation_report_inputs(results_root: Path) -> Dict[str, Optional[Path]]:
    """Discover report-relevant RAFM validation outputs beneath a results root."""

    root = results_root.resolve()
    candidates = {
        "validation_summary_json": root / "validation_summary.json",
        "validation_summary_md": root / "validation_summary.md",
        "analysis_json_dir": root / "analysis_json",
        "tables_dir": root / "tables",
        "reports_dir": root / "reports",
    }
    unfolding_dir = root / "unfolding"
    preferred_unfold = None
    for name in ("gls.json", "mlem.json", "gravel.json", "discrete.json"):
        candidate = unfolding_dir / name
        if candidate.exists():
            preferred_unfold = candidate
            break
    candidates["preferred_unfold_file"] = preferred_unfold
    return {
        key: value if value is not None and value.exists() else None
        for key, value in candidates.items()
    }


def _aggregate_isotope_activity_rows(
    lines: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for line in lines:
        isotope = str(line.get("isotope") or "unknown")
        grouped.setdefault(isotope, []).append(line)

    rows: List[Dict[str, Any]] = []
    for isotope, isotope_lines in grouped.items():
        activities = [_safe_float(item.get("activity_Bq")) for item in isotope_lines]
        uncertainties = [
            _safe_float(item.get("activity_unc_Bq")) for item in isotope_lines
        ]
        weighted_activity, weighted_unc = _weighted_mean_and_uncertainty(
            activities, uncertainties
        )
        half_life_s = next(
            (
                _safe_float(item.get("half_life_s"))
                for item in isotope_lines
                if _safe_float(item.get("half_life_s")) > 0.0
            ),
            0.0,
        )
        sample_mass_g = next(
            (
                _safe_float(item.get("sample_mass_g"), default=-1.0)
                for item in isotope_lines
                if _safe_float(item.get("sample_mass_g"), default=-1.0) > 0.0
            ),
            None,
        )
        if sample_mass_g is not None and sample_mass_g <= 0.0:
            sample_mass_g = None
        metrics = activation_study_metrics(
            activity_bq=weighted_activity,
            activity_unc_bq=weighted_unc,
            half_life_s=half_life_s,
            isotope=isotope,
            sample_mass_g=sample_mass_g,
        )
        energy_list = sorted(
            {
                _safe_float(item.get("energy_keV"))
                for item in isotope_lines
                if _safe_float(item.get("energy_keV")) > 0.0
            }
        )
        row = {
            "isotope": isotope,
            "n_lines": len(isotope_lines),
            "activity_Bq": weighted_activity,
            "activity_unc_Bq": weighted_unc,
            "total_net_counts": float(
                sum(_safe_float(item.get("net_counts")) for item in isotope_lines)
            ),
            "energies_keV": ", ".join(f"{value:.1f}" for value in energy_list),
            "half_life_s": half_life_s,
        }
        row.update(metrics)
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -_safe_float(item.get("activity_Bq")),
            str(item.get("isotope")),
        )
    )
    return rows


def _aggregate_validation_analysis_rows(
    analysis_payloads: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_rows: List[Dict[str, Any]] = []
    isotope_rows: List[Dict[str, Any]] = []
    for payload in analysis_payloads:
        sample_id = str(payload.get("sample_id") or "unknown")
        timing = payload.get("timing", {}) or {}
        isotopes = payload.get("isotopes", {}) or {}
        peaks = payload.get("peaks", []) or []
        total_counts = float(
            sum(
                _safe_float(item.get("gross_counts"))
                for item in peaks
                if isinstance(item, dict)
            )
        )
        total_activity = 0.0
        total_unc_sq = 0.0
        total_radioactive_mass = 0.0
        for isotope, values in isotopes.items():
            if not isinstance(values, dict):
                continue
            activity = _safe_float(values.get("activity_eoi_bq"), default=float("nan"))
            if not np.isfinite(activity) or activity <= 0.0:
                activity = _safe_float(values.get("activity_bq"))
                unc = _safe_float(values.get("activity_unc_bq"))
                radioactive_mass = _safe_float(values.get("radioactive_mass_g"))
                specific_activity = _safe_float(
                    values.get("specific_activity_Bq_g"), default=float("nan")
                )
            else:
                unc = _safe_float(values.get("activity_eoi_unc_bq"))
                radioactive_mass = _safe_float(values.get("eoi_radioactive_mass_g"))
                specific_activity = _safe_float(
                    values.get("eoi_specific_activity_Bq_g"), default=float("nan")
                )
            if radioactive_mass <= 0.0:
                radioactive_mass = _safe_float(values.get("radioactive_mass_g"))
            if not np.isfinite(specific_activity) or specific_activity <= 0.0:
                specific_activity = _safe_float(
                    values.get("specific_activity_Bq_g"), default=float("nan")
                )
            total_activity += activity
            total_unc_sq += unc * unc
            total_radioactive_mass += radioactive_mass
            isotope_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_group": payload.get("sample_group"),
                    "measurement_time": timing.get("measurement_time"),
                    "isotope": isotope,
                    "activity_Bq": activity,
                    "activity_unc_Bq": unc,
                    "radioactive_mass_g": radioactive_mass,
                    "sample_mass_g": values.get("sample_mass_g"),
                    "specific_activity_Bq_g": (
                        None
                        if not np.isfinite(specific_activity)
                        else specific_activity
                    ),
                    "n_peaks": values.get("n_peaks"),
                }
            )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "sample_group": payload.get("sample_group"),
                "measurement_time": timing.get("measurement_time"),
                "n_detected_peaks": payload.get("n_detected_peaks"),
                "n_unidentified_peaks": payload.get("n_unidentified_peaks"),
                "total_counts": total_counts,
                "total_activity_Bq": total_activity,
                "total_activity_unc_Bq": float(np.sqrt(max(total_unc_sq, 0.0))),
                "total_radioactive_mass_g": total_radioactive_mass,
                "isotope_count": sum(
                    1 for value in isotopes.values() if isinstance(value, dict)
                ),
            }
        )
    sample_rows.sort(key=lambda item: str(item.get("sample_id")))
    isotope_rows.sort(
        key=lambda item: (
            str(item.get("sample_id")),
            -_safe_float(item.get("activity_Bq")),
            str(item.get("isotope")),
        )
    )
    return sample_rows, isotope_rows


def _build_standard_report_text(
    *,
    output_path: Path,
    inputs: Dict[str, Any],
    summary: Dict[str, Any],
    spectrum_payload: Optional[Dict[str, Any]],
    peak_report: Optional[Dict[str, Any]],
    line_payload: Optional[Dict[str, Any]],
    isotope_rows: List[Dict[str, Any]],
    rates_payload: Optional[Dict[str, Any]],
    masking_payload: Optional[Dict[str, Any]],
    masking_line_rows: List[Dict[str, Any]],
    masking_isotope_rows: List[Dict[str, Any]],
    masking_recommendations: List[Dict[str, Any]],
    optimization_payload: Optional[Dict[str, Any]],
    optimization_recommendation: Optional[Dict[str, Any]],
    optimization_top_rows: List[Dict[str, Any]],
    unfold_payload: Optional[Dict[str, Any]],
    validation_payload: Optional[Dict[str, Any]],
    validation_results_summary: Optional[Dict[str, Any]],
    validation_sample_rows: List[Dict[str, Any]],
    validation_isotope_rows: List[Dict[str, Any]],
) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    sections: List[str] = [
        "FluxForge Standard Activation Report",
        "=" * 36,
        f"Generated: {generated_at}",
        f"Report bundle: {output_path}",
    ]

    sections.append(
        _format_key_value_section(
            "Inputs",
            [(key, value) for key, value in inputs.items()],
        )
    )

    if spectrum_payload is not None:
        spectrum = spectrum_payload.get("spectrum", {}) or {}
        counts = spectrum.get("counts", []) or []
        sections.append(
            _format_key_value_section(
                "Measurement Summary",
                [
                    ("Spectrum ID", spectrum.get("spectrum_id") or "unknown"),
                    ("Start time", spectrum.get("start_time") or "unknown"),
                    ("Live time (s)", _safe_float(spectrum.get("live_time"))),
                    ("Real time (s)", _safe_float(spectrum.get("real_time"))),
                    (
                        "Total counts",
                        float(sum(_safe_float(value) for value in counts)),
                    ),
                    ("Channel count", len(counts)),
                ],
            )
        )

    if peak_report is not None:
        peaks = [
            item
            for item in (peak_report.get("peaks", []) or [])
            if isinstance(item, dict)
        ]
        strongest_peak = max(
            peaks,
            key=lambda item: _safe_float(
                item.get("area") or item.get("raw_counts") or item.get("amplitude")
            ),
            default=None,
        )
        sections.append(
            _format_key_value_section(
                "Peak Summary",
                [
                    ("Peak count", len(peaks)),
                    (
                        "Total net peak counts",
                        float(
                            sum(
                                _safe_float(
                                    item.get("area")
                                    or item.get("raw_counts")
                                    or item.get("amplitude")
                                )
                                for item in peaks
                            )
                        ),
                    ),
                    (
                        "Strongest peak",
                        (
                            "n/a"
                            if strongest_peak is None
                            else f"{_safe_float(strongest_peak.get('energy_keV')):.1f} keV ({_safe_float(strongest_peak.get('area') or strongest_peak.get('raw_counts') or strongest_peak.get('amplitude')):.6g} counts)"
                        ),
                    ),
                ],
            )
        )

    sections.append(
        _format_key_value_section(
            "Aggregate Summary",
            sorted(summary.items(), key=lambda item: item[0]),
        )
    )

    sections.append(
        _format_table_section(
            "Isotope Activity Summary",
            [
                "Isotope",
                "Lines",
                "Activity (Bq)",
                "σ (Bq)",
                "Net counts",
                "Rad. mass (g)",
                "Specific act. (Bq/g)",
                "Energies (keV)",
            ],
            [
                [
                    row.get("isotope"),
                    row.get("n_lines"),
                    row.get("activity_Bq"),
                    row.get("activity_unc_Bq"),
                    row.get("total_net_counts"),
                    row.get("radioactive_mass_g"),
                    row.get("specific_activity_Bq_g"),
                    row.get("energies_keV"),
                ]
                for row in isotope_rows
            ],
        )
    )

    if line_payload is not None:
        lines = [
            item
            for item in (line_payload.get("lines", []) or [])
            if isinstance(item, dict)
        ]
        sections.append(
            _format_table_section(
                "Gamma Line Detail",
                [
                    "Isotope",
                    "Energy (keV)",
                    "Net counts",
                    "Activity (Bq)",
                    "σ (Bq)",
                    "Sample mass (g)",
                    "Specific act. (Bq/g)",
                ],
                [
                    [
                        line.get("isotope"),
                        _safe_float(line.get("energy_keV")),
                        _safe_float(line.get("net_counts")),
                        _safe_float(line.get("activity_Bq")),
                        _safe_float(line.get("activity_unc_Bq")),
                        _safe_float(line.get("sample_mass_g"), default=float("nan")),
                        _safe_float(
                            line.get("specific_activity_Bq_g"), default=float("nan")
                        ),
                    ]
                    for line in lines
                ],
            )
        )

    if rates_payload is not None:
        rates = [
            item
            for item in (rates_payload.get("rates", []) or [])
            if isinstance(item, dict)
        ]
        sections.append(
            _format_table_section(
                "Reaction Rate Summary",
                ["Reaction", "Rate (reactions/s)", "σ (reactions/s)", "Half-life (s)"],
                [
                    [
                        row.get("reaction_id"),
                        _safe_float(row.get("rate")),
                        _safe_float(row.get("uncertainty")),
                        _safe_float(row.get("half_life_s")),
                    ]
                    for row in rates
                ],
            )
        )

    if masking_payload is not None:
        sections.append(
            _format_key_value_section(
                "Masking Review Summary",
                [
                    (
                        "Energy window (keV)",
                        masking_payload.get("energy_window_keV"),
                    ),
                    (
                        "Masking interactions",
                        len(masking_line_rows),
                    ),
                    (
                        "Masking isotope entries",
                        len(masking_isotope_rows),
                    ),
                    (
                        "Alternate-line recommendations",
                        len(masking_recommendations),
                    ),
                ],
            )
        )
        sections.append(
            _format_table_section(
                "Masking Isotope Ranking",
                [
                    "Rank",
                    "Masking isotope",
                    "Masking score",
                    "Interference counts",
                    "Continuum counts",
                    "Target lines",
                ],
                [
                    [
                        row.get("rank"),
                        row.get("masking_nuclide"),
                        row.get("total_masking_score"),
                        row.get("total_interference_counts"),
                        row.get("total_continuum_counts"),
                        row.get("target_line_count"),
                    ]
                    for row in masking_isotope_rows
                ],
            )
        )
        sections.append(
            _format_table_section(
                "Alternate-Line Recommendations",
                [
                    "Rank",
                    "Nuclide",
                    "Preferred line (keV)",
                    "Preferred masking score",
                    "Strongest line (keV)",
                    "Guidance",
                ],
                [
                    [
                        row.get("rank"),
                        row.get("nuclide"),
                        row.get("preferred_line_energy_keV"),
                        row.get("preferred_masking_score"),
                        row.get("strongest_line_energy_keV"),
                        row.get("guidance"),
                    ]
                    for row in masking_recommendations
                ],
            )
        )

    if optimization_payload is not None:
        recommendation = optimization_recommendation or {}
        sections.append(
            _format_key_value_section(
                "Optimization Recommendation",
                [
                    ("Objective", recommendation.get("objective")),
                    ("Best candidate", recommendation.get("recommended_label")),
                    (
                        "Irradiation time (s)",
                        recommendation.get("recommended_irradiation_time_s"),
                    ),
                    (
                        "Cooldown time (s)",
                        recommendation.get("recommended_cooldown_time_s"),
                    ),
                    (
                        "Count time (s)",
                        recommendation.get("recommended_count_time_s"),
                    ),
                    (
                        "Objective score",
                        recommendation.get("recommended_objective_score"),
                    ),
                    (
                        "Mask isotope",
                        recommendation.get("recommended_mask_isotope"),
                    ),
                    (
                        "Isotope of interest",
                        recommendation.get("recommended_isotope_of_interest"),
                    ),
                    (
                        "Expected dose (uSv)",
                        recommendation.get("recommended_expected_dose_uSv"),
                    ),
                ],
            )
        )
        sections.append(
            _format_table_section(
                "Top Optimization Schedules",
                [
                    "Rank",
                    "Label",
                    "Irradiation (s)",
                    "Cooldown (s)",
                    "Count (s)",
                    "Score",
                ],
                [
                    [
                        row.get("rank"),
                        row.get("label"),
                        row.get("irradiation_time_s"),
                        row.get("cooldown_time_s"),
                        row.get("count_time_s"),
                        row.get("objective_score"),
                    ]
                    for row in optimization_top_rows
                ],
            )
        )

    if unfold_payload is not None:
        diagnostics = unfold_payload.get("diagnostics", {}) or {}
        energy_edges = (
            unfold_payload.get("boundaries_eV")
            or unfold_payload.get("energy_edges_eV")
            or []
        )
        sections.append(
            _format_key_value_section(
                "Flux Unfolding Summary",
                [
                    ("Method", unfold_payload.get("method") or "unknown"),
                    ("Energy groups", max(len(energy_edges) - 1, 0)),
                    (
                        "Integral flux",
                        float(
                            sum(
                                _safe_float(value)
                                for value in unfold_payload.get("flux", []) or []
                            )
                        ),
                    ),
                    (
                        "Chi2",
                        unfold_payload.get("chi2", unfold_payload.get("chi_squared")),
                    ),
                    (
                        "Iterations",
                        diagnostics.get("iterations", unfold_payload.get("iterations")),
                    ),
                    (
                        "Converged",
                        diagnostics.get("converged", unfold_payload.get("converged")),
                    ),
                ],
            )
        )

    if validation_payload is not None:
        metrics = validation_payload.get("metrics", {}) or {}
        sections.append(
            _format_key_value_section(
                "Validation Summary",
                [(key.upper(), value) for key, value in metrics.items()],
            )
        )

    if validation_results_summary is not None:
        sections.append(
            _format_key_value_section(
                "RAFM Validation Run Summary",
                [
                    (
                        "Overall passed",
                        validation_results_summary.get("overall_passed"),
                    ),
                    (
                        "Raw spectra analyzed",
                        validation_results_summary.get("n_raw_analyzed"),
                    ),
                    (
                        "Matched raw/QG pairs",
                        validation_results_summary.get("n_matched_pairs"),
                    ),
                    (
                        "Unmatched raw",
                        validation_results_summary.get("n_unmatched_raw"),
                    ),
                    ("Unmatched QG", validation_results_summary.get("n_unmatched_qg")),
                    (
                        "QG consistency flags",
                        validation_results_summary.get("qg_internal_consistency_flags"),
                    ),
                    (
                        "FluxForge consistency flags",
                        validation_results_summary.get(
                            "fluxforge_line_consistency_flags"
                        ),
                    ),
                    (
                        "Measurement QC flags",
                        validation_results_summary.get("measurement_qc_flags"),
                    ),
                ],
            )
        )

    if validation_sample_rows:
        sections.append(
            _format_table_section(
                "Validation Sample Summary",
                [
                    "Sample",
                    "Group",
                    "Measurement time",
                    "Peaks",
                    "Total counts",
                    "Activity (Bq)",
                    "σ (Bq)",
                    "Rad. mass (g)",
                ],
                [
                    [
                        row.get("sample_id"),
                        row.get("sample_group"),
                        row.get("measurement_time"),
                        row.get("n_detected_peaks"),
                        row.get("total_counts"),
                        row.get("total_activity_Bq"),
                        row.get("total_activity_unc_Bq"),
                        row.get("total_radioactive_mass_g"),
                    ]
                    for row in validation_sample_rows
                ],
            )
        )

    if validation_isotope_rows:
        sections.append(
            _format_table_section(
                "Validation Sample-Isotope Activity",
                [
                    "Sample",
                    "Isotope",
                    "Activity (Bq)",
                    "σ (Bq)",
                    "Rad. mass (g)",
                    "Specific act. (Bq/g)",
                    "Peaks",
                ],
                [
                    [
                        row.get("sample_id"),
                        row.get("isotope"),
                        row.get("activity_Bq"),
                        row.get("activity_unc_Bq"),
                        row.get("radioactive_mass_g"),
                        row.get("specific_activity_Bq_g"),
                        row.get("n_peaks"),
                    ]
                    for row in validation_isotope_rows
                ],
            )
        )

    return "\n\n".join(sections) + "\n"


def cmd_report(args: argparse.Namespace) -> None:
    inputs = {}
    summary = {}
    spectrum_payload: Optional[Dict[str, Any]] = None
    peak_report: Optional[Dict[str, Any]] = None
    line_payload: Optional[Dict[str, Any]] = None
    rates_payload: Optional[Dict[str, Any]] = None
    masking_payload: Optional[Dict[str, Any]] = None
    masking_line_rows: List[Dict[str, Any]] = []
    masking_isotope_rows: List[Dict[str, Any]] = []
    masking_recommendation_rows: List[Dict[str, Any]] = []
    optimization_payload: Optional[Dict[str, Any]] = None
    optimization_recommendation: Optional[Dict[str, Any]] = None
    optimization_top_rows: List[Dict[str, Any]] = []
    unfold_payload: Optional[Dict[str, Any]] = None
    validation_payload: Optional[Dict[str, Any]] = None
    isotope_rows: List[Dict[str, Any]] = []
    validation_results_summary: Optional[Dict[str, Any]] = None
    validation_sample_rows: List[Dict[str, Any]] = []
    validation_isotope_rows: List[Dict[str, Any]] = []

    validation_results_root = getattr(args, "validation_results_root", None)
    if validation_results_root:
        discovered = discover_validation_report_inputs(validation_results_root)
        inputs["validation_results_root"] = str(validation_results_root)
        summary_json = discovered.get("validation_summary_json")
        if summary_json is not None:
            validation_results_summary = _load_json(summary_json)
            summary["validation_overall_passed"] = bool(
                validation_results_summary.get("overall_passed", False)
            )
            summary["validation_n_raw_analyzed"] = int(
                validation_results_summary.get("n_raw_analyzed", 0) or 0
            )
            summary["validation_n_matched_pairs"] = int(
                validation_results_summary.get("n_matched_pairs", 0) or 0
            )
            summary["validation_qg_internal_consistency_flags"] = int(
                validation_results_summary.get("qg_internal_consistency_flags", 0) or 0
            )
            summary["validation_fluxforge_line_consistency_flags"] = int(
                validation_results_summary.get("fluxforge_line_consistency_flags", 0)
                or 0
            )
            summary["validation_measurement_qc_flags"] = int(
                validation_results_summary.get("measurement_qc_flags", 0) or 0
            )
        analysis_dir = discovered.get("analysis_json_dir")
        if analysis_dir is not None:
            analysis_payloads = [
                _load_json(path) for path in sorted(analysis_dir.glob("*.json"))
            ]
            validation_sample_rows, validation_isotope_rows = (
                _aggregate_validation_analysis_rows(analysis_payloads)
            )
            summary["validation_sample_count"] = len(validation_sample_rows)
            summary["validation_isotope_rows"] = len(validation_isotope_rows)
            summary["validation_total_activity_Bq"] = float(
                sum(
                    _safe_float(row.get("activity_Bq"))
                    for row in validation_isotope_rows
                )
            )
            summary["validation_total_radioactive_mass_g"] = float(
                sum(
                    _safe_float(row.get("radioactive_mass_g"))
                    for row in validation_isotope_rows
                )
            )
        if (
            unfold_payload is None
            and discovered.get("preferred_unfold_file") is not None
        ):
            preferred_unfold = discovered["preferred_unfold_file"]
            assert preferred_unfold is not None
            inputs["unfold_file"] = str(preferred_unfold)
            unfold_payload = read_unfold_result(preferred_unfold)
            summary["chi2"] = unfold_payload.get(
                "chi2", unfold_payload.get("chi_squared")
            )
            summary["integral_flux"] = float(
                sum(float(value) for value in unfold_payload.get("flux", []) or [])
            )

    if args.spectrum_file:
        inputs["spectrum_file"] = str(args.spectrum_file)
        spectrum_payload = read_spectrum_file(args.spectrum_file)
        if args.validate:
            validate_or_raise(spectrum_payload)
        spectrum = spectrum_payload.get("spectrum", {}) or {}
        counts = spectrum.get("counts", []) or []
        summary["spectrum_id"] = spectrum.get("spectrum_id") or ""
        summary["measurement_start_time"] = spectrum.get("start_time")
        summary["live_time_s"] = _safe_float(spectrum.get("live_time"))
        summary["real_time_s"] = _safe_float(spectrum.get("real_time"))
        summary["total_counts"] = float(sum(_safe_float(value) for value in counts))
    if args.peaks_file:
        inputs["peaks_file"] = str(args.peaks_file)
        peak_report = read_peak_report(args.peaks_file)
        if args.validate:
            validate_or_raise(peak_report)
        peak_items = peak_report.get("peaks", []) or []
        peaks = [item for item in peak_items if isinstance(item, dict)]
        summary["peak_count"] = len(peak_items)
        summary["total_net_peak_counts"] = float(
            sum(
                _safe_float(
                    item.get("area") or item.get("raw_counts") or item.get("amplitude")
                )
                for item in peaks
            )
        )
    if args.lines_file:
        inputs["lines_file"] = str(args.lines_file)
        line_payload = read_line_activities(args.lines_file)
        if args.validate:
            validate_or_raise(line_payload)
        lines = [
            item
            for item in (line_payload.get("lines", []) or [])
            if isinstance(item, dict)
        ]
        summary["line_count"] = len(lines)
        isotope_rows = _aggregate_isotope_activity_rows(lines)
        summary["isotope_count"] = len(isotope_rows)
        if lines:
            activities = [float(item.get("activity_Bq", 0.0) or 0.0) for item in lines]
            radioactive_mass = [
                float(item.get("radioactive_mass_g", 0.0) or 0.0) for item in lines
            ]
            specific_activity = [
                float(item.get("specific_activity_Bq_g", 0.0) or 0.0) for item in lines
            ]
            intrinsic_specific_activity = [
                float(item.get("radioisotope_specific_activity_Bq_g", 0.0) or 0.0)
                for item in lines
            ]
            summary["total_activity_Bq"] = float(sum(activities))
            isotope_mass_total = (
                float(
                    sum(
                        _safe_float(item.get("radioactive_mass_g"))
                        for item in isotope_rows
                    )
                )
                if isotope_rows
                else 0.0
            )
            summary["total_radioactive_mass_g"] = (
                isotope_mass_total
                if isotope_mass_total > 0.0
                else float(sum(radioactive_mass))
            )
            if any(value > 0.0 for value in specific_activity):
                summary["max_specific_activity_Bq_g"] = float(max(specific_activity))
            if any(value > 0.0 for value in intrinsic_specific_activity):
                summary["max_radioisotope_specific_activity_Bq_g"] = float(
                    max(intrinsic_specific_activity)
                )
        else:
            summary["total_activity_Bq"] = 0.0
            summary["total_radioactive_mass_g"] = 0.0
    if args.rates_file:
        inputs["rates_file"] = str(args.rates_file)
        rates_payload = read_reaction_rates(args.rates_file)
        if args.validate:
            validate_or_raise(rates_payload)
        rates = [
            item
            for item in (rates_payload.get("rates", []) or [])
            if isinstance(item, dict)
        ]
        summary["rate_count"] = len(rates)
        if rates:
            summary["total_rate_reactions_s"] = float(
                sum(float(item.get("rate", 0.0) or 0.0) for item in rates)
            )
    masking_file = getattr(args, "masking_file", None)
    if masking_file:
        masking_file_path = Path(masking_file)
        inputs["masking_file"] = str(masking_file_path)
        masking_payload = _load_json(masking_file_path)
        masking_line_rows = [
            item
            for item in (masking_payload.get("line_masking_results", []) or [])
            if isinstance(item, dict)
        ]
        masking_isotope_rows = [
            item
            for item in (masking_payload.get("masking_isotope_ranking", []) or [])
            if isinstance(item, dict)
        ]
        masking_recommendation_rows = [
            item
            for item in (masking_payload.get("alternate_line_recommendations", []) or [])
            if isinstance(item, dict)
        ]
        summary["masking_line_interactions"] = len(masking_line_rows)
        summary["masking_isotope_count"] = len(masking_isotope_rows)
        if masking_isotope_rows:
            summary["dominant_masking_isotope"] = masking_isotope_rows[0].get(
                "masking_nuclide"
            )
        if masking_recommendation_rows:
            summary["alternate_line_recommendations"] = len(
                masking_recommendation_rows
            )
    optimization_file = getattr(args, "optimization_file", None)
    if optimization_file:
        optimization_file_path = Path(optimization_file)
        inputs["optimization_file"] = str(optimization_file_path)
        optimization_payload = _load_json(optimization_file_path)
        optimization_recommendation, optimization_top_rows = (
            _recommendation_from_optimization_payload(
                optimization_payload,
                optimization_file=optimization_file_path,
            )
        )
        summary["optimization_objective"] = optimization_recommendation.get(
            "objective"
        )
        summary["recommended_schedule_label"] = optimization_recommendation.get(
            "recommended_label"
        )
        summary["recommended_irradiation_time_s"] = optimization_recommendation.get(
            "recommended_irradiation_time_s"
        )
        summary["recommended_cooldown_time_s"] = optimization_recommendation.get(
            "recommended_cooldown_time_s"
        )
        summary["recommended_count_time_s"] = optimization_recommendation.get(
            "recommended_count_time_s"
        )
        summary["recommended_mask_isotope"] = optimization_recommendation.get(
            "recommended_mask_isotope"
        )
        summary["recommended_isotope_of_interest"] = (
            optimization_recommendation.get("recommended_isotope_of_interest")
        )
        if "recommended_expected_dose_uSv" in optimization_recommendation:
            summary["recommended_expected_dose_uSv"] = optimization_recommendation.get(
                "recommended_expected_dose_uSv"
            )
    if args.unfold_file:
        inputs["unfold_file"] = str(args.unfold_file)
        unfold_payload = read_unfold_result(args.unfold_file)
        if args.validate:
            validate_or_raise(unfold_payload)
        summary["chi2"] = unfold_payload.get("chi2", unfold_payload.get("chi_squared"))
        summary["integral_flux"] = float(
            sum(float(value) for value in unfold_payload.get("flux", []) or [])
        )
    if args.validation_file:
        inputs["validation_file"] = str(args.validation_file)
        validation_payload = read_validation_bundle(args.validation_file)
        if args.validate:
            validate_or_raise(validation_payload)
        summary["mae"] = validation_payload.get("metrics", {}).get("mae")
        summary["rmse"] = validation_payload.get("metrics", {}).get("rmse")
        summary["validation_chi2"] = validation_payload.get("metrics", {}).get("chi2")

    _ensure_parent_dir(args.output)
    text_report_path = args.output.with_suffix(".txt")
    tables_dir = args.output.with_name(f"{args.output.stem}_tables")
    table_items: Dict[str, Dict[str, Any]] = {}

    if line_payload is not None:
        line_rows = [
            item
            for item in (line_payload.get("lines", []) or [])
            if isinstance(item, dict)
        ]
        line_table = _write_csv_table(
            tables_dir / "line_activity_detail.csv", line_rows
        )
        isotope_table = _write_csv_table(
            tables_dir / "isotope_activity_summary.csv", isotope_rows
        )
        if line_table is not None:
            table_items["line_activity_detail"] = line_table
        if isotope_table is not None:
            table_items["isotope_activity_summary"] = isotope_table
    if validation_sample_rows:
        sample_table = _write_csv_table(
            tables_dir / "validation_sample_summary.csv", validation_sample_rows
        )
        if sample_table is not None:
            table_items["validation_sample_summary"] = sample_table
    if validation_isotope_rows:
        validation_isotope_table = _write_csv_table(
            tables_dir / "validation_isotope_activity.csv", validation_isotope_rows
        )
        if validation_isotope_table is not None:
            table_items["validation_isotope_activity"] = validation_isotope_table
    if rates_payload is not None:
        rate_rows = [
            item
            for item in (rates_payload.get("rates", []) or [])
            if isinstance(item, dict)
        ]
        rate_table = _write_csv_table(
            tables_dir / "reaction_rates_summary.csv", rate_rows
        )
        if rate_table is not None:
            table_items["reaction_rates_summary"] = rate_table
    if masking_payload is not None:
        masking_line_table = _write_csv_table(
            tables_dir / "masking_line_results.csv", masking_line_rows
        )
        masking_isotope_table = _write_csv_table(
            tables_dir / "masking_isotope_ranking.csv", masking_isotope_rows
        )
        masking_recommendation_table = _write_csv_table(
            tables_dir / "alternate_line_recommendations.csv",
            masking_recommendation_rows,
        )
        if masking_line_table is not None:
            table_items["masking_line_results"] = masking_line_table
        if masking_isotope_table is not None:
            table_items["masking_isotope_ranking"] = masking_isotope_table
        if masking_recommendation_table is not None:
            table_items[
                "alternate_line_recommendations"
            ] = masking_recommendation_table
    if optimization_payload is not None:
        recommendation_rows = []
        if optimization_recommendation:
            recommendation_rows = [optimization_recommendation]
        recommendation_table = _write_csv_table(
            tables_dir / "optimization_recommendation.csv",
            recommendation_rows,
        )
        top_table = _write_csv_table(
            tables_dir / "optimization_top_candidates.csv",
            optimization_top_rows,
        )
        if recommendation_table is not None:
            table_items["optimization_recommendation"] = recommendation_table
        if top_table is not None:
            table_items["optimization_top_candidates"] = top_table
    if unfold_payload is not None:
        flux_rows = []
        boundaries = list(
            unfold_payload.get("boundaries_eV")
            or unfold_payload.get("energy_edges_eV")
            or []
        )
        flux = list(unfold_payload.get("flux", []) or [])
        covariance = list(unfold_payload.get("covariance", []) or [])
        for idx, value in enumerate(flux):
            variance = (
                _safe_float(covariance[idx][idx])
                if idx < len(covariance) and idx < len(covariance[idx])
                else 0.0
            )
            flux_rows.append(
                {
                    "group_index": idx + 1,
                    "lower_eV": (
                        _safe_float(boundaries[idx]) if idx < len(boundaries) else None
                    ),
                    "upper_eV": (
                        _safe_float(boundaries[idx + 1])
                        if idx + 1 < len(boundaries)
                        else None
                    ),
                    "flux": _safe_float(value),
                    "flux_uncertainty": float(np.sqrt(max(variance, 0.0))),
                }
            )
        flux_table = _write_csv_table(tables_dir / "unfold_flux_groups.csv", flux_rows)
        if flux_table is not None:
            table_items["unfold_flux_groups"] = flux_table

    report_text = _build_standard_report_text(
        output_path=args.output,
        inputs=inputs,
        summary=summary,
        spectrum_payload=spectrum_payload,
        peak_report=peak_report,
        line_payload=line_payload,
        isotope_rows=isotope_rows,
        rates_payload=rates_payload,
        masking_payload=masking_payload,
        masking_line_rows=masking_line_rows,
        masking_isotope_rows=masking_isotope_rows,
        masking_recommendations=masking_recommendation_rows,
        optimization_payload=optimization_payload,
        optimization_recommendation=optimization_recommendation,
        optimization_top_rows=optimization_top_rows,
        unfold_payload=unfold_payload,
        validation_payload=validation_payload,
        validation_results_summary=validation_results_summary,
        validation_sample_rows=validation_sample_rows,
        validation_isotope_rows=validation_isotope_rows,
    )
    text_report_path.write_text(report_text, encoding="utf-8")

    text_report = {
        "path": text_report_path.name,
        "format": "text/plain",
    }
    tables = (
        {"directory": tables_dir.name, "items": table_items} if table_items else None
    )

    write_report_bundle(
        args.output,
        summary=summary,
        inputs=inputs or None,
        tables=tables,
        text_report=text_report,
    )
    print(f"Wrote report bundle to {args.output}")
    print(f"Wrote standardized text report to {text_report_path}")


def cmd_k0_normalize(args: argparse.Namespace) -> None:
    peak_payload = read_peak_report(args.peaks_file)
    if args.validate:
        validate_or_raise(peak_payload)
    spectrum_payload = (
        read_spectrum_file(args.spectrum_file) if args.spectrum_file else None
    )
    if spectrum_payload is not None and args.validate:
        validate_or_raise(spectrum_payload)
    detector_payload = (
        read_detector_characterization(args.detector_characterization_file)
        if args.detector_characterization_file
        else None
    )
    if detector_payload is not None and args.validate:
        validate_or_raise(detector_payload)

    observations = peak_report_to_observations(
        peak_payload,
        spectrum_payload=spectrum_payload,
        detector_payload=detector_payload,
        detector_id=args.detector_id,
        geometry_id=args.geometry_id,
        irradiation_time_s=args.irradiation_time_s,
        decay_time_s=args.decay_time_s,
        counting_time_s=args.counting_time_s,
        import_format=args.import_format,
        project_id=args.project_id,
        sample_id=args.sample_id,
        irradiation_id=args.irradiation_id,
        measurement_id=args.measurement_id,
        expert_override=args.expert_override,
        allow_advanced=args.allow_advanced_lines,
    )
    summary = {
        "observation_count": len(observations),
        "accepted_count": sum(1 for item in observations if item.eligibility_accepted),
        "rejected_count": sum(
            1 for item in observations if not item.eligibility_accepted
        ),
        "project_id": args.project_id,
        "sample_id": args.sample_id,
        "irradiation_id": args.irradiation_id,
        "measurement_id": args.measurement_id,
    }
    spectrum_id = str(
        peak_payload.get("spectrum_id")
        or (spectrum_payload or {}).get("spectrum", {}).get("spectrum_id")
        or ""
    )
    detector_id = str(
        args.detector_id
        or (spectrum_payload or {}).get("spectrum", {}).get("detector_id")
        or ""
    )
    geometry_id = str(
        args.geometry_id
        or (spectrum_payload or {})
        .get("spectrum", {})
        .get("metadata", {})
        .get("geometry_id")
        or ""
    )
    write_peak_observation_bundle(
        args.output,
        spectrum_id=spectrum_id,
        detector_id=detector_id,
        geometry_id=geometry_id,
        observations=[item.to_dict() for item in observations],
        summary=summary,
        capability_flags=K0_CAPABILITY_FLAGS,
        source_path=args.peaks_file,
    )
    print(f"Wrote k0 peak observations to {args.output}")


def cmd_k0_detector(args: argparse.Namespace) -> None:
    rows = _load_structured_rows(args.points_file)
    if not isinstance(rows, list):
        raise ValueError("Detector characterization input must be a JSON/CSV row list.")
    artifact = build_detector_characterization(
        rows,
        detector_id=args.detector_id,
        reference_position_mm=args.reference_position_mm,
        degree=args.degree,
        peak_to_total_ratio=args.peak_to_total_ratio,
        coincidence_mode=args.coincidence_mode,
    )
    write_detector_characterization(
        args.output,
        detector_id=artifact["detector_id"],
        reference_position_mm=artifact["reference_position_mm"],
        characterized_positions_mm=artifact["characterized_positions_mm"],
        calibration_points=artifact["calibration_points"],
        efficiency_model=artifact["efficiency_model"],
        geometry_conversions=artifact.get("geometry_conversions"),
        peak_to_total_model=artifact.get("peak_to_total_model"),
        coincidence_model=artifact.get("coincidence_model"),
        capability_flags=artifact.get("capability_flags"),
        source_path=args.points_file,
    )
    print(f"Wrote detector characterization to {args.output}")


def cmd_k0_facility(args: argparse.Namespace) -> None:
    payload = _load_structured_rows(args.input)
    if not isinstance(payload, dict):
        raise ValueError("Facility characterization input must be a JSON object.")
    artifact = build_facility_characterization(payload)
    write_facility_characterization(
        args.output,
        facility_id=artifact["facility_id"],
        method=artifact["method"],
        monitor_definitions=artifact.get("monitor_definitions", []),
        irradiation=artifact.get("irradiation"),
        flux_parameters=artifact["flux_parameters"],
        temperature=artifact.get("temperature"),
        gradients=artifact.get("gradients"),
        fast_flux=artifact.get("fast_flux"),
        capability_flags=artifact.get("capability_flags"),
        source_path=args.input,
    )
    print(f"Wrote facility characterization to {args.output}")


def cmd_k0_analyze(args: argparse.Namespace) -> None:
    observation_payload = read_peak_observation_bundle(args.observations_file)
    facility_payload = read_facility_characterization(args.facility_file)
    if args.validate:
        validate_or_raise(observation_payload)
        validate_or_raise(facility_payload)
    standard_library, auxiliary_library = resolve_governed_libraries(
        k0_library_file=args.k0_library_file,
        auxiliary_library_file=args.auxiliary_library_file,
    )
    bundle = analyze_k0_observations(
        observation_payload,
        facility_payload,
        sample_mass_g=args.sample_mass_g,
        reference_isotope=args.reference_isotope,
        reference_mass_g=args.reference_mass_g,
        standard_library=standard_library,
        auxiliary_library=auxiliary_library,
    )
    write_k0_analysis_bundle(
        args.output,
        summary=bundle["summary"],
        line_results=bundle["line_results"],
        element_results=bundle["element_results"],
        rejected_observations=bundle.get("rejected_observations"),
        applied_corrections=bundle.get("applied_corrections"),
        recognized_but_not_applied=bundle.get("recognized_but_not_applied"),
        user_supplied_corrections=bundle.get("user_supplied_corrections"),
        default_assumptions=bundle.get("default_assumptions"),
        capability_flags=bundle.get("capability_flags"),
        libraries=bundle.get("libraries"),
        inputs=bundle.get("inputs"),
        source_path=args.observations_file,
    )
    print(f"Wrote k0 analysis bundle to {args.output}")


def cmd_k0_aggregate(args: argparse.Namespace) -> None:
    analysis_payloads = [read_k0_analysis_bundle(path) for path in args.analysis_files]
    if args.validate:
        for payload in analysis_payloads:
            validate_or_raise(payload)
    bundle = aggregate_k0_analysis_bundles(analysis_payloads)
    write_k0_aggregation_bundle(
        args.output,
        summary=bundle["summary"],
        aggregated_results=bundle["aggregated_results"],
        irradiation_summaries=bundle.get("irradiation_summaries"),
        inputs={"analysis_files": [str(path) for path in args.analysis_files]},
        source_path=args.analysis_files[0] if args.analysis_files else None,
    )
    print(f"Wrote k0 aggregation bundle to {args.output}")


def cmd_k0_qaqc(args: argparse.Namespace) -> None:
    plan = _load_structured_rows(args.plan_file)
    if not isinstance(plan, dict):
        raise ValueError("k0 QA/QC plan input must be a JSON object.")
    records = []
    for row in plan.get("records") or []:
        analysis_file = Path(row["analysis_file"])
        analysis_payload = read_k0_analysis_bundle(analysis_file)
        if args.validate:
            validate_or_raise(analysis_payload)
        records.append({**dict(row), "analysis_payload": analysis_payload})
    bundle = evaluate_k0_qaqc(
        records,
        default_blank_limit_ug_g=float(
            plan.get("default_blank_limit_ug_g", 0.0) or 0.0
        ),
    )
    write_k0_qaqc_bundle(
        args.output,
        summary=bundle["summary"],
        records=bundle["records"],
        inputs={"plan_file": str(args.plan_file)},
        source_path=args.plan_file,
    )
    print(f"Wrote k0 QA/QC bundle to {args.output}")


def cmd_k0_report(args: argparse.Namespace) -> None:
    analysis_payload = read_k0_analysis_bundle(args.analysis_file)
    aggregation_payload = (
        read_k0_aggregation_bundle(args.aggregation_file)
        if args.aggregation_file
        else None
    )
    qaqc_payload = read_k0_qaqc_bundle(args.qaqc_file) if args.qaqc_file else None
    if args.validate:
        validate_or_raise(analysis_payload)
        if aggregation_payload is not None:
            validate_or_raise(aggregation_payload)
        if qaqc_payload is not None:
            validate_or_raise(qaqc_payload)

    report_payload = build_k0_report_payload(
        analysis_payload,
        aggregation_payload=aggregation_payload,
        qaqc_payload=qaqc_payload,
    )
    tables_dir = args.output.with_suffix("").with_name(args.output.stem + "_tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_items = {}
    for name, rows in report_payload["tables"].items():
        table_path = tables_dir / f"{name}.csv"
        _write_dict_rows(table_path, list(rows))
        table_items[name] = {"path": str(table_path.name), "format": "text/csv"}

    text_report_path = args.output.with_suffix(".txt")
    text_report_path.write_text(report_payload["text"], encoding="utf-8")
    write_report_bundle(
        args.output,
        summary=report_payload["summary"],
        inputs={
            "analysis_file": str(args.analysis_file),
            "aggregation_file": (
                None if args.aggregation_file is None else str(args.aggregation_file)
            ),
            "qaqc_file": None if args.qaqc_file is None else str(args.qaqc_file),
        },
        tables={"directory": tables_dir.name, "items": table_items},
        text_report={"path": text_report_path.name, "format": "text/plain"},
        source_path=args.analysis_file,
    )
    print(f"Wrote k0 report bundle to {args.output}")
    print(f"Wrote k0 text report to {text_report_path}")


def cmd_k0_import_kayzero(args: argparse.Namespace) -> None:
    result = import_kayzero_k0_library(
        args.input, preferred_version=args.preferred_version
    )
    _ensure_parent_dir(args.output)
    write_governed_library_json(args.output, result.library)
    report_output = args.report_output
    if report_output is None:
        report_output = args.output.with_name(f"{args.output.stem}_import_report.json")
    _ensure_parent_dir(report_output)
    write_import_report_json(report_output, result.report)
    print(f"Wrote governed k0 library to {args.output}")
    print(f"Wrote Kayzero import report to {report_output}")


def cmd_reactions(args: argparse.Namespace) -> None:
    """Browse IRDFF-II dosimetry reactions."""
    from fluxforge.data.irdff import IRDFF_REACTIONS

    # Get category filter
    category = args.category

    # Get all reactions (optionally filtered)
    if category and category != "all":
        if category not in IRDFF_REACTIONS:
            print(f"Unknown category: {category}")
            print(f"Available categories: {', '.join(IRDFF_REACTIONS.keys())}")
            return
        reactions = {category: IRDFF_REACTIONS[category]}
    else:
        reactions = IRDFF_REACTIONS

    # Format output
    if args.format == "json":
        import json

        print(json.dumps(reactions, indent=2))
        return

    # Default table format
    print(f"\n{'='*80}")
    print("  IRDFF-II DOSIMETRY REACTIONS")
    print(f"{'='*80}\n")

    total = 0
    for cat, rxns in reactions.items():
        print(f"\n{cat.upper()} REACTIONS")
        print("-" * 60)
        print(f"{'Reaction':<30} {'Target':<10} {'Product':<10} {'Thresh (MeV)':<12}")
        print("-" * 60)

        for rxn_name, rxn_info in sorted(rxns.items()):
            threshold = rxn_info.get("threshold", 0.0)
            print(
                f"{rxn_name:<30} {rxn_info['target']:<10} {rxn_info['product']:<10} {threshold:>12.2f}"
            )
            total += 1

    print(f"\nTotal: {total} reactions")

    # Show additional info if requested
    if args.target:
        print(f"\n\nFiltering for target: {args.target}")
        for cat, rxns in IRDFF_REACTIONS.items():
            for rxn_name, rxn_info in rxns.items():
                if args.target.lower() in rxn_info["target"].lower():
                    print(f"  {rxn_name} ({cat})")


def cmd_gui(args: argparse.Namespace) -> None:
    """Launch FluxForge desktop GUI."""
    if args.dry_run:
        print(f"GUI dry run: project_dir={args.project_dir}")
        return

    try:
        from fluxforge.gui.app import launch_modern_gui
        from fluxforge.gui.qt_compat import QT_AVAILABLE
    except Exception:  # pragma: no cover - import/runtime environment specific
        launch_modern_gui = None
        QT_AVAILABLE = False

    if QT_AVAILABLE and launch_modern_gui is not None:
        launch_modern_gui(project_dir=args.project_dir)
        return

    try:
        from fluxforge_gui.app import launch_gui
    except Exception as exc:  # pragma: no cover - import/runtime environment specific
        raise RuntimeError(
            "Unable to start FluxForge GUI. The modern Qt shell is unavailable and "
            "the archived Tk fallback could not be imported either."
        ) from exc

    print(
        "Modern Qt GUI extras are unavailable in this environment; launching the "
        "archived Tk fallback.",
        file=sys.stderr,
    )
    launch_gui(project_dir=args.project_dir)


def cmd_plots(args: argparse.Namespace) -> None:
    """Generate master-plan plot suite in headless mode."""
    if args.dry_run:
        mode = "example" if args.example else "artifacts"
        print(f"Plots dry run: mode={mode}, output_dir={args.output_dir}")
        return

    # Make plotting robust over SSH/headless sessions.
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)

    from fluxforge.plots.master_suite import (
        generate_master_plan_plots,
        load_example_plot_inputs,
        load_plot_inputs_from_artifacts,
        normalize_plot_formats,
    )

    output_dir = Path(args.output_dir)
    formats = normalize_plot_formats(args.format)

    if args.example:
        plot_inputs = load_example_plot_inputs()
    else:
        missing = [
            name
            for name, value in (
                ("--unfold-file", args.unfold_file),
                ("--response-file", args.response_file),
                ("--rates-file", args.rates_file),
                ("--prior-flux-file", args.prior_flux_file),
            )
            if value is None
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                "Artifact mode requires all of: --unfold-file, --response-file, "
                f"--rates-file, --prior-flux-file. Missing: {missing_list}"
            )

        plot_inputs = load_plot_inputs_from_artifacts(
            unfold_file=Path(args.unfold_file),
            response_file=Path(args.response_file),
            rates_file=Path(args.rates_file),
            prior_flux_file=Path(args.prior_flux_file),
            validate=args.validate,
        )

    produced = generate_master_plan_plots(
        plot_inputs,
        output_dir=output_dir,
        formats=formats,
        include_response_plot=args.include_response_plot,
    )

    print(f"Wrote {sum(len(v) for v in produced.values())} plot files to {output_dir}")
    for key in sorted(produced):
        for path in produced[key]:
            print(f"  {key}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UWNR Flux-Wire–Driven Neutron Spectrum Reconstruction Tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="Ingest spectrum files into schema artifacts"
    )
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=Path("spectrum.json"))
    ingest.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset to apply before explicit user overrides",
    )
    ingest.add_argument(
        "--background-file",
        type=Path,
        default=None,
        help="Measured background spectrum file (.asc/.txt/.spe or artifact)",
    )
    ingest.add_argument(
        "--background-scale-mode",
        choices=["live", "real", "manual"],
        default="live",
        help="Normalization mode for measured background subtraction",
    )
    ingest.add_argument(
        "--background-scale-factor",
        type=float,
        default=None,
        help="Manual background scale factor (used only for --background-scale-mode manual)",
    )
    ingest.add_argument(
        "--energy-calibration",
        type=str,
        default=None,
        help="User override calibration coefficients as CSV: A,B[,C]",
    )
    ingest.add_argument(
        "--efficiency-coefficients",
        type=str,
        default=None,
        help="User override efficiency coefficients as CSV: C1,C2,C3,C4[,DetModel]",
    )
    ingest.add_argument(
        "--save-background-adjusted",
        type=Path,
        default=None,
        help="Optional CSV output for background-adjusted per-channel counts",
    )
    ingest.add_argument(
        "--save-final-corrected",
        type=Path,
        default=None,
        help="Optional CSV output for background-adjusted, calibrated, efficiency-corrected per-channel counts",
    )
    _add_validate_option(ingest)
    ingest.set_defaults(func=cmd_ingest)

    ingest_batch = subparsers.add_parser(
        "ingest-batch",
        help="Ingest a directory tree of spectra with one shared background file",
    )
    ingest_batch.add_argument("--input-dir", type=Path, required=True)
    ingest_batch.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset to apply before explicit user overrides",
    )
    ingest_batch.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/batch_spectra"),
        help="Directory for per-spectrum artifact JSON files",
    )
    ingest_batch.add_argument(
        "--background-file",
        type=Path,
        default=None,
        help="Measured background spectrum file applied to every input spectrum",
    )
    ingest_batch.add_argument(
        "--background-scale-mode",
        choices=["live", "real", "manual"],
        default="live",
        help="Normalization mode for measured background subtraction",
    )
    ingest_batch.add_argument(
        "--background-scale-factor",
        type=float,
        default=None,
        help="Manual background scale factor (used only for --background-scale-mode manual)",
    )
    ingest_batch.add_argument(
        "--energy-calibration",
        type=str,
        default=None,
        help="User override calibration coefficients as CSV: A,B[,C]",
    )
    ingest_batch.add_argument(
        "--efficiency-coefficients",
        type=str,
        default=None,
        help="User override efficiency coefficients as CSV: C1,C2,C3,C4[,DetModel]",
    )
    ingest_batch.add_argument(
        "--background-adjusted-dir",
        type=Path,
        default=None,
        help="Optional directory for background-adjusted per-channel CSV exports",
    )
    ingest_batch.add_argument(
        "--final-corrected-dir",
        type=Path,
        default=None,
        help="Optional directory for calibrated and efficiency-corrected per-channel CSV exports",
    )
    _add_validate_option(ingest_batch)
    ingest_batch.set_defaults(func=cmd_ingest_batch)

    spectrum_plot = subparsers.add_parser(
        "spectrum-plot",
        help="Save a calibrated gamma-spectrum plot with optional background subtraction and manual ROI overlays",
    )
    spectrum_plot.add_argument("--input", type=Path, required=True)
    spectrum_plot.add_argument("--output", type=Path, required=True)
    spectrum_plot.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset to apply before explicit user overrides",
    )
    spectrum_plot.add_argument(
        "--background-file",
        type=Path,
        default=None,
        help="Measured background spectrum file (.asc/.txt/.spe or artifact)",
    )
    spectrum_plot.add_argument(
        "--background-scale-mode",
        choices=["live", "real", "manual"],
        default="live",
        help="Normalization mode for measured background subtraction",
    )
    spectrum_plot.add_argument(
        "--background-scale-factor",
        type=float,
        default=None,
        help="Manual background scale factor (used only for --background-scale-mode manual)",
    )
    spectrum_plot.add_argument(
        "--energy-calibration",
        type=str,
        default=None,
        help="User override calibration coefficients as CSV: A,B[,C]",
    )
    spectrum_plot.add_argument(
        "--efficiency-coefficients",
        type=str,
        default=None,
        help="Optional efficiency override CSV: C1,C2,C3,C4[,DetModel]",
    )
    spectrum_plot.add_argument(
        "--background-subtracted",
        action="store_true",
        help="Plot the measured-background-subtracted spectrum instead of the raw counts",
    )
    spectrum_plot.add_argument(
        "--manual-peaks-file",
        type=Path,
        default=None,
        help="Optional CSV/JSON file with manual peak ROI definitions to overlay on the plot",
    )
    spectrum_plot.add_argument(
        "--save-peak-report",
        type=Path,
        default=None,
        help="Optional peak-report artifact generated from the manual ROI definitions",
    )
    spectrum_plot.add_argument(
        "--feature-photopeak-keV",
        type=float,
        default=None,
        help="Manual photopeak energy used to compute Compton/escape/backscatter guides",
    )
    spectrum_plot.add_argument(
        "--feature-line-keV",
        type=float,
        default=None,
        help="Manual library-line energy used for spectral-feature overlays",
    )
    spectrum_plot.add_argument(
        "--feature-isotope",
        type=str,
        default=None,
        help="Nuclide name for selecting a library gamma line before feature calculations",
    )
    spectrum_plot.add_argument(
        "--auto-feature-driver",
        action="store_true",
        help="Automatically detect the dominant continuum-driving line from matched peaks",
    )
    spectrum_plot.add_argument(
        "--feature-detector-material",
        choices=["hpge", "nai"],
        default="hpge",
        help="Detector material assumptions used for spectral-feature estimation",
    )
    spectrum_plot.add_argument(
        "--feature-tolerance-keV",
        type=float,
        default=2.0,
        help="Matching tolerance used for automatic continuum-driver detection",
    )
    spectrum_plot.add_argument(
        "--feature-min-intensity",
        type=float,
        default=0.02,
        help="Minimum gamma-line intensity for isotope/auto feature selection",
    )
    spectrum_plot.add_argument(
        "--save-feature-report",
        type=Path,
        default=None,
        help="Optional JSON report containing calculated spectral-feature energies and estimated heights",
    )
    spectrum_plot.add_argument(
        "--title", type=str, default=None, help="Optional plot title override"
    )
    spectrum_plot.add_argument("--x-min-keV", type=float, default=None)
    spectrum_plot.add_argument("--x-max-keV", type=float, default=None)
    spectrum_plot.add_argument("--y-log", action="store_true", help="Use a log y-axis")
    _add_validate_option(spectrum_plot)
    spectrum_plot.set_defaults(func=cmd_spectrum_plot)

    peaks = subparsers.add_parser("peaks", help="Detect peaks from a spectrum artifact")
    peaks.add_argument("--spectrum-file", type=Path, required=True)
    peaks.add_argument("--output", type=Path, default=Path("peaks.json"))
    peaks.add_argument(
        "--sensitivity",
        choices=["default", "sensitive", "conservative"],
        default="default",
    )
    peaks.add_argument("--fit-window", type=int, default=6)
    peaks.add_argument(
        "--manual-peaks-file",
        type=Path,
        default=None,
        help="Optional CSV/JSON file with manual ROI definitions; bypasses auto peak detection",
    )
    peaks.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset used when --manual-peaks-file is supplied",
    )
    peaks.add_argument(
        "--background-file",
        type=Path,
        default=None,
        help="Measured background spectrum file used when --manual-peaks-file is supplied",
    )
    peaks.add_argument(
        "--background-scale-mode",
        choices=["live", "real", "manual"],
        default="live",
        help="Normalization mode for measured background subtraction in manual-ROI mode",
    )
    peaks.add_argument(
        "--background-scale-factor",
        type=float,
        default=None,
        help="Manual background scale factor for manual-ROI mode",
    )
    peaks.add_argument(
        "--energy-calibration",
        type=str,
        default=None,
        help="User override calibration coefficients as CSV: A,B[,C] for manual-ROI mode",
    )
    peaks.add_argument(
        "--efficiency-coefficients",
        type=str,
        default=None,
        help="Optional efficiency override CSV for manual-ROI mode",
    )
    peaks.add_argument(
        "--background-subtracted",
        action="store_true",
        help="Use the background-subtracted spectrum to compute manual ROI areas",
    )
    peaks.add_argument(
        "--method",
        choices=list(_peak_search_cli_choices()),
        default="segmented",
        help="Peak-search method for automatic detection from a spectrum artifact",
    )
    peaks.add_argument(
        "--max-peaks",
        type=int,
        default=12,
        help="Maximum number of auto-detected peaks when using a parity peak-search method",
    )
    _add_validate_option(peaks)
    peaks.set_defaults(func=cmd_peaks)

    roi_analyze = subparsers.add_parser(
        "roi-analyze",
        help="Analyze one explicit ROI with sideband/SNIP background handling and optional overlap decomposition",
    )
    roi_analyze.add_argument("--input", type=Path, required=True)
    roi_analyze.add_argument("--output", type=Path, default=Path("roi_analysis.json"))
    roi_analyze.add_argument("--label", type=str, default=None)
    roi_analyze.add_argument("--left-keV", type=float, default=None)
    roi_analyze.add_argument("--right-keV", type=float, default=None)
    roi_analyze.add_argument("--left-channel", type=int, default=None)
    roi_analyze.add_argument("--right-channel", type=int, default=None)
    roi_analyze.add_argument(
        "--background-method",
        choices=["roi_sideband", "snip", "linear_minima"],
        default="roi_sideband",
    )
    roi_analyze.add_argument(
        "--peak-search-method",
        choices=list(_peak_search_cli_choices()),
        default="mariscotti",
    )
    roi_analyze.add_argument("--sideband-width-keV", type=float, default=4.0)
    roi_analyze.add_argument("--decompose-overlaps", action="store_true")
    roi_analyze.add_argument("--max-components", type=int, default=3)
    roi_analyze.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset to apply before explicit user overrides",
    )
    roi_analyze.add_argument("--background-file", type=Path, default=None)
    roi_analyze.add_argument(
        "--background-scale-mode",
        choices=["live", "real", "manual"],
        default="live",
    )
    roi_analyze.add_argument("--background-scale-factor", type=float, default=None)
    roi_analyze.add_argument("--energy-calibration", type=str, default=None)
    roi_analyze.add_argument("--efficiency-coefficients", type=str, default=None)
    _add_validate_option(roi_analyze)
    roi_analyze.set_defaults(func=cmd_roi_analyze)

    roi_statistics = subparsers.add_parser(
        "roi-statistics",
        help="Run the same ROI across many spectra and summarize detector-consistency statistics",
    )
    roi_statistics.add_argument("--inputs", type=Path, nargs="+", required=True)
    roi_statistics.add_argument(
        "--output",
        type=Path,
        default=Path("roi_statistics.json"),
    )
    roi_statistics.add_argument("--label", type=str, default=None)
    roi_statistics.add_argument("--left-keV", type=float, default=None)
    roi_statistics.add_argument("--right-keV", type=float, default=None)
    roi_statistics.add_argument("--left-channel", type=int, default=None)
    roi_statistics.add_argument("--right-channel", type=int, default=None)
    roi_statistics.add_argument(
        "--background-method",
        choices=["roi_sideband", "snip", "linear_minima"],
        default="roi_sideband",
    )
    roi_statistics.add_argument(
        "--peak-search-method",
        choices=list(_peak_search_cli_choices()),
        default="mariscotti",
    )
    roi_statistics.add_argument("--sideband-width-keV", type=float, default=4.0)
    roi_statistics.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list_rafm_profiles(),
        help="Bundled detector/background preset to apply before explicit user overrides",
    )
    roi_statistics.add_argument("--energy-calibration", type=str, default=None)
    roi_statistics.add_argument("--efficiency-coefficients", type=str, default=None)
    _add_validate_option(roi_statistics)
    roi_statistics.set_defaults(func=cmd_roi_statistics)

    file_query = subparsers.add_parser(
        "file-query",
        help="Query spectrum/artifact files in a workspace tree for archive-style review",
    )
    file_query.add_argument("--root", type=Path, default=Path("."))
    file_query.add_argument(
        "--patterns",
        type=str,
        default="**/*.json,**/*.csv,**/*.ASC,**/*.spe,**/*.txt",
        help="Comma-separated glob patterns relative to --root",
    )
    file_query.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter against relative path",
    )
    file_query.add_argument(
        "--suffixes",
        type=str,
        default=None,
        help="Optional comma-separated suffix whitelist (for example .json,.csv)",
    )
    file_query.add_argument("--min-size-bytes", type=int, default=0)
    file_query.add_argument("--max-size-bytes", type=int, default=None)
    file_query.add_argument(
        "--modified-after",
        type=str,
        default=None,
        help="ISO timestamp lower bound, for example 2026-01-01T00:00:00",
    )
    file_query.add_argument("--limit", type=int, default=2000)
    file_query.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
    )
    file_query.add_argument("--output", type=Path, default=Path("file_query.json"))
    file_query.set_defaults(func=cmd_file_query)

    batch_compare = subparsers.add_parser(
        "batch-compare",
        help="Compare two tabular JSON/CSV artifacts by key fields and numeric deltas",
    )
    batch_compare.add_argument("--baseline", type=Path, required=True)
    batch_compare.add_argument("--candidate", type=Path, required=True)
    batch_compare.add_argument(
        "--keys",
        type=str,
        required=True,
        help="Comma-separated key columns used to match rows",
    )
    batch_compare.add_argument(
        "--numeric-fields",
        type=str,
        default=None,
        help="Optional comma-separated numeric fields to compare (auto-inferred when omitted)",
    )
    batch_compare.add_argument("--max-rows", type=int, default=2000)
    batch_compare.add_argument(
        "--output", type=Path, default=Path("batch_compare.json")
    )
    batch_compare.set_defaults(func=cmd_batch_compare)

    parity_check = subparsers.add_parser(
        "parity-check",
        help="Run algorithm/workflow parity fixtures and report pass/fail summaries",
    )
    parity_check.add_argument(
        "--reference-root",
        type=Path,
        default=Path("tests/spectra/reference_parity"),
        help="Root path that contains reference parity manifests and case fixtures",
    )
    parity_check.add_argument(
        "--activation-root",
        type=Path,
        default=Path("tests/activation_inventory/fixtures"),
        help="Root path that contains activation-inventory fixture manifests",
    )
    parity_check.add_argument(
        "--scope",
        choices=["all", "algorithm", "workflow"],
        default="all",
    )
    parity_check.add_argument(
        "--fixture-id",
        type=str,
        default=None,
        help="Optional fixture_id filter to run one case",
    )
    parity_check.add_argument(
        "--no-activation",
        dest="include_activation",
        action="store_false",
        help="Disable activation-inventory fixtures and run only reference-parity cases",
    )
    parity_check.set_defaults(include_activation=True)
    parity_check.add_argument(
        "--output",
        type=Path,
        default=Path("parity_check.json"),
    )
    parity_check.set_defaults(func=cmd_parity_check)

    phase5_crosswalk_report = subparsers.add_parser(
        "phase5-crosswalk-report",
        help=(
            "Validate and summarize the machine-readable Phase 5 writeup crosswalk "
            "tracker"
        ),
    )
    phase5_crosswalk_report.add_argument(
        "--crosswalk",
        type=Path,
        default=Path(".github/project-management/phase5_crosswalk.json"),
        help="Path to the Phase 5 crosswalk tracker JSON file",
    )
    phase5_crosswalk_report.add_argument(
        "--include-parity-summary",
        action="store_true",
        help="Include reference parity suite summary in the generated report",
    )
    phase5_crosswalk_report.add_argument(
        "--reference-root",
        type=Path,
        default=Path("tests/spectra/reference_parity"),
        help="Reference parity fixture root used when --include-parity-summary is set",
    )
    phase5_crosswalk_report.add_argument(
        "--activation-root",
        type=Path,
        default=Path("tests/activation_inventory/fixtures"),
        help="Activation fixture root used when --include-parity-summary is set",
    )
    phase5_crosswalk_report.add_argument(
        "--output",
        type=Path,
        default=Path("phase5_crosswalk_report.json"),
    )
    phase5_crosswalk_report.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("phase5_crosswalk_report.md"),
    )
    phase5_crosswalk_report.set_defaults(func=cmd_phase5_crosswalk_report)

    gui_acceptance_check = subparsers.add_parser(
        "gui-acceptance-check",
        help="Validate 3.27 GUI release-checklist and probe artifact readiness",
    )
    gui_acceptance_check.add_argument(
        "--checklist",
        type=Path,
        default=Path("docs/PHASE3_27_RELEASE_CHECKLIST.md"),
        help="Markdown checklist path for GUI release acceptance",
    )
    gui_acceptance_check.add_argument(
        "--artifact-dir",
        action="append",
        default=None,
        help=(
            "Probe artifact directory (repeatable). Defaults to phase325/326/327 "
            "review directories."
        ),
    )
    gui_acceptance_check.add_argument(
        "--output",
        type=Path,
        default=Path("gui_acceptance_check.json"),
    )
    gui_acceptance_check.set_defaults(func=cmd_gui_acceptance_checklist)

    library_list = subparsers.add_parser(
        "library-list",
        help="List bundled and user-registered nuclear-data sources",
    )
    library_list.add_argument("--capability", type=str, default=None)
    library_list.add_argument("--kind", type=str, default=None)
    library_list.add_argument("--json", action="store_true", dest="json")
    library_list.set_defaults(func=cmd_library_list)

    library_register = subparsers.add_parser(
        "library-register",
        help="Register a user gamma-line library by alias and locator",
    )
    library_register.add_argument("--alias", type=str, required=True)
    library_register.add_argument("--locator", type=str, required=True)
    library_register.add_argument("--description", type=str, default=None)
    library_register.set_defaults(func=cmd_library_register)

    library_remove = subparsers.add_parser(
        "library-remove",
        help="Remove a registered user gamma-line library",
    )
    library_remove.add_argument("--source-id", type=str, required=True)
    library_remove.set_defaults(func=cmd_library_remove)

    activity = subparsers.add_parser(
        "activity", help="Compute line activities from peak report"
    )
    activity.add_argument("--peaks-file", type=Path, required=True)
    activity.add_argument("--output", type=Path, default=Path("activities.json"))
    activity.add_argument("--live-time-s", type=float)
    activity.add_argument("--efficiency", type=float, default=1.0)
    activity.add_argument("--emission-probability", type=float, default=1.0)
    activity.add_argument("--half-life-s", type=float, default=1.0)
    activity.add_argument("--sample-mass-g", type=float)
    activity.add_argument("--isotope", type=str)
    activity.add_argument("--reaction-id", type=str)
    _add_validate_option(activity)
    activity.set_defaults(func=cmd_activity)

    activity_review = subparsers.add_parser(
        "activity-review",
        help="Review all matched isotope activities for one spectrum and export EOI tables/plots",
    )
    activity_review.add_argument("--peaks-file", type=Path, required=True)
    activity_review.add_argument(
        "--output",
        type=Path,
        default=Path("activity_review.json"),
    )
    activity_review.add_argument("--live-time-s", type=float)
    activity_review.add_argument("--cooling-time-s", type=float, default=0.0)
    activity_review.add_argument("--dead-time-fraction", type=float, default=0.0)
    activity_review.add_argument("--energy-tolerance-keV", type=float, default=2.0)
    activity_review.add_argument(
        "--source-id",
        type=str,
        default="fluxforge_bundled_gamma",
        choices=list(_activity_review_source_choices()),
    )
    activity_review.add_argument("--custom-gamma-path", type=Path, default=None)
    activity_review.add_argument(
        "--efficiency",
        type=float,
        default=1.0,
        help="Constant full-energy peak efficiency to use when no polynomial is supplied.",
    )
    activity_review.add_argument(
        "--efficiency-polynomial",
        type=str,
        default=None,
        help="Comma-separated log-polynomial coefficients a0,a1,... for ln(eff)=sum ai*(ln E)^i.",
    )
    activity_review.add_argument(
        "--efficiency-uncertainty",
        type=float,
        default=0.05,
        help="Relative efficiency uncertainty applied to the activity review.",
    )
    activity_review.add_argument("--sample-mass-g", type=float)
    activity_review.add_argument(
        "--eoi-export",
        type=str,
        default="activities",
        choices=("activities", "reaction-rates", "both"),
        help="Choose whether to export EOI activities CSVs, reaction-rate CSVs, or both.",
    )
    activity_review.add_argument(
        "--irradiation-time-s",
        type=float,
        default=0.0,
        help="Single-segment irradiation duration used for reaction-rate export when no segment file is supplied.",
    )
    activity_review.add_argument(
        "--irradiation-segments-file",
        type=Path,
        default=None,
        help="Optional JSON/CSV segment definition file with duration_s and relative_power columns.",
    )
    activity_review.add_argument("--isotope-csv-output", type=Path, default=None)
    activity_review.add_argument("--line-csv-output", type=Path, default=None)
    activity_review.add_argument("--reaction-rate-csv-output", type=Path, default=None)
    activity_review.add_argument(
        "--reaction-rates-output",
        type=Path,
        default=None,
        help="Optional JSON/YAML reaction-rates artifact output path in addition to CSV export.",
    )
    activity_review.add_argument("--decay-plot", type=Path, default=None)
    activity_review.add_argument("--bateman-plot", type=Path, default=None)
    _add_validate_option(activity_review)
    activity_review.set_defaults(func=cmd_activity_review)

    inventory_review = subparsers.add_parser(
        "inventory-review",
        help="Propagate an activity-review inventory to arbitrary times and export time-series tables/plots",
    )
    inventory_review.add_argument("--activity-review-file", type=Path, required=True)
    inventory_review.add_argument(
        "--output",
        type=Path,
        default=Path("inventory_review.json"),
    )
    inventory_review.add_argument(
        "--decay-source-id",
        type=str,
        default=DEFAULT_DECAY_SOURCE_ID,
        choices=list(_inventory_decay_source_choices()),
    )
    inventory_review.add_argument(
        "--time-origin",
        type=str,
        default="eoi",
        choices=list(TIME_ORIGINS),
    )
    inventory_review.add_argument(
        "--time-points-s",
        type=str,
        default=None,
        help="Comma-separated relative times in seconds. Overrides the range arguments when provided.",
    )
    inventory_review.add_argument("--time-start-s", type=float, default=0.0)
    inventory_review.add_argument("--time-stop-s", type=float, default=86400.0)
    inventory_review.add_argument("--time-count", type=int, default=25)
    inventory_review.add_argument(
        "--observable",
        type=str,
        default="activity",
        choices=("activity", "atoms", "mass", "dose"),
    )
    inventory_review.add_argument("--distance-cm", type=float, default=30.0)
    inventory_review.add_argument("--top-n", type=int, default=8)
    inventory_review.add_argument("--timeseries-csv-output", type=Path, default=None)
    inventory_review.add_argument("--eoi-csv-output", type=Path, default=None)
    inventory_review.add_argument("--count-start-csv-output", type=Path, default=None)
    inventory_review.add_argument("--count-end-csv-output", type=Path, default=None)
    inventory_review.add_argument("--plot-output", type=Path, default=None)
    inventory_review.set_defaults(func=cmd_inventory_review)

    second_irradiation = subparsers.add_parser(
        "second-irradiation-plan",
        help="Rank second-irradiation candidates from an inventory seed and schedule definition",
    )
    second_irradiation.add_argument("--inventory-file", type=Path, required=True)
    second_irradiation.add_argument("--schedule-file", type=Path, required=True)
    second_irradiation.add_argument("--candidates-file", type=Path, required=True)
    second_irradiation.add_argument(
        "--output",
        type=Path,
        default=Path("second_irradiation_plan.json"),
        help="Output JSON bundle path",
    )
    second_irradiation.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export of the selected post-second-irradiation inventory",
    )
    second_irradiation.set_defaults(func=cmd_second_irradiation_plan)

    ffexp_export = subparsers.add_parser(
        "ffexp-export",
        help="Package activity, inventory, masking, optimization, and second-irradiation products into a benchmark .ffexp bundle",
    )
    ffexp_export.add_argument("--activity-review-file", type=Path, default=None)
    ffexp_export.add_argument("--inventory-review-file", type=Path, default=None)
    ffexp_export.add_argument("--masking-file", type=Path, default=None)
    ffexp_export.add_argument("--optimization-file", type=Path, default=None)
    ffexp_export.add_argument("--second-irradiation-file", type=Path, default=None)
    ffexp_export.add_argument(
        "--plot-paths",
        type=str,
        default=None,
        help="Optional comma-separated list of plot paths to include in the plot manifest",
    )
    ffexp_export.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_bundle.ffexp"),
        help="Output ffexp bundle path",
    )
    ffexp_export.set_defaults(func=cmd_ffexp_export)

    isotope_priority = subparsers.add_parser(
        "isotope-priority",
        help="Rank important isotopes from activity-review gamma-spectrum outputs",
    )
    isotope_priority.add_argument("--activity-review-file", type=Path, required=True)
    isotope_priority.add_argument(
        "--output",
        type=Path,
        default=Path("isotope_priority.json"),
        help="Output JSON ranking bundle path",
    )
    isotope_priority.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export with ranked isotope rows",
    )
    isotope_priority.add_argument(
        "--isotopes-of-interest",
        type=str,
        default=None,
        help="Optional comma-separated isotope subset for focused ranking",
    )
    isotope_priority.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Optional limit on ranked isotopes (0 means include all)",
    )
    isotope_priority.add_argument(
        "--weight-activity",
        type=float,
        default=0.35,
        help="Weight for activity magnitude in priority score",
    )
    isotope_priority.add_argument(
        "--weight-detectability",
        type=float,
        default=0.25,
        help="Weight for net-count detectability in priority score",
    )
    isotope_priority.add_argument(
        "--weight-confidence",
        type=float,
        default=0.20,
        help="Weight for measurement-confidence term in priority score",
    )
    isotope_priority.add_argument(
        "--weight-line-support",
        type=float,
        default=0.10,
        help="Weight for multi-line support in priority score",
    )
    isotope_priority.add_argument(
        "--weight-dose",
        type=float,
        default=0.10,
        help="Weight for dose-rate relevance in priority score",
    )
    isotope_priority.set_defaults(func=cmd_isotope_priority)

    masking_review = subparsers.add_parser(
        "masking-review",
        help="Rank line-level masking interactions and alternate-line guidance from activity-review outputs",
    )
    masking_review.add_argument("--activity-review-file", type=Path, required=True)
    masking_review.add_argument(
        "--output",
        type=Path,
        default=Path("masking_review.json"),
        help="Output JSON masking-review bundle path",
    )
    masking_review.add_argument(
        "--isotopes-of-interest",
        type=str,
        default=None,
        help="Optional comma-separated isotope subset for focused masking review",
    )
    masking_review.add_argument(
        "--energy-window-keV",
        type=float,
        default=3.0,
        help="Maximum line-energy delta considered for masking interactions",
    )
    masking_review.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Optional limit for exported line-level/isotope-level masking rows",
    )
    masking_review.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export of line-level masking interactions",
    )
    masking_review.add_argument(
        "--isotope-csv-output",
        type=Path,
        default=None,
        help="Optional CSV export of aggregated masking-isotope ranking",
    )
    masking_review.add_argument(
        "--recommendation-csv-output",
        type=Path,
        default=None,
        help="Optional CSV export of alternate-line recommendations",
    )
    masking_review.set_defaults(func=cmd_masking_review)

    optimization_sweep = subparsers.add_parser(
        "optimization-sweep",
        help="Rank schedule candidates using DI-FOM, FIM, MWDCS, or advanced N1/N2 objectives",
    )
    optimization_sweep.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON payload with schedule candidates and optional isotope weights",
    )
    optimization_sweep.add_argument(
        "--activity-review-file",
        type=Path,
        default=None,
        help="Optional activity-review bundle used to auto-generate schedule candidates when --input is omitted.",
    )
    optimization_sweep.add_argument(
        "--output",
        type=Path,
        default=Path("optimization_sweep.json"),
        help="Output JSON bundle path",
    )
    optimization_sweep.add_argument(
        "--objective",
        type=str,
        default="di-fom",
        choices=("di-fom", "fim-d", "fim-a", "fim-c", "mwdcs", "bass-d", "stbd-mr"),
        help="Optimization objective",
    )
    optimization_sweep.add_argument(
        "--isotopes-of-interest",
        type=str,
        default=None,
        help="Comma-separated isotopes used to filter candidate line terms before scoring",
    )
    optimization_sweep.add_argument(
        "--irradiation-grid-s",
        type=str,
        default="1800,3600,7200,14400",
        help="Irradiation-duration grid (s) used when candidates are auto-generated from activity-review data.",
    )
    optimization_sweep.add_argument(
        "--cooldown-grid-s",
        type=str,
        default="0,1800,7200,21600",
        help="Cooldown-time grid (s) used when candidates are auto-generated from activity-review data.",
    )
    optimization_sweep.add_argument(
        "--count-grid-s",
        type=str,
        default="300,600,900,1800",
        help="Count-time grid (s) used when candidates are auto-generated from activity-review data.",
    )
    optimization_sweep.add_argument(
        "--reference-irradiation-time-s",
        type=float,
        default=3600.0,
        help="Reference irradiation duration used to scale activity-review EOI activities during candidate generation.",
    )
    optimization_sweep.add_argument(
        "--unfold-file",
        type=Path,
        default=None,
        help="Optional unfolded neutron spectrum bundle used to derive spectrum-aware candidate scaling.",
    )
    optimization_sweep.add_argument(
        "--neutron-spectrum-file",
        type=Path,
        default=None,
        help="Optional neutron source spectrum CSV (energy, flux) used to derive spectrum-aware candidate scaling.",
    )
    optimization_sweep.add_argument(
        "--flux-scale",
        type=float,
        default=1.0,
        help="Base multiplicative flux scale before spectrum-derived hardness/integral adjustments.",
    )
    optimization_sweep.add_argument(
        "--reference-flux-integral",
        type=float,
        default=0.0,
        help="Optional reference integral flux; when >0, scales candidates by integral_flux/reference_flux_integral.",
    )
    optimization_sweep.add_argument(
        "--generated-candidates-output",
        type=Path,
        default=None,
        help="Optional path to write the auto-generated candidate payload before optimization scoring.",
    )
    optimization_sweep.add_argument(
        "--enable-advanced-objectives",
        action="store_true",
        help="Required guard flag for advanced objectives such as bass-d and stbd-mr",
    )
    optimization_sweep.add_argument(
        "--target-nuclide",
        type=str,
        default=None,
        help="Target nuclide used by fim-c objective",
    )
    optimization_sweep.add_argument(
        "--nuisance-variance-fraction",
        type=float,
        default=0.0,
        help="Additional relative variance term for FIM line variances",
    )
    optimization_sweep.add_argument(
        "--fim-regularization",
        type=float,
        default=1.0e-6,
        help="Diagonal regularization added to Fisher matrices for FIM objectives",
    )
    optimization_sweep.add_argument(
        "--mwdcs-window-offsets-s",
        type=str,
        default="0,7200,86400",
        help="Comma-separated cooldown offsets (s) for generated MWDCS windows",
    )
    optimization_sweep.add_argument(
        "--mwdcs-window-count-time-s",
        type=float,
        default=900.0,
        help="Default count duration (s) per generated MWDCS window",
    )
    optimization_sweep.add_argument(
        "--mwdcs-full-spectrum-mode",
        action="store_true",
        help="Enable overlap penalty in MWDCS scoring to emulate full-spectrum fitting",
    )
    optimization_sweep.add_argument(
        "--mwdcs-overlap-penalty",
        type=float,
        default=0.0,
        help="Penalty weight for near-energy overlaps in MWDCS full-spectrum mode",
    )
    optimization_sweep.add_argument(
        "--bassd-dose-weight",
        type=float,
        default=0.02,
        help="Dose penalty weight for advanced bass-d objective",
    )
    optimization_sweep.add_argument(
        "--bassd-exploration-temperature",
        type=float,
        default=0.0,
        help="Stochastic exploration temperature for bass-d action utility",
    )
    optimization_sweep.add_argument(
        "--bassd-seed",
        type=int,
        default=17,
        help="Random seed used by bass-d exploration noise",
    )
    optimization_sweep.add_argument(
        "--stbdmr-window-offsets-s",
        type=str,
        default="0,7200,86400",
        help="Comma-separated cooldown offsets (s) for generated STBD-MR windows",
    )
    optimization_sweep.add_argument(
        "--stbdmr-window-count-time-s",
        type=float,
        default=900.0,
        help="Default count duration (s) per generated STBD-MR window",
    )
    optimization_sweep.add_argument(
        "--stbdmr-masking-regularization",
        type=float,
        default=0.1,
        help="Masking regularization weight for STBD-MR objective",
    )
    optimization_sweep.add_argument(
        "--stbdmr-differentiable-graph",
        action="store_true",
        help="Enable differentiable interference-graph bonus in STBD-MR objective",
    )
    optimization_sweep.add_argument(
        "--stbdmr-graph-temperature",
        type=float,
        default=2.0,
        help="Energy temperature (keV) for differentiable STBD-MR graph affinities",
    )
    optimization_sweep.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV export with ranked candidate summary",
    )
    optimization_sweep.set_defaults(func=cmd_optimization_sweep)

    rates = subparsers.add_parser(
        "rates", help="Compute reaction rates from line activities"
    )
    rates.add_argument("--lines-file", type=Path, required=True)
    rates.add_argument("--segments-file", type=Path)
    rates.add_argument("--duration-s", type=float, default=1.0)
    rates.add_argument("--half-life-s", type=float, default=1.0)
    rates.add_argument("--output", type=Path, default=Path("rates.json"))
    _add_validate_option(rates)
    rates.set_defaults(func=cmd_rates)

    astm_e2005 = subparsers.add_parser(
        "astm-e2005", help="Run the ASTM E2005 reactor dosimetry workflow"
    )
    astm_e2005.add_argument("--plan-file", type=Path, required=True)
    astm_e2005.add_argument("--output", type=Path, default=Path("astm_e2005.json"))
    _add_validate_option(astm_e2005)
    astm_e2005.set_defaults(func=cmd_astm_e2005)

    astm_e261 = subparsers.add_parser(
        "astm-e261", help="Run the ASTM E261 reactor dosimetry workflow"
    )
    astm_e261.add_argument("--plan-file", type=Path, required=True)
    astm_e261.add_argument("--output", type=Path, default=Path("astm_e261.json"))
    _add_validate_option(astm_e261)
    astm_e261.set_defaults(func=cmd_astm_e261)

    astm_e262 = subparsers.add_parser(
        "astm-e262", help="Run the ASTM E262 thermal neutron fluence workflow"
    )
    astm_e262.add_argument("--plan-file", type=Path, required=True)
    astm_e262.add_argument("--output", type=Path, default=Path("astm_e262.json"))
    _add_validate_option(astm_e262)
    astm_e262.set_defaults(func=cmd_astm_e262)

    astm_e3376 = subparsers.add_parser(
        "astm-e3376", help="Run the ASTM E3376 high-purity germanium detection workflow"
    )
    astm_e3376.add_argument("--plan-file", type=Path, required=True)
    astm_e3376.add_argument("--output", type=Path, default=Path("astm_e3376.json"))
    _add_validate_option(astm_e3376)
    astm_e3376.set_defaults(func=cmd_astm_e3376)

    rafm_validate = subparsers.add_parser(
        "rafm-validate",
        help="Run the committed RAFM raw-spectrum validation workflow against QG reference data",
    )
    rafm_validate.add_argument("--example-root", type=Path, required=True)
    rafm_validate.add_argument("--results-root", type=Path, default=None)
    rafm_validate.add_argument("--max-spectra", type=int, default=None)
    rafm_validate.add_argument(
        "--flux-wire-counting-method",
        type=str,
        default=None,
        help="Override flux-wire peak counting method for raw spectra",
    )
    rafm_validate.add_argument(
        "--generic-counting-method",
        type=str,
        default=None,
        help="Override targeted/generic peak counting method for raw spectra",
    )
    rafm_validate.add_argument(
        "--no-fail",
        action="store_true",
        help="Complete the workflow without raising when parity thresholds are violated",
    )
    rafm_validate.set_defaults(func=cmd_rafm_validate)

    rafm_qg_benchmark = subparsers.add_parser(
        "rafm-qg-benchmark",
        help="Process committed QG RAFM flux-wire data through reaction-rate and unfolding outputs",
    )
    rafm_qg_benchmark.add_argument("--example-root", type=Path, required=True)
    rafm_qg_benchmark.add_argument("--results-root", type=Path, default=None)
    rafm_qg_benchmark.add_argument("--max-spectra", type=int, default=None)
    rafm_qg_benchmark.set_defaults(func=cmd_rafm_qg_benchmark)

    rafm_compare = subparsers.add_parser(
        "rafm-compare-branches",
        help="Compare completed raw-branch and QG-branch RAFM results",
    )
    rafm_compare.add_argument("--raw-results-root", type=Path, required=True)
    rafm_compare.add_argument("--qg-results-root", type=Path, required=True)
    rafm_compare.add_argument("--output-root", type=Path, default=None)
    rafm_compare.set_defaults(func=cmd_rafm_compare_branches)

    phase6_ldrd_example = subparsers.add_parser(
        "phase6-ldrd-worked-example",
        help="Run the Phase 6 worked example on RAFM LDRD-backed irradiation data",
    )
    phase6_ldrd_example.add_argument(
        "--sample-id",
        type=str,
        default=PHASE6_LDRD_DEFAULT_SAMPLE_ID,
        help="Sample ID from examples/RAFM_irradiation/results/analysis_json",
    )
    phase6_ldrd_example.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory for worked-example artifacts",
    )
    phase6_ldrd_example.set_defaults(func=cmd_phase6_ldrd_worked_example)

    phase6_ldrd_second_repo = subparsers.add_parser(
        "phase6-ldrd-second-irradiation-repo",
        help="Build a reproducible RAFM LDRD second-irradiation decision repository bundle",
    )
    phase6_ldrd_second_repo.add_argument(
        "--sample-id",
        type=str,
        default=PHASE6_LDRD_DEFAULT_SAMPLE_ID,
        help="Anchor sample ID used to generate/reuse Phase 6 worked-example artifacts",
    )
    phase6_ldrd_second_repo.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory for decision-repository artifacts",
    )
    phase6_ldrd_second_repo.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top data-driven isotopes to keep before appending literature seed isotopes",
    )
    phase6_ldrd_second_repo.set_defaults(func=cmd_phase6_ldrd_second_irradiation_repo)

    response = subparsers.add_parser(
        "response", help="Build response matrix from cross sections"
    )
    response.add_argument("--cross-section-file", type=Path, required=True)
    response.add_argument("--number-densities-file", type=Path, required=True)
    response.add_argument("--boundaries-file", type=Path, required=True)
    response.add_argument("--output", type=Path, default=Path("response.json"))
    _add_validate_option(response)
    response.set_defaults(func=cmd_response)

    unfold = subparsers.add_parser(
        "unfold", help="Infer spectrum using GLS, GRAVEL, MLEM, MAXED, RMLE, or ML Seed"
    )
    unfold.add_argument("--rates-file", type=Path, required=True)
    unfold.add_argument("--response-file", type=Path, required=True)
    unfold.add_argument("--prior-flux-file", type=Path)
    unfold.add_argument(
        "--method",
        type=str,
        default="gls",
        choices=["gls", "gravel", "mlem", "maxed", "rmle", "ml_seed"],
        help="Unfolding method to use",
    )
    unfold.add_argument("--prior-uncertainty", type=float, default=0.25)
    unfold.add_argument(
        "--prior-cov-model",
        dest="prior_cov_model",
        type=str,
        default=PriorCovarianceModel.DIAGONAL.value,
        choices=[m.value for m in PriorCovarianceModel],
        help="Prior covariance model used by GLS",
    )
    unfold.add_argument(
        "--prior-correlation-length",
        dest="prior_correlation_length",
        type=float,
        default=1.0,
        help="Lethargy correlation length (used for lethargy_correlated model)",
    )
    unfold.add_argument(
        "--max-iters",
        type=int,
        default=250,
        help="Maximum iterations for iterative solvers",
    )
    unfold.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Convergence tolerance for iterative solvers",
    )
    unfold.add_argument(
        "--chi2-tolerance",
        type=float,
        default=0.01,
        help="Chi-squared tolerance for iterative solvers",
    )
    unfold.add_argument(
        "--relaxation",
        type=float,
        default=0.8,
        help="Relaxation factor for iterative solvers",
    )
    unfold.add_argument(
        "--floor",
        type=float,
        default=1e-20,
        help="Positive floor used by iterative solvers",
    )
    unfold.add_argument(
        "--convergence-mode",
        type=str,
        default="relative",
        choices=["relative", "ddJ"],
        help="MLEM convergence criterion",
    )
    unfold.add_argument(
        "--use-ml-seed",
        action="store_true",
        help="Use the ML Seed approximation to initialize GRAVEL or RMLE",
    )
    unfold.add_argument(
        "--ml-seed-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold used to accept an ML Seed solution",
    )
    unfold.add_argument(
        "--no-enforce-nonnegativity",
        dest="enforce_nonnegativity",
        action="store_false",
        help="Disable non-negativity clipping for GLS",
    )
    unfold.add_argument(
        "--verbose-solver",
        action="store_true",
        help="Print iterative solver progress to stdout",
    )
    unfold.set_defaults(enforce_nonnegativity=True)
    unfold.add_argument("--output", type=Path, default=Path("spectrum.json"))
    _add_validate_option(unfold)
    unfold.set_defaults(func=cmd_unfold)

    compare = subparsers.add_parser(
        "compare", help="Compare unfolded spectrum with reference"
    )
    compare.add_argument("--unfold-file", type=Path, required=True)
    compare.add_argument("--truth-flux-file", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("validation.json"))
    _add_validate_option(compare)
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser(
        "report", help="Compile a report bundle from artifacts"
    )
    report.add_argument("--spectrum-file", type=Path)
    report.add_argument("--peaks-file", type=Path)
    report.add_argument("--lines-file", type=Path)
    report.add_argument("--rates-file", type=Path)
    report.add_argument("--masking-file", type=Path)
    report.add_argument("--optimization-file", type=Path)
    report.add_argument("--unfold-file", type=Path)
    report.add_argument("--validation-file", type=Path)
    report.add_argument("--validation-results-root", type=Path)
    report.add_argument("--output", type=Path, default=Path("report.json"))
    _add_validate_option(report)
    report.set_defaults(func=cmd_report)

    k0_normalize = subparsers.add_parser(
        "k0-normalize",
        help="Normalize a peak report into standards-oriented k0 peak observations",
    )
    k0_normalize.add_argument("--peaks-file", type=Path, required=True)
    k0_normalize.add_argument("--spectrum-file", type=Path)
    k0_normalize.add_argument("--detector-characterization-file", type=Path)
    k0_normalize.add_argument("--detector-id", type=str, default=None)
    k0_normalize.add_argument("--geometry-id", type=str, default=None)
    k0_normalize.add_argument("--irradiation-time-s", type=float, default=0.0)
    k0_normalize.add_argument("--decay-time-s", type=float, default=0.0)
    k0_normalize.add_argument("--counting-time-s", type=float, default=None)
    k0_normalize.add_argument("--import-format", type=str, default="peak_report")
    k0_normalize.add_argument("--project-id", type=str, default=None)
    k0_normalize.add_argument("--sample-id", type=str, default=None)
    k0_normalize.add_argument("--irradiation-id", type=str, default=None)
    k0_normalize.add_argument("--measurement-id", type=str, default=None)
    k0_normalize.add_argument("--allow-advanced-lines", action="store_true")
    k0_normalize.add_argument("--expert-override", action="store_true")
    k0_normalize.add_argument(
        "--output", type=Path, default=Path("k0_observations.json")
    )
    _add_validate_option(k0_normalize)
    k0_normalize.set_defaults(func=cmd_k0_normalize)

    k0_detector = subparsers.add_parser(
        "k0-detector",
        help="Build a reusable detector-characterization artifact for k0 workflows",
    )
    k0_detector.add_argument("--points-file", type=Path, required=True)
    k0_detector.add_argument("--detector-id", type=str, required=True)
    k0_detector.add_argument("--reference-position-mm", type=float, required=True)
    k0_detector.add_argument("--degree", type=int, default=2)
    k0_detector.add_argument("--peak-to-total-ratio", type=float, default=None)
    k0_detector.add_argument("--coincidence-mode", type=str, default="not_applied")
    k0_detector.add_argument(
        "--output", type=Path, default=Path("detector_characterization.json")
    )
    _add_validate_option(k0_detector)
    k0_detector.set_defaults(func=cmd_k0_detector)

    k0_facility = subparsers.add_parser(
        "k0-facility",
        help="Characterize a thermal irradiation facility using a bare triple-monitor workflow",
    )
    k0_facility.add_argument("--input", type=Path, required=True)
    k0_facility.add_argument(
        "--output", type=Path, default=Path("facility_characterization.json")
    )
    _add_validate_option(k0_facility)
    k0_facility.set_defaults(func=cmd_k0_facility)

    k0_analyze = subparsers.add_parser(
        "k0-analyze",
        help="Run a first-pass k0 analysis from normalized observations and facility characterization",
    )
    k0_analyze.add_argument("--observations-file", type=Path, required=True)
    k0_analyze.add_argument("--facility-file", type=Path, required=True)
    k0_analyze.add_argument("--sample-mass-g", type=float, required=True)
    k0_analyze.add_argument("--reference-isotope", type=str, default="Au-198")
    k0_analyze.add_argument("--reference-mass-g", type=float, default=None)
    k0_analyze.add_argument("--k0-library-file", type=Path, default=None)
    k0_analyze.add_argument("--auxiliary-library-file", type=Path, default=None)
    k0_analyze.add_argument("--output", type=Path, default=Path("k0_analysis.json"))
    _add_validate_option(k0_analyze)
    k0_analyze.set_defaults(func=cmd_k0_analyze)

    k0_aggregate = subparsers.add_parser(
        "k0-aggregate",
        help="Aggregate k0 analysis bundles across measurements and irradiations",
    )
    k0_aggregate.add_argument("--analysis-files", type=Path, nargs="+", required=True)
    k0_aggregate.add_argument(
        "--output", type=Path, default=Path("k0_aggregation.json")
    )
    _add_validate_option(k0_aggregate)
    k0_aggregate.set_defaults(func=cmd_k0_aggregate)

    k0_qaqc = subparsers.add_parser(
        "k0-qaqc",
        help="Evaluate blank and CRM QA/QC from k0 analysis bundles",
    )
    k0_qaqc.add_argument("--plan-file", type=Path, required=True)
    k0_qaqc.add_argument("--output", type=Path, default=Path("k0_qaqc.json"))
    _add_validate_option(k0_qaqc)
    k0_qaqc.set_defaults(func=cmd_k0_qaqc)

    k0_report = subparsers.add_parser(
        "k0-report",
        help="Write a richer k0 report bundle with optional aggregation and QA/QC summaries",
    )
    k0_report.add_argument("--analysis-file", type=Path, required=True)
    k0_report.add_argument("--aggregation-file", type=Path, default=None)
    k0_report.add_argument("--qaqc-file", type=Path, default=None)
    k0_report.add_argument("--output", type=Path, default=Path("k0_report.json"))
    _add_validate_option(k0_report)
    k0_report.set_defaults(func=cmd_k0_report)

    k0_import_kayzero = subparsers.add_parser(
        "k0-import-kayzero",
        help="Import a user-supplied Kayzero library folder or zip into governed FluxForge k0 JSON",
    )
    k0_import_kayzero.add_argument("--input", type=Path, required=True)
    k0_import_kayzero.add_argument("--preferred-version", type=str, default=None)
    k0_import_kayzero.add_argument(
        "--output", type=Path, default=Path("kayzero_k0_library.json")
    )
    k0_import_kayzero.add_argument("--report-output", type=Path, default=None)
    k0_import_kayzero.set_defaults(func=cmd_k0_import_kayzero)

    # Reaction browser command
    reactions = subparsers.add_parser(
        "reactions", help="Browse IRDFF-II dosimetry reactions"
    )
    reactions.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "thermal", "epithermal", "fast", "fission"],
        help="Filter by reaction category",
    )
    reactions.add_argument(
        "--target",
        type=str,
        default=None,
        help="Filter by target nuclide (e.g., 'Au', 'Fe')",
    )
    reactions.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format",
    )
    reactions.set_defaults(func=cmd_reactions)

    gui = subparsers.add_parser("gui", help="Launch FluxForge desktop GUI")
    gui.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory used as the default file root in the GUI",
    )
    gui.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI parsing for GUI launch without opening a window",
    )
    gui.set_defaults(func=cmd_gui)

    plots = subparsers.add_parser(
        "plots",
        help="Generate master-plan plots (headless; SSH-safe)",
    )
    plots.add_argument(
        "--example",
        action="store_true",
        help="Use bundled FluxForge example data (no external dependencies)",
    )
    plots.add_argument(
        "--unfold-file",
        type=Path,
        help="Unfold result artifact (required unless --example)",
    )
    plots.add_argument(
        "--response-file",
        type=Path,
        help="Response bundle artifact (required unless --example)",
    )
    plots.add_argument(
        "--rates-file",
        type=Path,
        help="Reaction-rates artifact (required unless --example)",
    )
    plots.add_argument(
        "--prior-flux-file",
        type=Path,
        help="Prior flux JSON file (required unless --example)",
    )
    plots.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/plots"),
        help="Directory where plot files will be written",
    )
    plots.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output plot format",
    )
    plots.add_argument(
        "--no-response-plot",
        dest="include_response_plot",
        action="store_false",
        help="Skip response matrix plot output",
    )
    plots.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI parsing for plot generation without writing files",
    )
    plots.set_defaults(include_response_plot=True)
    _add_validate_option(plots)
    plots.set_defaults(func=cmd_plots)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        print("No command provided.")


if __name__ == "__main__":
    main()
