"""Shared phase-6 irradiation optimization and export helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fluxforge.analysis.masking_review import (
    rank_line_masking_from_activity_review_payload,
    recommend_alternate_lines,
    summarize_masking_isotopes,
)
from fluxforge.core.inventory_timeline import (
    InventoryState,
    build_inventory_state_from_payload,
    compute_inventory_time_evolution,
)


_SECONDS_PER_DAY = 24.0 * 3600.0
_SECONDS_PER_WEEK = 7.0 * _SECONDS_PER_DAY
_SECONDS_PER_YEAR = 365.25 * _SECONDS_PER_DAY


@dataclass(frozen=True)
class SecondIrradiationCandidate:
    """One candidate second-irradiation schedule."""

    label: str
    flux_scale: float = 1.0
    duration_factor: float = 1.0
    second_cooling_time_s: float = 0.0
    pulse_delay_s: float = 0.0


@dataclass(frozen=True)
class SecondIrradiationNuclideScore:
    """One nuclide projection for a second-irradiation candidate."""

    candidate: str
    nuclide: str
    activity_bq: float
    weight: float
    weighted_activity: float

    def to_row(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "nuclide": self.nuclide,
            "activity_bq": float(self.activity_bq),
            "weight": float(self.weight),
            "weighted_activity": float(self.weighted_activity),
        }


@dataclass(frozen=True)
class SecondIrradiationScore:
    """Aggregate score for one second-irradiation candidate."""

    label: str
    score: float
    flux_scale: float
    duration_factor: float
    second_cooling_time_s: float
    pulse_delay_s: float
    nuclide_rows: tuple[SecondIrradiationNuclideScore, ...]

    def to_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "label": self.label,
            "score": float(self.score),
            "flux_scale": float(self.flux_scale),
            "duration_factor": float(self.duration_factor),
            "second_cooling_time_s": float(self.second_cooling_time_s),
            "pulse_delay_s": float(self.pulse_delay_s),
            "nuclide_count": len(self.nuclide_rows),
        }


@dataclass(frozen=True)
class SecondIrradiationPlan:
    """Ranked second-irradiation planning result."""

    inventory_state: InventoryState
    first_cooling_time_s: float
    second_irradiation_time_s: float
    target_weights: dict[str, float]
    ranked_candidates: tuple[SecondIrradiationScore, ...]

    @property
    def selected_candidate(self) -> SecondIrradiationScore | None:
        if not self.ranked_candidates:
            return None
        return self.ranked_candidates[0]

    def selected_inventory_rows(self) -> list[dict[str, Any]]:
        selected = self.selected_candidate
        if selected is None:
            return []
        return [row.to_row() for row in selected.nuclide_rows]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _candidate_row_from_payload(
    objective: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    first_window = (candidate.get("window_scores") or [{}])[0]
    first_action = (candidate.get("action_scores") or [{}])[0]
    row: dict[str, Any] = {
        "rank": int(candidate.get("rank") or 0),
        "label": str(candidate.get("label") or ""),
        "objective": str(objective),
        "irradiation_time_s": float(candidate.get("irradiation_time_s") or 0.0),
        "cooldown_time_s": float(
            candidate.get("cooldown_time_s")
            or first_window.get("cooldown_time_s")
            or first_action.get("cooldown_time_s")
            or 0.0
        ),
        "count_time_s": float(
            candidate.get("count_time_s")
            or first_window.get("count_time_s")
            or first_action.get("count_time_s")
            or 0.0
        ),
    }
    if objective == "di-fom":
        row["objective_score"] = float(candidate.get("difom_score") or 0.0)
    elif objective == "bass-d":
        row["objective_score"] = float(candidate.get("total_utility") or 0.0)
        row["expected_dose_uSv"] = float(first_action.get("expected_dose_uSv") or 0.0)
    else:
        row["objective_score"] = float(
            candidate.get("objective_score") or candidate.get("total_score") or 0.0
        )
        diagnostics = candidate.get("matrix_diagnostics") or candidate.get("diagnostics") or {}
        if diagnostics:
            if "condition_number" in diagnostics:
                row["condition_number"] = float(diagnostics.get("condition_number") or 0.0)
            if "effective_rank" in diagnostics:
                row["effective_rank"] = int(diagnostics.get("effective_rank") or 0)
            if "graph_density" in diagnostics:
                row["graph_density"] = float(diagnostics.get("graph_density") or 0.0)
            if "masking_penalty" in diagnostics:
                row["masking_penalty"] = float(diagnostics.get("masking_penalty") or 0.0)
    return row


def default_phase6_endpoint_grid() -> tuple[tuple[str, float], ...]:
    """Return the standard shutdown-to-100-year endpoint grid."""

    return (
        ("shutdown", 0.0),
        ("1_d", _SECONDS_PER_DAY),
        ("1_wk", _SECONDS_PER_WEEK),
        ("1_y", _SECONDS_PER_YEAR),
        ("100_y", 100.0 * _SECONDS_PER_YEAR),
    )


def build_phase6_support_artifacts(
    activity_review_payload: Mapping[str, Any],
    optimization_payload: Mapping[str, Any] | None,
    *,
    isotopes_of_interest: Sequence[str] = (),
    masking_top_n: int = 100,
    endpoint_grid: Sequence[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Build deterministic phase-6 export tables from activity/optimization inputs."""

    inventory_state = build_inventory_state_from_payload(dict(activity_review_payload))
    resolved_endpoint_grid = tuple(endpoint_grid or default_phase6_endpoint_grid())
    timeline = compute_inventory_time_evolution(
        inventory_state,
        relative_times_s=tuple(offset_s for _label, offset_s in resolved_endpoint_grid),
        time_origin="eoi",
        decay_source_id=str(
            activity_review_payload.get("decay_source_id")
            or inventory_state.decay_source_id
        ),
        distance_cm=30.0,
    )

    inventory_rows = timeline.time_series_rows()
    dose_endpoints: list[dict[str, Any]] = []
    for endpoint_label, offset_s in resolved_endpoint_grid:
        total_rows = [
            row
            for row in inventory_rows
            if str(row.get("nuclide")) == "Total"
            and abs(float(row.get("relative_time_s") or 0.0) - float(offset_s)) < 1.0e-6
        ]
        if not total_rows:
            continue
        row = dict(total_rows[0])
        dose_endpoints.append(
            {
                "endpoint": endpoint_label,
                "relative_time_s": float(row.get("relative_time_s") or 0.0),
                "absolute_time_s": float(row.get("absolute_time_s") or 0.0),
                "activity_bq": float(row.get("activity_bq") or 0.0),
                "activity_uncertainty_bq": float(
                    row.get("activity_uncertainty_bq") or 0.0
                ),
                "dose_rate_uSv_h": float(row.get("dose_rate_uSv_h") or 0.0),
                "dose_rate_uncertainty_uSv_h": float(
                    row.get("dose_rate_uncertainty_uSv_h") or 0.0
                ),
            }
        )

    selected_isotopes = [str(item) for item in isotopes_of_interest if str(item).strip()]
    masking_ranked = rank_line_masking_from_activity_review_payload(
        activity_review_payload,
        isotopes_of_interest=selected_isotopes,
        top_n=masking_top_n,
    )
    masking_candidates = [item.to_row(index) for index, item in enumerate(masking_ranked, start=1)]
    masking_isotopes = summarize_masking_isotopes(masking_ranked, top_n=masking_top_n)
    alternate_lines = [
        item.to_row(index)
        for index, item in enumerate(
            recommend_alternate_lines(
                activity_review_payload,
                masking_ranked,
                top_n=masking_top_n,
            ),
            start=1,
        )
    ]

    optimization_grid: list[dict[str, Any]] = []
    recommended_schedules: list[dict[str, Any]] = []
    objective = ""
    if optimization_payload is not None:
        objective = str(optimization_payload.get("objective") or "")
        ranked_candidates = optimization_payload.get("ranked_candidates") or []
        if isinstance(ranked_candidates, list):
            optimization_grid = [
                _candidate_row_from_payload(objective, item)
                for item in ranked_candidates
                if isinstance(item, Mapping)
            ]
            recommended_schedules = optimization_grid[: min(5, len(optimization_grid))]

    return {
        "activities_at_irradiation_rows": list(timeline.reference_rows("eoi")),
        "activities_at_count_start_rows": list(timeline.reference_rows("count_start")),
        "activities_at_count_end_rows": list(timeline.reference_rows("count_end")),
        "inventory_timeseries_rows": inventory_rows,
        "dose_endpoints_rows": dose_endpoints,
        "masking_candidates_rows": masking_candidates,
        "masking_isotope_rows": masking_isotopes,
        "alternate_line_rows": alternate_lines,
        "optimization_grid_rows": optimization_grid,
        "recommended_schedule_rows": recommended_schedules,
        "optimization_objective": objective,
    }


def parse_second_irradiation_candidates(
    payload: Mapping[str, Any],
) -> tuple[SecondIrradiationCandidate, ...]:
    """Parse candidate definitions from a JSON-compatible payload."""

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(
            "Second-irradiation candidate payload requires a non-empty 'candidates' list."
        )

    parsed: list[SecondIrradiationCandidate] = []
    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Second-irradiation candidate #{index} must be an object.")
        parsed.append(
            SecondIrradiationCandidate(
                label=str(item.get("label") or f"candidate_{index}"),
                flux_scale=max(_safe_float(item.get("flux_scale"), 1.0), 0.0),
                duration_factor=max(_safe_float(item.get("duration_factor"), 1.0), 0.0),
                second_cooling_time_s=max(
                    _safe_float(item.get("second_cooling_time_s"), 0.0),
                    0.0,
                ),
                pulse_delay_s=max(_safe_float(item.get("pulse_delay_s"), 0.0), 0.0),
            )
        )
    return tuple(parsed)


def build_second_irradiation_candidates(
    *,
    flux_scales: Sequence[float],
    duration_factors: Sequence[float],
    cooling_times_s: Sequence[float],
) -> tuple[SecondIrradiationCandidate, ...]:
    """Build a compact candidate grid for second-irradiation planning."""

    candidates: list[SecondIrradiationCandidate] = []
    for flux_scale in flux_scales:
        for duration_factor in duration_factors:
            for cooling_time_s in cooling_times_s:
                label = (
                    f"flux_{float(flux_scale):.2f}_"
                    f"dur_{float(duration_factor):.2f}_"
                    f"cool_{int(round(float(cooling_time_s)))}s"
                )
                candidates.append(
                    SecondIrradiationCandidate(
                        label=label,
                        flux_scale=max(float(flux_scale), 0.0),
                        duration_factor=max(float(duration_factor), 0.0),
                        second_cooling_time_s=max(float(cooling_time_s), 0.0),
                    )
                )
    return tuple(candidates)


def plan_second_irradiation(
    inventory_state: InventoryState,
    *,
    first_cooling_time_s: float,
    second_irradiation_time_s: float,
    target_weights: Mapping[str, float] | None = None,
    candidates: Sequence[SecondIrradiationCandidate],
) -> SecondIrradiationPlan:
    """Score and rank second-irradiation schedules from an inventory seed."""

    if not inventory_state.seeds:
        raise ValueError("No irradiation-time inventory seeds are available.")
    if not candidates:
        raise ValueError("At least one second-irradiation candidate is required.")

    weights = {str(key): float(value) for key, value in (target_weights or {}).items()}
    ranked: list[SecondIrradiationScore] = []
    for candidate in candidates:
        total_score = 0.0
        rows: list[SecondIrradiationNuclideScore] = []
        for seed in inventory_state.seeds:
            decay_constant = math.log(2.0) / max(float(seed.half_life_s), 1.0e-12)
            activity_after_first = float(seed.activity_eoi_bq) * math.exp(
                -decay_constant * max(float(first_cooling_time_s), 0.0)
            )
            if float(second_irradiation_time_s) > 0.0:
                saturation = 1.0 - math.exp(
                    -decay_constant * max(float(second_irradiation_time_s), 0.0)
                )
            else:
                saturation = 1.0
            replenishment = (
                float(seed.activity_eoi_bq)
                * max(float(candidate.flux_scale), 0.0)
                * max(float(candidate.duration_factor), 0.0)
            )
            activity_after_second = activity_after_first + replenishment * saturation
            activity_at_measurement = activity_after_second * math.exp(
                -decay_constant
                * (
                    max(float(candidate.pulse_delay_s), 0.0)
                    + max(float(candidate.second_cooling_time_s), 0.0)
                )
            )
            weight = float(weights.get(seed.nuclide, weights.get(seed.nuclide.replace("-", ""), 1.0)))
            weighted_activity = weight * activity_at_measurement
            total_score += weighted_activity
            rows.append(
                SecondIrradiationNuclideScore(
                    candidate=str(candidate.label),
                    nuclide=str(seed.nuclide),
                    activity_bq=float(activity_at_measurement),
                    weight=float(weight),
                    weighted_activity=float(weighted_activity),
                )
            )
        rows.sort(key=lambda item: item.nuclide)
        ranked.append(
            SecondIrradiationScore(
                label=str(candidate.label),
                score=float(total_score),
                flux_scale=float(candidate.flux_scale),
                duration_factor=float(candidate.duration_factor),
                second_cooling_time_s=float(candidate.second_cooling_time_s),
                pulse_delay_s=float(candidate.pulse_delay_s),
                nuclide_rows=tuple(rows),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return SecondIrradiationPlan(
        inventory_state=inventory_state,
        first_cooling_time_s=max(float(first_cooling_time_s), 0.0),
        second_irradiation_time_s=max(float(second_irradiation_time_s), 0.0),
        target_weights=weights,
        ranked_candidates=tuple(ranked),
    )


def serialize_second_irradiation_plan(
    plan: SecondIrradiationPlan,
    *,
    inventory_source: str | None = None,
    schedule_source: str | None = None,
    candidates_source: str | None = None,
) -> dict[str, Any]:
    """Serialize second-irradiation planning results to a CLI artifact shape."""

    ranked_rows = [
        candidate.to_row(rank)
        for rank, candidate in enumerate(plan.ranked_candidates, start=1)
    ]
    selected = plan.selected_candidate
    return {
        "schema": "fluxforge.second_irradiation_plan.v1",
        "inventory_source": inventory_source,
        "schedule_source": schedule_source,
        "candidates_source": candidates_source,
        "first_cooling_time_s": float(plan.first_cooling_time_s),
        "second_irradiation_time_s": float(plan.second_irradiation_time_s),
        "target_weights": dict(plan.target_weights),
        "selected_candidate": (
            None
            if selected is None
            else {
                "label": selected.label,
                "score": float(selected.score),
                "flux_scale": float(selected.flux_scale),
                "duration_factor": float(selected.duration_factor),
                "second_cooling_time_s": float(selected.second_cooling_time_s),
                "pulse_delay_s": float(selected.pulse_delay_s),
            }
        ),
        "ranked_candidates": ranked_rows,
        "selected_inventory_rows": plan.selected_inventory_rows(),
    }


__all__ = [
    "SecondIrradiationCandidate",
    "SecondIrradiationNuclideScore",
    "SecondIrradiationPlan",
    "SecondIrradiationScore",
    "build_phase6_support_artifacts",
    "build_second_irradiation_candidates",
    "default_phase6_endpoint_grid",
    "parse_second_irradiation_candidates",
    "plan_second_irradiation",
    "serialize_second_irradiation_plan",
]
