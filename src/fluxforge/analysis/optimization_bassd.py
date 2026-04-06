"""Bayesian adaptive schedule scoring with dose-aware utility."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from fluxforge.analysis.optimization_difom import DIFOMScheduleCandidate

_EPSILON = 1.0e-12
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class BASSDLineState:
    """Belief state for one gamma line before an adaptive action."""

    nuclide: str
    line_energy_keV: float
    prior_mean_counts: float
    prior_variance_counts2: float
    signal_counts: float
    background_counts: float = 0.0
    dose_rate_uSv_h: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class BASSDAction:
    """One adaptive action (cooldown/count window) to evaluate."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    lines: tuple[BASSDLineState, ...]


@dataclass(frozen=True)
class BASSDActionScore:
    """Scored utility contributions for one adaptive action."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    value_of_information: float
    expected_dose_uSv: float
    posterior_variance_reduction: float
    utility_score: float


@dataclass(frozen=True)
class BASSDEvaluation:
    """Full action trace score for one schedule candidate."""

    total_utility: float
    action_scores: tuple[BASSDActionScore, ...]


@dataclass(frozen=True)
class BASSDScheduleCandidate:
    """Adaptive schedule candidate containing ordered actions."""

    label: str
    irradiation_time_s: float
    actions: tuple[BASSDAction, ...]


@dataclass(frozen=True)
class BASSDScheduleScore:
    """Ranking row for one BASS-D schedule candidate."""

    label: str
    irradiation_time_s: float
    total_utility: float
    action_scores: tuple[BASSDActionScore, ...]


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(float(value) / sqrt(2.0)))


def posterior_variance_update(
    prior_variance_counts2: float,
    signal_counts: float,
    background_counts: float,
    *,
    variance_floor: float = 1.0e-9,
) -> float:
    """One-step scalar Bayesian variance update with Poisson-like observation noise."""

    prior_variance = max(float(prior_variance_counts2), variance_floor)
    observation_variance = max(
        _nonnegative(signal_counts) + _nonnegative(background_counts),
        variance_floor,
    )
    posterior_precision = (1.0 / prior_variance) + (1.0 / observation_variance)
    posterior_variance = 1.0 / max(posterior_precision, _EPSILON)
    return max(float(posterior_variance), variance_floor)


def _value_of_information(prior_variance: float, posterior_variance: float) -> float:
    prior = max(float(prior_variance), _EPSILON)
    posterior = max(float(posterior_variance), _EPSILON)
    if posterior >= prior:
        return 0.0
    return 0.5 * float(np.log(prior / posterior))


def _detection_probability(signal_counts: float, background_counts: float) -> float:
    signal = _nonnegative(signal_counts)
    if signal <= 0.0:
        return 0.0
    denominator = sqrt(max(signal + _nonnegative(background_counts), 1.0))
    z_score = signal / denominator
    return float(_normal_cdf(z_score - 1.0))


def evaluate_bassd(
    actions: Sequence[BASSDAction],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    dose_weight: float = 0.02,
    exploration_temperature: float = 0.0,
    seed: int = 17,
) -> BASSDEvaluation:
    """Evaluate an adaptive action sequence using dose-weighted expected utility."""

    rng = np.random.default_rng(int(seed))
    weights = isotope_weights or {}
    dose_weight_value = max(float(dose_weight), 0.0)
    exploration_value = max(float(exploration_temperature), 0.0)

    posterior_by_line: dict[tuple[str, float], float] = {}
    total_utility = 0.0
    action_scores: list[BASSDActionScore] = []

    for action in actions:
        value_of_information = 0.0
        dose_uSv = 0.0
        variance_reduction = 0.0

        for line in action.lines:
            key = (str(line.nuclide), float(line.line_energy_keV))
            prior_variance = posterior_by_line.get(
                key,
                max(float(line.prior_variance_counts2), 1.0e-9),
            )
            posterior_variance = posterior_variance_update(
                prior_variance,
                line.signal_counts,
                line.background_counts,
            )
            posterior_by_line[key] = posterior_variance

            voi = _value_of_information(prior_variance, posterior_variance)
            detect_prob = _detection_probability(
                line.signal_counts,
                line.background_counts,
            )
            weight = _nonnegative(float(weights.get(line.nuclide, line.weight)))
            value_of_information += voi * detect_prob * weight
            variance_reduction += max(prior_variance - posterior_variance, 0.0) * weight

            dose_uSv += (
                _nonnegative(line.dose_rate_uSv_h)
                * max(float(action.count_time_s), 0.0)
                / _SECONDS_PER_HOUR
            )

        stochastic_bonus = 0.0
        if exploration_value > 0.0:
            stochastic_bonus = float(rng.normal(loc=0.0, scale=exploration_value))

        utility = value_of_information - dose_weight_value * dose_uSv + stochastic_bonus
        total_utility += utility
        action_scores.append(
            BASSDActionScore(
                label=str(action.label),
                cooldown_time_s=float(action.cooldown_time_s),
                count_time_s=float(action.count_time_s),
                value_of_information=float(value_of_information),
                expected_dose_uSv=float(dose_uSv),
                posterior_variance_reduction=float(variance_reduction),
                utility_score=float(utility),
            )
        )

    return BASSDEvaluation(
        total_utility=float(total_utility),
        action_scores=tuple(action_scores),
    )


def rank_bassd_schedules(
    candidates: Sequence[BASSDScheduleCandidate],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    dose_weight: float = 0.02,
    exploration_temperature: float = 0.0,
    seed: int = 17,
) -> tuple[BASSDScheduleScore, ...]:
    """Score and rank schedules under the BASS-D adaptive utility objective."""

    ranked: list[BASSDScheduleScore] = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = evaluate_bassd(
            candidate.actions,
            isotope_weights=isotope_weights,
            dose_weight=dose_weight,
            exploration_temperature=exploration_temperature,
            seed=int(seed) + index,
        )
        ranked.append(
            BASSDScheduleScore(
                label=str(candidate.label),
                irradiation_time_s=float(candidate.irradiation_time_s),
                total_utility=float(evaluation.total_utility),
                action_scores=evaluation.action_scores,
            )
        )
    ranked.sort(key=lambda item: item.total_utility, reverse=True)
    return tuple(ranked)


def _parse_float(value: Any, *, field_name: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires numeric {field_name!r}.") from exc


def _parse_line_state(item: Any, *, context: str) -> BASSDLineState:
    if not isinstance(item, Mapping):
        raise ValueError(f"{context} line entries must be objects.")

    nuclide = str(item.get("nuclide") or item.get("isotope") or "").strip()
    if not nuclide:
        raise ValueError(f"{context} requires line 'nuclide' or 'isotope'.")

    prior_variance = item.get("prior_variance_counts2")
    if prior_variance is None:
        prior_uncertainty = item.get("prior_uncertainty_counts")
        if prior_uncertainty is not None:
            prior_variance = float(prior_uncertainty) ** 2
        else:
            prior_variance = max(
                _parse_float(
                    item.get("signal_counts", item.get("expected_signal_counts", 0.0)),
                    field_name="signal_counts",
                    context=context,
                ),
                1.0,
            )

    return BASSDLineState(
        nuclide=nuclide,
        line_energy_keV=_parse_float(
            item.get("line_energy_keV", item.get("energy_keV", 0.0)),
            field_name="line_energy_keV",
            context=context,
        ),
        prior_mean_counts=_parse_float(
            item.get("prior_mean_counts", item.get("signal_counts", 0.0)),
            field_name="prior_mean_counts",
            context=context,
        ),
        prior_variance_counts2=max(float(prior_variance), 1.0e-9),
        signal_counts=_parse_float(
            item.get("signal_counts", item.get("expected_signal_counts", 0.0)),
            field_name="signal_counts",
            context=context,
        ),
        background_counts=_parse_float(
            item.get("background_counts", item.get("expected_background_counts", 0.0)),
            field_name="background_counts",
            context=context,
        ),
        dose_rate_uSv_h=_parse_float(
            item.get("dose_rate_uSv_h", 0.0),
            field_name="dose_rate_uSv_h",
            context=context,
        ),
        weight=max(
            _parse_float(item.get("weight", 1.0), field_name="weight", context=context),
            0.0,
        ),
    )


def parse_bassd_sweep_payload(
    payload: Mapping[str, Any],
    *,
    default_count_time_s: float = 900.0,
) -> tuple[tuple[BASSDScheduleCandidate, ...], dict[str, float]]:
    """Parse payload consumed by CLI optimization-sweep for bass-d objective."""

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
        raise ValueError("Optimization payload requires a non-empty 'candidates' list.")

    candidates: list[BASSDScheduleCandidate] = []
    for index, item in enumerate(candidates_raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Candidate #{index} must be an object.")

        label = str(item.get("label") or f"candidate_{index}")
        irradiation_time_s = _parse_float(
            item.get("irradiation_time_s", 0.0),
            field_name="irradiation_time_s",
            context=f"Candidate {label!r}",
        )

        actions_raw = item.get("actions")
        actions: list[BASSDAction] = []
        if isinstance(actions_raw, list) and len(actions_raw) > 0:
            for action_index, action_raw in enumerate(actions_raw, start=1):
                action_context = f"Candidate {label!r} action #{action_index}"
                if not isinstance(action_raw, Mapping):
                    raise ValueError(f"{action_context} must be an object.")
                lines_raw = action_raw.get("lines")
                if not isinstance(lines_raw, list) or len(lines_raw) == 0:
                    raise ValueError(f"{action_context} requires a non-empty lines list.")
                lines = tuple(
                    _parse_line_state(line_item, context=action_context)
                    for line_item in lines_raw
                )
                actions.append(
                    BASSDAction(
                        label=str(action_raw.get("label") or f"action_{action_index}"),
                        cooldown_time_s=_parse_float(
                            action_raw.get("cooldown_time_s", 0.0),
                            field_name="cooldown_time_s",
                            context=action_context,
                        ),
                        count_time_s=_parse_float(
                            action_raw.get("count_time_s", default_count_time_s),
                            field_name="count_time_s",
                            context=action_context,
                        ),
                        lines=lines,
                    )
                )
        else:
            lines_raw = item.get("lines")
            if not isinstance(lines_raw, list) or len(lines_raw) == 0:
                raise ValueError(
                    f"Candidate {label!r} requires either non-empty 'actions' or 'lines'."
                )
            lines = tuple(
                _parse_line_state(line_item, context=f"Candidate {label!r}")
                for line_item in lines_raw
            )
            actions.append(
                BASSDAction(
                    label="action_1",
                    cooldown_time_s=_parse_float(
                        item.get("cooldown_time_s", 0.0),
                        field_name="cooldown_time_s",
                        context=f"Candidate {label!r}",
                    ),
                    count_time_s=_parse_float(
                        item.get("count_time_s", default_count_time_s),
                        field_name="count_time_s",
                        context=f"Candidate {label!r}",
                    ),
                    lines=lines,
                )
            )

        candidates.append(
            BASSDScheduleCandidate(
                label=label,
                irradiation_time_s=float(irradiation_time_s),
                actions=tuple(actions),
            )
        )

    weights_raw = payload.get("isotope_weights") or {}
    if not isinstance(weights_raw, Mapping):
        raise ValueError("'isotope_weights' must be a mapping of nuclide -> weight.")

    isotope_weights: dict[str, float] = {}
    for key, value in weights_raw.items():
        nuclide = str(key).strip()
        if not nuclide:
            continue
        isotope_weights[nuclide] = max(
            _parse_float(value, field_name="weight", context="isotope_weights"),
            0.0,
        )

    return tuple(candidates), isotope_weights


def serialize_bassd_ranking(
    ranked: Sequence[BASSDScheduleScore],
) -> dict[str, Any]:
    """Serialize BASS-D ranking output into CLI bundle shape."""

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank_index,
                "label": item.label,
                "irradiation_time_s": item.irradiation_time_s,
                "total_utility": item.total_utility,
                "action_scores": [
                    {
                        "label": action.label,
                        "cooldown_time_s": action.cooldown_time_s,
                        "count_time_s": action.count_time_s,
                        "value_of_information": action.value_of_information,
                        "expected_dose_uSv": action.expected_dose_uSv,
                        "posterior_variance_reduction": action.posterior_variance_reduction,
                        "utility_score": action.utility_score,
                    }
                    for action in item.action_scores
                ],
            }
        )

    return {
        "schema": "fluxforge.optimization_sweep.bassd.v1",
        "objective": "bass-d",
        "ranked_candidates": rows,
    }


def build_bassd_candidate_from_activity_results(
    activity_results: Sequence[Any],
    *,
    label: str = "activity_preview",
    irradiation_time_s: float = 0.0,
    cooldown_time_s: float = 0.0,
    count_time_s: float = 900.0,
    dose_scale_uSv_h_per_Bq: float = 1.0e-6,
) -> BASSDScheduleCandidate:
    """Build a preview candidate from activity-review style rows for GUI usage."""

    lines: list[BASSDLineState] = []
    for result in activity_results:
        nuclide = str(getattr(result, "nuclide", "")).strip()
        if not nuclide:
            continue
        mean_counts = _nonnegative(
            getattr(
                result,
                "age_corrected_activity_bq",
                getattr(result, "activity_bq", 0.0),
            )
        )
        if mean_counts <= 0.0:
            continue
        uncertainty = _nonnegative(
            getattr(
                result,
                "age_corrected_uncertainty_bq",
                getattr(result, "uncertainty_bq", 0.0),
            )
        )
        variance = max(float(uncertainty) ** 2, 1.0)
        lines.append(
            BASSDLineState(
                nuclide=nuclide,
                line_energy_keV=float(getattr(result, "line_energy_keV", 0.0) or 0.0),
                prior_mean_counts=float(mean_counts),
                prior_variance_counts2=float(variance),
                signal_counts=float(mean_counts),
                background_counts=float(uncertainty),
                dose_rate_uSv_h=float(mean_counts) * float(dose_scale_uSv_h_per_Bq),
                weight=1.0,
            )
        )

    action = BASSDAction(
        label="action_1",
        cooldown_time_s=max(float(cooldown_time_s), 0.0),
        count_time_s=max(float(count_time_s), 1.0),
        lines=tuple(lines),
    )
    return BASSDScheduleCandidate(
        label=str(label),
        irradiation_time_s=float(irradiation_time_s),
        actions=(action,),
    )


def candidate_from_difom_candidate(
    candidate: DIFOMScheduleCandidate,
    *,
    action_count: int = 2,
    count_time_s: float = 900.0,
) -> BASSDScheduleCandidate:
    """Build a baseline BASS-D candidate from a DI-FOM candidate."""

    n_actions = max(int(action_count), 1)
    actions: list[BASSDAction] = []
    for action_index in range(n_actions):
        cooldown = float(action_index) * 7200.0
        lines = tuple(
            BASSDLineState(
                nuclide=line.nuclide,
                line_energy_keV=float(line.line_energy_keV),
                prior_mean_counts=float(line.signal_counts),
                prior_variance_counts2=max(float(line.signal_counts + line.background_counts), 1.0),
                signal_counts=float(line.signal_counts),
                background_counts=float(line.background_counts + line.interference_counts),
                dose_rate_uSv_h=max(float(line.signal_counts) * 1.0e-4, 0.0),
                weight=1.0,
            )
            for line in candidate.lines
        )
        actions.append(
            BASSDAction(
                label=f"action_{action_index + 1}",
                cooldown_time_s=cooldown,
                count_time_s=max(float(count_time_s), 1.0),
                lines=lines,
            )
        )

    return BASSDScheduleCandidate(
        label=str(candidate.label),
        irradiation_time_s=float(candidate.irradiation_time_s),
        actions=tuple(actions),
    )


__all__ = [
    "BASSDAction",
    "BASSDActionScore",
    "BASSDLineState",
    "BASSDEvaluation",
    "BASSDScheduleCandidate",
    "BASSDScheduleScore",
    "build_bassd_candidate_from_activity_results",
    "candidate_from_difom_candidate",
    "evaluate_bassd",
    "parse_bassd_sweep_payload",
    "posterior_variance_update",
    "rank_bassd_schedules",
    "serialize_bassd_ranking",
]
