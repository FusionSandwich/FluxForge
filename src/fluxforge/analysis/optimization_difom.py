"""DI-FOM schedule scoring primitives for irradiation optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class DIFOMLineTerm:
    """One nuclide line contribution used in DI-FOM scoring."""

    nuclide: str
    line_energy_keV: float
    signal_counts: float
    background_counts: float = 0.0
    interference_counts: float = 0.0


@dataclass(frozen=True)
class DIFOMLineScore:
    """Scoring breakdown for one nuclide line."""

    nuclide: str
    line_energy_keV: float
    score: float
    weight: float
    weighted_score: float


@dataclass(frozen=True)
class DIFOMEvaluation:
    """Aggregated DI-FOM score with per-line diagnostics."""

    total_score: float
    line_scores: tuple[DIFOMLineScore, ...]


@dataclass(frozen=True)
class DIFOMScheduleCandidate:
    """Schedule candidate represented by one set of expected line terms."""

    label: str
    irradiation_time_s: float
    cooldown_time_s: float
    count_time_s: float
    lines: tuple[DIFOMLineTerm, ...]


@dataclass(frozen=True)
class DIFOMScheduleScore:
    """Scored candidate payload used by ranking and CLI serialization."""

    label: str
    irradiation_time_s: float
    cooldown_time_s: float
    count_time_s: float
    total_score: float
    line_scores: tuple[DIFOMLineScore, ...]


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def difom_line_score(
    signal_counts: float,
    background_counts: float = 0.0,
    interference_counts: float = 0.0,
) -> float:
    """Compute one line-level DI-FOM contribution.

    Uses $s^2 / (s + b + i)$ with nonnegative clipping for all terms.
    """

    signal = _nonnegative(signal_counts)
    background = _nonnegative(background_counts)
    interference = _nonnegative(interference_counts)
    if signal <= 0.0:
        return 0.0
    denominator = signal + background + interference
    if denominator <= _EPSILON:
        return 0.0
    return (signal * signal) / denominator


def evaluate_difom(
    line_terms: Sequence[DIFOMLineTerm],
    *,
    isotope_weights: Mapping[str, float] | None = None,
) -> DIFOMEvaluation:
    """Evaluate DI-FOM for a collection of line terms."""

    weights = isotope_weights or {}
    score_rows: list[DIFOMLineScore] = []
    total = 0.0
    for line in line_terms:
        base_score = difom_line_score(
            line.signal_counts,
            line.background_counts,
            line.interference_counts,
        )
        weight = _nonnegative(float(weights.get(line.nuclide, 1.0)))
        weighted = base_score * weight
        score_rows.append(
            DIFOMLineScore(
                nuclide=line.nuclide,
                line_energy_keV=float(line.line_energy_keV),
                score=float(base_score),
                weight=float(weight),
                weighted_score=float(weighted),
            )
        )
        total += weighted
    return DIFOMEvaluation(total_score=float(total), line_scores=tuple(score_rows))


def compute_difom_score(
    line_terms: Sequence[DIFOMLineTerm],
    *,
    isotope_weights: Mapping[str, float] | None = None,
) -> float:
    """Return only the scalar DI-FOM score for convenience call sites."""

    return evaluate_difom(
        line_terms,
        isotope_weights=isotope_weights,
    ).total_score


def rank_difom_schedules(
    candidates: Sequence[DIFOMScheduleCandidate],
    *,
    isotope_weights: Mapping[str, float] | None = None,
) -> tuple[DIFOMScheduleScore, ...]:
    """Score and rank schedule candidates from highest to lowest DI-FOM."""

    ranked: list[DIFOMScheduleScore] = []
    for candidate in candidates:
        evaluation = evaluate_difom(
            candidate.lines,
            isotope_weights=isotope_weights,
        )
        ranked.append(
            DIFOMScheduleScore(
                label=str(candidate.label),
                irradiation_time_s=float(candidate.irradiation_time_s),
                cooldown_time_s=float(candidate.cooldown_time_s),
                count_time_s=float(candidate.count_time_s),
                total_score=float(evaluation.total_score),
                line_scores=evaluation.line_scores,
            )
        )
    ranked.sort(key=lambda item: item.total_score, reverse=True)
    return tuple(ranked)


def _parse_float(value: Any, *, field_name: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires numeric {field_name!r}.") from exc


def _parse_line_term(item: Any, *, schedule_label: str, index: int) -> DIFOMLineTerm:
    context = f"Candidate {schedule_label!r} line #{index + 1}"
    if not isinstance(item, Mapping):
        raise ValueError(f"{context} must be an object.")

    nuclide = str(item.get("nuclide") or item.get("isotope") or "").strip()
    if not nuclide:
        raise ValueError(f"{context} requires 'nuclide' (or 'isotope').")

    line_energy_keV = _parse_float(
        item.get("line_energy_keV", item.get("energy_keV", 0.0)),
        field_name="line_energy_keV",
        context=context,
    )

    signal_raw = item.get("signal_counts", item.get("expected_signal_counts", item.get("signal")))
    if signal_raw is None:
        raise ValueError(
            f"{context} requires 'signal_counts' (or 'expected_signal_counts')."
        )

    return DIFOMLineTerm(
        nuclide=nuclide,
        line_energy_keV=float(line_energy_keV),
        signal_counts=_parse_float(
            signal_raw,
            field_name="signal_counts",
            context=context,
        ),
        background_counts=_parse_float(
            item.get("background_counts", item.get("expected_background_counts", 0.0)),
            field_name="background_counts",
            context=context,
        ),
        interference_counts=_parse_float(
            item.get("interference_counts", item.get("expected_interference_counts", 0.0)),
            field_name="interference_counts",
            context=context,
        ),
    )


def _parse_schedule_candidate(item: Any, *, index: int) -> DIFOMScheduleCandidate:
    if not isinstance(item, Mapping):
        raise ValueError(f"Candidate #{index + 1} must be an object.")

    label = str(item.get("label") or f"candidate_{index + 1}")
    lines_raw = item.get("lines")
    if not isinstance(lines_raw, list) or len(lines_raw) == 0:
        raise ValueError(f"Candidate {label!r} requires a non-empty 'lines' list.")

    lines = tuple(
        _parse_line_term(line_item, schedule_label=label, index=line_index)
        for line_index, line_item in enumerate(lines_raw)
    )
    return DIFOMScheduleCandidate(
        label=label,
        irradiation_time_s=_parse_float(
            item.get("irradiation_time_s", 0.0),
            field_name="irradiation_time_s",
            context=f"Candidate {label!r}",
        ),
        cooldown_time_s=_parse_float(
            item.get("cooldown_time_s", 0.0),
            field_name="cooldown_time_s",
            context=f"Candidate {label!r}",
        ),
        count_time_s=_parse_float(
            item.get("count_time_s", 0.0),
            field_name="count_time_s",
            context=f"Candidate {label!r}",
        ),
        lines=lines,
    )


def parse_difom_sweep_payload(
    payload: Mapping[str, Any],
) -> tuple[tuple[DIFOMScheduleCandidate, ...], dict[str, float]]:
    """Parse JSON payload consumed by the CLI optimization sweep command."""

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
        raise ValueError("Optimization payload requires a non-empty 'candidates' list.")

    candidates = tuple(
        _parse_schedule_candidate(item, index=index)
        for index, item in enumerate(candidates_raw)
    )

    weights_raw = payload.get("isotope_weights") or {}
    if not isinstance(weights_raw, Mapping):
        raise ValueError("'isotope_weights' must be a mapping of nuclide -> weight.")

    isotope_weights: dict[str, float] = {}
    for key, value in weights_raw.items():
        nuclide = str(key).strip()
        if not nuclide:
            continue
        isotope_weights[nuclide] = _nonnegative(
            _parse_float(value, field_name="weight", context="isotope_weights")
        )

    return candidates, isotope_weights


def serialize_difom_ranking(
    ranked: Sequence[DIFOMScheduleScore],
) -> dict[str, Any]:
    """Serialize DI-FOM ranking output into the CLI bundle shape."""

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank_index,
                "label": item.label,
                "irradiation_time_s": item.irradiation_time_s,
                "cooldown_time_s": item.cooldown_time_s,
                "count_time_s": item.count_time_s,
                "difom_score": item.total_score,
                "line_scores": [
                    {
                        "nuclide": line.nuclide,
                        "line_energy_keV": line.line_energy_keV,
                        "score": line.score,
                        "weight": line.weight,
                        "weighted_score": line.weighted_score,
                    }
                    for line in item.line_scores
                ],
            }
        )
    return {
        "schema": "fluxforge.optimization_sweep.difom.v1",
        "objective": "di-fom",
        "ranked_candidates": rows,
    }


def build_difom_terms_from_activity_results(
    activity_results: Sequence[Any],
) -> tuple[DIFOMLineTerm, ...]:
    """Build DI-FOM line terms from activity-review outputs.

    The GUI preview uses age-corrected activity as proxy signal and corresponding
    uncertainty as proxy background until full transport/line-interference coupling
    is implemented.
    """

    terms: list[DIFOMLineTerm] = []
    for result in activity_results:
        nuclide = str(getattr(result, "nuclide", "")).strip()
        if not nuclide:
            continue
        signal = _nonnegative(
            getattr(
                result,
                "age_corrected_activity_bq",
                getattr(result, "activity_bq", 0.0),
            )
        )
        if signal <= 0.0:
            continue
        uncertainty = _nonnegative(
            getattr(
                result,
                "age_corrected_uncertainty_bq",
                getattr(result, "uncertainty_bq", 0.0),
            )
        )
        terms.append(
            DIFOMLineTerm(
                nuclide=nuclide,
                line_energy_keV=float(getattr(result, "line_energy_keV", 0.0) or 0.0),
                signal_counts=float(signal),
                background_counts=float(uncertainty),
                interference_counts=0.0,
            )
        )
    return tuple(terms)


__all__ = [
    "DIFOMLineTerm",
    "DIFOMLineScore",
    "DIFOMEvaluation",
    "DIFOMScheduleCandidate",
    "DIFOMScheduleScore",
    "build_difom_terms_from_activity_results",
    "compute_difom_score",
    "difom_line_score",
    "evaluate_difom",
    "parse_difom_sweep_payload",
    "rank_difom_schedules",
    "serialize_difom_ranking",
]
