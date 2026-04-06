"""Multi-window decay-curve schedule scoring with optional full-spectrum penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fluxforge.analysis.optimization_difom import DIFOMScheduleCandidate

LN2 = np.log(2.0)
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class MWDCSLineTerm:
    """One line contribution inside one cooldown/count window."""

    nuclide: str
    line_energy_keV: float
    signal_counts: float
    background_counts: float = 0.0
    interference_counts: float = 0.0
    half_life_s: float | None = None


@dataclass(frozen=True)
class MWDCSWindow:
    """One planned cooldown/count window in a schedule."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    lines: tuple[MWDCSLineTerm, ...]


@dataclass(frozen=True)
class MWDCSWindowScore:
    """Score breakdown for one window."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    base_score: float
    overlap_penalty: float
    marginal_score: float


@dataclass(frozen=True)
class MWDCSEvaluation:
    """Aggregated MWDCS score and per-window diagnostics."""

    total_score: float
    window_scores: tuple[MWDCSWindowScore, ...]


@dataclass(frozen=True)
class MWDCSScheduleCandidate:
    """Schedule candidate composed of multiple windows."""

    label: str
    irradiation_time_s: float
    windows: tuple[MWDCSWindow, ...]


@dataclass(frozen=True)
class MWDCSScheduleScore:
    """Scored schedule payload for ranking/serialization."""

    label: str
    irradiation_time_s: float
    total_score: float
    full_spectrum_mode: bool
    window_scores: tuple[MWDCSWindowScore, ...]


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def _line_information(
    signal_counts: float,
    background_counts: float,
    interference_counts: float,
) -> float:
    signal = _nonnegative(signal_counts)
    if signal <= 0.0:
        return 0.0
    denominator = signal + _nonnegative(background_counts) + _nonnegative(interference_counts)
    if denominator <= _EPSILON:
        return 0.0
    return (signal * signal) / denominator


def _window_overlap_penalty(
    lines: Sequence[MWDCSLineTerm],
    *,
    overlap_penalty: float,
    energy_tolerance_keV: float = 3.0,
) -> float:
    penalty_scale = max(float(overlap_penalty), 0.0)
    if penalty_scale <= 0.0 or len(lines) < 2:
        return 0.0

    penalty = 0.0
    for idx in range(len(lines)):
        left = lines[idx]
        for jdx in range(idx + 1, len(lines)):
            right = lines[jdx]
            if abs(float(left.line_energy_keV) - float(right.line_energy_keV)) > energy_tolerance_keV:
                continue
            if left.nuclide == right.nuclide:
                continue
            left_info = _line_information(
                left.signal_counts,
                left.background_counts,
                left.interference_counts,
            )
            right_info = _line_information(
                right.signal_counts,
                right.background_counts,
                right.interference_counts,
            )
            penalty += penalty_scale * min(left_info, right_info)
    return penalty


def evaluate_mwdcs(
    windows: Sequence[MWDCSWindow],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    full_spectrum_mode: bool = False,
    overlap_penalty: float = 0.0,
) -> MWDCSEvaluation:
    """Evaluate additive information over multiple windows with diminishing returns."""

    weights = isotope_weights or {}
    seen_nuclides: dict[str, int] = {}
    total = 0.0
    score_rows: list[MWDCSWindowScore] = []

    for window in windows:
        base = 0.0
        for line in window.lines:
            info = _line_information(
                line.signal_counts,
                line.background_counts,
                line.interference_counts,
            )
            if info <= 0.0:
                continue
            weight = _nonnegative(float(weights.get(line.nuclide, 1.0)))
            seen = seen_nuclides.get(line.nuclide, 0)
            diminishing_factor = 1.0 / np.sqrt(1.0 + float(seen))
            base += info * weight * diminishing_factor
            seen_nuclides[line.nuclide] = seen + 1

        penalty = 0.0
        if full_spectrum_mode:
            penalty = _window_overlap_penalty(
                window.lines,
                overlap_penalty=overlap_penalty,
            )

        marginal = max(base - penalty, 0.0)
        total += marginal
        score_rows.append(
            MWDCSWindowScore(
                label=str(window.label),
                cooldown_time_s=float(window.cooldown_time_s),
                count_time_s=float(window.count_time_s),
                base_score=float(base),
                overlap_penalty=float(penalty),
                marginal_score=float(marginal),
            )
        )

    return MWDCSEvaluation(total_score=float(total), window_scores=tuple(score_rows))


def rank_mwdcs_schedules(
    candidates: Sequence[MWDCSScheduleCandidate],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    full_spectrum_mode: bool = False,
    overlap_penalty: float = 0.0,
) -> tuple[MWDCSScheduleScore, ...]:
    """Score and rank schedules under the MWDCS objective."""

    ranked: list[MWDCSScheduleScore] = []
    for candidate in candidates:
        evaluation = evaluate_mwdcs(
            candidate.windows,
            isotope_weights=isotope_weights,
            full_spectrum_mode=full_spectrum_mode,
            overlap_penalty=overlap_penalty,
        )
        ranked.append(
            MWDCSScheduleScore(
                label=str(candidate.label),
                irradiation_time_s=float(candidate.irradiation_time_s),
                total_score=float(evaluation.total_score),
                full_spectrum_mode=bool(full_spectrum_mode),
                window_scores=evaluation.window_scores,
            )
        )
    ranked.sort(key=lambda item: item.total_score, reverse=True)
    return tuple(ranked)


def _parse_float(value: Any, *, field_name: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} requires numeric {field_name!r}.") from exc


def _parse_line_term(item: Any, *, context: str) -> MWDCSLineTerm:
    if not isinstance(item, Mapping):
        raise ValueError(f"{context} line entries must be objects.")
    nuclide = str(item.get("nuclide") or item.get("isotope") or "").strip()
    if not nuclide:
        raise ValueError(f"{context} requires line 'nuclide' or 'isotope'.")

    half_life_raw = item.get("half_life_s")
    half_life_s = None
    if half_life_raw is not None:
        parsed_half_life = _parse_float(
            half_life_raw,
            field_name="half_life_s",
            context=context,
        )
        if parsed_half_life > 0.0:
            half_life_s = float(parsed_half_life)

    return MWDCSLineTerm(
        nuclide=nuclide,
        line_energy_keV=_parse_float(
            item.get("line_energy_keV", item.get("energy_keV", 0.0)),
            field_name="line_energy_keV",
            context=context,
        ),
        signal_counts=_parse_float(
            item.get("signal_counts", item.get("expected_signal_counts", item.get("signal", 0.0))),
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
        half_life_s=half_life_s,
    )


def _project_signal(signal: float, half_life_s: float | None, cooldown_time_s: float) -> float:
    if signal <= 0.0:
        return 0.0
    if half_life_s is None or half_life_s <= 0.0:
        return float(signal)
    lam = LN2 / float(half_life_s)
    return float(signal) * float(np.exp(-lam * max(float(cooldown_time_s), 0.0)))


def _windows_from_base_lines(
    lines: Sequence[MWDCSLineTerm],
    *,
    offsets_s: Sequence[float],
    window_count_time_s: float,
) -> tuple[MWDCSWindow, ...]:
    windows: list[MWDCSWindow] = []
    for index, offset in enumerate(offsets_s, start=1):
        cooldown = max(float(offset), 0.0)
        projected_lines: list[MWDCSLineTerm] = []
        for line in lines:
            projected_signal = _project_signal(
                line.signal_counts,
                line.half_life_s,
                cooldown,
            )
            projected_lines.append(
                MWDCSLineTerm(
                    nuclide=line.nuclide,
                    line_energy_keV=float(line.line_energy_keV),
                    signal_counts=projected_signal,
                    background_counts=float(line.background_counts),
                    interference_counts=float(line.interference_counts),
                    half_life_s=line.half_life_s,
                )
            )
        windows.append(
            MWDCSWindow(
                label=f"window_{index}",
                cooldown_time_s=cooldown,
                count_time_s=max(float(window_count_time_s), _EPSILON),
                lines=tuple(projected_lines),
            )
        )
    return tuple(windows)


def parse_mwdcs_sweep_payload(
    payload: Mapping[str, Any],
    *,
    default_window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    default_window_count_time_s: float = 900.0,
) -> tuple[tuple[MWDCSScheduleCandidate, ...], dict[str, float]]:
    """Parse payload consumed by CLI optimization-sweep for mwdcs objective."""

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
        raise ValueError("Optimization payload requires a non-empty 'candidates' list.")

    offsets = tuple(float(max(item, 0.0)) for item in default_window_offsets_s)
    if len(offsets) == 0:
        raise ValueError("MWDCS requires at least one window offset.")

    candidates: list[MWDCSScheduleCandidate] = []
    for index, item in enumerate(candidates_raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Candidate #{index} must be an object.")

        label = str(item.get("label") or f"candidate_{index}")
        irradiation_time_s = _parse_float(
            item.get("irradiation_time_s", 0.0),
            field_name="irradiation_time_s",
            context=f"Candidate {label!r}",
        )

        windows_raw = item.get("windows")
        if isinstance(windows_raw, list) and len(windows_raw) > 0:
            windows: list[MWDCSWindow] = []
            for window_index, window_raw in enumerate(windows_raw, start=1):
                window_context = f"Candidate {label!r} window #{window_index}"
                if not isinstance(window_raw, Mapping):
                    raise ValueError(f"{window_context} must be an object.")
                lines_raw = window_raw.get("lines")
                if not isinstance(lines_raw, list) or len(lines_raw) == 0:
                    raise ValueError(f"{window_context} requires a non-empty lines list.")
                lines = tuple(
                    _parse_line_term(line_item, context=window_context)
                    for line_item in lines_raw
                )
                windows.append(
                    MWDCSWindow(
                        label=str(window_raw.get("label") or f"window_{window_index}"),
                        cooldown_time_s=_parse_float(
                            window_raw.get("cooldown_time_s", 0.0),
                            field_name="cooldown_time_s",
                            context=window_context,
                        ),
                        count_time_s=_parse_float(
                            window_raw.get("count_time_s", default_window_count_time_s),
                            field_name="count_time_s",
                            context=window_context,
                        ),
                        lines=lines,
                    )
                )
        else:
            lines_raw = item.get("lines")
            if not isinstance(lines_raw, list) or len(lines_raw) == 0:
                raise ValueError(
                    f"Candidate {label!r} requires either non-empty 'windows' or 'lines'."
                )
            base_lines = tuple(
                _parse_line_term(line_item, context=f"Candidate {label!r}")
                for line_item in lines_raw
            )
            windows = list(
                _windows_from_base_lines(
                    base_lines,
                    offsets_s=offsets,
                    window_count_time_s=default_window_count_time_s,
                )
            )

        candidates.append(
            MWDCSScheduleCandidate(
                label=label,
                irradiation_time_s=float(irradiation_time_s),
                windows=tuple(windows),
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
        isotope_weights[nuclide] = _nonnegative(
            _parse_float(value, field_name="weight", context="isotope_weights")
        )

    return tuple(candidates), isotope_weights


def serialize_mwdcs_ranking(
    ranked: Sequence[MWDCSScheduleScore],
) -> dict[str, Any]:
    """Serialize MWDCS ranking output into CLI bundle shape."""

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank_index,
                "label": item.label,
                "irradiation_time_s": item.irradiation_time_s,
                "total_score": item.total_score,
                "full_spectrum_mode": item.full_spectrum_mode,
                "window_scores": [
                    {
                        "label": window.label,
                        "cooldown_time_s": window.cooldown_time_s,
                        "count_time_s": window.count_time_s,
                        "base_score": window.base_score,
                        "overlap_penalty": window.overlap_penalty,
                        "marginal_score": window.marginal_score,
                    }
                    for window in item.window_scores
                ],
            }
        )

    return {
        "schema": "fluxforge.optimization_sweep.mwdcs.v1",
        "objective": "mwdcs",
        "ranked_candidates": rows,
    }


def candidate_from_difom_candidate(
    candidate: DIFOMScheduleCandidate,
    *,
    window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    window_count_time_s: float = 900.0,
) -> MWDCSScheduleCandidate:
    """Construct an MWDCS candidate from a DI-FOM candidate definition."""

    base_lines = tuple(
        MWDCSLineTerm(
            nuclide=line.nuclide,
            line_energy_keV=float(line.line_energy_keV),
            signal_counts=float(line.signal_counts),
            background_counts=float(line.background_counts),
            interference_counts=float(line.interference_counts),
            half_life_s=None,
        )
        for line in candidate.lines
    )
    windows = _windows_from_base_lines(
        base_lines,
        offsets_s=window_offsets_s,
        window_count_time_s=window_count_time_s,
    )
    return MWDCSScheduleCandidate(
        label=str(candidate.label),
        irradiation_time_s=float(candidate.irradiation_time_s),
        windows=windows,
    )


def build_mwdcs_candidate_from_activity_results(
    activity_results: Sequence[Any],
    *,
    label: str = "activity_preview",
    irradiation_time_s: float = 0.0,
    window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    window_count_time_s: float = 900.0,
) -> MWDCSScheduleCandidate:
    """Build a preview MWDCS candidate from activity-review style rows."""

    base_lines: list[MWDCSLineTerm] = []
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
        background = _nonnegative(
            getattr(
                result,
                "age_corrected_uncertainty_bq",
                getattr(result, "uncertainty_bq", 0.0),
            )
        )
        half_life_value = getattr(result, "half_life_s", None)
        half_life_s = None
        if half_life_value is not None:
            parsed = float(half_life_value)
            if parsed > 0.0:
                half_life_s = parsed
        base_lines.append(
            MWDCSLineTerm(
                nuclide=nuclide,
                line_energy_keV=float(getattr(result, "line_energy_keV", 0.0) or 0.0),
                signal_counts=float(signal),
                background_counts=float(background),
                interference_counts=0.0,
                half_life_s=half_life_s,
            )
        )

    windows = _windows_from_base_lines(
        tuple(base_lines),
        offsets_s=window_offsets_s,
        window_count_time_s=window_count_time_s,
    )
    return MWDCSScheduleCandidate(
        label=str(label),
        irradiation_time_s=float(irradiation_time_s),
        windows=windows,
    )


__all__ = [
    "MWDCSLineTerm",
    "MWDCSWindow",
    "MWDCSWindowScore",
    "MWDCSEvaluation",
    "MWDCSScheduleCandidate",
    "MWDCSScheduleScore",
    "build_mwdcs_candidate_from_activity_results",
    "candidate_from_difom_candidate",
    "evaluate_mwdcs",
    "parse_mwdcs_sweep_payload",
    "rank_mwdcs_schedules",
    "serialize_mwdcs_ranking",
]
