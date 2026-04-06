"""Spectro-temporal Bayesian design with masking regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fluxforge.analysis.optimization_difom import DIFOMScheduleCandidate

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class STBDMRLineTerm:
    """One line observation inside a spectro-temporal window."""

    nuclide: str
    line_energy_keV: float
    signal_counts: float
    background_counts: float = 0.0
    continuum_counts: float = 0.0
    overlap_group: str = ""
    half_life_s: float | None = None


@dataclass(frozen=True)
class STBDMRWindow:
    """One cooldown/count window considered by STBD-MR."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    lines: tuple[STBDMRLineTerm, ...]


@dataclass(frozen=True)
class STBDMRWindowScore:
    """Window-level score contributions."""

    label: str
    cooldown_time_s: float
    count_time_s: float
    base_information: float
    masking_penalty: float
    graph_bonus: float
    marginal_score: float


@dataclass(frozen=True)
class STBDMRDiagnostics:
    """Interference-graph diagnostics for one evaluated schedule."""

    graph_density: float
    average_degree: float
    masking_penalty: float
    differentiable_graph_mode: bool
    gradient_norm: float


@dataclass(frozen=True)
class STBDMREvaluation:
    """Aggregated STBD-MR score and diagnostics."""

    total_score: float
    diagnostics: STBDMRDiagnostics
    window_scores: tuple[STBDMRWindowScore, ...]


@dataclass(frozen=True)
class STBDMRScheduleCandidate:
    """Schedule candidate for STBD-MR ranking."""

    label: str
    irradiation_time_s: float
    windows: tuple[STBDMRWindow, ...]


@dataclass(frozen=True)
class STBDMRScheduleScore:
    """Rank row for a candidate scored under STBD-MR."""

    label: str
    irradiation_time_s: float
    total_score: float
    diagnostics: STBDMRDiagnostics
    window_scores: tuple[STBDMRWindowScore, ...]


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def _line_information(term: STBDMRLineTerm) -> float:
    signal = _nonnegative(term.signal_counts)
    if signal <= 0.0:
        return 0.0
    denominator = (
        signal
        + _nonnegative(term.background_counts)
        + _nonnegative(term.continuum_counts)
    )
    return (signal * signal) / max(denominator, _EPSILON)


def _pair_is_adjacent(
    left: STBDMRLineTerm,
    right: STBDMRLineTerm,
    *,
    energy_tolerance_keV: float,
) -> bool:
    if left.nuclide == right.nuclide:
        return False
    if left.overlap_group and right.overlap_group and left.overlap_group == right.overlap_group:
        return True
    return abs(float(left.line_energy_keV) - float(right.line_energy_keV)) <= float(energy_tolerance_keV)


def build_interference_graph(
    lines: Sequence[STBDMRLineTerm],
    *,
    energy_tolerance_keV: float = 3.0,
) -> np.ndarray:
    """Build a symmetric adjacency matrix for interfering lines."""

    n_lines = len(lines)
    graph = np.zeros((n_lines, n_lines), dtype=float)
    for i in range(n_lines):
        left = lines[i]
        for j in range(i + 1, n_lines):
            right = lines[j]
            if _pair_is_adjacent(
                left,
                right,
                energy_tolerance_keV=energy_tolerance_keV,
            ):
                graph[i, j] = 1.0
                graph[j, i] = 1.0
    return graph


def _window_graph_penalty(
    lines: Sequence[STBDMRLineTerm],
    line_information: Sequence[float],
    *,
    masking_regularization: float,
    energy_tolerance_keV: float,
) -> tuple[float, float, float]:
    graph = build_interference_graph(lines, energy_tolerance_keV=energy_tolerance_keV)
    if graph.size == 0:
        return 0.0, 0.0, 0.0

    penalty = 0.0
    edge_count = 0
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            if graph[i, j] <= 0.0:
                continue
            edge_count += 1
            penalty += min(float(line_information[i]), float(line_information[j]))

    max_edges = max(n_lines * (n_lines - 1) / 2.0, 1.0)
    density = float(edge_count) / max_edges
    average_degree = (2.0 * float(edge_count)) / max(float(n_lines), 1.0)
    penalty *= max(float(masking_regularization), 0.0)
    return float(penalty), float(density), float(average_degree)


def _window_graph_bonus(
    lines: Sequence[STBDMRLineTerm],
    line_information: Sequence[float],
    *,
    graph_temperature: float,
) -> float:
    temperature = max(float(graph_temperature), 1.0e-6)
    bonus = 0.0
    n_lines = len(lines)
    for i in range(n_lines):
        left = lines[i]
        for j in range(i + 1, n_lines):
            right = lines[j]
            if left.nuclide == right.nuclide:
                continue
            distance = abs(float(left.line_energy_keV) - float(right.line_energy_keV))
            affinity = float(np.exp(-distance / temperature))
            info_term = float(np.sqrt(max(line_information[i], 0.0) * max(line_information[j], 0.0)))
            bonus += affinity * info_term
    return 0.1 * bonus


def evaluate_stbdmr(
    windows: Sequence[STBDMRWindow],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    masking_regularization: float = 0.1,
    differentiable_graph_mode: bool = False,
    graph_temperature: float = 2.0,
    energy_tolerance_keV: float = 3.0,
) -> STBDMREvaluation:
    """Evaluate STBD-MR schedule utility with masking regularization."""

    weights = isotope_weights or {}
    total_score = 0.0
    window_scores: list[STBDMRWindowScore] = []

    density_rows: list[float] = []
    degree_rows: list[float] = []
    total_penalty = 0.0

    for window in windows:
        line_information: list[float] = []
        base_information = 0.0
        for line in window.lines:
            info = _line_information(line)
            weight = max(float(weights.get(line.nuclide, 1.0)), 0.0)
            weighted_info = info * weight
            line_information.append(weighted_info)
            base_information += weighted_info

        masking_penalty, graph_density, average_degree = _window_graph_penalty(
            window.lines,
            line_information,
            masking_regularization=masking_regularization,
            energy_tolerance_keV=energy_tolerance_keV,
        )
        graph_bonus = 0.0
        if differentiable_graph_mode:
            graph_bonus = _window_graph_bonus(
                window.lines,
                line_information,
                graph_temperature=graph_temperature,
            )

        marginal = base_information - masking_penalty + graph_bonus
        total_score += marginal
        total_penalty += masking_penalty
        density_rows.append(graph_density)
        degree_rows.append(average_degree)

        window_scores.append(
            STBDMRWindowScore(
                label=str(window.label),
                cooldown_time_s=float(window.cooldown_time_s),
                count_time_s=float(window.count_time_s),
                base_information=float(base_information),
                masking_penalty=float(masking_penalty),
                graph_bonus=float(graph_bonus),
                marginal_score=float(marginal),
            )
        )

    average_density = float(np.mean(density_rows)) if density_rows else 0.0
    average_degree = float(np.mean(degree_rows)) if degree_rows else 0.0
    gradient_norm = abs(float(total_penalty))

    diagnostics = STBDMRDiagnostics(
        graph_density=average_density,
        average_degree=average_degree,
        masking_penalty=float(total_penalty),
        differentiable_graph_mode=bool(differentiable_graph_mode),
        gradient_norm=float(gradient_norm),
    )
    return STBDMREvaluation(
        total_score=float(total_score),
        diagnostics=diagnostics,
        window_scores=tuple(window_scores),
    )


def rank_stbdmr_schedules(
    candidates: Sequence[STBDMRScheduleCandidate],
    *,
    isotope_weights: Mapping[str, float] | None = None,
    masking_regularization: float = 0.1,
    differentiable_graph_mode: bool = False,
    graph_temperature: float = 2.0,
) -> tuple[STBDMRScheduleScore, ...]:
    """Score and rank schedule candidates under STBD-MR."""

    ranked: list[STBDMRScheduleScore] = []
    for candidate in candidates:
        evaluation = evaluate_stbdmr(
            candidate.windows,
            isotope_weights=isotope_weights,
            masking_regularization=masking_regularization,
            differentiable_graph_mode=differentiable_graph_mode,
            graph_temperature=graph_temperature,
        )
        ranked.append(
            STBDMRScheduleScore(
                label=str(candidate.label),
                irradiation_time_s=float(candidate.irradiation_time_s),
                total_score=float(evaluation.total_score),
                diagnostics=evaluation.diagnostics,
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


def _parse_line_term(item: Any, *, context: str) -> STBDMRLineTerm:
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

    return STBDMRLineTerm(
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
        continuum_counts=_parse_float(
            item.get("continuum_counts", item.get("continuum_burden", 0.0)),
            field_name="continuum_counts",
            context=context,
        ),
        overlap_group=str(item.get("overlap_group") or "").strip(),
        half_life_s=half_life_s,
    )


def _project_signal(signal: float, half_life_s: float | None, cooldown_time_s: float) -> float:
    if signal <= 0.0:
        return 0.0
    if half_life_s is None or half_life_s <= 0.0:
        return float(signal)
    lam = np.log(2.0) / float(half_life_s)
    return float(signal) * float(np.exp(-lam * max(float(cooldown_time_s), 0.0)))


def _windows_from_base_lines(
    lines: Sequence[STBDMRLineTerm],
    *,
    offsets_s: Sequence[float],
    window_count_time_s: float,
) -> tuple[STBDMRWindow, ...]:
    windows: list[STBDMRWindow] = []
    for index, offset in enumerate(offsets_s, start=1):
        cooldown = max(float(offset), 0.0)
        projected_lines: list[STBDMRLineTerm] = []
        for line in lines:
            projected_lines.append(
                STBDMRLineTerm(
                    nuclide=line.nuclide,
                    line_energy_keV=float(line.line_energy_keV),
                    signal_counts=_project_signal(
                        line.signal_counts,
                        line.half_life_s,
                        cooldown,
                    ),
                    background_counts=float(line.background_counts),
                    continuum_counts=float(line.continuum_counts),
                    overlap_group=str(line.overlap_group),
                    half_life_s=line.half_life_s,
                )
            )
        windows.append(
            STBDMRWindow(
                label=f"window_{index}",
                cooldown_time_s=cooldown,
                count_time_s=max(float(window_count_time_s), 1.0),
                lines=tuple(projected_lines),
            )
        )
    return tuple(windows)


def parse_stbdmr_sweep_payload(
    payload: Mapping[str, Any],
    *,
    default_window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    default_window_count_time_s: float = 900.0,
) -> tuple[tuple[STBDMRScheduleCandidate, ...], dict[str, float]]:
    """Parse payload consumed by CLI optimization-sweep for stbd-mr objective."""

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
        raise ValueError("Optimization payload requires a non-empty 'candidates' list.")

    offsets = tuple(float(max(item, 0.0)) for item in default_window_offsets_s)
    if len(offsets) == 0:
        raise ValueError("STBD-MR requires at least one window offset.")

    candidates: list[STBDMRScheduleCandidate] = []
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
            windows: list[STBDMRWindow] = []
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
                    STBDMRWindow(
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
            STBDMRScheduleCandidate(
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
        isotope_weights[nuclide] = max(
            _parse_float(value, field_name="weight", context="isotope_weights"),
            0.0,
        )

    return tuple(candidates), isotope_weights


def serialize_stbdmr_ranking(
    ranked: Sequence[STBDMRScheduleScore],
) -> dict[str, Any]:
    """Serialize STBD-MR ranking output into CLI bundle shape."""

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank_index,
                "label": item.label,
                "irradiation_time_s": item.irradiation_time_s,
                "total_score": item.total_score,
                "diagnostics": {
                    "graph_density": item.diagnostics.graph_density,
                    "average_degree": item.diagnostics.average_degree,
                    "masking_penalty": item.diagnostics.masking_penalty,
                    "differentiable_graph_mode": item.diagnostics.differentiable_graph_mode,
                    "gradient_norm": item.diagnostics.gradient_norm,
                },
                "window_scores": [
                    {
                        "label": window.label,
                        "cooldown_time_s": window.cooldown_time_s,
                        "count_time_s": window.count_time_s,
                        "base_information": window.base_information,
                        "masking_penalty": window.masking_penalty,
                        "graph_bonus": window.graph_bonus,
                        "marginal_score": window.marginal_score,
                    }
                    for window in item.window_scores
                ],
            }
        )

    return {
        "schema": "fluxforge.optimization_sweep.stbdmr.v1",
        "objective": "stbd-mr",
        "ranked_candidates": rows,
    }


def candidate_from_difom_candidate(
    candidate: DIFOMScheduleCandidate,
    *,
    window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    window_count_time_s: float = 900.0,
) -> STBDMRScheduleCandidate:
    """Construct STBD-MR candidate from DI-FOM schedule candidate."""

    base_lines = tuple(
        STBDMRLineTerm(
            nuclide=line.nuclide,
            line_energy_keV=float(line.line_energy_keV),
            signal_counts=float(line.signal_counts),
            background_counts=float(line.background_counts),
            continuum_counts=float(line.interference_counts),
            overlap_group="",
            half_life_s=None,
        )
        for line in candidate.lines
    )
    windows = _windows_from_base_lines(
        base_lines,
        offsets_s=window_offsets_s,
        window_count_time_s=window_count_time_s,
    )
    return STBDMRScheduleCandidate(
        label=str(candidate.label),
        irradiation_time_s=float(candidate.irradiation_time_s),
        windows=windows,
    )


def build_stbdmr_candidate_from_activity_results(
    activity_results: Sequence[Any],
    *,
    label: str = "activity_preview",
    irradiation_time_s: float = 0.0,
    window_offsets_s: Sequence[float] = (0.0, 7200.0, 86400.0),
    window_count_time_s: float = 900.0,
) -> STBDMRScheduleCandidate:
    """Build a preview STBD-MR candidate from activity-review rows."""

    base_lines: list[STBDMRLineTerm] = []
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
            STBDMRLineTerm(
                nuclide=nuclide,
                line_energy_keV=float(getattr(result, "line_energy_keV", 0.0) or 0.0),
                signal_counts=float(signal),
                background_counts=float(background),
                continuum_counts=float(background),
                overlap_group="",
                half_life_s=half_life_s,
            )
        )

    windows = _windows_from_base_lines(
        tuple(base_lines),
        offsets_s=window_offsets_s,
        window_count_time_s=window_count_time_s,
    )
    return STBDMRScheduleCandidate(
        label=str(label),
        irradiation_time_s=float(irradiation_time_s),
        windows=windows,
    )


__all__ = [
    "STBDMRLineTerm",
    "STBDMRWindow",
    "STBDMRWindowScore",
    "STBDMRDiagnostics",
    "STBDMREvaluation",
    "STBDMRScheduleCandidate",
    "STBDMRScheduleScore",
    "build_interference_graph",
    "build_stbdmr_candidate_from_activity_results",
    "candidate_from_difom_candidate",
    "evaluate_stbdmr",
    "parse_stbdmr_sweep_payload",
    "rank_stbdmr_schedules",
    "serialize_stbdmr_ranking",
]
