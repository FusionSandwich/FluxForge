"""Fisher-information schedule scoring for irradiation optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMScheduleCandidate,
)

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class FIMDiagnostics:
    """Matrix diagnostics for one Fisher-information evaluation."""

    nuclides: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    regularization: float
    condition_number: float
    determinant: float
    log_determinant: float
    trace_inverse: float
    effective_rank: int
    min_eigenvalue: float
    max_eigenvalue: float
    target_nuclide: str | None = None


@dataclass(frozen=True)
class FIMEvaluation:
    """Objective score and diagnostics for one line-term collection."""

    objective: str
    objective_score: float
    diagnostics: FIMDiagnostics


@dataclass(frozen=True)
class FIMScheduleScore:
    """Scored schedule payload used by CLI ranking output."""

    label: str
    irradiation_time_s: float
    cooldown_time_s: float
    count_time_s: float
    objective: str
    objective_score: float
    diagnostics: FIMDiagnostics


def _nonnegative(value: float) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        return 0.0
    return numeric


def _matrix_as_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def build_fisher_information(
    line_terms: Sequence[DIFOMLineTerm],
    *,
    nuisance_variance_fraction: float = 0.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct a Fisher-information matrix from line-term count expectations."""

    nuclides = tuple(sorted({str(line.nuclide) for line in line_terms if line.nuclide}))
    if len(nuclides) == 0:
        return np.zeros((0, 0), dtype=float), ()

    index_by_nuclide = {label: idx for idx, label in enumerate(nuclides)}
    matrix = np.zeros((len(nuclides), len(nuclides)), dtype=float)
    nuisance_fraction = max(float(nuisance_variance_fraction), 0.0)

    for line in line_terms:
        nuclide = str(line.nuclide)
        if nuclide not in index_by_nuclide:
            continue

        signal = _nonnegative(line.signal_counts)
        if signal <= 0.0:
            continue

        variance = (
            signal
            + _nonnegative(line.background_counts)
            + _nonnegative(line.interference_counts)
        )
        if nuisance_fraction > 0.0:
            variance += (nuisance_fraction * signal) ** 2

        weight = 1.0 / max(variance, _EPSILON)
        sensitivity = np.zeros(len(nuclides), dtype=float)
        sensitivity[index_by_nuclide[nuclide]] = signal
        matrix += weight * np.outer(sensitivity, sensitivity)

    return matrix, nuclides


def _regularized_matrix(matrix: np.ndarray, *, regularization: float) -> np.ndarray:
    if matrix.shape[0] == 0:
        return matrix
    reg = max(float(regularization), _EPSILON)
    return matrix + np.eye(matrix.shape[0], dtype=float) * reg


def _build_diagnostics(
    matrix_regularized: np.ndarray,
    *,
    nuclides: tuple[str, ...],
    regularization: float,
    target_nuclide: str | None,
) -> FIMDiagnostics:
    if matrix_regularized.shape[0] == 0:
        return FIMDiagnostics(
            nuclides=nuclides,
            matrix=(),
            regularization=float(max(regularization, _EPSILON)),
            condition_number=0.0,
            determinant=0.0,
            log_determinant=float("-inf"),
            trace_inverse=0.0,
            effective_rank=0,
            min_eigenvalue=0.0,
            max_eigenvalue=0.0,
            target_nuclide=target_nuclide,
        )

    determinant = float(np.linalg.det(matrix_regularized))
    sign, log_determinant = np.linalg.slogdet(matrix_regularized)
    if sign <= 0:
        log_determinant_value = float("-inf")
    else:
        log_determinant_value = float(log_determinant)

    inverse = np.linalg.inv(matrix_regularized)
    trace_inverse = float(np.trace(inverse))
    condition_number = float(np.linalg.cond(matrix_regularized))
    eigenvalues = np.linalg.eigvalsh(matrix_regularized)
    max_eigenvalue = float(np.max(eigenvalues))
    min_eigenvalue = float(np.min(eigenvalues))

    threshold = max(max_eigenvalue * 1.0e-9, _EPSILON)
    effective_rank = int(np.sum(eigenvalues > threshold))

    return FIMDiagnostics(
        nuclides=nuclides,
        matrix=_matrix_as_tuple(matrix_regularized),
        regularization=float(max(regularization, _EPSILON)),
        condition_number=condition_number,
        determinant=determinant,
        log_determinant=log_determinant_value,
        trace_inverse=trace_inverse,
        effective_rank=effective_rank,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        target_nuclide=target_nuclide,
    )


def evaluate_fim(
    line_terms: Sequence[DIFOMLineTerm],
    *,
    objective: str = "fim-d",
    target_nuclide: str | None = None,
    nuisance_variance_fraction: float = 0.0,
    regularization: float = 1.0e-6,
) -> FIMEvaluation:
    """Evaluate one Fisher-information objective for a line-term collection."""

    objective_key = str(objective).lower().strip()
    if objective_key not in {"fim-d", "fim-a", "fim-c"}:
        raise ValueError(
            "Unsupported FIM objective. Use one of: fim-d, fim-a, fim-c."
        )

    fisher_matrix, nuclides = build_fisher_information(
        line_terms,
        nuisance_variance_fraction=nuisance_variance_fraction,
    )
    matrix_regularized = _regularized_matrix(
        fisher_matrix,
        regularization=regularization,
    )

    selected_target = target_nuclide
    if objective_key == "fim-c" and len(nuclides) > 0:
        if selected_target is None:
            selected_target = nuclides[0]
        if selected_target not in nuclides:
            raise ValueError(
                f"Target nuclide {selected_target!r} was not found in the candidate lines."
            )

    diagnostics = _build_diagnostics(
        matrix_regularized,
        nuclides=nuclides,
        regularization=regularization,
        target_nuclide=selected_target,
    )

    if matrix_regularized.shape[0] == 0:
        score = 0.0
    elif objective_key == "fim-d":
        score = diagnostics.log_determinant
    else:
        inverse = np.linalg.inv(matrix_regularized)
        if objective_key == "fim-a":
            score = -float(np.trace(inverse))
        else:
            target_index = nuclides.index(str(selected_target))
            target_vector = np.zeros((len(nuclides),), dtype=float)
            target_vector[target_index] = 1.0
            score = -float(target_vector.T @ inverse @ target_vector)

    return FIMEvaluation(
        objective=objective_key,
        objective_score=float(score),
        diagnostics=diagnostics,
    )


def rank_fim_schedules(
    candidates: Sequence[DIFOMScheduleCandidate],
    *,
    objective: str = "fim-d",
    target_nuclide: str | None = None,
    nuisance_variance_fraction: float = 0.0,
    regularization: float = 1.0e-6,
) -> tuple[FIMScheduleScore, ...]:
    """Score and rank schedule candidates using a FIM objective."""

    ranked: list[FIMScheduleScore] = []
    for candidate in candidates:
        evaluation = evaluate_fim(
            candidate.lines,
            objective=objective,
            target_nuclide=target_nuclide,
            nuisance_variance_fraction=nuisance_variance_fraction,
            regularization=regularization,
        )
        ranked.append(
            FIMScheduleScore(
                label=str(candidate.label),
                irradiation_time_s=float(candidate.irradiation_time_s),
                cooldown_time_s=float(candidate.cooldown_time_s),
                count_time_s=float(candidate.count_time_s),
                objective=evaluation.objective,
                objective_score=float(evaluation.objective_score),
                diagnostics=evaluation.diagnostics,
            )
        )

    ranked.sort(key=lambda item: item.objective_score, reverse=True)
    return tuple(ranked)


def serialize_fim_ranking(
    ranked: Sequence[FIMScheduleScore],
    *,
    objective: str,
) -> dict[str, Any]:
    """Serialize FIM ranking output into the CLI bundle shape."""

    rows = []
    for rank_index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank_index,
                "label": item.label,
                "irradiation_time_s": item.irradiation_time_s,
                "cooldown_time_s": item.cooldown_time_s,
                "count_time_s": item.count_time_s,
                "objective": item.objective,
                "objective_score": item.objective_score,
                "matrix_diagnostics": {
                    "nuclides": list(item.diagnostics.nuclides),
                    "regularization": item.diagnostics.regularization,
                    "condition_number": item.diagnostics.condition_number,
                    "determinant": item.diagnostics.determinant,
                    "log_determinant": item.diagnostics.log_determinant,
                    "trace_inverse": item.diagnostics.trace_inverse,
                    "effective_rank": item.diagnostics.effective_rank,
                    "min_eigenvalue": item.diagnostics.min_eigenvalue,
                    "max_eigenvalue": item.diagnostics.max_eigenvalue,
                    "target_nuclide": item.diagnostics.target_nuclide,
                },
            }
        )

    return {
        "schema": "fluxforge.optimization_sweep.fim.v1",
        "objective": str(objective).lower(),
        "ranked_candidates": rows,
    }


__all__ = [
    "FIMDiagnostics",
    "FIMEvaluation",
    "FIMScheduleScore",
    "build_fisher_information",
    "evaluate_fim",
    "rank_fim_schedules",
    "serialize_fim_ranking",
]
