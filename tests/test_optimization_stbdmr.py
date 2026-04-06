from __future__ import annotations

import pytest

from fluxforge.analysis.optimization_stbdmr import (
    STBDMRLineTerm,
    STBDMRWindow,
    STBDMRScheduleCandidate,
    build_interference_graph,
    evaluate_stbdmr,
    rank_stbdmr_schedules,
)


def test_build_interference_graph_is_symmetric_and_detects_overlap() -> None:
    lines = (
        STBDMRLineTerm("Mo-99", 140.5, 80.0, 10.0),
        STBDMRLineTerm("Tc-99m", 141.0, 75.0, 11.0),
        STBDMRLineTerm("Co-60", 1332.5, 50.0, 8.0),
    )
    graph = build_interference_graph(lines, energy_tolerance_keV=2.0)

    assert graph.shape == (3, 3)
    assert graph[0, 1] == pytest.approx(graph[1, 0])
    assert graph[0, 1] > 0.0
    assert graph[0, 2] == 0.0


def test_evaluate_stbdmr_differentiable_mode_changes_score_and_reports_gradient() -> None:
    windows = (
        STBDMRWindow(
            label="w1",
            cooldown_time_s=0.0,
            count_time_s=900.0,
            lines=(
                STBDMRLineTerm("Mo-99", 140.5, 70.0, 12.0, 4.0),
                STBDMRLineTerm("Tc-99m", 141.0, 65.0, 11.0, 4.0),
            ),
        ),
    )

    baseline = evaluate_stbdmr(
        windows,
        masking_regularization=0.2,
        differentiable_graph_mode=False,
    )
    differentiable = evaluate_stbdmr(
        windows,
        masking_regularization=0.2,
        differentiable_graph_mode=True,
        graph_temperature=2.0,
    )

    assert baseline.diagnostics.gradient_norm > 0.0
    assert differentiable.diagnostics.differentiable_graph_mode is True
    assert differentiable.total_score != pytest.approx(baseline.total_score)


def test_rank_stbdmr_schedules_penalizes_overlap_when_regularization_is_high() -> None:
    candidate_separated = STBDMRScheduleCandidate(
        label="separated",
        irradiation_time_s=3600.0,
        windows=(
            STBDMRWindow(
                label="w1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    STBDMRLineTerm("Mo-99", 140.5, 65.0, 10.0),
                    STBDMRLineTerm("Co-60", 1332.5, 62.0, 10.0),
                ),
            ),
        ),
    )
    candidate_overlap = STBDMRScheduleCandidate(
        label="overlap",
        irradiation_time_s=3600.0,
        windows=(
            STBDMRWindow(
                label="w1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    STBDMRLineTerm("Mo-99", 140.5, 65.0, 10.0),
                    STBDMRLineTerm("Tc-99m", 140.7, 62.0, 10.0),
                ),
            ),
        ),
    )

    ranked = rank_stbdmr_schedules(
        (candidate_separated, candidate_overlap),
        masking_regularization=0.5,
        differentiable_graph_mode=False,
    )

    assert len(ranked) == 2
    assert ranked[0].label == "separated"
    assert ranked[0].total_score > ranked[1].total_score
