from __future__ import annotations

import pytest

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMScheduleCandidate,
    build_difom_terms_from_activity_results,
    difom_line_score,
    evaluate_difom,
    parse_difom_sweep_payload,
    rank_difom_schedules,
)
from fluxforge.core.analysis_workspace import ActivityCalculationResult


def test_difom_line_score_handles_zero_or_negative_signal() -> None:
    assert difom_line_score(0.0, 5.0, 1.0) == pytest.approx(0.0)
    assert difom_line_score(-3.0, 5.0, 1.0) == pytest.approx(0.0)


def test_evaluate_difom_applies_isotope_weights() -> None:
    lines = (
        DIFOMLineTerm(
            nuclide="Mo-99",
            line_energy_keV=140.5,
            signal_counts=120.0,
            background_counts=20.0,
            interference_counts=10.0,
        ),
        DIFOMLineTerm(
            nuclide="Tc-99m",
            line_energy_keV=140.5,
            signal_counts=30.0,
            background_counts=15.0,
            interference_counts=0.0,
        ),
    )

    unweighted = evaluate_difom(lines)
    weighted = evaluate_difom(lines, isotope_weights={"Mo-99": 2.0, "Tc-99m": 0.5})

    assert weighted.total_score > unweighted.total_score
    assert len(weighted.line_scores) == 2
    assert weighted.line_scores[0].weight == pytest.approx(2.0)
    assert weighted.line_scores[1].weight == pytest.approx(0.5)


def test_rank_difom_schedules_orders_by_score_descending() -> None:
    candidates = (
        DIFOMScheduleCandidate(
            label="slow_cooldown",
            irradiation_time_s=3600.0,
            cooldown_time_s=86400.0,
            count_time_s=1200.0,
            lines=(
                DIFOMLineTerm(
                    nuclide="Mo-99",
                    line_energy_keV=140.5,
                    signal_counts=40.0,
                    background_counts=10.0,
                    interference_counts=5.0,
                ),
            ),
        ),
        DIFOMScheduleCandidate(
            label="fast_cooldown",
            irradiation_time_s=3600.0,
            cooldown_time_s=3600.0,
            count_time_s=1200.0,
            lines=(
                DIFOMLineTerm(
                    nuclide="Mo-99",
                    line_energy_keV=140.5,
                    signal_counts=90.0,
                    background_counts=5.0,
                    interference_counts=2.0,
                ),
            ),
        ),
    )

    ranked = rank_difom_schedules(candidates)

    assert len(ranked) == 2
    assert ranked[0].label == "fast_cooldown"
    assert ranked[0].total_score > ranked[1].total_score


def test_parse_difom_sweep_payload_validates_candidates() -> None:
    with pytest.raises(ValueError, match="candidates"):
        parse_difom_sweep_payload({"candidates": []})


def test_build_difom_terms_from_activity_results_uses_age_corrected_activity() -> None:
    results = (
        ActivityCalculationResult(
            nuclide="Mo-99",
            line_energy_keV=140.5,
            activity_bq=800.0,
            uncertainty_bq=25.0,
            age_corrected_activity_bq=900.0,
            mda_bq=0.0,
            half_life_s=65.94 * 3600.0,
            source_age_s=7200.0,
            chain_summary="Mo-99 feed",
            age_corrected_uncertainty_bq=40.0,
        ),
    )

    terms = build_difom_terms_from_activity_results(results)

    assert len(terms) == 1
    assert terms[0].nuclide == "Mo-99"
    assert terms[0].signal_counts == pytest.approx(900.0)
    assert terms[0].background_counts == pytest.approx(40.0)
