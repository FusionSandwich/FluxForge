from __future__ import annotations

import pytest

from fluxforge.analysis.optimization_bassd import (
    BASSDAction,
    BASSDLineState,
    BASSDScheduleCandidate,
    evaluate_bassd,
    posterior_variance_update,
    rank_bassd_schedules,
)


def test_posterior_variance_update_reduces_uncertainty() -> None:
    prior_variance = 100.0
    posterior_variance = posterior_variance_update(
        prior_variance,
        signal_counts=80.0,
        background_counts=20.0,
    )

    assert posterior_variance > 0.0
    assert posterior_variance < prior_variance


def test_rank_bassd_schedules_prefers_lower_dose_for_similar_information() -> None:
    low_dose = BASSDScheduleCandidate(
        label="low_dose",
        irradiation_time_s=3600.0,
        actions=(
            BASSDAction(
                label="a1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    BASSDLineState(
                        nuclide="Mo-99",
                        line_energy_keV=140.5,
                        prior_mean_counts=100.0,
                        prior_variance_counts2=120.0,
                        signal_counts=80.0,
                        background_counts=15.0,
                        dose_rate_uSv_h=2.0,
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )
    high_dose = BASSDScheduleCandidate(
        label="high_dose",
        irradiation_time_s=3600.0,
        actions=(
            BASSDAction(
                label="a1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    BASSDLineState(
                        nuclide="Mo-99",
                        line_energy_keV=140.5,
                        prior_mean_counts=100.0,
                        prior_variance_counts2=120.0,
                        signal_counts=82.0,
                        background_counts=15.0,
                        dose_rate_uSv_h=25.0,
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )

    ranked = rank_bassd_schedules(
        (low_dose, high_dose),
        dose_weight=0.05,
        exploration_temperature=0.0,
        seed=5,
    )

    assert len(ranked) == 2
    assert ranked[0].label == "low_dose"
    assert ranked[0].total_utility > ranked[1].total_utility


def test_evaluate_bassd_is_seed_reproducible_with_exploration_noise() -> None:
    actions = (
        BASSDAction(
            label="a1",
            cooldown_time_s=0.0,
            count_time_s=600.0,
            lines=(
                BASSDLineState(
                    nuclide="Tc-99m",
                    line_energy_keV=140.5,
                    prior_mean_counts=60.0,
                    prior_variance_counts2=80.0,
                    signal_counts=45.0,
                    background_counts=10.0,
                    dose_rate_uSv_h=4.0,
                    weight=1.0,
                ),
            ),
        ),
    )

    result_a = evaluate_bassd(
        actions,
        dose_weight=0.01,
        exploration_temperature=0.3,
        seed=17,
    )
    result_b = evaluate_bassd(
        actions,
        dose_weight=0.01,
        exploration_temperature=0.3,
        seed=17,
    )

    assert result_a.total_utility == pytest.approx(result_b.total_utility)
    assert result_a.action_scores[0].utility_score == pytest.approx(
        result_b.action_scores[0].utility_score
    )
