from __future__ import annotations

import pytest

from fluxforge.analysis.optimization_difom import (
    DIFOMLineTerm,
    DIFOMScheduleCandidate,
)
from fluxforge.analysis.optimization_fim import (
    build_fisher_information,
    evaluate_fim,
    rank_fim_schedules,
)


def _balanced_lines() -> tuple[DIFOMLineTerm, ...]:
    return (
        DIFOMLineTerm(
            nuclide="Mo-99",
            line_energy_keV=140.5,
            signal_counts=70.0,
            background_counts=15.0,
            interference_counts=3.0,
        ),
        DIFOMLineTerm(
            nuclide="Tc-99m",
            line_energy_keV=140.5,
            signal_counts=60.0,
            background_counts=15.0,
            interference_counts=4.0,
        ),
    )


def test_build_fisher_information_returns_symmetric_matrix() -> None:
    matrix, nuclides = build_fisher_information(_balanced_lines())

    assert nuclides == ("Mo-99", "Tc-99m")
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])


def test_evaluate_fim_supports_all_objectives() -> None:
    for objective in ("fim-d", "fim-a", "fim-c"):
        result = evaluate_fim(
            _balanced_lines(),
            objective=objective,
            target_nuclide="Mo-99",
            nuisance_variance_fraction=0.05,
            regularization=1.0e-6,
        )
        assert result.objective == objective
        assert result.diagnostics.effective_rank >= 1
        assert result.objective_score == pytest.approx(result.objective_score)


def test_evaluate_fim_c_raises_for_unknown_target_nuclide() -> None:
    with pytest.raises(ValueError, match="Target nuclide"):
        evaluate_fim(_balanced_lines(), objective="fim-c", target_nuclide="Co-60")


def test_rank_fim_schedules_prefers_balanced_information_for_d_opt() -> None:
    candidates = (
        DIFOMScheduleCandidate(
            label="balanced",
            irradiation_time_s=3600.0,
            cooldown_time_s=7200.0,
            count_time_s=900.0,
            lines=_balanced_lines(),
        ),
        DIFOMScheduleCandidate(
            label="single_nuclide_dominant",
            irradiation_time_s=3600.0,
            cooldown_time_s=7200.0,
            count_time_s=900.0,
            lines=(
                DIFOMLineTerm(
                    nuclide="Mo-99",
                    line_energy_keV=140.5,
                    signal_counts=120.0,
                    background_counts=20.0,
                    interference_counts=6.0,
                ),
                DIFOMLineTerm(
                    nuclide="Tc-99m",
                    line_energy_keV=140.5,
                    signal_counts=5.0,
                    background_counts=12.0,
                    interference_counts=2.0,
                ),
            ),
        ),
    )

    ranked = rank_fim_schedules(candidates, objective="fim-d")

    assert len(ranked) == 2
    assert ranked[0].label == "balanced"
    assert ranked[0].objective_score > ranked[1].objective_score


def test_rank_fim_schedules_accepts_c_opt_target() -> None:
    ranked = rank_fim_schedules(
        (
            DIFOMScheduleCandidate(
                label="candidate",
                irradiation_time_s=3600.0,
                cooldown_time_s=7200.0,
                count_time_s=900.0,
                lines=_balanced_lines(),
            ),
        ),
        objective="fim-c",
        target_nuclide="Mo-99",
    )

    assert len(ranked) == 1
    assert ranked[0].diagnostics.target_nuclide == "Mo-99"
