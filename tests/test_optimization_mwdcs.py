from __future__ import annotations

import pytest

from fluxforge.analysis.optimization_mwdcs import (
    MWDCSLineTerm,
    MWDCSWindow,
    MWDCSScheduleCandidate,
    evaluate_mwdcs,
    parse_mwdcs_sweep_payload,
    rank_mwdcs_schedules,
)


def test_parse_mwdcs_payload_builds_default_windows_with_decay() -> None:
    payload = {
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "V-52",
                        "line_energy_keV": 1434.0,
                        "signal_counts": 100.0,
                        "background_counts": 10.0,
                        "half_life_s": 300.0,
                    }
                ],
            }
        ]
    }

    candidates, weights = parse_mwdcs_sweep_payload(
        payload,
        default_window_offsets_s=(0.0, 3600.0),
        default_window_count_time_s=900.0,
    )

    assert weights == {}
    assert len(candidates) == 1
    assert len(candidates[0].windows) == 2
    signal_early = candidates[0].windows[0].lines[0].signal_counts
    signal_late = candidates[0].windows[1].lines[0].signal_counts
    assert signal_early > signal_late


def test_evaluate_mwdcs_full_spectrum_penalty_reduces_score() -> None:
    overlapping_lines = (
        MWDCSLineTerm("Mo-99", 140.5, 80.0, 10.0, 3.0),
        MWDCSLineTerm("Tc-99m", 140.9, 70.0, 12.0, 4.0),
    )
    windows = (
        MWDCSWindow(
            label="window_1",
            cooldown_time_s=0.0,
            count_time_s=900.0,
            lines=overlapping_lines,
        ),
    )

    unconstrained = evaluate_mwdcs(windows, full_spectrum_mode=False)
    penalized = evaluate_mwdcs(
        windows,
        full_spectrum_mode=True,
        overlap_penalty=0.25,
    )

    assert unconstrained.total_score > 0.0
    assert penalized.total_score < unconstrained.total_score


def test_rank_mwdcs_schedules_prefers_richer_multi_window_schedule() -> None:
    candidate_balanced = MWDCSScheduleCandidate(
        label="balanced",
        irradiation_time_s=3600.0,
        windows=(
            MWDCSWindow(
                label="w1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    MWDCSLineTerm("Mo-99", 140.5, 60.0, 10.0, 2.0),
                    MWDCSLineTerm("Tc-99m", 140.6, 58.0, 11.0, 2.0),
                ),
            ),
            MWDCSWindow(
                label="w2",
                cooldown_time_s=7200.0,
                count_time_s=900.0,
                lines=(
                    MWDCSLineTerm("Mo-99", 140.5, 45.0, 9.0, 2.0),
                    MWDCSLineTerm("Tc-99m", 140.6, 42.0, 9.0, 2.0),
                ),
            ),
        ),
    )
    candidate_weak = MWDCSScheduleCandidate(
        label="weak",
        irradiation_time_s=3600.0,
        windows=(
            MWDCSWindow(
                label="w1",
                cooldown_time_s=0.0,
                count_time_s=900.0,
                lines=(
                    MWDCSLineTerm("Mo-99", 140.5, 25.0, 10.0, 3.0),
                    MWDCSLineTerm("Tc-99m", 140.6, 10.0, 10.0, 3.0),
                ),
            ),
        ),
    )

    ranked = rank_mwdcs_schedules((candidate_balanced, candidate_weak))

    assert len(ranked) == 2
    assert ranked[0].label == "balanced"
    assert ranked[0].total_score > ranked[1].total_score
