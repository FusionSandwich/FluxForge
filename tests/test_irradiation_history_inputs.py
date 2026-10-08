"""Timing rows preserve decay gaps and reject missing or nonphysical inputs."""

import copy

import pytest

from fluxforge.gui.irradiation_history import (
    irradiation_history_payload,
    prepare_irradiation_history,
    read_irradiation_history_payload,
)
from fluxforge.physics.activation import IrradiationSegment, irradiation_buildup_factor


def test_history_retains_leading_shutdown_and_internal_gaps():
    _, segments = prepare_irradiation_history([(10, 20, 1), (30, 50, 0.5), (50, 60, 0)])
    assert all(isinstance(segment, IrradiationSegment) for segment in segments)
    assert [(s.duration_s, s.relative_power) for s in segments] == [
        (10, 0),
        (10, 1),
        (10, 0),
        (20, 0.5),
        (10, 0),
    ]
    # Half-life 10 s: first exposure decays over 40 s; second over 10 s.
    assert irradiation_buildup_factor(segments, 10) == pytest.approx(
        (1 - 0.5) * 0.5**4 + 0.5 * (1 - 0.5**2) * 0.5
    )


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "at least one"),
        ([(0, 10)], "Start, Stop"),
        ([(0, 10, "")], "Relative power"),
        ([(0, 10, None)], "finite"),
        ([(0, 10, True)], "finite"),
        ([(0, float("inf"), 1)], "finite"),
        ([(float("nan"), 10, 1)], "finite"),
        ([(-1, 10, 1)], "Start < Stop"),
        ([(10, 10, 1)], "Start < Stop"),
        ([(0, 10, -0.1)], "negative"),
        ([(0, 10, 1), (9, 20, 1)], "overlap"),
        ([(10, 20, 1), (0, 10, 1)], "chronological"),
    ],
)
def test_invalid_history_is_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        prepare_irradiation_history(rows)


def test_editor_json_roundtrip_preserves_explicit_power_and_backend_shape():
    payload = irradiation_history_payload([("0", "10", "1.2"), (20, 30, 0)])
    intervals = read_irradiation_history_payload(payload)
    assert [vars(row) for row in intervals] == payload["timeline"]
    assert payload["scientific_admission"] is False
    assert payload["irradiation"]["segments"] == [
        {"duration_s": 10.0, "relative_power": 1.2},
        {"duration_s": 10.0, "relative_power": 0.0},
        {"duration_s": 10.0, "relative_power": 0.0},
    ]
    inconsistent = copy.deepcopy(payload)
    inconsistent["irradiation"]["segments"][0]["relative_power"] = 1
    with pytest.raises(ValueError, match="disagree"):
        read_irradiation_history_payload(inconsistent)
    inconsistent = copy.deepcopy(payload)
    inconsistent["time_unit"] = "hours"
    with pytest.raises(ValueError, match="seconds"):
        read_irradiation_history_payload(inconsistent)
