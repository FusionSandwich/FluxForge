from __future__ import annotations

import pytest

from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind


@pytest.mark.parametrize(
    "intent",
    (
        CanvasIntent(
            CanvasIntentKind.CREATE_ROI,
            spectrum_id="spectrum-1",
            bounds=(100.0, 120.0),
            left_background=(90.0, 98.0),
            right_background=(122.0, 130.0),
        ),
        CanvasIntent(
            CanvasIntentKind.MOVE_ROI,
            spectrum_id="spectrum-1",
            roi_id="roi-1",
            bounds=(101.0, 121.0),
            drag_token="drag-7",
        ),
        CanvasIntent(
            CanvasIntentKind.ADD_PEAK,
            spectrum_id="spectrum-1",
            position=1173.2,
        ),
        CanvasIntent(
            CanvasIntentKind.MERGE_PEAKS,
            spectrum_id="spectrum-1",
            component_ids=("component-a", "component-b"),
        ),
        CanvasIntent(
            CanvasIntentKind.ASSIGN_NUCLIDE,
            spectrum_id="spectrum-1",
            peak_id="peak-1",
            nuclide="Co60",
        ),
        CanvasIntent(CanvasIntentKind.PIN_NUCLIDE, nuclide="Co60"),
        CanvasIntent(
            CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
            spectrum_id="spectrum-1",
            role="foreground",
        ),
        CanvasIntent(
            CanvasIntentKind.CHANGE_VIEWPORT,
            spectrum_id="spectrum-1",
            viewport_id="spectrum-main",
            x_range=(100.0, 1500.0),
            y_range=(0.0, 10000.0),
        ),
        CanvasIntent(
            CanvasIntentKind.SET_LOG_SCALE,
            spectrum_id="spectrum-1",
            enabled=True,
        ),
    ),
)
def test_canvas_intents_round_trip_without_qt(intent: CanvasIntent) -> None:
    assert CanvasIntent.from_dict(intent.to_dict()) == intent


@pytest.mark.parametrize(
    "kwargs",
    (
        {"kind": CanvasIntentKind.CREATE_ROI, "spectrum_id": "s", "bounds": (2, 1)},
        {"kind": CanvasIntentKind.MOVE_PEAK, "spectrum_id": "s", "peak_id": "p"},
        {
            "kind": CanvasIntentKind.MERGE_PEAKS,
            "spectrum_id": "s",
            "component_ids": ("a",),
        },
        {"kind": CanvasIntentKind.ASSIGN_NUCLIDE, "spectrum_id": "s", "peak_id": "p"},
        {
            "kind": CanvasIntentKind.CHANGE_VIEWPORT,
            "spectrum_id": "s",
            "viewport_id": "v",
        },
        {"kind": CanvasIntentKind.SET_LOG_SCALE, "spectrum_id": "s"},
    ),
)
def test_canvas_intents_reject_incomplete_or_nonphysical_payloads(kwargs) -> None:
    with pytest.raises(ValueError):
        CanvasIntent(**kwargs)


def test_canvas_intent_rejects_future_schema_versions() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        CanvasIntent.from_dict(
            {
                "schema": "fluxforge.canvas_intent.v2",
                "version": 2,
                "kind": "nuclide.pin",
                "nuclide": "Co60",
            }
        )


def test_canvas_intent_rejects_string_boolean_values() -> None:
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        CanvasIntent.from_dict(
            {
                "schema": "fluxforge.canvas_intent.v1",
                "version": 1,
                "kind": "viewport.log_scale",
                "spectrum_id": "spectrum-1",
                "enabled": "false",
            }
        )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"version": "1"}, "version must be an integer"),
        ({"temporary": True}, "unknown fields"),
        ({"position": True}, "finite number"),
    ],
)
def test_canvas_intent_parser_is_strict(change, message) -> None:
    payload = CanvasIntent(
        CanvasIntentKind.ADD_PEAK,
        spectrum_id="spectrum-1",
        position=12.0,
    ).to_dict()
    payload.update(change)
    with pytest.raises(ValueError, match=message):
        CanvasIntent.from_dict(payload)
