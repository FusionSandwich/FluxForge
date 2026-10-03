"""Validate elapsed-time irradiation rows without assuming missing power."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from fluxforge.physics.activation import IrradiationSegment


@dataclass(frozen=True)
class IrradiationInterval:
    start_s: float
    stop_s: float
    relative_power: float


def prepare_irradiation_history(
    rows: Iterable[Iterable[object]],
) -> tuple[tuple[IrradiationInterval, ...], tuple[IrradiationSegment, ...]]:
    """Map chronological Start/Stop/Power rows to contiguous duration segments.

    Times are elapsed seconds from a common zero. Leading and intermediate
    gaps become zero-power segments, so decay during shutdown is retained.
    Input order is preserved and overlapping or out-of-order rows are rejected.
    Power is relative to the user's common reference, not an inferred MW value.
    """
    intervals, segments = [], []
    previous_stop = 0.0
    for index, row in enumerate(rows, start=1):
        fields = tuple(row)
        if len(fields) != 3:
            raise ValueError(f"Row {index}: supply Start, Stop and Relative power.")
        values = []
        for name, value in zip(("Start", "Stop", "Relative power"), fields):
            if isinstance(value, bool):
                raise ValueError(f"Row {index}: {name} must be a finite number.")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Row {index}: {name} must be a finite number."
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(f"Row {index}: {name} must be a finite number.")
            values.append(numeric)
        start, stop, power = values
        if start < 0 or stop <= start:
            raise ValueError(f"Row {index}: require 0 ≤ Start < Stop.")
        if start < previous_stop:
            raise ValueError(
                f"Row {index}: rows must be chronological and cannot overlap."
            )
        if power < 0:
            raise ValueError(f"Row {index}: Relative power cannot be negative.")
        if start > previous_stop:
            segments.append(IrradiationSegment(start - previous_stop, 0.0))
        intervals.append(IrradiationInterval(start, stop, power))
        segments.append(IrradiationSegment(stop - start, power))
        previous_stop = stop
    if not intervals:
        raise ValueError("Add at least one irradiation interval.")
    return tuple(intervals), tuple(segments)


def irradiation_history_payload(rows: Iterable[Iterable[object]]) -> dict:
    intervals, segments = prepare_irradiation_history(rows)
    return {
        "schema": "fluxforge.irradiation_history.editor.v1",
        "time_unit": "s",
        "power_basis": "relative_to_user_reference",
        "source_kind": "user_entered",
        "scientific_admission": False,
        "timeline": [vars(interval) for interval in intervals],
        "irradiation": {"segments": [vars(segment) for segment in segments]},
    }


def read_irradiation_history_payload(payload: dict) -> tuple[IrradiationInterval, ...]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "fluxforge.irradiation_history.editor.v1"
    ):
        raise ValueError("Expected an irradiation-history editor JSON file.")
    if (
        payload.get("time_unit") != "s"
        or payload.get("power_basis") != "relative_to_user_reference"
    ):
        raise ValueError("History must use elapsed seconds and relative power.")
    timeline = payload.get("timeline")
    if not isinstance(timeline, list) or any(
        not isinstance(row, dict) for row in timeline
    ):
        raise ValueError(
            "History timeline must contain Start/Stop/Relative power rows."
        )
    intervals, segments = prepare_irradiation_history(
        (row.get("start_s"), row.get("stop_s"), row.get("relative_power"))
        for row in timeline
    )
    if payload.get("irradiation") != {
        "segments": [vars(segment) for segment in segments]
    }:
        raise ValueError("History duration segments disagree with its timeline.")
    return intervals
