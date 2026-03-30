"""ASTM E1297 minimum detectable activity helpers."""

from __future__ import annotations

import math

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
    _format_float,
)


def currie_mda(
    *,
    background_counts: float,
    live_time_s: float,
    efficiency: float,
    emission_probability: float = 1.0,
) -> float:
    """Return Currie-style minimum detectable activity in Bq."""

    background = max(float(background_counts), 0.0)
    live_time = max(float(live_time_s), 1e-9)
    eff = max(float(efficiency), 1e-9)
    emission = max(float(emission_probability), 1e-9)
    decision_level = 2.71 + 4.65 * math.sqrt(background)
    return decision_level / (live_time * eff * emission)


class E1297Module(StandardsModule):
    """ASTM E1297 MDA surface using the Currie method."""

    standard_id = "ASTM E1297"
    display_name = "ASTM E1297"
    summary = "Currie-style minimum detectable activity checks for reported activities."

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return (
            LockedSetting(
                field_id="mda_method",
                value="Currie",
                standard_section="ASTM E1297 §7",
            ),
        )

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        payload = dict(context.extra.get("mda", {}))
        mda_value = currie_mda(
            background_counts=float(payload.get("background_counts", 0.0)),
            live_time_s=float(payload.get("live_time_s", 1.0)),
            efficiency=float(payload.get("efficiency", 1.0)),
            emission_probability=float(payload.get("emission_probability", 1.0)),
        )
        activity_bq = float(payload.get("activity_bq", 0.0))
        status = "green" if activity_bq >= mda_value else "amber"
        checks = (
            StandardsCheck(
                key="currie_mda",
                label="Currie MDA",
                status="green",
                message="Minimum detectable activity is computed with the locked Currie formulation.",
                section="ASTM E1297 §7",
                value=_format_float(mda_value, suffix=" Bq"),
                limit="Currie method",
            ),
            StandardsCheck(
                key="activity_vs_mda",
                label="Activity above MDA",
                status=status,
                message=(
                    "Reported activity exceeds the Currie MDA."
                    if status == "green"
                    else "Reported activity is below the Currie MDA and should be flagged."
                ),
                section="ASTM E1297 §8",
                value=_format_float(activity_bq, suffix=" Bq"),
                limit=f"MDA {_format_float(mda_value, suffix=' Bq')}",
            ),
        )
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=checks,
            locked_settings=self.locked_settings(),
            summary="ASTM E1297 forces Currie MDA reporting inside the activity table.",
        )


__all__ = ["E1297Module", "currie_mda"]
