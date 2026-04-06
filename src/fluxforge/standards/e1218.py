"""ASTM E1218 calibration-source bracketing checks."""

from __future__ import annotations

from datetime import timedelta

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
)


class E1218Module(StandardsModule):
    """ASTM E1218 calibration-source bracketing helper."""

    standard_id = "ASTM E1218"
    display_name = "ASTM E1218"
    summary = "Ensures measurements are bracketed by valid calibration or check-source observations."

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return (
            LockedSetting(
                field_id="calibration_bracketing",
                value="before-and-after-checks-required",
                standard_section="ASTM E1218 §8",
            ),
        )

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        before = context.before_calibration
        measured = context.measured_at
        after = context.after_calibration
        if before and measured and after and before <= measured <= after:
            status = "green"
            message = "Measurement time is bracketed by valid reference observations."
        elif measured and (before or after):
            status = "amber"
            message = "Measurement is only partially bracketed by reference observations."
        else:
            status = "red"
            message = "Calibration bracketing data is missing."
        checks = (
            StandardsCheck(
                key="calibration_bracketing",
                label="Calibration bracketing",
                status=status,
                message=message,
                section="ASTM E1218 §8",
                value=(
                    f"{before.isoformat() if before else 'N/A'} → "
                    f"{after.isoformat() if after else 'N/A'}"
                ),
                limit="measurement within bracket",
            ),
        )
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=checks,
            locked_settings=self.locked_settings(),
            summary="ASTM E1218 requires calibration-source bracketing around the measurement window.",
        )


__all__ = ["E1218Module"]
