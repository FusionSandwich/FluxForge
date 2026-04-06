"""ASTM E181 standards checks for energy calibration and detector QA."""

from __future__ import annotations

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
    _format_float,
)


class E181Module(StandardsModule):
    """ASTM E181 calibration and check-source compliance surface."""

    standard_id = "ASTM E181"
    display_name = "ASTM E181"
    summary = (
        "Calibration residuals, fit-order locking, and counting-statistics checks "
        "for standards-compliant gamma spectroscopy."
    )

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return (
            LockedSetting(
                field_id="energy_fit_order",
                value="2",
                standard_section="ASTM E181 §5.4.2",
            ),
            LockedSetting(
                field_id="calibration_sequence",
                value="calibrate-before-quantification",
                standard_section="ASTM E181 §6",
            ),
        )

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        order_ok = context.calibration_order is not None and context.calibration_order <= 2
        residual = context.max_residual_keV
        residual_ok = residual is not None and residual <= 0.5
        primary_counts = float(context.net_counts.get("primary", 0.0))
        count_status = "green" if primary_counts >= 1000.0 else "amber"

        checks = (
            StandardsCheck(
                key="fit_order",
                label="Energy fit order",
                status="green" if order_ok else "red",
                message=(
                    "Polynomial order satisfies ASTM E181."
                    if order_ok
                    else "Energy fit order exceeds the ASTM E181 2nd-order cap."
                ),
                section="ASTM E181 §5.4.2",
                value=str(context.calibration_order or "N/A"),
                limit="≤ 2",
            ),
            StandardsCheck(
                key="energy_residual",
                label="Maximum residual",
                status="green" if residual_ok else "red",
                message=(
                    "Residuals remain inside the ASTM acceptance band."
                    if residual_ok
                    else "Residual exceeds the ±0.5 keV ASTM acceptance band."
                ),
                section="ASTM E181 §6.3",
                value=_format_float(residual, suffix=" keV"),
                limit="±0.5 keV",
            ),
            StandardsCheck(
                key="counting_statistics",
                label="Counting statistics",
                status=count_status,
                message=(
                    "Primary check-source statistics are strong."
                    if count_status == "green"
                    else "Primary check-source net counts are low for a robust ASTM trend line."
                ),
                section="ASTM E181 §6",
                value=_format_float(primary_counts, precision=0),
                limit="≥ 1000 counts",
            ),
        )
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=checks,
            locked_settings=self.locked_settings(),
            summary=(
                "ASTM E181 locks the energy calibration to a 2nd-order fit and "
                "requires residuals inside the ±0.5 keV band."
            ),
        )


__all__ = ["E181Module"]
