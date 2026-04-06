"""ASTM C1232 laboratory QA checks."""

from __future__ import annotations

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
    _format_float,
)


class C1232Module(StandardsModule):
    """ASTM C1232-style QA gate for drift and resolution degradation."""

    standard_id = "ASTM C1232"
    display_name = "ASTM C1232"
    summary = "Flags drift and detector resolution degradation for laboratory QA review."

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return (
            LockedSetting(
                field_id="qa_baseline",
                value="historical-check-source-baseline",
                standard_section="ASTM C1232 §9",
            ),
        )

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        drift = context.qa_centroid_drift_keV
        drift_status = "green"
        if drift is not None and drift > 1.0:
            drift_status = "red"
        elif drift is not None and drift > 0.5:
            drift_status = "amber"

        fwhm = context.qa_fwhm_degradation_pct
        fwhm_status = "green"
        if fwhm is not None and fwhm > 20.0:
            fwhm_status = "red"
        elif fwhm is not None and fwhm > 10.0:
            fwhm_status = "amber"

        checks = (
            StandardsCheck(
                key="centroid_drift",
                label="Centroid drift",
                status=drift_status,
                message=(
                    "Centroid drift remains within the laboratory QA band."
                    if drift_status == "green"
                    else "Centroid drift exceeds the laboratory QA acceptance band."
                ),
                section="ASTM C1232 §9",
                value=_format_float(drift, suffix=" keV"),
                limit="≤ 0.5 keV amber / ≤ 1.0 keV red",
            ),
            StandardsCheck(
                key="fwhm_degradation",
                label="FWHM degradation",
                status=fwhm_status,
                message=(
                    "Resolution degradation remains within the QA band."
                    if fwhm_status == "green"
                    else "Resolution degradation exceeds the QA acceptance band."
                ),
                section="ASTM C1232 §9",
                value=_format_float(fwhm, suffix="%"),
                limit="≤ 10% amber / ≤ 20% red",
            ),
        )
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=checks,
            locked_settings=self.locked_settings(),
            summary="ASTM C1232 uses QA baselines to detect centroid and FWHM drift.",
        )


__all__ = ["C1232Module"]
