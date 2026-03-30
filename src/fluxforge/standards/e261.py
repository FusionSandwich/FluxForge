"""ASTM E261 stub for future neutron-fluence standards workflows."""

from __future__ import annotations

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
)


class E261Module(StandardsModule):
    """Stub module kept in the standards registry for roadmap completeness."""

    standard_id = "ASTM E261"
    display_name = "ASTM E261"
    summary = "Stub neutron-fluence standards module reserved for future scope."

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return ()

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=(
                StandardsCheck(
                    key="stub",
                    label="Scope",
                    status="amber",
                    message="ASTM E261 is registered as a roadmap stub and is not yet a full neutron workflow.",
                    section="roadmap",
                    value="stub",
                    limit="future",
                ),
            ),
            locked_settings=(),
            summary="Roadmap stub only.",
        )


__all__ = ["E261Module"]
