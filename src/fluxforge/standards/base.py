"""Standards module contracts and evaluation payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class LockedSetting:
    """Represents a standards-locked parameter."""

    field_id: str
    value: str
    standard_section: str


@dataclass(frozen=True)
class StandardsCheck:
    """One standards-compliance check surfaced to the GUI and reports."""

    key: str
    label: str
    status: str
    message: str
    section: str
    value: str = ""
    limit: str = ""


@dataclass(frozen=True)
class StandardsEvaluation:
    """Aggregate standards-compliance result for one standard."""

    standard_id: str
    display_name: str
    checks: tuple[StandardsCheck, ...]
    locked_settings: tuple[LockedSetting, ...] = ()
    summary: str = ""

    @property
    def overall_status(self) -> str:
        if any(check.status == "red" for check in self.checks):
            return "red"
        if any(check.status == "amber" for check in self.checks):
            return "amber"
        return "green"

    @property
    def compliant(self) -> bool:
        return self.overall_status == "green"


@dataclass(frozen=True)
class StandardsEvaluationContext:
    """Normalized runtime inputs consumed by standards modules."""

    calibration_order: int | None = None
    calibration_rms_keV: float | None = None
    max_residual_keV: float | None = None
    efficiency_uncertainty_pct: float | None = None
    fwhm_at_413_keV: float | None = None
    qa_centroid_drift_keV: float | None = None
    qa_fwhm_degradation_pct: float | None = None
    before_calibration: datetime | None = None
    measured_at: datetime | None = None
    after_calibration: datetime | None = None
    net_counts: Mapping[str, float] = field(default_factory=dict)
    line_observations: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


class StandardsModule(ABC):
    """Base class for standards-constrained workflows."""

    standard_id: str
    display_name: str

    @classmethod
    def description(cls) -> str:
        """Return a short human-readable description."""

        return getattr(cls, "summary", cls.display_name)

    @abstractmethod
    def locked_settings(self) -> tuple[LockedSetting, ...]:
        """Return the parameters locked by this standard."""

    @abstractmethod
    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        """Evaluate the active workflow state against this standard."""


def _format_float(value: float | None, *, precision: int = 3, suffix: str = "") -> str:
    """Return a compact text representation used in standards summaries."""

    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}{suffix}"


__all__ = [
    "LockedSetting",
    "StandardsCheck",
    "StandardsEvaluation",
    "StandardsEvaluationContext",
    "StandardsModule",
    "_format_float",
]
