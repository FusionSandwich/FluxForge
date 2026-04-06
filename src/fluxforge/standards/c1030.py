"""ASTM C1030 plutonium isotopics helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
    _format_float,
)


PU241_HALF_LIFE_YEARS = 14.29


@dataclass(frozen=True)
class PuLineObservation:
    """One fitted plutonium-region line used by the C1030 wizard."""

    nuclide: str
    energy_keV: float
    net_counts: float
    efficiency: float
    uncertainty_counts: float


@dataclass(frozen=True)
class PuIsotopicsResult:
    """Computed plutonium isotopic ratios and classification."""

    pu240_to_pu239: float
    pu241_to_pu239: float
    am241_to_pu239: float
    age_years: float | None
    classification: str
    ratio_uncertainty: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-12)


def compute_pu_isotopics(
    observations: tuple[PuLineObservation, ...],
    *,
    source_age_years: float | None = None,
) -> PuIsotopicsResult:
    """Return ratio-based plutonium isotopics from fitted line observations."""

    totals: dict[str, float] = {"Pu-239": 0.0, "Pu-240": 0.0, "Pu-241": 0.0, "Am-241": 0.0}
    variance = 0.0
    for observation in observations:
        signal = observation.net_counts / max(observation.efficiency, 1e-9)
        totals[observation.nuclide] = totals.get(observation.nuclide, 0.0) + signal
        variance += max(observation.uncertainty_counts, 0.0) ** 2

    pu239 = max(totals["Pu-239"], 1e-9)
    pu240_ratio = _safe_ratio(totals["Pu-240"], pu239)
    pu241_ratio = _safe_ratio(totals["Pu-241"], pu239)
    am241_ratio = _safe_ratio(totals["Am-241"], pu239)

    age_years = source_age_years
    if age_years is None and totals["Pu-241"] > 0.0 and totals["Am-241"] > 0.0:
        ratio = min(max(_safe_ratio(totals["Am-241"], totals["Pu-241"]), 1e-9), 0.999999)
        age_years = -PU241_HALF_LIFE_YEARS * math.log(1.0 - ratio) / math.log(2.0)

    if age_years is not None:
        ingrowth = 1.0 - math.exp(-math.log(2.0) * max(age_years, 0.0) / PU241_HALF_LIFE_YEARS)
        am241_ratio *= max(ingrowth, 1e-6)

    if pu240_ratio < 0.07:
        classification = "weapons-grade"
    elif pu240_ratio < 0.19:
        classification = "fuel-grade"
    else:
        classification = "reactor-grade"

    ratio_uncertainty = math.sqrt(max(variance, 0.0)) / pu239
    return PuIsotopicsResult(
        pu240_to_pu239=pu240_ratio,
        pu241_to_pu239=pu241_ratio,
        am241_to_pu239=am241_ratio,
        age_years=age_years,
        classification=classification,
        ratio_uncertainty=ratio_uncertainty,
    )


class C1030Module(StandardsModule):
    """ASTM C1030 plutonium isotopics compliance module."""

    standard_id = "ASTM C1030"
    display_name = "ASTM C1030"
    summary = "Plutonium isotopic ratios, age correction, and classification checks."

    def locked_settings(self) -> tuple[LockedSetting, ...]:
        return (
            LockedSetting(
                field_id="branching_ratio_table",
                value="ASTM C1030 locked values",
                standard_section="ASTM C1030 §8",
            ),
            LockedSetting(
                field_id="age_correction",
                value="Am-241 ingrowth from Pu-241",
                standard_section="ASTM C1030 §9",
            ),
        )

    def evaluate(self, context: StandardsEvaluationContext) -> StandardsEvaluation:
        observations = tuple(context.line_observations.get("c1030", ()))
        result = compute_pu_isotopics(
            observations,
            source_age_years=context.extra.get("source_age_years"),
        ) if observations else None

        pu240_peak_counts = float(context.net_counts.get("Pu-240 160.3", 0.0))
        resolution_ok = (
            context.fwhm_at_413_keV is not None and context.fwhm_at_413_keV <= 1.2
        )
        stats_ok = pu240_peak_counts >= 1000.0
        efficiency_ok = (
            context.efficiency_uncertainty_pct is not None
            and context.efficiency_uncertainty_pct <= 3.0
        )
        checks = (
            StandardsCheck(
                key="resolution_413",
                label="Resolution at 413.7 keV",
                status="green" if resolution_ok else "red",
                message=(
                    "Detector resolution satisfies the C1030 line-separation requirement."
                    if resolution_ok
                    else "Detector resolution is insufficient for ASTM C1030 isotopics."
                ),
                section="ASTM C1030 §8",
                value=_format_float(context.fwhm_at_413_keV, suffix=" keV"),
                limit="≤ 1.2 keV",
            ),
            StandardsCheck(
                key="pu240_statistics",
                label="Pu-240 160.3 keV statistics",
                status="green" if stats_ok else "red",
                message=(
                    "Pu-240 statistics support ASTM C1030 ratios."
                    if stats_ok
                    else "Pu-240 statistics are too low for ASTM C1030 ratios."
                ),
                section="ASTM C1030 §8",
                value=_format_float(pu240_peak_counts, precision=0),
                limit="≥ 1000 counts",
            ),
            StandardsCheck(
                key="efficiency_uncertainty",
                label="Efficiency uncertainty",
                status="green" if efficiency_ok else "red",
                message=(
                    "Efficiency calibration uncertainty satisfies ASTM C1030."
                    if efficiency_ok
                    else "Efficiency calibration uncertainty exceeds the ASTM C1030 cap."
                ),
                section="ASTM C1030 §8",
                value=_format_float(context.efficiency_uncertainty_pct, suffix="%"),
                limit="≤ 3%",
            ),
        )
        if result is not None:
            checks += (
                StandardsCheck(
                    key="classification",
                    label="Isotopic classification",
                    status="green",
                    message=f"Computed plutonium vector is {result.classification}.",
                    section="ASTM C1030 §10",
                    value=result.classification,
                    limit="reported",
                ),
            )
        return StandardsEvaluation(
            standard_id=self.standard_id,
            display_name=self.display_name,
            checks=checks,
            locked_settings=self.locked_settings(),
            summary="ASTM C1030 locks the plutonium isotopics workflow, branching ratios, and age correction.",
        )


__all__ = [
    "C1030Module",
    "PuIsotopicsResult",
    "PuLineObservation",
    "compute_pu_isotopics",
]
