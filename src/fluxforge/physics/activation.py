"""Activation and decay relationships for flux-wire analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from fluxforge.core.linalg import vector_average
from fluxforge.data.elements import atomic_mass, parse_isotope


AVOGADRO = 6.02214076e23


def count_decay_factor(half_life_s: float, real_time_s: float) -> float:
    """Count-average/count-start activity for a clock-time acquisition.

    Zero duration means the supplied activity is already at count start.
    Live-time acceptance is separate; callers must apply it only once.
    """
    if not math.isfinite(half_life_s) or half_life_s <= 0.0:
        raise ValueError("half_life_s must be finite and positive")
    if not math.isfinite(real_time_s) or real_time_s < 0.0:
        raise ValueError("real_time_s must be finite and nonnegative")
    exponent = (math.log(2.0) / half_life_s) * real_time_s
    return -math.expm1(-exponent) / exponent if exponent > 0.0 else 1.0


@dataclass
class GammaLineMeasurement:
    """Represents a single gamma-line observation used to infer activity."""

    net_counts: float
    live_time_s: float
    efficiency: float
    gamma_intensity: float
    half_life_s: float
    cooling_time_s: float = 0.0
    dead_time_fraction: float = 0.0
    real_time_s: float | None = None
    net_counts_unc: float | None = None

    def activity_at_reference(self) -> float:
        """Return the activity at the chosen reference (usually EOI)."""

        for name in ("live_time_s", "efficiency", "gamma_intensity", "half_life_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("net_counts", "cooling_time_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not math.isfinite(self.dead_time_fraction)
            or not 0 <= self.dead_time_fraction < 1
        ):
            raise ValueError("dead_time_fraction must be finite and in [0, 1)")
        # Uniform acceptance: C = eps * yield * A_start * live * C_decay(real).
        # Live time already accounts for lost counts; dividing counts by the
        # acceptance as well would correct dead time twice.
        real_time = self.real_time_s
        if real_time is None:
            real_time = self.live_time_s / (1.0 - self.dead_time_fraction)
        if not math.isfinite(real_time) or real_time < self.live_time_s:
            raise ValueError("real_time_s must be finite and at least live_time_s")
        if self.dead_time_fraction and not math.isclose(
            self.live_time_s / real_time,
            1.0 - self.dead_time_fraction,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("real_time_s and dead_time_fraction are inconsistent")
        buildup = self.live_time_s * count_decay_factor(self.half_life_s, real_time)
        decay_const = math.log(2.0) / self.half_life_s
        activity_at_count_start = self.net_counts / (
            self.efficiency * self.gamma_intensity * buildup
        )
        activity_ref = activity_at_count_start * math.exp(
            decay_const * self.cooling_time_s
        )
        return activity_ref

    def activity_uncertainty_at_reference(self) -> float:
        """Propagate absolute count sigma with other inputs held fixed.

        Missing sigma retains the legacy Poisson net-count assumption. This is
        conditional and is not a background-subtraction uncertainty model.
        """
        self.activity_at_reference()  # Validate the same clock and count inputs.
        sigma = self.net_counts_unc
        if sigma is None:
            sigma = math.sqrt(self.net_counts)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("net_counts_unc must be finite and nonnegative")
        real = self.real_time_s
        if real is None:
            real = self.live_time_s / (1.0 - self.dead_time_fraction)
        derivative = math.exp(
            math.log(2.0) * self.cooling_time_s / self.half_life_s
        ) / (
            self.efficiency
            * self.gamma_intensity
            * self.live_time_s
            * count_decay_factor(self.half_life_s, real)
        )
        result = derivative * sigma
        if not math.isfinite(result):
            raise ValueError("Propagated activity uncertainty must be finite")
        return result

    def count_uncertainty_metadata(self) -> dict[str, object]:
        return {
            "source": (
                "supplied net_counts_unc"
                if self.net_counts_unc is not None
                else "assumed Poisson net counts; background uncertainty unavailable"
            ),
            "assumed": self.net_counts_unc is None,
            "uncertainty_scope": "conditional",
            "scientific_admission": False,
            "fixed_inputs": [
                "efficiency",
                "gamma_intensity",
                "half_life_s",
                "cooling_time_s",
                "live_time_s",
                "real_time_s",
            ],
        }


@dataclass
class IrradiationSegment:
    """Segment of an irradiation timeline."""

    duration_s: float
    relative_power: float = 1.0


@dataclass
class ReactionRateEstimate:
    """Container holding a reaction rate estimate and propagated uncertainty."""

    rate: float
    uncertainty: float | None
    uncertainty_scope: str = "unavailable"
    uncertainty_unavailable_reason: str = "Activity uncertainty was not supplied."


def weighted_activity(
    gamma_lines: Iterable[GammaLineMeasurement],
) -> tuple[float, float]:
    """Compute a weighted mean activity and uncertainty from multiple lines."""

    activities = []
    variances = []
    for line in gamma_lines:
        activity = line.activity_at_reference()
        variance = line.activity_uncertainty_at_reference() ** 2
        if variance <= 0.0:
            raise ValueError("Weighted activity requires positive line uncertainties")
        activities.append(activity)
        variances.append(variance)

    if not activities:
        raise ValueError("At least one gamma-line measurement is required.")

    weights = [1.0 / v for v in variances]
    weighted_mean = vector_average(activities, weights)
    combined_variance = 1.0 / sum(weights)
    return weighted_mean, math.sqrt(combined_variance)


def irradiation_buildup_factor(
    segments: Sequence[IrradiationSegment], half_life_s: float
) -> float:
    if not math.isfinite(half_life_s) or half_life_s <= 0:
        raise ValueError("half_life_s must be finite and positive")
    for segment in segments:
        if not math.isfinite(segment.duration_s) or segment.duration_s < 0:
            raise ValueError("Irradiation duration must be finite and nonnegative")
        if not math.isfinite(segment.relative_power) or segment.relative_power < 0:
            raise ValueError("Relative power must be finite and nonnegative")
    decay_const = math.log(2.0) / half_life_s
    total_duration = sum(seg.duration_s for seg in segments)
    elapsed = 0.0
    factor = 0.0
    for segment in segments:
        elapsed += segment.duration_s
        segment_term = segment.relative_power * (
            -math.expm1(-decay_const * segment.duration_s)
        )
        decay_after = math.exp(-decay_const * (total_duration - elapsed))
        factor += segment_term * decay_after
    return factor


def reaction_rate_from_activity(
    activity_eoi: float,
    segments: Sequence[IrradiationSegment],
    half_life_s: float,
    activity_uncertainty_bq: float | None = None,
) -> ReactionRateEstimate:
    """Propagate supplied activity sigma conditional on the declared irradiation.

    Activity in Bq is not a count and cannot supply a Poisson variance. Missing
    activity uncertainty remains unavailable; this does not qualify a complete
    activation uncertainty budget.
    """
    if not math.isfinite(activity_eoi) or activity_eoi < 0:
        raise ValueError("Activity must be finite and non-negative.")
    if activity_uncertainty_bq is not None and (
        not math.isfinite(activity_uncertainty_bq) or activity_uncertainty_bq < 0
    ):
        raise ValueError("Activity uncertainty must be finite and non-negative.")
    factor = irradiation_buildup_factor(segments, half_life_s)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Irradiation factor must be positive.")
    rate = activity_eoi / factor
    uncertainty = (
        activity_uncertainty_bq / factor
        if activity_uncertainty_bq is not None
        else None
    )
    if not math.isfinite(rate) or (
        uncertainty is not None and not math.isfinite(uncertainty)
    ):
        raise ValueError("Propagated rate and uncertainty must be finite.")
    return ReactionRateEstimate(
        rate=rate,
        uncertainty=uncertainty,
        uncertainty_scope=(
            "activity_only_conditional" if uncertainty is not None else "unavailable"
        ),
        uncertainty_unavailable_reason=(
            "" if uncertainty is not None else "Activity uncertainty was not supplied."
        ),
    )


def activity_to_atoms(activity_bq: float, half_life_s: float) -> float:
    """Convert activity in Bq to radioactive atom count."""

    if half_life_s <= 0:
        return 0.0
    decay_const = math.log(2.0) / half_life_s
    return activity_bq / max(decay_const, 1e-30)


def infer_nuclide_atomic_mass_g_mol(isotope: str | None) -> float | None:
    """Infer an isotope molar mass from an isotope label like `Co60` or `Sc46`."""

    if not isotope:
        return None
    try:
        _element, mass_number, _isomeric = parse_isotope(str(isotope))
    except ValueError:
        symbol = "".join(ch for ch in str(isotope) if ch.isalpha())
        return atomic_mass(symbol) or None
    return float(mass_number)


def activity_to_radioactive_mass_g(
    activity_bq: float, half_life_s: float, isotope: str | None = None
) -> float:
    """Infer radioactive product mass from activity and half-life."""

    molar_mass = infer_nuclide_atomic_mass_g_mol(isotope)
    if molar_mass is None or molar_mass <= 0.0:
        return 0.0
    atoms = activity_to_atoms(activity_bq, half_life_s)
    return (atoms / AVOGADRO) * molar_mass


def radioisotope_specific_activity_bq_g(
    half_life_s: float, isotope: str | None = None
) -> float:
    """Return the specific activity of a pure radioactive isotope in Bq/g."""

    molar_mass = infer_nuclide_atomic_mass_g_mol(isotope)
    if half_life_s <= 0.0 or molar_mass is None or molar_mass <= 0.0:
        return 0.0
    decay_const = math.log(2.0) / half_life_s
    return decay_const * AVOGADRO / molar_mass


def activation_study_metrics(
    *,
    activity_bq: float,
    activity_unc_bq: float,
    half_life_s: float,
    isotope: str | None = None,
    sample_mass_g: float | None = None,
) -> dict[str, float]:
    """Build common activation-study metrics derived from activity."""

    decay_constant = math.log(2.0) / half_life_s if half_life_s > 0.0 else 0.0
    radioisotope_specific_activity = radioisotope_specific_activity_bq_g(
        half_life_s, isotope=isotope
    )
    atoms = activity_to_atoms(activity_bq, half_life_s) if activity_bq > 0.0 else 0.0
    atoms_unc = (
        activity_to_atoms(activity_unc_bq, half_life_s)
        if activity_unc_bq > 0.0
        else 0.0
    )
    radioactive_mass_g = activity_to_radioactive_mass_g(
        activity_bq, half_life_s, isotope=isotope
    )
    radioactive_mass_unc_g = activity_to_radioactive_mass_g(
        activity_unc_bq, half_life_s, isotope=isotope
    )

    payload = {
        "decay_constant_s": float(decay_constant),
        "radioisotope_specific_activity_Bq_g": float(radioisotope_specific_activity),
        "atoms": float(atoms),
        "atoms_unc": float(atoms_unc),
        "radioactive_mass_g": float(radioactive_mass_g),
        "radioactive_mass_unc_g": float(radioactive_mass_unc_g),
    }
    if sample_mass_g is not None and sample_mass_g > 0.0:
        payload["sample_mass_g"] = float(sample_mass_g)
        payload["specific_activity_Bq_g"] = float(activity_bq / sample_mass_g)
        payload["specific_activity_unc_Bq_g"] = float(activity_unc_bq / sample_mass_g)
        payload["radioactive_mass_fraction"] = float(radioactive_mass_g / sample_mass_g)
    return payload
