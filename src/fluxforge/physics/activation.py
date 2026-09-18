"""Activation and decay relationships for flux-wire analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from fluxforge.core.linalg import vector_average
from fluxforge.data.elements import atomic_mass, parse_isotope


AVOGADRO = 6.02214076e23


@dataclass
class GammaLineMeasurement:
    """Represents a single gamma-line observation used to infer activity.

    Dead time is treated as a uniform live fraction over the real counting
    interval.  This is the only timing model available when event timestamps
    or a time-dependent live-time history are not supplied.
    """

    net_counts: float
    live_time_s: float
    efficiency: float
    gamma_intensity: float
    half_life_s: float
    cooling_time_s: float = 0.0
    dead_time_fraction: float | None = None
    real_time_s: float | None = None

    def resolved_count_timing(self) -> tuple[float, float]:
        """Return ``(real_time_s, live_fraction)`` after consistency checks."""

        live_time = float(self.live_time_s)
        if not math.isfinite(live_time) or live_time <= 0.0:
            raise ValueError("Live time must be finite and positive.")

        supplied_fraction = self.dead_time_fraction
        if supplied_fraction is not None:
            supplied_fraction = float(supplied_fraction)
            if not math.isfinite(supplied_fraction) or not 0.0 <= supplied_fraction < 1.0:
                raise ValueError("Dead-time fraction must be finite and in [0, 1).")

        if self.real_time_s is None:
            live_fraction = 1.0 - (supplied_fraction or 0.0)
            return live_time / live_fraction, live_fraction

        real_time = float(self.real_time_s)
        if not math.isfinite(real_time) or real_time <= 0.0:
            raise ValueError("Real time must be finite and positive when supplied.")
        if live_time > real_time and not math.isclose(live_time, real_time, rel_tol=1e-12):
            raise ValueError("Live time cannot exceed real time.")
        live_fraction = live_time / real_time
        if supplied_fraction is not None and not math.isclose(
            supplied_fraction,
            1.0 - live_fraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Real time is inconsistent with dead-time fraction.")
        return real_time, live_fraction

    def effective_counting_duration_s(self) -> float:
        """Return the decay-weighted live exposure under the uniform-live assumption."""

        if not math.isfinite(float(self.half_life_s)) or self.half_life_s <= 0.0:
            raise ValueError("Half-life must be finite and positive.")
        real_time, live_fraction = self.resolved_count_timing()
        decay_const = math.log(2.0) / float(self.half_life_s)
        decay_integral = -math.expm1(-decay_const * real_time) / decay_const
        return live_fraction * decay_integral

    def activity_per_net_count_at_reference(self) -> float:
        """Return the activity conversion factor for one signed net count."""

        efficiency = float(self.efficiency)
        gamma_intensity = float(self.gamma_intensity)
        cooling_time = float(self.cooling_time_s)
        if not math.isfinite(efficiency) or efficiency <= 0.0:
            raise ValueError("Efficiency must be finite and positive.")
        if not math.isfinite(gamma_intensity) or gamma_intensity <= 0.0:
            raise ValueError("Gamma intensity must be finite and positive.")
        if not math.isfinite(cooling_time) or cooling_time < 0.0:
            raise ValueError("Cooling time must be finite and non-negative.")
        exposure = self.effective_counting_duration_s()
        decay_const = math.log(2.0) / float(self.half_life_s)
        return math.exp(decay_const * cooling_time) / (
            efficiency * gamma_intensity * exposure
        )

    def activity_at_reference(self) -> float:
        """Return the activity at the chosen reference (usually EOI)."""

        net_counts = float(self.net_counts)
        if not math.isfinite(net_counts):
            raise ValueError("Net counts must be finite.")
        return net_counts * self.activity_per_net_count_at_reference()


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


def weighted_activity(
    gamma_lines: Iterable[GammaLineMeasurement],
) -> tuple[float, float]:
    """Compute a weighted mean activity and uncertainty from multiple lines."""

    activities = []
    variances = []
    for line in gamma_lines:
        activity = line.activity_at_reference()
        variance = activity * activity / max(line.net_counts, 1.0)
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
    activity_uncertainty: float | None = None,
) -> ReactionRateEstimate:
    if activity_eoi < 0:
        raise ValueError("Activity must be non-negative.")
    factor = irradiation_buildup_factor(segments, half_life_s)
    if factor <= 0:
        raise ValueError("Irradiation factor must be positive.")
    rate = activity_eoi / factor
    uncertainty = None
    if activity_uncertainty is not None:
        resolved_uncertainty = float(activity_uncertainty)
        if not math.isfinite(resolved_uncertainty) or resolved_uncertainty < 0.0:
            raise ValueError("Activity uncertainty must be finite and non-negative.")
        uncertainty = resolved_uncertainty / factor
    return ReactionRateEstimate(rate=rate, uncertainty=uncertainty)


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
