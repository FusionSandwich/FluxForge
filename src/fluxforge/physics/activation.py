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
    """Represents a single gamma-line observation used to infer activity."""

    net_counts: float
    live_time_s: float
    efficiency: float
    gamma_intensity: float
    half_life_s: float
    cooling_time_s: float = 0.0
    dead_time_fraction: float = 0.0

    def activity_at_reference(self) -> float:
        """Return the activity at the chosen reference (usually EOI)."""

        decay_const = math.log(2.0) / self.half_life_s
        corrected_counts = self.net_counts / max(1.0 - self.dead_time_fraction, 1e-12)
        buildup = (1.0 - math.exp(-decay_const * self.live_time_s)) / max(decay_const, 1e-12)
        if buildup <= 0:
            raise ValueError("Live time must be positive to compute activity.")
        activity_at_count_start = corrected_counts / (self.efficiency * self.gamma_intensity * buildup)
        activity_ref = activity_at_count_start * math.exp(decay_const * self.cooling_time_s)
        return activity_ref


@dataclass
class IrradiationSegment:
    """Segment of an irradiation timeline."""

    duration_s: float
    relative_power: float = 1.0


@dataclass
class ReactionRateEstimate:
    """Container holding a reaction rate estimate and propagated uncertainty."""

    rate: float
    uncertainty: float


def weighted_activity(gamma_lines: Iterable[GammaLineMeasurement]) -> tuple[float, float]:
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


def irradiation_buildup_factor(segments: Sequence[IrradiationSegment], half_life_s: float) -> float:
    decay_const = math.log(2.0) / half_life_s
    total_duration = sum(seg.duration_s for seg in segments)
    elapsed = 0.0
    factor = 0.0
    for segment in segments:
        elapsed += segment.duration_s
        segment_term = segment.relative_power * (1.0 - math.exp(-decay_const * segment.duration_s))
        decay_after = math.exp(-decay_const * (total_duration - elapsed))
        factor += segment_term * decay_after
    return factor


def reaction_rate_from_activity(activity_eoi: float, segments: Sequence[IrradiationSegment], half_life_s: float) -> ReactionRateEstimate:
    if activity_eoi < 0:
        raise ValueError("Activity must be non-negative.")
    factor = irradiation_buildup_factor(segments, half_life_s)
    if factor <= 0:
        raise ValueError("Irradiation factor must be positive.")
    rate = activity_eoi / factor
    uncertainty = rate / math.sqrt(max(activity_eoi, 1e-12))
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


def activity_to_radioactive_mass_g(activity_bq: float, half_life_s: float, isotope: str | None = None) -> float:
    """Infer radioactive product mass from activity and half-life."""

    molar_mass = infer_nuclide_atomic_mass_g_mol(isotope)
    if molar_mass is None or molar_mass <= 0.0:
        return 0.0
    atoms = activity_to_atoms(activity_bq, half_life_s)
    return (atoms / AVOGADRO) * molar_mass


def radioisotope_specific_activity_bq_g(half_life_s: float, isotope: str | None = None) -> float:
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
    radioisotope_specific_activity = radioisotope_specific_activity_bq_g(half_life_s, isotope=isotope)
    atoms = activity_to_atoms(activity_bq, half_life_s) if activity_bq > 0.0 else 0.0
    atoms_unc = activity_to_atoms(activity_unc_bq, half_life_s) if activity_unc_bq > 0.0 else 0.0
    radioactive_mass_g = activity_to_radioactive_mass_g(activity_bq, half_life_s, isotope=isotope)
    radioactive_mass_unc_g = activity_to_radioactive_mass_g(activity_unc_bq, half_life_s, isotope=isotope)

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
