"""Activation and decay relationships for flux-wire analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from fluxforge.core.linalg import vector_average


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
