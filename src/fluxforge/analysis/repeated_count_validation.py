"""Conditional repeated-count decay diagnostics (issue #241).

No irradiation buildup, reaction rate, calibration qualification, or source-ID
inference occurs here. Callers supply source-bound identities and explicit
method choices. Agreement tests decay consistency only; it cannot identify the
cause of a discrepancy or establish absolute activity/flux accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Sequence

from fluxforge.physics.activation import GammaLineMeasurement, count_decay_factor

UNIFORM_ACCEPTANCE = "uniform_live_fraction"
SIMPLE_DECAY = "simple_decay_no_feeding_or_reirradiation"
METHOD = "repeated_count_log_ratio_v1"


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class CountObservation:
    """One line/count or a processed activity, with no implicit conventions.

    ``value`` is net accepted counts for ``net_counts``; an already live-time
    normalized count-average for ``count_average``; or a point activity for
    ``reference_activity``. The latter requires includes_count_decay=True and
    an explicit reference. ``count_average`` requires False and gets exactly
    one finite-window correction. Neither processed basis gets a live-time
    correction. Raw counts must not have been dead-time normalized already.

    Point references: count_start, count_end, end_of_irradiation, timestamp.
    Unknowns remain None. A known duration can derive count end, but no missing
    acquisition start, EOI, time zone or irradiation history is invented.
    ``channel_id`` binds the same gamma line or same declared summary method.
    A shared calibration ID asserts comparable response/geometry, not accuracy.
    ``count_basis`` separates physical reductions from comparison scenarios.
    """

    measurement_id: str
    specimen_id: str | None
    nuclide: str | None
    channel_id: str | None
    calibration_id: str | None
    background_id: str | None
    count_basis: str | None
    source_sha256: str | None
    identity_basis: str | None
    half_life_s: float | None
    value: float | None
    input_kind: str
    count_start: datetime | None
    real_time_s: float | None
    live_time_s: float | None = None
    count_end: datetime | None = None
    efficiency: float | None = None
    gamma_intensity: float | None = None
    acceptance_assumption: str | None = None
    includes_count_decay: bool | None = None
    activity_reference: str | None = None
    reference_timestamp: datetime | None = None
    eoi: datetime | None = None
    activity_unit: str = "Bq"
    exclusion_reason: str | None = None


def _base(status: str, reasons: list[str]) -> dict:
    return {
        "status": status,
        "reasons": reasons,
        "method": METHOD,
        "scientific_admission": False,
        "qualification": "conditional decay consistency only",
    }


def _elapsed(later: datetime, earlier: datetime, common_naive_clock: str | None):
    """UTC arithmetic handles DST folds; naive clocks need a named scenario."""
    for timestamp in (later, earlier):
        if timestamp.utcoffset() is not None:
            roundtrip = timestamp.astimezone(timezone.utc).astimezone(timestamp.tzinfo)
            if (
                roundtrip.replace(tzinfo=None) != timestamp.replace(tzinfo=None)
                or roundtrip.fold != timestamp.fold
            ):
                raise ValueError("invalid_local_timestamp_or_dst_gap")
    aware_later = later.utcoffset() is not None
    aware_earlier = earlier.utcoffset() is not None
    if aware_later != aware_earlier:
        raise ValueError("mixed_aware_and_naive_timestamps")
    if aware_later:
        later, earlier = later.astimezone(timezone.utc), earlier.astimezone(
            timezone.utc
        )
    elif not common_naive_clock or not common_naive_clock.strip():
        raise ValueError("unknown_time_zone")
    return (later - earlier).total_seconds()


def _end(observation: CountObservation) -> datetime:
    start = observation.count_start
    if start.utcoffset() is not None:
        return start.astimezone(timezone.utc) + timedelta(
            seconds=observation.real_time_s
        )
    return start + timedelta(seconds=observation.real_time_s)


def _reference(observation: CountObservation, reference: str) -> datetime:
    if reference == "count_start":
        return observation.count_start
    if reference == "count_end":
        return observation.count_end or _end(observation)
    if reference == "end_of_irradiation":
        if observation.eoi is None:
            raise ValueError("unknown_eoi")
        return observation.eoi
    if reference == "timestamp":
        if observation.reference_timestamp is None:
            raise ValueError("unknown_reference_timestamp")
        return observation.reference_timestamp
    raise ValueError("unknown_activity_reference")


def normalize_activity(
    observation: CountObservation,
    *,
    target_reference: str = "count_start",
    common_naive_clock: str | None = None,
) -> dict:
    """Return a conditional activity or UNAVAILABLE/EXCLUDED with null values.

    The canonical GammaLineMeasurement/count_decay_factor own the integral:
    C = epsilon * p * A_start * L * count_decay_factor(half_life, R).
    This is equivalent to uniform live acceptance times the real-time emission
    integral. No Poisson uncertainty or saturation history is assumed.
    """
    out = _base("UNAVAILABLE", [])
    out.update(
        measurement_id=observation.measurement_id,
        activity_value=None,
        activity_unit=observation.activity_unit,
        activity_reference=target_reference,
        reference_timestamp=None,
        count_window_corrections=0,
        common_naive_clock_assumption=common_naive_clock,
        input_kind=observation.input_kind,
        source_sha256=observation.source_sha256,
    )
    if observation.exclusion_reason is not None and not _text(
        observation.exclusion_reason
    ):
        out["reasons"] = ["invalid_exclusion_reason"]
        return out
    if observation.exclusion_reason:
        out.update(status="EXCLUDED", reasons=[observation.exclusion_reason])
        return out
    missing = [
        name
        for name in ("count_start", "real_time_s", "half_life_s", "value")
        if getattr(observation, name) is None
    ]
    if missing:
        out["reasons"] = ["unknown_" + name for name in missing]
        return out
    try:
        if not isinstance(observation.count_start, datetime):
            raise ValueError("invalid_count_start")
        _elapsed(observation.count_start, observation.count_start, common_naive_clock)
        for name in ("real_time_s", "half_life_s"):
            value = getattr(observation, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid_" + name)
        if (
            isinstance(observation.value, bool)
            or not math.isfinite(observation.value)
            or observation.value < 0
        ):
            raise ValueError("invalid_value")
        if observation.count_end is not None:
            duration = _elapsed(
                observation.count_end, observation.count_start, common_naive_clock
            )
            if duration <= 0 or not math.isclose(
                duration, observation.real_time_s, abs_tol=1e-6, rel_tol=1e-9
            ):
                raise ValueError("inconsistent_count_end_and_real_time")
        if observation.live_time_s is not None and (
            isinstance(observation.live_time_s, bool)
            or not math.isfinite(observation.live_time_s)
            or not 0 < observation.live_time_s <= observation.real_time_s
        ):
            raise ValueError("invalid_live_time_s")
        if (
            observation.eoi is not None
            and _elapsed(observation.count_start, observation.eoi, common_naive_clock)
            < 0
        ):
            raise ValueError("eoi_after_count_start")
        lam = math.log(2.0) / observation.half_life_s
        if observation.input_kind == "net_counts":
            if (
                observation.includes_count_decay is not None
                and observation.includes_count_decay is not False
            ):
                raise ValueError("net_counts_conflicting_count_decay_declaration")
            if observation.activity_reference is not None:
                raise ValueError("net_counts_conflicting_activity_reference")
            if observation.acceptance_assumption != UNIFORM_ACCEPTANCE:
                raise ValueError("unknown_or_unsupported_live_acceptance")
            if observation.activity_unit != "Bq":
                raise ValueError("raw_count_activity_unit_must_be_Bq")
            if any(
                x is None
                for x in (
                    observation.live_time_s,
                    observation.efficiency,
                    observation.gamma_intensity,
                )
            ):
                raise ValueError("unknown_raw_count_response_or_live_time")
            if (
                isinstance(observation.efficiency, bool)
                or isinstance(observation.gamma_intensity, bool)
                or not 0 < observation.efficiency <= 1
                or not 0 < observation.gamma_intensity <= 1
            ):
                raise ValueError("invalid_efficiency_or_gamma_intensity")
            # Explicit R bypasses legacy fallback from live time/dead fraction.
            value = GammaLineMeasurement(
                observation.value,
                observation.live_time_s,
                observation.efficiency,
                observation.gamma_intensity,
                observation.half_life_s,
                real_time_s=observation.real_time_s,
            ).activity_at_reference()
            source_time = observation.count_start
            corrections = 1
        elif observation.input_kind == "count_average":
            if observation.activity_reference is not None:
                raise ValueError("count_average_conflicting_point_reference")
            if observation.includes_count_decay is not False:
                raise ValueError("declare_count_average_includes_count_decay_false")
            if observation.acceptance_assumption != UNIFORM_ACCEPTANCE:
                raise ValueError("unknown_or_unsupported_live_acceptance")
            value = observation.value / count_decay_factor(
                observation.half_life_s, observation.real_time_s
            )
            source_time = observation.count_start
            corrections = 1
        elif observation.input_kind == "reference_activity":
            if observation.includes_count_decay is not True:
                raise ValueError("declare_reference_activity_includes_count_decay_true")
            value = observation.value
            source_time = _reference(observation, observation.activity_reference)
            corrections = 0
        else:
            raise ValueError("unsupported_input_kind")
        target_time = _reference(observation, target_reference)
        dt = _elapsed(target_time, source_time, common_naive_clock)
        value *= math.exp(-lam * dt)
        if not math.isfinite(value) or (value == 0 and observation.value > 0):
            raise ValueError("activity_outside_numeric_range")
    except (ValueError, ArithmeticError, TypeError, AttributeError) as exc:
        out["reasons"] = [str(exc) or type(exc).__name__]
        return out
    out.update(
        status="AVAILABLE",
        activity_value=value,
        reference_timestamp=target_time.isoformat(),
        count_window_corrections=corrections,
    )
    return out


def compare_repeated_counts(
    observations: Sequence[CountObservation],
    *,
    relative_tolerance: float,
    decay_assumption: str | None,
    common_naive_clock: str | None = None,
) -> dict:
    """Compare every distinct pair; no fitting of half-life or target activities.

    The user-selected reporting tolerance applies symmetrically in log space:
    abs(log(A_j/A_i) + lambda*(t_j-t_i)) <= log(1+relative_tolerance).
    It is a deterministic diagnostic, not a significance test or an uncertainty
    budget. All pairs must be evaluable and consistent for overall CONSISTENT.
    Missing identities and incompatible inputs remain separate from residuals.
    """
    try:
        valid_tolerance = (
            not isinstance(relative_tolerance, bool)
            and math.isfinite(relative_tolerance)
            and relative_tolerance > 0
        )
    except TypeError:
        valid_tolerance = False
    if not valid_tolerance:
        raise ValueError("relative_tolerance must be finite and positive")
    out = _base("UNAVAILABLE", [])
    normalized = [
        normalize_activity(o, common_naive_clock=common_naive_clock)
        for o in observations
    ]
    out.update(
        relative_tolerance=relative_tolerance,
        tolerance_definition=(
            "abs(log(observed_ratio/expected_ratio)) " "<= log1p(relative_tolerance)"
        ),
        decay_assumption=decay_assumption,
        common_naive_clock_assumption=common_naive_clock,
        activities=normalized,
        pairs=[],
    )
    if len(observations) < 2:
        out["reasons"] = ["need_at_least_two_counts"]
        return out
    for i, j in combinations(range(len(observations)), 2):
        a, b = observations[i], observations[j]
        pair = _base("UNAVAILABLE", [])
        pair.update(
            measurements=[a.measurement_id, b.measurement_id],
            log_observed_ratio=None,
            log_expected_ratio=None,
            log_residual=None,
        )
        out["pairs"].append(pair)
        if _text(a.exclusion_reason) or _text(b.exclusion_reason):
            pair.update(
                status="EXCLUDED",
                reasons=[
                    r for r in (a.exclusion_reason, b.exclusion_reason) if _text(r)
                ],
            )
            continue
        mismatches, missing = [], []
        if not _text(a.measurement_id) or not _text(b.measurement_id):
            missing.append("unknown_measurement_id")
        elif a.measurement_id == b.measurement_id:
            mismatches.append("duplicate_measurement_identity")
        for name in (
            "specimen_id",
            "nuclide",
            "channel_id",
            "calibration_id",
            "background_id",
            "count_basis",
            "activity_unit",
        ):
            av, bv = getattr(a, name), getattr(b, name)
            if not _text(av) or not _text(bv):
                missing.append("unknown_" + name)
            elif av != bv:
                mismatches.append("incompatible_" + name)
        for o in (a, b):
            if not _text(o.identity_basis):
                missing.append("unknown_identity_basis")
            if (
                not isinstance(o.source_sha256, str)
                or len(o.source_sha256) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in o.source_sha256)
            ):
                missing.append("unknown_or_invalid_source_sha256")
        if (
            isinstance(a.source_sha256, str)
            and isinstance(b.source_sha256, str)
            and a.source_sha256.lower() == b.source_sha256.lower()
        ):
            mismatches.append("duplicate_source_sha256")
        if (
            a.half_life_s is not None
            and b.half_life_s is not None
            and a.half_life_s != b.half_life_s
        ):
            mismatches.append("incompatible_half_life_s")
        if mismatches:
            pair.update(status="INCOMPATIBLE", reasons=mismatches + missing)
            continue
        if decay_assumption != SIMPLE_DECAY:
            missing.append("unknown_or_unsupported_decay_history")
        for activity in (normalized[i], normalized[j]):
            if activity["status"] != "AVAILABLE":
                missing.extend(activity["reasons"])
        if missing:
            pair["reasons"] = sorted(set(missing))
            continue
        va, vb = normalized[i]["activity_value"], normalized[j]["activity_value"]
        if va <= 0 or vb <= 0:
            pair["reasons"] = ["nonpositive_activity_ratio_unavailable"]
            continue
        try:
            if (
                a.eoi is not None
                and b.eoi is not None
                and _elapsed(a.eoi, b.eoi, common_naive_clock) != 0
            ):
                pair.update(status="INCOMPATIBLE", reasons=["incompatible_eoi"])
                continue
            dt = _elapsed(b.count_start, a.count_start, common_naive_clock)
            if -b.real_time_s < dt < a.real_time_s:
                pair.update(
                    status="INCOMPATIBLE", reasons=["overlapping_count_windows"]
                )
                continue
            expected = -(math.log(2.0) / a.half_life_s) * dt
            observed = math.log(vb) - math.log(va)
            residual = observed - expected
            if not all(math.isfinite(x) for x in (expected, observed, residual)):
                raise ValueError("ratio_outside_numeric_range")
        except (ValueError, OverflowError) as exc:
            pair["reasons"] = [str(exc)]
            continue
        pair.update(
            status=(
                "CONSISTENT"
                if abs(residual) <= math.log1p(relative_tolerance)
                else "DISCREPANCY"
            ),
            log_observed_ratio=observed,
            log_expected_ratio=expected,
            log_residual=residual,
        )
    statuses = {p["status"] for p in out["pairs"]}
    out["status"] = (
        "CONSISTENT"
        if statuses == {"CONSISTENT"}
        else (
            "DISCREPANCY"
            if statuses <= {"CONSISTENT", "DISCREPANCY"}
            else (
                "PARTIAL"
                if statuses & {"CONSISTENT", "DISCREPANCY"}
                else (
                    "EXCLUDED"
                    if statuses == {"EXCLUDED"}
                    else (
                        "INCOMPATIBLE"
                        if statuses == {"INCOMPATIBLE"}
                        else "UNAVAILABLE"
                    )
                )
            )
        )
    )
    out["reasons"] = sorted({reason for p in out["pairs"] for reason in p["reasons"]})
    return out
