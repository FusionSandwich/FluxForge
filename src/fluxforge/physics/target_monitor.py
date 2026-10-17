"""Thin-target activation calibration of incident charged-particle current.

The caller supplies isotope areal density and an effective production cross
section. No lithium abundance, beam charge state, or negligible-decay assumption
is silently imposed. This is not a thick-target neutron-yield calculation.
"""

from __future__ import annotations

import math

import numpy as np

from fluxforge.core.probability import covariance_matrix

ELEMENTARY_CHARGE_C = 1.602176634e-19


def target_monitor_calibration(
    activity_bq,
    areal_density_atoms_cm2,
    cross_section_cm2,
    currents_uA,
    durations_s,
    *,
    half_life_s,
    cooling_time_s=0,
    charge_state=1,
    joint_covariance=None,
):
    """Infer a common multiplicative current scale from measured activity.

    Activity refers to cooling_time_s after EOI. Currents are TOTAL beam current
    intercepted by the target. Uniform target isotope areal density is atoms/cm²
    along the beam. Effective cross section must account for the actual beam
    energy distribution; thin-target/no depletion and unchanged response across
    segments are assumptions. Durations include explicit zero-current shutdowns.

    Optional joint covariance order is [activity, areal density, cross section,
    current_0, ...], in the input units. First-order propagation preserves shared
    current/source errors. Timing, half-life, geometry and energy-model errors
    are conditional inputs, not implicitly included in that covariance.
    """
    current, duration = (np.asarray(v, dtype=float) for v in (currents_uA, durations_s))
    if (
        current.ndim != 1
        or not current.size
        or current.shape != duration.shape
        or not np.isfinite(current).all()
        or not np.isfinite(duration).all()
        or np.any(current < 0)
        or np.any(duration <= 0)
    ):
        raise ValueError(
            "currents/durations require matching nonnegative/positive vectors"
        )
    for name, value in (
        ("areal_density_atoms_cm2", areal_density_atoms_cm2),
        ("cross_section_cm2", cross_section_cm2),
        ("half_life_s", half_life_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if (
        not math.isfinite(activity_bq)
        or activity_bq < 0
        or not math.isfinite(cooling_time_s)
        or cooling_time_s < 0
    ):
        raise ValueError("activity and cooling time must be finite and nonnegative")
    if (
        isinstance(charge_state, bool)
        or not isinstance(charge_state, int)
        or charge_state < 1
    ):
        raise ValueError("charge_state must be a positive integer magnitude")
    decay = math.log(2) / half_life_s
    later = np.cumsum(duration[::-1])[::-1] - duration
    weights = -np.expm1(-decay * duration) * np.exp(-decay * later)
    weighted_current = float(weights @ current)
    conversion = 1e-6 / (ELEMENTARY_CHARGE_C * charge_state)
    expected = (
        areal_density_atoms_cm2
        * cross_section_cm2
        * weighted_current
        * conversion
        * math.exp(-decay * cooling_time_s)
    )
    if not math.isfinite(expected) or expected <= 0:
        raise ValueError("expected reference activity must be finite and positive")
    scale = activity_bq / expected
    if not math.isfinite(scale):
        raise ValueError("current scale is not finite")
    gradient = np.r_[
        1 / expected,
        -scale / areal_density_atoms_cm2,
        -scale / cross_section_cm2,
        -scale * weights / weighted_current,
    ]
    uncertainty = None
    if joint_covariance is not None:
        covariance = covariance_matrix(
            joint_covariance, len(gradient), "joint_covariance"
        )
        terms = gradient[:, None] * covariance * gradient[None, :]
        variance = float(terms.sum())
        if variance < -1e-12 * np.abs(terms).sum():
            raise ValueError("propagated current-scale variance is negative")
        uncertainty = math.sqrt(max(variance, 0))
    return {
        "current_scale": scale,
        "current_scale_uncertainty": uncertainty,
        "expected_reference_activity_bq": expected,
        "calibrated_particle_rates_s": (scale * current * conversion).tolist(),
        "gradient": gradient.tolist(),
        "uncertainty_scope": (
            "first_order_declared_joint_inputs_fixed_times_and_half_life"
            if uncertainty is not None
            else "unavailable"
        ),
        "physical_qualified": False,
    }


def target_monitor_scale_draws(joint_draws, durations_s, **fixed_inputs):
    """Transform caller-supplied JOINT input draws, retaining their correlations.

    Column order matches the covariance order above; no independent-error
    assumption or Gaussian perturbation is manufactured here.
    """
    values = np.asarray(joint_draws, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3 + len(durations_s)
    ):
        raise ValueError(
            "joint_draws columns must be A, areal density, sigma, currents"
        )
    if "joint_covariance" in fixed_inputs:
        raise ValueError(
            "joint draws and linearized covariance are separate calculations"
        )
    return np.array(
        [
            target_monitor_calibration(
                row[0], row[1], row[2], row[3:], durations_s, **fixed_inputs
            )["current_scale"]
            for row in values
        ]
    )
