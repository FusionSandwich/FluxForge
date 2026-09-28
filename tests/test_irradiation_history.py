"""Irradiation history and EOI handling in activity-to-rate conversion (issue #197)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fluxforge.analysis.flux_unfold import (
    activity_to_reaction_rate,
    irradiation_history_factor,
)
from fluxforge.examples.rafm_workflow import (
    TimingInfo,
    build_flux_wire_reactions,
    load_rafm_example_metadata,
)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "RAFM_irradiation"
CU64_HALF_LIFE_S = 12.701 * 3600.0
CO60_HALF_LIFE_S = 5.2714 * 365.25 * 86400.0


def _bateman(half_life_s: float, segments, steps: int = 200_000) -> float:
    """Integrate dA/dt = lambda (p(t) - A) with R*N = 1 numerically."""
    lam = math.log(2) / half_life_s
    activity = 0.0
    for duration, power in segments:
        dt = duration / steps
        for _ in range(steps if duration > 0 else 0):
            # exact step for constant power over dt
            activity = power + (activity - power) * math.exp(-lam * dt)
    return activity


def test_constant_interval_matches_saturation() -> None:
    lam = math.log(2) / CU64_HALF_LIFE_S
    assert irradiation_history_factor(CU64_HALF_LIFE_S, 7200.0) == pytest.approx(
        1 - math.exp(-lam * 7200.0)
    )


def test_segmented_history_matches_bateman() -> None:
    segments = [(4 * 3600.0, 1.0), (16 * 3600.0, 0.0), (2 * 3600.0, 0.5), (3600.0, 1.0)]
    expected = _bateman(CU64_HALF_LIFE_S, segments, steps=2000)
    observed = irradiation_history_factor(CU64_HALF_LIFE_S, irradiation_history=segments)
    assert observed == pytest.approx(expected, rel=1e-9)


def test_short_lived_product_depends_on_segment_order() -> None:
    early = irradiation_history_factor(CU64_HALF_LIFE_S, irradiation_history=[(3600, 1), (36000, 0)])
    late = irradiation_history_factor(CU64_HALF_LIFE_S, irradiation_history=[(36000, 0), (3600, 1)])
    assert late > 1.5 * early


def test_missing_timing_is_an_error_not_saturation() -> None:
    with pytest.raises(ValueError, match="Irradiation time or history is required"):
        activity_to_reaction_rate(1000.0, 4e19, CO60_HALF_LIFE_S, 0.0)
    with pytest.raises(ValueError):
        activity_to_reaction_rate(1000.0, 4e19, CO60_HALF_LIFE_S, None)
    saturated = activity_to_reaction_rate(
        1000.0, 4e19, CO60_HALF_LIFE_S, None, assume_saturated=True
    )
    assert saturated == pytest.approx(1000.0 / 4e19)


def test_rate_roundtrip_with_history() -> None:
    history = [(3600.0, 1.0), (1800.0, 0.0), (3600.0, 0.8)]
    rate = 2.5e-13
    n_atoms = 1e19
    activity = rate * n_atoms * irradiation_history_factor(CU64_HALF_LIFE_S, irradiation_history=history)
    recovered = activity_to_reaction_rate(
        activity, n_atoms, CU64_HALF_LIFE_S, None, irradiation_history=history
    )
    assert recovered == pytest.approx(rate, rel=1e-12)


def _timing() -> TimingInfo:
    return TimingInfo(
        sample_group="flux_wires",
        compare_eoi=True,
        irradiation_phase="phase2_whale_tube",
        irradiation_end=None,
        irradiation_time_s=7200.0,
        decay_time_s=1989660.0,
        measurement_time=None,
        decay_label=None,
        schedule_source="test",
    )


def test_measurement_activity_is_never_used_as_eoi_activity() -> None:
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    payload = {"Co60": {"activity_bq": 2000.0, "activity_unc_bq": 50.0, "activity_eoi_bq": None}}
    reactions = build_flux_wire_reactions(
        "Co-RAFM-1_25cm", "co-rafm-1", payload, _timing(), metadata
    )
    assert len(reactions) == 1
    assert reactions[0].reaction_rate == 0.0
    assert "no end-of-irradiation activity" in reactions[0].rate_note


def test_eoi_activity_uses_schedule_history() -> None:
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    payload = {"Co60": {"activity_bq": 2000.0, "activity_unc_bq": 50.0,
                        "activity_eoi_bq": 2052.0, "activity_eoi_unc_bq": 51.0}}
    timing = _timing()
    constant = build_flux_wire_reactions("Co-RAFM-1_25cm", "co-rafm-1", payload, timing, metadata)[0]
    timing.irradiation_history = [(3600.0, 1.0), (3600.0, 0.5)]
    segmented = build_flux_wire_reactions("Co-RAFM-1_25cm", "co-rafm-1", payload, timing, metadata)[0]
    # Same EOI activity from 3/4 of the fluence -> 4/3 higher rate (Co-60 is long-lived)
    assert segmented.reaction_rate / constant.reaction_rate == pytest.approx(4 / 3, rel=1e-4)
