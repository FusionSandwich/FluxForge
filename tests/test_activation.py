import math

from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    irradiation_buildup_factor,
    reaction_rate_from_activity,
    weighted_activity,
)


def test_activity_from_counts_no_decay():
    line = GammaLineMeasurement(
        net_counts=1000,
        live_time_s=10.0,
        efficiency=0.1,
        gamma_intensity=0.5,
        half_life_s=1e9,
        cooling_time_s=0.0,
    )
    activity = line.activity_at_reference()
    expected = 1000 / (0.1 * 0.5 * 10)
    assert math.isclose(activity, expected, rel_tol=1e-6)


def test_weighted_activity_combines_lines():
    lines = [
        GammaLineMeasurement(1000, 10.0, 0.1, 0.5, 1e9),
        GammaLineMeasurement(2000, 10.0, 0.1, 0.5, 1e9),
    ]
    activity, sigma = weighted_activity(lines)
    assert activity > 0
    assert sigma > 0


def test_reaction_rate_single_segment():
    activity = 5.0
    half_life = 60.0
    segments = [IrradiationSegment(duration_s=30.0, relative_power=1.0)]
    factor = irradiation_buildup_factor(segments, half_life)
    rate_estimate = reaction_rate_from_activity(activity, segments, half_life)
    assert math.isclose(rate_estimate.rate * factor, activity, rel_tol=1e-6)

