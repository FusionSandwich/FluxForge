import pytest

from fluxforge.k0_physics import (
    calculate_q0_alpha,
    counting_factor,
    decay_factor,
    saturation_factor,
    sdc_factor,
)


def test_q0_alpha_identity_at_zero_alpha():
    assert calculate_q0_alpha(15.7, 0.0, 5.65) == pytest.approx(15.7)


def test_sdc_factor_matches_components():
    half_life_s = 3600.0
    t_irr_s = 1200.0
    t_decay_s = 300.0
    t_count_s = 900.0

    s, d, c, sdc = sdc_factor(half_life_s, t_irr_s, t_decay_s, t_count_s)
    decay_const = 0.6931471805599453 / half_life_s

    assert s == pytest.approx(saturation_factor(decay_const, t_irr_s))
    assert d == pytest.approx(decay_factor(decay_const, t_decay_s))
    assert c == pytest.approx(counting_factor(decay_const, t_count_s))
    assert sdc == pytest.approx(s * d * c)


def test_sdc_factor_handles_nonpositive_half_life():
    assert sdc_factor(0.0, 1.0, 1.0, 1.0) == (0.0, 0.0, 0.0, 0.0)
