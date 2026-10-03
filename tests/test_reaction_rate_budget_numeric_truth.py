import numpy as np
import pytest
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    rate_covariance,
)


def test_small_rate_large_relative_has_finite_absolute_sigma():
    b = RateUncertaintyBudget(
        "tiny", 1e-200, [UncertaintyComponent("activity", 1e200, source="truth")]
    )
    assert b.total_relative == pytest.approx(1e200)
    assert b.total_absolute == pytest.approx(1)
    np.testing.assert_allclose(rate_covariance([b]), [[1]], rtol=1e-14)


def test_tiny_relative_large_rate_does_not_underflow_total():
    b = RateUncertaintyBudget(
        "large", 1e200, [UncertaintyComponent("activity", 1e-200, source="truth")]
    )
    assert b.total_absolute == pytest.approx(1)
    np.testing.assert_allclose(rate_covariance([b]), [[1]], rtol=1e-14)


def test_signed_common_source_anticorrelation():
    a = RateUncertaintyBudget(
        "a",
        10,
        [UncertaintyComponent("activity", 0.1, "shared", "truth", sensitivity_sign=1)],
    )
    b = RateUncertaintyBudget(
        "b",
        20,
        [UncertaintyComponent("activity", 0.2, "shared", "truth", sensitivity_sign=-1)],
    )
    np.testing.assert_allclose(rate_covariance([a, b]), [[1, -4], [-4, 16]], rtol=1e-14)


def test_full_covariance_signed_jacobian_preserves_exact_cancellation():
    covariance = [[4, 2], [2, 1]]
    component = UncertaintyComponent.from_covariance(
        "joint",
        covariance,
        [1, -2],
        input_names=["x", "y"],
        input_units=["m", "s"],
        source="truth",
        correlation_group="joint",
    )
    b = RateUncertaintyBudget("cancel", 3, [component])
    assert b.total_absolute == pytest.approx(0, abs=1e-12)
    assert rate_covariance([b])[0, 0] == pytest.approx(0, abs=1e-24)


def test_missing_components_stay_missing_even_with_reported_activity():
    b = RateUncertaintyBudget(
        "incomplete", 100, [UncertaintyComponent("activity", 0.03, source="reported")]
    )
    assert "target_mass" in b.missing
    assert not b.complete
    assert b.as_row()["scientific_admission"] is False
    with pytest.raises(ValueError, match="incomplete"):
        rate_covariance([b], require_complete=True)


def test_full_covariance_tiny_input_keeps_nonzero_relative():
    c = UncertaintyComponent.from_covariance(
        "tinycov",
        [[1e-300]],
        [1e-50],
        input_names=["x"],
        input_units=["s"],
        source="truth",
        correlation_group="tinycov",
    )
    assert c.relative == pytest.approx(1e-200, rel=1e-14, abs=0)
    b = RateUncertaintyBudget("large", 1e200, [c])
    assert b.total_absolute == pytest.approx(1)
    np.testing.assert_allclose(rate_covariance([b]), [[1]], rtol=1e-14)


def test_full_covariance_large_effect_keeps_finite_relative():
    c = UncertaintyComponent.from_covariance(
        "bigcov",
        [[1e300]],
        [1e50],
        input_names=["x"],
        input_units=["s"],
        source="truth",
        correlation_group="bigcov",
    )
    assert c.relative == pytest.approx(1e200, rel=1e-14)
    b = RateUncertaintyBudget("tiny", 1e-200, [c])
    assert b.total_absolute == pytest.approx(1)
    np.testing.assert_allclose(rate_covariance([b]), [[1]], rtol=1e-14)
