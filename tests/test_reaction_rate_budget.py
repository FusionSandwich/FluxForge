"""Reaction-rate uncertainty budgets and shared covariance (issue #199)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxforge.uncertainty.reaction_rate_budget import (
    REQUIRED_COMPONENTS,
    RateUncertaintyBudget,
    UncertaintyComponent,
    floor_as_component,
    rate_covariance,
)


def _budget(row_id, rate, counting, efficiency_group="det:South"):
    return RateUncertaintyBudget(
        row_id,
        rate,
        [
            UncertaintyComponent("activity", counting),
            UncertaintyComponent("detector_efficiency", 0.03, efficiency_group, "cert"),
        ],
    )


def test_shared_efficiency_correlates_rows_on_same_detector() -> None:
    a = _budget("a", 2.0, 0.04)
    b = _budget("b", 5.0, 0.02)
    c = _budget("c", 1.0, 0.01, efficiency_group="det:North")
    cov = rate_covariance([a, b, c])
    assert cov[0, 0] == pytest.approx((0.04**2 + 0.03**2) * 4.0)
    assert cov[0, 1] == pytest.approx(0.03 * 0.03 * 2.0 * 5.0)
    assert cov[0, 2] == 0.0 and cov[1, 2] == 0.0
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-15)


def test_missing_components_are_listed_not_zero() -> None:
    budget = _budget("a", 1.0, 0.05)
    assert "half_life" in budget.missing and "activity" not in budget.missing
    assert set(budget.missing) <= set(REQUIRED_COMPONENTS)
    assert budget.as_row()["missing_components"].count(";") == len(budget.missing) - 1


def test_floor_is_expressed_as_explicit_component() -> None:
    component = floor_as_component("model_floor", 0.1, 0.25, "config")
    assert math.hypot(0.1, component.relative) == pytest.approx(0.25)
    assert floor_as_component("model_floor", 0.3, 0.25, "config") is None
