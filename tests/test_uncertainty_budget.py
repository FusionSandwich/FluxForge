"""Tests for fluxforge.uncertainty.budget."""

from __future__ import annotations

import numpy as np
import pytest

from fluxforge.uncertainty.budget import (
    UncertaintyBudget,
    UncertaintyCategory,
    UncertaintyComponent,
    combine_budgets,
    create_activation_budget,
    create_flux_unfolding_budget,
    create_k0_naa_budget,
)


def test_uncertainty_component_factories() -> None:
    rel = UncertaintyComponent.from_relative(
        UncertaintyCategory.COUNTING_STATISTICS,
        relative=0.1,
        measurement=50.0,
        description="counting",
    )
    assert rel.value == 5.0
    assert rel.relative == 0.1

    abs_comp = UncertaintyComponent.from_absolute(
        UncertaintyCategory.EFFICIENCY,
        value=2.5,
        measurement=50.0,
        description="eff",
    )
    assert abs_comp.value == 2.5
    assert abs(abs_comp.relative - 0.05) < 1e-12


def test_uncertainty_budget_core_operations() -> None:
    budget = UncertaintyBudget(measurement=100.0, units="a.u.", name="demo")
    budget.add_relative(UncertaintyCategory.COUNTING_STATISTICS, 0.1, "counting")
    budget.add_absolute(UncertaintyCategory.EFFICIENCY, 5.0, "eff")

    total_quad = budget.compute_total(method="quadrature")
    assert abs(total_quad - np.sqrt(10.0**2 + 5.0**2)) < 1e-12
    assert abs(budget.relative_total - total_quad / 100.0) < 1e-12

    dominant = budget.dominant_component()
    assert dominant is not None
    assert dominant.category == UncertaintyCategory.COUNTING_STATISTICS

    frac_counting = budget.fraction_by_category(UncertaintyCategory.COUNTING_STATISTICS)
    assert abs(frac_counting - (10.0**2) / (10.0**2 + 5.0**2)) < 1e-12

    total_linear = budget.compute_total(method="linear")
    assert total_linear == 15.0

    summary = budget.summary_table()
    assert "Uncertainty Budget: demo" in summary
    assert "Component Breakdown" in summary

    as_dict = budget.to_dict()
    assert as_dict["measurement"] == 100.0
    assert as_dict["name"] == "demo"
    assert len(as_dict["components"]) == 2

    with pytest.raises(ValueError):
        budget.compute_total(method="bad-method")


def test_uncertainty_budget_zero_measurement_relative_behavior() -> None:
    budget = UncertaintyBudget(measurement=0.0)
    budget.add_absolute(UncertaintyCategory.OTHER, 1.0)
    budget.compute_total()
    assert budget.relative_total == 0.0
    assert budget.fraction_by_category(UncertaintyCategory.OTHER) == 1.0


def test_combine_budgets_variants() -> None:
    b1 = UncertaintyBudget(measurement=10.0, name="b1")
    b1.add_absolute(UncertaintyCategory.COUNTING_STATISTICS, 1.0)
    b1.compute_total()

    b2 = UncertaintyBudget(measurement=20.0, name="b2")
    b2.add_absolute(UncertaintyCategory.COUNTING_STATISTICS, 2.0)
    b2.compute_total()

    combined = combine_budgets([b1, b2])
    assert combined.name == "combined"
    assert combined.measurement == 15.0
    expected_unc = np.sqrt(1.0**2 + 2.0**2) / 2.0
    assert abs(combined.total_uncertainty - expected_unc) < 1e-12
    assert len(combined.components) >= 1

    combined_corr = combine_budgets(
        [b1, b2],
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    assert combined_corr.total_uncertainty >= combined.total_uncertainty

    assert combine_budgets([]).measurement == 0.0
    assert combine_budgets([b1]) is b1


def test_factory_budgets_include_expected_categories() -> None:
    activation = create_activation_budget(1_000.0)
    assert activation.name == "activation_analysis"
    assert activation.total_uncertainty > 0.0
    assert len(activation.components) == 8

    k0 = create_k0_naa_budget(100.0)
    categories = {c.category for c in k0.components}
    assert UncertaintyCategory.K0_FACTOR in categories
    assert UncertaintyCategory.ALPHA_PARAMETER in categories
    assert k0.total_uncertainty > 0.0

    unfolding = create_flux_unfolding_budget(1e10)
    assert unfolding.name == "flux_unfolding"
    assert len(unfolding.components) == 5
    assert unfolding.total_uncertainty > 0.0
