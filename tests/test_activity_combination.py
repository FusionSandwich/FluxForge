import json
from dataclasses import replace

import numpy as np
import pytest

from fluxforge.analysis.activity_combination import (
    ActivityLine,
    CovarianceComponent,
    METHODS,
    combine_activity_lines,
)


def lines(activities=(10.0, 20.0), sigmas=(2.0, 4.0)):
    return [
        ActivityLine(
            str(i),
            "Co60",
            a,
            s,
            True,
            "frozen upstream qualification",
            "source fixture",
            "physical net counts",
            "count start",
        )
        for i, (a, s) in enumerate(zip(activities, sigmas))
    ]


def combine(rows, method, **kwargs):
    return combine_activity_lines(
        rows,
        method=method,
        uncertainty_definition="absolute 1 sigma Bq",
        engine_identity="a7bcc680d1f5e06b1d9dae405241fc380087ca2b",
        analysis_role="method_control",
        **kwargs,
    )


def component(matrix, name="total"):
    return CovarianceComponent(
        name, matrix, "fixture covariance", "absolute Bq squared"
    )


def test_hand_calculated_same_input_control():
    # A/sigma = [5,5] => mean 15; inverse variance = [1/4,1/16] => 12.
    historical = combine(lines(), METHODS[0])
    inverse = combine(lines(), METHODS[1])
    assert historical["activity_bq"] == pytest.approx(15)
    assert historical["normalized_weights"] == pytest.approx([0.5, 0.5])
    assert historical["sigma_bq"] == pytest.approx(np.sqrt(5))
    assert inverse["activity_bq"] == pytest.approx(12)
    assert inverse["normalized_weights"] == pytest.approx([0.8, 0.2])
    assert inverse["sigma_bq"] == pytest.approx(np.sqrt(3.2))
    assert historical["input_lines"] == inverse["input_lines"]
    json.dumps(historical, allow_nan=False)


@pytest.mark.parametrize("method", METHODS)
def test_single_line_is_unchanged(method):
    out = combine(
        lines((13.0,), (3.0,)), method, covariance_components=[component([[9.0]])]
    )
    assert out["activity_bq"] == pytest.approx(13)
    assert out["sigma_bq"] == pytest.approx(3)
    assert out["normalized_weights"] == [1.0]


def test_fully_shared_uncertainty_does_not_shrink():
    for n in (1, 2, 5):
        out = combine(
            lines((10.0,) * n, (3.0,) * n),
            "gls",
            covariance_components=[component(np.full((n, n), 9.0))],
            singular_policy="exact_constraints",
        )
        assert out["sigma_bq"] == pytest.approx(3)
        assert out["status"] == "available"


def test_gls_keeps_negative_weights_and_correct_residual_covariance():
    c = [[1.0, 1.5], [1.5, 4.0]]
    out = combine(
        lines((10.0, 20.0), (1.0, 2.0)), "gls", covariance_components=[component(c)]
    )
    assert out["normalized_weights"] == pytest.approx([1.25, -0.25])
    assert out["activity_bq"] == pytest.approx(7.5)
    assert out["sigma_bq"] ** 2 == pytest.approx(0.875)
    assert out["diagnostics"]["negative_weight_line_ids"] == ["1"]
    assert out["diagnostics"]["residual_sigmas_bq"] == pytest.approx(
        [np.sqrt(0.125), np.sqrt(3.125)]
    )


@pytest.mark.parametrize(
    "bad, message",
    [
        ([[1]], "shape"),
        ([[1, 0.2], [0.1, 1]], "symmetric"),
        ([[1, 2], [2, 1]], "positive semidefinite"),
        ([[1, 0], [0, -1]], "positive semidefinite"),
        ([[np.nan, 0], [0, 1]], "finite"),
        ([[np.inf, 0], [0, 1]], "finite"),
    ],
)
def test_invalid_covariance(bad, message):
    with pytest.raises(ValueError, match=message):
        combine(lines(), "gls", covariance_components=[component(bad)])


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_uncertainties_are_not_floored(sigma, method):
    with pytest.raises(ValueError):
        combine(lines((10.0,), (sigma,)), method)


def test_tiny_resolved_uncertainty_is_preserved():
    for method in METHODS:
        out = combine(
            lines((10.0,), (1e-100,)),
            method,
            covariance_components=[component([[1e-200]])],
        )
        assert out["sigma_bq"] == pytest.approx(1e-100, rel=1e-12, abs=0)


def test_diagonal_mismatch_and_indefinite_subcomponent_rejected():
    with pytest.raises(ValueError, match="diagonal"):
        combine(lines(), "gls", covariance_components=[component(np.eye(2))])
    with pytest.raises(ValueError, match="positive semidefinite"):
        combine(
            lines(),
            "gls",
            covariance_components=[
                component([[1, 2], [2, 1]]),
                component(20 * np.eye(2), "other"),
            ],
        )


def test_missing_components_never_become_zero():
    components = [
        component(np.diag([4.0, 16.0]), "count"),
        component(None, "efficiency"),
        component(None, "yield"),
    ]
    for method in METHODS:
        absent = combine(lines(), method, covariance_components=components)
        assert absent["status"] == "unavailable"
        assert absent["activity_bq"] is None
        assert absent["unavailable_components"] == ["efficiency", "yield"]
        partial = combine(
            lines(),
            method,
            covariance_components=components,
            allow_incomplete_uncertainty=True,
        )
        assert partial["status"] == "conditional"
        assert partial["covariance_sources"][1]["matrix_bq2"] is None
        json.dumps(partial, allow_nan=False)
    assert combine(lines(), "gls")["status"] == "unavailable"
    assert (
        combine([replace(lines()[0], sigma_bq=None)], "inverse_variance")["status"]
        == "unavailable"
    )


def test_shared_plus_independent_error_has_calibration_lower_bound():
    for n in (2, 4, 8):
        out = combine(
            lines((10.0,) * n, (np.sqrt(13),) * n),
            "gls",
            covariance_components=[
                component(4 * np.eye(n), "count"),
                component(9 * np.ones((n, n)), "shared efficiency/yield"),
            ],
        )
        assert out["sigma_bq"] ** 2 == pytest.approx(9 + 4 / n)


def test_singular_requires_explicit_exact_model_and_flags_contradiction():
    c = component([[9.0, 9.0], [9.0, 9.0]])
    out = combine(lines((10.0, 12.0), (3.0, 3.0)), "gls", covariance_components=[c])
    assert out["status"] == "unavailable"
    out = combine(
        lines((10.0, 12.0), (3.0, 3.0)),
        "gls",
        covariance_components=[c],
        singular_policy="exact_constraints",
    )
    assert out["status"] == "inconsistent"
    assert out["diagnostics"]["incompatible_exact_constraints"]
    assert out["diagnostics"]["covariance_rank"] == 1
    # A naive pseudoinverse gives mean 11 and quietly ignores the exact contrast.
    assert out["activity_bq"] == pytest.approx(11)


def test_singular_nullspace_can_identify_exact_mean():
    # Perfect anticorrelation: equal weights have zero uncertainty.
    out = combine(
        lines((10.0, 12.0), (1.0, 1.0)),
        "gls",
        covariance_components=[component([[1.0, -1.0], [-1.0, 1.0]])],
        singular_policy="exact_constraints",
    )
    assert out["status"] == "available"
    assert out["activity_bq"] == pytest.approx(11)
    assert out["sigma_bq"] == pytest.approx(0)


@pytest.mark.parametrize("epsilon", [1e-13, 1e-18])
def test_near_singular_is_unavailable_without_regularization(epsilon):
    out = combine(
        lines((10.0, 11.0), (1.0, np.sqrt(epsilon))),
        "gls",
        covariance_components=[component(np.diag([1.0, epsilon]))],
    )
    assert out["status"] == "unavailable"
    assert out["normalized_weights"] is None
    assert out["diagnostics"]["regularization"] is None
    # These controls don't invert the covariance and remain valid.
    out = combine(
        lines((10.0, 11.0), (1.0, np.sqrt(epsilon))),
        "inverse_variance",
        covariance_components=[component(np.diag([1.0, epsilon]))],
    )
    assert out["status"] == "available"


@pytest.mark.parametrize("method", METHODS)
def test_excluded_ni57_sc48_not_admitted_by_method(method):
    excluded = [
        replace(
            lines()[0],
            isotope=iso,
            qualified=False,
            qualification=reason,
            activity_bq=None,
            sigma_bq=None,
        )
        for iso, reason in [
            ("Ni57", "unqualified assignment"),
            ("Sc48", "inconsistent source yield"),
        ]
    ]
    for row in excluded:
        out = combine([row], method)
        assert out["qualified_line_ids"] == []
        assert out["status"] == "unavailable"
        assert out["exclusions"] == [{"line_id": "0", "reason": row.qualification}]
    accepted = lines()
    out = combine(
        accepted
        + [replace(excluded[0], line_id="ni"), replace(excluded[1], line_id="sc")],
        method,
        covariance_components=[component(np.diag([4.0, 16.0]))],
    )
    assert out["qualified_line_ids"] == ["0", "1"]
    assert len(out["exclusions"]) == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("isotope", "Sc48"),
        ("count_basis", "QG comparison"),
        ("activity_reference", "EOI"),
    ],
)
def test_mixed_inputs_rejected(field, value):
    rows = lines()
    rows[1] = replace(rows[1], **{field: value})
    with pytest.raises(ValueError, match="share"):
        combine(rows, "inverse_variance")


def test_duplicate_and_missing_identities_rejected():
    with pytest.raises(ValueError, match="unique"):
        combine([lines()[0], lines()[0]], "inverse_variance")
    with pytest.raises(ValueError, match="source_identity"):
        combine([replace(lines()[0], source_identity="")], "inverse_variance")
    with pytest.raises(ValueError, match="boolean"):
        combine([replace(lines()[0], qualified="yes")], "inverse_variance")


def test_historical_weights_cannot_be_silently_used_for_signed_lines():
    with pytest.raises(ValueError, match="positive activities"):
        combine(lines((-10.0, 20.0)), METHODS[0])
    assert combine(lines((-10.0, 20.0)), "inverse_variance")[
        "activity_bq"
    ] == pytest.approx(-4)


def test_residual_contrast_and_scatter_not_hidden_by_reselection():
    out = combine(
        lines((10.0, 20.0), (1.0, 1.0)),
        "gls",
        covariance_components=[component(np.eye(2))],
    )
    assert out["qualified_line_ids"] == ["0", "1"]
    assert out["diagnostics"]["chi_square"] == pytest.approx(50)
    assert out["diagnostics"]["chi_square_dof"] == 1
    assert out["diagnostics"]["line_inconsistency_flags"] == [True, True]
    assert out["diagnostics"]["standardized_residuals"] == pytest.approx(
        [-np.sqrt(50), np.sqrt(50)]
    )


def test_gls_matches_independent_constrained_solve():
    c = np.array([[4.0, 0.3, 0.8], [0.3, 9.0, 1.0], [0.8, 1.0, 16.0]])
    rows = lines((12.0, 14.0, 20.0), np.sqrt(np.diag(c)))
    out = combine(rows, "gls", covariance_components=[component(c)])
    # KKT equality-constrained variance minimization, separate solver oracle.
    kkt = np.block([[c, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
    w = np.linalg.solve(kkt, [0.0, 0.0, 0.0, 1.0])[:3]
    assert out["normalized_weights"] == pytest.approx(w)
    assert out["sigma_bq"] ** 2 == pytest.approx(w @ c @ w)


def test_large_common_offset_cannot_hide_exact_contradiction():
    for baseline in (0.0, 1e15):
        out = combine(
            lines((baseline, baseline + 10), (1.0, 1.0)),
            "gls",
            covariance_components=[component(np.ones((2, 2)))],
            singular_policy="exact_constraints",
        )
        assert out["status"] == "inconsistent"
        assert out["diagnostics"]["residuals_bq"] == pytest.approx([-5, 5])
        assert out["diagnostics"]["line_inconsistency_flags"] == [True, True]


def test_numpy_scalars_export_as_json_and_overflow_is_not_available():
    out = combine(
        lines((np.float32(10), np.int64(20)), (np.float32(2), np.float32(4))),
        "inverse_variance",
    )
    json.dumps(out, allow_nan=False)
    with pytest.raises(ValueError, match="floating point range"):
        combine(lines((1e200, 2e200), (1.0, 1.0)), "inverse_variance")


def test_fixed_weight_singular_cancellation_is_reported_without_inverse():
    out = combine(
        lines((10.0, 10.0), (1.0, 1.0)),
        "inverse_variance",
        covariance_components=[component([[1.0, -1.0], [-1.0, 1.0]])],
    )
    assert out["sigma_bq"] == 0
    assert out["diagnostics"]["exact_cancellation"]
    assert out["diagnostics"]["chi_square"] is None
    assert out["diagnostics"]["rank_diagnostics_available"] is False


def test_large_baseline_preserves_stochastic_inconsistency_flags():
    out = combine(
        lines((1e15, 1e15 + 10), (1.0, 1.0)),
        "gls",
        covariance_components=[component(np.eye(2))],
    )
    assert out["diagnostics"]["chi_square"] == pytest.approx(50)
    assert out["diagnostics"]["line_inconsistency_flags"] == [True, True]


@pytest.mark.parametrize("method", METHODS)
def test_no_available_budget_is_unavailable_even_with_opt_in(method):
    out = combine(
        lines(),
        method,
        covariance_components=[component(None)],
        allow_incomplete_uncertainty=True,
    )
    assert out["status"] == "unavailable"


def test_example_source_hashes_and_no_new_isotope_admission():
    import runpy
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "examples/activity_combination/compare_same_lines.py"
    )
    payload = runpy.run_path(str(script))["build_example"]()
    assert payload["same_input_relative_method_change_percent"] == pytest.approx(
        -0.16697448068921927
    )
    for control in ("fixed_ASC_control", "historical_report_control"):
        group = payload[control]
        assert group["complete_uncertainty_gls"]["status"] == "unavailable"
        methods = group["conditional_same_input_methods"]
        assert all(m["status"] == "conditional" for m in methods)
        assert all(m["input_lines"] == methods[0]["input_lines"] for m in methods)
        assert {r["isotope"] for m in methods for r in m["input_lines"]} == {"Co60"}
    physical = payload["physical_synthetic_shared_covariance_control"]
    assert physical["sigma_bq"] ** 2 == pytest.approx(26)
    json.dumps(payload, allow_nan=False)
