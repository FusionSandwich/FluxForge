"""Physical and diagnostic regression checks for HPGe efficiency fits."""

import numpy as np
import pytest

from fluxforge.analysis.detector_calibration import EfficiencyPoint, fit_efficiency_curve
from fluxforge.analysis.efficiency_models import (
    semi_empirical_efficiency,
    semi_empirical_efficiency_uncertainty,
)
from fluxforge.core.analysis_workspace import fit_efficiency_model
from fluxforge.data.efficiency import EfficiencyCurve, calculate_efficiency_from_source


def _points(energies, efficiencies, relative_uncertainty=0.01):
    return tuple(
        EfficiencyPoint(
            energy_keV=float(energy),
            net_counts=float(efficiency * 1e8),
            live_time_s=1.0,
            activity_bq=1e8,
            emission_probability=1.0,
            count_uncertainty=float(efficiency * 1e8 * relative_uncertainty),
        )
        for energy, efficiency in zip(energies, efficiencies)
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("measured_counts", 0.0),
        ("measured_counts", -1.0),
        ("live_time", 0.0),
        ("source_activity", float("nan")),
        ("emission_probability", 0.0),
        ("emission_probability", 1.1),
        ("geometry_factor", 0.0),
        ("count_uncertainty", -1.0),
        ("activity_uncertainty", float("inf")),
        ("probability_uncertainty", -0.1),
    ],
)
def test_invalid_source_inputs_fail_explicitly(field, invalid):
    kwargs = dict(
        measured_counts=100.0,
        live_time=10.0,
        source_activity=1000.0,
        emission_probability=0.5,
        geometry_factor=1.0,
        count_uncertainty=10.0,
    )
    kwargs[field] = invalid
    with pytest.raises(ValueError):
        calculate_efficiency_from_source(**kwargs)


def test_nonphysical_efficiency_and_bad_energy_are_not_clipped_or_skipped():
    with pytest.raises(ValueError, match="absolute efficiency"):
        calculate_efficiency_from_source(2000, 1, 1000, 1)
    points = list(_points([100, 200, 400, 800], [0.01, 0.008, 0.005, 0.003]))
    points[1].energy_keV = 0.0
    with pytest.raises(ValueError, match="energy"):
        fit_efficiency_curve(points)
    with pytest.raises(ValueError, match="energy"):
        fit_efficiency_model(points, model_key="gray_functional")


def test_gray_fits_actual_inverse_energy_basis_and_covariance_order():
    energies = np.array([70, 100, 150, 220, 350, 500, 800, 1200, 1800], dtype=float)
    true = np.array([-2.0, -0.45, 0.015, -28.0])
    log_e = np.log(energies)
    values = np.exp(true[0] + true[1] * log_e + true[2] * log_e**2 + true[3] / energies)
    fit = fit_efficiency_model(_points(energies, values), model_key="gray_functional")
    actual = np.array([fit.curve.parameters[name] for name in ("a", "b", "c", "d")])
    assert actual == pytest.approx(true, rel=1e-6, abs=1e-6)
    assert fit.covariance_parameters == ("a", "b", "c", "d")
    assert np.shape(fit.covariance) == (4, 4)
    basis = np.column_stack((np.ones_like(energies), log_e, log_e**2, 1 / energies))
    expected_covariance = np.linalg.inv((basis / 0.01).T @ (basis / 0.01))
    assert np.asarray(fit.covariance) == pytest.approx(expected_covariance)
    assert max(abs(value) for value in fit.percentage_residuals) < 1e-8
    assert set(fit.point_status) == {"within_3_percent"}


def test_semi_empirical_full_fit_recovers_reference_parameters():
    energies = np.array([30, 45, 60, 80, 120, 180, 280, 450, 700, 1100, 1700, 2600], dtype=float)
    true = np.array([0.02, 2.0, 1.5, 1.0, 0.5])
    values = semi_empirical_efficiency(energies, true)
    fit = fit_efficiency_model(_points(energies, values), model_key="semi_empirical_hpge")
    assert fit.curve.parameters["coefficients"] == pytest.approx(true, rel=1e-5)
    assert fit.fit_quality["identifiability"] == "full"
    assert fit.covariance_parameters == (
        "scale", "length_g_cm2", "alpha", "length0_g_cm2", "kappa"
    )
    assert np.shape(fit.covariance) == (5, 5)
    assert fit.fit_quality["review_status"] == "pass"


def test_sparse_semi_empirical_fit_declares_fixed_assumptions_and_poor_fit():
    energies = [121.78, 356.01, 661.657, 1173.228, 1332.492]
    values = [0.0082, 0.0041, 0.0019, 0.00125, 0.00103]
    fit = fit_efficiency_model(_points(energies, values), model_key="semi_empirical_hpge")
    assert fit.fit_quality["identifiability"] == "conditional_sparse"
    assert fit.fit_quality["fixed_parameters"] == ("length_g_cm2", "length0_g_cm2")
    assert fit.covariance_parameters == ("scale", "alpha", "kappa")
    assert np.shape(fit.covariance) == (3, 3)
    assert fit.curve.parameters["fixed_coefficients"] == {
        "length_g_cm2": 8.3,
        "length0_g_cm2": 1.66,
    }
    assert np.isfinite(fit.curve.efficiency_uncertainty(661.657))
    assert fit.fit_quality["review_status"] == "review_required"
    assert "outside_3_percent" in fit.point_status


def test_semi_empirical_rejects_insufficient_distinct_energies():
    points = _points([100, 100, 200, 300], [0.01, 0.009, 0.008, 0.007])
    with pytest.raises(ValueError, match="distinct energies"):
        fit_efficiency_model(points, model_key="semi_empirical_hpge")


def test_negative_covariance_cannot_turn_into_zero_efficiency_uncertainty():
    coefficients = [0.02, 2.0, 1.5, 1.0, 0.5]
    with pytest.raises(ValueError, match="positive semidefinite"):
        semi_empirical_efficiency_uncertainty(
            np.array([661.657]), coefficients, np.array([[-1.0]]),
            parameter_indices=(0,),
        )
    curve = EfficiencyCurve(
        model_type="functional",
        parameters={"form": "semi_empirical_hpge", "coefficients": coefficients},
        uncertainty_model={
            "type": "fit_covariance", "parameter_names": ["scale"],
            "covariance": [[-1.0]],
        },
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        curve.efficiency_uncertainty(661.657)


def test_small_percentage_residuals_do_not_hide_statistically_bad_fit():
    energies = [100, 200, 400, 800, 1600, 2400]
    values = [0.01, 0.00808, 0.0065, 0.00505, 0.00408, 0.0035]
    fit = fit_efficiency_model(
        _points(energies, values, relative_uncertainty=0.001),
        model_key="log_poly_2",
    )
    assert fit.fit_quality["max_absolute_percentage_residual"] < 3
    assert fit.fit_quality["chi_square_p_value"] < 0.05
    assert fit.fit_quality["review_status"] == "review_required"
