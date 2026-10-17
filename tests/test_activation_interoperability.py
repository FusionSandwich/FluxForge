"""Independent thin-target, covariance and inventory/history contracts."""

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from fluxforge.analysis.efficiency_covariance import efficiency_evaluation_covariance
from fluxforge.core.probability import covariance_matrix
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.io.fispact import (
    fispact_history_text,
    read_fispact_activities,
    write_fispact_history,
)
from fluxforge.physics.target_monitor import (
    ELEMENTARY_CHARGE_C,
    target_monitor_calibration,
    target_monitor_scale_draws,
)


def test_thin_target_with_decay_pause_cooling_and_doubly_charged_beam():
    lam = math.log(2) / 10
    particle_rate = 3e-6 / (2 * ELEMENTARY_CHARGE_C)
    areal_density, sigma = 1e20, 1e-25
    expected = quad(
        lambda t: areal_density
        * sigma
        * particle_rate
        * lam
        * math.exp(-lam * (20 - t)),
        0,
        10,
    )[0]
    result = target_monitor_calibration(
        expected * 1.7,
        areal_density,
        sigma,
        [3, 0],
        [10, 5],
        half_life_s=10,
        cooling_time_s=5,
        charge_state=2,
    )
    assert result["current_scale"] == pytest.approx(1.7)
    assert result["expected_reference_activity_bq"] == pytest.approx(expected)
    assert result["calibrated_particle_rates_s"][0] == pytest.approx(
        particle_rate * 1.7
    )
    assert result["current_scale_uncertainty"] is None


def test_target_joint_gradient_against_finite_differences():
    inputs = np.array([2.0, 1e20, 1e-25, 3.0, 4.0])

    def calculate(v, **kw):
        return target_monitor_calibration(*v[:3], v[3:], [10, 20], half_life_s=30, **kw)

    gradient = calculate(inputs)["gradient"]
    for i in range(len(inputs)):
        delta = abs(inputs[i]) * 1e-5
        high, low = inputs.copy(), inputs.copy()
        high[i] += delta
        low[i] -= delta
        derivative = (
            calculate(high)["current_scale"] - calculate(low)["current_scale"]
        ) / (2 * delta)
        assert gradient[i] == pytest.approx(derivative, rel=1e-6)
    # One shared current calibration term: it must not average down per segment.
    nuisance = np.array([0, 0, 0, 0.3, 0.4])
    result = calculate(inputs, joint_covariance=np.outer(nuisance, nuisance))
    assert result["current_scale_uncertainty"] / result[
        "current_scale"
    ] == pytest.approx(0.1)


def test_joint_input_draws_preserve_common_activity_current_cancellation():
    draws = [[2, 1e20, 1e-25, 3], [4, 1e20, 1e-25, 6]]
    result = target_monitor_scale_draws(draws, [10], half_life_s=30)
    assert result[0] == pytest.approx(result[1])


def test_covariance_validation_does_not_depend_on_mixed_units():
    bad = np.array([[1e24, 1], [1, 1e-48]])
    with pytest.raises(ValueError, match="semidefinite"):
        covariance_matrix(bad, 2, "mixed")
    good = np.array([[1e24, 6e-13], [6e-13, 1e-48]])
    np.testing.assert_allclose(covariance_matrix(good, 2, "mixed"), good)


@pytest.mark.parametrize(
    "bad", [[[0, 1e-50], [1e-50, 1]], [[1e-40, 1e-41], [0, 1e-40]]]
)
def test_invalid_zero_and_tiny_asymmetric_covariance(bad):
    with pytest.raises(ValueError):
        covariance_matrix(bad, 2, "bad")


def inventory(tmp_path, rows):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"inventory_data": [{"nuclides": rows}]}))
    return path


def test_fispact_isomer_identity_zero_and_missing_are_distinct(tmp_path):
    path = inventory(
        tmp_path, [{"zai": 270600, "activity": 0}, {"zai": 270601, "activity": 3.2}]
    )
    assert read_fispact_activities(path, interval_index=0) == {270600: 0, 270601: 3.2}
    with pytest.raises(KeyError, match="absent"):
        read_fispact_activities(path, interval_index=0, required_zai=[270602])
    with pytest.raises(ValueError, match="unavailable"):
        read_fispact_activities(path, interval_index=1)


@pytest.mark.parametrize(
    "rows",
    [
        [{"zai": 270600, "activity": -1}],
        [{"zai": 270600.1, "activity": 1}],
        [{"zai": 270600, "activity": 1}, {"zai": 270600, "activity": 2}],
    ],
)
def test_bad_inventory_records_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        read_fispact_activities(inventory(tmp_path, rows), interval_index=0)


def test_history_export_keeps_shutdown_and_refuses_overwrite(tmp_path):
    text = fispact_history_text([1e12, 0], [10, 20])
    assert (
        text == "FLUX 1000000000000\nTIME 10 SECS\nATOMS\nFLUX 0\nTIME 20 SECS\nATOMS\n"
    )
    path = write_fispact_history(tmp_path / "history.txt", [1], [10])
    with pytest.raises(FileExistsError):
        write_fispact_history(path, [1], [10])


def curve():
    return EfficiencyCurve.from_polynomial(
        [-4, -0.1],
        (50, 2000),
        uncertainty_model={
            "type": "fit_covariance",
            "parameter_names": ["a0", "a1"],
            "covariance": [[0.01, -0.0001], [-0.0001, 0.0002]],
            "relative_systematic": 0.03,
        },
    )


def test_cross_energy_covariance_matches_jacobian_and_diagonal_uncertainties():
    fitted = curve()
    energy = np.array([100, 1000])
    nominal = np.atleast_1d(fitted.efficiency(energy))
    jacobian = nominal[:, None] * np.c_[np.ones(2), np.log(energy)]
    covariance = np.array(fitted.uncertainty_model["covariance"])
    expected = jacobian @ covariance @ jacobian.T + np.outer(nominal, nominal) * 0.03**2
    actual = efficiency_evaluation_covariance(
        fitted, energy, systematic_correlation="shared"
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        np.sqrt(np.diag(actual)) / nominal, fitted.efficiency_uncertainty(energy)
    )
    assert actual[0, 1] != 0


def test_efficiency_correlation_and_missing_budget_not_invented():
    fitted = curve()
    with pytest.raises(ValueError, match="systematic_correlation"):
        efficiency_evaluation_covariance(fitted, [100, 1000])
    with pytest.raises(ValueError, match="extrapolate"):
        efficiency_evaluation_covariance(fitted, [20], systematic_correlation="shared")
    fitted.uncertainty_model = None
    with pytest.raises(ValueError, match="covariance is required"):
        efficiency_evaluation_covariance(fitted, [100])


def test_spe_sum_preserves_covariance_and_acquisition_clocks(tmp_path, monkeypatch):
    from scipy.sparse import csr_matrix
    from fluxforge.io import spe

    paths = [tmp_path / "one.spe", tmp_path / "two.spe"]
    for i, path in enumerate(paths):
        path.write_text(str(i))
    spectra = [
        spe.GammaSpectrum(
            counts=[4, 9],
            counts_covariance=csr_matrix([[4, 1], [1, 9]]),
            channels=[0, 1],
            live_time=10,
            real_time=11,
            detector_id="hpge",
        ),
        spe.GammaSpectrum(
            counts=[1, 4],
            channels=[0, 1],
            live_time=20,
            real_time=22,
            detector_id="hpge",
        ),
    ]
    monkeypatch.setattr(spe, "read_spe_file", lambda path: spectra[paths.index(path)])
    result = spe.read_multiple_spe(paths, sum_spectra=True)
    np.testing.assert_allclose(result.counts, [5, 13])
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[5, 1], [1, 13]])
    assert result.live_time == 30 and result.real_time == 33
    assert result.start_time is None
    assert result.metadata["single_contiguous_acquisition"] is False
    assert len(result.metadata["acquisition_intervals"]) == 2
    with pytest.raises(ValueError, match="repeated paths"):
        spe.read_multiple_spe([paths[0], paths[0]], sum_spectra=True)
    spectra[1].channels = np.array([1, 2])
    with pytest.raises(ValueError, match="identical native grids"):
        spe.read_multiple_spe(paths, sum_spectra=True)
