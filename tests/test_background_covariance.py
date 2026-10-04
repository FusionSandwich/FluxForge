"""Counterexamples for shared counts, signed sidebands, and serialization."""

import json

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import (
    estimate_peak_area,
    estimate_peak_area_local_background,
)
from fluxforge.analysis.spectrum_math import (
    add_spectra,
    moving_average,
    subtract_measured_background,
    subtract_spectra,
)
from fluxforge.io.spe import GammaSpectrum


def fractional_grid_pair():
    sample = GammaSpectrum(counts=[10, 20], energies=np.array([0.5, 1.5]))
    background = GammaSpectrum(counts=[4, 9, 16], energies=np.array([0, 1, 2]))
    return sample, background


def test_shared_interpolation_counts_raise_sum_variance_without_changing_counts():
    sample, background = fractional_grid_pair()
    corrected = subtract_measured_background(
        sample, background, mode="manual", manual_scale=2
    )
    np.testing.assert_allclose(corrected.counts, [-3, -5])
    # W=[[.5,.5,0],[0,.5,.5]]: source channel 1 contributes to both outputs.
    expected = np.array([[23, 9], [9, 45]])
    np.testing.assert_allclose(corrected.counts_covariance.toarray(), expected)
    assert corrected.weighted_counts_variance(np.ones(2)) == pytest.approx(86)
    # Diagonal-only propagation gives 68 and would pass a per-bin-only test.
    assert np.sum(corrected.counts_uncertainty**2) == pytest.approx(68)
    assert corrected.counts_in_range(0, 1, use_energy=False)[1] == pytest.approx(
        np.sqrt(86)
    )
    _, uncertainty, _ = estimate_peak_area(
        corrected.counts,
        0,
        np.zeros(2),
        fwhm_channels=1,
        spectrum_data=corrected,
    )
    assert uncertainty == pytest.approx(np.sqrt(86))


def test_sideband_cross_terms_are_included_with_their_negative_sign():
    # Shared-mode covariance is cancelled by a continuum-subtracted count sum.
    covariance = np.eye(7) * 4 + np.ones((7, 7)) * 9
    spectrum = GammaSpectrum(
        counts=[10, 10, 20, 40, 20, 10, 10], counts_covariance=covariance
    )
    net, uncertainty, gross, background, bounds = estimate_peak_area_local_background(
        spectrum.counts,
        3,
        1,
        roi_width_fwhm=2,
        spectrum_data=spectrum,
    )
    assert (net, gross, background, bounds) == (50, 80, 30, (2, 4))
    # Net weights [0,-1.5,1,1,1,-1.5,0], sum=0; variance=4*(3+2*2.25)=30.
    assert uncertainty**2 == pytest.approx(30)
    # Dropping covariance, or summing ROI and sideband variances separately,
    # would incorrectly retain the shared background mode.
    assert uncertainty**2 != pytest.approx(13 * 7.5)


def test_covariance_serialization_and_missing_background_preserve_off_diagonal():
    sample, background = fractional_grid_pair()
    corrected = subtract_measured_background(
        sample, background, mode="manual", manual_scale=2
    )
    restored = GammaSpectrum.from_dict(json.loads(json.dumps(corrected.to_dict())))
    clone = subtract_measured_background(restored, None, warn_missing=False)
    np.testing.assert_allclose(clone.counts_covariance.toarray(), [[23, 9], [9, 45]])
    assert clone.weighted_counts_variance(np.ones(2)) == pytest.approx(86)
    legacy = corrected.to_dict()
    legacy.pop("counts_covariance")
    assert GammaSpectrum.from_dict(legacy).weighted_counts_variance(
        np.ones(2)
    ) == pytest.approx(68)


@pytest.mark.parametrize("operation", [add_spectra, subtract_spectra])
def test_arithmetic_retains_covariance_and_pads_shorter_spectrum(operation):
    left = GammaSpectrum(counts=[5, 6], counts_covariance=[[4, 2], [2, 9]])
    right = GammaSpectrum(counts=[1])
    result = operation(left, right)
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[5, 2], [2, 9]])
    assert result.weighted_counts_variance(np.ones(2)) == pytest.approx(18)


def test_smoothing_propagates_overlapping_window_covariance():
    sample = GammaSpectrum(counts=np.ones(4) * 9)
    result = moving_average(sample, width=1)
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[3, 2], [2, 3]])
    assert result.weighted_counts_variance(np.ones(2)) == pytest.approx(10)


@pytest.mark.parametrize(
    "covariance",
    [
        [[1]],
        [[1, 2], [0, 1]],
        [[-1, 0], [0, 1]],
        [[np.nan, 0], [0, 1]],
    ],
)
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="counts_covariance"):
        GammaSpectrum(counts=[1, 2], counts_covariance=covariance)


def test_inconsistent_covariance_diagonal_is_rejected():
    with pytest.raises(ValueError, match="diagonal must match"):
        GammaSpectrum(
            counts=[1, 2], counts_uncertainty=[1, 1], counts_covariance=np.eye(2) * 2
        )


def test_negative_weighted_variance_is_not_reported_as_zero():
    spectrum = GammaSpectrum(counts=[1, 1], counts_covariance=[[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="negative variance"):
        spectrum.weighted_counts_variance(np.array([1, -1]))


def test_covariance_clone_does_not_alias_source_matrix():
    from scipy.sparse import csr_matrix

    covariance = csr_matrix([[4.0, 2.0], [2.0, 9.0]])
    sample = GammaSpectrum(counts=[5, 6], counts_covariance=covariance)
    clone = subtract_measured_background(sample, None, warn_missing=False)
    clone.counts_covariance.data[0] = 100
    np.testing.assert_allclose(sample.counts_covariance.toarray(), [[4, 2], [2, 9]])
    sample.counts_covariance.data[0] = 200
    np.testing.assert_allclose(covariance.toarray(), [[4, 2], [2, 9]])


@pytest.mark.parametrize("uncertainty", [[-1, 1], [np.nan, 1], [np.inf, 1]])
def test_invalid_count_uncertainty_is_rejected(uncertainty):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GammaSpectrum(counts=[5, 6], counts_uncertainty=uncertainty)


def test_covariance_cannot_be_taken_from_a_different_count_array():
    data = GammaSpectrum(counts=[1, 2])
    with pytest.raises(ValueError, match="counts must match"):
        estimate_peak_area([2, 3], 0, np.zeros(2), spectrum_data=data)
    with pytest.raises(ValueError, match="counts must match"):
        estimate_peak_area_local_background([2, 3], 0, 1, spectrum_data=data)


def test_source_bound_south_joint_poisson_pilot_preserves_failures_and_controls(tmp_path):
    import importlib.util
    from pathlib import Path

    pilot_path = (
        Path(__file__).resolve().parents[1]
        / "examples/RAFM_irradiation/joint_poisson_south_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("joint_poisson_south_pilot", pilot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module.run_pilot()
    assert payload["vendor_targets_loaded"] is False
    assert payload["physical_qualification_status"] == "NOT_QUALIFIED"
    assert payload["source_bound_inputs_unchanged"]
    assert payload["south_background"]["source_sha256"] == module.SOUTH_SHA256
    assert payload["south_background"]["temporal_applicability"] == "UNRESOLVED"
    assert payload["calibration_covariance"].startswith("UNAVAILABLE")
    assert payload["current_integrated_iec_control"]["activity_bq"] == pytest.approx(
        134.32510950634438
    )

    expected = {
        1173.228: (7367.3, 7407.1),
        1332.492: (6872.7, 7513.1),
    }
    for row in payload["rows"]:
        joint = row["joint_fits"]
        south_linear = joint["south_native_linear"]
        south_step = joint["south_native_step"]
        ambient_off = joint["ambient_off_linear"]
        free = joint["south_native_free_normalization"]
        assert south_linear["success"] and south_step["success"] and ambient_off["success"]
        assert south_linear["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit"
        assert south_step["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit"
        assert ambient_off["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit"
        assert not free["success"] and free["status"] == "unidentifiable"
        assert free["identifiability_ratio"] < 1e-6
        assert row["fixed_roi_iec_covell_control"]["south_native"]["roi"] == row[
            "fixed_roi_sample_channels_inclusive"
        ]
        target_joint, target_iec = expected[row["energy_keV"]]
        assert south_linear["area_full_response_counts"] == pytest.approx(
            target_joint, abs=1.0
        )
        assert row["fixed_roi_iec_covell_control"]["south_native"][
            "net_counts"
        ] == pytest.approx(target_iec, abs=1.0)
        assert (
            south_linear["sample_residual_diagnostics"]["n_abs_deviance_residual_gt3"]
            >= 1
        )

    synthetic = {row["case"]: row for row in payload["synthetic_challenges"]}
    assert synthetic["well_specified_rounded_asimov"]["success"]
    assert (
        synthetic["well_specified_rounded_asimov"]["adequacy_flag"]
        != "strong_lack_of_fit"
    )
    assert synthetic["unmodelled_shifted_shoulder"]["success"]
    assert (
        synthetic["unmodelled_shifted_shoulder"]["adequacy_flag"]
        == "strong_lack_of_fit"
    )
    assert not synthetic["free_normalization_tradeoff"]["success"]
    assert synthetic["free_normalization_tradeoff"]["status"] == "unidentifiable"

    output = tmp_path / "south_joint_poisson"
    module.write_outputs(payload, output)
    assert {path.name for path in output.iterdir()} == {
        "PROVENANCE.json",
        "residuals.csv",
        "residuals.svg",
        "south_joint_poisson_pilot.json",
        "summary.csv",
    }
    saved = json.loads((output / "south_joint_poisson_pilot.json").read_text())
    assert saved["vendor_targets_loaded"] is False
