"""Source identity, fixed-data controls and independent adequacy challenges."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import xlogy


@pytest.fixture(scope="module")
def pilot():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/RAFM_irradiation/south_joint_poisson_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("south_pilot_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload(pilot):
    return pilot.run_pilot()


def test_native_south_source_and_count_conservation(payload):
    assert payload["originals_byte_identical_after_run"]
    sample, ambient = payload["sample_identity"], payload["ambient_identity"]
    assert sample["measurement_id"] == "Co-Cd-RAFM-1"
    assert sample["detector"] == ambient["detector"] == "South"
    assert (
        ambient["source_sha256"]
        == "96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344"
    )
    assert sample["live_time_s"] == 129600
    assert sample["real_time_s"] == 129691
    assert ambient["live_time_s"] == 14400
    assert ambient["real_time_s"] == 14409.5
    assert ambient["used_energy_polynomial_keV"] == [
        -1.6994324922561646,
        0.49959567189216614,
        9.02407251146542e-08,
    ]
    assert ambient["temporal_applicability"] == "UNRESOLVED"
    assert (
        sample["timeline"]["acquisition"]["clock_relation_status"]
        == "unresolved_clock_conflict"
    )
    conservation = payload["conservation"]
    assert conservation["native_total"] == 543427
    assert abs(conservation["residual"]) < 1e-8
    assert conservation["comparison_covariance_propagated"]
    assert not conservation["joint_observations_rebinned"]


def test_fixed_roi_and_original_integer_counts_across_every_candidate(payload):
    for energy in (1173.228, 1332.492):
        rows = [r for r in payload["rows"] if r["energy_keV"] == energy]
        original = rows[0]
        for row in rows:
            assert (
                row["roi_sample_channels_inclusive"]
                == original["roi_sample_channels_inclusive"]
            )
            np.testing.assert_array_equal(
                row["sample_original_counts"], original["sample_original_counts"]
            )
            counts = np.asarray(row["sample_original_counts"])
            assert np.all(counts >= 0) and np.all(counts % 1 == 0)
            if row["scenario"] == "ambient_off":
                assert row["ambient_original_counts"] is None
                assert row["joint"]["provenance"]["background_acquisition_id"] is None
            else:
                np.testing.assert_array_equal(
                    row["ambient_original_counts"], original["ambient_original_counts"]
                )
                assert np.all(np.asarray(row["ambient_original_counts"]) % 1 == 0)
                assert (
                    row["joint"]["provenance"]["acquisition_chronology"]
                    == "background_later"
                )
            if "iec" in row:
                assert (
                    row["iec"]["roi_sample_channels_inclusive"]
                    == row["roi_sample_channels_inclusive"]
                )
        assert not np.array_equal(
            original["sample_native_edges_keV"], original["ambient_native_edges_keV"]
        )


def test_nominal_lack_of_fit_and_free_normalization_failure_are_retained(payload):
    nominal = [
        r
        for r in payload["rows"]
        if r["response"] == "nominal" and r["scenario"] != "south_free_normalization"
    ]
    assert len(nominal) == 8
    assert all(r["joint"]["success"] for r in nominal)
    assert all(
        r["joint"]["model_diagnostics"]["adequacy_flag"] == "strong_lack_of_fit"
        for r in nominal
    )
    failed = [
        r["joint"]
        for r in payload["rows"]
        if r["scenario"] == "south_free_normalization"
    ]
    assert len(failed) == 2
    for fit in failed:
        assert not fit["success"] and fit["status"] == "unidentifiable"
        assert fit["identifiability_ratio"] < 1e-6
        assert fit["count_average_activity_bq"] is None
        assert fit["profile"] is None
        assert len(fit["sample_expected"]) > 0  # candidate evidence survives
    assert payload["calibration_covariance"].startswith("UNKNOWN")
    assert not payload["scientific_admission"]
    assert not payload["exact_vendor_parity"]
    assert not payload["physical_inversion_qualified"]


def test_poisson_residuals_match_independent_log_likelihood_oracle(pilot):
    counts, means = np.array([0.0, 2.0, 7.0, 3.0]), np.array([2.0, 2.0, 4.0, 5.0])
    result = pilot.poisson_diagnostics(counts, means, np.arange(5.0), 2.0)
    oracle = 2 * (xlogy(counts, counts / means) - counts + means)
    np.testing.assert_allclose(np.square(result["signed_deviance"]), oracle)
    np.testing.assert_allclose(result["pearson"], (counts - means) / np.sqrt(means))
    assert sum(r["deviance"] for r in result["segments"].values()) == pytest.approx(
        oracle.sum()
    )
    zeros = pilot.poisson_diagnostics(
        np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.arange(3.0), 1.0
    )
    assert zeros["status"] == "INFINITE_DEVIANCE"
    assert zeros["nonfinite_bins"] == [1]
    assert zeros["signed_deviance"] == [0.0, None]
    assert zeros["pearson"] == [0.0, None]


def test_every_saved_deviance_is_accounted_for_by_original_observations(payload):
    for row in payload["rows"]:
        fit = row["joint"]
        parts = fit["sample_poisson_residuals"]["signed_deviance"]
        if row["scenario"] != "ambient_off":
            parts = parts + fit["ambient_poisson_residuals"]["signed_deviance"]
        assert np.square(parts).sum() == pytest.approx(fit["deviance"], rel=1e-11)


def test_independent_synthetic_truth_and_failure_challenges(payload):
    cases = {r["name"]: r for r in payload["synthetic_challenges"]}
    for name in ("zero_signal", "weak_signal", "strong_signal"):
        case, fit = cases[name], cases[name]["joint"]
        assert fit["success"], fit["message"]
        assert fit["candidate_area"] == pytest.approx(case["truth_area"], abs=6)
        assert fit["profile"]["lower"] <= case["truth_area"] <= fit["profile"]["upper"]
        assert fit["model_diagnostics"]["adequacy_flag"] == "not_qualified"
    for name in (
        "shifted_response_misspecification",
        "broad_response_misspecification",
    ):
        assert cases[name]["joint"]["success"]
        assert (
            cases[name]["joint"]["model_diagnostics"]["adequacy_flag"]
            == "strong_lack_of_fit"
        )
    assert cases["normalization_unidentifiable"]["joint"]["status"] == "unidentifiable"
    assert cases["optimizer_budget_failure"]["joint"]["status"] == "optimizer_failed"


def test_fixed_iec_retains_roi_sideband_covariance(pilot):
    from scipy.sparse import csr_matrix
    from fluxforge.io.spe import GammaSpectrum

    counts = np.array([10.0, 20.0, 40.0, 20.0, 10.0])
    covariance = np.diag(counts)
    covariance[0, 1] = covariance[1, 0] = 2.0
    spectrum = GammaSpectrum(
        counts=counts, channels=np.arange(5), counts_covariance=csr_matrix(covariance)
    )
    config = dict(
        flux_wire_roi_width_fwhm=2.0,
        flux_wire_background_width_channels=1,
        flux_wire_background_gap_fwhm=0.0,
    )
    result = pilot.fixed_iec_control(spectrum, 2, 1.0, config, 1.0)
    weights = np.array([-1.5, 1.0, 1.0, 1.0, -1.5])
    assert result["roi_sample_channels_inclusive"] == [1, 3]
    assert result["net"] == weights @ counts
    assert result["std_full_covariance"] ** 2 == pytest.approx(
        weights @ covariance @ weights
    )


def test_all_response_sensitivities_retained_without_selecting_a_winner(payload):
    expected = {
        "nominal",
        "centroid_minus_half_bin",
        "centroid_plus_half_bin",
        "width_times_0.8",
        "width_times_1.2",
    }
    assert len(payload["rows"]) == 26
    for energy in (1173.228, 1332.492):
        for mode in ("south_native", "ambient_off"):
            rows = [
                r
                for r in payload["rows"]
                if r["energy_keV"] == energy and r["scenario"] == mode
            ]
            assert {r["response"] for r in rows} == expected
            assert len(rows) == 6
            assert all(not r["joint"]["physical_activity_qualified"] for r in rows)


def test_source_manifest_tamper_is_rejected(pilot, monkeypatch):
    driver = pilot.load_driver()
    original = driver.bound_path
    target = driver.MANIFEST_PATH.parent / "originals/ASC/Co-Cd-RAFM-1.ASC"

    class Tampered:
        def read_bytes(self):
            return b"tampered original"

    def replaced(repo, name):
        path = original(repo, name)
        return Tampered() if path == target else path

    monkeypatch.setattr(driver, "bound_path", replaced)
    with pytest.raises(ValueError, match="Source hash/size mismatch"):
        driver.verify_inputs(pilot.ROOT, driver.MANIFEST_PATH)


def test_output_refuses_overwrite(pilot, payload, tmp_path):
    with pytest.raises(FileExistsError):
        pilot.save_receipts(tmp_path, payload)
