from __future__ import annotations

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import (
    GammaLine,
    IdentifiedPeak,
    _window_metrics_local_background,
    combine_peak_activities,
)


def test_window_metrics_linear_background_tracks_linear_continuum() -> None:
    counts = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    net, net_unc, gross, background_sum = _window_metrics_local_background(
        counts,
        2,
        7,
        background_width_channels=2,
        background_model="linear",
    )

    assert gross == pytest.approx(np.sum(counts[2:8]))
    assert background_sum == pytest.approx(np.sum(counts[2:8]))
    assert net == pytest.approx(0.0)
    assert net_unc >= 0.0


def test_window_metrics_linear_background_matches_constant_sum_on_linear_background() -> (
    None
):
    counts = np.array([10.0, 11.0, 12.0, 20.0, 25.0, 24.0, 18.0, 17.0, 18.0, 19.0])
    const_net, _, _, _ = _window_metrics_local_background(
        counts,
        3,
        6,
        background_width_channels=2,
        background_model="constant",
    )
    linear_net, _, _, _ = _window_metrics_local_background(
        counts,
        3,
        6,
        background_width_channels=2,
        background_model="linear",
    )

    assert linear_net == pytest.approx(const_net)


def test_combine_peak_activities_emits_single_vs_all_diagnostics() -> None:
    peaks = [
        IdentifiedPeak(
            channel=100,
            energy_keV=300.1,
            net_counts=1200.0,
            net_counts_unc=40.0,
            gross_counts=1500.0,
            background=15.0,
            fwhm=2.0,
            significance=12.0,
            isotope="Sc48",
            gamma_line=GammaLine(
                energy_keV=300.1,
                intensity=0.9,
                isotope="Sc48",
            ),
            activity_bq=100.0,
            activity_unc_bq=5.0,
        ),
        IdentifiedPeak(
            channel=220,
            energy_keV=450.2,
            net_counts=980.0,
            net_counts_unc=35.0,
            gross_counts=1200.0,
            background=11.0,
            fwhm=2.1,
            significance=11.0,
            isotope="Sc48",
            gamma_line=GammaLine(
                energy_keV=450.2,
                intensity=0.8,
                isotope="Sc48",
            ),
            activity_bq=105.0,
            activity_unc_bq=5.0,
        ),
        IdentifiedPeak(
            channel=350,
            energy_keV=650.3,
            net_counts=400.0,
            net_counts_unc=30.0,
            gross_counts=700.0,
            background=9.0,
            fwhm=2.4,
            significance=6.0,
            isotope="Sc48",
            gamma_line=GammaLine(
                energy_keV=650.3,
                intensity=0.12,
                isotope="Sc48",
            ),
            activity_bq=180.0,
            activity_unc_bq=6.0,
        ),
    ]

    result = combine_peak_activities(peaks)
    row = result["Sc48"]

    assert row["n_peaks"] >= 2
    assert "single_peak_activity_diagnostics" in row
    assert row["single_peak_activity_diagnostics"]
    assert "variance_line_activity_bq2" in row
    assert "max_abs_relative_line_delta" in row
    assert row["max_abs_relative_line_delta"] > 0.0
    assert row["single_peak_outlier_energies"] or row["excluded_peak_energies"]

    first_diag = row["single_peak_activity_diagnostics"][0]
    assert "relative_delta_vs_combined" in first_diag
    assert "leave_one_out_activity_bq" in first_diag
    assert "all_vs_leave_one_out_relative_delta" in first_diag



def test_source_bound_south_joint_poisson_pilot_preserves_failures_and_controls(
    tmp_path,
) -> None:
    import importlib.util
    import json
    from pathlib import Path

    pilot_path = (
        Path(__file__).resolve().parents[1]
        / "examples/RAFM_irradiation/joint_poisson_south_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "joint_poisson_south_pilot", pilot_path
    )
    assert spec is not None and spec.loader is not None
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

        assert south_linear["success"]
        assert south_step["success"]
        assert ambient_off["success"]
        south_flag = south_linear["model_diagnostics"]["adequacy_flag"]
        step_flag = south_step["model_diagnostics"]["adequacy_flag"]
        ambient_flag = ambient_off["model_diagnostics"]["adequacy_flag"]
        assert south_flag == "strong_lack_of_fit"
        assert step_flag == "strong_lack_of_fit"
        assert ambient_flag == "strong_lack_of_fit"
        assert not free["success"]
        assert free["status"] == "unidentifiable"
        assert free["identifiability_ratio"] < 1e-6

        fixed_control = row["fixed_roi_iec_covell_control"]["south_native"]
        assert fixed_control["roi"] == row["fixed_roi_sample_channels_inclusive"]
        target_joint, target_iec = expected[row["energy_keV"]]
        assert south_linear["area_full_response_counts"] == pytest.approx(
            target_joint, abs=1.0
        )
        assert fixed_control["net_counts"] == pytest.approx(target_iec, abs=1.0)
        residual_diagnostics = south_linear["sample_residual_diagnostics"]
        assert residual_diagnostics["n_abs_deviance_residual_gt3"] >= 1

    synthetic = {row["case"]: row for row in payload["synthetic_challenges"]}
    well = synthetic["well_specified_rounded_asimov"]
    shoulder = synthetic["unmodelled_shifted_shoulder"]
    tradeoff = synthetic["free_normalization_tradeoff"]
    assert well["success"]
    assert well["adequacy_flag"] != "strong_lack_of_fit"
    assert shoulder["success"]
    assert shoulder["adequacy_flag"] == "strong_lack_of_fit"
    assert not tradeoff["success"]
    assert tradeoff["status"] == "unidentifiable"

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
