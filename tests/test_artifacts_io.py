import pytest

from fluxforge.core.schemas import validate_artifact
from fluxforge.io.artifacts import (
    read_detector_characterization,
    read_facility_characterization,
    read_k0_aggregation_bundle,
    read_k0_analysis_bundle,
    read_k0_qaqc_bundle,
    read_line_activities,
    read_peak_observation_bundle,
    read_peak_report,
    read_reaction_rates,
    read_report_bundle,
    read_response_bundle,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_detector_characterization,
    write_facility_characterization,
    write_k0_aggregation_bundle,
    write_k0_analysis_bundle,
    write_k0_qaqc_bundle,
    write_line_activities,
    write_peak_observation_bundle,
    write_peak_report,
    write_reaction_rates,
    write_report_bundle,
    write_response_bundle,
    write_spectrum_file,
    write_unfold_result,
    write_validation_bundle,
)


def _require_numpy():
    np = pytest.importorskip("numpy")
    from fluxforge.io.spe import GammaSpectrum

    return np, GammaSpectrum


def test_spectrum_file_roundtrip(tmp_path):
    np, GammaSpectrum = _require_numpy()
    spectrum = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        energies=np.array([0.0, 1.0, 2.0]),
        live_time=12.0,
        real_time=15.0,
        spectrum_id="demo",
    )
    output = tmp_path / "spectrum.json"
    write_spectrum_file(output, spectrum)
    payload = read_spectrum_file(output)
    assert validate_artifact(payload) == []
    assert payload["spectrum"]["counts"] == [10.0, 20.0, 30.0]
    assert payload["spectrum"]["live_time"] == 12.0
    assert payload["provenance"]["definitions"]["counts"] == "raw counts per channel"
    assert payload["provenance"]["units"]["energies"] == "keV"


def test_peak_report_roundtrip(tmp_path):
    _require_numpy()
    peaks = [
        {
            "channel": 12,
            "energy_keV": 511.0,
            "amplitude": 200.0,
            "raw_counts": 180.0,
            "sigma_keV": 1.2,
            "area": 500.0,
            "region": "mid",
            "is_report": False,
            "report_isotope": "",
            "report_file": "",
        }
    ]
    output = tmp_path / "peaks.json"
    write_peak_report(output, spectrum_id="demo", live_time_s=10.0, peaks=peaks)
    payload = read_peak_report(output)
    assert validate_artifact(payload) == []
    assert payload["peaks"][0]["energy_keV"] == 511.0
    assert payload["provenance"]["definitions"]["area"] == "net peak area"
    assert payload["provenance"]["units"]["live_time_s"] == "s"


def test_line_activities_roundtrip(tmp_path):
    _require_numpy()
    lines = [
        {
            "energy_keV": 511.0,
            "isotope": "Na-22",
            "reaction_id": "Na-22",
            "net_counts": 500.0,
            "activity_Bq": 5.0,
            "activity_unc_Bq": 0.5,
            "efficiency": 0.2,
            "emission_probability": 0.9,
            "half_life_s": 10.0,
            "radioisotope_specific_activity_Bq_g": 1.2e21,
            "atoms": 72.0,
            "radioactive_mass_g": 2.6e-21,
            "sample_mass_g": 0.25,
            "specific_activity_Bq_g": 20.0,
        }
    ]
    output = tmp_path / "activities.json"
    write_line_activities(output, spectrum_id="demo", lines=lines)
    payload = read_line_activities(output)
    assert validate_artifact(payload) == []
    assert payload["lines"][0]["activity_Bq"] == 5.0
    assert payload["lines"][0]["atoms"] == 72.0
    assert payload["lines"][0]["radioisotope_specific_activity_Bq_g"] == 1.2e21
    assert payload["lines"][0]["specific_activity_Bq_g"] == 20.0
    assert payload["provenance"]["definitions"]["activity_Bq"] == "activity at count time unless corrected"
    assert payload["provenance"]["units"]["activity_Bq"] == "Bq"


def test_reaction_rates_roundtrip(tmp_path):
    _require_numpy()
    rates = [
        {"reaction_id": "Fe-59", "rate": 1.5, "uncertainty": 0.1, "half_life_s": 12.0}
    ]
    segments = [{"duration_s": 5.0, "relative_power": 1.0}]
    output = tmp_path / "rates.json"
    write_reaction_rates(output, rates=rates, segments=segments)
    payload = read_reaction_rates(output)
    assert validate_artifact(payload) == []
    assert payload["rates"][0]["rate"] == 1.5
    assert payload["provenance"]["definitions"]["rate"] == "reaction rate at EOI per reaction"
    assert payload["provenance"]["units"]["rate"] == "reactions/s"


def test_response_bundle_roundtrip(tmp_path):
    _require_numpy()
    output = tmp_path / "response.json"
    write_response_bundle(
        output,
        matrix=[[1.0, 0.5], [0.2, 0.1]],
        reactions=["rx1", "rx2"],
        boundaries_eV=[0.0, 1.0, 2.0],
    )
    payload = read_response_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["matrix"][0][0] == 1.0
    assert payload["provenance"]["definitions"]["matrix"] == "response matrix with rows as reactions and columns as energy groups"
    assert payload["provenance"]["units"]["boundaries_eV"] == "eV"


def test_unfold_result_roundtrip(tmp_path):
    _require_numpy()
    output = tmp_path / "unfold.json"
    write_unfold_result(
        output,
        boundaries_eV=[0.0, 1.0],
        reactions=["rx1"],
        flux=[1.2],
        covariance=[[0.04]],
        chi2=1.1,
        method="gls",
    )
    payload = read_unfold_result(output)
    assert validate_artifact(payload) == []
    assert payload["flux"] == [1.2]
    assert payload["provenance"]["definitions"]["flux"] == "group-integrated flux per energy bin"
    assert payload["provenance"]["units"]["flux"] == "a.u."


def test_validation_bundle_roundtrip(tmp_path):
    _require_numpy()
    output = tmp_path / "validation.json"
    write_validation_bundle(
        output,
        metrics={"mae": 0.1},
        truth_flux=[1.0, 2.0],
        predicted_flux=[1.1, 1.9],
        residuals=[0.1, -0.1],
    )
    payload = read_validation_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["metrics"]["mae"] == 0.1
    assert payload["provenance"]["definitions"]["residuals"] == "predicted_flux - truth_flux"
    assert payload["provenance"]["units"]["residuals"] == "a.u."


def test_report_bundle_roundtrip(tmp_path):
    _require_numpy()
    output = tmp_path / "report.json"
    write_report_bundle(
        output,
        summary={"peak_count": 3},
        inputs={"peaks_file": "peaks.json"},
        figures={"directory": "report_figures", "items": {"spectrum_preview": {"path": "spectrum_preview.png"}}},
        tables={"directory": "report_tables", "items": {"activity_summary": {"path": "activity_summary.csv"}}},
        text_report={"path": "report.txt", "format": "text/plain"},
    )
    payload = read_report_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["peak_count"] == 3
    assert payload["figures"]["directory"] == "report_figures"
    assert payload["figures"]["items"]["spectrum_preview"]["path"] == "spectrum_preview.png"
    assert payload["tables"]["directory"] == "report_tables"
    assert payload["tables"]["items"]["activity_summary"]["path"] == "activity_summary.csv"
    assert payload["text_report"]["path"] == "report.txt"
    assert payload["provenance"]["definitions"]["summary"] == "aggregate summary across artifacts"
    assert payload["provenance"]["units"]["summary"] == "mixed"


def test_peak_observation_bundle_roundtrip(tmp_path):
    output = tmp_path / "observations.json"
    write_peak_observation_bundle(
        output,
        spectrum_id="demo",
        detector_id="hpge-01",
        geometry_id="pos-200mm",
        observations=[
            {
                "peak_id": "p1",
                "source_spectrum_id": "demo",
                "detector_id": "hpge-01",
                "geometry_id": "pos-200mm",
                "line_energy_keV": 411.8,
                "line_id": "Au-198@411.8keV",
                "assigned_radionuclide": "Au-198",
                "net_peak_area": 1500.0,
                "area_uncertainty": 38.7,
                "live_time_s": 100.0,
                "real_time_s": 101.0,
                "irradiation_time_s": 300.0,
                "decay_time_s": 3600.0,
                "counting_time_s": 100.0,
                "eligibility_class": "direct_k0_eligible",
                "eligibility_accepted": True,
            }
        ],
        summary={"observation_count": 1, "accepted_count": 1, "rejected_count": 0},
        capability_flags={"supports_thermal_inaa": "validated"},
    )
    payload = read_peak_observation_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["accepted_count"] == 1
    assert payload["observations"][0]["assigned_radionuclide"] == "Au-198"


def test_detector_characterization_roundtrip(tmp_path):
    output = tmp_path / "detector.json"
    write_detector_characterization(
        output,
        detector_id="hpge-01",
        reference_position_mm=200.0,
        characterized_positions_mm=[100.0, 200.0],
        calibration_points=[{"position_mm": 200.0, "reference_energy_keV": 661.7, "efficiency": 0.01}],
        efficiency_model={"model_type": "log_poly", "coefficients": [-5.0, -0.8, 0.02], "energy_range_keV": [50.0, 2000.0]},
        geometry_conversions={"items": {"100": {"ratio_mean": 3.0, "ratio_std": 0.1}}},
        peak_to_total_model={"method": "constant_ratio", "value": 0.32},
        coincidence_model={"mode": "not_applied", "applied": False},
        capability_flags={"supports_thermal_inaa": "validated"},
    )
    payload = read_detector_characterization(output)
    assert validate_artifact(payload) == []
    assert payload["detector_id"] == "hpge-01"
    assert payload["reference_position_mm"] == 200.0


def test_facility_characterization_roundtrip(tmp_path):
    output = tmp_path / "facility.json"
    write_facility_characterization(
        output,
        facility_id="whale-position-a",
        method="bare_triple_monitor",
        monitor_definitions=[{"monitor_id": "Au-197", "activity": 1000.0}],
        irradiation={"irradiation_time_s": 600.0},
        flux_parameters={"f": 25.0, "alpha": 0.01, "phi_thermal": 1.0e12, "phi_epithermal": 4.0e10},
        temperature={"value_K": 300.0, "method": "assumed_cooling_water"},
        capability_flags={"supports_thermal_inaa": "validated"},
    )
    payload = read_facility_characterization(output)
    assert validate_artifact(payload) == []
    assert payload["flux_parameters"]["f"] == 25.0


def test_k0_analysis_bundle_roundtrip(tmp_path):
    output = tmp_path / "k0_analysis.json"
    write_k0_analysis_bundle(
        output,
        summary={"element_count": 1},
        line_results=[{"element": "Co", "concentration_ug_g": 12.0, "concentration_unc_ug_g": 0.6}],
        element_results=[{"element": "Co", "concentration_ug_g": 12.0, "concentration_unc_ug_g": 0.6}],
        rejected_observations=[{"peak_id": "bad_line"}],
        applied_corrections=["saturation_decay_counting"],
        recognized_but_not_applied=["westcott_gT_not_applied:Lu-176"],
        capability_flags={"supports_thermal_inaa": "validated"},
        libraries={"standard_k0_library": {"version": "demo"}},
    )
    payload = read_k0_analysis_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["element_count"] == 1
    assert payload["element_results"][0]["element"] == "Co"


def test_k0_aggregation_bundle_roundtrip(tmp_path):
    output = tmp_path / "k0_aggregation.json"
    write_k0_aggregation_bundle(
        output,
        summary={"aggregated_result_count": 1},
        aggregated_results=[{"element": "Co", "concentration_ug_g": 11.0, "concentration_unc_ug_g": 0.8}],
        irradiation_summaries=[{"element": "Co", "irradiation_id": "irr-1", "concentration_ug_g": 11.0, "concentration_unc_ug_g": 0.8}],
    )
    payload = read_k0_aggregation_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["aggregated_result_count"] == 1


def test_k0_qaqc_bundle_roundtrip(tmp_path):
    output = tmp_path / "k0_qaqc.json"
    write_k0_qaqc_bundle(
        output,
        summary={"pass_count": 1, "fail_count": 0},
        records=[{"role": "blank", "sample_id": "blank-1", "status": "pass"}],
    )
    payload = read_k0_qaqc_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["pass_count"] == 1
