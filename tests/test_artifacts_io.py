import pytest

from fluxforge.core.schemas import validate_artifact
from fluxforge.io.artifacts import (
    read_line_activities,
    read_peak_report,
    read_reaction_rates,
    read_report_bundle,
    read_response_bundle,
    read_spectrum_file,
    read_unfold_result,
    read_validation_bundle,
    write_line_activities,
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
        }
    ]
    output = tmp_path / "activities.json"
    write_line_activities(output, spectrum_id="demo", lines=lines)
    payload = read_line_activities(output)
    assert validate_artifact(payload) == []
    assert payload["lines"][0]["activity_Bq"] == 5.0
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
    write_report_bundle(output, summary={"peak_count": 3}, inputs={"peaks_file": "peaks.json"})
    payload = read_report_bundle(output)
    assert validate_artifact(payload) == []
    assert payload["summary"]["peak_count"] == 3
    assert payload["provenance"]["definitions"]["summary"] == "aggregate summary across artifacts"
    assert payload["provenance"]["units"]["summary"] == "mixed"
