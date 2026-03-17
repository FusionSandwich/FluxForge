from __future__ import annotations

from pathlib import Path
import warnings

import pytest

from fluxforge.analysis.flux_wire_analysis import analyze_raw_spectrum
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.data.rafm_profile import list_rafm_profiles, load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.genie import read_genie_spectrum


REPO_ROOT = Path(__file__).resolve().parents[1]
RAFM_ROOT = REPO_ROOT / "examples" / "RAFM_irradiation"
BACKGROUND_ASC = RAFM_ROOT / "background.ASC"
SAMPLE_ASC = RAFM_ROOT / "raw_gamma_spec" / "flux_wires" / "Co-Cd-RAFM-1_25cm.ASC"


@pytest.mark.skipif(not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(), reason="RAFM example data not present")
def test_rafm_shared_background_subtraction_live_mode():
    background = read_genie_spectrum(BACKGROUND_ASC)
    sample = read_genie_spectrum(SAMPLE_ASC)

    corrected = subtract_measured_background(sample, background, mode="live")
    expected_scale = sample.live_time / background.live_time

    assert corrected.counts.shape == sample.counts.shape
    assert corrected.metadata["background_subtraction"]["scale_factor"] == pytest.approx(expected_scale)
    assert corrected.counts_uncertainty.shape == sample.counts.shape


def test_astm_profile_aliases_are_listed_and_match_rafm_defaults():
    profiles = list_rafm_profiles()
    assert "rafm_25cm" in profiles
    assert "astm_inl_dosimetry" in profiles
    assert "us_astm_reactor_dosimetry" in profiles

    rafm = load_rafm_profile("rafm_25cm")
    astm = load_rafm_profile("astm_inl_dosimetry")
    us_astm = load_rafm_profile("us_astm_reactor_dosimetry")

    assert astm.background_relative_path == rafm.background_relative_path
    assert astm.energy_calibration == pytest.approx(rafm.energy_calibration)
    assert astm.resolution == pytest.approx(rafm.resolution)
    assert astm.efficiency == pytest.approx(rafm.efficiency)
    assert us_astm.energy_calibration == pytest.approx(rafm.energy_calibration)
    assert us_astm.resolution == pytest.approx(rafm.resolution)
    assert us_astm.efficiency == pytest.approx(rafm.efficiency)


@pytest.mark.skipif(not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(), reason="RAFM example data not present")
def test_rafm_analysis_runs_with_shared_background():
    background = read_genie_spectrum(BACKGROUND_ASC)
    sample = read_genie_spectrum(SAMPLE_ASC)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        peaks = analyze_raw_spectrum(
            sample,
            peak_threshold=5.0,
            sample_name="Co-Cd-RAFM-1_25cm",
            background_spectrum=background,
            background_scale_mode="live",
            background_subtract=True,
        )
    assert len(peaks) > 0
    assert not any("requires non-negative counts" in str(item.message) for item in caught)


@pytest.mark.skipif(not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(), reason="RAFM example data not present")
@pytest.mark.parametrize("profile_name", ["rafm_25cm", "astm_inl_dosimetry"])
def test_rafm_profile_supplies_background_and_efficiency_defaults(profile_name: str):
    data = read_raw_asc(SAMPLE_ASC, profile_name=profile_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        peaks = analyze_raw_spectrum(
            data.spectrum,
            efficiency=data.efficiency,
            peak_threshold=5.0,
            sample_name="Co-Cd-RAFM-1_25cm",
            background_subtract=True,
            profile_name=profile_name,
        )

    assert data.energy_calibration[0] == pytest.approx(0.541, rel=0, abs=1e-3)
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(-20.26)
    assert len(peaks) > 0
    assert not any("no background spectrum was provided" in str(item.message) for item in caught)
    assert not any("requires non-negative counts" in str(item.message) for item in caught)
