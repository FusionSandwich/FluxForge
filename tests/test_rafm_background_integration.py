from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import (
    _snip_background_from_signed_counts,
    GammaLine,
    analyze_raw_spectrum,
    analyze_raw_spectrum_targeted,
    estimate_peak_area_local_background,
)
from fluxforge.analysis.peak_finders import snip_background
from fluxforge.analysis.spectrum_math import (
    nonnegative_counts_for_algorithm,
    subtract_measured_background,
)
from fluxforge.data.rafm_profile import list_rafm_profiles, load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.genie import read_genie_spectrum


REPO_ROOT = Path(__file__).resolve().parents[1]
RAFM_ROOT = REPO_ROOT / "examples" / "RAFM_irradiation"
BACKGROUND_ASC = RAFM_ROOT / "background.ASC"
SAMPLE_ASC = RAFM_ROOT / "raw_gamma_spec" / "flux_wires" / "Co-Cd-RAFM-1_25cm.ASC"


def test_rafm_profile_resolves_committed_shared_background():
    profile_background = load_rafm_profile("rafm_25cm").resolve_background_path()
    fixture_background = REPO_ROOT / "tests/data/flux_wires/raw/background.ASC"
    assert profile_background is not None
    assert profile_background.samefile(BACKGROUND_ASC)
    assert fixture_background.is_file()
    assert fixture_background.read_bytes() == profile_background.read_bytes()


def test_signed_snip_offsets_working_copy_and_preserves_signed_input():
    signed = np.array([-3.0, 1.0, 4.0, 20.0, 4.0, -1.0, 2.0])
    original = signed.copy()
    background, working, offset = _snip_background_from_signed_counts(
        signed, n_iterations=2
    )

    assert offset == 3.0
    assert np.array_equal(working, original + offset)
    assert np.array_equal(signed, original)
    assert np.all(working >= 0.0)
    assert np.all(background >= 0.0)
    assert background == pytest.approx(
        np.maximum(snip_background(original + offset, n_iterations=2) - offset, 0.0)
    )


@pytest.mark.skipif(
    not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(),
    reason="RAFM example data not present",
)
def test_rafm_shared_background_subtraction_live_mode():
    background = read_genie_spectrum(BACKGROUND_ASC)
    sample = read_genie_spectrum(SAMPLE_ASC)

    corrected = subtract_measured_background(sample, background, mode="live")
    expected_scale = sample.live_time / background.live_time

    assert corrected.counts.shape == sample.counts.shape
    assert corrected.metadata["background_subtraction"][
        "scale_factor"
    ] == pytest.approx(expected_scale)
    assert corrected.counts_uncertainty.shape == sample.counts.shape


@pytest.mark.skipif(
    not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(),
    reason="RAFM example data not present",
)
def test_rafm_hybrid_keeps_signed_counts_until_an_algorithm_requires_clipping():
    background = read_genie_spectrum(BACKGROUND_ASC)
    sample = read_genie_spectrum(SAMPLE_ASC)
    corrected = subtract_measured_background(sample, background, mode="live")
    signed = corrected.counts.copy()
    assert (signed < 0.0).any()
    assert corrected.metadata["background_subtraction"]["negative_bins"] == int(
        (signed < 0.0).sum()
    )

    with pytest.warns(RuntimeWarning, match="requires non-negative counts"):
        algorithm_counts = nonnegative_counts_for_algorithm(corrected, "test algorithm")
    assert (algorithm_counts >= 0.0).all()
    assert (corrected.counts == signed).all()
    assert (corrected.counts_uncertainty >= 0.0).all()


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


@pytest.mark.skipif(
    not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(),
    reason="RAFM example data not present",
)
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
    assert not any(
        "requires non-negative counts" in str(item.message) for item in caught
    )


@pytest.mark.skipif(
    not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(),
    reason="RAFM example data not present",
)
def test_flux_wire_roi_uncertainty_responds_to_background_uncertainty():
    sample = read_genie_spectrum(SAMPLE_ASC)
    background = read_genie_spectrum(BACKGROUND_ASC)
    inflated_background = read_genie_spectrum(BACKGROUND_ASC)
    inflated_background.counts_uncertainty *= 10.0

    baseline = analyze_raw_spectrum(
        sample,
        background_spectrum=background,
        sample_name="Co-Cd-RAFM-1_25cm",
        peak_threshold=2.0,
    )
    inflated = analyze_raw_spectrum(
        sample,
        background_spectrum=inflated_background,
        sample_name="Co-Cd-RAFM-1_25cm",
        peak_threshold=2.0,
    )
    by_channel = {peak.channel: peak for peak in inflated}
    shared = [
        (peak, by_channel[peak.channel])
        for peak in baseline
        if peak.channel in by_channel
    ]
    assert shared
    assert any(
        changed.net_counts_unc > original.net_counts_unc for original, changed in shared
    )
    for original, changed in shared:
        assert changed.net_counts == pytest.approx(original.net_counts)
        assert changed.net_counts_unc >= original.net_counts_unc


@pytest.mark.parametrize("counting_method", ["qg", "covell", "gilmore", "iec_tiered"])
def test_real_rafm_targeted_uncertainty_keeps_propagated_roi_floor(counting_method):
    data = read_raw_asc(SAMPLE_ASC, profile_name="rafm_25cm")
    background = read_raw_asc(BACKGROUND_ASC, profile_name="rafm_25cm").spectrum
    background.counts_uncertainty *= 10.0
    corrected = subtract_measured_background(data.spectrum, background, mode="live")
    lines = [
        GammaLine(energy_keV=1173.23, intensity=0.9985, isotope="Co60"),
        GammaLine(energy_keV=1332.49, intensity=0.9998, isotope="Co60"),
    ]
    peaks = analyze_raw_spectrum_targeted(
        data,
        lines,
        background_spectrum=background,
        profile_name="rafm_25cm",
        counting_method=counting_method,
        peak_threshold=0.0,
    )

    assert len(peaks) == 2
    for peak in peaks:
        slope = (
            data.energy_calibration[1] + 2.0 * data.energy_calibration[2] * peak.channel
        )
        _, roi_unc, _, _, _ = estimate_peak_area_local_background(
            corrected.counts,
            peak.channel,
            peak.fwhm / slope,
            spectrum_uncertainty=corrected.counts_uncertainty,
        )
        assert peak.net_counts_unc >= roi_unc - 1.0e-9
        assert peak.comparison_net_counts_unc < roi_unc


@pytest.mark.skipif(
    not BACKGROUND_ASC.exists() or not SAMPLE_ASC.exists(),
    reason="RAFM example data not present",
)
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
    assert not any(
        "no background spectrum was provided" in str(item.message) for item in caught
    )
    assert not any(
        "requires non-negative counts" in str(item.message) for item in caught
    )
