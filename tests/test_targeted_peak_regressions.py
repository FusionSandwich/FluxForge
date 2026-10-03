import numpy as np
import pytest
from pathlib import Path
from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.flux_wire import FluxWireData
from fluxforge.analysis.flux_wire_analysis import (
    GammaLine,
    analyze_raw_spectrum_targeted,
    build_gamma_library,
)


def synthetic_data(sigma_factor=1.0, multiplet=False):
    x = np.arange(512, dtype=float)
    counts = 40.0 + 2000.0 * np.exp(-0.5 * ((x - 200.0) / 1.2) ** 2)
    if multiplet:
        counts += 1000.0 * np.exp(-0.5 * ((x - 205.0) / 1.2) ** 2)
    spectrum = GammaSpectrum(
        counts=counts,
        counts_uncertainty=np.sqrt(counts) * sigma_factor,
        live_time=100.0,
        real_time=100.0,
        calibration={"energy": [0.0, 1.0]},
    )
    return FluxWireData(
        sample_id="synthetic",
        spectrum=spectrum,
        live_time=100.0,
        real_time=100.0,
        energy_calibration=[0.0, 1.0],
        resolution=[2.0, 0.0],
    )


@pytest.mark.parametrize("method", ["covell", "gilmore", "iec_tiered", "qg"])
def test_custom_counting_sigma_is_preserved(method):
    line = GammaLine(200.0, 0.5, "Present")
    low = analyze_raw_spectrum_targeted(
        synthetic_data(1.0), [line], background_subtract=False, counting_method=method
    )[0]
    high = analyze_raw_spectrum_targeted(
        synthetic_data(10.0), [line], background_subtract=False, counting_method=method
    )[0]
    assert high.net_counts == pytest.approx(low.net_counts, rel=1e-5)
    assert high.net_counts_unc == pytest.approx(10.0 * low.net_counts_unc, rel=1e-4)


def test_multiplet_custom_counting_sigma_is_preserved():
    lines = [GammaLine(200.0, 0.5, "Present"), GammaLine(205.0, 0.5, "Present")]
    low = analyze_raw_spectrum_targeted(
        synthetic_data(1.0, True),
        lines,
        background_subtract=False,
        counting_method="iec_tiered",
    )
    high = analyze_raw_spectrum_targeted(
        synthetic_data(10.0, True),
        lines,
        background_subtract=False,
        counting_method="iec_tiered",
    )
    assert len(low) == len(high) == 2
    for one, two in zip(low, high):
        assert two.net_counts_unc == pytest.approx(10.0 * one.net_counts_unc, rel=1e-4)


def test_nearby_strong_line_does_not_satisfy_wrong_expected_identity():
    peaks = analyze_raw_spectrum_targeted(
        synthetic_data(),
        [GammaLine(205.0, 0.5, "Absent")],
        peak_threshold=3.0,
        background_subtract=False,
        counting_method="iec_tiered",
    )
    assert peaks == []


def test_absent_neighbor_does_not_erase_real_supported_line():
    peaks = analyze_raw_spectrum_targeted(
        synthetic_data(),
        [GammaLine(200.0, 0.5, "Present"), GammaLine(203.0, 0.1, "Absent")],
        peak_threshold=3.0,
        background_subtract=False,
        counting_method="iec_tiered",
    )
    supported = [peak for peak in peaks if peak.isotope == "Present"]
    assert len(supported) == 1
    assert supported[0].energy_keV == pytest.approx(200.0, abs=0.25)
    assert supported[0].net_counts == pytest.approx(
        2000.0 * 1.2 * np.sqrt(2 * np.pi), rel=0.1
    )


def test_resolved_weak_neighbor_is_recovered_without_biasing_strong_peak():
    x = np.arange(512, dtype=float)
    counts = (
        40.0
        + 2000.0 * np.exp(-0.5 * ((x - 200.0) / 1.2) ** 2)
        + 500.0 * np.exp(-0.5 * ((x - 204.0) / 1.2) ** 2)
    )
    spectrum = GammaSpectrum(
        counts=counts,
        live_time=100.0,
        real_time=100.0,
        calibration={"energy": [0.0, 1.0]},
    )
    data = FluxWireData(
        sample_id="real_doublet",
        spectrum=spectrum,
        energy_calibration=[0.0, 1.0],
        resolution=[2.0, 0.0],
    )
    peaks = analyze_raw_spectrum_targeted(
        data,
        [GammaLine(200.0, 0.5, "Strong"), GammaLine(204.0, 0.5, "Weak")],
        peak_threshold=3.0,
        background_subtract=False,
        counting_method="iec_tiered",
    )
    by_isotope = {peak.isotope: peak for peak in peaks}
    assert set(by_isotope) == {"Strong", "Weak"}
    assert by_isotope["Strong"].net_counts == pytest.approx(
        2000.0 * 1.2 * np.sqrt(2 * np.pi), rel=0.01
    )
    assert by_isotope["Weak"].net_counts == pytest.approx(
        500.0 * 1.2 * np.sqrt(2 * np.pi), rel=0.01
    )


def test_declared_high_variance_bin_does_not_bias_peak_fit():
    data = synthetic_data(multiplet=True)
    data.spectrum.counts[201] += 1000.0
    data.spectrum.counts_uncertainty[201] = 1.0e6
    peaks = analyze_raw_spectrum_targeted(
        data,
        [GammaLine(200.0, 0.5, "Strong"), GammaLine(205.0, 0.5, "Weak")],
        background_subtract=False,
        counting_method="iec_tiered",
    )
    by_isotope = {peak.isotope: peak for peak in peaks}
    assert set(by_isotope) == {"Strong", "Weak"}
    assert by_isotope["Strong"].net_counts == pytest.approx(
        2000.0 * 1.2 * np.sqrt(2 * np.pi), rel=0.01
    )
    assert by_isotope["Weak"].net_counts == pytest.approx(
        1000.0 * 1.2 * np.sqrt(2 * np.pi), rel=0.01
    )


@pytest.mark.parametrize("method", ["covell", "gilmore", "iec_tiered", "qg"])
def test_gross_sigma_retains_supplied_channel_variance(method):
    line = GammaLine(200.0, 0.5, "Present")
    low = analyze_raw_spectrum_targeted(
        synthetic_data(1.0), [line], background_subtract=False, counting_method=method
    )[0]
    high = analyze_raw_spectrum_targeted(
        synthetic_data(10.0), [line], background_subtract=False, counting_method=method
    )[0]
    assert high.gross_counts_unc == pytest.approx(10 * low.gross_counts_unc, rel=1e-5)
    assert high.comparison_gross_counts_unc == pytest.approx(
        10 * low.comparison_gross_counts_unc, rel=1e-5
    )


@pytest.mark.parametrize("method", ["covell", "gilmore", "iec_tiered", "qg"])
def test_background_corrected_sample_net_is_separate_from_raw_comparison(method):
    sample = synthetic_data()
    ambient = synthetic_data()
    x = np.arange(512, dtype=float)
    true_counts = 500.0 * np.exp(-0.5 * ((x - 200.0) / 1.2) ** 2)
    sample.spectrum.counts += true_counts
    sample.spectrum.counts_uncertainty = np.sqrt(sample.spectrum.counts)
    peaks = analyze_raw_spectrum_targeted(
        sample,
        [GammaLine(200.0, 0.5, "Present")],
        peak_threshold=3.0,
        background_spectrum=ambient.spectrum,
        counting_method=method,
    )
    assert len(peaks) == 1
    assert peaks[0].net_counts == pytest.approx(true_counts.sum(), rel=0.01)
    assert peaks[0].comparison_net_counts > 4 * peaks[0].net_counts


@pytest.mark.parametrize("method", ["covell", "gilmore", "iec_tiered", "qg"])
def test_measured_background_peak_with_insignificant_residual_is_not_sample_activity(
    method,
):
    x = np.arange(512, dtype=float)
    ambient_counts = 40.0 + 2000.0 * np.exp(-0.5 * ((x - 200.0) / 1.2) ** 2)
    sample_counts = ambient_counts + np.exp(-0.5 * ((x - 200.0) / 1.2) ** 2)
    sample = GammaSpectrum(
        counts=sample_counts,
        live_time=100.0,
        real_time=100.0,
        calibration={"energy": [0.0, 1.0]},
    )
    ambient = GammaSpectrum(
        counts=ambient_counts,
        live_time=100.0,
        real_time=100.0,
        calibration={"energy": [0.0, 1.0]},
    )
    data = FluxWireData(
        sample_id="ambient_only",
        spectrum=sample,
        energy_calibration=[0.0, 1.0],
        resolution=[2.0, 0.0],
    )
    peaks = analyze_raw_spectrum_targeted(
        data,
        [GammaLine(200.0, 0.5, "Ambient")],
        peak_threshold=3.0,
        background_spectrum=ambient,
        counting_method=method,
    )
    assert peaks == []


@pytest.mark.parametrize("method", ["qg", "iec_tiered"])
def test_multiplet_raw_comparison_and_physical_counts_use_their_own_spectra(method):
    sample = synthetic_data(multiplet=True)
    ambient = synthetic_data(multiplet=True)
    x = np.arange(512, dtype=float)
    for energy, amplitude in [(200.0, 500.0), (205.0, 300.0)]:
        sample.spectrum.counts += amplitude * np.exp(-0.5 * ((x - energy) / 1.2) ** 2)
    sample.spectrum.counts_uncertainty = np.sqrt(sample.spectrum.counts)
    peaks = analyze_raw_spectrum_targeted(
        sample,
        [GammaLine(200.0, 0.5, "Strong"), GammaLine(205.0, 0.5, "Weak")],
        peak_threshold=3.0,
        background_spectrum=ambient.spectrum,
        counting_method=method,
    )
    by_isotope = {peak.isotope: peak for peak in peaks}
    assert set(by_isotope) == {"Strong", "Weak"}
    area_per_height = 1.2 * np.sqrt(2 * np.pi)
    for isotope, physical_height, raw_height in [
        ("Strong", 500.0, 2500.0),
        ("Weak", 300.0, 1300.0),
    ]:
        peak = by_isotope[isotope]
        assert peak.net_counts == pytest.approx(
            physical_height * area_per_height, rel=0.01
        )
        assert peak.comparison_net_counts == pytest.approx(
            raw_height * area_per_height, rel=0.01
        )


@pytest.mark.parametrize("stem", ["Ti-RAFM-1_25cm", "Ti-RAFM-1b_25cm"])
def test_titanium_sc47_recovery_retries_closer_noise_maximum(stem):
    from fluxforge.io.flux_wire import read_raw_asc

    fixture_root = Path(__file__).parent / "data/flux_wires/raw"
    data = read_raw_asc(fixture_root / f"{stem}.ASC")
    background = read_raw_asc(fixture_root / "background.ASC").spectrum
    line = build_gamma_library(isotope_filter=["Sc47"])[0]
    peaks = analyze_raw_spectrum_targeted(
        data,
        [line],
        background_spectrum=background,
        profile_name="rafm_25cm",
        peak_threshold=3.0,
    )
    assert len(peaks) == 1
    assert peaks[0].isotope == "Sc47"
    assert peaks[0].channel > 316  # Closer maximum316 is a failed noise fit.
    assert abs(peaks[0].energy_keV - line.energy_keV) <= 2.0
    assert peaks[0].net_counts > 0.0
