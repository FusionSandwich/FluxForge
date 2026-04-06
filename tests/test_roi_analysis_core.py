from __future__ import annotations

from datetime import datetime

import numpy as np

from fluxforge.core.analysis_workspace import (
    analyze_roi_region,
    compute_roi_statistics,
    detect_peak_candidates,
)
from fluxforge.gui.panels.modern_shell import build_demo_spectrum
from fluxforge.io.spe import GammaSpectrum


def _multiplet_spectrum(scale: float = 1.0) -> GammaSpectrum:
    channels = np.arange(512, dtype=float)
    background = 20.0 + 0.02 * channels
    peak_a = 820.0 * np.exp(-0.5 * ((channels - 210.0) / 4.0) ** 2)
    peak_b = 610.0 * np.exp(-0.5 * ((channels - 219.0) / 4.8) ** 2)
    counts = (background + peak_a + peak_b) * float(scale)
    return GammaSpectrum(
        counts=np.asarray(counts, dtype=float),
        channels=channels,
        calibration={"energy": [0.0, 1.0]},
        live_time=120.0,
        real_time=126.0,
        start_time=datetime(2026, 4, 1, 12, 0),
        spectrum_id=f"multiplet-{scale:.2f}",
    )


def test_peak_search_methods_detect_candidates_for_demo_workspace():
    spectrum = build_demo_spectrum()

    for method in ("mariscotti", "second_difference", "nasa_peaksearch"):
        peaks = detect_peak_candidates(spectrum, method=method, max_peaks=8)
        assert len(peaks) >= 2
        assert all(peak.energy_keV > 0.0 for peak in peaks[:2])


def test_roi_analysis_region_supports_sidebands_and_overlap_decomposition():
    spectrum = _multiplet_spectrum()

    result = analyze_roi_region(
        spectrum,
        roi_bounds_keV=(202.0, 226.0),
        label="close-doublet",
        background_method="roi_sideband",
        peak_search_method="mariscotti",
        sideband_width_keV=6.0,
        decompose_overlaps=True,
        max_components=3,
    )

    assert result.gross_counts > result.background_counts
    assert result.net_counts > 0.0
    assert result.significance > 1.0
    assert len(result.overlap_components) >= 2
    assert result.sideband_bounds_keV[0][1] == 202.0
    assert result.sideband_bounds_keV[1][0] == 226.0


def test_roi_statistics_summary_captures_variation_across_loaded_spectra():
    result = compute_roi_statistics(
        (
            ("run-a", _multiplet_spectrum(1.0)),
            ("run-b", _multiplet_spectrum(1.1)),
            ("run-c", _multiplet_spectrum(0.92)),
        ),
        roi_bounds_keV=(202.0, 226.0),
        background_method="snip",
        peak_search_method="nasa_peaksearch",
        sideband_width_keV=5.0,
    )

    assert result.sample_count == 3
    assert result.mean_net_counts > 0.0
    assert result.stdev_net_counts > 0.0
    assert result.relative_std > 0.0
    assert len(result.samples) == 3
