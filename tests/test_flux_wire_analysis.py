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
    assert (
        row["single_peak_outlier_energies"]
        or row["excluded_peak_energies"]
    )

    first_diag = row["single_peak_activity_diagnostics"][0]
    assert "relative_delta_vs_combined" in first_diag
    assert "leave_one_out_activity_bq" in first_diag
    assert "all_vs_leave_one_out_relative_delta" in first_diag
