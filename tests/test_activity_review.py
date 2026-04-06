from __future__ import annotations

import json

import pytest

from fluxforge.core.activity_review import review_spectrum_activation
from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.data.efficiency import EfficiencyCurve


def _constant_efficiency_curve(value: float, rel_uncertainty: float) -> EfficiencyCurve:
    return EfficiencyCurve(
        model_type="empirical",
        parameters={
            "energies": [1.0, 10000.0],
            "efficiencies": [value, value],
            "interpolation": "linear",
        },
        uncertainty_model={"type": "constant", "value": rel_uncertainty},
    )


def test_review_spectrum_activation_builds_eoi_tables_and_bateman_series(tmp_path):
    gamma_path = tmp_path / "gamma_lines.json"
    gamma_path.write_text(
        json.dumps(
            [
                {
                    "nuclide": "Co-60",
                    "energy_keV": 1173.228,
                    "intensity": 0.999,
                    "intensity_unc": 0.005,
                    "half_life_s": 5.2714 * 365.25 * 24.0 * 3600.0,
                },
                {
                    "nuclide": "Co-60",
                    "energy_keV": 1332.492,
                    "intensity": 0.998,
                    "intensity_unc": 0.005,
                    "half_life_s": 5.2714 * 365.25 * 24.0 * 3600.0,
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    review = review_spectrum_activation(
        (
            PeakCandidate(
                peak_id="peak-1",
                channel=100.0,
                energy_keV=1173.20,
                significance=8.0,
                roi_bounds_keV=(1168.0, 1178.0),
                net_counts=10000.0,
                fit_quality=1.1,
                status="matched",
                nuclide="Co60",
            ),
            PeakCandidate(
                peak_id="peak-2",
                channel=120.0,
                energy_keV=1332.50,
                significance=7.5,
                roi_bounds_keV=(1327.0, 1338.0),
                net_counts=8500.0,
                fit_quality=1.0,
                status="matched",
                nuclide="Co60",
            ),
        ),
        live_time_s=120.0,
        efficiency_curve=_constant_efficiency_curve(0.18, 0.03),
        cooling_time_s=3600.0,
        source_id="custom_gamma_file",
        custom_gamma_path=str(gamma_path),
        energy_tolerance_keV=1.0,
    )

    assert len(review.line_results) == 2
    assert len(review.isotope_summaries) == 1

    summary = review.isotope_summaries[0]
    assert summary.nuclide == "Co-60"
    assert summary.irradiation_time_activity_bq > summary.count_time_activity_bq > 0.0
    assert "retains" in summary.chain_summary

    count_only_unc = review.line_results[0].count_time_activity_bq / 100.0
    assert review.line_results[0].count_time_uncertainty_bq > count_only_unc

    isotope_rows = review.isotope_rows()
    assert isotope_rows[0]["irradiation_time_activity_Bq"] == pytest.approx(
        summary.irradiation_time_activity_bq
    )
    assert "Co-60 parent" in review.bateman_plot_data
    assert "Co-60 daughter eq" in review.bateman_plot_data
    assert review.decay_plot_data["Co-60"][0][0] == pytest.approx(0.0)
