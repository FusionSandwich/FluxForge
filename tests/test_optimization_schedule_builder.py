from __future__ import annotations

import json

import pytest

from fluxforge.analysis.optimization_schedule_builder import (
    build_difom_payload_from_activity_review,
    summarize_neutron_spectrum_source,
)


def test_summarize_neutron_spectrum_source_from_csv(tmp_path):
    spectrum_csv = tmp_path / "neutron_flux.csv"
    spectrum_csv.write_text(
        "energy_eV,flux\n"
        "1.0e2,1.0\n"
        "1.0e4,2.0\n"
        "2.0e5,3.0\n",
        encoding="utf-8",
    )

    summary = summarize_neutron_spectrum_source(
        neutron_spectrum_csv=spectrum_csv,
        base_scale=1.0,
        reference_integral_flux=0.0,
    )

    assert summary["source"] == "csv"
    assert summary["integral_flux"] == pytest.approx(6.0)
    assert summary["high_energy_fraction"] == pytest.approx(0.5)
    assert summary["applied_flux_scale"] > 0.0


def test_build_difom_payload_from_activity_review():
    activity_review_payload = {
        "schema": "fluxforge.activity_review.v1",
        "live_time_s": 300.0,
        "cooling_time_s": 3600.0,
        "line_results": [
            {
                "nuclide": "Mo-99",
                "matched_line_energy_keV": 140.5,
            },
            {
                "nuclide": "Sc-46",
                "matched_line_energy_keV": 889.3,
            },
        ],
        "isotope_summaries": [
            {
                "nuclide": "Mo-99",
                "line_count": 2,
                "total_net_counts": 12000.0,
                "half_life_s": 65.94 * 3600.0,
                "cooling_time_s": 3600.0,
                "irradiation_time_activity_Bq": 900.0,
                "irradiation_time_activity_unc_Bq": 45.0,
            },
            {
                "nuclide": "Sc-46",
                "line_count": 1,
                "total_net_counts": 6000.0,
                "half_life_s": 83.79 * 24.0 * 3600.0,
                "cooling_time_s": 3600.0,
                "irradiation_time_activity_Bq": 400.0,
                "irradiation_time_activity_unc_Bq": 30.0,
            },
        ],
    }

    payload = build_difom_payload_from_activity_review(
        activity_review_payload,
        irradiation_grid_s=(1800.0, 3600.0),
        cooldown_grid_s=(0.0, 3600.0),
        count_grid_s=(300.0,),
        reference_irradiation_time_s=3600.0,
        flux_scale=1.1,
    )

    assert payload["schema"] == "fluxforge.optimization_candidates.activity_review.v1"
    assert len(payload["candidates"]) == 4
    assert "Mo-99" in payload["isotope_weights"]
    first = payload["candidates"][0]
    assert first["irradiation_time_s"] > 0.0
    assert first["lines"]
    assert all("signal_counts" in row for row in first["lines"])
    assert all("interference_counts" in row for row in first["lines"])

    # Ensure the generated payload is JSON serializable for CLI export paths.
    json.dumps(payload)
