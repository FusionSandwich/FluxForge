from __future__ import annotations

import pytest

from fluxforge.core.analysis_workspace import ActivityCalculationResult
from fluxforge.core.inventory_timeline import (
    build_inventory_state_from_activity_results,
    build_inventory_state_from_payload,
    compute_inventory_time_evolution,
)


def test_inventory_time_evolution_tracks_reference_states_and_daughter_growth():
    state = build_inventory_state_from_payload(
        {
            "spectrum_id": "mo99-demo",
            "source_id": "nasa_common_lab_sources",
            "cooling_time_s": 7200.0,
            "live_time_s": 300.0,
            "isotope_summaries": [
                {
                    "nuclide": "Mo-99",
                    "line_count": 2,
                    "half_life_s": 65.94 * 3600.0,
                    "count_time_activity_Bq": 850.0,
                    "count_time_activity_unc_Bq": 40.0,
                    "irradiation_time_activity_Bq": 900.0,
                    "irradiation_time_activity_unc_Bq": 50.0,
                }
            ],
        }
    )

    result = compute_inventory_time_evolution(
        state,
        relative_times_s=(0.0, 6.0 * 3600.0, 12.0 * 3600.0),
        time_origin="eoi",
        distance_cm=25.0,
    )

    assert "Mo-99" in result.activity_series
    assert "Tc-99m" in result.activity_series
    assert result.activity_series["Mo-99"].points[0][1] > result.activity_series["Mo-99"].points[-1][1]
    assert result.activity_series["Tc-99m"].points[-1][1] > 0.0
    assert result.parents_by_nuclide["Tc-99m"] == ("Mo-99",)
    assert "Tc-99m" in result.daughters_by_nuclide["Mo-99"]

    eoi_rows = result.reference_rows("eoi")
    count_start_rows = result.reference_rows("count_start")
    count_end_rows = result.reference_rows("count_end")
    assert any(row["nuclide"] == "Mo-99" for row in eoi_rows)
    assert any(row["reference_state"] == "count_start" for row in count_start_rows)
    assert any(row["reference_state"] == "count_end" for row in count_end_rows)

    time_rows = result.time_series_rows()
    assert any(row["nuclide"] == "Total" for row in time_rows)
    assert all("dose_rate_uSv_h" in row for row in time_rows)
    assert result.notes


def test_build_inventory_state_from_activity_results_preserves_age_corrected_uncertainty():
    state = build_inventory_state_from_activity_results(
        (
            ActivityCalculationResult(
                nuclide="Mo-99",
                line_energy_keV=140.5,
                activity_bq=850.0,
                uncertainty_bq=30.0,
                age_corrected_activity_bq=900.0,
                mda_bq=0.0,
                half_life_s=65.94 * 3600.0,
                source_age_s=7200.0,
                chain_summary="demo",
                age_corrected_uncertainty_bq=45.0,
            ),
        ),
        live_time_s=300.0,
        gamma_source_id="nasa_common_lab_sources",
        sample_id="demo-spectrum",
    )

    assert state.sample_id == "demo-spectrum"
    assert state.schedule.cooling_time_s == pytest.approx(7200.0)
    assert state.seeds[0].activity_eoi_bq == pytest.approx(900.0)
    assert state.seeds[0].uncertainty_eoi_bq == pytest.approx(45.0)
