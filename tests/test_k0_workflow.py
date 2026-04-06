from __future__ import annotations

from fluxforge.analysis import (
    K0_CAPABILITY_FLAGS,
    PeakObservation,
    aggregate_k0_analysis_bundles,
    analyze_k0_observations,
    build_detector_characterization,
    classify_peak_observation,
    evaluate_k0_qaqc,
    peak_report_to_observations,
    resolve_governed_libraries,
)


def test_classify_peak_observation_rejects_escape_peak():
    eligibility, accepted, reasons = classify_peak_observation(
        "single_escape",
        gamma_yield=1.0,
        reaction_family="thermal_capture",
    )
    assert eligibility == "escape_peak"
    assert accepted is False
    assert "escape_peaks_are_not_base_k0_lines" in reasons


def test_peak_report_to_observations_preserves_k0_metadata():
    peak_payload = {
        "spectrum_id": "spec-1",
        "live_time_s": 120.0,
        "peaks": [
            {
                "label": "gold_ref",
                "energy_keV": 411.8,
                "net_counts": 1000.0,
                "net_counts_unc": 31.6,
                "report_isotope": "Au-198",
                "sample_role": "reference_monitor",
                "sample_mass_g": 0.001,
                "efficiency": 0.02,
                "emission_probability": 0.9558,
            },
            {
                "label": "co_sample",
                "energy_keV": 1332.5,
                "net_counts": 250.0,
                "net_counts_unc": 15.8,
                "report_isotope": "Co-60",
                "sample_mass_g": 0.1,
                "efficiency": 0.01,
                "emission_probability": 0.9998,
            },
        ],
    }
    spectrum_payload = {
        "spectrum": {
            "spectrum_id": "spec-1",
            "detector_id": "hpge-01",
            "live_time": 120.0,
            "real_time": 122.0,
            "start_time": "2025-01-01T00:00:00",
            "metadata": {"geometry_id": "pos-200mm"},
        }
    }
    observations = peak_report_to_observations(
        peak_payload,
        spectrum_payload=spectrum_payload,
        irradiation_time_s=600.0,
        decay_time_s=3600.0,
        project_id="proj-1",
        sample_id="sample-1",
        irradiation_id="irr-1",
        measurement_id="meas-1",
    )
    assert len(observations) == 2
    assert observations[0].detector_id == "hpge-01"
    assert observations[0].sample_role == "reference_monitor"
    assert observations[1].eligibility_class == "direct_k0_eligible"
    assert observations[1].irradiation_time_s == 600.0
    assert observations[1].sample_id == "sample-1"


def test_build_detector_characterization_includes_conversion_models():
    artifact = build_detector_characterization(
        [
            {
                "position_mm": 200.0,
                "reference_energy_keV": 411.8,
                "net_counts": 4000.0,
                "live_time_s": 100.0,
                "activity_bq": 50000.0,
                "emission_probability": 0.95,
            },
            {
                "position_mm": 200.0,
                "reference_energy_keV": 1332.5,
                "net_counts": 500.0,
                "live_time_s": 100.0,
                "activity_bq": 50000.0,
                "emission_probability": 0.99,
            },
            {
                "position_mm": 100.0,
                "reference_energy_keV": 411.8,
                "net_counts": 16000.0,
                "live_time_s": 100.0,
                "activity_bq": 50000.0,
                "emission_probability": 0.95,
            },
            {
                "position_mm": 100.0,
                "reference_energy_keV": 1332.5,
                "net_counts": 2000.0,
                "live_time_s": 100.0,
                "activity_bq": 50000.0,
                "emission_probability": 0.99,
            },
        ],
        detector_id="hpge-01",
        reference_position_mm=200.0,
    )
    assert artifact["detector_id"] == "hpge-01"
    assert "100" in artifact["geometry_conversions"]["items"]


def test_analyze_k0_observations_generates_element_result():
    observation_payload = {
        "schema": "fluxforge.peak_observation_bundle.v1",
        "observations": [
            PeakObservation(
                peak_id="gold_ref",
                source_spectrum_id="spec-1",
                detector_id="hpge-01",
                geometry_id="pos-200mm",
                position_mm=200.0,
                line_energy_keV=411.8,
                line_id="Au-198@411.8keV",
                assigned_radionuclide="Au-198",
                net_peak_area=100000.0,
                area_uncertainty=316.0,
                live_time_s=100.0,
                real_time_s=102.0,
                count_start_time="2025-01-01T00:00:00",
                reference_time=None,
                irradiation_time_s=600.0,
                decay_time_s=3600.0,
                counting_time_s=100.0,
                dead_time_correction_method="live_time_real_time",
                baseline_method="local",
                deconvolution_status="resolved",
                analyst_review_status="reviewed",
                peak_area_provenance="manual_review",
                efficiency=0.02,
                emission_probability=0.9558,
                sample_role="reference_monitor",
                sample_mass_g=0.001,
                project_id="proj-1",
                sample_id="sample-1",
                irradiation_id="irr-1",
                measurement_id="meas-1",
            ).to_dict(),
            PeakObservation(
                peak_id="co_line",
                source_spectrum_id="spec-1",
                detector_id="hpge-01",
                geometry_id="pos-200mm",
                position_mm=200.0,
                line_energy_keV=1332.5,
                line_id="Co-60@1332.5keV",
                assigned_radionuclide="Co-60",
                net_peak_area=2500.0,
                area_uncertainty=50.0,
                live_time_s=100.0,
                real_time_s=102.0,
                count_start_time="2025-01-01T00:00:00",
                reference_time=None,
                irradiation_time_s=600.0,
                decay_time_s=3600.0,
                counting_time_s=100.0,
                dead_time_correction_method="live_time_real_time",
                baseline_method="local",
                deconvolution_status="resolved",
                analyst_review_status="reviewed",
                peak_area_provenance="manual_review",
                efficiency=0.01,
                emission_probability=0.9998,
                sample_role="sample",
                sample_mass_g=0.1,
                project_id="proj-1",
                sample_id="sample-1",
                irradiation_id="irr-1",
                measurement_id="meas-1",
            ).to_dict(),
        ],
    }
    facility_payload = {
        "schema": "fluxforge.facility_characterization.v1",
        "flux_parameters": {
            "f": 25.0,
            "f_uncertainty": 1.0,
            "alpha": 0.01,
            "alpha_uncertainty": 0.005,
            "phi_thermal": 1.0e12,
            "phi_epithermal": 4.0e10,
            "phi_fast": 0.0,
        },
        "temperature": {"value_K": 300.0},
    }
    bundle = analyze_k0_observations(
        observation_payload,
        facility_payload,
        sample_mass_g=0.1,
        reference_isotope="Au-198",
        reference_mass_g=0.001,
    )
    assert bundle["summary"]["element_count"] == 1
    assert bundle["element_results"][0]["element"] == "Co"
    assert bundle["capability_flags"] == K0_CAPABILITY_FLAGS
    assert bundle["libraries"]["standard_k0_library"]["status"] == "partial"
    assert bundle["summary"]["sample_id"] == "sample-1"


def test_resolve_governed_libraries_loads_external_library(tmp_path):
    library_path = tmp_path / "full_library.json"
    library_path.write_text(
        """
        {
          "library_id": "custom.k0.full",
          "version": "2026-test",
          "scope": "Full test library",
          "status": "external",
          "provenance_note": "test",
          "records": {
            "Au-198": {
              "product_isotope": "Au-198",
              "target_isotope": "Au-197",
              "element": "Au",
              "gamma_energy_keV": 411.8,
              "gamma_intensity": 0.9558,
              "half_life_s": 232796.16,
              "k0_Au": 1.0,
              "Q0": 15.71,
              "atomic_mass_g_mol": 196.967
            }
          }
        }
        """,
        encoding="utf-8",
    )
    standard, auxiliary = resolve_governed_libraries(k0_library_file=library_path)
    assert standard.library_id == "custom.k0.full"
    assert auxiliary.library_id.startswith("fluxforge.k0.")


def test_aggregate_and_qaqc_k0_bundles():
    analysis_payloads = [
        {
            "summary": {"project_id": "proj-1", "sample_id": "sample-1"},
            "element_results": [
                {
                    "project_id": "proj-1",
                    "sample_id": "sample-1",
                    "element": "Co",
                    "measurement_ids": ["m1"],
                    "irradiation_ids": ["irr-1"],
                    "line_ids": ["Co-60@1332.5keV"],
                    "concentration_ug_g": 10.0,
                    "concentration_unc_ug_g": 1.0,
                }
            ],
        },
        {
            "summary": {"project_id": "proj-1", "sample_id": "sample-1"},
            "element_results": [
                {
                    "project_id": "proj-1",
                    "sample_id": "sample-1",
                    "element": "Co",
                    "measurement_ids": ["m2"],
                    "irradiation_ids": ["irr-2"],
                    "line_ids": ["Co-60@1173.2keV"],
                    "concentration_ug_g": 12.0,
                    "concentration_unc_ug_g": 2.0,
                }
            ],
        },
    ]
    aggregate = aggregate_k0_analysis_bundles(analysis_payloads)
    assert aggregate["summary"]["aggregated_result_count"] == 1
    assert aggregate["aggregated_results"][0]["measurement_ids"] == ["m1", "m2"]

    qaqc = evaluate_k0_qaqc(
        [
            {
                "role": "blank",
                "sample_id": "blank-1",
                "default_limit_ug_g": 0.5,
                "analysis_payload": {
                    "element_results": [{"element": "Co", "concentration_ug_g": 0.1}]
                },
            },
            {
                "role": "crm",
                "sample_id": "crm-1",
                "analysis_payload": {
                    "element_results": [
                        {
                            "element": "Co",
                            "concentration_ug_g": 10.5,
                            "concentration_unc_ug_g": 0.5,
                        }
                    ]
                },
                "certified_values": {"Co": {"value_ug_g": 10.0, "unc_ug_g": 0.5}},
            },
        ]
    )
    assert qaqc["summary"]["pass_count"] == 2
