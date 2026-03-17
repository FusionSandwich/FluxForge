from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxforge.examples.rafm_workflow import (
    analyze_flux_wire_sample,
    analyze_generic_sample,
    default_paths,
    estimate_rafm_sample_mass_g,
    ensure_results_tree,
    load_rafm_example_metadata,
    normalize_pairing_key,
    prune_generic_targeted_lines,
    select_generic_targeted_lines,
    resolve_measurement_timing,
    run_rafm_validation,
    build_generic_gamma_library,
    build_line_diagnostic_records,
    workflow_profile_energy_calibration,
    merge_detected_and_targeted_peaks,
)
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.analysis.flux_wire_analysis import GammaLine, analyze_raw_spectrum, analyze_raw_spectrum_targeted
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "RAFM_irradiation"


def test_metadata_and_pairing_aliases_load():
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    assert metadata.config["profile_name"] == "rafm_25cm"
    assert metadata.config["use_profile_energy_calibration"] is True
    assert "Fe59" in metadata.sample_gamma_library
    assert normalize_pairing_key("Cu-RAFM-1_25cm", metadata.pairing_aliases) == "cu-rafm-1"
    assert normalize_pairing_key("Fe-Cd-RAFM-1_0cm", metadata.pairing_aliases) == "fe-cd-rafm-1"


def test_workflow_profile_energy_calibration_supports_astm_inl_alias():
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    astm_config = dict(metadata.config)
    astm_config["profile_name"] = "astm_inl_dosimetry"

    expected = load_rafm_profile("rafm_25cm").energy_calibration
    assert workflow_profile_energy_calibration(astm_config) == pytest.approx(expected)


def test_prune_generic_targeted_lines_drops_weak_nearby_nuisance_lines():
    lines = [
        GammaLine(energy_keV=1099.25, intensity=0.5659, isotope="Fe59"),
        GammaLine(energy_keV=1102.43, intensity=0.0027, isotope="Tb154m"),
        GammaLine(energy_keV=1173.23, intensity=0.9985, isotope="Co60"),
        GammaLine(energy_keV=1177.71, intensity=0.0029, isotope="Tb154m"),
        GammaLine(energy_keV=1231.02, intensity=0.1130, isotope="Ta182"),
        GammaLine(energy_keV=1229.42, intensity=0.0050, isotope="Tb154m"),
    ]
    pruned = prune_generic_targeted_lines(lines, neighbor_window_keV=5.0, intensity_ratio=20.0)
    got = {(line.isotope, round(line.energy_keV, 2)) for line in pruned}
    assert ("Fe59", 1099.25) in got
    assert ("Co60", 1173.23) in got
    assert ("Ta182", 1231.02) in got
    assert ("Tb154m", 1102.43) not in got
    assert ("Tb154m", 1177.71) not in got
    assert ("Tb154m", 1229.42) not in got


def test_select_generic_targeted_lines_keeps_supported_sets_and_limits_dense_unsupported_isotopes():
    detected = [
        SimpleNamespace(isotope="Fe59", energy_keV=1291.6),
        SimpleNamespace(isotope=None, energy_keV=685.8),
    ]
    lines = [
        GammaLine(energy_keV=1099.25, intensity=0.5659, isotope="Fe59"),
        GammaLine(energy_keV=1291.59, intensity=0.4321, isotope="Fe59"),
        GammaLine(energy_keV=479.49, intensity=0.21681, isotope="W187"),
        GammaLine(energy_keV=685.74, intensity=0.27, isotope="W187"),
        GammaLine(energy_keV=551.49, intensity=0.05049, isotope="W187"),
        GammaLine(energy_keV=123.07, intensity=0.30, isotope="Tb154m"),
        GammaLine(energy_keV=247.94, intensity=0.22, isotope="Tb154m"),
        GammaLine(energy_keV=1004.73, intensity=0.109, isotope="Tb154m"),
        GammaLine(energy_keV=1102.43, intensity=0.0027, isotope="Tb154m"),
        GammaLine(energy_keV=1177.71, intensity=0.0029, isotope="Tb154m"),
    ]
    config = {
        "min_peak_energy_keV": 80.0,
        "max_peak_energy_keV": 3000.0,
        "generic_targeted_support_window_keV": 3.0,
        "generic_targeted_support_min_intensity": 0.15,
        "generic_targeted_fallback_top_lines_per_isotope": 2,
        "generic_targeted_fallback_min_intensity": 0.03,
        "generic_targeted_prune_neighbor_window_keV": 5.0,
        "generic_targeted_prune_intensity_ratio": 20.0,
    }
    selected = select_generic_targeted_lines(detected, lines, config)
    got = {(line.isotope, round(line.energy_keV, 2)) for line in selected}
    assert ("Fe59", 1099.25) in got
    assert ("Fe59", 1291.59) in got
    assert ("W187", 685.74) in got
    assert ("W187", 479.49) in got
    assert ("Tb154m", 123.07) in got
    assert ("Tb154m", 247.94) in got
    assert ("Tb154m", 1004.73) not in got
    assert ("Tb154m", 1102.43) not in got
    assert ("Tb154m", 1177.71) not in got


def test_raw_loader_uses_file_energy_calibration_and_profile_efficiency():
    data = read_raw_asc(
        EXAMPLE_ROOT / "raw_gamma_spec" / "flux_wires" / "Co-Cd-RAFM-1_25cm.ASC",
        profile_name="rafm_25cm",
    )
    assert data.energy_calibration[0] == pytest.approx(0.5410, abs=1e-6)
    assert data.energy_calibration[1] == pytest.approx(0.4980, abs=1e-6)
    assert data.energy_calibration[2] == pytest.approx(2.605e-7, abs=1e-12)
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(-20.26)
    assert data.efficiency.source_distance_cm == pytest.approx(25.0)


def test_resolve_measurement_timing_for_rafm_and_flux_wires():
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)

    rafm3 = resolve_measurement_timing("RAFM3-B_2hrEOI", None, metadata)
    assert rafm3.sample_group == "RAFM3"
    assert rafm3.compare_eoi is True
    assert rafm3.decay_time_s is not None and rafm3.decay_time_s > 0

    rafm4 = resolve_measurement_timing("RAFM4-B_15dEOI", None, metadata)
    assert rafm4.sample_group == "RAFM4"
    assert rafm4.compare_eoi is True
    assert rafm4.decay_time_s is not None and rafm4.decay_time_s > 0
    assert rafm4.irradiation_phase == "phase2_whale_tube"

    wire = resolve_measurement_timing("Co-RAFM-1_25cm", None, metadata)
    assert wire.sample_group == "flux_wires"
    assert wire.compare_eoi is True
    assert wire.irradiation_time_s == 7200
    assert wire.irradiation_phase == "phase2_whale_tube"


def test_estimate_rafm_sample_mass_from_geometry_metadata():
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)

    mass_g = estimate_rafm_sample_mass_g("RAFM4-B_15dEOI", metadata)

    assert mass_g == pytest.approx(0.076026542216873 * 7.87)


def test_analyze_generic_sample_writes_artifacts(tmp_path):
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = default_paths(EXAMPLE_ROOT, results_root=tmp_path / "results")
    tree = ensure_results_tree(paths.results_root)
    gamma_library, half_lives = build_generic_gamma_library(metadata)
    background = read_raw_asc(paths.background_path, profile_name=metadata.config["profile_name"]).spectrum
    assert background is not None

    raw_path = paths.raw_root / "RAFM4" / "RAFM4-B_15dEOI.ASC"
    qg_path = paths.qg_root / "RAFM4" / "RAFM4-B_15dEOI.txt"
    artifact = analyze_generic_sample(raw_path, metadata, paths, tree, gamma_library, half_lives, background, qg_path)

    assert artifact["sample_id"] == "RAFM4-B_15dEOI"
    assert artifact["sample_group"] == "RAFM4"
    assert artifact["n_detected_peaks"] > 0
    assert "Cr51" in artifact["isotopes"]
    assert Path(artifact["counts_csv"]).exists()
    assert Path(artifact["comparison_report_txt"]).exists()
    assert Path(artifact["line_diagnostics_csv"]).exists()
    assert Path(artifact["qg_consistency_csv"]).exists()
    assert artifact["analysis_configuration"]["efficiency_source"] == "profile:rafm_25cm"
    assert artifact["analysis_configuration"]["energy_calibration_source"] == "profile_override"
    assert artifact["analysis_configuration"]["qg_comparison_stage"] == "raw_fluxforge_activity_pre_cd_compensation"
    assert artifact["isotopes"]["Cr51"]["sample_mass_g"] == pytest.approx(0.076026542216873 * 7.87)
    assert artifact["isotopes"]["Cr51"]["specific_activity_Bq_g"] > 0.0
    assert (tree["artifacts"] / "RAFM4-B_15dEOI.json").exists()
    assert (tree["plots_comparisons"] / "RAFM4-B_15dEOI_vs_qg.png").exists()


def test_generic_targeted_selection_recovers_known_rafm4_missing_lines():
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = default_paths(EXAMPLE_ROOT)
    gamma_library, _ = build_generic_gamma_library(metadata)
    energy_override = workflow_profile_energy_calibration(metadata.config)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=energy_override,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    assert background is not None

    raw_path = paths.raw_root / "RAFM4" / "RAFM4-A_15dEOI.ASC"
    qg_path = paths.qg_root / "RAFM4" / "RAFM4-A_15dEOI.txt"
    raw_data = read_raw_asc(
        raw_path,
        energy_calibration_override=energy_override,
        profile_name=metadata.config["profile_name"],
    )
    adjusted = subtract_measured_background(raw_data.spectrum, background, mode="live", negative_policy="hybrid", warn_missing=True)
    detected_peaks = analyze_raw_spectrum(
        adjusted,
        efficiency=raw_data.efficiency,
        gamma_library=list(gamma_library),
        peak_threshold=float(metadata.config.get("peak_significance_sigma", 3.0)),
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
        background_subtract=False,
    )
    targeted_gamma_library = select_generic_targeted_lines(detected_peaks, list(gamma_library), metadata.config)
    targeted_peaks = analyze_raw_spectrum_targeted(
        data=raw_data,
        expected_lines=targeted_gamma_library,
        peak_threshold=float(metadata.config.get("targeted_peak_significance_sigma", metadata.config.get("peak_significance_sigma", 3.0))),
        min_energy_keV=float(metadata.config.get("min_peak_energy_keV", 80.0)),
        max_energy_keV=float(metadata.config.get("max_peak_energy_keV", 3000.0)),
        background_spectrum=background,
        background_subtract=True,
        profile_name=metadata.config["profile_name"],
        roi_width_fwhm=float(metadata.config.get("flux_wire_roi_width_fwhm", 4.0)),
        background_width_channels=int(metadata.config.get("flux_wire_background_width_channels", 1)),
        background_gap_fwhm=float(metadata.config.get("flux_wire_background_gap_fwhm", 0.0)),
        comparison_background_model=str(metadata.config.get("generic_comparison_background_model", "linear")),
        broad_window_max_raw_gross_ratio=float(metadata.config.get("generic_broad_window_max_raw_gross_ratio", 1.35)),
    )
    peaks = merge_detected_and_targeted_peaks(detected_peaks, targeted_peaks, metadata.config)
    reference = read_processed_txt(qg_path, profile_name=metadata.config["profile_name"])
    line_rows, _ = build_line_diagnostic_records("RAFM4-A_15dEOI", "RAFM4", peaks, reference, metadata.config)
    missing = {
        (str(row["reference_isotope"]), round(float(row["reference_energy_keV"]), 2))
        for row in line_rows
        if row["diagnostic_bucket"] == "missing_in_fluxforge"
    }
    assert ("Fe59", 1099.25) not in missing
    assert ("Co60", 1173.17) not in missing
    assert ("Ta182", 1231.00) not in missing
    assert ("Ta182", 229.55) not in missing


def test_analyze_flux_wire_sample_writes_reactions(tmp_path):
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = default_paths(EXAMPLE_ROOT, results_root=tmp_path / "results")
    tree = ensure_results_tree(paths.results_root)
    background = read_raw_asc(paths.background_path, profile_name=metadata.config["profile_name"]).spectrum
    assert background is not None

    raw_path = paths.raw_root / "flux_wires" / "Co-RAFM-1_25cm.ASC"
    qg_path = paths.qg_root / "flux_wires" / "Co-RAFM-1_25cm.txt"
    sample_key = normalize_pairing_key(raw_path.stem, metadata.pairing_aliases)
    artifact = analyze_flux_wire_sample(raw_path, metadata, paths, tree, background, qg_path, sample_key)

    assert artifact["sample_group"] == "flux_wires"
    assert artifact["reactions"]
    assert any(row["reaction_id"] == "Co-59(n,g)Co-60" for row in artifact["reactions"])
    assert "Co60" in artifact["isotopes"]
    assert Path(artifact["comparison_report_txt"]).exists()
    assert Path(artifact["line_diagnostics_csv"]).exists()
    assert Path(artifact["qg_consistency_csv"]).exists()
    assert artifact["analysis_configuration"]["energy_calibration_source"] == "profile_override"
    assert artifact["analysis_configuration"]["qg_comparison_stage"] == "raw_fluxforge_activity_pre_cd_compensation"
    with open(artifact["line_diagnostics_csv"], newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    assert "reference_gross_counts" in first_row
    assert "raw_gross_counts" in first_row


def test_flux_wire_count_parity_representative_lines(tmp_path):
    metadata = load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = default_paths(EXAMPLE_ROOT, results_root=tmp_path / "results")
    tree = ensure_results_tree(paths.results_root)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=[-1.694, 0.4996, 6.710e-08],
        profile_name=metadata.config["profile_name"],
    ).spectrum
    assert background is not None

    expectations = {
        "Co-Cd-RAFM-1_25cm": {
            ("Co60", 1173.13): {"gross_max": 0.15, "net_max": 0.02},
            ("Co60", 1332.44): {"gross_max": 0.16, "net_max": 0.08},
        },
        "Sc-RAFM-1_25cm": {
            ("Sc46", 889.36): {"gross_max": 0.02, "net_max": 0.01},
            ("Sc46", 1120.41): {"gross_max": 0.03, "net_max": 0.03},
        },
        "Ti-RAFM-1a_25cm": {
            ("Sc47", 159.30): {"gross_max": 0.05, "net_max": 0.02},
            ("Sc48", 175.28): {"gross_max": 0.12, "net_max": 0.05},
            ("Sc46", 889.11): {"gross_max": 0.03, "net_max": 0.08},
        },
    }

    for stem, line_expectations in expectations.items():
        raw_path = paths.raw_root / "flux_wires" / f"{stem}.ASC"
        qg_path = paths.qg_root / "flux_wires" / f"{stem}.txt"
        sample_key = normalize_pairing_key(raw_path.stem, metadata.pairing_aliases)
        artifact = analyze_flux_wire_sample(raw_path, metadata, paths, tree, background, qg_path, sample_key)
        rows = {
            (str(row["reference_isotope"]), round(float(row["reference_energy_keV"]), 2)): row
            for row in artifact["line_diagnostics"]
        }
        for (isotope, energy), limits in line_expectations.items():
            row = rows[(isotope, round(energy, 2))]
            assert abs(float(row["relative_gross_error"])) <= limits["gross_max"]
            assert abs(float(row["relative_count_error"])) <= limits["net_max"]


def test_run_rafm_validation_subset_generates_summary(tmp_path):
    summary = run_rafm_validation(
        example_root=EXAMPLE_ROOT,
        results_root=tmp_path / "results",
        enforce_thresholds=False,
        max_spectra=2,
    )
    assert summary["n_raw_analyzed"] == 2
    assert Path(summary["results_root"]).exists()
    assert (Path(summary["results_root"]) / "validation_summary.json").exists()
    assert (Path(summary["results_root"]) / "tables" / "raw_qg_pairing.csv").exists()
    assert (Path(summary["results_root"]) / "tables" / "line_diagnostics.csv").exists()
    assert (Path(summary["results_root"]) / "tables" / "qg_internal_consistency.csv").exists()
    assert (Path(summary["results_root"]) / "tables" / "flux_wire_count_disagreement.csv").exists()
    assert (Path(summary["results_root"]) / "reports").exists()
