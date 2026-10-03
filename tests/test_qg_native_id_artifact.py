from __future__ import annotations

import json
from pathlib import Path

from fluxforge.examples import rafm_workflow


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "RAFM_irradiation"


def test_qg_compatibility_overlay_cannot_create_native_peak_identifications(
    tmp_path, monkeypatch
):
    metadata = rafm_workflow.load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = rafm_workflow.default_paths(EXAMPLE_ROOT, results_root=tmp_path / "results")
    tree = rafm_workflow.ensure_results_tree(paths.results_root)
    gamma_library, half_lives = rafm_workflow.build_generic_gamma_library(metadata)
    background = rafm_workflow.read_raw_asc(
        paths.background_path,
        energy_calibration_override=rafm_workflow.workflow_profile_energy_calibration(
            metadata.config
        ),
        profile_name=metadata.config["profile_name"],
    ).spectrum
    assert background is not None

    # The real raw spectrum and QG report exercise artifact production; only the
    # two extraction stages are controlled so this is an empty-native-peaks case.
    monkeypatch.setattr(rafm_workflow, "analyze_raw_spectrum", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rafm_workflow, "analyze_raw_spectrum_targeted", lambda *args, **kwargs: []
    )

    sample_id = "RAFM4-B_15dEOI"
    raw_path = paths.raw_root / "RAFM4" / f"{sample_id}.ASC"
    qg_path = paths.qg_root / "RAFM4" / f"{sample_id}.txt"
    artifact = rafm_workflow.analyze_generic_sample(
        raw_path,
        metadata,
        paths,
        tree,
        gamma_library,
        half_lives,
        background,
        qg_path,
    )

    reference = rafm_workflow.read_processed_txt(
        qg_path, profile_name=metadata.config["profile_name"]
    )
    min_energy = float(metadata.config.get("min_peak_energy_keV", 0.0))
    max_energy = float(metadata.config.get("max_peak_energy_keV", 1.0e9))
    min_counts = float(metadata.config.get("minimum_qg_net_counts", 1.0))
    expected_reference_count = sum(
        min_energy <= peak["energy_keV"] <= max_energy
        and peak["net_counts"] >= min_counts
        for peak in rafm_workflow.qg_reference_peaks(reference)
    )

    # QG parity creates report-aligned peaks in the compatibility artifact.
    assert artifact["n_detected_peaks"] > 0
    assert artifact["peaks"]
    assert any(peak["isotope"] for peak in artifact["peaks"])
    assert artifact["native_peak_identifications"] == []

    validation = artifact["peak_id_validation"]
    assert validation["reference_peaks"] == expected_reference_count > 0
    assert validation["same_id"] == 0
    assert validation["passed"] is False
    assert all(not row["isotope_match"] for row in artifact["peak_id_comparison"])

    saved = json.loads(
        (tree["artifacts"] / f"{sample_id}.json").read_text(encoding="utf-8")
    )
    assert saved["native_peak_identifications"] == []
    assert saved["peak_id_validation"] == validation


def test_corrected_tb_activity_is_not_copied_to_ta_in_artifact(tmp_path, monkeypatch):
    metadata = rafm_workflow.load_rafm_example_metadata(EXAMPLE_ROOT)
    paths = rafm_workflow.default_paths(EXAMPLE_ROOT, results_root=tmp_path / "results")
    tree = rafm_workflow.ensure_results_tree(paths.results_root)
    gamma_library, half_lives = rafm_workflow.build_generic_gamma_library(metadata)
    background = rafm_workflow.read_raw_asc(
        paths.background_path,
        energy_calibration_override=rafm_workflow.workflow_profile_energy_calibration(
            metadata.config
        ),
        profile_name=metadata.config["profile_name"],
    ).spectrum
    assert background is not None
    monkeypatch.setattr(rafm_workflow, "analyze_raw_spectrum", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rafm_workflow, "analyze_raw_spectrum_targeted", lambda *args, **kwargs: []
    )

    sample_id = "RAFM4-N_15dEOI"
    raw_path = paths.raw_root / "RAFM4" / f"{sample_id}.ASC"
    qg_path = paths.qg_root / "RAFM4" / f"{sample_id}.txt"
    artifact = rafm_workflow.analyze_generic_sample(
        raw_path,
        metadata,
        paths,
        tree,
        gamma_library,
        half_lives,
        background,
        qg_path,
    )
    report = rafm_workflow.read_processed_txt(
        qg_path, profile_name=metadata.config["profile_name"]
    )
    ta_report = next(nuclide for nuclide in report.nuclides if nuclide.isotope == "Ta182")
    tb_report = next(nuclide for nuclide in report.nuclides if nuclide.isotope == "Tb154m")
    tb_source_row = next(
        peak for peak in tb_report.peaks if peak["source_line_number"] == 92
    )

    # The source-bound correction changes comparison identity, while the
    # report's Tb summary activity stays excluded from the modeled isotope map.
    correction = artifact["qg_peak_identity_corrections"]
    assert len(correction) == 1
    assert correction[0]["reported_isotope"] == "Tb154m"
    assert correction[0]["corrected_isotope"] == "Ta182"
    assert correction[0]["source_line_number"] == 92
    assert artifact["excluded_misassigned_report_activity_isotopes"] == ["Tb154m"]
    assert "Tb154m" not in artifact["isotopes"]
    assert artifact["isotopes"]["Ta182"]["activity_bq"] == ta_report.activity_bq
    assert artifact["isotopes"]["Ta182"]["activity_bq"] != tb_report.activity_bq

    corrected_row = next(
        row for row in artifact["peak_comparison"]
        if row.get("identity_correction") is not None
    )
    assert corrected_row["reported_reference_isotope"] == "Tb154m"
    assert corrected_row["reference_isotope"] == "Ta182"
    assert corrected_row["matched"] is False
    assert all(
        not (
            peak["isotope"] == "Ta182"
            and abs(peak["energy_keV"] - tb_source_row["center_keV"]) <= 0.1
        )
        for peak in artifact["peaks"]
    )

    saved = json.loads(
        (tree["artifacts"] / f"{sample_id}.json").read_text(encoding="utf-8")
    )
    assert "Tb154m" not in saved["isotopes"]
    assert saved["isotopes"]["Ta182"]["activity_bq"] == ta_report.activity_bq
