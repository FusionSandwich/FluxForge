from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fluxforge.analysis.k0_workflow import PeakObservation, analyze_k0_observations
from fluxforge.data.kayzero_k0 import import_kayzero_k0_library


def _write_sample_kayzero_tree(root: Path) -> Path:
    library_dir = root / "KayWinV4" / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    (library_dir / "k0-2023.uk0").write_text(
        "\n".join(
            [
                "Nuclide E (keV) k0 dk0 k0code",
                "Au-198 411.8 1.0 0.0 1",
                "Co-60 1332.5 1.320 0.9 1",
                "Co-60 1173.2 1.318 1.0 1",
                "Ag-110m 657.8 0.0123 2.0 1",
                "Tc-99m 140.5 0.036 2.0 1",
                "U-239 74.7 0.0133 3.0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.uQ0").write_text(
        "\n".join(
            [
                "Nuclide Q0 dQ0 Er dEr",
                "Au-198 15.71 1.5 5.65 0.1",
                "Co-60 1.99 2.0 132.0 1.0",
                "Ag-110m 12.4 3.0 42.0 0.5",
                "Tc-99m 2.24 3.0 132.0 1.0",
                "U-239 103.4 4.0 16.9 0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.uT12").write_text(
        "\n".join(
            [
                "T1/2 dT",
                "Au-198 3880.8 0.3",
                "Co-60 2771834 100.0",
                "Ag-110m 359712 57.6",
                "Tc-99m 21624 30.0",
                "U-239 1408.2 12.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.MDcode").write_text(
        "\n".join(["Au-198 1", "Co-60 10", "Ag-110m 2", "Tc-99m 3", "U-239 4"]) + "\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2020.FCd").write_text(
        "\n".join(["FCd", "Fe-59,1.002", "Zn-65,1.003"]) + "\n",
        encoding="utf-8",
    )
    (library_dir / "IRI_MB1.TXT").write_text(
        "\n".join(
            [
                "Au",
                "  M: 196.97",
                "  Reactions:",
                "  Au-197   -> Au-198    ( 2.69 d)",
                "        (theta: 1.00E+00, sigma0: 9.865E+01 b, Q0: 1.571E+01, Er: 5.65 eV)",
                "",
                "Co",
                "  M: 58.93",
                "  Reactions:",
                "  Co-59   -> Co-60    ( 5.27 y)",
                "        (theta: 1.00E+00, sigma0: 3.718E+01 b, Q0: 1.99, Er: 132.0 eV)",
                "",
                "Ag",
                "  M: 107.87",
                "  Reactions:",
                "  Ag-109   -> Ag-110m    (249.76 d)",
                "        (theta: 4.81E-01, sigma0: 9.100E+01 b, Q0: 1.240E+01, Er: 42.0 eV)",
                "",
                "Mo",
                "  M: 95.95",
                "  Reactions:",
                "  Mo-98   -> Mo-99    (66.0 h)",
                "        (theta: 2.41E-01, sigma0: 1.300E-01 b, Q0: 2.240E+00, Er: 132.0 eV)",
                "      Mo-99   -> Tc-99m    (6.01 h)",
                "",
                "U",
                "  M: 238.03",
                "  Reactions:",
                "  U-238   -> U-239    (23.47 m)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "IRI_MB2.TXT").write_text(
        "\n".join(
            [
                "Au-198 ( 2.69 d)",
                "Peaks:",
                " 411.80 keV:   95.58    *",
                " 675.90 keV:    0.806",
                "",
                "Co-60 ( 5.27 y)",
                "Peaks:",
                "1173.20 keV:   99.85",
                "1332.50 keV:   99.98    *",
                "",
                "Ag-110m (249.76 d)",
                "Peaks:",
                "657.80 keV:   94.60    *",
                "884.70 keV:   72.60",
                "",
                "Tc-99m ( 6.01 h)",
                "Peaks:",
                "140.50 keV:   89.00    *",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "IRI_MB3.TXT").write_text(
        "\n".join(
            [
                "140.50  Tc-99m   6.01 h  89.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.LB1").write_bytes(b"opaque lb1")
    (library_dir / "k0-2023.LB2").write_bytes(b"opaque lb2")
    return root


def test_import_kayzero_k0_library_from_directory_reports_unresolved_fields(tmp_path):
    source_root = _write_sample_kayzero_tree(tmp_path)

    result = import_kayzero_k0_library(source_root)

    assert result.library.library_id == "fluxforge.k0.kayzero.2023"
    assert result.library.records["Au-198"].target_isotope == "Au-197"
    assert abs(result.library.records["Au-198"].half_life_s - 232848.0) < 1.0
    assert result.library.records["Co-60"].gamma_intensity > 0.0
    assert result.library.records["Ag-110m"].sigma_0_barn == 91.0
    assert result.library.records["Ag-110m"].I0_barn == 91.0 * 12.4
    assert result.library.records["Ag-110m"].target_isotope == "Ag-109"
    assert result.library.records["Ag-110m"].gamma_intensity == 0.946
    assert result.library.records["Tc-99m"].target_isotope == "Mo-98"
    assert result.library.records["Tc-99m"].element == "Mo"
    assert result.library.records["Tc-99m"].gamma_intensity == 0.89
    assert result.library.records["U-239"].target_isotope == "U-238"
    assert result.library.records["U-239"].gamma_intensity == 0.492
    assert result.library.records["U-239"].sigma_0_barn > 0.0
    assert result.library.records["U-239"].I0_barn > result.library.records["U-239"].sigma_0_barn
    assert result.report["summary"]["record_count"] == 5
    assert result.report["summary"]["opaque_binary_files_present"] is True
    assert result.report["summary"]["iri_text_files_present"] is True
    unresolved = {row["product_isotope"]: row for row in result.report["unresolved_records"]}
    assert "Ag-110m" not in unresolved
    assert "Tc-99m" not in unresolved
    assert "U-239" not in unresolved


def test_import_kayzero_k0_library_from_zip(tmp_path):
    source_root = _write_sample_kayzero_tree(tmp_path / "tree")
    archive_path = tmp_path / "kayzero.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for child in source_root.rglob("*"):
            if child.is_file():
                archive.write(child, child.relative_to(source_root))

    result = import_kayzero_k0_library(archive_path)

    assert result.library.version == "2023"
    assert result.report["source_files"]["uk0"].endswith("k0-2023.uk0")
    assert result.library.records["Co-60"].k0_Au == 1.320


def test_imported_kayzero_library_can_drive_k0_analysis(tmp_path):
    source_root = _write_sample_kayzero_tree(tmp_path / "analysis_tree")
    result = import_kayzero_k0_library(source_root)

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
        standard_library=result.library,
    )

    assert bundle["summary"]["element_count"] == 1
    assert bundle["element_results"][0]["element"] == "Co"
    assert bundle["libraries"]["standard_k0_library"]["library_id"] == result.library.library_id
