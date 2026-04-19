from __future__ import annotations

import json
import sys
from argparse import Namespace
from types import SimpleNamespace
import types
from pathlib import Path

import numpy as np
import pytest

from fluxforge.cli import app
from fluxforge.io.spe import GammaSpectrum
from tests._phase6_real_data import (
    DEFAULT_PHASE6_SAMPLE_ID,
    RAFM_UNFOLD_MLEM_PATH,
    load_phase6_real_activity_review_payload,
    load_phase6_real_optimization_grids,
    write_phase6_real_second_irradiation_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _dummy_spectrum() -> GammaSpectrum:
    return GammaSpectrum(
        counts=np.array([0.0, 10.0, 0.0], dtype=float),
        channels=np.array([0, 1, 2], dtype=float),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [0.0, 1.0, 0.0]},
        spectrum_id="dummy",
    )


def _write_phase6_real_activity_review(tmp_path: Path) -> Path:
    activity_review_path = tmp_path / f"{DEFAULT_PHASE6_SAMPLE_ID}_activity_review.json"
    activity_review_path.write_text(
        json.dumps(load_phase6_real_activity_review_payload(), indent=2),
        encoding="utf-8",
    )
    return activity_review_path


def test_build_parser_and_reactions_json(capsys):
    parser = app.build_parser()
    args = parser.parse_args(["reactions", "--format", "json", "--category", "thermal"])
    assert args.command == "reactions"
    args.func(args)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "thermal" in payload


def test_build_parser_gui_dry_run(capsys, tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(["gui", "--project-dir", str(tmp_path), "--dry-run"])
    assert args.command == "gui"
    args.func(args)
    out = capsys.readouterr().out
    assert "GUI dry run" in out


def test_build_parser_phase6_ldrd_worked_example(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "phase6-ldrd-worked-example",
            "--sample-id",
            DEFAULT_PHASE6_SAMPLE_ID,
            "--output-root",
            str(tmp_path / "phase6_ldrd"),
        ]
    )
    assert args.command == "phase6-ldrd-worked-example"
    assert args.sample_id == DEFAULT_PHASE6_SAMPLE_ID
    assert args.output_root == tmp_path / "phase6_ldrd"


def test_cmd_phase6_ldrd_worked_example_invokes_workflow(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_run_phase6_ldrd_worked_example(*, sample_id, output_root):
        called["sample_id"] = sample_id
        called["output_root"] = Path(output_root)
        summary = Path(output_root) / "WORKED_EXAMPLE_SUMMARY.md"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("# ok\n", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        app,
        "run_phase6_ldrd_worked_example",
        fake_run_phase6_ldrd_worked_example,
    )

    output_root = tmp_path / "phase6_cli"
    app.cmd_phase6_ldrd_worked_example(
        Namespace(sample_id=DEFAULT_PHASE6_SAMPLE_ID, output_root=output_root)
    )

    out = capsys.readouterr().out
    assert "Wrote Phase 6 LDRD worked example artifacts" in out
    assert "Summary:" in out
    assert called["sample_id"] == DEFAULT_PHASE6_SAMPLE_ID
    assert called["output_root"] == output_root


def test_build_parser_k0_commands(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "k0-normalize",
            "--peaks-file",
            str(tmp_path / "peaks.json"),
            "--output",
            str(tmp_path / "observations.json"),
        ]
    )
    assert args.command == "k0-normalize"
    assert args.output.name == "observations.json"
    aggregate_args = parser.parse_args(
        [
            "k0-aggregate",
            "--analysis-files",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
        ]
    )
    assert aggregate_args.command == "k0-aggregate"
    import_args = parser.parse_args(
        [
            "k0-import-kayzero",
            "--input",
            str(tmp_path / "kayzero"),
        ]
    )
    assert import_args.command == "k0-import-kayzero"


def test_cmd_gui_launches_module(monkeypatch, tmp_path):
    called = {}

    modern_module = types.ModuleType("fluxforge.gui.app")
    qt_module = types.ModuleType("fluxforge.gui.qt_compat")

    def fake_launch_gui(project_dir):
        called["project_dir"] = Path(project_dir)

    modern_module.launch_modern_gui = fake_launch_gui
    qt_module.QT_AVAILABLE = True
    monkeypatch.setitem(sys.modules, "fluxforge.gui.app", modern_module)
    monkeypatch.setitem(sys.modules, "fluxforge.gui.qt_compat", qt_module)

    app.cmd_gui(Namespace(project_dir=tmp_path, dry_run=False))
    assert called["project_dir"] == tmp_path


def test_cmd_gui_falls_back_to_legacy_when_qt_is_unavailable(monkeypatch, tmp_path):
    called = {}

    modern_module = types.ModuleType("fluxforge.gui.app")
    qt_module = types.ModuleType("fluxforge.gui.qt_compat")
    legacy_module = types.ModuleType("fluxforge_gui.app")

    def fake_modern_launch(project_dir):
        called["modern_project_dir"] = Path(project_dir)

    def fake_legacy_launch(project_dir):
        called["legacy_project_dir"] = Path(project_dir)

    modern_module.launch_modern_gui = fake_modern_launch
    qt_module.QT_AVAILABLE = False
    legacy_module.launch_gui = fake_legacy_launch

    monkeypatch.setitem(sys.modules, "fluxforge.gui.app", modern_module)
    monkeypatch.setitem(sys.modules, "fluxforge.gui.qt_compat", qt_module)
    monkeypatch.setitem(sys.modules, "fluxforge_gui.app", legacy_module)

    app.cmd_gui(Namespace(project_dir=tmp_path, dry_run=False))

    assert "modern_project_dir" not in called
    assert called["legacy_project_dir"] == tmp_path


def test_build_parser_plots_dry_run(capsys, tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        ["plots", "--example", "--dry-run", "--output-dir", str(tmp_path)]
    )
    assert args.command == "plots"
    args.func(args)
    out = capsys.readouterr().out
    assert "Plots dry run" in out


def test_build_parser_spectrum_plot(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "spectrum-plot",
            "--input",
            str(tmp_path / "sample.ASC"),
            "--output",
            str(tmp_path / "sample.png"),
            "--background-subtracted",
            "--feature-isotope",
            "Co-60",
            "--feature-line-keV",
            "1332.5",
            "--save-feature-report",
            str(tmp_path / "features.json"),
        ]
    )
    assert args.command == "spectrum-plot"
    assert args.background_subtracted is True
    assert args.feature_isotope == "Co-60"
    assert args.feature_line_keV == pytest.approx(1332.5)


def test_build_parser_roi_commands(tmp_path):
    parser = app.build_parser()
    roi_args = parser.parse_args(
        [
            "roi-analyze",
            "--input",
            str(tmp_path / "sample.ASC"),
            "--left-keV",
            "200.0",
            "--right-keV",
            "220.0",
        ]
    )
    assert roi_args.command == "roi-analyze"
    assert roi_args.peak_search_method == "mariscotti"

    stats_args = parser.parse_args(
        [
            "roi-statistics",
            "--inputs",
            str(tmp_path / "a.ASC"),
            str(tmp_path / "b.ASC"),
            "--left-channel",
            "10",
            "--right-channel",
            "20",
        ]
    )
    assert stats_args.command == "roi-statistics"
    assert len(stats_args.inputs) == 2


def test_build_parser_file_query_and_batch_compare_commands(tmp_path):
    parser = app.build_parser()
    file_args = parser.parse_args(
        [
            "file-query",
            "--root",
            str(tmp_path),
            "--contains",
            "sample",
            "--output",
            str(tmp_path / "file_query.json"),
        ]
    )
    assert file_args.command == "file-query"
    assert file_args.output.name == "file_query.json"

    compare_args = parser.parse_args(
        [
            "batch-compare",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--candidate",
            str(tmp_path / "candidate.json"),
            "--keys",
            "sample_id,reaction_id",
        ]
    )
    assert compare_args.command == "batch-compare"
    assert compare_args.keys == "sample_id,reaction_id"

    parity_args = parser.parse_args(
        [
            "parity-check",
            "--reference-root",
            str(tmp_path / "reference"),
            "--activation-root",
            str(tmp_path / "activation"),
            "--scope",
            "workflow",
        ]
    )
    assert parity_args.command == "parity-check"
    assert parity_args.scope == "workflow"

    gui_acceptance_args = parser.parse_args(
        [
            "gui-acceptance-check",
            "--checklist",
            str(tmp_path / "checklist.md"),
            "--artifact-dir",
            str(tmp_path / "phase327_probe"),
        ]
    )
    assert gui_acceptance_args.command == "gui-acceptance-check"
    assert gui_acceptance_args.checklist.name == "checklist.md"


def test_cmd_file_query_writes_rows(tmp_path):
    (tmp_path / "sample_a.json").write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "sample_b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "ignore.log").write_text("ignored", encoding="utf-8")

    output = tmp_path / "query.json"
    app.cmd_file_query(
        Namespace(
            root=tmp_path,
            patterns="**/*",
            contains="sample",
            suffixes=".json,.csv",
            min_size_bytes=0,
            max_size_bytes=None,
            modified_after=None,
            limit=100,
            format="json",
            output=output,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.file_query.v1"
    assert payload["result_count"] == 2
    assert {row["path"] for row in payload["rows"]} == {
        "sample_a.json",
        "sample_b.csv",
    }


def test_cmd_batch_compare_computes_summary(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "sample_id": "s1",
                    "reaction_id": "r1",
                    "rate": 10.0,
                    "uncertainty": 1.0,
                },
                {
                    "sample_id": "s1",
                    "reaction_id": "r2",
                    "rate": 20.0,
                    "uncertainty": 2.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                {
                    "sample_id": "s1",
                    "reaction_id": "r1",
                    "rate": 12.0,
                    "uncertainty": 1.5,
                },
                {
                    "sample_id": "s1",
                    "reaction_id": "r2",
                    "rate": 18.0,
                    "uncertainty": 1.8,
                },
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "compare.json"
    app.cmd_batch_compare(
        Namespace(
            baseline=baseline,
            candidate=candidate,
            keys="sample_id,reaction_id",
            numeric_fields="rate,uncertainty",
            max_rows=100,
            output=output,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.batch_compare.v1"
    assert payload["summary"]["matched_rows"] == 2
    assert payload["summary"]["baseline_only_rows"] == 0
    assert payload["summary"]["candidate_only_rows"] == 0
    assert "rate" in payload["summary"]["field_stats"]


def test_cmd_parity_check_writes_summary_payload(tmp_path):
    output = tmp_path / "parity.json"
    app.cmd_parity_check(
        Namespace(
            reference_root=Path("tests/spectra/reference_parity"),
            activation_root=Path("tests/activation_inventory/fixtures"),
            scope="all",
            fixture_id=None,
            include_activation=True,
            output=output,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.reference_parity.run.v1"
    assert payload["summary"]["total"] >= 5


def test_cmd_gui_acceptance_checklist_writes_readiness_payload(tmp_path):
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "\n".join(
            [
                "# Checklist",
                "- [x] Qt tests passed",
                "- [x] Probes generated",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phase327 = tmp_path / "phase327_probe"
    phase327.mkdir(parents=True, exist_ok=True)
    (phase327 / "index.html").write_text("<html></html>\n", encoding="utf-8")

    output = tmp_path / "gui_acceptance.json"
    app.cmd_gui_acceptance_checklist(
        Namespace(
            checklist=checklist,
            artifact_dir=[phase327],
            output=output,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.gui_acceptance_check.v1"
    assert payload["checklist"]["exists"] is True
    assert payload["checklist"]["unchecked_items"] == []
    assert payload["ready"] is True


def test_build_parser_activity_review_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "activity-review",
            "--peaks-file",
            str(tmp_path / "peaks.json"),
            "--cooling-time-s",
            "3600",
            "--source-id",
            "custom_gamma_file",
            "--custom-gamma-path",
            str(tmp_path / "gamma.json"),
            "--efficiency-polynomial=-2.0,-0.1",
        ]
    )
    assert args.command == "activity-review"
    assert args.source_id == "custom_gamma_file"
    assert args.cooling_time_s == pytest.approx(3600.0)


def test_build_parser_activity_review_supports_reaction_export_options(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "activity-review",
            "--peaks-file",
            str(tmp_path / "peaks.json"),
            "--eoi-export",
            "reaction-rates",
            "--irradiation-time-s",
            "7200",
            "--reaction-rate-csv-output",
            str(tmp_path / "reaction_rates.csv"),
        ]
    )
    assert args.command == "activity-review"
    assert args.eoi_export == "reaction-rates"
    assert args.irradiation_time_s == pytest.approx(7200.0)


def test_build_parser_inventory_review_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "inventory-review",
            "--activity-review-file",
            str(tmp_path / "activity_review.json"),
            "--time-origin",
            "count_start",
            "--observable",
            "dose",
            "--time-points-s",
            "0,3600,7200",
        ]
    )
    assert args.command == "inventory-review"
    assert args.time_origin == "count_start"
    assert args.observable == "dose"


def test_build_parser_second_irradiation_plan_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "second-irradiation-plan",
            "--inventory-file",
            str(tmp_path / "inventory_seed.json"),
            "--schedule-file",
            str(tmp_path / "schedule.json"),
            "--candidates-file",
            str(tmp_path / "candidates.json"),
        ]
    )
    assert args.command == "second-irradiation-plan"
    assert args.output.name == "second_irradiation_plan.json"


def test_build_parser_ffexp_export_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "ffexp-export",
            "--activity-review-file",
            str(tmp_path / "activity_review.json"),
            "--optimization-file",
            str(tmp_path / "optimization.json"),
            "--plot-paths",
            "inventory.png,pareto.png",
        ]
    )
    assert args.command == "ffexp-export"
    assert args.output.name == "benchmark_bundle.ffexp"
    assert args.plot_paths == "inventory.png,pareto.png"


def test_build_parser_isotope_priority_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "isotope-priority",
            "--activity-review-file",
            str(tmp_path / "activity_review.json"),
            "--isotopes-of-interest",
            "Mo-99,Sc-46",
            "--top-n",
            "5",
            "--weight-dose",
            "0.2",
        ]
    )
    assert args.command == "isotope-priority"
    assert args.isotopes_of_interest == "Mo-99,Sc-46"
    assert args.top_n == 5
    assert args.weight_dose == pytest.approx(0.2)


def test_build_parser_masking_review_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "masking-review",
            "--activity-review-file",
            str(tmp_path / "activity_review.json"),
            "--isotopes-of-interest",
            "Mo-99,Sc-46",
            "--energy-window-keV",
            "2.5",
            "--top-n",
            "20",
        ]
    )
    assert args.command == "masking-review"
    assert args.activity_review_file.name == "activity_review.json"
    assert args.isotopes_of_interest == "Mo-99,Sc-46"
    assert args.energy_window_keV == pytest.approx(2.5)
    assert args.top_n == 20


def test_build_parser_optimization_sweep_command(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--input",
            str(tmp_path / "optimization_candidates.json"),
            "--objective",
            "di-fom",
            "--isotopes-of-interest",
            "Mo-99,Tc-99m",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.output.name == "optimization_sweep.json"
    assert args.isotopes_of_interest == "Mo-99,Tc-99m"


def test_build_parser_optimization_sweep_supports_activity_review_generation(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--activity-review-file",
            str(tmp_path / "activity_review.json"),
            "--objective",
            "di-fom",
            "--neutron-spectrum-file",
            str(tmp_path / "neutron_flux.csv"),
            "--irradiation-grid-s",
            "1800,3600",
            "--cooldown-grid-s",
            "0,3600",
            "--count-grid-s",
            "300,900",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.input is None
    assert args.activity_review_file.name == "activity_review.json"
    assert args.neutron_spectrum_file.name == "neutron_flux.csv"


def test_build_parser_optimization_sweep_supports_fim_options(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--input",
            str(tmp_path / "optimization_candidates.json"),
            "--objective",
            "fim-c",
            "--target-nuclide",
            "Mo-99",
            "--nuisance-variance-fraction",
            "0.05",
            "--fim-regularization",
            "1e-5",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.objective == "fim-c"
    assert args.target_nuclide == "Mo-99"
    assert args.nuisance_variance_fraction == pytest.approx(0.05)
    assert args.fim_regularization == pytest.approx(1.0e-5)


def test_build_parser_optimization_sweep_supports_mwdcs_options(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--input",
            str(tmp_path / "optimization_candidates.json"),
            "--objective",
            "mwdcs",
            "--mwdcs-window-offsets-s",
            "0,1800,7200",
            "--mwdcs-window-count-time-s",
            "1200",
            "--mwdcs-full-spectrum-mode",
            "--mwdcs-overlap-penalty",
            "0.2",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.objective == "mwdcs"
    assert args.mwdcs_window_offsets_s == "0,1800,7200"
    assert args.mwdcs_window_count_time_s == pytest.approx(1200.0)
    assert args.mwdcs_full_spectrum_mode is True
    assert args.mwdcs_overlap_penalty == pytest.approx(0.2)


def test_build_parser_optimization_sweep_supports_bassd_options(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--input",
            str(tmp_path / "optimization_candidates.json"),
            "--objective",
            "bass-d",
            "--enable-advanced-objectives",
            "--bassd-dose-weight",
            "0.03",
            "--bassd-exploration-temperature",
            "0.2",
            "--bassd-seed",
            "19",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.objective == "bass-d"
    assert args.enable_advanced_objectives is True
    assert args.bassd_dose_weight == pytest.approx(0.03)
    assert args.bassd_exploration_temperature == pytest.approx(0.2)
    assert args.bassd_seed == 19


def test_build_parser_optimization_sweep_supports_stbdmr_options(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "optimization-sweep",
            "--input",
            str(tmp_path / "optimization_candidates.json"),
            "--objective",
            "stbd-mr",
            "--enable-advanced-objectives",
            "--stbdmr-window-offsets-s",
            "0,1800,7200",
            "--stbdmr-window-count-time-s",
            "1200",
            "--stbdmr-masking-regularization",
            "0.2",
            "--stbdmr-differentiable-graph",
            "--stbdmr-graph-temperature",
            "1.5",
        ]
    )
    assert args.command == "optimization-sweep"
    assert args.objective == "stbd-mr"
    assert args.enable_advanced_objectives is True
    assert args.stbdmr_window_offsets_s == "0,1800,7200"
    assert args.stbdmr_window_count_time_s == pytest.approx(1200.0)
    assert args.stbdmr_masking_regularization == pytest.approx(0.2)
    assert args.stbdmr_differentiable_graph is True
    assert args.stbdmr_graph_temperature == pytest.approx(1.5)


def test_build_parser_library_registry_commands(tmp_path):
    parser = app.build_parser()
    list_args = parser.parse_args(["library-list", "--json"])
    assert list_args.command == "library-list"

    register_args = parser.parse_args(
        [
            "library-register",
            "--alias",
            "Lab Ref",
            "--locator",
            str(tmp_path / "gamma.csv"),
        ]
    )
    assert register_args.command == "library-register"

    remove_args = parser.parse_args(
        [
            "library-remove",
            "--source-id",
            "user_gamma_lab_ref",
        ]
    )
    assert remove_args.command == "library-remove"


def test_activity_review_source_choices_only_include_gamma_libraries():
    source_ids = set(app._activity_review_source_choices())

    assert "nasa_common_lab_sources" in source_ids
    assert "nasa_capture_capgam" not in source_ids
    assert "nasa_capture_iaea" not in source_ids
    assert "radioactivedecay_icrp107_kayzero_2023" not in source_ids


def test_build_parser_peak_search_supports_all_modern_methods(tmp_path):
    parser = app.build_parser()
    args = parser.parse_args(
        [
            "peaks",
            "--spectrum-file",
            str(tmp_path / "sample.json"),
            "--method",
            "consensus",
        ]
    )
    assert args.method == "consensus"

    roi_args = parser.parse_args(
        [
            "roi-analyze",
            "--input",
            str(tmp_path / "sample.json"),
            "--left-keV",
            "100.0",
            "--right-keV",
            "200.0",
            "--peak-search-method",
            "wavelet",
        ]
    )
    assert roi_args.peak_search_method == "wavelet"


def test_cmd_activity_review_writes_json_csv_and_plot_artifacts(monkeypatch, tmp_path):
    gamma_path = tmp_path / "gamma.json"
    gamma_path.write_text(
        json.dumps(
            [
                {
                    "nuclide": "Co60",
                    "energy_keV": 1173.228,
                    "intensity": 0.999,
                    "intensity_unc": 0.004,
                    "half_life_s": 5.2714 * 365.25 * 24.0 * 3600.0,
                },
                {
                    "nuclide": "Co60",
                    "energy_keV": 1332.492,
                    "intensity": 0.998,
                    "intensity_unc": 0.004,
                    "half_life_s": 5.2714 * 365.25 * 24.0 * 3600.0,
                },
                {
                    "nuclide": "Sc46",
                    "energy_keV": 889.277,
                    "intensity": 0.999,
                    "intensity_unc": 0.006,
                    "half_life_s": 83.79 * 24.0 * 3600.0,
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    peaks_path = tmp_path / "peaks.json"
    peaks_path.write_text(
        json.dumps(
            {
                "spectrum_id": "demo-activation",
                "live_time_s": 120.0,
                "peaks": [
                    {
                        "peak_id": "peak-1",
                        "channel": 100,
                        "energy_keV": 1173.23,
                        "area": 12000.0,
                        "report_isotope": "Co60",
                    },
                    {
                        "peak_id": "peak-2",
                        "channel": 120,
                        "energy_keV": 1332.49,
                        "area": 10000.0,
                        "report_isotope": "Co60",
                    },
                    {
                        "peak_id": "peak-3",
                        "channel": 80,
                        "energy_keV": 889.28,
                        "area": 7000.0,
                        "report_isotope": "Sc46",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_plot_decay_curves(*_args, save_path=None, **_kwargs):
        Path(save_path).write_text("plot", encoding="utf-8")
        return object(), object()

    monkeypatch.setattr(app, "plot_decay_curves", fake_plot_decay_curves)

    output = tmp_path / "activity_review.json"
    app.cmd_activity_review(
        Namespace(
            peaks_file=peaks_path,
            output=output,
            live_time_s=None,
            cooling_time_s=7200.0,
            dead_time_fraction=0.0,
            energy_tolerance_keV=1.0,
            source_id="custom_gamma_file",
            custom_gamma_path=gamma_path,
            efficiency=0.2,
            efficiency_polynomial=None,
            efficiency_uncertainty=0.04,
            sample_mass_g=2.5,
            isotope_csv_output=None,
            line_csv_output=None,
            decay_plot=None,
            bateman_plot=None,
            validate=False,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.activity_review.v1"
    assert payload["spectrum_id"] == "demo-activation"
    assert len(payload["isotope_summaries"]) == 2
    assert (tmp_path / "activity_review_isotopes.csv").exists()
    assert (tmp_path / "activity_review_lines.csv").exists()
    assert (tmp_path / "activity_review_decay.png").exists()
    assert (tmp_path / "activity_review_bateman.png").exists()
    assert "irradiation_time_activity_Bq" in (tmp_path / "activity_review_isotopes.csv").read_text(
        encoding="utf-8"
    )


def test_cmd_activity_review_can_export_reaction_rate_csv(monkeypatch, tmp_path):
    gamma_path = tmp_path / "gamma.json"
    gamma_path.write_text(
        json.dumps(
            [
                {
                    "nuclide": "Co60",
                    "energy_keV": 1173.228,
                    "intensity": 0.999,
                    "half_life_s": 5.2714 * 365.25 * 24.0 * 3600.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    peaks_path = tmp_path / "peaks.json"
    peaks_path.write_text(
        json.dumps(
            {
                "spectrum_id": "demo-activation",
                "live_time_s": 120.0,
                "peaks": [
                    {
                        "peak_id": "peak-1",
                        "channel": 100,
                        "energy_keV": 1173.23,
                        "area": 12000.0,
                        "report_isotope": "Co60",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_plot_decay_curves(*_args, save_path=None, **_kwargs):
        Path(save_path).write_text("plot", encoding="utf-8")
        return object(), object()

    monkeypatch.setattr(app, "plot_decay_curves", fake_plot_decay_curves)

    output = tmp_path / "activity_review.json"
    app.cmd_activity_review(
        Namespace(
            peaks_file=peaks_path,
            output=output,
            live_time_s=None,
            cooling_time_s=3600.0,
            dead_time_fraction=0.0,
            energy_tolerance_keV=1.0,
            source_id="custom_gamma_file",
            custom_gamma_path=gamma_path,
            efficiency=0.2,
            efficiency_polynomial=None,
            efficiency_uncertainty=0.04,
            sample_mass_g=2.5,
            eoi_export="reaction-rates",
            irradiation_time_s=3600.0,
            irradiation_segments_file=None,
            isotope_csv_output=None,
            line_csv_output=None,
            reaction_rate_csv_output=None,
            reaction_rates_output=None,
            decay_plot=None,
            bateman_plot=None,
            validate=False,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["eoi_export_mode"] == "reaction-rates"
    assert "reaction_rate_rows" in payload
    assert payload["reaction_rate_rows"]
    reaction_csv = tmp_path / "activity_review_reaction_rates.csv"
    assert reaction_csv.exists()
    assert "reaction_rate_s" in reaction_csv.read_text(encoding="utf-8")


def test_cmd_inventory_review_writes_json_csv_and_plot_artifacts(monkeypatch, tmp_path):
    activity_review_path = tmp_path / "activity_review.json"
    activity_review_path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.activity_review.v1",
                "spectrum_id": "mo99-demo",
                "source_id": "nasa_common_lab_sources",
                "custom_gamma_path": None,
                "live_time_s": 300.0,
                "cooling_time_s": 7200.0,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_plot_decay_curves(*_args, save_path=None, **_kwargs):
        Path(save_path).write_text("plot", encoding="utf-8")
        return object(), object()

    monkeypatch.setattr(app, "plot_decay_curves", fake_plot_decay_curves)

    output = tmp_path / "inventory_review.json"
    app.cmd_inventory_review(
        Namespace(
            activity_review_file=activity_review_path,
            output=output,
            decay_source_id="radioactivedecay_icrp107_kayzero_2023",
            time_origin="count_start",
            time_points_s="0,3600,7200",
            time_start_s=0.0,
            time_stop_s=86400.0,
            time_count=25,
            observable="activity",
            distance_cm=25.0,
            top_n=5,
            timeseries_csv_output=None,
            eoi_csv_output=None,
            count_start_csv_output=None,
            count_end_csv_output=None,
            plot_output=None,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.inventory_time_evolution.v1"
    assert payload["sample_id"] == "mo99-demo"
    assert payload["time_origin"] == "count_start"
    assert len(payload["time_series_rows"]) >= 3
    assert (tmp_path / "inventory_review_timeseries.csv").exists()
    assert (tmp_path / "inventory_review_activities_at_irradiation.csv").exists()
    assert (tmp_path / "inventory_review_activities_at_count_start.csv").exists()
    assert (tmp_path / "inventory_review_activities_at_count_end.csv").exists()
    assert (tmp_path / "inventory_review_activity.png").exists()
    assert "dose_rate_uSv_h" in (tmp_path / "inventory_review_timeseries.csv").read_text(
        encoding="utf-8"
    )


def test_cmd_second_irradiation_plan_writes_json_and_csv_outputs(tmp_path):
    phase6_inputs = write_phase6_real_second_irradiation_inputs(tmp_path)
    inventory_path = phase6_inputs["inventory"]
    schedule_path = phase6_inputs["schedule"]
    candidates_path = phase6_inputs["candidates"]

    output_path = tmp_path / "second_irradiation_plan.json"
    csv_path = tmp_path / "second_irradiation_selected.csv"
    app.cmd_second_irradiation_plan(
        Namespace(
            inventory_file=inventory_path,
            schedule_file=schedule_path,
            candidates_file=candidates_path,
            output=output_path,
            csv_output=csv_path,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.second_irradiation_plan.v1"
    assert payload["selected_candidate"]["label"].startswith(f"{DEFAULT_PHASE6_SAMPLE_ID}_window_")
    assert payload["selected_inventory_rows"]
    assert csv_path.exists()
    assert "weighted_activity" in csv_path.read_text(encoding="utf-8")


def test_cmd_isotope_priority_writes_json_and_csv_outputs(tmp_path):
    activity_review_path = tmp_path / "activity_review.json"
    activity_review_path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.activity_review.v1",
                "spectrum_id": "priority-demo",
                "isotope_summaries": [
                    {
                        "nuclide": "Mo-99",
                        "line_count": 2,
                        "total_net_counts": 8200.0,
                        "irradiation_time_activity_Bq": 1200.0,
                        "irradiation_time_activity_unc_Bq": 80.0,
                        "dose_rate_uSv_h": 30.0,
                    },
                    {
                        "nuclide": "Co-60",
                        "line_count": 1,
                        "total_net_counts": 10000.0,
                        "irradiation_time_activity_Bq": 800.0,
                        "irradiation_time_activity_unc_Bq": 200.0,
                        "dose_rate_uSv_h": 40.0,
                    },
                    {
                        "nuclide": "Sc-46",
                        "line_count": 3,
                        "total_net_counts": 6000.0,
                        "irradiation_time_activity_Bq": 600.0,
                        "irradiation_time_activity_unc_Bq": 50.0,
                        "dose_rate_uSv_h": 20.0,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "isotope_priority.json"
    csv_path = tmp_path / "isotope_priority.csv"
    app.cmd_isotope_priority(
        Namespace(
            activity_review_file=activity_review_path,
            output=output_path,
            csv_output=csv_path,
            isotopes_of_interest="Mo-99,Sc-46",
            top_n=1,
            weight_activity=0.5,
            weight_detectability=0.2,
            weight_confidence=0.2,
            weight_line_support=0.1,
            weight_dose=0.0,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.isotope_priority.v1"
    assert payload["isotopes_of_interest"] == ["Mo-99", "Sc-46"]
    assert len(payload["ranked_isotopes"]) == 1
    assert payload["ranked_isotopes"][0]["nuclide"] == "Mo-99"
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "priority_score" in csv_text
    assert "nuclide" in csv_text


def test_cmd_masking_review_writes_json_and_csv_outputs(tmp_path):
    activity_review_path = tmp_path / "activity_review.json"
    activity_review_path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.activity_review.v1",
                "line_results": [
                    {
                        "nuclide": "Mo-99",
                        "matched_line_energy_keV": 140.5,
                        "net_counts": 7000.0,
                        "net_counts_uncertainty": 90.0,
                    },
                    {
                        "nuclide": "Sc-46",
                        "matched_line_energy_keV": 140.6,
                        "net_counts": 12000.0,
                        "net_counts_uncertainty": 120.0,
                    },
                    {
                        "nuclide": "Co-60",
                        "matched_line_energy_keV": 1332.5,
                        "net_counts": 4000.0,
                        "net_counts_uncertainty": 80.0,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "masking_review.json"
    csv_path = tmp_path / "masking_lines.csv"
    isotope_csv_path = tmp_path / "masking_isotopes.csv"
    recommendation_csv_path = tmp_path / "masking_recommendations.csv"
    app.cmd_masking_review(
        Namespace(
            activity_review_file=activity_review_path,
            output=output_path,
            isotopes_of_interest="Mo-99,Sc-46",
            energy_window_keV=3.0,
            top_n=10,
            csv_output=csv_path,
            isotope_csv_output=isotope_csv_path,
            recommendation_csv_output=recommendation_csv_path,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.masking_review.v1"
    assert payload["isotopes_of_interest"] == ["Mo-99", "Sc-46"]
    assert payload["line_masking_results"]
    assert payload["masking_isotope_ranking"]
    assert payload["alternate_line_recommendations"]
    assert csv_path.exists()
    assert "masking_score" in csv_path.read_text(encoding="utf-8")
    assert isotope_csv_path.exists()
    assert "masking_nuclide" in isotope_csv_path.read_text(encoding="utf-8")
    assert recommendation_csv_path.exists()
    assert "guidance" in recommendation_csv_path.read_text(encoding="utf-8")


def test_cmd_optimization_sweep_writes_json_and_csv_outputs(tmp_path):
    input_payload = {
        "isotope_weights": {"Mo-99": 1.2, "Tc-99m": 0.8},
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 120.0,
                        "background_counts": 30.0,
                        "interference_counts": 10.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 60.0,
                        "background_counts": 25.0,
                        "interference_counts": 15.0,
                    },
                ],
            },
            {
                "label": "candidate_b",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 14400.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 65.0,
                        "background_counts": 20.0,
                        "interference_counts": 8.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 22.0,
                        "background_counts": 12.0,
                        "interference_counts": 6.0,
                    },
                ],
            },
        ],
    }
    input_path = tmp_path / "optimization_candidates.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep.json"
    csv_path = tmp_path / "optimization_sweep.csv"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="di-fom",
            csv_output=csv_path,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.optimization_sweep.difom.v1"
    assert payload["objective"] == "di-fom"
    assert len(payload["ranked_candidates"]) == 2
    assert payload["ranked_candidates"][0]["difom_score"] >= payload["ranked_candidates"][1]["difom_score"]
    assert csv_path.exists()
    assert "difom_score" in csv_path.read_text(encoding="utf-8")


def test_cmd_optimization_sweep_builds_candidates_from_activity_review(tmp_path):
    activity_review_payload = load_phase6_real_activity_review_payload()
    activity_review_path = _write_phase6_real_activity_review(tmp_path)
    grids = load_phase6_real_optimization_grids()
    isotopes_of_interest = ",".join(
        row["nuclide"]
        for row in activity_review_payload["isotope_summaries"][:3]
    )

    output_path = tmp_path / "optimization_sweep.json"
    generated_path = tmp_path / "generated_candidates.json"
    app.cmd_optimization_sweep(
        Namespace(
            input=None,
            activity_review_file=activity_review_path,
            output=output_path,
            objective="di-fom",
            csv_output=None,
            isotopes_of_interest=isotopes_of_interest,
            irradiation_grid_s=grids["irradiation_grid_s"],
            cooldown_grid_s=grids["cooldown_grid_s"],
            count_grid_s=grids["count_grid_s"],
            reference_irradiation_time_s=3600.0,
            unfold_file=RAFM_UNFOLD_MLEM_PATH,
            neutron_spectrum_file=None,
            flux_scale=1.0,
            reference_flux_integral=0.0,
            generated_candidates_output=generated_path,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
            enable_advanced_objectives=False,
            bassd_dose_weight=0.02,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
            stbdmr_window_offsets_s="0,7200,86400",
            stbdmr_window_count_time_s=900.0,
            stbdmr_masking_regularization=0.1,
            stbdmr_differentiable_graph=False,
            stbdmr_graph_temperature=2.0,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_source"] == "activity-review"
    assert payload["objective"] == "di-fom"
    assert payload["ranked_candidates"]
    assert payload["neutron_source"]["source"] == "unfold"
    assert "support_artifacts" in payload
    assert (tmp_path / "optimization_sweep_optimization_grid.csv").exists()
    assert (tmp_path / "optimization_sweep_recommended_schedules.csv").exists()
    assert (tmp_path / "optimization_sweep_dose_endpoints.csv").exists()
    assert (tmp_path / "optimization_sweep_masking_candidates.csv").exists()
    assert (tmp_path / "optimization_sweep_inventory_timeseries.csv").exists()
    assert (tmp_path / "optimization_sweep_activities_at_irradiation.csv").exists()
    assert generated_path.exists()


def test_cmd_optimization_sweep_filters_isotopes_of_interest(tmp_path):
    input_payload = {
        "isotope_weights": {"Mo-99": 2.0, "Co-60": 1.0},
        "candidates": [
            {
                "label": "mixed",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 90.0,
                        "background_counts": 10.0,
                    },
                    {
                        "nuclide": "Co-60",
                        "line_energy_keV": 1332.5,
                        "signal_counts": 80.0,
                        "background_counts": 10.0,
                    },
                ],
            },
            {
                "label": "co60_only",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Co-60",
                        "line_energy_keV": 1332.5,
                        "signal_counts": 150.0,
                        "background_counts": 10.0,
                    }
                ],
            },
        ],
    }
    input_path = tmp_path / "optimization_candidates_filtered.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep_filtered.json"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="di-fom",
            csv_output=None,
            isotopes_of_interest="Mo-99",
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective"] == "di-fom"
    assert payload["isotopes_of_interest"] == ["Mo-99"]
    assert payload["isotope_weights"] == {"Mo-99": 2.0}
    summary = payload["isotope_filter_summary"]
    assert summary["candidates_before"] == 2
    assert summary["candidates_after"] == 1
    assert summary["line_terms_before"] == 3
    assert summary["line_terms_after"] == 1
    assert [item["label"] for item in payload["ranked_candidates"]] == ["mixed"]


def test_cmd_optimization_sweep_writes_fim_outputs(tmp_path):
    input_payload = {
        "candidates": [
            {
                "label": "balanced",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 70.0,
                        "background_counts": 15.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 60.0,
                        "background_counts": 15.0,
                    },
                ],
            },
            {
                "label": "single_nuclide_dominant",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 120.0,
                        "background_counts": 20.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 5.0,
                        "background_counts": 12.0,
                    },
                ],
            },
        ]
    }
    input_path = tmp_path / "optimization_candidates_fim.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep_fim.json"
    csv_path = tmp_path / "optimization_sweep_fim.csv"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="fim-d",
            csv_output=csv_path,
            target_nuclide=None,
            nuisance_variance_fraction=0.05,
            fim_regularization=1.0e-6,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.optimization_sweep.fim.v1"
    assert payload["objective"] == "fim-d"
    assert len(payload["ranked_candidates"]) == 2
    assert "matrix_diagnostics" in payload["ranked_candidates"][0]
    assert csv_path.exists()
    assert "objective_score" in csv_path.read_text(encoding="utf-8")


def test_cmd_optimization_sweep_writes_mwdcs_outputs(tmp_path):
    input_payload = {
        "isotope_weights": {"Mo-99": 1.5},
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 100.0,
                        "background_counts": 10.0,
                        "half_life_s": 18000.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.6,
                        "signal_counts": 55.0,
                        "background_counts": 12.0,
                    },
                ],
            },
            {
                "label": "candidate_b",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 65.0,
                        "background_counts": 12.0,
                        "half_life_s": 18000.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.6,
                        "signal_counts": 20.0,
                        "background_counts": 12.0,
                    },
                ],
            },
        ],
    }
    input_path = tmp_path / "optimization_candidates_mwdcs.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep_mwdcs.json"
    csv_path = tmp_path / "optimization_sweep_mwdcs.csv"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="mwdcs",
            csv_output=csv_path,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,3600,14400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=True,
            mwdcs_overlap_penalty=0.1,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.optimization_sweep.mwdcs.v1"
    assert payload["objective"] == "mwdcs"
    assert len(payload["ranked_candidates"]) == 2
    assert "window_scores" in payload["ranked_candidates"][0]
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "objective_score" in csv_text
    assert "window_count" in csv_text


def test_cmd_optimization_sweep_bassd_requires_advanced_guard(tmp_path):
    input_payload = {
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 80.0,
                        "background_counts": 10.0,
                    }
                ],
            }
        ]
    }
    input_path = tmp_path / "optimization_candidates_bassd_guard.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="enable-advanced-objectives"):
        app.cmd_optimization_sweep(
            Namespace(
                input=input_path,
                output=tmp_path / "bassd_guard.json",
                objective="bass-d",
                csv_output=None,
                enable_advanced_objectives=False,
                bassd_dose_weight=0.02,
                bassd_exploration_temperature=0.0,
                bassd_seed=17,
                target_nuclide=None,
                nuisance_variance_fraction=0.0,
                fim_regularization=1.0e-6,
                mwdcs_window_offsets_s="0,7200,86400",
                mwdcs_window_count_time_s=900.0,
                mwdcs_full_spectrum_mode=False,
                mwdcs_overlap_penalty=0.0,
            )
        )


def test_cmd_optimization_sweep_writes_bassd_outputs(tmp_path):
    input_payload = {
        "isotope_weights": {"Mo-99": 1.2},
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "actions": [
                    {
                        "label": "a1",
                        "cooldown_time_s": 0.0,
                        "count_time_s": 900.0,
                        "lines": [
                            {
                                "nuclide": "Mo-99",
                                "line_energy_keV": 140.5,
                                "signal_counts": 100.0,
                                "background_counts": 10.0,
                                "dose_rate_uSv_h": 3.0,
                                "prior_variance_counts2": 120.0,
                            }
                        ],
                    }
                ],
            },
            {
                "label": "candidate_b",
                "irradiation_time_s": 3600.0,
                "actions": [
                    {
                        "label": "a1",
                        "cooldown_time_s": 0.0,
                        "count_time_s": 900.0,
                        "lines": [
                            {
                                "nuclide": "Mo-99",
                                "line_energy_keV": 140.5,
                                "signal_counts": 60.0,
                                "background_counts": 12.0,
                                "dose_rate_uSv_h": 12.0,
                                "prior_variance_counts2": 120.0,
                            }
                        ],
                    }
                ],
            },
        ],
    }
    input_path = tmp_path / "optimization_candidates_bassd.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep_bassd.json"
    csv_path = tmp_path / "optimization_sweep_bassd.csv"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="bass-d",
            csv_output=csv_path,
            enable_advanced_objectives=True,
            bassd_dose_weight=0.03,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.optimization_sweep.bassd.v1"
    assert payload["objective"] == "bass-d"
    assert payload["advanced_objective"] is True
    assert len(payload["ranked_candidates"]) == 2
    assert "action_scores" in payload["ranked_candidates"][0]
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "objective_score" in csv_text
    assert "action_count" in csv_text


def test_cmd_optimization_sweep_stbdmr_requires_advanced_guard(tmp_path):
    input_payload = {
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 80.0,
                        "background_counts": 10.0,
                    }
                ],
            }
        ]
    }
    input_path = tmp_path / "optimization_candidates_stbdmr_guard.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="enable-advanced-objectives"):
        app.cmd_optimization_sweep(
            Namespace(
                input=input_path,
                output=tmp_path / "stbdmr_guard.json",
                objective="stbd-mr",
                csv_output=None,
                enable_advanced_objectives=False,
                stbdmr_window_offsets_s="0,7200,86400",
                stbdmr_window_count_time_s=900.0,
                stbdmr_masking_regularization=0.1,
                stbdmr_differentiable_graph=False,
                stbdmr_graph_temperature=2.0,
                target_nuclide=None,
                nuisance_variance_fraction=0.0,
                fim_regularization=1.0e-6,
                mwdcs_window_offsets_s="0,7200,86400",
                mwdcs_window_count_time_s=900.0,
                mwdcs_full_spectrum_mode=False,
                mwdcs_overlap_penalty=0.0,
                bassd_dose_weight=0.02,
                bassd_exploration_temperature=0.0,
                bassd_seed=17,
            )
        )


def test_cmd_optimization_sweep_writes_stbdmr_outputs(tmp_path):
    input_payload = {
        "isotope_weights": {"Mo-99": 1.2},
        "candidates": [
            {
                "label": "candidate_a",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 100.0,
                        "background_counts": 10.0,
                        "continuum_counts": 4.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.7,
                        "signal_counts": 60.0,
                        "background_counts": 10.0,
                        "continuum_counts": 4.0,
                    },
                ],
            },
            {
                "label": "candidate_b",
                "irradiation_time_s": 3600.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 70.0,
                        "background_counts": 10.0,
                        "continuum_counts": 4.0,
                    },
                    {
                        "nuclide": "Co-60",
                        "line_energy_keV": 1332.5,
                        "signal_counts": 60.0,
                        "background_counts": 10.0,
                        "continuum_counts": 4.0,
                    },
                ],
            },
        ],
    }
    input_path = tmp_path / "optimization_candidates_stbdmr.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    output_path = tmp_path / "optimization_sweep_stbdmr.json"
    csv_path = tmp_path / "optimization_sweep_stbdmr.csv"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=output_path,
            objective="stbd-mr",
            csv_output=csv_path,
            enable_advanced_objectives=True,
            stbdmr_window_offsets_s="0,3600,21600",
            stbdmr_window_count_time_s=900.0,
            stbdmr_masking_regularization=0.2,
            stbdmr_differentiable_graph=True,
            stbdmr_graph_temperature=2.0,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
            bassd_dose_weight=0.02,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.optimization_sweep.stbdmr.v1"
    assert payload["objective"] == "stbd-mr"
    assert payload["advanced_objective"] is True
    assert len(payload["ranked_candidates"]) == 2
    assert "diagnostics" in payload["ranked_candidates"][0]
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "objective_score" in csv_text
    assert "graph_density" in csv_text


def test_cmd_optimization_sweep_compares_difom_and_fim_on_shared_fixture(tmp_path):
    payload = {
        "isotope_weights": {"Mo-99": 2.0, "Tc-99m": 0.2},
        "candidates": [
            {
                "label": "balanced",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 50.0,
                        "background_counts": 10.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 50.0,
                        "background_counts": 10.0,
                    },
                ],
            },
            {
                "label": "mo99_dominant",
                "irradiation_time_s": 3600.0,
                "cooldown_time_s": 7200.0,
                "count_time_s": 900.0,
                "lines": [
                    {
                        "nuclide": "Mo-99",
                        "line_energy_keV": 140.5,
                        "signal_counts": 90.0,
                        "background_counts": 10.0,
                    },
                    {
                        "nuclide": "Tc-99m",
                        "line_energy_keV": 140.5,
                        "signal_counts": 5.0,
                        "background_counts": 10.0,
                    },
                ],
            },
        ],
    }
    input_path = tmp_path / "shared_candidates.json"
    input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    difom_output = tmp_path / "difom.json"
    fim_output = tmp_path / "fim.json"
    mwdcs_output = tmp_path / "mwdcs.json"
    bassd_output = tmp_path / "bassd.json"
    stbdmr_output = tmp_path / "stbdmr.json"
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=difom_output,
            objective="di-fom",
            csv_output=None,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
        )
    )
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=fim_output,
            objective="fim-d",
            csv_output=None,
            target_nuclide=None,
            nuisance_variance_fraction=0.05,
            fim_regularization=1.0e-6,
        )
    )
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=mwdcs_output,
            objective="mwdcs",
            csv_output=None,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,3600,21600",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
        )
    )
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=bassd_output,
            objective="bass-d",
            csv_output=None,
            enable_advanced_objectives=True,
            bassd_dose_weight=0.03,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
        )
    )
    app.cmd_optimization_sweep(
        Namespace(
            input=input_path,
            output=stbdmr_output,
            objective="stbd-mr",
            csv_output=None,
            enable_advanced_objectives=True,
            stbdmr_window_offsets_s="0,3600,21600",
            stbdmr_window_count_time_s=900.0,
            stbdmr_masking_regularization=0.2,
            stbdmr_differentiable_graph=True,
            stbdmr_graph_temperature=2.0,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
            bassd_dose_weight=0.03,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
        )
    )

    difom_payload = json.loads(difom_output.read_text(encoding="utf-8"))
    fim_payload = json.loads(fim_output.read_text(encoding="utf-8"))
    mwdcs_payload = json.loads(mwdcs_output.read_text(encoding="utf-8"))
    bassd_payload = json.loads(bassd_output.read_text(encoding="utf-8"))
    stbdmr_payload = json.loads(stbdmr_output.read_text(encoding="utf-8"))

    assert difom_payload["schema"] == "fluxforge.optimization_sweep.difom.v1"
    assert fim_payload["schema"] == "fluxforge.optimization_sweep.fim.v1"
    assert mwdcs_payload["schema"] == "fluxforge.optimization_sweep.mwdcs.v1"
    assert bassd_payload["schema"] == "fluxforge.optimization_sweep.bassd.v1"
    assert stbdmr_payload["schema"] == "fluxforge.optimization_sweep.stbdmr.v1"
    assert difom_payload["ranked_candidates"][0]["label"] != fim_payload["ranked_candidates"][0]["label"]
    assert len(mwdcs_payload["ranked_candidates"]) == 2
    assert "window_scores" in mwdcs_payload["ranked_candidates"][0]
    assert len(bassd_payload["ranked_candidates"]) == 2
    assert "action_scores" in bassd_payload["ranked_candidates"][0]
    assert len(stbdmr_payload["ranked_candidates"]) == 2
    assert "diagnostics" in stbdmr_payload["ranked_candidates"][0]


def test_cmd_ffexp_export_packages_phase6_products(tmp_path):
    activity_review_path = _write_phase6_real_activity_review(tmp_path)
    grids = load_phase6_real_optimization_grids()
    optimization_path = tmp_path / "optimization.json"
    app.cmd_optimization_sweep(
        Namespace(
            input=None,
            activity_review_file=activity_review_path,
            output=optimization_path,
            objective="di-fom",
            csv_output=None,
            isotopes_of_interest=None,
            irradiation_grid_s=grids["irradiation_grid_s"],
            cooldown_grid_s=grids["cooldown_grid_s"],
            count_grid_s=grids["count_grid_s"],
            reference_irradiation_time_s=3600.0,
            unfold_file=RAFM_UNFOLD_MLEM_PATH,
            neutron_spectrum_file=None,
            flux_scale=1.0,
            reference_flux_integral=0.0,
            generated_candidates_output=None,
            target_nuclide=None,
            nuisance_variance_fraction=0.0,
            fim_regularization=1.0e-6,
            mwdcs_window_offsets_s="0,7200,86400",
            mwdcs_window_count_time_s=900.0,
            mwdcs_full_spectrum_mode=False,
            mwdcs_overlap_penalty=0.0,
            enable_advanced_objectives=False,
            bassd_dose_weight=0.02,
            bassd_exploration_temperature=0.0,
            bassd_seed=17,
            stbdmr_window_offsets_s="0,7200,86400",
            stbdmr_window_count_time_s=900.0,
            stbdmr_masking_regularization=0.1,
            stbdmr_differentiable_graph=False,
            stbdmr_graph_temperature=2.0,
        )
    )

    output_path = tmp_path / "benchmark.ffexp"
    app.cmd_ffexp_export(
        Namespace(
            activity_review_file=activity_review_path,
            inventory_review_file=None,
            masking_file=None,
            optimization_file=optimization_path,
            second_irradiation_file=None,
            plot_paths="inventory.png,pareto.png",
            output=output_path,
        )
    )

    payload = app.read_ffexp_bundle(output_path)
    optimization_payload = json.loads(optimization_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.ffexp_bundle.v1"
    assert payload["format"] == ".ffexp"
    assert payload["summary"]["optimization_candidate_count"] == len(
        optimization_payload["ranked_candidates"]
    )
    assert payload["inventory"]["dose_endpoints"]
    assert payload["masking"]["line_masking_results"]
    assert payload["optimization"]["recommended_schedules"]
    assert payload["plot_manifest"]["paths"] == ["inventory.png", "pareto.png"]


def test_cmd_library_register_list_and_remove(monkeypatch, tmp_path, capsys):
    registry_path = tmp_path / "library_registry.json"
    gamma_path = tmp_path / "gamma.csv"
    gamma_path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\nCo60,1332.5,1.0,166344192.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry_path))

    app.cmd_library_register(
        Namespace(
            alias="Lab Ref",
            locator=str(gamma_path),
            description=None,
        )
    )
    register_out = capsys.readouterr().out
    assert "user_gamma_lab_ref" in register_out

    app.cmd_library_list(Namespace(capability="peak-identification", kind=None, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert any(item["source_id"] == "user_gamma_lab_ref" for item in payload)

    app.cmd_library_remove(Namespace(source_id="user_gamma_lab_ref"))
    remove_out = capsys.readouterr().out
    assert "Removed user_gamma_lab_ref" in remove_out


def test_cmd_plots_uses_master_suite(monkeypatch, tmp_path):
    called = {}

    dummy_module = types.ModuleType("fluxforge.plots.master_suite")

    def fake_load_plot_inputs_from_artifacts(**kwargs):
        called["artifact_kwargs"] = kwargs
        return "plot-inputs"

    def fake_load_example_plot_inputs():
        called["example_called"] = True
        return "example-inputs"

    def fake_normalize_plot_formats(fmt):
        called["format"] = fmt
        return ("png",)

    def fake_generate_master_plan_plots(
        inputs, output_dir, formats, include_response_plot
    ):
        called["generate_args"] = {
            "inputs": inputs,
            "output_dir": output_dir,
            "formats": formats,
            "include_response_plot": include_response_plot,
        }
        return {"g1": [output_dir / "g1.png"]}

    dummy_module.load_plot_inputs_from_artifacts = fake_load_plot_inputs_from_artifacts
    dummy_module.load_example_plot_inputs = fake_load_example_plot_inputs
    dummy_module.normalize_plot_formats = fake_normalize_plot_formats
    dummy_module.generate_master_plan_plots = fake_generate_master_plan_plots
    monkeypatch.setitem(sys.modules, "fluxforge.plots.master_suite", dummy_module)

    app.cmd_plots(
        Namespace(
            example=False,
            unfold_file=tmp_path / "unfold.json",
            response_file=tmp_path / "response.json",
            rates_file=tmp_path / "rates.json",
            prior_flux_file=tmp_path / "prior.json",
            output_dir=tmp_path / "plots",
            format="png",
            include_response_plot=True,
            dry_run=False,
            validate=False,
        )
    )

    assert called["format"] == "png"
    assert called["artifact_kwargs"]["validate"] is False
    assert called["generate_args"]["inputs"] == "plot-inputs"
    assert called["generate_args"]["formats"] == ("png",)


def test_cmd_ingest_reads_spe_and_writes_artifact(monkeypatch, tmp_path):
    written = {}

    def fake_read_spe(path):
        assert Path(path).suffix.lower() == ".spe"
        return _dummy_spectrum()

    def fake_write(output, spectrum, source_path=None):
        written["output"] = output
        written["spectrum_id"] = spectrum.spectrum_id
        written["source"] = source_path

    monkeypatch.setattr(app, "read_spe_file", fake_read_spe)
    monkeypatch.setattr(app, "write_spectrum_file", fake_write)

    in_file = tmp_path / "in.spe"
    in_file.write_text("dummy", encoding="utf-8")
    out_file = tmp_path / "out.json"

    app.cmd_ingest(Namespace(input=in_file, output=out_file, validate=True))

    assert written["output"] == out_file
    assert written["spectrum_id"] == "dummy"
    assert written["source"] == in_file


def test_cmd_ingest_applies_manual_background_scaling(monkeypatch, tmp_path):
    captured = {}

    sample = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.array([2.0, 4.0, 6.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="background",
    )

    def fake_read_genie(path, **kwargs):
        return background if "background" in str(path) else sample

    def fake_write(output, spectrum, source_path=None):
        captured["counts"] = spectrum.counts
        captured["output"] = output

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)
    monkeypatch.setattr(app, "write_spectrum_file", fake_write)

    in_file = tmp_path / "sample.ASC"
    bg_file = tmp_path / "background.ASC"
    in_file.write_text("dummy", encoding="utf-8")
    bg_file.write_text("dummy", encoding="utf-8")
    out_file = tmp_path / "out.json"

    app.cmd_ingest(
        Namespace(
            input=in_file,
            output=out_file,
            validate=False,
            background_file=bg_file,
            background_scale_mode="manual",
            background_scale_factor=0.5,
            energy_calibration=None,
            efficiency_coefficients=None,
        )
    )

    assert captured["output"] == out_file
    assert np.allclose(captured["counts"], [9.0, 18.0, 27.0])


def test_cmd_ingest_writes_optional_adjusted_and_final_exports(
    monkeypatch, tmp_path, capsys
):
    sample = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [1.0, 1.0, 0.0]},
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.array([2.0, 4.0, 6.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [1.0, 1.0, 0.0]},
        spectrum_id="background",
    )

    def fake_read_genie(path, **kwargs):
        if "background" in str(path):
            return GammaSpectrum.from_dict(background.to_dict())
        return GammaSpectrum.from_dict(sample.to_dict())

    written = {}

    def fake_write(output, spectrum, source_path=None):
        written["output"] = output
        written["source"] = source_path

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)
    monkeypatch.setattr(app, "write_spectrum_file", fake_write)

    input_file = tmp_path / "sample.ASC"
    background_file = tmp_path / "background.ASC"
    adjusted_file = tmp_path / "exports" / "sample_background_adjusted.csv"
    final_file = tmp_path / "exports" / "sample_final_corrected.csv"
    input_file.write_text("dummy", encoding="utf-8")
    background_file.write_text("dummy", encoding="utf-8")

    app.cmd_ingest(
        Namespace(
            input=input_file,
            output=tmp_path / "sample.json",
            validate=False,
            background_file=background_file,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients="1,0,0,0,1",
            save_background_adjusted=adjusted_file,
            save_final_corrected=final_file,
        )
    )

    adjusted_lines = adjusted_file.read_text(encoding="utf-8").splitlines()
    final_lines = final_file.read_text(encoding="utf-8").splitlines()
    out = capsys.readouterr().out

    assert written["output"] == tmp_path / "sample.json"
    assert "Wrote background-adjusted counts to" in out
    assert "Wrote final corrected counts to" in out
    assert "Warnings: none" in out
    assert adjusted_lines[-1].startswith("2,3.0,24.0")
    assert final_lines[-1].startswith("2,3.0,24.0,6.0,1.0,24.0,6.0")


def test_cmd_ingest_profile_supplies_background_and_efficiency(monkeypatch, tmp_path):
    captured = {}

    sample = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [0.0, 1.0, 0.0]},
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.array([1.0, 2.0, 3.0]),
        channels=np.array([0, 1, 2]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="background",
    )

    def fake_read_genie(path, **kwargs):
        return background if "profile_background" in str(path) else sample

    def fake_write(output, spectrum, source_path=None):
        captured["counts"] = spectrum.counts
        captured["efficiency"] = dict(spectrum.metadata.get("efficiency", {}))

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)
    monkeypatch.setattr(app, "write_spectrum_file", fake_write)
    monkeypatch.setattr(
        app, "_profile_background_file", lambda _: tmp_path / "profile_background.ASC"
    )
    monkeypatch.setattr(
        app,
        "_profile_efficiency_override",
        lambda _: {
            "C1": -20.26,
            "C2": 10.29,
            "C3": -1.655,
            "C4": 0.08666,
            "geometry_factor_A": 0.00348,
        },
    )

    in_file = tmp_path / "sample.ASC"
    in_file.write_text("dummy", encoding="utf-8")

    app.cmd_ingest(
        Namespace(
            input=in_file,
            output=tmp_path / "out.json",
            validate=False,
            profile="rafm_25cm",
            background_file=None,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            save_background_adjusted=None,
            save_final_corrected=None,
        )
    )

    assert np.allclose(captured["counts"], [9.0, 18.0, 27.0])
    assert captured["efficiency"]["C1"] == -20.26
    assert captured["efficiency"]["geometry_factor_A"] == 0.00348


def test_cmd_ingest_batch_uses_shared_background_for_all_raw_files(
    monkeypatch, tmp_path, capsys
):
    input_dir = tmp_path / "raw_gamma_spec"
    (input_dir / "RAFM4").mkdir(parents=True)
    (input_dir / "flux_wires").mkdir(parents=True)
    sample_a = input_dir / "RAFM4" / "sample_a.ASC"
    sample_b = input_dir / "flux_wires" / "sample_b.ASC"
    background_file = tmp_path / "background.ASC"
    sample_a.write_text("dummy", encoding="utf-8")
    sample_b.write_text("dummy", encoding="utf-8")
    background_file.write_text("dummy", encoding="utf-8")

    def _make_spectrum(counts, spectrum_id):
        return GammaSpectrum(
            counts=np.array(counts, dtype=float),
            channels=np.array([0, 1], dtype=float),
            live_time=10.0,
            real_time=10.0,
            calibration={"energy": [1.0, 1.0, 0.0]},
            spectrum_id=spectrum_id,
        )

    def fake_read_genie(path, **kwargs):
        path = Path(path)
        if path == background_file:
            return _make_spectrum([1.0, 2.0], "background")
        if path.name == "sample_a.ASC":
            return _make_spectrum([10.0, 20.0], "sample_a")
        return _make_spectrum([5.0, 5.0], "sample_b")

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)

    artifact_dir = tmp_path / "results" / "artifacts"
    adjusted_dir = tmp_path / "results" / "background_adjusted"
    final_dir = tmp_path / "results" / "final_corrected"

    app.cmd_ingest_batch(
        Namespace(
            input_dir=input_dir,
            output_dir=artifact_dir,
            validate=False,
            background_file=background_file,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            background_adjusted_dir=adjusted_dir,
            final_corrected_dir=final_dir,
        )
    )

    adjusted_a = adjusted_dir / "RAFM4" / "sample_a_background_adjusted.csv"
    adjusted_b = adjusted_dir / "flux_wires" / "sample_b_background_adjusted.csv"
    out = capsys.readouterr().out

    assert (artifact_dir / "RAFM4" / "sample_a.json").exists()
    assert (artifact_dir / "flux_wires" / "sample_b.json").exists()
    assert adjusted_a.exists()
    assert adjusted_b.exists()
    assert not any(final_dir.rglob("*_final_corrected.csv"))
    assert "Processed 2 spectra from" in out
    assert f"Spectrum artifacts directory: {artifact_dir}" in out
    assert f"Background-adjusted counts directory: {adjusted_dir} (2 files)" in out
    assert f"Final corrected counts directory: {final_dir} (0 files)" in out
    assert (
        "Final corrected export requested but no usable efficiency coefficients were available for sample_a"
        in out
    )
    assert (
        "Final corrected export requested but no usable efficiency coefficients were available for sample_b"
        in out
    )


def test_cmd_spectrum_plot_writes_plot_and_manual_peak_report(
    monkeypatch, tmp_path, capsys
):
    sample = GammaSpectrum(
        counts=np.array([1.0, 3.0, 12.0, 20.0, 12.0, 3.0, 1.0], dtype=float),
        channels=np.arange(7, dtype=float),
        energies=np.arange(100.0, 107.0, 1.0),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [100.0, 1.0, 0.0]},
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.array([0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0], dtype=float),
        channels=np.arange(7, dtype=float),
        energies=np.arange(100.0, 107.0, 1.0),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [100.0, 1.0, 0.0]},
        spectrum_id="background",
    )

    def fake_read_genie(path, **kwargs):
        if "background" in str(path):
            return GammaSpectrum.from_dict(background.to_dict())
        return GammaSpectrum.from_dict(sample.to_dict())

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)

    input_file = tmp_path / "sample.ASC"
    background_file = tmp_path / "background.ASC"
    manual_file = tmp_path / "manual_peaks.csv"
    plot_file = tmp_path / "plots" / "sample.png"
    peak_report = tmp_path / "manual_peaks.json"

    input_file.write_text("dummy", encoding="utf-8")
    background_file.write_text("dummy", encoding="utf-8")
    manual_file.write_text(
        "label,left_keV,right_keV,isotope\nmain_peak,102.0,104.0,Co60\n",
        encoding="utf-8",
    )

    app.cmd_spectrum_plot(
        Namespace(
            input=input_file,
            output=plot_file,
            validate=False,
            profile=None,
            background_file=background_file,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            background_subtracted=True,
            manual_peaks_file=manual_file,
            save_peak_report=peak_report,
            feature_photopeak_keV=None,
            feature_line_keV=None,
            feature_isotope=None,
            auto_feature_driver=False,
            feature_detector_material="hpge",
            feature_tolerance_keV=2.0,
            feature_min_intensity=0.02,
            save_feature_report=None,
            title=None,
            x_min_keV=None,
            x_max_keV=None,
            y_log=False,
        )
    )

    payload = json.loads(peak_report.read_text(encoding="utf-8"))
    out = capsys.readouterr().out

    assert plot_file.exists()
    assert peak_report.exists()
    assert payload["peaks"][0]["label"] == "main_peak"
    assert payload["peaks"][0]["gross_counts"] == 44.0
    assert payload["peaks"][0]["background_subtracted"] is True
    assert "Wrote spectrum plot to" in out
    assert "Wrote manual peak report to" in out
    assert "Warnings:" in out


def test_cmd_spectrum_plot_writes_spectral_feature_report(
    monkeypatch, tmp_path, capsys
):
    sample = GammaSpectrum(
        counts=np.array([2.0, 5.0, 25.0, 60.0, 24.0, 6.0, 2.0], dtype=float),
        channels=np.arange(7, dtype=float),
        energies=np.arange(100.0, 107.0, 1.0),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [100.0, 1.0, 0.0]},
        spectrum_id="sample",
    )

    monkeypatch.setattr(
        app,
        "read_genie_spectrum",
        lambda path, **kwargs: GammaSpectrum.from_dict(sample.to_dict()),
    )

    input_file = tmp_path / "sample.ASC"
    plot_file = tmp_path / "plots" / "sample_features.png"
    feature_report = tmp_path / "feature_report.json"
    input_file.write_text("dummy", encoding="utf-8")

    app.cmd_spectrum_plot(
        Namespace(
            input=input_file,
            output=plot_file,
            validate=False,
            profile=None,
            background_file=None,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            background_subtracted=False,
            manual_peaks_file=None,
            save_peak_report=None,
            feature_photopeak_keV=104.0,
            feature_line_keV=None,
            feature_isotope=None,
            auto_feature_driver=False,
            feature_detector_material="hpge",
            feature_tolerance_keV=2.0,
            feature_min_intensity=0.02,
            save_feature_report=feature_report,
            title=None,
            x_min_keV=None,
            x_max_keV=None,
            y_log=False,
        )
    )

    payload = json.loads(feature_report.read_text(encoding="utf-8"))
    out = capsys.readouterr().out

    assert plot_file.exists()
    assert feature_report.exists()
    assert payload["photopeak_energy_keV"] == pytest.approx(104.0)
    assert any(item["kind"] == "compton_edge" for item in payload["features"])
    assert any(
        item["estimated_height_counts"] is not None for item in payload["features"]
    )
    assert "Wrote spectral feature report to" in out


def test_cmd_peaks_manual_roi_mode(monkeypatch, tmp_path, capsys):
    sample = GammaSpectrum(
        counts=np.array([2.0, 4.0, 10.0, 18.0, 10.0, 4.0, 2.0], dtype=float),
        channels=np.arange(7, dtype=float),
        energies=np.arange(200.0, 207.0, 1.0),
        live_time=12.0,
        real_time=12.0,
        calibration={"energy": [200.0, 1.0, 0.0]},
        spectrum_id="sample",
    )

    def fake_read_genie(path, **kwargs):
        return GammaSpectrum.from_dict(sample.to_dict())

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)

    spectrum_file = tmp_path / "sample.ASC"
    manual_file = tmp_path / "manual.json"
    output_file = tmp_path / "manual_peaks.json"
    spectrum_file.write_text("dummy", encoding="utf-8")
    manual_file.write_text(
        json.dumps(
            [
                {
                    "label": "manual_roi_1",
                    "left_channel": 2,
                    "right_channel": 4,
                    "isotope": "Sc46",
                }
            ]
        ),
        encoding="utf-8",
    )

    app.cmd_peaks(
        Namespace(
            spectrum_file=spectrum_file,
            output=output_file,
            sensitivity="default",
            fit_window=6,
            validate=False,
            manual_peaks_file=manual_file,
            profile=None,
            background_file=None,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            background_subtracted=False,
        )
    )

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    out = capsys.readouterr().out

    assert payload["peaks"][0]["label"] == "manual_roi_1"
    assert payload["peaks"][0]["left_channel"] == 2
    assert payload["peaks"][0]["right_channel"] == 4
    assert payload["peaks"][0]["gross_counts"] == 38.0
    assert payload["peaks"][0]["net_counts"] == 38.0
    assert "Wrote peak report to" in out
    assert "Warnings: none" in out


def test_cmd_peaks_activity_and_rates(monkeypatch, tmp_path):
    peak_written = {}
    lines_written = {}
    rates_written = {}

    spectrum_payload = {
        "schema_version": "1.0",
        "type": "spectrum",
        "spectrum": _dummy_spectrum().to_dict(),
    }

    fake_peak = SimpleNamespace(
        channel=1.0,
        energy_keV=1.0,
        amplitude=10.0,
        raw_counts=10.0,
        sigma_keV=0.1,
        area=12.0,
        region=[0, 2],
        is_report=True,
        report_isotope="Co-60",
        report_file="report.txt",
    )

    def fake_write_peak_report(output, **kwargs):
        peak_written["output"] = output
        peak_written["payload"] = kwargs

    monkeypatch.setattr(app, "read_spectrum_file", lambda _: spectrum_payload)
    monkeypatch.setattr(app, "detect_peaks_segmented", lambda *a, **k: [fake_peak])
    monkeypatch.setattr(app, "write_peak_report", fake_write_peak_report)

    peak_out = tmp_path / "peaks.json"
    app.cmd_peaks(
        Namespace(
            spectrum_file=tmp_path / "spectrum.json",
            output=peak_out,
            sensitivity="default",
            fit_window=6,
            validate=False,
        )
    )
    assert peak_written["output"] == peak_out
    assert len(peak_written["payload"]["peaks"]) == 1

    def fake_write_line_activities(output, **kwargs):
        lines_written["output"] = output
        lines_written["payload"] = kwargs

    monkeypatch.setattr(
        app,
        "read_peak_report",
        lambda _: {
            "spectrum_id": "dummy",
            "live_time_s": 10.0,
            "peaks": peak_written["payload"]["peaks"],
        },
    )
    monkeypatch.setattr(app, "write_line_activities", fake_write_line_activities)

    lines_out = tmp_path / "lines.json"
    app.cmd_activity(
        Namespace(
            peaks_file=peak_out,
            output=lines_out,
            live_time_s=None,
            efficiency=0.5,
            emission_probability=0.25,
            half_life_s=1.0,
            sample_mass_g=2.0,
            isotope=None,
            reaction_id=None,
            validate=False,
        )
    )
    assert lines_written["output"] == lines_out
    line = lines_written["payload"]["lines"][0]
    assert line["reaction_id"] == "Co-60"
    assert line["activity_Bq"] > 0
    assert line["atoms"] > 0
    assert line["radioactive_mass_g"] > 0
    assert line["specific_activity_Bq_g"] == pytest.approx(line["activity_Bq"] / 2.0)

    monkeypatch.setattr(
        app,
        "read_line_activities",
        lambda _: {"lines": lines_written["payload"]["lines"]},
    )
    monkeypatch.setattr(
        app,
        "reaction_rate_from_activity",
        lambda *a, **k: SimpleNamespace(rate=5.0, uncertainty=0.5),
    )

    def fake_write_reaction_rates(output, **kwargs):
        rates_written["output"] = output
        rates_written["payload"] = kwargs

    monkeypatch.setattr(app, "write_reaction_rates", fake_write_reaction_rates)

    rates_out = tmp_path / "rates.json"
    app.cmd_rates(
        Namespace(
            lines_file=lines_out,
            segments_file=None,
            duration_s=10.0,
            half_life_s=1.0,
            output=rates_out,
            validate=False,
        )
    )
    assert rates_written["output"] == rates_out
    assert rates_written["payload"]["rates"][0]["rate"] == 5.0


def test_cmd_peaks_passes_fit_window_through_config(monkeypatch, tmp_path):
    spectrum_payload = {
        "schema_version": "1.0",
        "type": "spectrum",
        "spectrum": _dummy_spectrum().to_dict(),
    }
    captured = {}

    fake_peak = SimpleNamespace(
        channel=1.0,
        energy_keV=1.0,
        amplitude=10.0,
        raw_counts=10.0,
        sigma_keV=0.1,
        area=12.0,
        region=[0, 2],
        is_report=False,
        report_isotope="",
        report_file="",
    )

    def fake_detect(channels, energies, counts, raw_counts=None, config=None):
        captured["fit_window"] = None if config is None else config.fit_window
        return [fake_peak]

    monkeypatch.setattr(app, "read_spectrum_file", lambda _: spectrum_payload)
    monkeypatch.setattr(app, "detect_peaks_segmented", fake_detect)
    monkeypatch.setattr(app, "write_peak_report", lambda *args, **kwargs: None)

    peak_out = tmp_path / "peaks.json"
    app.cmd_peaks(
        Namespace(
            spectrum_file=tmp_path / "spectrum.json",
            output=peak_out,
            sensitivity="default",
            fit_window=9,
            validate=False,
        )
    )

    assert captured["fit_window"] == 9


def test_cmd_peaks_parity_method_uses_core_detection(monkeypatch, tmp_path):
    spectrum_payload = {
        "schema_version": "1.0",
        "type": "spectrum",
        "spectrum": _dummy_spectrum().to_dict(),
    }
    written = {}

    fake_peak = SimpleNamespace(
        channel=1.2,
        energy_keV=1.2,
        net_counts=42.0,
        roi_bounds_keV=(0.8, 1.6),
        nuclide="Co-60",
        peak_id="peak-1",
        significance=5.1,
        fit_quality=1.2,
    )

    monkeypatch.setattr(app, "read_spectrum_file", lambda _: spectrum_payload)
    monkeypatch.setattr(app, "detect_peak_candidates", lambda *a, **k: [fake_peak])
    monkeypatch.setattr(
        app,
        "write_peak_report",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    output = tmp_path / "peaks.json"
    app.cmd_peaks(
        Namespace(
            spectrum_file=tmp_path / "spectrum.json",
            output=output,
            sensitivity="default",
            fit_window=6,
            validate=False,
            manual_peaks_file=None,
            method="mariscotti",
            max_peaks=8,
        )
    )

    assert written["output"] == output
    assert written["payload"]["peaks"][0]["peak_search_method"] == "mariscotti"
    assert written["payload"]["peaks"][0]["report_isotope"] == "Co-60"


def test_cmd_roi_analyze_writes_background_and_overlap_payload(monkeypatch, tmp_path):
    channels = np.arange(256, dtype=float)
    sample = GammaSpectrum(
        counts=(
            12.0
            + 0.03 * channels
            + 220.0 * np.exp(-0.5 * ((channels - 90.0) / 4.0) ** 2)
            + 180.0 * np.exp(-0.5 * ((channels - 98.0) / 4.5) ** 2)
        ),
        channels=channels,
        live_time=60.0,
        real_time=63.0,
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.full_like(channels, 3.0, dtype=float),
        channels=channels,
        live_time=60.0,
        real_time=60.0,
        calibration={"energy": [0.0, 1.0]},
        spectrum_id="background",
    )

    def fake_read_genie(path, **kwargs):
        return background if "background" in str(path) else sample

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)

    input_file = tmp_path / "sample.ASC"
    background_file = tmp_path / "background.ASC"
    output_file = tmp_path / "roi_analysis.json"
    input_file.write_text("dummy", encoding="utf-8")
    background_file.write_text("dummy", encoding="utf-8")

    app.cmd_roi_analyze(
        Namespace(
            input=input_file,
            output=output_file,
            label="doublet",
            left_keV=84.0,
            right_keV=104.0,
            left_channel=None,
            right_channel=None,
            background_method="roi_sideband",
            peak_search_method="mariscotti",
            sideband_width_keV=5.0,
            decompose_overlaps=True,
            max_components=3,
            profile=None,
            background_file=background_file,
            background_scale_mode="live",
            background_scale_factor=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            validate=False,
        )
    )

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["analysis"]["background_method"] == "roi_sideband"
    assert payload["analysis"]["peak_search_method"] == "mariscotti"
    assert payload["analysis"]["net_counts"] > 0.0
    assert len(payload["analysis"]["overlap_components"]) >= 1


def test_cmd_roi_statistics_writes_summary_payload(monkeypatch, tmp_path):
    def _make(scale: float) -> GammaSpectrum:
        channels = np.arange(128, dtype=float)
        counts = (
            8.0
            + 0.02 * channels
            + scale * 120.0 * np.exp(-0.5 * ((channels - 44.0) / 3.0) ** 2)
        )
        return GammaSpectrum(
            counts=counts,
            channels=channels,
            live_time=30.0,
            real_time=30.0,
            calibration={"energy": [0.0, 1.0]},
            spectrum_id=f"sample-{scale:.2f}",
        )

    def fake_read_genie(path, **kwargs):
        name = Path(path).name
        if name.startswith("a"):
            return _make(1.0)
        if name.startswith("b"):
            return _make(1.1)
        return _make(0.9)

    monkeypatch.setattr(app, "read_genie_spectrum", fake_read_genie)

    files = [tmp_path / "a.ASC", tmp_path / "b.ASC", tmp_path / "c.ASC"]
    for path in files:
        path.write_text("dummy", encoding="utf-8")
    output_file = tmp_path / "roi_stats.json"

    app.cmd_roi_statistics(
        Namespace(
            inputs=files,
            output=output_file,
            label="roi-batch",
            left_keV=40.0,
            right_keV=48.0,
            left_channel=None,
            right_channel=None,
            background_method="snip",
            peak_search_method="nasa_peaksearch",
            sideband_width_keV=4.0,
            profile=None,
            energy_calibration=None,
            efficiency_coefficients=None,
            validate=False,
        )
    )

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["statistics"]["sample_count"] == 3
    assert payload["statistics"]["mean_net_counts"] > 0.0
    assert len(payload["statistics"]["samples"]) == 3


def test_cmd_unfold_compare_and_report(monkeypatch, tmp_path):
    unfold_written = {}
    validation_written = {}
    report_written = {}

    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {
            "matrix": [[1.0]],
            "boundaries_eV": [1e-5, 1.0],
            "reactions": ["r1"],
        },
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {"rates": [{"rate": 1.0, "uncertainty": 0.1}]},
    )
    monkeypatch.setattr(
        app,
        "gls_adjust",
        lambda *a, **k: SimpleNamespace(flux=[1.0], covariance=[[0.01]], chi2=0.1),
    )
    monkeypatch.setattr(
        app,
        "write_unfold_result",
        lambda output, **kwargs: unfold_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    unfold_out = tmp_path / "unfold.json"
    app.cmd_unfold(
        Namespace(
            rates_file=tmp_path / "rates.json",
            response_file=tmp_path / "response.json",
            prior_flux_file=None,
            prior_uncertainty=0.25,
            prior_cov_model="diagonal",
            prior_correlation_length=1.0,
            output=unfold_out,
            validate=False,
        )
    )
    assert unfold_written["output"] == unfold_out
    assert unfold_written["payload"]["flux"] == [1.0]

    truth_file = tmp_path / "truth.json"
    truth_file.write_text(json.dumps([1.0]), encoding="utf-8")

    monkeypatch.setattr(
        app, "read_unfold_result", lambda _: {"flux": [1.0], "chi2": 0.1}
    )
    monkeypatch.setattr(
        app,
        "write_validation_bundle",
        lambda output, **kwargs: validation_written.update(
            {"output": output, "payload": kwargs}
        ),
    )
    compare_out = tmp_path / "validation.json"
    app.cmd_compare(
        Namespace(
            unfold_file=unfold_out,
            truth_flux_file=truth_file,
            output=compare_out,
            validate=False,
        )
    )
    assert validation_written["output"] == compare_out
    assert "metrics" in validation_written["payload"]

    monkeypatch.setattr(
        app, "read_spectrum_file", lambda _: {"spectrum": _dummy_spectrum().to_dict()}
    )
    monkeypatch.setattr(app, "read_peak_report", lambda _: {"peaks": [1, 2, 3]})
    monkeypatch.setattr(app, "read_line_activities", lambda _: {"lines": [1, 2]})
    monkeypatch.setattr(app, "read_reaction_rates", lambda _: {"rates": [1]})
    monkeypatch.setattr(app, "read_unfold_result", lambda _: {"chi2": 0.1})
    monkeypatch.setattr(
        app, "read_validation_bundle", lambda _: {"metrics": {"mae": 0.0}}
    )
    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: report_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    report_out = tmp_path / "report.json"
    app.cmd_report(
        Namespace(
            spectrum_file=tmp_path / "spec.json",
            peaks_file=tmp_path / "peaks.json",
            lines_file=tmp_path / "lines.json",
            rates_file=tmp_path / "rates.json",
            unfold_file=tmp_path / "unfold.json",
            validation_file=tmp_path / "validation.json",
            output=report_out,
            validate=False,
        )
    )
    assert report_written["output"] == report_out
    assert report_written["payload"]["summary"]["peak_count"] == 3
    assert report_written["payload"]["summary"]["total_activity_Bq"] == 0.0
    assert report_written["payload"]["text_report"]["path"] == "report.txt"
    assert report_written["payload"]["tables"] is None
    assert (tmp_path / "report.txt").exists()


def test_cmd_report_summarizes_activation_metrics(monkeypatch, tmp_path):
    report_written = {}

    monkeypatch.setattr(
        app,
        "read_line_activities",
        lambda _: {
            "lines": [
                {
                    "activity_Bq": 12.0,
                    "radioactive_mass_g": 2.0e-12,
                    "specific_activity_Bq_g": 24.0,
                    "radioisotope_specific_activity_Bq_g": 1.5e12,
                },
                {
                    "activity_Bq": 8.0,
                    "radioactive_mass_g": 3.0e-12,
                    "specific_activity_Bq_g": 16.0,
                    "radioisotope_specific_activity_Bq_g": 1.1e12,
                },
            ]
        },
    )
    monkeypatch.setattr(
        app, "read_reaction_rates", lambda _: {"rates": [{"rate": 3.0}, {"rate": 2.0}]}
    )
    monkeypatch.setattr(
        app, "read_unfold_result", lambda _: {"chi2": 0.1, "flux": [1.0, 2.0, 3.0]}
    )
    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: report_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    report_out = tmp_path / "report.json"
    app.cmd_report(
        Namespace(
            spectrum_file=None,
            peaks_file=None,
            lines_file=tmp_path / "lines.json",
            rates_file=tmp_path / "rates.json",
            unfold_file=tmp_path / "unfold.json",
            validation_file=None,
            output=report_out,
            validate=False,
        )
    )

    summary = report_written["payload"]["summary"]
    assert summary["line_count"] == 2
    assert summary["total_activity_Bq"] == 20.0
    assert summary["total_radioactive_mass_g"] == pytest.approx(5.0e-12)
    assert summary["max_specific_activity_Bq_g"] == 24.0
    assert summary["max_radioisotope_specific_activity_Bq_g"] == 1.5e12
    assert summary["rate_count"] == 2
    assert summary["total_rate_reactions_s"] == 5.0
    assert summary["integral_flux"] == 6.0
    assert report_written["payload"]["text_report"]["path"] == "report.txt"
    assert report_written["payload"]["tables"]["directory"] == "report_tables"
    assert "line_activity_detail" in report_written["payload"]["tables"]["items"]
    assert "isotope_activity_summary" in report_written["payload"]["tables"]["items"]
    assert "reaction_rates_summary" in report_written["payload"]["tables"]["items"]
    assert "unfold_flux_groups" in report_written["payload"]["tables"]["items"]
    report_text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "FluxForge Standard Activation Report" in report_text
    assert "Isotope Activity Summary" in report_text
    assert "Reaction Rate Summary" in report_text


def test_cmd_report_includes_optimization_recommendation(monkeypatch, tmp_path):
    report_written = {}

    candidate_input_path = tmp_path / "optimization_candidates.json"
    candidate_input_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "candidate_a",
                        "lines": [
                            {
                                "nuclide": "Mo-99",
                                "line_energy_keV": 140.5,
                                "signal_counts": 120.0,
                                "background_counts": 20.0,
                                "interference_counts": 30.0,
                            },
                            {
                                "nuclide": "Sc-46",
                                "line_energy_keV": 140.6,
                                "signal_counts": 90.0,
                                "background_counts": 15.0,
                                "interference_counts": 60.0,
                            },
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    optimization_path = tmp_path / "optimization_sweep.json"
    optimization_path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.optimization_sweep.difom.v1",
                "objective": "di-fom",
                "input": str(candidate_input_path),
                "isotopes_of_interest": ["Mo-99"],
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "label": "candidate_a",
                        "irradiation_time_s": 3600.0,
                        "cooldown_time_s": 7200.0,
                        "count_time_s": 900.0,
                        "difom_score": 123.4,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: report_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    report_out = tmp_path / "report.json"
    app.cmd_report(
        Namespace(
            spectrum_file=None,
            peaks_file=None,
            lines_file=None,
            rates_file=None,
            optimization_file=optimization_path,
            unfold_file=None,
            validation_file=None,
            validation_results_root=None,
            output=report_out,
            validate=False,
        )
    )

    summary = report_written["payload"]["summary"]
    assert summary["optimization_objective"] == "di-fom"
    assert summary["recommended_schedule_label"] == "candidate_a"
    assert summary["recommended_mask_isotope"] == "Sc-46"
    assert summary["recommended_isotope_of_interest"] == "Mo-99"
    assert "optimization_recommendation" in report_written["payload"]["tables"]["items"]
    assert "optimization_top_candidates" in report_written["payload"]["tables"]["items"]

    report_text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "Optimization Recommendation" in report_text
    assert "Top Optimization Schedules" in report_text


def test_cmd_report_includes_masking_review_tables(monkeypatch, tmp_path):
    report_written = {}

    masking_path = tmp_path / "masking_review.json"
    masking_path.write_text(
        json.dumps(
            {
                "schema": "fluxforge.masking_review.v1",
                "energy_window_keV": 3.0,
                "line_masking_results": [
                    {
                        "rank": 1,
                        "target_nuclide": "Mo-99",
                        "masking_nuclide": "Sc-46",
                        "masking_score": 0.8,
                        "interference_counts": 120.0,
                        "continuum_counts": 20.0,
                    }
                ],
                "masking_isotope_ranking": [
                    {
                        "rank": 1,
                        "masking_nuclide": "Sc-46",
                        "total_masking_score": 0.8,
                        "total_interference_counts": 120.0,
                        "total_continuum_counts": 20.0,
                        "target_line_count": 1,
                    }
                ],
                "alternate_line_recommendations": [
                    {
                        "rank": 1,
                        "nuclide": "Mo-99",
                        "preferred_line_energy_keV": 739.5,
                        "preferred_masking_score": 0.1,
                        "strongest_line_energy_keV": 140.5,
                        "guidance": "switch_to_preferred_line",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: report_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    report_out = tmp_path / "report.json"
    app.cmd_report(
        Namespace(
            spectrum_file=None,
            peaks_file=None,
            lines_file=None,
            rates_file=None,
            masking_file=masking_path,
            optimization_file=None,
            unfold_file=None,
            validation_file=None,
            validation_results_root=None,
            output=report_out,
            validate=False,
        )
    )

    summary = report_written["payload"]["summary"]
    assert summary["masking_line_interactions"] == 1
    assert summary["dominant_masking_isotope"] == "Sc-46"
    assert "masking_line_results" in report_written["payload"]["tables"]["items"]
    assert "masking_isotope_ranking" in report_written["payload"]["tables"]["items"]
    assert "alternate_line_recommendations" in report_written["payload"]["tables"]["items"]

    report_text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "Masking Review Summary" in report_text
    assert "Masking Isotope Ranking" in report_text


def test_cmd_report_accepts_validation_results_root(monkeypatch, tmp_path):
    report_written = {}

    results_root = tmp_path / "rafm_validation"
    analysis_dir = results_root / "analysis_json"
    unfolding_dir = results_root / "unfolding"
    analysis_dir.mkdir(parents=True)
    unfolding_dir.mkdir(parents=True)

    (results_root / "validation_summary.json").write_text(
        json.dumps(
            {
                "overall_passed": True,
                "n_raw_analyzed": 2,
                "n_matched_pairs": 2,
                "qg_internal_consistency_flags": 1,
                "fluxforge_line_consistency_flags": 3,
                "measurement_qc_flags": 0,
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "sample_a.json").write_text(
        json.dumps(
            {
                "sample_id": "sample_a",
                "sample_group": "RAFM4",
                "timing": {"measurement_time": "2025-08-20T15:45:00"},
                "n_detected_peaks": 5,
                "n_unidentified_peaks": 1,
                "peaks": [{"gross_counts": 1000.0}, {"gross_counts": 2000.0}],
                "isotopes": {
                    "Cr51": {
                        "activity_bq": 10.0,
                        "activity_unc_bq": 1.0,
                        "radioactive_mass_g": 2.0e-12,
                        "specific_activity_Bq_g": 20.0,
                        "n_peaks": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (unfolding_dir / "gls.json").write_text(
        json.dumps(
            {
                "flux": [1.0, 2.0],
                "energy_edges_eV": [0.1, 1.0, 10.0],
                "chi_squared": 0.5,
                "method": "GLS",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: report_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    report_out = tmp_path / "report.json"
    app.cmd_report(
        Namespace(
            spectrum_file=None,
            peaks_file=None,
            lines_file=None,
            rates_file=None,
            unfold_file=None,
            validation_file=None,
            validation_results_root=results_root,
            output=report_out,
            validate=False,
        )
    )

    summary = report_written["payload"]["summary"]
    assert summary["validation_overall_passed"] is True
    assert summary["validation_sample_count"] == 1
    assert summary["validation_total_activity_Bq"] == 10.0
    assert summary["integral_flux"] == 3.0
    assert summary["chi2"] == 0.5
    assert report_written["payload"]["tables"]["directory"] == "report_tables"
    assert "validation_sample_summary" in report_written["payload"]["tables"]["items"]


def test_cmd_k0_normalize_writes_observation_bundle(monkeypatch, tmp_path):
    written = {}

    monkeypatch.setattr(
        app,
        "read_peak_report",
        lambda path: {
            "schema": "fluxforge.peak_report.v1",
            "spectrum_id": "spec-1",
            "live_time_s": 100.0,
            "peaks": [
                {
                    "label": "gold_ref",
                    "energy_keV": 411.8,
                    "net_counts": 1000.0,
                    "net_counts_unc": 31.0,
                    "report_isotope": "Au-198",
                    "sample_role": "reference_monitor",
                    "efficiency": 0.02,
                }
            ],
        },
    )
    monkeypatch.setattr(
        app,
        "write_peak_observation_bundle",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_normalize(
        Namespace(
            peaks_file=tmp_path / "peaks.json",
            spectrum_file=None,
            detector_characterization_file=None,
            detector_id="hpge-01",
            geometry_id="pos-200mm",
            irradiation_time_s=600.0,
            decay_time_s=3600.0,
            counting_time_s=None,
            import_format="peak_report",
            project_id="proj-1",
            sample_id="sample-1",
            irradiation_id="irr-1",
            measurement_id="meas-1",
            expert_override=False,
            allow_advanced_lines=False,
            output=tmp_path / "observations.json",
            validate=False,
        )
    )

    assert written["output"].name == "observations.json"
    assert written["payload"]["summary"]["accepted_count"] == 1
    assert written["payload"]["observations"][0]["assigned_radionuclide"] == "Au-198"
    assert written["payload"]["summary"]["sample_id"] == "sample-1"


def test_cmd_k0_detector_writes_characterization(monkeypatch, tmp_path):
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps(
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
            ]
        ),
        encoding="utf-8",
    )
    written = {}
    monkeypatch.setattr(
        app,
        "write_detector_characterization",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_detector(
        Namespace(
            points_file=points_file,
            detector_id="hpge-01",
            reference_position_mm=200.0,
            degree=1,
            peak_to_total_ratio=None,
            coincidence_mode="not_applied",
            output=tmp_path / "detector.json",
            validate=False,
        )
    )

    assert written["payload"]["detector_id"] == "hpge-01"
    assert written["output"].name == "detector.json"


def test_cmd_k0_analyze_writes_bundle(monkeypatch, tmp_path):
    written = {}
    monkeypatch.setattr(
        app,
        "read_peak_observation_bundle",
        lambda path: {
            "schema": "fluxforge.peak_observation_bundle.v1",
            "observations": [
                {
                    "peak_id": "gold_ref",
                    "source_spectrum_id": "spec-1",
                    "detector_id": "hpge-01",
                    "geometry_id": "pos-200mm",
                    "line_energy_keV": 411.8,
                    "line_id": "Au-198@411.8keV",
                    "assigned_radionuclide": "Au-198",
                    "net_peak_area": 100000.0,
                    "area_uncertainty": 316.0,
                    "live_time_s": 100.0,
                    "real_time_s": 102.0,
                    "irradiation_time_s": 600.0,
                    "decay_time_s": 3600.0,
                    "counting_time_s": 100.0,
                    "dead_time_correction_method": "live_time_real_time",
                    "baseline_method": "local",
                    "deconvolution_status": "resolved",
                    "efficiency": 0.02,
                    "sample_role": "reference_monitor",
                    "sample_mass_g": 0.001,
                    "eligibility_class": "direct_k0_eligible",
                    "eligibility_accepted": True,
                },
                {
                    "peak_id": "co_line",
                    "source_spectrum_id": "spec-1",
                    "detector_id": "hpge-01",
                    "geometry_id": "pos-200mm",
                    "line_energy_keV": 1332.5,
                    "line_id": "Co-60@1332.5keV",
                    "assigned_radionuclide": "Co-60",
                    "net_peak_area": 2500.0,
                    "area_uncertainty": 50.0,
                    "live_time_s": 100.0,
                    "real_time_s": 102.0,
                    "irradiation_time_s": 600.0,
                    "decay_time_s": 3600.0,
                    "counting_time_s": 100.0,
                    "dead_time_correction_method": "live_time_real_time",
                    "baseline_method": "local",
                    "deconvolution_status": "resolved",
                    "efficiency": 0.01,
                    "sample_role": "sample",
                    "sample_mass_g": 0.1,
                    "eligibility_class": "direct_k0_eligible",
                    "eligibility_accepted": True,
                },
            ],
        },
    )
    monkeypatch.setattr(
        app,
        "read_facility_characterization",
        lambda path: {
            "schema": "fluxforge.facility_characterization.v1",
            "flux_parameters": {
                "f": 25.0,
                "f_uncertainty": 1.0,
                "alpha": 0.01,
                "alpha_uncertainty": 0.005,
                "phi_thermal": 1.0e12,
                "phi_epithermal": 4.0e10,
            },
            "temperature": {"value_K": 300.0},
        },
    )
    monkeypatch.setattr(
        app,
        "write_k0_analysis_bundle",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_analyze(
        Namespace(
            observations_file=tmp_path / "observations.json",
            facility_file=tmp_path / "facility.json",
            sample_mass_g=0.1,
            reference_isotope="Au-198",
            reference_mass_g=0.001,
            k0_library_file=None,
            auxiliary_library_file=None,
            output=tmp_path / "k0_analysis.json",
            validate=False,
        )
    )

    assert written["output"].name == "k0_analysis.json"
    assert written["payload"]["summary"]["element_count"] == 1


def test_cmd_k0_aggregate_writes_bundle(monkeypatch, tmp_path):
    written = {}
    monkeypatch.setattr(
        app,
        "read_k0_analysis_bundle",
        lambda path: {
            "summary": {"sample_id": Path(path).stem},
            "element_results": [
                {
                    "sample_id": Path(path).stem,
                    "element": "Co",
                    "concentration_ug_g": 10.0,
                    "concentration_unc_ug_g": 1.0,
                    "measurement_ids": [Path(path).stem],
                    "irradiation_ids": ["irr-1"],
                    "line_ids": ["line-1"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        app,
        "write_k0_aggregation_bundle",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_aggregate(
        Namespace(
            analysis_files=[tmp_path / "a.json", tmp_path / "b.json"],
            output=tmp_path / "k0_aggregation.json",
            validate=False,
        )
    )

    assert written["output"].name == "k0_aggregation.json"
    assert written["payload"]["summary"]["source_bundle_count"] == 2


def test_cmd_k0_qaqc_writes_bundle(monkeypatch, tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "role": "blank",
                        "analysis_file": str(tmp_path / "blank.json"),
                        "default_limit_ug_g": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    written = {}
    monkeypatch.setattr(
        app,
        "read_k0_analysis_bundle",
        lambda path: {
            "element_results": [{"element": "Co", "concentration_ug_g": 0.1}]
        },
    )
    monkeypatch.setattr(
        app,
        "write_k0_qaqc_bundle",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_qaqc(
        Namespace(
            plan_file=plan_file,
            output=tmp_path / "k0_qaqc.json",
            validate=False,
        )
    )

    assert written["output"].name == "k0_qaqc.json"
    assert written["payload"]["summary"]["pass_count"] == 1


def test_cmd_k0_report_writes_report_bundle(monkeypatch, tmp_path):
    written = {}
    monkeypatch.setattr(
        app,
        "read_k0_analysis_bundle",
        lambda path: {
            "summary": {"element_count": 1},
            "element_results": [
                {
                    "element": "Co",
                    "concentration_ug_g": 10.0,
                    "concentration_unc_ug_g": 1.0,
                }
            ],
            "line_results": [],
            "recognized_but_not_applied": [],
            "libraries": {
                "standard_k0_library": {
                    "library_id": "demo",
                    "version": "1",
                    "status": "partial",
                }
            },
        },
    )
    monkeypatch.setattr(
        app,
        "write_report_bundle",
        lambda output, **kwargs: written.update({"output": output, "payload": kwargs}),
    )

    app.cmd_k0_report(
        Namespace(
            analysis_file=tmp_path / "k0_analysis.json",
            aggregation_file=None,
            qaqc_file=None,
            output=tmp_path / "k0_report.json",
            validate=False,
        )
    )

    assert written["output"].name == "k0_report.json"
    assert written["payload"]["summary"]["element_count"] == 1


def test_cmd_k0_import_kayzero_writes_outputs(tmp_path):
    library_dir = tmp_path / "KayWinV4" / "library"
    library_dir.mkdir(parents=True)
    (library_dir / "k0-2023.uk0").write_text(
        "Nuclide E (keV) k0 dk0 k0code\nAu-198 411.8 1.0 0.0 1\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.uQ0").write_text(
        "Nuclide Q0 dQ0 Er dEr\nAu-198 15.71 1.5 5.65 0.1\n",
        encoding="utf-8",
    )
    (library_dir / "k0-2023.uT12").write_text(
        "T1/2 dT\nAu-198 3880.8 0.3\n",
        encoding="utf-8",
    )

    output = tmp_path / "kayzero_library.json"
    report_output = tmp_path / "kayzero_report.json"

    app.cmd_k0_import_kayzero(
        Namespace(
            input=tmp_path,
            preferred_version=None,
            output=output,
            report_output=report_output,
        )
    )

    library_payload = json.loads(output.read_text(encoding="utf-8"))
    report_payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert library_payload["library_id"] == "fluxforge.k0.kayzero.2023"
    assert "Au-198" in library_payload["records"]
    assert report_payload["summary"]["record_count"] == 1


def test_cmd_unfold_supports_mlem(monkeypatch, tmp_path):
    unfold_written = {}

    monkeypatch.setattr(
        app,
        "read_response_bundle",
        lambda _: {
            "matrix": [[1.0, 0.4]],
            "boundaries_eV": [1e-5, 1.0, 1e3],
            "reactions": ["r1"],
        },
    )
    monkeypatch.setattr(
        app,
        "read_reaction_rates",
        lambda _: {"rates": [{"rate": 1.0, "uncertainty": 0.1}]},
    )
    monkeypatch.setattr(
        app,
        "mlem",
        lambda *a, **k: SimpleNamespace(
            flux=[1.2, 0.8],
            iterations=7,
            converged=True,
            chi_squared=0.25,
            chi_squared_history=[1.0, 0.4, 0.25],
            final_residuals=[0.0],
        ),
    )
    monkeypatch.setattr(
        app,
        "write_unfold_result",
        lambda output, **kwargs: unfold_written.update(
            {"output": output, "payload": kwargs}
        ),
    )

    app.cmd_unfold(
        Namespace(
            rates_file=tmp_path / "rates.json",
            response_file=tmp_path / "response.json",
            prior_flux_file=None,
            method="mlem",
            prior_uncertainty=0.25,
            prior_cov_model="diagonal",
            prior_correlation_length=1.0,
            max_iters=25,
            tolerance=1e-4,
            chi2_tolerance=0.01,
            relaxation=0.8,
            floor=1e-20,
            convergence_mode="ddJ",
            enforce_nonnegativity=True,
            verbose_solver=False,
            output=tmp_path / "unfold_mlem.json",
            validate=False,
        )
    )

    assert unfold_written["payload"]["method"] == "mlem"
    assert unfold_written["payload"]["flux"] == [1.2, 0.8]
    assert unfold_written["payload"]["diagnostics"]["iterations"] == 7
    assert unfold_written["payload"]["diagnostics"]["convergence_mode"] == "ddJ"
    assert unfold_written["payload"]["diagnostics"]["measured_rates"] == [1.0]
    assert unfold_written["payload"]["diagnostics"]["predicted_rates"]
    assert unfold_written["payload"]["diagnostics"]["rate_pulls"]
