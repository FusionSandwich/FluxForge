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


def _dummy_spectrum() -> GammaSpectrum:
    return GammaSpectrum(
        counts=np.array([0.0, 10.0, 0.0], dtype=float),
        channels=np.array([0, 1, 2], dtype=float),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [0.0, 1.0, 0.0]},
        spectrum_id="dummy",
    )


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
        ]
    )
    assert args.command == "spectrum-plot"
    assert args.background_subtracted is True


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
