"""Run bundle integrity and the supported Qt capture/export workflow."""

import base64
from copy import deepcopy
from hashlib import sha256
import json
import os
from zipfile import ZipFile

import pytest

from fluxforge.reporting.engine import ReportingEngine

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _context():
    pixels = b"captured PNG bytes"
    snapshot = {
        "schema": "fluxforge.gui_report_snapshot.v1",
        "captured_at": "2026-10-03T12:00:00+00:00",
        "scientific_admission": False,
        "parameters": {"background_scale": 0.42},
        "inputs": [{"control": "SampleName", "value": "<script>bad</script>"}],
        "tables": [
            {
                "control": "PeakTable",
                "headers": ["Nuclide"],
                "rows": [["<img src=x onerror=bad>"]],
            }
        ],
        "views": [
            {
                "control": "SpectrumPlot",
                "range": [[20, 40], [0, 80]],
                "log_x": False,
                "log_y": True,
                "png_base64": base64.b64encode(pixels).decode(),
                "sha256": sha256(pixels).hexdigest(),
                "asset": "views/000.png",
            }
        ],
        "workspace": {"document_id": "recorded"},
    }
    context = {
        key: ""
        for key in ReportingEngine().templates["standard_lab"].required_context_keys
    }
    context.update(
        title="FluxForge Module 3 Report",
        run_snapshot=snapshot,
        run_parameters_json=json.dumps(snapshot["parameters"]),
    )
    return context


def test_bundle_has_matching_typed_snapshot_assets_and_manifest(tmp_path):
    context = _context()
    before = deepcopy(context)
    output = ReportingEngine().export_bundle(
        "standard_lab", context, tmp_path / "run.zip"
    )
    assert context == before
    with ZipFile(output) as archive:
        snapshot = json.loads(archive.read("snapshot.json"))
        manifest = json.loads(archive.read("manifest.json"))
        html = archive.read("report.html").decode()
        assert snapshot["parameters"]["background_scale"] == 0.42
        assert "png_base64" not in snapshot["views"][0]
        assert "data:image/png;base64," in html
        assert "&lt;script&gt;bad&lt;/script&gt;" in html
        assert "<script>bad</script>" not in html
        assert "<img src=x onerror=bad>" not in html
        assert manifest["scientific_admission"] is False
        for name, receipt in manifest["files"].items():
            data = archive.read(name)
            assert sha256(data).hexdigest() == receipt["sha256"]
            assert len(data) == receipt["bytes"]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        ReportingEngine().export_bundle("standard_lab", context, output)
    assert output.read_bytes() == original


def test_bundle_rejects_tampered_plot_and_failed_pdf_before_creating_output(
    tmp_path, monkeypatch
):
    engine = ReportingEngine()
    context = _context()
    context["run_snapshot"]["views"][0]["sha256"] = "bad"
    target = tmp_path / "run.zip"
    with pytest.raises(ValueError, match="hash"):
        engine.export_bundle("standard_lab", context, target)
    assert not target.exists()

    def missing_backend():
        raise RuntimeError("PDF native backend unavailable")

    monkeypatch.setattr(
        "fluxforge.reporting.engine._load_weasyprint_html", missing_backend
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.export_bundle("standard_lab", _context(), target, include_pdf=True)
    assert not target.exists()


def test_qt_generate_report_preserves_view_and_captures_tables_once(
    tmp_path, monkeypatch, wait_for_report_export
):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.mode_manager import ModeManager
    from fluxforge.gui.selection_bus import SelectionBus

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(), selection_bus=SelectionBus(), load_example=True
    )
    window.show()
    app.processEvents()
    try:
        canvas = window.central_tabs.canvas
        canvas.set_log_scale(True)
        canvas.plot_item.setXRange(100, 400, padding=0)
        canvas.plot_item.setYRange(0, 3, padding=0)
        before = canvas.viewport_state().to_dict()
        window._open_report_export()
        dialog = window._report_dialog
        captures = []
        factory = dialog.context_factory

        def counted_factory(name):
            captured = factory(name)
            captures.append(captured)
            return captured

        monkeypatch.setattr(dialog, "context_factory", counted_factory)
        dialog.path_input.setText(str(tmp_path / "run.html"))
        QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
        wait_for_report_export(dialog)
        assert dialog.last_bundle_path is not None, dialog.export_status.text()
        assert len(captures) == 1
        assert canvas.viewport_state().to_dict() == before
        with ZipFile(dialog.last_bundle_path) as archive:
            snapshot = json.loads(archive.read("snapshot.json"))
            assert snapshot["parameters"]["canvas"] == before
            view = next(
                item for item in snapshot["views"] if item["control"] == "SpectrumPlot"
            )
            assert archive.read(view["asset"]).startswith(b"\x89PNG\r\n\x1a\n")
            assert view["log_y"] is True
            assert snapshot["tables"]
            assert any(
                item["control"] == "PeakTableWidget" for item in snapshot["tables"]
            )
            assert not any(
                item["control"].startswith("Report") for item in snapshot["inputs"]
            )
        assert (
            dialog.last_context["run_snapshot"]["views"]
            == captures[0]["run_snapshot"]["views"]
        )
        assert (
            dialog.last_context["run_snapshot"]["captured_at"]
            == captures[0]["run_snapshot"]["captured_at"]
        )
        assert (
            dialog.last_context["run_snapshot"]["workspace"]
            == captures[0]["run_snapshot"]["workspace"]
        )
        QTest.mouseClick(dialog.generate_button, Qt.LeftButton)
        wait_for_report_export(dialog)
        assert dialog.last_bundle_path is None
        assert "failed" in dialog.export_status.text()
        dialog.close()
    finally:
        window.close()


def test_html_preview_and_export_use_one_context_capture(
    tmp_path, wait_for_report_export
):
    pytest.importorskip("PySide6")
    from fluxforge.gui.dialogs.report_export_dialog import ReportExportDialog
    from fluxforge.gui.qt_compat import QApplication

    app = QApplication.instance() or QApplication([])
    contexts = []

    def factory(name):
        context = _context()
        context["title"] = f"Capture {len(contexts)}"
        contexts.append(context)
        return context

    dialog = ReportExportDialog(context_factory=factory)
    try:
        contexts.clear()
        dialog.path_input.setText(str(tmp_path / "report.html"))
        dialog.export_html()
        wait_for_report_export(dialog)
        app.processEvents()
        assert len(contexts) == 1
        assert "Capture 0" in dialog.preview.toPlainText()
        assert "Capture 0" in dialog.last_export_path.read_text(encoding="utf-8")
        dialog.path_input.clear()
        dialog.export_html()
        assert dialog.last_export_path is None
        assert "Enter an export" in dialog.export_status.text()
    finally:
        dialog.close()
