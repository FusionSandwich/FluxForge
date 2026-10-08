"""Slow rendering must leave Qt responsive and retain one captured snapshot."""

import os
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from fluxforge.gui.dialogs.report_export_dialog import ReportExportDialog  # noqa: E402
from fluxforge.reporting.engine import ReportingEngine  # noqa: E402


class BlockingEngine(ReportingEngine):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.calls = []
        self.fail = False

    def can_export_pdf(self):
        return True

    def template_backend_available(self):
        return True

    def render(self, name, context):
        return SimpleNamespace(html=f"<p>{context['value']}</p>")

    def _export(self, name, context, path):
        self.calls.append((get_ident(), name, context["value"], path))
        self.started.set()
        assert self.release.wait(3), "Test worker was not released"
        if self.fail:
            raise RuntimeError("test renderer failure")
        path.write_text(str(context["value"]), encoding="utf-8")
        return path

    export_html = _export
    export_pdf = _export

    def export_bundle(self, name, context, path, **options):
        return self._export(name, context, path)


@pytest.mark.parametrize(
    "operation, attribute",
    [
        ("export_html", "last_export_path"),
        ("export_pdf", "last_pdf_export_path"),
        ("generate_report", "last_bundle_path"),
    ],
)
def test_export_responsive_snapshot_bound_and_close_safe(
    tmp_path, wait_for_report_export, operation, attribute
):
    app = QApplication.instance() or QApplication([])
    engine = BlockingEngine()
    source = {"value": "original"}
    captures = []

    def capture(name):
        captures.append(get_ident())
        return source

    dialog = ReportExportDialog(engine=engine, context_factory=capture)
    dialog.show()
    captures.clear()
    dialog.path_input.setText(str(tmp_path / "original.html"))
    ticks = []
    timer = QTimer(dialog)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    try:
        getattr(dialog, operation)()
        assert dialog.worker is not None
        assert not dialog.generate_button.isEnabled()
        for _ in range(50):
            QTest.qWait(5)
            if engine.started.is_set() and ticks:
                break
        assert ticks and engine.started.is_set()
        source["value"] = "changed after capture"
        dialog.path_input.setText(str(tmp_path / "different.html"))
        getattr(dialog, operation)()
        assert captures == [get_ident()]
        event = QCloseEvent()
        dialog.closeEvent(event)
        assert not event.isAccepted()
        dialog.reject()
        assert dialog.isVisible()
        engine.release.set()
        wait_for_report_export(dialog)
        assert engine.calls[0][0] != get_ident()
        assert engine.calls[0][2] == "original"
        assert getattr(dialog, attribute).stem == "original"
        assert dialog.last_context["value"] == "original"
        assert dialog.generate_button.isEnabled()
    finally:
        engine.release.set()
        wait_for_report_export(dialog)
        dialog.close()
        app.processEvents()


def test_export_failure_restores_controls_for_retry(tmp_path, wait_for_report_export):
    app = QApplication.instance() or QApplication([])
    engine = BlockingEngine()
    engine.release.set()
    engine.fail = True
    dialog = ReportExportDialog(
        engine=engine, context_factory=lambda name: {"value": "captured"}
    )
    dialog.path_input.setText(str(tmp_path / "report.html"))
    try:
        dialog.export_html()
        wait_for_report_export(dialog)
        assert "test renderer failure" in dialog.export_status.text()
        assert dialog.last_export_path is None
        assert dialog.export_button.isEnabled()
        engine.fail = False
        dialog.export_html()
        wait_for_report_export(dialog)
        assert dialog.last_export_path.is_file()
    finally:
        wait_for_report_export(dialog)
        dialog.close()
        app.processEvents()
