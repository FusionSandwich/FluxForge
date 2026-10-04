"""Real worker-backed folder queue output and preservation of existing files."""

import os
import time

import pytest
from fluxforge.gui.qt_compat import QT_AVAILABLE

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="Native Qt is unavailable")


def test_active_conversion_remains_visible_through_escape_and_dialog_completion():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog
    from fluxforge.gui.dialogs.spectrum_file_queue_dialog import SpectrumFileQueueDialog

    app = QApplication.instance() or QApplication([])
    dialog = SpectrumFileQueueDialog()
    try:
        dialog.show()
        app.processEvents()
        dialog.worker = object()  # Exercise Qt dismissal routes while a job is active.
        QTest.keyClick(dialog, Qt.Key_Escape)
        app.processEvents()
        assert dialog.isVisible()
        for finish in (
            dialog.reject,
            dialog.accept,
            lambda: dialog.done(QDialog.Accepted),
        ):
            finish()
            app.processEvents()
            assert dialog.isVisible()
        assert "still running" in dialog.status_label.text()
        dialog.worker = None
        dialog.reject()
        assert not dialog.isVisible()
    finally:
        dialog.worker = None
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_queue_folder_worker_converts_and_preserves_outputs(tmp_path):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from fluxforge.gui.dialogs.spectrum_file_queue_dialog import SpectrumFileQueueDialog
    from fluxforge.io.session import read_ffs_session

    app = QApplication.instance() or QApplication([])
    for name in ("a", "b"):
        (tmp_path / f"{name}.csv").write_text("channel,counts\n0,2\n1,3\n")
    dialog = SpectrumFileQueueDialog()
    try:
        dialog.add_folder(tmp_path)
        dialog.add_folder(tmp_path)
        assert len(dialog.paths) == 2
        dialog.table.selectRow(1)
        dialog.remove_button.click()
        assert len(dialog.paths) == 1
        dialog.add_folder(tmp_path)
        dialog.mode_combo.setCurrentIndex(1)
        path = tmp_path / "sum.ffs"
        dialog.output_input.setText(str(path))
        dialog.run_button.click()
        assert dialog.worker is None and not path.exists()
        assert "independent" in dialog.status_label.text()
        dialog.independent_check.setChecked(True)
        dialog.run_button.click()
        assert dialog.worker is not None and not dialog.run_button.isEnabled()
        deadline = time.monotonic() + 10
        while dialog.worker is not None and time.monotonic() < deadline:
            QTest.qWait(20)
            app.processEvents()
        assert dialog.worker is None
        assert dialog.last_result["output_spectra"] == 1
        assert read_ffs_session(path).spectra[0].counts.tolist() == [4, 6]
        before = path.read_bytes()
        dialog.run_button.click()
        assert "exists" in dialog.status_label.text()
        assert path.read_bytes() == before
    finally:
        if dialog.worker is not None:
            dialog.worker.wait(10000)
            app.processEvents()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
