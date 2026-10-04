"""Native form editing, segment application and import/export behavior."""

import json
import os

import pytest

from fluxforge.gui.qt_compat import QT_AVAILABLE

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="Native Qt is unavailable")


def test_history_edit_apply_export_and_invalid_import(tmp_path):
    from PySide6.QtWidgets import QApplication, QTableWidgetItem
    from fluxforge.gui.dialogs.irradiation_history_dialog import (
        IrradiationHistoryDialog,
    )

    app = QApplication.instance() or QApplication([])
    dialog = IrradiationHistoryDialog()
    changes = []
    dialog.historyChanged.connect(changes.append)
    try:
        dialog.add_button.click()
        assert dialog.table.item(0, 2).text() == ""
        with pytest.raises(ValueError, match="Stop"):
            dialog.apply_history()
        dialog.set_rows([(0, 10, 1), (20, 30, 0.5)])
        dialog.apply_button.click()
        assert len(changes) == 1
        assert [(s.duration_s, s.relative_power) for s in changes[0]] == [
            (10, 1),
            (10, 0),
            (10, 0.5),
        ]
        path = tmp_path / "history.json"
        dialog.export_json(path)
        assert json.loads(path.read_text())["scientific_admission"] is False
        dialog.table.selectRow(1)
        dialog.remove_button.click()
        assert dialog.table.rowCount() == 1
        dialog.load_file(path)
        assert dialog.table.rowCount() == 2
        previous_rows = dialog.rows()
        path.write_text('{"schema": "bad"}')
        with pytest.raises(ValueError, match="editor JSON"):
            dialog.load_file(path)
        assert dialog.rows() == previous_rows
        dialog.table.setItem(1, 0, QTableWidgetItem("9"))
        with pytest.raises(ValueError, match="overlap"):
            dialog.export_json(path)
        assert path.read_text() == '{"schema": "bad"}'
        assert len(changes) == 1
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_history_menu_retains_edits_and_applied_segments(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication
    from fluxforge.gui.main_window import FluxForgeMainWindow

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        settings=QSettings(str(tmp_path / "gui.ini"), QSettings.IniFormat)
    )
    try:
        action = window.findChild(QAction, "OpenIrradiationHistoryAction")
        assert action is not None and action.isEnabled()
        action.trigger()
        dialog = window._irradiation_history_dialog
        dialog.set_rows([(0, 10, 0.25)])
        dialog.apply_button.click()
        assert window._irradiation_segments[0].relative_power == 0.25
        dialog.close()
        action.trigger()
        assert window._irradiation_history_dialog is dialog
        assert dialog.rows() == [("0.0", "10.0", "0.25")]
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
