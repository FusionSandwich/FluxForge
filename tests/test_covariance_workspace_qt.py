"""Behavioral heatmap, menu and image-export checks for issue #44."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fluxforge.gui.qt_compat import QT_AVAILABLE

pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="Native Qt is unavailable")


def test_heatmap_load_switch_and_export(tmp_path):
    from PySide6.QtWidgets import QApplication
    from fluxforge.gui.dialogs.covariance_dialog import CovarianceDialog

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "activities.json"
    path.write_text(
        json.dumps(
            {
                "covariance": [[4, -3, 0], [-3, 9, 0], [0, 0, 0]],
                "labels": ["A", "B", "fixed"],
                "title": "Activities (Bq²)",
            }
        )
    )
    dialog = CovarianceDialog()
    try:
        assert not dialog.export_button.isEnabled()
        dialog.load_file(path)
        dialog.show()
        app.processEvents()
        assert dialog.table.item(0, 1).text() == "-0.5"
        assert dialog.table.item(2, 2).text() == "N/A"
        assert dialog.table.horizontalHeaderItem(0).text() == "A"
        dialog.view_combo.setCurrentIndex(1)
        app.processEvents()
        assert dialog.table.item(0, 1).text() == "-3"
        image = tmp_path / "covariance.png"
        dialog.export_png(image)
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 5000
        path.write_text(json.dumps({"covariance": None}))
        with pytest.raises(ValueError, match="unavailable"):
            dialog.load_file(path)
        assert dialog.table.item(0, 1).text() == "-3"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_tools_menu_reuses_covariance_dialog(tmp_path):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from fluxforge.gui.main_window import FluxForgeMainWindow

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        settings=QSettings(str(tmp_path / "gui.ini"), QSettings.IniFormat)
    )
    try:
        action = window.findChild(QAction, "OpenCovarianceAction")
        assert action is not None and action.isEnabled()
        action.trigger()
        app.processEvents()
        first = window._covariance_dialog
        first.set_covariance([[1]], ["Nuclear datum"])
        first.close()
        action.trigger()
        app.processEvents()
        assert window._covariance_dialog is first
        assert first.table.item(0, 0).text() == "1"
    finally:
        if window._covariance_dialog:
            window._covariance_dialog.close()
        window.close()
        window.deleteLater()
        app.processEvents()
