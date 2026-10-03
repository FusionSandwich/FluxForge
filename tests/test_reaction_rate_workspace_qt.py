"""Translator fields, found-isotope reference handling and current-history export."""

import json
import os
from types import SimpleNamespace

import pytest
from fluxforge.gui.qt_compat import QT_AVAILABLE

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="Native Qt is unavailable")


def test_found_rows_leave_measured_constraints_and_eoi_activity_blank(tmp_path):
    from PySide6.QtWidgets import QApplication, QTableWidgetItem
    from fluxforge.gui.dialogs.reaction_rate_dialog import ReactionRateDialog
    from fluxforge.physics.activation import IrradiationSegment

    app = QApplication.instance() or QApplication([])
    history = [IrradiationSegment(10, 1)]
    observed = SimpleNamespace(
        nuclide="Co60", line_energy_keV=1173.2, activity_bq=12, half_life_s=10
    )
    dialog = ReactionRateDialog(
        history_provider=lambda: history, activity_provider=lambda: [observed]
    )
    try:
        dialog.found_button.click()
        dialog.found_button.click()
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 8).text() == "12"
        assert all(dialog.table.item(0, col).text() == "" for col in (1, 2, 3, 4, 6, 7))
        with pytest.raises(ValueError):
            dialog.convert()
        for col, value in enumerate(("Co60", "Co", "100", "1", "1", "10", "12", "")):
            dialog.table.setItem(0, col, QTableWidgetItem(value))
        dialog.convert_button.click()
        assert dialog.result_table.item(0, 5).text() == "Unavailable"
        rate_before = dialog.current_payload["rows"][0]["sigphi_per_atom_s"]
        history[0].relative_power = 0.5
        path = tmp_path / "rates.json"
        dialog.export_json(path)
        payload = json.loads(path.read_text())
        assert payload["rows"][0]["sigphi_per_atom_s"] == pytest.approx(2 * rate_before)
        assert payload["rows"][0]["sigphi_activity_sigma"] is None
        dialog.table.setItem(0, 2, QTableWidgetItem(""))
        assert dialog.current_payload is None and dialog.result_table.rowCount() == 0
        before = path.read_bytes()
        with pytest.raises(ValueError):
            dialog.export_json(path)
        assert path.read_bytes() == before
        dialog.table.selectRow(0)
        dialog.remove_button.click()
        assert dialog.table.rowCount() == 0
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_main_window_history_changes_clear_stale_reaction_rates(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.physics.activation import IrradiationSegment

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        settings=QSettings(str(tmp_path / "gui.ini"), QSettings.IniFormat)
    )
    try:
        window.findChild(QAction, "OpenReactionRateAction").trigger()
        dialog = window._reaction_rate_dialog
        window._set_irradiation_history((IrradiationSegment(10, 1),))
        dialog.add_row(("Co60", "Co", "100", "1", "1", "10", "12", "3"))
        dialog.convert()
        assert dialog.current_payload is not None
        window._set_irradiation_history((IrradiationSegment(20, 1),))
        assert dialog.current_payload is None and dialog.result_table.rowCount() == 0
        dialog.close()
        window.findChild(QAction, "OpenReactionRateAction").trigger()
        assert window._reaction_rate_dialog is dialog
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
