"""Offscreen contract tests for the HPGe efficiency review dialog."""

from __future__ import annotations

import csv

import pytest

from fluxforge.gui.backends.pyqtgraph_backend import PYQTGRAPH_AVAILABLE
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt plots unavailable"
)


def _dialog():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    dialog = EfficiencyCalibrationDialog(mode_manager=ModeManager())
    return app, dialog


def test_fit_populates_measured_fitted_and_percent_review():
    app, dialog = _dialog()
    assert dialog.table.rowCount() == 0  # never silently seed on startup
    dialog._seed_demo_points()
    dialog._fit_model()
    app.processEvents()
    assert dialog.accepted_fit() is not None
    assert len(dialog.calibration_points()) == 5
    assert dialog.point_diagnostics.rowCount() == 5
    assert dialog.point_diagnostics.item(0, 3).text()
    assert dialog.point_diagnostics.item(0, 4).text()
    assert len(dialog.efficiency_plot.listDataItems()) >= 2
    assert len(dialog.residual_plot.listDataItems()) >= 1
    assert "Model comparison" in dialog.model_comparison.toPlainText()
    assert "Covariance:" in dialog.summary.text()
    dialog.close()


def test_invalid_edit_blocks_fit_without_skipping_row():
    _app, dialog = _dialog()
    dialog._seed_demo_points()
    dialog.table.item(2, 0).setText("invalid")
    dialog._fit_model()
    assert dialog.accepted_fit() is None
    assert dialog.calibration_points() == ()
    assert "Row 3" in dialog.summary.text()
    dialog.table.item(2, 0).setText("661.657")
    dialog._fit_model()
    assert dialog.accepted_fit() is not None
    dialog.table.item(2, 0).setText("662")
    assert dialog.accepted_fit() is None
    dialog.close()


def test_csv_contract_is_atomic_and_certificate_unsupported(tmp_path):
    _app, dialog = _dialog()
    dialog._seed_demo_points()
    before = dialog.table.item(0, 0).text()
    path = tmp_path / "points.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(dialog.CSV_COLUMNS)
        writer.writerow(["bad", 100, 10, 100, 1000, 0, 0.5, 0, 1])
    with pytest.raises(ValueError, match="line 2"):
        dialog.import_csv(path)
    assert dialog.table.item(0, 0).text() == before
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(dialog.CSV_COLUMNS)
        writer.writerow([661.7, 1000, 32, 100, 1000, 0.02, 0.8, 0.01, 1])
    dialog.import_csv(path)
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 9).text() == ""
    dialog.table.item(0, 9).setText("certificate-A")
    assert dialog._read_points()[0].energy_keV == pytest.approx(661.7)
    assert dialog._read_points()[0].activity_source_id == "certificate-A"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((*dialog.CSV_COLUMNS, dialog.CSV_SOURCE_COLUMN))
        writer.writerow(
            [661.7, 1000, 32, 100, 1000, 0.02, 0.8, 0.01, 1, "certificate-B"]
        )
    dialog.import_csv(path)
    assert dialog._read_points()[0].activity_source_id == "certificate-B"
    with pytest.raises(ValueError, match="Only CSV"):
        dialog.import_csv(tmp_path / "certificate.pdf")
    dialog.close()


def test_optional_source_csv_is_atomic_with_reordered_headers(tmp_path):
    _app, dialog = _dialog()
    dialog._seed_demo_points()
    original_energy = dialog.table.item(0, 0).text()
    columns = tuple(reversed((*dialog.CSV_COLUMNS, dialog.CSV_SOURCE_COLUMN)))
    row = dict(
        zip(dialog.CSV_COLUMNS, [661.7, 1000, 32, 100, 1000, 0.02, 0.8, 0.01, 1])
    )
    row[dialog.CSV_SOURCE_COLUMN] = "certificate-C"
    path = tmp_path / "source-points.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow({**row, "net_counts": "bad"})
    with pytest.raises(ValueError, match="line 3"):
        dialog.import_csv(path)
    assert dialog.table.item(0, 0).text() == original_energy
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    dialog.import_csv(path)
    assert dialog._read_points()[0].activity_source_id == "certificate-C"
    dialog.close()


def test_add_delete_and_stable_action_ids():
    _app, dialog = _dialog()
    dialog._add_row()
    assert dialog.table.rowCount() == 1
    dialog.table.selectRow(0)
    dialog._delete_selected_rows()
    assert dialog.table.rowCount() == 0
    for name in (
        "AddEfficiencyPointButton", "DeleteEfficiencyPointsButton",
        "ImportEfficiencyCsvButton", "EfficiencyMeasuredFittedPlot",
        "EfficiencyPercentResidualPlot", "EfficiencyPointDiagnosticsTable",
        "EfficiencyModelComparison", "EfficiencyFitDiagnosticsLabel",
    ):
        assert dialog.findChild(__import__("PySide6.QtCore", fromlist=["QObject"]).QObject, name)
    dialog.close()
