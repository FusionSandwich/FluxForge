"""Behavioral coverage for the single supported desktop workflow."""

from pathlib import Path
import json
import os
import tomllib

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.main_window import FluxForgeMainWindow
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.selection_bus import SelectionBus
from fluxforge.gui.dialogs.unfolding_dialog import UnfoldingWorkspaceDialog
from fluxforge.io.artifacts import write_response_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def dispose_closed_windows():
    """Keep application-wide theme changes independent of earlier test windows."""
    def cleanup():
        if not QT_AVAILABLE:
            return
        app = QApplication.instance()
        if app is None:
            return
        from PySide6.QtCore import QCoreApplication, QEvent

        for widget in app.topLevelWidgets():
            if not widget.isVisible():
                widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    cleanup()
    yield
    cleanup()


class MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


def test_shipped_gui_and_native_bundle_use_qt():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]
    assert scripts["fluxforge-gui"] == "fluxforge.gui.app:main"
    assert "fluxforge-gui-legacy" not in scripts
    assert not (ROOT / "src/fluxforge_gui").exists()
    assert (ROOT / "archive/legacy_gui/src/fluxforge_gui/app.py").is_file()
    launcher = (ROOT / "tools/pyinstaller_launch_gui.py").read_text(encoding="utf-8")
    assert "from fluxforge.gui.app import main" in launcher


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt renderer unavailable"
)
def test_workflow_shortcuts_reveal_peak_review_and_follow_loaded_data(monkeypatch):
    from PySide6.QtWidgets import QToolButton
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog
    from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog

    monkeypatch.setattr(AutoPeakReviewDialog, "exec", lambda self: QDialog.Accepted)

    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
    )
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    buttons = {
        b.defaultAction().objectName(): b
        for b in window.analysis_workflow_toolbar.findChildren(QToolButton)
        if b.defaultAction()
    }
    assert not buttons["AutoFindPeaksAction"].isEnabled()
    assert buttons["OpenSpectrumUnfoldingAction"].isEnabled()
    window._load_example_workspace()
    app.processEvents()
    bottom = window.bottom_dock.widget()
    bottom.setCurrentWidget(bottom.roi_tools_panel)
    window.bottom_dock.hide()
    QTest.mouseClick(buttons["ReviewPeaksAction"], Qt.LeftButton)
    app.processEvents()
    assert window.bottom_dock.isVisible()
    assert bottom.currentWidget() is bottom.peak_table_panel
    QTest.mouseClick(buttons["AutoFindPeaksAction"], Qt.LeftButton)
    app.processEvents()
    assert bottom.currentWidget() is bottom.peak_table_panel
    assert window.analysis_workspace.state.peaks
    for button in buttons.values():
        point = button.mapTo(window, button.rect().topRight())
        assert 0 <= point.x() < window.width()
    window._reset_analysis_workspace()
    assert not buttons["ReviewPeaksAction"].isEnabled()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt renderer unavailable"
)
def test_analysis_forms_scroll_without_forcing_a_tall_window():
    from PySide6.QtWidgets import QScrollArea

    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
    )
    window.show()
    app.processEvents()
    bottom = window.bottom_dock.widget()
    assert bottom.minimumSizeHint().height() < 300
    scroll = bottom.peak_table_panel.findChild(QScrollArea, "AnalysisPanelScrollArea")
    assert scroll is not None
    scroll.resize(600, 240)
    app.processEvents()
    scroll.ensureWidgetVisible(bottom.peak_table_panel.peak_id_filter)
    assert scroll.verticalScrollBar().maximum() > 0
    assert bottom.currentWidget() is bottom.peak_table_panel
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt renderer unavailable"
)
def test_measured_data_unfolding_requires_matching_physical_response(tmp_path):
    app = QApplication.instance() or QApplication([])
    dialog = UnfoldingWorkspaceDialog(
        start_empty=True, mode_manager=ModeManager(settings=MemorySettings())
    )
    assert dialog.current_result is None
    assert not dialog.run_button.isEnabled()
    rates = tmp_path / "rates.json"
    rates.write_text(
        json.dumps(
            {
                "rates": [
                    {
                        "reaction_id": "Au-197(n,g)Au-198",
                        "rate": 40.0,
                        "uncertainty": 2.0,
                    },
                    {
                        "reaction_id": "Co-59(n,g)Co-60",
                        "rate": 20.0,
                        "uncertainty": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    dialog.rates_path_input.setText(str(rates))
    dialog._load_measured_rates()
    assert dialog.workspace_input.measured_rates.tolist() == [40.0, 20.0]
    assert not dialog.run_button.isEnabled()
    response = tmp_path / "response.json"
    labels = list(dialog.workspace_input.measurement_labels)
    write_response_bundle(
        response,
        matrix=[[1.0, 0.2], [0.1, 1.0]],
        reactions=labels[::-1],
        boundaries_eV=[1.0, 10.0, 100.0],
    )
    dialog.response_path_input.setText(str(response))
    dialog._load_selected_response_matrix()
    assert "row order" in dialog.summary_label.text()
    assert dialog.current_result is None
    write_response_bundle(
        response,
        matrix=[[1.0, 0.2], [0.1, 1.0]],
        reactions=labels,
        boundaries_eV=[1.0, 10.0, 100.0],
    )
    dialog._load_selected_response_matrix()
    app.processEvents()
    assert dialog.run_button.isEnabled()
    assert dialog.current_result is not None
    assert dialog.workspace_input.energy_unit == "eV"
    np.testing.assert_array_equal(dialog.workspace_input.measured_rates, [40.0, 20.0])
    np.testing.assert_array_equal(
        dialog.workspace_input.energy_edges, [1.0, 10.0, 100.0]
    )
    # A later rate import must not silently reuse a response in a different order.
    payload = json.loads(rates.read_text(encoding="utf-8"))
    payload["rates"].reverse()
    rates.write_text(json.dumps(payload), encoding="utf-8")
    dialog._load_measured_rates()
    assert "row order" in dialog.summary_label.text()
    np.testing.assert_array_equal(dialog.workspace_input.measured_rates, [40.0, 20.0])
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt renderer unavailable"
)
def test_empty_unfolding_imports_committed_rafm_rates():
    app = QApplication.instance() or QApplication([])
    dialog = UnfoldingWorkspaceDialog(
        start_empty=True, mode_manager=ModeManager(settings=MemorySettings())
    )
    dialog.rates_path_input.setText(
        str(
            ROOT
            / "examples/RAFM_irradiation/results/tables/flux_wire_reaction_rates.csv"
        )
    )
    dialog._load_measured_rates()
    app.processEvents()
    assert dialog.workspace_input.measured_rates.size == 18
    assert dialog.workspace_input.response_matrix.shape == (18, 20)
    assert "Simplified" in dialog.workspace_input.label
    assert dialog.current_result is not None
    dialog.close()
