"""Real Qt acceptance path from efficiency dialog to detector profile."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fluxforge.gui.analysis_workspace import (
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
    SpectrumSlot,
)
from fluxforge.gui.backends.pyqtgraph_backend import PYQTGRAPH_AVAILABLE
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog
from fluxforge.gui.library_manager import DataLibraryManager
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.panels.modern_shell import ActivityResultsPanel
from fluxforge.gui.panels.modern_shell_shared import build_demo_spectrum
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus
from fluxforge.io.flux_wire import EfficiencyCalibration


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt plots unavailable"
)


def test_dialog_apply_persists_profile_and_one_undo_step() -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QUndoStack
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    spectrum = build_demo_spectrum()
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(SpectrumSlot("foreground", "Foreground", spectrum),)
        )
    )
    stack = QUndoStack()
    panel = ActivityResultsPanel(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        workspace_controller=controller,
        library_manager=DataLibraryManager(),
        undo_stack=stack,
    )
    panel.show()
    app.processEvents()
    assert panel.fit_efficiency_button.isEnabled()

    def complete_dialog() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, EfficiencyCalibrationDialog)
        dialog._seed_demo_points()
        dialog._fit_model()
        assert dialog.accepted_fit() is not None
        assert dialog.accept_button.isEnabled()
        dialog.accept()

    QTimer.singleShot(0, complete_dialog)
    panel.fit_efficiency_button.click()
    app.processEvents()

    profile = controller.active_detector_profile()
    assert profile is not None and profile.efficiency_model is not None
    assert len(profile.efficiency_model.points) == 5
    assert controller.state.efficiency_fit is not None
    assert controller.spectrum() is spectrum
    assert stack.count() == 1
    assert controller.document.workflow_state["analysis_invalidation"][
        "requires_reanalysis"
    ] is True

    stack.undo()
    assert controller.active_detector_profile() is None
    assert controller.state.efficiency_fit is None
    assert controller.spectrum() is spectrum

    stack.redo()
    assert controller.active_detector_profile() == profile
    assert controller.state.efficiency_fit is not None
    stack.clear()
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_malformed_saved_points_report_error_without_opening_dialog(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    controller = AnalysisWorkspaceController(
        AnalysisWorkspaceState(
            spectra=(SpectrumSlot("foreground", "Foreground", build_demo_spectrum()),)
        )
    )
    panel = ActivityResultsPanel(
        mode_manager=ModeManager(), selection_bus=SelectionBus(),
        workspace_controller=controller, library_manager=DataLibraryManager(),
    )
    from fluxforge.analysis.detector_calibration import EfficiencyPoint
    from fluxforge.core.analysis_workspace import fit_efficiency_model
    points = (
        EfficiencyPoint(100, 1000, 100, 1000, 0.5),
        EfficiencyPoint(300, 800, 100, 1000, 0.5),
        EfficiencyPoint(600, 500, 100, 1000, 0.5),
        EfficiencyPoint(1000, 300, 100, 1000, 0.5),
    )
    fit = fit_efficiency_model(points)
    profile = controller.proposed_efficiency_profile(
        fit, EfficiencyCalibration(), points=points,
    )
    malformed_model = replace(profile.efficiency_model, points=({"unexpected": 1},))
    monkeypatch.setattr(
        controller, "active_detector_profile",
        lambda: replace(profile, efficiency_model=malformed_model),
    )
    panel.fit_efficiency_button.click()
    app.processEvents()
    assert "Saved efficiency points are invalid" in panel.summary.text()
    panel.close()
