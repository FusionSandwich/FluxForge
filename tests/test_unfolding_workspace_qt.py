import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.dialogs import UnfoldingWorkspaceDialog  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_runs_maxed_and_surfaces_uncertainties():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    dialog.method_selector.set_current_key("maxed")
    QTest.mouseClick(dialog.run_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.current_result is not None
    assert dialog.current_result.method_used == "MAXED"
    assert dialog.results_table.rowCount() == dialog.workspace_input.initial_flux.size
    assert dialog.results_table.item(0, 3).text() not in {"", "N/A"}
    assert dialog.measurements_table.rowCount() == dialog.workspace_input.measured_rates.size
    assert dialog.response_image.image is not None
    assert "uncertainties are visible" in dialog.summary_label.text().lower()
    assert dialog.show_uncertainty_bands_checkbox.isEnabled() is False
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_defaults_to_rmle_and_exposes_lambda_controls():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    assert dialog.method_selector.current_key() == "rmle"
    assert dialog.rmle_auto_checkbox.isChecked() is True
    assert dialog.rmle_penalty_combo.count() == 3
    assert dialog.rmle_lambda_spin.isEnabled() is False
    assert dialog.rmle_lambda_slider.isEnabled() is False
    assert dialog.method_selector.combo.count() == 4
    assert dialog.use_ml_seed_checkbox.isChecked() is False
    assert dialog.ml_seed_threshold_spin.value() == pytest.approx(0.6)
    assert dialog.show_uncertainty_bands_checkbox.isEnabled() is True

    dialog.rmle_auto_checkbox.setChecked(False)
    _qapp().processEvents()
    assert dialog.rmle_lambda_spin.isEnabled() is True
    assert dialog.rmle_lambda_slider.isEnabled() is True
    dialog.rmle_lambda_slider.setValue(250)
    _qapp().processEvents()
    assert dialog.rmle_lambda_spin.value() == pytest.approx(
        dialog._slider_to_lambda(250),
        rel=1e-2,
        abs=1e-4,
    )
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_supports_mouse_driven_algorithm_comparison():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    dialog.method_selector.set_current_key("rmle")
    dialog.compare_selector.set_current_key("gravel")
    dialog.compare_mode_checkbox.setChecked(True)
    _qapp().processEvents()

    QTest.mouseClick(dialog.compare_button, Qt.LeftButton)
    _qapp().processEvents()

    assert set(dialog.comparison_results) == {"rmle", "gravel"}
    assert dialog.comparison_table.rowCount() == 2
    assert dialog.negative_bin_label.text().lower().endswith("solution.")
    assert dialog.show_uncertainty_bands_checkbox.isChecked() is True
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_keeps_uncertainty_toggle_available_when_rmle_is_in_comparison():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    dialog.method_selector.set_current_key("maxed")
    dialog.compare_selector.set_current_key("rmle")
    dialog.compare_mode_checkbox.setChecked(True)
    _qapp().processEvents()

    assert dialog.show_uncertainty_bands_checkbox.isEnabled() is True
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_supports_ml_seed_selection_and_seeded_rmle():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    dialog.method_selector.set_current_key("ml_seed")
    dialog.ml_seed_threshold_spin.setValue(0.4)
    QTest.mouseClick(dialog.run_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.current_result is not None
    assert dialog.current_result.method_used == "ML Seed"
    assert "seed confidence" in dialog.summary_label.text().lower()

    dialog.method_selector.set_current_key("rmle")
    dialog.use_ml_seed_checkbox.setChecked(True)
    dialog.ml_seed_threshold_spin.setValue(0.4)
    QTest.mouseClick(dialog.run_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.current_result is not None
    assert dialog.current_result.method_used == "RMLE"
    assert dialog.current_result.parameters_used["seed_with_ml"] is True
    assert dialog.current_result.parameters_used["seed_accepted"] is True
    assert dialog.current_result.parameters_used["seed_confidence_score"] >= 0.4
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_dialog_loads_analytical_response_and_updates_heatmap(tmp_path):
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.show()
    _qapp().processEvents()

    dialog.response_source_combo.setCurrentIndex(
        dialog.response_source_combo.findData("analytical_hpge")
    )
    QTest.mouseClick(dialog.response_load_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.workspace_input.label == "Analytical HPGe"
    assert dialog.response_image.image is not None
    assert dialog.measurements_table.rowCount() == dialog.workspace_input.measured_rates.size
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_main_window_opens_unfolding_workspace_dialog():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window._open_unfolding_workspace()
    _qapp().processEvents()

    assert window._unfolding_dialog is not None
    assert window._unfolding_dialog.windowTitle() == "FluxForge Next - Unfolding Workspace"
    assert window._unfolding_dialog.method_selector.combo.count() >= 4

    window._unfolding_dialog.close()
    window.close()
