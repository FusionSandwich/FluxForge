import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.dialogs import UnfoldingWorkspaceDialog  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from fluxforge.io import write_reaction_rates  # noqa: E402

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402
    from PySide6.QtWidgets import QScrollArea  # noqa: E402


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
    assert (
        dialog.measurements_table.rowCount()
        == dialog.workspace_input.measured_rates.size
    )
    assert dialog.response_image.image is not None
    assert "uncertainties are visible" in dialog.summary_label.text().lower()
    assert dialog.show_uncertainty_bands_checkbox.isEnabled() is True
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
def test_unfolding_workspace_dialog_loads_analytical_response_and_updates_heatmap(
    tmp_path,
):
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
    assert (
        dialog.measurements_table.rowCount()
        == dialog.workspace_input.measured_rates.size
    )
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_controls_fit_common_desktop_width_and_scroll_vertically():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    dialog.resize(1280, 700)
    dialog.show()
    _qapp().processEvents()

    controls = dialog.findChild(QScrollArea, "UnfoldingControlsScrollArea")
    assert controls is not None
    assert dialog.minimumSizeHint().width() <= 1280
    assert controls.horizontalScrollBar().maximum() == 0
    assert controls.verticalScrollBar().maximum() >= 0
    viewport = controls.viewport()
    for widget in (
        dialog.response_load_button,
        dialog.rates_load_button,
        dialog.log_energy_checkbox,
        dialog.log_flux_checkbox,
        dialog.reset_plots_button,
        dialog.run_button,
        dialog.compare_button,
    ):
        top_left = widget.mapTo(viewport, widget.rect().topLeft())
        assert top_left.x() >= 0
        assert top_left.x() + widget.width() <= viewport.width()
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_loads_real_rate_artifact_and_preserves_rates(tmp_path):
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    rates_path = tmp_path / "rates.json"
    expected = [float(index + 2) for index in range(8)]
    write_reaction_rates(
        rates_path,
        rates=[
            {
                "reaction_id": f"wire-{index + 1}",
                "rate": value,
                "uncertainty": value * 0.05,
            }
            for index, value in enumerate(expected)
        ],
    )
    dialog.rates_path_input.setText(str(rates_path))
    QTest.mouseClick(dialog.rates_load_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.workspace_input.measured_rates.tolist() == expected
    assert dialog.measurements_table.item(0, 0).text() == "wire-1"

    dialog.response_source_combo.setCurrentIndex(
        dialog.response_source_combo.findData("analytical_hpge")
    )
    QTest.mouseClick(dialog.response_load_button, Qt.LeftButton)
    _qapp().processEvents()
    assert dialog.workspace_input.measured_rates.tolist() == expected
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_runs_bundled_uwnr_rafm_rate_csv():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    rates_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "RAFM_irradiation"
        / "results"
        / "tables"
        / "flux_wire_reaction_rates.csv"
    )

    dialog.rates_path_input.setText(str(rates_path))
    QTest.mouseClick(dialog.rates_load_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.workspace_input.label == "UWNR RAFM Simplified Flux-Wire Response"
    assert dialog.workspace_input.measured_rates.size == 18
    assert dialog.workspace_input.response_matrix.shape == (18, 20)
    assert dialog.results_table.rowCount() == 20
    assert dialog.current_result is not None
    assert "completed" in dialog.summary_label.text().lower()
    assert dialog.response_image.image is not None
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt unfolding workspace dependencies are unavailable.",
)
def test_unfolding_workspace_uses_physical_energy_centers_and_plot_controls():
    _qapp()
    dialog = UnfoldingWorkspaceDialog(mode_manager=ModeManager())
    result_curve = next(
        item
        for item in dialog.flux_plot.listDataItems()
        if item.name() == dialog.current_result.method_used
    )
    expected_centers = np.sqrt(
        dialog.workspace_input.energy_edges[:-1]
        * dialog.workspace_input.energy_edges[1:]
    )
    np.testing.assert_allclose(result_curve.xData, expected_centers)

    dialog.log_energy_checkbox.setChecked(True)
    dialog.log_flux_checkbox.setChecked(True)
    QTest.mouseClick(dialog.reset_plots_button, Qt.LeftButton)
    _qapp().processEvents()
    assert dialog.flux_plot.getPlotItem().ctrl.logXCheck.isChecked()
    assert dialog.flux_plot.getPlotItem().ctrl.logYCheck.isChecked()
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
        load_example=True,
    )
    window._open_unfolding_workspace()
    _qapp().processEvents()

    assert window._unfolding_dialog is not None
    assert window._unfolding_dialog.windowTitle() == "FluxForge — Unfolding Workspace"
    assert window._unfolding_dialog.method_selector.combo.count() >= 4

    window._unfolding_dialog.close()
    window.close()
