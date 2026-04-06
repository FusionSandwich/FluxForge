import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.core.calibration import (  # noqa: E402
    ASTM_E181_LOCKED_ORDER,
    EnergyCalibrationPoint,
    EnergyDeviationPair,
    FWHMCalibrationPoint,
    fit_energy_calibration,
    fit_fwhm_calibration,
    fit_quick_slider_calibration,
)
from fluxforge.core.peak_fitting import (  # noqa: E402
    available_background_models,
    peak_fitter_entries,
    register_builtin_peak_fitters,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.dialogs import CalibrationWorkspaceDialog  # noqa: E402
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog  # noqa: E402
from fluxforge.gui.library_manager import DataLibraryManager  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import GUIMode, ModeManager, ModeState  # noqa: E402
from fluxforge.gui.panels.modern_shell import build_demo_spectrum  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402
from fluxforge.gui.widgets import MethodSelectorWidget  # noqa: E402
from fluxforge.plugins import PluginRegistries  # noqa: E402

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    import pyqtgraph as pg  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402
    from PySide6.QtWidgets import QPushButton  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


def _plot_click_point(dialog, channel: float):
    counts = np.asarray(dialog._spectrum.counts, dtype=float)
    y_value = float(counts[int(round(channel))])
    scene_point = dialog.spectrum_plot.plotItem.vb.mapViewToScene(
        pg.Point(float(channel), y_value)
    )
    return dialog.spectrum_plot.mapFromScene(scene_point)


def test_fit_energy_calibration_locks_astm_order_and_flags_outliers():
    fit = fit_energy_calibration(
        [
            EnergyCalibrationPoint(channel=0.0, reference_energy_keV=0.0),
            EnergyCalibrationPoint(channel=500.0, reference_energy_keV=500.0),
            EnergyCalibrationPoint(channel=1000.0, reference_energy_keV=1000.0),
            EnergyCalibrationPoint(channel=1500.0, reference_energy_keV=1515.0),
        ],
        order=4,
        standard="ASTM E181",
    )

    assert fit.order == ASTM_E181_LOCKED_ORDER
    assert fit.locked_by is not None
    assert any(fit.out_of_tolerance)
    assert fit.chi_squared > 0.0


def test_fit_fwhm_calibration_returns_resolution_metrics():
    fit = fit_fwhm_calibration(
        [
            FWHMCalibrationPoint(energy_keV=121.78, fwhm_keV=0.95),
            FWHMCalibrationPoint(energy_keV=661.657, fwhm_keV=1.82),
            FWHMCalibrationPoint(energy_keV=1173.228, fwhm_keV=2.11),
        ]
    )

    assert fit.model == "sqrt_poly"
    assert len(fit.coefficients) == 3
    assert fit.fitted_fwhm_keV.shape == (3,)
    assert fit.rms_keV >= 0.0


def test_quick_slider_calibration_builds_linear_preview():
    preview = fit_quick_slider_calibration(
        anchor_channels=(100.0, 500.0),
        reference_energies_keV=(121.78, 661.657),
    )

    assert preview.slope_keV_per_channel == pytest.approx((661.657 - 121.78) / 400.0)
    assert preview.evaluate([100.0, 500.0]).tolist() == pytest.approx([121.78, 661.657])


def test_energy_calibration_accepts_deviation_pairs():
    fit = fit_energy_calibration(
        [
            EnergyCalibrationPoint(channel=100.0, reference_energy_keV=120.0),
            EnergyCalibrationPoint(channel=500.0, reference_energy_keV=660.0),
            EnergyCalibrationPoint(channel=900.0, reference_energy_keV=1180.0),
        ],
        order=2,
        deviation_pairs=(
            EnergyDeviationPair(energy_keV=660.0, correction_keV=0.25, label="mid"),
        ),
    )

    assert len(fit.deviation_pairs) == 1
    assert fit.correction_keV.shape == fit.fitted_keV.shape
    assert np.max(np.abs(fit.correction_keV)) >= 0.25 - 1e-6


def test_peak_fitter_registry_contains_gaussian_skew_and_bayesian_entries():
    entries = {entry.key: entry for entry in peak_fitter_entries()}

    assert "gaussian" in entries
    assert "gaussian_skew" in entries
    assert "bayesian_gaussian" in entries
    assert entries["gaussian"].metadata.recommended is True
    assert entries["gaussian"].metadata.standards_locked is True
    assert available_background_models("gaussian_skew") == ("linear", "constant")


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_method_selector_widget_tracks_modern_peak_fitter_registry():
    _qapp()
    registries = register_builtin_peak_fitters(PluginRegistries())
    manager = ModeManager()
    widget = MethodSelectorWidget(registries.peak_fitters, manager, title="Peak fitter")
    widget.show()
    _qapp().processEvents()

    assert widget.combo.count() == 3
    assert widget.current_key() == "gaussian"
    assert widget.badge_label.text() == "Recommended"

    widget.set_current_key("gaussian_skew")
    _qapp().processEvents()
    assert widget.current_key() == "gaussian_skew"
    assert widget.badge_label.text() == "Alternative"
    assert "skewed gaussian" in widget.detail_label.text().lower()

    manager.set_standard("ASTM E181")
    _qapp().processEvents()
    assert widget.combo.count() == 1
    assert widget.current_key() == "gaussian"
    assert widget.badge_label.text() == "🔒 Standards Locked"
    assert widget.combo.isEnabled() is False
    widget.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_efficiency_dialog_exposes_all_four_registered_models():
    _qapp()
    dialog = EfficiencyCalibrationDialog(
        mode_manager=ModeManager(),
    )
    dialog.show()
    _qapp().processEvents()

    keys = {
        dialog.method_selector.combo.itemData(index)
        for index in range(dialog.method_selector.combo.count())
    }
    assert {
        "log_poly_2",
        "log_poly_3",
        "gray_functional",
        "semi_empirical_hpge",
    }.issubset(keys)

    dialog.method_selector.set_current_key("semi_empirical_hpge")
    dialog._fit_model()
    _qapp().processEvents()

    assert dialog.accepted_fit() is not None
    assert dialog.accepted_fit().model_key == "semi_empirical_hpge"
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_locks_astm_order_and_applies_to_workspace_spectrum():
    _qapp()
    manager = ModeManager(
        initial_state=ModeState(
            mode=GUIMode.STANDARDS,
            standard="ASTM E181",
            theme="dark",
        )
    )
    spectrum = build_demo_spectrum()
    bus = SelectionBus()
    applied = []

    dialog = CalibrationWorkspaceDialog(
        spectrum=spectrum,
        mode_manager=manager,
        selection_bus=bus,
        on_apply=lambda spec, energy_fit, fwhm_fit: applied.append(
            (spec, energy_fit, fwhm_fit)
        ),
    )
    _qapp().processEvents()

    assert dialog.energy_order.value() == 2
    assert dialog.energy_order.isEnabled() is False
    assert dialog.energy_table.rowCount() == 3
    assert dialog.fwhm_table.rowCount() == 3
    assert dialog.apply_button.isEnabled() is True

    dialog._apply_workspace_results()

    assert len(applied) == 1
    applied_spectrum, energy_fit, fwhm_fit = applied[0]
    assert applied_spectrum.calibration["energy"] == pytest.approx(energy_fit.coefficients)
    assert isinstance(fwhm_fit.fitted_fwhm_keV, np.ndarray)
    assert bus.describe()["reference_lines_keV"]
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_main_window_opens_calibration_workspace_dialog():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window._open_energy_fwhm_workspace()
    _qapp().processEvents()

    assert window._calibration_dialog is not None
    assert (
        window._calibration_dialog.windowTitle()
        == "FluxForge Next - Unified Calibration Workspace"
    )
    assert window._calibration_dialog.energy_table.rowCount() >= 3

    window._calibration_dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_supports_mouse_peak_selection_and_library_assignment():
    _qapp()
    manager = ModeManager()
    bus = SelectionBus()
    dialog = CalibrationWorkspaceDialog(
        spectrum=build_demo_spectrum(),
        mode_manager=manager,
        selection_bus=bus,
        library_manager=DataLibraryManager(),
    )
    dialog.show()
    _qapp().processEvents()

    dialog.energy_table.selectRow(0)
    click_point = _plot_click_point(dialog, 1173.0)
    QTest.mouseClick(dialog.spectrum_plot.viewport(), Qt.LeftButton, Qt.NoModifier, click_point)
    _qapp().processEvents()

    channel = float(dialog.energy_table.item(0, 1).text())
    observed = float(dialog.energy_table.item(0, 2).text())
    assert channel == pytest.approx(1173.0, abs=20.0)
    assert observed == pytest.approx(1173.0, abs=20.0)
    assert bus.state.peak_energy_keV == pytest.approx(observed, abs=1.0)

    assert dialog.library_source_combo.count() >= 4
    dialog.library_search.setText("co")
    _qapp().processEvents()
    assert dialog.library_results.count() >= 1

    target_index = 0
    for index in range(dialog.library_results.count()):
        item = dialog.library_results.item(index)
        if "co" in item.text().lower():
            target_index = index
            break
    dialog.library_results.setCurrentRow(target_index)
    _qapp().processEvents()
    assert dialog.library_lines.count() >= 1

    line_index = 0
    for index in range(dialog.library_lines.count()):
        item = dialog.library_lines.item(index)
        if "1173" in item.text():
            line_index = index
            break
    dialog.library_lines.setCurrentRow(line_index)
    dialog._assign_selected_library_line()
    _qapp().processEvents()

    assert "co" in dialog.energy_table.item(0, 0).text().lower()
    assert float(dialog.energy_table.item(0, 3).text()) == pytest.approx(1173.228, abs=1.0)
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_main_window_exposes_manual_and_standards_workflows_and_library_selectors():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    assert sidebar.gamma_source_combo.count() >= 4
    assert sidebar.calibration_source_combo.count() >= 1
    assert sidebar.naa_source_combo.count() >= 1
    assert sidebar.dosimetry_source_combo.count() >= 1
    assert sidebar.activation_source_combo.count() >= 1
    assert "Identification" in sidebar.library_summary.toPlainText()

    bottom_tabs = window.bottom_dock.widget()
    bottom_tabs.setCurrentIndex(1)
    _qapp().processEvents()
    manual_button = bottom_tabs.findChild(QPushButton, "ManualCalibrationWorkflowButton")
    quick_button = bottom_tabs.findChild(QPushButton, "QuickSliderCalibrationWorkflowButton")
    standards_button = bottom_tabs.findChild(QPushButton, "StandardsCalibrationWorkflowButton")
    assert manual_button is not None
    assert quick_button is not None
    assert standards_button is not None

    QTest.mouseClick(manual_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window.mode_manager.state.mode is GUIMode.EXPERT
    assert window._calibration_dialog is not None
    assert window._calibration_dialog.energy_order.isEnabled() is True

    QTest.mouseClick(standards_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window.mode_manager.state.mode is GUIMode.STANDARDS
    assert window.mode_manager.state.standard == "ASTM E181"
    assert window._calibration_dialog.energy_order.isEnabled() is False

    QTest.mouseClick(quick_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window._calibration_dialog.advanced_tabs.currentWidget() is window._calibration_dialog.quick_slider_tab

    window._calibration_dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_exposes_advanced_tools_and_roi_fitting():
    _qapp()
    manager = ModeManager()
    dialog = CalibrationWorkspaceDialog(
        spectrum=build_demo_spectrum(),
        mode_manager=manager,
        selection_bus=SelectionBus(),
        library_manager=DataLibraryManager(),
    )
    dialog.show()
    _qapp().processEvents()

    dialog.advanced_tabs.setCurrentWidget(dialog.quick_slider_tab)
    dialog.quick_anchor_a_combo.setCurrentIndex(0)
    dialog.quick_anchor_b_combo.setCurrentIndex(dialog.quick_anchor_b_combo.count() - 1)
    dialog.quick_anchor_a_slider.setValue(662)
    dialog.quick_anchor_b_slider.setValue(1332)
    _qapp().processEvents()
    assert dialog._quick_fit is not None
    dialog.quick_promote_button.click()
    _qapp().processEvents()
    assert dialog.energy_table.item(0, 0).text() == "Quick Anchor A"

    dialog.energy_table.item(2, 3).setText("1515.0")
    _qapp().processEvents()
    dialog.seed_deviation_pairs_button.click()
    _qapp().processEvents()
    assert dialog.deviation_table.rowCount() >= 1
    assert len(dialog._energy_fit.deviation_pairs) >= 1

    dialog.advanced_tabs.setCurrentWidget(dialog.roi_fit_tab)
    dialog.roi_region.setRegion((1160.0, 1190.0))
    _qapp().processEvents()
    assert dialog._roi_fit is not None
    assert dialog._roi_fit.centroid_channel == pytest.approx(1173.0, abs=12.0)
    assert dialog.roi_method_selector.combo.count() >= 3
    dialog.roi_method_selector.set_current_key("gaussian_skew")
    _qapp().processEvents()
    assert dialog.roi_background_combo.count() >= 2

    dialog.energy_table.selectRow(1)
    dialog.apply_roi_energy_button.click()
    dialog.fwhm_table.selectRow(1)
    dialog.apply_roi_fwhm_button.click()
    _qapp().processEvents()
    assert float(dialog.energy_table.item(1, 1).text()) == pytest.approx(
        dialog._roi_fit.centroid_channel,
        abs=1.0,
    )
    assert float(dialog.fwhm_table.item(1, 2).text()) > 0.0

    manager.set_standard("ASTM E181")
    _qapp().processEvents()
    assert dialog.roi_method_selector.current_key() == "gaussian"
    assert dialog.roi_method_selector.combo.count() == 1
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_supports_preserve_slots_fine_tune_and_nasa_seed():
    _qapp()
    dialog = CalibrationWorkspaceDialog(
        spectrum=build_demo_spectrum(),
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        library_manager=DataLibraryManager(),
    )
    dialog.show()
    _qapp().processEvents()

    original_channel = float(dialog.energy_table.item(0, 1).text())
    dialog.energy_table.item(2, 3).setText("1515.0")
    _qapp().processEvents()
    dialog.seed_deviation_pairs_button.click()
    _qapp().processEvents()
    preserved_pair_count = dialog.deviation_table.rowCount()
    assert preserved_pair_count >= 1

    dialog.preserve_current_button.click()
    _qapp().processEvents()
    assert dialog.fine_tune_preserved_button.isEnabled() is True
    assert "Preserved calibration: Current workspace" in dialog.snapshot_summary.text()

    dialog.save_slot_button.click()
    _qapp().processEvents()
    assert "Stored the current calibration in detector slot" in dialog.snapshot_summary.text()
    assert dialog.load_slot_button.isEnabled() is True

    dialog.energy_table.item(0, 1).setText("25.0")
    dialog.clear_deviation_pairs_button.click()
    _qapp().processEvents()
    assert dialog.deviation_table.rowCount() == 0

    dialog.load_slot_button.click()
    _qapp().processEvents()
    assert float(dialog.energy_table.item(0, 1).text()) == pytest.approx(
        original_channel,
        abs=1.0,
    )
    assert dialog.deviation_table.rowCount() == preserved_pair_count

    dialog.energy_table.item(0, 1).setText("40.0")
    _qapp().processEvents()
    dialog.fine_tune_preserved_button.click()
    _qapp().processEvents()
    fine_tuned_channel = float(dialog.energy_table.item(0, 1).text())
    assert abs(fine_tuned_channel - 40.0) >= 1.0
    assert abs(fine_tuned_channel - original_channel) <= 48.0

    dialog.nasa_smart_seed_button.click()
    _qapp().processEvents()
    assert "NASA smart seed refreshed the calibration anchors" in dialog.snapshot_summary.text()
    assert dialog.energy_table.rowCount() >= 3
    assert "Cs-137" in dialog.energy_table.item(0, 0).text()
    dialog.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_apply_workspace_results_persists_calibration_workspace_state_to_spectrum():
    _qapp()
    spectrum = build_demo_spectrum()
    bus = SelectionBus()
    dialog = CalibrationWorkspaceDialog(
        spectrum=spectrum,
        mode_manager=ModeManager(),
        selection_bus=bus,
        library_manager=DataLibraryManager(),
    )
    dialog.show()
    _qapp().processEvents()

    dialog.advanced_tabs.setCurrentWidget(dialog.quick_slider_tab)
    dialog.quick_anchor_a_slider.setValue(662)
    dialog.quick_anchor_b_slider.setValue(1332)
    dialog.quick_promote_button.click()
    _qapp().processEvents()

    dialog.energy_table.item(2, 3).setText("1515.0")
    _qapp().processEvents()
    dialog.seed_deviation_pairs_button.click()
    _qapp().processEvents()

    dialog.advanced_tabs.setCurrentWidget(dialog.roi_fit_tab)
    dialog.roi_region.setRegion((1160.0, 1190.0))
    _qapp().processEvents()

    dialog._apply_workspace_results()

    assert spectrum.calibration["energy"] == pytest.approx(dialog._energy_fit.coefficients)
    assert spectrum.calibration["deviation_pairs"]
    assert bus.state.roi_bounds_keV is not None
    assert spectrum.energies is not None
    dialog.close()


def test_data_library_manager_tracks_gui_library_categories():
    manager = DataLibraryManager()

    gamma_ids = {record.source_id for record in manager.available_sources("gamma_identification")}
    activation_ids = {record.source_id for record in manager.available_sources("activation")}
    assert len(gamma_ids) >= 4
    assert {
        "decay_2012",
        "fluxforge_bundled_gamma",
        "actigamma_2012",
        "nndc_offline_activation",
        "custom_gamma_file",
    }.issubset(gamma_ids)
    assert manager.available_sources("calibration")[0].source_id == "calibration_standard_sources"
    assert manager.available_sources("naa_monitor")[0].source_id == "k0_naa_monitors"
    assert manager.available_sources("dosimetry")[0].source_id == "irdff_ii_dosimetry"
    assert "flux_wire_catalog" in activation_ids
    assert "nasa_capture_iaea" in activation_ids


def test_data_library_manager_registers_and_removes_user_sources(monkeypatch, tmp_path):
    registry_path = tmp_path / "library_registry.json"
    gamma_path = tmp_path / "gamma.csv"
    gamma_path.write_text(
        "nuclide,energy_keV,intensity,half_life_s\nCo60,1332.5,1.0,166344192.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXFORGE_LIBRARY_REGISTRY", str(registry_path))

    manager = DataLibraryManager()
    record = manager.register_user_gamma_source("Lab Ref", str(gamma_path))

    gamma_ids = {
        source.source_id for source in manager.available_sources("gamma_identification")
    }
    assert record.source_id == "user_gamma_lab_ref"
    assert record.source_id in gamma_ids
    assert manager.state.gamma_identification_source_id == "user_gamma_lab_ref"

    assert manager.remove_user_gamma_source("user_gamma_lab_ref") is True
    assert manager.state.gamma_identification_source_id == "fluxforge_bundled_gamma"


def test_data_library_manager_resolves_astm_locked_sources():
    manager = DataLibraryManager()

    standards_gamma = manager.available_sources(
        "gamma_identification",
        standard="ASTM E181",
    )
    standards_calibration = manager.available_sources(
        "calibration",
        standard="ASTM E181",
    )
    e261_dosimetry = manager.available_sources("dosimetry", standard="ASTM E261")

    assert [record.source_id for record in standards_gamma] == ["decay_2012"]
    assert [record.source_id for record in standards_calibration] == [
        "calibration_standard_sources"
    ]
    assert [record.source_id for record in e261_dosimetry] == ["irdff_ii_dosimetry"]
    assert (
        manager.resolved_state(standard="ASTM C1030").gamma_identification_source_id
        == "decay_2012"
    )


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt calibration workspace dependencies are unavailable.",
)
def test_calibration_dialog_locks_gamma_library_to_astm_sources():
    _qapp()
    manager = ModeManager(
        initial_state=ModeState(
            mode=GUIMode.STANDARDS,
            standard="ASTM E181",
            theme="dark",
        )
    )
    dialog = CalibrationWorkspaceDialog(
        spectrum=build_demo_spectrum(),
        mode_manager=manager,
        selection_bus=SelectionBus(),
        library_manager=DataLibraryManager(),
    )
    dialog.show()
    _qapp().processEvents()

    assert dialog.library_source_combo.count() == 1
    assert dialog.library_source_combo.currentData() == "decay_2012"
    assert dialog.library_source_combo.isEnabled() is False
    assert dialog.nuclide_controller.source_id == "decay_2012"
    assert "actigamma" in dialog.library_summary.text().lower()
    dialog.close()
