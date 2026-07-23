import json
import os
from pathlib import Path

import pytest

from fluxforge.gui import QT_AVAILABLE, FluxForgeMainWindow, ModeManager, SelectionBus
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.dialogs.calibration_dialog import CalibrationWorkspaceDialog
from fluxforge.gui.dialogs.efficiency_dialog import EfficiencyCalibrationDialog
from fluxforge.gui.dialogs.unfolding_dialog import UnfoldingWorkspaceDialog
from fluxforge.gui.panels.modern_shell_shared import build_demo_spectrum
from fluxforge.gui.production_contract import (
    ActionCatalogEntry,
    ActionKind,
    CopyScope,
    ProductionCopy,
    assert_production_copy,
    assert_valid_action_catalog,
)
from fluxforge.gui.qt_compat import QApplication
from fluxforge.standards import QAMonitor

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if QT_AVAILABLE:
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QTabBar,
        QTabWidget,
        QTableWidget,
        QTextEdit,
        QToolButton,
        QWidget,
    )


ROOT = Path(__file__).resolve().parents[1]


class MemorySettings:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value) -> None:
        self.values[key] = value

    def sync(self) -> None:
        return None


def _qapp():
    return QApplication.instance() or QApplication([])


def _window(
    *,
    developer_tools: bool = False,
    load_example: bool = False,
    qa_monitor: QAMonitor | None = None,
):
    return FluxForgeMainWindow(
        mode_manager=ModeManager(settings=MemorySettings()),
        selection_bus=SelectionBus(),
        settings=MemorySettings(),
        qa_monitor=qa_monitor,
        developer_tools=developer_tools,
        load_example=load_example,
    )


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_production_startup_is_empty_and_examples_are_explicit(tmp_path):
    app = _qapp()
    window = _window(qa_monitor=QAMonitor(tmp_path / "qa.sqlite"))
    window.show()
    app.processEvents()

    assert window.analysis_workspace.state.spectra == ()
    assert window.analysis_workspace.state.loaded_spectra == ()
    assert window.analysis_workspace.spectrum() is None
    assert window.file_label.text() == "File: none"
    assert window.qa_monitor.history() == ()
    assert (
        "No QA or standards result" in window.left_dock.widget().qa_note.toPlainText()
    )

    actions = {action.text(): action for action in window.findChildren(QAction)}
    actions["Open Example"].trigger()
    app.processEvents()

    assert len(window.analysis_workspace.state.loaded_spectra) == 3
    assert window.analysis_workspace.spectrum() is not None
    assert window.file_label.text() == "File: bundled HPGe example"
    assert len(window.qa_monitor.history()) == 3

    window._reset_analysis_workspace()
    app.processEvents()
    assert window.analysis_workspace.spectrum() is None
    assert window.central_tabs.current_spectrum() is None
    assert window.file_label.text() == "File: none"
    assert window.qa_monitor.history() == ()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_empty_production_workspace_does_not_open_demo_analysis_dialogs():
    app = _qapp()
    window = _window()
    window.show()
    app.processEvents()

    actions = {action.objectName(): action for action in window.findChildren(QAction)}
    for action_name in (
        "AutoFindPeaksAction",
        "OpenEnergyFwhmCalibrationAction",
        "OpenSpectrumUnfoldingAction",
        "OpenPuIsotopicsAction",
        "RunAstmCheckAction",
        "WorkspaceOpenStandardsReviewAction",
    ):
        assert actions[action_name].isEnabled() is False
    assert window.left_dock.widget().astm_check_button.isEnabled() is False

    window._open_energy_fwhm_workspace()
    window._open_unfolding_workspace()
    window._open_standards_review()
    app.processEvents()
    assert window._calibration_dialog is None
    assert window._unfolding_dialog is None
    assert window._standards_review_dialog is None
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_astm_review_enables_only_after_data_load_and_uses_workspace_values(tmp_path):
    app = _qapp()
    window = _window(qa_monitor=QAMonitor(tmp_path / "qa.sqlite"))
    window.show()
    app.processEvents()

    with pytest.raises(RuntimeError, match="Load a spectrum"):
        window._build_standards_context()

    actions = {action.objectName(): action for action in window.findChildren(QAction)}
    actions["OpenExampleAction"].trigger()
    app.processEvents()

    assert actions["RunAstmCheckAction"].isEnabled()
    assert actions["WorkspaceOpenStandardsReviewAction"].isEnabled()
    assert window.left_dock.widget().astm_check_button.isEnabled()

    context = window._build_standards_context()
    spectrum = window.analysis_workspace.spectrum()
    assert spectrum is not None
    assert context.calibration_order == len(spectrum.calibration["energy"]) - 1
    assert context.max_residual_keV is None
    assert context.efficiency_uncertainty_pct is None
    assert context.fwhm_at_413_keV is None
    assert context.net_counts == {}
    assert context.extra["spectrum_id"] == spectrum.spectrum_id
    assert context.extra["live_time_s"] == pytest.approx(spectrum.live_time)
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_polluted_active_workflow_cannot_restore_spectrum_path_until_load(tmp_path):
    app = _qapp()
    settings = MemorySettings()
    spectrum_path = (
        ROOT
        / "examples"
        / "RAFM_irradiation"
        / "raw_gamma_spec"
        / "flux_wires"
        / "Ti-RAFM-1a_25cm.ASC"
    )
    assert spectrum_path.exists()

    first = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
        qa_monitor=QAMonitor(tmp_path / "first-qa.sqlite"),
    )
    first.open_path(spectrum_path)
    first.workflow_presets.save_workflow(
        "polluted-path-workflow",
        first._snapshot_current_workflow(),
        description="Contains a persisted local spectrum path",
    )
    first.close()

    second = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
        qa_monitor=QAMonitor(tmp_path / "second-qa.sqlite"),
    )
    second.show()
    app.processEvents()

    assert second.workflow_presets.active_workflow_name() == "polluted-path-workflow"
    assert second.workflow_combo.currentData() == "polluted-path-workflow"
    assert second.analysis_workspace.state.loaded_spectra == ()
    assert second.analysis_workspace.spectrum() is None
    assert second.file_label.text() == "File: none"

    second.load_workflow_button.click()
    app.processEvents()

    assert second.analysis_workspace.spectrum() is not None
    assert len(second.analysis_workspace.state.loaded_spectra) == 1
    assert second.analysis_workspace.state.loaded_spectra[0].source_path == str(
        spectrum_path
    )
    second.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_production_mode_hides_prototypes_and_developer_diagnostics():
    app = _qapp()
    window = _window()
    window.show()
    app.processEvents()

    bottom = window.bottom_dock.widget()
    labels = [bottom.tabText(index) for index in range(bottom.count())]
    action_labels = {
        action.text() for action in window.findChildren(QAction) if action.text()
    }
    assert not any("Prototype" in label for label in labels)
    assert "Parity Evidence" not in labels
    assert "Developer Log" not in labels
    assert "Renderer Diagnostics" not in action_labels
    assert "Hardware Diagnostics (not available)" not in action_labels
    assert not any("(not available)" in label for label in action_labels)
    assert window.progress.isVisible() is False
    assert window.hardware_led.isVisible() is False
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_developer_mode_exposes_truthfully_labeled_prototypes():
    app = _qapp()
    window = _window(developer_tools=True)
    window.show()
    app.processEvents()

    bottom = window.bottom_dock.widget()
    labels = [bottom.tabText(index) for index in range(bottom.count())]
    assert {
        "Inventory Prototype",
        "Masking Prototype",
        "Optimization Prototype",
        "Second-Irradiation Prototype",
        "Parity Evidence",
        "Developer Log",
    } <= set(labels)
    assert window.hardware_led.isVisible()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_production_actions_have_stable_ids_and_covered_copy():
    app = _qapp()
    window = _window()
    window.show()
    app.processEvents()

    actions = [
        action
        for action in window.findChildren(QAction)
        if action.text().strip() and not action.isSeparator()
    ]
    object_names = [action.objectName() for action in actions]
    assert object_names
    assert all(object_names)
    assert len(object_names) == len(set(object_names))
    assert_production_copy(
        ProductionCopy(action.text(), source=action.objectName()) for action in actions
    )
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_machine_readable_action_catalog_matches_all_production_controls():
    app = _qapp()
    window = _window()
    window.show()
    dialogs = (
        CalibrationWorkspaceDialog(
            spectrum=build_demo_spectrum(),
            mode_manager=window.mode_manager,
            selection_bus=window.selection_bus,
            parent=window,
        ),
        EfficiencyCalibrationDialog(
            mode_manager=window.mode_manager,
            points=(),
            parent=window,
        ),
        UnfoldingWorkspaceDialog(
            mode_manager=window.mode_manager,
            parent=window,
        ),
    )
    for dialog in dialogs:
        dialog.show()
    app.processEvents()
    manifest = json.loads(
        (ROOT / ".github/project-management/gui_action_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "fluxforge.gui_action_catalog"
    assert manifest["version"] == 1

    bottom = window.bottom_dock.widget()
    developer_roots = {
        bottom.inventory_timeline_panel,
        bottom.masking_review_panel,
        bottom.optimization_workspace_panel,
        bottom.second_irradiation_panel,
        bottom.phase5_parity_panel,
    }

    def is_developer_descendant(widget) -> bool:
        current = widget
        while current is not None:
            if current in developer_roots:
                return True
            current = (
                current.parentWidget() if hasattr(current, "parentWidget") else None
            )
        return False

    actual_actions = [
        action.objectName()
        for action in window.findChildren(QAction)
        if action.text().strip()
        and not action.isSeparator()
        and not is_developer_descendant(
            action.parent() if isinstance(action.parent(), QWidget) else None
        )
    ]
    expected_actions = [
        object_name
        for group in manifest["action_groups"]
        for object_name in group["object_names"]
    ]
    assert "" not in actual_actions
    assert len(actual_actions) == len(set(actual_actions))
    assert len(expected_actions) == len(set(expected_actions))
    assert set(actual_actions) == set(expected_actions)

    interactive_types = (
        QAbstractButton,
        QLineEdit,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QSlider,
        QPlainTextEdit,
        QTextEdit,
    )
    actual_controls = []
    for control in window.findChildren(QWidget):
        if not isinstance(control, interactive_types):
            continue
        if is_developer_descendant(control):
            continue
        if isinstance(control, (QPlainTextEdit, QTextEdit)) and control.isReadOnly():
            continue
        if control.objectName().startswith("qt_"):
            continue
        if isinstance(control, QLineEdit) and isinstance(
            control.parent(), QAbstractSpinBox
        ):
            continue
        if isinstance(control, QToolButton) and control.defaultAction() is not None:
            continue
        if isinstance(control, QAbstractButton) and isinstance(
            control.parent(), QTabBar
        ):
            continue
        actual_controls.append(control.objectName())
    expected_controls = [
        object_name
        for surface in manifest["control_surfaces"]
        for object_name in surface["object_names"]
    ]
    assert "" not in actual_controls
    assert len(actual_controls) == len(set(actual_controls))
    assert len(expected_controls) == len(set(expected_controls))
    assert set(actual_controls) == set(expected_controls)

    actual_tab_ids = []
    for tabs in window.findChildren(QTabWidget):
        if is_developer_descendant(tabs):
            continue
        mapping = tabs.property("fluxforgeTabIds") or {}
        initial_index = tabs.currentIndex()
        for index in range(tabs.count()):
            label = tabs.tabText(index)
            assert label in mapping
            tabs.setCurrentIndex(index)
            app.processEvents()
            assert tabs.currentWidget() is tabs.widget(index)
            actual_tab_ids.append(mapping[label])
        tabs.setCurrentIndex(initial_index)
    expected_tab_ids = [entry["action_id"] for entry in manifest["tabs"]]
    assert len(actual_tab_ids) == len(set(actual_tab_ids))
    assert len(expected_tab_ids) == len(set(expected_tab_ids))
    assert set(actual_tab_ids) == set(expected_tab_ids)

    actual_catalog_ids = actual_actions + actual_controls + actual_tab_ids
    expected_catalog_ids = expected_actions + expected_controls + expected_tab_ids
    assert len(actual_catalog_ids) == len(set(actual_catalog_ids))
    assert len(expected_catalog_ids) == len(set(expected_catalog_ids))

    for section in ("action_groups", "control_surfaces"):
        for group in manifest[section]:
            assert group["test_ids"]
            for test_id in group["test_ids"]:
                test_path = ROOT / test_id
                assert test_path.is_file()
                assert "def test_" in test_path.read_text(encoding="utf-8")
    for section in ("tabs", "renderer_intents"):
        for entry in manifest[section]:
            assert entry["test_ids"]
            assert_valid_action_catalog(
                (
                    ActionCatalogEntry(
                        action_id=entry["action_id"],
                        kind=(
                            ActionKind.TAB
                            if section == "tabs"
                            else ActionKind.PLOT_TOOL
                        ),
                        label=entry["action_id"],
                        surface=section,
                        test_ids=tuple(entry["test_ids"]),
                        scope=CopyScope.PRODUCTION,
                    ),
                )
            )
    for dialog in dialogs:
        dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_production_visible_copy_contains_no_implementation_narration():
    app = _qapp()
    window = _window()
    window.show()
    app.processEvents()
    bottom = window.bottom_dock.widget()
    developer_roots = {
        bottom.inventory_timeline_panel,
        bottom.masking_review_panel,
        bottom.optimization_workspace_panel,
        bottom.second_irradiation_panel,
        bottom.phase5_parity_panel,
    }

    def is_developer_descendant(widget) -> bool:
        current = widget
        while current is not None:
            if current in developer_roots:
                return True
            current = current.parentWidget()
        return False

    copy = []
    for label in window.findChildren(QLabel):
        if label.text().strip() and not is_developer_descendant(label):
            copy.append(ProductionCopy(label.text(), source=label.objectName()))
    for button in window.findChildren(QAbstractButton):
        if button.text().strip() and not is_developer_descendant(button):
            copy.append(ProductionCopy(button.text(), source=button.objectName()))
    for tabs in window.findChildren(QTabWidget):
        if is_developer_descendant(tabs):
            continue
        for index in range(tabs.count()):
            copy.append(
                ProductionCopy(
                    tabs.tabText(index),
                    source=f"{tabs.objectName()}.tab[{index}]",
                )
            )
    for action in window.findChildren(QAction):
        if action.text().strip() and "Diagnostics" not in action.text():
            copy.append(ProductionCopy(action.text(), source=action.objectName()))
        if action.toolTip().strip() and "Diagnostics" not in action.text():
            copy.append(
                ProductionCopy(
                    action.toolTip(), source=f"{action.objectName()}.tooltip"
                )
            )
        if action.statusTip().strip() and "Diagnostics" not in action.text():
            copy.append(
                ProductionCopy(
                    action.statusTip(), source=f"{action.objectName()}.status"
                )
            )
    for group in window.findChildren(QGroupBox):
        if group.title().strip() and not is_developer_descendant(group):
            copy.append(ProductionCopy(group.title(), source=group.objectName()))
    for combo in window.findChildren(QComboBox):
        if is_developer_descendant(combo):
            continue
        for index in range(combo.count()):
            copy.append(
                ProductionCopy(
                    combo.itemText(index),
                    source=f"{combo.objectName()}.item[{index}]",
                )
            )
    for editor in window.findChildren(QLineEdit):
        if editor.placeholderText().strip() and not is_developer_descendant(editor):
            copy.append(
                ProductionCopy(
                    editor.placeholderText(),
                    source=f"{editor.objectName()}.placeholder",
                )
            )
    for table in window.findChildren(QTableWidget):
        if is_developer_descendant(table):
            continue
        for column in range(table.columnCount()):
            item = table.horizontalHeaderItem(column)
            if item is not None and item.text().strip():
                copy.append(
                    ProductionCopy(
                        item.text(),
                        source=f"{table.objectName()}.header[{column}]",
                    )
                )
    text_widgets = window.findChildren(QPlainTextEdit) + window.findChildren(QTextEdit)
    for text_widget in text_widgets:
        if text_widget.toPlainText().strip() and not is_developer_descendant(
            text_widget
        ):
            copy.append(
                ProductionCopy(
                    text_widget.toPlainText(),
                    source=text_widget.objectName(),
                )
            )

    assert_production_copy(copy)
    window.close()
