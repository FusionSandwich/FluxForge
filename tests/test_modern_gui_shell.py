import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fluxforge.gui import (
    QT_AVAILABLE,
    FluxForgeMainWindow,
    GUIMode,
    HierarchicalSpectrumBuffer,
    ModeManager,
    SelectionBus,
    WorkflowPresetManager,
    describe_gui_scaffold,
    modern_gui_unavailable_message,
    register_builtin_render_backends,
)
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, PyQtGraphSpectrumCanvas
from fluxforge.gui.qt_compat import QApplication
from fluxforge.gui.spectrum_canvas import ReferenceLine, SpectrumTrace
from fluxforge.plugins import PluginRegistries

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest


class FakeSettings:
    def __init__(self) -> None:
        self._values = {}
        self.sync_count = 0

    def value(self, key, default=None):
        return self._values.get(key, default)

    def setValue(self, key, value) -> None:
        self._values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


def _qapp():
    return QApplication.instance() or QApplication([])


def test_describe_gui_scaffold_exposes_primary_and_legacy_entrypoints():
    scaffold = describe_gui_scaffold()

    assert scaffold["modern_entrypoint"] == "fluxforge-gui"
    assert scaffold["legacy_entrypoint"] == "fluxforge-gui-legacy"
    assert scaffold["renderer_backends"][0]["key"] == "pyqtgraph"


def test_mode_manager_persists_state_with_settings_backend():
    settings = FakeSettings()
    manager = ModeManager(settings=settings)

    manager.save_theme_profile("warm-light", theme="light")
    manager.set_theme_profile("warm-light")
    manager.set_standard("ASTM E181")

    assert settings._values["gui/mode"] == "standards"
    assert settings._values["gui/standard"] == "ASTM E181"
    assert settings._values["gui/theme"] == "light"
    assert settings._values["gui/theme_profile"] == "warm-light"
    assert settings.sync_count >= 2

    reloaded = ModeManager(settings=settings)
    assert reloaded.state.mode is GUIMode.STANDARDS
    assert reloaded.state.standard == "ASTM E181"
    assert reloaded.state.theme == "light"
    assert reloaded.state.theme_profile == "warm-light"


def test_mode_manager_theme_profile_selection_and_describe_payload():
    manager = ModeManager()

    manager.save_theme_profile("analysis-night", theme="dark")
    manager.set_theme_profile("analysis-night")

    assert "analysis-night" in manager.available_theme_profiles()
    assert manager.theme_for_profile("analysis-night") == "dark"
    assert manager.describe()["theme_profile"] == "analysis-night"


def test_workflow_preset_manager_persists_builtin_and_user_presets():
    settings = FakeSettings()
    manager = WorkflowPresetManager(settings=settings)

    assert "quantumgold-workflow" in manager.workflow_names()
    assert "astm-ldrd-irradiation" in manager.workflow_names()

    manager.save_workflow(
        "lab-phase6-session",
        {
            "version": 1,
            "mode_state": {"mode": "expert", "theme": "dark", "theme_profile": "night-lab"},
        },
        description="Local saved workflow",
    )

    assert manager.active_workflow_name() == "lab-phase6-session"
    reloaded = WorkflowPresetManager(settings=settings)
    assert "lab-phase6-session" in reloaded.workflow_names()
    assert reloaded.active_workflow_name() == "lab-phase6-session"
    saved = reloaded.get_workflow("lab-phase6-session")
    assert saved is not None
    assert saved.built_in is False
    assert saved.description == "Local saved workflow"


def test_workflow_preset_manager_mode_and_library_helpers_apply_payloads():
    settings = FakeSettings()
    manager = WorkflowPresetManager(settings=settings)

    captured: dict[str, dict[str, object]] = {}
    applied = manager.apply_mode_and_library_state(
        name="astm-ldrd-irradiation",
        mode_state_applier=lambda payload: captured.setdefault("mode", dict(payload)),
        library_state_applier=lambda payload: captured.setdefault("library", dict(payload)),
    )

    assert applied is not None
    assert applied.name == "astm-ldrd-irradiation"
    assert captured["mode"]["mode"] == "standards"
    assert captured["library"]["gamma_identification_source_id"] == "decay_2012"


def test_workflow_preset_manager_extract_mode_and_library_state_handles_missing_payloads():
    mode_state, library_state = WorkflowPresetManager.extract_mode_and_library_state({})

    assert mode_state == {}
    assert library_state == {}


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt GUI dependencies are unavailable.")
def test_main_window_restores_saved_workflow_state_across_sessions():
    _qapp()
    settings = FakeSettings()

    first = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
    )
    first.show()
    _qapp().processEvents()

    first._log_scale_action.setChecked(True)
    first._toggle_log_scale(True)
    sidebar = first.left_dock.widget()
    bottom = first.bottom_dock.widget()
    sidebar.apply_workflow_state(
        {
            "nuclide_query": "co",
            "nuclide_age_days": 1.5,
            "saved_nuclides": ["Co-60"],
            "mixture_entries": [{"nuclide": "Co-60", "weight": 2.5}],
            "reference_tab": "User List",
        }
    )
    bottom.activity_results_panel.apply_workflow_state(
        {
            "background_mode": "scaled",
            "background_scale": 1.25,
            "background_visible": False,
            "source_age_hours": 18.0,
            "activity_unit": "kBq",
        }
    )
    bottom.inventory_timeline_panel.apply_workflow_state(
        {
            "time_origin": "count_start",
            "observable": "dose",
            "time_stop_hours": 72.0,
            "activity_unit": "MBq",
            "advanced_objectives": True,
        }
    )
    bottom.masking_review_panel.apply_workflow_state(
        {
            "energy_window_keV": 9.5,
            "isotopes_of_interest": "Co-60",
            "top_n": 12,
            "current_tab": "Recommendations",
        }
    )
    bottom.optimization_workspace_panel.apply_workflow_state(
        {
            "objective": "fim-a",
            "irradiation_grid_s": "3600,7200",
            "cooldown_grid_s": "1800,7200",
            "count_grid_s": "600,1200",
            "target_nuclide": "Co-60",
            "advanced_objectives": True,
            "current_tab": "Recommendation",
        }
    )
    bottom.second_irradiation_panel.apply_workflow_state(
        {
            "flux_scales": "1.0,1.4",
            "duration_factors": "1.0,1.2",
            "cooling_grid_s": "600,900",
            "target_weights": "Co-60:2.0",
        }
    )
    bottom.phase5_parity_panel.apply_workflow_state(
        {
            "replay_state_filter": "adapter-required",
            "parity_scope": "algorithm",
            "fixture_filter": "spectrum_io_normalization_algorithm_case",
        }
    )
    first.workflow_presets.save_workflow(
        "session-restore-check",
        first._snapshot_current_workflow(),
        description="Regression test workflow",
    )
    first.close()

    second = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
    )
    second.show()
    _qapp().processEvents()

    restored_sidebar = second.left_dock.widget()
    restored_bottom = second.bottom_dock.widget()
    assert second.workflow_presets.active_workflow_name() == "session-restore-check"
    assert second.workflow_combo.currentData() == "session-restore-check"
    assert second._log_scale_action.isChecked() is True
    assert restored_sidebar.nuclide_query.text() == "co"
    assert restored_sidebar.saved_nuclides.count() == 1
    assert restored_bottom.activity_results_panel.background_scale.value() == pytest.approx(1.25)
    assert restored_bottom.activity_results_panel.background_visible.isChecked() is False
    assert restored_bottom.inventory_timeline_panel.time_stop_hours.value() == pytest.approx(72.0)
    assert restored_bottom.masking_review_panel.energy_window_spin.value() == pytest.approx(9.5)
    assert restored_bottom.optimization_workspace_panel.target_nuclide_edit.text() == "Co-60"
    assert restored_bottom.second_irradiation_panel.target_weights_edit.text() == "Co-60:2.0"
    assert restored_bottom.phase5_parity_panel.replay_filter_combo.currentText() == "adapter-required"
    assert restored_bottom.phase5_parity_panel.parity_scope_combo.currentText() == "algorithm"
    assert (
        restored_bottom.phase5_parity_panel.fixture_filter_edit.text()
        == "spectrum_io_normalization_algorithm_case"
    )
    second.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt GUI dependencies are unavailable.")
def test_phase5_parity_panel_runs_workflow_fixture_bundle():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().phase5_parity_panel
    scope_index = panel.parity_scope_combo.findText("workflow")
    if scope_index >= 0:
        panel.parity_scope_combo.setCurrentIndex(scope_index)
    panel.fixture_filter_edit.setText("roi_statistics_workflow_case")

    payload = panel.run_parity_suite()
    assert payload is not None
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 0
    assert payload["results"][0]["fixture_id"] == "roi_statistics_workflow_case"
    assert "1 passed" in panel.parity_label.text()
    window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt GUI dependencies are unavailable.")
def test_phase5_parity_panel_runs_activation_inventory_fixture_bundle():
    _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    _qapp().processEvents()

    panel = window.bottom_dock.widget().phase5_parity_panel
    scope_index = panel.parity_scope_combo.findText("workflow")
    if scope_index >= 0:
        panel.parity_scope_combo.setCurrentIndex(scope_index)
    panel.fixture_filter_edit.setText("radioactivedecay_inventory_case")

    payload = panel.run_parity_suite()
    assert payload is not None
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 0
    assert payload["results"][0]["fixture_id"] == "radioactivedecay_inventory_case"
    assert "1 passed" in panel.parity_label.text()
    window.close()


def test_selection_bus_helper_publishers_preserve_shared_context():
    bus = SelectionBus()

    bus.publish_peak(661.657, nuclide="Cs-137")
    bus.publish_roi(640.0, 680.0)

    state = bus.publish_nuclide(
        "Ba-137m",
        annotation_lines=(ReferenceLine(energy_keV=661.657, label="Photopeak"),),
    )
    assert state.peak_energy_keV == 661.657
    assert state.roi_bounds_keV == (640.0, 680.0)
    assert bus.describe()["nuclide"] == "Ba-137m"
    assert bus.describe()["annotation_line_count"] == 1


def test_hierarchical_buffer_builds_multiple_levels():
    counts = [float(index % 17) for index in range(256)]
    buffer = HierarchicalSpectrumBuffer.from_counts(counts, max_levels=5)

    assert buffer.describe()["level_count"] == 5
    assert buffer.levels[0].stride == 1
    assert buffer.levels[-1].stride == 16
    assert buffer.choose_level(pixel_width=20).sample_count <= 40


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_spectrum_canvas_reset_and_draggable_roi_controls():
    app = _qapp()
    selection_bus = SelectionBus()
    canvas = PyQtGraphSpectrumCanvas(selection_bus=selection_bus)
    canvas.resize(1000, 640)
    canvas.show()
    canvas.set_traces(
        (
            SpectrumTrace(
                label="UWNR HPGe",
                counts=tuple(float((index % 50) + 1) for index in range(1001)),
                channels=tuple(float(index) for index in range(1001)),
                x_axis_label="Energy (keV)",
            ),
        )
    )
    app.processEvents()

    canvas.plot_item.setXRange(400.0, 500.0, padding=0.0)
    app.processEvents()
    before_zoom_width = canvas.plot_item.viewRange()[0][1] - canvas.plot_item.viewRange()[0][0]
    QTest.mouseClick(canvas.zoom_in_button, Qt.LeftButton)
    app.processEvents()
    zoomed_width = canvas.plot_item.viewRange()[0][1] - canvas.plot_item.viewRange()[0][0]
    assert zoomed_width < before_zoom_width
    QTest.mouseClick(canvas.zoom_out_button, Qt.LeftButton)
    app.processEvents()
    assert canvas.plot_item.viewRange()[0][1] - canvas.plot_item.viewRange()[0][0] > zoomed_width

    canvas.plot_item.setXRange(300.0, 600.0, padding=0.0)
    app.processEvents()
    before_pan = tuple(canvas.plot_item.viewRange()[0])
    viewport = canvas.plot.viewport()
    start = QPoint(viewport.width() // 2, viewport.height() // 2)
    finish = QPoint(start.x() + 100, start.y())
    QTest.mousePress(viewport, Qt.LeftButton, pos=start)
    QTest.mouseMove(viewport, finish, delay=30)
    QTest.mouseRelease(viewport, Qt.LeftButton, pos=finish)
    app.processEvents()
    after_pan = tuple(canvas.plot_item.viewRange()[0])
    assert after_pan != pytest.approx(before_pan)

    QTest.mouseClick(canvas.reset_view_button, Qt.LeftButton)
    app.processEvents()
    reset_range = canvas.plot_item.viewRange()[0]
    assert reset_range[0] <= 0.0
    assert reset_range[1] >= 1000.0

    QTest.mouseClick(canvas.roi_button, Qt.LeftButton)
    app.processEvents()
    assert canvas._roi_region.isVisible()
    canvas._roi_region.setRegion((640.0, 680.0))
    canvas._roi_region.sigRegionChangeFinished.emit(canvas._roi_region)
    app.processEvents()
    assert selection_bus.state.roi_bounds_keV == pytest.approx((640.0, 680.0))

    QTest.mouseClick(canvas.clear_roi_button, Qt.LeftButton)
    app.processEvents()
    assert selection_bus.state.roi_bounds_keV is None
    assert not canvas._roi_region.isVisible()
    canvas.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_modern_main_window_opens_bundled_uwnr_genie_asc():
    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    asc_path = (
        ROOT
        / "examples"
        / "RAFM_irradiation"
        / "raw_gamma_spec"
        / "flux_wires"
        / "Ti-RAFM-1a_25cm.ASC"
    )

    window.open_path(asc_path)
    app.processEvents()

    spectrum = window.analysis_workspace.spectrum()
    assert spectrum is not None
    assert len(spectrum.counts) == 8192
    assert window.central_tabs.canvas._x_axis_is_energy is True
    assert (
        window.central_tabs.canvas.plot.getPlotItem().getAxis("bottom").labelText
        == "Energy (keV)"
    )
    assert asc_path.name in window.file_label.text()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_hidden_phase5_harness_does_not_cover_analysis_tabs():
    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    app.processEvents()

    analysis_tabs = window.bottom_dock.widget()
    assert not analysis_tabs.phase5_parity_panel.isVisible()
    assert analysis_tabs.currentWidget() is analysis_tabs.peak_table_panel
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_modern_menu_actions_open_files_focus_search_and_explain_availability(
    monkeypatch,
):
    from fluxforge.gui import main_window as main_window_module

    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    app.processEvents()
    actions = {action.text(): action for action in window.findChildren(QAction)}

    asc_path = (
        ROOT
        / "examples"
        / "RAFM_irradiation"
        / "raw_gamma_spec"
        / "flux_wires"
        / "Ti-RAFM-1a_25cm.ASC"
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(asc_path), ""),
    )
    assert actions["Open Spectrum..."].isEnabled()
    actions["Open Spectrum..."].trigger()
    app.processEvents()
    assert window.analysis_workspace.spectrum() is not None
    assert asc_path.name in window.file_label.text()

    opened_sessions = []
    monkeypatch.setattr(window, "_open_dialog_path", opened_sessions.append)
    session_path = str(ROOT / "tests" / "spectra" / "example.ffs")
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (session_path, ""),
    )
    assert actions["Open Session..."].isEnabled()
    actions["Open Session..."].trigger()
    assert opened_sessions == [session_path]

    assert actions["Nuclide Search"].isEnabled()
    actions["Nuclide Search"].trigger()
    app.processEvents()
    assert window.left_dock.widget().nuclide_query.hasFocus()

    assert actions["Restore Default Layout"].isEnabled()
    window.left_dock.hide()
    actions["Restore Default Layout"].trigger()
    app.processEvents()
    assert window.left_dock.isVisible()

    unavailable = [
        action
        for action in window.findChildren(QAction)
        if "(not available)" in action.text()
    ]
    assert unavailable
    assert all(not action.isEnabled() for action in unavailable)
    assert all(action.toolTip() for action in unavailable)
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)
def test_modern_help_and_full_canvas_actions_are_clickable(monkeypatch):
    from fluxforge.gui import main_window as main_window_module

    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
    )
    window.show()
    app.processEvents()
    actions = {action.text(): action for action in window.findChildren(QAction)}
    messages = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, body: messages.append((title, body)),
    )

    actions["Shortcut Reference"].trigger()
    actions["About FluxForge"].trigger()
    assert [title for title, _body in messages] == [
        "FluxForge shortcuts",
        "About FluxForge",
    ]
    assert "Ctrl+O" in messages[0][1]

    assert not window.isFullScreen()
    actions["Toggle Full Canvas"].trigger()
    app.processEvents()
    assert window.isFullScreen()
    assert not window.left_dock.isVisible()
    assert not window.bottom_dock.isVisible()
    assert not window.right_dock.isVisible()
    assert not window.primary_toolbar.isVisible()
    actions["Toggle Full Canvas"].trigger()
    app.processEvents()
    assert not window.isFullScreen()
    assert window.left_dock.isVisible()
    assert window.bottom_dock.isVisible()
    assert window.right_dock.isVisible()
    assert window.primary_toolbar.isVisible()
    window.close()


def test_modern_shell_reuses_shared_demo_and_selection_helpers():
    modern_shell_path = ROOT / "src" / "fluxforge" / "gui" / "panels" / "modern_shell.py"
    parsed = ast.parse(modern_shell_path.read_text(encoding="utf-8"))
    local_defs = {
        node.name
        for node in ast.walk(parsed)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_selection_summary" not in local_defs
    assert "_format_duration" not in local_defs
    assert "_demo_counts" not in local_defs

    from fluxforge.gui.panels import modern_shell, modern_shell_shared

    assert modern_shell.build_demo_spectrum is modern_shell_shared.build_demo_spectrum
    assert (
        modern_shell.build_demo_background_spectrum
        is modern_shell_shared.build_demo_background_spectrum
    )
    assert (
        modern_shell.build_demo_overlay_spectrum
        is modern_shell_shared.build_demo_overlay_spectrum
    )


def test_register_builtin_render_backends_tracks_default_and_stub():
    registries = register_builtin_render_backends(PluginRegistries())

    assert registries.render_backends.default_key == "pyqtgraph"
    assert registries.render_backends.keys() == ("pyqtgraph", "vispy")


def test_modern_gui_unavailable_message_mentions_legacy_fallback():
    message = modern_gui_unavailable_message()

    assert "fluxforge-gui-legacy" in message


def test_gui_package_imports_without_reporting_extra():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockJinja2(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "jinja2" or fullname.startswith("jinja2."):
                    raise ModuleNotFoundError("blocked for test")
                return None

        sys.meta_path.insert(0, BlockJinja2())
        import fluxforge.gui
        print("ok")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_current_gui_regression_files_target_modern_qt_stack():
    modern_gui_files = (
        ROOT / "tests" / "test_modern_gui_shell.py",
        ROOT / "tests" / "test_calibration_workspace_qt.py",
        ROOT / "tests" / "test_unfolding_workspace_qt.py",
        ROOT / "tests" / "gui_calibration_workspace_probe.py",
        ROOT / "tests" / "gui_analysis_workspace_probe.py",
        ROOT / "tests" / "gui_unfolding_workspace_probe.py",
        ROOT / "tests" / "gui_module3_workflows_probe.py",
        ROOT / "tests" / "gui_phase6_optimization_probe.py",
        ROOT / "tests" / "gui_phase327_release_probe.py",
        ROOT / "tests" / "gui_predictive_dashboard_probe.py",
        ROOT / "tests" / "test_analysis_workspace_qt.py",
        ROOT / "tests" / "test_module3_workflows_qt.py",
        ROOT / "tests" / "test_predictive_dashboard_qt.py",
    )

    for path in modern_gui_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert all(not name.startswith("fluxforge_gui") for name in imports)
        assert any(name.startswith("fluxforge.gui") for name in imports)


def test_phase327_release_checklist_exists_with_action_items():
    checklist = ROOT / "docs" / "PHASE3_27_RELEASE_CHECKLIST.md"
    assert checklist.exists()
    text = checklist.read_text(encoding="utf-8")
    assert "release-blocking" in text.lower()
    assert "- [ ]" in text
