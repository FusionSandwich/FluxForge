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
from fluxforge.gui.qt_compat import QApplication
from fluxforge.gui.spectrum_canvas import ReferenceLine
from fluxforge.plugins import PluginRegistries

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
