import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from fluxforge.gui import (
    GUIMode,
    HierarchicalSpectrumBuffer,
    ModeManager,
    SelectionBus,
    describe_gui_scaffold,
    modern_gui_unavailable_message,
    register_builtin_render_backends,
)
from fluxforge.gui.spectrum_canvas import ReferenceLine
from fluxforge.plugins import PluginRegistries

ROOT = Path(__file__).resolve().parents[1]


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


def test_describe_gui_scaffold_exposes_primary_and_legacy_entrypoints():
    scaffold = describe_gui_scaffold()

    assert scaffold["modern_entrypoint"] == "fluxforge-gui"
    assert scaffold["legacy_entrypoint"] == "fluxforge-gui-legacy"
    assert scaffold["renderer_backends"][0]["key"] == "pyqtgraph"


def test_mode_manager_persists_state_with_settings_backend():
    settings = FakeSettings()
    manager = ModeManager(settings=settings)

    manager.set_standard("ASTM E181")
    manager.set_theme("light")

    assert settings._values["gui/mode"] == "standards"
    assert settings._values["gui/standard"] == "ASTM E181"
    assert settings._values["gui/theme"] == "light"
    assert settings.sync_count >= 2

    reloaded = ModeManager(settings=settings)
    assert reloaded.state.mode is GUIMode.STANDARDS
    assert reloaded.state.standard == "ASTM E181"
    assert reloaded.state.theme == "light"


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
