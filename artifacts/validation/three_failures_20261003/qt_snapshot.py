"""Render the current Qt interface with isolated in-memory settings."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from PySide6 import __version__ as qt_version
from PySide6.QtGui import QFontDatabase
from fluxforge.gui import app as gui_app
from fluxforge.gui.main_window import FluxForgeMainWindow
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QApplication
from fluxforge.gui.selection_bus import SelectionBus

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent

class MemorySettings:
    def __init__(self):
        self.values = {}
    def value(self, key, default=None):
        return self.values.get(key, default)
    def setValue(self, key, value):
        self.values[key] = value
    def sync(self):
        pass

assert Path(gui_app.__file__).resolve().is_relative_to(repo / "src")
scripts = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
assert scripts["fluxforge-gui"] == "fluxforge.gui.app:main"
assert "fluxforge-gui-legacy" not in scripts
assert not (repo / "src/fluxforge_gui").exists()
assert "from fluxforge.gui.app import main" in (repo / "tools/pyinstaller_launch_gui.py").read_text()
app = QApplication.instance() or QApplication([])
settings = MemorySettings()
window = FluxForgeMainWindow(mode_manager=ModeManager(settings=settings), selection_bus=SelectionBus(), settings=settings, load_example=False)
try:
    window.resize(1560, 980)
    window.show()
    app.processEvents()
    assert window.isVisible()
    assert window.analysis_workflow_toolbar.isVisible()
    print(json.dumps({"requested_size":[1560,980], "actual_size":[window.width(),window.height()], "minimum_size_hint":[window.minimumSizeHint().width(),window.minimumSizeHint().height()]}),flush=True)
    assert window.grab().save(str(out / "qt_empty_workspace.png"))
    window._load_example_workspace()
    app.processEvents()
    assert window.analysis_workspace.state.loaded_spectra
    assert window.grab().save(str(out / "qt_example_workspace.png"))
    assert not any(m == "fluxforge_gui" or m.startswith("fluxforge_gui.") for m in sys.modules)
    payload = {"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),"gui":"PySide6/Qt","qt_version":qt_version,"qpa_platform":app.platformName(),"font_families":QFontDatabase.families(),"default_font":app.font().family(),"python":sys.executable,"prefix":sys.prefix,"gui_module":gui_app.__file__,"window_size":[window.width(),window.height()],"settings":"isolated in memory","screenshots":["qt_empty_workspace.png","qt_example_workspace.png"],"example_kind":"bundled demo; not scientific qualification","example_peak_count":len(window.analysis_workspace.state.peaks),"legacy_gui_imported":False,"installed_entrypoints":scripts,"source_sha256":{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/"src/fluxforge/gui/main_window.py",repo/"src/fluxforge/gui/app.py",repo/"pyproject.toml")},"status":"ok"}
    (out / "qt_snapshot.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload))
finally:
    window.close()
    app.processEvents()
