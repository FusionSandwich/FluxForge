"""Capture native Qt layouts without using saved operator preferences."""
import os
os.environ.update(QT_QPA_PLATFORM="windows", FLUXFORGE_OFFLINE="1", MPLBACKEND="Agg")
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OUT = OUT / 'native-final'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton
from fluxforge.gui import FluxForgeMainWindow, ModeManager, SelectionBus

class MemorySettings:
    def __init__(self):
        self.values = {}
    def value(self, key, default=None):
        return self.values.get(key, default)
    def setValue(self, key, value):
        self.values[key] = value
    def sync(self):
        pass

QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(OUT / 'capture-settings'))
app = QApplication([])
settings = MemorySettings()
window = FluxForgeMainWindow(mode_manager=ModeManager(settings=settings),
                            settings=settings, selection_bus=SelectionBus())
window.resize(1280, 800)
window.show()
app.processEvents()
receipts = []

def capture(widget, name):
    app.processEvents()
    path = OUT / (name + '.png')
    assert widget.grab().save(str(path))
    controls = []
    for button in widget.findChildren(QPushButton) + widget.findChildren(QToolButton):
        if not button.isVisible():
            continue
        point = button.mapTo(widget, button.rect().center())
        controls.append(dict(name=button.objectName(), text=button.text(),
                             x=point.x(), y=point.y(), enabled=button.isEnabled(),
                             inside=widget.rect().contains(point)))
    receipts.append(dict(name=name, width=widget.width(), height=widget.height(),
                         font=widget.font().family(), controls=controls))

capture(window, '01-empty')
window._load_example_workspace()
capture(window, '02-loaded')
window._open_energy_fwhm_workspace()
dialog = window._calibration_dialog
dialog.resize(1280, 800)
capture(dialog, '03-calibration')
dialog.close()
window._open_report_export()
dialog = window._report_dialog
dialog.resize(1100, 800)
capture(dialog, '04-report')
dialog.path_input.setText(str(OUT / 'review-report.html'))
if not (OUT / 'review-report.zip').exists():
    dialog.generate_report()
receipts.append(dict(report_bundle=str(dialog.last_bundle_path),
                     report_status=dialog.export_status.text()))
dialog.close()
window._open_unfolding_workspace()
dialog = window._unfolding_dialog
dialog.resize(1280, 800)
capture(dialog, '05-unfolding')
dialog.close()
window.close()
(OUT / 'native-layout.json').write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')
print(json.dumps([dict(name=item.get('name'), width=item.get('width'), height=item.get('height'),
                      report_status=item.get('report_status')) for item in receipts]))
