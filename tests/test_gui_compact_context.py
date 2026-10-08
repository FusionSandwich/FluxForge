"""Context diagnostics must scroll instead of forcing a tall desktop window."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402
from fluxforge.gui import FluxForgeMainWindow, ModeManager, SelectionBus  # noqa: E402


class MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


def test_context_scrolls_on_720_pixel_desktop():
    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        settings=settings,
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    try:
        window.resize(1280, 720)
        window.show()
        app.processEvents()
        assert window.width() <= 1280
        assert window.height() <= 720
        context = window.right_dock.widget()
        assert isinstance(context, QScrollArea)
        assert context.verticalScrollBar().maximum() > 0
        context.ensureWidgetVisible(context.analysis_summary)
        app.processEvents()
        point = context.analysis_summary.mapTo(
            context.viewport(), context.analysis_summary.rect().bottomLeft()
        )
        assert context.viewport().rect().contains(point)
        assert window.central_tabs.canvas.plot.height() > 100
    finally:
        window.close()
        app.processEvents()
