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
        canvas = window.central_tabs.canvas
        assert canvas.plot.height() > 100
        canvas.status_label.setText("Long acquisition metadata " * 30)
        app.processEvents()
        assert window.width() <= 1280 and window.height() <= 720
        assert canvas.plot.height() > 100
        window.central_tabs.setCurrentIndex(1)
        app.processEvents()
        dashboard = window.central_tabs.predictive_dashboard
        assert isinstance(dashboard, QScrollArea)
        assert window.height() <= 720
        assert dashboard.verticalScrollBar().maximum() > 0
        dashboard.ensureWidgetVisible(dashboard.target_counts_spin)
        app.processEvents()
        point = dashboard.target_counts_spin.mapTo(
            dashboard.viewport(), dashboard.target_counts_spin.rect().center()
        )
        assert dashboard.viewport().rect().contains(point)
        dashboard.verticalScrollBar().setValue(dashboard.verticalScrollBar().maximum())
        dashboard.summary_browser.verticalScrollBar().setValue(
            dashboard.summary_browser.verticalScrollBar().maximum()
        )
        app.processEvents()
        point = dashboard.summary_browser.mapTo(
            dashboard.viewport(), dashboard.summary_browser.rect().bottomLeft()
        )
        assert dashboard.viewport().rect().contains(point)
        assert dashboard.count_rate_plot.height() >= 160
    finally:
        window.close()
        app.processEvents()
