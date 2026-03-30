import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.core.phase2_analysis import PeakCandidate  # noqa: E402
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt predictive dashboard dependencies are unavailable.",
)
def test_predictive_dashboard_shows_eta_dead_time_and_qa_forecast():
    _qapp()
    selection_bus = SelectionBus()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=selection_bus)
    window.phase2_workspace.replace_peaks(
        (
            PeakCandidate(
                peak_id="peak-1",
                channel=662.0,
                energy_keV=661.66,
                significance=9.8,
                roi_bounds_keV=(655.0, 668.0),
                net_counts=4200.0,
                fit_quality=0.98,
            ),
        )
    )
    window.phase2_workspace.select_peak("peak-1")
    selection_bus.publish_roi(655.0, 668.0)
    _qapp().processEvents()

    tab_bar = window.central_tabs.tabBar()
    QTest.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(1).center())
    _qapp().processEvents()

    dashboard = window.central_tabs.predictive_dashboard
    dashboard.target_counts_spin.setValue(12000.0)
    _qapp().processEvents()

    summary = dashboard.summary_browser.toPlainText()
    assert "Count target" in summary
    assert "Dead-time trend" in summary
    assert "QA recalibration forecast" in summary
    assert dashboard.count_rate_plot.plotItem.listDataItems()
    assert dashboard.dead_time_plot.plotItem.listDataItems()

    sidebar_summary = window.left_dock.widget().qa_note.toPlainText()
    assert "Predictive" in sidebar_summary
    assert "Recalibration forecast" in sidebar_summary
    assert "Predictive: ETA" in window.predictive_label.text()
    window.close()
