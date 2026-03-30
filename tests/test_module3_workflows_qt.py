import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.core.analysis_workspace import PeakCandidate  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import GUIMode, ModeManager  # noqa: E402
from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication  # noqa: E402
from fluxforge.gui.selection_bus import SelectionBus  # noqa: E402

if QT_AVAILABLE and PYQTGRAPH_AVAILABLE:  # noqa: E402
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtTest import QTest  # noqa: E402


def _qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_main_window_opens_qa_history_and_exports_csv(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    window._open_qa_history()
    _qapp().processEvents()

    dialog = window._qa_history_dialog
    dialog.export_path.setPlainText(str(tmp_path / "qa.csv"))
    QTest.mouseClick(dialog.export_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.table.rowCount() >= 1
    assert dialog.last_export_path is not None
    assert dialog.last_export_path.exists()
    dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_main_window_opens_report_export_and_writes_html(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    window._open_report_export()
    _qapp().processEvents()

    dialog = window._report_dialog
    dialog.path_input.setText(str(tmp_path / "report.html"))
    QTest.mouseClick(dialog.export_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.last_export_path is not None
    assert dialog.last_export_path.exists()
    assert "FluxForge Module 3 Report" in dialog.preview.toPlainText()
    dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_main_window_report_export_dialog_writes_pdf(tmp_path, monkeypatch):
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())

    def _fake_export_pdf(template_name, context, path):
        Path(path).write_bytes(b"%PDF-1.4\n% Qt dialog export\n")
        return Path(path)

    monkeypatch.setattr(window.reporting_engine, "can_export_pdf", lambda: True)
    monkeypatch.setattr(window.reporting_engine, "export_pdf", _fake_export_pdf)

    window._open_report_export()
    _qapp().processEvents()

    dialog = window._report_dialog
    dialog.path_input.setText(str(tmp_path / "report.html"))
    dialog._sync_pdf_status()
    QTest.mouseClick(dialog.pdf_button, Qt.LeftButton)
    _qapp().processEvents()

    assert dialog.last_pdf_export_path is not None
    assert dialog.last_pdf_export_path.exists()
    assert dialog.last_pdf_export_path.suffix == ".pdf"
    dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_sidebar_buttons_open_qa_history_and_astm_review():
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    window.show()
    _qapp().processEvents()
    sidebar = window.left_dock.widget()

    QTest.mouseClick(sidebar.qa_history_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window._qa_history_dialog is not None
    assert window._qa_history_dialog.isVisible()

    QTest.mouseClick(sidebar.astm_check_button, Qt.LeftButton)
    _qapp().processEvents()
    assert window._standards_review_dialog is not None
    assert window._standards_review_dialog.isVisible()
    assert window._standards_review_dialog.table.rowCount() >= 5
    assert "ASTM E181" in window._standards_review_dialog.detail.toPlainText()

    window._qa_history_dialog.close()
    window._standards_review_dialog.close()
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_main_window_opens_pu_isotopics_wizard_outside_simple_mode():
    _qapp()
    manager = ModeManager()
    window = FluxForgeMainWindow(mode_manager=manager, selection_bus=SelectionBus())

    manager.set_mode(GUIMode.EXPERT)
    _qapp().processEvents()
    assert window._pu_isotopics_action.isEnabled() is True

    window._open_pu_isotopics_wizard()
    _qapp().processEvents()

    dialog = window._pu_isotopics_dialog
    assert dialog.tabs.count() == 4
    assert dialog.ratio_table.rowCount() >= 1
    dialog.close()

    manager.set_mode(GUIMode.SIMPLE)
    _qapp().processEvents()
    assert window._pu_isotopics_action.isEnabled() is False
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_batch_queue_panel_runs_and_writes_outputs(tmp_path):
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    panel = window.bottom_dock.widget().batch_queue_panel
    panel.output_dir.setText(str(tmp_path / "batch"))

    QTest.mouseClick(panel.queue_button, Qt.LeftButton)
    _qapp().processEvents()
    QTest.mouseClick(panel.run_button, Qt.LeftButton)
    _qapp().processEvents()

    assert panel.table.rowCount() >= 1
    assert panel.last_output_dir is not None
    assert (panel.last_output_dir / "aggregate.csv").exists()
    assert panel.progress_bar.value() == 100
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_hardware_led_click_opens_dashboard_tab():
    _qapp()
    window = FluxForgeMainWindow(mode_manager=ModeManager(), selection_bus=SelectionBus())
    window.show()
    _qapp().processEvents()

    assert window.central_tabs.currentIndex() == 0
    QTest.mouseClick(window.hardware_led, Qt.LeftButton)
    _qapp().processEvents()

    assert window.central_tabs.currentIndex() == 1
    window.close()


@pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt module-3 workspace dependencies are unavailable.",
)
def test_peak_table_ml_button_updates_summary_and_sidebar_shows_qa_locks():
    _qapp()
    manager = ModeManager()
    window = FluxForgeMainWindow(mode_manager=manager, selection_bus=SelectionBus())
    bottom = window.bottom_dock.widget()
    peak_panel = bottom.peak_table_panel

    window.analysis_workspace.replace_peaks(
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
    QTest.mouseClick(peak_panel.ml_button, Qt.LeftButton)
    _qapp().processEvents()

    sidebar = window.left_dock.widget()
    manager.set_standard("ASTM E181")
    _qapp().processEvents()

    assert "ML" in peak_panel.ml_summary.text()
    assert "ASTM Status" in sidebar.qa_note.toPlainText()
    assert "energy_fit_order" in sidebar.qa_note.toPlainText()
    window.close()
