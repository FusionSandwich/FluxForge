"""Real INL CLI/GUI subtraction, persistence and unsupported-case rejection.

These are software workflow checks, not physical INL acceptance.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from fluxforge.io.reader_factory import read_spectrum_any
from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.artifacts import write_spectrum_file
from fluxforge.io.spectrum_csv import write_spectrum_csv
from fluxforge.analysis.spectrum_math import subtract_measured_background

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "tests/data/flux_wires/raw"
SAMPLE = RAW / "Co-Cd-RAFM-1_25cm.ASC"
BACKGROUND = RAW / "background.ASC"


@pytest.mark.parametrize('invalid', [False, True])
def test_inl_cli_subtraction_and_invalid_coverage(tmp_path, invalid):
    output = tmp_path / "spectrum.json"
    corrected = tmp_path / "corrected.csv"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    background_path = BACKGROUND
    if invalid:
        background = read_spectrum_any(BACKGROUND)
        background.energies = background.energies + 100
        background_path = tmp_path/'invalid-background.json'
        write_spectrum_file(background_path, background)
    run = subprocess.run(
        [sys.executable, "-m", "fluxforge.cli.app", "ingest",
         "--input", str(SAMPLE), "--background-file", str(background_path),
         "--output", str(output), "--save-background-adjusted", str(corrected)],
        capture_output=True, text=True, env=env, timeout=60,
    )
    if invalid:
        assert run.returncode != 0
        assert "strict coverage" in run.stdout + run.stderr
        assert not output.exists()
        assert not corrected.exists()
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        actual = GammaSpectrum.from_dict(json.loads(output.read_text())['spectrum'])
        expected = subtract_measured_background(read_spectrum_any(SAMPLE), read_spectrum_any(BACKGROUND))
        assert actual.to_dict() == expected.to_dict()
        exported = read_spectrum_any(corrected)
        assert exported.to_dict() == actual.to_dict()
        assert actual.counts_covariance.nnz > len(actual.counts)
        assert np.any(actual.counts < 0)


def test_inl_gui_real_files_subtraction_session_and_recovery(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from fluxforge.gui.qt_compat import QApplication
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.mode_manager import ModeManager
    from fluxforge.gui.selection_bus import SelectionBus
    from PySide6.QtCore import QSettings
    from fluxforge.standards import QAMonitor

    app = QApplication.instance() or QApplication([])
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(), selection_bus=SelectionBus(),
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
        qa_monitor=QAMonitor(tmp_path / "qa.db"),
    )
    try:
        window.show()
        window.open_path(SAMPLE)
        window.open_path(BACKGROUND)
        app.processEvents()
        controller = window.analysis_workspace
        slots = controller.state.loaded_spectra
        sample_key = next(item.key for item in slots if item.label == SAMPLE.name)
        background_key = next(item.key for item in slots if item.label == BACKGROUND.name)
        sidebar = window.left_dock.widget()
        sidebar.foreground_spectrum_combo.setCurrentIndex(
            sidebar.foreground_spectrum_combo.findData(sample_key))
        sidebar.background_spectrum_combo.setCurrentIndex(
            sidebar.background_spectrum_combo.findData(background_key))
        controller.set_background_config(mode="statistical", scale=1.0)
        app.processEvents()
        foreground = controller.spectrum("foreground")
        background = controller.spectrum("background")
        before = foreground.to_dict(), background.to_dict()
        canvas = window.central_tabs.canvas
        status = canvas.status_label.text().lower()
        assert "background not applied" not in status
        expected = subtract_measured_background(foreground, background)
        np.testing.assert_allclose(canvas.buffer.full_resolution, expected.counts, rtol=1e-6, atol=1e-6)
        session = tmp_path/'real-inl.ffs'
        assert window.save_session(session)
        window.open_path(session)
        app.processEvents()
        np.testing.assert_allclose(canvas.buffer.full_resolution, expected.counts, rtol=1e-6, atol=1e-6)
        # Exercise native ROI action, including error presentation and recovery.
        roi_panel = window.bottom_dock.widget().roi_tools_panel
        roi_panel.roi_left.setValue(1165)
        roi_panel.roi_right.setValue(1182)
        roi_panel.overlap_checkbox.setChecked(False)
        roi_panel.background_selector.set_current_key('roi_sideband')
        roi_panel.analyze_button.click()
        app.processEvents()
        assert controller.state.roi_analysis is not None
        assert controller.state.roi_analysis.net_counts_uncertainty > 0
        roi_panel.background_selector.set_current_key('snip')
        roi_panel.analyze_button.click()
        app.processEvents()
        assert 'not applied' in roi_panel.summary.toPlainText()
        assert controller.state.roi_analysis is None
        roi_panel.background_selector.set_current_key('roi_sideband')
        roi_panel.analyze_button.click()
        assert controller.state.roi_analysis is not None
        # A truncated energy interval must still leave foreground visible.
        invalid_background = GammaSpectrum.from_dict(background.to_dict())
        invalid_background.energies += 100
        controller.replace_spectrum_slot('background', invalid_background)
        app.processEvents()
        assert 'background not applied' in canvas.status_label.text().lower()
        assert 'strict coverage' in canvas.status_label.text().lower()
        np.testing.assert_array_equal(canvas.buffer.full_resolution, foreground.counts)
        controller.replace_spectrum_slot('background', background)
        app.processEvents()
        np.testing.assert_allclose(canvas.buffer.full_resolution, expected.counts, rtol=1e-6, atol=1e-6)
        assert (foreground.to_dict(), background.to_dict()) == before
        # Reopen the corrected CSV itself as foreground and persist its full
        # covariance, independently of recomputing subtraction from raw inputs.
        corrected_csv = tmp_path / 'corrected-for-gui.csv'
        write_spectrum_csv(corrected_csv, expected)
        window._reset_analysis_workspace()
        assert not controller.document.spectra
        assert not controller.document.spectrum_roles
        window.open_path(corrected_csv)
        app.processEvents()
        restored = controller.spectrum('foreground')
        np.testing.assert_array_equal(restored.counts, expected.counts)
        assert (restored.counts_covariance != expected.counts_covariance).nnz == 0
        corrected_session = tmp_path / 'corrected-inl.ffs'
        assert window.save_session(corrected_session)
        window.open_path(corrected_session)
        app.processEvents()
        restored = controller.spectrum('foreground')
        assert (restored.counts_covariance != expected.counts_covariance).nnz == 0
        np.testing.assert_allclose(canvas.buffer.full_resolution, expected.counts,
                                   rtol=1e-6, atol=1e-6)
    finally:
        window.close()
        app.processEvents()
