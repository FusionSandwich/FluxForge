"""The production Qt peak-review path must not silently discard weaker peaks."""

import numpy as np
import pytest

from fluxforge.core.analysis_workspace import detect_peak_candidates
from fluxforge.io.spe import GammaSpectrum


@pytest.fixture
def many_peaks():
    channels = np.arange(2048)
    counts = np.full(2048, 20.0)
    centers = np.arange(90, 1900, 90)
    for index, center in enumerate(centers):
        counts += (4000 - index * 80) * np.exp(-0.5 * ((channels - center) / 3) ** 2)
    return GammaSpectrum(channels=channels, counts=counts), centers


def test_uncapped_search_retains_every_supported_strong_peak(many_peaks):
    spectrum, centers = many_peaks
    peaks = detect_peak_candidates(spectrum, method="mariscotti", max_peaks=None)
    assert len(peaks) == len(centers) > 12
    assert all(any(abs(p.channel - center) < 1 for p in peaks) for center in centers)
    limited = detect_peak_candidates(spectrum, method="mariscotti", max_peaks=12)
    assert len(limited) == 12


def test_new_gui_passes_uncapped_candidates_into_peak_review(monkeypatch, many_peaks):
    from fluxforge.gui.qt_compat import QT_AVAILABLE, QApplication

    if not QT_AVAILABLE:
        pytest.skip("Qt unavailable")
    from PySide6.QtWidgets import QDialog
    from fluxforge.gui.main_window import FluxForgeMainWindow
    from fluxforge.gui.mode_manager import ModeManager
    from fluxforge.gui.selection_bus import SelectionBus
    from fluxforge.gui.dialogs.auto_peak_review_dialog import AutoPeakReviewDialog

    class MemorySettings:
        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

        def sync(self):
            pass

    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(),
        settings=settings,
    )
    spectrum, centers = many_peaks
    captured = {}

    def accept(dialog):
        captured["rows"] = dialog.table.rowCount()
        return QDialog.Accepted

    monkeypatch.setattr(AutoPeakReviewDialog, "exec", accept)
    try:
        window.analysis_workspace.replace_spectrum_slot("primary", spectrum)
        window.analysis_workspace.select_spectrum("primary")
        window.bottom_dock.widget().peak_table_panel.run_auto_peak_search()
        assert captured["rows"] == len(centers) > 12
        assert len(window.analysis_workspace.state.peaks) == len(centers)
    finally:
        window.close()
        app.processEvents()
