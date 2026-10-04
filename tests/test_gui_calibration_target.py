"""Calibration dialogs must retain the acquisition they were opened for."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.gui import QT_AVAILABLE, FluxForgeMainWindow, ModeManager, SelectionBus
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.qt_compat import QApplication


class MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE), reason="Qt renderer unavailable"
)


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    settings = MemorySettings()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(settings=settings),
        selection_bus=SelectionBus(), settings=settings, load_example=True,
    )
    yield window
    if window._calibration_dialog is not None:
        window._calibration_dialog.close()
    window.close()
    app.processEvents()


def _fit(dialog):
    dialog._energy_fit = SimpleNamespace(
        order=1, coefficients=(0.5, 2.0), deviation_pairs=(),
        chi_squared=0.2, reduced_chi_squared=0.1, rms_keV=0.03,
    )
    dialog._fwhm_fit = None


def test_apply_after_selection_switch_updates_original_and_undo(window):
    controller = window.analysis_workspace
    original_id = controller.document.active_spectrum_id
    other = next(item for item in controller.document.spectra
                 if item.spectrum_id != original_id)
    original = controller.document.spectrum_by_id(original_id).spectrum
    window._open_energy_fwhm_workspace()
    dialog = window._calibration_dialog
    _fit(dialog)
    controller.select_spectrum(other.spectrum_id)
    selection = window.selection_bus.state
    label = window.file_label.text()
    dialog._apply_workspace_results()
    assert controller.document.spectrum_by_id(original_id).spectrum.calibration["energy"] == [0.5, 2.0]
    assert controller.document.spectrum_by_id(other.spectrum_id).spectrum is other.spectrum
    assert controller.document.active_spectrum_id == other.spectrum_id
    assert window.file_label.text() == label
    assert window.selection_bus.state == selection
    window.undo_stack.undo()
    assert controller.document.spectrum_by_id(original_id).spectrum.calibration == original.calibration
    assert controller.document.active_spectrum_id == other.spectrum_id
    window.undo_stack.redo()
    assert controller.document.spectrum_by_id(original_id).spectrum.calibration["energy"] == [0.5, 2.0]


def test_stale_dialog_cannot_change_reloaded_example_with_same_ids(window):
    window._open_energy_fwhm_workspace()
    dialog = window._calibration_dialog
    _fit(dialog)
    window._load_example_workspace()
    before = window.analysis_workspace.document.to_dict()
    selection = window.selection_bus.state
    dialog._apply_workspace_results()
    assert window.analysis_workspace.document.to_dict() == before
    assert window.selection_bus.state == selection
    assert window.undo_stack.count() == 0
    assert "Applied to the workspace" not in dialog.energy_summary.text()


def test_replaced_acquisition_under_same_id_rejects_old_fit(window):
    controller = window.analysis_workspace
    original_id = controller.document.active_spectrum_id
    original = controller.document.spectrum_by_id(original_id)
    window._open_energy_fwhm_workspace()
    dialog = window._calibration_dialog
    _fit(dialog)
    replacement = replace(original, spectrum=replace(
        original.spectrum, counts=original.spectrum.counts.copy(), energies=None,
    ))
    controller.set_document(replace(controller.document, spectra=tuple(
        replacement if item.spectrum_id == original_id else item
        for item in controller.document.spectra
    )))
    before = controller.document.to_dict()
    dialog._apply_workspace_results()
    assert controller.document.to_dict() == before
    assert window.undo_stack.count() == 0
