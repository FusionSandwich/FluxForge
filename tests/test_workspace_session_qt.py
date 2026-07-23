from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluxforge.core.workspace_document import (
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    DetectorProfile,
    EfficiencyModelState,
    WorkspaceDocument,
)
from fluxforge.gui import QT_AVAILABLE, FluxForgeMainWindow, ModeManager, SelectionBus
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE
from fluxforge.gui.qt_compat import QApplication
from fluxforge.io import FluxForgeSession, read_ffs_session, write_ffs_session

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class MemorySettings:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value) -> None:
        self.values[key] = value

    def sync(self) -> None:
        return None


def _qapp():
    return QApplication.instance() or QApplication([])


def _window(*, load_example: bool = False) -> FluxForgeMainWindow:
    return FluxForgeMainWindow(
        mode_manager=ModeManager(settings=MemorySettings()),
        selection_bus=SelectionBus(),
        settings=MemorySettings(),
        load_example=load_example,
    )


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)


def test_session_actions_have_stable_ids_and_shortcuts() -> None:
    _qapp()
    window = _window()
    try:
        save = window.findChild(object, "SaveSessionAction")
        save_as = window.findChild(object, "SaveSessionAsAction")
        assert save is not None and save.shortcut().toString() == "Ctrl+S"
        assert save_as is not None and save_as.shortcut().toString() == "Ctrl+Shift+S"
    finally:
        window.close()


def test_save_and_open_session_replaces_complete_workspace(tmp_path: Path) -> None:
    app = _qapp()
    window = _window(load_example=True)
    window.show()
    app.processEvents()
    try:
        spectrum_id = window.analysis_workspace.document.active_spectrum_id
        assert spectrum_id is not None
        roi = AnalysisROI(
            roi_id="saved-roi",
            spectrum_id=spectrum_id,
            signal_range=(400.0, 700.0),
            left_background_range=(300.0, 400.0),
            right_background_range=(700.0, 800.0),
            label="review",
        )
        profile = DetectorProfile(
            detector_profile_id="saved-detector",
            energy_calibration=CalibrationModel(
                model_key="polynomial", coefficients=(0.1, 0.5)
            ),
            efficiency_model=EfficiencyModelState(
                model_key="log-polynomial",
                parameters={"coefficients": [-2.0, -0.6]},
                covariance=((0.01, 0.0), (0.0, 0.02)),
            ),
        )
        window.analysis_workspace.upsert_roi(roi)
        window.analysis_workspace.upsert_detector_profile(profile)
        window.analysis_workspace.upsert_viewport(
            CanvasViewport(
                viewport_id="primary-spectrum",
                spectrum_id=spectrum_id,
                x_range=(250.0, 850.0),
                y_range=(0.1, 200.0),
                log_y=True,
                residual_mode="compact",
                selected_roi_id="saved-roi",
                labels_visible=False,
            )
        )
        window.central_tabs.apply_viewport_state(
            window.analysis_workspace.document.viewport_by_id("primary-spectrum")
        )
        app.processEvents()

        target = tmp_path / "complete.ffs"
        assert window.save_session(target)
        persisted = read_ffs_session(target).document
        assert persisted.roi_by_id("saved-roi") == roi
        assert persisted.detector_profile_by_id("saved-detector") == profile
        assert persisted.viewport_by_id("primary-spectrum").log_y is True

        window.analysis_workspace.set_document(
            WorkspaceDocument(
                document_id="temporary",
                created_at="2026-07-22T00:00:00+00:00",
                updated_at="2026-07-22T00:00:00+00:00",
            )
        )
        window.open_path(target)
        app.processEvents()

        loaded = window.analysis_workspace.document
        loaded_payload = loaded.to_dict()
        persisted_payload = persisted.to_dict()
        loaded_payload.pop("updated_at")
        persisted_payload.pop("updated_at")
        assert loaded_payload == persisted_payload
        assert window._session_path == target.resolve()
        assert window._document_dirty is False
        assert window.undo_stack.isClean()
        assert window.selection_bus.state.roi_bounds_keV == pytest.approx(
            roi.signal_range
        )
        restored_view = window.central_tabs.viewport_state()
        assert restored_view.x_range == pytest.approx((250.0, 850.0))
        assert restored_view.log_y is True
        assert restored_view.labels_visible is False
    finally:
        window.close()


def test_opening_second_session_does_not_append_first_session(tmp_path: Path) -> None:
    _qapp()
    populated_path = tmp_path / "populated.ffs"
    empty_path = tmp_path / "empty.ffs"
    source = _window(load_example=True)
    try:
        assert source.save_session(populated_path)
    finally:
        source.close()

    empty_document = WorkspaceDocument(
        document_id="empty",
        created_at="2026-07-22T00:00:00+00:00",
        updated_at="2026-07-22T00:00:00+00:00",
    )
    write_ffs_session(empty_path, FluxForgeSession(document=empty_document))

    window = _window()
    try:
        window.open_path(populated_path)
        assert window.analysis_workspace.document.spectra
        window.open_path(empty_path)
        assert window.analysis_workspace.document.document_id == "empty"
        assert window.analysis_workspace.document.spectra == ()
        assert window.analysis_workspace.state.loaded_spectra == ()
        assert window.analysis_workspace.state.spectra == ()
    finally:
        window.close()


def test_gui_open_and_resave_preserves_session_envelope(tmp_path: Path) -> None:
    _qapp()
    source_path = tmp_path / "source.ffs"
    target_path = tmp_path / "resaved.ffs"
    embedded_recent = [tmp_path / "first.asc", tmp_path / "second.asc"]
    original = FluxForgeSession(
        document=WorkspaceDocument(
            document_id="envelope-test",
            created_at="2026-07-20T10:00:00+00:00",
            updated_at="2026-07-20T10:00:00+00:00",
        ),
        recent_files=embedded_recent,
        device_snapshot=[
            {
                "device_id": "sim-mca-1",
                "telemetry": {"dead_time_fraction": 0.012},
            }
        ],
        metadata={"campaign": "UWNR", "operator": "fixture"},
        created_at="2026-07-20T09:55:00+00:00",
    )
    write_ffs_session(source_path, original)

    window = _window()
    try:
        window.open_path(source_path)
        assert window.save_session(target_path)

        resaved = read_ffs_session(target_path)
        assert resaved.device_snapshot == original.device_snapshot
        assert resaved.metadata == original.metadata
        assert resaved.created_at == original.created_at
        assert str(source_path) in resaved.recent_files
        assert str(target_path) in resaved.recent_files
        assert all(str(item) in resaved.recent_files for item in embedded_recent)

        window._load_example_workspace()
        assert window._session_device_snapshot == []
        assert window._session_metadata == {}
        assert window._session_created_at is None
        assert window._session_recent_files == ()
    finally:
        window.close()


def test_main_window_calibration_apply_and_undo_are_non_mutating() -> None:
    _qapp()
    window = _window(load_example=True)
    try:
        spectrum_id = window.analysis_workspace.document.active_spectrum_id
        before_record = window.analysis_workspace.document.spectrum_by_id(spectrum_id)
        before_spectrum = before_record.spectrum
        before_counts = before_spectrum.counts
        before_energies = np.asarray(before_spectrum.energies, dtype=float).copy()
        before_coefficients = tuple(before_spectrum.calibration["energy"])
        before_payload = window.analysis_workspace.document.to_dict()
        before_payload.pop("updated_at")
        fit = SimpleNamespace(
            order=1,
            coefficients=(0.5, 2.0),
            deviation_pairs=(),
            chi_squared=0.2,
            reduced_chi_squared=0.1,
            rms_keV=0.03,
        )

        window._apply_calibration_workspace_result(before_spectrum, fit, None)
        applied_record = window.analysis_workspace.document.spectrum_by_id(spectrum_id)
        applied_spectrum = applied_record.spectrum
        assert applied_spectrum is not before_spectrum
        assert applied_spectrum.counts is before_counts
        assert before_spectrum.calibration["energy"] == list(before_coefficients)
        assert np.array_equal(before_spectrum.energies, before_energies)
        assert applied_spectrum.calibration["energy"] == [0.5, 2.0]
        assert applied_record.detector_profile_id is not None

        window.undo_stack.undo()
        restored = window.analysis_workspace.document.spectrum_by_id(
            spectrum_id
        ).spectrum
        assert restored.counts is before_counts
        assert restored.calibration["energy"] == list(before_coefficients)
        assert np.array_equal(restored.energies, before_energies)
        restored_payload = window.analysis_workspace.document.to_dict()
        restored_payload.pop("updated_at")
        assert restored_payload == before_payload

        window.undo_stack.redo()
        redone = window.analysis_workspace.document.spectrum_by_id(spectrum_id).spectrum
        assert redone.calibration["energy"] == [0.5, 2.0]
    finally:
        window.close()
