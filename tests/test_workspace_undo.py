from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QUndoStack

from fluxforge.core.workspace_document import (
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    DetectorProfile,
    FitDiagnostics,
    NuclideAssignment,
    PeakModel,
    WorkspaceDocument,
    WorkspaceSpectrum,
)
from fluxforge.gui.analysis_workspace import (
    AnalysisWorkspaceController,
    AnalysisWorkspaceState,
)
from fluxforge.gui.workspace_undo import (
    ApplyCalibrationCommand,
    AssignSpectrumRoleCommand,
    DeletePeakCommand,
    DeleteROICommand,
    MoveROIBoundsCommand,
    ReplacePeakSetCommand,
    TogglePinnedNuclideCommand,
    UpdateCanvasViewportCommand,
    UpdateDetectorProfileCommand,
    UpdatePeakAssignmentCommand,
    UpdatePeakCommand,
    UpsertROICommand,
)
from fluxforge.io.spe import GammaSpectrum


class FakeLeafController:
    def __init__(self) -> None:
        self.peaks: dict[str, tuple[PeakModel, ...]] = {}
        self.rois: dict[str, AnalysisROI] = {}
        self.roles: dict[str, tuple[str, ...]] = {}
        self.profiles: dict[str, DetectorProfile] = {}
        self.viewports: dict[str, CanvasViewport] = {}
        self.diagnostics: dict[str, FitDiagnostics] = {}
        self.pinned: tuple[str, ...] = ()
        self.calibrations: dict[str, CalibrationModel | DetectorProfile | None] = {}

    def replace_peak_models(self, peaks, *, spectrum_id=None):
        assert spectrum_id is not None
        self.peaks[spectrum_id] = tuple(peaks)

    def replace_all_peak_models(self, peaks):
        grouped: dict[str, list[PeakModel]] = {}
        for peak in peaks:
            grouped.setdefault(peak.spectrum_id, []).append(peak)
        self.peaks = {key: tuple(values) for key, values in grouped.items()}

    def upsert_peak_model(self, peak, *, index=None):
        items = list(self.peaks.get(peak.spectrum_id, ()))
        for index, current in enumerate(items):
            if current.peak_id == peak.peak_id:
                items[index] = peak
                break
        else:
            if index is None:
                items.append(peak)
            else:
                items.insert(max(0, min(index, len(items))), peak)
        self.peaks[peak.spectrum_id] = tuple(items)

    def delete_peak_model(self, spectrum_id, peak_id):
        self.peaks[spectrum_id] = tuple(
            peak for peak in self.peaks.get(spectrum_id, ()) if peak.peak_id != peak_id
        )

    def upsert_roi(self, roi, *, index=None):
        if index is None or roi.roi_id in self.rois:
            self.rois[roi.roi_id] = roi
            return
        items = list(self.rois.items())
        items.insert(max(0, min(index, len(items))), (roi.roi_id, roi))
        self.rois = dict(items)

    def delete_roi(self, roi_id):
        self.rois.pop(roi_id, None)

    def upsert_fit_diagnostic(self, diagnostic):
        self.diagnostics[diagnostic.diagnostic_id] = diagnostic

    def assign_spectrum_role(self, role, spectrum_ids):
        resolved = tuple(spectrum_ids or ())
        if resolved:
            self.roles[role] = resolved
        else:
            self.roles.pop(role, None)

    def upsert_detector_profile(self, profile):
        self.profiles[profile.detector_profile_id] = profile

    def delete_detector_profile(self, profile_id):
        self.profiles.pop(profile_id, None)

    def upsert_viewport(self, viewport):
        self.viewports[viewport.viewport_id] = viewport

    def delete_viewport(self, viewport_id):
        self.viewports.pop(viewport_id, None)

    def set_pinned_nuclides(self, pinned_nuclides):
        self.pinned = tuple(pinned_nuclides)

    def apply_calibration(self, spectrum_id, calibration_state):
        self.calibrations[spectrum_id] = calibration_state


def _peak(peak_id: str = "peak-1", centroid: float = 10.0) -> PeakModel:
    return PeakModel(
        peak_id=peak_id,
        spectrum_id="spec-1",
        roi_id=None,
        centroid_channel=centroid,
        centroid_energy_keV=centroid * 2.0,
    )


def _roi(signal=(9.0, 11.0)) -> AnalysisROI:
    return AnalysisROI(
        roi_id="roi-1",
        spectrum_id="spec-1",
        signal_range=signal,
        left_background_range=(6.0, 8.0),
        right_background_range=(12.0, 14.0),
    )


def _linked_document() -> WorkspaceDocument:
    spectrum = GammaSpectrum(
        counts=np.asarray([3.0, 8.0, 4.0]),
        channels=np.arange(3),
        energies=np.asarray([9.0, 10.0, 11.0]),
        live_time=10.0,
        real_time=11.0,
        spectrum_id="spec-1",
    )
    peak = replace(_peak(), roi_id="roi-1")
    roi = replace(_roi(), associated_peak_ids=("peak-1",))
    diagnostic = FitDiagnostics(
        diagnostic_id="diag-1",
        spectrum_id="spec-1",
        roi_id="roi-1",
        peak_id="peak-1",
        fit_revision=1,
        status="valid",
        x=(9.0, 10.0, 11.0),
        observed=(3.0, 8.0, 4.0),
        model=(3.2, 7.6, 4.1),
        uncertainty=(1.0, 1.0, 1.0),
        normalized_residuals=(-0.2, 0.4, -0.1),
    )
    return WorkspaceDocument(
        spectra=(
            WorkspaceSpectrum(
                spectrum_id="spec-1",
                spectrum=spectrum,
                label="Linked test",
            ),
        ),
        active_spectrum_id="spec-1",
        rois=(roi,),
        peaks=(peak,),
        fit_diagnostics=(diagnostic,),
        viewports=(
            CanvasViewport(
                viewport_id="main",
                spectrum_id="spec-1",
                selected_roi_id="roi-1",
            ),
        ),
    )


def _semantic_payload(document: WorkspaceDocument) -> dict:
    payload = document.to_dict()
    payload.pop("updated_at", None)
    return payload


def _ordered_linked_document() -> WorkspaceDocument:
    base = _linked_document()
    secondary_spectrum = GammaSpectrum(
        counts=np.asarray([1.0, 2.0, 1.0]),
        channels=np.arange(3),
        energies=np.asarray([9.0, 10.0, 11.0]),
        live_time=10.0,
        real_time=10.0,
        spectrum_id="spec-2",
    )
    interleaved = PeakModel(
        peak_id="other-peak",
        spectrum_id="spec-2",
        roi_id=None,
        centroid_channel=1.0,
        centroid_energy_keV=10.0,
    )
    second_roi = replace(
        _roi((20.0, 22.0)),
        roi_id="roi-2",
        left_background_range=(17.0, 19.0),
        right_background_range=(23.0, 25.0),
        associated_peak_ids=("peak-2",),
    )
    second_peak = replace(
        _peak("peak-2", 21.0),
        roi_id="roi-2",
        centroid_energy_keV=21.0,
    )
    return replace(
        base,
        spectra=base.spectra
        + (
            WorkspaceSpectrum(
                spectrum_id="spec-2",
                spectrum=secondary_spectrum,
                label="Interleaved spectrum",
            ),
        ),
        rois=(base.rois[0], second_roi),
        peaks=(base.peaks[0], interleaved, second_peak),
    )


@pytest.mark.parametrize("operation", ["delete_peak", "replace_peaks", "delete_roi"])
def test_destructive_focused_commands_restore_all_dependent_leaves(operation):
    original = _linked_document()
    controller = AnalysisWorkspaceController(original)

    if operation == "delete_peak":
        command = DeletePeakCommand(controller, peak=original.peaks[0])
    elif operation == "replace_peaks":
        command = ReplacePeakSetCommand(
            controller,
            spectrum_id="spec-1",
            before=original.peaks,
            after=(),
        )
    else:
        command = DeleteROICommand(controller, roi=original.rois[0])

    command.redo()
    assert _semantic_payload(controller.document) != _semantic_payload(original)
    command.undo()

    assert _semantic_payload(controller.document) == _semantic_payload(original)


@pytest.mark.parametrize("operation", ["delete_peak", "replace_peaks", "delete_roi"])
def test_destructive_undo_restores_global_leaf_order(operation):
    original = _ordered_linked_document()
    controller = AnalysisWorkspaceController(original)

    if operation == "delete_peak":
        command = DeletePeakCommand(controller, peak=original.peaks[0])
    elif operation == "replace_peaks":
        spectrum_peaks = tuple(
            item for item in original.peaks if item.spectrum_id == "spec-1"
        )
        command = ReplacePeakSetCommand(
            controller,
            spectrum_id="spec-1",
            before=spectrum_peaks,
            after=(spectrum_peaks[1],),
        )
    else:
        command = DeleteROICommand(controller, roi=original.rois[0])

    command.redo()
    command.undo()

    assert _semantic_payload(controller.document) == _semantic_payload(original)


def test_peak_commands_redo_and_undo_only_target_peak_leaves():
    controller = FakeLeafController()
    peak_before = _peak()
    peak_after = _peak(centroid=10.5)
    second = _peak("peak-2", 20.0)

    replace_set = ReplacePeakSetCommand(
        controller,
        spectrum_id="spec-1",
        before=(peak_before,),
        after=(peak_after, second),
    )
    replace_set.redo()
    assert controller.peaks["spec-1"] == (peak_after, second)
    replace_set.undo()
    assert controller.peaks["spec-1"] == (peak_before,)

    update = UpdatePeakCommand(controller, before=peak_before, after=peak_after)
    update.redo()
    assert controller.peaks["spec-1"] == (peak_after,)
    update.undo()
    assert controller.peaks["spec-1"] == (peak_before,)

    add = UpdatePeakCommand(controller, before=None, after=second)
    add.redo()
    assert second in controller.peaks["spec-1"]
    add.undo()
    assert second not in controller.peaks["spec-1"]

    delete = DeletePeakCommand(controller, peak=peak_before)
    delete.redo()
    assert controller.peaks["spec-1"] == ()
    delete.undo()
    assert controller.peaks["spec-1"] == (peak_before,)


def test_assignment_pin_roi_role_profile_viewport_and_calibration_commands():
    controller = FakeLeafController()
    peak_before = _peak()
    assignment = NuclideAssignment(nuclide="Cs-137", line_energy_keV=661.657)
    peak_after = replace(peak_before, assignments=(assignment,), tags=("reference",))
    assignment_command = UpdatePeakAssignmentCommand(
        controller, before=peak_before, after=peak_after
    )
    assignment_command.redo()
    assert controller.peaks["spec-1"] == (peak_after,)
    assignment_command.undo()
    assert controller.peaks["spec-1"] == (peak_before,)

    pin = TogglePinnedNuclideCommand(
        controller, before=("Co-60",), after=("Co-60", "Cs-137")
    )
    pin.redo()
    assert controller.pinned == ("Co-60", "Cs-137")
    pin.undo()
    assert controller.pinned == ("Co-60",)

    roi = _roi()
    upsert_roi = UpsertROICommand(controller, before=None, after=roi)
    upsert_roi.redo()
    assert controller.rois == {"roi-1": roi}
    upsert_roi.undo()
    assert controller.rois == {}
    controller.upsert_roi(roi)
    delete_roi = DeleteROICommand(controller, roi=roi)
    delete_roi.redo()
    assert controller.rois == {}
    delete_roi.undo()
    assert controller.rois == {"roi-1": roi}

    role = AssignSpectrumRoleCommand(
        controller,
        role="foreground",
        before=("old-spec",),
        after=("spec-1",),
    )
    role.redo()
    assert controller.roles["foreground"] == ("spec-1",)
    role.undo()
    assert controller.roles["foreground"] == ("old-spec",)

    profile = DetectorProfile(detector_profile_id="hpge-1", detector_id="HPGe")
    profile_command = UpdateDetectorProfileCommand(
        controller, before=None, after=profile
    )
    profile_command.redo()
    assert controller.profiles == {"hpge-1": profile}
    profile_command.undo()
    assert controller.profiles == {}

    viewport = CanvasViewport(
        viewport_id="main", spectrum_id="spec-1", x_range=(5.0, 25.0)
    )
    viewport_command = UpdateCanvasViewportCommand(
        controller, before=None, after=viewport
    )
    viewport_command.redo()
    assert controller.viewports == {"main": viewport}
    viewport_command.undo()
    assert controller.viewports == {}

    old_calibration = CalibrationModel(model_key="linear", coefficients=(0.0, 1.0))
    new_calibration = CalibrationModel(model_key="linear", coefficients=(0.1, 1.01))
    calibration_command = ApplyCalibrationCommand(
        controller,
        spectrum_id="spec-1",
        before=old_calibration,
        after=new_calibration,
    )
    calibration_command.redo()
    assert controller.calibrations["spec-1"] is new_calibration
    calibration_command.undo()
    assert controller.calibrations["spec-1"] is old_calibration


def test_roi_drag_commands_merge_only_for_same_gesture_and_identity():
    controller = FakeLeafController()
    first = _roi((9.0, 11.0))
    middle = _roi((9.2, 11.2))
    final = _roi((9.5, 11.5))
    stack = QUndoStack()

    stack.push(
        MoveROIBoundsCommand(
            controller, before=first, after=middle, drag_token="drag-17"
        )
    )
    stack.push(
        MoveROIBoundsCommand(
            controller, before=middle, after=final, drag_token="drag-17"
        )
    )

    assert stack.count() == 1
    assert controller.rois["roi-1"] is final
    stack.undo()
    assert controller.rois["roi-1"] is first
    stack.redo()
    assert controller.rois["roi-1"] is final

    stack.push(
        MoveROIBoundsCommand(
            controller,
            before=final,
            after=_roi((9.6, 11.6)),
            drag_token="drag-18",
        )
    )
    assert stack.count() == 2


def test_commands_never_retain_documents_legacy_states_or_spectrum_arrays():
    controller = FakeLeafController()
    peak = _peak()
    roi = _roi()
    profile = DetectorProfile(detector_profile_id="hpge-1")
    viewport = CanvasViewport(viewport_id="main", spectrum_id="spec-1")
    calibration = CalibrationModel(model_key="linear", coefficients=(0.0, 1.0))
    commands = (
        ReplacePeakSetCommand(
            controller,
            spectrum_id="spec-1",
            before=(),
            after=(peak,),
        ),
        UpdatePeakCommand(controller, before=None, after=peak),
        DeletePeakCommand(controller, peak=peak),
        UpdatePeakAssignmentCommand(controller, before=peak, after=peak),
        TogglePinnedNuclideCommand(controller, before=(), after=("Cs-137",)),
        UpsertROICommand(controller, before=None, after=roi),
        DeleteROICommand(controller, roi=roi),
        MoveROIBoundsCommand(controller, before=roi, after=roi, drag_token="drag-1"),
        AssignSpectrumRoleCommand(
            controller, role="foreground", before=(), after=("spec-1",)
        ),
        UpdateDetectorProfileCommand(controller, before=None, after=profile),
        UpdateCanvasViewportCommand(controller, before=None, after=viewport),
        ApplyCalibrationCommand(
            controller,
            spectrum_id="spec-1",
            before=None,
            after=calibration,
        ),
    )

    forbidden = (WorkspaceDocument, AnalysisWorkspaceState, GammaSpectrum, np.ndarray)
    for command in commands:
        assert not any(
            isinstance(value, forbidden) for value in vars(command).values()
        ), command.__class__.__name__


def test_commands_reject_identity_changes_and_non_leaf_calibration():
    controller = FakeLeafController()
    with pytest.raises(ValueError, match="same peak"):
        UpdatePeakCommand(controller, before=_peak(), after=_peak("other"))
    with pytest.raises(ValueError, match="same ROI"):
        MoveROIBoundsCommand(
            controller,
            before=_roi(),
            after=replace(_roi(), roi_id="other"),
            drag_token="drag",
        )
    with pytest.raises(TypeError, match="CalibrationModel"):
        ApplyCalibrationCommand(
            controller,
            spectrum_id="spec-1",
            before=None,
            after=np.arange(4.0),  # type: ignore[arg-type]
        )
