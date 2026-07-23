"""Focused undo commands for canonical analysis-workspace leaf edits.

These commands intentionally retain only immutable domain leaves and stable IDs.
In particular, they never retain a ``WorkspaceDocument``, the legacy projected
``AnalysisWorkspaceState``, a ``GammaSpectrum``, or any spectrum array.  A drag
may issue many previews, but ``MoveROIBoundsCommand`` merges committed updates
that share one explicit drag token into a single undo-stack entry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence, TypeAlias

from fluxforge.core.workspace_document import (
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    DetectorProfile,
    FitDiagnostics,
    PeakModel,
    WorkspaceDocument,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:  # pragma: no branch - selected once at import time
    from fluxforge.gui.qt_compat import QUndoCommand
else:  # pragma: no cover - import-safe fallback for non-GUI installations

    class QUndoCommand:  # type: ignore[no-redef]
        """Minimal import-safe stand-in; Qt is required to use an undo stack."""

        def __init__(self, text: str = "") -> None:
            self._text = text

        def text(self) -> str:
            return self._text

        def id(self) -> int:
            return -1

        def mergeWith(self, other: object) -> bool:  # noqa: N802 - Qt API
            return False


CalibrationLeaf: TypeAlias = CalibrationModel | DetectorProfile | None


def _copy_workflow_leaf(value: Any) -> Any:
    """Copy immutable JSON-like workflow leaves, including mapping proxies."""

    if isinstance(value, Mapping):
        return {str(key): _copy_workflow_leaf(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_workflow_leaf(item) for item in value)
    if isinstance(value, list):
        return [_copy_workflow_leaf(item) for item in value]
    return deepcopy(value)


def _invalidated_diagnostic(
    diagnostic: FitDiagnostics,
    reason: str,
) -> FitDiagnostics:
    """Return an explicit invalid state without obsolete fit arrays."""

    return replace(
        diagnostic,
        status="invalid",
        x=(),
        observed=(),
        model=(),
        uncertainty=(),
        normalized_residuals=(),
        goodness_of_fit={},
        warning_flags=tuple(
            dict.fromkeys((*diagnostic.warning_flags, "analysis-edit-invalidated"))
        ),
        provenance={
            **dict(diagnostic.provenance),
            "invalidated_by": reason,
        },
    )


def _invalidated_peak(peak: PeakModel, reason: str) -> PeakModel:
    """Return a peak leaf with no displayable residuals from an obsolete fit."""

    return replace(
        peak,
        status="invalidated",
        fit_quality=0.0,
        normalized_residuals=(),
        residual_channels=(),
        provenance={
            **dict(peak.provenance),
            "fit_invalidated_by": reason,
        },
    )


@dataclass(frozen=True)
class CalibrationUndoState:
    """Small exact snapshot of calibration-owned persisted leaves."""

    detector_profile_id: str | None
    detector_profiles: tuple[DetectorProfile, ...]
    calibration_payload: dict
    energies: tuple[float, ...] | None

    @classmethod
    def from_document(
        cls, document: WorkspaceDocument, spectrum_id: str
    ) -> "CalibrationUndoState":
        record = document.spectrum_by_id(spectrum_id)
        if record is None:
            raise KeyError(f"Unknown spectrum ID: {spectrum_id}")
        energies = (
            tuple(float(value) for value in record.spectrum.energies)
            if record.spectrum.energies is not None
            else None
        )
        return cls(
            detector_profile_id=record.detector_profile_id,
            detector_profiles=tuple(document.detector_profiles),
            calibration_payload=deepcopy(dict(record.spectrum.calibration or {})),
            energies=energies,
        )


class WorkspaceLeafController(Protocol):
    """Small controller surface required by the focused commands."""

    def replace_peak_models(
        self, peaks: Sequence[PeakModel], *, spectrum_id: str | None = None
    ) -> object: ...

    def replace_all_peak_models(self, peaks: Sequence[PeakModel]) -> object: ...

    def upsert_peak_model(
        self, peak: PeakModel, *, index: int | None = None
    ) -> object: ...

    def delete_peak_model(self, spectrum_id: str, peak_id: str) -> object: ...

    def upsert_roi(self, roi: AnalysisROI, *, index: int | None = None) -> object: ...

    def delete_roi(self, roi_id: str) -> object: ...

    def upsert_fit_diagnostic(self, diagnostic: FitDiagnostics) -> object: ...

    def set_fit_diagnostics(self, diagnostics: Sequence[FitDiagnostics]) -> object: ...

    def set_workflow_state(self, workflow_state: Mapping[str, Any]) -> object: ...

    def assign_spectrum_role(
        self, role: str, spectrum_ids: str | Sequence[str] | None
    ) -> object: ...

    def upsert_detector_profile(self, profile: DetectorProfile) -> object: ...

    def delete_detector_profile(self, profile_id: str) -> object: ...

    def upsert_viewport(self, viewport: CanvasViewport) -> object: ...

    def delete_viewport(self, viewport_id: str) -> object: ...

    def set_pinned_nuclides(self, pinned_nuclides: Sequence[str]) -> object: ...

    def set_nuclide_tags(self, nuclide_tags: Mapping[str, Sequence[str]]) -> object: ...

    def apply_calibration(
        self, spectrum_id: str, calibration_state: CalibrationLeaf
    ) -> object: ...

    def restore_calibration_state(
        self, spectrum_id: str, state: CalibrationUndoState
    ) -> object: ...


class _WorkspaceLeafCommand(QUndoCommand):
    """Shared command base that retains only the controller reference."""

    def __init__(self, controller: WorkspaceLeafController, description: str) -> None:
        super().__init__(description)
        self.controller = controller


class ReplacePeakSetCommand(_WorkspaceLeafCommand):
    """Replace all peak-model leaves for one spectrum."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        spectrum_id: str,
        before: Sequence[PeakModel],
        after: Sequence[PeakModel],
        before_document_peaks: Sequence[PeakModel] | None = None,
        related_rois: Sequence[AnalysisROI] | None = None,
        related_diagnostics: Sequence[FitDiagnostics] | None = None,
        description: str = "Replace peaks",
    ) -> None:
        super().__init__(controller, description)
        self.spectrum_id = spectrum_id
        self.before = tuple(before)
        self.after = tuple(after)
        if any(peak.spectrum_id != spectrum_id for peak in self.before + self.after):
            raise ValueError("all peaks must belong to spectrum_id")
        removed_peak_ids = {item.peak_id for item in self.before} - {
            item.peak_id for item in self.after
        }
        document = _controller_document(controller)
        if before_document_peaks is None and document is not None:
            before_document_peaks = document.peaks
        self.before_document_peaks = (
            tuple(before_document_peaks) if before_document_peaks is not None else None
        )
        if related_rois is None:
            related_rois = (
                tuple(
                    item
                    for item in document.rois
                    if item.spectrum_id == spectrum_id
                    and removed_peak_ids.intersection(item.associated_peak_ids)
                )
                if document is not None
                else ()
            )
        if related_diagnostics is None:
            related_diagnostics = (
                tuple(
                    item
                    for item in document.fit_diagnostics
                    if item.spectrum_id == spectrum_id
                    and item.peak_id in removed_peak_ids
                )
                if document is not None
                else ()
            )
        self.related_rois = tuple(related_rois)
        self.related_diagnostics = tuple(related_diagnostics)

    def undo(self) -> None:
        if self.before_document_peaks is not None:
            self.controller.replace_all_peak_models(self.before_document_peaks)
        else:
            self.controller.replace_peak_models(
                self.before, spectrum_id=self.spectrum_id
            )
        for roi in self.related_rois:
            self.controller.upsert_roi(roi)
        for diagnostic in self.related_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)

    def redo(self) -> None:
        self.controller.replace_peak_models(self.after, spectrum_id=self.spectrum_id)


class UpdatePeakCommand(_WorkspaceLeafCommand):
    """Add or replace one peak-model leaf."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: PeakModel | None,
        after: PeakModel,
        invalidate_diagnostics: bool = False,
        description: str = "Update peak",
    ) -> None:
        super().__init__(controller, description)
        if before is not None and (
            before.spectrum_id,
            before.peak_id,
        ) != (after.spectrum_id, after.peak_id):
            raise ValueError("before and after must identify the same peak")
        self.spectrum_id = after.spectrum_id
        self.peak_id = after.peak_id
        self.before = before
        self.after = after
        document = _controller_document(controller)
        self.related_diagnostics = (
            tuple(
                item
                for item in document.fit_diagnostics
                if item.spectrum_id == after.spectrum_id
                and (
                    item.peak_id == after.peak_id
                    or (after.roi_id is not None and item.roi_id == after.roi_id)
                )
            )
            if invalidate_diagnostics and document is not None
            else ()
        )
        self.invalid_diagnostics = tuple(
            _invalidated_diagnostic(item, description)
            for item in self.related_diagnostics
        )
        self.related_roi_peaks = (
            tuple(
                item
                for item in document.peaks
                if item.spectrum_id == after.spectrum_id
                and after.roi_id is not None
                and item.roi_id == after.roi_id
                and item.peak_id != after.peak_id
            )
            if invalidate_diagnostics and document is not None
            else ()
        )
        self.invalid_roi_peaks = tuple(
            _invalidated_peak(item, description) for item in self.related_roi_peaks
        )

    def undo(self) -> None:
        if self.before is None:
            self.controller.delete_peak_model(self.spectrum_id, self.peak_id)
        else:
            self.controller.upsert_peak_model(self.before)
        for peak in self.related_roi_peaks:
            self.controller.upsert_peak_model(peak)
        for diagnostic in self.related_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)

    def redo(self) -> None:
        self.controller.upsert_peak_model(self.after)
        for peak in self.invalid_roi_peaks:
            self.controller.upsert_peak_model(peak)
        for diagnostic in self.invalid_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)


class DeletePeakCommand(_WorkspaceLeafCommand):
    """Delete one peak and retain only directly affected leaves for undo."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        peak: PeakModel,
        peak_index: int | None = None,
        related_rois: Sequence[AnalysisROI] | None = None,
        related_diagnostics: Sequence[FitDiagnostics] | None = None,
        description: str = "Delete peak",
    ) -> None:
        super().__init__(controller, description)
        self.peak = peak
        self.spectrum_id = peak.spectrum_id
        self.peak_id = peak.peak_id
        document = _controller_document(controller)
        if peak_index is None and document is not None:
            peak_index = next(
                (
                    index
                    for index, item in enumerate(document.peaks)
                    if (item.spectrum_id, item.peak_id)
                    == (self.spectrum_id, self.peak_id)
                ),
                None,
            )
        self.peak_index = peak_index
        if related_rois is None:
            related_rois = (
                tuple(
                    item
                    for item in document.rois
                    if item.spectrum_id == self.spectrum_id
                    and self.peak_id in item.associated_peak_ids
                )
                if document is not None
                else ()
            )
        if related_diagnostics is None:
            related_diagnostics = (
                tuple(
                    item
                    for item in document.fit_diagnostics
                    if item.spectrum_id == self.spectrum_id
                    and item.peak_id == self.peak_id
                )
                if document is not None
                else ()
            )
        self.related_rois = tuple(related_rois)
        self.related_diagnostics = tuple(related_diagnostics)

    def undo(self) -> None:
        self.controller.upsert_peak_model(self.peak, index=self.peak_index)
        for roi in self.related_rois:
            self.controller.upsert_roi(roi)
        for diagnostic in self.related_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)

    def redo(self) -> None:
        self.controller.delete_peak_model(self.spectrum_id, self.peak_id)


class UpdatePeakAssignmentCommand(_WorkspaceLeafCommand):
    """Update assignments/tags represented by old and new peak leaves."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: PeakModel,
        after: PeakModel,
        description: str = "Update peak assignment",
    ) -> None:
        super().__init__(controller, description)
        if (before.spectrum_id, before.peak_id) != (
            after.spectrum_id,
            after.peak_id,
        ):
            raise ValueError("before and after must identify the same peak")
        self.before = before
        self.after = after

    def undo(self) -> None:
        self.controller.upsert_peak_model(self.before)

    def redo(self) -> None:
        self.controller.upsert_peak_model(self.after)


class TogglePinnedNuclideCommand(_WorkspaceLeafCommand):
    """Replace the small ordered set of pinned nuclide identifiers."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: Sequence[str],
        after: Sequence[str],
        description: str = "Update pinned nuclides",
    ) -> None:
        super().__init__(controller, description)
        self.before = tuple(before)
        self.after = tuple(after)

    def undo(self) -> None:
        self.controller.set_pinned_nuclides(self.before)

    def redo(self) -> None:
        self.controller.set_pinned_nuclides(self.after)


class UpdateNuclideTagsCommand(_WorkspaceLeafCommand):
    """Replace the small persisted map of analyst tags by nuclide."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: Mapping[str, Sequence[str]],
        after: Mapping[str, Sequence[str]],
        description: str = "Tag nuclide",
    ) -> None:
        super().__init__(controller, description)
        self.before = {key: tuple(value) for key, value in before.items()}
        self.after = {key: tuple(value) for key, value in after.items()}

    def undo(self) -> None:
        self.controller.set_nuclide_tags(self.before)

    def redo(self) -> None:
        self.controller.set_nuclide_tags(self.after)


class UpdateWorkflowStateCommand(_WorkspaceLeafCommand):
    """Replace derived workflow state so scientific edits cannot leave stale output."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        description: str = "Invalidate derived analysis",
    ) -> None:
        super().__init__(controller, description)
        self.before = _copy_workflow_leaf(before)
        self.after = _copy_workflow_leaf(after)

    def undo(self) -> None:
        self.controller.set_workflow_state(self.before)

    def redo(self) -> None:
        self.controller.set_workflow_state(self.after)


class UpsertROICommand(_WorkspaceLeafCommand):
    """Add or replace one ROI leaf."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: AnalysisROI | None,
        after: AnalysisROI,
        description: str = "Update ROI",
    ) -> None:
        super().__init__(controller, description)
        if before is not None and before.roi_id != after.roi_id:
            raise ValueError("before and after must identify the same ROI")
        self.roi_id = after.roi_id
        self.before = before
        self.after = after

    def undo(self) -> None:
        if self.before is None:
            self.controller.delete_roi(self.roi_id)
        else:
            self.controller.upsert_roi(self.before)

    def redo(self) -> None:
        self.controller.upsert_roi(self.after)


class DeleteROICommand(_WorkspaceLeafCommand):
    """Delete one ROI and retain only directly affected leaves for undo."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        roi: AnalysisROI,
        roi_index: int | None = None,
        related_peaks: Sequence[PeakModel] | None = None,
        related_diagnostics: Sequence[FitDiagnostics] | None = None,
        related_viewports: Sequence[CanvasViewport] | None = None,
        description: str = "Delete ROI",
    ) -> None:
        super().__init__(controller, description)
        self.roi = roi
        self.roi_id = roi.roi_id
        document = _controller_document(controller)
        if roi_index is None and document is not None:
            roi_index = next(
                (
                    index
                    for index, item in enumerate(document.rois)
                    if item.roi_id == self.roi_id
                ),
                None,
            )
        self.roi_index = roi_index
        if related_peaks is None:
            related_peaks = (
                tuple(item for item in document.peaks if item.roi_id == self.roi_id)
                if document is not None
                else ()
            )
        if related_diagnostics is None:
            related_diagnostics = (
                tuple(
                    item
                    for item in document.fit_diagnostics
                    if item.roi_id == self.roi_id
                )
                if document is not None
                else ()
            )
        if related_viewports is None:
            related_viewports = (
                tuple(
                    item
                    for item in document.viewports
                    if item.selected_roi_id == self.roi_id
                )
                if document is not None
                else ()
            )
        self.related_peaks = tuple(related_peaks)
        self.related_diagnostics = tuple(related_diagnostics)
        self.related_viewports = tuple(related_viewports)

    def undo(self) -> None:
        self.controller.upsert_roi(self.roi, index=self.roi_index)
        for peak in self.related_peaks:
            self.controller.upsert_peak_model(peak)
        for diagnostic in self.related_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)
        for viewport in self.related_viewports:
            self.controller.upsert_viewport(viewport)

    def redo(self) -> None:
        self.controller.delete_roi(self.roi_id)


def _controller_document(
    controller: WorkspaceLeafController,
) -> WorkspaceDocument | None:
    """Return a controller's canonical document without widening the protocol."""

    document = getattr(controller, "document", None)
    return document if isinstance(document, WorkspaceDocument) else None


class MoveROIBoundsCommand(_WorkspaceLeafCommand):
    """Commit one ROI-bound drag, merging updates from the same drag gesture."""

    COMMAND_ID = 0x464652  # stable, process-independent ``FFR`` identifier

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: AnalysisROI,
        after: AnalysisROI,
        drag_token: str,
        description: str = "Move ROI bounds",
    ) -> None:
        super().__init__(controller, description)
        if (before.spectrum_id, before.roi_id) != (
            after.spectrum_id,
            after.roi_id,
        ):
            raise ValueError("before and after must identify the same ROI")
        if not drag_token:
            raise ValueError("drag_token must be a non-empty stable gesture ID")
        self.spectrum_id = before.spectrum_id
        self.roi_id = before.roi_id
        self.drag_token = drag_token
        self.before = before
        self.after = after
        document = _controller_document(controller)
        self.related_peaks = (
            tuple(
                item
                for item in document.peaks
                if item.spectrum_id == before.spectrum_id
                and item.roi_id == before.roi_id
            )
            if document is not None
            else ()
        )
        self.invalid_peaks = tuple(
            _invalidated_peak(item, description) for item in self.related_peaks
        )
        self.related_diagnostics = (
            tuple(
                item
                for item in document.fit_diagnostics
                if item.spectrum_id == before.spectrum_id
                and item.roi_id == before.roi_id
            )
            if document is not None
            else ()
        )
        self.invalid_diagnostics = tuple(
            _invalidated_diagnostic(item, description)
            for item in self.related_diagnostics
        )

    def id(self) -> int:
        return self.COMMAND_ID

    def mergeWith(self, other: object) -> bool:  # noqa: N802 - Qt API
        if not isinstance(other, MoveROIBoundsCommand):
            return False
        if (
            self.controller is not other.controller
            or self.spectrum_id != other.spectrum_id
            or self.roi_id != other.roi_id
            or self.drag_token != other.drag_token
        ):
            return False
        self.after = other.after
        return True

    def undo(self) -> None:
        self.controller.upsert_roi(self.before)
        for peak in self.related_peaks:
            self.controller.upsert_peak_model(peak)
        for diagnostic in self.related_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)

    def redo(self) -> None:
        self.controller.upsert_roi(self.after)
        for peak in self.invalid_peaks:
            self.controller.upsert_peak_model(peak)
        for diagnostic in self.invalid_diagnostics:
            self.controller.upsert_fit_diagnostic(diagnostic)


class AssignSpectrumRoleCommand(_WorkspaceLeafCommand):
    """Replace one named spectrum-role assignment."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        role: str,
        before: Sequence[str],
        after: Sequence[str],
        description: str = "Assign spectrum role",
    ) -> None:
        super().__init__(controller, description)
        self.role = role
        self.before = tuple(before)
        self.after = tuple(after)

    def undo(self) -> None:
        self.controller.assign_spectrum_role(self.role, self.before)

    def redo(self) -> None:
        self.controller.assign_spectrum_role(self.role, self.after)


class UpdateDetectorProfileCommand(_WorkspaceLeafCommand):
    """Add or replace one detector-profile leaf."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: DetectorProfile | None,
        after: DetectorProfile,
        description: str = "Update detector profile",
    ) -> None:
        super().__init__(controller, description)
        if before is not None and (
            before.detector_profile_id != after.detector_profile_id
        ):
            raise ValueError("before and after must identify the same detector profile")
        self.profile_id = after.detector_profile_id
        self.before = before
        self.after = after

    def undo(self) -> None:
        if self.before is None:
            self.controller.delete_detector_profile(self.profile_id)
        else:
            self.controller.upsert_detector_profile(self.before)

    def redo(self) -> None:
        self.controller.upsert_detector_profile(self.after)


class UpdateCanvasViewportCommand(_WorkspaceLeafCommand):
    """Add or replace one persisted canvas viewport leaf."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        before: CanvasViewport | None,
        after: CanvasViewport,
        description: str = "Update canvas view",
    ) -> None:
        super().__init__(controller, description)
        if before is not None and before.viewport_id != after.viewport_id:
            raise ValueError("before and after must identify the same viewport")
        self.viewport_id = after.viewport_id
        self.before = before
        self.after = after

    def undo(self) -> None:
        if self.before is None:
            self.controller.delete_viewport(self.viewport_id)
        else:
            self.controller.upsert_viewport(self.before)

    def redo(self) -> None:
        self.controller.upsert_viewport(self.after)


class ApplyCalibrationCommand(_WorkspaceLeafCommand):
    """Apply a calibration leaf without retaining a spectrum or its arrays."""

    def __init__(
        self,
        controller: WorkspaceLeafController,
        *,
        spectrum_id: str,
        before: CalibrationLeaf,
        after: CalibrationLeaf,
        before_state: CalibrationUndoState | None = None,
        description: str = "Apply calibration",
    ) -> None:
        super().__init__(controller, description)
        for value in (before, after):
            if value is not None and not isinstance(
                value, (CalibrationModel, DetectorProfile)
            ):
                raise TypeError(
                    "calibration state must be CalibrationModel, DetectorProfile, or None"
                )
        self.spectrum_id = spectrum_id
        self.before = before
        self.after = after
        document = _controller_document(controller)
        self.before_state = (
            before_state
            if before_state is not None
            else (
                CalibrationUndoState.from_document(document, spectrum_id)
                if document is not None
                else None
            )
        )

    def undo(self) -> None:
        if self.before_state is not None:
            self.controller.restore_calibration_state(
                self.spectrum_id, self.before_state
            )
        else:
            self.controller.apply_calibration(self.spectrum_id, self.before)

    def redo(self) -> None:
        self.controller.apply_calibration(self.spectrum_id, self.after)


__all__ = [
    "ApplyCalibrationCommand",
    "AssignSpectrumRoleCommand",
    "CalibrationUndoState",
    "DeletePeakCommand",
    "DeleteROICommand",
    "MoveROIBoundsCommand",
    "ReplacePeakSetCommand",
    "TogglePinnedNuclideCommand",
    "UpdateCanvasViewportCommand",
    "UpdateDetectorProfileCommand",
    "UpdateNuclideTagsCommand",
    "UpdatePeakAssignmentCommand",
    "UpdatePeakCommand",
    "UpdateWorkflowStateCommand",
    "UpsertROICommand",
    "WorkspaceLeafController",
]
