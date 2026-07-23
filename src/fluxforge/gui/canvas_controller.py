"""Translate renderer-neutral canvas intents into focused workspace edits."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Callable
from uuid import uuid4

import numpy as np

from fluxforge.core.analysis_workspace import compute_cascade_sum_lines
from fluxforge.core.workspace_document import (
    AnalysisROI,
    CanvasViewport,
    NuclideAssignment,
    PeakComponent,
    PeakModel,
)
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.workspace_undo import (
    AssignSpectrumRoleCommand,
    DeletePeakCommand,
    DeleteROICommand,
    MoveROIBoundsCommand,
    TogglePinnedNuclideCommand,
    UpdateCanvasViewportCommand,
    UpdateNuclideTagsCommand,
    UpdatePeakCommand,
    UpdateWorkflowStateCommand,
    UpsertROICommand,
)


class CanvasIntentDispatcher:
    """Apply validated canvas events without placing analysis logic in widgets."""

    def __init__(
        self,
        workspace_controller: AnalysisWorkspaceController,
        selection_bus: SelectionBus,
        *,
        undo_stack=None,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self.workspace_controller = workspace_controller
        self.selection_bus = selection_bus
        self.undo_stack = undo_stack
        self.id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex[:12]}")

    def handle(self, intent: CanvasIntent) -> None:
        """Validate and apply one complete user intent."""

        intent.validate()
        handlers = {
            CanvasIntentKind.CREATE_ROI: self._create_roi,
            CanvasIntentKind.SELECT_ROI: self._select_roi,
            CanvasIntentKind.DELETE_ROI: self._delete_roi,
            CanvasIntentKind.MOVE_ROI: self._move_roi,
            CanvasIntentKind.MOVE_BACKGROUND: self._move_background,
            CanvasIntentKind.ADD_PEAK: self._add_peak,
            CanvasIntentKind.SELECT_PEAK: self._select_peak,
            CanvasIntentKind.MOVE_PEAK: self._move_peak,
            CanvasIntentKind.DELETE_PEAK: self._delete_peak,
            CanvasIntentKind.ADD_COMPONENT: self._add_component,
            CanvasIntentKind.SPLIT_PEAK: self._split_peak,
            CanvasIntentKind.MERGE_PEAKS: self._merge_components,
            CanvasIntentKind.ASSIGN_NUCLIDE: self._assign_nuclide,
            CanvasIntentKind.CLEAR_NUCLIDE: self._clear_nuclide,
            CanvasIntentKind.PIN_NUCLIDE: self._pin_nuclide,
            CanvasIntentKind.UNPIN_NUCLIDE: self._unpin_nuclide,
            CanvasIntentKind.TAG_NUCLIDE: self._tag_nuclide,
            CanvasIntentKind.TAG_PEAK: self._tag_peak,
            CanvasIntentKind.ASSIGN_SPECTRUM_ROLE: self._assign_role,
            CanvasIntentKind.CHANGE_VIEWPORT: self._change_viewport,
            CanvasIntentKind.TOGGLE_CROSSHAIR: self._toggle_crosshair,
            CanvasIntentKind.SET_LOG_SCALE: self._set_log_scale,
            CanvasIntentKind.SET_LABELS_VISIBLE: self._set_labels_visible,
        }
        handler = handlers.get(intent.kind)
        if handler is None:
            raise ValueError(
                f"Unsupported production canvas intent: {intent.kind.value}"
            )
        handler(intent)

    def _push(self, command) -> None:
        if self.undo_stack is None:
            command.redo()
        else:
            self.undo_stack.push(command)

    def _push_macro(self, description: str, commands: tuple[object, ...]) -> None:
        if self.undo_stack is None:
            for command in commands:
                command.redo()
            return
        self.undo_stack.beginMacro(description)
        try:
            for command in commands:
                self.undo_stack.push(command)
        finally:
            self.undo_stack.endMacro()

    def _invalidation_command(self, reason: str) -> UpdateWorkflowStateCommand:
        before = self.workspace_controller.document.workflow_state
        after = dict(before)
        legacy = dict(after.get("analysis_workspace_v1", {}))
        legacy.update(
            {
                "activity_results": [],
                "roi_analysis": None,
                "roi_statistics": None,
            }
        )
        after["analysis_workspace_v1"] = legacy
        after["analysis_invalidation"] = {
            "reason": reason,
            "requires_reanalysis": True,
        }
        return UpdateWorkflowStateCommand(
            self.workspace_controller,
            before=before,
            after=after,
            description="Invalidate derived analysis",
        )

    def _push_scientific(self, command, description: str) -> None:
        self._push_macro(
            description,
            (command, self._invalidation_command(description)),
        )

    def _cascade_workflow_command(
        self, pinned_nuclides: tuple[str, ...]
    ) -> UpdateWorkflowStateCommand:
        before = self.workspace_controller.document.workflow_state
        after = dict(before)
        legacy = dict(after.get("analysis_workspace_v1", {}))
        legacy["cascade_sum_lines_keV"] = list(
            compute_cascade_sum_lines(pinned_nuclides)
        )
        after["analysis_workspace_v1"] = legacy
        return UpdateWorkflowStateCommand(
            self.workspace_controller,
            before=before,
            after=after,
            description="Update cascade-sum overlays",
        )

    def _require_spectrum_id(self, intent: CanvasIntent) -> str:
        spectrum_id = intent.spectrum_id
        if (
            not spectrum_id
            or self.workspace_controller.document.spectrum_by_id(spectrum_id) is None
        ):
            raise KeyError(f"Unknown spectrum ID: {spectrum_id!r}")
        return spectrum_id

    def _require_roi(self, intent: CanvasIntent) -> AnalysisROI:
        roi = self.workspace_controller.document.roi_by_id(str(intent.roi_id or ""))
        if roi is None:
            raise KeyError(f"Unknown ROI ID: {intent.roi_id!r}")
        if intent.spectrum_id and roi.spectrum_id != intent.spectrum_id:
            raise ValueError("ROI does not belong to the intent spectrum")
        return roi

    def _require_peak(self, intent: CanvasIntent) -> PeakModel:
        spectrum_id = self._require_spectrum_id(intent)
        peak = self.workspace_controller.document.peak_by_id(
            spectrum_id, str(intent.peak_id or "")
        )
        if peak is None:
            raise KeyError(f"Unknown peak ID: {intent.peak_id!r}")
        return peak

    @staticmethod
    def _default_backgrounds(
        bounds: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        lower, upper = bounds
        width = max(upper - lower, 1.0e-6)
        sideband_width = max(0.4 * width, 1.0e-3)
        gap = max(0.08 * width, 1.0e-4)
        return (
            (lower - gap - sideband_width, lower - gap),
            (upper + gap, upper + gap + sideband_width),
        )

    def _viewport_selection_command(
        self, spectrum_id: str, roi_id: str | None
    ) -> UpdateCanvasViewportCommand:
        before = self.workspace_controller.document.viewport_by_id("primary-spectrum")
        after = (
            replace(before, spectrum_id=spectrum_id, selected_roi_id=roi_id)
            if before is not None
            else CanvasViewport(
                viewport_id="primary-spectrum",
                spectrum_id=spectrum_id,
                selected_roi_id=roi_id,
            )
        )
        return UpdateCanvasViewportCommand(
            self.workspace_controller,
            before=before,
            after=after,
            description="Select ROI",
        )

    def _publish_roi(self, roi: AnalysisROI) -> None:
        self.selection_bus.publish(
            SelectionState(
                spectrum_id=roi.spectrum_id,
                roi_id=roi.roi_id,
                roi_bounds_keV=roi.signal_range,
            )
        )

    def _publish_peak(self, peak: PeakModel) -> None:
        roi = (
            self.workspace_controller.document.roi_by_id(peak.roi_id)
            if peak.roi_id
            else None
        )
        candidate = self.workspace_controller._candidate_from_peak_model(peak)
        self.selection_bus.publish(
            SelectionState(
                spectrum_id=peak.spectrum_id,
                peak_id=peak.peak_id,
                roi_id=peak.roi_id,
                peak_energy_keV=peak.centroid_energy_keV,
                roi_bounds_keV=(
                    roi.signal_range if roi is not None else candidate.roi_bounds_keV
                ),
                nuclide=(peak.assignments[0].nuclide if peak.assignments else None),
                reference_lines_keV=peak.reference_lines_keV,
            )
        )

    def _create_roi(self, intent: CanvasIntent) -> None:
        spectrum_id = self._require_spectrum_id(intent)
        bounds = intent.bounds
        assert bounds is not None
        left, right = self._default_backgrounds(bounds)
        associated_peaks = tuple(
            peak
            for peak in self.workspace_controller.document.peaks
            if peak.spectrum_id == spectrum_id
            and peak.roi_id is None
            and bounds[0] <= peak.centroid_energy_keV <= bounds[1]
        )
        roi = AnalysisROI(
            roi_id=self.id_factory("roi"),
            spectrum_id=spectrum_id,
            signal_range=bounds,
            left_background_range=intent.left_background or left,
            right_background_range=intent.right_background or right,
            associated_peak_ids=tuple(peak.peak_id for peak in associated_peaks),
            label="ROI",
        )
        roi.validate("roi")
        commands: list[object] = [
            UpsertROICommand(
                self.workspace_controller,
                before=None,
                after=roi,
                description="Create ROI",
            )
        ]
        commands.extend(
            UpdatePeakCommand(
                self.workspace_controller,
                before=peak,
                after=self._invalidate_peak_fit(
                    replace(peak, roi_id=roi.roi_id),
                    "Create ROI",
                ),
                invalidate_diagnostics=True,
                description="Associate peak with ROI",
            )
            for peak in associated_peaks
        )
        commands.extend(
            (
                self._viewport_selection_command(spectrum_id, roi.roi_id),
                self._invalidation_command("Create ROI"),
            )
        )
        self._push_macro(
            "Create ROI",
            tuple(commands),
        )
        self._publish_roi(roi)

    def _select_roi(self, intent: CanvasIntent) -> None:
        roi = self._require_roi(intent)
        self._push(self._viewport_selection_command(roi.spectrum_id, roi.roi_id))
        self._publish_roi(roi)

    def _delete_roi(self, intent: CanvasIntent) -> None:
        roi = self._require_roi(intent)
        self._push_scientific(
            DeleteROICommand(
                self.workspace_controller,
                roi=roi,
                description="Delete ROI",
            ),
            "Delete ROI",
        )
        self.selection_bus.clear()

    @staticmethod
    def _backgrounds_for_signal(
        roi: AnalysisROI, bounds: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        left_width = roi.left_background_range[1] - roi.left_background_range[0]
        right_width = roi.right_background_range[1] - roi.right_background_range[0]
        left_gap = max(roi.signal_range[0] - roi.left_background_range[1], 0.0)
        right_gap = max(roi.right_background_range[0] - roi.signal_range[1], 0.0)
        return (
            (bounds[0] - left_gap - left_width, bounds[0] - left_gap),
            (bounds[1] + right_gap, bounds[1] + right_gap + right_width),
        )

    def _move_roi(self, intent: CanvasIntent) -> None:
        roi = self._require_roi(intent)
        assert intent.bounds is not None
        left, right = self._backgrounds_for_signal(roi, intent.bounds)
        updated = replace(
            roi,
            signal_range=intent.bounds,
            left_background_range=left,
            right_background_range=right,
            fit_revision=roi.fit_revision + 1,
        )
        self._push_scientific(
            MoveROIBoundsCommand(
                self.workspace_controller,
                before=roi,
                after=updated,
                drag_token=intent.drag_token or self.id_factory("drag"),
                description="Move ROI bounds",
            ),
            "Move ROI bounds",
        )
        self._publish_roi(updated)

    def _move_background(self, intent: CanvasIntent) -> None:
        roi = self._require_roi(intent)
        updated = replace(
            roi,
            left_background_range=intent.left_background or roi.left_background_range,
            right_background_range=intent.right_background
            or roi.right_background_range,
            fit_revision=roi.fit_revision + 1,
        )
        updated.validate("roi")
        self._push_scientific(
            MoveROIBoundsCommand(
                self.workspace_controller,
                before=roi,
                after=updated,
                drag_token=intent.drag_token or self.id_factory("drag"),
                description="Move ROI background",
            ),
            "Move ROI background",
        )
        self._publish_roi(updated)

    def _channel_and_energy(
        self, spectrum_id: str, position: float
    ) -> tuple[float, float]:
        record = self.workspace_controller.document.spectrum_by_id(spectrum_id)
        assert record is not None
        spectrum = record.spectrum
        if spectrum.energies is None or len(spectrum.energies) == 0:
            channel = float(position)
            return channel, channel
        energies = np.asarray(spectrum.energies, dtype=float)
        channels = np.asarray(spectrum.channels, dtype=float)
        index = int(np.argmin(np.abs(energies - float(position))))
        return float(channels[index]), float(position)

    def _add_peak(self, intent: CanvasIntent) -> None:
        spectrum_id = self._require_spectrum_id(intent)
        assert intent.position is not None
        channel, energy = self._channel_and_energy(spectrum_id, intent.position)
        record = self.workspace_controller.document.spectrum_by_id(spectrum_id)
        counts = np.asarray(record.spectrum.counts, dtype=float)
        count_index = min(max(int(round(channel)), 0), max(len(counts) - 1, 0))
        net_counts = float(counts[count_index]) if len(counts) else 0.0
        containing_rois = tuple(
            roi
            for roi in self.workspace_controller.document.rois
            if roi.spectrum_id == spectrum_id
            and roi.signal_range[0] <= energy <= roi.signal_range[1]
        )
        selected_roi = self.workspace_controller.document.roi_by_id(
            str(self.selection_bus.state.roi_id or "")
        )
        target_roi = (
            selected_roi
            if selected_roi in containing_rois
            else (containing_rois[0] if containing_rois else None)
        )
        peak = PeakModel(
            peak_id=self.id_factory("peak"),
            spectrum_id=spectrum_id,
            roi_id=target_roi.roi_id if target_roi is not None else None,
            centroid_channel=channel,
            centroid_energy_keV=energy,
            status="manual",
            manual_overrides={"roi_bounds_keV": [max(energy - 2.0, 0.0), energy + 2.0]},
            provenance={"source": "canvas_manual_peak"},
            net_counts=net_counts,
        )
        commands: list[object] = [
            UpdatePeakCommand(
                self.workspace_controller,
                before=None,
                after=peak,
                description="Add manual peak",
            )
        ]
        if target_roi is not None:
            commands.append(
                UpsertROICommand(
                    self.workspace_controller,
                    before=target_roi,
                    after=replace(
                        target_roi,
                        associated_peak_ids=tuple(
                            dict.fromkeys(
                                (*target_roi.associated_peak_ids, peak.peak_id)
                            )
                        ),
                    ),
                    description="Associate peak with ROI",
                )
            )
        commands.append(self._invalidation_command("Add manual peak"))
        self._push_macro("Add manual peak", tuple(commands))
        self.workspace_controller.select_peak(peak.peak_id)
        self._publish_peak(peak)

    def _select_peak(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        self.workspace_controller.select_peak(peak.peak_id)
        self._publish_peak(peak)

    def _move_peak(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        assert intent.position is not None
        channel, energy = self._channel_and_energy(peak.spectrum_id, intent.position)
        energy_delta = energy - peak.centroid_energy_keV
        overrides = dict(peak.manual_overrides)
        raw_bounds = overrides.get("roi_bounds_keV")
        if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 2:
            overrides["roi_bounds_keV"] = [
                float(raw_bounds[0]) + energy_delta,
                float(raw_bounds[1]) + energy_delta,
            ]
        components = tuple(
            replace(item, centroid=item.centroid + energy_delta)
            for item in peak.components
        )
        updated = self._invalidate_peak_fit(
            replace(
                peak,
                centroid_channel=channel,
                centroid_energy_keV=energy,
                components=components,
                manual_overrides=overrides,
                status="manual",
            ),
            "Move peak centroid",
        )
        self._push_scientific(
            UpdatePeakCommand(
                self.workspace_controller,
                before=peak,
                after=updated,
                invalidate_diagnostics=True,
                description="Move peak centroid",
            ),
            "Move peak centroid",
        )
        self.workspace_controller.select_peak(updated.peak_id)
        self._publish_peak(updated)

    def _delete_peak(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        self._push_scientific(
            DeletePeakCommand(
                self.workspace_controller,
                peak=peak,
                description="Delete peak",
            ),
            "Delete peak",
        )
        self.workspace_controller.select_peak(None)
        self.selection_bus.clear()

    def _update_peak(
        self,
        before: PeakModel,
        after: PeakModel,
        description: str,
        *,
        invalidate: bool = True,
        invalidate_fit: bool = False,
    ) -> None:
        if invalidate_fit:
            after = self._invalidate_peak_fit(after, description)
        command = UpdatePeakCommand(
            self.workspace_controller,
            before=before,
            after=after,
            invalidate_diagnostics=invalidate_fit,
            description=description,
        )
        if invalidate:
            self._push_scientific(command, description)
        else:
            self._push(command)
        self.workspace_controller.select_peak(after.peak_id)
        self._publish_peak(after)

    @staticmethod
    def _invalidate_peak_fit(peak: PeakModel, reason: str) -> PeakModel:
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

    def _add_component(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        position = (
            float(intent.position)
            if intent.position is not None
            else peak.centroid_energy_keV
        )
        component = PeakComponent(
            component_id=self.id_factory("component"),
            shape=peak.shape,
            centroid=position,
            area=max(peak.net_counts, 0.0),
        )
        self._update_peak(
            peak,
            replace(peak, components=(*peak.components, component), status="manual"),
            "Add overlap component",
            invalidate_fit=True,
        )

    def _split_peak(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        components = list(peak.components)
        if components:
            target_index = 0
            if intent.component_ids:
                requested = intent.component_ids[0]
                target_index = next(
                    (
                        index
                        for index, item in enumerate(components)
                        if item.component_id == requested
                    ),
                    0,
                )
            target = components.pop(target_index)
        else:
            target_index = 0
            target = PeakComponent(
                component_id=self.id_factory("component"),
                shape=peak.shape,
                centroid=peak.centroid_energy_keV,
                area=max(peak.net_counts, 0.0),
            )
        delta = max(target.fwhm * 0.25, 0.25)
        children = (
            replace(
                target,
                component_id=self.id_factory("component"),
                centroid=target.centroid - delta,
                area=target.area * 0.5,
                amplitude=target.amplitude * 0.5,
            ),
            replace(
                target,
                component_id=self.id_factory("component"),
                centroid=target.centroid + delta,
                area=target.area * 0.5,
                amplitude=target.amplitude * 0.5,
            ),
        )
        components[target_index:target_index] = children
        self._update_peak(
            peak,
            replace(peak, components=tuple(components), status="manual"),
            "Split overlap component",
            invalidate_fit=True,
        )

    def _merge_components(self, intent: CanvasIntent) -> None:
        document = self.workspace_controller.document
        peak = (
            document.peak_by_id(str(intent.spectrum_id), str(intent.peak_id))
            if intent.spectrum_id and intent.peak_id
            else next(
                (
                    item
                    for item in document.peaks
                    if set(intent.component_ids).issubset(
                        {component.component_id for component in item.components}
                    )
                ),
                None,
            )
        )
        if peak is None:
            raise KeyError("No peak owns all requested overlap components")
        selected = [
            item
            for item in peak.components
            if item.component_id in intent.component_ids
        ]
        if len(selected) < 2:
            raise ValueError("At least two existing components are required")
        total_area = sum(item.area for item in selected)
        centroid = (
            sum(item.centroid * item.area for item in selected) / total_area
            if total_area > 0.0
            else sum(item.centroid for item in selected) / len(selected)
        )
        merged = PeakComponent(
            component_id=self.id_factory("component"),
            shape=selected[0].shape,
            centroid=centroid,
            area=total_area,
            amplitude=max(item.amplitude for item in selected),
            fwhm=max(item.fwhm for item in selected),
            uncertainty=sqrt(sum(item.uncertainty**2 for item in selected)),
            provenance={"merged_from": list(intent.component_ids)},
        )
        selected_ids = set(intent.component_ids)
        first_index = min(
            index
            for index, item in enumerate(peak.components)
            if item.component_id in selected_ids
        )
        remaining = [
            item for item in peak.components if item.component_id not in selected_ids
        ]
        remaining.insert(first_index, merged)
        self._update_peak(
            peak,
            replace(peak, components=tuple(remaining), status="manual"),
            "Merge overlap components",
            invalidate_fit=True,
        )

    def _assign_nuclide(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        assert intent.nuclide is not None
        assignment = NuclideAssignment(
            nuclide=intent.nuclide,
            line_energy_keV=peak.centroid_energy_keV,
            manual=True,
            provenance={"source": "canvas_context_menu"},
        )
        updated = replace(
            peak,
            assignments=(assignment, *peak.assignments[1:]),
            candidate_nuclides=tuple(
                dict.fromkeys((intent.nuclide, *peak.candidate_nuclides))
            ),
            status="manual",
        )
        self._update_peak(peak, updated, "Assign peak nuclide")

    def _clear_nuclide(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        if not peak.assignments and not peak.reference_lines_keV:
            return
        self._update_peak(
            peak,
            replace(peak, assignments=(), reference_lines_keV=(), status="manual"),
            "Clear peak nuclide",
        )

    def _pin_nuclide(self, intent: CanvasIntent) -> None:
        assert intent.nuclide is not None
        before = self.workspace_controller.document.pinned_nuclides
        if intent.nuclide in before:
            return
        after = tuple(dict.fromkeys((*before, intent.nuclide)))
        self._push_macro(
            "Pin nuclide",
            (
                TogglePinnedNuclideCommand(
                    self.workspace_controller,
                    before=before,
                    after=after,
                    description="Pin nuclide",
                ),
                self._cascade_workflow_command(after),
            ),
        )

    def _unpin_nuclide(self, intent: CanvasIntent) -> None:
        assert intent.nuclide is not None
        before = self.workspace_controller.document.pinned_nuclides
        if intent.nuclide not in before:
            return
        after = tuple(item for item in before if item != intent.nuclide)
        self._push_macro(
            "Unpin nuclide",
            (
                TogglePinnedNuclideCommand(
                    self.workspace_controller,
                    before=before,
                    after=after,
                    description="Unpin nuclide",
                ),
                self._cascade_workflow_command(after),
            ),
        )

    def _tag_peak(self, intent: CanvasIntent) -> None:
        peak = self._require_peak(intent)
        assert intent.tag is not None
        if intent.tag in peak.tags:
            return
        self._update_peak(
            peak,
            replace(peak, tags=tuple(dict.fromkeys((*peak.tags, intent.tag)))),
            "Tag peak",
            invalidate=False,
        )

    def _tag_nuclide(self, intent: CanvasIntent) -> None:
        assert intent.nuclide is not None
        assert intent.tag is not None
        before = self.workspace_controller.document.nuclide_tags
        if intent.tag in before.get(intent.nuclide, ()):
            return
        after = {key: tuple(value) for key, value in before.items()}
        after[intent.nuclide] = tuple(
            dict.fromkeys((*after.get(intent.nuclide, ()), intent.tag))
        )
        self._push(
            UpdateNuclideTagsCommand(
                self.workspace_controller,
                before=before,
                after=after,
                description="Tag nuclide",
            )
        )

    def _assign_role(self, intent: CanvasIntent) -> None:
        spectrum_id = self._require_spectrum_id(intent)
        assert intent.role is not None
        existing = next(
            (
                item
                for item in self.workspace_controller.document.spectrum_roles
                if item.role == intent.role
            ),
            None,
        )
        before = existing.spectrum_ids if existing is not None else ()
        if before == (spectrum_id,):
            return
        self._push_scientific(
            AssignSpectrumRoleCommand(
                self.workspace_controller,
                role=intent.role,
                before=before,
                after=(spectrum_id,),
                description=f"Assign {intent.role} spectrum",
            ),
            f"Assign {intent.role} spectrum",
        )

    def _viewport(self, intent: CanvasIntent) -> CanvasViewport:
        viewport_id = intent.viewport_id or "primary-spectrum"
        current = self.workspace_controller.document.viewport_by_id(viewport_id)
        if current is not None and (
            intent.spectrum_id is None
            or current.spectrum_id in {None, intent.spectrum_id}
        ):
            return current
        return CanvasViewport(
            viewport_id=viewport_id,
            spectrum_id=intent.spectrum_id,
        )

    def _commit_viewport(
        self, before: CanvasViewport | None, after: CanvasViewport, description: str
    ) -> None:
        self._push(
            UpdateCanvasViewportCommand(
                self.workspace_controller,
                before=before,
                after=after,
                description=description,
            )
        )

    def _change_viewport(self, intent: CanvasIntent) -> None:
        current = self._viewport(intent)
        before = self.workspace_controller.document.viewport_by_id(current.viewport_id)
        after = replace(
            current,
            x_range=intent.x_range or current.x_range,
            y_range=intent.y_range or current.y_range,
        )
        self._commit_viewport(before, after, "Change spectrum viewport")

    def _toggle_crosshair(self, intent: CanvasIntent) -> None:
        current = self._viewport(intent)
        before = self.workspace_controller.document.viewport_by_id(current.viewport_id)
        self._commit_viewport(
            before,
            replace(current, crosshair_enabled=bool(intent.enabled)),
            "Toggle crosshair",
        )

    def _set_log_scale(self, intent: CanvasIntent) -> None:
        current = self._viewport(intent)
        before = self.workspace_controller.document.viewport_by_id(current.viewport_id)
        self._commit_viewport(
            before,
            replace(current, log_y=bool(intent.enabled)),
            "Change spectrum scale",
        )

    def _set_labels_visible(self, intent: CanvasIntent) -> None:
        current = self._viewport(intent)
        before = self.workspace_controller.document.viewport_by_id(current.viewport_id)
        self._commit_viewport(
            before,
            replace(current, labels_visible=bool(intent.enabled)),
            "Change spectrum labels",
        )


__all__ = ["CanvasIntentDispatcher"]
