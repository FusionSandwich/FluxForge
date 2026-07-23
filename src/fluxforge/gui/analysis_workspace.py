"""Shared state container for the modern analysis GUI workflow."""

from __future__ import annotations

import dataclasses
import re
from copy import copy as shallow_copy, deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fluxforge.core.analysis_workspace import (
    ActivityCalculationResult,
    EfficiencyCalibrationFitResult,
    PeakCandidate,
    ROIComponentFit,
    ROIAnalysisResult,
    ROISpectrumStatistic,
    ROIStatisticsResult,
    SurveyPoint,
)
from fluxforge.core.workspace_document import (
    AnalysisROI,
    CalibrationModel,
    CanvasViewport,
    DetectorProfile,
    FitDiagnostics,
    NuclideAssignment,
    PeakModel,
    SpectrumRoleAssignment,
    WorkspaceDocument,
    WorkspaceSpectrum,
)
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.io.flux_wire import EfficiencyCalibration
from fluxforge.io.spe import GammaSpectrum


@dataclass(frozen=True)
class SpectrumSlot:
    """One spectrum lane shown in the central workspace."""

    key: str
    label: str
    spectrum: GammaSpectrum
    source_key: str | None = None
    source_label: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class LoadedSpectrumRecord:
    """One spectrum loaded into the modern GUI inventory."""

    key: str
    label: str
    spectrum: GammaSpectrum
    source_path: str | None = None


@dataclass(frozen=True)
class AnalysisWorkspaceState:
    """Serializable state for the remaining analysis GUI surfaces."""

    spectra: tuple[SpectrumSlot, ...] = ()
    loaded_spectra: tuple[LoadedSpectrumRecord, ...] = ()
    active_spectrum_key: str = "foreground"
    peaks: tuple[PeakCandidate, ...] = ()
    selected_peak_id: str | None = None
    pinned_nuclides: tuple[str, ...] = ()
    peak_search_method: str = "mariscotti"
    bayesian_source_id: str = "fluxforge_bundled_gamma"
    ml_source_id: str = "fluxforge_bundled_gamma"
    roi_background_method: str = "roi_sideband"
    background_mode: str = "simple"
    background_scale: float = 1.0
    background_visible: bool = True
    efficiency_fit: EfficiencyCalibrationFitResult | None = None
    detector_efficiency: EfficiencyCalibration | None = None
    activity_results: tuple[ActivityCalculationResult, ...] = ()
    roi_analysis: ROIAnalysisResult | None = None
    roi_statistics: ROIStatisticsResult | None = None
    survey_points: tuple[SurveyPoint, ...] = ()
    cascade_sum_lines_keV: tuple[float, ...] = ()


WorkspaceListener = Callable[[AnalysisWorkspaceState], None]
WorkspaceDocumentListener = Callable[[WorkspaceDocument], None]


class AnalysisWorkspaceController:
    """Observable adapter over the canonical, persisted workspace document.

    ``AnalysisWorkspaceState`` remains available while the Qt surfaces migrate, but
    it is a projection.  All persisted state is owned by ``WorkspaceDocument``.
    The adapter replaces only document leaves, so spectrum arrays and unrelated
    analysis results are never copied during ordinary UI mutations.
    """

    def __init__(
        self,
        initial_state: AnalysisWorkspaceState | WorkspaceDocument | None = None,
    ) -> None:
        self._listeners: list[WorkspaceListener] = []
        self._document_listeners: list[WorkspaceDocumentListener] = []
        if isinstance(initial_state, WorkspaceDocument):
            initial_state.validate()
            self._document = initial_state
            self._state = self._state_from_document(initial_state)
        else:
            self._state = initial_state or AnalysisWorkspaceState()
            self._document = self._document_from_state(self._state)

    @property
    def state(self) -> AnalysisWorkspaceState:
        return self._state

    @property
    def document(self) -> WorkspaceDocument:
        """Return the immutable canonical analysis document."""

        return self._document

    def subscribe(self, listener: WorkspaceListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: WorkspaceListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def subscribe_document(self, listener: WorkspaceDocumentListener) -> None:
        if listener not in self._document_listeners:
            self._document_listeners.append(listener)

    def unsubscribe_document(self, listener: WorkspaceDocumentListener) -> None:
        if listener in self._document_listeners:
            self._document_listeners.remove(listener)

    def set_document(self, document: WorkspaceDocument) -> WorkspaceDocument:
        """Replace the canonical document and rebuild the legacy projection."""

        if not isinstance(document, WorkspaceDocument):
            raise TypeError("document must be a WorkspaceDocument")
        document.validate()
        self._document = document
        self._state = self._state_from_document(document)
        self._notify()
        return document

    def set_state(self, state: AnalysisWorkspaceState) -> AnalysisWorkspaceState:
        """Accept a legacy projection and synchronize its persisted leaves."""

        if not isinstance(state, AnalysisWorkspaceState):
            raise TypeError("state must be an AnalysisWorkspaceState")
        previous_state = self._state
        self._state = state
        self._document = self._document_from_state(
            state,
            base=self._document,
            previous_state=previous_state,
        )
        self._notify()
        return state

    def update(self, **changes) -> AnalysisWorkspaceState:
        return self.set_state(replace(self._state, **changes))

    def _notify(self) -> None:
        for listener in tuple(self._document_listeners):
            listener(self._document)
        for listener in tuple(self._listeners):
            listener(self._state)

    def spectrum(self, key: str | None = None) -> GammaSpectrum | None:
        target = key or self._state.active_spectrum_key
        for slot in self._state.spectra:
            if slot.key == target:
                return slot.spectrum
        return None

    def spectrum_slots(self) -> tuple[SpectrumSlot, ...]:
        return self._state.spectra

    def loaded_spectrum_records(self) -> tuple[LoadedSpectrumRecord, ...]:
        return self._state.loaded_spectra

    def slot(self, key: str) -> SpectrumSlot | None:
        for slot in self._state.spectra:
            if slot.key == key:
                return slot
        return None

    def loaded_spectrum_record(self, key: str) -> LoadedSpectrumRecord | None:
        for record in self._state.loaded_spectra:
            if record.key == key:
                return record
        return None

    def register_loaded_spectrum(
        self,
        spectrum: GammaSpectrum,
        *,
        label: str | None = None,
        source_path: str | None = None,
        key: str | None = None,
    ) -> str:
        resolved_label = str(
            label
            or spectrum.spectrum_id
            or (source_path.rsplit("/", 1)[-1] if source_path else "Spectrum")
        )
        resolved_key = key or self._unique_loaded_key(resolved_label)
        records = [
            record
            for record in self._state.loaded_spectra
            if record.key != resolved_key
        ]
        records.append(
            LoadedSpectrumRecord(
                key=resolved_key,
                label=resolved_label,
                spectrum=spectrum,
                source_path=source_path,
            )
        )
        self.update(loaded_spectra=tuple(records))
        return resolved_key

    def replace_spectrum_slot(
        self,
        key: str,
        spectrum: GammaSpectrum,
        *,
        source_key: str | None = None,
        source_label: str | None = None,
        source_path: str | None = None,
    ) -> AnalysisWorkspaceState:
        slots = []
        matched = False
        for slot in self._state.spectra:
            if slot.key == key:
                slots.append(
                    SpectrumSlot(
                        key=slot.key,
                        label=slot.label,
                        spectrum=spectrum,
                        source_key=(
                            source_key if source_key is not None else slot.source_key
                        ),
                        source_label=(
                            source_label
                            if source_label is not None
                            else slot.source_label
                        ),
                        source_path=(
                            source_path if source_path is not None else slot.source_path
                        ),
                    )
                )
                matched = True
            else:
                slots.append(slot)
        if not matched:
            slots.append(
                SpectrumSlot(
                    key=key,
                    label=(
                        "Secondary Overlay"
                        if key == "overlay"
                        else key.replace("_", " ").title()
                    ),
                    spectrum=spectrum,
                    source_key=source_key,
                    source_label=source_label,
                    source_path=source_path,
                )
            )
        return self.update(spectra=tuple(slots))

    def assign_loaded_spectrum_to_slot(
        self,
        loaded_key: str,
        slot_key: str,
    ) -> AnalysisWorkspaceState:
        record = self.loaded_spectrum_record(loaded_key)
        if record is None:
            raise KeyError(f"Unknown loaded spectrum key: {loaded_key}")
        return self.replace_spectrum_slot(
            slot_key,
            record.spectrum,
            source_key=record.key,
            source_label=record.label,
            source_path=record.source_path,
        )

    def select_spectrum(self, key: str) -> AnalysisWorkspaceState:
        role = next(
            (item for item in self._document.spectrum_roles if item.role == key),
            None,
        )
        spectrum_id = (
            role.spectrum_ids[0] if role is not None and role.spectrum_ids else key
        )
        if self._document.spectrum_by_id(spectrum_id) is None:
            raise KeyError(f"Unknown spectrum role or ID: {key!r}")
        self._replace_document(active_spectrum_id=spectrum_id)
        return self._state

    def replace_peaks(self, peaks: Sequence[PeakCandidate]) -> AnalysisWorkspaceState:
        selected = self._state.selected_peak_id
        if selected and all(peak.peak_id != selected for peak in peaks):
            selected = peaks[0].peak_id if peaks else None
        return self.update(peaks=tuple(peaks), selected_peak_id=selected)

    def select_peak(self, peak_id: str | None) -> AnalysisWorkspaceState:
        return self.update(selected_peak_id=peak_id)

    def selected_peak(self) -> PeakCandidate | None:
        if self._state.selected_peak_id is None:
            return None
        for peak in self._state.peaks:
            if peak.peak_id == self._state.selected_peak_id:
                return peak
        return None

    def replace_peak(self, updated_peak: PeakCandidate) -> AnalysisWorkspaceState:
        peaks = list(self._state.peaks)
        for index, peak in enumerate(peaks):
            if peak.peak_id == updated_peak.peak_id:
                peaks[index] = updated_peak
                break
        else:
            peaks.append(updated_peak)
        return self.update(peaks=tuple(peaks))

    # Canonical leaf-edit API used by focused undo commands.  These methods
    # intentionally retain every document field outside the named leaf.

    def replace_peak_models(
        self,
        peaks: Sequence[PeakModel],
        *,
        spectrum_id: str | None = None,
    ) -> WorkspaceDocument:
        target = spectrum_id or self._document.active_spectrum_id
        replacements = tuple(peaks)
        if target is None and replacements:
            target = replacements[0].spectrum_id
        if target is None:
            merged = self._document.peaks
        else:
            if any(item.spectrum_id != target for item in replacements):
                raise ValueError(
                    "all replacement peaks must target the selected spectrum"
                )
            pending = list(replacements)
            merged_items: list[PeakModel] = []
            last_target_position: int | None = None
            for item in self._document.peaks:
                if item.spectrum_id != target:
                    merged_items.append(item)
                    continue
                if pending:
                    merged_items.append(pending.pop(0))
                    last_target_position = len(merged_items)
            if pending:
                insert_at = (
                    last_target_position
                    if last_target_position is not None
                    else len(merged_items)
                )
                merged_items[insert_at:insert_at] = pending
            merged = tuple(merged_items)
        valid_peak_ids = {item.peak_id for item in replacements}
        projected_rois = tuple(
            replace(
                roi,
                associated_peak_ids=tuple(
                    peak_id
                    for peak_id in roi.associated_peak_ids
                    if roi.spectrum_id != target or peak_id in valid_peak_ids
                ),
            )
            for roi in self._document.rois
        )
        rois = (
            self._document.rois
            if projected_rois == self._document.rois
            else projected_rois
        )
        diagnostics = tuple(
            (
                _invalid_fit_diagnostic(
                    item,
                    reason="peak set replaced",
                    peak_id=None,
                )
                if item.spectrum_id == target
                and item.peak_id is not None
                and item.peak_id not in valid_peak_ids
                else item
            )
            for item in self._document.fit_diagnostics
        )
        return self._replace_document(
            peaks=merged,
            rois=rois,
            fit_diagnostics=diagnostics,
        )

    def replace_all_peak_models(self, peaks: Sequence[PeakModel]) -> WorkspaceDocument:
        """Replace the ordered peak-leaf collection without retaining spectra."""

        return self._replace_document(peaks=tuple(peaks))

    def upsert_peak_model(
        self, peak: PeakModel, *, index: int | None = None
    ) -> WorkspaceDocument:
        items = list(self._document.peaks)
        key = (peak.spectrum_id, peak.peak_id)
        for current_index, current in enumerate(items):
            if (current.spectrum_id, current.peak_id) == key:
                items[current_index] = peak
                break
        else:
            if index is None:
                items.append(peak)
            else:
                items.insert(max(0, min(index, len(items))), peak)
        return self._replace_document(peaks=tuple(items))

    def delete_peak_model(self, spectrum_id: str, peak_id: str) -> WorkspaceDocument:
        peaks = tuple(
            item
            for item in self._document.peaks
            if (item.spectrum_id, item.peak_id) != (spectrum_id, peak_id)
        )
        rois = tuple(
            (
                replace(
                    item,
                    associated_peak_ids=tuple(
                        value for value in item.associated_peak_ids if value != peak_id
                    ),
                )
                if item.spectrum_id == spectrum_id
                else item
            )
            for item in self._document.rois
        )
        diagnostics = tuple(
            (
                _invalid_fit_diagnostic(
                    item,
                    reason="peak deleted",
                    peak_id=None,
                )
                if item.spectrum_id == spectrum_id and item.peak_id == peak_id
                else item
            )
            for item in self._document.fit_diagnostics
        )
        return self._replace_document(
            peaks=peaks,
            rois=rois,
            fit_diagnostics=diagnostics,
        )

    def upsert_roi(
        self, roi: AnalysisROI, *, index: int | None = None
    ) -> WorkspaceDocument:
        items = list(self._document.rois)
        for current_index, current in enumerate(items):
            if current.roi_id == roi.roi_id:
                items[current_index] = roi
                break
        else:
            if index is None:
                items.append(roi)
            else:
                items.insert(max(0, min(index, len(items))), roi)
        return self._replace_document(rois=tuple(items))

    def delete_roi(self, roi_id: str) -> WorkspaceDocument:
        rois = tuple(item for item in self._document.rois if item.roi_id != roi_id)
        peaks = tuple(
            (
                _invalid_peak_model_fit(item, reason="ROI deleted", roi_id=None)
                if item.roi_id == roi_id
                else item
            )
            for item in self._document.peaks
        )
        diagnostics = tuple(
            (
                _invalid_fit_diagnostic(
                    item,
                    reason="ROI deleted",
                    roi_id=None,
                )
                if item.roi_id == roi_id
                else item
            )
            for item in self._document.fit_diagnostics
        )
        viewports = tuple(
            (
                replace(item, selected_roi_id=None)
                if item.selected_roi_id == roi_id
                else item
            )
            for item in self._document.viewports
        )
        return self._replace_document(
            rois=rois,
            peaks=peaks,
            fit_diagnostics=diagnostics,
            viewports=viewports,
        )

    def upsert_fit_diagnostic(self, diagnostic: FitDiagnostics) -> WorkspaceDocument:
        items = list(self._document.fit_diagnostics)
        for index, current in enumerate(items):
            if current.diagnostic_id == diagnostic.diagnostic_id:
                items[index] = diagnostic
                break
        else:
            items.append(diagnostic)
        return self._replace_document(fit_diagnostics=tuple(items))

    def set_fit_diagnostics(
        self, diagnostics: Sequence[FitDiagnostics]
    ) -> WorkspaceDocument:
        """Replace canonical fit diagnostics without retaining spectrum arrays."""

        return self._replace_document(fit_diagnostics=tuple(diagnostics))

    def set_workflow_state(
        self, workflow_state: Mapping[str, Any]
    ) -> WorkspaceDocument:
        """Replace the persisted workflow leaf without touching spectrum arrays."""

        return self._replace_document(workflow_state=_json_safe(workflow_state))

    def assign_spectrum_role(
        self,
        role: str,
        spectrum_ids: str | Sequence[str] | None,
    ) -> WorkspaceDocument:
        if isinstance(spectrum_ids, str):
            resolved_ids = (spectrum_ids,)
        else:
            resolved_ids = tuple(spectrum_ids or ())
        roles = [item for item in self._document.spectrum_roles if item.role != role]
        if resolved_ids:
            roles.append(SpectrumRoleAssignment(role=role, spectrum_ids=resolved_ids))
        return self._replace_document(spectrum_roles=tuple(roles))

    def upsert_detector_profile(self, profile: DetectorProfile) -> WorkspaceDocument:
        items = list(self._document.detector_profiles)
        for index, current in enumerate(items):
            if current.detector_profile_id == profile.detector_profile_id:
                items[index] = profile
                break
        else:
            items.append(profile)
        return self._replace_document(detector_profiles=tuple(items))

    def delete_detector_profile(self, profile_id: str) -> WorkspaceDocument:
        profiles = tuple(
            item
            for item in self._document.detector_profiles
            if item.detector_profile_id != profile_id
        )
        spectra = tuple(
            (
                replace(item, detector_profile_id=None)
                if item.detector_profile_id == profile_id
                else item
            )
            for item in self._document.spectra
        )
        return self._replace_document(detector_profiles=profiles, spectra=spectra)

    def apply_calibration(
        self,
        spectrum_id: str,
        calibration_state: CalibrationModel | DetectorProfile | None,
    ) -> WorkspaceDocument:
        """Apply one calibration/profile leaf to a copied spectrum container.

        Counts, uncertainties, and channel arrays remain shared read-only inputs;
        only the mutable ``GammaSpectrum`` container, calibration mapping, and
        derived energy array are replaced.  This prevents a calibration preview
        or undo operation from mutating the spectrum held by an earlier state.
        """

        workspace_spectrum = self._document.spectrum_by_id(spectrum_id)
        if workspace_spectrum is None:
            raise KeyError(f"Unknown spectrum ID: {spectrum_id}")
        if calibration_state is not None and not isinstance(
            calibration_state, (CalibrationModel, DetectorProfile)
        ):
            raise TypeError(
                "calibration_state must be CalibrationModel, DetectorProfile, or None"
            )

        profiles = list(self._document.detector_profiles)
        profile_id = workspace_spectrum.detector_profile_id
        if isinstance(calibration_state, DetectorProfile):
            profile = calibration_state
            profile_id = profile.detector_profile_id
            calibration = profile.energy_calibration
        else:
            profile = (
                self._document.detector_profile_by_id(profile_id)
                if profile_id is not None
                else None
            )
            if profile is None and calibration_state is not None:
                profile_id = f"{spectrum_id}-detector"
                profile = DetectorProfile(
                    detector_profile_id=profile_id,
                    detector_id=str(workspace_spectrum.spectrum.detector_id or ""),
                    energy_calibration=calibration_state,
                )
            elif profile is not None:
                profile = replace(profile, energy_calibration=calibration_state)
            calibration = calibration_state

        if profile is not None:
            for index, current in enumerate(profiles):
                if current.detector_profile_id == profile.detector_profile_id:
                    profiles[index] = profile
                    break
            else:
                profiles.append(profile)

        current_spectrum = workspace_spectrum.spectrum
        calibrated_spectrum = shallow_copy(current_spectrum)
        calibration_payload = dict(current_spectrum.calibration or {})
        if calibration is None:
            calibration_payload.pop("energy", None)
            calibration_payload.pop("deviation_pairs", None)
            calibrated_spectrum.energies = None
        else:
            calibration_payload["energy"] = [
                float(value) for value in calibration.coefficients
            ]
            calibration_payload["deviation_pairs"] = [
                {
                    "energy_keV": float(energy),
                    "correction_keV": float(correction),
                }
                for energy, correction in calibration.deviation_pairs
            ]
        calibrated_spectrum.calibration = calibration_payload
        if calibration is not None:
            calibrated_spectrum.energies = calibrated_spectrum.calibrate_channels()

        spectra = tuple(
            (
                replace(
                    item,
                    spectrum=calibrated_spectrum,
                    detector_profile_id=profile_id,
                )
                if item.spectrum_id == spectrum_id
                else item
            )
            for item in self._document.spectra
        )
        return self._replace_document(
            spectra=spectra,
            detector_profiles=tuple(profiles),
        )

    def restore_calibration_state(
        self, spectrum_id: str, state: Any
    ) -> WorkspaceDocument:
        """Restore the exact persisted leaves captured before calibration."""

        from fluxforge.gui.workspace_undo import CalibrationUndoState

        if not isinstance(state, CalibrationUndoState):
            raise TypeError("state must be a CalibrationUndoState")
        workspace_spectrum = self._document.spectrum_by_id(spectrum_id)
        if workspace_spectrum is None:
            raise KeyError(f"Unknown spectrum ID: {spectrum_id}")

        restored_spectrum = shallow_copy(workspace_spectrum.spectrum)
        restored_spectrum.calibration = deepcopy(state.calibration_payload)
        restored_spectrum.energies = (
            np.asarray(state.energies, dtype=float)
            if state.energies is not None
            else None
        )
        spectra = tuple(
            (
                replace(
                    item,
                    spectrum=restored_spectrum,
                    detector_profile_id=state.detector_profile_id,
                )
                if item.spectrum_id == spectrum_id
                else item
            )
            for item in self._document.spectra
        )
        return self._replace_document(
            spectra=spectra,
            detector_profiles=state.detector_profiles,
        )

    def upsert_viewport(self, viewport: CanvasViewport) -> WorkspaceDocument:
        items = list(self._document.viewports)
        for index, current in enumerate(items):
            if current.viewport_id == viewport.viewport_id:
                items[index] = viewport
                break
        else:
            items.append(viewport)
        return self._replace_document(viewports=tuple(items))

    def delete_viewport(self, viewport_id: str) -> WorkspaceDocument:
        return self._replace_document(
            viewports=tuple(
                item
                for item in self._document.viewports
                if item.viewport_id != viewport_id
            )
        )

    def set_pinned_nuclides(
        self, pinned_nuclides: Sequence[str]
    ) -> AnalysisWorkspaceState:
        deduped: list[str] = []
        for nuclide in pinned_nuclides:
            if nuclide and nuclide not in deduped:
                deduped.append(nuclide)
        return self.update(pinned_nuclides=tuple(deduped))

    def set_nuclide_tags(
        self, nuclide_tags: Mapping[str, Sequence[str]]
    ) -> WorkspaceDocument:
        """Replace persisted analyst tags keyed by nuclide identifier."""

        normalized: dict[str, tuple[str, ...]] = {}
        for raw_nuclide, raw_tags in nuclide_tags.items():
            nuclide = str(raw_nuclide).strip()
            if not nuclide:
                raise ValueError("nuclide tag keys must be non-empty")
            tags = tuple(
                dict.fromkeys(str(tag).strip() for tag in raw_tags if str(tag).strip())
            )
            if tags:
                normalized[nuclide] = tags
        return self._replace_document(nuclide_tags=normalized)

    def toggle_pinned_nuclide(self, nuclide: str) -> AnalysisWorkspaceState:
        current = list(self._state.pinned_nuclides)
        if nuclide in current:
            current.remove(nuclide)
        elif nuclide:
            current.append(nuclide)
        return self.set_pinned_nuclides(current)

    def set_background_config(
        self,
        *,
        mode: str | None = None,
        scale: float | None = None,
        visible: bool | None = None,
    ) -> AnalysisWorkspaceState:
        payload = {
            "background_mode": (
                mode if mode is not None else self._state.background_mode
            ),
            "background_scale": (
                float(scale) if scale is not None else self._state.background_scale
            ),
            "background_visible": (
                bool(visible) if visible is not None else self._state.background_visible
            ),
        }
        return self.update(**payload)

    def set_peak_search_method(self, method: str) -> AnalysisWorkspaceState:
        return self.update(peak_search_method=str(method))

    def set_bayesian_source_id(self, source_id: str) -> AnalysisWorkspaceState:
        return self.update(bayesian_source_id=str(source_id))

    def set_ml_source_id(self, source_id: str) -> AnalysisWorkspaceState:
        return self.update(ml_source_id=str(source_id))

    def set_roi_background_method(self, method: str) -> AnalysisWorkspaceState:
        return self.update(roi_background_method=str(method))

    def set_efficiency_fit(
        self,
        fit: EfficiencyCalibrationFitResult | None,
    ) -> AnalysisWorkspaceState:
        return self.update(efficiency_fit=fit)

    def set_detector_efficiency(
        self,
        calibration: EfficiencyCalibration | None,
    ) -> AnalysisWorkspaceState:
        return self.update(
            detector_efficiency=(
                dataclasses.replace(calibration) if calibration is not None else None
            )
        )

    def set_activity_results(
        self,
        results: Sequence[ActivityCalculationResult],
    ) -> AnalysisWorkspaceState:
        self.update(activity_results=tuple(results))
        self._clear_analysis_invalidation()
        return self.state

    def set_roi_analysis(
        self,
        result: ROIAnalysisResult | None,
    ) -> AnalysisWorkspaceState:
        self.update(roi_analysis=result)
        self._clear_analysis_invalidation()
        return self.state

    def set_roi_statistics(
        self,
        result: ROIStatisticsResult | None,
    ) -> AnalysisWorkspaceState:
        self.update(roi_statistics=result)
        self._clear_analysis_invalidation()
        return self.state

    def _clear_analysis_invalidation(self) -> None:
        workflow = dict(self._document.workflow_state)
        if "analysis_invalidation" not in workflow:
            return
        workflow.pop("analysis_invalidation", None)
        self.set_workflow_state(workflow)

    def set_survey_points(
        self, survey_points: Sequence[SurveyPoint]
    ) -> AnalysisWorkspaceState:
        return self.update(survey_points=tuple(survey_points))

    def set_cascade_sum_lines(
        self, energies_keV: Sequence[float]
    ) -> AnalysisWorkspaceState:
        return self.update(
            cascade_sum_lines_keV=tuple(float(value) for value in energies_keV)
        )

    def describe(self) -> dict[str, object]:
        return {
            "active_spectrum_key": self._state.active_spectrum_key,
            "spectrum_keys": [slot.key for slot in self._state.spectra],
            "slot_sources": {
                slot.key: slot.source_label or slot.label
                for slot in self._state.spectra
            },
            "loaded_spectrum_count": len(self._state.loaded_spectra),
            "peak_count": len(self._state.peaks),
            "selected_peak_id": self._state.selected_peak_id,
            "pinned_nuclides": list(self._state.pinned_nuclides),
            "peak_search_method": self._state.peak_search_method,
            "bayesian_source_id": self._state.bayesian_source_id,
            "ml_source_id": self._state.ml_source_id,
            "roi_background_method": self._state.roi_background_method,
            "background_mode": self._state.background_mode,
            "background_visible": self._state.background_visible,
            "detector_id": (
                self._state.detector_efficiency.detector_id
                if self._state.detector_efficiency is not None
                else None
            ),
            "activity_result_count": len(self._state.activity_results),
            "has_roi_analysis": self._state.roi_analysis is not None,
            "has_roi_statistics": self._state.roi_statistics is not None,
            "survey_point_count": len(self._state.survey_points),
            "cascade_sum_line_count": len(self._state.cascade_sum_lines_keV),
        }

    def _replace_document(self, **changes: Any) -> WorkspaceDocument:
        changes.setdefault("updated_at", _workspace_timestamp())
        document = replace(self._document, **changes)
        return self.set_document(document)

    def _document_from_state(
        self,
        state: AnalysisWorkspaceState,
        *,
        base: WorkspaceDocument | None = None,
        previous_state: AnalysisWorkspaceState | None = None,
    ) -> WorkspaceDocument:
        base = base or WorkspaceDocument()
        spectra_by_id = {item.spectrum_id: item for item in base.spectra}

        # Loaded records define stable spectrum identities.  A slot that has not
        # yet been registered is added under its source key (or role key).
        ordered_ids: list[str] = []
        for record in state.loaded_spectra:
            prior = spectra_by_id.get(record.key)
            spectra_by_id[record.key] = WorkspaceSpectrum(
                spectrum_id=record.key,
                spectrum=record.spectrum,
                label=record.label,
                source_path=record.source_path,
                source_hash=prior.source_hash if prior else None,
                detector_profile_id=prior.detector_profile_id if prior else None,
                provenance=prior.provenance if prior else {},
                extensions=prior.extensions if prior else {},
            )
            ordered_ids.append(record.key)

        roles: list[SpectrumRoleAssignment] = []
        base_roles_by_name = {item.role: item for item in base.spectrum_roles}
        projected_role_names: set[str] = set()
        slot_spectrum_ids: dict[str, str] = {}
        for slot in state.spectra:
            spectrum_id = slot.source_key or slot.key
            # Avoid making two different slot spectra share an identity.
            if (
                spectrum_id in spectra_by_id
                and spectra_by_id[spectrum_id].spectrum is not slot.spectrum
                and slot.source_key is None
            ):
                spectrum_id = self._available_spectrum_id(
                    spectrum_id, set(spectra_by_id)
                )
            prior = spectra_by_id.get(spectrum_id)
            spectra_by_id[spectrum_id] = WorkspaceSpectrum(
                spectrum_id=spectrum_id,
                spectrum=slot.spectrum,
                label=slot.source_label or slot.label,
                source_path=slot.source_path,
                source_hash=prior.source_hash if prior else None,
                detector_profile_id=prior.detector_profile_id if prior else None,
                provenance=prior.provenance if prior else {},
                extensions={
                    **dict(prior.extensions if prior else {}),
                    "slot_label": slot.label,
                },
            )
            if spectrum_id not in ordered_ids:
                ordered_ids.append(spectrum_id)
            prior_role = base_roles_by_name.get(slot.key)
            retained_secondary_ids = (
                tuple(
                    item for item in prior_role.spectrum_ids[1:] if item != spectrum_id
                )
                if prior_role is not None
                else ()
            )
            roles.append(
                SpectrumRoleAssignment(
                    role=slot.key,
                    spectrum_ids=(spectrum_id,) + retained_secondary_ids,
                )
            )
            projected_role_names.add(slot.key)
            slot_spectrum_ids[slot.key] = spectrum_id

        # The legacy UI projects one primary spectrum per role. Preserve any
        # additional role members, and roles with no legacy slot, whenever a
        # spectrum-inventory edit is synchronized back into the v2 document.
        roles.extend(
            item
            for item in base.spectrum_roles
            if item.role not in projected_role_names
        )

        # Retain document-only spectra (for example secondary spectra not yet
        # represented by a legacy slot) after the projected inventory.
        for spectrum_id in tuple(spectra_by_id):
            if spectrum_id not in ordered_ids:
                ordered_ids.append(spectrum_id)
        spectra = tuple(spectra_by_id[item] for item in ordered_ids)

        active_spectrum_id = slot_spectrum_ids.get(state.active_spectrum_key)
        if active_spectrum_id is None and state.active_spectrum_key in spectra_by_id:
            active_spectrum_id = state.active_spectrum_key
        if active_spectrum_id is None and spectra:
            active_spectrum_id = spectra[0].spectrum_id

        existing_peaks = {(item.spectrum_id, item.peak_id): item for item in base.peaks}
        active_legacy_peaks = tuple(
            self._peak_model_from_candidate(
                item,
                active_spectrum_id or state.active_spectrum_key,
                existing_peaks.get(
                    (active_spectrum_id or state.active_spectrum_key, item.peak_id)
                ),
            )
            for item in state.peaks
        )
        if active_spectrum_id is None:
            peaks = base.peaks if not active_legacy_peaks else active_legacy_peaks
        else:
            peaks = (
                tuple(
                    item
                    for item in base.peaks
                    if item.spectrum_id != active_spectrum_id
                )
                + active_legacy_peaks
            )

        active_peak_ids = {item.peak_id for item in active_legacy_peaks}
        projected_rois = tuple(
            replace(
                roi,
                associated_peak_ids=tuple(
                    item
                    for item in roi.associated_peak_ids
                    if roi.spectrum_id != active_spectrum_id or item in active_peak_ids
                ),
            )
            for roi in base.rois
        )
        rois = base.rois if projected_rois == base.rois else projected_rois

        spectrum_inventory_changed = previous_state is None or not (
            _spectrum_slots_equivalent(state.spectra, previous_state.spectra)
            and _loaded_records_equivalent(
                state.loaded_spectra, previous_state.loaded_spectra
            )
        )
        if not spectrum_inventory_changed:
            spectra = base.spectra
            roles = list(base.spectrum_roles)
        if (
            previous_state is not None
            and state.active_spectrum_key == previous_state.active_spectrum_key
        ):
            active_spectrum_id = base.active_spectrum_id
        if previous_state is not None and state.peaks == previous_state.peaks:
            peaks = base.peaks
            rois = base.rois

        workflow = dict(base.workflow_state)
        workflow["analysis_workspace_v1"] = self._legacy_workflow_payload(state)
        document = replace(
            base,
            updated_at=_workspace_timestamp(),
            spectra=spectra,
            spectrum_roles=tuple(roles),
            active_spectrum_id=active_spectrum_id,
            rois=rois,
            peaks=peaks,
            pinned_nuclides=(
                base.pinned_nuclides
                if previous_state is not None
                and state.pinned_nuclides == previous_state.pinned_nuclides
                else state.pinned_nuclides
            ),
            workflow_state=workflow,
        )
        return document

    def _state_from_document(
        self, document: WorkspaceDocument
    ) -> AnalysisWorkspaceState:
        by_id = {item.spectrum_id: item for item in document.spectra}
        loaded = tuple(
            LoadedSpectrumRecord(
                key=item.spectrum_id,
                label=item.label or item.spectrum_id,
                spectrum=item.spectrum,
                source_path=item.source_path,
            )
            for item in document.spectra
        )
        slots: list[SpectrumSlot] = []
        for role in document.spectrum_roles:
            if not role.spectrum_ids:
                continue
            workspace_spectrum = by_id.get(role.spectrum_ids[0])
            if workspace_spectrum is None:
                continue
            slot_label = str(
                workspace_spectrum.extensions.get("slot_label")
                or role.role.replace("_", " ").title()
            )
            slots.append(
                SpectrumSlot(
                    key=role.role,
                    label=slot_label,
                    spectrum=workspace_spectrum.spectrum,
                    source_key=workspace_spectrum.spectrum_id,
                    source_label=workspace_spectrum.label
                    or workspace_spectrum.spectrum_id,
                    source_path=workspace_spectrum.source_path,
                )
            )

        active_key = document.active_spectrum_id or "foreground"
        for role in document.spectrum_roles:
            if document.active_spectrum_id in role.spectrum_ids:
                active_key = role.role
                break
        peaks = tuple(
            self._candidate_from_peak_model(item, document=document)
            for item in document.peaks
            if item.spectrum_id == document.active_spectrum_id
        )
        legacy_payload = document.workflow_state.get("analysis_workspace_v1", {})
        legacy = self._legacy_state_values(legacy_payload)
        selected = legacy.pop("selected_peak_id", None)
        if selected and all(item.peak_id != selected for item in peaks):
            selected = peaks[0].peak_id if peaks else None
        return AnalysisWorkspaceState(
            spectra=tuple(slots),
            loaded_spectra=loaded,
            active_spectrum_key=active_key,
            peaks=peaks,
            selected_peak_id=selected,
            pinned_nuclides=document.pinned_nuclides,
            **legacy,
        )

    @staticmethod
    def _peak_model_from_candidate(
        candidate: PeakCandidate,
        spectrum_id: str,
        prior: PeakModel | None = None,
    ) -> PeakModel:
        assignments: tuple[NuclideAssignment, ...] = ()
        if candidate.nuclide:
            if (
                prior is not None
                and prior.assignments
                and prior.assignments[0].nuclide == candidate.nuclide
            ):
                assignments = prior.assignments
            else:
                line_energy = (
                    candidate.reference_lines_keV[0]
                    if candidate.reference_lines_keV
                    else candidate.energy_keV
                )
                assignments = (
                    NuclideAssignment(
                        nuclide=candidate.nuclide,
                        line_energy_keV=float(line_energy),
                        manual=True,
                        provenance={"source": "legacy_peak_editor"},
                    ),
                )
        overrides = dict(prior.manual_overrides if prior else {})
        overrides["roi_bounds_keV"] = [
            float(candidate.roi_bounds_keV[0]),
            float(candidate.roi_bounds_keV[1]),
        ]
        return PeakModel(
            peak_id=candidate.peak_id,
            spectrum_id=spectrum_id,
            roi_id=prior.roi_id if prior else None,
            centroid_channel=float(candidate.channel),
            centroid_energy_keV=float(candidate.energy_keV),
            shape=prior.shape if prior else "gaussian",
            components=prior.components if prior else (),
            assignments=assignments,
            tags=candidate.tags,
            status=candidate.status,
            manual_overrides=overrides,
            provenance=(
                prior.provenance if prior else {"source": "legacy_peak_candidate"}
            ),
            net_counts=float(candidate.net_counts),
            significance=float(candidate.significance),
            fit_quality=float(candidate.fit_quality),
            candidate_nuclides=candidate.candidate_nuclides,
            reference_lines_keV=candidate.reference_lines_keV,
            normalized_residuals=candidate.normalized_residuals,
            residual_channels=candidate.residual_channels,
        )

    def _candidate_from_peak_model(
        self,
        peak: PeakModel,
        *,
        document: WorkspaceDocument | None = None,
    ) -> PeakCandidate:
        source = document or self._document
        roi = source.roi_by_id(peak.roi_id or "")
        raw_bounds = peak.manual_overrides.get("roi_bounds_keV")
        if roi is not None and roi.spectrum_id == peak.spectrum_id:
            roi_bounds = roi.signal_range
        elif isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 2:
            roi_bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
        else:
            roi_bounds = (
                max(float(peak.centroid_energy_keV) - 2.0, 0.0),
                float(peak.centroid_energy_keV) + 2.0,
            )
        nuclide = peak.assignments[0].nuclide if peak.assignments else None
        return PeakCandidate(
            peak_id=peak.peak_id,
            channel=peak.centroid_channel,
            energy_keV=peak.centroid_energy_keV,
            significance=peak.significance,
            roi_bounds_keV=roi_bounds,
            net_counts=peak.net_counts,
            fit_quality=peak.fit_quality,
            status=peak.status,
            nuclide=nuclide,
            candidate_nuclides=peak.candidate_nuclides,
            reference_lines_keV=peak.reference_lines_keV,
            tags=peak.tags,
            normalized_residuals=peak.normalized_residuals,
            residual_channels=peak.residual_channels,
        )

    @staticmethod
    def _legacy_workflow_payload(state: AnalysisWorkspaceState) -> dict[str, Any]:
        return {
            "selected_peak_id": state.selected_peak_id,
            "peak_search_method": state.peak_search_method,
            "bayesian_source_id": state.bayesian_source_id,
            "ml_source_id": state.ml_source_id,
            "roi_background_method": state.roi_background_method,
            "background_mode": state.background_mode,
            "background_scale": state.background_scale,
            "background_visible": state.background_visible,
            "efficiency_fit": _json_safe(state.efficiency_fit),
            "detector_efficiency": _json_safe(state.detector_efficiency),
            "activity_results": _json_safe(state.activity_results),
            "roi_analysis": _json_safe(state.roi_analysis),
            "roi_statistics": _json_safe(state.roi_statistics),
            "survey_points": _json_safe(state.survey_points),
            "cascade_sum_lines_keV": list(state.cascade_sum_lines_keV),
        }

    @staticmethod
    def _legacy_state_values(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            payload = {}
        return {
            "peak_search_method": str(payload.get("peak_search_method", "mariscotti")),
            "bayesian_source_id": str(
                payload.get("bayesian_source_id", "fluxforge_bundled_gamma")
            ),
            "ml_source_id": str(payload.get("ml_source_id", "fluxforge_bundled_gamma")),
            "roi_background_method": str(
                payload.get("roi_background_method", "roi_sideband")
            ),
            "background_mode": str(payload.get("background_mode", "simple")),
            "background_scale": float(payload.get("background_scale", 1.0)),
            "background_visible": bool(payload.get("background_visible", True)),
            "efficiency_fit": _decode_efficiency_fit(payload.get("efficiency_fit")),
            "detector_efficiency": _decode_dataclass(
                EfficiencyCalibration, payload.get("detector_efficiency")
            ),
            "activity_results": tuple(
                item
                for item in (
                    _decode_dataclass(ActivityCalculationResult, raw)
                    for raw in _payload_sequence(payload.get("activity_results"))
                )
                if item is not None
            ),
            "roi_analysis": _decode_roi_analysis(payload.get("roi_analysis")),
            "roi_statistics": _decode_roi_statistics(payload.get("roi_statistics")),
            "survey_points": tuple(
                item
                for item in (
                    _decode_dataclass(SurveyPoint, raw)
                    for raw in _payload_sequence(payload.get("survey_points"))
                )
                if item is not None
            ),
            "cascade_sum_lines_keV": tuple(
                float(item)
                for item in _payload_sequence(payload.get("cascade_sum_lines_keV"))
            ),
        }

    def _unique_loaded_key(self, label: str) -> str:
        parts = re.findall(r"[a-z0-9]+", label.lower())
        base = "-".join(parts) or "spectrum"
        existing = {record.key for record in self._state.loaded_spectra}
        if base not in existing:
            return base
        index = 2
        while f"{base}-{index}" in existing:
            index += 1
        return f"{base}-{index}"

    @staticmethod
    def _available_spectrum_id(base: str, existing: set[str]) -> str:
        index = 2
        while f"{base}-{index}" in existing:
            index += 1
        return f"{base}-{index}"


def _payload_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _workspace_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _spectrum_slots_equivalent(
    left: Sequence[SpectrumSlot], right: Sequence[SpectrumSlot]
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.key == b.key
        and a.label == b.label
        and a.spectrum is b.spectrum
        and a.source_key == b.source_key
        and a.source_label == b.source_label
        and a.source_path == b.source_path
        for a, b in zip(left, right)
    )


def _loaded_records_equivalent(
    left: Sequence[LoadedSpectrumRecord], right: Sequence[LoadedSpectrumRecord]
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.key == b.key
        and a.label == b.label
        and a.spectrum is b.spectrum
        and a.source_path == b.source_path
        for a, b in zip(left, right)
    )


def _invalid_fit_diagnostic(
    diagnostic: FitDiagnostics,
    *,
    reason: str,
    **changes: Any,
) -> FitDiagnostics:
    """Return an explicit invalid state with no displayable stale residuals."""

    flags = tuple(
        dict.fromkeys((*diagnostic.warning_flags, "analysis-edit-invalidated"))
    )
    provenance = {
        **dict(diagnostic.provenance),
        "invalidated_by": reason,
    }
    return replace(
        diagnostic,
        status="invalid",
        x=(),
        observed=(),
        model=(),
        uncertainty=(),
        normalized_residuals=(),
        goodness_of_fit={},
        warning_flags=flags,
        provenance=provenance,
        **changes,
    )


def _invalid_peak_model_fit(
    peak: PeakModel,
    *,
    reason: str,
    **changes: Any,
) -> PeakModel:
    """Return a peak leaf that cannot expose residuals from an obsolete fit."""

    provenance = {
        **dict(peak.provenance),
        "fit_invalidated_by": reason,
    }
    return replace(
        peak,
        status="invalidated",
        fit_quality=0.0,
        normalized_residuals=(),
        residual_channels=(),
        provenance=provenance,
        **changes,
    )


def _json_safe(value: Any) -> Any:
    """Convert legacy DTOs to strict JSON data for ``workflow_state``."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, EfficiencyCurve):
        return {
            "model_type": value.model_type,
            "parameters": _json_safe(value.parameters),
            "energy_range": _json_safe(value.energy_range),
            "detector_id": value.detector_id,
            "calibration_date": value.calibration_date,
            "calibration_sources": _json_safe(value.calibration_sources),
            "geometry": _json_safe(value.geometry),
            "uncertainty_model": _json_safe(value.uncertainty_model),
        }
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    raise TypeError(f"Unsupported workspace workflow value: {type(value).__name__}")


def _decode_dataclass(cls, payload: Any):
    if not isinstance(payload, Mapping):
        return None
    allowed = {field.name for field in dataclasses.fields(cls) if field.init}
    try:
        return cls(**{key: payload[key] for key in allowed if key in payload})
    except (TypeError, ValueError):
        return None


def _decode_efficiency_curve(payload: Any) -> EfficiencyCurve | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        energy_range = tuple(
            float(value) for value in payload.get("energy_range", (0.0, 10000.0))
        )
        if len(energy_range) != 2:
            return None
        return EfficiencyCurve(
            model_type=str(payload.get("model_type") or "polynomial"),
            parameters=dict(payload.get("parameters") or {}),
            energy_range=(energy_range[0], energy_range[1]),
            detector_id=str(payload.get("detector_id") or ""),
            calibration_date=str(payload.get("calibration_date") or ""),
            calibration_sources=[
                str(item) for item in payload.get("calibration_sources", ())
            ],
            geometry=dict(payload.get("geometry") or {}),
            uncertainty_model=(
                dict(payload["uncertainty_model"])
                if isinstance(payload.get("uncertainty_model"), Mapping)
                else None
            ),
        )
    except (TypeError, ValueError):
        return None


def _decode_efficiency_fit(payload: Any) -> EfficiencyCalibrationFitResult | None:
    if not isinstance(payload, Mapping):
        return None
    curve = _decode_efficiency_curve(payload.get("curve"))
    if curve is None:
        return None
    try:
        return EfficiencyCalibrationFitResult(
            model_key=str(payload.get("model_key") or ""),
            model_label=str(payload.get("model_label") or ""),
            curve=curve,
            residuals=tuple(float(item) for item in payload.get("residuals", ())),
            rmse=float(payload.get("rmse", 0.0)),
            points_used=int(payload.get("points_used", 0)),
        )
    except (TypeError, ValueError):
        return None


def _decode_roi_analysis(payload: Any) -> ROIAnalysisResult | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        values = dict(payload)
        values["roi_bounds_keV"] = tuple(values.get("roi_bounds_keV", ()))
        values["sideband_bounds_keV"] = tuple(
            tuple(item) for item in values.get("sideband_bounds_keV", ())
        )
        values["overlap_components"] = tuple(
            item
            for item in (
                _decode_dataclass(ROIComponentFit, raw)
                for raw in _payload_sequence(values.get("overlap_components"))
            )
            if item is not None
        )
        values["notes"] = tuple(values.get("notes", ()))
        return ROIAnalysisResult(**values)
    except (TypeError, ValueError):
        return None


def _decode_roi_statistics(payload: Any) -> ROIStatisticsResult | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        values = dict(payload)
        values["roi_bounds_keV"] = tuple(values.get("roi_bounds_keV", ()))
        values["samples"] = tuple(
            item
            for item in (
                _decode_dataclass(ROISpectrumStatistic, raw)
                for raw in _payload_sequence(values.get("samples"))
            )
            if item is not None
        )
        return ROIStatisticsResult(**values)
    except (TypeError, ValueError):
        return None
