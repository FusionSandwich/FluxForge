"""Shared state container for the modern analysis GUI workflow."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from fluxforge.core.analysis_workspace import (
    ActivityCalculationResult,
    EfficiencyCalibrationFitResult,
    PeakCandidate,
    ROIAnalysisResult,
    ROIStatisticsResult,
    SurveyPoint,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.io.spe import GammaSpectrum

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.qt_compat import QUndoCommand


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
    roi_background_method: str = "roi_sideband"
    background_mode: str = "simple"
    background_scale: float = 1.0
    background_visible: bool = True
    efficiency_fit: EfficiencyCalibrationFitResult | None = None
    activity_results: tuple[ActivityCalculationResult, ...] = ()
    roi_analysis: ROIAnalysisResult | None = None
    roi_statistics: ROIStatisticsResult | None = None
    survey_points: tuple[SurveyPoint, ...] = ()
    cascade_sum_lines_keV: tuple[float, ...] = ()


WorkspaceListener = Callable[[AnalysisWorkspaceState], None]


class AnalysisWorkspaceController:
    """Observable state store for the modern analysis shell."""

    def __init__(self, initial_state: AnalysisWorkspaceState | None = None) -> None:
        self._state = initial_state or AnalysisWorkspaceState()
        self._listeners: list[WorkspaceListener] = []

    @property
    def state(self) -> AnalysisWorkspaceState:
        return self._state

    def subscribe(self, listener: WorkspaceListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: WorkspaceListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def set_state(self, state: AnalysisWorkspaceState) -> AnalysisWorkspaceState:
        self._state = state
        for listener in tuple(self._listeners):
            listener(state)
        return state

    def update(self, **changes) -> AnalysisWorkspaceState:
        return self.set_state(replace(self._state, **changes))

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
            label or spectrum.spectrum_id or (source_path.rsplit("/", 1)[-1] if source_path else "Spectrum")
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
                        source_key=source_key if source_key is not None else slot.source_key,
                        source_label=(
                            source_label if source_label is not None else slot.source_label
                        ),
                        source_path=source_path if source_path is not None else slot.source_path,
                    )
                )
                matched = True
            else:
                slots.append(slot)
        if not matched:
            slots.append(
                SpectrumSlot(
                    key=key,
                    label=key.title(),
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
        return self.update(active_spectrum_key=key)

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
        peaks = [
            updated_peak if peak.peak_id == updated_peak.peak_id else peak
            for peak in self._state.peaks
        ]
        return self.update(peaks=tuple(peaks))

    def set_pinned_nuclides(self, pinned_nuclides: Sequence[str]) -> AnalysisWorkspaceState:
        deduped: list[str] = []
        for nuclide in pinned_nuclides:
            if nuclide and nuclide not in deduped:
                deduped.append(nuclide)
        return self.update(pinned_nuclides=tuple(deduped))

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
            "background_mode": mode if mode is not None else self._state.background_mode,
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

    def set_roi_background_method(self, method: str) -> AnalysisWorkspaceState:
        return self.update(roi_background_method=str(method))

    def set_efficiency_fit(
        self,
        fit: EfficiencyCalibrationFitResult | None,
    ) -> AnalysisWorkspaceState:
        return self.update(efficiency_fit=fit)

    def set_activity_results(
        self,
        results: Sequence[ActivityCalculationResult],
    ) -> AnalysisWorkspaceState:
        return self.update(activity_results=tuple(results))

    def set_roi_analysis(
        self,
        result: ROIAnalysisResult | None,
    ) -> AnalysisWorkspaceState:
        return self.update(roi_analysis=result)

    def set_roi_statistics(
        self,
        result: ROIStatisticsResult | None,
    ) -> AnalysisWorkspaceState:
        return self.update(roi_statistics=result)

    def set_survey_points(self, survey_points: Sequence[SurveyPoint]) -> AnalysisWorkspaceState:
        return self.update(survey_points=tuple(survey_points))

    def set_cascade_sum_lines(self, energies_keV: Sequence[float]) -> AnalysisWorkspaceState:
        return self.update(
            cascade_sum_lines_keV=tuple(float(value) for value in energies_keV)
        )

    def describe(self) -> dict[str, object]:
        return {
            "active_spectrum_key": self._state.active_spectrum_key,
            "spectrum_keys": [slot.key for slot in self._state.spectra],
            "slot_sources": {
                slot.key: slot.source_label or slot.label for slot in self._state.spectra
            },
            "loaded_spectrum_count": len(self._state.loaded_spectra),
            "peak_count": len(self._state.peaks),
            "selected_peak_id": self._state.selected_peak_id,
            "pinned_nuclides": list(self._state.pinned_nuclides),
            "peak_search_method": self._state.peak_search_method,
            "roi_background_method": self._state.roi_background_method,
            "background_mode": self._state.background_mode,
            "background_visible": self._state.background_visible,
            "activity_result_count": len(self._state.activity_results),
            "has_roi_analysis": self._state.roi_analysis is not None,
            "has_roi_statistics": self._state.roi_statistics is not None,
            "survey_point_count": len(self._state.survey_points),
            "cascade_sum_line_count": len(self._state.cascade_sum_lines_keV),
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


if QT_AVAILABLE:  # pragma: no cover - optional dependency branch

    class WorkspaceStateCommand(QUndoCommand):
        """Undoable replacement of the shared analysis workspace state."""

        def __init__(
            self,
            controller: AnalysisWorkspaceController,
            *,
            description: str,
            before: AnalysisWorkspaceState,
            after: AnalysisWorkspaceState,
        ) -> None:
            super().__init__(description)
            self.controller = controller
            self.before = copy.deepcopy(before)
            self.after = copy.deepcopy(after)

        def undo(self) -> None:
            self.controller.set_state(copy.deepcopy(self.before))

        def redo(self) -> None:
            self.controller.set_state(copy.deepcopy(self.after))


else:

    class WorkspaceStateCommand:  # pragma: no cover - import-safe placeholder
        def __init__(
            self,
            controller: AnalysisWorkspaceController,
            *,
            description: str,
            before: AnalysisWorkspaceState,
            after: AnalysisWorkspaceState,
        ) -> None:
            self.controller = controller
            self.description = description
            self.before = before
            self.after = after
