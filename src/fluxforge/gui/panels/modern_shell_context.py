"""Context-side modern-shell panel module."""

from __future__ import annotations

from collections.abc import Mapping

from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    from fluxforge.gui.panels.modern_shell_shared import (
        selection_summary as _selection_summary,
    )
    from fluxforge.gui.qt_compat import QLabel, QVBoxLayout, QWidget

    class ToolContextPanel(QWidget):
        """Right-side live analysis context."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QLabel("Tool Context", self)
            title.setObjectName("PanelHeading")
            layout.addWidget(title)

            self.mode_summary = QLabel(self)
            self.mode_summary.setWordWrap(True)
            self.mode_summary.setObjectName("PanelBody")
            layout.addWidget(self.mode_summary)

            self.selection_summary = QLabel("No active selection", self)
            self.selection_summary.setWordWrap(True)
            self.selection_summary.setObjectName("PanelBody")
            layout.addWidget(self.selection_summary)

            self.workspace_summary = QLabel(self)
            self.workspace_summary.setWordWrap(True)
            self.workspace_summary.setObjectName("PanelBody")
            layout.addWidget(self.workspace_summary)

            self.analysis_summary = QLabel(self)
            self.analysis_summary.setWordWrap(True)
            self.analysis_summary.setObjectName("PanelBody")
            layout.addWidget(self.analysis_summary)
            layout.addStretch(1)

            self.mode_manager.subscribe(self._sync_mode)
            self.selection_bus.subscribe(self._sync_selection)
            self.workspace_controller.subscribe(self._sync_workspace)
            self._sync_mode(self.mode_manager.state)
            self._sync_selection(self.selection_bus.state)
            self._sync_workspace(self.workspace_controller.state)

        def _sync_mode(self, state) -> None:
            description = f"Mode: {state.mode.value.title()}"
            if state.standard:
                description += f" ({state.standard})"
            description += f"\nTheme: {state.theme}"
            self.mode_summary.setText(description)

        def _sync_selection(self, state: SelectionState) -> None:
            self.selection_summary.setText(_selection_summary(state))

        def _sync_workspace(self, state) -> None:
            foreground = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "foreground"
                ),
                "none",
            )
            background = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "background"
                ),
                "none",
            )
            overlay = next(
                (
                    slot.source_label or slot.label
                    for slot in state.spectra
                    if slot.key == "overlay"
                ),
                "none",
            )
            self.workspace_summary.setText(
                (
                    f"Active spectrum: {state.active_spectrum_key}\n"
                    f"Foreground source: {foreground}\n"
                    f"Background source: {background}\n"
                    f"Overlay source: {overlay}\n"
                    f"Peaks: {len(state.peaks)}\n"
                    f"Pinned nuclides: {', '.join(state.pinned_nuclides) or 'none'}\n"
                    f"Background: {state.background_mode} "
                    f"({'visible' if state.background_visible else 'hidden'})"
                )
            )
            roi = state.roi_analysis
            efficiency = state.efficiency_fit
            invalidation = self.workspace_controller.document.workflow_state.get(
                "analysis_invalidation"
            )
            invalidation_line = ""
            if isinstance(invalidation, Mapping) and invalidation.get(
                "requires_reanalysis"
            ):
                reason = str(invalidation.get("reason") or "scientific edit")
                invalidation_line = f"\nAnalysis status: re-run required ({reason})"
            self.analysis_summary.setText(
                (
                    f"Efficiency calibration: {'ready' if efficiency is not None else 'not fitted'}\n"
                    f"Activity results: {len(state.activity_results)}\n"
                    f"ROI analysis: "
                    f"{f'{roi.roi_bounds_keV[0]:.2f}-{roi.roi_bounds_keV[1]:.2f} keV' if roi is not None else 'not run'}"
                    f"{invalidation_line}"
                )
            )

else:

    class ToolContextPanel:  # pragma: no cover - placeholder without Qt
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


__all__ = ["ToolContextPanel"]
