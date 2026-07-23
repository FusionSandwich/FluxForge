"""Central modern-shell panel modules."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from fluxforge.core.analysis_workspace import subtract_background_counts
from fluxforge.core.workspace_document import CanvasViewport, WorkspaceValidationError
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_dead_time_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.canvas_controller import CanvasIntentDispatcher
from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, pyqtgraph_backend_status
from fluxforge.gui.backends.pyqtgraph_backend import catalog_pyqtgraph_export_action
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.panels.modern_shell_shared import (
    current_tab_label,
    format_duration as _format_duration,
    format_percent as _format_percent,
    set_tab_label,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import (
    CanvasPeakOverlay,
    CanvasROIOverlay,
    SpectrumTrace,
)
from fluxforge.standards import QAMonitor

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    if PYQTGRAPH_AVAILABLE:
        import pyqtgraph as pg

    from fluxforge.gui.backends import PyQtGraphSpectrumCanvas
    from fluxforge.gui.qt_compat import (
        QDoubleSpinBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QTabBar,
        QTabWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    class PredictiveDashboardPanel(QWidget):
        """Offline predictive dashboard derived from current spectra and QA history."""

        def __init__(
            self,
            *,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            qa_monitor: QAMonitor,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.qa_monitor = qa_monitor
            self._selection_state = SelectionState()

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Target ROI counts", self))
            self.target_counts_spin = QDoubleSpinBox(self)
            self.target_counts_spin.setObjectName("PredictiveTargetCountsSpin")
            self.target_counts_spin.setRange(100.0, 1_000_000.0)
            self.target_counts_spin.setDecimals(0)
            self.target_counts_spin.setSingleStep(500.0)
            self.target_counts_spin.setValue(10000.0)
            controls.addWidget(self.target_counts_spin)
            controls.addStretch(1)
            layout.addLayout(controls)

            self.metrics_browser = QTextBrowser(self)
            self.metrics_browser.setObjectName("PredictiveMetricsBrowser")
            layout.addWidget(self.metrics_browser)

            if PYQTGRAPH_AVAILABLE:
                self.count_rate_plot = pg.PlotWidget(self)
                self.count_rate_plot.setObjectName("PredictiveCountRatePlot")
                self.count_rate_plot.setBackground("#0f172a")
                self.count_rate_plot.setLabel("left", "ROI cps")
                self.count_rate_plot.setLabel("bottom", "History Index")
                catalog_pyqtgraph_export_action(
                    self.count_rate_plot,
                    "PredictiveCountRatePlot",
                )
                layout.addWidget(self.count_rate_plot, 1)

                self.dead_time_plot = pg.PlotWidget(self)
                self.dead_time_plot.setObjectName("PredictiveDeadTimePlot")
                self.dead_time_plot.setBackground("#0f172a")
                self.dead_time_plot.setLabel("left", "Dead Time (%)")
                self.dead_time_plot.setLabel("bottom", "History Index")
                catalog_pyqtgraph_export_action(
                    self.dead_time_plot,
                    "PredictiveDeadTimePlot",
                )
                layout.addWidget(self.dead_time_plot, 1)

            self.summary_browser = QTextBrowser(self)
            self.summary_browser.setObjectName("PredictiveSummaryBrowser")
            layout.addWidget(self.summary_browser, 1)

            self.selection_bus.subscribe(self._selection_changed)
            self.workspace_controller.subscribe(lambda _state: self.refresh())
            self.target_counts_spin.valueChanged.connect(lambda _value: self.refresh())
            self.refresh()

        def _selection_changed(self, state: SelectionState) -> None:
            self._selection_state = state
            self.refresh()

        def _history_spectra(self):
            records = list(self.workspace_controller.loaded_spectrum_records())
            records.sort(
                key=lambda record: (
                    record.spectrum.start_time or datetime.max,
                    record.label,
                )
            )
            return tuple(record.spectrum for record in records)

        def _selected_roi_bounds(self) -> tuple[float, float] | None:
            if self._selection_state.roi_bounds_keV is not None:
                return self._selection_state.roi_bounds_keV
            selected_peak = self.workspace_controller.selected_peak()
            if selected_peak is not None:
                return selected_peak.roi_bounds_keV
            return None

        def refresh(self) -> None:
            active = self.workspace_controller.spectrum()
            if active is None:
                self.metrics_browser.setHtml("<p>No active spectrum available.</p>")
                self.summary_browser.setHtml("<p>No predictive forecast available.</p>")
                return

            history = self._history_spectra() or (active,)
            roi_bounds = self._selected_roi_bounds()
            count_forecast = estimate_count_target_forecast(
                active,
                roi_bounds_keV=roi_bounds,
                target_counts=float(self.target_counts_spin.value()),
                history_spectra=history,
            )
            dead_time_forecast = estimate_dead_time_forecast(history)
            recalibration_forecast = estimate_recalibration_forecast(
                self.qa_monitor.history()
            )

            input_rate = float(
                active.metadata.get("input_count_rate_cps", active.count_rate)
            )
            metrics_lines = [
                "<h3>Predictive Dashboard</h3>",
                (
                    "<p><strong>Current metrics:</strong> "
                    f"Input {input_rate:,.1f} cps | "
                    f"Output {active.count_rate:,.1f} cps | "
                    f"Dead time {_format_percent(active.dead_time_fraction)} | "
                    f"Live {active.live_time:.0f}s</p>"
                ),
                (
                    "<p><strong>ROI scope:</strong> "
                    + (
                        f"{roi_bounds[0]:.1f}-{roi_bounds[1]:.1f} keV"
                        if roi_bounds is not None
                        else "Full spectrum"
                    )
                    + "</p>"
                ),
            ]
            self.metrics_browser.setHtml("".join(metrics_lines))

            summary_lines = [
                "<h3>Predictions</h3>",
                (
                    "<p><strong>Count target:</strong> "
                    f"{count_forecast.current_counts:.1f} ± {count_forecast.current_uncertainty:.1f} counts | "
                    f"{count_forecast.count_rate_cps:.2f} ± {count_forecast.count_rate_uncertainty_cps:.2f} cps | "
                    f"ETA {_format_duration(count_forecast.eta_seconds)}"
                ),
            ]
            if count_forecast.eta_uncertainty_seconds is not None:
                summary_lines[
                    -1
                ] += (
                    f" ± {_format_duration(count_forecast.eta_uncertainty_seconds)}</p>"
                )
            else:
                summary_lines[-1] += "</p>"
            summary_lines.append(
                (
                    "<p><strong>Count-rate trend:</strong> "
                    f"{count_forecast.trend.slope:+.2f} ± {count_forecast.trend.slope_stderr:.2f} cps/h "
                    f"(R² {count_forecast.trend.r_squared:.3f})</p>"
                )
            )
            summary_lines.append(
                (
                    "<p><strong>Dead-time trend:</strong> "
                    f"{_format_percent(dead_time_forecast.current_dead_time_fraction)} now | "
                    f"{_format_percent(dead_time_forecast.projected_dead_time_fraction_1h)} projected in 1h | "
                    f"saturation {_format_duration(dead_time_forecast.eta_to_saturation_seconds)}</p>"
                )
            )
            if recalibration_forecast is not None:
                summary_lines.append(
                    (
                        "<p><strong>QA recalibration forecast:</strong> "
                        f"{recalibration_forecast.nuclide} {recalibration_forecast.energy_keV:.2f} keV | "
                        f"trigger {recalibration_forecast.trigger_metric} | "
                        f"target date "
                        + (
                            recalibration_forecast.predicted_recalibration_at.strftime(
                                "%Y-%m-%d"
                            )
                            if recalibration_forecast.predicted_recalibration_at
                            is not None
                            else "stable"
                        )
                        + (
                            f" ({recalibration_forecast.days_until_recalibration:.1f} d)"
                            if recalibration_forecast.days_until_recalibration
                            is not None
                            else ""
                        )
                        + "</p>"
                    )
                )
            self.summary_browser.setHtml("".join(summary_lines))

            if PYQTGRAPH_AVAILABLE and hasattr(self, "count_rate_plot"):
                indices = np.arange(len(history), dtype=float)
                if roi_bounds is not None:
                    rate_values = [
                        spectrum.counts_in_range(*roi_bounds)[0]
                        / max(float(spectrum.live_time), 1e-12)
                        for spectrum in history
                    ]
                else:
                    rate_values = [spectrum.count_rate for spectrum in history]
                dead_values = [
                    float(spectrum.dead_time_fraction) * 100.0 for spectrum in history
                ]
                self.count_rate_plot.clear()
                self.dead_time_plot.clear()
                self.count_rate_plot.plot(
                    indices,
                    rate_values,
                    pen=pg.mkPen(color="#72d6ff", width=2),
                    symbol="o",
                    symbolBrush=pg.mkBrush("#72d6ff"),
                )
                self.dead_time_plot.plot(
                    indices,
                    dead_values,
                    pen=pg.mkPen(color="#f59e0b", width=2),
                    symbol="o",
                    symbolBrush=pg.mkBrush("#f59e0b"),
                )

    class CentralWorkspaceTabs(QTabWidget):
        """Center-zone spectrum and forecasting workspaces."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            qa_monitor: QAMonitor,
            undo_stack=None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.qa_monitor = qa_monitor
            self.undo_stack = undo_stack
            self.canvas_intent_dispatcher = CanvasIntentDispatcher(
                self.workspace_controller,
                self.selection_bus,
                undo_stack=self.undo_stack,
            )
            self._last_canvas_viewport: CanvasViewport | None | object = object()
            self._last_canvas_spectrum_id: str | None | object = object()
            self._current_spectrum = workspace_controller.spectrum()
            self.setObjectName("CentralWorkspaceTabs")
            self.addTab(self._build_spectrum_tab(), "Spectrum")
            self.addTab(self._build_dashboard_tab(), "Forecasts")
            self.setProperty(
                "fluxforgeTabIds",
                {
                    "Spectrum": "canvas.spectrum.open",
                    "Forecasts": "workspace.forecasts.open",
                },
            )
            self.mode_manager.subscribe(self._sync_mode_banner)
            self.mode_manager.subscribe(self._sync_workspace_mode)
            self.selection_bus.subscribe(self._sync_canvas_selection)
            self.workspace_controller.subscribe(self._sync_workspace_state)
            self._sync_mode_banner(self.mode_manager.state)
            self._sync_workspace_state(self.workspace_controller.state)

        def _build_spectrum_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)

            self.standards_banner = QLabel(widget)
            self.standards_banner.setObjectName("StandardsBanner")
            self.standards_banner.setVisible(False)
            layout.addWidget(self.standards_banner)

            self.spectrum_slot_tabs = QTabBar(widget)
            self.spectrum_slot_tabs.setObjectName("CanvasSpectrumTabs")
            self.spectrum_slot_tabs.currentChanged.connect(self._slot_tab_changed)
            layout.addWidget(self.spectrum_slot_tabs)

            if PYQTGRAPH_AVAILABLE:
                self.canvas = PyQtGraphSpectrumCanvas(
                    selection_bus=self.selection_bus,
                    parent=widget,
                )
                if hasattr(self.canvas, "set_intent_sink"):
                    self.canvas.set_intent_sink(self._dispatch_canvas_intent)
                elif hasattr(self.canvas, "subscribe_intents"):
                    self.canvas.subscribe_intents(self._dispatch_canvas_intent)
                layout.addWidget(self.canvas, 1)
            else:
                status = pyqtgraph_backend_status()
                reason = status["reason"] or "Install the native GUI extras."
                renderer_status = QLabel(
                    f"Spectrum renderer unavailable: {reason}",
                    widget,
                )
                renderer_status.setObjectName("PanelBody")
                renderer_status.setWordWrap(True)
                layout.addWidget(renderer_status, 1)

            return widget

        def load_spectrum(
            self,
            spectrum,
            *,
            slot_key: str | None = None,
            source_key: str | None = None,
            source_label: str | None = None,
            source_path: str | None = None,
        ) -> None:
            """Load a GammaSpectrum-like object into the primary canvas."""

            self._current_spectrum = spectrum
            target_slot = (
                slot_key or self.workspace_controller.state.active_spectrum_key
            )
            self.workspace_controller.replace_spectrum_slot(
                target_slot,
                spectrum,
                source_key=source_key,
                source_label=source_label,
                source_path=source_path,
            )

        def current_spectrum(self):
            """Return the current spectrum object shown in the central workspace."""

            return self.workspace_controller.spectrum()

        def set_log_scale(self, enabled: bool) -> None:
            if (
                PYQTGRAPH_AVAILABLE
                and hasattr(self, "canvas")
                and hasattr(self.canvas, "set_log_scale")
            ):
                spectrum_id = self.workspace_controller.document.active_spectrum_id
                self._dispatch_canvas_intent(
                    CanvasIntent(
                        CanvasIntentKind.SET_LOG_SCALE,
                        spectrum_id=spectrum_id,
                        viewport_id="primary-spectrum",
                        enabled=bool(enabled),
                    )
                )

        def set_peak_labels_visible(self, visible: bool) -> None:
            if (
                PYQTGRAPH_AVAILABLE
                and hasattr(self, "canvas")
                and hasattr(self.canvas, "set_peak_labels_visible")
            ):
                spectrum_id = self.workspace_controller.document.active_spectrum_id
                self._dispatch_canvas_intent(
                    CanvasIntent(
                        CanvasIntentKind.SET_LABELS_VISIBLE,
                        spectrum_id=spectrum_id,
                        viewport_id="primary-spectrum",
                        enabled=bool(visible),
                    )
                )

        def _dispatch_canvas_intent(self, intent: CanvasIntent) -> None:
            """Apply one canvas edit and restore the canonical view if rejected."""

            try:
                self.canvas_intent_dispatcher.handle(intent)
            except (WorkspaceValidationError, ValueError, KeyError) as exc:
                self._sync_analysis_overlays(self.selection_bus.state)
                if hasattr(self.canvas, "show_interaction_error"):
                    self.canvas.show_interaction_error(str(exc))

        def viewport_state(self) -> CanvasViewport | None:
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return None
            document = self.workspace_controller.document
            current = document.viewport_by_id("primary-spectrum")
            selected_roi_id = current.selected_roi_id if current is not None else None
            return self.canvas.viewport_state(
                viewport_id="primary-spectrum",
                spectrum_id=document.active_spectrum_id,
                selected_roi_id=selected_roi_id,
            )

        def apply_viewport_state(self, viewport: CanvasViewport) -> None:
            if PYQTGRAPH_AVAILABLE and hasattr(self, "canvas"):
                self.canvas.apply_viewport_state(viewport)

        def _sync_canvas_selection(self, selection: SelectionState) -> None:
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return
            roi = (
                self.workspace_controller.document.roi_by_id(selection.roi_id)
                if selection.roi_id is not None
                else None
            )
            bounds = roi.signal_range if roi is not None else selection.roi_bounds_keV
            if (
                selection.zoom_requested
                and bounds is not None
                and hasattr(self.canvas, "zoom_to_range")
            ):
                self.canvas.zoom_to_range(float(bounds[0]), float(bounds[1]))
            self._sync_analysis_overlays(selection)

        def _sync_analysis_overlays(
            self, selection: SelectionState | None = None
        ) -> None:
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return
            document = self.workspace_controller.document
            spectrum_id = document.active_spectrum_id
            if hasattr(self.canvas, "set_interaction_context"):
                self.canvas.set_interaction_context(spectrum_id)
            if not hasattr(self.canvas, "set_analysis_overlays"):
                return
            selection = selection or self.selection_bus.state
            viewport = document.viewport_by_id("primary-spectrum")
            selected_roi_id = (
                selection.roi_id
                if document.roi_by_id(selection.roi_id or "") is not None
                else None
            ) or (viewport.selected_roi_id if viewport is not None else None)
            requested_peak_id = (
                selection.peak_id or self.workspace_controller.state.selected_peak_id
            )
            selected_peak_id = (
                requested_peak_id
                if requested_peak_id is not None
                and document.peak_by_id(spectrum_id, requested_peak_id) is not None
                else None
            )
            rois = tuple(
                CanvasROIOverlay(
                    roi_id=roi.roi_id,
                    spectrum_id=roi.spectrum_id,
                    signal_range=roi.signal_range,
                    left_background_range=roi.left_background_range,
                    right_background_range=roi.right_background_range,
                    color=roi.color,
                    selected=roi.roi_id == selected_roi_id,
                )
                for roi in document.rois
                if roi.spectrum_id == spectrum_id
            )
            pinned = set(document.pinned_nuclides)
            peaks = tuple(
                CanvasPeakOverlay(
                    peak_id=peak.peak_id,
                    spectrum_id=peak.spectrum_id,
                    position=peak.centroid_energy_keV,
                    y_value=max(float(peak.net_counts), 0.0),
                    component_ids=tuple(
                        component.component_id for component in peak.components
                    ),
                    nuclide=(peak.assignments[0].nuclide if peak.assignments else None),
                    tags=peak.tags,
                    nuclide_tags=(
                        tuple(
                            document.nuclide_tags.get(peak.assignments[0].nuclide, ())
                        )
                        if peak.assignments
                        else ()
                    ),
                    pinned=any(
                        assignment.nuclide in pinned for assignment in peak.assignments
                    ),
                    selected=peak.peak_id == selected_peak_id,
                )
                for peak in document.peaks
                if peak.spectrum_id == spectrum_id
            )
            self.canvas.set_analysis_overlays(
                rois,
                peaks,
                selected_roi_id=selected_roi_id,
                selected_peak_id=selected_peak_id,
            )

        def _slot_tab_changed(self, index: int) -> None:
            if index < 0 or index >= len(self.workspace_controller.state.spectra):
                return
            self.workspace_controller.select_spectrum(
                self.workspace_controller.state.spectra[index].key
            )

        def _build_dashboard_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(18)

            title = QLabel("Count and QA Forecasts", widget)
            title.setObjectName("HeroHeader")
            layout.addWidget(title)

            subtitle = QLabel(
                "Forecasts derived from the active spectrum and recorded QA history.",
                widget,
            )
            subtitle.setWordWrap(True)
            subtitle.setObjectName("HeroSubhead")
            layout.addWidget(subtitle)

            self.predictive_dashboard = PredictiveDashboardPanel(
                selection_bus=self.selection_bus,
                workspace_controller=self.workspace_controller,
                qa_monitor=self.qa_monitor,
                parent=widget,
            )
            layout.addWidget(self.predictive_dashboard, 1)
            layout.addStretch(1)
            return widget

        def _sync_workspace_state(self, state) -> None:
            self._current_spectrum = self.workspace_controller.spectrum()
            self._reconcile_selection_with_document()
            self._sync_slot_tabs(state)
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return

            foreground_slot = None
            background_slot = None
            overlay_slot = None
            for slot in state.spectra:
                if slot.key == state.active_spectrum_key:
                    foreground_slot = slot
                elif slot.key == "background":
                    background_slot = slot
                elif slot.key in {"overlay", "secondary"}:
                    overlay_slot = slot
            foreground = (
                foreground_slot.spectrum if foreground_slot is not None else None
            )
            background = (
                background_slot.spectrum if background_slot is not None else None
            )
            overlay = overlay_slot.spectrum if overlay_slot is not None else None
            traces: list[SpectrumTrace] = []
            if foreground is not None:
                primary_counts = np.asarray(foreground.counts, dtype=float)
                if (
                    background is not None
                    and state.active_spectrum_key != "background"
                    and state.background_mode in {"simple", "scaled", "statistical"}
                ):
                    primary_counts = subtract_background_counts(
                        foreground,
                        background,
                        mode=state.background_mode,
                        scale=state.background_scale,
                    )
                traces.append(
                    SpectrumTrace(
                        label=(
                            foreground_slot.source_label
                            if foreground_slot is not None
                            and foreground_slot.source_label
                            else "Foreground"
                        ),
                        counts=tuple(float(value) for value in primary_counts),
                        channels=tuple(
                            float(value)
                            for value in np.asarray(
                                (
                                    foreground.energies
                                    if foreground.energies is not None
                                    else foreground.channels
                                ),
                                dtype=float,
                            )
                        ),
                        color="#72d6ff",
                        x_axis_label=(
                            "Energy (keV)"
                            if foreground.energies is not None
                            else "Channel"
                        ),
                    )
                )
            if background is not None and state.background_visible:
                traces.append(
                    SpectrumTrace(
                        label=(
                            background_slot.source_label
                            if background_slot is not None
                            and background_slot.source_label
                            else "Background"
                        ),
                        counts=tuple(
                            float(value)
                            for value in np.asarray(background.counts, dtype=float)
                        ),
                        channels=tuple(
                            float(value)
                            for value in np.asarray(
                                (
                                    background.energies
                                    if background.energies is not None
                                    else background.channels
                                ),
                                dtype=float,
                            )
                        ),
                        color="#f59e0b",
                        x_axis_label=(
                            "Energy (keV)"
                            if background.energies is not None
                            else "Channel"
                        ),
                    )
                )
            if overlay is not None:
                traces.append(
                    SpectrumTrace(
                        label=(
                            overlay_slot.source_label
                            if overlay_slot is not None and overlay_slot.source_label
                            else "Secondary Overlay"
                        ),
                        counts=tuple(
                            float(value)
                            for value in np.asarray(overlay.counts, dtype=float)
                        ),
                        channels=tuple(
                            float(value)
                            for value in np.asarray(
                                (
                                    overlay.energies
                                    if overlay.energies is not None
                                    else overlay.channels
                                ),
                                dtype=float,
                            )
                        ),
                        color="#10b981",
                        x_axis_label=(
                            "Energy (keV)"
                            if overlay.energies is not None
                            else "Channel"
                        ),
                    )
                )
            self.canvas.set_traces(traces)
            self._apply_document_viewport()
            self.canvas.set_peak_candidates(state.peaks)
            self._sync_analysis_overlays()
            self.canvas.set_cascade_sum_lines(state.cascade_sum_lines_keV)
            self.canvas.set_peak_residuals(
                state.peaks[:3],
                visible=self.mode_manager.state.mode is not GUIMode.SIMPLE,
            )
            if state.peaks and state.selected_peak_id:
                selected_peak = next(
                    (
                        peak
                        for peak in state.peaks
                        if peak.peak_id == state.selected_peak_id
                    ),
                    None,
                )
                if selected_peak is not None:
                    self.selection_bus.publish(
                        SelectionState(
                            spectrum_id=(
                                self.workspace_controller.document.active_spectrum_id
                            ),
                            peak_id=selected_peak.peak_id,
                            peak_energy_keV=selected_peak.energy_keV,
                            roi_bounds_keV=selected_peak.roi_bounds_keV,
                            nuclide=selected_peak.nuclide,
                            reference_lines_keV=selected_peak.reference_lines_keV,
                        )
                    )

        def _reconcile_selection_with_document(self) -> None:
            """Remove exact IDs that no longer exist after undo or session load."""

            document = self.workspace_controller.document
            state = self.selection_bus.state
            spectrum_exists = state.spectrum_id is None or (
                state.spectrum_id == document.active_spectrum_id
                and document.spectrum_by_id(state.spectrum_id) is not None
            )
            peak_exists = state.peak_id is None or (
                spectrum_exists
                and document.peak_by_id(
                    state.spectrum_id or document.active_spectrum_id,
                    state.peak_id,
                )
                is not None
            )
            selected_roi = (
                document.roi_by_id(state.roi_id) if state.roi_id is not None else None
            )
            roi_exists = state.roi_id is None or (
                selected_roi is not None
                and selected_roi.spectrum_id == document.active_spectrum_id
            )
            if spectrum_exists and peak_exists and roi_exists:
                return
            reconciled = SelectionState(
                spectrum_id=state.spectrum_id if spectrum_exists else None,
                peak_id=state.peak_id if peak_exists else None,
                roi_id=state.roi_id if roi_exists else None,
                peak_energy_keV=(state.peak_energy_keV if peak_exists else None),
                roi_bounds_keV=(state.roi_bounds_keV if roi_exists else None),
                nuclide=state.nuclide if peak_exists else None,
                reference_lines_keV=(state.reference_lines_keV if peak_exists else ()),
                annotation_lines=state.annotation_lines if peak_exists else (),
                zoom_requested=False,
            )
            self.selection_bus.publish(reconciled)

        def _apply_document_viewport(self) -> None:
            """Reapply canonical viewport changes, including undo to no viewport."""

            if not PYQTGRAPH_AVAILABLE or not hasattr(self, "canvas"):
                return
            document = self.workspace_controller.document
            active_spectrum_id = document.active_spectrum_id
            persisted_viewport = document.viewport_by_id("primary-spectrum")
            viewport = (
                persisted_viewport
                if persisted_viewport is None
                or persisted_viewport.spectrum_id in {None, active_spectrum_id}
                else None
            )
            if (
                viewport == self._last_canvas_viewport
                and active_spectrum_id == self._last_canvas_spectrum_id
            ):
                return
            previous = self._last_canvas_viewport
            self._last_canvas_viewport = viewport
            spectrum_changed = active_spectrum_id != self._last_canvas_spectrum_id
            self._last_canvas_spectrum_id = active_spectrum_id
            if viewport is not None:
                self.canvas.apply_viewport_state(viewport)
            elif isinstance(previous, CanvasViewport) or spectrum_changed:
                if hasattr(self.canvas, "reset_persisted_viewport_state"):
                    self.canvas.reset_persisted_viewport_state()
            previous_roi_id = (
                previous.selected_roi_id
                if isinstance(previous, CanvasViewport)
                else None
            )
            selected_roi_id = viewport.selected_roi_id if viewport is not None else None
            if selected_roi_id != previous_roi_id:
                state = self.selection_bus.state
                roi = (
                    document.roi_by_id(selected_roi_id)
                    if selected_roi_id is not None
                    else None
                )
                self.selection_bus.publish(
                    SelectionState(
                        spectrum_id=state.spectrum_id or document.active_spectrum_id,
                        peak_id=state.peak_id,
                        roi_id=roi.roi_id if roi is not None else None,
                        peak_energy_keV=state.peak_energy_keV,
                        roi_bounds_keV=(roi.signal_range if roi is not None else None),
                        nuclide=state.nuclide,
                        reference_lines_keV=state.reference_lines_keV,
                        annotation_lines=state.annotation_lines,
                        zoom_requested=False,
                    )
                )

        def _sync_slot_tabs(self, state) -> None:
            self.spectrum_slot_tabs.blockSignals(True)
            while self.spectrum_slot_tabs.count():
                self.spectrum_slot_tabs.removeTab(0)
            current_index = 0
            for index, slot in enumerate(state.spectra):
                self.spectrum_slot_tabs.addTab(slot.label)
                self.spectrum_slot_tabs.setTabData(index, slot.key)
                if slot.key == state.active_spectrum_key:
                    current_index = index
            self.spectrum_slot_tabs.setCurrentIndex(current_index)
            self.spectrum_slot_tabs.blockSignals(False)

        def _sync_workspace_mode(self, state) -> None:
            if PYQTGRAPH_AVAILABLE and hasattr(self, "canvas"):
                self.canvas.set_peak_residuals(
                    self.workspace_controller.state.peaks[:3],
                    visible=state.mode is not GUIMode.SIMPLE,
                )

        def _sync_mode_banner(self, state) -> None:
            locked = state.mode is GUIMode.STANDARDS and state.standard
            self.standards_banner.setVisible(bool(locked))
            if locked:
                self.standards_banner.setText(
                    f"Standards mode locked to {state.standard}. Alternate methods remain available in Expert mode."
                )

        def workflow_state(self) -> dict[str, object]:
            return {
                "current_tab": current_tab_label(self),
            }

        def apply_workflow_state(self, payload: dict[str, object] | None) -> None:
            if not payload:
                return
            if "current_tab" in payload:
                set_tab_label(self, str(payload["current_tab"]))

else:

    class PredictiveDashboardPanel:  # pragma: no cover - placeholder without Qt
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class CentralWorkspaceTabs:  # pragma: no cover - placeholder without Qt
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


__all__ = ["PredictiveDashboardPanel", "CentralWorkspaceTabs"]
