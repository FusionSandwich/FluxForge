"""Central modern-shell panel modules."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from fluxforge.core.analysis_workspace import subtract_background_counts
from fluxforge.core.predictive import (
    estimate_count_target_forecast,
    estimate_dead_time_forecast,
    estimate_recalibration_forecast,
)
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, pyqtgraph_backend_status
from fluxforge.gui.mode_manager import GUIMode, ModeManager
from fluxforge.gui.panels.modern_shell_shared import (
    build_demo_spectrum,
    current_tab_label,
    format_duration as _format_duration,
    format_percent as _format_percent,
    set_tab_label,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import SpectrumTrace
from fluxforge.standards import QAMonitor

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    if PYQTGRAPH_AVAILABLE:
        import pyqtgraph as pg

    from fluxforge.gui.backends import PyQtGraphSpectrumCanvas
    from fluxforge.gui.panels.modern_shell_shared import card as _card
    from fluxforge.gui.qt_compat import (
        QFrame,
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

            intro = QLabel(
                (
                    "Predictive analytics use the current ROI, loaded-spectrum history, "
                    "and QA trend data to estimate target-count timing, dead-time saturation, "
                    "and recalibration risk without requiring live MCA acquisition."
                ),
                self,
            )
            intro.setObjectName("PanelBody")
            intro.setWordWrap(True)
            layout.addWidget(intro)

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
                layout.addWidget(self.count_rate_plot, 1)

                self.dead_time_plot = pg.PlotWidget(self)
                self.dead_time_plot.setObjectName("PredictiveDeadTimePlot")
                self.dead_time_plot.setBackground("#0f172a")
                self.dead_time_plot.setLabel("left", "Dead Time (%)")
                self.dead_time_plot.setLabel("bottom", "History Index")
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

            input_rate = float(active.metadata.get("input_count_rate_cps", active.count_rate))
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
                summary_lines[-1] += (
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
                            recalibration_forecast.predicted_recalibration_at.strftime("%Y-%m-%d")
                            if recalibration_forecast.predicted_recalibration_at is not None
                            else "stable"
                        )
                        + (
                            f" ({recalibration_forecast.days_until_recalibration:.1f} d)"
                            if recalibration_forecast.days_until_recalibration is not None
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
        """Center-zone tab stack with the modern spectrum and survey surfaces."""

        def __init__(
            self,
            mode_manager: ModeManager,
            selection_bus: SelectionBus,
            workspace_controller: AnalysisWorkspaceController,
            qa_monitor: QAMonitor,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.mode_manager = mode_manager
            self.selection_bus = selection_bus
            self.workspace_controller = workspace_controller
            self.qa_monitor = qa_monitor
            self._current_spectrum = workspace_controller.spectrum() or build_demo_spectrum()
            self.setObjectName("CentralWorkspaceTabs")
            self.addTab(self._build_spectrum_tab(), "Spectrum")
            self.addTab(self._build_dashboard_tab(), "Dashboard")
            self.mode_manager.subscribe(self._sync_mode_banner)
            self.mode_manager.subscribe(self._sync_workspace_mode)
            self.workspace_controller.subscribe(self._sync_workspace_state)
            self._sync_mode_banner(self.mode_manager.state)
            self._sync_workspace_state(self.workspace_controller.state)

        def _build_spectrum_tab(self) -> QWidget:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(18)

            self.standards_banner = QLabel(widget)
            self.standards_banner.setObjectName("StandardsBanner")
            self.standards_banner.setVisible(False)
            layout.addWidget(self.standards_banner)

            hero = QFrame(widget)
            hero.setObjectName("HeroCanvas")
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(28, 28, 28, 28)
            hero_layout.setSpacing(10)

            eyebrow = QLabel("FluxForge Next", hero)
            eyebrow.setObjectName("HeroEyebrow")
            hero_layout.addWidget(eyebrow)

            title = QLabel("Native, dockable HPGe workspace", hero)
            title.setObjectName("HeroHeader")
            hero_layout.addWidget(title)

            subtitle = QLabel(
                (
                    "PySide6 shell with a PyQtGraph-first spectrum canvas, standards-aware "
                    "workflow modes, and a clearly separated legacy fallback."
                ),
                hero,
            )
            subtitle.setWordWrap(True)
            subtitle.setObjectName("HeroSubhead")
            hero_layout.addWidget(subtitle)
            layout.addWidget(hero)

            self.spectrum_slot_tabs = QTabBar(widget)
            self.spectrum_slot_tabs.setObjectName("CanvasSpectrumTabs")
            self.spectrum_slot_tabs.currentChanged.connect(self._slot_tab_changed)
            layout.addWidget(self.spectrum_slot_tabs)

            body = QHBoxLayout()
            body.setSpacing(18)
            layout.addLayout(body, 1)

            if PYQTGRAPH_AVAILABLE:
                self.canvas = PyQtGraphSpectrumCanvas(
                    selection_bus=self.selection_bus,
                    parent=widget,
                )
                body.addWidget(self.canvas, 3)
            else:
                status = pyqtgraph_backend_status()
                reason = status["reason"] or "Install the native GUI extras."
                body.addWidget(
                    _card(
                        "Renderer pending optional extras",
                        "The production canvas is wired, but this local environment does not have PySide6 + PyQtGraph installed.",
                        f"Import status: {reason}",
                    ),
                    3,
                )

            rail = QWidget(widget)
            rail_layout = QVBoxLayout(rail)
            rail_layout.setContentsMargins(0, 0, 0, 0)
            rail_layout.setSpacing(18)
            rail_layout.addWidget(
                _card(
                    "Visual Feedback First",
                    "Peak fits, calibration, results, and standards context stay visible around the canvas instead of hiding behind modal-only flows.",
                    "Design target: bGamma polish with InterSpec-grade canvas interaction.",
                )
            )
            rail_layout.addWidget(
                _card(
                    "Shared Analytical State",
                    "SelectionBus synchronizes the peak table, sidebar, and tool inspector around the same active ROI or nuclide.",
                    "Current shell wiring already reflects cross-panel selection state.",
                )
            )
            rail_layout.addWidget(
                _card(
                    "Offline-First Reporting",
                    "Session provenance, native exports, and report templates stay local and reproducible.",
                    "Roadmap target: Jinja2 templates with residuals embedded by default.",
                )
            )
            rail_layout.addStretch(1)
            body.addWidget(rail, 2)

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
            target_slot = slot_key or self.workspace_controller.state.active_spectrum_key
            self.workspace_controller.replace_spectrum_slot(
                target_slot,
                spectrum,
                source_key=source_key,
                source_label=source_label,
                source_path=source_path,
            )

        def current_spectrum(self):
            """Return the current spectrum object shown in the central workspace."""

            return self.workspace_controller.spectrum() or self._current_spectrum

        def set_log_scale(self, enabled: bool) -> None:
            if PYQTGRAPH_AVAILABLE and hasattr(self, "canvas") and hasattr(self.canvas, "set_log_scale"):
                self.canvas.set_log_scale(bool(enabled))

        def set_peak_labels_visible(self, visible: bool) -> None:
            if (
                PYQTGRAPH_AVAILABLE
                and hasattr(self, "canvas")
                and hasattr(self.canvas, "set_peak_labels_visible")
            ):
                self.canvas.set_peak_labels_visible(bool(visible))

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

            title = QLabel("Digital twin hardware dashboard", widget)
            title.setObjectName("HeroHeader")
            layout.addWidget(title)

            subtitle = QLabel(
                "Reserved for MCA device discovery, status telemetry, live acquisition, and the future spectrogram surface.",
                widget,
            )
            subtitle.setWordWrap(True)
            subtitle.setObjectName("HeroSubhead")
            layout.addWidget(subtitle)

            predictive_note = QLabel(
                (
                    "Live MCA transport remains deferred, but the predictive subset from the "
                    "GUI plan is now active here using offline spectra and QA history."
                ),
                widget,
            )
            predictive_note.setObjectName("HeroCardAccent")
            predictive_note.setWordWrap(True)
            layout.addWidget(predictive_note)

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
            self._current_spectrum = self.workspace_controller.spectrum() or self._current_spectrum
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
                elif slot.key == "overlay":
                    overlay_slot = slot
            foreground = (
                foreground_slot.spectrum if foreground_slot is not None else self._current_spectrum
            )
            background = background_slot.spectrum if background_slot is not None else None
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
                            if foreground_slot is not None and foreground_slot.source_label
                            else "Foreground"
                        ),
                        counts=tuple(float(value) for value in primary_counts),
                        channels=tuple(float(value) for value in np.asarray(foreground.channels, dtype=float)),
                        color="#72d6ff",
                    )
                )
            if background is not None and state.background_visible:
                traces.append(
                    SpectrumTrace(
                        label=(
                            background_slot.source_label
                            if background_slot is not None and background_slot.source_label
                            else "Background"
                        ),
                        counts=tuple(float(value) for value in np.asarray(background.counts, dtype=float)),
                        channels=tuple(float(value) for value in np.asarray(background.channels, dtype=float)),
                        color="#f59e0b",
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
                        counts=tuple(float(value) for value in np.asarray(overlay.counts, dtype=float)),
                        channels=tuple(float(value) for value in np.asarray(overlay.channels, dtype=float)),
                        color="#10b981",
                    )
                )
            self.canvas.set_traces(traces)
            self.canvas.set_peak_candidates(state.peaks)
            self.canvas.set_cascade_sum_lines(state.cascade_sum_lines_keV)
            self.canvas.set_peak_residuals(
                state.peaks[:3],
                visible=self.mode_manager.state.mode is not GUIMode.SIMPLE,
            )
            if state.peaks and state.selected_peak_id:
                selected_peak = next(
                    (peak for peak in state.peaks if peak.peak_id == state.selected_peak_id),
                    None,
                )
                if selected_peak is not None:
                    self.selection_bus.publish(
                        SelectionState(
                            peak_energy_keV=selected_peak.energy_keV,
                            roi_bounds_keV=selected_peak.roi_bounds_keV,
                            nuclide=selected_peak.nuclide,
                            reference_lines_keV=selected_peak.reference_lines_keV,
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
