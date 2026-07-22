"""PyQtGraph-backed spectrum canvas implementation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import (
    HierarchicalSpectrumBuffer,
    ReferenceLine,
    RendererCapabilities,
    SpectrumCanvas,
    SpectrumTrace,
)


PYQTGRAPH_AVAILABLE = False
PYQTGRAPH_IMPORT_ERROR = QT_IMPORT_ERROR

if QT_AVAILABLE:  # pragma: no cover - optional dependency branch
    try:
        import pyqtgraph as pg

        from fluxforge.gui.qt_compat import (
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        PYQTGRAPH_AVAILABLE = True
        PYQTGRAPH_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - optional dependency branch
        PYQTGRAPH_IMPORT_ERROR = exc


def pyqtgraph_backend_status() -> dict[str, object]:
    """Return import-safe availability metadata for the default renderer."""

    reason = None
    if PYQTGRAPH_IMPORT_ERROR is not None:
        reason = f"{type(PYQTGRAPH_IMPORT_ERROR).__name__}: {PYQTGRAPH_IMPORT_ERROR}"
    return {
        "key": "pyqtgraph",
        "display_name": "PyQtGraph",
        "recommended": True,
        "available": PYQTGRAPH_AVAILABLE,
        "reason": reason,
    }


if PYQTGRAPH_AVAILABLE:  # pragma: no cover - optional dependency branch

    class PyQtGraphSpectrumCanvas(QWidget, SpectrumCanvas):
        """Production-ready Qt spectrum canvas for the redesigned shell."""

        backend_key = "pyqtgraph"
        capabilities = RendererCapabilities()

        def __init__(self, selection_bus: SelectionBus | None = None, parent=None) -> None:
            super().__init__(parent)
            self.selection_bus = selection_bus
            self.buffer = HierarchicalSpectrumBuffer.from_counts(())
            self._current_traces: tuple[SpectrumTrace, ...] = ()
            self._peak_candidates_by_id: dict[str, PeakCandidate] = {}
            self._annotation_specs: tuple[ReferenceLine, ...] = ()
            self._reference_lines: list[object] = []
            self._annotation_label_items: list[object] = []
            self._cascade_sum_lines: list[object] = []
            self._overlay_traces: list[object] = []
            self._peak_scatter = None
            self._residual_visible = False
            self._log_scale = False
            self._peak_labels_visible = True
            self._x_axis_is_energy = False
            self._syncing_roi_region = False

            shell = QVBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.setSpacing(12)

            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)

            self.header_label = QLabel("Live spectrum canvas", self)
            self.header_label.setObjectName("CanvasHeader")
            header.addWidget(self.header_label)

            header.addStretch(1)

            self.status_label = QLabel("No spectrum loaded", self)
            self.status_label.setObjectName("CanvasMeta")
            header.addWidget(self.status_label)

            self.roi_button = QPushButton("Select ROI", self)
            self.roi_button.setObjectName("SpectrumSelectRoiButton")
            self.roi_button.setCheckable(True)
            self.roi_button.setToolTip(
                "Show draggable ROI boundaries. Drag either handle to set the analysis limits."
            )
            self.roi_button.toggled.connect(self._set_roi_visible)
            header.addWidget(self.roi_button)

            self.clear_roi_button = QPushButton("Clear ROI", self)
            self.clear_roi_button.setObjectName("SpectrumClearRoiButton")
            self.clear_roi_button.clicked.connect(self._clear_roi)
            header.addWidget(self.clear_roi_button)

            self.zoom_in_button = QPushButton("Zoom +", self)
            self.zoom_in_button.setObjectName("SpectrumZoomInButton")
            self.zoom_in_button.setToolTip("Zoom into the center of the current spectrum view.")
            self.zoom_in_button.clicked.connect(lambda: self._zoom_view(0.65))
            header.addWidget(self.zoom_in_button)

            self.zoom_out_button = QPushButton("Zoom −", self)
            self.zoom_out_button.setObjectName("SpectrumZoomOutButton")
            self.zoom_out_button.setToolTip(
                "Zoom out from the center of the current spectrum view."
            )
            self.zoom_out_button.clicked.connect(lambda: self._zoom_view(1.5))
            header.addWidget(self.zoom_out_button)

            self.reset_view_button = QPushButton("Reset View", self)
            self.reset_view_button.setObjectName("SpectrumResetViewButton")
            self.reset_view_button.setToolTip(
                "Fit the full spectrum. Use the mouse wheel to zoom and left-drag to pan."
            )
            self.reset_view_button.clicked.connect(self.reset_view)
            header.addWidget(self.reset_view_button)

            shell.addLayout(header)

            self.plot = pg.PlotWidget(self)
            self.plot.setObjectName("SpectrumPlot")
            self.plot.setBackground("#0f172a")
            self.plot.showGrid(x=True, y=True, alpha=0.14)
            self.plot.setMenuEnabled(False)
            self.plot.setMouseEnabled(x=True, y=True)
            self.plot.setLabel("bottom", "Channel")
            self.plot.setLabel("left", "Counts")
            self.plot_item = self.plot.getPlotItem()
            self.plot_item.setLogMode(x=False, y=False)
            self.trace_item = self.plot_item.plot(
                pen=pg.mkPen(color="#72d6ff", width=2),
                fillLevel=0,
                brush=(114, 214, 255, 70),
            )
            self._peak_scatter = pg.ScatterPlotItem(
                symbol="o",
                size=9,
                brush=pg.mkBrush("#f59e0b"),
                pen=pg.mkPen(color="#f59e0b", width=1.5),
            )
            self._peak_scatter.sigClicked.connect(self._peak_scatter_clicked)
            self.plot_item.addItem(self._peak_scatter)
            self._roi_region = pg.LinearRegionItem(
                values=(0.0, 1.0),
                orientation="vertical",
                movable=True,
                brush=pg.mkBrush(15, 118, 110, 48),
                pen=pg.mkPen(color="#2dd4bf", width=1.5),
                hoverPen=pg.mkPen(color="#f8fafc", width=2.0),
            )
            self._roi_region.setZValue(20)
            self._roi_region.setVisible(False)
            self._roi_region.sigRegionChangeFinished.connect(
                self._publish_roi_region
            )
            self.plot_item.addItem(self._roi_region)
            shell.addWidget(self.plot, 1)

            self.residual_row = QWidget(self)
            residual_layout = QHBoxLayout(self.residual_row)
            residual_layout.setContentsMargins(0, 0, 0, 0)
            residual_layout.setSpacing(8)
            self.residual_plots: list[pg.PlotWidget] = []
            self.residual_curves: list[object] = []
            self.residual_zero_lines: list[object] = []
            for index in range(3):
                residual_plot = pg.PlotWidget(self.residual_row)
                residual_plot.setObjectName(f"SpectrumResidualPlot{index + 1}")
                residual_plot.setBackground("#0f172a")
                residual_plot.showGrid(x=True, y=True, alpha=0.1)
                residual_plot.setMouseEnabled(x=False, y=False)
                residual_plot.setMenuEnabled(False)
                residual_plot.setLabel("left", "z")
                residual_plot.setLabel("bottom", "Ch")
                curve = residual_plot.plot(
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=pg.mkBrush("#72d6ff"),
                    symbolPen=pg.mkPen(color="#72d6ff", width=1.0),
                )
                zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False)
                residual_plot.addItem(zero)
                residual_layout.addWidget(residual_plot, 1)
                self.residual_plots.append(residual_plot)
                self.residual_curves.append(curve)
                self.residual_zero_lines.append(zero)
            self.residual_row.setVisible(False)
            shell.addWidget(self.residual_row)

            if self.selection_bus is not None:
                self.selection_bus.subscribe(self._on_selection_changed)

        def set_spectrum(self, counts: Sequence[float]) -> None:
            self.set_traces(
                (
                    SpectrumTrace(
                        label="Foreground",
                        counts=tuple(float(value) for value in counts),
                        channels=tuple(float(index) for index in range(len(counts))),
                    ),
                )
            )

        def set_traces(self, traces: Sequence[SpectrumTrace]) -> None:
            if not traces:
                self.clear()
                return

            self._current_traces = tuple(traces)
            self._x_axis_is_energy = traces[0].x_axis_label.lower().startswith(
                "energy"
            )
            self.plot.setLabel("bottom", traces[0].x_axis_label)
            for item in self._overlay_traces:
                self.plot_item.removeItem(item)
            self._overlay_traces.clear()

            primary = traces[0]
            values = tuple(float(value) for value in primary.counts)
            self.buffer = HierarchicalSpectrumBuffer.from_counts(values)
            level = self.buffer.choose_level(pixel_width=1400)
            if primary.channels:
                full_channels = tuple(float(value) for value in primary.channels)
            else:
                full_channels = tuple(float(index) for index in range(len(values)))
            channels = [
                full_channels[index * level.stride]
                for index in range(min(len(level.counts), len(full_channels[:: level.stride or 1])))
            ]
            if len(channels) < len(level.counts):
                channels = [index * level.stride for index in range(len(level.counts))]
            display_primary = [self._display_value(value) for value in level.counts]
            self.trace_item.setData(channels, display_primary)
            self.trace_item.setPen(pg.mkPen(color=primary.color, width=2))

            for overlay in traces[1:]:
                if not overlay.visible:
                    continue
                item = self.plot_item.plot(
                    tuple(float(value) for value in overlay.channels),
                    tuple(self._display_value(value) for value in overlay.counts),
                    pen=pg.mkPen(color=overlay.color, width=1.35),
                )
                self._overlay_traces.append(item)

            visible_labels = ", ".join(trace.label for trace in traces if trace.visible)
            self.status_label.setText(
                f"{len(values):,} channels · {len(self.buffer.levels)} LOD levels · {visible_labels}"
            )
            if self._annotation_specs:
                self.set_annotation_lines(self._annotation_specs)

        def set_reference_lines(self, energies_keV: Sequence[float]) -> None:
            self.set_annotation_lines(
                tuple(
                    ReferenceLine(
                        energy_keV=float(energy),
                        color="#f59e0b",
                    )
                    for energy in energies_keV
                )
            )

        def set_annotation_lines(self, lines: Sequence[ReferenceLine]) -> None:
            self._annotation_specs = tuple(lines)
            for line in self._reference_lines:
                self.plot_item.removeItem(line)
            self._reference_lines.clear()
            for label_item in self._annotation_label_items:
                self.plot_item.removeItem(label_item)
            self._annotation_label_items.clear()

            for marker in lines:
                line = pg.InfiniteLine(
                    pos=float(marker.energy_keV),
                    angle=90,
                    pen=pg.mkPen(
                        color=marker.color,
                        width=1,
                        style=pg.QtCore.Qt.DashLine,
                    ),
                )
                self.plot_item.addItem(line)
                self._reference_lines.append(line)
                if self._peak_labels_visible and marker.label:
                    label_item = pg.TextItem(
                        html=(
                            "<div style='font-size:10px; color:#e2e8f0; "
                            "background:rgba(15,23,42,0.72); padding:2px 4px; border-radius:4px;'>"
                            f"{marker.label}"
                            "</div>"
                        ),
                        anchor=(0.0, 1.0),
                    )
                    label_item.setPos(
                        float(marker.energy_keV),
                        float(self._annotation_label_height()),
                    )
                    self.plot_item.addItem(label_item)
                    self._annotation_label_items.append(label_item)

        def set_cascade_sum_lines(self, energies_keV: Sequence[float]) -> None:
            for line in self._cascade_sum_lines:
                self.plot_item.removeItem(line)
            self._cascade_sum_lines.clear()

            for energy in energies_keV:
                line = pg.InfiniteLine(
                    pos=float(energy),
                    angle=90,
                    pen=pg.mkPen(color="#d946ef", width=1, style=pg.QtCore.Qt.DotLine),
                )
                self.plot_item.addItem(line)
                self._cascade_sum_lines.append(line)

        def set_peak_candidates(self, peaks: Sequence[PeakCandidate]) -> None:
            if self._peak_scatter is None:
                return
            if not peaks:
                self._peak_candidates_by_id = {}
                self._peak_scatter.setData([], [])
                return
            self._peak_candidates_by_id = {
                str(peak.peak_id): peak for peak in peaks if peak.peak_id
            }
            x_values = [
                float(peak.energy_keV if self._x_axis_is_energy else peak.channel)
                for peak in peaks
            ]
            y_values = [
                self._display_value(
                    float(
                        self.buffer.full_resolution[
                            min(int(round(peak.channel)), len(self.buffer.full_resolution) - 1)
                        ]
                    )
                )
                if self.buffer.full_resolution
                else 0.0
                for peak in peaks
            ]
            brushes = [
                pg.mkBrush("#10b981" if peak.status == "matched" else "#f59e0b")
                for peak in peaks
            ]
            self._peak_scatter.setData(
                x=x_values,
                y=y_values,
                brush=brushes,
                data=[peak.peak_id for peak in peaks],
            )

        def _peak_scatter_clicked(self, _scatter, points) -> None:
            if self.selection_bus is None or not points:
                return
            peak_id = points[0].data()
            if peak_id is None:
                return
            peak = self._peak_candidates_by_id.get(str(peak_id))
            if peak is None:
                return
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=float(peak.energy_keV),
                    roi_bounds_keV=peak.roi_bounds_keV,
                    nuclide=peak.nuclide,
                    reference_lines_keV=peak.reference_lines_keV,
                )
            )

        def set_peak_residuals(self, peaks: Sequence[PeakCandidate], *, visible: bool) -> None:
            self._residual_visible = bool(visible)
            self.residual_row.setVisible(bool(visible and peaks))
            for index, curve in enumerate(self.residual_curves):
                if not visible or index >= len(peaks):
                    curve.setData([], [])
                    self.residual_plots[index].setTitle("Residuals")
                    continue
                peak = peaks[index]
                curve.setData(
                    np.asarray(peak.residual_channels, dtype=float),
                    np.asarray(peak.normalized_residuals, dtype=float),
                )
                severity = max((abs(value) for value in peak.normalized_residuals), default=0.0)
                if severity >= 3.0:
                    color = "#ef4444"
                elif severity >= 2.0:
                    color = "#f59e0b"
                else:
                    color = "#10b981"
                curve.setSymbolBrush(pg.mkBrush(color))
                curve.setSymbolPen(pg.mkPen(color=color, width=1.0))
                self.residual_zero_lines[index].setPen(pg.mkPen(color="#94a3b8", width=1))
                self.residual_plots[index].setTitle(
                    f"{peak.energy_keV:.1f} keV · max |z| {severity:.2f}"
                )

        def clear(self) -> None:
            self.buffer = HierarchicalSpectrumBuffer.from_counts(())
            self._current_traces = ()
            self._annotation_specs = ()
            self.trace_item.setData([], [])
            self.set_annotation_lines(())
            self.set_cascade_sum_lines(())
            self.set_peak_candidates(())
            self.set_peak_residuals((), visible=False)
            self.status_label.setText("No spectrum loaded")
            self._roi_region.setVisible(False)
            self.roi_button.setChecked(False)

        def reset_view(self) -> None:
            """Fit all visible traces while retaining mouse pan and wheel zoom."""

            self.plot_item.enableAutoRange(x=True, y=True)
            self.plot_item.autoRange()

        def _zoom_view(self, factor: float) -> None:
            """Apply a centered zoom while keeping free mouse pan/zoom enabled."""

            self.plot_item.disableAutoRange()
            self.plot_item.getViewBox().scaleBy((float(factor), float(factor)))

        def _set_roi_visible(self, visible: bool) -> None:
            bounds = (
                self.selection_bus.state.roi_bounds_keV
                if visible and self.selection_bus is not None
                else None
            )
            if visible and bounds is None:
                x_range = self.plot_item.viewRange()[0]
                width = max(float(x_range[1] - x_range[0]), 1.0)
                bounds = (
                    float(x_range[0] + 0.4 * width),
                    float(x_range[0] + 0.6 * width),
                )
            if bounds is not None:
                self._syncing_roi_region = True
                self._roi_region.setRegion(bounds)
                self._syncing_roi_region = False
            self._roi_region.setVisible(bool(visible))

        def _publish_roi_region(self) -> None:
            if self._syncing_roi_region or self.selection_bus is None:
                return
            lower, upper = self._roi_region.getRegion()
            self.selection_bus.publish_roi(float(lower), float(upper))

        def _clear_roi(self) -> None:
            self.roi_button.setChecked(False)
            if self.selection_bus is None:
                return
            state = self.selection_bus.state
            self.selection_bus.publish(
                SelectionState(
                    peak_energy_keV=state.peak_energy_keV,
                    roi_bounds_keV=None,
                    nuclide=state.nuclide,
                    reference_lines_keV=state.reference_lines_keV,
                    annotation_lines=state.annotation_lines,
                )
            )

        def set_log_scale(self, enabled: bool) -> None:
            self._log_scale = bool(enabled)
            self.plot_item.setLogMode(x=False, y=self._log_scale)
            if self._current_traces:
                self.set_traces(self._current_traces)

        def set_peak_labels_visible(self, visible: bool) -> None:
            self._peak_labels_visible = bool(visible)
            self.set_annotation_lines(self._annotation_specs)

        def _on_selection_changed(self, state: SelectionState) -> None:
            if state.roi_bounds_keV is not None:
                self._syncing_roi_region = True
                self._roi_region.setRegion(state.roi_bounds_keV)
                self._syncing_roi_region = False
                self._roi_region.setVisible(True)
                self.roi_button.blockSignals(True)
                self.roi_button.setChecked(True)
                self.roi_button.blockSignals(False)
            else:
                self._roi_region.setVisible(False)
                self.roi_button.blockSignals(True)
                self.roi_button.setChecked(False)
                self.roi_button.blockSignals(False)
            if state.reference_lines_keV or state.annotation_lines:
                annotations = list(state.annotation_lines)
                if not annotations:
                    annotations = [
                        ReferenceLine(energy_keV=float(energy))
                        for energy in state.reference_lines_keV
                    ]
                self.set_annotation_lines(tuple(annotations))
            else:
                self.set_annotation_lines(())
            fragments = []
            if state.peak_energy_keV is not None:
                fragments.append(f"{state.peak_energy_keV:.3f} keV")
            if state.nuclide:
                fragments.append(state.nuclide)
            if state.roi_bounds_keV:
                fragments.append(
                    f"ROI {state.roi_bounds_keV[0]:.1f}-{state.roi_bounds_keV[1]:.1f} keV"
                )
            if state.annotation_lines:
                fragments.append(f"{len(state.annotation_lines)} guides")
            if fragments:
                self.header_label.setText("Selection: " + " | ".join(fragments))
            else:
                self.header_label.setText("Live spectrum canvas")

        def _display_value(self, value: float) -> float:
            if not self._log_scale:
                return float(value)
            return max(float(value), 1.0e-3)

        def _annotation_label_height(self) -> float:
            if not self._current_traces:
                return 1.0
            max_value = 0.0
            for trace in self._current_traces:
                if not trace.visible or not trace.counts:
                    continue
                max_value = max(max_value, max(float(value) for value in trace.counts))
            if self._log_scale:
                return max(max_value, 1.0e-3)
            return max_value if max_value > 0.0 else 1.0


else:

    class PyQtGraphSpectrumCanvas(SpectrumCanvas):  # pragma: no cover - placeholder without optional deps
        """Import-safe placeholder when Qt or PyQtGraph is unavailable."""

        backend_key = "pyqtgraph"
        capabilities = RendererCapabilities()

        def __init__(self, selection_bus: SelectionBus | None = None, parent=None) -> None:
            self.selection_bus = selection_bus
            self.parent = parent

        def set_spectrum(self, counts: Sequence[float]) -> None:
            del counts
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )

        def set_reference_lines(self, energies_keV: Sequence[float]) -> None:
            del energies_keV
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )

        def set_annotation_lines(self, lines: Sequence[ReferenceLine]) -> None:
            del lines
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )

        def clear(self) -> None:
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )
