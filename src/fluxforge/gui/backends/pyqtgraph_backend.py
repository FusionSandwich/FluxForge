"""PyQtGraph-backed spectrum canvas implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import uuid4

import numpy as np

from fluxforge.core.analysis_workspace import PeakCandidate
from fluxforge.core.workspace_document import CanvasViewport
from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind
from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import (
    CanvasPeakOverlay,
    CanvasROIOverlay,
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
        from PySide6.QtCore import QEvent, QTimer

        from fluxforge.gui.qt_compat import (
            QAction,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QMenu,
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

    @dataclass
    class _ROIItems:
        overlay: CanvasROIOverlay
        signal: object
        left_background: object
        right_background: object

    def catalog_pyqtgraph_export_action(plot_widget, object_name: str) -> None:
        """Give PyQtGraph's production context-menu export action a stable ID."""

        for action in plot_widget.scene().findChildren(QAction):
            if action.text() == "Export...":
                action.setObjectName(f"Export{object_name}Action")

    class PyQtGraphSpectrumCanvas(QWidget, SpectrumCanvas):
        """Production-ready Qt spectrum canvas for the redesigned shell."""

        backend_key = "pyqtgraph"
        capabilities = RendererCapabilities()

        def __init__(
            self, selection_bus: SelectionBus | None = None, parent=None
        ) -> None:
            super().__init__(parent)
            self.selection_bus = selection_bus
            self.buffer = HierarchicalSpectrumBuffer.from_counts(())
            self._current_traces: tuple[SpectrumTrace, ...] = ()
            self._peak_candidates_by_id: dict[str, PeakCandidate] = {}
            self._peak_overlays_by_id: dict[str, CanvasPeakOverlay] = {}
            self._roi_items_by_id: dict[str, _ROIItems] = {}
            self._interaction_spectrum_id: str | None = None
            self._selected_roi_id: str | None = None
            self._selected_peak_id: str | None = None
            self._selected_peak_line = None
            self._syncing_peak_line = False
            self._syncing_analysis_overlays = False
            self._shift_drag_active = False
            self._shift_drag_start: float | None = None
            self._shift_drag_preview = None
            self._context_position = 0.0
            self._context_peak_id: str | None = None
            self._context_roi_id: str | None = None
            self._annotation_specs: tuple[ReferenceLine, ...] = ()
            self._reference_lines: list[object] = []
            self._annotation_label_items: list[object] = []
            self._cascade_sum_lines: list[object] = []
            self._overlay_traces: list[object] = []
            self._peak_scatter = None
            self._residual_peaks: tuple[PeakCandidate, ...] = ()
            self._residual_visible = False
            self._residual_mode = "off"
            self._crosshair_enabled = False
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

            self.crosshair_readout = QLabel("x -- | y --", self)
            self.crosshair_readout.setObjectName("SpectrumCrosshairReadout")
            self.crosshair_readout.setVisible(False)
            header.addWidget(self.crosshair_readout)

            self.crosshair_button = QPushButton("Crosshair", self)
            self.crosshair_button.setObjectName("SpectrumCrosshairButton")
            self.crosshair_button.setCheckable(True)
            self.crosshair_button.setToolTip(
                "Show exact spectrum coordinates under the mouse pointer."
            )
            self.crosshair_button.toggled.connect(self._crosshair_button_toggled)
            header.addWidget(self.crosshair_button)

            self.roi_button = QPushButton("Select ROI", self)
            self.roi_button.setObjectName("SpectrumSelectRoiButton")
            self.roi_button.setCheckable(True)
            self.roi_button.setToolTip(
                "Show temporary analysis bounds. Shift-drag the plot to create a "
                "persisted ROI with background sidebands."
            )
            self.roi_button.toggled.connect(self._set_roi_visible)
            header.addWidget(self.roi_button)

            self.clear_roi_button = QPushButton("Clear ROI", self)
            self.clear_roi_button.setObjectName("SpectrumClearRoiButton")
            self.clear_roi_button.clicked.connect(self._clear_roi)
            header.addWidget(self.clear_roi_button)

            self.zoom_in_button = QPushButton("Zoom +", self)
            self.zoom_in_button.setObjectName("SpectrumZoomInButton")
            self.zoom_in_button.setToolTip(
                "Zoom into the center of the current spectrum view."
            )
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
                "Fit the full spectrum. Use the mouse wheel to zoom and "
                "left-drag to pan."
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
            self._viewport_commit_timer = QTimer(self)
            self._viewport_commit_timer.setSingleShot(True)
            self._viewport_commit_timer.setInterval(120)
            self._viewport_commit_timer.timeout.connect(self._commit_viewport_intent)
            self.plot_item.getViewBox().sigRangeChangedManually.connect(
                lambda _mask: self._schedule_viewport_commit()
            )
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
            self._selected_peak_line = pg.InfiniteLine(
                angle=90,
                movable=True,
                pen=pg.mkPen(color="#f8fafc", width=2.5),
                hoverPen=pg.mkPen(color="#38bdf8", width=3.0),
                label="selected peak",
                labelOpts={"color": "#f8fafc", "position": 0.92},
            )
            self._selected_peak_line.setZValue(28)
            self._selected_peak_line.setVisible(False)
            self._selected_peak_line.sigPositionChangeFinished.connect(
                self._selected_peak_move_finished
            )
            self.plot_item.addItem(self._selected_peak_line, ignoreBounds=True)
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
            self._roi_region.sigRegionChangeFinished.connect(self._publish_roi_region)
            self.plot_item.addItem(self._roi_region)

            self._crosshair_vertical = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(color="#94a3b8", width=1, style=pg.QtCore.Qt.DashLine),
            )
            self._crosshair_horizontal = pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(color="#94a3b8", width=1, style=pg.QtCore.Qt.DashLine),
            )
            for line in (self._crosshair_vertical, self._crosshair_horizontal):
                line.setZValue(30)
                line.setVisible(False)
                self.plot_item.addItem(line, ignoreBounds=True)
            self.plot.scene().sigMouseMoved.connect(self._move_crosshair)
            self.plot.viewport().setMouseTracking(True)
            self.plot.viewport().installEventFilter(self)
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
                catalog_pyqtgraph_export_action(
                    residual_plot,
                    f"SpectrumResidualPlot{index + 1}",
                )
            catalog_pyqtgraph_export_action(self.plot, "SpectrumPlot")
            self._build_context_menu()
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
            self._x_axis_is_energy = traces[0].x_axis_label.lower().startswith("energy")
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
                for index in range(
                    min(len(level.counts), len(full_channels[:: level.stride or 1]))
                )
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
                f"{len(values):,} channels · {len(self.buffer.levels)} LOD levels · "
                f"{visible_labels}"
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
                            "background:rgba(15,23,42,0.72); padding:2px 4px; "
                            "border-radius:4px;'>"
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
                self._peak_overlays_by_id = {}
                self._render_peak_overlays(())
                return
            self._peak_candidates_by_id = {
                str(peak.peak_id): peak for peak in peaks if peak.peak_id
            }
            overlays = tuple(
                CanvasPeakOverlay(
                    peak_id=str(peak.peak_id),
                    spectrum_id=self._interaction_spectrum_id or "legacy-spectrum",
                    position=float(
                        peak.energy_keV if self._x_axis_is_energy else peak.channel
                    ),
                    y_value=(
                        self._display_value(
                            float(
                                self.buffer.full_resolution[
                                    min(
                                        max(int(round(peak.channel)), 0),
                                        len(self.buffer.full_resolution) - 1,
                                    )
                                ]
                            )
                        )
                        if self.buffer.full_resolution
                        else 0.0
                    ),
                    nuclide=peak.nuclide,
                    tags=tuple(peak.tags),
                    selected=peak.peak_id == self._selected_peak_id,
                )
                for peak in peaks
            )
            self._peak_overlays_by_id = {item.peak_id: item for item in overlays}
            self._render_peak_overlays(overlays)

        def set_interaction_context(self, spectrum_id: str | None) -> None:
            self._interaction_spectrum_id = str(spectrum_id) if spectrum_id else None

        def set_analysis_overlays(
            self,
            rois: Sequence[CanvasROIOverlay],
            peaks: Sequence[CanvasPeakOverlay],
            *,
            selected_roi_id: str | None = None,
            selected_peak_id: str | None = None,
        ) -> None:
            """Render exact canonical leaves without modifying the document."""

            self._syncing_analysis_overlays = True
            try:
                self._selected_roi_id = selected_roi_id
                self._selected_peak_id = selected_peak_id
                if self._interaction_spectrum_id is None:
                    identities = {
                        item.spectrum_id for item in tuple(rois) + tuple(peaks)
                    }
                    if len(identities) == 1:
                        self._interaction_spectrum_id = next(iter(identities))
                self._replace_roi_items(tuple(rois))
                normalized_peaks = tuple(
                    CanvasPeakOverlay(
                        peak_id=item.peak_id,
                        spectrum_id=item.spectrum_id,
                        position=item.position,
                        y_value=item.y_value,
                        component_ids=item.component_ids,
                        nuclide=item.nuclide,
                        tags=item.tags,
                        pinned=item.pinned,
                        selected=(
                            item.peak_id == selected_peak_id
                            if selected_peak_id is not None
                            else item.selected
                        ),
                    )
                    for item in peaks
                )
                self._peak_overlays_by_id = {
                    item.peak_id: item for item in normalized_peaks
                }
                self._render_peak_overlays(normalized_peaks)
            finally:
                self._syncing_analysis_overlays = False

        def _render_peak_overlays(self, peaks: Sequence[CanvasPeakOverlay]) -> None:
            self._peak_overlays_by_id = {item.peak_id: item for item in peaks}
            if not peaks:
                self._peak_scatter.setData([], [])
                self._selected_peak_line.setVisible(False)
                return
            self._peak_scatter.setData(
                x=[float(item.position) for item in peaks],
                y=[self._peak_y_at_position(item) for item in peaks],
                size=[13 if item.selected else 9 for item in peaks],
                brush=[
                    pg.mkBrush("#38bdf8" if item.selected else "#f59e0b")
                    for item in peaks
                ],
                pen=[
                    pg.mkPen(
                        color="#f8fafc" if item.selected else "#f59e0b",
                        width=2.0 if item.selected else 1.25,
                    )
                    for item in peaks
                ],
                data=[item.peak_id for item in peaks],
            )
            selected = next((item for item in peaks if item.selected), None)
            self._syncing_peak_line = True
            try:
                if selected is None:
                    self._selected_peak_line.setVisible(False)
                else:
                    self._selected_peak_id = selected.peak_id
                    self._selected_peak_line.setPos(float(selected.position))
                    self._selected_peak_line.setVisible(True)
            finally:
                self._syncing_peak_line = False

        def _peak_y_at_position(self, overlay: CanvasPeakOverlay) -> float:
            if self._current_traces:
                primary = self._current_traces[0]
                if primary.channels and primary.counts:
                    channels = np.asarray(primary.channels, dtype=float)
                    index = int(np.argmin(np.abs(channels - float(overlay.position))))
                    if 0 <= index < len(primary.counts):
                        return self._display_value(float(primary.counts[index]))
            return self._display_value(float(overlay.y_value))

        def _replace_roi_items(self, rois: Sequence[CanvasROIOverlay]) -> None:
            for bundle in self._roi_items_by_id.values():
                for item in (
                    bundle.signal,
                    bundle.left_background,
                    bundle.right_background,
                ):
                    self.plot_item.removeItem(item)
            self._roi_items_by_id = {}
            for overlay in rois:
                selected = (
                    overlay.roi_id == self._selected_roi_id
                    if self._selected_roi_id is not None
                    else overlay.selected
                )
                signal = self._analysis_region_item(
                    overlay,
                    "signal",
                    overlay.signal_range,
                    selected=selected,
                )
                left = self._analysis_region_item(
                    overlay,
                    "left",
                    overlay.left_background_range,
                    selected=selected,
                )
                right = self._analysis_region_item(
                    overlay,
                    "right",
                    overlay.right_background_range,
                    selected=selected,
                )
                self._roi_items_by_id[overlay.roi_id] = _ROIItems(
                    overlay=overlay,
                    signal=signal,
                    left_background=left,
                    right_background=right,
                )
                for item in (left, right, signal):
                    self.plot_item.addItem(item)

        def _analysis_region_item(
            self,
            overlay: CanvasROIOverlay,
            part: str,
            bounds: tuple[float, float],
            *,
            selected: bool,
        ):
            signal = part == "signal"
            color = overlay.color if signal else "#64748b"
            alpha = 62 if selected else 34
            brush_color = pg.mkColor(color)
            brush_color.setAlpha(alpha)
            item = pg.LinearRegionItem(
                values=tuple(float(value) for value in bounds),
                orientation="vertical",
                movable=True,
                brush=pg.mkBrush(brush_color),
                pen=pg.mkPen(
                    color="#f8fafc" if selected else color,
                    width=2.2 if selected else 1.2,
                ),
                hoverPen=pg.mkPen(color="#38bdf8", width=2.5),
            )
            item.setZValue(24 if signal else 22)
            item.sigRegionChangeFinished.connect(
                lambda _item=item, roi_id=overlay.roi_id, region_part=part: (
                    self._analysis_region_move_finished(roi_id, region_part)
                )
            )
            return item

        def _peak_scatter_clicked(self, _scatter, points) -> None:
            if not points:
                return
            peak_id = points[0].data()
            if peak_id is None:
                return
            peak_id = str(peak_id)
            overlay = self._peak_overlays_by_id.get(peak_id)
            if overlay is not None:
                self._selected_peak_id = peak_id
                self._emit_for_spectrum(
                    CanvasIntentKind.SELECT_PEAK,
                    spectrum_id=overlay.spectrum_id,
                    peak_id=peak_id,
                )
                self._render_peak_overlays(
                    tuple(
                        CanvasPeakOverlay(
                            peak_id=item.peak_id,
                            spectrum_id=item.spectrum_id,
                            position=item.position,
                            y_value=item.y_value,
                            component_ids=item.component_ids,
                            nuclide=item.nuclide,
                            tags=item.tags,
                            pinned=item.pinned,
                            selected=item.peak_id == peak_id,
                        )
                        for item in self._peak_overlays_by_id.values()
                    )
                )
            if self.selection_bus is None:
                return
            peak = self._peak_candidates_by_id.get(peak_id)
            if peak is None:
                return
            self.selection_bus.publish(
                SelectionState(
                    spectrum_id=(
                        overlay.spectrum_id
                        if overlay is not None
                        else self._interaction_spectrum_id
                    ),
                    peak_id=peak_id,
                    roi_id=self.selection_bus.state.roi_id,
                    peak_energy_keV=float(peak.energy_keV),
                    roi_bounds_keV=peak.roi_bounds_keV,
                    nuclide=peak.nuclide,
                    reference_lines_keV=peak.reference_lines_keV,
                )
            )

        def _selected_peak_move_finished(self) -> None:
            if self._syncing_peak_line or self._selected_peak_id is None:
                return
            overlay = self._peak_overlays_by_id.get(self._selected_peak_id)
            if overlay is None:
                return
            self._emit_for_spectrum(
                CanvasIntentKind.MOVE_PEAK,
                spectrum_id=overlay.spectrum_id,
                peak_id=overlay.peak_id,
                position=float(self._selected_peak_line.value()),
                drag_token=f"peak-{uuid4().hex}",
            )

        def _analysis_region_move_finished(self, roi_id: str, part: str) -> None:
            if self._syncing_analysis_overlays:
                return
            bundle = self._roi_items_by_id.get(roi_id)
            if bundle is None:
                return
            signal = self._ordered_region(bundle.signal.getRegion())
            left = self._ordered_region(bundle.left_background.getRegion())
            right = self._ordered_region(bundle.right_background.getRegion())
            if part == "signal":
                self._emit_for_spectrum(
                    CanvasIntentKind.MOVE_ROI,
                    spectrum_id=bundle.overlay.spectrum_id,
                    roi_id=roi_id,
                    bounds=signal,
                    left_background=left,
                    right_background=right,
                    drag_token=f"roi-{uuid4().hex}",
                )
            else:
                self._emit_for_spectrum(
                    CanvasIntentKind.MOVE_BACKGROUND,
                    spectrum_id=bundle.overlay.spectrum_id,
                    roi_id=roi_id,
                    left_background=left,
                    right_background=right,
                    drag_token=f"roi-background-{uuid4().hex}",
                )

        @staticmethod
        def _ordered_region(values) -> tuple[float, float]:
            lower, upper = sorted((float(values[0]), float(values[1])))
            return lower, upper

        def _emit_for_spectrum(
            self,
            kind: CanvasIntentKind,
            *,
            spectrum_id: str | None = None,
            **values,
        ) -> None:
            target = spectrum_id or self._interaction_spectrum_id
            if target is None and kind not in {
                CanvasIntentKind.PIN_NUCLIDE,
                CanvasIntentKind.UNPIN_NUCLIDE,
                CanvasIntentKind.TAG_NUCLIDE,
            }:
                return
            self._emit_canvas_intent(
                CanvasIntent(kind=kind, spectrum_id=target, **values)
            )

        def eventFilter(self, watched, event):  # noqa: N802 - Qt API
            if watched is self.plot.viewport():
                event_type = event.type()
                if event_type == QEvent.MouseButtonPress:
                    if (
                        event.button() == pg.QtCore.Qt.LeftButton
                        and event.modifiers() & pg.QtCore.Qt.ShiftModifier
                        and self._interaction_spectrum_id is not None
                    ):
                        self._begin_shift_roi_drag(self._event_data_x(event))
                        event.accept()
                        return True
                elif event_type == QEvent.MouseMove:
                    if self._shift_drag_active:
                        self._update_shift_roi_drag(self._event_data_x(event))
                        event.accept()
                        return True
                    if self._crosshair_enabled:
                        self._move_crosshair(self._event_scene_position(event))
                elif event_type == QEvent.MouseButtonRelease:
                    if self._shift_drag_active:
                        self._finish_shift_roi_drag(self._event_data_x(event))
                        event.accept()
                        return True
                    if event.button() == pg.QtCore.Qt.RightButton:
                        self._prepare_context_menu(self._event_data_x(event))
                        position = (
                            event.position().toPoint()
                            if hasattr(event, "position")
                            else event.pos()
                        )
                        self.context_menu.popup(
                            self.plot.viewport().mapToGlobal(position)
                        )
                        event.accept()
                        return True
                elif event_type == QEvent.ContextMenu:
                    x_position = self._event_data_x(event)
                    self._prepare_context_menu(x_position)
                    self.context_menu.popup(event.globalPos())
                    event.accept()
                    return True
            return super().eventFilter(watched, event)

        def _event_data_x(self, event) -> float:
            scene_position = self._event_scene_position(event)
            data_position = self.plot_item.getViewBox().mapSceneToView(scene_position)
            return float(data_position.x())

        def _event_scene_position(self, event):
            position = (
                event.position().toPoint()
                if hasattr(event, "position")
                else event.pos()
            )
            return self.plot.mapToScene(position)

        def _begin_shift_roi_drag(self, x_position: float) -> None:
            self._shift_drag_active = True
            self._shift_drag_start = float(x_position)
            self._shift_drag_preview = pg.LinearRegionItem(
                values=(float(x_position), float(x_position)),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(45, 212, 191, 72),
                pen=pg.mkPen(color="#5eead4", width=2.0),
            )
            self._shift_drag_preview.setZValue(40)
            self.plot_item.addItem(self._shift_drag_preview)

        def _update_shift_roi_drag(self, x_position: float) -> None:
            if self._shift_drag_start is None or self._shift_drag_preview is None:
                return
            self._shift_drag_preview.setRegion(
                sorted((float(self._shift_drag_start), float(x_position)))
            )

        def _finish_shift_roi_drag(self, x_position: float) -> None:
            start = self._shift_drag_start
            preview = self._shift_drag_preview
            self._shift_drag_active = False
            self._shift_drag_start = None
            self._shift_drag_preview = None
            if preview is not None:
                self.plot_item.removeItem(preview)
            if start is None:
                return
            lower, upper = sorted((float(start), float(x_position)))
            minimum_width = max(
                abs(
                    float(self.plot_item.viewRange()[0][1])
                    - float(self.plot_item.viewRange()[0][0])
                )
                / max(float(self.plot.viewport().width()), 1.0),
                1.0e-9,
            )
            if upper - lower < minimum_width:
                return
            sideband_width = upper - lower
            self._emit_for_spectrum(
                CanvasIntentKind.CREATE_ROI,
                bounds=(lower, upper),
                left_background=(lower - sideband_width, lower),
                right_background=(upper, upper + sideband_width),
                drag_token=f"roi-create-{uuid4().hex}",
            )

        def _build_context_menu(self) -> None:
            self.context_menu = QMenu("Spectrum tools", self)
            self.context_menu.setObjectName("SpectrumContextMenu")
            self.context_menu.menuAction().setObjectName(
                "OpenSpectrumContextMenuAction"
            )
            self.add_peak_action = self._menu_action(
                "Add Peak Here", "CanvasContextAddPeakAction", self._context_add_peak
            )
            self.select_peak_action = self._menu_action(
                "Select Peak",
                "CanvasContextSelectPeakAction",
                self._context_select_peak,
            )
            self.move_peak_action = self._menu_action(
                "Move Peak Here", "CanvasContextMovePeakAction", self._context_move_peak
            )
            self.delete_peak_action = self._menu_action(
                "Delete Peak",
                "CanvasContextDeletePeakAction",
                self._context_delete_peak,
            )
            self.context_menu.addSeparator()
            self.assign_nuclide_action = self._menu_action(
                "Assign Nuclide...",
                "CanvasContextAssignNuclideAction",
                self._context_assign_nuclide,
            )
            self.clear_nuclide_action = self._menu_action(
                "Clear Nuclide",
                "CanvasContextClearNuclideAction",
                self._context_clear_nuclide,
            )
            self.pin_nuclide_action = self._menu_action(
                "Pin Nuclide",
                "CanvasContextPinNuclideAction",
                self._context_pin_nuclide,
            )
            self.unpin_nuclide_action = self._menu_action(
                "Unpin Nuclide",
                "CanvasContextUnpinNuclideAction",
                self._context_unpin_nuclide,
            )
            self.tag_peak_action = self._menu_action(
                "Tag Peak...", "CanvasContextTagPeakAction", self._context_tag_peak
            )
            self.tag_nuclide_action = self._menu_action(
                "Tag Nuclide...",
                "CanvasContextTagNuclideAction",
                self._context_tag_nuclide,
            )
            self.context_menu.addSeparator()
            self.add_component_action = self._menu_action(
                "Add Overlap Component Here",
                "CanvasContextAddComponentAction",
                self._context_add_component,
            )
            self.split_peak_action = self._menu_action(
                "Split Peak Here",
                "CanvasContextSplitPeakAction",
                self._context_split_peak,
            )
            self.merge_peaks_action = self._menu_action(
                "Merge Components",
                "CanvasContextMergePeaksAction",
                self._context_merge_components,
            )
            self.context_menu.addSeparator()
            self.select_roi_action = self._menu_action(
                "Select ROI", "CanvasContextSelectRoiAction", self._context_select_roi
            )
            self.delete_roi_action = self._menu_action(
                "Delete ROI", "CanvasContextDeleteRoiAction", self._context_delete_roi
            )
            self.create_roi_action = self._menu_action(
                "Create ROI Around Cursor",
                "CanvasContextCreateRoiAction",
                self._context_create_roi,
            )
            self.context_menu.addSeparator()
            roles_menu = self.context_menu.addMenu("Assign Spectrum Role")
            roles_menu.setObjectName("CanvasContextSpectrumRoleMenu")
            roles_menu.menuAction().setObjectName(
                "OpenCanvasContextSpectrumRoleMenuAction"
            )
            for label, role, object_name in (
                ("Foreground", "foreground", "CanvasContextForegroundRoleAction"),
                ("Background", "background", "CanvasContextBackgroundRoleAction"),
                (
                    "Secondary",
                    "secondary",
                    "CanvasContextSecondaryRoleAction",
                ),
            ):
                action = QAction(label, roles_menu)
                action.setObjectName(object_name)
                action.triggered.connect(
                    lambda _checked=False, value=role: self._context_assign_role(value)
                )
                roles_menu.addAction(action)
            self.context_menu.addSeparator()
            self.reset_context_action = self._menu_action(
                "Reset View", "CanvasContextResetViewAction", self.reset_view
            )
            self.crosshair_action = self._menu_action(
                "Crosshair",
                "CanvasContextCrosshairAction",
                self._context_crosshair_toggled,
                checkable=True,
            )
            export_action = next(
                (
                    action
                    for action in self.plot.scene().findChildren(QAction)
                    if action.objectName() == "ExportSpectrumPlotAction"
                ),
                None,
            )
            if export_action is not None:
                self.context_menu.addAction(export_action)

        def _menu_action(
            self,
            text: str,
            object_name: str,
            callback,
            *,
            checkable: bool = False,
        ) -> QAction:
            action = QAction(text, self.context_menu)
            action.setObjectName(object_name)
            action.setCheckable(checkable)
            if checkable:
                action.toggled.connect(callback)
            else:
                action.triggered.connect(callback)
            self.context_menu.addAction(action)
            return action

        def _prepare_context_menu(self, x_position: float) -> None:
            self._context_position = float(x_position)
            # PyQtGraph's export action normally receives this target from its
            # own scene menu. FluxForge hosts the action in a stable-ID menu,
            # so provide the same target explicitly before it can be invoked.
            self.plot.scene().contextMenuItem = self.plot_item
            x_range = self.plot_item.viewRange()[0]
            tolerance = (
                abs(float(x_range[1]) - float(x_range[0]))
                * 14.0
                / max(float(self.plot.viewport().width()), 1.0)
            )
            selected_peak = self._peak_overlays_by_id.get(
                str(self._selected_peak_id or "")
            )
            nearest_peak = min(
                self._peak_overlays_by_id.values(),
                key=lambda item: abs(float(item.position) - self._context_position),
                default=None,
            )
            if selected_peak is not None and (
                abs(float(selected_peak.position) - self._context_position) <= tolerance
            ):
                self._context_peak_id = selected_peak.peak_id
            elif nearest_peak is not None and (
                abs(float(nearest_peak.position) - self._context_position) <= tolerance
            ):
                self._context_peak_id = nearest_peak.peak_id
            else:
                self._context_peak_id = self._selected_peak_id
            containing_rois = [
                bundle.overlay
                for bundle in self._roi_items_by_id.values()
                if any(
                    bounds[0] <= self._context_position <= bounds[1]
                    for bounds in (
                        bundle.overlay.signal_range,
                        bundle.overlay.left_background_range,
                        bundle.overlay.right_background_range,
                    )
                )
            ]
            selected_roi = self._context_roi_overlay_for_id(self._selected_roi_id)
            selected_contains_cursor = selected_roi is not None and any(
                bounds[0] <= self._context_position <= bounds[1]
                for bounds in (
                    selected_roi.signal_range,
                    selected_roi.left_background_range,
                    selected_roi.right_background_range,
                )
            )
            self._context_roi_id = (
                selected_roi.roi_id
                if selected_contains_cursor
                else (containing_rois[0].roi_id if containing_rois else None)
            )
            peak = self._context_peak_overlay()
            has_peak = peak is not None
            for action in (
                self.select_peak_action,
                self.move_peak_action,
                self.delete_peak_action,
                self.assign_nuclide_action,
                self.clear_nuclide_action,
                self.tag_peak_action,
                self.add_component_action,
                self.split_peak_action,
            ):
                action.setEnabled(has_peak)
            self.pin_nuclide_action.setEnabled(
                bool(peak and peak.nuclide and not peak.pinned)
            )
            self.unpin_nuclide_action.setEnabled(
                bool(peak and peak.nuclide and peak.pinned)
            )
            self.tag_nuclide_action.setEnabled(bool(peak and peak.nuclide))
            self.tag_nuclide_action.setText(
                "Tag Nuclide..."
                if not peak or not peak.nuclide_tags
                else "Tag Nuclide... [" + ", ".join(peak.nuclide_tags) + "]"
            )
            self.merge_peaks_action.setEnabled(
                bool(peak and len(peak.component_ids) >= 2)
            )
            self.select_roi_action.setEnabled(self._context_roi_id is not None)
            self.delete_roi_action.setEnabled(self._context_roi_id is not None)

        def _context_peak_overlay(self) -> CanvasPeakOverlay | None:
            if self._context_peak_id is None:
                return None
            return self._peak_overlays_by_id.get(self._context_peak_id)

        def _context_roi_overlay(self) -> CanvasROIOverlay | None:
            return self._context_roi_overlay_for_id(self._context_roi_id)

        def _context_roi_overlay_for_id(
            self, roi_id: str | None
        ) -> CanvasROIOverlay | None:
            if roi_id is None:
                return None
            bundle = self._roi_items_by_id.get(roi_id)
            return bundle.overlay if bundle is not None else None

        def _context_add_peak(self) -> None:
            self._emit_for_spectrum(
                CanvasIntentKind.ADD_PEAK, position=self._context_position
            )

        def _context_select_peak(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.SELECT_PEAK,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                )

        def _context_move_peak(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.MOVE_PEAK,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    position=self._context_position,
                    drag_token=f"peak-menu-{uuid4().hex}",
                )

        def _context_delete_peak(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.DELETE_PEAK,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                )

        def _context_assign_nuclide(self) -> None:
            peak = self._context_peak_overlay()
            if peak is None:
                return
            nuclide, accepted = QInputDialog.getText(self, "Assign nuclide", "Nuclide:")
            if accepted and str(nuclide).strip():
                self._emit_for_spectrum(
                    CanvasIntentKind.ASSIGN_NUCLIDE,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    nuclide=str(nuclide).strip(),
                )

        def _context_clear_nuclide(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.CLEAR_NUCLIDE,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                )

        def _context_pin_nuclide(self) -> None:
            self._context_pin_state(True)

        def _context_unpin_nuclide(self) -> None:
            self._context_pin_state(False)

        def _context_pin_state(self, pinned: bool) -> None:
            peak = self._context_peak_overlay()
            if peak is not None and peak.nuclide:
                self._emit_for_spectrum(
                    (
                        CanvasIntentKind.PIN_NUCLIDE
                        if pinned
                        else CanvasIntentKind.UNPIN_NUCLIDE
                    ),
                    spectrum_id=peak.spectrum_id,
                    nuclide=peak.nuclide,
                )

        def _context_tag_peak(self) -> None:
            peak = self._context_peak_overlay()
            if peak is None:
                return
            tag, accepted = QInputDialog.getText(self, "Tag peak", "Tag:")
            if accepted and str(tag).strip():
                self._emit_for_spectrum(
                    CanvasIntentKind.TAG_PEAK,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    tag=str(tag).strip(),
                )

        def _context_tag_nuclide(self) -> None:
            peak = self._context_peak_overlay()
            if peak is None or not peak.nuclide:
                return
            tag, accepted = QInputDialog.getText(
                self,
                "Tag nuclide",
                f"Tag for {peak.nuclide}:",
            )
            if accepted and str(tag).strip():
                self._emit_for_spectrum(
                    CanvasIntentKind.TAG_NUCLIDE,
                    spectrum_id=peak.spectrum_id,
                    nuclide=peak.nuclide,
                    tag=str(tag).strip(),
                )

        def _context_add_component(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.ADD_COMPONENT,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    position=self._context_position,
                )

        def _context_split_peak(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.SPLIT_PEAK,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    position=self._context_position,
                    component_ids=peak.component_ids,
                )

        def _context_merge_components(self) -> None:
            peak = self._context_peak_overlay()
            if peak is not None and len(peak.component_ids) >= 2:
                self._emit_for_spectrum(
                    CanvasIntentKind.MERGE_PEAKS,
                    spectrum_id=peak.spectrum_id,
                    peak_id=peak.peak_id,
                    component_ids=peak.component_ids,
                )

        def _context_select_roi(self) -> None:
            roi = self._context_roi_overlay()
            if roi is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.SELECT_ROI,
                    spectrum_id=roi.spectrum_id,
                    roi_id=roi.roi_id,
                )

        def _context_delete_roi(self) -> None:
            roi = self._context_roi_overlay()
            if roi is not None:
                self._emit_for_spectrum(
                    CanvasIntentKind.DELETE_ROI,
                    spectrum_id=roi.spectrum_id,
                    roi_id=roi.roi_id,
                )

        def _context_create_roi(self) -> None:
            view_range = self.plot_item.viewRange()[0]
            width = max(abs(float(view_range[1]) - float(view_range[0])) * 0.04, 1.0e-6)
            lower = self._context_position - width
            upper = self._context_position + width
            self._emit_for_spectrum(
                CanvasIntentKind.CREATE_ROI,
                bounds=(lower, upper),
                left_background=(lower - 2.0 * width, lower),
                right_background=(upper, upper + 2.0 * width),
                drag_token=f"roi-menu-{uuid4().hex}",
            )

        def _context_assign_role(self, role: str) -> None:
            self._emit_for_spectrum(CanvasIntentKind.ASSIGN_SPECTRUM_ROLE, role=role)

        def _context_crosshair_toggled(self, enabled: bool) -> None:
            self.set_crosshair_enabled(bool(enabled))
            self._emit_for_spectrum(
                CanvasIntentKind.TOGGLE_CROSSHAIR, enabled=bool(enabled)
            )

        def set_peak_residuals(
            self, peaks: Sequence[PeakCandidate], *, visible: bool
        ) -> None:
            self._residual_peaks = tuple(peaks)
            self._residual_mode = "compact" if visible else "off"
            for index, curve in enumerate(self.residual_curves):
                if index >= len(peaks):
                    curve.setData([], [])
                    self.residual_plots[index].setTitle("Residuals")
                    continue
                peak = peaks[index]
                curve.setData(
                    np.asarray(peak.residual_channels, dtype=float),
                    np.asarray(peak.normalized_residuals, dtype=float),
                )
                severity = max(
                    (abs(value) for value in peak.normalized_residuals), default=0.0
                )
                if severity >= 3.0:
                    color = "#ef4444"
                elif severity >= 2.0:
                    color = "#f59e0b"
                else:
                    color = "#10b981"
                curve.setSymbolBrush(pg.mkBrush(color))
                curve.setSymbolPen(pg.mkPen(color=color, width=1.0))
                self.residual_zero_lines[index].setPen(
                    pg.mkPen(color="#94a3b8", width=1)
                )
                self.residual_plots[index].setTitle(
                    f"{peak.energy_keV:.1f} keV · max |z| {severity:.2f}"
                )
            self._apply_residual_visibility()

        def set_residual_mode(self, mode: str) -> None:
            """Show or hide the already-rendered residual diagnostics."""

            if mode not in {"off", "compact", "full"}:
                raise ValueError("Residual mode must be off, compact, or full")
            self._residual_mode = mode
            self._apply_residual_visibility()

        def _apply_residual_visibility(self) -> None:
            self._residual_visible = self._residual_mode != "off"
            self.residual_row.setVisible(
                bool(self._residual_visible and self._residual_peaks)
            )

        def set_crosshair_enabled(self, enabled: bool) -> None:
            """Toggle the plot crosshair without changing the current viewport."""

            self._crosshair_enabled = bool(enabled)
            if self._crosshair_enabled:
                x_range, y_range = self.plot_item.viewRange()
                self._crosshair_vertical.setPos((x_range[0] + x_range[1]) / 2.0)
                self._crosshair_horizontal.setPos((y_range[0] + y_range[1]) / 2.0)
            self._crosshair_vertical.setVisible(self._crosshair_enabled)
            self._crosshair_horizontal.setVisible(self._crosshair_enabled)
            self.crosshair_readout.setVisible(self._crosshair_enabled)
            self.crosshair_button.blockSignals(True)
            self.crosshair_button.setChecked(self._crosshair_enabled)
            self.crosshair_button.blockSignals(False)
            if hasattr(self, "crosshair_action"):
                self.crosshair_action.blockSignals(True)
                self.crosshair_action.setChecked(self._crosshair_enabled)
                self.crosshair_action.blockSignals(False)

        def _crosshair_button_toggled(self, enabled: bool) -> None:
            self.set_crosshair_enabled(enabled)
            self._emit_for_spectrum(
                CanvasIntentKind.TOGGLE_CROSSHAIR,
                enabled=bool(enabled),
            )

        def _move_crosshair(self, scene_position) -> None:
            if not self._crosshair_enabled:
                return
            if not self.plot.sceneBoundingRect().contains(scene_position):
                return
            data_position = self.plot_item.getViewBox().mapSceneToView(scene_position)
            self._crosshair_vertical.setPos(float(data_position.x()))
            self._crosshair_horizontal.setPos(float(data_position.y()))
            self.crosshair_readout.setText(
                f"x {float(data_position.x()):.3f} | y {float(data_position.y()):.3f}"
            )

        def clear(self) -> None:
            self._replace_roi_items(())
            self._peak_overlays_by_id = {}
            self._selected_roi_id = None
            self._selected_peak_id = None
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

        def show_interaction_error(self, message: str) -> None:
            """Surface a rejected edit next to the canvas without a modal dialog."""

            detail = " ".join(str(message).split())
            self.status_label.setText(f"Edit rejected: {detail}")
            self.status_label.setToolTip(detail)

        def reset_persisted_viewport_state(self) -> None:
            """Restore default view controls after undo removes the viewport leaf."""

            self._viewport_commit_timer.stop()
            self.set_log_scale(False)
            self.set_peak_labels_visible(True)
            self.set_crosshair_enabled(False)
            self.plot_item.enableAutoRange(x=True, y=True)
            self.plot_item.autoRange()

        def reset_view(self) -> None:
            """Fit all visible traces while retaining mouse pan and wheel zoom."""

            self.plot_item.enableAutoRange(x=True, y=True)
            self.plot_item.autoRange()
            self._schedule_viewport_commit()

        def zoom_to_range(
            self,
            lower: float,
            upper: float,
            *,
            padding_fraction: float = 0.12,
        ) -> None:
            lower = float(lower)
            upper = float(upper)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("zoom range must be finite and increasing")
            if not np.isfinite(padding_fraction) or padding_fraction < 0.0:
                raise ValueError("padding_fraction must be finite and non-negative")
            padding = (upper - lower) * float(padding_fraction)
            self.plot_item.disableAutoRange()
            self.plot_item.setXRange(lower - padding, upper + padding, padding=0.0)

        def _zoom_view(self, factor: float) -> None:
            """Apply a centered zoom while keeping free mouse pan/zoom enabled."""

            self.plot_item.disableAutoRange()
            self.plot_item.getViewBox().scaleBy((float(factor), float(factor)))
            self._schedule_viewport_commit()

        def _schedule_viewport_commit(self) -> None:
            if self._interaction_spectrum_id is not None:
                self._viewport_commit_timer.start()

        def _commit_viewport_intent(self) -> None:
            x_range, y_range = self.plot_item.viewRange()
            self._emit_for_spectrum(
                CanvasIntentKind.CHANGE_VIEWPORT,
                viewport_id="primary-spectrum",
                x_range=(float(x_range[0]), float(x_range[1])),
                y_range=(float(y_range[0]), float(y_range[1])),
            )

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

        def viewport_state(
            self,
            *,
            viewport_id: str = "primary-spectrum",
            spectrum_id: str | None = None,
            selected_roi_id: str | None = None,
        ) -> CanvasViewport:
            """Capture the current pan/zoom and display toggles without Qt types."""

            x_range, y_range = self.plot_item.viewRange()
            viewport = CanvasViewport(
                viewport_id=viewport_id,
                spectrum_id=spectrum_id,
                x_range=(float(x_range[0]), float(x_range[1])),
                y_range=(float(y_range[0]), float(y_range[1])),
                x_unit="energy_keV" if self._x_axis_is_energy else "channel",
                y_unit="counts",
                log_y=self._log_scale,
                overlays=tuple(
                    name
                    for name, enabled in (
                        ("peak-labels", self._peak_labels_visible),
                        (
                            "roi",
                            self._roi_region.isVisible() or bool(self._roi_items_by_id),
                        ),
                    )
                    if enabled
                ),
                residual_mode=self._residual_mode,
                selected_roi_id=selected_roi_id,
                crosshair_enabled=self._crosshair_enabled,
                labels_visible=self._peak_labels_visible,
            )
            viewport.validate("viewport")
            return viewport

        def apply_viewport_state(self, viewport: CanvasViewport) -> None:
            """Restore a validated view after spectrum data has been loaded."""

            viewport.validate("viewport")
            self.set_log_scale(viewport.log_y)
            self.set_peak_labels_visible(viewport.labels_visible)
            roi_visible = "roi" in viewport.overlays
            self.roi_button.blockSignals(True)
            self.roi_button.setChecked(roi_visible)
            self.roi_button.blockSignals(False)
            self._set_roi_visible(roi_visible)
            self.set_residual_mode(viewport.residual_mode)
            if viewport.x_range is not None or viewport.y_range is not None:
                self.plot_item.disableAutoRange()
            if viewport.x_range is not None:
                self.plot_item.setXRange(*viewport.x_range, padding=0.0)
            if viewport.y_range is not None:
                self.plot_item.setYRange(*viewport.y_range, padding=0.0)
            self.set_crosshair_enabled(viewport.crosshair_enabled)

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
                    f"ROI {state.roi_bounds_keV[0]:.1f}-"
                    f"{state.roi_bounds_keV[1]:.1f} keV"
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

    def catalog_pyqtgraph_export_action(plot_widget, object_name: str) -> None:
        """No-op unless the PyQtGraph backend is available."""

        del plot_widget, object_name

    class PyQtGraphSpectrumCanvas(
        SpectrumCanvas
    ):  # pragma: no cover - placeholder without optional deps
        """Import-safe placeholder when Qt or PyQtGraph is unavailable."""

        backend_key = "pyqtgraph"
        capabilities = RendererCapabilities()

        def __init__(
            self, selection_bus: SelectionBus | None = None, parent=None
        ) -> None:
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

        def viewport_state(
            self,
            *,
            viewport_id: str = "primary-spectrum",
            spectrum_id: str | None = None,
            selected_roi_id: str | None = None,
        ) -> CanvasViewport:
            del viewport_id, spectrum_id, selected_roi_id
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )

        def apply_viewport_state(self, viewport: CanvasViewport) -> None:
            del viewport
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )
