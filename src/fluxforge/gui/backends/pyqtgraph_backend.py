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

        from fluxforge.gui.qt_compat import QHBoxLayout, QLabel, QVBoxLayout, QWidget

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
            self._reference_lines: list[object] = []
            self._cascade_sum_lines: list[object] = []
            self._overlay_traces: list[object] = []
            self._peak_scatter = None
            self._residual_visible = False

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
            self.plot_item.addItem(self._peak_scatter)
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
            self.trace_item.setData(channels, level.counts)
            self.trace_item.setPen(pg.mkPen(color=primary.color, width=2))

            for overlay in traces[1:]:
                if not overlay.visible:
                    continue
                item = self.plot_item.plot(
                    tuple(float(value) for value in overlay.channels),
                    tuple(float(value) for value in overlay.counts),
                    pen=pg.mkPen(color=overlay.color, width=1.35),
                )
                self._overlay_traces.append(item)

            visible_labels = ", ".join(trace.label for trace in traces if trace.visible)
            self.status_label.setText(
                f"{len(values):,} channels · {len(self.buffer.levels)} LOD levels · {visible_labels}"
            )

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
            for line in self._reference_lines:
                self.plot_item.removeItem(line)
            self._reference_lines.clear()

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
                self._peak_scatter.setData([], [])
                return
            x_values = [float(peak.channel) for peak in peaks]
            y_values = [
                float(self.buffer.full_resolution[min(int(round(peak.channel)), len(self.buffer.full_resolution) - 1)])
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
            self.trace_item.setData([], [])
            self.set_annotation_lines(())
            self.set_cascade_sum_lines(())
            self.set_peak_candidates(())
            self.set_peak_residuals((), visible=False)
            self.status_label.setText("No spectrum loaded")

        def _on_selection_changed(self, state: SelectionState) -> None:
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
