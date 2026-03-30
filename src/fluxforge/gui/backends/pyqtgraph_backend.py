"""PyQtGraph-backed spectrum canvas implementation."""

from __future__ import annotations

from typing import Sequence

from fluxforge.gui.qt_compat import QT_AVAILABLE, QT_IMPORT_ERROR
from fluxforge.gui.selection_bus import SelectionBus, SelectionState
from fluxforge.gui.spectrum_canvas import HierarchicalSpectrumBuffer, RendererCapabilities, SpectrumCanvas


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
            shell.addWidget(self.plot, 1)

            if self.selection_bus is not None:
                self.selection_bus.subscribe(self._on_selection_changed)

        def set_spectrum(self, counts: Sequence[float]) -> None:
            values = tuple(float(value) for value in counts)
            self.buffer = HierarchicalSpectrumBuffer.from_counts(values)
            level = self.buffer.choose_level(pixel_width=1400)
            channels = [index * level.stride for index in range(len(level.counts))]
            self.trace_item.setData(channels, level.counts)
            self.status_label.setText(
                f"{len(values):,} channels · {len(self.buffer.levels)} LOD levels"
            )

        def set_reference_lines(self, energies_keV: Sequence[float]) -> None:
            for line in self._reference_lines:
                self.plot_item.removeItem(line)
            self._reference_lines.clear()

            for energy in energies_keV:
                line = pg.InfiniteLine(
                    pos=float(energy),
                    angle=90,
                    pen=pg.mkPen(color="#f59e0b", width=1, style=pg.QtCore.Qt.DashLine),
                )
                self.plot_item.addItem(line)
                self._reference_lines.append(line)

        def clear(self) -> None:
            self.buffer = HierarchicalSpectrumBuffer.from_counts(())
            self.trace_item.setData([], [])
            self.set_reference_lines(())
            self.status_label.setText("No spectrum loaded")

        def _on_selection_changed(self, state: SelectionState) -> None:
            fragments = []
            if state.peak_energy_keV is not None:
                fragments.append(f"{state.peak_energy_keV:.3f} keV")
            if state.nuclide:
                fragments.append(state.nuclide)
            if state.roi_bounds_keV:
                fragments.append(
                    f"ROI {state.roi_bounds_keV[0]:.1f}-{state.roi_bounds_keV[1]:.1f} keV"
                )
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

        def clear(self) -> None:
            raise RuntimeError(
                "PyQtGraph renderer is unavailable. Install the `native-gui` extra."
            )
